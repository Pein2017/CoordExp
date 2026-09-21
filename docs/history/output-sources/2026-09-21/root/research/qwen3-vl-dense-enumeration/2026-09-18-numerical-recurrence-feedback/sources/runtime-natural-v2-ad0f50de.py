"""Native numerical-feedback capture, paired qualification, and release.

CPU selection owns episode meaning.  This module only binds the frozen native
histories to the real FP32/SDPA model and records replay evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
from transformers import (
    GenerationConfig,
    LogitsProcessor,
    LogitsProcessorList,
    StoppingCriteria,
    StoppingCriteriaList,
)

from probes.training_set_completion.readout_norm_fresh import _binding, _input_identity
from probes.training_set_completion.untied_shared import load_model
from probes.training_set_completion.numerical_feedback.select import (
    ROLES,
    replacements,
    rows as parsed_rows,
)
from src.inference.bound_requests import build_bound_native_requests
from src.qwen.native import (
    _STALE_HISTORY_FIELDS,
    exact_history_inputs,
    prepare_native_inputs,
)


EOS, COORD, END = 151645, 151670, 151649
RELEASE_ROWS, RELEASE_TOKENS = 32, 512


def write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def token_hash(tokens: list[int]) -> str:
    return canonical_sha256(tokens)


def _as_list(value: torch.Tensor) -> list[float]:
    values = value.detach().float().cpu().tolist()
    if not all(torch.isfinite(torch.tensor(v)) for v in values):
        raise ValueError("nonfinite captured logits")
    return [float(v) for v in values]


def _source_key(boundary: dict[str, Any]) -> str:
    return f"{boundary['model']}-{boundary['group']}-{boundary['image_id']}-{boundary['kind']}"


def _slot_key(role: str, offset: int) -> str:
    return f"{role}@{offset}"


def _prefix_tokens(raw: list[dict[str, Any]], offset: int, pad: int) -> list[list[int]]:
    """Make equal-length prefixes, ending short companions with EOS then pad."""
    result: list[list[int]] = []
    for row in raw:
        tokens = list(row["token_ids"][:offset])
        if len(tokens) < offset:
            if EOS not in tokens:
                tokens.append(EOS)
            else:
                tokens = tokens[: tokens.index(EOS) + 1]
            tokens.extend([pad] * (offset - len(tokens)))
        result.append(tokens[:offset])
    if any(len(row) != offset for row in result):
        raise ValueError("prefix materialization did not produce equal lengths")
    return result


def full_prefix(
    batch: Any,
    raw: list[dict[str, Any]],
    offset: int,
    pad: int,
    device: torch.device | str,
    mutation: tuple[int, int, int, int] | None = None,
) -> dict[str, Any]:
    """Build a same-width native prefix while retaining the image tensors."""
    suffix = _prefix_tokens(raw, offset, pad)
    if mutation is not None:
        batch_index, position, old, new = mutation
        if not 0 <= batch_index < len(suffix) or not 0 <= position < offset:
            raise ValueError("mutation position is outside the native prefix")
        if suffix[batch_index][position] != old:
            raise ValueError("mutation does not name the original prefix token")
        suffix[batch_index][position] = new
    inputs = {
        key: value
        for key, value in batch.inputs.items()
        if key not in _STALE_HISTORY_FIELDS
        and key not in ("use_cache", "return_dict", "logits_to_keep")
    }
    ext = torch.tensor(suffix, device=device, dtype=torch.long)
    inputs["input_ids"] = torch.cat((batch.inputs["input_ids"], ext), dim=1)
    inputs["attention_mask"] = torch.cat(
        (
            batch.inputs["attention_mask"],
            torch.ones(
                len(raw), offset, device=device, dtype=batch.inputs["attention_mask"].dtype
            ),
        ),
        dim=1,
    )
    return inputs


def _trim_generated(tokens: list[int], cap: int) -> tuple[list[int], str]:
    if EOS in tokens[:cap]:
        end = tokens.index(EOS) + 1
        return tokens[:end], "eos"
    return tokens[:cap], "cap"


class Runtime:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.output = args.output
        self.output.mkdir(parents=True, exist_ok=False)
        self.started = time.monotonic()
        self.panel = json.loads(args.panel.read_text())
        self.handles: list[Any] = []
        self.last: dict[str, torch.Tensor] = {}
        self.ledger: dict[str, Any] = {
            "schema": "numerical_feedback.runtime.v2",
            "status": "running",
            "pid": os.getpid(),
            "model": args.model,
            "mode": args.mode,
            "model_forwards": 0,
            "vision_forwards": 0,
            "tensor_bytes": 0,
            "panel": _binding(args.panel),
            "producer": _binding(Path(__file__)),
            "artifacts": [],
        }
        self.persist()
        self.q, self.ledger["identity"] = load_model(args.model, args.device)
        self.model = self.q.model
        self.head = self.model.get_output_embeddings()
        ids = self.head.selected_token_ids
        if ids[4:].tolist() != list(range(COORD, COORD + 1000)):
            raise ValueError("unexpected coordinate token IDs")
        self.ids = ids[4:]
        self.U = (self.head.base.weight[ids] + self.head.shared_embed_delta).detach()[4:]
        self.E = self.model.get_input_embeddings()(ids).detach()[4:]
        norms = self.U.double().norm(dim=1)
        self.factors = norms.median() / norms
        self.versions = {
            name: parameter._version for name, parameter in self.model.named_parameters()
        }

        def count(_module: Any, _inputs: tuple[Any, ...]) -> None:
            self.ledger["model_forwards"] += 1
            if self.ledger["model_forwards"] > args.max_forwards:
                raise RuntimeError("allocated forward budget exhausted")
            if time.monotonic() - self.started > args.max_seconds:
                raise RuntimeError("allocated wall-time budget exhausted")

        def vision(_module: Any, _inputs: tuple[Any, ...]) -> None:
            self.ledger["vision_forwards"] += 1

        def hidden(_module: Any, inputs: tuple[Any, ...]) -> None:
            value = inputs[0]
            if not isinstance(value, torch.Tensor):
                raise TypeError("lm head input hook did not receive a tensor")
            self.last["head_input"] = value.detach()

        def logits(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            value = output[0] if isinstance(output, tuple) else output
            if not isinstance(value, torch.Tensor):
                raise TypeError("lm head output hook did not receive a tensor")
            self.last["logits"] = value.detach()

        self.handles = [
            self.model.register_forward_pre_hook(count),
            self.model.model.visual.register_forward_pre_hook(vision),
            self.head.register_forward_pre_hook(hidden),
            self.head.register_forward_hook(logits),
        ]
        self.save(
            "weights.pt",
            {
                "coordinate_ids": self.ids.cpu(),
                "input_rows": self.E.cpu(),
                "output_rows": self.U.cpu(),
                "factors": self.factors.cpu(),
            },
        )
        self.persist()

    def persist(self) -> None:
        self.ledger["gpu_seconds"] = time.monotonic() - self.started
        write(self.output / "receipt.json", self.ledger)

    def save(self, name: str, payload: Any) -> Path:
        path = self.output / name
        torch.save(payload, path)
        self.ledger["tensor_bytes"] += path.stat().st_size
        if self.ledger["tensor_bytes"] > self.args.max_bytes:
            raise RuntimeError("allocated tensor payload budget exhausted")
        self.ledger["artifacts"].append(_binding(path))
        return path

    def _config(self, model: str, group: dict[str, Any]) -> dict[str, Any]:
        config = dict(self.panel["configs"][model])
        config["data"] = dict(input_jsonl=group["input_jsonl"])
        return config

    def _group(self, key: str) -> dict[str, Any]:
        return next(group for group in self.panel["groups"] if group["key"] == key)

    def _prepare_full(self, group: dict[str, Any]) -> Any:
        requests, _ = build_bound_native_requests(
            self.q, self._config(self.args.model, group), group["cases"]
        )
        return prepare_native_inputs(
            self.q.processor, requests, device=self.args.device, record_media_identity=True
        )

    def _prepare_group(
        self, group: dict[str, Any], receipt: dict[str, Any]
    ) -> Any:
        """Reopen the complete native group; companions stay in the replay batch."""
        batch = self._prepare_full(group)
        if _input_identity(batch) != receipt["input_identity"]:
            raise ValueError("source heterogeneous batch identity changed")
        return batch

    def _source(
        self, boundary: dict[str, Any]
    ) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        group = self._group(boundary["group"])
        raw_path = Path(boundary["raw_path"])
        trace_path = Path(boundary["trace_path"])
        receipt_path = Path(boundary["receipt_path"])
        expected = next(
            source
            for source in self.panel["saved_sources"]
            if source["condition"] == f"{self.args.model}-original"
            and source["group"] == boundary["group"]
        )
        for key, path in (
            ("raw", raw_path),
            ("trace", trace_path),
            ("receipt", receipt_path),
        ):
            if _binding(path) != expected[key]:
                raise ValueError(f"source {key} binding changed")
        raw = json.loads(raw_path.read_text())["rows"]
        trace = json.loads(trace_path.read_text())["steps"]
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("status") != "candidate_complete":
            raise ValueError("source natural receipt is not complete")
        if len(raw) != len(group["cases"]):
            raise ValueError("source companion count changed")
        if int(raw[boundary["batch_index"]]["image_id"]) != int(boundary["image_id"]):
            raise ValueError("source batch index/image identity changed")
        native = raw[boundary["batch_index"]]["token_ids"]
        if native != boundary["native_tokens"] or token_hash(native) != boundary["native_token_hash"]:
            raise ValueError("source native token identity changed")
        if boundary["source_row"]["end"] > len(native):
            raise ValueError("source row is outside native history")
        return group, raw, {"steps": trace}, receipt

    @staticmethod
    def _fixed_suffix_end(native: list[int], boundary: dict[str, Any]) -> int:
        """Return the exclusive end of the latest complete target row."""
        last_row = max(int(slot["row_index"]) for slot in boundary["target_slots"])
        row = next((item for item in parsed_rows(native) if item["index"] == last_row), None)
        if row is None:
            raise ValueError("last target row is not a complete parsed native row")
        declared = boundary.get("fixed_suffix_end")
        if declared is not None:
            end = int(declared)
            if end != int(row["end"]):
                raise ValueError("declared fixed suffix end does not close the final target row")
            return end
        return int(row["end"])

    def _score_history(
        self,
        batch: Any,
        raw: list[dict[str, Any]],
        target_index: int,
        keep_start: int,
        max_token: int,
        mutation: tuple[int, int, int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        if not 0 <= target_index < len(raw):
            raise ValueError("target batch index is outside native group")
        native = list(raw[target_index]["token_ids"])
        if not 0 <= keep_start <= max_token < len(native):
            raise ValueError("score span is outside native history")
        prefixes = _prefix_tokens(
            raw, max_token, self.q.tokenizer.pad_token_id
        )
        if mutation is not None:
            batch_index, position, old, new = mutation
            if not 0 <= batch_index < len(prefixes):
                raise ValueError("score mutation batch index is outside native group")
            if position >= len(prefixes[batch_index]) or prefixes[batch_index][position] != old:
                raise ValueError("score mutation does not match native token")
            prefixes[batch_index][position] = new
        prompt_rows = [list(row) for row in batch.prompt_token_ids]
        histories = [prompt + prefix for prompt, prefix in zip(prompt_rows, prefixes, strict=True)]
        keep = max_token - keep_start + 1
        inputs = exact_history_inputs(
            self.model,
            batch.inputs,
            histories,
            pad_token_id=self.q.tokenizer.pad_token_id,
            logits_to_keep=keep,
        )
        self.last.clear()
        with torch.inference_mode():
            outputs = self.model(**inputs)
        scores = outputs.logits.detach().float()
        hidden = self.last.get("head_input")
        if hidden is None:
            raise RuntimeError("native replay did not capture lm-head inputs")
        hidden = hidden.detach().float()
        if (
            scores.ndim != 3
            or hidden.ndim != 3
            or scores.shape[0] != len(raw)
            or hidden.shape[0] != len(raw)
            or scores.shape[1] != keep
            or hidden.shape[1] != keep
        ):
            raise RuntimeError("compact replay logits/head inputs are misaligned")
        return scores, hidden, {
            "batch_size": len(raw),
            "target_index": target_index,
            "keep_start": keep_start,
            "max_token": max_token,
            "history_token_counts": [len(history) for history in histories],
            "prompt_token_counts": [len(prompt) for prompt in prompt_rows],
            "padded_history_width": int(inputs["input_ids"].shape[1]),
            "mutation": mutation,
        }

    def _slot(
        self,
        scores: torch.Tensor,
        hidden: torch.Tensor,
        offset: int,
        role: str,
        native: list[int],
        boundary: dict[str, Any],
        score_meta: dict[str, Any],
    ) -> dict[str, Any]:
        index = offset - int(score_meta["keep_start"])
        batch_index = int(score_meta["target_index"])
        if not 0 <= index < scores.shape[1] or not 0 <= batch_index < scores.shape[0]:
            raise ValueError("slot is outside compact replay span")
        score = scores[batch_index, index]
        head_input = hidden[batch_index, index]
        coordinate = score[self.ids]
        normalized = (
            coordinate.double() * self.factors.to(device=coordinate.device)
        ).to(coordinate.dtype)
        top = torch.topk(score, 5)
        native_token = int(native[offset])
        logprobs = torch.log_softmax(score.double(), dim=0)
        native_value = float(score[native_token].item())
        native_rank = int((score > score[native_token]).sum().item()) + 1
        coord_winner = int(torch.argmax(coordinate).item())
        coord_top = torch.topk(coordinate, 2)
        return {
            "offset": offset,
            "role": role,
            "native_token_id": native_token,
            "native_logit": native_value,
            "native_logprob": float(logprobs[native_token].item()),
            "native_rank": native_rank,
            "head_input": _as_list(head_input),
            "coordinate_logits": _as_list(coordinate),
            "normalized_coordinate_logits": _as_list(normalized),
            "normalized_minus_original": _as_list(normalized - coordinate),
            "top_tokens": [
                {"token_id": int(token), "logit": float(value)}
                for token, value in zip(
                    top.indices.tolist(), top.values.tolist(), strict=True
                )
            ],
            "winner_margin": float((top.values[0] - top.values[1]).item()),
            "eos_logit": float(score[EOS].item()),
            "eos_probability": float(torch.exp(logprobs[EOS]).item()),
            "current_pattern_support": {
                "source_values": list(boundary["source_row"]["values"]),
                "valid": bool(boundary["source_row"]["valid"]),
                "coordinate_argmax_value": coord_winner,
                "coordinate_top2_margin": float(
                    (coord_top.values[0] - coord_top.values[1]).item()
                ),
                "native_is_coordinate": COORD <= native_token < COORD + 1000,
            },
        }

    def _trace_parity(
        self,
        score: torch.Tensor,
        offset: int,
        batch_index: int,
        trace: dict[str, Any],
    ) -> dict[str, Any] | None:
        steps = trace["steps"]
        if offset >= len(steps) or batch_index >= len(steps[offset].get("raw_top2", [])):
            return None
        step = steps[offset]
        top = torch.topk(score, 2)
        error = max(
            abs(float(top.values[i].item()) - float(step["raw_top2"][batch_index][i]))
            for i in (0, 1)
        )
        winner = int(top.indices[0].item()) == int(step["raw_winners"][batch_index])
        tolerance = float(self.panel["tolerances"]["logit_atol"])
        return {
            "offset": offset,
            "top2_max_abs_error": error,
            "winner_match": winner,
            "tolerance": tolerance,
            "passed": bool(winner and error <= tolerance),
        }

    def _capture_boundary(self, boundary: dict[str, Any]) -> dict[str, Any]:
        group, raw, trace, receipt = self._source(boundary)
        batch = self._prepare_group(group, receipt)
        native = list(boundary["native_tokens"])
        target_index = int(boundary["batch_index"])
        prompt = list(batch.prompt_token_ids[target_index])
        target_slots = list(boundary["target_slots"])
        fixed_end = self._fixed_suffix_end(native, boundary)
        if fixed_end <= int(boundary["source_row"]["end"]):
            raise ValueError("fixed suffix must extend through a complete target row")
        max_target = fixed_end - 1
        keep_start = int(boundary["source_row"]["start"])
        scores, hidden, score_meta = self._score_history(
            batch, raw, target_index, keep_start, max_target
        )
        score_meta["fixed_suffix_end_exclusive"] = fixed_end
        slot_records: dict[str, Any] = {}
        parity: list[dict[str, Any]] = []
        for index, offset in enumerate(boundary["source_row"]["coordinate_offsets"]):
            role = ROLES[index]
            record = self._slot(scores, hidden, offset, role, native, boundary, score_meta)
            record["trace_parity"] = self._trace_parity(
                score=scores[target_index, offset - keep_start],
                offset=offset,
                batch_index=target_index,
                trace=trace,
            )
            if record["trace_parity"] is not None:
                parity.append(record["trace_parity"])
            slot_records[_slot_key(f"source_{role}", offset)] = record
        for target in target_slots:
            offset = int(target["offset"])
            record = self._slot(
                scores,
                hidden,
                offset,
                str(target["role"]),
                native,
                boundary,
                score_meta,
            )
            record["row_delay"] = int(target["row_delay"])
            record["row_index"] = int(target["row_index"])
            record["trace_parity"] = self._trace_parity(
                score=scores[target_index, offset - keep_start],
                offset=offset,
                batch_index=target_index,
                trace=trace,
            )
            if record["trace_parity"] is not None:
                parity.append(record["trace_parity"])
            slot_records[_slot_key(str(target["role"]), offset)] = record
        fixed = self._fixed_suffix(
            scores,
            native,
            int(boundary["source_row"]["end"]),
            fixed_end,
            keep_start,
            target_index,
        )
        expected_parity_slots = len(boundary["source_row"]["coordinate_offsets"]) + len(
            target_slots
        )
        if len(parity) != expected_parity_slots:
            raise RuntimeError("native source trace is missing a captured slot")
        if any(not item["passed"] for item in parity):
            raise RuntimeError("native source replay parity failed")
        return {
            "schema": "numerical_feedback.source_capture.v1",
            "boundary_id": boundary["id"],
            "model": boundary["model"],
            "group": boundary["group"],
            "image_id": boundary["image_id"],
            "batch_index": boundary["batch_index"],
            "source_row": boundary["source_row"],
            "target_slots": target_slots,
            "native_tokens": native,
            "native_token_hash": token_hash(native),
            "prefix_hash": token_hash(native[: int(boundary["source_row"]["end"])]),
            "prompt_token_hash": token_hash(prompt),
            "score_span": score_meta,
            "slots": slot_records,
            "fixed_suffix": fixed,
            "fixed_suffix_end_exclusive": fixed_end,
            "parity": parity,
            "input_identity": _input_identity(batch),
        }

    def capture_source(self, selection_path: Path, manifest_path: Path) -> None:
        selection_binding = _binding(selection_path)
        selection = json.loads(selection_path.read_text())
        if selection.get("status") != "frozen" or any(
            item.get("status") == "pending_natural"
            for item in selection.get("exclusions", [])
        ):
            raise ValueError("selection is not frozen after natural source completion")
        boundaries = [
            item for item in selection["boundaries"] if item["model"] == self.args.model
        ]
        if self.args.group:
            boundaries = [item for item in boundaries if item["group"] == self.args.group]
        if not boundaries:
            raise ValueError("no selected boundaries for model/group")
        entries = []
        for boundary in boundaries:
            capture = self._capture_boundary(boundary)
            path = self.output / f"{_source_key(boundary)}.source.pt"
            torch.save(capture, path)
            self.ledger["tensor_bytes"] += path.stat().st_size
            self.ledger["artifacts"].append(_binding(path))
            source_offsets = {
                role: capture["slots"][
                    _slot_key(
                        f"source_{role}",
                        boundary["source_row"]["coordinate_offsets"][index],
                    )
                ]["coordinate_logits"]
                for index, role in enumerate(ROLES)
            }
            candidate_rows = []
            for role in ROLES:
                for candidate in replacements(boundary, role, source_offsets[role]):
                    candidate = dict(candidate)
                    original = int(candidate["original_value"])
                    value = int(candidate["value"])
                    candidate["embedding_distance"] = float(
                        (self.E[value] - self.E[original]).norm().item()
                    )
                    candidate_rows.append(candidate)
            entries.append(
                {
                    "boundary": boundary,
                    "source_capture": _binding(path),
                    "candidates": candidate_rows,
                }
            )
            self.ledger["captures"] = len(entries)
            self.persist()
        manifest = {
            "schema": "numerical_feedback.candidate_manifest.v1",
            "status": "frozen",
            "selection": selection_binding,
            "panel": _binding(self.args.panel),
            "model": self.args.model,
            "identity": self.ledger["identity"],
            "boundaries": entries,
            "candidate_policy": "parent select.replacements; frozen before paired replay/release",
        }
        if manifest_path.exists():
            raise FileExistsError(f"candidate manifest already exists: {manifest_path}")
        write(manifest_path, manifest)
        self.ledger["candidate_manifest"] = _binding(manifest_path)
        self.ledger["status"] = "source_capture_complete"
        self.persist()

    def _load_manifest_entry(self, manifest_path: Path) -> dict[str, Any]:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") != "frozen" or manifest.get("model") != self.args.model:
            raise ValueError("candidate manifest is not frozen for this model")
        if manifest["panel"] != _binding(self.args.panel):
            raise ValueError("candidate manifest panel binding changed")
        if manifest.get("identity") != self.ledger["identity"]:
            raise ValueError("candidate manifest model identity changed")
        selection_path = Path(manifest["selection"]["path"])
        if _binding(selection_path) != manifest["selection"]:
            raise ValueError("candidate manifest selection binding changed")
        boundary_id = self.args.boundary_id or manifest["boundaries"][0]["boundary"]["id"]
        entry = next(
            item for item in manifest["boundaries"] if item["boundary"]["id"] == boundary_id
        )
        candidate_kind = self.args.candidate_kind or "adjacent"
        role = self.args.role or entry["candidates"][0]["role"]
        candidates = [
            item
            for item in entry["candidates"]
            if item["role"] == role and item["kind"] == candidate_kind
        ]
        if not candidates:
            raise ValueError("requested candidate is absent from frozen manifest")
        return {"manifest": manifest, "entry": entry, "candidate": candidates[0]}

    def _paired_scores(
        self,
        boundary: dict[str, Any],
        capture: dict[str, Any],
        candidate: dict[str, Any],
    ) -> dict[str, Any]:
        group, raw, _trace, receipt = self._source(boundary)
        batch = self._prepare_group(group, receipt)
        native = list(boundary["native_tokens"])
        target_index = int(boundary["batch_index"])
        fixed_end = self._fixed_suffix_end(native, boundary)
        if fixed_end <= int(boundary["source_row"]["end"]):
            raise ValueError("fixed suffix must extend through a complete target row")
        max_target = fixed_end - 1
        keep_start = int(boundary["source_row"]["start"])
        offset = int(candidate["source_offset"])
        old_token = COORD + int(candidate["original_value"])
        new_token = COORD + int(candidate["value"])
        if native[offset] != old_token:
            raise ValueError("candidate source token identity changed")
        native_scores, native_hidden, score_meta = self._score_history(
            batch, raw, target_index, keep_start, max_target
        )
        noop_scores, noop_hidden, _ = self._score_history(
            batch,
            raw,
            target_index,
            keep_start,
            max_target,
            mutation=(target_index, offset, old_token, old_token),
        )
        candidate_scores, candidate_hidden, _ = self._score_history(
            batch,
            raw,
            target_index,
            keep_start,
            max_target,
            mutation=(target_index, offset, old_token, new_token),
        )
        diff = (native_scores - noop_scores).abs()
        target_diff = diff[target_index]
        tolerance = float(self.panel["tolerances"]["logit_atol"])
        native_choice = native_scores.argmax(dim=-1)
        noop_choice = noop_scores.argmax(dim=-1)
        parity = {
            "max_abs_error": float(diff.max().item()),
            "target_max_abs_error": float(target_diff.max().item()),
            "exact_choices": bool(torch.equal(native_choice, noop_choice)),
            "tolerance": tolerance,
            "source_token_id": old_token,
            "noop_token_id": old_token,
            "batch_size": len(raw),
            "target_batch_index": target_index,
        }
        source_replay = {}
        candidate_replay = {}
        for slot in boundary["target_slots"]:
            slot_offset = int(slot["offset"])
            source_replay[_slot_key(slot["role"], slot_offset)] = self._slot(
                native_scores,
                native_hidden,
                slot_offset,
                slot["role"],
                native,
                boundary,
                score_meta,
            )
            candidate_replay[_slot_key(slot["role"], slot_offset)] = self._slot(
                candidate_scores,
                candidate_hidden,
                slot_offset,
                slot["role"],
                native,
                boundary,
                score_meta,
            )
            candidate_replay[_slot_key(slot["role"], slot_offset)]["coordinate_logit_diff"] = [
                float(value - base)
                for value, base in zip(
                    candidate_replay[_slot_key(slot["role"], slot_offset)]["coordinate_logits"],
                    source_replay[_slot_key(slot["role"], slot_offset)]["coordinate_logits"],
                    strict=True,
                )
            ]
        capture_parity = []
        for key, prior in capture["slots"].items():
            offset_value = int(prior["offset"])
            if offset_value < keep_start or offset_value > max_target:
                continue
            observed = native_scores[target_index, offset_value - keep_start]
            expected = torch.tensor(prior["coordinate_logits"], device=observed.device)
            actual = observed[self.ids]
            capture_parity.append(
                {
                    "slot": key,
                    "coordinate_max_abs_error": float((actual - expected).abs().max().item()),
                }
            )
        source_capture_max_error = max(
            (item["coordinate_max_abs_error"] for item in capture_parity), default=0.0
        )
        if len(capture_parity) != len(capture["slots"]):
            raise RuntimeError("paired replay does not cover every captured slot")
        parity["source_capture_coordinate_max_abs_error"] = source_capture_max_error
        parity["passed"] = bool(
            parity["max_abs_error"] <= tolerance
            and parity["exact_choices"]
            and source_capture_max_error <= tolerance
        )
        return {
            "boundary": boundary,
            "candidate": candidate,
            "source_token_identity": {
                "offset": offset,
                "original_value": int(candidate["original_value"]),
                "original_token_id": old_token,
                "replacement_value": int(candidate["value"]),
                "replacement_token_id": new_token,
                "native_token_hash": token_hash(native),
            },
            "score_span": score_meta,
            "parity": parity,
            "source_capture_parity": capture_parity,
            "native_slots": source_replay,
            "candidate_slots": candidate_replay,
            "fixed_suffix": self._fixed_suffix(
                native_scores,
                native,
                int(boundary["source_row"]["end"]),
                fixed_end,
                keep_start,
                target_index,
            ),
            "candidate_fixed_suffix": self._fixed_suffix(
                candidate_scores,
                native,
                int(boundary["source_row"]["end"]),
                fixed_end,
                keep_start,
                target_index,
            ),
            "fixed_suffix_end_exclusive": fixed_end,
        }

    def _fixed_suffix(
        self,
        scores: torch.Tensor,
        native: list[int],
        start: int,
        end_exclusive: int,
        keep_start: int,
        batch_index: int,
    ) -> dict[str, Any]:
        if not start <= end_exclusive <= len(native):
            raise ValueError("fixed suffix bounds are outside native history")
        values = []
        for offset in range(start, end_exclusive):
            score = scores[batch_index, offset - keep_start]
            token = int(native[offset])
            values.append(
                {
                    "offset": offset,
                    "token_id": token,
                    "logprob": float(torch.log_softmax(score.double(), dim=0)[token].item()),
                }
            )
        return {
            "start_offset": start,
            "end_offset_exclusive": end_exclusive,
            "end_offset_inclusive": end_exclusive - 1,
            "tokens": values,
            "logprob_sum": float(sum(item["logprob"] for item in values)),
        }

    def release(
        self,
        group: dict[str, Any],
        raw: list[dict[str, Any]],
        boundary: dict[str, Any],
        candidate: dict[str, Any],
    ) -> dict[str, Any]:
        batch = self._prepare_full(group)
        source_receipt = json.loads(Path(boundary["receipt_path"]).read_text())
        if _input_identity(batch) != source_receipt["input_identity"]:
            raise ValueError("release native batch identity changed")
        source_end = int(boundary["source_row"]["end"])
        target_index = int(boundary["batch_index"])
        old = COORD + int(candidate["original_value"])
        new = COORD + int(candidate["value"])
        target_prefix = list(raw[target_index]["token_ids"][:source_end])
        if len(target_prefix) != source_end or target_prefix[int(candidate["source_offset"])] != old:
            raise ValueError("release prefix does not contain the frozen source token")
        inputs = full_prefix(
            batch,
            raw,
            source_end,
            self.q.tokenizer.pad_token_id,
            self.args.device,
            mutation=(target_index, int(candidate["source_offset"]), old, new),
        )
        width = int(inputs["input_ids"].shape[1])
        stop_state: dict[str, Any] = {"complete_rows": 0, "eos": False}

        class RowStop(StoppingCriteria):
            def __call__(_self, ids: torch.Tensor, _scores: Any, **_kwargs: Any) -> bool:
                target = ids[target_index, width:].tolist()
                complete = len(parsed_rows(target))
                stop_state["complete_rows"] = complete
                stop_state["eos"] = EOS in target
                # Count parsed rows, so malformed END tokens cannot terminate a release.
                return bool(stop_state["eos"] or complete >= RELEASE_ROWS)

        result = self.generate(inputs, RELEASE_TOKENS, stopping=[RowStop()])
        rows = []
        for index, item in enumerate(raw):
            tokens, stop = _trim_generated(result[index, width:].tolist(), RELEASE_TOKENS)
            rows.append(
                {
                    "image_id": int(item["image_id"]),
                    "batch_index": index,
                    "token_ids": tokens,
                    "text": self.q.tokenizer.decode(
                        tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False
                    ),
                    "stop": stop,
                    "complete_rows": len(parsed_rows(tokens)),
                }
            )
        target = rows[target_index]
        return {
            "schema": "numerical_feedback.release.v1",
            "source_end": source_end,
            "target_batch_index": target_index,
            "mutation": {
                "offset": int(candidate["source_offset"]),
                "old_token_id": old,
                "new_token_id": new,
            },
            "stop": {
                "reason": "eos"
                if stop_state["eos"]
                else "rows"
                if stop_state["complete_rows"] >= RELEASE_ROWS
                else "cap",
                "complete_rows": int(stop_state["complete_rows"]),
                "token_cap": RELEASE_TOKENS,
                "row_cap": RELEASE_ROWS,
                "parser": "select.rows",
            },
            "rows": rows,
            "target": target,
        }

    def paired(self, manifest_path: Path) -> None:
        selected = self._load_manifest_entry(manifest_path)
        entry = selected["entry"]
        boundary = entry["boundary"]
        group, raw, _trace, _receipt = self._source(boundary)
        capture_path = Path(entry["source_capture"]["path"])
        if _binding(capture_path) != entry["source_capture"]:
            raise ValueError("source capture binding changed")
        capture = torch.load(capture_path, map_location="cpu", weights_only=False)
        replay = self._paired_scores(boundary, capture, selected["candidate"])
        if not replay["parity"]["passed"]:
            raise RuntimeError("paired native/noop parity gate failed")
        release = self.release(group, raw, boundary, selected["candidate"])
        replay_path = self.output / "paired-replay.pt"
        torch.save(replay, replay_path)
        release_path = self.output / "release.json"
        write(release_path, release)
        summary = {
            "schema": "numerical_feedback.paired_gate.v1",
            "status": "candidate_complete",
            "manifest": _binding(manifest_path),
            "boundary_id": boundary["id"],
            "candidate": selected["candidate"],
            "parity": replay["parity"],
            "source_capture_parity": replay["source_capture_parity"],
            "source_token_identity": replay["source_token_identity"],
            "replay": _binding(replay_path),
            "release": _binding(release_path),
            "release_stop": release["stop"],
        }
        summary_path = self.output / "paired.json"
        write(summary_path, summary)
        self.ledger.update(
            status="paired_gate_passed",
            manifest=_binding(manifest_path),
            boundary_id=boundary["id"],
            parity=replay["parity"],
            release_stop=release["stop"],
        )
        self.ledger["artifacts"].extend(
            [_binding(replay_path), _binding(release_path), _binding(summary_path)]
        )
        self.persist()

    def generate(
        self,
        inputs: dict[str, Any],
        cap: int,
        processors: list[Any] | None = None,
        stopping: list[Any] | None = None,
    ) -> torch.Tensor:
        config = GenerationConfig(
            max_new_tokens=cap,
            do_sample=False,
            repetition_penalty=1,
            eos_token_id=EOS,
            pad_token_id=self.q.tokenizer.pad_token_id,
        )
        with torch.inference_mode():
            return self.model.generate(
                **inputs,
                generation_config=config,
                use_model_defaults=False,
                logits_processor=LogitsProcessorList(processors or []),
                stopping_criteria=StoppingCriteriaList(stopping or []),
            )

    def natural(self) -> None:
        group = self._group(self.args.group)
        batch = self._prepare_full(group)
        self.ledger["input_identity"] = _input_identity(batch)
        self.ledger.update(
            condition=f"{self.args.model}-{self.args.policy}", group=self.args.group
        )
        width = batch.inputs["input_ids"].shape[1]
        traces: list[dict[str, Any]] = []
        runtime = self

        class Policy(LogitsProcessor):
            def __call__(_self, tokens: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
                transformed = scores.clone()
                transformed[:, runtime.ids] = (
                    scores[:, runtime.ids].double() * runtime.factors
                ).to(scores.dtype)
                used = transformed if runtime.args.policy == "normalized" else scores
                top = scores.topk(2)
                chosen = used.argmax(-1)
                traces.append(
                    {
                        "offset": tokens.shape[1] - width,
                        "raw_winners": top.indices[:, 0].tolist(),
                        "raw_top2": top.values.tolist(),
                        "raw_runnerups": top.indices[:, 1].tolist(),
                        "chosen": chosen.tolist(),
                        "eos_logits": scores[:, EOS].tolist(),
                        "logsumexp": scores.logsumexp(-1).tolist(),
                        "chosen_raw_logits": scores.gather(1, chosen[:, None]).squeeze(1).tolist(),
                    }
                )
                return used

        result = self.generate(batch.inputs, 3084, processors=[Policy()])
        rows = []
        for case, tokens in zip(group["cases"], result[:, width:].tolist(), strict=True):
            token_ids, stop = _trim_generated(tokens, 3084)
            rows.append(
                {
                    "image_id": int(case["input_record"]["image_id"]),
                    "row_id": case["row_id"],
                    "token_ids": token_ids,
                    "text": self.q.tokenizer.decode(
                        token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
                    ),
                    "stop": stop,
                }
            )
        write(self.output / "raw.json", {"rows": rows})
        write(self.output / "trace.json", {"steps": traces})
        self.ledger.update(
            input_identity=_input_identity(batch),
            raw=_binding(self.output / "raw.json"),
            trace=_binding(self.output / "trace.json"),
        )

    def close(self, error: BaseException | None = None) -> None:
        for handle in self.handles:
            handle.remove()
        if hasattr(self, "model") and hasattr(self, "versions"):
            current = {
                name: parameter._version for name, parameter in self.model.named_parameters()
            }
            if current != self.versions:
                error = error or RuntimeError("parameter mutation detected")
        if error is not None:
            self.ledger.update(status="partial_candidate", error=repr(error))
        elif self.ledger.get("status") == "running":
            self.ledger["status"] = "candidate_complete"
        self.ledger.update(live_jobs=[], ended_pid=os.getpid())
        self.persist()


def selfcheck() -> None:
    row = [151646, 9, 151647, 151648, COORD, COORD + 1, COORD + 2, COORD + 3, 151649]
    assert len(parsed_rows(row * 3 + [EOS])) == 3
    raw = [{"token_ids": [1, 2, 3]}, {"token_ids": [1]}]
    assert _prefix_tokens(raw, 3, 0)[1] == [1, EOS, 0]
    assert _trim_generated([1, 2, EOS, 0], 4) == ([1, 2, EOS], "eos")
    print("PASS native prefix, complete-row stop parser, and EOS trimming")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["natural", "capture-source", "paired"], nargs="?")
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--model", choices=["tied", "untied"], required=True)
    parser.add_argument("--group")
    parser.add_argument("--policy", choices=["original", "normalized"], default="original")
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--candidate-manifest", type=Path)
    parser.add_argument("--boundary-id")
    parser.add_argument("--role", choices=ROLES)
    parser.add_argument("--candidate-kind", choices=["adjacent", "far", "opposite"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-forwards", type=int, default=400000)
    parser.add_argument("--max-seconds", type=float, default=57600)
    parser.add_argument("--max-bytes", type=int, default=32 * 1024**3)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        selfcheck()
        return
    if args.mode is None:
        parser.error("mode is required unless --self-check is used")
    runtime = Runtime.__new__(Runtime)
    try:
        runtime.__init__(args)
        if args.mode == "natural":
            if not args.group:
                raise ValueError("natural mode requires --group")
            runtime.natural()
        elif args.mode == "capture-source":
            selection = args.selection or args.panel.parent / "selection.json"
            manifest = args.candidate_manifest or args.panel.parent / f"candidate-manifest-{args.model}.json"
            runtime.capture_source(selection, manifest)
        else:
            manifest = args.candidate_manifest or args.panel.parent / f"candidate-manifest-{args.model}.json"
            runtime.paired(manifest)
    except BaseException as exc:
        if hasattr(runtime, "ledger"):
            runtime.close(exc)
        raise
    else:
        runtime.close()


if __name__ == "__main__":
    main()
