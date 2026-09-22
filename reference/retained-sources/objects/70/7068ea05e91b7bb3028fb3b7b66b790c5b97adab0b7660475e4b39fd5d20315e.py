"""Bounded full-prefix source-path barrier runtime for Lane A.

The runtime deliberately does not use a generation KV cache.  Every score and
every released token is recomputed from the literal prompt plus the current
target suffix, which is the contract being tested here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

from probes.training_set_completion.numerical_feedback.runtime import _prefix_tokens
from probes.training_set_completion.numerical_feedback.select import rows as parsed_rows
from probes.training_set_completion.readout_norm_fresh import _binding, _input_identity
from probes.training_set_completion.untied_shared import load_model
from src.inference.bound_requests import build_bound_native_requests
from src.qwen.native import exact_history_inputs, prepare_native_inputs


EOS = 151645
COORD = 151670
ROW_OPEN, REF_END, BOX_START, ROW_END = 151646, 151647, 151648, 151649
ATOL = 2e-4
PLAN = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-21-history-source-sign/selection/shared-admission.json"
)
ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-21-history-source-sign"
)


def _json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _tensor_hash(value: torch.Tensor) -> str:
    data = value.detach().cpu().contiguous()
    return hashlib.sha256(data.view(torch.uint8).numpy().tobytes()).hexdigest()


def _write_once(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _finite(value: torch.Tensor) -> list[float]:
    values = value.detach().float().cpu().tolist()
    if not all(math.isfinite(float(item)) for item in values):
        raise ValueError("nonfinite full-vocabulary logit/logprob payload")
    return [float(item) for item in values]


def _cell_specs(plan: dict[str, Any]) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for lane_name, key in (("failure", "failure_boundaries"), ("control", "control_boundaries")):
        for boundary in plan["lane_a"][key]:
            for condition in ("native", "cut_A", "cut_C"):
                cell = dict(boundary)
                cell["lane"] = lane_name
                cell["condition"] = condition
                cell["id"] = f"{boundary['id']}--{condition}"
                cells.append(cell)
    return cells


def _source_record(plan: dict[str, Any], source_boundary_id: str) -> dict[str, Any]:
    matches = [
        item for item in plan["source_pool"]
        if item["source_boundary_id"] == source_boundary_id
    ]
    if len(matches) != 1:
        raise ValueError(f"source boundary is not unique: {source_boundary_id}")
    return matches[0]


def _group(panel: dict[str, Any], key: str) -> dict[str, Any]:
    matches = [item for item in panel["groups"] if item.get("key") == key]
    if len(matches) != 1:
        raise ValueError(f"native source group is not unique: {key}")
    return matches[0]


def _binding_matches(path_value: dict[str, Any]) -> Path:
    path = Path(path_value["path"])
    if _binding(path) != path_value:
        raise ValueError(f"bound source changed: {path}")
    return path


def _source_row(native: list[int], row_index: int) -> dict[str, Any]:
    parsed = parsed_rows(native)
    matches = [item for item in parsed if int(item["index"]) == int(row_index)]
    if len(matches) != 1:
        raise ValueError(f"native row index is not a complete row: {row_index}")
    return matches[0]


def _layout_mask(
    attention_mask: torch.Tensor,
    *,
    target_index: int,
    prompt_length: int,
    suffix_length: int,
    source_start: int | None = None,
    source_end: int | None = None,
    leak: bool = False,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return an additive [batch,1,q,k] causal mask with one target barrier."""
    if attention_mask.ndim != 2:
        raise ValueError("custom barrier requires a 2D native padding mask")
    batch, width = attention_mask.shape
    valid = attention_mask.to(dtype=torch.bool)
    causal = torch.ones((width, width), dtype=torch.bool, device=attention_mask.device).tril()
    allowed = causal.unsqueeze(0).expand(batch, -1, -1).clone()
    # A left-padded query row has no valid causal key.  Keep its own padding
    # path alive; valid queries still cannot attend any padding key.
    allowed &= valid[:, None, :] | ~valid[:, :, None]
    if source_start is not None or source_end is not None:
        if source_start is None or source_end is None or source_start >= source_end:
            raise ValueError("barrier endpoints must be a nonempty half-open interval")
        if not 0 <= target_index < batch:
            raise ValueError("barrier target is outside the native batch")
        target_length = prompt_length + suffix_length
        left_pad = width - target_length
        key_start = left_pad + prompt_length + int(source_start)
        key_stop = left_pad + prompt_length + int(source_end) - 1
        query_start = left_pad + prompt_length + (int(source_end) if leak else int(source_end) - 1)
        if not (0 <= key_start < key_stop <= width and 0 <= query_start <= width):
            raise ValueError("barrier interval is outside padded replay coordinates")
        allowed[target_index, query_start:, key_start:key_stop] = False
    result = torch.zeros((batch, 1, width, width), dtype=dtype, device=attention_mask.device)
    result.masked_fill_(~allowed[:, None], torch.finfo(dtype).min)
    return result


def _allowed(mask: torch.Tensor) -> torch.Tensor:
    """Convert an additive mask back to a CPU boolean layout for validation."""
    if mask.ndim != 4:
        raise ValueError("expected a 4D additive mask")
    return mask[0, 0].detach().cpu() == 0


def selfcheck() -> None:
    padding = torch.ones((1, 20), dtype=torch.long)
    good = _layout_mask(
        padding,
        target_index=0,
        prompt_length=0,
        suffix_length=20,
        source_start=5,
        source_end=10,
    )
    good_layout = _allowed(good)
    assert not bool(good_layout[9, 5])  # q=end-1 is quarantined from row content
    assert bool(good_layout[9, 9])  # the box_end key itself remains legal
    assert not bool(good_layout[10, 5])
    leaky = _layout_mask(
        padding,
        target_index=0,
        prompt_length=0,
        suffix_length=20,
        source_start=5,
        source_end=10,
        leak=True,
    )
    leaky_layout = _allowed(leaky)
    assert bool(leaky_layout[9, 5])  # q>=end omits the required endpoint
    assert not bool(leaky_layout[10, 5])
    print("PASS half-open row-end barrier and leaky endpoint detection")


class Runtime:
    def __init__(self, args: argparse.Namespace, cell: dict[str, Any] | None = None) -> None:
        self.args = args
        self.started = time.monotonic()
        self.plan_path = args.plan
        self.plan = json.loads(args.plan.read_text())
        if self.plan.get("status") != "frozen_before_model_comparisons":
            raise ValueError("shared admission is not frozen")
        if args.mode == "qualify":
            target = [item for item in _cell_specs(self.plan) if item["id"].endswith("tied-14038-failure-before-row8--cut_A")]
            if len(target) != 1:
                raise ValueError("qualification cell is not unique")
            self.cell = target[0]
            self.qualification_attempt = int(getattr(args, "attempt", 1))
            self.job_dir = args.output_root / "qualification" / f"attempt-{self.qualification_attempt}"
        else:
            if cell is None:
                raise ValueError("cell mode needs a cell")
            self.cell = cell
            self.job_dir = args.output_root / "runtime" / str(cell["id"])
        self.job_dir.mkdir(parents=True, exist_ok=False)
        self.launch = {
            "schema": "history_source_sign.launch.v1",
            "status": "launched",
            "mode": args.mode,
            "cell_id": self.cell["id"],
            "qualification_attempt": getattr(self, "qualification_attempt", None),
            "device": str(args.device),
            "pid": os.getpid(),
            "argv": list(sys.argv),
            "plan": _binding(args.plan),
            "producer": _binding(Path(__file__).resolve()),
            "started_unix": time.time(),
            "budget": {
                "max_forwards": int(args.max_forwards),
                "max_seconds": float(args.max_seconds),
                "max_bytes": int(args.max_bytes),
            },
        }
        _write_once(self.job_dir / "launch.json", self.launch)
        self.count = 0
        self.vision_count = 0
        self.bytes = 0
        self.q, self.identity = load_model(self.cell["model"], args.device)
        self.model = self.q.model.eval()
        self.handles: list[Any] = []

        def count_forward(_module: Any, _inputs: tuple[Any, ...]) -> None:
            self.count += 1
            if self.count > int(args.max_forwards):
                raise RuntimeError("model-forward budget exhausted")
            if time.monotonic() - self.started > float(args.max_seconds):
                raise RuntimeError("allocated wall-time budget exhausted")

        def count_vision(_module: Any, _inputs: tuple[Any, ...]) -> None:
            self.vision_count += 1

        self.handles.append(self.model.register_forward_pre_hook(count_forward))
        self.handles.append(self.model.model.visual.register_forward_pre_hook(count_vision))
        self.groups: dict[str, dict[str, Any]] = {}
        self.contexts: dict[str, dict[str, Any]] = {}

    def _artifact(self, path: Path) -> dict[str, Any]:
        binding = _binding(path)
        self.bytes += int(binding["size_bytes"])
        if self.bytes > int(self.args.max_bytes):
            raise RuntimeError("retained payload budget exhausted")
        return binding

    def _load_context(self, cell: dict[str, Any]) -> dict[str, Any]:
        source_id = str(cell["source_boundary_id"])
        if source_id in self.contexts:
            return self.contexts[source_id]
        record = _source_record(self.plan, source_id)
        panel_path = _binding_matches(record["source_panel"])
        panel = json.loads(panel_path.read_text())
        group = _group(panel, str(record["group"]))
        raw_path = _binding_matches(record["raw"])
        trace_path = _binding_matches(record["trace"])
        receipt_path = _binding_matches(record["receipt"])
        raw_obj = json.loads(raw_path.read_text())
        raw = raw_obj.get("rows")
        trace_obj = json.loads(trace_path.read_text())
        receipt = json.loads(receipt_path.read_text())
        if not isinstance(raw, list) or not isinstance(trace_obj.get("steps"), list):
            raise ValueError("native source artifacts have an invalid schema")
        if receipt.get("status") != "candidate_complete":
            raise ValueError("native source receipt is not complete")
        if receipt.get("condition") != f"{record['model']}-original" or receipt.get("group") != record["group"]:
            raise ValueError("native source condition/group changed")
        if receipt.get("raw") != _binding(raw_path) or receipt.get("trace") != _binding(trace_path):
            raise ValueError("native source receipt bindings changed")
        config = dict(panel["configs"][record["model"]])
        config["data"] = {"input_jsonl": group["input_jsonl"]}
        requests, _ = build_bound_native_requests(self.q, config, group["cases"])
        batch = prepare_native_inputs(
            self.q.processor,
            requests,
            device=self.args.device,
            record_media_identity=True,
        )
        if receipt.get("input_identity") != _input_identity(batch):
            raise ValueError("native input identity changed")
        target_index = int(record["batch_index"])
        native = [int(item) for item in raw[target_index]["token_ids"]]
        if _json_hash(native) != record["native_token_sha256"]:
            raise ValueError("native target token identity changed")
        context = {
            "record": record,
            "panel": panel,
            "group": group,
            "raw": raw,
            "trace": trace_obj["steps"],
            "receipt": receipt,
            "batch": batch,
            "target_index": target_index,
            "native": native,
            "prompt_tokens": [list(item) for item in batch.prompt_token_ids],
        }
        self.contexts[source_id] = context
        return context

    def _inputs(
        self,
        context: dict[str, Any],
        target_suffix: list[int],
        *,
        keep: int,
        source_row: dict[str, Any] | None = None,
        leak: bool = False,
    ) -> dict[str, Any]:
        raw = context["raw"]
        batch = context["batch"]
        target_index = int(context["target_index"])
        if not target_suffix:
            raise ValueError("replay suffix cannot be empty")
        suffixes = _prefix_tokens(raw, len(target_suffix), int(self.q.tokenizer.pad_token_id))
        suffixes[target_index] = list(target_suffix)
        histories = [
            prompt + suffix
            for prompt, suffix in zip(context["prompt_tokens"], suffixes, strict=True)
        ]
        inputs = exact_history_inputs(
            self.model,
            batch.inputs,
            histories,
            pad_token_id=int(self.q.tokenizer.pad_token_id),
            logits_to_keep=int(keep),
        )
        if source_row is not None:
            inputs["attention_mask"] = _layout_mask(
                inputs["attention_mask"],
                target_index=target_index,
                prompt_length=len(context["prompt_tokens"][target_index]),
                suffix_length=len(target_suffix),
                source_start=int(source_row["start"]),
                source_end=int(source_row["end"]),
                leak=leak,
                dtype=self.model.model.language_model.embed_tokens.weight.dtype,
            )
        return inputs

    def _replay(
        self,
        context: dict[str, Any],
        target_suffix: list[int],
        *,
        keep_start: int,
        max_token: int,
        source_row: dict[str, Any] | None = None,
        leak: bool = False,
    ) -> torch.Tensor:
        if not 0 <= keep_start <= max_token <= len(target_suffix):
            raise ValueError("replay score interval is outside target suffix")
        inputs = self._inputs(
            context,
            target_suffix,
            keep=max_token - keep_start + 1,
            source_row=source_row,
            leak=leak,
        )
        with torch.inference_mode():
            outputs = self.model(**inputs)
        scores = outputs.logits.detach().float()
        target = scores[int(context["target_index"])]
        if target.ndim != 2 or target.shape[0] != max_token - keep_start + 1:
            raise RuntimeError("compact full-prefix replay shape mismatch")
        if not torch.isfinite(target).all():
            raise RuntimeError("nonfinite full-vocabulary replay logits")
        return target

    @staticmethod
    def _top2(score: torch.Tensor) -> dict[str, Any]:
        top = torch.topk(score, 2)
        return {
            "token_ids": [int(item) for item in top.indices.tolist()],
            "logits": [float(item) for item in top.values.tolist()],
            "margin": float((top.values[0] - top.values[1]).item()),
        }

    def _row_candidate_score(
        self,
        context: dict[str, Any],
        cell: dict[str, Any],
        condition: str,
        candidate: dict[str, Any],
        source_row: dict[str, Any],
    ) -> dict[str, Any]:
        native = context["native"]
        prefix_end = int(cell["prefix_end"])
        row_tokens = [int(item) for item in candidate["tokens"]]
        if len(row_tokens) != int(source_row["end"]) - int(source_row["start"]):
            raise ValueError("candidate replacement is not same-length with source row")
        # The final scored token is predicted from the preceding token; omit
        # its own input token so the compact replay ends at that query.
        target_suffix = native[:prefix_end] + row_tokens[:-1]
        keep_start = prefix_end - 1
        scores = self._replay(
            context,
            target_suffix,
            keep_start=keep_start,
            max_token=len(target_suffix) - 1,
            source_row=None if condition == "native" else source_row,
        )
        logprobs = torch.log_softmax(scores.double(), dim=-1)
        values = [float(logprobs[index, token].item()) for index, token in enumerate(row_tokens)]
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError("candidate row contains a nonfinite token log probability")
        first = scores[0]
        return {
            "id": candidate["id"],
            "owner": candidate["owner"],
            "source": candidate["source"],
            "construction_rule": candidate["construction_rule"],
            "token_ids": row_tokens,
            "token_sha256": candidate["token_sha256"],
            "values": list(candidate["values"]),
            "valid": bool(candidate["valid"]),
            "token_logprobs": values,
            "logprob_sum": float(sum(values)),
            "logprob_includes": "opener_and_terminator",
            "boundary_first_fork": {
                "opener_token_id": int(row_tokens[0]),
                "opener_logprob": values[0],
                "full_logits_sha256": _tensor_hash(first),
            },
        }

    def _candidate_panel(
        self,
        context: dict[str, Any],
        cell: dict[str, Any],
        condition: str,
        source_row: dict[str, Any],
    ) -> dict[str, Any]:
        sets: dict[str, list[dict[str, Any]]] = {}
        first_full: list[float] | None = None
        first_meta: dict[str, Any] | None = None
        for name in ("A", "C", "N"):
            values: list[dict[str, Any]] = []
            for candidate in cell["candidate_sets"][name]:
                result = self._row_candidate_score(context, cell, condition, candidate, source_row)
                if first_full is None:
                    prefix_end = int(cell["prefix_end"])
                    suffix = context["native"][:prefix_end]
                    first_score = self._replay(
                        context,
                        suffix,
                        keep_start=prefix_end - 1,
                        max_token=prefix_end - 1,
                        source_row=None if condition == "native" else source_row,
                    )[0]
                    first_full = _finite(first_score)
                    first_meta = self._top2(first_score)
                values.append(result)
            sets[name] = values
        masses: dict[str, float] = {}
        for name, values in sets.items():
            mass = torch.logsumexp(torch.tensor([item["logprob_sum"] for item in values], dtype=torch.float64), dim=0)
            masses[name] = float(mass.item())
        return {
            "sets": sets,
            "finite_set_log_masses": masses,
            "finite_set_contrasts": {
                "A_minus_N": masses["A"] - masses["N"],
                "C_minus_N": masses["C"] - masses["N"],
                "A_minus_C": masses["A"] - masses["C"],
            },
            "first_candidate_fork": {
                "full_logits": first_full,
                "full_logits_sha256": _json_hash(first_full),
                "top2": first_meta,
                "logit_scope": "full_vocabulary",
            },
        }

    def _release(self, context: dict[str, Any], cell: dict[str, Any], source_row: dict[str, Any]) -> dict[str, Any]:
        condition = str(cell["condition"])
        native = context["native"]
        prefix_end = int(cell["prefix_end"])
        prefix = native[:prefix_end]
        max_tokens = max(0, len(native) - prefix_end)
        generated: list[int] = []
        steps: list[dict[str, Any]] = []
        barrier_active = condition != "native"
        barrier_release_step: int | None = None
        max_rows = 9  # one barrier-active row, then at most eight rebuilt rows
        stop = "source_cap"
        while len(generated) < max_tokens and len(parsed_rows(generated)) < max_rows:
            suffix = prefix + generated
            last = len(suffix) - 1
            scores = self._replay(
                context,
                suffix,
                keep_start=last,
                max_token=last,
                source_row=source_row if barrier_active else None,
            )[0]
            choice = int(torch.argmax(scores).item())
            top = self._top2(scores)
            steps.append({
                "offset": len(generated),
                "barrier_active": barrier_active,
                "top2": top,
                "chosen_token": choice,
            })
            generated.append(choice)
            if choice == EOS:
                stop = "eos"
                break
            complete = len(parsed_rows(generated))
            if barrier_active and complete >= 1:
                barrier_active = False
                barrier_release_step = len(generated)
            if complete >= max_rows:
                stop = "rows"
                break
        if stop == "source_cap" and len(parsed_rows(generated)) >= max_rows:
            stop = "rows"
        first_row = parsed_rows(generated)[0] if parsed_rows(generated) else None
        return {
            "condition": condition,
            "token_ids": generated,
            "token_sha256": _json_hash(generated),
            "text": self.q.tokenizer.decode(
                generated, skip_special_tokens=False, clean_up_tokenization_spaces=False
            ),
            "stop": {
                "reason": stop,
                "token_count": len(generated),
                "complete_rows": len(parsed_rows(generated)),
                "source_cap": max_tokens,
                "row_cap": max_rows,
                "parser": "numerical_feedback.select.rows",
            },
            "first_complete_row": first_row,
            "barrier_release_step": barrier_release_step,
            "steps": steps,
            "state_rebuild": {
                "method": "unmasked full-prefix replay from actual emitted tokens",
                "performed": barrier_release_step is not None,
                "native_uses_same_replay": True,
            },
        }

    def qualify(self) -> dict[str, Any]:
        context = self._load_context(self.cell)
        native = context["native"]
        source_row = _source_row(native, int(self.cell["source_row_index_A"]))
        source_end = int(self.cell["prefix_end"])
        ordinary = self._replay(
            context,
            native[: source_end - 1],
            keep_start=0,
            max_token=source_end - 1,
        )
        noop = self._replay(
            context,
            native[: source_end - 1],
            keep_start=0,
            max_token=source_end - 1,
            source_row=None,
        )
        trace_checks: list[dict[str, Any]] = []
        for offset in range(source_end):
            expected = context["trace"][offset]
            top = self._top2(ordinary[offset])
            expected_ids = [int(expected["raw_winners"][context["target_index"]]), int(expected["raw_runnerups"][context["target_index"]])]
            expected_values = [float(item) for item in expected["raw_top2"][context["target_index"]]]
            errors = [abs(top["logits"][i] - expected_values[i]) for i in range(2)]
            margin = max(abs(expected_values[0] - expected_values[1]), 1e-12)
            trace_checks.append({
                "offset": offset,
                "winner_match": top["token_ids"][0] == expected_ids[0],
                "top2_ids_match": top["token_ids"] == expected_ids,
                "top2_max_abs_error": max(errors),
                "margin": margin,
                "error_over_margin": max(errors) / margin,
            })
        noop_diff = (ordinary - noop).abs()
        trace_pass = all(
            item["top2_ids_match"]
            and item["top2_max_abs_error"] <= ATOL
            and item["error_over_margin"] <= 0.25
            for item in trace_checks
        )
        noop_pass = float(noop_diff.max().item()) <= ATOL

        donor = next(item for item in self.cell["candidate_sets"]["A"] if item["id"] == "A-annotation")
        horizon = min(len(native), source_end + max(12, 2 * (int(source_row["end"]) - int(source_row["start"]))))
        original_suffix = native[:horizon]
        replacement_suffix = (
            native[: int(source_row["start"])]
            + [int(item) for item in donor["tokens"]]
            + native[int(source_row["end"]):horizon]
        )
        keep_start = int(source_row["end"]) - 1
        original_cut = self._replay(
            context,
            original_suffix,
            keep_start=keep_start,
            max_token=horizon - 1,
            source_row=source_row,
        )
        replacement_cut = self._replay(
            context,
            replacement_suffix,
            keep_start=keep_start,
            max_token=horizon - 1,
            source_row=source_row,
        )
        checkpoints = [keep_start] + [
            offset for offset in (int(source_row["end"]), int(source_row["end"]) + 1,
                                   int(source_row["end"]) + 4, int(source_row["end"]) + 8)
            if offset < horizon
        ]
        invariance = []
        for offset in checkpoints:
            index = offset - keep_start
            diff = float((original_cut[index] - replacement_cut[index]).abs().max().item())
            invariance.append({"offset": offset, "max_abs": diff, "passed": diff <= ATOL})
        cut_pass = all(item["passed"] for item in invariance)

        leaky_original = self._replay(
            context,
            original_suffix,
            keep_start=keep_start,
            max_token=horizon - 1,
            source_row=source_row,
            leak=True,
        )
        leaky_replacement = self._replay(
            context,
            replacement_suffix,
            keep_start=keep_start,
            max_token=horizon - 1,
            source_row=source_row,
            leak=True,
        )
        leaky_boundary_diff = float((leaky_original[0] - leaky_replacement[0]).abs().max().item())
        layout = _layout_mask(
            torch.ones((1, len(context["prompt_tokens"][context["target_index"]]) + len(original_suffix)), dtype=torch.long, device=self.args.device),
            target_index=0,
            prompt_length=len(context["prompt_tokens"][context["target_index"]]),
            suffix_length=len(original_suffix),
            source_start=int(source_row["start"]),
            source_end=int(source_row["end"]),
            leak=True,
        )
        prompt_offset = len(context["prompt_tokens"][context["target_index"]])
        leak_rejected = bool(
            _allowed(layout)[
                prompt_offset + int(source_row["end"]) - 1,
                prompt_offset + int(source_row["start"]),
            ]
        )
        leak_pass = leaky_boundary_diff > ATOL and leak_rejected
        passed = bool(trace_pass and noop_pass and cut_pass and leak_pass)
        return {
            "schema": "history_source_sign.qualification.v1",
            "status": "candidate_complete" if passed else "technical_HOLD",
            "passed": passed,
            "cell_id": self.cell["id"],
            "model": self.cell["model"],
            "boundary_id": self.cell["source_boundary_id"],
            "tolerances": {"full_vocab_atol": ATOL, "error_over_margin_max": 0.25},
            "ordinary_2d_vs_source_trace": {
                "passed": trace_pass,
                "max_top2_abs_error": max(item["top2_max_abs_error"] for item in trace_checks),
                "max_error_over_margin": max(item["error_over_margin"] for item in trace_checks),
                "checks": trace_checks,
            },
            "causal_4d_noop": {
                "passed": noop_pass,
                "max_full_vocab_abs_error": float(noop_diff.max().item()),
                "vector_shape": list(noop.shape),
                "ordinary_hash": _tensor_hash(ordinary),
                "noop_hash": _tensor_hash(noop),
            },
            "cut_replacement_invariance": {
                "passed": cut_pass,
                "source_row": source_row,
                "replacement": {
                    "id": donor["id"],
                    "token_sha256": donor["token_sha256"],
                    "same_length": len(donor["tokens"]) == int(source_row["end"]) - int(source_row["start"]),
                },
                "checkpoints": invariance,
            },
            "leaky_mask_rejection": {
                "passed": leak_pass,
                "boundary_max_abs_error": leaky_boundary_diff,
                "rejected_for_endpoint": leak_rejected,
                "leaky_rule": "q >= end",
                "required_rule": "q >= end-1",
            },
            "input_identity": context["receipt"]["input_identity"],
            "source": {
                "raw": _binding(Path(_source_record(self.plan, self.cell["source_boundary_id"])["raw"]["path"])),
                "trace": _binding(Path(_source_record(self.plan, self.cell["source_boundary_id"])["trace"]["path"])),
                "receipt": _binding(Path(_source_record(self.plan, self.cell["source_boundary_id"])["receipt"]["path"])),
            },
        }

    def cell(self) -> dict[str, Any]:
        candidates = sorted(self.args.output_root.glob("qualification/attempt-*/result.json"))
        passed = []
        for path in candidates:
            value = json.loads(path.read_text())
            if value.get("status") == "candidate_complete" and value.get("passed"):
                passed.append((int(path.parent.name.removeprefix("attempt-")), path, value))
        if not passed:
            raise RuntimeError("qualification result is absent or failed; cell is technical HOLD")
        _attempt, qualification_path, qualification = max(passed, key=lambda item: item[0])
        if qualification.get("status") != "candidate_complete" or not qualification.get("passed"):
            raise RuntimeError("qualification did not pass; cell is technical HOLD")
        if qualification.get("boundary_id") != "tied-14038-failure":
            raise RuntimeError("qualification boundary changed")
        context = self._load_context(self.cell)
        native = context["native"]
        source_index = int(self.cell["source_row_index_A"] if self.cell["condition"] != "cut_C" else self.cell["source_row_index_C"])
        source_row = _source_row(native, source_index)
        candidates = self._candidate_panel(context, self.cell, str(self.cell["condition"]), source_row)
        release = self._release(context, self.cell, source_row)
        return {
            "schema": "history_source_sign.cell.v1",
            "status": "candidate_complete",
            "cell_id": self.cell["id"],
            "lane": self.cell["lane"],
            "condition": self.cell["condition"],
            "model": self.cell["model"],
            "boundary_id": self.cell["source_boundary_id"],
            "image_id": int(self.cell["image_id"]),
            "prefix_end": int(self.cell["prefix_end"]),
            "source_row": source_row,
            "native_token_sha256": _json_hash(native),
            "candidate_scores": candidates,
            "release": release,
            "qualification": _binding(qualification_path),
            "source": {
                "raw": _binding(Path(_source_record(self.plan, self.cell["source_boundary_id"])["raw"]["path"])),
                "trace": _binding(Path(_source_record(self.plan, self.cell["source_boundary_id"])["trace"]["path"])),
                "receipt": _binding(Path(_source_record(self.plan, self.cell["source_boundary_id"])["receipt"]["path"])),
            },
        }

    def run(self) -> None:
        result_path = self.job_dir / "result.json"
        receipt_path = self.job_dir / "receipt.json"
        status = "candidate_complete"
        error: str | None = None
        result: dict[str, Any] | None = None
        try:
            result = self.qualify() if self.args.mode == "qualify" else self.cell()
            if self.args.mode == "qualify":
                _write_once(result_path, result)
                self.bytes += result_path.stat().st_size
            else:
                _write_once(self.job_dir / "release.json", result)
                self.bytes += (self.job_dir / "release.json").stat().st_size
            if self.bytes > int(self.args.max_bytes):
                raise RuntimeError("retained payload budget exhausted")
        except BaseException as exc:
            status = "technical_HOLD"
            error = repr(exc)
        finally:
            for handle in self.handles:
                handle.remove()
            versions = {
                name: parameter._version for name, parameter in self.model.named_parameters()
            }
            if hasattr(self, "initial_versions") and versions != self.initial_versions:
                status = "technical_HOLD"
                error = error or "parameter mutation detected"
            receipt = {
                "schema": "history_source_sign.receipt.v1",
                "status": status,
                "mode": self.args.mode,
                "cell_id": self.cell["id"],
                "model": self.cell["model"],
                "device": str(self.args.device),
                "plan": _binding(self.plan_path),
                "producer": _binding(Path(__file__).resolve()),
                "identity": self.identity,
                "model_forwards": self.count,
                "vision_forwards": self.vision_count,
                "free_cells": 0 if self.args.mode == "qualify" else 1,
                "retained_tensor_bytes": self.bytes,
                "elapsed_gpu_seconds": time.monotonic() - self.started,
                "ended_unix": time.time(),
            }
            if result is not None:
                receipt["result"] = _binding(result_path if self.args.mode == "qualify" else self.job_dir / "release.json")
            if error is not None:
                receipt["error"] = error
            _write_once(receipt_path, receipt)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("qualify", "cell"))
    parser.add_argument("--cell-id")
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--output-root", type=Path, default=ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-forwards", type=int, default=40000)
    parser.add_argument("--max-seconds", type=float, default=4 * 3600)
    parser.add_argument("--max-bytes", type=int, default=16 * 1024**3)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
        return
    if args.mode is None:
        parser.error("--mode is required unless --selfcheck is used")
    cells = _cell_specs(json.loads(args.plan.read_text()))
    selected = None
    if args.mode == "cell":
        if not args.cell_id:
            parser.error("--cell-id is required in cell mode")
        matches = [item for item in cells if item["id"] == args.cell_id]
        if len(matches) != 1:
            parser.error(f"unknown cell ID: {args.cell_id}")
        selected = matches[0]
    runtime: Runtime | None = None
    try:
        runtime = Runtime(args, selected)
        runtime.initial_versions = {
            name: parameter._version for name, parameter in runtime.model.named_parameters()
        }
        runtime.run()
    except BaseException as exc:
        if runtime is None:
            raise
        raise RuntimeError(f"runtime failed before receipt: {exc}") from exc


if __name__ == "__main__":
    main()
