"""Native full-vocabulary capture for the frozen recurrence phase unit.

This is a thin replay caller.  The phase plan and source binding stay owned by
``common``/``prepare``; this module only loads the selected native model,
captures registered causal rows, and writes immutable shard artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

from probes.training_set_completion.readout_norm_fresh import _binding, _input_identity
from probes.training_set_completion.recurrence_phase_decision.common import (
    bound_source,
    prefix as verify_prefix,
    read_plan,
)
from probes.training_set_completion.recurrence_phase_decision.prepare import PANEL
from probes.training_set_completion.untied_shared import load_model
from probes.training_set_completion.numerical_feedback.select import rows as parsed_rows
import src.qwen.native as native_module
from src.qwen.native import exact_history_inputs


COORD_BASE = 151670
COORD_COUNT = 1000
EOS = 151645
ROLES = ("x1", "y1", "x2", "y2")
PARITY_ATOL = 2e-4
DEFAULT_PLAN = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-19-recurrence-phase-decision/selection/plan.json"
)
DEFAULT_WINDOWS = DEFAULT_PLAN.with_name("windows.json")


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _tensor_hash(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._+-" else "_" for char in value)


def _write_json_new(path: Path, value: Any) -> int:
    data = json.dumps(value, indent=2, allow_nan=False).encode() + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    return len(data)


def _write_json(path: Path, value: Any) -> int:
    data = json.dumps(value, indent=2, allow_nan=False).encode() + b"\n"
    with path.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    return len(data)


def _stage_slots(stage: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the frozen stage positions in causal target order."""
    slots: list[dict[str, Any]] = [{"name": "opener", "offset": int(stage["opener"])}]
    slots.extend(
        {"name": f"description_{index}", "offset": int(offset)}
        for index, offset in enumerate(stage.get("description", []))
    )
    slots.extend(
        [
            {"name": "description_end", "offset": int(stage["description_end"])},
            {"name": "bbox_start", "offset": int(stage["bbox_start"])},
        ]
    )
    slots.extend(
        {"name": role, "offset": int(stage["coordinates"][role])} for role in ROLES
    )
    slots.append({"name": "bbox_end", "offset": int(stage["bbox_end"])})
    if stage.get("next_boundary") is not None:
        slots.append({"name": "next_boundary", "offset": int(stage["next_boundary"])})
    offsets = [item["offset"] for item in slots]
    if offsets != sorted(set(offsets)):
        raise ValueError(f"stage slots are not increasing and unique: {stage.get('row_index')}")
    return slots


def _validate_stage(native: list[int], profile: dict[str, Any]) -> list[dict[str, Any]]:
    stage = profile["stage"]
    row = profile["row"]
    if int(stage["row_index"]) != int(profile["row_index"]):
        raise ValueError("profile stage row differs from frozen row")
    slots = _stage_slots(stage)
    for item in slots:
        offset = item["offset"]
        if not 0 <= offset < len(native):
            raise ValueError(f"stage slot outside native tokens: {offset}")
    if native[int(stage["opener"])] != 151646:
        raise ValueError("native opener token changed")
    if native[int(stage["description_end"])] != 151647:
        raise ValueError("native description delimiter changed")
    if native[int(stage["bbox_start"])] != 151648:
        raise ValueError("native bbox delimiter changed")
    if native[int(stage["bbox_end"])] != 151649:
        raise ValueError("native bbox end changed")
    values = list(row["values"])
    for role, value in zip(ROLES, values, strict=True):
        offset = int(stage["coordinates"][role])
        if native[offset] != COORD_BASE + int(value):
            raise ValueError(f"native {role} token changed at {offset}")
    # The frozen next-boundary slot can be another opener or terminal EOS;
    # preserve whichever literal token the source actually consumed.
    return slots


def _slot_physical_index(
    *,
    prompt_length: int,
    left_pad: int,
    target_offset: int,
    first_target_offset: int,
) -> tuple[int, int]:
    """Map an action target to its consumed input row and compact logit row."""
    physical = left_pad + prompt_length + target_offset - 1
    compact = target_offset - first_target_offset
    if physical < 0 or compact < 0:
        raise ValueError("target offset does not have a preceding causal input")
    return physical, compact


def _trace_top2(trace: dict[str, Any], batch_index: int, offset: int) -> dict[str, Any] | None:
    steps = trace.get("steps")
    if not isinstance(steps, list) or not 0 <= offset < len(steps):
        return None
    step = steps[offset]
    try:
        winner = int(step["raw_winners"][batch_index])
        values = step["raw_top2"][batch_index]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"source trace lacks comparable offset {offset}") from exc
    if len(values) != 2:
        raise ValueError("source trace raw_top2 must have two entries")
    if isinstance(values[0], (list, tuple)):
        expected = [[int(value[0]), float(value[1])] for value in values]
    else:
        runner = int(step["raw_runnerups"][batch_index])
        expected = [[winner, float(values[0])], [runner, float(values[1])]]
    return {"winner": winner, "top2": expected}


def _parity(score: torch.Tensor, expected: dict[str, Any] | None) -> dict[str, Any] | None:
    if expected is None:
        return None
    top = torch.topk(score, 2)
    observed_ids = [int(value) for value in top.indices.tolist()]
    observed_values = [float(value) for value in top.values.tolist()]
    errors = [
        abs(observed_values[index] - float(expected["top2"][index][1])) for index in range(2)
    ]
    return {
        "winner": observed_ids[0],
        "expected_winner": int(expected["winner"]),
        "winner_match": observed_ids[0] == int(expected["winner"]),
        "top2_max_abs_error": max(errors),
        "tolerance": PARITY_ATOL,
        "passed": observed_ids[0] == int(expected["winner"]) and max(errors) <= PARITY_ATOL,
    }


def _source_manifests(plan: dict[str, Any], boundary_ids: Iterable[str]) -> list[dict[str, Any]]:
    wanted = set(boundary_ids)
    entries: list[dict[str, Any]] = []
    for summary in plan["source_summaries"]:
        if summary["boundary_id"] not in wanted:
            continue
        sources: dict[str, Any] = {}
        for key in ("raw", "trace", "receipt"):
            declared = summary["source"][key]
            actual = _binding(Path(declared["path"]))
            if actual["sha256"] != declared["sha256"] or actual["size_bytes"] != declared["size_bytes"]:
                raise ValueError(f"source {key} binding changed: {summary['boundary_id']}")
            sources[key] = actual
        source_panel = summary["source_panel"]
        actual_panel = _binding(Path(source_panel["path"]))
        if actual_panel["sha256"] != source_panel["sha256"]:
            raise ValueError(f"source panel binding changed: {summary['boundary_id']}")
        entries.append(
            {
                "boundary_id": summary["boundary_id"],
                "source": sources,
                "source_panel": actual_panel,
            }
        )
    if {item["boundary_id"] for item in entries} != wanted:
        raise ValueError("source manifest does not cover requested boundaries")
    return entries


def _read_ids_file(path: Path) -> list[str]:
    value = json.loads(path.read_text())
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError("--boundary-ids-file must contain a JSON list of strings")
    if len(set(value)) != len(value):
        raise ValueError("--boundary-ids-file contains duplicate boundary IDs")
    return value


def _requested_ids(args: argparse.Namespace, panel: dict[str, Any]) -> list[str]:
    panel_by_id = {str(item["id"]): item for item in panel["all_boundaries"]}
    values: list[str] = []
    values.extend(args.boundary_ids)
    values.extend(args.boundary_id)
    if args.boundary_ids_file is not None:
        values.extend(_read_ids_file(args.boundary_ids_file))
    if len(set(values)) != len(values):
        raise ValueError("requested boundary IDs contain duplicates")
    if not values:
        return [str(item["id"]) for item in panel["all_boundaries"] if item["model"] == args.model]
    unknown = [value for value in values if value not in panel_by_id]
    if unknown:
        raise ValueError(f"unknown boundary IDs: {unknown}")
    wrong_model = [value for value in values if panel_by_id[value]["model"] != args.model]
    if wrong_model:
        raise ValueError(f"boundary IDs belong to another model: {wrong_model}")
    return values


def _variant_groups(plan: dict[str, Any], boundary_id: str) -> dict[tuple[str, str], dict[str, Any]]:
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for cell in plan["cells"]:
        if cell["boundary_id"] != boundary_id or cell["status"] != "ready":
            continue
        key = (str(cell["phase"]), str(cell["release_mode"]))
        group = groups.setdefault(key, {"cells": {}, "all_cells": []})
        variant = "native" if int(cell["delta"]) == 0 else f"{cell['role']}{int(cell['delta']):+d}"
        group["cells"].setdefault(variant, []).append(cell)
        group["all_cells"].append(cell)
    return groups


def _release_slots(
    native_rows: list[dict[str, Any]], native_length: int, phase: dict[str, Any], mode: str
) -> tuple[int, int | None, list[dict[str, Any]], str | None]:
    row_index = int(phase["row_index"])
    if mode == "immediate":
        prefix_end = int(phase["immediate_prefix_end"])
        target_index = row_index + 1
    else:
        declared = phase.get("bridge_row_indices")
        if not declared or phase.get("bridge_prefix_end") is None:
            raise ValueError("bridge release is held without two complete rows")
        prefix_end = int(phase["bridge_prefix_end"])
        target_index = int(declared[-1]) + 1
    matches = [row for row in native_rows if int(row["index"]) == target_index]
    if len(matches) != 1:
        # A complete bridge can end at the native EOS without a following
        # complete row.  Keep the first unforced native token and disclose the
        # absent coordinate suffix instead of manufacturing a target row.
        if prefix_end < native_length:
            return (
                prefix_end,
                None,
                [{"name": "first_unforced_token", "offset": prefix_end}],
                "next_complete_row_missing",
            )
        raise ValueError(f"release target row is missing: {target_index}")
    row = matches[0]
    if int(row["start"]) != prefix_end:
        raise ValueError("release prefix does not end at the next row opener")
    slots = [{"name": "first_unforced_opener", "offset": int(row["start"])}]
    slots.extend(
        {"name": f"next_row_{role}", "offset": int(offset)}
        for role, offset in zip(ROLES, row["coordinate_offsets"], strict=True)
    )
    return prefix_end, target_index, slots, None


class CaptureRuntime:
    def __init__(
        self,
        *,
        args: argparse.Namespace,
        plan: dict[str, Any],
        panel: dict[str, Any],
        boundary_ids: list[str],
        windows_binding: dict[str, Any],
    ) -> None:
        self.args = args
        self.plan = plan
        self.panel = panel
        self.boundary_ids = boundary_ids
        self.windows_binding = windows_binding
        self.output = args.output
        self.started = time.monotonic()
        self.forward_count = 0
        self.saved_bytes = 0
        self.pad_token_id: int | None = None
        self.artifacts: list[dict[str, Any]] = []
        self.handles: list[Any] = []
        self.last_head: torch.Tensor | None = None
        self.receipt: dict[str, Any] = {
            "schema": "recurrence_phase_decision.capture_receipt.v1",
            "status": "pre_model_snapshot",
            "pid": os.getpid(),
            "model": args.model,
            "device": str(args.device),
            "boundary_ids": list(boundary_ids),
            "plan": _binding(args.plan),
            "panel": _binding(args.panel),
            "windows": windows_binding,
            "producer": _binding(Path(__file__).resolve()),
            "max_forwards": int(args.max_forwards),
            "max_seconds": float(args.max_seconds),
            "max_bytes": int(args.max_bytes),
            "model_forwards": 0,
            "artifact_bytes": 0,
            "artifacts": self.artifacts,
        }

    def persist_receipt(self, status: str | None = None, error: str | None = None) -> None:
        self.receipt["model_forwards"] = self.forward_count
        self.receipt["artifact_bytes"] = self.saved_bytes
        self.receipt["elapsed_seconds"] = time.monotonic() - self.started
        if status is not None:
            self.receipt["status"] = status
        if error is not None:
            self.receipt["error"] = error
        _write_json(self.output / "receipt.json", self.receipt)

    def check_budget(self) -> None:
        if self.forward_count > int(self.args.max_forwards):
            raise RuntimeError("per-shard model-forward budget exhausted")
        if time.monotonic() - self.started > float(self.args.max_seconds):
            raise RuntimeError("per-shard wall-time budget exhausted")
        if self.saved_bytes > int(self.args.max_bytes):
            raise RuntimeError("per-shard artifact-byte budget exhausted")

    def save_tensor_payload(self, path: Path, payload: dict[str, Any], descriptor: dict[str, Any]) -> None:
        self.check_budget()
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        data = buffer.getbuffer()
        if self.saved_bytes + len(data) > int(self.args.max_bytes):
            raise RuntimeError("per-shard artifact-byte budget exhausted before write")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        self.saved_bytes += len(data)
        item = dict(descriptor)
        item.update({"path": str(path), "size_bytes": len(data)})
        self.artifacts.append(item)
        self.persist_receipt()

    def _count_forward(self, _module: Any, _inputs: tuple[Any, ...]) -> None:
        self.forward_count += 1
        self.check_budget()

    def _head_hook(self, _module: Any, inputs: tuple[Any, ...]) -> None:
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            raise TypeError("lm-head pre-hook did not receive a hidden tensor")
        self.last_head = inputs[0].detach().float().cpu().clone()

    def install_hooks(self, model: Any, head: Any) -> None:
        self.handles = [
            model.register_forward_pre_hook(self._count_forward),
            head.register_forward_pre_hook(self._head_hook),
        ]

    def remove_hooks(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    @staticmethod
    def _histories(
        batch: Any,
        raw: list[dict[str, Any]],
        target_index: int,
        history_end: int,
        mutation: tuple[int, int, int] | None,
    ) -> tuple[list[list[int]], dict[str, Any]]:
        prompts = [list(row) for row in batch.prompt_token_ids]
        histories: list[list[int]] = []
        for index, item in enumerate(raw):
            action = [int(value) for value in item["token_ids"][:history_end]]
            if index == target_index and mutation is not None:
                offset, old, new = mutation
                if not 0 <= offset < history_end or action[offset] != old:
                    raise ValueError("edited role is outside or differs from consumed prefix")
                action[offset] = new
            histories.append(prompts[index] + action)
        if any(not history for history in histories):
            raise ValueError("exact native history is empty")
        target = histories[target_index]
        return histories, {
            "prompt_tokens": prompts[target_index],
            "prefix_tokens": target[len(prompts[target_index]) :],
            "target_history_tokens": target,
            "history_end_exclusive": history_end,
            "mutation": None
            if mutation is None
            else {"offset": mutation[0], "old_token_id": mutation[1], "new_token_id": mutation[2]},
        }

    def run_scores(
        self,
        *,
        model: Any,
        batch: Any,
        raw: list[dict[str, Any]],
        boundary: dict[str, Any],
        trace: dict[str, Any],
        target_offsets: list[int],
        history_end: int,
        mutation: tuple[int, int, int] | None,
        parity: bool,
        singleton_check: bool,
    ) -> dict[str, Any]:
        offsets = sorted(set(int(offset) for offset in target_offsets))
        if not offsets or max(offsets) > history_end:
            raise ValueError("target offsets must be below the consumed history end")
        target_index = int(boundary["batch_index"])
        histories, context = self._histories(batch, raw, target_index, history_end, mutation)
        keep = max(offsets) - min(offsets) + 1
        inputs = exact_history_inputs(
            model,
            batch.inputs,
            histories,
            pad_token_id=int(self.pad_token_id),
            logits_to_keep=keep,
        )
        self.last_head = None
        with torch.inference_mode():
            output = model(**inputs)
        logits = output.logits.detach().float().cpu()
        if logits.ndim != 3 or logits.shape[1] != keep:
            raise RuntimeError("compact full-vocabulary replay shape is wrong")
        if not torch.isfinite(logits).all():
            raise RuntimeError("native replay produced nonfinite vocabulary scores")
        if self.last_head is None or self.last_head.shape[:2] != logits.shape[:2]:
            raise RuntimeError("lm-head input and compact logits are misaligned")
        head = self.last_head
        if not torch.isfinite(head).all():
            raise RuntimeError("native replay produced nonfinite lm-head input")
        input_ids = inputs["input_ids"].detach().cpu()
        attention = inputs["attention_mask"].detach().cpu()
        positions = inputs.get("position_ids")
        if not isinstance(positions, torch.Tensor):
            raise RuntimeError("exact replay did not return position IDs")
        positions = positions.detach().cpu()
        target_history_length = len(histories[target_index])
        left_pad = int(input_ids.shape[1]) - target_history_length
        prompt_length = len(batch.prompt_token_ids[target_index])
        slots: list[dict[str, Any]] = []
        all_passed = True
        for offset in offsets:
            physical, compact = _slot_physical_index(
                prompt_length=prompt_length,
                left_pad=left_pad,
                target_offset=offset,
                first_target_offset=min(offsets),
            )
            if compact >= logits.shape[1] or physical >= input_ids.shape[1]:
                raise RuntimeError("causal slot alignment is outside replay tensors")
            expected = _trace_top2(trace, target_index, offset) if parity else None
            check = _parity(logits[target_index, compact], expected)
            if check is not None:
                all_passed = all_passed and bool(check["passed"])
            slots.append(
                {
                    "offset": offset,
                    "compact_logit_index": compact,
                    "physical_consumed_input_index": physical,
                    "consumed_input_token_id": int(input_ids[target_index, physical]),
                    "target_token_id": int(boundary["native_tokens"][offset]),
                    "position_ids_at_consumed_input": positions[:, target_index, physical].clone(),
                    "logits": logits[target_index, compact].clone(),
                    "head_input": head[target_index, compact].clone(),
                    "trace_parity": check,
                }
            )
        if parity and not all_passed:
            raise RuntimeError("native replay differs from saved source trace")

        single: dict[str, Any] | None = None
        if singleton_check:
            singleton = self.run_scores(
                model=model,
                batch=batch,
                raw=raw,
                boundary=boundary,
                trace=trace,
                target_offsets=[max(offsets)],
                history_end=history_end,
                mutation=mutation,
                parity=parity,
                singleton_check=False,
            )
            grouped = next(item for item in slots if item["offset"] == max(offsets))
            single_slot = singleton["slots"][0]
            grouped_top = torch.topk(grouped["logits"], 2)
            single_top = torch.topk(single_slot["logits"], 2)
            error = max(
                abs(float(grouped_top.values[index]) - float(single_top.values[index]))
                for index in range(2)
            )
            observed = int(single_slot["compact_logit_index"])
            last_nonpad = int(attention[target_index].nonzero()[-1].item())
            if observed != 0 or last_nonpad != int(single_slot["physical_consumed_input_index"]):
                raise RuntimeError("logits_to_keep=1 did not select the last consumed input")
            if error > PARITY_ATOL or int(grouped_top.indices[0]) != int(single_top.indices[0]):
                raise RuntimeError("singleton and grouped native replay disagree")
            single = {
                "target_offset": max(offsets),
                "logits_to_keep": 1,
                "last_nonpad_input_index": last_nonpad,
                "last_consumed_token_id": int(input_ids[target_index, last_nonpad]),
                "top2_max_abs_error_vs_grouped": error,
                "winner_match_vs_grouped": int(grouped_top.indices[0]) == int(single_top.indices[0]),
                "trace_parity": single_slot["trace_parity"],
            }
        context.update(
            {
                "input_ids": input_ids[target_index].clone(),
                "attention_mask": attention[target_index].clone(),
                "position_ids": positions[:, target_index].clone(),
                "padded_history_width": int(input_ids.shape[1]),
                "left_pad": left_pad,
                "prompt_length": prompt_length,
                "target_offsets": offsets,
                "logits_to_keep": keep,
                "target_index": target_index,
                "prompt_token_hash": _digest(context["prompt_tokens"]),
                "prefix_token_hash": _digest(context["prefix_tokens"]),
                "target_history_hash": _digest(context["target_history_tokens"]),
            }
        )
        return {"context": context, "slots": slots, "singleton_check": single}

    def weights(self, model: Any, identity: dict[str, Any]) -> dict[str, Any]:
        embeddings = model.get_input_embeddings()
        head = model.get_output_embeddings()
        ids = head.selected_token_ids.detach()
        if ids[4:].tolist() != list(range(COORD_BASE, COORD_BASE + COORD_COUNT)):
            raise ValueError("runtime coordinate token IDs are not the frozen 1000-bin family")
        input_rows = embeddings(ids).detach().float().cpu()[4:]
        output_rows = (
            head.base.weight[ids].detach() + head.shared_embed_delta.detach()
        ).float().cpu()[4:]
        return {
            "schema": "recurrence_phase_decision.coordinate_weights.v1",
            "model": self.args.model,
            "coordinate_ids": ids[4:].cpu(),
            "input_rows": input_rows,
            "output_rows": output_rows,
            "input_rows_sha256": _tensor_hash(input_rows),
            "output_rows_sha256": _tensor_hash(output_rows),
            "model_identity": identity,
        }

    def profile_boundary(
        self,
        *,
        model: Any,
        q: Any,
        identity: dict[str, Any],
        boundary_id: str,
        batch: Any,
        raw: list[dict[str, Any]],
        trace: dict[str, Any],
        meta: dict[str, Any],
    ) -> None:
        boundary = meta["boundary"]
        summary = meta["summary"]
        native = [int(value) for value in boundary["native_tokens"]]
        rows: list[dict[str, Any]] = []
        for profile in summary["profile_rows"]:
            if profile["status"] != "available":
                rows.append({"row_index": int(profile["row_index"]), "status": "missing"})
                continue
            slots = _validate_stage(native, profile)
            offsets = [item["offset"] for item in slots]
            result = self.run_scores(
                model=model,
                batch=batch,
                raw=raw,
                boundary=boundary,
                trace=trace,
                target_offsets=offsets,
                history_end=max(offsets),
                mutation=None,
                parity=True,
                singleton_check=True,
            )
            rows.append(
                {
                    "row_index": int(profile["row_index"]),
                    "status": "captured",
                    "row": profile["row"],
                    "stage": profile["stage"],
                    "registered_slots": slots,
                    "capture": result,
                }
            )
        payload = {
            "schema": "recurrence_phase_decision.profile.v1",
            "status": "candidate_complete",
            "boundary_id": boundary_id,
            "model": self.args.model,
            "image_id": int(boundary["image_id"]),
            "kind": boundary["kind"],
            "plan": _binding(self.args.plan),
            "panel": _binding(self.args.panel),
            "source": meta["summary"]["source"],
            "source_panel": _binding(self.args.panel),
            "native_source_panel": meta["summary"]["source_panel"],
            "windows": self.windows_binding,
            "input_identity": _input_identity(batch),
            "native_tokens": native,
            "native_token_hash": _digest(native),
            "model_identity": identity,
            "rows": rows,
        }
        path = self.output / "profiles" / f"{_safe_name(boundary_id)}.pt"
        self.save_tensor_payload(
            path,
            payload,
            {"kind": "profile", "boundary_id": boundary_id, "row_count": len(rows)},
        )

    def release_boundary(
        self,
        *,
        model: Any,
        q: Any,
        identity: dict[str, Any],
        boundary_id: str,
        batch: Any,
        raw: list[dict[str, Any]],
        trace: dict[str, Any],
        meta: dict[str, Any],
    ) -> None:
        boundary = meta["boundary"]
        native = [int(value) for value in boundary["native_tokens"]]
        native_rows = parsed_rows(native)
        groups = _variant_groups(self.plan, boundary_id)
        index: list[dict[str, Any]] = []
        phase_order = {"seed": 0, "first_repeat": 1, "third_row": 2, "proxy": 3}
        for (phase_name, mode), group in sorted(
            groups.items(), key=lambda item: (phase_order.get(item[0][0], 9), item[0][1])
        ):
            phase = next(
                item for item in meta["summary"]["phases"] if item["name"] == phase_name
            )
            prefix_end, target_row_index, slots, missing_suffix = _release_slots(
                native_rows, len(native), phase, mode
            )
            offsets = [item["offset"] for item in slots]
            variants: list[dict[str, Any]] = []
            variant_order = ("native", "x1-1", "x1+1", "y2-1", "y2+1")
            for variant in variant_order:
                cells = group["cells"].get(variant)
                if not cells:
                    continue
                cell = cells[0]
                mutation = None
                site = None
                if variant != "native":
                    site = (
                        int(cell["site_offset"]),
                        int(cell["old_token_id"]),
                        int(cell["new_token_id"]),
                    )
                    mutation = site
                # Verify the same role-specific prefix through the shared helper
                # before deriving exact MRoPE inputs for the actual forward.
                verify_prefix(batch, raw, boundary, prefix_end, q, self.args.device, site)
                result = self.run_scores(
                    model=model,
                    batch=batch,
                    raw=raw,
                    boundary=boundary,
                    trace=trace,
                    target_offsets=offsets,
                    history_end=max(offsets),
                    mutation=mutation,
                    parity=variant == "native",
                    singleton_check=False,
                )
                path_variant = _safe_name(variant)
                path = (
                    self.output
                    / "releases"
                    / f"{_safe_name(boundary_id)}__{_safe_name(phase_name)}__{mode}__{path_variant}.pt"
                )
                payload = {
                    "schema": "recurrence_phase_decision.release.v1",
                    "status": "candidate_complete",
                    "boundary_id": boundary_id,
                    "model": self.args.model,
                    "kind": boundary["kind"],
                    "phase": phase_name,
                    "phase_row_index": int(phase["row_index"]),
                    "release_mode": mode,
                    "plan": _binding(self.args.plan),
                    "panel": _binding(self.args.panel),
                    "prefix_end_exclusive": prefix_end,
                    "target_row_index": target_row_index,
                    "row_index": -1 if target_row_index is None else target_row_index,
                    "missing_suffix": missing_suffix,
                    "registered_slots": slots,
                    "variant": variant,
                    "cell_ids": [item["id"] for item in cells],
                    "cell_records": cells,
                    "source": meta["summary"]["source"],
                    "source_panel": _binding(self.args.panel),
                    "native_source_panel": meta["summary"]["source_panel"],
                    "windows": self.windows_binding,
                    "input_identity": _input_identity(batch),
                    "native_tokens": native,
                    "native_token_hash": _digest(native),
                    "model_identity": identity,
                    "capture": result,
                }
                self.save_tensor_payload(
                    path,
                    payload,
                    {
                        "kind": "release",
                        "boundary_id": boundary_id,
                        "phase": phase_name,
                        "release_mode": mode,
                        "variant": variant,
                        "cell_ids": [item["id"] for item in cells],
                    },
                )
                variants.append(
                    {
                        "variant": variant,
                        "cell_ids": [item["id"] for item in cells],
                        "path": str(path),
                    }
                )
            index.append(
                {
                    "boundary_id": boundary_id,
                    "phase": phase_name,
                    "release_mode": mode,
                    "prefix_end_exclusive": prefix_end,
                    "target_row_index": target_row_index,
                    "missing_suffix": missing_suffix,
                    "variants": variants,
                }
            )
        if index:
            self.receipt.setdefault("release_groups", []).extend(index)
            self.persist_receipt()


def _snapshot(
    *,
    output: Path,
    args: argparse.Namespace,
    plan: dict[str, Any],
    panel: dict[str, Any],
    boundary_ids: list[str],
    windows_binding: dict[str, Any],
) -> dict[str, Any]:
    source_entries = _source_manifests(plan, boundary_ids)
    snapshot = {
        "schema": "recurrence_phase_decision.capture_snapshot.v1",
        "status": "frozen_before_model_load",
        "model": args.model,
        "device": str(args.device),
        "boundary_ids": list(boundary_ids),
        "plan": _binding(args.plan),
        "panel": _binding(args.panel),
        "windows": windows_binding,
        "producer": _binding(Path(__file__).resolve()),
        "common": _binding(Path(__file__).resolve().with_name("common.py")),
        "prepare": _binding(Path(__file__).resolve().with_name("prepare.py")),
        "loader": _binding(Path(__file__).resolve().parents[1] / "untied_shared.py"),
        "native": _binding(Path(native_module.__file__)),
        "source_artifacts": source_entries,
        "boundary_ids_file": None
        if args.boundary_ids_file is None
        else _binding(args.boundary_ids_file),
        "selected_release_cells": [
            item["id"]
            for item in plan["cells"]
            if item["boundary_id"] in set(boundary_ids) and item["model"] == args.model
        ],
        "panel_boundary_count": len(panel["all_boundaries"]),
    }
    _write_json_new(output / "snapshot.json", snapshot)
    return snapshot


def _validate_windows(path: Path, plan_path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if value.get("schema") != "recurrence_phase_decision.windows.v1":
        raise ValueError("coordinate-window sidecar schema changed")
    if value.get("status") != "frozen_before_broad_capture":
        raise ValueError("coordinate-window sidecar is not frozen")
    if value.get("plan", {}).get("sha256") != _binding(plan_path)["sha256"]:
        # The plan path is fixed by read_plan; keep this comparison explicit so
        # a sidecar from another phase cannot silently enter a shard.
        raise ValueError("coordinate-window sidecar plan binding changed")
    if not isinstance(value.get("slots"), list):
        raise ValueError("coordinate-window sidecar has no slots")
    return _binding(path)


def run(args: argparse.Namespace) -> None:
    args.plan = args.plan.resolve()
    args.panel = args.panel.resolve()
    args.windows = args.windows.resolve()
    args.output = args.output.resolve()
    if args.boundary_ids_file is not None:
        args.boundary_ids_file = args.boundary_ids_file.resolve()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite shard output: {args.output}")
    plan = read_plan(args.plan)
    panel = json.loads(args.panel.read_text())
    if _binding(args.panel)["sha256"] != plan["panel"]["sha256"]:
        raise ValueError("shared panel binding differs from frozen plan")
    boundary_ids = _requested_ids(args, panel)
    if not boundary_ids:
        raise ValueError("no boundaries selected")
    windows_binding = _validate_windows(args.windows, args.plan)
    args.output.mkdir(parents=True, exist_ok=False)
    _snapshot(
        output=args.output,
        args=args,
        plan=plan,
        panel=panel,
        boundary_ids=boundary_ids,
        windows_binding=windows_binding,
    )
    runtime = CaptureRuntime(
        args=args,
        plan=plan,
        panel=panel,
        boundary_ids=boundary_ids,
        windows_binding=windows_binding,
    )
    runtime.receipt["snapshot"] = _binding(args.output / "snapshot.json")
    runtime.persist_receipt()
    try:
        runtime.check_budget()
        q, identity = load_model(args.model, torch.device(args.device))
        runtime.pad_token_id = int(q.tokenizer.pad_token_id)
        runtime.receipt["runtime_pad_token_id"] = runtime.pad_token_id
        tokenizer_ids: dict[str, Any] = {"eos": int(q.tokenizer.eos_token_id), "pad": runtime.pad_token_id}
        if tokenizer_ids["eos"] != EOS:
            raise ValueError(f"runtime EOS ID changed: {tokenizer_ids['eos']}")
        for token_id in (EOS, 151646, 151647, 151648, 151649, COORD_BASE, COORD_BASE + COORD_COUNT - 1):
            token = q.tokenizer.convert_ids_to_tokens(token_id)
            if q.tokenizer.convert_tokens_to_ids(token) != token_id:
                raise ValueError(f"tokenizer ID roundtrip changed at {token_id}: {token}")
        runtime.receipt["tokenizer_ids"] = tokenizer_ids
        model = q.model.eval()
        head = model.get_output_embeddings()
        runtime.install_hooks(model, head)
        runtime.receipt["model_identity"] = identity
        weights = runtime.weights(model, identity)
        runtime.save_tensor_payload(
            args.output / "weights.pt",
            weights,
            {"kind": "coordinate_weights", "coordinate_count": COORD_COUNT},
        )
        summary_by_id = {item["boundary_id"]: item for item in plan["source_summaries"]}
        for boundary_id in boundary_ids:
            runtime.check_budget()
            batch, raw, trace, meta = bound_source(
                plan, panel, boundary_id, q, torch.device(args.device)
            )
            if meta["summary"] != summary_by_id[boundary_id]:
                raise ValueError("bound source summary differs from frozen plan")
            runtime.profile_boundary(
                model=model,
                q=q,
                identity=identity,
                boundary_id=boundary_id,
                batch=batch,
                raw=raw,
                trace=trace,
                meta=meta,
            )
            if any(
                cell["boundary_id"] == boundary_id and cell["status"] == "ready"
                for cell in plan["cells"]
            ):
                runtime.release_boundary(
                    model=model,
                    q=q,
                    identity=identity,
                    boundary_id=boundary_id,
                    batch=batch,
                    raw=raw,
                    trace=trace,
                    meta=meta,
                )
        _write_json_new(
            args.output / "artifact-index.json",
            {
                "schema": "recurrence_phase_decision.capture_index.v1",
                "status": "candidate_complete",
                "model": args.model,
                "boundary_ids": boundary_ids,
                "artifacts": runtime.artifacts,
            },
        )
        runtime.persist_receipt("candidate_complete")
    except BaseException as exc:
        runtime.persist_receipt("technical_invalid", repr(exc))
        raise
    finally:
        runtime.remove_hooks()


def selfcheck() -> None:
    stage = {
        "row_index": 2,
        "opener": 4,
        "description": [5, 6],
        "description_end": 7,
        "bbox_start": 8,
        "coordinates": {"x1": 9, "y1": 10, "x2": 11, "y2": 12},
        "bbox_end": 13,
        "next_boundary": 14,
    }
    tokens = [151646] * 15
    tokens[7], tokens[8], tokens[13], tokens[14] = 151647, 151648, 151649, 151646
    for offset, value in zip((9, 10, 11, 12), (10, 20, 30, 40), strict=True):
        tokens[offset] = COORD_BASE + value
    profile = {
        "row_index": 2,
        "row": {"values": [10, 20, 30, 40]},
        "stage": stage,
    }
    slots = _validate_stage(tokens, profile)
    assert [item["offset"] for item in slots] == list(range(4, 15))
    physical, compact = _slot_physical_index(
        prompt_length=5, left_pad=2, target_offset=9, first_target_offset=9
    )
    assert (physical, compact) == (15, 0)
    physical, compact = _slot_physical_index(
        prompt_length=5, left_pad=2, target_offset=14, first_target_offset=9
    )
    assert (physical, compact) == (20, 5)
    histories, context = CaptureRuntime._histories(
        type("Batch", (), {"prompt_token_ids": ((101, 102, 103),)})(),
        [{"token_ids": [1, 2, 3, 4]}],
        0,
        4,
        (2, 3, 99),
    )
    assert histories == [[101, 102, 103, 1, 2, 99, 4]]
    assert context["mutation"] == {"offset": 2, "old_token_id": 3, "new_token_id": 99}
    with __import__("tempfile").TemporaryDirectory() as directory:
        path = Path(directory) / "snapshot.json"
        _write_json_new(path, {"status": "frozen_before_model_load"})
        try:
            _write_json_new(path, {"status": "changed"})
        except FileExistsError:
            pass
        else:
            raise AssertionError("immutable snapshot writer accepted overwrite")
    print("PASS slot offsets, logits_to_keep alignment, role edit, immutable snapshot")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selfcheck", action="store_true")
    parser.add_argument("--model", choices=("tied", "untied"))
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--panel", type=Path, default=PANEL)
    parser.add_argument("--windows", type=Path, default=DEFAULT_WINDOWS)
    parser.add_argument("--boundary-ids", nargs="*", default=[])
    parser.add_argument("--boundary-id", action="append", default=[])
    parser.add_argument("--boundary-ids-file", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-forwards", type=int, default=160_000)
    parser.add_argument("--max-seconds", type=float, default=12 * 3600)
    parser.add_argument("--max-bytes", type=int, default=32 * 1024**3)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.selfcheck:
        selfcheck()
        return
    if args.model is None or args.output is None:
        raise SystemExit("--model and --output are required unless --selfcheck is used")
    run(args)


if __name__ == "__main__":
    main()
