"""Run the frozen original-greedy recurrence phase release matrix.

The plan and source bindings are snapshotted before loading the model.  A
ready cell is one immutable torch file; held cells get a small JSON receipt.
"""
from __future__ import annotations

import argparse
import io
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

from probes.training_set_completion.numerical_feedback.metrics import release_metrics
from probes.training_set_completion.numerical_feedback.select import ROLES, rows as parsed_rows
from probes.training_set_completion.readout_norm_fresh import _input_identity
from probes.training_set_completion.recurrence_phase_decision.common import (
    PANEL,
    binding,
    bound_source,
    prefix,
    read_plan,
    token_digest,
)
from probes.training_set_completion.untied_shared import load_model


EOS = 151645
ROW_OPEN = 151646
REF_END = 151647
BOX_START = 151648
COORD_BASE = 151670
COORD_COUNT = 1000
ROW_END = 151649
DEFAULT_PLAN = PANEL.parent.parent / "2026-09-19-recurrence-phase-decision" / "selection" / "plan.json"
TOKEN_CAP = 256
ROW_CAP = 16
TOP_K = 8


def _json_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _safe_cell_name(cell_id: str) -> str:
    if not cell_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_+-" for char in cell_id):
        raise ValueError(f"unsafe cell ID: {cell_id!r}")
    return cell_id


def _tokenizer_ids(tokenizer: Any) -> dict[str, Any]:
    special = {
        "eos": int(tokenizer.eos_token_id),
        "object_ref_start": int(tokenizer.convert_tokens_to_ids("<|object_ref_start|>")),
        "object_ref_end": int(tokenizer.convert_tokens_to_ids("<|object_ref_end|>")),
        "box_start": int(tokenizer.convert_tokens_to_ids("<|box_start|>")),
        "box_end": int(tokenizer.convert_tokens_to_ids("<|box_end|>")),
    }
    coordinates = [int(tokenizer.convert_tokens_to_ids(f"<|coord_{index}|>")) for index in range(COORD_COUNT)]
    actual = {**special, "coordinate_ids": coordinates}
    expected = {
        "eos": EOS,
        "object_ref_start": ROW_OPEN,
        "object_ref_end": REF_END,
        "box_start": BOX_START,
        "box_end": ROW_END,
        "coordinate_ids": list(range(COORD_BASE, COORD_BASE + COORD_COUNT)),
    }
    if actual != expected:
        raise ValueError(f"runtime tokenizer IDs differ from the frozen native vocabulary: {actual}")
    return actual


def _row_summary(tokens: list[int]) -> dict[str, Any]:
    parsed = parsed_rows(tokens)
    return {
        "complete_rows": len(parsed),
        "invalid_rows": sum(not bool(row["valid"]) for row in parsed),
        "malformed_openers": max(0, tokens.count(ROW_OPEN) - len(parsed)),
        "unparsed_tokens": len(tokens) - sum(row["end"] - row["start"] for row in parsed) - int(EOS in tokens),
        "eos": EOS in tokens,
        "parsed_rows": parsed,
    }


def _requested_ids(args: argparse.Namespace) -> list[str] | None:
    if args.cell_ids_file is None and args.cell_ids is None:
        return None
    if args.cell_ids_file is not None:
        value = json.loads(args.cell_ids_file.read_text())
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError("--cell-ids-file must contain a JSON list of string cell IDs")
        ids = list(value)
    else:
        ids = list(args.cell_ids)
    if len(ids) != len(set(ids)):
        raise ValueError("explicit cell IDs contain duplicates")
    return ids


def _select_cells(plan: dict[str, Any], model: str, requested: list[str] | None) -> list[dict[str, Any]]:
    by_id = {str(cell["id"]): cell for cell in plan["cells"]}
    model_cells = [cell for cell in plan["cells"] if cell["model"] == model]
    if requested is None:
        return model_cells
    if len(requested) != len(set(requested)):
        raise ValueError("explicit cell IDs contain duplicates")
    unknown = sorted(set(requested) - set(by_id))
    if unknown:
        raise ValueError(f"cell IDs are absent from the frozen plan: {unknown}")
    wrong_model = sorted(cell_id for cell_id in requested if by_id[cell_id]["model"] != model)
    if wrong_model:
        raise ValueError(f"cell IDs belong to another model: {wrong_model}")
    held = sorted(cell_id for cell_id in requested if by_id[cell_id]["status"] != "ready")
    if held:
        raise ValueError(f"explicit cell IDs include held cells: {held}")
    wanted = set(requested)
    return [cell for cell in model_cells if cell["id"] in wanted]


def _boundary(panel: dict[str, Any], boundary_id: str) -> dict[str, Any]:
    matches = [item for item in panel["all_boundaries"] if item["id"] == boundary_id]
    if len(matches) != 1:
        raise ValueError(f"boundary is not unique: {boundary_id}")
    return matches[0]


def _phase_row(boundary: dict[str, Any], cell: dict[str, Any]) -> dict[str, Any]:
    rows = {int(row["index"]): row for row in parsed_rows([int(token) for token in boundary["native_tokens"]])}
    try:
        return rows[int(cell["phase_row_index"])]
    except KeyError as exc:
        raise ValueError(f"phase row is absent from the bound native history: {cell['id']}") from exc


def _site(boundary: dict[str, Any], cell: dict[str, Any]) -> tuple[int, int, int, str]:
    native = [int(token) for token in boundary["native_tokens"]]
    offset = int(cell["site_offset"])
    old = int(cell["old_token_id"])
    new = int(cell["new_token_id"])
    if not 0 <= offset < len(native) or native[offset] != old:
        raise ValueError(f"cell site does not match the bound native token: {cell['id']}")
    row = _phase_row(boundary, cell)
    role_by_offset = dict(zip(ROLES, row["coordinate_offsets"], strict=True))
    delta = int(cell["delta"])
    role = "x1" if delta == 0 else str(cell["role"])
    if role not in ("x1", "y2") or int(role_by_offset[role]) != offset:
        raise ValueError(f"cell role/site binding is inconsistent: {cell['id']}")
    value = int(row["values"][ROLES.index(role)])
    if old != COORD_BASE + value or new != old + delta:
        raise ValueError(f"cell coordinate edit is inconsistent: {cell['id']}")
    edited = list(cell["edited_values"])
    expected = list(row["values"])
    expected[ROLES.index(role)] += delta
    if edited != expected:
        raise ValueError(f"cell edited geometry is inconsistent: {cell['id']}")
    if cell["release_mode"] == "immediate":
        expected_end = int(row["end"])
        bridge_rows: list[int] = []
    elif cell["release_mode"] == "bridge":
        bridge_rows = [int(item) for item in cell["bridge_row_indices"]]
        if bridge_rows != [int(row["index"]) + 1, int(row["index"]) + 2]:
            raise ValueError(f"bridge rows are not the frozen +2 native rows: {cell['id']}")
        all_rows = {int(item["index"]): item for item in parsed_rows(native)}
        try:
            expected_end = int(all_rows[bridge_rows[-1]]["end"])
        except KeyError as exc:
            raise ValueError(f"bridge row is absent from the bound native history: {cell['id']}") from exc
    else:
        raise ValueError(f"unknown release mode: {cell['release_mode']}")
    if cell["prefix_end"] is None or int(cell["prefix_end"]) != expected_end:
        raise ValueError(f"cell prefix end does not close its declared release boundary: {cell['id']}")
    if not 0 <= offset < int(cell["prefix_end"]):
        raise ValueError(f"cell edit is outside its consumed prefix: {cell['id']}")
    return offset, old, new, role


def _prefix_readback(
    inputs: dict[str, Any], batch: Any, boundary: dict[str, Any], cell: dict[str, Any], site: tuple[int, int, int, str]
) -> dict[str, Any]:
    target = int(boundary["batch_index"])
    end = int(cell["prefix_end"])
    prompt_width = int(batch.inputs["input_ids"].shape[1])
    actual = inputs["input_ids"][target, prompt_width:].detach().cpu().tolist()
    expected = [int(token) for token in boundary["native_tokens"][:end]]
    expected[site[0]] = site[2]
    if actual != expected:
        raise ValueError(f"prefix position readback differs for {cell['id']}")
    prefixes = inputs["input_ids"][:, prompt_width:].detach().cpu().tolist()
    return {
        "prompt_width": prompt_width,
        "consumed_prefix_length": end,
        "target_prefix_token_ids": actual,
        "target_prefix_sha256": token_digest(actual),
        "target_prefix_mutation": {"offset": site[0], "old_token_id": site[1], "new_token_id": site[2]},
        "companion_prefix_token_ids": prefixes,
        "companion_prefix_sha256": [token_digest(row) for row in prefixes],
    }


class _CompleteRowStop(StoppingCriteria):
    def __init__(self, target_index: int, width: int, token_cap: int = TOKEN_CAP, row_cap: int = ROW_CAP) -> None:
        self.target_index = target_index
        self.width = width
        self.token_cap = token_cap
        self.row_cap = row_cap
        self.reason: str | None = None
        self.complete_rows = 0

    def __call__(self, input_ids: torch.Tensor, _scores: Any, **_: Any) -> bool:
        target = [int(token) for token in input_ids[self.target_index, self.width:].tolist()]
        self.complete_rows = len(parsed_rows(target))
        if EOS in target:
            self.reason = "eos"
        elif self.complete_rows >= self.row_cap:
            self.reason = "rows"
        elif len(target) >= self.token_cap:
            self.reason = "cap"
        return self.reason is not None


class _DecisionCapture(LogitsProcessor):
    def __init__(self, target_index: int, width: int, prefix_end: int) -> None:
        self.target_index = target_index
        self.width = width
        self.prefix_end = prefix_end
        self.first_logits: torch.Tensor | None = None
        self.first_top: list[dict[str, Any]] | None = None
        self.subsequent: list[dict[str, Any]] = []

    @staticmethod
    def _top(score: torch.Tensor) -> tuple[list[dict[str, Any]], float]:
        values, indices = torch.topk(score, min(TOP_K, score.numel()))
        margin = float((values[0] - values[1]).item()) if values.numel() > 1 else float("inf")
        top = [
            {"rank": rank + 1, "token_id": int(index), "logit": float(value), "margin_to_top": float(values[0] - value)}
            for rank, (index, value) in enumerate(zip(indices.tolist(), values.tolist(), strict=True))
        ]
        return top, margin

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        step = int(input_ids.shape[1] - self.width)
        score = scores[self.target_index].detach().float()
        if not torch.isfinite(score).all():
            raise ValueError("nonfinite free-release vocabulary scores")
        top, margin = self._top(score)
        if step == 0:
            self.first_logits = score.cpu().clone()
            self.first_top = top
        else:
            self.subsequent.append(
                {
                    "generated_step": step,
                    "consumed_prefix_length": int(self.prefix_end + step),
                    "next_action_offset": int(self.prefix_end + step),
                    "top_competitors": top,
                    "top1_margin": margin,
                }
            )
        return scores


class _HeadCapture:
    def __init__(self, target_index: int) -> None:
        self.target_index = target_index
        self.value: torch.Tensor | None = None

    def __call__(self, _module: Any, values: tuple[Any, ...]) -> None:
        if self.value is not None:
            return
        value = values[0]
        if not isinstance(value, torch.Tensor):
            raise TypeError("lm-head pre-hook did not receive a tensor")
        if value.ndim == 3:
            self.value = value[self.target_index, -1].detach().float().cpu().clone()
        elif value.ndim == 2:
            self.value = value[self.target_index].detach().float().cpu().clone()
        else:
            raise ValueError(f"unexpected lm-head input shape: {tuple(value.shape)}")


class ReleaseRuntime:
    def __init__(
        self,
        *,
        args: argparse.Namespace,
        plan: dict[str, Any],
        panel: dict[str, Any],
        cells: list[dict[str, Any]],
        launch_path: Path,
        held_paths: list[Path],
    ) -> None:
        self.args = args
        self.plan = plan
        self.panel = panel
        self.cells = cells
        self.output = args.output
        self.device = torch.device(args.device)
        self.started = time.monotonic()
        self.model_forwards = 0
        self.vision_forwards = 0
        self.artifact_bytes = sum(path.stat().st_size for path in [launch_path, *held_paths])
        self.cell_paths: list[Path] = list(held_paths)
        self.handles: list[Any] = []
        self.sources: dict[str, tuple[Any, list[dict[str, Any]], dict[str, Any], dict[str, Any]]] = {}
        self.identity: dict[str, Any] | None = None
        self.q: Any = None
        self.launch_path = launch_path

    def _install_counters(self) -> None:
        def count(_module: Any, _inputs: tuple[Any, ...]) -> None:
            self.model_forwards += 1
            if self.model_forwards > self.args.max_forwards:
                raise RuntimeError("allocated model-forward budget exhausted")
            if time.monotonic() - self.started > self.args.max_seconds:
                raise RuntimeError("allocated wall-time budget exhausted")

        def vision(_module: Any, _inputs: tuple[Any, ...]) -> None:
            self.vision_forwards += 1

        self.handles = [self.q.model.register_forward_pre_hook(count), self.q.model.model.visual.register_forward_pre_hook(vision)]

    def _source(self, boundary_id: str) -> tuple[Any, list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        if boundary_id not in self.sources:
            self.sources[boundary_id] = bound_source(
                self.plan, self.panel, boundary_id, self.q, self.device
            )
        return self.sources[boundary_id]

    def _generate(self, inputs: dict[str, Any], target_index: int, width: int, prefix_end: int) -> tuple[torch.Tensor, _CompleteRowStop, _DecisionCapture, torch.Tensor]:
        stop = _CompleteRowStop(target_index, width)
        capture = _DecisionCapture(target_index, width, prefix_end)
        head_capture = _HeadCapture(target_index)
        handle = self.q.model.get_output_embeddings().register_forward_pre_hook(head_capture)
        try:
            config = GenerationConfig(
                max_new_tokens=TOKEN_CAP,
                do_sample=False,
                repetition_penalty=1.0,
                eos_token_id=EOS,
                pad_token_id=int(self.q.tokenizer.pad_token_id),
            )
            with torch.inference_mode():
                result = self.q.model.generate(
                    **inputs,
                    generation_config=config,
                    use_model_defaults=False,
                    logits_processor=LogitsProcessorList([capture]),
                    stopping_criteria=StoppingCriteriaList([stop]),
                )
        finally:
            handle.remove()
        if capture.first_logits is None or capture.first_top is None or head_capture.value is None:
            raise RuntimeError("free-release first decision capture is missing")
        return result, stop, capture, head_capture.value

    def _cell(self, cell: dict[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        boundary = _boundary(self.panel, cell["boundary_id"])
        batch, raw, trace, source_info = self._source(cell["boundary_id"])
        phase_row = _phase_row(boundary, cell)
        site = _site(boundary, cell)
        inputs = prefix(
            batch,
            raw,
            boundary,
            int(cell["prefix_end"]),
            self.q,
            self.device,
            site=(site[0], site[1], site[2]),
        )
        prefix_info = _prefix_readback(inputs, batch, boundary, cell, site)
        target_index = int(boundary["batch_index"])
        prompt_width = int(prefix_info["prompt_width"])
        width = prompt_width + int(cell["prefix_end"])
        if int(inputs["input_ids"].shape[1]) != width:
            raise ValueError(f"generated width does not bind consumed prefix: {cell['id']}")
        before_forwards, before_vision = self.model_forwards, self.vision_forwards
        result, stopper, capture, head_input = self._generate(
            inputs, target_index, width, int(cell["prefix_end"])
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        generated = [[int(token) for token in result[index, width:].detach().cpu().tolist()] for index in range(result.shape[0])]
        target_generated = generated[target_index]
        reason = stopper.reason
        if reason is None:
            reason = "eos" if EOS in target_generated else "cap" if len(target_generated) >= TOKEN_CAP else "generation_complete"
        target_rows = parsed_rows(target_generated[: target_generated.index(EOS) + 1] if EOS in target_generated else target_generated)
        if reason == "rows" and len(target_rows) < ROW_CAP:
            raise RuntimeError(f"row stopper reported rows before complete parsed rows: {cell['id']}")
        subsequent = list(capture.subsequent)
        for record in subsequent:
            step = int(record["generated_step"])
            record["chosen_token_id"] = target_generated[step] if step < len(target_generated) else None
        outputs: list[dict[str, Any]] = []
        for index, item in enumerate(raw):
            full = generated[index]
            trimmed = full[: full.index(EOS) + 1] if EOS in full else full
            summary = _row_summary(trimmed)
            output = {
                "image_id": int(item["image_id"]),
                "batch_index": index,
                "generated_token_ids": full,
                "token_ids": trimmed,
                "text": self.q.tokenizer.decode(trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False),
                "stop_reason": "eos" if EOS in full else "shared_release_stop",
                **summary,
            }
            if index == target_index:
                role = None if int(cell["delta"]) == 0 else str(cell["role"])
                output["release_metrics"] = release_metrics(
                    trimmed,
                    phase_row,
                    role,
                    int(cell["edited_values"][ROLES.index(role)]) if role else None,
                )
            outputs.append(output)
        elapsed = time.monotonic() - started
        first_logits = capture.first_logits
        first_head = head_input
        payload: dict[str, Any] = {
            "schema": "recurrence_phase_decision.release_cell.v1",
            "status": "candidate_complete",
            "cell": cell,
            "source": {
                "boundary_id": boundary["id"],
                "model": boundary["model"],
                "group": boundary["group"],
                "image_id": int(boundary["image_id"]),
                "batch_index": target_index,
                "native_token_hash": boundary["native_token_hash"],
                "native_token_count": len(boundary["native_tokens"]),
                "source_row": boundary["source_row"],
                "phase_row": phase_row,
                "source_bindings": source_info["summary"]["source"],
                "source_panel": source_info["summary"]["source_panel"],
                "input_identity": _input_identity(batch),
                "trace_steps": len(trace["steps"]),
            },
            "mutation": {
                "offset": site[0],
                "role": site[3],
                "delta": int(cell["delta"]),
                "old_token_id": site[1],
                "new_token_id": site[2],
            },
            "prefix": prefix_info,
            "release": {
                "mode": cell["release_mode"],
                "bridge_row_indices": cell["bridge_row_indices"],
                "token_cap": TOKEN_CAP,
                "row_cap": ROW_CAP,
                "stop_reason": reason,
                "complete_rows": len(target_rows),
                "generated_token_count": len(target_generated),
                "parser": "numerical_feedback.select.rows",
            },
            "first_free": {
                "consumed_prefix_length": int(cell["prefix_end"]),
                "next_action_offset": int(cell["prefix_end"]),
                "prompt_width": prompt_width,
                "generated_step": 0,
                "logits_dtype": "float32",
                "head_input_dtype": "float32",
                "logits_shape": list(first_logits.shape),
                "head_input_shape": list(first_head.shape),
                "winner_token_id": int(first_logits.argmax().item()),
                "top_competitors": capture.first_top,
            },
            "first_free_logits": first_logits,
            "first_free_head_input": first_head,
            "subsequent_decisions": subsequent,
            "outputs": outputs,
            "target": outputs[target_index],
            "execution": {
                "device": str(self.device),
                "elapsed_seconds": elapsed,
                "model_forwards": self.model_forwards - before_forwards,
                "vision_forwards": self.vision_forwards - before_vision,
            },
        }
        if first_logits.dtype != torch.float32 or first_head.dtype != torch.float32:
            raise RuntimeError("free-release captures were not stored as FP32")
        return payload

    def _save_cell(self, cell: dict[str, Any], payload: dict[str, Any]) -> Path:
        path = self.output / "cells" / f"{_safe_cell_name(cell['id'])}.pt"
        if path.exists():
            raise FileExistsError(path)
        size = 0
        while True:
            payload["execution"]["artifact_bytes"] = size
            buffer = io.BytesIO()
            torch.save(payload, buffer)
            data = buffer.getvalue()
            if len(data) == size:
                break
            size = len(data)
        if self.artifact_bytes + len(data) > self.args.max_bytes:
            raise RuntimeError("allocated artifact budget exhausted")
        with path.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        self.artifact_bytes += len(data)
        self.cell_paths.append(path)
        return path

    def run(self) -> None:
        if not any(cell["status"] == "ready" for cell in self.cells):
            self._write_receipt(status="candidate_complete")
            return
        try:
            self.q, self.identity = load_model(self.args.model, self.device)
            self.q.model.eval()
            self._versions = {
                name: parameter._version for name, parameter in self.q.model.named_parameters()
            }
            runtime_ids = _tokenizer_ids(self.q.tokenizer)
            self._install_counters()
            for cell in self.cells:
                if cell["status"] != "ready":
                    continue
                try:
                    payload = self._cell(cell)
                    self._save_cell(cell, payload)
                except BaseException as exc:
                    _json_new(
                        self.output / "cells" / f"{_safe_cell_name(cell['id'])}.error.json",
                        {"schema": "recurrence_phase_decision.cell_error.v1", "status": "technical_invalid", "cell": cell, "error": repr(exc)},
                    )
                    raise
            versions = {name: parameter._version for name, parameter in self.q.model.named_parameters()}
            if versions != self._versions:
                raise RuntimeError("model parameter mutation detected")
            self.runtime_ids = runtime_ids
            self._write_receipt(status="candidate_complete")
        except BaseException as exc:
            if not (self.output / "shard-error.json").exists():
                _json_new(
                    self.output / "shard-error.json",
                    {
                        "schema": "recurrence_phase_decision.shard_error.v1",
                        "status": "technical_invalid",
                        "model": self.args.model,
                        "error": repr(exc),
                        "model_forwards": self.model_forwards,
                        "vision_forwards": self.vision_forwards,
                        "elapsed_seconds": time.monotonic() - self.started,
                    },
                )
            raise
        finally:
            for handle in self.handles:
                handle.remove()

    def _write_receipt(self, *, status: str) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        value = {
            "schema": "recurrence_phase_decision.shard_receipt.v1",
            "status": status,
            "model": self.args.model,
            "device": str(self.device),
            "plan": binding(self.args.plan),
            "panel": binding(self.args.panel),
            "producer": binding(Path(__file__)),
            "launch": binding(self.launch_path),
            "model_identity": self.identity,
            "runtime_tokenizer_ids": getattr(self, "runtime_ids", None),
            "selected_cell_ids": [cell["id"] for cell in self.cells],
            "cell_files": [binding(path) for path in self.cell_paths],
            "model_forwards": self.model_forwards,
            "vision_forwards": self.vision_forwards,
            "artifact_bytes": self.artifact_bytes,
            "elapsed_seconds": time.monotonic() - self.started,
            "limits": {"max_forwards": self.args.max_forwards, "max_seconds": self.args.max_seconds, "max_bytes": self.args.max_bytes},
        }
        _json_new(self.output / "shard-receipt.json", value)


def _launch_snapshot(
    *,
    output: Path,
    plan_path: Path,
    panel_path: Path,
    plan: dict[str, Any],
    panel: dict[str, Any],
    cells: list[dict[str, Any]],
    model: str,
    device: str,
) -> Path:
    by_id = {item["boundary_id"]: item for item in plan["source_summaries"]}
    boundary_bindings: dict[str, Any] = {}
    for cell in cells:
        boundary_id = cell["boundary_id"]
        if boundary_id in boundary_bindings:
            continue
        summary = by_id[boundary_id]
        boundary_bindings[boundary_id] = {
            "source": {key: binding(Path(value["path"])) for key, value in summary["source"].items()},
            "source_panel": binding(Path(summary["source_panel"]["path"])),
        }
    value = {
        "schema": "recurrence_phase_decision.launch_snapshot.v1",
        "status": "frozen_before_model_calls",
        "producer": binding(Path(__file__)),
        "plan": binding(plan_path),
        "panel": binding(panel_path),
        "sources": binding(Path(plan["sources"]["path"])),
        "model": model,
        "device": device,
        "selected_cells": cells,
        "source_bindings": boundary_bindings,
    }
    path = output / "launch.json"
    _json_new(path, value)
    return path


def selfcheck() -> None:
    def row(values: list[int]) -> list[int]:
        return [ROW_OPEN, 9, REF_END, BOX_START, *[COORD_BASE + value for value in values], ROW_END]

    native = row([1, 2, 3, 4])
    boundary = {
        "id": "cpu",
        "batch_index": 0,
        "native_tokens": native,
        "source_row": {"index": 0, "start": 0, "end": 9, "coordinate_offsets": [4, 5, 6, 7], "values": [1, 2, 3, 4], "valid": True},
    }
    cell = {
        "id": "cpu__seed__x1+1__immediate",
        "boundary_id": "cpu",
        "phase_row_index": 0,
        "role": "x1",
        "delta": 1,
        "site_offset": 4,
        "old_token_id": COORD_BASE + 1,
        "new_token_id": COORD_BASE + 2,
        "release_mode": "immediate",
        "prefix_end": len(native),
        "bridge_row_indices": [],
        "edited_values": [2, 2, 3, 4],
    }
    site = _site(boundary, cell)
    fake_batch = type("Batch", (), {"inputs": {"input_ids": torch.tensor([[91, 92]])}})()
    fake_inputs = {"input_ids": torch.tensor([[91, 92, *native[:4], COORD_BASE + 2, *native[5:]]])}
    readback = _prefix_readback(fake_inputs, fake_batch, boundary, cell, site)
    assert readback["target_prefix_token_ids"][site[0]] == COORD_BASE + 2
    stopper = _CompleteRowStop(0, width=0, token_cap=100, row_cap=2)
    assert not stopper(torch.tensor([row([1, 2, 3, 4])]), None)
    assert stopper(torch.tensor([row([1, 2, 3, 4]) + row([1, 2, 3, 4])]), None)
    malformed = [ROW_OPEN, 9, ROW_END]
    assert len(parsed_rows(malformed)) == 0
    assert not _CompleteRowStop(0, width=0, token_cap=100, row_cap=1)(torch.tensor([malformed]), None)
    print("PASS generated-suffix stopper, role edit, exact prefix readback, malformed retention")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=PANEL)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--model", choices=("tied", "untied"))
    ids = parser.add_mutually_exclusive_group()
    ids.add_argument("--cell-id", dest="cell_ids", action="append")
    ids.add_argument("--cell-ids-file", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-forwards", type=int)
    parser.add_argument("--max-seconds", type=float, default=12 * 3600)
    parser.add_argument("--max-bytes", type=int, default=32 * 1024**3)
    parser.add_argument("--selfcheck", "--self-check", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
        return
    if args.model is None or args.output is None:
        parser.error("--model and --output are required unless --selfcheck is used")
    args.panel = args.panel.resolve(strict=True)
    args.plan = args.plan.resolve(strict=True)
    panel = json.loads(args.panel.read_text())
    plan = read_plan(args.plan)
    if binding(args.panel) != plan["panel"]:
        raise ValueError("--panel does not match the frozen plan panel binding")
    requested = _requested_ids(args)
    cells = _select_cells(plan, args.model, requested)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "cells").mkdir()
    launch_path = _launch_snapshot(
        output=args.output,
        plan_path=args.plan,
        panel_path=args.panel,
        plan=plan,
        panel=panel,
        cells=cells,
        model=args.model,
        device=args.device,
    )
    held_paths: list[Path] = []
    for cell in cells:
        if cell["status"] == "ready":
            continue
        path = args.output / "cells" / f"{_safe_cell_name(cell['id'])}.json"
        _json_new(
            path,
            {"schema": "recurrence_phase_decision.held_cell.v1", "status": "held", "cell": cell, "reason": cell["status"]},
        )
        held_paths.append(path)
    ready_count = sum(cell["status"] == "ready" for cell in cells)
    args.max_forwards = args.max_forwards if args.max_forwards is not None else 256 * ready_count + 32
    runtime = ReleaseRuntime(
        args=args,
        plan=plan,
        panel=panel,
        cells=cells,
        launch_path=launch_path,
        held_paths=held_paths,
    )
    runtime.run()
    print(json.dumps({"status": "candidate_complete", "model": args.model, "ready_cells": ready_count, "held_cells": len(cells) - ready_count, "model_forwards": runtime.model_forwards}))


if __name__ == "__main__":
    main()
