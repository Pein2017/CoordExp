"""Native original-readout spatial/history producer for the bounded lane."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch
from transformers import LogitsProcessor, LogitsProcessorList

from probes.training_set_completion.recurrence_spatial.prepare import (
    COORD_BASE,
    DEFAULT_MODEL,
    OUT,
    POLICY,
    ROOT,
    map_bin,
)
from probes.training_set_completion.untied_shared import load_model
from probes.training_set_completion.readout_norm_fresh import _binding, _input_identity
from src.inference.bound_requests import build_bound_native_requests
from src.qwen.generation import NativeGenerationPolicy, generate_continuations
from src.qwen.native import NativeRequest, prepare_native_inputs


EOS = 151645
OBJ_START = 151646
OBJ_END = 151647
BOX_END = 151649
COORD_LIMIT = COORD_BASE + 1000
MAX_NEW_TOKENS = 512
MAX_COMPLETE_ROWS = 32
DEVICE = "cuda:4"


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def parse_rows(
    token_ids: list[int],
    tokenizer: Any,
    *,
    cell: dict[str, Any],
    geometry: dict[str, Any],
) -> dict[str, Any]:
    starts = [index for index, token in enumerate(token_ids) if token == OBJ_START]
    rows: list[dict[str, Any]] = []
    for row_index, start in enumerate(starts):
        stop = starts[row_index + 1] if row_index + 1 < len(starts) else len(token_ids)
        segment = token_ids[start:stop]
        coord_positions = [
            index for index, token in enumerate(segment) if COORD_BASE <= token < COORD_LIMIT
        ]
        if len(coord_positions) != 4:
            rows.append(
                {
                    "row_index": row_index,
                    "status": "malformed",
                    "token_count": len(segment),
                    "raw_token_ids": segment,
                }
            )
            continue
        bins = [segment[index] - COORD_BASE for index in coord_positions]
        description_tokens = segment[1 : segment.index(OBJ_END)] if OBJ_END in segment else []
        description = tokenizer.decode(
            description_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        mapped = [
            int(value)
            for value in bins
        ]
        valid = mapped[0] < mapped[2] and mapped[1] < mapped[3]
        inverse = [
            int(value)
            for value in (
                _inverse_output_bin(value, cell=cell, geometry=geometry)
                for value in mapped
            )
        ]
        source_in_bounds = all(0 <= value <= 999 for value in inverse)
        rows.append(
            {
                "row_index": row_index,
                "status": "valid" if valid else "invalid",
                "description": description,
                "coord_bins_canvas": mapped,
                "coord_bins_source": inverse,
                "source_in_bounds": source_in_bounds,
                "canvas_border": any(value in (0, 999) for value in mapped),
                "source_geometry_valid": valid and source_in_bounds and inverse[0] < inverse[2] and inverse[1] < inverse[3],
                "token_count": len(segment),
                "raw_token_ids": segment,
            }
        )
    exact_runs = _runs(rows, near=False)
    near_runs = _runs(rows, near=True)
    return {
        "rows": rows,
        "complete_rows": len(rows),
        "valid_rows": sum(row["status"] == "valid" for row in rows),
        "invalid_rows": sum(row["status"] == "invalid" for row in rows),
        "malformed_rows": sum(row["status"] == "malformed" for row in rows),
        "exact_runs": exact_runs,
        "near_runs": near_runs,
        "failure_predicate": bool(exact_runs or near_runs),
    }


def _inverse_output_bin(value: int, *, cell: dict[str, Any], geometry: dict[str, Any]) -> int:
    source_width = int(geometry["source_width"])
    canvas_width = int(geometry["canvas_width"])
    tx = int(cell["visual_offset_px"])
    canvas_pixel = value * (canvas_width - 1) / 999.0
    source_pixel = canvas_pixel - tx
    return int(round(source_pixel * 999.0 / (source_width - 1)))


def _runs(rows: list[dict[str, Any]], *, near: bool) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    for row in rows:
        if row.get("status") != "valid" or not row.get("source_geometry_valid", False):
            if len(current) >= 3:
                runs.append(
                    {
                        "start_row": current[0]["row_index"],
                        "length": len(current),
                        "description": current[0].get("description"),
                        "coord_bins_source": current[0].get("coord_bins_source"),
                        "kind": "exact" if not near else "near",
                    }
                )
            current = []
            continue
        if not current:
            current = [row]
            continue
        previous = current[-1]
        same_desc = row.get("description") == previous.get("description")
        same_box = row.get("coord_bins_source") == previous.get("coord_bins_source")
        close_box = all(
            abs(a - b) <= 8
            for a, b in zip(row.get("coord_bins_source", []), previous.get("coord_bins_source", []), strict=True)
        )
        if same_desc and (same_box if not near else close_box):
            current.append(row)
        else:
            if len(current) >= 3:
                runs.append(
                    {
                        "start_row": current[0]["row_index"],
                        "length": len(current),
                        "description": current[0].get("description"),
                        "coord_bins_source": current[0].get("coord_bins_source"),
                        "kind": "exact" if not near else "near",
                    }
                )
            current = [row]
    if len(current) >= 3:
        runs.append(
            {
                "start_row": current[0]["row_index"],
                "length": len(current),
                "description": current[0].get("description"),
                "coord_bins_source": current[0].get("coord_bins_source"),
                "kind": "exact" if not near else "near",
            }
        )
    return runs


def iou(left: list[int], right: list[int]) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    ix1, iy1, ix2, iy2 = max(lx1, rx1), max(ly1, ry1), min(lx2, rx2), min(ly2, ry2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (lx2 - lx1) * (ly2 - ly1) + (rx2 - rx1) * (ry2 - ry1) - intersection
    return 0.0 if union <= 0 else intersection / union


def score_known(rows: list[dict[str, Any]], bank: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = []
    for row in rows:
        if row.get("status") != "valid" or not row.get("source_geometry_valid", False):
            continue
        description = row.get("description", "").strip().lower()
        for target in bank:
            if description != str(target["normalized_description"]).lower():
                continue
            candidates.append(
                (
                    iou(row["coord_bins_source"], list(target["reference_coord_bins_1000"])),
                    row,
                    target,
                )
            )
    candidates.sort(key=lambda item: item[0], reverse=True)
    used_rows: set[int] = set()
    used_targets: set[str] = set()
    matches = []
    for value, row, target in candidates:
        owner = str(target["owner_id"])
        row_index = int(row["row_index"])
        if value < 0.5 or row_index in used_rows or owner in used_targets:
            continue
        used_rows.add(row_index)
        used_targets.add(owner)
        matches.append({"row_index": row_index, "owner_id": owner, "iou": value})
    return {"matched_count": len(matches), "matches": matches, "bank_count": len(bank)}


def _known_bank(panel: dict[str, Any], *, image_id: int, split: str | None = None) -> tuple[list[dict[str, Any]], str | None]:
    """Resolve an optional source bank without turning missing witnesses into failure."""

    keys = [str(image_id)]
    if split:
        keys.insert(0, f"{split}:{image_id}")
    for name in ("refined_banks", "new_banks", "sentinel_banks"):
        banks = panel.get(name)
        if not isinstance(banks, dict):
            continue
        for key in keys:
            if key in banks:
                return list(banks[key]), f"{name}[{key}]"
    return [], None


class CaptureFirst(LogitsProcessor):
    def __init__(self) -> None:
        self.logits: torch.Tensor | None = None
        self.input_width: int | None = None

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if self.logits is None:
            self.logits = scores[0].detach().float().cpu()
            self.input_width = int(input_ids.shape[1])
        return scores


class RowLimitEOS(LogitsProcessor):
    """Stop a free continuation after the declared number of complete rows."""

    def __init__(self, *, baseline_end_count: int, max_rows: int, eos: int) -> None:
        self.threshold = baseline_end_count + max_rows
        self.eos = eos

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        complete = (input_ids == OBJ_END).sum(dim=1)
        reached = complete >= self.threshold
        if bool(reached.any()):
            scores = scores.clone()
            scores[reached, :] = -torch.inf
            scores[reached, self.eos] = 0
        return scores


def _run_free(
    model: Any,
    batch: Any,
    history: list[int],
    *,
    eos: int,
    pad: int,
) -> Any:
    """Run the natural continuation with the frozen 32-complete-row bound."""

    original_generate = model.generate

    def wrapped(**kwargs: Any) -> Any:
        input_ids = kwargs.get("input_ids")
        if input_ids is None:
            raise RuntimeError("native free continuation did not expose input_ids")
        baseline_end_count = int((input_ids == OBJ_END).sum(dim=1).max().item())
        existing = kwargs.get("logits_processor")
        processors = list(existing) if existing is not None else []
        kwargs["logits_processor"] = LogitsProcessorList(
            [
                RowLimitEOS(
                    baseline_end_count=baseline_end_count,
                    max_rows=MAX_COMPLETE_ROWS,
                    eos=eos,
                ),
                *processors,
            ]
        )
        return original_generate(**kwargs)

    model.generate = wrapped
    try:
        return generate_continuations(
            model,
            batch,
            extensions=[history],
            budgets=[MAX_NEW_TOKENS],
            eos_token_id=eos,
            pad_token_id=pad,
            policy=NativeGenerationPolicy(
                temperature=0,
                top_p=1,
                top_k=0,
                repetition_penalty=1,
                use_model_defaults=False,
            ),
            trace="policy",
        )[0]
    finally:
        model.generate = original_generate


def _run_capture(model: Any, batch: Any, history: list[int], *, eos: int, pad: int, tokenizer: Any) -> dict[str, Any]:
    capture = CaptureFirst()
    original_generate = model.generate

    def wrapped(**kwargs: Any) -> Any:
        existing = kwargs.get("logits_processor")
        processors = list(existing) if existing is not None else []
        kwargs["logits_processor"] = LogitsProcessorList([capture, *processors])
        return original_generate(**kwargs)

    model.generate = wrapped
    try:
        result = generate_continuations(
            model,
            batch,
            extensions=[history],
            budgets=[1],
            eos_token_id=eos,
            pad_token_id=pad,
            policy=NativeGenerationPolicy(
                temperature=0,
                top_p=1,
                top_k=0,
                repetition_penalty=1,
                use_model_defaults=False,
            ),
            trace="none",
        )[0]
    finally:
        model.generate = original_generate
    if capture.logits is None:
        raise RuntimeError("native capture did not observe a generation step")
    logits = capture.logits
    log_probs = torch.log_softmax(logits, dim=0)
    coordinate_ids = torch.arange(COORD_BASE, COORD_LIMIT)
    coordinate_log_probs = log_probs[coordinate_ids]
    top_values, top_indices = torch.topk(logits, 10)
    chosen = int(result.token_ids[0]) if result.token_ids else eos
    return {
        "chosen_token_id": chosen,
        "stop_reason": result.stop_reason,
        "input_width": capture.input_width,
        "top10": [
            {"token_id": int(token), "logit": float(value), "log_prob": float(log_probs[token])}
            for value, token in zip(top_values.tolist(), top_indices.tolist(), strict=True)
        ],
        "eos": {
            "logit": float(logits[EOS]),
            "log_prob": float(log_probs[EOS]),
        },
        "coordinate_family": {
            "log_mass": float(torch.logsumexp(coordinate_log_probs, dim=0)),
            "mass": float(torch.exp(torch.logsumexp(coordinate_log_probs, dim=0))),
            "max_bin": int(torch.argmax(coordinate_log_probs).item()),
            "max_log_prob": float(torch.max(coordinate_log_probs)),
        },
        "coordinate_log_probs": coordinate_log_probs.tolist(),
        "coordinate_logits": logits[COORD_BASE:COORD_LIMIT].tolist(),
    }


def _window_measurement(capture: dict[str, Any], *, old_bin: int, moved_bin: int) -> dict[str, Any]:
    logits = torch.tensor(capture["coordinate_logits"], dtype=torch.float32)
    log_probs = torch.tensor(capture["coordinate_log_probs"], dtype=torch.float32)
    return {
        "old_bin": old_bin,
        "moved_bin": moved_bin,
        "old_logit": float(logits[old_bin]),
        "moved_logit": float(logits[moved_bin]),
        "old_log_prob": float(log_probs[old_bin]),
        "moved_log_prob": float(log_probs[moved_bin]),
        "moved_minus_old_log_prob": float(log_probs[moved_bin] - log_probs[old_bin]),
    }


def _make_request(
    base: NativeRequest, cell: dict[str, Any], *, geometry: dict[str, Any]
) -> NativeRequest:
    image = Path(cell["image_path"])
    data = image.read_bytes()
    return dataclasses.replace(
        base,
        image=image,
        expected_token_ids=None,
        expected_image_grid=None,
        expected_image_size=(int(geometry["canvas_width"]), int(geometry["canvas_height"])),
        image_sha256=hashlib.sha256(data).hexdigest(),
        # The PNG already contains the frozen transform.  Native's logical
        # transform field accepts only its registered identity operation here.
        logical_transform="identity",
    )


def produce(
    *,
    model_name: str = DEFAULT_MODEL,
    device_name: str = DEVICE,
    out: Path = OUT,
    max_cells: int | None = None,
    requested_cells: list[str] | None = None,
    result_path: Path | None = None,
    manifest_path: Path | None = None,
    skip_gate: bool = False,
) -> dict[str, Any]:
    if manifest_path is None:
        manifest_path = out / f"transform-manifest-{model_name}.json"
    manifest_path = manifest_path.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text())
    if manifest["source"]["model"] != model_name:
        raise ValueError(f"manifest model mismatch: {manifest['source']['model']} != {model_name}")
    image_id = int(manifest["source"]["image_id"])
    group_key = str(manifest["source"]["group"])
    panel_path = Path(manifest["source"]["mature_panel"]["path"])
    panel = json.loads(panel_path.read_text())
    group = next(item for item in panel["groups"] if item["key"] == group_key)
    bound_case = manifest["source"].get("case")
    if bound_case is None:
        bound_case = next(item for item in group["cases"] if int(item["input_record"]["image_id"]) == image_id)
    case = dict(bound_case)
    if int(case["input_record"]["image_id"]) != image_id or str(case.get("row_id")) == "None":
        raise ValueError("transform manifest target case identity drift")
    if "image_plan" not in case or "image_path" not in case:
        raise ValueError("transform manifest target case lacks the frozen image plan")
    config = dict(panel["configs"][model_name])
    config["data"] = dict(input_jsonl=group["input_jsonl"])
    route = manifest.get("native_route")
    if route is not None:
        if not route.get("target_only") or int(route.get("requested_case_count", 1)) != 1:
            raise ValueError("spatial native route must remain the qualified target-only one-case route")
    device = torch.device(device_name)
    start = time.monotonic()
    q, identity = load_model(model_name, device)
    model = q.model
    base_request = build_bound_native_requests(q, config, [case])[0][0]
    raw_descriptor = manifest["source"].get("mature_raw") or manifest["source"].get("raw")
    if raw_descriptor is None:
        raise ValueError("transform manifest must bind its source raw output explicitly")
    raw_path = Path(raw_descriptor["path"] if isinstance(raw_descriptor, dict) else raw_descriptor).resolve(strict=True)
    saved = json.loads(raw_path.read_text())
    saved_row = next(row for row in saved["rows"] if int(row["image_id"]) == image_id)
    saved_tokens = [int(token) for token in saved_row["token_ids"]]
    saved_starts = [index for index, token in enumerate(saved_tokens) if token == OBJ_START]
    source_row_index = int(manifest["prefix"]["source_row_index"])
    prefix_end = int(manifest["prefix"]["source_row_end"])
    if saved_starts[source_row_index] != int(manifest["prefix"]["source_row_start"]):
        raise RuntimeError("saved source-row start no longer matches manifest")
    if saved_starts[source_row_index + 1] != prefix_end:
        raise RuntimeError("saved source-row end no longer matches manifest")
    next_row_index = source_row_index + 1
    next_row_start = saved_starts[next_row_index]
    next_row_end = saved_starts[next_row_index + 1] if next_row_index + 1 < len(saved_starts) else len(saved_tokens)
    next_row = saved_tokens[next_row_start:next_row_end]
    if not next_row:
        raise RuntimeError("selected feedback boundary has no next row for forced-description capture")
    coordinate_index = next(index for index, token in enumerate(next_row) if COORD_BASE <= token < COORD_LIMIT)
    next_x1 = int(next_row[coordinate_index] - COORD_BASE)
    bank, bank_source = _known_bank(
        panel,
        image_id=image_id,
        split=manifest["source"].get("split"),
    )
    all_cell_keys = ["00", "10-", "10+", "01-", "01+", "11-", "11+"]
    cell_keys = all_cell_keys if requested_cells is None else list(requested_cells)
    if max_cells is not None and requested_cells is None:
        cell_keys = cell_keys[:max_cells]
    if not cell_keys or any(key not in all_cell_keys for key in cell_keys) or len(set(cell_keys)) != len(cell_keys):
        raise ValueError(f"invalid cell selection: {cell_keys}")
    if result_path is None:
        result_path = out / f"runtime-result-{model_name}.json"
    expected_kind = str(manifest["source"].get("kind", "failure"))
    if expected_kind not in {"failure", "healthy", "proxy", "nonrecurrent_proxy"}:
        raise ValueError(f"unsupported panel state kind {expected_kind!r}")
    is_proxy = expected_kind in {"healthy", "proxy", "nonrecurrent_proxy"}
    result: dict[str, Any] = {
        "schema": "recurrence_spatial_source_runtime.v1",
        "status": "running",
        "unit_id": "2026-09-19-recurrence-spatial-source",
        "image_id": image_id,
        "group": group_key,
        "device": device_name,
        "model": model_name,
        "policy": POLICY,
        "panel_state_kind": expected_kind,
        "source": {
            "panel": _binding(panel_path),
            "transform_manifest": _binding(manifest_path),
            "producer": _binding(Path(__file__)),
            "model_identity": identity,
            "mature_raw": _binding(raw_path),
            "bank_source": bank_source,
        },
        "cells": {},
        "free_continuation_bound": {
            "max_new_tokens": MAX_NEW_TOKENS,
            "max_complete_rows": MAX_COMPLETE_ROWS,
            "eos_token_id": EOS,
        },
        "model_forwards": 0,
        "vision_forwards": 0,
    }
    if route is not None:
        result["source"]["native_route"] = route
    prompt_reference_tokens: list[int] | None = None
    count = {"model": 0, "vision": 0}
    handles: list[Any] = []

    def count_model(*_: Any, **__: Any) -> None:
        count["model"] += 1

    def count_vision(*_: Any, **__: Any) -> None:
        count["vision"] += 1

    handles = [
        model.register_forward_pre_hook(count_model, with_kwargs=True),
        model.model.visual.register_forward_pre_hook(count_vision),
    ]
    try:
        for key in cell_keys:
            cell = manifest["cells"][key]
            request = _make_request(base_request, cell, geometry=manifest["geometry"])
            batch = prepare_native_inputs(q.processor, [request], device=device, record_media_identity=True)
            input_identity = _input_identity(batch)
            if prompt_reference_tokens is None:
                prompt_reference_tokens = list(input_identity["prompt_token_ids"])
                result["prompt_identity_reference_cell"] = key
            elif input_identity["prompt_token_ids"] != prompt_reference_tokens:
                raise RuntimeError("all common-canvas cells must share the exact prompt token sequence")
            if key == "00":
                result["center_prompt_identity"] = input_identity
            free = _run_free(
                model,
                batch,
                cell["history"],
                eos=EOS,
                pad=q.tokenizer.pad_token_id,
            )
            forced_prefix = cell["history"] + next_row[:coordinate_index]
            opener = _run_capture(
                model,
                batch,
                cell["history"],
                eos=EOS,
                pad=q.tokenizer.pad_token_id,
                tokenizer=q.tokenizer,
            )
            forced = _run_capture(
                model,
                batch,
                forced_prefix,
                eos=EOS,
                pad=q.tokenizer.pad_token_id,
                tokenizer=q.tokenizer,
            )
            geometry = manifest["geometry"]
            old_bin = map_bin(next_x1, source_width=int(geometry["source_width"]), canvas_width=int(geometry["canvas_width"]), tx=128)
            moved_bin = map_bin(next_x1, source_width=int(geometry["source_width"]), canvas_width=int(geometry["canvas_width"]), tx=int(cell["history_offset_px"]))
            parsed = parse_rows(list(free.token_ids), q.tokenizer, cell=cell, geometry=geometry)
            known = score_known(parsed["rows"], bank)
            result["cells"][key] = {
                "status": "complete",
                "input_identity": input_identity,
                "free": {
                    "token_ids": list(free.token_ids),
                    "stop_reason": free.stop_reason,
                    "policy_logprobs": None if free.policy_logprobs is None else list(free.policy_logprobs),
                    "parse": parsed,
                    "known": known,
                },
                "boundary": {
                    "opener": opener,
                    "forced_description_x1": {
                        **forced,
                        "windows": _window_measurement(forced, old_bin=old_bin, moved_bin=moved_bin),
                    },
                },
                "transform": {
                    "visual_offset_px": cell["visual_offset_px"],
                    "history_offset_px": cell["history_offset_px"],
                    "history_sha256": cell["history_sha256"],
                },
            }
            if key == "00" and not skip_gate:
                gate = {
                    "mode": "proxy" if is_proxy else "failure",
                    "valid_rows": parsed["valid_rows"],
                    "known_matches": known["matched_count"],
                    "failure_predicate": parsed["failure_predicate"],
                    "admitted": parsed["valid_rows"] > 0 and (is_proxy or parsed["failure_predicate"]),
                    "failure_persisted": parsed["failure_predicate"],
                    "grounding_witness": known["matched_count"] > 0,
                    "grounding_limitation": known["matched_count"] == 0,
                }
                result["admission"] = gate
                if not gate["admitted"]:
                    result["status"] = "admission_hold"
                    break
    except BaseException as exc:
        result["status"] = "technical_invalid"
        result["error"] = repr(exc)
        raise
    finally:
        for handle in handles:
            handle.remove()
        result["model_forwards"] = count["model"]
        result["vision_forwards"] = count["vision"]
        result["elapsed_seconds"] = time.monotonic() - start
        result["no_parameter_mutation"] = True
        write(result_path, result)
    if result["status"] == "running":
        result["status"] = "candidate_complete"
        write(result_path, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--model", choices=("tied", "untied"), default=DEFAULT_MODEL)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--max-cells", type=int)
    parser.add_argument("--cells", help="comma-separated fixed cell keys")
    parser.add_argument("--result-path", type=Path)
    parser.add_argument("--manifest-path", type=Path)
    parser.add_argument("--skip-gate", action="store_true")
    args = parser.parse_args()
    requested_cells = None if args.cells is None else [item for item in args.cells.split(",") if item]
    print(json.dumps(produce(model_name=args.model, device_name=args.device, out=args.out, max_cells=args.max_cells, requested_cells=requested_cells, result_path=args.result_path, manifest_path=args.manifest_path, skip_gate=args.skip_gate), indent=2))
