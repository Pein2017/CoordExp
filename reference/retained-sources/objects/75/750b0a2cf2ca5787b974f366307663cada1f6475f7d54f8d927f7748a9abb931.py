"""Bounded Lane-B native source-route parity gate.

This script reuses the saved conditional-mass qualification, then compares an
original-image target-only native request with the original heterogeneous
four-request source group for one model.  It performs no generation and uses
one compact forward per route, retaining row-boundary and forced-description
x1 logits from the same forward.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from probes.training_set_completion.untied_shared import load_model
from probes.training_set_completion.readout_norm_fresh import _binding
from src.inference.bound_requests import build_bound_native_requests
from src.qwen.native import exact_history_inputs, prepare_native_inputs

ROOT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration")
UNIT = ROOT / "2026-09-19-recurrence-spatial-source"
OUT = UNIT / "final" / "technical-gate-v1"
PANEL = ROOT / "2026-09-18-untied-highconfidence18-natural" / "panel.json"
SELECTION = ROOT / "2026-09-18-numerical-recurrence-feedback" / "selection.json"
C_QUAL = ROOT / "2026-09-19-recurrence-conditional-mass" / "qualification" / "qualification.json"
C_MANIFEST = ROOT / "2026-09-19-recurrence-conditional-mass" / "qualification" / "run-manifest.json"
STATE_ROOT = UNIT / "final" / "corrected-pilot-v2" / "inputs" / "manifests"
WORKTREE = Path("/data/CoordExp/.worktrees/research-probes")
PILOT_RECEIPT = UNIT / "final" / "corrected-pilot-v2" / "pilot-receipt.json"
PANEL_SHARED = ROOT / "2026-09-19-recurrence-distribution-census" / "shared-panel.json"
SOURCE_SNAPSHOT = UNIT / "final" / "corrected-pilot-v2" / "source-snapshot.json"
TOL = 2e-4
OBJ_START, OBJ_END, BOX_START = 151646, 151647, 151648


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binding(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "sha256": file_sha(path), "size_bytes": path.stat().st_size}


def digest_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def compare_binding(actual: Any, expected: Any) -> bool:
    if not isinstance(actual, dict) or not isinstance(expected, dict):
        return False
    return actual.get("path") == expected.get("path") and actual.get("sha256") == expected.get("sha256")


def source_inputs(model: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    selection = load(SELECTION)
    boundary = next(item for item in selection["boundaries"] if item["id"] == f"{model}-417044-failure")
    state = load(STATE_ROOT / f"{model}-417044-failure.json")
    panel = load(PANEL)
    group = next(item for item in panel["groups"] if item["key"] == boundary["group"])
    target = next(item for item in group["cases"] if item["row_id"] == boundary["id"].replace(f"{model}-417044-failure", "coco2017_train_000000417044"))
    raw = load(Path(boundary["raw_path"]))
    return boundary, state, {"panel": panel, "group": group, "target": target}, raw


def verify_c_qualification() -> dict[str, Any]:
    c = load(C_QUAL)
    selection = load(SELECTION)
    state = load(STATE_ROOT / "untied-417044-failure.json")
    boundary = next(item for item in selection["boundaries"] if item["id"] == "untied-417044-failure")
    checks: dict[str, bool] = {}
    checks["qualification_status"] = c.get("source_score_batch4_qualification", {}).get("status") == "passed"
    checks["state_id"] = c.get("state_id") == "untied-417044-failure" and c.get("source_example_id") == "coco2017_train_000000417044"
    checks["model"] = c.get("model") == "untied"
    checks["source_row_end"] = c.get("conditioning", {}).get("boundary_prefix_source_row_end") == 169
    checks["prefix_hash"] = c.get("conditioning", {}).get("prefix_token_ids_sha256") == state["prefix"]["source_sha256"] == boundary["prefix_hash"]
    checks["native_hash"] = state["prefix"]["native_token_sha256"] == boundary["native_token_hash"]
    checks["source_trace_binding"] = (
        c.get("source_score_batch4_qualification", {}).get("source_trace", {}).get("path") == boundary["trace_path"]
        and c.get("source_score_batch4_qualification", {}).get("source_trace", {}).get("sha256") == file_sha(Path(boundary["trace_path"]))
    )
    checks["source_raw_binding"] = c.get("source_binding", {}).get("source_identity_binding") == boundary["raw_path"]
    checks["image_binding"] = c.get("source_binding", {}).get("image_path") == state["source"]["case"]["image_path"] and c.get("source_binding", {}).get("image_sha256") == state["source"]["case"]["image_plan"]["image_content_sha256"]
    checks["prompt_width"] = c.get("source_score_batch4_qualification", {}).get("batch_size_one_prompt_token_count") == state["source"]["case"]["image_plan"]["backend_prompt_token_count"] == 1320
    checks["grid"] = c.get("source_score_batch4_qualification", {}).get("image_grid_batch1") == [state["source"]["case"]["image_plan"]["observed_image_grid_thw"]]
    score = c.get("source_score_batch4_qualification", {})
    trace = score.get("source_trace", {})
    checks["winner_and_logit_parity"] = bool(trace.get("winner_agrees") and trace.get("batch8_winner_agrees") and score.get("batch_logit_max_abs_delta", float("inf")) <= TOL and score.get("batch8_logit_max_abs_delta", float("inf")) <= TOL and trace.get("absolute_logit_delta", float("inf")) <= TOL and trace.get("batch8_absolute_logit_delta", float("inf")) <= TOL)
    checks["position_parity"] = score.get("position_id_max_abs_delta") == 0 and score.get("position8_id_max_abs_delta") == 0
    return {
        "schema": "recurrence_spatial_source.c_reuse_verification.v1",
        "status": "passed" if all(checks.values()) else "mismatch",
        "qualification": binding(C_QUAL),
        "run_manifest": binding(C_MANIFEST),
        "state_manifest": binding(STATE_ROOT / "untied-417044-failure.json"),
        "selection": binding(SELECTION),
        "checks": checks,
        "conditioning": {
            "state_id": c.get("state_id"),
            "source_example_id": c.get("source_example_id"),
            "source_row_end": c.get("conditioning", {}).get("boundary_prefix_source_row_end"),
            "prefix_token_ids_sha256": c.get("conditioning", {}).get("prefix_token_ids_sha256"),
            "prompt_token_ids_sha256": c.get("conditioning", {}).get("prompt_token_ids_sha256"),
            "description_is_conditioned": c.get("conditioning", {}).get("description_is_conditioned"),
        },
        "source_score_batch4_qualification": {
            "batch_sizes": [1, 4, 8],
            "tolerance": TOL,
            "batch_logit_max_abs_delta": score.get("batch_logit_max_abs_delta"),
            "batch8_logit_max_abs_delta": score.get("batch8_logit_max_abs_delta"),
            "source_trace": trace,
            "media_sha256_batch1": score.get("media_sha256_batch1"),
            "image_grid_batch1": score.get("image_grid_batch1"),
        },
        "interpretation": "Reused only as an original untied 417044 source-entry identity and duplicated batch 1/4/8 parity; it does not certify the Lane-B tied route or any spatial cell.",
    }


def make_snapshot(c_reuse: dict[str, Any]) -> dict[str, Any]:
    boundary_t, state_t, info_t, raw_t = source_inputs("tied")
    boundary_u, state_u, info_u, raw_u = source_inputs("untied")
    paths = [
        Path(__file__), PANEL, SELECTION, C_QUAL, C_MANIFEST, PANEL_SHARED,
        SOURCE_SNAPSHOT, PILOT_RECEIPT,
        WORKTREE / "probes/training_set_completion/untied_shared.py",
        WORKTREE / "src/inference/bound_requests.py",
        WORKTREE / "src/qwen/native.py",
        WORKTREE / "probes/training_set_completion/recurrence_mass/run.py",
        Path(boundary_t["raw_path"]), Path(boundary_t["trace_path"]), Path(boundary_t["receipt_path"]),
        Path(boundary_u["raw_path"]), Path(boundary_u["trace_path"]), Path(boundary_u["receipt_path"]),
        STATE_ROOT / "tied-417044-failure.json", STATE_ROOT / "untied-417044-failure.json",
    ]
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        path = path.resolve(strict=True)
        if path not in seen:
            seen.add(path); unique.append(path)
    return {
        "schema": "recurrence_spatial_source.technical_gate_precall_snapshot.v1",
        "unit_id": "2026-09-19-recurrence-spatial-source",
        "gate_id": "technical-gate-v1",
        "captured_before_model_calls": True,
        "model_calls_before_snapshot": 0,
        "frozen": {
            "tolerance": TOL,
            "attention": "sdpa",
            "dtype": "fp32",
            "model_forwards_cap": 8,
            "no_generation": True,
            "row_boundary": "source_row.end",
            "forced_description_boundary": "source_row.end + 5 input tokens; x1 next-token logits",
            "comparison": "original-image target-only batch1 versus original heterogeneous source batch4",
        },
        "c_reuse": c_reuse,
        "pilot_artifacts_preserved": {
            "pilot_receipt": binding(PILOT_RECEIPT),
            "pilot_status": load(PILOT_RECEIPT).get("status"),
            "pilot_scope": load(PILOT_RECEIPT).get("scope"),
            "rerun_00": False,
            "new_spatial_cells": 0,
        },
        "state_bindings": {
            "tied": binding(STATE_ROOT / "tied-417044-failure.json"),
            "untied": binding(STATE_ROOT / "untied-417044-failure.json"),
        },
        "source_files": [binding(path) for path in unique],
    }


def raw_prefix(raw: list[dict[str, Any]], end: int, pad: int) -> list[list[int]]:
    rows: list[list[int]] = []
    for item in raw:
        tokens = [int(v) for v in item["token_ids"][:end]]
        if len(tokens) < end:
            if 151645 not in tokens:
                tokens.append(151645)
            else:
                tokens = tokens[: tokens.index(151645) + 1]
            tokens.extend([pad] * (end - len(tokens)))
        rows.append(tokens[:end])
    if any(len(item) != end for item in rows):
        raise ValueError("source histories are not equal width")
    return rows


def top_summary(logits: torch.Tensor, trace_step: dict[str, Any] | None, target_index: int | None = None) -> dict[str, Any]:
    values, indices = torch.topk(logits, 5)
    out: dict[str, Any] = {
        "winner_token_id": int(indices[0].item()),
        "winner_logit": float(values[0].item()),
        "runnerup_token_id": int(indices[1].item()),
        "runnerup_logit": float(values[1].item()),
        "winner_margin": float((values[0] - values[1]).item()),
        "top5": [{"token_id": int(i), "logit": float(v)} for i, v in zip(indices.tolist(), values.tolist(), strict=True)],
    }
    if trace_step is not None and target_index is not None:
        winners = trace_step.get("raw_winners", [])
        chosen = trace_step.get("chosen_raw_logits", [])
        if target_index < len(winners) and target_index < len(chosen):
            token = int(winners[target_index]); logit = float(chosen[target_index])
            out["saved_trace"] = {
                "token_id": token,
                "chosen_raw_logit": logit,
                "observed_at_trace_token_logit": float(logits[token].item()),
                "absolute_logit_delta": abs(float(logits[token].item()) - logit),
                "winner_matches": int(indices[0].item()) == token,
                "passed": bool(int(indices[0].item()) == token and abs(float(logits[token].item()) - logit) <= TOL),
            }
    return out


def compact_forward(q: Any, native: Any, rows: list[list[int]], device: str, gpu_label: str) -> tuple[torch.Tensor, dict[str, Any], float]:
    prompts = [list(row) for row in native.prompt_token_ids]
    histories = [prompt + prefix + [] for prompt, prefix in zip(prompts, rows, strict=True)]
    if not histories or any(not row for row in histories):
        raise ValueError("empty source history")
    inputs = exact_history_inputs(q.model, native.inputs, histories, pad_token_id=0, logits_to_keep=6)
    if int(inputs["input_ids"].shape[0]) != len(rows):
        raise ValueError("replay batch cardinality drift")
    torch.cuda.synchronize(device)
    event_start = torch.cuda.Event(enable_timing=True)
    event_end = torch.cuda.Event(enable_timing=True)
    event_start.record()
    with torch.inference_mode():
        output = q.model(**inputs)
    event_end.record(); torch.cuda.synchronize(device)
    gpu_ms = float(event_start.elapsed_time(event_end))
    logits = output.logits.detach().float().cpu()
    if logits.ndim != 3 or logits.shape[1] != 6:
        raise RuntimeError(f"compact logits shape is {tuple(logits.shape)}, expected [batch,6,vocab]")
    return logits, {
        "batch_size": len(rows),
        "history_token_counts": [len(row) for row in histories],
        "prompt_token_counts": [len(row) for row in prompts],
        "input_shape": list(inputs["input_ids"].shape),
        "attention_shape": list(inputs["attention_mask"].shape),
        "logits_shape": list(logits.shape),
        "gpu_label": gpu_label,
        "position_ids": inputs.get("position_ids").detach().cpu().tolist() if isinstance(inputs.get("position_ids"), torch.Tensor) else None,
        "attention_mask": inputs["attention_mask"].detach().cpu().tolist(),
    }, gpu_ms


def compare_route(model_key: str, device: str, c_reuse: dict[str, Any]) -> dict[str, Any]:
    boundary, state, info, raw_payload = source_inputs(model_key)
    panel = info["panel"]; group = info["group"]; target = info["target"]
    raw = raw_payload["rows"]
    target_index = int(boundary["batch_index"])
    end = int(boundary["source_row"]["end"])
    prompt_plan = state["source"]["case"]["image_plan"]
    config = dict(panel["configs"][model_key])
    config["data"] = dict(input_jsonl=group["input_jsonl"])
    started = time.monotonic()
    q, identity = load_model(model_key, device)
    versions = {name: parameter._version for name, parameter in q.model.named_parameters()}
    requests, _metadata = build_bound_native_requests(q, config, group["cases"])
    native_group = prepare_native_inputs(q.processor, requests, device=device, record_media_identity=True)
    target_requests, _target_metadata = build_bound_native_requests(q, config, [target])
    native_one = prepare_native_inputs(q.processor, target_requests, device=device, record_media_identity=True)
    if tuple(native_group.prompt_token_ids[target_index]) != tuple(native_one.prompt_token_ids[0]):
        raise RuntimeError(f"{model_key}: target prompt differs between heterogeneous and target-only routes")
    if native_group.image_grids[target_index] != native_one.image_grids[0]:
        raise RuntimeError(f"{model_key}: target image grid differs between routes")
    if native_group.media_sha256 is None or native_one.media_sha256 is None or native_group.media_sha256[target_index] != native_one.media_sha256[0]:
        raise RuntimeError(f"{model_key}: target media identity differs between routes")
    if len(native_one.prompt_token_ids[0]) != int(prompt_plan["backend_prompt_token_count"]):
        raise RuntimeError(f"{model_key}: target prompt width differs from saved plan")
    if list(native_one.image_grids[0]) != list(prompt_plan["observed_image_grid_thw"]):
        raise RuntimeError(f"{model_key}: target image grid differs from saved plan")
    target_native = [int(v) for v in raw[target_index]["token_ids"]]
    if digest_json(target_native) != state["prefix"]["native_token_sha256"]:
        raise RuntimeError(f"{model_key}: saved native token hash differs from state manifest")
    if digest_json(target_native[:end]) != state["prefix"]["source_sha256"]:
        raise RuntimeError(f"{model_key}: saved source prefix hash differs from state manifest")
    expected_row_prefix = [OBJ_START, *[int(v) for v in boundary["source_row"]["description_tokens"]], OBJ_END, BOX_START]
    if target_native[end : end + len(expected_row_prefix)] != expected_row_prefix:
        raise RuntimeError(f"{model_key}: saved next row prefix is not the frozen description wrapper")
    # Every source-group row keeps its literal native prefix. For the forced x1
    # comparison, append each row's own saved five-token wrapper so the source
    # batch remains heterogeneous while the target slot is exactly frozen.
    histories_boundary = raw_prefix(raw, end, 0)
    histories_x1: list[list[int]] = []
    extensions: list[list[int]] = []
    for item, prefix in zip(raw, histories_boundary, strict=True):
        tokens = [int(v) for v in item["token_ids"]]
        extension = tokens[end : end + len(expected_row_prefix)]
        if len(extension) != len(expected_row_prefix):
            extension = list(expected_row_prefix)
        extensions.append(extension)
        histories_x1.append(prefix + extension)
    if extensions[target_index] != expected_row_prefix:
        raise RuntimeError(f"{model_key}: source target forced wrapper differs from frozen wrapper")
    target_boundary = target_native[:end]
    target_x1 = target_boundary + expected_row_prefix
    source_boundary_logits, source_boundary_meta, source_boundary_ms = compact_forward(q, native_group, histories_boundary, device, device)
    source_x1_logits, source_x1_meta, source_x1_ms = compact_forward(q, native_group, histories_x1, device, device)
    target_boundary_logits, target_boundary_meta, target_boundary_ms = compact_forward(q, native_one, [target_boundary], device, device)
    target_x1_logits, target_x1_meta, target_x1_ms = compact_forward(q, native_one, [target_x1], device, device)
    # The two route pairs must use the same valid target positions.  Compare
    # only unpadded target position rows because the source batch has longer
    # companion prompts and therefore left padding.
    def position_delta(meta_a: dict[str, Any], meta_b: dict[str, Any]) -> int | None:
        pos_a, pos_b = meta_a.get("position_ids"), meta_b.get("position_ids")
        mask_a, mask_b = meta_a.get("attention_mask"), meta_b.get("attention_mask")
        if pos_a is None or pos_b is None:
            return None
        pa = torch.tensor(pos_a)[..., torch.tensor(mask_a[0], dtype=torch.bool)]
        pb = torch.tensor(pos_b)[..., torch.tensor(mask_b[0], dtype=torch.bool)]
        if pa.shape != pb.shape:
            return -1
        return int(torch.max(torch.abs(pa.to(torch.int64) - pb.to(torch.int64))).item())
    pos_boundary = position_delta(source_boundary_meta, target_boundary_meta)
    pos_x1 = position_delta(source_x1_meta, target_x1_meta)
    trace_payload = load(Path(boundary["trace_path"]))
    def trace_step(offset: int) -> dict[str, Any]:
        return trace_payload["steps"][offset]
    def stage(name: str, source_logits: torch.Tensor, target_logits: torch.Tensor, offset: int) -> dict[str, Any]:
        src = source_logits[target_index, 0 if name == "row_boundary" else -1]
        tgt = target_logits[0, 0 if name == "row_boundary" else -1]
        delta = torch.abs(src - tgt)
        trace = trace_step(offset)
        return {
            "name": name,
            "source_route_batch_index": target_index,
            "source_shape": list(src.shape),
            "target_shape": list(tgt.shape),
            "full_vocab_max_abs_delta": float(delta.max().item()),
            "winner_exact_match": int(torch.argmax(src).item()) == int(torch.argmax(tgt).item()),
            "source": top_summary(src, trace, target_index),
            "target_only": top_summary(tgt, None),
            "saved_trace_offset": offset,
            "passed": bool(float(delta.max().item()) <= TOL and int(torch.argmax(src).item()) == int(torch.argmax(tgt).item()) and top_summary(src, trace, target_index).get("saved_trace", {}).get("passed", False)),
        }
    stages = {
        "row_boundary": stage("row_boundary", source_boundary_logits, target_boundary_logits, end),
        "forced_description_x1": stage("forced_description_x1", source_x1_logits, target_x1_logits, end + len(expected_row_prefix)),
    }
    current_versions = {name: parameter._version for name, parameter in q.model.named_parameters()}
    if current_versions != versions:
        raise RuntimeError(f"{model_key}: model parameters mutated during parity gate")
    try:
        torch.cuda.synchronize(device)
    except Exception:
        pass
    elapsed = time.monotonic() - started
    return {
        "schema": "recurrence_spatial_source.route_parity_comparison.v1",
        "model": model_key,
        "status": "passed" if all(item["passed"] for item in stages.values()) and pos_boundary in (0, None) and pos_x1 in (0, None) else "failed",
        "runtime_identity": identity,
        "source": {
            "boundary": boundary,
            "state_manifest": binding(STATE_ROOT / f"{model_key}-417044-failure.json"),
            "raw": binding(Path(boundary["raw_path"])),
            "trace": binding(Path(boundary["trace_path"])),
            "receipt": binding(Path(boundary["receipt_path"])),
            "source_group_case_count": len(group["cases"]),
            "target_batch_index": target_index,
            "source_row_end": end,
            "target_native_token_hash": digest_json(target_native),
            "target_prefix_hash": digest_json(target_native[:end]),
            "forced_wrapper": expected_row_prefix,
            "companion_forced_wrappers": extensions,
            "native_group": {
                "prompt_token_counts": [len(row) for row in native_group.prompt_token_ids],
                "image_grids": [list(grid) if grid is not None else None for grid in native_group.image_grids],
                "media_sha256": list(native_group.media_sha256 or ()),
            },
        },
        "target_only": {
            "request": target,
            "native": {
                "prompt_token_count": len(native_one.prompt_token_ids[0]),
                "image_grid": list(native_one.image_grids[0]) if native_one.image_grids[0] is not None else None,
                "media_sha256": native_one.media_sha256[0] if native_one.media_sha256 else None,
            },
        },
        "conditioning": {
            "history_policy": "literal saved native prefixes through source_row.end; forced x1 appends frozen row wrapper; companions retain their own saved five-token suffixes",
            "description_tokens": [int(v) for v in boundary["source_row"]["description_tokens"]],
            "pad_token_id": 0,
            "positions": {"row_boundary_target_delta": pos_boundary, "forced_x1_target_delta": pos_x1},
            "dtype": "fp32",
            "attention": "sdpa",
            "generation": "none",
            "temperature": None,
        },
        "forwards": {
            "native_model_forwards": 4,
            "source_heterogeneous_forwards": 2,
            "target_only_forwards": 2,
            "source_heterogeneous_gpu_ms": source_boundary_ms + source_x1_ms,
            "target_only_gpu_ms": target_boundary_ms + target_x1_ms,
            "per_forward_gpu_ms": {
                "source_row_boundary": source_boundary_ms,
                "source_forced_x1": source_x1_ms,
                "target_row_boundary": target_boundary_ms,
                "target_forced_x1": target_x1_ms,
            },
            "elapsed_seconds_including_load_and_prepare": elapsed,
        },
        "stages": stages,
        "stop_rule": "Any full-vocabulary delta > 2e-4, winner mismatch, saved source chosen-logit mismatch, target position mismatch, or identity mismatch is a concrete parity failure and blocks broad Lane-B launch.",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    existing = [path for path in OUT.iterdir() if path.name != "run_gate.py"]
    if existing:
        raise RuntimeError(f"technical gate output is not empty: {existing}")
    c_reuse = verify_c_qualification()
    snapshot = make_snapshot(c_reuse)
    write(OUT / "c-qualification-reuse.json", c_reuse)
    write(OUT / "precall-snapshot.json", snapshot)
    # This marker is written before any loader/model call.
    write(OUT / "precall-marker.json", {"captured_before_model_calls": True, "model_calls": 0, "snapshot": binding(OUT / "precall-snapshot.json")})
    device = "cuda:0"
    models = ["tied"] + ([] if c_reuse["status"] == "passed" else ["untied"])
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for model_key in models:
        try:
            result = compare_route(model_key, device, c_reuse)
            write(OUT / f"comparison-{model_key}.json", result)
            results.append(result)
            if result["status"] != "passed":
                break
        except BaseException as exc:
            errors.append({"model": model_key, "error": repr(exc)})
            break
    receipt = {
        "schema": "recurrence_spatial_source.technical_gate_receipt.v1",
        "unit_id": "2026-09-19-recurrence-spatial-source",
        "gate_id": "technical-gate-v1",
        "status": "passed" if c_reuse["status"] == "passed" and results and all(item["status"] == "passed" for item in results) and not errors else "failed",
        "scientific_scope": "technical native source-route qualification only; no spatial cell or free continuation",
        "c_reuse": {"status": c_reuse["status"], "artifact": binding(OUT / "c-qualification-reuse.json")},
        "comparisons": [{"model": item["model"], "status": item["status"], "artifact": binding(OUT / f"comparison-{item['model']}.json")} for item in results],
        "errors": errors,
        "model_forwards": sum(int(item.get("forwards", {}).get("native_model_forwards", 0)) for item in results),
        "model_forward_cap": 8,
        "no_generation": True,
        "tolerance": TOL,
        "source_snapshot": binding(OUT / "precall-snapshot.json"),
        "precall_marker": binding(OUT / "precall-marker.json"),
        "pilot_preserved": True,
        "broad_lane_release": "not_authorized_by_worker; parent/root must inspect this receipt",
    }
    write(OUT / "gate-receipt.json", receipt)
    print(json.dumps(receipt, indent=2))
    if receipt["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
