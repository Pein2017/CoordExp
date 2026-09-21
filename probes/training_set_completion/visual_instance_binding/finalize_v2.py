"""Freeze the corrected, qualification-only Lane B CPU integration.

This finalizer is deliberately separate from the first integration pass.  The
unversioned files are historical audit records and are never rewritten here.
It validates the saved primary receipts, checks the decoded image contracts,
compares the redundant pre-ruling execution, and writes four new ``*-v2``
files only after all checks pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from .compositor import binding


ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-21-visual-instance-binding"
)
ADMISSION = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-21-spatial-progress-gate/selection/shared-admission.json"
)
ADMISSION_BINDING = ROOT / "selection" / "shared-admission-binding.json"
BUDGET = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-21-spatial-progress-gate/selection/budget-estimate.json"
)
WALL_LIMIT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-21-spatial-progress-gate/selection/wall-limit-amendment.json"
)
WALL_START = ROOT / "wall-start.json"
RULING = ROOT / "coordination" / "lane-b-redundant-replay-ruling-03.json"
MILESTONE_RULING = ROOT / "coordination" / "lane-b-milestone-ruling.json"
PRIMARY_CAMPAIGN = "qualification-01"
AUDIT_CAMPAIGN = "scientific-01"
AUDIT_DIR = "states"
CONDITIONS = ("clean", "ablate_A", "ablate_N", "ablate_unrelated")
REGIONS = ("A", "N", "unrelated")
ROW_FIELDS = (
    "row_sum_logprob",
    "x1_logprob",
    "y1_given_x1_logprob",
    "x1_y1_conditional_logprob",
)
ROW_OPEN = 151646
ROW_END = 151649
ATOL = 1e-5
PARITY_ATOL = 2e-4


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def stat_record(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "mtime_ns": stat.st_mtime_ns,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }


def file_record(path: Path, *, include_stat: bool = False) -> dict[str, Any]:
    result = binding(path)
    if include_stat:
        result["filesystem"] = stat_record(path)
    return result


def finite(value: Any) -> float:
    number = float(value)
    require(math.isfinite(number), f"non-finite numeric value: {value!r}")
    return number


def logsumexp(values: list[float]) -> float | None:
    if not values:
        return None
    peak = max(values)
    return peak + math.log(sum(math.exp(value - peak) for value in values))


def check_condition(condition: dict[str, Any], *, label: str) -> None:
    rows = condition.get("rows", {})
    require(rows, f"{label}: no rows")
    for row_id, row in rows.items():
        tokens = [int(token) for token in row.get("token_ids", [])]
        logps = [finite(value) for value in row.get("token_logprobs", [])]
        require(tokens and tokens[0] == ROW_OPEN, f"{label}/{row_id}: missing row entry")
        require(tokens[-1] == ROW_END, f"{label}/{row_id}: missing row terminator")
        require(len(tokens) == len(logps), f"{label}/{row_id}: token/logprob length mismatch")
        require(
            abs(sum(logps) - finite(row["row_sum_logprob"])) <= ATOL,
            f"{label}/{row_id}: complete-row sum mismatch",
        )
    for set_name, group in condition.get("candidate_sets", {}).items():
        row_ids = [str(item) for item in group.get("row_ids", [])]
        expected = group.get("row_logsumexp")
        observed = logsumexp([finite(rows[item]["row_sum_logprob"]) for item in row_ids])
        if expected is None:
            require(observed is None, f"{label}/{set_name}: unexpected candidate-set value")
        else:
            require(observed is not None, f"{label}/{set_name}: missing candidate-set value")
            require(
                abs(observed - finite(expected)) <= ATOL,
                f"{label}/{set_name}: full-vocabulary denominator changed",
            )


def source_for_state(admission: dict[str, Any], state_id: str) -> dict[str, Any]:
    states = [item for item in admission["lane_b"]["states"] if item["id"] == state_id]
    require(len(states) == 1, f"state is not unique in admission: {state_id}")
    return states[0]


def source_record(admission: dict[str, Any], boundary_id: str) -> dict[str, Any]:
    records = [item for item in admission["source_pool"] if item["source_boundary_id"] == boundary_id]
    require(len(records) == 1, f"source is not unique in admission: {boundary_id}")
    return records[0]


def expected_box(mask: dict[str, Any], size: tuple[int, int]) -> list[int]:
    x1, y1, x2, y2 = (int(value) for value in mask["pixel_xyxy"])
    width, height = size
    box = [max(0, x1), max(0, y1), min(width, x2), min(height, y2)]
    require(box[0] < box[2] and box[1] < box[3], f"empty clipped mask: {box}")
    return box


def decoded(path: Path) -> tuple[tuple[int, int], bytes]:
    with Image.open(path) as opened:
        image = opened.convert("RGB")
        return image.size, image.tobytes()


def check_mask_image(path: Path, box: list[int], size: tuple[int, int], label: str) -> None:
    with Image.open(path) as opened:
        mask = opened.convert("L")
        require(mask.size == size, f"{label}: mask dimensions changed")
        pixels = mask.load()
        x1, y1, x2, y2 = box
        for y in range(size[1]):
            for x in range(size[0]):
                expected = 255 if x1 <= x < x2 and y1 <= y < y2 else 0
                require(pixels[x, y] == expected, f"{label}: mask rectangle changed")


def check_compositor(
    score: dict[str, Any],
    admission: dict[str, Any],
    score_path: Path,
) -> dict[str, Any]:
    state = source_for_state(admission, score["state_id"])
    source = source_record(admission, score["source_boundary_id"])
    source_path = Path(source["image"]["path"])
    source_size, source_bytes = decoded(source_path)
    compositor = score["compositor"]
    require(set(compositor) == {"clean", "ablate_A", "ablate_N", "ablate_unrelated"}, f"{score_path}: compositor conditions changed")
    clean_image = Path(compositor["clean"]["image"]["path"])
    clean_size, clean_bytes = decoded(clean_image)
    require(clean_size == source_size and clean_bytes == source_bytes, f"{score_path}: clean decoded RGB changed")
    require(compositor["clean"]["decoded_size"] == list(source_size), f"{score_path}: clean dimensions changed")
    target_index = int(source["batch_index"])
    clean_grid = score["conditions"]["clean"]["input_identity"]["image_grids"][target_index]
    require(clean_grid == source["image_plan"]["observed_image_grid_thw"], f"{score_path}: clean grid changed")
    checks = []
    total_pixels = source_size[0] * source_size[1]
    for region in ("A", "N", "unrelated"):
        condition = f"ablate_{region}"
        comp = compositor[condition]
        box = expected_box(state["masks"][region], source_size)
        require(comp["decoded_size"] == list(source_size), f"{score_path}/{region}: dimensions changed")
        require(comp["pixel_xyxy"] == box, f"{score_path}/{region}: rectangle changed")
        require(comp["ring"] == {"radius_px": 8, "clipped": True, "excludes_mask": True, "rounding": "round-half-even"}, f"{score_path}/{region}: ring rule changed")
        image_path = Path(comp["image"]["path"])
        mask_path = Path(comp["mask"]["path"])
        altered_size, altered_bytes = decoded(image_path)
        require(altered_size == source_size, f"{score_path}/{region}: image dimensions changed")
        check_mask_image(mask_path, box, source_size, f"{score_path}/{region}")
        x1, y1, x2, y2 = box
        changed = 0
        outside_changed = 0
        with Image.open(source_path) as source_opened, Image.open(image_path) as altered_opened:
            source_rgb = source_opened.convert("RGB")
            altered_rgb = altered_opened.convert("RGB")
            source_pixels, altered_pixels = source_rgb.load(), altered_rgb.load()
            for y in range(source_size[1]):
                for x in range(source_size[0]):
                    differs = source_pixels[x, y] != altered_pixels[x, y]
                    if differs:
                        changed += 1
                        if not (x1 <= x < x2 and y1 <= y < y2):
                            outside_changed += 1
        require(changed > 0, f"{score_path}/{region}: changed mask is empty")
        require(outside_changed == 0, f"{score_path}/{region}: complement changed")
        require(comp["changed_pixel_count"] == changed, f"{score_path}/{region}: changed count mismatch")
        require(comp["unchanged_complement_pixel_count"] == total_pixels - (x2 - x1) * (y2 - y1), f"{score_path}/{region}: complement accounting mismatch")
        checks.append({"region": region, "image": file_record(image_path), "mask": file_record(mask_path), "pixel_xyxy": box, "changed_pixel_count": changed, "unchanged_complement_pixel_count": comp["unchanged_complement_pixel_count"]})
    return {"state_id": score["state_id"], "source_image": file_record(source_path), "clean_decoded_rgb_sha256": hashlib.sha256(source_bytes).hexdigest(), "clean_grid": clean_grid, "image_checks": checks}


def check_primary_score(score_path: Path, receipt_path: Path, admission: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    require(score_path.parent.parent.name == PRIMARY_CAMPAIGN, f"primary score escaped qualification path: {score_path}")
    require(score_path.parent.parent.parent == ROOT, f"primary score root changed: {score_path}")
    scores, receipt = load(score_path), load(receipt_path)
    require(receipt["status"] == "candidate_complete", f"primary receipt incomplete: {receipt_path}")
    require(receipt["campaign"] == PRIMARY_CAMPAIGN and receipt["mode"] == "qualify", f"primary campaign/mode changed: {receipt_path}")
    require(receipt.get("scores") == file_record(score_path), f"primary score binding changed: {score_path}")
    require(scores["campaign"] == PRIMARY_CAMPAIGN and scores["mode"] == "qualify", f"primary score campaign changed: {score_path}")
    require(scores["state_id"] == receipt["state_id"], f"primary state mismatch: {score_path}")
    require(set(scores["conditions"]) == set(CONDITIONS), f"primary conditions changed: {score_path}")
    for condition in CONDITIONS:
        check_condition(scores["conditions"][condition], label=f"{score_path}/{condition}")
    require(scores["source_native_replay_parity"]["passed"], f"native source parity failed: {score_path}")
    require(scores["source_native_replay_parity"]["entry_included"], f"entry omitted: {score_path}")
    require(scores["source_native_replay_parity"]["terminator_included"], f"terminator omitted: {score_path}")
    require(scores["clean_compositor_full_logit_parity"]["passed"], f"clean compositor parity failed: {score_path}")
    require(finite(scores["clean_compositor_full_logit_parity"]["max_abs_logit_error"]) <= PARITY_ATOL, f"clean logit parity tolerance failed: {score_path}")
    require(scores["actual_condition_difference"]["passed"], f"actual condition difference missing: {score_path}")
    image_check = check_compositor(scores, admission, score_path)
    return scores, receipt, image_check


def check_duplicate_equality(
    primary: dict[str, tuple[dict[str, Any], dict[str, Any]]],
    duplicate: dict[str, tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    per_state = []
    max_delta = 0.0
    compared_rows = 0
    for state_id in sorted(primary):
        primary_score, primary_receipt = primary[state_id]
        duplicate_score, duplicate_receipt = duplicate.get(state_id, (None, None))  # type: ignore[assignment]
        require(duplicate_score is not None and duplicate_receipt is not None, f"missing redundant state: {state_id}")
        state_rows = 0
        state_max = 0.0
        for condition in CONDITIONS:
            p_rows = primary_score["conditions"][condition]["rows"]
            d_rows = duplicate_score["conditions"][condition]["rows"]
            require(set(p_rows) == set(d_rows), f"duplicate row set changed: {state_id}/{condition}")
            for row_id in sorted(p_rows):
                for field in ROW_FIELDS:
                    delta = abs(finite(p_rows[row_id][field]) - finite(d_rows[row_id][field]))
                    state_max = max(state_max, delta)
                    max_delta = max(max_delta, delta)
                state_rows += 1
        require(state_rows == 16, f"duplicate comparison row count changed: {state_id}")
        compared_rows += state_rows
        per_state.append({"state_id": state_id, "compared_rows": state_rows, "max_abs_delta": state_max, "primary_scores": file_record(ROOT / PRIMARY_CAMPAIGN / state_id / "scores.json"), "duplicate_scores": file_record(ROOT / AUDIT_DIR / state_id / "scores.json"), "primary_receipt": file_record(ROOT / PRIMARY_CAMPAIGN / state_id / "receipt.json"), "duplicate_receipt": file_record(ROOT / AUDIT_DIR / state_id / "receipt.json")})
        require(primary_receipt["state_id"] == duplicate_receipt["state_id"] == state_id, f"duplicate receipt state mismatch: {state_id}")
    require(compared_rows == 64, f"expected 64 duplicate rows, got {compared_rows}")
    require(max_delta == 0.0, f"redundant score delta is not zero: {max_delta}")
    return {"compared_rows": compared_rows, "fields": list(ROW_FIELDS), "max_abs_delta": max_delta, "per_state": per_state, "disposition": "audit_only_redundant_pre_ruling_execution_excluded_from_primary_evidence_denominators_and_replication_claims"}


def sum_counters(receipts: list[dict[str, Any]]) -> dict[str, float | int]:
    keys = ("model_forwards", "vision_forwards", "candidate_rows", "gpu_seconds", "retained_tensor_bytes")
    totals: dict[str, float | int] = {}
    for key in keys:
        values = [receipt["counters"].get(key, 0) for receipt in receipts]
        if key in ("gpu_seconds",):
            totals[key] = sum(float(value) for value in values)
        else:
            totals[key] = sum(int(value) for value in values)
    return totals


def matrix_for_score(scores: dict[str, Any]) -> list[dict[str, Any]]:
    clean = scores["conditions"]["clean"]["rows"]
    candidate_sets = {
        str(row_id): set_name
        for set_name in ("A", "N")
        for row_id in scores["candidate_sets"][set_name]["row_ids"]
    }
    rows = []
    for row_id in sorted(candidate_sets):
        prefix_route = "native_reachable" if row_id == scores["actual_row_id"] else "supplied_candidate_under_native_prefix"
        regions: dict[str, dict[str, float]] = {}
        for region in REGIONS:
            altered = scores["conditions"][f"ablate_{region}"]["rows"][row_id]
            regions[region] = {field: finite(altered[field]) - finite(clean[row_id][field]) for field in ROW_FIELDS}
        rows.append({"candidate_id": row_id, "candidate_set": candidate_sets[row_id], "prefix_route": prefix_route, "regions": regions})
    return rows


def selectivity(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    return {
        "A_removal_A_effects": [row["regions"]["A"]["row_sum_logprob"] for row in rows if row["candidate_set"] == "A"],
        "A_removal_N_effects": [row["regions"]["A"]["row_sum_logprob"] for row in rows if row["candidate_set"] == "N"],
        "N_removal_A_effects": [row["regions"]["N"]["row_sum_logprob"] for row in rows if row["candidate_set"] == "A"],
        "N_removal_N_effects": [row["regions"]["N"]["row_sum_logprob"] for row in rows if row["candidate_set"] == "N"],
    }


def job_record(root: Path, directory: str, state_id: str, receipt: dict[str, Any]) -> dict[str, Any]:
    """Bind every pre-entry and terminal record for one saved job."""
    receipt_path = root / directory / state_id / "receipt.json"
    score_path = root / directory / state_id / "scores.json"
    launch_path = root / directory / state_id / "launch.json"
    snapshot_path = root / directory / state_id / "source-snapshot.json"
    return {
        "state_id": state_id,
        "campaign": receipt["campaign"],
        "model": receipt["model"],
        "device": receipt["device"],
        "status": receipt["status"],
        "job_state": "terminal",
        "scores": file_record(score_path),
        "receipt": file_record(receipt_path),
        "launch": file_record(launch_path),
        "source_snapshot": file_record(snapshot_path),
        "counters": receipt["counters"],
        "parity": receipt.get("parity"),
    }


def chronology(primary_receipts: list[dict[str, Any]], duplicate_receipts: list[dict[str, Any]]) -> dict[str, Any]:
    records = []
    for campaign, directory, receipts in ((PRIMARY_CAMPAIGN, PRIMARY_CAMPAIGN, primary_receipts), (AUDIT_CAMPAIGN, AUDIT_DIR, duplicate_receipts)):
        for receipt in receipts:
            state_id = receipt["state_id"]
            launch_path = ROOT / directory / state_id / "launch.json"
            receipt_path = ROOT / directory / state_id / "receipt.json"
            log_path = ROOT / "logs" / ("qualification-01" if campaign == PRIMARY_CAMPAIGN else "scientific") / f"{state_id}.log"
            records.append({"campaign": campaign, "state_id": state_id, "launch": file_record(launch_path, include_stat=True), "terminal_receipt": file_record(receipt_path, include_stat=True), "terminal_log": file_record(log_path, include_stat=True)})
    launch_times = [record["launch"]["filesystem"]["mtime_ns"] for record in records]
    terminal_times = [record["terminal_receipt"]["filesystem"]["mtime_ns"] for record in records]
    return {"basis": "retained launch.json, terminal receipt.json, and terminal log records with filesystem timestamps", "lead_timestamp_caveat": "The lead did not independently establish this chronology from score receipts alone; this record binds the retained launch and terminal records.", "earliest_launch_utc": datetime.fromtimestamp(min(launch_times) / 1e9, timezone.utc).isoformat(), "latest_terminal_receipt_utc": datetime.fromtimestamp(max(terminal_times) / 1e9, timezone.utc).isoformat(), "milestone_ruling": file_record(MILESTONE_RULING, include_stat=True), "redundant_replay_ruling": file_record(RULING, include_stat=True), "interpretation": "Retained records place the redundant scientific-01 launch and terminal records before the 08:03:12Z milestone ruling; this is recorded as pre-ruling redundant execution and is not characterized as a violation of the later instruction.", "records": records}


def artifact_bytes(directory: Path) -> int:
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def write_new(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    require(not path.exists(), f"refusing to overwrite existing versioned output: {path}")
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return file_record(path)


def producer_record() -> dict[str, Any]:
    return file_record(Path(__file__))


def run(root: Path) -> dict[str, Any]:
    require(root.resolve() == ROOT.resolve(), "v2 finalizer is bound to the frozen Lane B output root")
    outputs = {name: root / name for name in ("reduction-v2.json", "response-matrices-v2.json", "cpu-acceptance-v2.json", "candidate-manifest-v2.json")}
    for path in outputs.values():
        require(not path.exists(), f"versioned output already exists: {path}")
    require(ADMISSION.is_file() and ADMISSION_BINDING.is_file(), "frozen admission files missing")
    admission_binding = load(ADMISSION_BINDING)
    require(admission_binding["status"] == "frozen", "shared admission binding is not frozen")
    require(admission_binding["shared_admission"] == file_record(ADMISSION), "shared admission bytes changed")
    admission = load(ADMISSION)
    require(admission["status"] == "frozen_before_intervention_scores", "shared admission status changed")
    require(len(admission["lane_b"]["states"]) == 4, "Lane B state count changed")
    require(BUDGET.is_file() and WALL_LIMIT.is_file() and WALL_START.is_file() and RULING.is_file() and MILESTONE_RULING.is_file(), "required governance receipt missing")
    ruling = load(RULING)
    milestone_ruling = load(MILESTONE_RULING)
    require(ruling["primary_evidence"] == PRIMARY_CAMPAIGN and ruling["audit_only"] == "states/scientific-01", "lead ruling changed")
    require(ruling["score_max_abs_delta"] == 0, "lead ruling score delta changed")
    require(ruling["additional_model_calls_authorized"] is False, "lead ruling permits more model calls")
    require(milestone_ruling["qualification_cells_reused_as_primary"] is True and milestone_ruling["additional_model_calls_authorized"] is False and milestone_ruling["optional_release"] == "skipped", "lead milestone ruling changed")

    primary: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    duplicate: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    primary_images: dict[str, dict[str, Any]] = {}
    for score_path in sorted((root / PRIMARY_CAMPAIGN).glob("*/scores.json")):
        receipt_path = score_path.with_name("receipt.json")
        score, receipt, image_check = check_primary_score(score_path, receipt_path, admission)
        require(score["state_id"] not in primary, f"duplicate primary state: {score['state_id']}")
        primary[score["state_id"]] = (score, receipt)
        primary_images[score["state_id"]] = image_check
    require(set(primary) == {state["id"] for state in admission["lane_b"]["states"]}, "primary states do not match frozen admission")

    for score_path in sorted((root / AUDIT_DIR).glob("*/scores.json")):
        receipt_path = score_path.with_name("receipt.json")
        score, receipt = load(score_path), load(receipt_path)
        require(score["state_id"] not in duplicate, f"duplicate audit state: {score['state_id']}")
        require(receipt["status"] == "candidate_complete", f"audit receipt incomplete: {receipt_path}")
        require(receipt["campaign"] == AUDIT_CAMPAIGN and receipt["mode"] == "state", f"audit campaign/mode changed: {receipt_path}")
        require(receipt.get("scores") == file_record(score_path), f"audit score binding changed: {score_path}")
        duplicate[score["state_id"]] = (score, receipt)
    require(set(duplicate) == set(primary), "audit state set changed")
    equality = check_duplicate_equality(primary, duplicate)

    primary_receipts = [primary[state_id][1] for state_id in sorted(primary)]
    duplicate_receipts = [duplicate[state_id][1] for state_id in sorted(duplicate)]
    primary_cost = sum_counters(primary_receipts)
    duplicate_cost = sum_counters(duplicate_receipts)
    require(primary_cost["model_forwards"] == 80 and primary_cost["vision_forwards"] == 80, "primary forward count changed")
    require(duplicate_cost["model_forwards"] == 80 and duplicate_cost["vision_forwards"] == 80, "duplicate forward count changed")
    expected_duplicate = ruling["duplicate_cost"]
    require(duplicate_cost["model_forwards"] == expected_duplicate["model_forwards"], "duplicate model cost disagrees with ruling")
    require(duplicate_cost["vision_forwards"] == expected_duplicate["vision_forwards"], "duplicate vision cost disagrees with ruling")
    require(abs(float(duplicate_cost["gpu_seconds"]) - float(expected_duplicate["gpu_seconds"])) <= 1e-9, "duplicate GPU cost disagrees with ruling")

    matrices = []
    for state_id in sorted(primary):
        score, receipt = primary[state_id]
        rows = matrix_for_score(score)
        matrices.append({"state_id": state_id, "model": score["model"], "image_id": score["image_id"], "stratum": "control" if "control" in state_id else "target_first_revisit", "fixed_text_history": True, "candidate_prefix_labels": {row["candidate_id"]: row["prefix_route"] for row in rows}, "rows": rows, "selectivity": selectivity(rows), "primary_scores": file_record(root / PRIMARY_CAMPAIGN / state_id / "scores.json"), "primary_receipt": file_record(root / PRIMARY_CAMPAIGN / state_id / "receipt.json")})

    reduction_states = []
    for state_id in sorted(primary):
        score, receipt = primary[state_id]
        rows = []
        for matrix_row in matrix_for_score(score):
            rows.append({"candidate_id": matrix_row["candidate_id"], "candidate_set": matrix_row["candidate_set"], "prefix_route": matrix_row["prefix_route"], "region_effects": matrix_row["regions"]})
        reduction_states.append({"state_id": state_id, "model": score["model"], "image_id": score["image_id"], "stratum": "control" if "control" in state_id else "target_first_revisit", "response_matrix": rows, "selectivity": selectivity(matrix_for_score(score)), "parity": {"native": score["source_native_replay_parity"], "clean": score["clean_compositor_full_logit_parity"]}, "scores": file_record(root / PRIMARY_CAMPAIGN / state_id / "scores.json"), "receipt": file_record(root / PRIMARY_CAMPAIGN / state_id / "receipt.json")})

    primary_jobs = [job_record(root, PRIMARY_CAMPAIGN, state_id, primary[state_id][1]) for state_id in sorted(primary)]
    audit_jobs = []
    for state_id in sorted(duplicate):
        score, receipt = duplicate[state_id]
        audit_jobs.append({**job_record(root, AUDIT_DIR, state_id, receipt), "disposition": "redundant_pre_ruling_execution_excluded_from_primary_evidence_denominators_and_replication_claims"})

    admission_ref = file_record(ADMISSION_BINDING)
    common_governance = {"admission": admission_ref, "admission_sha256": file_record(ADMISSION)["sha256"], "budget_estimate": file_record(BUDGET), "wall_limit_amendment": file_record(WALL_LIMIT), "wall_start": file_record(WALL_START), "lead_milestone_ruling": file_record(MILESTONE_RULING), "lead_redundant_replay_ruling": file_record(RULING), "wall_limit_seconds": load(WALL_START)["wall_limit_seconds"]}
    chronology_record = chronology(primary_receipts, duplicate_receipts)
    superseded = {name: file_record(root / name) for name in ("candidate-manifest.json", "reduction.json", "response-matrices.json", "cpu-acceptance.json")}
    preentry_path = root / "qualification-00-preentry-failures.json"
    require(preentry_path.is_file(), "pre-entry failure receipt missing")

    reduction = {"schema": "visual_instance_binding.reduction.v2", "status": "candidate", "primary_campaign": PRIMARY_CAMPAIGN, "primary_denominator": {"states": 4, "cells": 16, "jobs": 4, "model_forwards": primary_cost["model_forwards"], "vision_forwards": primary_cost["vision_forwards"]}, "states": reduction_states, "failure_or_control_strata": {"target_first_revisit": sum(item["stratum"] == "target_first_revisit" for item in reduction_states), "control": sum(item["stratum"] == "control" for item in reduction_states)}, "counters": primary_cost, "failed_attempts": [{"receipt": file_record(preentry_path), "status": "technical_invalid_pre_entry", "model_entry": False, "disposition": "superseded_by_qualification-01"}], "response_matrix_definition": "logP(candidate | region ablated) - logP(candidate | clean), complete row including entry and terminator; x1 then y1 given candidate x1 fields are retained per row", "candidate_set_accounting": "full vocabulary denominator with entry and terminator included; no candidate renormalization", "audit_only_redundant_execution": {"campaign": AUDIT_CAMPAIGN, "root": str(root / AUDIT_DIR), "jobs": audit_jobs, "cells": 16, "counters": duplicate_cost, "score_equality": equality, "chronology": chronology_record}, "governance": common_governance, "interpretation": "Finite visual binding candidate from qualification-01 only; A/N interventions show relative local dependence, with no physical-owner or training-origin claim.", "producer": producer_record()}

    response_matrices = {"schema": "visual_instance_binding.response_matrices.v2", "status": "candidate", "primary_campaign": PRIMARY_CAMPAIGN, "definition": "Each region entry reports ablated minus clean log probability for the same candidate row under the fixed native text prefix.", "columns": ["x1", "y1_given_candidate_x1", "joint_corner", "full_row"], "field_bindings": {"x1": "x1_logprob", "y1_given_candidate_x1": "y1_given_x1_logprob", "joint_corner": "x1_y1_conditional_logprob", "full_row": "row_sum_logprob"}, "prefix_route_labels": {"native_reachable": "the frozen native next row at the boundary", "supplied_candidate_under_native_prefix": "a frozen A/N candidate scored conditionally after the same native prefix"}, "states": matrices, "audit_exclusion": "scientific-01 states/ is excluded from these primary matrices; its 64-row exact equality check is recorded in reduction-v2 and cpu-acceptance-v2.", "governance": common_governance, "producer": producer_record()}

    cpu_states = []
    for state_id in sorted(primary):
        score = primary[state_id][0]
        cpu_states.append({"state_id": state_id, "primary_scores": file_record(root / PRIMARY_CAMPAIGN / state_id / "scores.json"), "primary_receipt": file_record(root / PRIMARY_CAMPAIGN / state_id / "receipt.json"), **primary_images[state_id], "conditions": list(score["conditions"]), "candidate_count": sum(len(group["row_ids"]) for group in score["candidate_sets"].values())})
    cpu_acceptance = {"schema": "visual_instance_binding.cpu_acceptance.v2", "status": "passed", "primary_campaign": PRIMARY_CAMPAIGN, "primary_states": cpu_states, "primary_assertions": {"decoded_rgb_clean_equal_native": True, "dimensions_and_grid_verified": True, "exact_mask_rectangle_verified": True, "nonempty_changed_mask_verified": True, "unchanged_complement_verified": True, "full_row_entry_and_terminator_verified": True, "candidate_set_full_vocab_accounting_verified": True, "source_replay_atol": PARITY_ATOL, "clean_full_logit_atol": PARITY_ATOL}, "audit_only_redundant_execution": {"root": str(root / AUDIT_DIR), "campaign": AUDIT_CAMPAIGN, "score_equality": equality, "cost": duplicate_cost, "disposition": "excluded_from_primary_cpu_acceptance_and_scientific_denominators"}, "governance": common_governance, "producer": producer_record()}

    candidate = {"schema": "visual_instance_binding.candidate_manifest.v2", "status": "candidate", "lane": "B_visual_instance_binding", "question": admission["lane_b"]["question"], "primary_evidence": {"campaign": PRIMARY_CAMPAIGN, "cells": 16, "states": 4, "jobs": 4, "denominator_rule": "qualification-01 only", "jobs_terminal": all(receipt["status"] == "candidate_complete" for receipt in primary_receipts), "job_records": primary_jobs, "cost": primary_cost, "response_matrices": "response-matrices-v2.json", "reduction": "reduction-v2.json", "cpu_acceptance": "cpu-acceptance-v2.json"}, "audit_only_redundant_execution": {"campaign": AUDIT_CAMPAIGN, "root": str(root / AUDIT_DIR), "cells": 16, "jobs": 4, "jobs_terminal": all(receipt["status"] == "candidate_complete" for receipt in duplicate_receipts), "job_records": audit_jobs, "cost": duplicate_cost, "score_equality": equality, "disposition": "pre-ruling redundant execution debt; excluded from scientific evidence, denominators, selection, aggregation, replication claims, and response matrices", "chronology": chronology_record}, "cost": {"primary": primary_cost, "audit_only_redundant": duplicate_cost, "combined_executed": {key: primary_cost[key] + duplicate_cost[key] for key in primary_cost}}, "governance": common_governance, "superseded_unversioned_outputs": superseded, "preentry_failure": file_record(preentry_path), "holds": ["Optional release skipped under lead ruling; no further model calls authorized.", "Finite candidate rows and matched control are observational; H1, physical-owner recovery, and training origin remain HOLD."], "interpretation": {"observation": "Across the four primary states, A-removal and N-removal effects are reported relatively at the complete-row, x1, y1-given-x1, and joint-corner paths.", "inference": "The finite qualification cells show separated local dependence for A and N interventions.", "claim_limit": "This does not establish H1, a physical owner, or training origin."}, "artifact_bytes": {"primary_run_root": artifact_bytes(root / PRIMARY_CAMPAIGN), "audit_only_run_root": artifact_bytes(root / AUDIT_DIR), "combined_run_roots": artifact_bytes(root / PRIMARY_CAMPAIGN) + artifact_bytes(root / AUDIT_DIR)}, "outputs": {"reduction": "reduction-v2.json", "response_matrices": "response-matrices-v2.json", "cpu_acceptance": "cpu-acceptance-v2.json"}, "commands": ["python -m probes.training_set_completion.visual_instance_binding.selfcheck", "python -m probes.training_set_completion.visual_instance_binding.finalize_v2 --root /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-21-visual-instance-binding"], "producer": producer_record()}

    reduction_ref = write_new(outputs["reduction-v2.json"], reduction)
    response_ref = write_new(outputs["response-matrices-v2.json"], response_matrices)
    cpu_ref = write_new(outputs["cpu-acceptance-v2.json"], cpu_acceptance)
    candidate["outputs"]["reduction_binding"] = reduction_ref
    candidate["outputs"]["response_matrices_binding"] = response_ref
    candidate["outputs"]["cpu_acceptance_binding"] = cpu_ref
    candidate_ref = write_new(outputs["candidate-manifest-v2.json"], candidate)
    return {"status": "candidate", "primary_cost": primary_cost, "audit_cost": duplicate_cost, "duplicate_score_max_abs_delta": equality["max_abs_delta"], "outputs": {name: file_record(path) for name, path in outputs.items()}, "candidate_binding": candidate_ref}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    result = run(args.root)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
