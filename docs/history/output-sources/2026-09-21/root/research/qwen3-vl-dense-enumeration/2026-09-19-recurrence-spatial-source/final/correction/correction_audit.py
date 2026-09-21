#!/usr/bin/env python3
"""CPU audit of the frozen 45-state candidate against corrected Lane-B semantics."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

WORKTREE = Path("/data/CoordExp/.worktrees/research-probes")
ARTIFACT_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-19-recurrence-spatial-source"
)
FINAL = ARTIFACT_ROOT / "final"
CORRECTION = FINAL / "correction"
INDEX_PATH = FINAL / "execution-index.json"
PANEL_SHA = "005bc6deff209bc96ce469e8cbc20d3dbd30a7e424a7ff18a0f7f8942a1bdb49"
SHIFT_PX = 128
COORD_BASE = 151670
COORD_LIMIT = COORD_BASE + 1000
OBJ_START = 151646
OBJ_END = 151647
BOX_START = 151648
BOX_END = 151649
ROW_LIMIT = 32
sys.path.insert(0, str(WORKTREE))

from probes.training_set_completion.recurrence_spatial import prepare  # noqa: E402
from probes.training_set_completion.recurrence_spatial.recurrence_semantics import complete_box_count, parse_rows  # noqa: E402


class DummyTokenizer:
    def decode(self, tokens, **kwargs):
        return ""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binding(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size}


def path_value(value: Any) -> Path:
    if isinstance(value, dict):
        value = value["path"]
    return Path(str(value)).resolve(strict=True)


def runtime_for_state(state_ref: dict[str, Any]) -> dict[str, Any]:
    """Join the two retained pilot runtime receipts into one cell view."""

    if "runtime_path" in state_ref:
        return json.loads(path_value(state_ref["runtime_path"]).read_text())
    runtime = state_ref.get("runtime")
    if not isinstance(runtime, dict) or "center" not in runtime or "transforms" not in runtime:
        raise AssertionError(f"{state_ref.get('id')}: no runtime binding")
    center = json.loads(path_value(runtime["center"]).read_text())
    transforms = json.loads(path_value(runtime["transforms"]).read_text())
    merged = dict(center)
    merged["cells"] = dict(center.get("cells", {}))
    merged["cells"].update(transforms.get("cells", {}))
    merged["joined_runtime_receipts"] = [runtime["center"], runtime["transforms"]]
    return merged


def logsumexp(values: list[float]) -> float:
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def source_tokens(manifest: dict[str, Any]) -> list[int]:
    raw_path = path_value(manifest["source"]["mature_raw"])
    raw = json.loads(raw_path.read_text())
    image_id = int(manifest["source"]["image_id"])
    rows = [row for row in raw["rows"] if int(row["image_id"]) == image_id]
    if len(rows) != 1:
        raise AssertionError(f"raw source identity mismatch for {manifest['state_id']}")
    return [int(value) for value in rows[0]["token_ids"]]


def row_starts(tokens: list[int]) -> list[int]:
    return [index for index, token in enumerate(tokens) if token == OBJ_START]


def next_x1(manifest: dict[str, Any], tokens: list[int]) -> int:
    starts = row_starts(tokens)
    next_index = int(manifest["prefix"]["source_row_index"]) + 1
    if next_index >= len(starts):
        raise AssertionError(f"no next row for {manifest['state_id']}")
    stop = starts[next_index + 1] if next_index + 1 < len(starts) else len(tokens)
    values = [value - COORD_BASE for value in tokens[starts[next_index] : stop] if COORD_BASE <= value < COORD_LIMIT]
    if len(values) != 4:
        raise AssertionError(f"next row has {len(values)} coords for {manifest['state_id']}")
    return int(values[0])


def windows(log_probs: list[float], center: int, radius: int = 8) -> dict[str, Any]:
    lo, hi = max(0, center - radius), min(999, center + radius)
    values = log_probs[lo : hi + 1]
    return {
        "lo": lo,
        "hi": hi,
        "log_mass": logsumexp(values),
        "mass": math.exp(logsumexp(values)),
        "max_log_prob": max(values),
    }


def utc_seconds(start: str, end: str) -> int:
    parse = lambda value: datetime.fromisoformat(value.replace("Z", "+00:00"))
    return int((parse(end) - parse(start)).total_seconds())


def main() -> None:
    index = json.loads(INDEX_PATH.read_text())
    states = index["states"]
    invariant_failures: list[str] = []
    history_counts = Counter()
    row_counts = Counter()
    predicate_counts = Counter()
    window_counts = Counter()
    state_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    fixed_windows: list[dict[str, Any]] = []
    termination = Counter()
    by_cell = Counter()

    if len(states) != 45 or int(index["declared_states"]) != 45:
        invariant_failures.append(f"declared state count is {len(states)}")

    for state_ref in states:
        state_id = str(state_ref["id"])
        manifest = json.loads(path_value(state_ref["manifest"]).read_text())
        runtime = runtime_for_state(state_ref)
        geometry = manifest["geometry"]
        source_width = int(geometry["source_width"])
        source_height = int(geometry["source_height"])
        canvas_width = int(geometry["canvas_width"])
        canvas_height = int(geometry["canvas_height"])
        if manifest.get("state_id") not in (None, state_id):
            invariant_failures.append(f"{state_id}: manifest state id drift")
        if state_ref.get("mode") == "reuse_pilot":
            if manifest["source"].get("feedback_panel_sha256") is None:
                invariant_failures.append(f"{state_id}: pilot feedback panel binding missing")
        elif manifest["source"].get("shared_panel", {}).get("sha256") != PANEL_SHA:
            invariant_failures.append(f"{state_id}: panel hash drift")
        if canvas_width != source_width + 2 * SHIFT_PX or canvas_height != source_height:
            invariant_failures.append(f"{state_id}: canvas geometry drift")
        if any(value % 32 for value in (source_width, source_height, canvas_width, canvas_height)):
            invariant_failures.append(f"{state_id}: processor grid divisibility drift")
        tokens = source_tokens(manifest)
        starts = row_starts(tokens)
        source_row_index = int(manifest["prefix"]["source_row_index"])
        prefix_end = int(manifest["prefix"]["source_row_end"])
        if source_row_index >= len(starts) or starts[source_row_index + 1] != prefix_end:
            invariant_failures.append(f"{state_id}: selected prefix boundary drift")
        prefix = tokens[:prefix_end]
        if len(prefix) != int(manifest["prefix"]["source_token_count"]):
            invariant_failures.append(f"{state_id}: source prefix token count drift")
        nx1 = next_x1(manifest, tokens)
        state_cell_count = 0
        state_history_rows = 0
        state_history_y_changed = 0
        state_history_x_changed = 0
        state_history_noncoord_drift = 0
        state_complete_old = 0
        state_complete_new = 0
        state_truncated = 0
        state_predicate_changed = 0
        state_inverse_changed = 0
        state_window_wrong = 0
        state_window_records = 0
        image_sizes_ok = 0
        for key, cell in manifest["cells"].items():
            state_cell_count += 1
            image_path = path_value(cell["image_path"])
            if binding(image_path) != cell["image"]:
                invariant_failures.append(f"{state_id}/{key}: image binding drift")
            with Image.open(image_path) as image:
                if image.size != (canvas_width, canvas_height):
                    invariant_failures.append(f"{state_id}/{key}: image canvas size drift")
                else:
                    image_sizes_ok += 1
            history = [int(value) for value in cell["history"]]
            if len(history) != len(prefix):
                invariant_failures.append(f"{state_id}/{key}: history token count drift")
            source_coordinate_positions = [index for index, value in enumerate(prefix) if COORD_BASE <= value < COORD_LIMIT]
            history_coordinate_positions = [index for index, value in enumerate(history) if COORD_BASE <= value < COORD_LIMIT]
            if source_coordinate_positions != history_coordinate_positions:
                invariant_failures.append(f"{state_id}/{key}: coordinate token position drift")
            for index, (left, right) in enumerate(zip(prefix, history, strict=True)):
                if index not in source_coordinate_positions and left != right:
                    state_history_noncoord_drift += 1
            tx = int(cell["history_offset_px"])
            for box in cell.get("history_boxes", []):
                state_history_rows += 1
                source = [int(value) for value in box["source_bins"]]
                expected = [
                    prepare.map_coordinate(
                        value,
                        coordinate_index=coordinate_index,
                        source_width=source_width,
                        source_height=source_height,
                        canvas_width=canvas_width,
                        canvas_height=canvas_height,
                        tx=tx,
                        ty=0,
                    )
                    for coordinate_index, value in enumerate(source)
                ]
                actual = [int(value) for value in box["mapped_bins"]]
                if actual[0] != expected[0] or actual[2] != expected[2]:
                    state_history_x_changed += 1
                if actual[1] != expected[1]:
                    state_history_y_changed += 1
                if actual[3] != expected[3]:
                    state_history_y_changed += 1
                if any(value < 0 or value > 999 for value in expected):
                    invariant_failures.append(f"{state_id}/{key}: corrected history map clipped")
            runtime_cell = runtime.get("cells", {}).get(key)
            if runtime_cell is None:
                continue
            state_cell_count += 0
            state_rows_current = runtime_cell.get("free", {}).get("parse", {})
            token_ids = [int(value) for value in runtime_cell["free"]["token_ids"]]
            corrected = parse_rows(
                token_ids,
                DummyTokenizer(),
                cell=cell,
                geometry=geometry,
            )
            current_complete = int(state_rows_current.get("complete_rows", -1))
            corrected_complete = int(corrected["complete_rows"])
            serialized_complete = complete_box_count(token_ids)
            if serialized_complete != corrected_complete:
                invariant_failures.append(f"{state_id}/{key}: complete-box counter/parser drift")
            state_complete_old += current_complete
            state_complete_new += corrected_complete
            row_counts["cells_with_complete_count_change"] += current_complete != corrected_complete
            row_counts["current_complete_rows"] += current_complete
            row_counts["corrected_complete_rows"] += corrected_complete
            row_counts["serialized_complete_box_rows"] += serialized_complete
            row_counts["current_box_end_tokens"] += token_ids.count(BOX_END)
            row_limit_receipt = runtime_cell.get("free", {}).get("row_limit")
            if isinstance(row_limit_receipt, dict):
                row_counts["row_cap_provenance_present_cells"] += 1
                if row_limit_receipt.get("mode") != "free_suffix_complete_row_cap":
                    invariant_failures.append(f"{state_id}/{key}: unknown row-cap provenance mode")
            else:
                # Candidate-v1 was produced before the corrected producer
                # exposed the free-boundary/injected-EOS receipt.  Its
                # historical row-cap class below is therefore a parser
                # reconstruction, not a native stop-reason claim.
                row_counts["row_cap_provenance_missing_cells"] += 1
            if current_complete != corrected_complete:
                row_counts["complete_count_changed_cells"] += 1
            starts_free = row_starts(token_ids)
            truncated_segments = 0
            for row_index, start in enumerate(starts_free):
                stop = starts_free[row_index + 1] if row_index + 1 < len(starts_free) else len(token_ids)
                segment = token_ids[start:stop]
                if OBJ_END in segment and BOX_END not in segment:
                    truncated_segments += 1
            state_truncated += truncated_segments
            row_counts["truncated_description_without_box_rows"] += truncated_segments
            row_counts["cells_with_truncated_row"] += bool(truncated_segments)
            old_cap = current_complete >= ROW_LIMIT
            new_cap = corrected_complete >= ROW_LIMIT
            row_counts["current_row_cap_cells"] += old_cap
            row_counts["corrected_row_cap_cells"] += new_cap
            row_counts["row_cap_class_changed_cells"] += old_cap != new_cap
            termination["corrected_row_cap" if new_cap else "natural_or_other"] += 1
            by_cell[f"{key}|corrected_row_cap" if new_cap else f"{key}|natural_or_other"] += 1

            current_rows = state_rows_current.get("rows", [])
            for current_row in current_rows:
                if current_row.get("status") in {"invalid", "malformed"} or not current_row.get("source_geometry_valid", False):
                    row_counts["current_rows_unusable_for_runs"] += 1
            for corrected_row in corrected["rows"]:
                if corrected_row.get("status") in {"invalid", "malformed"} or not corrected_row.get("source_geometry_valid", False):
                    row_counts["corrected_rows_invalid_or_out_of_source"] += 1
            current_predicate = bool(state_rows_current.get("failure_predicate", False))
            corrected_predicate = bool(corrected.get("failure_predicate", False))
            predicate_counts["current_true"] += current_predicate
            predicate_counts["corrected_true"] += corrected_predicate
            predicate_counts["changed_cells"] += current_predicate != corrected_predicate
            predicate_counts["current_exact_true"] += bool(state_rows_current.get("exact_runs"))
            predicate_counts["corrected_exact_true"] += bool(corrected.get("exact_runs"))
            predicate_counts["current_near_true"] += bool(state_rows_current.get("near_runs"))
            predicate_counts["corrected_near_true"] += bool(corrected.get("near_runs"))
            predicate_counts["current_near_runs"] += len(state_rows_current.get("near_runs", []))
            predicate_counts["corrected_near_witnesses"] += len(corrected.get("near_runs", []))
            state_predicate_changed += current_predicate != corrected_predicate
            if current_predicate != corrected_predicate:
                predicate_counts[f"changed|{key}"] += 1
            # Compare source-coordinate inverses for aligned complete rows.
            for current_row, corrected_row in zip(
                [row for row in current_rows if row.get("status") != "malformed"],
                [row for row in corrected["rows"] if row.get("status") != "malformed"],
                strict=False,
            ):
                if current_row.get("coord_bins_source") != corrected_row.get("coord_bins_source"):
                    state_inverse_changed += 1
                    row_counts["rows_with_role_aware_inverse_change"] += 1

            forced = runtime_cell.get("boundary", {}).get("forced_description_x1")
            if forced is not None and len(forced.get("coordinate_log_probs", [])) == 1000:
                state_window_records += 1
                state_window_records += 0
                log_probs = [float(value) for value in forced["coordinate_log_probs"]]
                old_bin = prepare.map_bin(nx1, source_width=source_width, canvas_width=canvas_width, tx=SHIFT_PX)
                current_moved = int(forced["windows"]["moved_bin"])
                current_old = int(forced["windows"]["old_bin"])
                # Recompute both fixed sign windows from the one saved full
                # vocabulary capture.  The historical scalar is comparable
                # to the cell's old selected sign, and to both signs for 00.
                selected_sign = None if key == "00" else key[-1]
                for sign, moved_tx in (("-", 0), ("+", 2 * SHIFT_PX)):
                    expected_moved = prepare.map_bin(nx1, source_width=source_width, canvas_width=canvas_width, tx=moved_tx)
                    old_window = windows(log_probs, old_bin)
                    moved_window = windows(log_probs, expected_moved)
                    comparable = key == "00" or selected_sign == sign
                    fixed_windows.append(
                        {
                            "state_id": state_id,
                            "cell": key,
                            "sign": sign,
                            "old_bin": old_bin,
                            "current_old_bin": current_old,
                            "expected_moved_bin": expected_moved,
                            "current_moved_bin": current_moved,
                            "historical_scalar_comparable": comparable,
                            "current_window_radius": forced["windows"].get("window_radius"),
                            "old_window": old_window,
                            "moved_window": moved_window,
                            "moved_minus_old_log_mass": moved_window["log_mass"] - old_window["log_mass"],
                        }
                    )
                    window_counts["records"] += 1
                    window_counts["current_single_bin_records"] += "window_radius" not in forced["windows"]
                    window_counts["historical_scalar_comparable_records"] += comparable
                    if comparable:
                        window_counts["current_moved_bin_wrong"] += current_moved != expected_moved
                        window_counts["current_same_as_old_but_should_move"] += current_moved == old_bin and expected_moved != old_bin
                        window_counts["current_different_but_should_stay"] += current_moved != old_bin and expected_moved == old_bin
                        window_counts[f"wrong|{key}"] += current_moved != expected_moved
                        state_window_wrong += current_moved != expected_moved
            cell_rows.append(
                {
                    "state_id": state_id,
                    "cell": key,
                    "current_complete_rows": current_complete,
                    "corrected_complete_rows": corrected_complete,
                    "truncated_description_without_box_rows": truncated_segments,
                    "current_failure_predicate": current_predicate,
                    "corrected_failure_predicate": corrected_predicate,
                    "role_aware_inverse_changed_rows": state_inverse_changed,
                    "row_cap_provenance": (
                        "present" if isinstance(row_limit_receipt, dict) else "missing_in_candidate_v1"
                    ),
                }
            )
        if state_cell_count != 7:
            invariant_failures.append(f"{state_id}: manifest cell count {state_cell_count}")
        if state_history_noncoord_drift:
            invariant_failures.append(f"{state_id}: {state_history_noncoord_drift} non-coordinate history tokens changed")
        history_counts["state_count"] += 1
        history_counts["history_rows"] += state_history_rows
        history_counts["history_y_coordinate_changes"] += state_history_y_changed
        history_counts["history_x_box_changes"] += state_history_x_changed
        history_counts["image_sizes_verified"] += image_sizes_ok
        history_counts["history_noncoord_drift"] += state_history_noncoord_drift
        state_rows.append(
            {
                "state_id": state_id,
                "model": state_ref["model"],
                "kind": state_ref["kind"],
                "manifest_cells": state_cell_count,
                "runtime_cells": len(runtime.get("cells", {})),
                "history_rows": state_history_rows,
                "history_y_coordinate_changes": state_history_y_changed,
                "complete_rows_current_sum": state_complete_old,
                "complete_rows_corrected_sum": state_complete_new,
                "truncated_description_without_box_rows": state_truncated,
                "predicate_changed_cells": state_predicate_changed,
                "forced_window_wrong_cells": state_window_wrong,
            }
        )

    package_start = "2026-09-19T06:56:12Z"
    first_final_wave_start = "2026-09-19T08:17:19Z"
    final_enclosing_end = "2026-09-19T08:34:23Z"
    a_enclosing_end = "2026-09-19T08:03:39Z"
    d_enclosing_end = "2026-09-19T08:37:50Z"
    b_pre_seconds = utc_seconds(package_start, first_final_wave_start)
    b_final_seconds = utc_seconds(first_final_wave_start, final_enclosing_end)
    a_seconds = utc_seconds(package_start, a_enclosing_end)
    d_seconds = utc_seconds(package_start, d_enclosing_end)
    b_gpu_seconds = 2 * b_pre_seconds + 6 * b_final_seconds
    a_gpu_seconds = 4 * a_seconds
    d_gpu_seconds = d_seconds
    reserved_c_gpu_seconds = 4 * 60 * 60
    package_ceiling_gpu_seconds = 24 * 60 * 60
    prior_upper_gpu_seconds = b_gpu_seconds + a_gpu_seconds + d_gpu_seconds
    reserved_and_prior_gpu_seconds = prior_upper_gpu_seconds + reserved_c_gpu_seconds

    # The input identity audit is independent of the corrected output parser:
    # every state had one bound source, common canvas dimensions, seven cells,
    # and a preserved non-coordinate prefix stream before this pass.
    receipt = {
        "schema": "recurrence_spatial_source.correction_audit.v1",
        "unit_id": "2026-09-19-recurrence-spatial-source",
        "status": "cpu_correction_gate_passed" if not invariant_failures else "cpu_correction_gate_failed",
        "model_work_launched": False,
        "gpu_forwards": 0,
        "candidate_v1_snapshot": binding(ARTIFACT_ROOT.parent / "2026-09-19-recurrence-spatial-source-candidate-v1" / "candidate-v1-snapshot-receipt.json"),
        "panel": {"sha256": PANEL_SHA, "declared_states": 45, "failure": 21, "proxy": 24},
        "input_invariants": {
            "states_checked": history_counts["state_count"],
            "manifests_with_seven_cells": sum(item["manifest_cells"] == 7 for item in state_rows),
            "runtime_cells": sum(item["runtime_cells"] for item in state_rows),
            "history_rows_checked": history_counts["history_rows"],
            "images_size_verified": history_counts["image_sizes_verified"],
            "noncoordinate_history_token_drift": history_counts["history_noncoord_drift"],
            "panel_hash_all_new_states": not any("panel hash drift" in failure for failure in invariant_failures),
            "invariant_failures": invariant_failures,
        },
        "affected_cell_counts": {
            "history": dict(history_counts),
            "row_stop": dict(row_counts),
            "recurrence_predicate": dict(predicate_counts),
            "forced_windows": dict(window_counts),
        },
        "corrected_termination": {
            "cell_count": sum(termination.values()),
            "row_cap_cells_by_corrected_complete_serialized_row": termination["corrected_row_cap"],
            "natural_or_other_cells": termination["natural_or_other"],
            "by_cell": dict(by_cell),
        },
        "state_rows": state_rows,
        "cell_rows": cell_rows,
        "fixed_sign_windows_from_saved_logits": fixed_windows,
        "semantics": {
            "history_map": "x1/x2 use width and signed tx; y1/y2 use height and ty=0",
            "output_inverse": "x1/x2 use width and visual tx; y1/y2 use height and visual ty=0",
            "complete_row": "OBJ_START, description OBJ_END, BOX_START, exactly four coordinate bins, BOX_END",
            "row_cap": "BOX_END count, including invalid or out-of-source complete boxes",
            "recurrence": "accepted numerical_feedback.same primitive on the three pairs of each consecutive complete-row triple; exact then <=8-bin near; invalid/out-of-source complete rows retained",
            "forced_window": "old centered and moved sign-specific x1 windows are fixed +/-8 bins over saved full-vocabulary coordinate log probabilities for both signs in every cell (00 stores both); no window renormalization",
        },
        "historical_occupancy_bound": {
            "status": "conservative_enclosing_interval_receipt",
            "package_start_utc": package_start,
            "ceiling_gpu_hours": 24.0,
            "device_ownership": {
                "b_pre_final": {
                    "devices": [4, 5],
                    "start_utc": package_start,
                    "end_utc": first_final_wave_start,
                    "upper_bound_wall_seconds": b_pre_seconds,
                    "upper_bound_gpu_seconds": 2 * b_pre_seconds,
                    "basis": "enclosing B pilot interval; saved B launch/runtime receipts use cuda:4 and cuda:5",
                },
                "b_final_enclosing": {
                    "devices": [0, 1, 2, 3, 4, 5],
                    "start_utc": first_final_wave_start,
                    "end_utc": final_enclosing_end,
                    "upper_bound_wall_seconds": b_final_seconds,
                    "upper_bound_gpu_seconds": 6 * b_final_seconds,
                    "basis": "encloses both final waves and the intervening gap, intentionally conservative",
                },
                "a_enclosing": {
                    "devices": [0, 1, 2, 3],
                    "start_utc": package_start,
                    "end_utc": a_enclosing_end,
                    "upper_bound_wall_seconds": a_seconds,
                    "upper_bound_gpu_seconds": a_gpu_seconds,
                    "basis": "enclosing A worker interval from parent-supplied worker receipts",
                },
                "d_enclosing": {
                    "devices": [7],
                    "start_utc": package_start,
                    "end_utc": d_enclosing_end,
                    "upper_bound_wall_seconds": d_seconds,
                    "upper_bound_gpu_seconds": d_gpu_seconds,
                    "basis": "enclosing D producer interval; GPU7 was released by B closure and assigned to D",
                },
            },
            "earlier_pilot_wall_seconds_recorded": 725.0,
            "gpu7_b_owned_pre_final": False,
            "gpu7_evidence": "B receipts observed devices 4/5 before the final wave; final/job-closure.json releases GPU7 with D-owned work, so no B GPU7 interval is counted",
            "prior_upper_bound_gpu_seconds": prior_upper_gpu_seconds,
            "prior_upper_bound_gpu_hours": prior_upper_gpu_seconds / 3600.0,
            "reserved_c_gpu_hours": reserved_c_gpu_seconds / 3600.0,
            "prior_plus_reserved_c_gpu_seconds": reserved_and_prior_gpu_seconds,
            "prior_plus_reserved_c_gpu_hours": reserved_and_prior_gpu_seconds / 3600.0,
            "remaining_upper_budget_gpu_seconds": package_ceiling_gpu_seconds - reserved_and_prior_gpu_seconds,
            "remaining_upper_budget_gpu_hours": (package_ceiling_gpu_seconds - reserved_and_prior_gpu_seconds) / 3600.0,
            "missing_intervals_counted_as_zero": False,
            "gpu_time_instrumentation": "producer did not record device time; all values are enclosing reservation upper bounds, not measured GPU utilization",
        },
        "acceptance_commands": [
            "python3 /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/correction/red_witness.py",
            "python3 /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/correction/green_witness.py",
            "python3 /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/correction/correction_audit.py",
            "python3 -m py_compile probes/training_set_completion/recurrence_spatial/*.py",
            "python3 -m probes.training_set_completion.recurrence_spatial.reduce --model untied --manifest-path /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/manifests/untied-885-healthy.json --runtime-path /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/runtime/untied-885-healthy.json --reduced-path /tmp/recur-spatial-corrected-reduced-885.json",
        ],
        "artifacts": {
            "red_witness": str(CORRECTION / "red-witness.json"),
            "green_witness": str(CORRECTION / "green-witness.json"),
            "correction_audit": str(CORRECTION / "correction-receipt.json"),
            "fixed_window_table": str(CORRECTION / "correction-receipt.json"),
            "saved_reducer_receipt": str(CORRECTION / "saved-reducer-receipt.json"),
            "saved_reducer_output": str(CORRECTION / "saved-reducer-untied-885-healthy.json"),
        },
        "stop": {
            "no_gpu_rerun": True,
            "no_replacement_states": True,
            "parent_release_required_for_corrected_pilot": True,
        },
    }
    CORRECTION.mkdir(parents=True, exist_ok=True)
    (CORRECTION / "correction-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    if invariant_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
