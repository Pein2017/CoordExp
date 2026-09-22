from __future__ import annotations

import json
from pathlib import Path

from probes.training_set_completion.coordinate_address_readout.reduce_production import reduce_production


def _native(row_id: str, condition: str, *, duplicate: bool = False, status: str | None = None) -> dict:
    predictions = [
        {"generated_order": 0, "description": "person", "coord_bins": [0, 0, 500, 500]},
        {"generated_order": 1, "description": "person", "coord_bins": [0, 0, 500, 500]},
        {"generated_order": 2, "description": "car", "coord_bins": [700, 700, 800, 800]},
    ]
    if duplicate:
        predictions = predictions[:1]
    result = {"row_id": row_id, "condition": condition, "token_ids": [1, 2, 3, 4], "stop_reason": "length", "cap": 4, "parsed": {"predictions": predictions, "dropped_predictions": []}}
    if status is not None:
        result["status"] = status
        if status == "HOLD":
            result.pop("parsed")
            result["partial_generation"] = {"token_ids": [1, 2]}
    return result


def test_reduction_keeps_unknown_separate_and_holds_missing_cells(tmp_path: Path) -> None:
    case = {"row_id": "r1", "image_id": 1, "cohort": "fresh32", "input_record": {"image_id": 1, "objects": [{"coco_ann_id": 11, "desc": "person", "bbox_2d": ["<|coord_0|>", "<|coord_0|>", "<|coord_500|>", "<|coord_500|>"]}]}}
    second = {"row_id": "r2", "image_id": 2, "cohort": "development6", "input_record": {"image_id": 2, "objects": []}}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"conditions": ["original", "aligned"], "native_cases": [case, second], "calibration_cases": []}))
    cells = tmp_path / "cells"
    cells.mkdir()
    (cells / "r1-original.json").write_text(json.dumps(_native("r1", "original")))
    (cells / "r1-aligned.json").write_text(json.dumps(_native("r1", "aligned", duplicate=True)))
    (cells / "r2-original.json").write_text(json.dumps(_native("r2", "original", status="HOLD")))
    result = reduce_production(manifest_path=manifest, cells_root=cells, output_path=tmp_path / "reduction.json", cap=4)
    native = result["native"]
    assert native["observed"] == {"saved_cells": 3, "missing_cells": 1, "held_cells": 1, "complete_cells": 2, "complete_blocks": 1, "partial_blocks": 1}
    row = next(item for item in native["per_image"] if item["row_id"] == "r1")
    original = row["conditions"]["original"]
    assert original["owner_proxy"]["covered_owner_ids"] == ["11"]
    assert original["owner_proxy"]["same_owner_revisits"] == 1
    assert original["prediction_counts"]["owner_proxy_unknown"] == 1
    assert original["termination"]["cap_debt"] == 1
    assert result["status"] == "candidate_reduction_with_HOLD_cells"


def test_calibration_free_diagnostic_resolves_unique_manifest_referent(tmp_path: Path) -> None:
    case = {"row_id": "cal1", "image_id": 3, "cohort": "calibration", "input_record": {"image_id": 3, "objects": [{"desc": "person", "bbox_2d": ["<|coord_90|>", "<|coord_190|>", "<|coord_310|>", "<|coord_390|>"]}]}}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"conditions": ["original", "aligned"], "calibration_referents": {"cal1": "person"}, "native_cases": [], "calibration_cases": [case]}))
    cells = tmp_path / "cells"
    cells.mkdir()
    for condition, predictions in (("original", [{"generated_order": 0, "coord_bins": [100, 200, 300, 400]}]), ("aligned", [])):
        (cells / f"{condition}.json").write_text(json.dumps({"row_id": "cal1", "condition": condition, "teacher_coordinate_ce_sum": 4.0, "coordinate_tokens": 4, "coordinate_absolute_error_sum": 10.0, "free": {"token_ids": [1], "parsed": {"predictions": predictions, "dropped_predictions": []}}}))
    result = reduce_production(manifest_path=manifest, cells_root=cells, output_path=tmp_path / "reduction.json")
    rows = result["calibration"]["per_image"]
    assert rows[0]["conditions"]["original"]["free"]["diagnostic"]["status"] == "first_box_scored"
    assert rows[0]["conditions"]["aligned"]["free"]["diagnostic"]["status"] == "missing_first_box"
    agg = result["calibration"]["aggregate_complete_blocks"]["original"]
    assert agg["free_diagnostic_case_count"] == 1
    assert agg["free_first_box_scored"] == 1


def test_calibration_referent_ambiguity_is_hold(tmp_path: Path) -> None:
    case = {"row_id": "cal1", "image_id": 3, "cohort": "calibration", "input_record": {"image_id": 3, "objects": [{"desc": "person", "bbox_2d": ["<|coord_0|>", "<|coord_0|>", "<|coord_500|>", "<|coord_500|>"]}, {"desc": "person", "bbox_2d": ["<|coord_500|>", "<|coord_500|>", "<|coord_900|>", "<|coord_900|>"]}]}}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"conditions": ["original"], "calibration_referents": {"cal1": "person"}, "native_cases": [], "calibration_cases": [case]}))
    cells = tmp_path / "cells"
    cells.mkdir()
    (cells / "cell.json").write_text(json.dumps({"row_id": "cal1", "condition": "original", "teacher_coordinate_ce_sum": 4.0, "coordinate_tokens": 4, "coordinate_absolute_error_sum": 10.0, "free": {"token_ids": [1], "parsed": {"predictions": [{"generated_order": 0, "coord_bins": [100, 200, 300, 400]}], "dropped_predictions": []}}}))
    result = reduce_production(manifest_path=manifest, cells_root=cells, output_path=tmp_path / "reduction.json")
    diagnostic = result["calibration"]["per_image"][0]["conditions"]["original"]["free"]["diagnostic"]
    assert diagnostic["status"] == "reference_unavailable_HOLD"
