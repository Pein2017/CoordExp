from __future__ import annotations

import json
from pathlib import Path

import pytest

from probes.coordinate_representation.coordinate_codebook_alignment.scale_reduce import reduce_scale


def _row(path: Path, row_id: str, *, density: str = "ordinary") -> None:
    path.write_text(json.dumps({"image_id": 1, "_admission": {"row_id": row_id, "cohort": "fit_refined5", "stratum": {"density": density}}}) + "\n")


def _cell(path: Path, row_id: str, condition: str, *, status: str = "complete") -> None:
    path.write_text(json.dumps({"schema": "coordinate_codebook_alignment.evaluation_cell.v1", "status": status, "condition": condition, "case": {"row_id": row_id, "cohort": "fit_refined5"}, "gt": [{"owner_id": "1", "description": "cat", "bbox": [0, 0, 1000, 1000]}], "parser": {"parse_status": "accepted", "predictions": [{"description": "cat", "bbox": [0, 0, 1000, 1000], "coord_bins": [0, 0, 999, 999]}], "dropped_predictions": []}, "generation": {"stop_reason": "im_end", "cap": 3084}, "teacher": {"token_count": 1, "ce_sum": 0.1, "ce_mean": 0.1, "minimum_target_margin": 1.0, "mean_target_margin": 1.0}}))


def test_scale_missing_duplicate_and_mutated_cells_hold(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    _row(train, "rdup")
    specs = [{"cell_key": "source-rdup", "condition": "source", "panel": "train", "row_id": "rdup", "dataset": str(train), "reuse": {"path": str(tmp_path / "old.json")}}]
    analytical = {"specs": specs * 2752}
    # Duplicate frozen identities are rejected before any saved-cell read.
    with pytest.raises(ValueError, match="duplicate frozen"):
        reduce_scale({"schema": "test"}, analytical, tmp_path)

    # Use the real denominator shape with unique identities and one reused source.
    specs = []
    next_id = 0
    def add_specs(condition: str, count: int, panel: str, reuse_count: int = 0) -> None:
        nonlocal next_id
        for _ in range(count):
            row_id = f"r{next_id}"
            with train.open("a") as handle:
                handle.write(json.dumps({"image_id": next_id, "_admission": {"row_id": row_id, "cohort": "fit_refined5", "stratum": {"density": "ordinary"}}}) + "\n")
            path = tmp_path / f"old-{next_id}.json"
            if len([s for s in specs if s["condition"] == condition and "reuse" in s]) < reuse_count:
                _cell(path, row_id, condition)
                spec = {"cell_key": f"{condition}-{next_id}", "condition": condition, "panel": panel, "row_id": row_id, "dataset": str(train), "reuse": {"path": str(path)}}
            else:
                spec = {"cell_key": f"{condition}-{next_id}", "condition": condition, "panel": panel, "row_id": row_id, "dataset": str(train)}
            specs.append(spec)
            next_id += 1
    add_specs("source", 32, "train", reuse_count=32)
    add_specs("source", 992, "train")
    add_specs("source", 256, "validation")
    add_specs("epoch32", 1024, "train")
    add_specs("epoch32", 256, "validation")
    add_specs("epoch4", 96, "train")
    add_specs("epoch16", 96, "train")
    with pytest.raises(ValueError, match="retained32 membership"):
        reduce_scale({"schema": "test"}, {"specs": specs, "retained32_row_ids": ["r0"]}, tmp_path)
    # A deliberately mutated reused cell is visible as HOLD rather than being accepted.
    _cell(tmp_path / "old-0.json", "wrong", "source")
    result = reduce_scale({"schema": "test"}, {"specs": specs, "retained32_row_ids": [f"r{i}" for i in range(32)]}, tmp_path)
    assert result["status"] == "HOLD"
    assert result["denominator"]["mutation_cells"] == 1
    assert result["denominator"]["missing_cells"] == 2720
    assert result["per_image"]
    source = result["aggregate"]["source"]["cells"]
    assert source["teacher"]["successful_images"] == 31
    assert source["matched_target_denominators"]["iou50_class_consistent"]["target"] > 0
    assert result["guardrails"]["by_condition"]["epoch32"]["train1024"]["eligibility"] == "HOLD"
