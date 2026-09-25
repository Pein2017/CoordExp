from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from probes.coordinate_representation.coordinate_codebook_alignment.early_edge_reduce import reduce_early_edge


def _cell(path: Path, row_id: str, condition: str, *, bad: bool = False, repeat: int = 1) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    cell = {"status": "complete", "condition": condition, "case": {"row_id": row_id},
            "gt": [{"owner_id": "owner-a", "description": "cat", "bbox": [0, 0, 1000, 1000]}],
            "parser": {"parse_status": "accepted", "predictions": [
                {"description": "cat", "bbox": [0, 0, 1000, 1000], "coord_bins": [0, 0, 999, 999]}
                for _ in range(repeat)],
                "dropped_predictions": [{"reason": "malformed_object_span"}] if bad else []},
            "generation": {"stop_reason": "im_end", "cap": 3084},
            "teacher": {"token_count": 1, "ce_sum": 0.1, "ce_mean": 0.1,
                        "minimum_target_margin": 1.0, "mean_target_margin": 1.0}}
    path.write_text(json.dumps(cell))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_early_edge_actual_caller_pairs_cohorts_and_holds_mutated_reuse(tmp_path: Path) -> None:
    dataset = tmp_path / "cases.jsonl"
    specs = []
    rows = [f"r{i}" for i in range(1280)]
    dataset.write_text("\n".join(json.dumps({"_admission": {
        "row_id": row_id, "cohort": "fit_refined5" if i < 1024 else "monitor_v3",
        "stratum": {"density": "dense" if i == 40 else "ordinary"}}})
        for i, row_id in enumerate(rows)) + "\n")

    def add(condition: str, row_ids: list[str], panel: str, reused: bool) -> None:
        for i, row_id in enumerate(row_ids):
            spec = {"cell_key": f"{condition}-{row_id}", "condition": condition,
                    "panel": panel, "row_id": row_id, "dataset": str(dataset)}
            cell_condition = condition
            if reused:
                path = tmp_path / "old" / f"{condition}-{row_id}.json"
                # The first 32 source files are reused, while accepted retained32
                # is deliberately r100-r131 to falsify reuse-derived membership.
                digest = _cell(path, row_id, cell_condition,
                               bad=condition == "source" and row_id == "r0",
                               repeat=6 if condition == "source" and row_id == "r1" else 1)
                spec["reuse"] = {"path": str(path), "sha256": digest}
            else:
                path = tmp_path / "production" / condition / "cells" / f"{spec['cell_key']}.json"
                _cell(path, row_id, cell_condition)
            specs.append(spec)

    train = rows[:1024]
    validation = rows[1024:]
    add("source", train, "train", True)
    add("source", validation, "validation", True)
    add("three_loss_epoch16", train, "train", True)
    add("three_loss_epoch16", validation, "validation", True)
    add("three_loss_epoch8", train[:96], "train", True)
    add("early_edge_epoch16", train, "train", False)
    add("early_edge_epoch16", validation, "validation", False)
    add("early_edge_epoch8", train[:96], "train", False)

    retained = [f"r{i}" for i in range(100, 132)]
    analytical = {"specs": specs, "retained32_row_ids": retained}
    result = reduce_early_edge({"schema": "synthetic"}, analytical, tmp_path)
    assert result["status"] == "complete"
    assert result["denominator"] == {"analytical_cells": 4032, "distinct_cells": 4032,
        "new_cells": 1376, "reused_cells": 2656,
        "expected_by_condition": {"source": 1280, "three_loss_epoch16": 1280, "three_loss_epoch8": 96, "early_edge_epoch16": 1280, "early_edge_epoch8": 96},
        "reused_by_condition": {"source": 1280, "three_loss_epoch16": 1280, "three_loss_epoch8": 96, "early_edge_epoch16": 0, "early_edge_epoch8": 0},
        "missing_cells": 0, "mutation_cells": 0}
    assert result["retained_membership"]["row_ids"] == retained
    source_retained = next(row for row in result["per_image"] if row["condition"] == "source" and row["row_id"] == "r100")
    source_addition = next(row for row in result["per_image"] if row["condition"] == "source" and row["row_id"] == "r0")
    assert source_retained["group"] == "retained32" and source_addition["group"] == "additions992"
    assert result["transition_counts"]["early_edge_epoch16_vs_source"]["additions992"]["bad"]["repaired"] == 1
    assert result["transition_counts"]["early_edge_epoch16_vs_source"]["dense"]["bad"]["repaired"] == 0
    severe = next(row for row in result["per_image_comparisons"]
                  if row["baseline_condition"] == "source" and row["row_id"] == "r1" and row["condition"] == "early_edge_epoch16")
    assert severe["transitions"]["severe"]["repaired"]
    assert severe["baseline_repeat_proxy"]["exact_row_revisit_count"] == 5
    assert severe["baseline_repeat_proxy"]["owner_revisit_count_iou50"] == 5
    assert result["unknown_policy"].startswith("annotation-unmatched")

    # Current binding and row identity must be required for accepted reuse.
    reused = next(s for s in specs if s["condition"] == "source" and s["row_id"] == "r0")
    expected_hash = reused["reuse"]["sha256"]
    del reused["reuse"]["sha256"]
    unbound = reduce_early_edge({"schema": "synthetic"}, analytical, tmp_path)
    assert unbound["status"] == "HOLD"
    assert unbound["denominator"]["mutation_cells"] == 1
    reused["reuse"]["sha256"] = expected_hash
    Path(reused["reuse"]["path"]).write_text("{")
    held = reduce_early_edge({"schema": "synthetic"}, analytical, tmp_path)
    assert held["status"] == "HOLD"
    assert held["denominator"]["mutation_cells"] == 1
    assert held["paired_complete_by_comparison"]["early_edge_epoch16_vs_source"] == 1279
    broken = {"specs": specs, "retained32_row_ids": ["not-a-source-row"] * 32}
    with pytest.raises(ValueError, match="explicit retained32"):
        reduce_early_edge({"schema": "synthetic"}, broken, tmp_path)


def test_frozen_early_edge_packet_reduces_reused_cells_to_hold(tmp_path: Path) -> None:
    root = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-early-edge-codebook")
    if not (root / "analytical-cells.json").exists():
        pytest.skip("frozen early-edge output packet is unavailable")
    packet = json.loads((root / "packet-v1.json").read_text())
    analytical = json.loads((root / "analytical-cells.json").read_text())
    result = reduce_early_edge(packet, analytical, tmp_path)
    assert result["status"] == "HOLD"
    assert result["denominator"]["analytical_cells"] == 4032
    assert result["denominator"]["reused_cells"] == 2656
    assert result["denominator"]["new_cells"] == 1376
    assert result["denominator"]["missing_cells"] == 1376
    assert result["denominator"]["mutation_cells"] == 0
    assert sum(item["cells_complete"] for item in result["aggregates"].values()) == 2656
    assert {item["condition"] for item in result["missing_cells"]} == {
        "early_edge_epoch8", "early_edge_epoch16"}
    reused = [item for item in result["input_bindings"] if item["source_kind"] == "reused_source"]
    assert len(reused) == 2656
    assert all(item["expected_sha256"] == item["sha256"] for item in reused)
