from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from probes.training_set_completion.coordinate_codebook_alignment.off_reduce import reduce_off
from probes.training_set_completion.coordinate_codebook_alignment.off_reduce import _summarize_pairs


def test_paired_iou_image_delta_counts_mixed_gain_loss_as_improvement() -> None:
    pair = {"clean_delta": 0}
    for metric in ("iou50_class_consistent", "iou80_class_consistent"):
        pair[metric] = {"gained_owner_ids": [], "lost_owner_ids": []}
    pair["iou80_class_consistent"] = {"gained_owner_ids": ["a", "b"],
                                      "lost_owner_ids": ["c"]}
    pair["failure_transitions"] = {name: {"repaired": False, "persistent": False,
                                          "newly_introduced": False}
                                   for name in ("bad", "cap", "owner_recurrent", "severe")}
    result = _summarize_pairs([pair])["iou80_class_consistent"]
    assert result["image_improvements"] == 1
    assert result["image_regressions"] == 0
    assert result["images_with_both_gain_and_loss"] == 1


def test_off_reducer_cli_refuses_existing_output(tmp_path: Path, monkeypatch) -> None:
    from probes.training_set_completion.coordinate_codebook_alignment.off_reduce import main

    output = tmp_path / "reduction.json"
    output.write_text("accepted artifact\n")
    before = output.read_bytes()
    monkeypatch.setattr("sys.argv", ["off_reduce", "--manifest", str(tmp_path / "missing-manifest"),
        "--analytical", str(tmp_path / "missing-analytical"), "--run-root", str(tmp_path),
        "--output", str(output)])
    with pytest.raises(FileExistsError):
        main()
    assert output.read_bytes() == before


def _cell(path: Path, row_id: str, condition: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"status": "complete", "condition": condition,
               "case": {"row_id": row_id},
               "gt": [{"owner_id": "owner-a", "description": "cat", "bbox": [0, 0, 1000, 1000]}],
               "parser": {"parse_status": "accepted", "predictions": [
                   {"description": "cat", "bbox": [0, 0, 1000, 1000], "coord_bins": [0, 0, 999, 999]}],
                   "dropped_predictions": []},
               "generation": {"stop_reason": "im_end", "cap": 3084},
               "teacher": {"token_count": 1, "ce_sum": 0.1, "ce_mean": 0.1,
                           "minimum_target_margin": 1.0, "mean_target_margin": 1.0,
                           "coordinate_mean_absolute_error": 0.01}}
    raw = json.dumps(payload).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _packet(tmp_path: Path):
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    rows = []
    for i in range(1280):
        # Some retained identities have a COCO val-origin name but belong to
        # this experiment's train panel; panel comes from the frozen spec.
        row_id = f"coco2017_val_{i:012d}" if 5 <= i < 32 else f"coco2017_train_{i:012d}"
        rows.append(row_id)
    train.write_text("\n".join(json.dumps({"_admission": {
        "row_id": rid, "cohort": "fit_refined5",
        "stratum": {"density": "dense" if i == 40 else "ordinary"}}})
        for i, rid in enumerate(rows[:1024])) + "\n")
    validation_rows = [f"coco2017_val_{i + 2000:012d}" for i in range(256)]
    validation.write_text("\n".join(json.dumps({"_admission": {
        "row_id": rid, "cohort": "monitor_v3",
        "stratum": {"density": "middle" if i == 0 else "ordinary"}}})
        for i, rid in enumerate(validation_rows)) + "\n")
    rows = rows[:1024] + validation_rows
    retained = rows[0:32]
    specs = []
    for condition in ("source", "three_loss_epoch16", "early_edge_epoch16", "off_epoch16"):
        for i, row_id in enumerate(rows):
            panel = "train" if i < 1024 else "validation"
            spec = {"cell_key": f"{condition}-{i}", "condition": condition,
                    "panel": panel, "row_id": row_id,
                    "dataset": str(train if panel == "train" else validation),
                    "density": "dense" if i == 40 else "ordinary"}
            if condition != "off_epoch16":
                path = tmp_path / "reused" / f"{condition}-{i}.json"
                digest = _cell(path, row_id, condition)
                spec["reuse"] = {"path": str(path), "sha256": digest}
            else:
                path = tmp_path / "production" / condition / "cells" / f"{spec['cell_key']}.json"
                _cell(path, row_id, condition)
            specs.append(spec)
    return {"specs": specs, "retained32_row_ids": retained}, tmp_path


def test_off_actual_caller_full_denominator_and_holds_mutations(tmp_path: Path) -> None:
    analytical, run_root = _packet(tmp_path)
    manifest = {"schema": "synthetic", "effective_losses": {
        "ce": 1.0, "typegate": 0.2, "axis": 0.01, "gaussian": 0.0}}
    result = reduce_off(manifest, analytical, run_root)
    assert result["status"] == "complete"
    assert result["denominator"] == {
        "analytical_cells": 5120, "distinct_cells": 5120, "images": 1280,
        "new_cells": 1280, "reused_cells": 3840,
        "expected_by_condition": {"source": 1280, "three_loss_epoch16": 1280,
                                  "early_edge_epoch16": 1280, "off_epoch16": 1280},
        "reused_by_condition": {"source": 1280, "three_loss_epoch16": 1280,
                                "early_edge_epoch16": 1280, "off_epoch16": 0},
        "missing_cells": 0, "mutation_cells": 0, "technical_invalid_cells": 0}
    assert len(result["per_image_comparisons"]) == 5 * 1280
    assert result["per_image_comparisons"][0]["teacher_fidelity"]["coordinate_mean_absolute_error"]["delta"] == 0.0
    assert result["retained_membership"]["row_ids"] == sorted(analytical["retained32_row_ids"])
    assert result["aggregates"]["source"]["by_group"]["retained32"]["cells_expected"] == 32
    assert result["aggregates"]["source"]["by_group"]["additions992"]["cells_expected"] == 992
    assert result["aggregates"]["source"]["by_stratum"]["dense"]["cells_expected"] == 1
    assert result["teacher_ce_scope"].startswith("teacher CE")
    assert result["training_objective"] == manifest["effective_losses"]

    # Preserve a frozen reused digest but mutate its bytes: HOLD before parsing.
    reused = next(s for s in analytical["specs"] if s["condition"] == "source")
    Path(reused["reuse"]["path"]).write_text("{")
    # A new cell with a readable but wrong row identity also remains HOLD.
    new = next(s for s in analytical["specs"] if s["condition"] == "off_epoch16")
    new_path = run_root / "production" / "off_epoch16" / "cells" / f"{new['cell_key']}.json"
    broken = json.loads(new_path.read_text())
    broken["case"]["row_id"] = "wrong-row"
    new_path.write_text(json.dumps(broken))
    malformed_new = next(s for s in analytical["specs"]
                         if s["condition"] == "off_epoch16" and s["row_id"] != new["row_id"])
    (run_root / "production" / "off_epoch16" / "cells" /
     f"{malformed_new['cell_key']}.json").write_text("{")
    held = reduce_off(manifest, analytical, run_root)
    assert held["status"] == "HOLD"
    assert held["denominator"]["mutation_cells"] == 2
    assert held["denominator"]["technical_invalid_cells"] == 1
    assert held["paired_summaries"]["off_epoch16_vs_source"]["all"]["paired_complete"] == 1278
    assert any(m["row_id"] == reused["row_id"] and "reuse_sha256_missing_or_mismatch" in m["reason"]
               for m in held["mutation_cells"])
    assert any(m["row_id"] == new["row_id"] and "cell_identity_mismatch" in m["reason"]
               for m in held["mutation_cells"])
    assert any(m["row_id"] == malformed_new["row_id"] and m["reason"] == "invalid_json"
               for m in held["technical_invalid_cells"])
    assert any(row["row_id"] == malformed_new["row_id"] and row["reason"] == "invalid_new_saved_cell"
               for row in held["per_image"] if row["status"] == "HOLD")

    invalid_membership = dict(analytical, retained32_row_ids=[f"not-a-train-row-{i}" for i in range(32)])
    with pytest.raises(ValueError, match="must name admitted source rows"):
        reduce_off(manifest, invalid_membership, run_root)


def test_frozen_missing_new_cells_are_hold(tmp_path: Path) -> None:
    analytical, run_root = _packet(tmp_path)
    new = next(s for s in analytical["specs"] if s["condition"] == "off_epoch16")
    (run_root / "production" / "off_epoch16" / "cells" / f"{new['cell_key']}.json").unlink()
    result = reduce_off({"schema": "synthetic"}, analytical, run_root)
    assert result["status"] == "HOLD"
    assert result["denominator"]["missing_cells"] == 1
    assert result["per_image"][0]["status"] == "complete"
    assert any(row["row_id"] == new["row_id"] and row["reason"] == "missing_saved_cell"
               for row in result["per_image"] if row["status"] == "HOLD")
