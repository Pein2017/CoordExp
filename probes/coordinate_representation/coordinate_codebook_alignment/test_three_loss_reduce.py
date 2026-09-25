from __future__ import annotations

import json
from pathlib import Path

import pytest

from probes.coordinate_representation.coordinate_codebook_alignment.three_loss_reduce import _reduce


def _write_row(path: Path, row_id: str, cohort: str = "fit_refined5") -> None:
    with path.open("a") as handle:
        handle.write(json.dumps({"image_id": int(row_id[1:]), "_admission": {"row_id": row_id, "cohort": cohort, "stratum": {"density": "ordinary"}}}) + "\n")


def _write_cell(path: Path, row_id: str, condition: str) -> None:
    path.write_text(json.dumps({"schema": "coordinate_codebook_alignment.evaluation_cell.v1", "status": "complete", "condition": condition, "case": {"row_id": row_id, "cohort": "fit_refined5"}, "gt": [{"owner_id": "1", "description": "cat", "bbox": [0, 0, 1000, 1000]}], "parser": {"parse_status": "accepted", "predictions": [{"description": "cat", "bbox": [0, 0, 1000, 1000], "coord_bins": [0, 0, 999, 999]}], "dropped_predictions": []}, "generation": {"stop_reason": "im_end", "cap": 3084}, "teacher": {"token_count": 1, "ce_sum": 0.1, "ce_mean": 0.1, "minimum_target_margin": 1.0, "mean_target_margin": 1.0}}))


def test_three_loss_denominators_hold_and_ce_relative_flags(tmp_path: Path) -> None:
    dataset = tmp_path / "cases.jsonl"
    specs = []
    retained = [f"r{i}" for i in range(32)]
    next_id = 0
    def add(condition: str, count: int, panel: str, reuse: int = 0) -> None:
        nonlocal next_id
        for _ in range(count):
            row_id = f"r{next_id}"
            _write_row(dataset, row_id, "fit_refined5" if panel == "train" and next_id < 32 else "next")
            path = tmp_path / f"{condition}-{next_id}.json"
            spec = {"cell_key": f"{condition}-{next_id}", "condition": condition, "panel": panel, "row_id": row_id, "dataset": str(dataset)}
            if reuse and reuse > 0:
                _write_cell(path, row_id, condition)
                spec["reuse"] = {"path": str(path)}
                reuse -= 1
            specs.append(spec)
            next_id += 1
    add("source", 1024, "train", 1024)
    add("source", 256, "validation", 256)
    add("ce_only_epoch16", 1024, "train", 96)
    add("ce_only_epoch16", 256, "validation")
    add("three_loss_epoch8", 96, "train")
    add("three_loss_epoch16", 1024, "train")
    add("three_loss_epoch16", 256, "validation")
    # The fixture intentionally leaves most cells absent: the frozen denominator remains HOLD.
    result = _reduce({"schema": "test"}, {"specs": specs, "retained32_row_ids": retained}, tmp_path)
    assert result["status"] == "HOLD"
    assert result["denominator"]["analytical_cells"] == 3936
    assert result["denominator"]["new_cells"] == 2560
    assert result["denominator"]["reused_cells"] == 1376
    assert result["denominator"]["missing_cells"] == 2560
    assert result["retained_membership"]["count"] == 32
    assert result["guardrails"]["source_vs_final"]["train1024"]["eligibility"] == "HOLD"


def test_summary_preserves_span_and_recurrence_distinctions():
    from probes.coordinate_representation.coordinate_codebook_alignment.three_loss_summary import burden, band
    row={'status':'complete','stop_reason':'im_end','parser_dropped':1,'invalid_geometry':0,
         'format_summary':{'malformed_spans':[{'reason':'malformed_object_span','raw_span_text':'9'*100}]},
         'repeat_proxy':{'exact_row_revisit_count':0,'owner_revisit_count_iou50':2,'owner_max_run_iou50':1}}
    b=burden([row])
    assert b['bad_images']==1 and b['parser_drops']==1 and b['dropped_span_characters']==100
    assert b['exact_revisits']==0 and b['owner_revisits']==2 and b['max_owner_run']==1
    assert [band(n) for n in [4,5,9,10,19,20,39,40]]==['1-4','5-9','5-9','10-19','10-19','20-39','20-39','40+']


def test_saved_span_taxonomy_has_overlapping_causes_and_no_false_unknown():
    from probes.coordinate_representation.coordinate_codebook_alignment.three_loss_summary import span_causes
    def box(s):return '<|box_start|>'+s+'<|box_end|>'
    assert 'literal_digit_in_box' in span_causes({'reason':'malformed_object_span','raw_span_text':box('<|coord_831|>9<|coord_999|><|coord_805|>')})
    causes=span_causes({'reason':'geometry_invalid','raw_span_text':box('<|coord_500|><|coord_800|><|coord_400|><|coord_800|>')})
    assert set(causes)=={'parser_invalid_geometry','reversed_x','degenerate_y'}
    assert span_causes({'reason':'annotation_unmatched','raw_span_text':box('<|coord_100|><|coord_200|><|coord_300|><|coord_400|>')})==[]
