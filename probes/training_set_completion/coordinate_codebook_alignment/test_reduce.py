from __future__ import annotations

import json
from pathlib import Path

from probes.training_set_completion.coordinate_codebook_alignment.evaluation import _target_ids
from probes.training_set_completion.coordinate_codebook_alignment.reduce import reduce_cell, reduce_cells


def _case(path, row_id, cohort):
    image_id = 1 if row_id == "r1" else 2
    row = {"image_id": image_id, "objects": [{"coco_ann_id": 1, "desc": "cat", "bbox_2d": ["<|coord_0|>", "<|coord_0|>", "<|coord_999|>", "<|coord_999|>"]}], "metadata": {"split": "val"}, "file_name": f"images/val2017/{image_id:012d}.jpg", "width": 1000, "height": 1000, "images": ["/tmp/image.jpg"], "_admission": {"row_id": row_id, "cohort": cohort}}
    path.write_text(json.dumps(row) + "\n")


def _cell(row_id="r1", condition="source", *, status="complete", unknown=False, teacher=False):
    pred = {"description": "cat", "bbox": [0.0, 0.0, 1000.0, 1000.0], "coord_bins": [0, 0, 999, 999]}
    if unknown:
        pred["description"] = "dog"
    cell = {"status": status, "condition": condition, "case": {"row_id": row_id, "cohort": "fit"}, "gt": [{"owner_id": "1", "description": "cat", "bbox": [0.0, 0.0, 1000.0, 1000.0]}], "parser": {"parse_status": "accepted", "predictions": [pred], "dropped_predictions": []}, "generation": {"stop_reason": "im_end", "cap": 3084}, "timing": {"wall_seconds": 0.1}}
    if teacher:
        cell["teacher"] = {"token_count": 10, "ce_sum": 4.0, "ce_mean": 0.4, "minimum_target_margin": 0.2, "mean_target_margin": 0.5, "coordinate_mean_absolute_error": 0.01}
    return cell


def test_unmatched_prediction_remains_unknown():
    reduced = reduce_cell(_cell(unknown=True))
    assert reduced["parser_status"] == "accepted"
    assert reduced["iou50_class_agnostic"]["matched_count"] == 1
    assert reduced["iou50_class_consistent"]["matched_count"] == 0
    assert reduced["unknown_prediction_indices"] == []
    assert reduced["clean_known_positive_proxy"]["eligible"] is False


def test_missing_denominator_is_hold(tmp_path):
    fit = tmp_path / "fit.jsonl"
    monitor = tmp_path / "monitor.jsonl"
    _case(fit, "r1", "fit")
    _case(monitor, "r2", "monitor")
    manifest = {"fit": {"count": 1, "cases_path": str(fit)}, "monitor": {"count": 1, "cases_path": str(monitor)}}
    result = reduce_cells(manifest, [_cell("r1", "source")], ["source", "candidate"])
    assert result["status"] == "HOLD"
    assert result["observed"]["missing_cells"] == 3
    assert {tuple(item.values()) for item in result["missing_cells"]} == {("r1", "candidate"), ("r2", "source"), ("r2", "candidate")}
    assert all(item["teacher"]["status"] == "missing" for item in result["per_image"] if item["status"] == "HOLD")


def test_cli_ignores_noncell_json_but_retains_missing_denominator(tmp_path, monkeypatch):
    from probes.training_set_completion.coordinate_codebook_alignment.reduce import main
    fit, monitor = tmp_path / 'fit.jsonl', tmp_path / 'monitor.jsonl'
    _case(fit, 'r1', 'fit')
    _case(monitor, 'r2', 'monitor')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'fit': {'count': 1, 'cases_path': str(fit)},
                                    'monitor': {'count': 1, 'cases_path': str(monitor)}}))
    (tmp_path / 'unrelated-check.json').write_text('[{"passed": true}]')
    output = tmp_path / 'result.json'
    monkeypatch.setattr('sys.argv', ['reduce', '--manifest', str(manifest),
        '--input-root', str(tmp_path), '--output', str(output), '--conditions', 'source'])
    main()
    result = json.loads(output.read_text())
    assert result['status'] == 'HOLD'
    assert result['observed']['missing_cells'] == 2
    cell = _cell()
    cell['schema'] = 'coordinate_codebook_alignment.evaluation_cell.v1'
    cell_path = tmp_path / 'cell.json'
    cell_path.write_text(json.dumps(cell))
    # An explicit empty input set must not discover the nearby valid cell.
    cell_list = tmp_path / 'cell-list.json'
    cell_list.write_text('[]')
    monkeypatch.setattr('sys.argv', ['reduce', '--manifest', str(manifest),
        '--input-root', str(tmp_path), '--output', str(output), '--conditions', 'source',
        '--cell-list', str(cell_list)])
    main()
    assert json.loads(output.read_text())['observed']['cells_loaded'] == 0


def test_teacher_metrics_are_preserved_and_aggregated(tmp_path):
    fit = tmp_path / "fit.jsonl"
    monitor = tmp_path / "monitor.jsonl"
    _case(fit, "r1", "fit")
    _case(monitor, "r2", "monitor")
    manifest = {"fit": {"count": 1, "cases_path": str(fit)}, "monitor": {"count": 1, "cases_path": str(monitor)}}
    result = reduce_cells(manifest, [_cell("r1", teacher=True)], ["source"], planned_cells=[{"row_id": "r1", "condition": "source"}])
    image = result["per_image"][0]
    assert image["teacher"]["status"] == "present"
    assert image["teacher"]["metrics"]["ce_sum"] == 4.0
    teacher = result["aggregate"]["source"]["teacher"]
    assert teacher["successful_images"] == 1
    assert teacher["missing_images"] == 0
    assert teacher["ce_macro_image"] == 0.4
    assert teacher["ce_token_weighted"] == 0.4
    assert teacher["mean_target_margin_macro_image"] == 0.5
    assert image["clean_known_positive_proxy"]["eligible"] is True


def test_cap_parser_drop_and_technical_invalid_are_not_clean():
    cap = _cell()
    cap["generation"]["stop_reason"] = "length"
    assert reduce_cell(cap)["clean_known_positive_proxy"]["eligible"] is False
    dropped = _cell()
    dropped["parser"]["dropped_predictions"] = [{"reason": "unmatched_text"}]
    assert reduce_cell(dropped)["clean_known_positive_proxy"]["eligible"] is False
    invalid = reduce_cell(_cell(status="technical_invalid"))
    assert invalid["status"] == "HOLD"
    assert invalid["clean_known_positive_proxy"]["eligible"] is False


def test_owner_recurrence_allows_nonidentical_boxes():
    cell = _cell()
    cell["parser"]["predictions"] = [
        {"description": "cat", "bbox": [0.0, 0.0, 900.0, 900.0], "coord_bins": [0, 0, 899, 899]},
        {"description": "cat", "bbox": [10.0, 10.0, 890.0, 890.0], "coord_bins": [10, 10, 889, 889]},
    ]
    repeat = reduce_cell(cell)["repeat_proxy"]
    assert repeat["exact_row_revisit_count"] == 0
    assert repeat["owner_revisit_count_iou50"] == 1
    assert repeat["owner_max_run_iou50"] == 2


def test_explicit_planned_cells_freeze_intermediate_denominator(tmp_path):
    fit = tmp_path / "fit.jsonl"
    monitor = tmp_path / "monitor.jsonl"
    _case(fit, "r1", "fit")
    _case(monitor, "r2", "monitor")
    manifest = {"fit": {"count": 1, "cases_path": str(fit)}, "monitor": {"count": 1, "cases_path": str(monitor)}}
    result = reduce_cells(manifest, [_cell("r1", "source")], ["source", "candidate"], planned_cells=[{"row_id": "r1", "condition": "source"}])
    assert result["status"] == "complete"
    assert result["denominator"]["cells_expected"] == 1
    assert result["aggregate"]["source"]["cells_expected"] == 1


def test_target_ids_use_original_input_record(tmp_path):
    admission_path = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-coordinate-codebook-alignment/selection-v4/admission.json")
    admission = json.loads(admission_path.read_text())
    row = json.loads(Path(admission["fit"]["cases_path"]).read_text().splitlines()[0])

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            assert "_admission" not in text
            return [17, 18]

    class Qwen:
        tokenizer = Tokenizer()

    ids = _target_ids(Qwen(), row, admission["source_config"], Path(admission["fit"]["cases_path"]))
    assert ids == [17, 18]
