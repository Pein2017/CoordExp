from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from src.common.errors import ArtifactContractError
from src.inference.backend import DecodeResult, TokenTrace
from src.inference.parsing import parse_compact_object_box_closed


OBJECT_TEXT = (
    "<|object_ref_start|>cat<|object_ref_end|>"
    "<|box_start|><|coord_100|><|coord_200|><|coord_300|><|coord_400|><|box_end|>"
)


def test_detection_consumer_writes_official_coco_metrics_for_perfect_prediction(
    tmp_path: Path,
) -> None:
    from src.eval.detection_consumer import evaluate_scored_detection_artifacts
    from src.eval.detection_consumer import sha256_file

    artifact_dir = _write_scored_fixture(
        tmp_path / "artifacts",
        rows=[_raw_row("row-1", 0, text=OBJECT_TEXT)],
    )
    result = evaluate_scored_detection_artifacts(
        artifact_dir=artifact_dir,
        output_dir=tmp_path / "eval",
    )

    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))

    assert result.metrics_path == tmp_path / "eval" / "metrics.json"
    assert metrics["metric_artifact_name"] == "metrics.json"
    assert metrics["metric_family"] == "coordexp_swift_detection_coco_bbox_v1"
    assert metrics["benchmark_metric"] is False
    assert metrics["metric_scope"] == "coco_bbox"
    assert metrics["row_count"] == 1
    assert metrics["gt_object_count"] == 1
    assert metrics["pred_object_count"] == 1
    assert metrics["scored_pred_count"] == 1
    assert metrics["mAP"] == pytest.approx(1.0)
    assert metrics["mAP_50"] == pytest.approx(1.0)
    assert metrics["mAP_75"] == pytest.approx(1.0)
    assert metrics["mRecall"] == pytest.approx(1.0)
    assert metrics["bbox_AP"] == pytest.approx(1.0)
    assert metrics["bbox_AR100"] == pytest.approx(1.0)
    assert (tmp_path / "eval" / "coco_gt.json").is_file()
    assert (tmp_path / "eval" / "coco_predictions.json").is_file()
    coco_gt = json.loads((tmp_path / "eval" / "coco_gt.json").read_text(encoding="utf-8"))
    coco_predictions = json.loads(
        (tmp_path / "eval" / "coco_predictions.json").read_text(encoding="utf-8")
    )
    assert "_ignore" not in coco_gt["annotations"][0]
    assert "ignore" not in coco_gt["annotations"][0]
    assert set(coco_predictions[0]) == {"image_id", "category_id", "bbox", "score"}
    receipt_path = tmp_path / "eval" / "evaluation_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert result.receipt_path == receipt_path
    assert metrics["evaluation_receipt_json"] == "evaluation_receipt.json"
    assert metrics["evaluation_receipt"] == receipt
    assert receipt["artifact_dir"] == artifact_dir.as_posix()
    assert receipt["benchmark_metric"] is False
    assert receipt["run_manifest"]["benchmark_eligible"] is False
    assert receipt["run_manifest"]["terminal_status"] == "completed"
    assert receipt["artifacts"]["gt_vs_pred.jsonl"]["sha256"] == sha256_file(
        artifact_dir / "gt_vs_pred.jsonl"
    )
    assert receipt["artifacts"]["gt_vs_pred_scored.jsonl"]["sha256"] == sha256_file(
        artifact_dir / "gt_vs_pred_scored.jsonl"
    )
    assert receipt["artifacts"]["gt_vs_pred_scored.jsonl.provenance.json"][
        "sha256"
    ] == sha256_file(artifact_dir / "gt_vs_pred_scored.jsonl.provenance.json")
    assert receipt["artifacts"]["run_manifest.json"]["sha256"] == sha256_file(
        artifact_dir / "run_manifest.json"
    )
    assert receipt["row_binding"]["row_count"] == 1
    assert receipt["generation_config_fingerprint"] == "gen-fp"


def test_detection_consumer_scales_gt_coord_bins_to_prediction_pixel_space(
    tmp_path: Path,
) -> None:
    from src.eval.detection_consumer import evaluate_scored_detection_artifacts

    artifact_dir = _write_scored_fixture(
        tmp_path / "artifacts",
        rows=[
            _raw_row(
                "row-1",
                0,
                text=OBJECT_TEXT,
                image_width=1248,
                image_height=832,
            )
        ],
    )

    result = evaluate_scored_detection_artifacts(
        artifact_dir=artifact_dir,
        output_dir=tmp_path / "eval",
    )

    coco_gt = json.loads((tmp_path / "eval" / "coco_gt.json").read_text(encoding="utf-8"))
    coco_predictions = json.loads(
        (tmp_path / "eval" / "coco_predictions.json").read_text(encoding="utf-8")
    )

    assert coco_gt["annotations"][0]["bbox"] == [125, 166, 249, 167]
    assert coco_predictions[0]["bbox"] == [125, 166, 249, 167]
    assert result.metrics["mAP"] == pytest.approx(1.0)
    assert result.metrics["mRecall"] == pytest.approx(1.0)


def test_detection_consumer_keeps_empty_predictions_as_false_negatives(
    tmp_path: Path,
) -> None:
    from src.eval.detection_consumer import evaluate_scored_detection_artifacts

    artifact_dir = _write_scored_fixture(
        tmp_path / "artifacts",
        rows=[_raw_row("row-1", 0, text="malformed")],
        decode_results={},
    )

    result = evaluate_scored_detection_artifacts(
        artifact_dir=artifact_dir,
        output_dir=tmp_path / "eval",
    )

    coco_gt = json.loads((tmp_path / "eval" / "coco_gt.json").read_text(encoding="utf-8"))
    coco_predictions = json.loads(
        (tmp_path / "eval" / "coco_predictions.json").read_text(encoding="utf-8")
    )

    assert len(coco_gt["images"]) == 1
    assert len(coco_gt["annotations"]) == 1
    assert coco_predictions == []
    assert result.metrics["empty_pred_row_count"] == 1
    assert result.metrics["raw_dropped_prediction_count"] == 1
    assert result.metrics["parse_status_counts"] == {"all_spans_dropped": 1}
    assert result.metrics["mAP"] == pytest.approx(0.0)
    assert result.metrics["mRecall"] == pytest.approx(0.0)


def test_detection_consumer_counts_parser_drops_without_reparsing_raw_text(
    tmp_path: Path,
) -> None:
    from src.eval.detection_consumer import evaluate_scored_detection_artifacts

    text = OBJECT_TEXT + "not a valid object"
    artifact_dir = _write_scored_fixture(
        tmp_path / "artifacts",
        rows=[_raw_row("row-1", 0, text=text)],
        decode_results={"row-1": _decode_result("row-1", text=OBJECT_TEXT)},
    )

    result = evaluate_scored_detection_artifacts(
        artifact_dir=artifact_dir,
        output_dir=tmp_path / "eval",
    )

    assert result.metrics["pred_object_count"] == 1
    assert result.metrics["raw_dropped_prediction_count"] == 1
    assert result.metrics["parse_status_counts"] == {"accepted_with_drops": 1}


def test_detection_consumer_drops_unknown_prediction_category_with_counter(
    tmp_path: Path,
) -> None:
    from src.eval.detection_consumer import evaluate_scored_detection_artifacts

    unknown_text = OBJECT_TEXT.replace("cat", "spaceship")
    artifact_dir = _write_scored_fixture(
        tmp_path / "artifacts",
        rows=[_raw_row("row-1", 0, text=unknown_text)],
        decode_results={"row-1": _decode_result("row-1", text=unknown_text)},
    )

    result = evaluate_scored_detection_artifacts(
        artifact_dir=artifact_dir,
        output_dir=tmp_path / "eval",
    )

    coco_predictions = json.loads(
        (tmp_path / "eval" / "coco_predictions.json").read_text(encoding="utf-8")
    )
    assert coco_predictions == []
    assert result.metrics["unknown_category_pred_count"] == 1
    assert result.metrics["mAP"] == pytest.approx(0.0)


def test_detection_consumer_refuses_unknown_gt_category(tmp_path: Path) -> None:
    from src.eval.detection_consumer import evaluate_scored_detection_artifacts

    artifact_dir = _write_scored_fixture(
        tmp_path / "artifacts",
        rows=[
            _raw_row(
                "row-1",
                0,
                text=OBJECT_TEXT,
                gt_description="spaceship",
            )
        ],
    )

    with pytest.raises(ArtifactContractError) as exc_info:
        evaluate_scored_detection_artifacts(
            artifact_dir=artifact_dir,
            output_dir=tmp_path / "eval",
        )

    assert exc_info.value.code == "eval_detection.unknown_gt_category"


def test_detection_consumer_coco80_registry_matches_prompt_contract() -> None:
    from src.eval.detection_categories import COCO_80_CATEGORY_IDS, COCO_80_CLASS_NAMES

    assert len(COCO_80_CLASS_NAMES) == 80
    assert len(set(COCO_80_CLASS_NAMES)) == 80
    assert COCO_80_CATEGORY_IDS["person"] == 1
    assert COCO_80_CATEGORY_IDS["cat"] == 16
    assert COCO_80_CATEGORY_IDS["toothbrush"] == 80


def test_detection_consumer_refuses_missing_provenance_before_metrics(tmp_path: Path) -> None:
    from src.eval.detection_consumer import evaluate_scored_detection_artifacts

    artifact_dir = _write_scored_fixture(tmp_path / "artifacts")
    (artifact_dir / "gt_vs_pred_scored.jsonl.provenance.json").unlink()

    with pytest.raises(ArtifactContractError) as exc_info:
        evaluate_scored_detection_artifacts(
            artifact_dir=artifact_dir,
            output_dir=tmp_path / "eval",
        )

    assert exc_info.value.code == "eval_detection.missing_provenance"
    assert not (tmp_path / "eval" / "metrics.json").exists()


def test_detection_consumer_refuses_provenance_binding_mismatch_before_metrics(tmp_path: Path) -> None:
    from src.eval.detection_consumer import evaluate_scored_detection_artifacts

    artifact_dir = _write_scored_fixture(tmp_path / "artifacts")
    provenance_path = artifact_dir / "gt_vs_pred_scored.jsonl.provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["row_binding"]["row_count"] = 999
    provenance_path.write_text(json.dumps(provenance, sort_keys=True), encoding="utf-8")

    with pytest.raises(ArtifactContractError) as exc_info:
        evaluate_scored_detection_artifacts(
            artifact_dir=artifact_dir,
            output_dir=tmp_path / "eval",
        )

    assert exc_info.value.code == "eval_detection.provenance_row_count_mismatch"
    assert not (tmp_path / "eval" / "metrics.json").exists()


def test_detection_consumer_refuses_raw_artifact_sha_mismatch_before_metrics(tmp_path: Path) -> None:
    from src.eval.detection_consumer import evaluate_scored_detection_artifacts

    artifact_dir = _write_scored_fixture(tmp_path / "artifacts")
    provenance_path = artifact_dir / "gt_vs_pred_scored.jsonl.provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["raw_artifact"]["sha256"] = "definitely-not-the-raw-sha"
    provenance_path.write_text(json.dumps(provenance, sort_keys=True), encoding="utf-8")

    with pytest.raises(ArtifactContractError) as exc_info:
        evaluate_scored_detection_artifacts(
            artifact_dir=artifact_dir,
            output_dir=tmp_path / "eval",
        )

    assert exc_info.value.code == "eval_detection.raw_sha_mismatch"
    assert not (tmp_path / "eval" / "metrics.json").exists()


def test_detection_consumer_wraps_malformed_scored_jsonl_row(tmp_path: Path) -> None:
    from src.eval.detection_consumer import evaluate_scored_detection_artifacts

    artifact_dir = _write_scored_fixture(tmp_path / "artifacts")
    (artifact_dir / "gt_vs_pred_scored.jsonl").write_text("{not-json}\n", encoding="utf-8")

    with pytest.raises(ArtifactContractError) as exc_info:
        evaluate_scored_detection_artifacts(
            artifact_dir=artifact_dir,
            output_dir=tmp_path / "eval",
        )

    assert exc_info.value.code == "eval_detection.json_decode"
    assert exc_info.value.context["path"].endswith("gt_vs_pred_scored.jsonl")
    assert exc_info.value.context["row_index"] == 0
    assert not (tmp_path / "eval" / "metrics.json").exists()


def test_detection_consumer_wraps_malformed_provenance_json(tmp_path: Path) -> None:
    from src.eval.detection_consumer import evaluate_scored_detection_artifacts

    artifact_dir = _write_scored_fixture(tmp_path / "artifacts")
    (artifact_dir / "gt_vs_pred_scored.jsonl.provenance.json").write_text("{not-json}\n", encoding="utf-8")

    with pytest.raises(ArtifactContractError) as exc_info:
        evaluate_scored_detection_artifacts(
            artifact_dir=artifact_dir,
            output_dir=tmp_path / "eval",
        )

    assert exc_info.value.code == "eval_detection.json_decode"
    assert exc_info.value.context["path"].endswith("gt_vs_pred_scored.jsonl.provenance.json")
    assert "row_index" not in exc_info.value.context
    assert not (tmp_path / "eval" / "metrics.json").exists()


def test_detection_consumer_refuses_invalid_row_local_score_provenance(tmp_path: Path) -> None:
    from src.eval.detection_consumer import evaluate_scored_detection_artifacts
    from src.inference.artifacts import sha256_file

    artifact_dir = _write_scored_fixture(tmp_path / "artifacts")
    scored_path = artifact_dir / "gt_vs_pred_scored.jsonl"
    rows = _read_jsonl(scored_path)
    del rows[0]["pred"][0]["pred_score_source"]
    _write_jsonl(scored_path, rows)
    provenance_path = artifact_dir / "gt_vs_pred_scored.jsonl.provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["scored_artifact"]["sha256"] = sha256_file(scored_path)
    provenance_path.write_text(json.dumps(provenance, sort_keys=True), encoding="utf-8")

    with pytest.raises(ArtifactContractError) as exc_info:
        evaluate_scored_detection_artifacts(
            artifact_dir=artifact_dir,
            output_dir=tmp_path / "eval",
        )

    assert exc_info.value.code == "eval_detection.row_score_provenance_missing"
    assert not (tmp_path / "eval" / "metrics.json").exists()


def test_detection_consumer_refuses_mismatched_score_policy_provenance(tmp_path: Path) -> None:
    from src.eval.detection_consumer import evaluate_scored_detection_artifacts
    from src.inference.artifacts import sha256_file

    artifact_dir = _write_scored_fixture(tmp_path / "artifacts")
    scored_path = artifact_dir / "gt_vs_pred_scored.jsonl"
    rows = _read_jsonl(scored_path)
    rows[0]["pred"][0]["pred_score_source"]["score_policy_fingerprint"] = "wrong-policy"
    _write_jsonl(scored_path, rows)
    provenance_path = artifact_dir / "gt_vs_pred_scored.jsonl.provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["scored_artifact"]["sha256"] = sha256_file(scored_path)
    provenance_path.write_text(json.dumps(provenance, sort_keys=True), encoding="utf-8")

    with pytest.raises(ArtifactContractError) as exc_info:
        evaluate_scored_detection_artifacts(
            artifact_dir=artifact_dir,
            output_dir=tmp_path / "eval",
        )

    assert exc_info.value.code == "eval_detection.row_score_policy_mismatch"
    assert not (tmp_path / "eval" / "metrics.json").exists()


def test_detection_consumer_refuses_mismatched_score_row_provenance(tmp_path: Path) -> None:
    from src.eval.detection_consumer import evaluate_scored_detection_artifacts
    from src.inference.artifacts import sha256_file

    artifact_dir = _write_scored_fixture(tmp_path / "artifacts")
    scored_path = artifact_dir / "gt_vs_pred_scored.jsonl"
    rows = _read_jsonl(scored_path)
    rows[0]["pred"][0]["pred_score_source"]["row_id"] = "other-row"
    _write_jsonl(scored_path, rows)
    provenance_path = artifact_dir / "gt_vs_pred_scored.jsonl.provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["scored_artifact"]["sha256"] = sha256_file(scored_path)
    provenance_path.write_text(json.dumps(provenance, sort_keys=True), encoding="utf-8")

    with pytest.raises(ArtifactContractError) as exc_info:
        evaluate_scored_detection_artifacts(
            artifact_dir=artifact_dir,
            output_dir=tmp_path / "eval",
        )

    assert exc_info.value.code == "eval_detection.row_score_provenance_mismatch"
    assert not (tmp_path / "eval" / "metrics.json").exists()


def test_detection_consumer_refuses_invalid_score_value(tmp_path: Path) -> None:
    from src.eval.detection_consumer import evaluate_scored_detection_artifacts
    from src.inference.artifacts import sha256_file

    artifact_dir = _write_scored_fixture(tmp_path / "artifacts")
    scored_path = artifact_dir / "gt_vs_pred_scored.jsonl"
    rows = _read_jsonl(scored_path)
    rows[0]["pred"][0]["score"] = 2.0
    _write_jsonl(scored_path, rows)
    provenance_path = artifact_dir / "gt_vs_pred_scored.jsonl.provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["scored_artifact"]["sha256"] = sha256_file(scored_path)
    provenance_path.write_text(json.dumps(provenance, sort_keys=True), encoding="utf-8")

    with pytest.raises(ArtifactContractError) as exc_info:
        evaluate_scored_detection_artifacts(
            artifact_dir=artifact_dir,
            output_dir=tmp_path / "eval",
        )

    assert exc_info.value.code == "eval_detection.invalid_score"
    assert not (tmp_path / "eval" / "metrics.json").exists()


def test_detection_consumer_refuses_raw_scored_row_id_mismatch(tmp_path: Path) -> None:
    from src.eval.detection_consumer import evaluate_scored_detection_artifacts
    from src.inference.artifacts import sha256_file

    artifact_dir = _write_scored_fixture(tmp_path / "artifacts")
    raw_path = artifact_dir / "gt_vs_pred.jsonl"
    raw_rows = _read_jsonl(raw_path)
    raw_rows[0]["row_id"] = "different-row"
    _write_jsonl(raw_path, raw_rows)
    provenance_path = artifact_dir / "gt_vs_pred_scored.jsonl.provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["raw_artifact"]["sha256"] = sha256_file(raw_path)
    provenance_path.write_text(json.dumps(provenance, sort_keys=True), encoding="utf-8")

    with pytest.raises(ArtifactContractError) as exc_info:
        evaluate_scored_detection_artifacts(
            artifact_dir=artifact_dir,
            output_dir=tmp_path / "eval",
        )

    assert exc_info.value.code == "eval_detection.raw_scored_row_id_mismatch"
    assert not (tmp_path / "eval" / "metrics.json").exists()


def test_detection_consumer_refuses_raw_scored_gt_mismatch(tmp_path: Path) -> None:
    from src.eval.detection_consumer import evaluate_scored_detection_artifacts
    from src.inference.artifacts import sha256_file

    artifact_dir = _write_scored_fixture(tmp_path / "artifacts")
    scored_path = artifact_dir / "gt_vs_pred_scored.jsonl"
    scored_rows = _read_jsonl(scored_path)
    scored_rows[0]["gt"][0]["description"] = "dog"
    _write_jsonl(scored_path, scored_rows)
    provenance_path = artifact_dir / "gt_vs_pred_scored.jsonl.provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["scored_artifact"]["sha256"] = sha256_file(scored_path)
    provenance_path.write_text(json.dumps(provenance, sort_keys=True), encoding="utf-8")

    with pytest.raises(ArtifactContractError) as exc_info:
        evaluate_scored_detection_artifacts(
            artifact_dir=artifact_dir,
            output_dir=tmp_path / "eval",
        )

    assert exc_info.value.code == "eval_detection.raw_scored_payload_mismatch"
    assert exc_info.value.context["field"] == "gt"
    assert not (tmp_path / "eval" / "metrics.json").exists()


def test_detection_consumer_refuses_raw_scored_image_mismatch(tmp_path: Path) -> None:
    from src.eval.detection_consumer import evaluate_scored_detection_artifacts
    from src.inference.artifacts import sha256_file

    artifact_dir = _write_scored_fixture(tmp_path / "artifacts")
    scored_path = artifact_dir / "gt_vs_pred_scored.jsonl"
    scored_rows = _read_jsonl(scored_path)
    scored_rows[0]["image_path"] = "different.jpg"
    _write_jsonl(scored_path, scored_rows)
    provenance_path = artifact_dir / "gt_vs_pred_scored.jsonl.provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["scored_artifact"]["sha256"] = sha256_file(scored_path)
    provenance_path.write_text(json.dumps(provenance, sort_keys=True), encoding="utf-8")

    with pytest.raises(ArtifactContractError) as exc_info:
        evaluate_scored_detection_artifacts(
            artifact_dir=artifact_dir,
            output_dir=tmp_path / "eval",
        )

    assert exc_info.value.code == "eval_detection.raw_scored_payload_mismatch"
    assert exc_info.value.context["field"] == "image_path"
    assert not (tmp_path / "eval" / "metrics.json").exists()


def test_evaluate_detection_cli_consumes_swift_artifact_dir(tmp_path: Path) -> None:
    artifact_dir = _write_scored_fixture(
        tmp_path / "artifacts",
        rows=[_raw_row("row-1", 0, text=OBJECT_TEXT)],
    )
    output_dir = tmp_path / "eval"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_detection.py",
            "--artifact-dir",
            str(artifact_dir),
            "--out-dir",
            str(output_dir),
        ],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["mAP"] == pytest.approx(1.0)
    assert completed.stdout.splitlines()[0].startswith("metrics: ")


def test_evaluate_detection_cli_consumes_swift_pred_jsonl_alias(tmp_path: Path) -> None:
    artifact_dir = _write_scored_fixture(
        tmp_path / "artifacts",
        rows=[_raw_row("row-1", 0, text=OBJECT_TEXT)],
    )
    output_dir = tmp_path / "eval"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_detection.py",
            "--pred-jsonl",
            str(artifact_dir / "gt_vs_pred_scored.jsonl"),
            "--out-dir",
            str(output_dir),
        ],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert (output_dir / "metrics.json").is_file()
    assert (output_dir / "coco_gt.json").is_file()
    assert (output_dir / "coco_predictions.json").is_file()
    stdout_lines = completed.stdout.splitlines()
    assert stdout_lines[0].startswith("metrics: ")
    assert json.loads(stdout_lines[1])["mAP"] == pytest.approx(1.0)


def test_evaluate_detection_cli_reports_contract_error_without_traceback(tmp_path: Path) -> None:
    output_dir = tmp_path / "eval"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_detection.py",
            "--artifact-dir",
            str(tmp_path / "missing"),
            "--out-dir",
            str(output_dir),
        ],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "Traceback" not in completed.stderr
    error = json.loads(completed.stderr)
    assert error["code"] == "eval_detection.missing_scored_artifact"


def _write_scored_fixture(
    output_dir: Path,
    *,
    rows: list[dict[str, Any]] | None = None,
    decode_results: dict[str, DecodeResult] | None = None,
) -> Path:
    from src.inference.artifacts import write_inference_artifacts

    if rows is None:
        rows = [
            _raw_row("row-1", 0, text=OBJECT_TEXT),
            _raw_row("row-2", 1, text="malformed"),
        ]
    if decode_results is None:
        decode_results = {
            str(row["row_id"]): _decode_result(
                str(row["row_id"]),
                text=str(row["raw_decode_text"]),
            )
            for row in rows
            if row["parse"].predictions
        }
    write_inference_artifacts(
        output_dir=output_dir,
        rows=rows,
        decode_results=decode_results,
        image_plan_rows=[
            {"row_id": row["row_id"], "row_index": row["row_index"]}
            for row in rows
        ],
        metadata={
            "artifact_schema_version": 1,
            "detection_template_id": "compact-object-box-closed",
            "prompt_policy_fingerprint": "prompt-fp",
            "generation_config_fingerprint": "gen-fp",
            "model_identity_fingerprint": "model-fp",
            "processor_identity_fingerprint": "processor-fp",
            "template_identity": {"id": "template-v1"},
            "parser_policy": "compact_object_box_closed_only",
            "dataset_identity": {"name": "unit"},
            "backend": "hf",
            "backend_mode": "generate",
            "response_family": "hf",
        },
    )
    return output_dir


def _raw_row(
    row_id: str,
    row_index: int,
    *,
    text: str,
    gt_description: str = "cat",
    image_width: int = 1000,
    image_height: int = 1000,
) -> dict[str, Any]:
    parse_row = parse_compact_object_box_closed(
        text,
        row_id=row_id,
        row_index=row_index,
        image_width=image_width,
        image_height=image_height,
    )
    return {
        "row_id": row_id,
        "row_index": row_index,
        "example_id": row_id,
        "image_path": f"{row_id}.jpg",
        "image_width": image_width,
        "image_height": image_height,
        "gt": [{"description": gt_description, "bbox": [100, 200, 300, 400]}],
        "raw_decode_text": text,
        "parse": parse_row,
    }


def _decode_result(row_id: str, *, text: str = OBJECT_TEXT) -> DecodeResult:
    pieces = [
        "<|object_ref_start|>",
        _description_from_text(text),
        "<|object_ref_end|>",
        "<|box_start|>",
        "<|coord_100|>",
        "<|coord_200|>",
        "<|coord_300|>",
        "<|coord_400|>",
        "<|box_end|>",
    ]
    trace = [
        TokenTrace(
            step_index=index,
            token_id=151646 + index,
            token_text=piece,
            logprob=math.log(0.25),
            is_stop=False,
            is_pad=False,
            backend="hf",
            backend_mode="generate",
            response_family="hf",
        )
        for index, piece in enumerate(pieces)
    ]
    return DecodeResult(
        request_id=row_id,
        backend="hf",
        backend_mode="generate",
        response_family="hf",
        prompt_token_ids=[11, 12],
        generated_token_ids=[item.token_id for item in trace],
        raw_generated_text=text,
        parser_text=text,
        strip_policy="none",
        stop_reason="length",
        model_identity={"family": "unit"},
        tokenizer_identity={"sha256": "tok"},
        generation_config_fingerprint="gen-fp",
        token_trace=trace,
    )


def _description_from_text(text: str) -> str:
    start = text.index("<|object_ref_start|>") + len("<|object_ref_start|>")
    end = text.index("<|object_ref_end|>")
    return text[start:end]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
