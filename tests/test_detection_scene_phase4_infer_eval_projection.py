from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.eval.artifacts import with_constant_scores
from src.infer.artifacts import load_comparable_artifact
from src.infer.parsing import (
    DecodedDetectionResult,
    DetectionParserResult,
    diagnostic_parser_result,
    require_metric_bearing,
    strict_parser_result,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_decoded_detection_result_is_canonical_metric_boundary() -> None:
    strict = strict_parser_result(
        predictions=({"type": "bbox_2d", "points": [1, 2, 3, 4], "desc": "cat"},),
        parser_id="coordjson",
        diagnostics={"dropped_invalid": 0},
    )

    assert isinstance(strict, DecodedDetectionResult)
    assert DetectionParserResult is DecodedDetectionResult
    assert require_metric_bearing(strict, consumer="official_eval") is strict
    assert strict.metric_bearing is True
    assert strict.salvage_recovered is False
    assert strict.predictions == (
        {"type": "bbox_2d", "points": [1, 2, 3, 4], "desc": "cat"},
    )
    assert strict.diagnostics["dropped_invalid"] == 0
    assert not hasattr(strict, "raw_text")
    assert not hasattr(strict, "raw_output_json")
    assert not hasattr(strict, "eval_record")


def test_diagnostic_decoded_detection_result_is_excluded_from_metrics() -> None:
    diagnostic = diagnostic_parser_result(
        predictions=({"type": "bbox_2d", "points": [1, 2, 3, 4], "desc": "cat"},),
        parser_id="coordjson_salvage",
        errors=("dropped_invalid",),
        diagnostics={"dropped_invalid": 1, "drop_reason": "bbox_invalid"},
        salvage_recovered=True,
    )

    assert isinstance(diagnostic, DecodedDetectionResult)
    assert diagnostic.metric_bearing is False
    assert diagnostic.salvage_recovered is True
    assert diagnostic.errors == ("dropped_invalid",)
    assert diagnostic.diagnostics["dropped_invalid"] == 1
    with pytest.raises(ValueError, match="metric_bearing=false"):
        require_metric_bearing(diagnostic, consumer="official_eval")


def test_offline_eval_record_materialization_consumes_decoded_result() -> None:
    from src.infer.runtime import (
        decode_offline_detection_result,
        materialize_offline_gt_vs_pred_record,
    )

    class _Coord:
        def process_prediction_text(self, raw_text, *, width, height, errors):
            assert raw_text == "backend text that is not the semantic object"
            assert (width, height) == (64, 48)
            return [
                {
                    "type": "bbox_2d",
                    "points": [0, 0, 63, 47],
                    "desc": "box",
                    "backend_private": "drop-me",
                }
            ]

    owner = SimpleNamespace(
        detection_sequence_format="coordjson",
        coord=_Coord(),
        cfg=SimpleNamespace(compact_full_parse_mode="strict"),
    )
    decoded = decode_offline_detection_result(
        owner,
        "backend text that is not the semantic object",
        width=64,
        height=48,
    )

    record = materialize_offline_gt_vs_pred_record(
        image="img.png",
        width=64,
        height=48,
        mode="text",
        gt=[],
        decoded_result=decoded,
        raw_output_json={"objects": [{"raw": "kept only as compatibility metadata"}]},
        raw_special_tokens=["<|im_end|>"],
        raw_ends_with_im_end=True,
    )

    assert isinstance(decoded, DecodedDetectionResult)
    assert record["pred"] == [
        {"type": "bbox_2d", "points": [0, 0, 63, 47], "desc": "box"}
    ]
    assert record["errors"] == []
    assert record["raw_output_json"] == {
        "objects": [{"raw": "kept only as compatibility metadata"}]
    }
    assert not hasattr(decoded, "raw_text")
    assert not hasattr(decoded, "raw_output_json")


def test_official_gt_vs_pred_materialization_rejects_diagnostic_result() -> None:
    from src.infer.runtime import materialize_offline_gt_vs_pred_record

    diagnostic = diagnostic_parser_result(
        predictions=({"type": "bbox_2d", "points": [1, 2, 3, 4], "desc": "cat"},),
        parser_id="coordjson",
        errors=("dropped_invalid",),
        diagnostics={"dropped_invalid": 1},
        salvage_recovered=False,
    )

    with pytest.raises(ValueError, match="metric_bearing=false"):
        materialize_offline_gt_vs_pred_record(
            image="img.png",
            width=64,
            height=48,
            mode="text",
            gt=[],
            decoded_result=diagnostic,
            raw_output_json={"objects": []},
            raw_special_tokens=[],
            raw_ends_with_im_end=False,
        )


def test_raw_eval_jsonl_loads_detection_eval_records_without_schema_change(
    tmp_path: Path,
) -> None:
    from src.eval.detection_records import DetectionEvalRecord, load_jsonl, preds_to_gt_records

    row = {
        "image": "img.png",
        "width": 64,
        "height": 48,
        "mode": "text",
        "gt": [{"type": "bbox_2d", "points": [0, 0, 10, 10], "desc": "cat"}],
        "pred": [{"type": "bbox_2d", "points": [1, 1, 9, 9], "desc": "cat"}],
        "coord_mode": "pixel",
        "raw_output_json": {"objects": []},
        "raw_special_tokens": [],
        "raw_ends_with_im_end": False,
        "errors": [],
        "error_entries": [],
    }
    path = tmp_path / "gt_vs_pred.jsonl"
    _write_jsonl(path, [row])

    records = load_jsonl(path)
    gt_records = preds_to_gt_records(records)

    assert records and isinstance(records[0], DetectionEvalRecord)
    assert records[0]["image"] == "img.png"
    assert records[0].get("width") == 64
    assert records[0].to_json_record() == row
    assert json.loads(json.dumps(records[0].to_json_record())) == row
    assert gt_records and isinstance(gt_records[0], DetectionEvalRecord)
    assert gt_records[0].to_json_record() == {
        "images": ["img.png"],
        "width": 64,
        "height": 48,
        "objects": [{"bbox_2d": [0, 0, 10, 10], "desc": "cat"}],
        "image_id": None,
        "metadata": {},
    }


def test_detection_eval_and_scored_records_are_detached_schema_views() -> None:
    from src.eval.detection_records import (
        DetectionEvalRecord,
        ScoredDetectionEvalRecord,
    )

    raw_row = {
        "image": "img.png",
        "width": 64,
        "height": 48,
        "gt": [],
        "pred": [{"type": "bbox_2d", "points": [0, 0, 63, 47], "desc": "box"}],
    }

    raw_record = DetectionEvalRecord.from_json_record(raw_row)
    raw_json = raw_record.to_json_record()
    scored_rows = with_constant_scores(
        records=[raw_json],
        pred_score_source="constant_test",
        pred_score_version=1,
        constant_score=0.5,
    )
    scored_record = ScoredDetectionEvalRecord.from_json_record(
        scored_rows[0],
        score_provenance={"score_policy_fingerprint": "score:test"},
    )

    assert raw_record.to_json_record() == raw_row
    assert "pred_score_source" not in raw_row
    assert "score" not in raw_row["pred"][0]
    assert scored_record.to_json_record() == {
        "image": "img.png",
        "width": 64,
        "height": 48,
        "gt": [],
        "pred": [
            {
                "type": "bbox_2d",
                "points": [0, 0, 63, 47],
                "desc": "box",
                "score": 0.5,
            }
        ],
        "pred_score_source": "constant_test",
        "pred_score_version": 1,
    }
    assert scored_record.score_provenance["score_policy_fingerprint"] == "score:test"


def test_confidence_postop_scored_path_uses_scored_eval_record(
    tmp_path: Path,
) -> None:
    from src.eval.confidence_postop import (
        PRED_SCORE_SOURCE,
        PRED_SCORE_VERSION,
        ConfidencePostOpPaths,
        _build_scored_record,
        run_confidence_postop,
    )
    from src.eval.detection_records import ScoredDetectionEvalRecord

    row = {
        "image": "img.png",
        "width": 64,
        "height": 48,
        "gt": [],
        "pred": [],
    }
    scored_record = _build_scored_record(record=row, confidence_objects=[])
    assert isinstance(scored_record, ScoredDetectionEvalRecord)
    assert scored_record.to_json_record() == {
        **row,
        "pred_score_source": PRED_SCORE_SOURCE,
        "pred_score_version": PRED_SCORE_VERSION,
    }

    raw_path = tmp_path / "gt_vs_pred.jsonl"
    trace_path = tmp_path / "pred_token_trace.jsonl"
    confidence_path = tmp_path / "pred_confidence.jsonl"
    scored_path = tmp_path / "gt_vs_pred_scored.jsonl"
    summary_path = tmp_path / "confidence_postop_summary.json"
    _write_jsonl(raw_path, [row])
    trace_path.write_text("", encoding="utf-8")

    summary = run_confidence_postop(
        ConfidencePostOpPaths(
            gt_vs_pred_jsonl=raw_path,
            pred_token_trace_jsonl=trace_path,
            pred_confidence_jsonl=confidence_path,
            gt_vs_pred_scored_jsonl=scored_path,
            confidence_postop_summary_json=summary_path,
        )
    )

    scored_rows = [
        json.loads(line)
        for line in scored_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert summary["total_samples"] == 1
    assert scored_rows == [scored_record.to_json_record()]
    assert scored_path.name == "gt_vs_pred_scored.jsonl"


def test_stable_artifact_filenames_and_scored_provenance_remain_loadable(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "gt_vs_pred.jsonl"
    scored_path = tmp_path / "gt_vs_pred_scored.jsonl"
    row = {
        "image": "img.png",
        "width": 64,
        "height": 48,
        "gt": [],
        "pred": [],
    }
    _write_jsonl(raw_path, [row])
    _write_jsonl(scored_path, [{**row, "pred_score_source": "test", "pred_score_version": 1}])
    scored_path.with_suffix(scored_path.suffix + ".provenance.json").write_text(
        json.dumps(
            {
                "prompt_policy_fingerprint": "prompt:test",
                "decode_policy_fingerprint": "decode:test",
                "model_identity_fingerprint": "model:test",
                "score_policy_fingerprint": "score:test",
                "artifact_path": str(scored_path),
                "metric_bearing": True,
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )

    loaded = load_comparable_artifact(scored_path, require_score=True)

    assert raw_path.name == "gt_vs_pred.jsonl"
    assert scored_path.name == "gt_vs_pred_scored.jsonl"
    assert loaded["artifact_path"] == str(scored_path)
    assert loaded["provenance"]["score_policy_fingerprint"] == "score:test"
