from __future__ import annotations

import json
import math
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


def test_detection_consumer_writes_explicit_minimal_metrics_for_scored_fixture(tmp_path: Path) -> None:
    from src.eval.detection_consumer import evaluate_scored_detection_artifacts

    artifact_dir = _write_scored_fixture(tmp_path / "artifacts")
    result = evaluate_scored_detection_artifacts(
        artifact_dir=artifact_dir,
        output_dir=tmp_path / "eval",
    )

    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))

    assert result.metrics_path == tmp_path / "eval" / "metrics.json"
    assert metrics["metric_artifact_name"] == "metrics.json"
    assert metrics["metric_family"] == "coordexp_swift_detection_counts_v1"
    assert metrics["benchmark_metric"] is False
    assert metrics["row_count"] == 2
    assert metrics["gt_object_count"] == 2
    assert metrics["pred_object_count"] == 1
    assert metrics["scored_pred_count"] == 1
    assert "map" not in metrics


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


def _write_scored_fixture(output_dir: Path) -> Path:
    from src.inference.artifacts import write_inference_artifacts

    write_inference_artifacts(
        output_dir=output_dir,
        rows=[
            _raw_row("row-1", 0, text=OBJECT_TEXT),
            _raw_row("row-2", 1, text="malformed"),
        ],
        decode_results={"row-1": _decode_result("row-1")},
        image_plan_rows=[{"row_id": "row-1"}, {"row_id": "row-2"}],
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


def _raw_row(row_id: str, row_index: int, *, text: str) -> dict[str, Any]:
    parse_row = parse_compact_object_box_closed(
        text,
        row_id=row_id,
        row_index=row_index,
        image_width=1000,
        image_height=1000,
    )
    return {
        "row_id": row_id,
        "row_index": row_index,
        "example_id": row_id,
        "image_path": f"{row_id}.jpg",
        "image_width": 1000,
        "image_height": 1000,
        "gt": [{"description": "gt-cat", "bbox": [100, 100, 300, 300]}],
        "raw_decode_text": text,
        "parse": parse_row,
    }


def _decode_result(row_id: str) -> DecodeResult:
    pieces = [
        "<|object_ref_start|>",
        "cat",
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
        raw_generated_text=OBJECT_TEXT,
        parser_text=OBJECT_TEXT,
        strip_policy="none",
        stop_reason="length",
        model_identity={"family": "unit"},
        tokenizer_identity={"sha256": "tok"},
        generation_config_fingerprint="gen-fp",
        token_trace=trace,
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
