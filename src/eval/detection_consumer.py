"""Minimal CoordExp-swift detection consumer for scored inference artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.common.errors import ArtifactContractError
from src.inference.artifacts import PROVENANCE_NAME, SCORED_NAME, sha256_file


METRICS_NAME = "metrics.json"
METRIC_FAMILY = "coordexp_swift_detection_counts_v1"


@dataclass(frozen=True)
class DetectionConsumerResult:
    metrics_path: Path
    metrics: dict[str, Any]


def evaluate_scored_detection_artifacts(
    *,
    artifact_dir: str | Path,
    output_dir: str | Path,
    metrics_name: str = METRICS_NAME,
) -> DetectionConsumerResult:
    artifact_root = Path(artifact_dir)
    scored_path = artifact_root / SCORED_NAME
    provenance_path = artifact_root / PROVENANCE_NAME
    _require_file(scored_path, code="eval_detection.missing_scored_artifact")
    _require_file(provenance_path, code="eval_detection.missing_provenance")
    rows = _read_jsonl(scored_path)
    provenance = _read_json(provenance_path)
    _validate_provenance(scored_path=scored_path, rows=rows, provenance=provenance)
    metrics = _count_metrics(rows, metric_artifact_name=metrics_name)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    metrics_path = output_root / metrics_name
    _write_json(metrics_path, metrics)
    return DetectionConsumerResult(metrics_path=metrics_path, metrics=metrics)


def _validate_provenance(
    *,
    scored_path: Path,
    rows: list[dict[str, Any]],
    provenance: dict[str, Any],
) -> None:
    scored_artifact = provenance.get("scored_artifact")
    if not isinstance(scored_artifact, dict) or scored_artifact.get("path") != SCORED_NAME:
        raise ArtifactContractError(
            "scored artifact provenance does not name gt_vs_pred_scored.jsonl",
            code="eval_detection.scored_artifact_binding_missing",
            context={"expected": SCORED_NAME},
        )
    if scored_artifact.get("sha256") != sha256_file(scored_path):
        raise ArtifactContractError(
            "scored artifact sha does not match provenance",
            code="eval_detection.scored_sha_mismatch",
            context={"path": str(scored_path)},
        )
    row_binding = provenance.get("row_binding")
    if not isinstance(row_binding, dict):
        raise ArtifactContractError(
            "scored provenance is missing row binding evidence",
            code="eval_detection.row_binding_missing",
        )
    if int(row_binding.get("row_count", -1)) != len(rows):
        raise ArtifactContractError(
            "scored provenance row count does not match scored artifact",
            code="eval_detection.provenance_row_count_mismatch",
            context={"expected": row_binding.get("row_count"), "observed": len(rows)},
        )
    row_ids = [str(row.get("row_id")) for row in rows]
    expected_row_ids_sha = hashlib.sha256(
        json.dumps(row_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if row_binding.get("row_ids_sha256") != expected_row_ids_sha:
        raise ArtifactContractError(
            "scored provenance row identity binding does not match scored artifact",
            code="eval_detection.provenance_row_ids_mismatch",
        )
    for field in (
        "raw_artifact",
        "detection_template_id",
        "prompt_policy_fingerprint",
        "decode_policy_fingerprint",
        "model_identity_fingerprint",
        "processor_identity_fingerprint",
        "template_identity",
        "parser_policy",
        "score_policy_fingerprint",
    ):
        if not provenance.get(field):
            raise ArtifactContractError(
                "scored provenance is missing required evaluator binding field",
                code="eval_detection.provenance_field_missing",
                context={"field": field},
            )
    _validate_row_local_scores(rows)


def _validate_row_local_scores(rows: list[dict[str, Any]]) -> None:
    for row_index, row in enumerate(rows):
        if "row_id" not in row:
            raise ArtifactContractError(
                "scored row is missing row_id",
                code="eval_detection.row_id_missing",
                context={"row_index": row_index},
            )
        pred = row.get("pred")
        if not isinstance(pred, list):
            raise ArtifactContractError(
                "scored row pred must be a list",
                code="eval_detection.pred_shape",
                context={"row_id": row.get("row_id")},
            )
        for pred_index, item in enumerate(pred):
            if not item.get("pred_score_source") or not isinstance(item.get("pred_score_version"), int):
                raise ArtifactContractError(
                    "scored prediction is missing row-local score provenance",
                    code="eval_detection.row_score_provenance_missing",
                    context={"row_id": row.get("row_id"), "pred_index": pred_index},
                )
            score = item.get("score")
            if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
                raise ArtifactContractError(
                    "scored prediction score must be finite",
                    code="eval_detection.invalid_score",
                    context={"row_id": row.get("row_id"), "pred_index": pred_index},
                )
            if float(score) < 0.0 or float(score) > 1.0:
                raise ArtifactContractError(
                    "scored prediction score must be in [0.0, 1.0]",
                    code="eval_detection.invalid_score",
                    context={"row_id": row.get("row_id"), "pred_index": pred_index},
                )


def _count_metrics(rows: list[dict[str, Any]], *, metric_artifact_name: str) -> dict[str, Any]:
    gt_object_count = sum(len(row.get("gt") or []) for row in rows)
    pred_object_count = sum(len(row.get("pred") or []) for row in rows)
    return {
        "metric_artifact_name": metric_artifact_name,
        "metric_family": METRIC_FAMILY,
        "benchmark_metric": False,
        "metric_scope": "minimal_scored_artifact_consumer",
        "row_count": len(rows),
        "gt_object_count": gt_object_count,
        "pred_object_count": pred_object_count,
        "scored_pred_count": pred_object_count,
        "empty_pred_row_count": sum(1 for row in rows if not row.get("pred")),
    }


def _require_file(path: Path, *, code: str) -> None:
    if not path.is_file():
        raise ArtifactContractError(
            "required detection consumer input artifact is missing",
            code=code,
            context={"path": str(path)},
        )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for row_index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ArtifactContractError(
                "scored artifact row must be a JSON object",
                code="eval_detection.row_shape",
                context={"path": str(path), "row_index": row_index},
            )
        rows.append(payload)
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ArtifactContractError(
            "provenance artifact must be a JSON object",
            code="eval_detection.provenance_shape",
            context={"path": str(path)},
        )
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
