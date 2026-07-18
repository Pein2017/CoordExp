"""Minimal CoordExp-swift detection consumer for scored inference artifacts."""

from __future__ import annotations

from collections import Counter
import contextlib
import copy
import hashlib
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.common.errors import ArtifactContractError, DataContractError
from src.data.geometry import coord_bins_to_pixel_xyxy
from src.eval.detection_categories import (
    COCO_80_CATEGORY_IDS,
    COCO_80_CLASS_NAMES,
    normalize_coco_category_name,
)
from src.inference.artifacts import benchmark_scope_eligible


RAW_NAME = "gt_vs_pred.jsonl"
SCORED_NAME = "gt_vs_pred_scored.jsonl"
PROVENANCE_NAME = "gt_vs_pred_scored.jsonl.provenance.json"
METRICS_NAME = "metrics.json"
RECEIPT_NAME = "evaluation_receipt.json"
METRIC_FAMILY = "coordexp_swift_detection_coco_bbox_v1"
COCO_GT_NAME = "coco_gt.json"
COCO_PREDICTIONS_NAME = "coco_predictions.json"
RUN_MANIFEST_NAME = "run_manifest.json"
SUMMARY_NAME = "summary.json"
SUPPORTED_PRED_SCORE_VERSION = 1
SUPPORTED_SCORE_SOURCE_KIND = "token_trace_selected_logprob_mean"


@dataclass(frozen=True)
class DetectionConsumerResult:
    metrics_path: Path
    receipt_path: Path
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
    _validate_provenance(
        artifact_root=artifact_root,
        scored_path=scored_path,
        rows=rows,
        provenance=provenance,
    )
    raw_rows = _read_jsonl(artifact_root / RAW_NAME)
    normalized_rows = _normalize_eval_rows(scored_rows=rows, raw_rows=raw_rows)
    coco_gt, coco_predictions, conversion_metrics = _to_coco_eval_artifacts(normalized_rows)
    official_metrics = _run_coco_bbox_eval(coco_gt, coco_predictions)
    receipt = _evaluation_receipt(
        artifact_root=artifact_root,
        provenance=provenance,
        rows=rows,
    )
    metrics = _count_metrics(
        rows,
        normalized_rows=normalized_rows,
        conversion_metrics=conversion_metrics,
        official_metrics=official_metrics,
        metric_artifact_name=metrics_name,
        receipt=receipt,
    )
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / COCO_GT_NAME, coco_gt)
    _write_json(output_root / COCO_PREDICTIONS_NAME, coco_predictions)
    receipt_path = output_root / RECEIPT_NAME
    _write_json(receipt_path, receipt)
    metrics_path = output_root / metrics_name
    _write_json(metrics_path, metrics)
    return DetectionConsumerResult(
        metrics_path=metrics_path,
        receipt_path=receipt_path,
        metrics=metrics,
    )


def _validate_provenance(
    *,
    artifact_root: Path,
    scored_path: Path,
    rows: list[dict[str, Any]],
    provenance: dict[str, Any],
) -> None:
    raw_artifact = provenance.get("raw_artifact")
    if not isinstance(raw_artifact, dict) or raw_artifact.get("path") != RAW_NAME:
        raise ArtifactContractError(
            "scored artifact provenance does not name gt_vs_pred.jsonl",
            code="eval_detection.raw_artifact_binding_missing",
            context={"expected": RAW_NAME},
        )
    raw_path = artifact_root / RAW_NAME
    _require_file(raw_path, code="eval_detection.missing_raw_artifact")
    if raw_artifact.get("sha256") != sha256_file(raw_path):
        raise ArtifactContractError(
            "raw artifact sha does not match provenance",
            code="eval_detection.raw_sha_mismatch",
            context={"path": str(raw_path)},
        )
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
    _validate_row_local_scores(
        rows,
        score_policy_fingerprint=str(provenance["score_policy_fingerprint"]),
    )


def _validate_row_local_scores(
    rows: list[dict[str, Any]],
    *,
    score_policy_fingerprint: str,
) -> None:
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
        row_id = str(row["row_id"])
        for pred_index, item in enumerate(pred):
            pred_score_version = item.get("pred_score_version")
            pred_score_source = item.get("pred_score_source")
            if not pred_score_source or not isinstance(pred_score_version, int):
                raise ArtifactContractError(
                    "scored prediction is missing row-local score provenance",
                    code="eval_detection.row_score_provenance_missing",
                    context={"row_id": row_id, "pred_index": pred_index},
                )
            if pred_score_version != SUPPORTED_PRED_SCORE_VERSION:
                raise ArtifactContractError(
                    "scored prediction score provenance version is unsupported",
                    code="eval_detection.row_score_version_unsupported",
                    context={
                        "row_id": row_id,
                        "pred_index": pred_index,
                        "pred_score_version": pred_score_version,
                        "expected": SUPPORTED_PRED_SCORE_VERSION,
                    },
                )
            _validate_pred_score_source(
                pred_score_source,
                row_id=row_id,
                pred_index=pred_index,
                object_span_id=item.get("object_span_id"),
                score_policy_fingerprint=score_policy_fingerprint,
            )
            score = item.get("score")
            if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
                raise ArtifactContractError(
                    "scored prediction score must be finite",
                    code="eval_detection.invalid_score",
                    context={"row_id": row_id, "pred_index": pred_index},
                )
            if float(score) < 0.0 or float(score) > 1.0:
                raise ArtifactContractError(
                    "scored prediction score must be in [0.0, 1.0]",
                    code="eval_detection.invalid_score",
                    context={"row_id": row_id, "pred_index": pred_index},
                )


def _validate_pred_score_source(
    source: Any,
    *,
    row_id: str,
    pred_index: int,
    object_span_id: Any,
    score_policy_fingerprint: str,
) -> None:
    if not isinstance(source, dict):
        raise ArtifactContractError(
            "scored prediction score provenance must be a mapping",
            code="eval_detection.row_score_provenance_shape",
            context={"row_id": row_id, "pred_index": pred_index},
        )
    if source.get("kind") != SUPPORTED_SCORE_SOURCE_KIND:
        raise ArtifactContractError(
            "scored prediction score source kind is unsupported",
            code="eval_detection.row_score_source_kind_unsupported",
            context={
                "row_id": row_id,
                "pred_index": pred_index,
                "kind": source.get("kind"),
                "expected": SUPPORTED_SCORE_SOURCE_KIND,
            },
        )
    if str(source.get("row_id")) != row_id:
        raise ArtifactContractError(
            "scored prediction score source row_id does not match scored row",
            code="eval_detection.row_score_provenance_mismatch",
            context={
                "row_id": row_id,
                "pred_index": pred_index,
                "source_row_id": source.get("row_id"),
            },
        )
    if object_span_id is None or str(source.get("object_span_id")) != str(object_span_id):
        raise ArtifactContractError(
            "scored prediction score source object_span_id does not match prediction",
            code="eval_detection.row_score_provenance_mismatch",
            context={
                "row_id": row_id,
                "pred_index": pred_index,
                "object_span_id": object_span_id,
                "source_object_span_id": source.get("object_span_id"),
            },
        )
    if source.get("score_policy_fingerprint") != score_policy_fingerprint:
        raise ArtifactContractError(
            "scored prediction score source policy does not match artifact provenance",
            code="eval_detection.row_score_policy_mismatch",
            context={
                "row_id": row_id,
                "pred_index": pred_index,
                "source_score_policy_fingerprint": source.get("score_policy_fingerprint"),
                "expected": score_policy_fingerprint,
            },
        )
    selected_count = source.get("selected_count")
    if not isinstance(selected_count, int) or isinstance(selected_count, bool) or selected_count <= 0:
        raise ArtifactContractError(
            "scored prediction score source must include positive selected_count",
            code="eval_detection.row_score_provenance_shape",
            context={"row_id": row_id, "pred_index": pred_index},
        )
    for field in (
        "generated_step_indices",
        "token_ids",
        "token_text",
        "selected_logprobs",
    ):
        values = source.get(field)
        if not isinstance(values, list) or len(values) != selected_count:
            raise ArtifactContractError(
                "scored prediction score source selected-token evidence is incomplete",
                code="eval_detection.row_score_provenance_shape",
                context={
                    "row_id": row_id,
                    "pred_index": pred_index,
                    "field": field,
                    "selected_count": selected_count,
                },
            )
    for selected_index, logprob in enumerate(source["selected_logprobs"]):
        if not isinstance(logprob, (int, float)) or not math.isfinite(float(logprob)):
            raise ArtifactContractError(
                "scored prediction score source selected logprobs must be finite",
                code="eval_detection.row_score_provenance_shape",
                context={
                    "row_id": row_id,
                    "pred_index": pred_index,
                    "selected_index": selected_index,
                },
            )


def _normalize_eval_rows(
    *,
    scored_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(scored_rows) != len(raw_rows):
        raise ArtifactContractError(
            "raw and scored detection artifacts must have the same row count",
            code="eval_detection.raw_scored_row_count_mismatch",
            context={
                "raw_row_count": len(raw_rows),
                "scored_row_count": len(scored_rows),
            },
        )
    normalized: list[dict[str, Any]] = []
    for row_index, (scored, raw) in enumerate(zip(scored_rows, raw_rows, strict=True)):
        scored_row_id = str(scored.get("row_id"))
        raw_row_id = str(raw.get("row_id"))
        if scored_row_id != raw_row_id:
            raise ArtifactContractError(
                "raw and scored detection rows must preserve row identity",
                code="eval_detection.raw_scored_row_id_mismatch",
                context={
                    "row_index": row_index,
                    "raw_row_id": raw_row_id,
                    "scored_row_id": scored_row_id,
                },
            )
        _validate_raw_scored_payload_parity(
            raw=raw,
            scored=scored,
            row_id=scored_row_id,
            row_index=row_index,
        )
        normalized.append(
            {
                "row_id": scored_row_id,
                "row_index": int(raw.get("row_index", row_index)),
                "image_id": row_index + 1,
                "image_path": str(
                    raw.get("image_path") or scored.get("image_path") or scored_row_id
                ),
                "image_width": _positive_int(
                    raw.get("image_width", scored.get("image_width")),
                    field="image_width",
                    row_id=scored_row_id,
                ),
                "image_height": _positive_int(
                    raw.get("image_height", scored.get("image_height")),
                    field="image_height",
                    row_id=scored_row_id,
                ),
                "gt": _require_object_list(raw.get("gt"), field="gt", row_id=scored_row_id),
                "pred": _require_object_list(scored.get("pred"), field="pred", row_id=scored_row_id),
                "parse_status": str(raw.get("parse_status", "unknown")),
                "raw_metric_bearing": bool(raw.get("metric_bearing", False)),
                "raw_dropped_prediction_count": int(raw.get("dropped_prediction_count") or 0),
                "raw_valid_prediction_count": int(raw.get("valid_prediction_count") or 0),
            }
        )
    return normalized


def _validate_raw_scored_payload_parity(
    *,
    raw: dict[str, Any],
    scored: dict[str, Any],
    row_id: str,
    row_index: int,
) -> None:
    scalar_fields = (
        "row_index",
        "example_id",
        "image_path",
        "image_width",
        "image_height",
    )
    for field in scalar_fields:
        if field in raw or field in scored:
            _require_raw_scored_field_equal(
                raw.get(field),
                scored.get(field),
                field=field,
                row_id=row_id,
                row_index=row_index,
            )
    _require_raw_scored_field_equal(
        _canonical_json_value(raw.get("gt", [])),
        _canonical_json_value(scored.get("gt", [])),
        field="gt",
        row_id=row_id,
        row_index=row_index,
    )


def _require_raw_scored_field_equal(
    raw_value: Any,
    scored_value: Any,
    *,
    field: str,
    row_id: str,
    row_index: int,
) -> None:
    if raw_value == scored_value:
        return
    raise ArtifactContractError(
        "raw and scored detection artifacts must preserve immutable row payload fields",
        code="eval_detection.raw_scored_payload_mismatch",
        context={
            "row_id": row_id,
            "row_index": row_index,
            "field": field,
            "raw_value": raw_value,
            "scored_value": scored_value,
        },
    )


def _canonical_json_value(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, separators=(",", ":")))


def _to_coco_eval_artifacts(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, int]]:
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    counters: Counter[str] = Counter(
        {
            "unknown_category_pred_count": 0,
            "invalid_pred_bbox_count": 0,
        }
    )
    annotation_id = 1
    for row in rows:
        image_id = int(row["image_id"])
        images.append(
            {
                "id": image_id,
                "file_name": row["image_path"],
                "width": int(row["image_width"]),
                "height": int(row["image_height"]),
                "row_id": row["row_id"],
            }
        )
        for gt_index, obj in enumerate(row["gt"]):
            category_id = _category_id_for_gt(obj, row=row, object_index=gt_index)
            x1, y1, x2, y2 = _gt_coord_bins_to_pixel_xyxy(
                obj,
                row=row,
                object_index=gt_index,
            )
            width = x2 - x1
            height = y2 - y1
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x1, y1, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
        for pred_index, obj in enumerate(row["pred"]):
            category_id = _category_id_for_pred(obj)
            if category_id is None:
                counters["unknown_category_pred_count"] += 1
                continue
            try:
                x1, y1, x2, y2 = _xyxy_bbox(
                    obj,
                    row=row,
                    object_index=pred_index,
                    object_kind="pred",
                )
            except ArtifactContractError:
                counters["invalid_pred_bbox_count"] += 1
                continue
            predictions.append(
                {
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": _score(obj, row=row, pred_index=pred_index),
                }
            )
    coco_gt = {
        "info": {},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": index + 1, "name": name}
            for index, name in enumerate(COCO_80_CLASS_NAMES)
        ],
    }
    counters["coco_prediction_count"] = len(predictions)
    return coco_gt, predictions, dict(counters)


def _run_coco_bbox_eval(
    coco_gt_payload: dict[str, Any],
    predictions: list[dict[str, Any]],
) -> dict[str, float]:
    metric_suffixes = (
        "AP",
        "AP50",
        "AP75",
        "APs",
        "APm",
        "APl",
        "AR1",
        "AR10",
        "AR100",
        "ARs",
        "ARm",
        "ARl",
    )
    if not coco_gt_payload["annotations"] or not predictions:
        zeros = {f"bbox_{suffix}": 0.0 for suffix in metric_suffixes}
        return {
            **zeros,
            "mAP": 0.0,
            "mAP_50": 0.0,
            "mAP_75": 0.0,
            "mRecall": 0.0,
            "mRecall_1": 0.0,
            "mRecall_10": 0.0,
            "mRecall_100": 0.0,
            "map": 0.0,
            "map_50": 0.0,
            "map_75": 0.0,
            "mrecall": 0.0,
        }

    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise ArtifactContractError(
            "pycocotools is required for official COCO bbox evaluation",
            code="eval_detection.pycocotools_unavailable",
            cause=exc,
        ) from exc

    coco_gt = COCO()
    coco_gt.dataset = copy.deepcopy(coco_gt_payload)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(copy.deepcopy(predictions))
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.params.imgIds = [
            int(image["id"]) for image in coco_gt_payload["images"]
        ]
        coco_eval.params.catIds = [
            int(category["id"]) for category in coco_gt_payload["categories"]
        ]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    stats = [float(value) for value in coco_eval.stats]
    return {
        "bbox_AP": stats[0],
        "bbox_AP50": stats[1],
        "bbox_AP75": stats[2],
        "bbox_APs": stats[3],
        "bbox_APm": stats[4],
        "bbox_APl": stats[5],
        "bbox_AR1": stats[6],
        "bbox_AR10": stats[7],
        "bbox_AR100": stats[8],
        "bbox_ARs": stats[9],
        "bbox_ARm": stats[10],
        "bbox_ARl": stats[11],
        "mAP": stats[0],
        "mAP_50": stats[1],
        "mAP_75": stats[2],
        "mRecall": stats[8],
        "mRecall_1": stats[6],
        "mRecall_10": stats[7],
        "mRecall_100": stats[8],
        "map": stats[0],
        "map_50": stats[1],
        "map_75": stats[2],
        "mrecall": stats[8],
    }


def _count_metrics(
    rows: list[dict[str, Any]],
    *,
    normalized_rows: list[dict[str, Any]],
    conversion_metrics: dict[str, int],
    official_metrics: dict[str, float],
    metric_artifact_name: str,
    receipt: dict[str, Any],
) -> dict[str, Any]:
    gt_object_count = sum(len(row.get("gt") or []) for row in rows)
    pred_object_count = sum(len(row.get("pred") or []) for row in rows)
    parse_status_counts = Counter(str(row["parse_status"]) for row in normalized_rows)
    return {
        "metric_artifact_name": metric_artifact_name,
        "metric_family": METRIC_FAMILY,
        "benchmark_metric": bool(receipt["benchmark_metric"]),
        "benchmark_eligible": bool(receipt["run_manifest"]["benchmark_eligible"]),
        "metric_scope": "coco_bbox",
        "artifact_dir": receipt["artifact_dir"],
        "evaluation_receipt_json": RECEIPT_NAME,
        "evaluation_receipt": receipt,
        "row_count": len(rows),
        "gt_object_count": gt_object_count,
        "input_scored_pred_count": pred_object_count,
        "pred_object_count": pred_object_count,
        "scored_pred_count": pred_object_count,
        "empty_pred_row_count": sum(1 for row in rows if not row.get("pred")),
        "parse_status_counts": dict(sorted(parse_status_counts.items())),
        "raw_dropped_prediction_count": sum(
            int(row["raw_dropped_prediction_count"]) for row in normalized_rows
        ),
        "raw_metric_bearing_false_row_count": sum(
            1 for row in normalized_rows if not row["raw_metric_bearing"]
        ),
        "coco_gt_json": COCO_GT_NAME,
        "coco_predictions_json": COCO_PREDICTIONS_NAME,
        **conversion_metrics,
        **official_metrics,
    }


def _evaluation_receipt(
    *,
    artifact_root: Path,
    provenance: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    artifacts = {
        RAW_NAME: _artifact_receipt(artifact_root / RAW_NAME),
        SCORED_NAME: _artifact_receipt(artifact_root / SCORED_NAME),
        PROVENANCE_NAME: _artifact_receipt(artifact_root / PROVENANCE_NAME),
    }
    manifest = _optional_json_artifact_receipt(artifact_root / RUN_MANIFEST_NAME)
    summary = _optional_json_artifact_receipt(artifact_root / SUMMARY_NAME)
    row_ids = [str(row.get("row_id")) for row in rows]
    row_ids_sha256 = hashlib.sha256(
        json.dumps(row_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    run_manifest_payload = manifest.get("payload")
    if isinstance(run_manifest_payload, dict):
        terminal_status = str(run_manifest_payload.get("terminal_status", "completed"))
        manifest_benchmark_eligible = bool(
            run_manifest_payload.get("benchmark_eligible", False)
        )
        benchmark_eligible = manifest_benchmark_eligible and benchmark_scope_eligible(
            len(rows)
        )
        evaluator_consumer_status = run_manifest_payload.get("evaluator_consumer_status")
        artifacts[RUN_MANIFEST_NAME] = {
            "path": RUN_MANIFEST_NAME,
            "sha256": manifest["sha256"],
        }
    else:
        terminal_status = "missing"
        manifest_benchmark_eligible = False
        benchmark_eligible = False
        evaluator_consumer_status = "manifest_missing"
    if summary.get("payload") is not None:
        artifacts[SUMMARY_NAME] = {
            "path": SUMMARY_NAME,
            "sha256": summary["sha256"],
        }

    return {
        "artifact_dir": artifact_root.as_posix(),
        "metric_family": METRIC_FAMILY,
        "artifacts": artifacts,
        "row_binding": {
            "row_count": len(rows),
            "row_ids_sha256": row_ids_sha256,
            "provenance_row_ids_sha256": (provenance.get("row_binding") or {}).get(
                "row_ids_sha256"
            ),
        },
        "generation_config_fingerprint": provenance.get(
            "generation_config_fingerprint"
        )
        or provenance.get("decode_policy_fingerprint"),
        "generation_policy": dict(provenance.get("generation_policy") or {}),
        "parallelism": dict(provenance.get("parallelism") or {}),
        "model_identity_fingerprint": provenance.get("model_identity_fingerprint"),
        "processor_identity_fingerprint": provenance.get("processor_identity_fingerprint"),
        "score_policy_fingerprint": provenance.get("score_policy_fingerprint"),
        "run_manifest": {
            "path": RUN_MANIFEST_NAME,
            "sha256": manifest.get("sha256"),
            "terminal_status": terminal_status,
            "manifest_benchmark_eligible": manifest_benchmark_eligible,
            "benchmark_eligible": benchmark_eligible,
            "evaluator_consumer_status": evaluator_consumer_status,
        },
        "benchmark_metric": bool(
            terminal_status == "completed" and benchmark_eligible
        ),
    }


def _artifact_receipt(path: Path) -> dict[str, Any]:
    _require_file(path, code="eval_detection.missing_receipt_artifact")
    return {"path": path.name, "sha256": sha256_file(path)}


def _optional_json_artifact_receipt(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": path.name, "sha256": None, "payload": None}
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "payload": _read_json(path),
    }


def _positive_int(value: Any, *, field: str, row_id: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ArtifactContractError(
            "detection row image dimension must be a positive integer",
            code="eval_detection.invalid_image_dimension",
            context={"row_id": row_id, "field": field, "value": value},
        )
    return value


def _require_object_list(value: Any, *, field: str, row_id: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ArtifactContractError(
            "detection row object field must be a list",
            code="eval_detection.object_list_shape",
            context={"row_id": row_id, "field": field},
        )
    result: list[dict[str, Any]] = []
    for object_index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ArtifactContractError(
                "detection row object must be a mapping",
                code="eval_detection.object_shape",
                context={"row_id": row_id, "field": field, "object_index": object_index},
            )
        result.append(item)
    return result


def _category_text(obj: dict[str, Any]) -> str:
    for key in ("description", "desc", "category_name"):
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return normalize_coco_category_name(value)
    return ""


def _category_id_for_gt(
    obj: dict[str, Any],
    *,
    row: dict[str, Any],
    object_index: int,
) -> int:
    category = _category_text(obj)
    category_id = COCO_80_CATEGORY_IDS.get(category)
    if category_id is None:
        raise ArtifactContractError(
            "GT object category is not in the canonical COCO-80 registry",
            code="eval_detection.unknown_gt_category",
            context={
                "row_id": row["row_id"],
                "object_index": object_index,
                "category": category,
            },
        )
    return category_id


def _category_id_for_pred(obj: dict[str, Any]) -> int | None:
    return COCO_80_CATEGORY_IDS.get(_category_text(obj))


def _gt_coord_bins_to_pixel_xyxy(
    obj: dict[str, Any],
    *,
    row: dict[str, Any],
    object_index: int,
) -> tuple[float, float, float, float]:
    raw_bbox = obj.get("bbox", obj.get("bbox_2d"))
    try:
        x1, y1, x2, y2 = coord_bins_to_pixel_xyxy(
            raw_bbox,
            image_width=int(row["image_width"]),
            image_height=int(row["image_height"]),
            field="gt.bbox",
        )
    except DataContractError as exc:
        raise ArtifactContractError(
            "GT object bbox must be norm1000 xyxy coordinate bins",
            code="eval_detection.invalid_gt_bbox",
            context={
                "row_id": row["row_id"],
                "object_index": object_index,
                "bbox": raw_bbox,
                "image_width": row["image_width"],
                "image_height": row["image_height"],
            },
            cause=exc,
        ) from exc
    return float(x1), float(y1), float(x2), float(y2)


def _xyxy_bbox(
    obj: dict[str, Any],
    *,
    row: dict[str, Any],
    object_index: int,
    object_kind: str,
) -> tuple[float, float, float, float]:
    raw_bbox = obj.get("bbox", obj.get("bbox_2d"))
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        raise ArtifactContractError(
            "detection object bbox must be a four-value xyxy list",
            code=f"eval_detection.invalid_{object_kind}_bbox",
            context={"row_id": row["row_id"], "object_index": object_index},
        )
    try:
        bbox = tuple(float(value) for value in raw_bbox)
    except (TypeError, ValueError) as exc:
        raise ArtifactContractError(
            "detection object bbox values must be numeric",
            code=f"eval_detection.invalid_{object_kind}_bbox",
            context={
                "row_id": row["row_id"],
                "object_index": object_index,
                "bbox": list(raw_bbox),
            },
            cause=exc,
        ) from exc
    if not all(math.isfinite(value) for value in bbox):
        raise ArtifactContractError(
            "detection object bbox values must be finite",
            code=f"eval_detection.invalid_{object_kind}_bbox",
            context={"row_id": row["row_id"], "object_index": object_index, "bbox": list(raw_bbox)},
        )
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        raise ArtifactContractError(
            "detection object bbox must have positive width and height",
            code=f"eval_detection.invalid_{object_kind}_bbox",
            context={"row_id": row["row_id"], "object_index": object_index, "bbox": list(raw_bbox)},
        )
    return x1, y1, x2, y2


def _score(obj: dict[str, Any], *, row: dict[str, Any], pred_index: int) -> float:
    value = obj.get("score")
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ArtifactContractError(
            "scored prediction score must be finite",
            code="eval_detection.invalid_score",
            context={"row_id": row["row_id"], "pred_index": pred_index},
        )
    score = float(value)
    if score < 0.0 or score > 1.0:
        raise ArtifactContractError(
            "scored prediction score must be in [0.0, 1.0]",
            code="eval_detection.invalid_score",
            context={"row_id": row["row_id"], "pred_index": pred_index, "score": score},
        )
    return score


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
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ArtifactContractError(
                "scored artifact row is not valid JSON",
                code="eval_detection.json_decode",
                context={"path": str(path), "row_index": row_index},
                cause=exc,
            ) from exc
        if not isinstance(payload, dict):
            raise ArtifactContractError(
                "scored artifact row must be a JSON object",
                code="eval_detection.row_shape",
                context={"path": str(path), "row_index": row_index},
            )
        rows.append(payload)
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArtifactContractError(
            "provenance artifact is not valid JSON",
            code="eval_detection.json_decode",
            context={"path": str(path)},
            cause=exc,
        ) from exc
    if not isinstance(payload, dict):
        raise ArtifactContractError(
            "provenance artifact must be a JSON object",
            code="eval_detection.provenance_shape",
            context={"path": str(path)},
        )
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
