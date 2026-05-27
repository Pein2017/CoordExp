"""
Offline detection evaluator for CoordExp (pixel-space schema).

Features:
- Ingests standardized ``pred.jsonl`` (pixel-space ``gt`` / ``pred`` objects) or legacy GT JSONL.
- Converts geometries to COCO-format GT and prediction artifacts (bbox + segm for polygons).
- Runs COCOeval (bbox + segm) and/or a set-matching "F1-ish" metric and emits metrics plus robustness counters.
"""

from __future__ import annotations

import copy
import datetime
import json
import math
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, cast

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
from tqdm import tqdm

from src.common.geometry import (
    bbox_from_points,
    bbox_to_quadrilateral,
    coerce_point_list,
    denorm_and_clamp,
    is_degenerate_bbox,
)
from src.common.geometry.object_geometry import extract_single_geometry
from src.common.lvis_semantics import (
    LvisCategory,
    LvisImagePolicy,
    build_lvis_category_catalog,
    extract_lvis_image_policy,
)
from src.common.duplicate_control import (
    DuplicateControlConfig,
    DuplicateControlObject,
    DuplicateControlResult,
    apply_duplicate_policy,
    duplicate_control_object_from_mapping,
    validate_duplicate_control_config,
)
from src.common.prediction_parsing import GEOM_KEYS
from src.common.semantic_desc import SemanticDescEncoder, normalize_desc
from src.common.io import load_jsonl_with_diagnostics
from src.eval.artifacts import (
    build_per_image_report,
    resolve_duplicate_guard_report_path,
    resolve_guarded_prediction_artifact_path,
    resolve_matches_artifact_path,
    write_jsonl_records,
    write_outputs,
)
from src.utils import get_logger
from src.vis import materialize_gt_vs_pred_vis_resource, render_gt_vs_pred_review

logger = get_logger(__name__)

from src.eval.detection_coco import (
    _run_coco_eval,
    _to_coco_gt,
    _to_coco_preds,
    _validate_score_provenance_for_coco,
)
from src.eval.detection_duplicate_guard import (
    _apply_offline_duplicate_control,
    _resolve_duplicate_guard_report_path_for_options,
    _resolve_guarded_pred_path,
)
from src.eval.detection_f1ish import (
    _fmt_iou_thr,
    _select_primary_f1ish_iou_thr,
    evaluate_f1ish,
)
from src.eval.detection_lvis import (
    _matches_by_record_idx,
    _maybe_backfill_lvis_metadata_for_eval,
    _prepare_lvis_artifacts,
    _run_lvis_eval,
    _use_lvis_backend,
)
from src.eval.detection_records import (
    EvalCounters,
    EvalOptions,
    Sample,
    _build_categories,
    _prepare_gt_record,
    _prepare_pred_objects,
    _prepare_pred_objects_detached,
    _wants_official_metrics,
    load_jsonl,
    preds_to_gt_records,
)
from src.infer.artifacts import load_comparable_artifact

def _prepare_all_from_records(
    gt_records: List[Dict[str, Any]],
    pred_records: List[Dict[str, Any]],
    options: EvalOptions,
    counters: EvalCounters,
    *,
    prepare_coco: bool,
) -> Tuple[
    List[Sample],
    List[Tuple[int, List[Dict[str, Any]]]],
    Dict[str, int],
    Dict[str, Any],
    List[Dict[str, Any]],
    bool,
    List[Dict[str, Any]],
]:
    if prepare_coco:
        for record_idx, rec in enumerate(pred_records):
            _validate_score_provenance_for_coco(rec, record_idx)

    gt_samples: List[Sample] = []
    for idx, rec in enumerate(
        tqdm(gt_records, desc="GT", unit="img", disable=len(gt_records) < 10)
    ):
        sample = _prepare_gt_record(rec, idx, counters, strict=options.strict_parse)
        if sample:
            gt_samples.append(sample)
        elif options.strict_parse:
            raise ValueError(f"Failed to prepare GT for index {idx}")

    pred_map: Dict[int, Dict[str, Any]] = {}
    for i, rec in enumerate(pred_records):
        image_id = rec.get("index", i)
        pred_map[int(image_id)] = rec

    pred_samples: List[Tuple[int, List[Dict[str, Any]]]] = []
    invalid_preds: Dict[int, List[Dict[str, Any]]] = {}

    if options.num_workers and options.num_workers > 0:
        num_workers = min(options.num_workers, cpu_count())
        args_list = [
            (
                sample.image_id,
                pred_map.get(sample.image_id, {}),
                sample.width,
                sample.height,
            )
            for sample in gt_samples
        ]
        with Pool(processes=num_workers) as pool:
            for image_id, preds, invalid, local_counts in tqdm(
                pool.imap_unordered(
                    partial(_prepare_pred_objects_detached, options=options), args_list
                ),
                total=len(args_list),
                desc="Pred",
                unit="img",
                disable=len(args_list) < 10,
            ):
                pred_samples.append((image_id, preds))
                if invalid:
                    invalid_preds[image_id] = invalid
                for key, value in local_counts.items():
                    setattr(counters, key, getattr(counters, key) + value)
        pred_samples.sort(key=lambda x: x[0])
    else:
        for sample in tqdm(
            gt_samples, desc="Pred", unit="img", disable=len(gt_samples) < 10
        ):
            rec = pred_map.get(sample.image_id, {})
            preds, invalid = _prepare_pred_objects(
                rec,
                width=sample.width,
                height=sample.height,
                options=options,
                counters=counters,
            )
            pred_samples.append((sample.image_id, preds))
            if invalid:
                invalid_preds[sample.image_id] = invalid

    per_image = build_per_image_report(gt_samples, pred_samples, invalid_preds)

    if not prepare_coco:
        return (
            gt_samples,
            pred_samples,
            {},
            {},
            [],
            False,
            per_image,
        )

    if _use_lvis_backend(gt_samples, options=options):
        categories, coco_gt_dict, results, run_segm = _prepare_lvis_artifacts(
            gt_samples,
            pred_samples,
            options=options,
            counters=counters,
        )
    else:
        categories = _build_categories(gt_samples)
        coco_gt_dict = _to_coco_gt(
            gt_samples, categories, add_box_segmentation=options.use_segm
        )
        results = _to_coco_preds(
            pred_samples, categories, options=options, counters=counters
        )
        run_segm = options.use_segm and any("segmentation" in r for r in results)
    return (
        gt_samples,
        pred_samples,
        categories,
        coco_gt_dict,
        results,
        run_segm,
        per_image,
    )


def _prepare_all(
    pred_records: List[Dict[str, Any]],
    options: EvalOptions,
    counters: EvalCounters,
    *,
    prepare_coco: bool,
) -> Tuple[
    List[Sample],
    List[Tuple[int, List[Dict[str, Any]]]],
    Dict[str, int],
    Dict[str, Any],
    List[Dict[str, Any]],
    bool,
    List[Dict[str, Any]],
]:
    gt_records = preds_to_gt_records(pred_records)
    return _prepare_all_from_records(
        gt_records,
        pred_records,
        options,
        counters,
        prepare_coco=prepare_coco,
    )


def compute_coco_metrics_from_records(
    pred_records: Sequence[Mapping[str, Any]],
    *,
    options: EvalOptions,
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """Compute official detection metrics from in-memory scored ``gt_vs_pred`` records.

    Returns ``(metrics, counters_dict)`` where ``metrics`` uses the same
    ``bbox_*`` / ``segm_*`` keys as offline evaluation. When ``options.metrics``
    requests LVIS and the records carry federated LVIS metadata, this dispatches
    to the LVIS-aware backend; otherwise it uses the COCO backend.
    """

    counters = EvalCounters()
    pred_records_list = [dict(r) for r in pred_records]
    (
        gt_samples,
        _pred_samples,
        _categories,
        coco_gt_dict,
        results,
        run_segm,
        _per_image,
    ) = _prepare_all(
        pred_records_list,
        options,
        counters,
        prepare_coco=True,
    )

    coco_gt = COCO()
    coco_gt.dataset = copy.deepcopy(coco_gt_dict)
    coco_gt.createIndex()

    if _use_lvis_backend(gt_samples, options=options):
        metrics, _per_class = _run_lvis_eval(
            coco_gt,
            results,
            options=options,
            run_segm=run_segm,
        )
    else:
        metrics, _per_class = _run_coco_eval(
            coco_gt,
            results,
            options=options,
            run_segm=run_segm,
        )
    return metrics, counters.to_dict()


def _prepare_all_separate(
    gt_records: List[Dict[str, Any]],
    pred_records: List[Dict[str, Any]],
    options: EvalOptions,
    counters: EvalCounters,
    *,
    prepare_coco: bool,
) -> Tuple[
    List[Sample],
    List[Tuple[int, List[Dict[str, Any]]]],
    Dict[str, int],
    Dict[str, Any],
    List[Dict[str, Any]],
    bool,
    List[Dict[str, Any]],
]:
    return _prepare_all_from_records(
        gt_records,
        pred_records,
        options,
        counters,
        prepare_coco=prepare_coco,
    )


def _evaluate_preloaded_records(
    pred_records: List[Dict[str, Any]],
    *,
    pred_path: Path,
    options: EvalOptions,
    materialize_vis_resource: bool,
    render_overlay: bool,
) -> Dict[str, Any]:
    from src.infer.pipeline import resolve_root_image_dir_for_jsonl

    counters = EvalCounters()
    metrics_mode = str(options.metrics).strip().lower()
    want_official = _wants_official_metrics(metrics_mode)
    want_f1ish = metrics_mode in {"f1ish", "both"}

    (
        gt_samples,
        pred_samples,
        categories,
        coco_gt_dict,
        results,
        run_segm,
        per_image,
    ) = _prepare_all(
        pred_records,
        options,
        counters,
        prepare_coco=want_official,
    )

    metrics: Dict[str, float] = {}
    per_class: Dict[str, float] = {}
    if want_official:
        coco_gt = COCO()
        coco_gt.dataset = copy.deepcopy(coco_gt_dict)
        coco_gt.createIndex()
        if _use_lvis_backend(gt_samples, options=options):
            metrics, per_class = _run_lvis_eval(
                coco_gt,
                results,
                options=options,
                run_segm=run_segm,
            )
        else:
            metrics, per_class = _run_coco_eval(
                coco_gt,
                results,
                options=options,
                run_segm=run_segm,
            )

    if _use_lvis_backend(gt_samples, options=options):
        metrics.update(
            {
                "lvis_diag_matched_verified_positive": float(
                    counters.lvis_matched_verified_positive
                ),
                "lvis_diag_verified_negative_unmatched": float(
                    counters.lvis_verified_negative_unmatched
                ),
                "lvis_diag_ignored_not_exhaustive": float(
                    counters.lvis_ignored_not_exhaustive
                ),
                "lvis_diag_ignored_unevaluable": float(
                    counters.lvis_ignored_unevaluable
                ),
            }
        )

    summary = {
        "metrics": metrics,
        "per_class": per_class,
        "counters": counters.to_dict(),
        "categories": categories,
    }

    if want_f1ish:
        f1ish_summary = evaluate_f1ish(
            gt_samples,
            pred_samples,
            per_image,
            options=options,
        )
        summary["metrics"].update(f1ish_summary["metrics"])
    else:
        f1ish_summary = {"matches_by_thr": {}}

    vis_resource_path: Optional[Path] = None
    if materialize_vis_resource:
        vis_matches: Dict[int, Dict[str, Any]] | None = None
        if want_f1ish:
            primary_thr = _select_primary_f1ish_iou_thr(options.f1ish_iou_thrs)
            primary_key = _fmt_iou_thr(primary_thr)
            vis_matches = _matches_by_record_idx(
                f1ish_summary.get("matches_by_thr", {}).get(primary_key, [])
            )

        vis_resource_path = materialize_gt_vs_pred_vis_resource(
            pred_path,
            source_kind="detection_eval",
            external_matches=vis_matches,
            materialize_matching=True,
        )

    write_outputs(
        options.output_dir,
        coco_gt=coco_gt_dict if want_official else None,
        coco_preds=results if want_official else None,
        summary=summary,
        per_image=per_image,
        name_suffix=options.artifact_name_suffix,
    )

    if render_overlay and options.overlay and vis_resource_path is not None:
        root_dir, root_source = resolve_root_image_dir_for_jsonl(pred_path)
        if root_dir is not None:
            logger.info(
                "Overlay image root resolved (source=%s): %s",
                root_source,
                root_dir,
            )

        overlay_dir = options.output_dir / "overlays"
        render_gt_vs_pred_review(
            vis_resource_path,
            out_dir=overlay_dir,
            limit=options.overlay_k,
            root_image_dir=root_dir,
            root_source=root_source,
            record_order="error_first",
        )

    return summary


def evaluate_detection(
    gt_path: Path,
    pred_path: Path | None = None,
    *,
    options: EvalOptions,
) -> Dict[str, Any]:
    counters = EvalCounters()
    metrics_mode = str(options.metrics).strip().lower()
    want_official = metrics_mode in {"coco", "lvis", "both"}

    if pred_path is None:
        pred_records = load_jsonl(gt_path, counters, strict=options.strict_parse)
        (
            gt_samples,
            _pred_samples,
            categories,
            coco_gt_dict,
            results,
            run_segm,
            _per_image,
        ) = _prepare_all(
            pred_records,
            options,
            counters,
            prepare_coco=want_official,
        )
    else:
        gt_records = load_jsonl(gt_path, counters, strict=options.strict_parse)
        pred_records = load_jsonl(pred_path, counters, strict=options.strict_parse)
        (
            gt_samples,
            _pred_samples,
            categories,
            coco_gt_dict,
            results,
            run_segm,
            _per_image,
        ) = _prepare_all_separate(
            gt_records,
            pred_records,
            options,
            counters,
            prepare_coco=want_official,
        )

    metrics: Dict[str, float] = {}
    per_class: Dict[str, float] = {}
    if want_official:
        coco_gt = COCO()
        coco_gt.dataset = copy.deepcopy(coco_gt_dict)
        coco_gt.createIndex()
        if _use_lvis_backend(gt_samples, options=options):
            metrics, per_class = _run_lvis_eval(
                coco_gt,
                results,
                options=options,
                run_segm=run_segm,
            )
        else:
            metrics, per_class = _run_coco_eval(
                coco_gt,
                results,
                options=options,
                run_segm=run_segm,
            )

    return {
        "metrics": metrics,
        "per_class": per_class,
        "counters": counters.to_dict(),
        "categories": categories,
    }


def evaluate_and_save(
    pred_path: Path,
    *,
    options: EvalOptions,
) -> Dict[str, Any]:
    if _wants_official_metrics(options.metrics):
        load_comparable_artifact(pred_path, require_score=True)
    load_counters = EvalCounters()
    pred_records = load_jsonl(pred_path, load_counters, strict=options.strict_parse)
    pred_records = _maybe_backfill_lvis_metadata_for_eval(
        pred_records,
        pred_path=pred_path,
        options=options,
    )

    if load_counters.invalid_json:
        logger.info(
            "Evaluation ingestion skipped %d invalid JSONL records from %s",
            load_counters.invalid_json,
            pred_path,
        )

    summary = _evaluate_preloaded_records(
        pred_records,
        pred_path=pred_path,
        options=options,
        materialize_vis_resource=True,
        render_overlay=True,
    )
    if load_counters.invalid_json:
        summary["counters"]["invalid_json"] = (
            int(summary["counters"].get("invalid_json", 0)) + load_counters.invalid_json
        )
        metrics_path = options.output_dir / f"metrics{options.artifact_name_suffix}.json"
        if metrics_path.is_file():
            metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            counters_payload = metrics_payload.get("counters", {})
            if isinstance(counters_payload, dict):
                counters_payload["invalid_json"] = summary["counters"]["invalid_json"]
                metrics_payload["counters"] = counters_payload
                metrics_path.write_text(
                    json.dumps(metrics_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

    if not options.duplicate_control_enabled:
        return summary

    guarded_pred_path = _resolve_guarded_pred_path(pred_path=pred_path, options=options)
    duplicate_guard_report_path = _resolve_duplicate_guard_report_path_for_options(options)
    guarded_records, duplicate_guard_report = _apply_offline_duplicate_control(pred_records)
    write_jsonl_records(guarded_pred_path, guarded_records)
    duplicate_guard_report_path.parent.mkdir(parents=True, exist_ok=True)
    duplicate_guard_report_path.write_text(
        json.dumps(duplicate_guard_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info(
        "Duplicate-control guard emitted raw+guarded outputs: guarded_pred=%s report=%s",
        guarded_pred_path,
        duplicate_guard_report_path,
    )

    guarded_options = copy.copy(options)
    guarded_options.artifact_name_suffix = "_guarded"
    guarded_options.overlay = False
    guarded_options.duplicate_control_enabled = False
    guarded_options.guarded_pred_path = None
    guarded_options.duplicate_guard_report_path = None

    guarded_summary = _evaluate_preloaded_records(
        guarded_records,
        pred_path=guarded_pred_path,
        options=guarded_options,
        materialize_vis_resource=False,
        render_overlay=False,
    )
    summary["guarded"] = guarded_summary
    summary["duplicate_control"] = duplicate_guard_report
    return summary
