from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict


def evaluate_detection_summary(
    *,
    gt_path: Path,
    pred_path: Path | None,
    options: Any,
    counters_cls: Any,
    load_jsonl_fn: Any,
    prepare_all_fn: Any,
    prepare_all_separate_fn: Any,
    run_coco_eval_fn: Any,
    coco_cls: Any,
) -> Dict[str, Any]:
    counters = counters_cls()
    want_coco = str(options.metrics).lower() in {"coco", "both"}
    if pred_path is None:
        pred_records = load_jsonl_fn(gt_path, counters, strict=options.strict_parse)
        (
            gt_samples,
            pred_samples,
            categories,
            coco_gt_dict,
            results,
            run_segm,
            _,
        ) = prepare_all_fn(pred_records, options, counters, prepare_coco=want_coco)
    else:
        gt_records = load_jsonl_fn(gt_path, counters, strict=options.strict_parse)
        pred_records = load_jsonl_fn(pred_path, counters, strict=options.strict_parse)
        (
            gt_samples,
            pred_samples,
            categories,
            coco_gt_dict,
            results,
            run_segm,
            _,
        ) = prepare_all_separate_fn(
            gt_records,
            pred_records,
            options,
            counters,
            prepare_coco=want_coco,
        )

    metrics: Dict[str, float] = {}
    per_class: Dict[str, float] = {}
    if want_coco:
        coco_gt = coco_cls()
        coco_gt.dataset = copy.deepcopy(coco_gt_dict)
        coco_gt.createIndex()
        metrics, per_class = run_coco_eval_fn(
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


def evaluate_and_save_outputs(
    *,
    pred_path: Path,
    options: Any,
    counters_cls: Any,
    load_jsonl_fn: Any,
    prepare_all_fn: Any,
    run_coco_eval_fn: Any,
    evaluate_f1ish_fn: Any,
    select_primary_f1ish_iou_thr_fn: Any,
    fmt_iou_thr_fn: Any,
    matches_by_record_idx_fn: Any,
    materialize_gt_vs_pred_vis_resource_fn: Any,
    write_outputs_fn: Any,
    resolve_root_image_dir_for_jsonl_fn: Any,
    render_gt_vs_pred_review_fn: Any,
    logger: Any,
    coco_cls: Any,
) -> Dict[str, Any]:
    counters = counters_cls()
    metrics_mode = str(options.metrics).lower()
    want_coco = metrics_mode in {"coco", "both"}
    want_f1ish = metrics_mode in {"f1ish", "both"}
    pred_records = load_jsonl_fn(pred_path, counters, strict=options.strict_parse)
    (
        gt_samples,
        pred_samples,
        categories,
        coco_gt_dict,
        results,
        run_segm,
        per_image,
    ) = prepare_all_fn(pred_records, options, counters, prepare_coco=want_coco)

    metrics: Dict[str, float] = {}
    per_class: Dict[str, float] = {}
    if want_coco:
        coco_gt = coco_cls()
        coco_gt.dataset = copy.deepcopy(coco_gt_dict)
        coco_gt.createIndex()
        metrics, per_class = run_coco_eval_fn(
            coco_gt,
            results,
            options=options,
            run_segm=run_segm,
        )

    summary = {
        "metrics": metrics,
        "per_class": per_class,
        "counters": counters.to_dict(),
        "categories": categories,
    }

    if want_f1ish:
        f1ish_summary = evaluate_f1ish_fn(
            gt_samples,
            pred_samples,
            per_image,
            options=options,
        )
        summary["metrics"].update(f1ish_summary["metrics"])
    else:
        f1ish_summary = {"matches_by_thr": {}}

    vis_matches: Dict[int, Dict[str, Any]] | None = None
    if want_f1ish:
        primary_thr = select_primary_f1ish_iou_thr_fn(options.f1ish_iou_thrs)
        primary_key = fmt_iou_thr_fn(primary_thr)
        vis_matches = matches_by_record_idx_fn(
            f1ish_summary.get("matches_by_thr", {}).get(primary_key, [])
        )

    vis_resource_path = materialize_gt_vs_pred_vis_resource_fn(
        pred_path,
        source_kind="detection_eval",
        external_matches=vis_matches,
        materialize_matching=True,
    )

    write_outputs_fn(
        options.output_dir,
        coco_gt=coco_gt_dict if want_coco else None,
        coco_preds=results if want_coco else None,
        summary=summary,
        per_image=per_image,
        name_suffix=getattr(options, "artifact_name_suffix", ""),
    )

    if options.overlay:
        root_dir, root_source = resolve_root_image_dir_for_jsonl_fn(pred_path)
        if root_dir is not None:
            logger.info(
                "Overlay image root resolved (source=%s): %s",
                root_source,
                root_dir,
            )

        overlay_dir = options.output_dir / "overlays"
        render_gt_vs_pred_review_fn(
            vis_resource_path,
            out_dir=overlay_dir,
            limit=options.overlay_k,
            root_image_dir=root_dir,
            root_source=root_source,
            record_order="error_first",
        )

    return summary
