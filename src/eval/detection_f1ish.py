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

from src.eval.detection_geometry import _greedy_match_by_iou
from src.eval.detection_records import (
    EvalOptions,
    Sample,
    _normalize_desc,
    _resolve_semantic_desc_encoder,
)

def _fmt_iou_thr(iou_thr: float) -> str:
    return f"{float(iou_thr):.2f}"


def _f1ish_filter_gt_objects(sample: Sample) -> List[Dict[str, Any]]:
    return list(sample.objects)


def _compute_prf_from_counts(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    tp_f = float(tp)
    fp_f = float(fp)
    fn_f = float(fn)
    p = tp_f / (tp_f + fp_f) if (tp_f + fp_f) > 0.0 else 1.0
    r = tp_f / (tp_f + fn_f) if (tp_f + fn_f) > 0.0 else 1.0
    f1 = (2.0 * p * r / (p + r)) if (p + r) > 0.0 else 0.0
    return float(p), float(r), float(f1)


def _select_primary_f1ish_iou_thr(iou_thrs: List[float]) -> float:
    thrs = [float(t) for t in (iou_thrs or [])]
    if not thrs:
        return 0.5
    if any(abs(t - 0.5) < 1e-9 for t in thrs):
        return 0.5
    return max(thrs)


def _try_build_semantic_embeddings(
    unique_norm_texts: List[str],
    *,
    options: EvalOptions,
) -> Dict[str, np.ndarray]:
    if not unique_norm_texts:
        return {}
    model_name = str(options.semantic_model or "").strip()
    if not model_name:
        raise ValueError(
            "semantic_model must be a non-empty HuggingFace model id for F1-ish evaluation"
        )
    device = options.semantic_device or "auto"
    bs = max(1, int(options.semantic_batch_size))

    encoder_cls = _resolve_semantic_desc_encoder()
    encoder = encoder_cls(
        model_name=model_name, device=str(device), batch_size=int(bs)
    )

    try:
        embs = encoder.encode_norm_texts(unique_norm_texts)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(
            "F1-ish semantic filtering requires the semantic encoder "
            f"'{model_name}', but loading/encoding failed. Ensure the model is "
            "available in the local HuggingFace cache or that downloads are allowed."
        ) from exc

    out: Dict[str, np.ndarray] = {}
    for text in unique_norm_texts:
        v = embs.get(text)
        if v is None:
            continue
        out[text] = v
    return out


def evaluate_f1ish(
    gt_samples: List[Sample],
    pred_samples: List[Tuple[int, List[Dict[str, Any]]]],
    per_image: List[Dict[str, Any]],
    *,
    options: EvalOptions,
) -> Dict[str, Any]:
    """Compute F1-ish (set matching) metrics and emit match diagnostics.

    Notes on prediction scope:
    - In ``f1ish_pred_scope=annotated`` (default), predictions whose desc is not semantically
      close to any GT desc in the image are ignored (not counted as FP).
    - In ``f1ish_pred_scope=all``, all predictions are counted (strict; extra open-vocab
      objects become FP).
    """
    iou_thrs_in = options.f1ish_iou_thrs or [0.3, 0.5]
    iou_thrs = sorted({float(t) for t in iou_thrs_in})
    primary_thr = _select_primary_f1ish_iou_thr(iou_thrs)

    pred_scope = str(options.f1ish_pred_scope or "annotated").strip().lower()
    if pred_scope not in {"annotated", "all"}:
        logger.warning(
            "Unknown f1ish_pred_scope='%s'; defaulting to 'annotated'", pred_scope
        )
        pred_scope = "annotated"

    pred_lookup = {img_id: preds for img_id, preds in pred_samples}
    per_image_lookup = {row.get("image_id"): row for row in per_image}

    unique_norm_texts: set[str] = set()

    # Collect texts for semantic matching/filtering. This is also used by the
    # per-match semantic scoring below.
    for sample in gt_samples:
        gts = _f1ish_filter_gt_objects(sample)
        for gt in gts:
            gt_desc = _normalize_desc(str(gt.get("desc", "")))
            if gt_desc:
                unique_norm_texts.add(gt_desc)
        for pred in pred_lookup.get(sample.image_id, []):
            pred_desc = _normalize_desc(str(pred.get("desc", "")))
            if pred_desc:
                unique_norm_texts.add(pred_desc)

    # Build semantic embedding cache once (if available)
    norm_texts_sorted = sorted(unique_norm_texts)
    emb = _try_build_semantic_embeddings(norm_texts_sorted, options=options)
    sem_thr = float(options.semantic_threshold)
    use_embeddings = bool(emb)

    def _pred_in_gt_label_space(pred_desc: str, gt_descs: List[str]) -> bool:
        if pred_scope == "all":
            return True
        if not pred_desc or not gt_descs:
            return False
        if pred_desc in set(gt_descs):
            return True
        if not use_embeddings:
            return False
        pred_vec = emb.get(pred_desc)
        if pred_vec is None:
            return False
        best = -1.0
        for gt_desc in gt_descs:
            gt_vec = emb.get(gt_desc)
            if gt_vec is None:
                continue
            best = max(best, float(pred_vec @ gt_vec))
        return best >= sem_thr

    # Pre-filter predictions based on the requested scope. This is independent of IoU
    # threshold and can be reused across all thresholds.
    gts_by_image: Dict[int, List[Dict[str, Any]]] = {}
    preds_eval_by_image: Dict[int, List[Dict[str, Any]]] = {}
    preds_eval_orig_idx_by_image: Dict[int, List[int]] = {}
    preds_ignored_orig_idx_by_image: Dict[int, List[int]] = {}

    for sample in gt_samples:
        gts = _f1ish_filter_gt_objects(sample)
        gts_by_image[sample.image_id] = gts

        gt_descs_set: set[str] = set()
        for gt in gts:
            d = _normalize_desc(str(gt.get("desc", "")))
            if d:
                gt_descs_set.add(d)
        gt_descs = sorted(gt_descs_set)

        preds_total = pred_lookup.get(sample.image_id, [])
        preds_eval: List[Dict[str, Any]] = []
        preds_eval_orig_idx: List[int] = []
        preds_ignored_orig_idx: List[int] = []
        for idx, pred in enumerate(preds_total):
            pred_desc = _normalize_desc(str(pred.get("desc", "")))
            if _pred_in_gt_label_space(pred_desc, gt_descs):
                preds_eval.append(pred)
                preds_eval_orig_idx.append(int(idx))
            else:
                preds_ignored_orig_idx.append(int(idx))
        preds_eval_by_image[sample.image_id] = preds_eval
        preds_eval_orig_idx_by_image[sample.image_id] = preds_eval_orig_idx
        preds_ignored_orig_idx_by_image[sample.image_id] = preds_ignored_orig_idx

    # Location matching per threshold.
    matches_by_thr: Dict[str, Dict[int, List[Tuple[int, int, float]]]] = {}
    matches_records_by_thr: Dict[str, List[Dict[str, Any]]] = {}

    for thr in iou_thrs:
        thr_key = _fmt_iou_thr(thr)
        matches_by_thr[thr_key] = {}

    for sample in gt_samples:
        preds_eval = preds_eval_by_image.get(sample.image_id, [])
        gts = gts_by_image.get(sample.image_id, [])
        width = int(sample.width)
        height = int(sample.height)
        for thr in iou_thrs:
            thr_key = _fmt_iou_thr(thr)
            matches_by_thr[thr_key][sample.image_id] = _greedy_match_by_iou(
                preds_eval, gts, iou_thr=thr, width=width, height=height
            )

    # Accumulators
    metrics_out: Dict[str, float] = {}

    for thr in iou_thrs:
        thr_key = _fmt_iou_thr(thr)

        sum_tp = 0
        sum_fp = 0
        sum_fn = 0
        sum_sem_ok = 0
        sum_sem_bad = 0
        sum_pred_total = 0
        sum_pred_eval = 0
        sum_pred_ignored = 0

        sum_tp_full = 0
        sum_fp_full = 0
        sum_fn_full = 0

        precisions: List[float] = []
        recalls: List[float] = []
        f1s: List[float] = []

        precisions_full: List[float] = []
        recalls_full: List[float] = []
        f1s_full: List[float] = []

        matches_records: List[Dict[str, Any]] = []

        for sample in gt_samples:
            preds_total = pred_lookup.get(sample.image_id, [])
            preds = preds_eval_by_image.get(sample.image_id, [])
            preds_eval_orig = preds_eval_orig_idx_by_image.get(sample.image_id, [])
            preds_ignored_orig = preds_ignored_orig_idx_by_image.get(
                sample.image_id, []
            )
            gts = gts_by_image.get(sample.image_id, [])
            matches = matches_by_thr[thr_key].get(sample.image_id, [])

            matched_pred = {
                pred_idx for pred_idx, _, _ in matches
            }  # pred_idx is in *eval* space
            matched_gt = {gt_idx for _, gt_idx, _ in matches}

            tp_loc = len(matches)
            fp_loc = max(0, len(preds) - len(matched_pred))
            fn_loc = max(0, len(gts) - len(matched_gt))

            sum_pred_total += len(preds_total)
            sum_pred_eval += len(preds)
            sum_pred_ignored += len(preds_ignored_orig)

            sem_ok = 0
            sem_bad = 0
            match_rows: List[Dict[str, Any]] = []
            for pred_idx, gt_idx, iou in matches:
                pred_desc_raw = str(preds[pred_idx].get("desc", ""))
                gt_desc_raw = str(gts[gt_idx].get("desc", ""))
                pred_desc = _normalize_desc(pred_desc_raw)
                gt_desc = _normalize_desc(gt_desc_raw)
                pred_idx_orig = (
                    int(preds_eval_orig[pred_idx])
                    if 0 <= int(pred_idx) < len(preds_eval_orig)
                    else int(pred_idx)
                )

                exact_ok = bool(pred_desc) and (pred_desc == gt_desc)
                sim: float | None = None
                if use_embeddings and pred_desc in emb and gt_desc in emb:
                    sim = float(emb[pred_desc] @ emb[gt_desc])
                ok = bool(exact_ok or (sim is not None and sim >= sem_thr))
                if ok:
                    sem_ok += 1
                else:
                    sem_bad += 1

                match_rows.append(
                    {
                        "pred_idx": int(pred_idx_orig),
                        "gt_idx": int(gt_idx),
                        "iou": float(iou),
                        "pred_desc": pred_desc_raw,
                        "gt_desc": gt_desc_raw,
                        "sem_sim": float(sim) if sim is not None else None,
                        "sem_ok": bool(ok),
                        "pred_bbox": preds[pred_idx].get("bbox"),
                        "gt_bbox": gts[gt_idx].get("bbox"),
                        "pred_type": preds[pred_idx].get("type"),
                        "gt_type": gts[gt_idx].get("type"),
                    }
                )

            sum_tp += tp_loc
            sum_fp += fp_loc
            sum_fn += fn_loc
            sum_sem_ok += sem_ok
            sum_sem_bad += sem_bad

            tp_full = sem_ok
            fp_full = fp_loc + sem_bad
            fn_full = fn_loc + sem_bad

            sum_tp_full += tp_full
            sum_fp_full += fp_full
            sum_fn_full += fn_full

            p_i, r_i, f1_i = _compute_prf_from_counts(tp_loc, fp_loc, fn_loc)
            precisions.append(p_i)
            recalls.append(r_i)
            f1s.append(f1_i)

            p_f, r_f, f1_f = _compute_prf_from_counts(tp_full, fp_full, fn_full)
            precisions_full.append(p_f)
            recalls_full.append(r_f)
            f1s_full.append(f1_f)

            image_row = per_image_lookup.get(sample.image_id)
            if image_row is not None:
                f1ish_field = image_row.setdefault("f1ish", {})
                f1ish_field[thr_key] = {
                    "tp_loc": int(tp_loc),
                    "fp_loc": int(fp_loc),
                    "fn_loc": int(fn_loc),
                    "pred_count_eval": int(len(preds)),
                    "pred_count_ignored": int(len(preds_ignored_orig)),
                    "matched_sem_ok": int(sem_ok),
                    "matched_sem_bad": int(sem_bad),
                    "sem_acc_on_matched": float(sem_ok / tp_loc) if tp_loc > 0 else 1.0,
                    "tp_full": int(tp_full),
                    "fp_full": int(fp_full),
                    "fn_full": int(fn_full),
                }

            is_primary = abs(float(thr) - float(primary_thr)) < 1e-9
            if is_primary or len(iou_thrs) > 1:
                unmatched_pred_indices = [
                    int(preds_eval_orig[i])
                    for i in range(len(preds))
                    if i not in matched_pred and i < len(preds_eval_orig)
                ]
                matches_records.append(
                    {
                        "image_id": int(sample.image_id),
                        "file_name": sample.file_name,
                        "width": int(sample.width),
                        "height": int(sample.height),
                        "iou_thr": float(thr),
                        "gt_count": int(len(gts)),
                        "pred_count": int(len(preds_total)),
                        "pred_count_eval": int(len(preds)),
                        "pred_count_ignored": int(len(preds_ignored_orig)),
                        "pred_scope": str(pred_scope),
                        "tp_loc": int(tp_loc),
                        "fp_loc": int(fp_loc),
                        "fn_loc": int(fn_loc),
                        "matches": match_rows,
                        "unmatched_pred_indices": unmatched_pred_indices,
                        "ignored_pred_indices": [int(i) for i in preds_ignored_orig],
                        "unmatched_gt_indices": [
                            int(i) for i in range(len(gts)) if i not in matched_gt
                        ],
                    }
                )

        p_micro, r_micro, f1_micro = _compute_prf_from_counts(sum_tp, sum_fp, sum_fn)
        p_macro = float(np.mean(precisions)) if precisions else 0.0
        r_macro = float(np.mean(recalls)) if recalls else 0.0
        f1_macro = float(np.mean(f1s)) if f1s else 0.0

        p_full_micro, r_full_micro, f1_full_micro = _compute_prf_from_counts(
            sum_tp_full, sum_fp_full, sum_fn_full
        )
        p_full_macro = float(np.mean(precisions_full)) if precisions_full else 0.0
        r_full_macro = float(np.mean(recalls_full)) if recalls_full else 0.0
        f1_full_macro = float(np.mean(f1s_full)) if f1s_full else 0.0

        prefix = f"f1ish@{thr_key}_"
        metrics_out.update(
            {
                f"{prefix}tp_loc": float(sum_tp),
                f"{prefix}fp_loc": float(sum_fp),
                f"{prefix}fn_loc": float(sum_fn),
                f"{prefix}pred_total": float(sum_pred_total),
                f"{prefix}pred_eval": float(sum_pred_eval),
                f"{prefix}pred_ignored": float(sum_pred_ignored),
                f"{prefix}precision_loc_micro": float(p_micro),
                f"{prefix}recall_loc_micro": float(r_micro),
                f"{prefix}f1_loc_micro": float(f1_micro),
                f"{prefix}precision_loc_macro": float(p_macro),
                f"{prefix}recall_loc_macro": float(r_macro),
                f"{prefix}f1_loc_macro": float(f1_macro),
                f"{prefix}matched_sem_ok": float(sum_sem_ok),
                f"{prefix}matched_sem_bad": float(sum_sem_bad),
                f"{prefix}sem_acc_on_matched": float(sum_sem_ok / sum_tp)
                if sum_tp > 0
                else 1.0,
                f"{prefix}tp_full": float(sum_tp_full),
                f"{prefix}fp_full": float(sum_fp_full),
                f"{prefix}fn_full": float(sum_fn_full),
                f"{prefix}precision_full_micro": float(p_full_micro),
                f"{prefix}recall_full_micro": float(r_full_micro),
                f"{prefix}f1_full_micro": float(f1_full_micro),
                f"{prefix}precision_full_macro": float(p_full_macro),
                f"{prefix}recall_full_macro": float(r_full_macro),
                f"{prefix}f1_full_macro": float(f1_full_macro),
            }
        )

        # Emit matches artifacts
        out_dir = options.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        is_primary = abs(float(thr) - float(primary_thr)) < 1e-9
        matches_path = resolve_matches_artifact_path(
            out_dir=out_dir,
            iou_thr_key=None if is_primary else thr_key,
            name_suffix=options.artifact_name_suffix,
        )
        if is_primary or len(iou_thrs) > 1:
            write_jsonl_records(matches_path, matches_records)
        matches_records_by_thr[thr_key] = matches_records

    return {
        "metrics": metrics_out,
        "matches_by_thr": matches_records_by_thr,
    }
