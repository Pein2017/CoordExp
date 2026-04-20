from __future__ import annotations

import argparse
import gc
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

from src.analysis.duplication_collapse_analysis import (
    Qwen3VLSurgeryProber,
    _append_ready_prefix,
    _bbox_center,
    _bbox_diag,
    _bbox_iou,
    _bbox_xyxy,
    _choose_gt_next_candidate,
    _coord_split_candidate,
    _interpolate_object_bbox,
    _load_image_for_case,
    _load_yaml,
    _pair_duplicate_metrics,
    _read_jsonl,
    _write_json,
    _write_jsonl,
)
from src.common.semantic_desc import normalize_desc
from src.utils.coordjson_transpiler import parse_coordjson


def _norm1000_to_pixel_bbox(
    bins: Sequence[int],
    *,
    width: int,
    height: int,
) -> List[int]:
    if len(bins) != 4:
        raise ValueError("expected four normalized bbox bins")
    return [
        int(round(int(bins[0]) * float(width) / 1000.0)),
        int(round(int(bins[1]) * float(height) / 1000.0)),
        int(round(int(bins[2]) * float(width) / 1000.0)),
        int(round(int(bins[3]) * float(height) / 1000.0)),
    ]


def _distribution(values: Iterable[Optional[float]]) -> Dict[str, Optional[float]]:
    xs = [float(v) for v in values if v is not None]
    if not xs:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }
    xs_sorted = sorted(xs)
    return {
        "n": len(xs_sorted),
        "mean": float(mean(xs_sorted)),
        "median": float(median(xs_sorted)),
        "min": float(xs_sorted[0]),
        "max": float(xs_sorted[-1]),
    }


def _case_kind(row: Mapping[str, Any]) -> str:
    case_id = str(row.get("case_id") or "")
    selection_reason = str(row.get("selection_reason") or "")
    if case_id.endswith("-dup") or "duplicate case" in selection_reason:
        return "dup"
    if case_id.endswith("-normal") or "normal same-desc control" in selection_reason:
        return "normal"
    return "unknown"


def _value_or(default: Any, *values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return default


def _same_desc_crowding_metrics(gt_objects: Sequence[Mapping[str, Any]], desc: str) -> Dict[str, Any]:
    desc_norm = normalize_desc(str(desc))
    boxes = [
        _bbox_xyxy(obj)
        for obj in gt_objects
        if normalize_desc(str(obj.get("desc") or "")) == desc_norm
    ]
    boxes = [box for box in boxes if box is not None]
    n = len(boxes)
    if n == 0:
        return {
            "same_desc_gt_count": 0,
            "max_same_desc_iou": None,
            "mean_pair_iou": None,
            "min_norm_center_distance": None,
            "median_min_norm_center_distance": None,
        }
    if n == 1:
        return {
            "same_desc_gt_count": 1,
            "max_same_desc_iou": 0.0,
            "mean_pair_iou": 0.0,
            "min_norm_center_distance": None,
            "median_min_norm_center_distance": None,
        }
    pair_ious: List[float] = []
    nearest_norm_distances: List[float] = []
    for idx, box in enumerate(boxes):
        center_i = _bbox_center(box)
        diag_i = max(_bbox_diag(box), 1e-6)
        nearest = None
        for jdx, other in enumerate(boxes):
            if idx == jdx:
                continue
            pair_ious.append(_bbox_iou(box, other))
            center_j = _bbox_center(other)
            dist = math.hypot(center_i[0] - center_j[0], center_i[1] - center_j[1]) / diag_i
            nearest = dist if nearest is None else min(nearest, dist)
        if nearest is not None:
            nearest_norm_distances.append(float(nearest))
    return {
        "same_desc_gt_count": int(n),
        "max_same_desc_iou": float(max(pair_ious)) if pair_ious else 0.0,
        "mean_pair_iou": float(mean(pair_ious)) if pair_ious else 0.0,
        "min_norm_center_distance": (
            float(min(nearest_norm_distances)) if nearest_norm_distances else None
        ),
        "median_min_norm_center_distance": (
            float(median(nearest_norm_distances)) if nearest_norm_distances else None
        ),
    }


def _load_case_manifest(path: Path) -> List[Dict[str, Any]]:
    return [dict(row) for row in _read_jsonl(path)]


def _compare_lookup(paths: Sequence[Path]) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        for row in _read_jsonl(path):
            rows[str(row.get("case_id") or "")] = dict(row)
    return rows


def run_crowding_and_prior_audit(config_path: Path) -> Dict[str, Any]:
    cfg = _load_yaml(config_path)
    run_cfg = cfg.get("run") or {}
    crowd_cfg = cfg.get("crowding_audit") or {}
    run_name = str(run_cfg.get("name") or "duplication-followup-crowding")
    output_root = Path(str(run_cfg.get("output_root") or "research/duplication_followup"))
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    case_manifest = _load_case_manifest(Path(str(crowd_cfg["cases_manifest_jsonl"])))
    compare_paths = [Path(str(p)) for p in (crowd_cfg.get("compare_case_rows_jsonl") or [])]
    compare_rows = _compare_lookup(compare_paths)

    case_rows: List[Dict[str, Any]] = []
    for case in case_manifest:
        case_id = str(case.get("case_id") or "")
        desc = str(((case.get("onset") or {}).get("desc")) or case.get("top_desc") or "")
        gt_rows = _read_jsonl(Path(str(case["source_gt_vs_pred_jsonl"])))
        line_idx = int(case["line_idx"])
        if not (0 <= line_idx < len(gt_rows)):
            continue
        source_row = gt_rows[line_idx]
        gt_objects = list(source_row.get("gt") or [])
        crowd = _same_desc_crowding_metrics(gt_objects, desc)
        compare = compare_rows.get(case_id) or {}
        margins = compare.get("margins") or {}
        predicted = compare.get("predicted_object") or {}
        duplicate = compare.get("exact_duplicate") or {}
        row = {
            "case_id": case_id,
            "image_id": int(_value_or(-1, case.get("image_id"), source_row.get("image_id"))),
            "checkpoint_alias": str(case.get("checkpoint_alias") or ""),
            "case_kind": _case_kind(case),
            "desc": desc,
            "gt_count": int(case.get("gt_count") or len(gt_objects)),
            **crowd,
            "predicted_minus_duplicate": margins.get("predicted_minus_duplicate"),
            "gt_next_minus_duplicate": margins.get("gt_next_minus_duplicate"),
            "oracle_x1y1_minus_duplicate": margins.get("oracle_x1y1_minus_duplicate"),
            "oracle_x2y2_minus_duplicate": margins.get("oracle_x2y2_minus_duplicate"),
            "oracle_interp_minus_duplicate": margins.get("oracle_interp_minus_duplicate"),
            "predicted_prev_neighborhood_mass": predicted.get("coord_previous_box_neighborhood_mass_mean"),
            "duplicate_prev_neighborhood_mass": duplicate.get("coord_previous_box_neighborhood_mass_mean"),
            "predicted_top1_mean": predicted.get("coord_top1_prob_mean"),
            "duplicate_top1_mean": duplicate.get("coord_top1_prob_mean"),
        }
        case_rows.append(row)

    by_kind: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_desc: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in case_rows:
        by_kind[str(row["case_kind"])].append(row)
        by_desc[str(row["desc"])].append(row)

    summary = {
        "num_cases": len(case_rows),
        "by_case_kind": {
            kind: {
                "num_cases": len(rows),
                "same_desc_gt_count": _distribution(row.get("same_desc_gt_count") for row in rows),
                "max_same_desc_iou": _distribution(row.get("max_same_desc_iou") for row in rows),
                "min_norm_center_distance": _distribution(
                    row.get("min_norm_center_distance") for row in rows
                ),
                "predicted_minus_duplicate": _distribution(
                    row.get("predicted_minus_duplicate") for row in rows
                ),
                "predicted_prev_neighborhood_mass": _distribution(
                    row.get("predicted_prev_neighborhood_mass") for row in rows
                ),
                "duplicate_prev_neighborhood_mass": _distribution(
                    row.get("duplicate_prev_neighborhood_mass") for row in rows
                ),
            }
            for kind, rows in sorted(by_kind.items())
        },
        "top_desc_counts": Counter(row["desc"] for row in case_rows).most_common(),
        "per_desc": {
            desc: {
                "num_cases": len(rows),
                "case_kind_counts": dict(Counter(row["case_kind"] for row in rows)),
                "same_desc_gt_count": _distribution(row.get("same_desc_gt_count") for row in rows),
                "max_same_desc_iou": _distribution(row.get("max_same_desc_iou") for row in rows),
            }
            for desc, rows in sorted(by_desc.items())
        },
    }

    _write_jsonl(run_dir / "case_rows.jsonl", case_rows)
    _write_json(run_dir / "summary.json", summary)
    return {"run_dir": str(run_dir), "summary": summary}


def _tokenize_prefix(prober: Qwen3VLSurgeryProber, prefix_text: str) -> List[int]:
    return list(
        prober.tokenizer.encode(
            prefix_text,
            add_special_tokens=False,
        )
    )


def _generate_from_prefix(
    prober: Qwen3VLSurgeryProber,
    *,
    image_path: str,
    prefix_objects: Sequence[Mapping[str, Any]],
    width: int,
    height: int,
    object_field_order: str,
    max_new_tokens: int,
) -> Dict[str, Any]:
    with _load_image_for_case(image_path) as image:
        prompt_inputs = prober._prompt_inputs(image)
        prefix_text = _append_ready_prefix(
            prefix_objects,
            width=width,
            height=height,
            object_field_order=object_field_order,
        )
        prefix_ids = _tokenize_prefix(prober, prefix_text)
        input_ids = prompt_inputs["input_ids"]
        prefix_tensor = torch.tensor(
            [prefix_ids],
            device=input_ids.device,
            dtype=input_ids.dtype,
        )
        full_input_ids = torch.cat([input_ids, prefix_tensor], dim=1)
        attention_mask = prompt_inputs.get("attention_mask")
        model_inputs = dict(prompt_inputs)
        model_inputs["input_ids"] = full_input_ids
        if isinstance(attention_mask, torch.Tensor):
            prefix_mask = torch.ones(
                (attention_mask.shape[0], len(prefix_ids)),
                device=attention_mask.device,
                dtype=attention_mask.dtype,
            )
            model_inputs["attention_mask"] = torch.cat([attention_mask, prefix_mask], dim=1)
        generated = prober.model.generate(
            **model_inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            temperature=0.0,
            top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=prober.tokenizer.pad_token_id,
            eos_token_id=prober.tokenizer.eos_token_id,
            use_cache=True,
        )
        new_ids = generated[0, full_input_ids.shape[1] :].detach().cpu().tolist()
        new_text = prober.tokenizer.decode(
            new_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        assistant_text = prefix_text + new_text
        parsed = parse_coordjson(
            assistant_text,
            mode="salvage",
            object_field_order=object_field_order,
        )
        records = list(parsed.records)
        next_record = records[len(prefix_objects)] if len(records) > len(prefix_objects) else None
        next_object = None
        if next_record is not None and next_record.geometry_key == "bbox_2d":
            bbox_px = _norm1000_to_pixel_bbox(
                next_record.geometry_values,
                width=width,
                height=height,
            )
            next_object = {
                "type": "bbox_2d",
                "points": bbox_px,
                "desc": str(next_record.desc or ""),
            }
        return {
            "prefix_text": prefix_text,
            "generated_text": new_text,
            "generated_token_ids": new_ids,
            "parse_truncated": bool(parsed.truncated),
            "parse_failed": bool(parsed.parse_failed),
            "num_records": len(records),
            "next_object": next_object,
        }


def _perturbation_variants(
    *,
    prefix_objects: Sequence[Mapping[str, Any]],
    source_index_in_prefix: int,
    gt_next: Optional[Mapping[str, Any]],
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    variants: List[Tuple[str, List[Dict[str, Any]]]] = []
    prefix_copy = [dict(obj) for obj in prefix_objects]
    if 0 <= int(source_index_in_prefix) < len(prefix_copy):
        dropped = [
            dict(obj)
            for idx, obj in enumerate(prefix_copy)
            if idx != int(source_index_in_prefix)
        ]
        variants.append(("drop_source", dropped))
    if gt_next is not None and 0 <= int(source_index_in_prefix) < len(prefix_copy):
        replaced = [dict(obj) for obj in prefix_copy]
        replaced[int(source_index_in_prefix)] = dict(gt_next)
        variants.append(("replace_source_with_gt_next", replaced))
        interp = _interpolate_object_bbox(
            source_object=prefix_copy[int(source_index_in_prefix)],
            target_object=gt_next,
            alpha=0.5,
        )
        if interp is not None:
            interp_prefix = [dict(obj) for obj in prefix_copy]
            interp_prefix[int(source_index_in_prefix)] = interp
            variants.append(("interp_source_to_gt_next_0p5", interp_prefix))
        x1y1 = _coord_split_candidate(
            base_object=prefix_copy[int(source_index_in_prefix)],
            alt_object=gt_next,
            alt_slots=(0, 1),
        )
        if x1y1 is not None:
            x1y1_prefix = [dict(obj) for obj in prefix_copy]
            x1y1_prefix[int(source_index_in_prefix)] = x1y1
            variants.append(("source_x1y1_from_gt_next", x1y1_prefix))
    return variants


def build_prefix_perturbation_variants(
    *,
    prefix_objects: Sequence[Mapping[str, Any]],
    source_index_in_prefix: int,
    gt_next: Optional[Mapping[str, Any]],
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    return _perturbation_variants(
        prefix_objects=prefix_objects,
        source_index_in_prefix=source_index_in_prefix,
        gt_next=gt_next,
    )


def _escape_label(
    *,
    source_object: Mapping[str, Any],
    next_object: Optional[Mapping[str, Any]],
) -> str:
    if next_object is None:
        return "undercount_or_parse_fail"
    source_desc = normalize_desc(str(source_object.get("desc") or ""))
    next_desc = normalize_desc(str(next_object.get("desc") or ""))
    if not next_desc:
        return "undercount_or_parse_fail"
    if next_desc != source_desc:
        return "semantic_escape"
    cfg_stub = type(
        "CfgStub",
        (),
        {
            "subset": type(
                "SubsetStub",
                (),
                {
                    "duplicate_iou_threshold": 0.65,
                    "local_center_radius_scale": 0.9,
                    "size_ratio_min": 0.7,
                },
            )()
        },
    )()
    pair = _pair_duplicate_metrics(source_object, next_object, cfg=cfg_stub)
    if pair is not None and bool(pair.get("duplicate_like")):
        return "duplicate_basin"
    return "same_desc_escape"


def run_prefix_perturbation(config_path: Path) -> Dict[str, Any]:
    cfg = _load_yaml(config_path)
    run_cfg = cfg.get("run") or {}
    perturb_cfg = cfg.get("prefix_perturbation") or {}
    exec_cfg = cfg.get("execution") or {}
    run_name = str(run_cfg.get("name") or "duplication-followup-prefix-perturb")
    output_root = Path(str(run_cfg.get("output_root") or "research/duplication_followup"))
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    cuda_visible = exec_cfg.get("cuda_visible_devices")
    if cuda_visible is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible)
    device = str(exec_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    attn_implementation = str(exec_cfg.get("attn_implementation") or "sdpa")
    max_new_tokens = int(perturb_cfg.get("max_new_tokens") or 64)

    case_manifest = _load_case_manifest(Path(str(perturb_cfg["cases_manifest_jsonl"])))
    case_filters = set(int(v) for v in (perturb_cfg.get("line_idx_filter") or []))
    if case_filters:
        case_manifest = [
            row
            for row in case_manifest
            if int(_value_or(-1, row.get("line_idx"))) in case_filters
        ]

    out_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    for target in perturb_cfg.get("targets") or []:
        alias = str(target["alias"])
        reproduction_rows = _read_jsonl(Path(str(target["reproduce_gt_vs_pred_jsonl"])))
        by_image_id = {
            int(_value_or(-1, row.get("image_id"))): row
            for row in reproduction_rows
            if row.get("image_id") is not None
        }
        checkpoint_stub = type(
            "CheckpointStub",
            (),
            {
                "path": Path(str(target["checkpoint_path"])),
                "prompt_variant": str(target["prompt_variant"]),
                "object_field_order": str(target["object_field_order"]),
            },
        )()
        prober = Qwen3VLSurgeryProber(
            checkpoint=checkpoint_stub,
            device=device,
            attn_implementation=attn_implementation,
        )
        try:
            for case in case_manifest:
                if _case_kind(case) != "dup":
                    continue
                image_id = int(case["image_id"])
                reproduce_row = by_image_id.get(image_id)
                if reproduce_row is None:
                    continue
                onset = dict(case.get("onset") or {})
                object_idx = onset.get("object_idx")
                source_idx = onset.get("source_object_idx")
                preds = list(reproduce_row.get("pred") or [])
                gts = list(reproduce_row.get("gt") or [])
                row_base = {
                    "target_alias": alias,
                    "case_id": str(case.get("case_id") or ""),
                    "image_id": image_id,
                    "line_idx": int(_value_or(-1, case.get("line_idx"))),
                }
                if object_idx is None or source_idx is None:
                    out_rows.append({**row_base, "error": "missing_onset"})
                    continue
                object_idx = int(object_idx)
                source_idx = int(source_idx)
                if not (0 <= source_idx < len(preds)) or not (0 <= object_idx <= len(preds)):
                    out_rows.append({**row_base, "error": "insufficient_reproduced_prefix"})
                    continue
                prefix_objects = [dict(obj) for obj in preds[:object_idx]]
                if not (0 <= source_idx < len(prefix_objects)):
                    out_rows.append({**row_base, "error": "source_not_in_prefix"})
                    continue
                source_object = dict(preds[source_idx])
                gt_next = _choose_gt_next_candidate(
                    prefix_objects=prefix_objects,
                    gt_objects=gts,
                    source_object=source_object,
                    cfg=type(
                        "CfgStub",
                        (),
                        {"controls": type("ControlsStub", (), {"same_desc_iou_threshold": 0.5})()},
                    )(),
                )
                perturbations = _perturbation_variants(
                    prefix_objects=prefix_objects,
                    source_index_in_prefix=source_idx,
                    gt_next=gt_next,
                )
                baseline_next = preds[object_idx] if object_idx < len(preds) else None
                out_rows.append(
                    {
                        **row_base,
                        "perturbation": "baseline_reproduced",
                        "next_object": baseline_next,
                        "escape_label": _escape_label(
                            source_object=source_object,
                            next_object=baseline_next,
                        ),
                    }
                )
                for perturbation_id, perturbed_prefix in perturbations:
                    generated = _generate_from_prefix(
                        prober,
                        image_path=str(reproduce_row.get("image") or case.get("image") or ""),
                        prefix_objects=perturbed_prefix,
                        width=int(_value_or(0, reproduce_row.get("width"), case.get("width"))),
                        height=int(_value_or(0, reproduce_row.get("height"), case.get("height"))),
                        object_field_order=str(target["object_field_order"]),
                        max_new_tokens=max_new_tokens,
                    )
                    next_object = generated.get("next_object")
                    out_rows.append(
                        {
                            **row_base,
                            "perturbation": perturbation_id,
                            "prefix_len": len(perturbed_prefix),
                            "source_desc": str(source_object.get("desc") or ""),
                            "gt_next_desc": (str(gt_next.get("desc") or "") if gt_next is not None else None),
                            "next_object": next_object,
                            "escape_label": _escape_label(
                                source_object=source_object,
                                next_object=next_object,
                            ),
                            "generated_text": generated.get("generated_text"),
                            "parse_failed": generated.get("parse_failed"),
                            "parse_truncated": generated.get("parse_truncated"),
                            "num_records": generated.get("num_records"),
                        }
                    )
        finally:
            prober.close()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        target_rows = [row for row in out_rows if row.get("target_alias") == alias]
        summary_rows.append(
            {
                "target_alias": alias,
                "num_rows": len(target_rows),
                "escape_label_counts": dict(Counter(row.get("escape_label") for row in target_rows)),
            }
        )

    summary = {
        "num_rows": len(out_rows),
        "by_target": summary_rows,
        "by_perturbation": {
            perturbation: dict(Counter(row.get("escape_label") for row in rows))
            for perturbation, rows in defaultdict(list, (
                (str(row.get("perturbation") or "unknown"), [])
                for row in []
            )).items()
        },
    }
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in out_rows:
        grouped[str(row.get("perturbation") or "unknown")].append(row)
    summary["by_perturbation"] = {
        key: {
            "num_rows": len(rows),
            "escape_label_counts": dict(Counter(row.get("escape_label") for row in rows)),
        }
        for key, rows in sorted(grouped.items())
    }

    _write_jsonl(run_dir / "case_rows.jsonl", out_rows)
    _write_json(run_dir / "summary.json", summary)
    return {"run_dir": str(run_dir), "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run duplication follow-up analyses.")
    parser.add_argument("--config", required=True, help="Path to YAML follow-up config.")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    cfg = _load_yaml(config_path)
    mode = str((cfg.get("run") or {}).get("mode") or "").strip().lower()
    if mode == "crowding_audit":
        result = run_crowding_and_prior_audit(config_path)
    elif mode == "prefix_perturbation":
        result = run_prefix_perturbation(config_path)
    else:
        raise ValueError(f"Unsupported follow-up mode: {mode!r}")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
