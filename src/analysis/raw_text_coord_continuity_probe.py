from __future__ import annotations

import gc
from dataclasses import dataclass
import json
from pathlib import Path
import random
from statistics import mean

from PIL import Image
import yaml

from src.common.object_field_order import build_object_payload
from src.common.paths import resolve_image_path_strict
from src.infer.checkpoints import resolve_inference_checkpoint
from src.utils.assistant_json import (
    CANONICAL_JSON_SEPARATORS,
    dumps_canonical_json,
    dumps_coordjson,
)
from src.analysis.raw_text_coord_continuity_report import (
    compute_basin_metrics,
    summarize_wrong_anchor_advantage,
    write_report_bundle,
)

_VALID_STAGES = ("audit", "pilot", "canonical", "bad_basin", "dense_scene", "report")
REPO_ROOT = Path(__file__).resolve().parents[2]
_CANONICAL_JSON_SURFACE = "pretty_inline"
_SLOT_TO_INDEX = {"x1": 0, "y1": 1, "x2": 2, "y2": 3}


@dataclass(frozen=True)
class RunConfig:
    name: str
    output_dir: str
    stages: tuple[str, ...]


@dataclass(frozen=True)
class ModelConfig:
    alias: str
    path: str
    prompt_surface: str
    coord_mode: str = "norm1000_text"
    prompt_variant: str = "coco_80"
    object_field_order: str = "desc_first"
    json_surface: str = _CANONICAL_JSON_SURFACE


@dataclass(frozen=True)
class CohortConfig:
    jsonl_path: str
    sample_count: int
    seed: int


@dataclass(frozen=True)
class StudyModels:
    base: ModelConfig
    pure_ce: ModelConfig


@dataclass(frozen=True)
class StudyCohorts:
    val_headline: CohortConfig
    train_supplemental: CohortConfig


@dataclass(frozen=True)
class ScoringConfig:
    device: str = "cuda:0"
    attn_implementation: str = "auto"
    pilot_cohort: str = "val_headline"
    pilot_max_rows_per_model: int = 2
    pilot_max_objects_per_row: int = 1
    pilot_candidate_radius: int = 1
    pilot_slots: tuple[str, ...] = ("x1", "y1", "x2", "y2")


@dataclass(frozen=True)
class StudyConfig:
    run: RunConfig
    models: StudyModels
    cohorts: StudyCohorts
    scoring: ScoringConfig


def _load_yaml(config_path: Path) -> dict[str, object]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("study config root must be a mapping")
    return raw


def _require_mapping(parent: dict[str, object], key: str) -> dict[str, object]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _resolve_input_path(path_str: str, *, config_dir: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    config_relative = config_dir / path
    if config_relative.exists():
        return config_relative
    return REPO_ROOT / path


def _resolve_output_dir(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"jsonl row in {path} must be a mapping")
        rows.append(payload)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{json.dumps(row, ensure_ascii=False)}\n" for row in rows),
        encoding="utf-8",
    )


def _resolve_audit_model_info(model_path: Path) -> dict[str, object]:
    resolved = resolve_inference_checkpoint(model_checkpoint=str(model_path))
    processor_source = str(resolved.resolved_base_model_checkpoint)
    processor_source_path = Path(processor_source)
    return {
        "requested_model_path": str(model_path),
        "checkpoint_mode": resolved.checkpoint_mode,
        "resolved_base_model_checkpoint": resolved.resolved_base_model_checkpoint,
        "resolved_adapter_checkpoint": resolved.resolved_adapter_checkpoint,
        "processor_source": processor_source,
        "processor_source_is_local": processor_source_path.exists(),
        "has_coord_offset_adapter": bool(
            resolved.adapter_info is not None
            and resolved.adapter_info.coord_offset_spec is not None
        ),
    }


def _load_tokenizer_for_audit(model_path: Path) -> object:
    from transformers import AutoProcessor

    model_info = _resolve_audit_model_info(model_path)
    processor_source = str(model_info["processor_source"])
    processor = AutoProcessor.from_pretrained(
        processor_source,
        trust_remote_code=True,
        local_files_only=bool(model_info["processor_source_is_local"]),
    )
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError(
            f"processor at {processor_source} did not expose a tokenizer"
        )
    return tokenizer


def _tokenize_text_for_audit(tokenizer: object, text: str) -> dict[str, object]:
    if hasattr(tokenizer, "encode"):
        input_ids = list(tokenizer.encode(text, add_special_tokens=False))
    else:
        tokens = list(getattr(tokenizer, "tokenize")(text))
        return {
            "text": text,
            "token_ids": list(range(len(tokens))),
            "tokens": tokens,
            "token_count": len(tokens),
        }
    if hasattr(tokenizer, "convert_ids_to_tokens"):
        tokens = list(tokenizer.convert_ids_to_tokens(input_ids))
    else:
        tokens = list(getattr(tokenizer, "tokenize")(text))
    return {
        "text": text,
        "token_ids": input_ids,
        "tokens": tokens,
        "token_count": len(input_ids),
    }


def _first_diff_index(left: list[int], right: list[int]) -> int | None:
    for idx, (left_id, right_id) in enumerate(zip(left, right)):
        if int(left_id) != int(right_id):
            return idx
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def _build_surface_form_audit(tokenizer: object) -> dict[str, object]:
    sample_payload = {
        "objects": [
            {"desc": "book", "bbox_2d": [199, 200, 210, 250]},
            {"desc": "book", "bbox_2d": [231, 200, 260, 280]},
        ]
    }
    pretty_inline = dumps_canonical_json(sample_payload)
    compact = json.dumps(sample_payload, ensure_ascii=False, separators=(",", ":"))
    pretty_multiline = json.dumps(sample_payload, ensure_ascii=False, indent=2)
    variants = [
        ("pretty_inline", pretty_inline),
        ("compact", compact),
        ("pretty_multiline", pretty_multiline),
    ]
    tokenized = [
        {
            "label": label,
            **_tokenize_text_for_audit(tokenizer, text),
        }
        for label, text in variants
    ]
    canonical_token_ids = list(tokenized[0]["token_ids"])
    for row in tokenized[1:]:
        row["first_diff_vs_pretty_inline"] = _first_diff_index(
            canonical_token_ids,
            list(row["token_ids"]),
        )
    tokenized[0]["first_diff_vs_pretty_inline"] = None
    return {
        "canonical_label": _CANONICAL_JSON_SURFACE,
        "canonical_separators": list(CANONICAL_JSON_SEPARATORS),
        "sample_payload": sample_payload,
        "variants": tokenized,
    }


def render_pretty_inline_assistant_text(
    row: dict[str, object],
    *,
    object_field_order: str,
) -> str:
    objects_raw = row.get("objects")
    if not isinstance(objects_raw, list):
        raise ValueError("row.objects must be a list")
    rendered_objects: list[dict[str, object]] = []
    for idx, obj in enumerate(objects_raw):
        if not isinstance(obj, dict):
            raise ValueError(f"row.objects[{idx}] must be a mapping")
        desc = str(obj.get("desc") or "").strip()
        bbox = obj.get("bbox_2d")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(
                f"row.objects[{idx}].bbox_2d must be a 4-element list for raw-text xyxy probing"
            )
        rendered_objects.append(
            build_object_payload(
                desc=desc,
                geometry_key="bbox_2d",
                geometry_value=list(bbox),
                object_field_order=object_field_order,
            )
        )
    return dumps_coordjson({"objects": rendered_objects})


def build_candidate_values_around(
    center_value: int,
    *,
    radius: int,
) -> list[int]:
    lower = max(0, int(center_value) - int(radius))
    upper = min(999, int(center_value) + int(radius))
    return list(range(lower, upper + 1))


def run_phase0_audit(scorer: object) -> dict[str, object]:
    numbers = [0, 1, 9, 10, 99, 100, 199, 200, 210, 999]
    tokenizer = getattr(scorer, "tokenizer")
    rows = []
    for value in numbers:
        tokenized = _tokenize_text_for_audit(tokenizer, str(value))
        rows.append(
            {
                "value": value,
                "tokens": tokenized["tokens"],
                "token_ids": tokenized["token_ids"],
                "token_count": tokenized["token_count"],
            }
        )
    return {
        "numbers": rows,
        "surface_forms": _build_surface_form_audit(tokenizer),
    }


def build_random_cohort(
    rows: list[dict[str, object]],
    *,
    sample_count: int,
    seed: int,
) -> list[dict[str, object]]:
    cohort = list(rows)
    random.Random(seed).shuffle(cohort)
    return cohort[:sample_count]


def build_study_hard_cases(
    rows: list[dict[str, object]],
    *,
    max_cases: int,
) -> list[dict[str, object]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            int(row.get("same_desc_duplicate_pair_count") or 0),
            int(row.get("max_desc_count") or 0),
            int(row.get("pred_count") or 0),
        ),
        reverse=True,
    )
    return ordered[:max_cases]


def _bbox_points_from_object(obj: dict[str, object]) -> list[int] | None:
    bbox = obj.get("bbox_2d")
    if isinstance(bbox, list) and len(bbox) == 4:
        return [int(value) for value in bbox]
    points = obj.get("points")
    if isinstance(points, list) and len(points) == 4:
        return [int(value) for value in points]
    return None


def _bbox_to_norm1000(
    bbox_xyxy: list[int],
    *,
    width: int,
    height: int,
) -> list[int]:
    denom_w = max(float(width), 1.0)
    denom_h = max(float(height), 1.0)
    x1, y1, x2, y2 = bbox_xyxy
    return [
        int(max(0, min(999, round((float(x1) / denom_w) * 1000.0)))),
        int(max(0, min(999, round((float(y1) / denom_h) * 1000.0)))),
        int(max(0, min(999, round((float(x2) / denom_w) * 1000.0)))),
        int(max(0, min(999, round((float(y2) / denom_h) * 1000.0)))),
    ]


def _object_to_norm1000(
    obj: dict[str, object],
    *,
    width: int,
    height: int,
) -> dict[str, object] | None:
    bbox = _bbox_points_from_object(obj)
    if bbox is None:
        return None
    return {
        "desc": str(obj.get("desc") or ""),
        "bbox_2d": _bbox_to_norm1000(bbox, width=width, height=height),
    }


def select_self_prefix_duplicate_anchor(
    source_row: dict[str, object],
    *,
    duplicate_iou_threshold: float = 0.5,
    size_ratio_min: float = 0.6,
    local_center_radius_scale: float = 1.2,
    same_desc_iou_threshold: float = 0.5,
) -> dict[str, object] | None:
    from src.analysis.duplication_collapse_analysis import (
        _build_anchor_from_object_pair,
        _choose_gt_next_candidate,
        _pair_duplicate_metrics,
    )

    preds = list(source_row.get("pred") or [])
    gt_objects = list(source_row.get("gt") or [])
    if len(preds) < 2:
        return None
    subset_stub = type(
        "SubsetStub",
        (),
        {
            "duplicate_iou_threshold": float(duplicate_iou_threshold),
            "size_ratio_min": float(size_ratio_min),
            "local_center_radius_scale": float(local_center_radius_scale),
        },
    )()
    controls_stub = type(
        "ControlsStub",
        (),
        {"same_desc_iou_threshold": float(same_desc_iou_threshold)},
    )()
    cfg_stub = type("CfgStub", (), {"subset": subset_stub, "controls": controls_stub})()
    candidates: list[tuple[tuple[float, float, float, int, int], dict[str, object]]] = []
    for object_index in range(1, len(preds)):
        for source_index in range(object_index):
            pair_metrics = _pair_duplicate_metrics(
                preds[source_index],
                preds[object_index],
                cfg=cfg_stub,
            )
            if pair_metrics is None or not bool(pair_metrics.get("duplicate_like")):
                continue
            anchor = _build_anchor_from_object_pair(
                preds,
                confidence_record=None,
                cfg=cfg_stub,
                object_idx=object_index,
                source_object_idx=source_index,
                anchor_source="self_prefix_duplicate_like",
            )
            if anchor is None:
                continue
            sort_key = (
                float(pair_metrics["iou"]),
                float(pair_metrics["size_ratio"]),
                -float(pair_metrics["center_distance"]),
                -int(object_index),
                -int(source_index),
            )
            candidates.append((sort_key, anchor))
    if not candidates:
        return None
    best_anchor = max(candidates, key=lambda item: item[0])[1]
    object_index = int(best_anchor["object_idx"])
    source_index = int(best_anchor["source_object_idx"])
    prefix_objects = preds[:object_index]
    source_object = dict(preds[source_index])
    gt_next = _choose_gt_next_candidate(
        prefix_objects=prefix_objects,
        gt_objects=gt_objects,
        source_object=source_object,
        cfg=cfg_stub,
    )
    return {
        "object_index": object_index,
        "source_object_index": source_index,
        "pair_metrics": dict(best_anchor["pair_metrics"]),
        "pred_object": dict(preds[object_index]),
        "source_object": source_object,
        "gt_next": dict(gt_next) if gt_next is not None else None,
    }


def _load_probe_image(
    record: dict[str, object],
    *,
    subset_path: Path,
    source_jsonl_path: Path,
) -> Image.Image:
    image_rel = str(((record.get("images") or [None])[0] or record.get("image") or ""))
    resolved = resolve_image_path_strict(
        image_rel,
        jsonl_dir=subset_path.parent,
        root_image_dir=source_jsonl_path.parent,
    )
    if resolved is None:
        raise FileNotFoundError(f"Unable to resolve image path {image_rel!r}")
    return Image.open(resolved).convert("RGB")


def summarize_pilot_coordinate_records(
    rows: list[dict[str, object]],
) -> dict[str, object]:
    ok_rows = [row for row in rows if str(row.get("scoring_status")) == "ok"]
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in ok_rows:
        key = (
            row.get("model_alias"),
            row.get("cohort_name"),
            row.get("image_id"),
            row.get("object_index"),
            row.get("slot"),
        )
        grouped.setdefault(key, []).append(row)

    probe_metrics: list[dict[str, object]] = []
    for group_rows in grouped.values():
        exemplar = group_rows[0]
        best_row = max(group_rows, key=lambda row: float(row["score"]))
        metrics = compute_basin_metrics(group_rows, center_key="gt_value")
        probe_metrics.append(
            {
                "model_alias": exemplar["model_alias"],
                "cohort_name": exemplar["cohort_name"],
                "image_id": exemplar["image_id"],
                "object_index": exemplar["object_index"],
                "slot": exemplar["slot"],
                "desc": exemplar["desc"],
                "gt_value": exemplar["gt_value"],
                "num_candidates": len(group_rows),
                "best_candidate_value": best_row["candidate_value"],
                "best_candidate_is_gt": int(
                    int(best_row["candidate_value"]) == int(exemplar["gt_value"])
                ),
                **metrics,
            }
        )

    by_model_slot: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in probe_metrics:
        key = (str(row["model_alias"]), str(row["slot"]))
        by_model_slot.setdefault(key, []).append(row)
    slot_metrics: list[dict[str, object]] = []
    metric_keys = (
        "mass_at_1",
        "mass_at_2",
        "mass_at_4",
        "mass_at_8",
        "mass_at_16",
        "local_expected_abs_error",
        "half_height_width",
        "best_candidate_is_gt",
    )
    for (model_alias, slot), metric_rows in sorted(by_model_slot.items()):
        slot_metrics.append(
            {
                "model_alias": model_alias,
                "slot": slot,
                "num_probes": len(metric_rows),
                **{
                    key: float(mean(float(row[key]) for row in metric_rows))
                    for key in metric_keys
                },
            }
        )

    return {
        "candidate_rows_total": len(rows),
        "candidate_rows_ok": len(ok_rows),
        "candidate_rows_failed": len(rows) - len(ok_rows),
        "num_probes": len(probe_metrics),
        "slot_metrics": slot_metrics,
        "probe_metrics": probe_metrics,
    }


def summarize_bad_basin_coordinate_records(
    rows: list[dict[str, object]],
) -> dict[str, object]:
    ok_rows = [
        row
        for row in rows
        if str(row.get("scoring_status")) == "ok"
        and row.get("gt_value") is not None
        and row.get("pred_value") is not None
    ]
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in ok_rows:
        key = (
            row.get("model_alias"),
            row.get("case_id"),
            row.get("image_id"),
            row.get("object_index"),
            row.get("slot"),
        )
        grouped.setdefault(key, []).append(row)

    probe_metrics: list[dict[str, object]] = []
    for group_rows in grouped.values():
        exemplar = group_rows[0]
        pred_metrics = compute_basin_metrics(group_rows, center_key="pred_value")
        gt_metrics = compute_basin_metrics(group_rows, center_key="gt_value")
        previous_metrics = None
        if all(row.get("previous_value") is not None for row in group_rows):
            previous_metrics = compute_basin_metrics(
                group_rows,
                center_key="previous_value",
            )
        wrong_anchor = summarize_wrong_anchor_advantage(group_rows)
        probe_row = {
            "model_alias": exemplar["model_alias"],
            "case_id": exemplar["case_id"],
            "image_id": exemplar["image_id"],
            "object_index": exemplar["object_index"],
            "slot": exemplar["slot"],
            "pred_value": exemplar["pred_value"],
            "gt_value": exemplar["gt_value"],
            "num_candidates": len(group_rows),
            **{f"pred_center_{key}": value for key, value in pred_metrics.items()},
            **{f"gt_center_{key}": value for key, value in gt_metrics.items()},
            **wrong_anchor,
        }
        if previous_metrics is not None:
            probe_row["previous_value"] = exemplar["previous_value"]
            probe_row.update(
                {
                    f"previous_center_{key}": value
                    for key, value in previous_metrics.items()
                }
            )
        probe_metrics.append(probe_row)

    by_model_slot_center: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    metric_keys = (
        "mass_at_1",
        "mass_at_2",
        "mass_at_4",
        "mass_at_8",
        "mass_at_16",
        "local_expected_abs_error",
        "half_height_width",
    )
    for probe in probe_metrics:
        center_kinds = ["pred", "gt"]
        if probe.get("previous_value") is not None:
            center_kinds.append("previous")
        for center_kind in center_kinds:
            by_model_slot_center.setdefault(
                (str(probe["model_alias"]), str(probe["slot"]), center_kind),
                [],
            ).append(probe)
    center_metrics: list[dict[str, object]] = []
    for (model_alias, slot, center_kind), metric_rows in sorted(by_model_slot_center.items()):
        center_metrics.append(
            {
                "model_alias": model_alias,
                "slot": slot,
                "center_kind": center_kind,
                "num_probes": len(metric_rows),
                **{
                    key: float(
                        mean(
                            float(row[f"{center_kind}_center_{key}"])
                            for row in metric_rows
                        )
                    )
                    for key in metric_keys
                },
            }
        )

    return {
        "candidate_rows_total": len(rows),
        "candidate_rows_ok": len(ok_rows),
        "candidate_rows_failed": len(rows) - len(ok_rows),
        "num_probes": len(probe_metrics),
        "center_metrics": center_metrics,
        "probe_metrics": probe_metrics,
    }


def _run_bad_basin_scoring(
    *,
    cfg: StudyConfig,
    config_dir: Path,
    run_dir: Path,
) -> dict[str, object]:
    from src.analysis.raw_text_coord_continuity_scoring import (
        score_candidate_coordinate_sequences_batch,
        score_candidate_coordinate_sequence,
    )
    from src.analysis.unmatched_proposal_verifier import TeacherForcedScorer
    import torch

    bad_basin_dir = run_dir / "bad_basin"
    pilot_cohort = str(cfg.scoring.pilot_cohort)
    subset_path = run_dir / "cohorts" / f"{pilot_cohort}.jsonl"
    if not subset_path.exists():
        raise FileNotFoundError(f"bad_basin cohort manifest not found: {subset_path}")
    cohort_cfg = getattr(cfg.cohorts, pilot_cohort, None)
    if cohort_cfg is None:
        raise ValueError(f"unknown bad_basin cohort: {pilot_cohort}")
    source_jsonl_path = _resolve_input_path(
        cohort_cfg.jsonl_path,
        config_dir=config_dir,
    )
    selected_slots = tuple(str(slot) for slot in cfg.scoring.pilot_slots)
    invalid_slots = tuple(slot for slot in selected_slots if slot not in _SLOT_TO_INDEX)
    if invalid_slots:
        raise ValueError(f"unsupported bad_basin slot(s): {', '.join(invalid_slots)}")
    source_rows = _read_jsonl(subset_path)[: int(cfg.scoring.pilot_max_rows_per_model)]

    model_summaries: dict[str, object] = {}
    all_rows: list[dict[str, object]] = []
    models = {
        "base": cfg.models.base,
        "pure_ce": cfg.models.pure_ce,
    }
    for model_key, model_cfg in models.items():
        resolved_model_path = _resolve_input_path(model_cfg.path, config_dir=config_dir)
        scorer = TeacherForcedScorer(
            checkpoint_path=resolved_model_path,
            device=cfg.scoring.device,
            attn_implementation=cfg.scoring.attn_implementation,
            coord_mode=model_cfg.coord_mode,
        )
        model_rows: list[dict[str, object]] = []
        gt_cache: dict[Path, list[dict[str, object]]] = {}
        try:
            for row_idx, row in enumerate(source_rows):
                base_row = {
                    "model_key": model_key,
                    "model_alias": model_cfg.alias,
                    "cohort_name": pilot_cohort,
                    "row_index": row_idx,
                    "image_id": row.get("image_id"),
                    "prompt_surface": model_cfg.prompt_surface,
                    "coord_mode": model_cfg.coord_mode,
                    "prompt_variant": model_cfg.prompt_variant,
                    "object_field_order": model_cfg.object_field_order,
                    "json_surface": model_cfg.json_surface,
                }
                try:
                    image = _load_probe_image(
                        row,
                        subset_path=subset_path,
                        source_jsonl_path=source_jsonl_path,
                    )
                    source_gt_vs_pred = Path(str(row.get("source_gt_vs_pred_jsonl") or ""))
                    line_idx = int(row.get("line_idx") or 0)
                    if source_gt_vs_pred not in gt_cache:
                        gt_cache[source_gt_vs_pred] = _read_jsonl(source_gt_vs_pred)
                    source_payload = gt_cache[source_gt_vs_pred][line_idx]
                    anchor = select_self_prefix_duplicate_anchor(source_payload)
                    if anchor is None or anchor.get("gt_next") is None:
                        raise ValueError("unable_to_select_self_prefix_anchor")
                    width = int(source_payload.get("width") or row.get("width") or 0)
                    height = int(source_payload.get("height") or row.get("height") or 0)
                    prefix_objects = []
                    for pred_obj in list(source_payload.get("pred") or [])[: int(anchor["object_index"])]:
                        norm_obj = _object_to_norm1000(
                            dict(pred_obj),
                            width=width,
                            height=height,
                        )
                        if norm_obj is not None:
                            prefix_objects.append(norm_obj)
                    pred_object = _object_to_norm1000(
                        dict(anchor["pred_object"]),
                        width=width,
                        height=height,
                    )
                    gt_next = _object_to_norm1000(
                        dict(anchor["gt_next"]),
                        width=width,
                        height=height,
                    )
                    source_object = _object_to_norm1000(
                        dict(anchor["source_object"]),
                        width=width,
                        height=height,
                    )
                    if pred_object is None or gt_next is None or source_object is None:
                        raise ValueError("failed_to_normalize_bad_basin_objects")
                    assistant_text = render_pretty_inline_assistant_text(
                        {"objects": prefix_objects + [pred_object]},
                        object_field_order=model_cfg.object_field_order,
                    )
                    object_index = len(prefix_objects)
                    pred_bbox = list(pred_object["bbox_2d"])
                    gt_bbox = list(gt_next["bbox_2d"])
                    source_bbox = list(source_object["bbox_2d"])
                    case_id = (
                        f"{base_row['image_id']}:{line_idx}:"
                        f"{int(anchor['source_object_index'])}->{int(anchor['object_index'])}"
                    )
                except (FileNotFoundError, IndexError, ValueError) as exc:
                    model_rows.append(
                        {
                            **base_row,
                            "scoring_status": "failed",
                            "failure_reason": str(exc),
                        }
                    )
                    continue
                try:
                    desc = str(pred_object.get("desc") or "")
                    for slot in selected_slots:
                        slot_idx = int(_SLOT_TO_INDEX[slot])
                        pred_value = int(pred_bbox[slot_idx])
                        gt_value = int(gt_bbox[slot_idx])
                        candidate_values = sorted(
                            set(
                                build_candidate_values_around(
                                    pred_value,
                                    radius=cfg.scoring.pilot_candidate_radius,
                                )
                                + build_candidate_values_around(
                                    gt_value,
                                    radius=cfg.scoring.pilot_candidate_radius,
                                )
                            )
                        )
                        try:
                            scored_rows = score_candidate_coordinate_sequences_batch(
                                scorer=scorer,
                                image=image,
                                assistant_text=assistant_text,
                                slot=slot,
                                original_bbox=pred_bbox,
                                candidate_values=candidate_values,
                                prompt_variant=model_cfg.prompt_variant,
                                object_field_order=model_cfg.object_field_order,
                                object_index=object_index,
                            )
                            for scored in scored_rows:
                                candidate_value = int(scored["candidate_value"])
                                model_rows.append(
                                    {
                                        **base_row,
                                        "case_id": case_id,
                                        "scoring_status": "ok",
                                        "failure_reason": None,
                                        "object_index": object_index,
                                        "source_object_index": int(anchor["source_object_index"]),
                                        "desc": desc,
                                        "slot": slot,
                                        "pred_value": pred_value,
                                        "gt_value": gt_value,
                                        "previous_value": int(source_bbox[slot_idx]),
                                        "candidate_value": candidate_value,
                                        "numeric_distance_to_pred": abs(candidate_value - pred_value),
                                        "numeric_distance_to_gt": abs(candidate_value - gt_value),
                                        "score": float(scored["sum_logprob"]),
                                        "sum_logprob": float(scored["sum_logprob"]),
                                        "mean_logprob": float(scored["mean_logprob"]),
                                        "token_count": int(scored["count"]),
                                        "candidate_token_span": len(list(scored["absolute_positions"])),
                                    }
                                )
                            continue
                        except (RuntimeError, ValueError):
                            pass
                        for candidate_value in candidate_values:
                            try:
                                scored = score_candidate_coordinate_sequence(
                                    scorer=scorer,
                                    image=image,
                                    assistant_text=assistant_text,
                                    slot=slot,
                                    original_bbox=pred_bbox,
                                    candidate_value=candidate_value,
                                    prompt_variant=model_cfg.prompt_variant,
                                    object_field_order=model_cfg.object_field_order,
                                    object_index=object_index,
                                )
                                model_rows.append(
                                    {
                                        **base_row,
                                        "case_id": case_id,
                                        "scoring_status": "ok",
                                        "failure_reason": None,
                                        "object_index": object_index,
                                        "source_object_index": int(anchor["source_object_index"]),
                                        "desc": desc,
                                        "slot": slot,
                                        "pred_value": pred_value,
                                        "gt_value": gt_value,
                                        "previous_value": int(source_bbox[slot_idx]),
                                        "candidate_value": int(candidate_value),
                                        "numeric_distance_to_pred": abs(int(candidate_value) - pred_value),
                                        "numeric_distance_to_gt": abs(int(candidate_value) - gt_value),
                                        "score": float(scored["sum_logprob"]),
                                        "sum_logprob": float(scored["sum_logprob"]),
                                        "mean_logprob": float(scored["mean_logprob"]),
                                        "token_count": int(scored["count"]),
                                        "candidate_token_span": len(list(scored["absolute_positions"])),
                                    }
                                )
                            except (RuntimeError, ValueError) as exc:
                                model_rows.append(
                                    {
                                        **base_row,
                                        "case_id": case_id,
                                        "scoring_status": "failed",
                                        "failure_reason": str(exc),
                                        "object_index": object_index,
                                        "source_object_index": int(anchor["source_object_index"]),
                                        "desc": desc,
                                        "slot": slot,
                                        "pred_value": pred_value,
                                        "gt_value": gt_value,
                                        "candidate_value": int(candidate_value),
                                    }
                                )
                finally:
                    image.close()
        finally:
            del scorer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        per_model_path = bad_basin_dir / f"{model_key}_per_coord_scores.jsonl"
        per_model_summary = summarize_bad_basin_coordinate_records(model_rows)
        _write_jsonl(per_model_path, model_rows)
        _write_json(bad_basin_dir / f"{model_key}_summary.json", per_model_summary)
        model_summaries[model_key] = {
            **per_model_summary,
            "per_coord_scores_path": str(per_model_path),
        }
        all_rows.extend(model_rows)

    combined_summary = summarize_bad_basin_coordinate_records(all_rows)
    combined_path = bad_basin_dir / "per_coord_scores.jsonl"
    _write_jsonl(combined_path, all_rows)
    summary = {
        "cohort_name": pilot_cohort,
        "models": model_summaries,
        "per_coord_scores_path": str(combined_path),
        **combined_summary,
    }
    _write_json(bad_basin_dir / "summary.json", summary)
    return summary


def _materialize_random_cohort(
    cohort_name: str,
    cohort_cfg: CohortConfig,
    *,
    config_dir: Path,
    run_dir: Path,
) -> dict[str, object]:
    source_path = _resolve_input_path(cohort_cfg.jsonl_path, config_dir=config_dir)
    source_rows = _read_jsonl(source_path)
    selected_rows = build_random_cohort(
        source_rows,
        sample_count=cohort_cfg.sample_count,
        seed=cohort_cfg.seed,
    )
    manifest_path = run_dir / "cohorts" / f"{cohort_name}.jsonl"
    _write_jsonl(manifest_path, selected_rows)
    return {
        "jsonl_path": cohort_cfg.jsonl_path,
        "resolved_jsonl_path": str(source_path),
        "sample_count": cohort_cfg.sample_count,
        "seed": cohort_cfg.seed,
        "manifest_path": str(manifest_path),
        "num_rows": len(selected_rows),
    }


def _run_tokenization_audit(
    *,
    cfg: StudyConfig,
    config_dir: Path,
    run_dir: Path,
) -> dict[str, object]:
    audit_dir = run_dir / "audit"
    models = {
        "base": cfg.models.base,
        "pure_ce": cfg.models.pure_ce,
    }
    artifacts: dict[str, object] = {}
    for model_key, model_cfg in models.items():
        resolved_model_path = _resolve_input_path(model_cfg.path, config_dir=config_dir)
        model_info = _resolve_audit_model_info(resolved_model_path)
        tokenizer = _load_tokenizer_for_audit(resolved_model_path)
        audit = run_phase0_audit(type("AuditSurface", (), {"tokenizer": tokenizer})())
        audit["model_alias"] = model_cfg.alias
        audit["model_path"] = model_cfg.path
        audit["resolved_model_path"] = str(resolved_model_path)
        audit["serialization_surface"] = _CANONICAL_JSON_SURFACE
        audit["model_resolution"] = model_info
        out_path = audit_dir / f"{model_key}_tokenization.json"
        _write_json(out_path, audit)
        artifacts[model_key] = str(out_path)
    summary = {
        "artifacts": artifacts,
        "serialization_surface": _CANONICAL_JSON_SURFACE,
    }
    _write_json(audit_dir / "summary.json", summary)
    return summary


def _run_pilot_scoring(
    *,
    cfg: StudyConfig,
    config_dir: Path,
    run_dir: Path,
) -> dict[str, object]:
    from src.analysis.raw_text_coord_continuity_scoring import (
        score_candidate_coordinate_sequences_batch,
        score_candidate_coordinate_sequence,
    )
    from src.analysis.unmatched_proposal_verifier import TeacherForcedScorer
    import torch

    pilot_dir = run_dir / "pilot"
    pilot_cohort = str(cfg.scoring.pilot_cohort)
    subset_path = run_dir / "cohorts" / f"{pilot_cohort}.jsonl"
    if not subset_path.exists():
        raise FileNotFoundError(f"pilot cohort manifest not found: {subset_path}")
    cohort_cfg = getattr(cfg.cohorts, pilot_cohort, None)
    if cohort_cfg is None:
        raise ValueError(f"unknown pilot cohort: {pilot_cohort}")
    source_jsonl_path = _resolve_input_path(
        cohort_cfg.jsonl_path,
        config_dir=config_dir,
    )
    source_rows = _read_jsonl(subset_path)[: int(cfg.scoring.pilot_max_rows_per_model)]
    selected_slots = tuple(str(slot) for slot in cfg.scoring.pilot_slots)
    invalid_slots = tuple(slot for slot in selected_slots if slot not in _SLOT_TO_INDEX)
    if invalid_slots:
        raise ValueError(f"unsupported pilot slot(s): {', '.join(invalid_slots)}")

    model_summaries: dict[str, object] = {}
    all_rows: list[dict[str, object]] = []
    models = {
        "base": cfg.models.base,
        "pure_ce": cfg.models.pure_ce,
    }
    for model_key, model_cfg in models.items():
        resolved_model_path = _resolve_input_path(model_cfg.path, config_dir=config_dir)
        scorer = TeacherForcedScorer(
            checkpoint_path=resolved_model_path,
            device=cfg.scoring.device,
            attn_implementation=cfg.scoring.attn_implementation,
            coord_mode=model_cfg.coord_mode,
        )
        model_rows: list[dict[str, object]] = []
        try:
            for row_idx, row in enumerate(source_rows):
                base_row = {
                    "model_key": model_key,
                    "model_alias": model_cfg.alias,
                    "cohort_name": pilot_cohort,
                    "row_index": row_idx,
                    "image_id": row.get("image_id"),
                    "prompt_surface": model_cfg.prompt_surface,
                    "coord_mode": model_cfg.coord_mode,
                    "prompt_variant": model_cfg.prompt_variant,
                    "object_field_order": model_cfg.object_field_order,
                    "json_surface": model_cfg.json_surface,
                }
                try:
                    assistant_text = render_pretty_inline_assistant_text(
                        row,
                        object_field_order=model_cfg.object_field_order,
                    )
                    image = _load_probe_image(
                        row,
                        subset_path=subset_path,
                        source_jsonl_path=source_jsonl_path,
                    )
                except (FileNotFoundError, ValueError) as exc:
                    model_rows.append(
                        {
                            **base_row,
                            "scoring_status": "failed",
                            "failure_reason": str(exc),
                        }
                    )
                    continue
                try:
                    objects = row.get("objects")
                    if not isinstance(objects, list):
                        raise ValueError("row.objects must be a list")
                    for object_index, obj in enumerate(objects):
                        if int(object_index) >= int(cfg.scoring.pilot_max_objects_per_row):
                            break
                        if not isinstance(obj, dict):
                            continue
                        bbox = obj.get("bbox_2d")
                        if not isinstance(bbox, list) or len(bbox) != 4:
                            continue
                        desc = str(obj.get("desc") or "").strip()
                        bbox_values = [int(value) for value in bbox]
                        for slot in selected_slots:
                            gt_value = int(bbox_values[int(_SLOT_TO_INDEX[slot])])
                            candidate_values = build_candidate_values_around(
                                gt_value,
                                radius=cfg.scoring.pilot_candidate_radius,
                            )
                            try:
                                scored_rows = score_candidate_coordinate_sequences_batch(
                                    scorer=scorer,
                                    image=image,
                                    assistant_text=assistant_text,
                                    slot=slot,
                                    original_bbox=bbox_values,
                                    candidate_values=candidate_values,
                                    prompt_variant=model_cfg.prompt_variant,
                                    object_field_order=model_cfg.object_field_order,
                                    object_index=object_index,
                                )
                                for scored in scored_rows:
                                    candidate_value = int(scored["candidate_value"])
                                    model_rows.append(
                                        {
                                            **base_row,
                                            "scoring_status": "ok",
                                            "failure_reason": None,
                                            "object_index": object_index,
                                            "desc": desc,
                                            "slot": slot,
                                            "gt_value": gt_value,
                                            "candidate_value": candidate_value,
                                            "numeric_distance": abs(
                                                candidate_value - int(gt_value)
                                            ),
                                            "score": float(scored["sum_logprob"]),
                                            "sum_logprob": float(scored["sum_logprob"]),
                                            "mean_logprob": float(
                                                scored["mean_logprob"]
                                            ),
                                            "token_count": int(scored["count"]),
                                            "candidate_token_span": len(
                                                list(scored["absolute_positions"])
                                            ),
                                        }
                                    )
                                continue
                            except (RuntimeError, ValueError):
                                pass
                            for candidate_value in candidate_values:
                                try:
                                    scored = score_candidate_coordinate_sequence(
                                        scorer=scorer,
                                        image=image,
                                        assistant_text=assistant_text,
                                        slot=slot,
                                        original_bbox=bbox_values,
                                        candidate_value=candidate_value,
                                        prompt_variant=model_cfg.prompt_variant,
                                        object_field_order=model_cfg.object_field_order,
                                        object_index=object_index,
                                    )
                                    model_rows.append(
                                        {
                                            **base_row,
                                            "scoring_status": "ok",
                                            "failure_reason": None,
                                            "object_index": object_index,
                                            "desc": desc,
                                            "slot": slot,
                                            "gt_value": gt_value,
                                            "candidate_value": int(candidate_value),
                                            "numeric_distance": abs(
                                                int(candidate_value) - int(gt_value)
                                            ),
                                            "score": float(scored["sum_logprob"]),
                                            "sum_logprob": float(scored["sum_logprob"]),
                                            "mean_logprob": float(scored["mean_logprob"]),
                                            "token_count": int(scored["count"]),
                                            "candidate_token_span": len(
                                                list(scored["absolute_positions"])
                                            ),
                                        }
                                    )
                                except (RuntimeError, ValueError) as exc:
                                    model_rows.append(
                                        {
                                            **base_row,
                                            "scoring_status": "failed",
                                            "failure_reason": str(exc),
                                            "object_index": object_index,
                                            "desc": desc,
                                            "slot": slot,
                                            "gt_value": gt_value,
                                            "candidate_value": int(candidate_value),
                                        }
                                    )
                finally:
                    image.close()
        finally:
            del scorer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        per_model_path = pilot_dir / f"{model_key}_per_coord_scores.jsonl"
        per_model_summary = summarize_pilot_coordinate_records(model_rows)
        _write_jsonl(per_model_path, model_rows)
        _write_json(pilot_dir / f"{model_key}_summary.json", per_model_summary)
        model_summaries[model_key] = {
            **per_model_summary,
            "per_coord_scores_path": str(per_model_path),
        }
        all_rows.extend(model_rows)

    combined_summary = summarize_pilot_coordinate_records(all_rows)
    combined_path = pilot_dir / "per_coord_scores.jsonl"
    _write_jsonl(combined_path, all_rows)
    summary = {
        "cohort_name": pilot_cohort,
        "models": model_summaries,
        "per_coord_scores_path": str(combined_path),
        **combined_summary,
    }
    _write_json(pilot_dir / "summary.json", summary)
    return summary


def load_study_config(config_path: Path) -> StudyConfig:
    raw = _load_yaml(config_path)
    run_raw = _require_mapping(raw, "run")
    models_raw = _require_mapping(raw, "models")
    cohorts_raw = _require_mapping(raw, "cohorts")
    scoring_raw = raw.get("scoring") or {}
    if not isinstance(scoring_raw, dict):
        raise ValueError("scoring must be a mapping")
    stages = tuple(str(value) for value in run_raw.get("stages") or ())
    invalid_stages = tuple(stage for stage in stages if stage not in _VALID_STAGES)
    if invalid_stages:
        raise ValueError(f"unsupported stage(s): {', '.join(invalid_stages)}")
    return StudyConfig(
        run=RunConfig(
            name=str(run_raw["name"]),
            output_dir=str(run_raw["output_dir"]),
            stages=stages,
        ),
        models=StudyModels(
            base=ModelConfig(
                alias=str(_require_mapping(models_raw, "base")["alias"]),
                path=str(_require_mapping(models_raw, "base")["path"]),
                prompt_surface=str(_require_mapping(models_raw, "base")["prompt_surface"]),
                coord_mode=str(
                    _require_mapping(models_raw, "base").get("coord_mode", "norm1000_text")
                ),
                prompt_variant=str(
                    _require_mapping(models_raw, "base").get("prompt_variant", "coco_80")
                ),
                object_field_order=str(
                    _require_mapping(models_raw, "base").get(
                        "object_field_order",
                        "desc_first",
                    )
                ),
                json_surface=str(
                    _require_mapping(models_raw, "base").get(
                        "json_surface",
                        _CANONICAL_JSON_SURFACE,
                    )
                ),
            ),
            pure_ce=ModelConfig(
                alias=str(_require_mapping(models_raw, "pure_ce")["alias"]),
                path=str(_require_mapping(models_raw, "pure_ce")["path"]),
                prompt_surface=str(_require_mapping(models_raw, "pure_ce")["prompt_surface"]),
                coord_mode=str(
                    _require_mapping(models_raw, "pure_ce").get(
                        "coord_mode",
                        "norm1000_text",
                    )
                ),
                prompt_variant=str(
                    _require_mapping(models_raw, "pure_ce").get(
                        "prompt_variant",
                        "coco_80",
                    )
                ),
                object_field_order=str(
                    _require_mapping(models_raw, "pure_ce").get(
                        "object_field_order",
                        "desc_first",
                    )
                ),
                json_surface=str(
                    _require_mapping(models_raw, "pure_ce").get(
                        "json_surface",
                        _CANONICAL_JSON_SURFACE,
                    )
                ),
            ),
        ),
        cohorts=StudyCohorts(
            val_headline=CohortConfig(
                jsonl_path=str(_require_mapping(cohorts_raw, "val_headline")["jsonl_path"]),
                sample_count=int(_require_mapping(cohorts_raw, "val_headline")["sample_count"]),
                seed=int(_require_mapping(cohorts_raw, "val_headline")["seed"]),
            ),
            train_supplemental=CohortConfig(
                jsonl_path=str(_require_mapping(cohorts_raw, "train_supplemental")["jsonl_path"]),
                sample_count=int(_require_mapping(cohorts_raw, "train_supplemental")["sample_count"]),
                seed=int(_require_mapping(cohorts_raw, "train_supplemental")["seed"]),
            ),
        ),
        scoring=ScoringConfig(
            device=str(scoring_raw.get("device", "cuda:0")),
            attn_implementation=str(scoring_raw.get("attn_implementation", "auto")),
            pilot_cohort=str(scoring_raw.get("pilot_cohort", "val_headline")),
            pilot_max_rows_per_model=int(
                scoring_raw.get("pilot_max_rows_per_model", 2)
            ),
            pilot_max_objects_per_row=int(
                scoring_raw.get("pilot_max_objects_per_row", 1)
            ),
            pilot_candidate_radius=int(
                scoring_raw.get("pilot_candidate_radius", 1)
            ),
            pilot_slots=tuple(
                str(slot)
                for slot in (scoring_raw.get("pilot_slots") or ("x1", "y1", "x2", "y2"))
            ),
        ),
    )


def run_study(config_path: Path) -> dict[str, object]:
    resolved_config_path = config_path.resolve()
    cfg = load_study_config(resolved_config_path)
    run_dir = _resolve_output_dir(cfg.run.output_dir) / cfg.run.name
    run_dir.mkdir(parents=True, exist_ok=True)
    val_cohort = _materialize_random_cohort(
        "val_headline",
        cfg.cohorts.val_headline,
        config_dir=resolved_config_path.parent,
        run_dir=run_dir,
    )
    train_cohort = _materialize_random_cohort(
        "train_supplemental",
        cfg.cohorts.train_supplemental,
        config_dir=resolved_config_path.parent,
        run_dir=run_dir,
    )
    summary = {
        "run_name": cfg.run.name,
        "stages": list(cfg.run.stages),
        "models": {
            "base": {
                "alias": cfg.models.base.alias,
                "path": cfg.models.base.path,
                "prompt_surface": cfg.models.base.prompt_surface,
                "coord_mode": cfg.models.base.coord_mode,
                "prompt_variant": cfg.models.base.prompt_variant,
                "object_field_order": cfg.models.base.object_field_order,
                "json_surface": cfg.models.base.json_surface,
            },
            "pure_ce": {
                "alias": cfg.models.pure_ce.alias,
                "path": cfg.models.pure_ce.path,
                "prompt_surface": cfg.models.pure_ce.prompt_surface,
                "coord_mode": cfg.models.pure_ce.coord_mode,
                "prompt_variant": cfg.models.pure_ce.prompt_variant,
                "object_field_order": cfg.models.pure_ce.object_field_order,
                "json_surface": cfg.models.pure_ce.json_surface,
            },
        },
        "cohorts": {
            "val_headline": val_cohort,
            "train_supplemental": train_cohort,
        },
    }
    if "audit" in cfg.run.stages:
        summary["audit"] = _run_tokenization_audit(
            cfg=cfg,
            config_dir=resolved_config_path.parent,
            run_dir=run_dir,
        )
    if "pilot" in cfg.run.stages:
        summary["pilot"] = _run_pilot_scoring(
            cfg=cfg,
            config_dir=resolved_config_path.parent,
            run_dir=run_dir,
        )
    if "bad_basin" in cfg.run.stages:
        summary["bad_basin"] = _run_bad_basin_scoring(
            cfg=cfg,
            config_dir=resolved_config_path.parent,
            run_dir=run_dir,
        )
    if "report" in cfg.run.stages:
        write_report_bundle(
            out_dir=run_dir,
            summary=summary,
            report_md="# Raw-Text Coordinate Continuity Probe\n",
            per_coord_rows=[],
            hard_cases=[],
        )
    else:
        _write_json(run_dir / "summary.json", summary)
    return {"run_dir": str(run_dir), "summary": summary}
