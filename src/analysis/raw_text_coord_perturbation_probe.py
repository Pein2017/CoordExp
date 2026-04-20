from __future__ import annotations

import gc
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from statistics import mean
from typing import Mapping, Sequence

import yaml

from src.analysis.duplication_followup import build_prefix_perturbation_variants
from src.analysis.raw_text_coord_continuity_probe import (
    _CANONICAL_JSON_SURFACE,
    _SLOT_TO_INDEX,
    _load_probe_image,
    _object_to_norm1000,
    _require_mapping,
    _resolve_input_path,
    _resolve_output_dir,
    _write_json,
    _write_jsonl,
    build_candidate_values_around,
    render_pretty_inline_assistant_text,
)
from src.analysis.raw_text_coord_continuity_report import (
    compute_basin_metrics,
    summarize_wrong_anchor_advantage,
)


@dataclass(frozen=True)
class PerturbationRunConfig:
    name: str
    output_dir: str


@dataclass(frozen=True)
class PerturbationSelectionConfig:
    bad_basin_summary_json: str
    cohort_jsonl: str
    max_cases: int = 8
    target_model_alias: str = "pure_ce"
    slots: tuple[str, ...] = ("x1", "y1")
    require_positive_advantage: bool = True


@dataclass(frozen=True)
class PerturbationModelConfig:
    alias: str
    path: str
    prompt_surface: str = "raw_text_xyxy"
    coord_mode: str = "norm1000_text"
    prompt_variant: str = "coco_80"
    object_field_order: str = "desc_first"
    json_surface: str = _CANONICAL_JSON_SURFACE


@dataclass(frozen=True)
class PerturbationModels:
    base: PerturbationModelConfig
    pure_ce: PerturbationModelConfig


@dataclass(frozen=True)
class PerturbationScoringConfig:
    device: str = "cuda:0"
    attn_implementation: str = "auto"
    candidate_radius: int = 8


@dataclass(frozen=True)
class PerturbationConfig:
    run: PerturbationRunConfig
    selection: PerturbationSelectionConfig
    models: PerturbationModels
    scoring: PerturbationScoringConfig


def _load_yaml(config_path: Path) -> dict[str, object]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("perturbation config root must be a mapping")
    return raw


def _parse_model_config(parent: dict[str, object], key: str) -> PerturbationModelConfig:
    raw = _require_mapping(parent, key)
    return PerturbationModelConfig(
        alias=str(raw["alias"]),
        path=str(raw["path"]),
        prompt_surface=str(raw.get("prompt_surface", "raw_text_xyxy")),
        coord_mode=str(raw.get("coord_mode", "norm1000_text")),
        prompt_variant=str(raw.get("prompt_variant", "coco_80")),
        object_field_order=str(raw.get("object_field_order", "desc_first")),
        json_surface=str(raw.get("json_surface", _CANONICAL_JSON_SURFACE)),
    )


def load_perturbation_config(config_path: Path) -> PerturbationConfig:
    raw = _load_yaml(config_path)
    run_raw = _require_mapping(raw, "run")
    selection_raw = _require_mapping(raw, "selection")
    models_raw = _require_mapping(raw, "models")
    scoring_raw = raw.get("scoring") or {}
    if not isinstance(scoring_raw, dict):
        raise ValueError("scoring must be a mapping")
    return PerturbationConfig(
        run=PerturbationRunConfig(
            name=str(run_raw["name"]),
            output_dir=str(run_raw["output_dir"]),
        ),
        selection=PerturbationSelectionConfig(
            bad_basin_summary_json=str(selection_raw["bad_basin_summary_json"]),
            cohort_jsonl=str(selection_raw["cohort_jsonl"]),
            max_cases=int(selection_raw.get("max_cases", 8)),
            target_model_alias=str(selection_raw.get("target_model_alias", "pure_ce")),
            slots=tuple(
                str(slot)
                for slot in selection_raw.get("slots", ("x1", "y1"))
            ),
            require_positive_advantage=bool(
                selection_raw.get("require_positive_advantage", True)
            ),
        ),
        models=PerturbationModels(
            base=_parse_model_config(models_raw, "base"),
            pure_ce=_parse_model_config(models_raw, "pure_ce"),
        ),
        scoring=PerturbationScoringConfig(
            device=str(scoring_raw.get("device", "cuda:0")),
            attn_implementation=str(scoring_raw.get("attn_implementation", "auto")),
            candidate_radius=int(scoring_raw.get("candidate_radius", 8)),
        ),
    )


def _extract_probe_rows_from_bad_basin_summary(
    summary_payload: Mapping[str, object],
) -> list[dict[str, object]]:
    models_raw = summary_payload.get("models")
    rows: list[dict[str, object]] = []
    if isinstance(models_raw, dict):
        for model_payload in models_raw.values():
            if not isinstance(model_payload, dict):
                continue
            probe_metrics = model_payload.get("probe_metrics") or []
            if not isinstance(probe_metrics, list):
                continue
            for row in probe_metrics:
                if isinstance(row, dict):
                    rows.append(dict(row))
    probe_metrics = summary_payload.get("probe_metrics") or []
    if isinstance(probe_metrics, list):
        for row in probe_metrics:
            if isinstance(row, dict):
                rows.append(dict(row))
    return rows


def select_perturbation_cases(
    probe_rows: Sequence[Mapping[str, object]],
    *,
    max_cases: int,
    target_model_alias: str,
    slots: Sequence[str],
    require_positive_advantage: bool,
) -> list[dict[str, object]]:
    filtered_rows = [
        dict(row)
        for row in probe_rows
        if str(row.get("model_alias") or "") == str(target_model_alias)
        and str(row.get("slot") or "") in set(str(slot) for slot in slots)
        and row.get("case_id") is not None
    ]
    best_by_case: dict[str, dict[str, object]] = {}
    for row in filtered_rows:
        advantage = float(row.get("wrong_anchor_advantage_at_4") or 0.0)
        if require_positive_advantage and advantage <= 0.0:
            continue
        case_id = str(row["case_id"])
        previous = best_by_case.get(case_id)
        if previous is None or float(previous["selection_wrong_anchor_advantage_at_4"]) < advantage:
            selected = dict(row)
            selected["selection_wrong_anchor_advantage_at_4"] = advantage
            selected["selection_slot"] = str(row.get("slot") or "")
            best_by_case[case_id] = selected
    ordered = sorted(
        best_by_case.values(),
        key=lambda row: (
            -float(row["selection_wrong_anchor_advantage_at_4"]),
            int(row.get("image_id") or -1),
            str(row["case_id"]),
        ),
    )
    return ordered[: max(0, int(max_cases))]


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"expected JSON object per line in {path}")
        rows.append(payload)
    return rows


def _parse_case_id(case_id: str) -> dict[str, int]:
    try:
        image_id_str, line_idx_str, object_part = str(case_id).split(":")
        source_idx_str, object_idx_str = object_part.split("->")
        return {
            "image_id": int(image_id_str),
            "line_idx": int(line_idx_str),
            "source_object_index": int(source_idx_str),
            "object_index": int(object_idx_str),
        }
    except ValueError as exc:
        raise ValueError(f"unsupported case_id format: {case_id!r}") from exc


def _cohort_lookup_key(row: Mapping[str, object]) -> tuple[int, int]:
    image_id = row.get("image_id")
    line_idx = row.get("line_idx")
    return (
        int(image_id if image_id is not None else -1),
        int(line_idx if line_idx is not None else -1),
    )


def _controls_stub() -> object:
    return type(
        "CfgStub",
        (),
        {
            "controls": type(
                "ControlsStub",
                (),
                {"same_desc_iou_threshold": 0.5},
            )()
        },
    )()


def _normalize_prefix_objects(
    objects: Sequence[Mapping[str, object]],
    *,
    width: int,
    height: int,
) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for obj in objects:
        norm_obj = _object_to_norm1000(dict(obj), width=width, height=height)
        if norm_obj is None:
            raise ValueError("failed_to_normalize_prefix_object")
        normalized.append(norm_obj)
    return normalized


def _canonicalize_variant_prefix_objects(
    prefix_objects: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    canonical: list[dict[str, object]] = []
    for obj in prefix_objects:
        points = obj.get("points")
        bbox = obj.get("bbox_2d")
        if isinstance(points, list) and len(points) >= 4:
            bbox_xyxy = [int(round(float(points[idx]))) for idx in range(4)]
        elif isinstance(bbox, list) and len(bbox) >= 4:
            bbox_xyxy = [int(round(float(bbox[idx]))) for idx in range(4)]
        else:
            raise ValueError("variant_object_missing_bbox")
        clone = dict(obj)
        clone["bbox_2d"] = bbox_xyxy
        canonical.append(clone)
    return canonical


def _variant_source_value(
    variant_prefix: Sequence[Mapping[str, object]],
    *,
    source_object_index: int,
    slot: str,
) -> int | None:
    if not (0 <= int(source_object_index) < len(variant_prefix)):
        return None
    bbox = variant_prefix[int(source_object_index)].get("bbox_2d")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    return int(bbox[int(_SLOT_TO_INDEX[slot])])


def summarize_perturbation_case_rows(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    summary_rows = [dict(row) for row in rows]
    baseline_by_key = {
        (
            str(row.get("model_alias") or ""),
            str(row.get("case_id") or ""),
            str(row.get("slot") or ""),
        ): row
        for row in summary_rows
        if str(row.get("variant") or "") == "baseline"
    }
    delta_rows: list[dict[str, object]] = []
    for row in summary_rows:
        variant = str(row.get("variant") or "")
        if variant == "baseline":
            continue
        key = (
            str(row.get("model_alias") or ""),
            str(row.get("case_id") or ""),
            str(row.get("slot") or ""),
        )
        baseline = baseline_by_key.get(key)
        if baseline is None:
            continue
        delta_rows.append(
            {
                "model_alias": str(row.get("model_alias") or ""),
                "case_id": str(row.get("case_id") or ""),
                "slot": str(row.get("slot") or ""),
                "variant": variant,
                "delta_wrong_anchor_advantage_at_4": (
                    float(row.get("wrong_anchor_advantage_at_4") or 0.0)
                    - float(baseline.get("wrong_anchor_advantage_at_4") or 0.0)
                ),
                "delta_pred_center_mass_at_4": (
                    float(row.get("pred_center_mass_at_4") or 0.0)
                    - float(baseline.get("pred_center_mass_at_4") or 0.0)
                ),
                "delta_gt_center_mass_at_4": (
                    float(row.get("gt_center_mass_at_4") or 0.0)
                    - float(baseline.get("gt_center_mass_at_4") or 0.0)
                ),
            }
        )
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in delta_rows:
        grouped[(row["model_alias"], row["variant"], row["slot"])].append(row)
    variant_metrics = [
        {
            "model_alias": model_alias,
            "variant": variant,
            "slot": slot,
            "num_cases": len(group_rows),
            "avg_delta_wrong_anchor_advantage_at_4": float(
                mean(float(row["delta_wrong_anchor_advantage_at_4"]) for row in group_rows)
            ),
            "avg_delta_pred_center_mass_at_4": float(
                mean(float(row["delta_pred_center_mass_at_4"]) for row in group_rows)
            ),
            "avg_delta_gt_center_mass_at_4": float(
                mean(float(row["delta_gt_center_mass_at_4"]) for row in group_rows)
            ),
        }
        for (model_alias, variant, slot), group_rows in sorted(grouped.items())
    ]
    variant_metrics.sort(
        key=lambda row: (
            float(row["avg_delta_wrong_anchor_advantage_at_4"]),
            str(row["model_alias"]),
            str(row["variant"]),
            str(row["slot"]),
        )
    )
    return {
        "num_summary_rows": len(summary_rows),
        "num_delta_rows": len(delta_rows),
        "variant_metrics": variant_metrics,
        "delta_rows": delta_rows,
    }


def _prepare_selected_cases(
    *,
    selection_rows: Sequence[Mapping[str, object]],
    cohort_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    cohort_by_key = {_cohort_lookup_key(row): dict(row) for row in cohort_rows}
    cohort_rows_by_image_id: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in cohort_rows:
        cohort_rows_by_image_id[int(row.get("image_id") or -1)].append(dict(row))
    selected_cases: list[dict[str, object]] = []
    for rank, row in enumerate(selection_rows, start=1):
        parsed: dict[str, int]
        cohort_row: dict[str, object] | None
        try:
            parsed = _parse_case_id(str(row["case_id"]))
            key = (parsed["image_id"], parsed["line_idx"])
            cohort_row = cohort_by_key.get(key)
        except ValueError:
            image_id_raw = row.get("image_id")
            image_id = int(image_id_raw if image_id_raw is not None else -1)
            candidates = cohort_rows_by_image_id.get(image_id, [])
            cohort_row = candidates[0] if len(candidates) == 1 else None
            row_line_idx = row.get("line_idx")
            row_source_idx = row.get("source_object_index")
            row_object_idx = row.get("object_index")
            parsed = {
                "image_id": image_id,
                "line_idx": int(
                    (cohort_row or {}).get("line_idx")
                    if (cohort_row or {}).get("line_idx") is not None
                    else (row_line_idx if row_line_idx is not None else -1)
                ),
                "source_object_index": int(
                    row_source_idx if row_source_idx is not None else -1
                ),
                "object_index": int(
                    row_object_idx if row_object_idx is not None else -1
                ),
            }
        if cohort_row is None:
            raise KeyError(f"selected case missing from cohort manifest: {row['case_id']}")
        selected_cases.append(
            {
                **cohort_row,
                **dict(row),
                **parsed,
                "selection_rank": rank,
            }
        )
    return selected_cases


def _build_case_context(
    case_row: Mapping[str, object],
    *,
    cohort_jsonl_path: Path,
    source_cache: dict[Path, list[dict[str, object]]],
) -> dict[str, object]:
    from src.analysis.duplication_collapse_analysis import _choose_gt_next_candidate

    source_jsonl_path = Path(str(case_row["source_gt_vs_pred_jsonl"]))
    if source_jsonl_path not in source_cache:
        source_cache[source_jsonl_path] = _read_jsonl(source_jsonl_path)
    line_idx = int(case_row["line_idx"])
    source_payload = source_cache[source_jsonl_path][line_idx]
    preds = list(source_payload.get("pred") or [])
    gt_objects = list(source_payload.get("gt") or [])
    object_index = int(case_row["object_index"])
    source_object_index = int(case_row["source_object_index"])
    if not (0 <= object_index < len(preds)):
        raise ValueError("object_index_out_of_range")
    if not (0 <= source_object_index < object_index):
        raise ValueError("source_object_index_out_of_range")
    width = int(source_payload.get("width") or case_row.get("width") or 0)
    height = int(source_payload.get("height") or case_row.get("height") or 0)
    prefix_pred_objects = [dict(obj) for obj in preds[:object_index]]
    source_object = dict(preds[source_object_index])
    pred_object = dict(preds[object_index])
    gt_next = _choose_gt_next_candidate(
        prefix_objects=prefix_pred_objects,
        gt_objects=gt_objects,
        source_object=source_object,
        cfg=_controls_stub(),
    )
    if gt_next is None:
        raise ValueError("unable_to_choose_gt_next")
    prefix_objects = _normalize_prefix_objects(
        prefix_pred_objects,
        width=width,
        height=height,
    )
    pred_object_norm = _object_to_norm1000(pred_object, width=width, height=height)
    source_object_norm = _object_to_norm1000(source_object, width=width, height=height)
    gt_next_norm = _object_to_norm1000(dict(gt_next), width=width, height=height)
    if pred_object_norm is None or source_object_norm is None or gt_next_norm is None:
        raise ValueError("failed_to_normalize_case_objects")
    probe_image_record = dict(source_payload)
    if case_row.get("images") is not None:
        probe_image_record["images"] = case_row.get("images")
    if case_row.get("image") is not None:
        probe_image_record["image"] = case_row.get("image")
    image_context_path = cohort_jsonl_path if (
        probe_image_record.get("images") is not None or probe_image_record.get("image") is not None
    ) else source_jsonl_path
    image = _load_probe_image(
        probe_image_record,
        subset_path=cohort_jsonl_path,
        source_jsonl_path=image_context_path,
    )
    return {
        "source_payload": source_payload,
        "source_jsonl_path": source_jsonl_path,
        "width": width,
        "height": height,
        "prefix_objects": prefix_objects,
        "pred_object": pred_object_norm,
        "source_object": source_object_norm,
        "gt_next": gt_next_norm,
        "image": image,
    }


def _score_variant_rows(
    *,
    scorer: object,
    model_cfg: PerturbationModelConfig,
    case_row: Mapping[str, object],
    variant: str,
    variant_prefix: Sequence[Mapping[str, object]],
    pred_object: Mapping[str, object],
    gt_next: Mapping[str, object],
    source_object_index: int,
    candidate_radius: int,
    slots: Sequence[str],
    image: object,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    from src.analysis.raw_text_coord_continuity_scoring import (
        score_candidate_coordinate_sequence,
        score_candidate_coordinate_sequences_batch,
    )

    assistant_text = render_pretty_inline_assistant_text(
        {"objects": [dict(obj) for obj in variant_prefix] + [dict(pred_object)]},
        object_field_order=model_cfg.object_field_order,
    )
    object_index = len(variant_prefix)
    pred_bbox = [int(value) for value in list(pred_object["bbox_2d"])]
    gt_bbox = [int(value) for value in list(gt_next["bbox_2d"])]
    case_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for slot in slots:
        slot_idx = int(_SLOT_TO_INDEX[slot])
        pred_value = int(pred_bbox[slot_idx])
        gt_value = int(gt_bbox[slot_idx])
        candidate_values = sorted(
            set(
                build_candidate_values_around(pred_value, radius=candidate_radius)
                + build_candidate_values_around(gt_value, radius=candidate_radius)
            )
        )
        variant_source_value = _variant_source_value(
            variant_prefix,
            source_object_index=source_object_index,
            slot=slot,
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
        except (RuntimeError, ValueError):
            scored_rows = []
            for candidate_value in candidate_values:
                scored_rows.append(
                    score_candidate_coordinate_sequence(
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
                )
                scored_rows[-1]["candidate_value"] = int(candidate_value)
        slot_rows: list[dict[str, object]] = []
        for scored in scored_rows:
            candidate_value = int(scored["candidate_value"])
            row = {
                "model_alias": model_cfg.alias,
                "case_id": str(case_row["case_id"]),
                "image_id": int(case_row["image_id"]),
                "line_idx": int(case_row["line_idx"]),
                "object_index": int(case_row["object_index"]),
                "source_object_index": int(case_row["source_object_index"]),
                "variant": variant,
                "slot": slot,
                "candidate_value": candidate_value,
                "pred_value": pred_value,
                "gt_value": gt_value,
                "baseline_source_value": int(
                    list(case_row["source_object_bbox"])[slot_idx]
                ),
                "variant_source_value": variant_source_value,
                "numeric_distance_to_pred": abs(candidate_value - pred_value),
                "numeric_distance_to_gt": abs(candidate_value - gt_value),
                "score": float(scored["sum_logprob"]),
                "sum_logprob": float(scored["sum_logprob"]),
                "mean_logprob": float(scored["mean_logprob"]),
                "token_count": int(scored["count"]),
                "candidate_token_span": len(list(scored["absolute_positions"])),
                "scoring_status": "ok",
            }
            case_rows.append(row)
            slot_rows.append(row)
        if slot_rows:
            pred_metrics = compute_basin_metrics(slot_rows, center_key="pred_value")
            gt_metrics = compute_basin_metrics(slot_rows, center_key="gt_value")
            wrong_anchor = summarize_wrong_anchor_advantage(slot_rows)
            summary_rows.append(
                {
                    "model_alias": model_cfg.alias,
                    "case_id": str(case_row["case_id"]),
                    "image_id": int(case_row["image_id"]),
                    "line_idx": int(case_row["line_idx"]),
                    "object_index": int(case_row["object_index"]),
                    "source_object_index": int(case_row["source_object_index"]),
                    "variant": variant,
                    "slot": slot,
                    "selection_wrong_anchor_advantage_at_4": float(
                        case_row["selection_wrong_anchor_advantage_at_4"]
                    ),
                    "baseline_source_value": int(
                        list(case_row["source_object_bbox"])[slot_idx]
                    ),
                    "variant_source_value": variant_source_value,
                    **{f"pred_center_{key}": value for key, value in pred_metrics.items()},
                    **{f"gt_center_{key}": value for key, value in gt_metrics.items()},
                    **wrong_anchor,
                }
            )
    return case_rows, summary_rows


def _run_selected_perturbation_cases(
    *,
    cfg: PerturbationConfig,
    config_dir: Path,
    selected_cases: Sequence[Mapping[str, object]],
) -> dict[str, list[dict[str, object]]]:
    from src.analysis.unmatched_proposal_verifier import TeacherForcedScorer
    import torch

    per_coord_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    cohort_jsonl_path = _resolve_input_path(
        cfg.selection.cohort_jsonl,
        config_dir=config_dir,
    )
    source_cache: dict[Path, list[dict[str, object]]] = {}
    models = (cfg.models.base, cfg.models.pure_ce)
    for model_cfg in models:
        scorer = TeacherForcedScorer(
            checkpoint_path=_resolve_input_path(model_cfg.path, config_dir=config_dir),
            device=cfg.scoring.device,
            attn_implementation=cfg.scoring.attn_implementation,
            coord_mode=model_cfg.coord_mode,
        )
        try:
            for selected_case in selected_cases:
                image = None
                try:
                    case_context = _build_case_context(
                        selected_case,
                        cohort_jsonl_path=cohort_jsonl_path,
                        source_cache=source_cache,
                    )
                    image = case_context["image"]
                    prefix_objects = list(case_context["prefix_objects"])
                    pred_object = dict(case_context["pred_object"])
                    gt_next = dict(case_context["gt_next"])
                    source_object = dict(case_context["source_object"])
                    variants = [("baseline", prefix_objects)]
                    variants.extend(
                        build_prefix_perturbation_variants(
                            prefix_objects=prefix_objects,
                            source_index_in_prefix=int(selected_case["source_object_index"]),
                            gt_next=gt_next,
                        )
                    )
                    case_row = {
                        **dict(selected_case),
                        "source_object_bbox": list(source_object["bbox_2d"]),
                    }
                    for variant, variant_prefix in variants:
                        canonical_variant_prefix = _canonicalize_variant_prefix_objects(
                            variant_prefix
                        )
                        variant_rows, variant_summary_rows = _score_variant_rows(
                            scorer=scorer,
                            model_cfg=model_cfg,
                            case_row=case_row,
                            variant=variant,
                            variant_prefix=canonical_variant_prefix,
                            pred_object=pred_object,
                            gt_next=gt_next,
                            source_object_index=int(selected_case["source_object_index"]),
                            candidate_radius=cfg.scoring.candidate_radius,
                            slots=cfg.selection.slots,
                            image=image,
                        )
                        per_coord_rows.extend(variant_rows)
                        summary_rows.extend(variant_summary_rows)
                except (FileNotFoundError, IndexError, KeyError, RuntimeError, ValueError) as exc:
                    per_coord_rows.append(
                        {
                            "model_alias": model_cfg.alias,
                            "case_id": str(selected_case.get("case_id") or ""),
                            "image_id": selected_case.get("image_id"),
                            "line_idx": selected_case.get("line_idx"),
                            "object_index": selected_case.get("object_index"),
                            "source_object_index": selected_case.get("source_object_index"),
                            "variant": "baseline",
                            "slot": None,
                            "scoring_status": "failed",
                            "failure_reason": str(exc),
                        }
                    )
                finally:
                    if image is not None:
                        image.close()
        finally:
            del scorer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return {
        "per_coord_rows": per_coord_rows,
        "summary_rows": summary_rows,
    }


def run_perturbation_probe(config_path: Path) -> dict[str, object]:
    resolved_config_path = config_path.resolve()
    cfg = load_perturbation_config(resolved_config_path)
    run_dir = _resolve_output_dir(cfg.run.output_dir) / cfg.run.name
    run_dir.mkdir(parents=True, exist_ok=True)

    bad_basin_summary_path = _resolve_input_path(
        cfg.selection.bad_basin_summary_json,
        config_dir=resolved_config_path.parent,
    )
    cohort_jsonl_path = _resolve_input_path(
        cfg.selection.cohort_jsonl,
        config_dir=resolved_config_path.parent,
    )
    bad_basin_summary = json.loads(
        bad_basin_summary_path.read_text(encoding="utf-8")
    )
    if not isinstance(bad_basin_summary, dict):
        raise ValueError("bad basin summary must be a JSON object")
    probe_rows = _extract_probe_rows_from_bad_basin_summary(bad_basin_summary)
    selected_probe_rows = select_perturbation_cases(
        probe_rows,
        max_cases=cfg.selection.max_cases,
        target_model_alias=cfg.selection.target_model_alias,
        slots=cfg.selection.slots,
        require_positive_advantage=cfg.selection.require_positive_advantage,
    )
    cohort_rows = _read_jsonl(cohort_jsonl_path)
    selected_cases = _prepare_selected_cases(
        selection_rows=selected_probe_rows,
        cohort_rows=cohort_rows,
    )
    _write_jsonl(run_dir / "selected_cases.jsonl", selected_cases)

    execution = _run_selected_perturbation_cases(
        cfg=cfg,
        config_dir=resolved_config_path.parent,
        selected_cases=selected_cases,
    )
    per_coord_rows = list(execution["per_coord_rows"])
    summary_rows = list(execution["summary_rows"])
    _write_jsonl(run_dir / "per_coord_scores.jsonl", per_coord_rows)
    _write_jsonl(run_dir / "summary_rows.jsonl", summary_rows)

    results_summary = summarize_perturbation_case_rows(summary_rows)
    summary = {
        "run_name": cfg.run.name,
        "selection": {
            "target_model_alias": cfg.selection.target_model_alias,
            "slots": list(cfg.selection.slots),
            "require_positive_advantage": cfg.selection.require_positive_advantage,
            "num_probe_rows": len(probe_rows),
            "num_selected_cases": len(selected_cases),
        },
        "results": {
            **results_summary,
            "num_per_coord_rows": len(per_coord_rows),
        },
    }
    _write_json(run_dir / "summary.json", summary)
    return {"run_dir": str(run_dir), "summary": summary}
