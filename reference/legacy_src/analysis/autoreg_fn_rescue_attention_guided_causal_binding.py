"""Phase-3 attention-guided causal binding analyses for FN-rescue rows."""

from __future__ import annotations

import json
import heapq
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from src.analysis.autoreg_fn_rescue_desc_x1_phase2 import (
    PHASE2_SUCCESS_FIELD,
    _box_tuple_from_any,
    _build_intervention_processor_inputs,
    _iter_jsonl,
    _norm1000_bbox_to_pixel_box,
    _optional_bool,
    _required_str,
    _summarize_intervention_execution,
    _write_json,
    _write_jsonl,
)


PHASE3_STAGES = ("target_mask", "competitor_source", "sink_triage", "case_linked", "report")
CASE_LINKED_INPUT_SUBDIRS = ("target_mask", "competitor_source")
SINK_TRIAGE_MAX_CANDIDATES = 5000


@dataclass(frozen=True)
class Phase3Paths:
    artifact_root: Path
    phase2_root: Path
    fn_rescue_root: Path


@dataclass(frozen=True)
class Phase3ExecutionConfig:
    fn_rescue_config_path: Path
    target_mask_max_cases: int
    competitor_source_max_cases: int


@dataclass(frozen=True)
class Phase3CaseLinkedConfig:
    max_attention_rows: int | None
    roles: tuple[str, ...]
    aggregation_scopes: tuple[str, ...]
    region_kinds: tuple[str, ...]


@dataclass(frozen=True)
class Phase3Config:
    paths: Phase3Paths
    evidence_scope: str
    execution: Phase3ExecutionConfig
    case_linked: Phase3CaseLinkedConfig


def load_phase3_config(path: Path) -> Phase3Config:
    config_dir = path.expanduser().resolve(strict=False).parent
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("phase3 config must be a mapping")
    paths_raw = _required_mapping(raw, "paths")
    execution_raw = _required_mapping(raw, "execution")
    case_linked_raw = _optional_mapping(raw, "case_linked")
    max_attention_rows_raw = case_linked_raw.get("max_attention_rows")
    return Phase3Config(
        paths=Phase3Paths(
            artifact_root=_required_path(paths_raw, "artifact_root"),
            phase2_root=_required_path(paths_raw, "phase2_root"),
            fn_rescue_root=_required_path(paths_raw, "fn_rescue_root"),
        ),
        evidence_scope=str(raw.get("evidence_scope") or "val200_fn_rescue_desc_x1_phase3_causal_binding"),
        execution=Phase3ExecutionConfig(
            fn_rescue_config_path=_required_path(
                execution_raw,
                "fn_rescue_config_path",
                base_dir=config_dir,
            ),
            target_mask_max_cases=_positive_int(
                execution_raw.get("target_mask_max_cases", 50),
                "execution.target_mask_max_cases",
            ),
            competitor_source_max_cases=_positive_int(
                execution_raw.get("competitor_source_max_cases", 16),
                "execution.competitor_source_max_cases",
            ),
        ),
        case_linked=Phase3CaseLinkedConfig(
            max_attention_rows=(
                None
                if max_attention_rows_raw is None
                else _nonnegative_int(max_attention_rows_raw, "case_linked.max_attention_rows")
            ),
            roles=_string_tuple(case_linked_raw.get("roles", ("pre_y1",)), field_name="case_linked.roles"),
            aggregation_scopes=_string_tuple(
                case_linked_raw.get("aggregation_scopes", ("union",)),
                field_name="case_linked.aggregation_scopes",
            ),
            region_kinds=_string_tuple(
                case_linked_raw.get(
                    "region_kinds",
                    (
                        "target_gt",
                        "same_desc_competitor_gt_object",
                        "same_desc_rollout_prediction",
                        "wrong_control_source_region",
                        "far_background",
                    ),
                ),
                field_name="case_linked.region_kinds",
            ),
        ),
    )


def build_phase3_dry_run_plan(config: Phase3Config, *, stages: Sequence[str]) -> dict[str, Any]:
    _validate_stages(stages)
    selected_interventions = config.paths.phase2_root / "intervention_plan" / "selected_interventions.jsonl"
    generation_rows = config.paths.fn_rescue_root / "rescue_generation_rows.jsonl"
    attention_rows = config.paths.fn_rescue_root / "rescue_attention_region_rows.jsonl"
    return {
        "artifact_root": str(config.paths.artifact_root),
        "phase2_root": str(config.paths.phase2_root),
        "fn_rescue_root": str(config.paths.fn_rescue_root),
        "evidence_scope": config.evidence_scope,
        "stages": list(stages),
        "selected_interventions_path": str(selected_interventions),
        "generation_rows_path": str(generation_rows),
        "attention_rows_path": str(attention_rows),
        "selected_interventions_exists": selected_interventions.exists(),
        "generation_rows_exists": generation_rows.exists(),
        "attention_rows_exists": attention_rows.exists(),
        "fn_rescue_config_path": str(config.execution.fn_rescue_config_path),
        "target_mask_max_cases": config.execution.target_mask_max_cases,
        "competitor_source_max_cases": config.execution.competitor_source_max_cases,
        "case_linked_max_attention_rows": config.case_linked.max_attention_rows,
        "case_linked_roles": list(config.case_linked.roles),
        "case_linked_region_kinds": list(config.case_linked.region_kinds),
    }


def materialize_target_mask(config: Phase3Config) -> dict[str, Any]:
    plan_rows = list(_iter_jsonl(_selected_interventions_path(config)))
    selected_rows = select_intervention_lane_rows(
        plan_rows,
        selection_kinds=("desc_x1_success_same_desc_competitor",),
        intervention_kinds_by_selection={
            "desc_x1_success_same_desc_competitor": ("no_op_control", "target_gt_mask"),
        },
        max_cases=config.execution.target_mask_max_cases,
    )
    return execute_intervention_rows(
        config,
        selected_rows=selected_rows,
        output_subdir="target_mask",
        execution_scope="target_mask_scale_up",
    )


def materialize_competitor_source(config: Phase3Config) -> dict[str, Any]:
    _require_target_mask_gate(config)
    plan_rows = list(_iter_jsonl(_selected_interventions_path(config)))
    selected_rows = select_intervention_lane_rows(
        plan_rows,
        selection_kinds=("desc_x1_success_same_desc_competitor", "desc_x1_wrong_control_failure"),
        intervention_kinds_by_selection={
            "desc_x1_success_same_desc_competitor": (
                "no_op_control",
                "same_desc_competitor_mask",
                "same_desc_rollout_prediction_mask",
            ),
            "desc_x1_wrong_control_failure": (
                "no_op_control",
                "wrong_control_source_region_mask",
            ),
        },
        max_cases=config.execution.competitor_source_max_cases,
    )
    return execute_intervention_rows(
        config,
        selected_rows=selected_rows,
        output_subdir="competitor_source",
        execution_scope="same_desc_competitor_and_wrong_source",
    )


def materialize_sink_triage(config: Phase3Config) -> dict[str, Any]:
    output_root = config.paths.artifact_root / "sink_triage"
    output_root.mkdir(parents=True, exist_ok=True)
    wanted_regions = {"far_background"}
    wanted_scopes = set(config.case_linked.aggregation_scopes)
    wanted_roles = set(config.case_linked.roles)
    heap: list[tuple[float, int, dict[str, Any]]] = []
    seen_candidates = 0
    processed_attention_rows = 0
    for row in _iter_jsonl(config.paths.fn_rescue_root / "rescue_attention_region_rows.jsonl"):
        if config.case_linked.max_attention_rows is not None and processed_attention_rows >= config.case_linked.max_attention_rows:
            break
        processed_attention_rows += 1
        if str(row.get("aggregation_scope") or "") not in wanted_scopes:
            continue
        if str(row.get("role") or "") not in wanted_roles:
            continue
        if str(row.get("region_kind") or "") not in wanted_regions:
            continue
        mass = float(row.get("attention_mass_normalized", row.get("attention_mass", 0.0)))
        candidate = {
            "case_id": row.get("case_id"),
            "rescue_tier": row.get("rescue_tier"),
            "role": row.get("role"),
            "layer": row.get("layer"),
            "head": row.get("head"),
            "aggregation_scope": row.get("aggregation_scope"),
            "region_kind": row.get("region_kind"),
            "attention_mass_normalized": mass,
            "sink_candidate_status": "candidate_only_not_intervened",
            "intervention_kind": "far_background_candidate_only",
            "causal_status": "not_intervened_candidate_triage",
        }
        seen_candidates += 1
        item = (mass, seen_candidates, candidate)
        if len(heap) < SINK_TRIAGE_MAX_CANDIDATES:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
    rows = [item[2] for item in heap]
    rows.sort(
        key=lambda row: (
            -float(row.get("attention_mass_normalized") or 0.0),
            str(row.get("case_id") or ""),
            str(row.get("rescue_tier") or ""),
            int(row.get("layer") or -1),
            int(row.get("head") or -1),
        )
    )
    _write_jsonl(output_root / "sink_candidate_rows.jsonl", rows)
    summary = {
        "artifact_root": str(config.paths.artifact_root),
        "fn_rescue_root": str(config.paths.fn_rescue_root),
        "evidence_scope": config.evidence_scope,
        "stage": "sink_triage",
        "row_count": len(rows),
        "max_candidates": SINK_TRIAGE_MAX_CANDIDATES,
        "seen_candidates": seen_candidates,
        "case_count": len({(_required_str(row, "case_id"), _required_str(row, "rescue_tier")) for row in rows}),
        "processed_attention_rows": processed_attention_rows,
        "roles": list(config.case_linked.roles),
        "aggregation_scopes": list(config.case_linked.aggregation_scopes),
        "region_kinds": sorted(wanted_regions),
        "causal_status": "candidate_only_not_intervened",
        "interpretation_boundary": (
            "sink triage only ranks far-background attention candidates; it does not mask, suppress, or train on background tokens"
        ),
    }
    _write_json(output_root / "summary.json", summary)
    _write_lane_report(config, "sink_triage", "Sink Candidate Triage", summary)
    return summary


def select_intervention_lane_rows(
    plan_rows: Sequence[Mapping[str, Any]],
    *,
    selection_kinds: Sequence[str],
    intervention_kinds_by_selection: Mapping[str, Sequence[str]],
    max_cases: int,
) -> list[dict[str, Any]]:
    wanted_selection_kinds = tuple(str(kind) for kind in selection_kinds)
    wanted_selection_set = set(wanted_selection_kinds)
    grouped: dict[tuple[str, str, str], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in plan_rows:
        selection_kind = str(row.get("selection_kind"))
        if selection_kind not in wanted_selection_set:
            continue
        wanted_interventions = set(intervention_kinds_by_selection.get(selection_kind, ()))
        intervention_kind = str(row.get("intervention_kind"))
        if intervention_kind not in wanted_interventions:
            continue
        key = (
            _required_str(row, "case_id"),
            _required_str(row, "rescue_tier"),
            selection_kind,
        )
        grouped[key][intervention_kind].append(dict(row))

    selected: list[dict[str, Any]] = []
    for selection_kind in wanted_selection_kinds:
        selected_cases_for_kind = 0
        wanted_interventions = tuple(intervention_kinds_by_selection.get(selection_kind, ()))
        for key, rows_by_kind in grouped.items():
            if key[2] != selection_kind:
                continue
            if "no_op_control" not in rows_by_kind:
                continue
            if not any(kind != "no_op_control" and rows_by_kind.get(kind) for kind in wanted_interventions):
                continue
            if selected_cases_for_kind >= max_cases:
                break
            selected_cases_for_kind += 1
            selected.append(rows_by_kind["no_op_control"][0])
            for kind in wanted_interventions:
                if kind == "no_op_control":
                    continue
                selected.extend(rows_by_kind.get(kind, []))
    return selected


def execute_intervention_rows(
    config: Phase3Config,
    *,
    selected_rows: Sequence[Mapping[str, Any]],
    output_subdir: str,
    execution_scope: str,
) -> dict[str, Any]:
    from src.analysis.autoreg_fn_rescue_continuation import (
        _build_fn_rescue_messages,
        _config_sha256,
        _default_fn_rescue_model_handle_loader,
        _fn_rescue_runtime_spec,
        _generation_kwargs_artifact_summary,
        _model_attn_implementation,
        _resolve_im_end_token_id,
        _resolve_tokenizer,
        decode_rescue_tail,
        load_fn_rescue_config,
        score_rescue_box,
    )

    fn_config = load_fn_rescue_config(config.execution.fn_rescue_config_path)
    model_handle = _default_fn_rescue_model_handle_loader(fn_config)
    tokenizer = _resolve_tokenizer(model_handle)
    runtime_spec = _fn_rescue_runtime_spec(fn_config)
    selected_attn = _model_attn_implementation(getattr(model_handle, "model", None))
    generation_by_key = {
        (_required_str(row, "case_id"), _required_str(row, "rescue_tier")): row
        for row in _iter_jsonl(config.paths.fn_rescue_root / "rescue_generation_rows.jsonl")
    }
    output_rows: list[dict[str, Any]] = []
    missing_generation_rows = 0
    for plan_row in selected_rows:
        key = (_required_str(plan_row, "case_id"), _required_str(plan_row, "rescue_tier"))
        generation_row = generation_by_key.get(key)
        if generation_row is None:
            missing_generation_rows += 1
            continue
        assistant_prefix_text = _required_str(generation_row, "assistant_prefix_text")
        assistant_prefix_token_ids = tuple(int(token_id) for token_id in generation_row.get("assistant_prefix_token_ids", []))
        if not assistant_prefix_token_ids:
            assistant_prefix_token_ids = tuple(
                int(token_id)
                for token_id in tokenizer.encode(assistant_prefix_text, add_special_tokens=False)
            )
        image_path = Path(_required_str(generation_row, "image_path"))
        messages = _build_fn_rescue_messages(
            runtime_spec=runtime_spec,
            image_path=image_path,
            assistant_prefix_text=assistant_prefix_text,
        )
        bbox_xyxy = plan_row.get("bbox_xyxy")
        mask_metadata = _intervention_mask_metadata(
            image_path=image_path,
            bbox_xyxy=bbox_xyxy,
            intervention_kind=str(plan_row.get("intervention_kind")),
        )
        if mask_metadata["intervention_error"] is not None:
            output_rows.append(
                _skipped_intervention_row(
                    generation_row,
                    plan_row,
                    mask_metadata=mask_metadata,
                    execution_scope=execution_scope,
                )
            )
            continue
        processor_inputs = _build_intervention_processor_inputs(
            model_handle=model_handle,
            messages=messages,
            image_path=image_path,
            bbox_xyxy=bbox_xyxy if plan_row.get("intervention_kind") != "no_op_control" else None,
        )
        tier = _required_str(generation_row, "rescue_tier")
        decode_result = decode_rescue_tail(
            model_handle=model_handle,
            processor_inputs=processor_inputs,
            tier=tier,
            assistant_prefix_token_ids=assistant_prefix_token_ids,
        )
        target_box = _box_tuple_from_any(generation_row.get("target_bbox_xyxy"))
        score = score_rescue_box(
            generated_box=decode_result.parsed_generation.generated_box_xyxy,
            target_box=target_box,
            same_desc_rollout_boxes=(),
            duplicate_iou_threshold=fn_config.selection.duplicate_iou_threshold,
        )
        eos_token_id = _resolve_im_end_token_id(tokenizer)
        baseline_iou = generation_row.get("target_iou")
        baseline_success = _optional_bool(generation_row.get(PHASE2_SUCCESS_FIELD))
        output_rows.append(
            {
                "case_id": key[0],
                "rescue_tier": tier,
                "source_line_idx": generation_row.get("source_line_idx"),
                "target_desc": generation_row.get("target_desc"),
                "target_gt_idx": generation_row.get("target_gt_idx"),
                "selection_kind": plan_row.get("selection_kind"),
                "intervention_kind": plan_row.get("intervention_kind"),
                "region_kind": plan_row.get("region_kind"),
                "region_instance_id": plan_row.get("region_instance_id"),
                "bbox_xyxy": bbox_xyxy,
                **mask_metadata,
                "prefix_quality": generation_row.get("prefix_quality"),
                "binding_bucket": generation_row.get("binding_bucket"),
                "depth_bucket": generation_row.get("depth_bucket"),
                "object_count_bucket": generation_row.get("object_count_bucket"),
                "target_bbox_xyxy": list(target_box),
                "baseline_generated_box_xyxy": generation_row.get("generated_box_xyxy"),
                "baseline_generated_tail_text": generation_row.get("generated_tail_text"),
                "baseline_target_iou": baseline_iou,
                "baseline_primary_rescue_success": baseline_success,
                "generated_token_ids": list(decode_result.generated_tail_ids),
                "generated_tail_text": decode_result.generated_tail_text,
                "generated_coord_tokens": list(decode_result.parsed_generation.generated_coord_tokens),
                "generated_box_xyxy": (
                    None
                    if decode_result.parsed_generation.generated_box_xyxy is None
                    else list(decode_result.parsed_generation.generated_box_xyxy)
                ),
                "parse_status": "ok" if decode_result.parsed_generation.valid_parse else "invalid_parse",
                "parse_errors": list(decode_result.parsed_generation.parse_errors),
                "stop_reason": (
                    "eos_token"
                    if decode_result.generated_tail_ids
                    and eos_token_id is not None
                    and int(decode_result.generated_tail_ids[-1]) == int(eos_token_id)
                    else "max_new_tokens"
                ),
                "target_iou": float(score.target_iou),
                "target_iou_delta": None if baseline_iou is None else float(score.target_iou) - float(baseline_iou),
                "success_iou50": bool(score.success_iou50),
                "primary_rescue_success": bool(score.primary_rescue_success),
                "primary_rescue_success_changed": (
                    None
                    if baseline_success is None
                    else bool(score.primary_rescue_success) != baseline_success
                ),
                "exact_tail_match_baseline": decode_result.generated_tail_text == generation_row.get("generated_tail_text"),
                "generation_kwargs": _generation_kwargs_artifact_summary(decode_result.generation_kwargs),
                "image_path": str(image_path),
                "config_sha256": _config_sha256(fn_config),
                "attn_implementation_selected": selected_attn,
                "causal_status": "executed",
                "execution_scope": execution_scope,
            }
        )
    output_root = config.paths.artifact_root / output_subdir
    output_root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_root / "intervention_rows.jsonl", output_rows)
    summary = _summarize_intervention_execution(
        config=_phase2_summary_adapter(config, execution_scope=execution_scope),
        rows=output_rows,
        missing_generation_rows=missing_generation_rows,
        fn_rescue_config_path=config.execution.fn_rescue_config_path,
    )
    summary.update(
        {
            "stage": output_subdir,
            "execution_scope": execution_scope,
            "selected_plan_rows": len(selected_rows),
            "output_dir": str(output_root),
            "denominator_chain": _denominator_chain(
                selected_rows=selected_rows,
                output_rows=output_rows,
                missing_generation_rows=missing_generation_rows,
            ),
            "strata_summaries": _strata_summaries(output_rows),
            "no_op_parity_definition": "tail_text_and_parsed_box_metric_level_not_token_id_level",
        }
    )
    _write_json(output_root / "summary.json", summary)
    _write_lane_report(config, output_subdir, output_subdir.replace("_", " ").title(), summary)
    return summary


def materialize_case_linked_table(config: Phase3Config) -> dict[str, Any]:
    intervention_rows = _load_intervention_output_rows(config)
    if not intervention_rows:
        raise ValueError("case_linked requires materialized target_mask and competitor_source intervention rows")
    attention_by_key, attention_summary = _case_attention_summary(config, intervention_rows)
    no_op_by_key = _paired_no_op_index(intervention_rows)
    output_rows: list[dict[str, Any]] = []
    for row in intervention_rows:
        key = (_required_str(row, "case_id"), _required_str(row, "rescue_tier"))
        attention = attention_by_key.get(key, {})
        target_mass = attention.get("target_gt")
        competitor_mass = _first_non_none(
            attention.get("same_desc_competitor_gt_object"),
            attention.get("same_desc_rollout_prediction"),
        )
        paired_no_op = no_op_by_key.get(key)
        paired_no_op_valid = _paired_no_op_valid(paired_no_op)
        is_no_op = row.get("intervention_kind") == "no_op_control"
        valid_paired_row = bool(is_no_op or paired_no_op_valid)
        output = dict(row)
        output.update(
            {
                "intervention_generated_box_xyxy": row.get("generated_box_xyxy"),
                "primary_success_changed": row.get("primary_rescue_success_changed"),
                "paired_no_op_valid": paired_no_op_valid,
                "valid_paired_row": valid_paired_row,
                "paired_no_op_case_id": None if paired_no_op is None else paired_no_op.get("case_id"),
                "paired_no_op_rescue_tier": None if paired_no_op is None else paired_no_op.get("rescue_tier"),
                "target_attention_mass": target_mass,
                "competitor_attention_mass": competitor_mass,
                "wrong_source_attention_mass": attention.get("wrong_control_source_region"),
                "far_background_attention_mass": attention.get("far_background"),
                "target_minus_competitor_attention": (
                    None
                    if target_mass is None or competitor_mass is None
                    else float(target_mass) - float(competitor_mass)
                ),
                "mechanism_bucket": _mechanism_bucket(row, valid_paired_row=valid_paired_row),
                "top_attention_heads_for_case": attention.get("__top_heads__", []),
                "x1_logit_lens_rank_when_available": None,
            }
        )
        output_rows.append(output)
    output_root = config.paths.artifact_root / "case_linked"
    output_root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_root / "case_mechanism_rows.jsonl", output_rows)
    bucket_counts = Counter(str(row["mechanism_bucket"]) for row in output_rows)
    summary = {
        "artifact_root": str(config.paths.artifact_root),
        "fn_rescue_root": str(config.paths.fn_rescue_root),
        "evidence_scope": config.evidence_scope,
        "stage": "case_linked",
        "row_count": len(output_rows),
        "case_count": len({(_required_str(row, "case_id"), _required_str(row, "rescue_tier")) for row in output_rows}),
        "mechanism_bucket_counts": dict(sorted(bucket_counts.items())),
        "attention_summary": attention_summary,
        "valid_paired_rows": sum(1 for row in output_rows if row.get("valid_paired_row") is True),
        "interpretation_boundary": (
            "case-linked buckets are gated by paired no-op replay at tail-text/parsed-box/metric level; "
            "top_attention_heads_for_case is diagnostic metadata, not head-level causal proof"
        ),
    }
    _write_json(output_root / "summary.json", summary)
    _write_lane_report(config, "case_linked", "Case-Linked Mechanism Table", summary)
    return summary


def _intervention_mask_metadata(
    *,
    image_path: Path,
    bbox_xyxy: Any,
    intervention_kind: str,
) -> dict[str, Any]:
    if intervention_kind == "no_op_control":
        return {
            "mask_applied": False,
            "mask_norm1000_box_xyxy": None,
            "mask_pixel_box_xyxy": None,
            "mask_area_pixels": 0,
            "image_size_wh": None,
            "intervention_error": None,
        }
    if bbox_xyxy is None:
        return {
            "mask_applied": False,
            "mask_norm1000_box_xyxy": None,
            "mask_pixel_box_xyxy": None,
            "mask_area_pixels": 0,
            "image_size_wh": None,
            "intervention_error": "missing_bbox_xyxy",
        }
    from PIL import Image

    with Image.open(image_path) as image:
        width, height = int(image.width), int(image.height)
    pixel_box = _norm1000_bbox_to_pixel_box(bbox_xyxy, width=width, height=height)
    if pixel_box is None:
        return {
            "mask_applied": False,
            "mask_norm1000_box_xyxy": list(bbox_xyxy) if isinstance(bbox_xyxy, Sequence) and not isinstance(bbox_xyxy, (str, bytes)) else None,
            "mask_pixel_box_xyxy": None,
            "mask_area_pixels": 0,
            "image_size_wh": [width, height],
            "intervention_error": "invalid_or_degenerate_bbox_xyxy",
        }
    x1, y1, x2, y2 = pixel_box
    return {
        "mask_applied": True,
        "mask_norm1000_box_xyxy": list(_box_tuple_from_any(bbox_xyxy)),
        "mask_pixel_box_xyxy": [x1, y1, x2, y2],
        "mask_area_pixels": max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1),
        "image_size_wh": [width, height],
        "intervention_error": None,
    }


def _skipped_intervention_row(
    generation_row: Mapping[str, Any],
    plan_row: Mapping[str, Any],
    *,
    mask_metadata: Mapping[str, Any],
    execution_scope: str,
) -> dict[str, Any]:
    baseline_iou = generation_row.get("target_iou")
    return {
        "case_id": generation_row.get("case_id"),
        "rescue_tier": generation_row.get("rescue_tier"),
        "source_line_idx": generation_row.get("source_line_idx"),
        "target_desc": generation_row.get("target_desc"),
        "target_gt_idx": generation_row.get("target_gt_idx"),
        "selection_kind": plan_row.get("selection_kind"),
        "intervention_kind": plan_row.get("intervention_kind"),
        "region_kind": plan_row.get("region_kind"),
        "region_instance_id": plan_row.get("region_instance_id"),
        "bbox_xyxy": plan_row.get("bbox_xyxy"),
        **dict(mask_metadata),
        "prefix_quality": generation_row.get("prefix_quality"),
        "binding_bucket": generation_row.get("binding_bucket"),
        "depth_bucket": generation_row.get("depth_bucket"),
        "object_count_bucket": generation_row.get("object_count_bucket"),
        "target_bbox_xyxy": generation_row.get("target_bbox_xyxy"),
        "baseline_generated_box_xyxy": generation_row.get("generated_box_xyxy"),
        "baseline_generated_tail_text": generation_row.get("generated_tail_text"),
        "baseline_target_iou": baseline_iou,
        "baseline_primary_rescue_success": _optional_bool(generation_row.get(PHASE2_SUCCESS_FIELD)),
        "generated_token_ids": [],
        "generated_tail_text": "",
        "generated_coord_tokens": [],
        "generated_box_xyxy": None,
        "parse_status": "skipped_invalid_intervention",
        "parse_errors": [str(mask_metadata["intervention_error"])],
        "stop_reason": "skipped_invalid_intervention",
        "target_iou": 0.0,
        "target_iou_delta": None,
        "success_iou50": False,
        "primary_rescue_success": False,
        "primary_rescue_success_changed": None,
        "exact_tail_match_baseline": False,
        "generation_kwargs": {},
        "image_path": generation_row.get("image_path"),
        "config_sha256": None,
        "attn_implementation_selected": None,
        "causal_status": "skipped_invalid_intervention",
        "execution_scope": execution_scope,
    }


def write_phase3_report(config: Phase3Config) -> Path:
    lines = [
        "# FN-Rescue Attention-Guided Causal Binding Phase 3",
        "",
        f"- Evidence scope: `{config.evidence_scope}`",
        f"- Artifact root: `{config.paths.artifact_root}`",
        f"- Phase-2 root: `{config.paths.phase2_root}`",
        f"- FN-rescue root: `{config.paths.fn_rescue_root}`",
        "",
        "## Interpretation Boundary",
        "",
        "This report separates no-op replay parity, masked-region causal evidence, and attention-linked diagnostic evidence. It does not claim broad background suppression is beneficial. No-op parity is checked at generated-tail text, parsed-box, IoU, and success levels; token-level parity is not claimed unless token IDs are explicitly persisted.",
        "",
    ]
    for subdir, title in (
        ("target_mask", "Target Mask Scale-Up"),
        ("competitor_source", "Competitor And Wrong-Source Intervention"),
        ("sink_triage", "Sink Candidate Triage"),
        ("case_linked", "Case-Linked Mechanism Table"),
    ):
        summary_path = config.paths.artifact_root / subdir / "summary.json"
        lines.extend([f"## {title}", ""])
        if not summary_path.exists():
            lines.extend(["Not materialized.", ""])
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        lines.append(f"- Rows: `{summary.get('row_count')}`")
        lines.append(f"- Cases: `{summary.get('case_count')}`")
        if "kind_summaries" in summary:
            lines.append("")
            lines.append("| Intervention | Rows | Mean IoU Delta | Success Changed | Exact Tail Match | Invalid Parse |")
            lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
            for kind, payload in sorted(summary.get("kind_summaries", {}).items()):
                lines.append(
                    "| {kind} | {rows} | {delta} | {changed} | {match} | {invalid} |".format(
                        kind=kind,
                        rows=payload.get("rows"),
                        delta=_fmt(payload.get("mean_target_iou_delta")),
                        changed=payload.get("primary_success_changed"),
                        match=payload.get("exact_tail_match_baseline"),
                        invalid=payload.get("invalid_parse"),
                    )
                )
        if "mechanism_bucket_counts" in summary:
            lines.append(f"- Mechanism buckets: `{summary.get('mechanism_bucket_counts')}`")
        if subdir == "competitor_source":
            lines.append(
                "- Wrong-control rows are interpreted separately: metric-neutral rows are not generation-neutral unless exact-tail parity is also true."
            )
        lines.append("")
    report_path = config.paths.artifact_root / "report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_phase3_manifest(config)
    _write_phase3_summary(config)
    return report_path


def _selected_interventions_path(config: Phase3Config) -> Path:
    return config.paths.phase2_root / "intervention_plan" / "selected_interventions.jsonl"


def _load_intervention_output_rows(config: Phase3Config) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for subdir in CASE_LINKED_INPUT_SUBDIRS:
        path = config.paths.artifact_root / subdir / "intervention_rows.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"case_linked requires {subdir} output: {path}")
        rows.extend(_iter_jsonl(path))
    return rows


def _case_attention_summary(
    config: Phase3Config,
    intervention_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    wanted_keys = {
        (_required_str(row, "case_id"), _required_str(row, "rescue_tier"))
        for row in intervention_rows
    }
    wanted_scopes = set(config.case_linked.aggregation_scopes)
    wanted_roles = set(config.case_linked.roles)
    wanted_regions = set(config.case_linked.region_kinds)
    totals: dict[tuple[str, str, str], float] = defaultdict(float)
    counts: dict[tuple[str, str, str], int] = defaultdict(int)
    head_totals: dict[tuple[str, str, str, str, int, int], float] = defaultdict(float)
    head_counts: dict[tuple[str, str, str, str, int, int], int] = defaultdict(int)
    processed_rows = 0
    joined_rows = 0
    for row in _iter_jsonl(config.paths.fn_rescue_root / "rescue_attention_region_rows.jsonl"):
        if config.case_linked.max_attention_rows is not None and processed_rows >= config.case_linked.max_attention_rows:
            break
        processed_rows += 1
        key = (_required_str(row, "case_id"), _required_str(row, "rescue_tier"))
        if key not in wanted_keys:
            continue
        if str(row.get("aggregation_scope") or "") not in wanted_scopes:
            continue
        if str(row.get("role") or "") not in wanted_roles:
            continue
        region_kind = str(row.get("region_kind") or "")
        if region_kind not in wanted_regions:
            continue
        stat_key = (key[0], key[1], region_kind)
        totals[stat_key] += float(row.get("attention_mass_normalized", row.get("attention_mass", 0.0)))
        counts[stat_key] += 1
        layer = int(row.get("layer") or -1)
        head = int(row.get("head") or -1)
        role = str(row.get("role") or "")
        scope = str(row.get("aggregation_scope") or "")
        head_key = (key[0], key[1], role, scope, layer, head)
        head_totals[head_key] += float(row.get("attention_mass_normalized", row.get("attention_mass", 0.0)))
        head_counts[head_key] += 1
        joined_rows += 1
    by_case: dict[tuple[str, str], dict[str, Any]] = defaultdict(dict)
    for (case_id, tier, region_kind), total in totals.items():
        by_case[(case_id, tier)][region_kind] = total / counts[(case_id, tier, region_kind)]
    heads_by_case: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for (case_id, tier, role, scope, layer, head), total in head_totals.items():
        heads_by_case[(case_id, tier)].append(
            {
                "role": role,
                "aggregation_scope": scope,
                "layer": layer,
                "head": head,
                "_mean_attention_mass": total / head_counts[(case_id, tier, role, scope, layer, head)],
            }
        )
    for key, heads in heads_by_case.items():
        heads.sort(key=lambda item: (-float(item["_mean_attention_mass"]), int(item["layer"]), int(item["head"])))
        by_case[key]["__top_heads__"] = [
            {k: v for k, v in item.items() if k != "_mean_attention_mass"} for item in heads[:5]
        ]
    return (
        dict(by_case),
        {
            "processed_attention_rows": processed_rows,
            "joined_attention_rows": joined_rows,
            "case_tier_count": len(wanted_keys),
            "roles": list(config.case_linked.roles),
            "aggregation_scopes": list(config.case_linked.aggregation_scopes),
        },
    )


def _mechanism_bucket(row: Mapping[str, Any], *, valid_paired_row: bool = True) -> str:
    if not valid_paired_row:
        return "invalid_or_uninterpretable"
    if row.get("parse_status") != "ok":
        return "invalid_or_uninterpretable"
    kind = row.get("intervention_kind")
    changed = row.get("primary_rescue_success_changed") is True
    delta = row.get("target_iou_delta")
    delta_value = None if delta is None else float(delta)
    if kind == "target_gt_mask" and (changed or (delta_value is not None and delta_value < -0.1)):
        return "target_dependent"
    if kind in {"same_desc_competitor_mask", "same_desc_rollout_prediction_mask"} and (
        changed or (delta_value is not None and delta_value > 0.1)
    ):
        return "competitor_dependent"
    if kind == "wrong_control_source_region_mask" and (
        changed or (delta_value is not None and abs(delta_value) > 0.1)
    ):
        return "wrong_source_dependent"
    if kind == "no_op_control":
        return "no_op_replay"
    return "robust_to_region_masks"


def _phase2_summary_adapter(config: Phase3Config, *, execution_scope: str) -> Any:
    class _Paths:
        artifact_root = config.paths.artifact_root
        fn_rescue_root = config.paths.fn_rescue_root

    class _Adapter:
        paths = _Paths()
        evidence_scope = config.evidence_scope

    return _Adapter()


def _first_non_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _require_target_mask_gate(config: Phase3Config) -> None:
    summary_path = config.paths.artifact_root / "target_mask" / "summary.json"
    if not summary_path.exists():
        raise ValueError(f"target_mask gate failed: missing {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    kind_summaries = summary.get("kind_summaries", {})
    if not isinstance(kind_summaries, Mapping):
        raise ValueError("target_mask gate failed: missing kind_summaries")
    no_op = kind_summaries.get("no_op_control", {})
    target_mask = kind_summaries.get("target_gt_mask", {})
    if not isinstance(no_op, Mapping) or not isinstance(target_mask, Mapping):
        raise ValueError("target_mask gate failed: missing no_op_control or target_gt_mask summary")
    no_op_rows = int(no_op.get("rows") or 0)
    no_op_exact = int(no_op.get("exact_tail_match_baseline") or 0)
    parity = 0.0 if no_op_rows == 0 else no_op_exact / no_op_rows
    target_rows = int(target_mask.get("rows") or 0)
    invalid_rate = 1.0 if target_rows == 0 else int(target_mask.get("invalid_parse") or 0) / target_rows
    mean_delta_raw = target_mask.get("mean_target_iou_delta")
    mean_delta = None if mean_delta_raw is None else float(mean_delta_raw)
    success_flips = int(target_mask.get("primary_success_changed") or 0)
    if parity < 0.95:
        raise ValueError(f"target_mask gate failed: no-op parity {parity:.3f} < 0.95")
    if invalid_rate > 0.05:
        raise ValueError(f"target_mask gate failed: invalid parse rate {invalid_rate:.3f} > 0.05")
    if not ((mean_delta is not None and mean_delta < -0.10) or success_flips >= 1):
        raise ValueError("target_mask gate failed: target mask did not show sufficient effect")


def _paired_no_op_index(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    index: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        if row.get("intervention_kind") != "no_op_control":
            continue
        key = (_required_str(row, "case_id"), _required_str(row, "rescue_tier"))
        index[key] = row
    return index


def _paired_no_op_valid(row: Mapping[str, Any] | None) -> bool:
    if row is None:
        return False
    if row.get("parse_status") != "ok":
        return False
    if row.get("exact_tail_match_baseline") is not True:
        return False
    delta = row.get("target_iou_delta")
    return delta is None or abs(float(delta)) <= 1e-12


def _denominator_chain(
    *,
    selected_rows: Sequence[Mapping[str, Any]],
    output_rows: Sequence[Mapping[str, Any]],
    missing_generation_rows: int,
) -> dict[str, int]:
    selected_cases = {
        (_required_str(row, "case_id"), _required_str(row, "rescue_tier"))
        for row in selected_rows
    }
    phase3_cases = {
        (_required_str(row, "case_id"), _required_str(row, "rescue_tier"))
        for row in output_rows
    }
    valid_paired_rows = sum(1 for row in output_rows if row.get("parse_status") == "ok")
    return {
        "selected_cases": len(selected_cases),
        "attempted_generation_rows": len(selected_rows),
        "reconstructable_rows": max(0, len(selected_rows) - int(missing_generation_rows)),
        "selected_generation_rows": len(output_rows),
        "phase3_cases": len(phase3_cases),
        "valid_paired_rows": valid_paired_rows,
    }


def _strata_summaries(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    fields = ("rescue_tier", "prefix_quality", "depth_bucket", "object_count_bucket", "binding_bucket")
    result: dict[str, dict[str, int]] = {}
    for field in fields:
        counter = Counter(str(row.get(field) or "unknown") for row in rows)
        result[field] = dict(sorted(counter.items()))
    return result


def _write_lane_report(config: Phase3Config, subdir: str, title: str, summary: Mapping[str, Any]) -> None:
    lines = [
        f"# {title}",
        "",
        f"- Evidence scope: `{config.evidence_scope}`",
        f"- Artifact root: `{config.paths.artifact_root}`",
        f"- Stage: `{summary.get('stage', subdir)}`",
        f"- Rows: `{summary.get('row_count')}`",
        f"- Cases: `{summary.get('case_count')}`",
        "",
    ]
    if "kind_summaries" in summary:
        lines.extend(
            [
                "| Intervention | Rows | Mean IoU Delta | Success Changed | Exact Tail Match | Invalid Parse |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for kind, payload in sorted(summary.get("kind_summaries", {}).items()):
            lines.append(
                "| {kind} | {rows} | {delta} | {changed} | {match} | {invalid} |".format(
                    kind=kind,
                    rows=payload.get("rows"),
                    delta=_fmt(payload.get("mean_target_iou_delta")),
                    changed=payload.get("primary_success_changed"),
                    match=payload.get("exact_tail_match_baseline"),
                    invalid=payload.get("invalid_parse"),
                )
            )
        lines.append("")
    if "mechanism_bucket_counts" in summary:
        lines.append(f"- Mechanism buckets: `{summary.get('mechanism_bucket_counts')}`")
    if "interpretation_boundary" in summary:
        lines.append(f"- Interpretation boundary: {summary.get('interpretation_boundary')}")
    output_path = config.paths.artifact_root / subdir / "report.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_phase3_manifest(config: Phase3Config) -> Path:
    manifest = {
        "artifact_root": str(config.paths.artifact_root),
        "phase2_root": str(config.paths.phase2_root),
        "fn_rescue_root": str(config.paths.fn_rescue_root),
        "evidence_scope": config.evidence_scope,
        "stages": list(PHASE3_STAGES),
        "outputs": {
            "target_mask": {
                "summary": "target_mask/summary.json",
                "rows": "target_mask/intervention_rows.jsonl",
                "report": "target_mask/report.md",
            },
            "competitor_source": {
                "summary": "competitor_source/summary.json",
                "rows": "competitor_source/intervention_rows.jsonl",
                "report": "competitor_source/report.md",
            },
            "sink_triage": {
                "summary": "sink_triage/summary.json",
                "rows": "sink_triage/sink_candidate_rows.jsonl",
                "report": "sink_triage/report.md",
            },
            "case_linked": {
                "summary": "case_linked/summary.json",
                "rows": "case_linked/case_mechanism_rows.jsonl",
                "report": "case_linked/report.md",
            },
            "root_report": "report.md",
            "root_summary": "summary.json",
            "manifest": "manifest.json",
        },
    }
    path = config.paths.artifact_root / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, manifest)
    return path


def _write_phase3_summary(config: Phase3Config) -> Path:
    stage_summaries: dict[str, Any] = {}
    for subdir in ("target_mask", "competitor_source", "sink_triage", "case_linked"):
        path = config.paths.artifact_root / subdir / "summary.json"
        if path.exists():
            stage_summaries[subdir] = json.loads(path.read_text(encoding="utf-8"))
    summary = {
        "artifact_root": str(config.paths.artifact_root),
        "phase2_root": str(config.paths.phase2_root),
        "fn_rescue_root": str(config.paths.fn_rescue_root),
        "evidence_scope": config.evidence_scope,
        "stages_materialized": sorted(stage_summaries),
        "stage_summaries": stage_summaries,
    }
    path = config.paths.artifact_root / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, summary)
    return path


def _validate_stages(stages: Sequence[str]) -> None:
    if not stages:
        raise ValueError("stages must not be empty")
    unknown = sorted(set(stages) - set(PHASE3_STAGES))
    if unknown:
        raise ValueError(f"unknown phase3 stage(s): {', '.join(unknown)}")


def _iter_existing_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return ()
    return _iter_jsonl(path)


def _required_mapping(row: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    value = row.get(field)
    if not isinstance(value, Mapping):
        raise ValueError(f"missing mapping field: {field}")
    return value


def _optional_mapping(row: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    value = row.get(field, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _required_path(row: Mapping[str, Any], field: str, *, base_dir: Path | None = None) -> Path:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"missing path field: {field}")
    path = Path(value).expanduser()
    if base_dir is not None and not path.is_absolute():
        path = base_dir / path
    return path.resolve(strict=False)


def _string_tuple(value: Any, *, field_name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be a string or sequence")
    result = tuple(str(item) for item in value)
    if not result:
        raise ValueError(f"{field_name} must not be empty")
    return result


def _positive_int(value: Any, field_name: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{field_name} must be positive")
    return result


def _nonnegative_int(value: Any, field_name: str) -> int:
    result = int(value)
    if result < 0:
        raise ValueError(f"{field_name} must be nonnegative")
    return result


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)
