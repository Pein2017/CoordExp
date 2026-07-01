"""Phase-2 FN-rescue desc->x1 evidence-routing analyses.

This module intentionally starts with CPU-only mining over the completed
FN-rescue continuation artifacts.  Later GPU probe/intervention stages should
stay analysis-only and must not edit upstream model files.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


PHASE2_STAGES = ("attention_mining", "intervention_plan", "intervention_smoke", "report")
PHASE2_REGION_KINDS = (
    "target_gt",
    "same_desc_competitor_gt_object",
    "same_desc_rollout_prediction",
    "wrong_control_source_region",
    "context_ring",
    "far_background",
)
PHASE2_PRIMARY_REGION_KINDS = frozenset(
    {
        "target_gt",
        "same_desc_competitor_gt_object",
        "same_desc_rollout_prediction",
        "wrong_control_source_region",
        "context_ring",
        "far_background",
    }
)
PHASE2_SUCCESS_FIELD = "primary_rescue_success"


@dataclass(frozen=True)
class Phase2Paths:
    artifact_root: Path
    fn_rescue_root: Path


@dataclass(frozen=True)
class Phase2MiningConfig:
    max_attention_rows: int | None
    ranking_limit: int
    min_rows_per_head: int
    aggregation_scopes: tuple[str, ...]
    region_kinds: tuple[str, ...]


@dataclass(frozen=True)
class Phase2InterventionConfig:
    max_cases: int | None
    fn_rescue_config_path: Path | None
    execute_max_cases: int
    execute_intervention_kinds: tuple[str, ...]


@dataclass(frozen=True)
class Phase2Config:
    paths: Phase2Paths
    evidence_scope: str
    mining: Phase2MiningConfig
    intervention_plan: Phase2InterventionConfig


@dataclass
class _CellStats:
    count: int = 0
    total: float = 0.0

    def add(self, value: float) -> None:
        self.count += 1
        self.total += float(value)

    @property
    def mean(self) -> float | None:
        if self.count <= 0:
            return None
        return self.total / self.count


@dataclass
class _OutcomeStats:
    success_count: int = 0
    success_total: float = 0.0
    failure_count: int = 0
    failure_total: float = 0.0

    def add(self, value: float, *, success: bool | None) -> None:
        if success is True:
            self.success_count += 1
            self.success_total += float(value)
        elif success is False:
            self.failure_count += 1
            self.failure_total += float(value)

    @property
    def success_mean(self) -> float | None:
        if self.success_count <= 0:
            return None
        return self.success_total / self.success_count

    @property
    def failure_mean(self) -> float | None:
        if self.failure_count <= 0:
            return None
        return self.failure_total / self.failure_count

    @property
    def success_minus_failure(self) -> float | None:
        success_mean = self.success_mean
        failure_mean = self.failure_mean
        if success_mean is None or failure_mean is None:
            return None
        return success_mean - failure_mean


def load_phase2_config(path: Path) -> Phase2Config:
    config_dir = path.expanduser().resolve(strict=False).parent
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("phase2 config must be a mapping")
    paths_raw = _required_mapping(raw, "paths")
    mining_raw = _optional_mapping(raw, "mining")
    intervention_raw = _optional_mapping(raw, "intervention_plan")
    artifact_root = _required_path(paths_raw, "artifact_root")
    fn_rescue_root = _required_path(paths_raw, "fn_rescue_root")
    evidence_scope = str(raw.get("evidence_scope") or "val200_fn_rescue_desc_x1_phase2")
    max_attention_rows_raw = mining_raw.get("max_attention_rows")
    max_attention_rows = (
        None if max_attention_rows_raw is None else _nonnegative_int(max_attention_rows_raw, "max_attention_rows")
    )
    ranking_limit = _positive_int(mining_raw.get("ranking_limit", 25), "ranking_limit")
    min_rows_per_head = _nonnegative_int(mining_raw.get("min_rows_per_head", 1), "min_rows_per_head")
    aggregation_scopes = _string_tuple(
        mining_raw.get("aggregation_scopes", ("union",)),
        field_name="aggregation_scopes",
    )
    region_kinds = _string_tuple(
        mining_raw.get("region_kinds", PHASE2_REGION_KINDS),
        field_name="region_kinds",
    )
    max_intervention_cases_raw = intervention_raw.get("max_cases")
    max_intervention_cases = (
        None
        if max_intervention_cases_raw is None
        else _nonnegative_int(max_intervention_cases_raw, "intervention_plan.max_cases")
    )
    fn_rescue_config_path = _optional_path_value(
        intervention_raw.get("fn_rescue_config_path"),
        base_dir=config_dir,
    )
    execute_max_cases = _positive_int(
        intervention_raw.get("execute_max_cases", 4),
        "intervention_plan.execute_max_cases",
    )
    execute_intervention_kinds = _string_tuple(
        intervention_raw.get("execute_intervention_kinds", ("no_op_control", "target_gt_mask")),
        field_name="intervention_plan.execute_intervention_kinds",
    )
    return Phase2Config(
        paths=Phase2Paths(
            artifact_root=artifact_root,
            fn_rescue_root=fn_rescue_root,
        ),
        evidence_scope=evidence_scope,
        mining=Phase2MiningConfig(
            max_attention_rows=max_attention_rows,
            ranking_limit=ranking_limit,
            min_rows_per_head=min_rows_per_head,
            aggregation_scopes=aggregation_scopes,
            region_kinds=region_kinds,
        ),
        intervention_plan=Phase2InterventionConfig(
            max_cases=max_intervention_cases,
            fn_rescue_config_path=fn_rescue_config_path,
            execute_max_cases=execute_max_cases,
            execute_intervention_kinds=execute_intervention_kinds,
        ),
    )


def build_phase2_dry_run_plan(config: Phase2Config, *, stages: Sequence[str]) -> dict[str, Any]:
    _validate_stages(stages)
    attention_path = config.paths.fn_rescue_root / "rescue_attention_region_rows.jsonl"
    generation_path = config.paths.fn_rescue_root / "rescue_generation_rows.jsonl"
    candidate_region_path = config.paths.fn_rescue_root / "rescue_candidate_region_rows.jsonl"
    return {
        "artifact_root": str(config.paths.artifact_root),
        "fn_rescue_root": str(config.paths.fn_rescue_root),
        "evidence_scope": config.evidence_scope,
        "stages": list(stages),
        "attention_rows_path": str(attention_path),
        "generation_rows_path": str(generation_path),
        "candidate_region_rows_path": str(candidate_region_path),
        "attention_rows_exists": attention_path.exists(),
        "generation_rows_exists": generation_path.exists(),
        "candidate_region_rows_exists": candidate_region_path.exists(),
        "max_attention_rows": config.mining.max_attention_rows,
        "max_intervention_cases": config.intervention_plan.max_cases,
        "fn_rescue_config_path": (
            None
            if config.intervention_plan.fn_rescue_config_path is None
            else str(config.intervention_plan.fn_rescue_config_path)
        ),
        "execute_max_cases": config.intervention_plan.execute_max_cases,
        "execute_intervention_kinds": list(config.intervention_plan.execute_intervention_kinds),
        "ranking_limit": config.mining.ranking_limit,
        "aggregation_scopes": list(config.mining.aggregation_scopes),
        "region_kinds": list(config.mining.region_kinds),
    }


def materialize_attention_mining(config: Phase2Config) -> dict[str, Any]:
    generation_path = config.paths.fn_rescue_root / "rescue_generation_rows.jsonl"
    attention_path = config.paths.fn_rescue_root / "rescue_attention_region_rows.jsonl"
    generation_outcomes = load_generation_outcomes(generation_path)
    mining_root = config.paths.artifact_root / "attention_mining"
    mining_root.mkdir(parents=True, exist_ok=True)
    summary_rows, ranking_payload, summary = mine_attention_rows(
        attention_path=attention_path,
        generation_outcomes=generation_outcomes,
        max_attention_rows=config.mining.max_attention_rows,
        aggregation_scopes=config.mining.aggregation_scopes,
        region_kinds=config.mining.region_kinds,
        ranking_limit=config.mining.ranking_limit,
        min_rows_per_head=config.mining.min_rows_per_head,
    )
    summary.update(
        {
            "artifact_root": str(config.paths.artifact_root),
            "fn_rescue_root": str(config.paths.fn_rescue_root),
            "evidence_scope": config.evidence_scope,
            "stage": "attention_mining",
            "generation_outcome_count": len(generation_outcomes),
            "max_attention_rows": config.mining.max_attention_rows,
        }
    )
    _write_jsonl(mining_root / "head_layer_region_summary.jsonl", summary_rows)
    _write_json(mining_root / "head_layer_rankings.json", ranking_payload)
    _write_json(mining_root / "summary.json", summary)
    report_path = write_phase2_report(config.paths.artifact_root)
    summary["report_path"] = str(report_path)
    _write_json(mining_root / "summary.json", summary)
    return summary


def materialize_intervention_plan(config: Phase2Config) -> dict[str, Any]:
    generation_path = config.paths.fn_rescue_root / "rescue_generation_rows.jsonl"
    candidate_region_path = config.paths.fn_rescue_root / "rescue_candidate_region_rows.jsonl"
    candidate_index, candidate_count = _load_candidate_region_index(candidate_region_path)
    plan_root = config.paths.artifact_root / "intervention_plan"
    plan_root.mkdir(parents=True, exist_ok=True)
    plan_rows: list[dict[str, Any]] = []
    selected_generation_rows = 0
    skipped_generation_rows = 0
    missing_regions: Counter[tuple[str, str]] = Counter()
    selected_cases: set[str] = set()

    for generation_row in _iter_jsonl(generation_path):
        plan_kind = _intervention_selection_kind(generation_row)
        if plan_kind is None:
            skipped_generation_rows += 1
            continue
        case_id = _required_str(generation_row, "case_id")
        if config.intervention_plan.max_cases is not None and case_id not in selected_cases:
            if len(selected_cases) >= config.intervention_plan.max_cases:
                skipped_generation_rows += 1
                continue
        tier = _required_str(generation_row, "rescue_tier")
        selected_generation_rows += 1
        selected_cases.add(case_id)
        candidates_for_tier = candidate_index.get((case_id, tier), {})
        plan_rows.append(
            _intervention_plan_row(
                generation_row,
                intervention_kind="no_op_control",
                selection_kind=plan_kind,
                region_row=None,
            )
        )
        for region_kind, intervention_kind in _planned_region_interventions(plan_kind):
            region_rows = candidates_for_tier.get(region_kind, [])
            if not region_rows:
                missing_regions[(plan_kind, region_kind)] += 1
                continue
            for region_row in region_rows:
                plan_rows.append(
                    _intervention_plan_row(
                        generation_row,
                        intervention_kind=intervention_kind,
                        selection_kind=plan_kind,
                        region_row=region_row,
                    )
                )

    intervention_kind_counts = Counter(str(row["intervention_kind"]) for row in plan_rows)
    selection_kind_counts = Counter(str(row["selection_kind"]) for row in plan_rows)
    summary = {
        "artifact_root": str(config.paths.artifact_root),
        "fn_rescue_root": str(config.paths.fn_rescue_root),
        "evidence_scope": config.evidence_scope,
        "stage": "intervention_plan",
        "generation_rows_path": str(generation_path),
        "candidate_region_rows_path": str(candidate_region_path),
        "candidate_region_rows": candidate_count,
        "selected_generation_rows": selected_generation_rows,
        "skipped_generation_rows": skipped_generation_rows,
        "case_count": len(selected_cases),
        "planned_intervention_rows": len(plan_rows),
        "causal_status": "planned_not_executed",
        "execution_required": "gpu_redecode_with_noop_parity_before_masked_intervention",
        "max_cases": config.intervention_plan.max_cases,
        "intervention_kind_counts": dict(sorted(intervention_kind_counts.items())),
        "selection_kind_counts": dict(sorted(selection_kind_counts.items())),
        "missing_region_counts": {
            f"{selection_kind}:{region_kind}": count
            for (selection_kind, region_kind), count in sorted(missing_regions.items())
        },
    }
    _write_jsonl(plan_root / "selected_interventions.jsonl", plan_rows)
    _write_json(plan_root / "summary.json", summary)
    return summary


def materialize_intervention_smoke(config: Phase2Config) -> dict[str, Any]:
    fn_rescue_config_path = config.intervention_plan.fn_rescue_config_path
    if fn_rescue_config_path is None:
        raise ValueError("intervention_smoke requires intervention_plan.fn_rescue_config_path")
    plan_path = config.paths.artifact_root / "intervention_plan" / "selected_interventions.jsonl"
    if not plan_path.exists():
        materialize_intervention_plan(config)

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

    fn_config = load_fn_rescue_config(fn_rescue_config_path)
    model_handle = _default_fn_rescue_model_handle_loader(fn_config)
    tokenizer = _resolve_tokenizer(model_handle)
    runtime_spec = _fn_rescue_runtime_spec(fn_config)
    selected_attn = _model_attn_implementation(getattr(model_handle, "model", None))
    generation_by_key = {
        (_required_str(row, "case_id"), _required_str(row, "rescue_tier")): row
        for row in _iter_jsonl(config.paths.fn_rescue_root / "rescue_generation_rows.jsonl")
    }
    plan_rows = list(_iter_jsonl(plan_path))
    execution_plan = _select_intervention_smoke_rows(
        plan_rows,
        intervention_kinds=config.intervention_plan.execute_intervention_kinds,
        max_cases=config.intervention_plan.execute_max_cases,
    )
    output_rows: list[dict[str, Any]] = []
    missing_generation_rows = 0
    for plan_row in execution_plan:
        key = (_required_str(plan_row, "case_id"), _required_str(plan_row, "rescue_tier"))
        generation_row = generation_by_key.get(key)
        if generation_row is None:
            missing_generation_rows += 1
            continue
        assistant_prefix_text = _required_str(generation_row, "assistant_prefix_text")
        assistant_prefix_token_ids = tuple(
            int(token_id)
            for token_id in generation_row.get("assistant_prefix_token_ids", [])
        )
        if not assistant_prefix_token_ids:
            assistant_prefix_token_ids = tuple(
                int(token_id)
                for token_id in tokenizer.encode(
                    assistant_prefix_text,
                    add_special_tokens=False,
                )
            )
        image_path = Path(_required_str(generation_row, "image_path"))
        messages = _build_fn_rescue_messages(
            runtime_spec=runtime_spec,
            image_path=image_path,
            assistant_prefix_text=assistant_prefix_text,
        )
        bbox_xyxy = plan_row.get("bbox_xyxy")
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
        stop_reason = (
            "eos_token"
            if decode_result.generated_tail_ids
            and eos_token_id is not None
            and int(decode_result.generated_tail_ids[-1]) == int(eos_token_id)
            else "max_new_tokens"
        )
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
                "target_bbox_xyxy": list(target_box),
                "baseline_generated_box_xyxy": generation_row.get("generated_box_xyxy"),
                "baseline_generated_tail_text": generation_row.get("generated_tail_text"),
                "baseline_target_iou": baseline_iou,
                "baseline_primary_rescue_success": baseline_success,
                "generated_token_ids": list(decode_result.generated_tail_ids),
                "generated_tail_text": decode_result.generated_tail_text,
                "generated_coord_tokens": list(
                    decode_result.parsed_generation.generated_coord_tokens
                ),
                "generated_box_xyxy": (
                    None
                    if decode_result.parsed_generation.generated_box_xyxy is None
                    else list(decode_result.parsed_generation.generated_box_xyxy)
                ),
                "parse_status": (
                    "ok"
                    if decode_result.parsed_generation.valid_parse
                    else "invalid_parse"
                ),
                "parse_errors": list(decode_result.parsed_generation.parse_errors),
                "stop_reason": stop_reason,
                "target_iou": float(score.target_iou),
                "target_iou_delta": (
                    None
                    if baseline_iou is None
                    else float(score.target_iou) - float(baseline_iou)
                ),
                "success_iou50": bool(score.success_iou50),
                "primary_rescue_success": bool(score.primary_rescue_success),
                "primary_rescue_success_changed": (
                    None
                    if baseline_success is None
                    else bool(score.primary_rescue_success) != baseline_success
                ),
                "exact_tail_match_baseline": decode_result.generated_tail_text
                == generation_row.get("generated_tail_text"),
                "generation_kwargs": _generation_kwargs_artifact_summary(
                    decode_result.generation_kwargs
                ),
                "image_path": str(image_path),
                "config_sha256": _config_sha256(fn_config),
                "attn_implementation_selected": selected_attn,
                "causal_status": "executed_smoke",
            }
        )

    intervention_root = config.paths.artifact_root / "intervention"
    intervention_root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(intervention_root / "intervention_rows.jsonl", output_rows)
    summary = _summarize_intervention_execution(
        config=config,
        rows=output_rows,
        missing_generation_rows=missing_generation_rows,
        fn_rescue_config_path=fn_rescue_config_path,
    )
    _write_json(intervention_root / "summary.json", summary)
    return summary


def _select_intervention_smoke_rows(
    plan_rows: Sequence[Mapping[str, Any]],
    *,
    intervention_kinds: Sequence[str],
    max_cases: int,
) -> list[dict[str, Any]]:
    wanted = tuple(str(kind) for kind in intervention_kinds)
    wanted_set = set(wanted)
    grouped: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    for row in plan_rows:
        intervention_kind = str(row.get("intervention_kind"))
        if intervention_kind not in wanted_set:
            continue
        key = (
            _required_str(row, "case_id"),
            _required_str(row, "rescue_tier"),
            _required_str(row, "selection_kind"),
        )
        grouped.setdefault(key, {}).setdefault(intervention_kind, dict(row))

    selected: list[dict[str, Any]] = []
    selected_cases = 0
    for rows_by_kind in grouped.values():
        if not wanted_set.issubset(rows_by_kind):
            continue
        if selected_cases >= max_cases:
            break
        selected_cases += 1
        for kind in wanted:
            selected.append(rows_by_kind[kind])
    return selected


def _build_intervention_processor_inputs(
    *,
    model_handle: Any,
    messages: Sequence[Mapping[str, Any]],
    image_path: Path,
    bbox_xyxy: Any,
) -> Mapping[str, Any]:
    import torch

    from src.analysis.autoreg_fn_rescue_continuation import (
        _load_fn_rescue_image,
        _model_device,
    )
    from src.common.qwen_generation import call_processor_with_qwen_geometry

    processor = getattr(model_handle, "processor", None)
    if processor is None:
        raise ValueError("model_handle.processor is required")
    prompt_text = processor.apply_chat_template(
        [{"role": message["role"], "content": message["content"]} for message in messages],
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=True,
    )
    image = _load_fn_rescue_image(image_path)
    if bbox_xyxy is not None:
        image = _occlude_norm1000_bbox(image, bbox_xyxy)
    inputs = call_processor_with_qwen_geometry(
        processor,
        text=[prompt_text],
        images=[image],
        return_tensors="pt",
        padding=False,
    )
    model = getattr(model_handle, "model", None)
    device = _model_device(model)
    if device is None:
        return inputs
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in dict(inputs).items()
    }


def _occlude_norm1000_bbox(image: Any, bbox_xyxy: Any) -> Any:
    from PIL import ImageDraw

    image = image.convert("RGB").copy()
    pixel_box = _norm1000_bbox_to_pixel_box(
        bbox_xyxy,
        width=int(image.width),
        height=int(image.height),
    )
    if pixel_box is None:
        return image
    ImageDraw.Draw(image).rectangle(pixel_box, fill=(127, 127, 127))
    return image


def _norm1000_bbox_to_pixel_box(
    bbox_xyxy: Any,
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    if width <= 0 or height <= 0:
        return None
    try:
        x1, y1, x2, y2 = (float(value) for value in _box_tuple_from_any(bbox_xyxy))
    except ValueError:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    x1 = max(0.0, min(999.0, x1))
    y1 = max(0.0, min(999.0, y1))
    x2 = max(0.0, min(999.0, x2))
    y2 = max(0.0, min(999.0, y2))
    if x2 <= x1 or y2 <= y1:
        return None

    def scale_x(value: float) -> int:
        return max(0, min(width - 1, int(round(value / 999.0 * (width - 1)))))

    def scale_y(value: float) -> int:
        return max(0, min(height - 1, int(round(value / 999.0 * (height - 1)))))

    return (scale_x(x1), scale_y(y1), scale_x(x2), scale_y(y2))


def _box_tuple_from_any(value: Any) -> tuple[int, int, int, int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("bbox_xyxy must be a sequence of four numbers")
    if len(value) != 4:
        raise ValueError("bbox_xyxy must have four values")
    return tuple(int(round(float(coord))) for coord in value)  # type: ignore[return-value]


def _summarize_intervention_execution(
    *,
    config: Phase2Config,
    rows: Sequence[Mapping[str, Any]],
    missing_generation_rows: int,
    fn_rescue_config_path: Path,
) -> dict[str, Any]:
    by_kind = Counter(str(row.get("intervention_kind")) for row in rows)
    kind_summaries: dict[str, dict[str, Any]] = {}
    for kind in sorted(by_kind):
        kind_rows = [row for row in rows if row.get("intervention_kind") == kind]
        deltas = [
            float(row["target_iou_delta"])
            for row in kind_rows
            if row.get("target_iou_delta") is not None
        ]
        kind_summaries[kind] = {
            "rows": len(kind_rows),
            "mean_target_iou_delta": None if not deltas else sum(deltas) / len(deltas),
            "primary_success_changed": sum(
                1 for row in kind_rows if row.get("primary_rescue_success_changed") is True
            ),
            "exact_tail_match_baseline": sum(
                1 for row in kind_rows if row.get("exact_tail_match_baseline") is True
            ),
            "invalid_parse": sum(1 for row in kind_rows if row.get("parse_status") != "ok"),
        }
    return {
        "artifact_root": str(config.paths.artifact_root),
        "fn_rescue_root": str(config.paths.fn_rescue_root),
        "fn_rescue_config_path": str(fn_rescue_config_path),
        "evidence_scope": config.evidence_scope,
        "stage": "intervention_smoke",
        "causal_status": "executed_smoke",
        "execution_scope": "no_op_and_target_mask_sanity",
        "row_count": len(rows),
        "case_count": len({str(row.get("case_id")) for row in rows}),
        "missing_generation_rows": missing_generation_rows,
        "intervention_kind_counts": dict(sorted(by_kind.items())),
        "kind_summaries": kind_summaries,
        "interpretation_boundary": (
            "target-mask smoke can validate causal directionality; "
            "far-background/sink suppression is not executed in this stage"
        ),
    }


def _load_candidate_region_index(path: Path) -> tuple[dict[tuple[str, str], dict[str, list[dict[str, Any]]]], int]:
    indexed: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    row_count = 0
    for row in _iter_jsonl(path):
        case_id = _required_str(row, "case_id")
        tier = _required_str(row, "planned_rescue_tier")
        region_kind = _required_str(row, "region_kind")
        indexed[(case_id, tier)][region_kind].append(row)
        row_count += 1
    return (
        {
            key: {region_kind: list(region_rows) for region_kind, region_rows in by_region.items()}
            for key, by_region in indexed.items()
        },
        row_count,
    )


def _intervention_selection_kind(generation_row: Mapping[str, Any]) -> str | None:
    tier = generation_row.get("rescue_tier")
    success = _optional_bool(generation_row.get(PHASE2_SUCCESS_FIELD))
    binding_bucket = generation_row.get("binding_bucket")
    if tier == "desc_x1" and success is True and binding_bucket == "same_desc_competitor":
        return "desc_x1_success_same_desc_competitor"
    if tier == "desc_x1_wrong_control" and success is False:
        return "desc_x1_wrong_control_failure"
    return None


def _planned_region_interventions(selection_kind: str) -> tuple[tuple[str, str], ...]:
    if selection_kind == "desc_x1_success_same_desc_competitor":
        return (
            ("target_gt", "target_gt_mask"),
            ("same_desc_competitor_gt_object", "same_desc_competitor_mask"),
            ("same_desc_rollout_prediction", "same_desc_rollout_prediction_mask"),
            ("far_background", "far_background_sink_mask"),
        )
    if selection_kind == "desc_x1_wrong_control_failure":
        return (("wrong_control_source_region", "wrong_control_source_region_mask"),)
    raise ValueError(f"unknown intervention selection kind: {selection_kind}")


def _intervention_plan_row(
    generation_row: Mapping[str, Any],
    *,
    intervention_kind: str,
    selection_kind: str,
    region_row: Mapping[str, Any] | None,
) -> dict[str, Any]:
    row = {
        "case_id": _required_str(generation_row, "case_id"),
        "source_line_idx": generation_row.get("source_line_idx"),
        "rescue_tier": _required_str(generation_row, "rescue_tier"),
        "selection_kind": selection_kind,
        "intervention_kind": intervention_kind,
        "target_desc": generation_row.get("target_desc"),
        "target_gt_idx": generation_row.get("target_gt_idx"),
        "target_bbox_xyxy": generation_row.get("target_bbox_xyxy"),
        "generated_box_xyxy": generation_row.get("generated_box_xyxy"),
        "generated_coord_tokens": generation_row.get("generated_coord_tokens"),
        "hint_x1": generation_row.get("hint_x1"),
        "hint_x1_raw": generation_row.get("hint_x1_raw"),
        "hint_x1_used": generation_row.get("hint_x1_used"),
        "hint_x1_clamped": generation_row.get("hint_x1_clamped"),
        "binding_bucket": generation_row.get("binding_bucket"),
        "primary_rescue_success": _optional_bool(generation_row.get(PHASE2_SUCCESS_FIELD)),
        "success_iou50": _optional_bool(generation_row.get("success_iou50")),
        "target_iou": generation_row.get("target_iou"),
        "prefix_quality": generation_row.get("prefix_quality"),
        "depth_bucket": generation_row.get("depth_bucket"),
        "object_count_bucket": generation_row.get("object_count_bucket"),
        "image_path": generation_row.get("image_path"),
        "region_kind": None,
        "region_instance_id": None,
        "source_region_kind": None,
        "source_index": None,
        "bbox_xyxy": None,
        "aggregation_scope": None,
        "causal_status": "planned_not_executed",
        "execution_required": "gpu_redecode_with_noop_parity_before_masked_intervention",
        "safety_note": _intervention_safety_note(intervention_kind),
    }
    if region_row is not None:
        row.update(
            {
                "region_kind": region_row.get("region_kind"),
                "region_instance_id": region_row.get("region_instance_id"),
                "source_region_kind": region_row.get("source_region_kind"),
                "source_index": region_row.get("source_index"),
                "bbox_xyxy": region_row.get("bbox_xyxy"),
                "aggregation_scope": region_row.get("aggregation_scope"),
            }
        )
    return row


def _intervention_safety_note(intervention_kind: str) -> str:
    if intervention_kind == "no_op_control":
        return "no visual intervention; validates decode parity before causal masking"
    if intervention_kind == "far_background_sink_mask":
        return "broad background candidate; execute only after token-level sink subset audit"
    return "planned region mask; execute only after no-op parity matches baseline decode"


def load_generation_outcomes(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    outcomes: dict[tuple[str, str], dict[str, Any]] = {}
    for row in _iter_jsonl(path):
        case_id = _required_str(row, "case_id")
        tier = _required_str(row, "rescue_tier")
        key = (case_id, tier)
        outcomes[key] = {
            "case_id": case_id,
            "rescue_tier": tier,
            "primary_rescue_success": _optional_bool(row.get(PHASE2_SUCCESS_FIELD)),
            "success_iou50": _optional_bool(row.get("success_iou50")),
            "outcome_bucket": row.get("outcome_bucket"),
            "target_iou": row.get("target_iou"),
            "binding_bucket": row.get("binding_bucket"),
            "prefix_quality": row.get("prefix_quality"),
            "object_count_bucket": row.get("object_count_bucket"),
        }
    return outcomes


def mine_attention_rows(
    *,
    attention_path: Path,
    generation_outcomes: Mapping[tuple[str, str], Mapping[str, Any]],
    max_attention_rows: int | None,
    aggregation_scopes: Sequence[str],
    region_kinds: Sequence[str],
    ranking_limit: int,
    min_rows_per_head: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    wanted_scopes = set(aggregation_scopes)
    wanted_regions = set(region_kinds)
    region_stats: dict[tuple[str, str, int, int, str], _CellStats] = defaultdict(_CellStats)
    outcome_stats: dict[tuple[str, str, int, int, str], _OutcomeStats] = defaultdict(_OutcomeStats)
    head_totals: dict[tuple[str, str, int, int], int] = defaultdict(int)
    tier_counts: dict[str, int] = defaultdict(int)
    role_counts: dict[str, int] = defaultdict(int)
    region_counts: dict[str, int] = defaultdict(int)
    processed_rows = 0
    joined_rows = 0
    skipped_scope_rows = 0
    skipped_region_rows = 0
    missing_generation_rows = 0
    for row in _iter_jsonl(attention_path):
        if max_attention_rows is not None and processed_rows >= max_attention_rows:
            break
        processed_rows += 1
        aggregation_scope = str(row.get("aggregation_scope") or "")
        if aggregation_scope not in wanted_scopes:
            skipped_scope_rows += 1
            continue
        region_kind = str(row.get("region_kind") or "")
        if region_kind not in wanted_regions:
            skipped_region_rows += 1
            continue
        case_id = _required_str(row, "case_id")
        tier = _required_str(row, "rescue_tier")
        outcome = generation_outcomes.get((case_id, tier))
        if outcome is None:
            missing_generation_rows += 1
            continue
        role = _required_str(row, "role")
        layer = _required_int(row, "layer")
        head = _required_int(row, "head")
        mass = float(row.get("attention_mass_normalized", row.get("attention_mass", 0.0)))
        key = (tier, role, layer, head, region_kind)
        head_key = (tier, role, layer, head)
        region_stats[key].add(mass)
        outcome_stats[key].add(
            mass,
            success=_optional_bool(outcome.get(PHASE2_SUCCESS_FIELD)),
        )
        head_totals[head_key] += 1
        tier_counts[tier] += 1
        role_counts[role] += 1
        region_counts[region_kind] += 1
        joined_rows += 1
    summary_rows = _build_summary_rows(region_stats, outcome_stats)
    ranking_payload = _build_rankings(
        summary_rows,
        head_totals=head_totals,
        ranking_limit=ranking_limit,
        min_rows_per_head=min_rows_per_head,
    )
    summary = {
        "processed_attention_rows": processed_rows,
        "joined_attention_rows": joined_rows,
        "skipped_scope_rows": skipped_scope_rows,
        "skipped_region_rows": skipped_region_rows,
        "missing_generation_rows": missing_generation_rows,
        "unique_head_role_cells": len(head_totals),
        "summary_row_count": len(summary_rows),
        "tier_counts": dict(sorted(tier_counts.items())),
        "role_counts": dict(sorted(role_counts.items())),
        "region_counts": dict(sorted(region_counts.items())),
        "ranking_sections": sorted(ranking_payload),
    }
    return summary_rows, ranking_payload, summary


def write_phase2_report(artifact_root: Path) -> Path:
    mining_root = artifact_root / "attention_mining"
    summary_path = mining_root / "summary.json"
    ranking_path = mining_root / "head_layer_rankings.json"
    intervention_summary_path = artifact_root / "intervention_plan" / "summary.json"
    intervention_smoke_summary_path = artifact_root / "intervention" / "summary.json"
    if not summary_path.exists() or not ranking_path.exists():
        raise FileNotFoundError("attention mining summary/ranking artifacts are required before report")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rankings = json.loads(ranking_path.read_text(encoding="utf-8"))
    lines = [
        "# FN-Rescue Desc-X1 Evidence Routing Phase 2",
        "",
        "## Scope",
        "",
        f"- Evidence scope: `{summary.get('evidence_scope')}`",
        f"- FN-rescue root: `{summary.get('fn_rescue_root')}`",
        f"- Processed attention rows: `{summary.get('processed_attention_rows')}`",
        f"- Joined attention rows: `{summary.get('joined_attention_rows')}`",
        f"- Generation outcome rows: `{summary.get('generation_outcome_count')}`",
        "",
        "## Interpretation Boundary",
        "",
        "This report is attention-mining evidence only. It ranks candidate heads and regions for later causal tests; it does not prove that background/sink regions are harmful.",
        "",
    ]
    if intervention_summary_path.exists():
        intervention_summary = json.loads(intervention_summary_path.read_text(encoding="utf-8"))
        lines.extend(
            [
                "## Causal Intervention Plan",
                "",
                f"- Status: `{intervention_summary.get('causal_status')}`",
                f"- Planned rows: `{intervention_summary.get('planned_intervention_rows')}`",
                f"- Case count: `{intervention_summary.get('case_count')}`",
                f"- Selected generation rows: `{intervention_summary.get('selected_generation_rows')}`",
                f"- Execution required: `{intervention_summary.get('execution_required')}`",
                "",
            ]
        )
    if intervention_smoke_summary_path.exists():
        intervention_smoke_summary = json.loads(
            intervention_smoke_summary_path.read_text(encoding="utf-8")
        )
        kind_summaries = intervention_smoke_summary.get("kind_summaries", {})
        no_op = kind_summaries.get("no_op_control", {}) if isinstance(kind_summaries, Mapping) else {}
        target_mask = (
            kind_summaries.get("target_gt_mask", {})
            if isinstance(kind_summaries, Mapping)
            else {}
        )
        lines.extend(
            [
                "## Causal Intervention Smoke",
                "",
                f"- Status: `{intervention_smoke_summary.get('causal_status')}`",
                f"- Scope: `{intervention_smoke_summary.get('execution_scope')}`",
                f"- Rows: `{intervention_smoke_summary.get('row_count')}`",
                f"- Cases: `{intervention_smoke_summary.get('case_count')}`",
                f"- No-op exact tail matches: `{no_op.get('exact_tail_match_baseline')}` / `{no_op.get('rows')}`",
                f"- Target-mask mean IoU delta: `{_fmt(target_mask.get('mean_target_iou_delta'))}`",
                f"- Target-mask primary-success changed: `{target_mask.get('primary_success_changed')}` / `{target_mask.get('rows')}`",
                "",
            ]
        )
    for section_name, title in (
        ("target_margin_top", "Target Margin Top Heads"),
        ("background_sink_top", "Background Sink Top Heads"),
        ("competitor_sink_top", "Same-Desc Competitor Sink Top Heads"),
        ("success_gap_top", "Success Gap Top Heads"),
    ):
        lines.extend([f"## {title}", ""])
        rows = rankings.get(section_name, [])
        if not rows:
            lines.extend(["No rows passed the ranking filters.", ""])
            continue
        lines.append(
            "| Tier | Role | Layer | Head | Score | Target | Competitor | Background | Rows |"
        )
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in rows[:10]:
            lines.append(
                "| {tier} | {role} | {layer} | {head} | {score} | {target} | {competitor} | {background} | {rows} |".format(
                    tier=row.get("tier"),
                    role=row.get("role"),
                    layer=row.get("layer"),
                    head=row.get("head"),
                    score=_fmt(row.get("score")),
                    target=_fmt(row.get("target_mean")),
                    competitor=_fmt(row.get("competitor_mean")),
                    background=_fmt(row.get("far_background_mean")),
                    rows=row.get("head_rows"),
                )
            )
        lines.append("")
    report_path = artifact_root / "report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _build_summary_rows(
    region_stats: Mapping[tuple[str, str, int, int, str], _CellStats],
    outcome_stats: Mapping[tuple[str, str, int, int, str], _OutcomeStats],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in sorted(region_stats):
        tier, role, layer, head, region_kind = key
        stats = region_stats[key]
        outcomes = outcome_stats[key]
        rows.append(
            {
                "tier": tier,
                "rescue_tier": tier,
                "role": role,
                "layer": layer,
                "head": head,
                "region_kind": region_kind,
                "row_count": stats.count,
                "attention_mass_mean": stats.mean,
                "success_count": outcomes.success_count,
                "success_attention_mass_mean": outcomes.success_mean,
                "failure_count": outcomes.failure_count,
                "failure_attention_mass_mean": outcomes.failure_mean,
                "success_minus_failure_attention": outcomes.success_minus_failure,
            }
        )
    return rows


def _build_rankings(
    summary_rows: Sequence[Mapping[str, Any]],
    *,
    head_totals: Mapping[tuple[str, str, int, int], int],
    ranking_limit: int,
    min_rows_per_head: int,
) -> dict[str, Any]:
    by_head: dict[tuple[str, str, int, int], dict[str, float]] = defaultdict(dict)
    success_gap_by_head: dict[tuple[str, str, int, int], dict[str, float]] = defaultdict(dict)
    for row in summary_rows:
        key = (
            str(row["tier"]),
            str(row["role"]),
            int(row["layer"]),
            int(row["head"]),
        )
        region_kind = str(row["region_kind"])
        mean = row.get("attention_mass_mean")
        if isinstance(mean, (int, float)):
            by_head[key][region_kind] = float(mean)
        gap = row.get("success_minus_failure_attention")
        if isinstance(gap, (int, float)):
            success_gap_by_head[key][region_kind] = float(gap)

    ranked: list[dict[str, Any]] = []
    for key, region_means in by_head.items():
        tier, role, layer, head = key
        head_rows = int(head_totals.get(key, 0))
        if head_rows < min_rows_per_head:
            continue
        target = region_means.get("target_gt")
        competitor = max(
            (
                region_means.get("same_desc_competitor_gt_object", 0.0),
                region_means.get("same_desc_rollout_prediction", 0.0),
                region_means.get("wrong_control_source_region", 0.0),
            )
        )
        background = region_means.get("far_background")
        context = region_means.get("context_ring")
        target_margin = None if target is None else target - competitor
        target_background_margin = (
            None if target is None or background is None else target - background
        )
        success_gap_target = success_gap_by_head.get(key, {}).get("target_gt")
        ranked.append(
            {
                "tier": tier,
                "role": role,
                "layer": layer,
                "head": head,
                "head_rows": head_rows,
                "target_mean": target,
                "competitor_mean": competitor,
                "far_background_mean": background,
                "context_ring_mean": context,
                "target_minus_competitor": target_margin,
                "target_minus_far_background": target_background_margin,
                "target_success_minus_failure": success_gap_target,
            }
        )

    def top_by(field: str, *, reverse: bool = True) -> list[dict[str, Any]]:
        rows = [row for row in ranked if isinstance(row.get(field), (int, float))]
        rows.sort(key=lambda row: float(row[field]), reverse=reverse)
        return [{**row, "score": row[field]} for row in rows[:ranking_limit]]

    return {
        "target_margin_top": top_by("target_minus_competitor", reverse=True),
        "target_margin_bottom": top_by("target_minus_competitor", reverse=False),
        "background_sink_top": top_by("far_background_mean", reverse=True),
        "competitor_sink_top": top_by("competitor_mean", reverse=True),
        "success_gap_top": top_by("target_success_minus_failure", reverse=True),
        "success_gap_bottom": top_by("target_success_minus_failure", reverse=False),
    }


def _validate_stages(stages: Sequence[str]) -> None:
    if not stages:
        raise ValueError("stages must not be empty")
    unknown = sorted(set(stages) - set(PHASE2_STAGES))
    if unknown:
        raise ValueError(f"unknown phase2 stage(s): {', '.join(unknown)}")


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} row is not a JSON object")
            yield row


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _required_mapping(row: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    value = row.get(field)
    if not isinstance(value, Mapping):
        raise ValueError(f"missing mapping field: {field}")
    return value


def _optional_mapping(row: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    value = row.get(field)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"field must be a mapping: {field}")
    return value


def _required_path(row: Mapping[str, Any], field: str) -> Path:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"missing path field: {field}")
    return Path(value)


def _optional_path_value(value: Any, *, base_dir: Path | None = None) -> Path | None:
    if value is None or value == "":
        return None
    path = Path(str(value)).expanduser()
    if base_dir is not None and not path.is_absolute():
        path = base_dir / path
    return path.resolve(strict=False)


def _required_str(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"missing string field: {field}")
    return value


def _required_int(row: Mapping[str, Any], field: str) -> int:
    value = row.get(field)
    if not isinstance(value, int):
        raise ValueError(f"missing int field: {field}")
    return value


def _optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _string_tuple(value: Any, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field_name} must be a sequence of strings")
    items = tuple(str(item) for item in value)
    if not items:
        raise ValueError(f"{field_name} must not be empty")
    return items


def _nonnegative_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a nonnegative integer")
    return value


def _positive_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, int):
        return str(value)
    return ""
