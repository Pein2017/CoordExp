from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

import torch

from ..teacher_forcing.contracts import PipelineResult, TeacherForcingContext
from ..teacher_forcing.objective_pipeline import run_teacher_forcing_pipeline


@dataclass(frozen=True)
class Stage2ObjectiveRunResult:
    objective_specs_ctx: list[Mapping[str, Any]]
    pipeline_ctx_result: PipelineResult
    pipeline_metrics_ctx: Dict[str, float]
    pipeline_ctx_total_loss: torch.Tensor
    total_loss: torch.Tensor


def _spec_name(spec: Mapping[str, Any]) -> str:
    return str(spec.get("name", "") or "").strip()


def _filter_channel_a_specs(
    specs: Sequence[Mapping[str, Any]],
    *,
    run_a_text: bool,
    run_a_bbox_geo: bool,
    run_a_bbox_size_aux: bool,
    run_a_coord_reg: bool,
) -> list[Mapping[str, Any]]:
    out: list[Mapping[str, Any]] = []
    for spec in list(specs or []):
        if not isinstance(spec, Mapping):
            continue
        name = _spec_name(spec)
        if name == "token_ce" and run_a_text:
            out.append(spec)
            continue
        if name == "bbox_geo" and run_a_bbox_geo:
            out.append(spec)
            continue
        if name == "bbox_size_aux" and run_a_bbox_size_aux:
            out.append(spec)
            continue
        if name == "coord_reg" and run_a_coord_reg:
            out.append(spec)
    return out


def build_teacher_forcing_context(
    *,
    channel: str,
    registry_context: str,
    input_ids: torch.Tensor,
    logits: torch.Tensor,
    logits_ce: torch.Tensor,
    meta: Sequence[Mapping[str, Any]],
    coord_token_ids: Sequence[int],
    temperature: float,
    token_type_masks: Optional[Mapping[str, torch.Tensor]] = None,
    rollout_subset_masks: Optional[Mapping[str, torch.Tensor]] = None,
) -> TeacherForcingContext:
    return TeacherForcingContext(
        channel=str(channel),
        registry_context=str(registry_context),
        input_ids=input_ids,
        logits=logits,
        logits_ce=logits_ce,
        meta=meta,
        coord_token_ids=coord_token_ids,
        temperature=float(temperature),
        token_type_masks=dict(token_type_masks or {}),
        rollout_subset_masks=dict(rollout_subset_masks or {}),
        extra={},
    )


def run_stage2_objective_pipelines(
    *,
    channel: str,
    objective_specs: Sequence[Mapping[str, Any]],
    diagnostic_specs: Sequence[Mapping[str, Any]],
    input_ids: torch.Tensor,
    logits: torch.Tensor,
    logits_ce: torch.Tensor,
    meta: Sequence[Mapping[str, Any]],
    coord_token_ids: Sequence[int],
    temperature: float,
    token_type_masks: Mapping[str, torch.Tensor],
    rollout_subset_masks: Mapping[str, torch.Tensor],
    run_a_text: bool,
    run_a_bbox_geo: bool,
    run_a_bbox_size_aux: bool,
    run_a_coord_reg: bool,
    warn_once_cache: set[str],
) -> Stage2ObjectiveRunResult:
    objective_specs_ctx = (
        _filter_channel_a_specs(
            objective_specs,
            run_a_text=run_a_text,
            run_a_bbox_geo=run_a_bbox_geo,
            run_a_bbox_size_aux=run_a_bbox_size_aux,
            run_a_coord_reg=run_a_coord_reg,
        )
        if channel == "A"
        else list(objective_specs or [])
    )

    tf_context = build_teacher_forcing_context(
        channel=str(channel),
        registry_context=("gt" if channel == "A" else "rollout"),
        input_ids=input_ids,
        logits=logits,
        logits_ce=logits_ce,
        meta=meta,
        coord_token_ids=coord_token_ids,
        temperature=float(temperature),
        token_type_masks=token_type_masks,
        rollout_subset_masks=rollout_subset_masks,
    )

    pipeline_ctx_result = run_teacher_forcing_pipeline(
        context=tf_context,
        objective_specs=objective_specs_ctx,
        diagnostics_specs=diagnostic_specs,
        initial_state=None,
        warn_once_cache=warn_once_cache,
    )
    pipeline_metrics_ctx = dict(pipeline_ctx_result.metrics)
    pipeline_ctx_total_loss = pipeline_ctx_result.total_loss

    return Stage2ObjectiveRunResult(
        objective_specs_ctx=objective_specs_ctx,
        pipeline_ctx_result=pipeline_ctx_result,
        pipeline_metrics_ctx=pipeline_metrics_ctx,
        pipeline_ctx_total_loss=pipeline_ctx_total_loss,
        total_loss=pipeline_ctx_total_loss,
    )


__all__ = [
    "build_stage2_core_loss_logs",
    "Stage2ObjectiveRunResult",
    "build_teacher_forcing_context",
    "run_stage2_objective_pipelines",
]


def build_stage2_core_loss_logs(
    *,
    channel: str,
    pipeline_metrics_ctx: Mapping[str, float],
    token_ce_module_w: float,
    bbox_geo_module_w: float,
    bbox_size_aux_module_w: float,
    coord_reg_module_w: float,
    duplicate_burst_unlikelihood_module_w: float,
    run_a_text: bool,
    run_a_bbox_geo: bool,
    run_a_bbox_size_aux: bool,
    run_a_coord_reg: bool,
    token_desc_ce_weight: float,
    fn_desc_ce_weight: float,
    bbox_smoothl1_w: float,
    bbox_ciou_w: float,
    bbox_log_wh_w: float,
    bbox_oversize_w: float,
    coord_ce_w: float,
    coord_soft_ce_w: float,
    coord_w1_w: float,
    coord_gate_w: float,
    text_gate_w: float,
) -> Dict[str, float]:
    stage2_logs: Dict[str, float] = {}

    if channel == "A":
        if float(token_ce_module_w) != 0.0 and run_a_text:
            token_struct = float(
                pipeline_metrics_ctx.get("loss/token_ce_struct", 0.0) or 0.0
            )
            token_desc = float(
                pipeline_metrics_ctx.get("loss/token_ce_desc", 0.0) or 0.0
            )

            stage2_logs["loss/text/struct_ce"] = float(
                float(token_ce_module_w) * float(token_struct)
            )
            if float(token_desc_ce_weight) != 0.0:
                stage2_logs["loss/text/desc_ce"] = float(
                    float(token_ce_module_w) * float(token_desc)
                )

        if float(bbox_geo_module_w) != 0.0 and run_a_bbox_geo:
            smoothl1 = float(
                pipeline_metrics_ctx.get("loss/bbox_smoothl1", 0.0) or 0.0
            )
            ciou = float(
                pipeline_metrics_ctx.get("loss/bbox_ciou", 0.0) or 0.0
            )
            if float(bbox_smoothl1_w) != 0.0:
                stage2_logs["loss/coord/bbox_smoothl1"] = float(
                    float(bbox_geo_module_w) * float(bbox_smoothl1_w) * float(smoothl1)
                )
            if float(bbox_ciou_w) != 0.0:
                stage2_logs["loss/coord/bbox_ciou"] = float(
                    float(bbox_geo_module_w) * float(bbox_ciou_w) * float(ciou)
                )

        if float(bbox_size_aux_module_w) != 0.0 and run_a_bbox_size_aux:
            def _emit_a_bbox_size(term: str, weight: float, raw_key: str) -> None:
                if float(weight) == 0.0:
                    return
                value = float(pipeline_metrics_ctx.get(raw_key, 0.0) or 0.0)
                stage2_logs[f"loss/coord/{term}"] = float(
                    float(bbox_size_aux_module_w) * float(weight) * float(value)
                )

            _emit_a_bbox_size("bbox_log_wh", bbox_log_wh_w, "loss/bbox_log_wh")
            _emit_a_bbox_size("bbox_oversize", bbox_oversize_w, "loss/bbox_oversize")

        if float(coord_reg_module_w) != 0.0 and run_a_coord_reg:
            def _emit_a(term: str, weight: float, raw_key: str) -> None:
                if float(weight) == 0.0:
                    return
                value = float(pipeline_metrics_ctx.get(raw_key, 0.0) or 0.0)
                stage2_logs[f"loss/coord/{term}"] = float(
                    float(coord_reg_module_w) * float(weight) * float(value)
                )

            _emit_a("coord_token_ce", coord_ce_w, "loss/coord_token_ce")
            _emit_a("coord_soft_ce", coord_soft_ce_w, "loss/coord_soft_ce")
            _emit_a("coord_w1", coord_w1_w, "loss/coord_w1")
            _emit_a("coord_gate", coord_gate_w, "loss/coord_gate")
            _emit_a("text_gate", text_gate_w, "loss/text_gate")
    else:
        if float(token_ce_module_w) != 0.0:
            token_struct = float(
                pipeline_metrics_ctx.get("loss/token_ce_struct", 0.0) or 0.0
            )
            token_desc = float(
                pipeline_metrics_ctx.get("loss/token_ce_desc", 0.0) or 0.0
            )

            stage2_logs["loss/B_rollout_text/struct_ce"] = float(
                float(token_ce_module_w) * float(token_struct)
            )
            if float(fn_desc_ce_weight) != 0.0:
                stage2_logs["loss/B_rollout_text/desc_ce"] = float(
                    float(token_ce_module_w) * float(token_desc)
                )

        if float(duplicate_burst_unlikelihood_module_w) != 0.0:
            duplicate_burst_loss = float(
                pipeline_metrics_ctx.get(
                    "train/optimization/loss_duplicate_burst_unlikelihood", 0.0
                )
                or 0.0
            )
            duplicate_burst_num_terms = float(
                pipeline_metrics_ctx.get(
                    "train/triage/duplicate_burst_unlikelihood_target_count", 0.0
                )
                or 0.0
            )
            duplicate_burst_num_boundaries = float(
                pipeline_metrics_ctx.get(
                    "train/triage/duplicate_burst_unlikelihood_boundary_count", 0.0
                )
                or 0.0
            )
            stage2_logs["train/optimization/loss_duplicate_burst_unlikelihood"] = float(
                duplicate_burst_loss
            )
            stage2_logs["loss/B_rollout_text/duplicate_burst_unlikelihood"] = float(
                duplicate_burst_loss
            )
            stage2_logs["diag/duplicate_burst/num_terms"] = float(
                duplicate_burst_num_terms
            )
            stage2_logs["diag/duplicate_burst/num_ul_boundaries"] = float(
                duplicate_burst_num_boundaries
            )
            stage2_logs["diag/duplicate_burst/loss_per_term"] = float(
                duplicate_burst_loss if duplicate_burst_num_terms > 0.0 else 0.0
            )

        if float(bbox_geo_module_w) != 0.0:
            smoothl1 = float(
                pipeline_metrics_ctx.get("loss/bbox_smoothl1", 0.0) or 0.0
            )
            ciou = float(
                pipeline_metrics_ctx.get("loss/bbox_ciou", 0.0) or 0.0
            )
            if float(bbox_smoothl1_w) != 0.0:
                stage2_logs["loss/B_coord/bbox_smoothl1"] = float(
                    float(bbox_geo_module_w) * float(bbox_smoothl1_w) * float(smoothl1)
                )
            if float(bbox_ciou_w) != 0.0:
                stage2_logs["loss/B_coord/bbox_ciou"] = float(
                    float(bbox_geo_module_w) * float(bbox_ciou_w) * float(ciou)
                )

        if float(bbox_size_aux_module_w) != 0.0:
            def _emit_b_bbox_size(term: str, weight: float, raw_key: str) -> None:
                if float(weight) == 0.0:
                    return
                value = float(pipeline_metrics_ctx.get(raw_key, 0.0) or 0.0)
                stage2_logs[f"loss/B_coord/{term}"] = float(
                    float(bbox_size_aux_module_w) * float(weight) * float(value)
                )

            _emit_b_bbox_size("bbox_log_wh", bbox_log_wh_w, "loss/bbox_log_wh")
            _emit_b_bbox_size("bbox_oversize", bbox_oversize_w, "loss/bbox_oversize")

        if float(coord_reg_module_w) != 0.0:
            def _emit_b(term: str, weight: float, raw_key: str) -> None:
                if float(weight) == 0.0:
                    return
                value = float(pipeline_metrics_ctx.get(raw_key, 0.0) or 0.0)
                stage2_logs[f"loss/B_coord/{term}"] = float(
                    float(coord_reg_module_w) * float(weight) * float(value)
                )

            _emit_b("coord_token_ce", coord_ce_w, "loss/coord_token_ce")
            _emit_b("coord_soft_ce", coord_soft_ce_w, "loss/coord_soft_ce")
            _emit_b("coord_w1", coord_w1_w, "loss/coord_w1")
            _emit_b("coord_gate", coord_gate_w, "loss/coord_gate")
            _emit_b("text_gate", text_gate_w, "loss/text_gate")

    return stage2_logs
