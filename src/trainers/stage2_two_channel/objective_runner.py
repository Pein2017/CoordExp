from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

import torch

from ..teacher_forcing.contracts import (
    PipelineModuleSpec,
    TeacherForcingContext,
    PipelineResult,
)
from ..teacher_forcing.objective_pipeline import run_teacher_forcing_pipeline


@dataclass(frozen=True)
class Stage2ObjectiveRunResult:
    objective_specs_ctx: list[Mapping[str, Any]]
    pipeline_ctx_result: PipelineResult
    pipeline_metrics_ctx: Dict[str, float]
    pipeline_ctx_total_loss: torch.Tensor
    pipeline_metrics_gt: Dict[str, float]
    pipeline_gt_total_loss: torch.Tensor
    pipeline_metrics_a1: Dict[str, float]
    pipeline_a1_total_loss: torch.Tensor
    a1_bbox_state: Optional[Mapping[str, Any]]
    a1_coord_state: Optional[Mapping[str, Any]]
    total_loss: torch.Tensor


def _spec_name(spec: Mapping[str, Any]) -> str:
    return str(spec.get("name", "") or "").strip()


def _filter_channel_a_gt_specs(
    specs: Sequence[Mapping[str, Any]],
    *,
    run_a1_text: bool,
) -> list[Mapping[str, Any]]:
    if not run_a1_text:
        return []
    return [
        spec
        for spec in list(specs or [])
        if isinstance(spec, Mapping) and _spec_name(spec) == "token_ce"
    ]


def _filter_channel_a_ctx_specs(
    specs: Sequence[Mapping[str, Any]],
    *,
    run_a2_text: bool,
    run_a2_bbox_geo: bool,
    run_a2_bbox_size_aux: bool,
    run_a2_coord_reg: bool,
) -> list[Mapping[str, Any]]:
    out: list[Mapping[str, Any]] = []
    for spec in list(specs or []):
        if not isinstance(spec, Mapping):
            continue
        name = _spec_name(spec)
        if name == "token_ce":
            if run_a2_text:
                out.append(spec)
            continue
        if name == "bbox_geo" and run_a2_bbox_geo:
            out.append(spec)
            continue
        if name == "bbox_size_aux" and run_a2_bbox_size_aux:
            out.append(spec)
            continue
        if name == "coord_reg" and run_a2_coord_reg:
            out.append(spec)
            continue
        if name == "loss_dead_anchor_suppression":
            out.append(spec)
    return out


def _filter_channel_a_a1_coord_specs(
    specs: Sequence[Mapping[str, Any]],
    *,
    run_a1_bbox_geo: bool,
    run_a1_bbox_size_aux: bool,
    run_a1_coord_reg: bool,
) -> list[Mapping[str, Any]]:
    out: list[Mapping[str, Any]] = []
    for spec in list(specs or []):
        if not isinstance(spec, Mapping):
            continue
        name = _spec_name(spec)
        if name == "bbox_geo" and run_a1_bbox_geo:
            out.append(spec)
            continue
        if name == "bbox_size_aux" and run_a1_bbox_size_aux:
            out.append(spec)
            continue
        if name == "coord_reg" and run_a1_coord_reg:
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
    decode_mode: str,
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
        decode_mode=str(decode_mode),
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
    coord_decode_mode: str,
    token_type_masks: Mapping[str, torch.Tensor],
    rollout_subset_masks: Mapping[str, torch.Tensor],
    run_a1_text: bool,
    run_a1_bbox_geo: bool,
    run_a1_bbox_size_aux: bool,
    run_a1_coord_reg: bool,
    run_a2_text: bool,
    run_a2_bbox_geo: bool,
    run_a2_bbox_size_aux: bool,
    run_a2_coord_reg: bool,
    warn_once_cache: set[str],
) -> Stage2ObjectiveRunResult:
    tf_context = build_teacher_forcing_context(
        channel=str(channel),
        registry_context=("self_context" if channel == "A" else "rollout"),
        input_ids=input_ids,
        logits=logits,
        logits_ce=logits,
        meta=meta,
        coord_token_ids=coord_token_ids,
        temperature=float(temperature),
        decode_mode=str(coord_decode_mode),
        token_type_masks=token_type_masks,
        rollout_subset_masks=rollout_subset_masks,
    )

    objective_specs_ctx = (
        _filter_channel_a_ctx_specs(
            objective_specs,
            run_a2_text=run_a2_text,
            run_a2_bbox_geo=run_a2_bbox_geo,
            run_a2_bbox_size_aux=run_a2_bbox_size_aux,
            run_a2_coord_reg=run_a2_coord_reg,
        )
        if channel == "A"
        else list(objective_specs or [])
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

    pipeline_metrics_gt: Dict[str, float] = {}
    pipeline_metrics_a1: Dict[str, float] = {}
    a1_bbox_state: Optional[Mapping[str, Any]] = None
    a1_coord_state: Optional[Mapping[str, Any]] = None
    pipeline_a1_total_loss = logits_ce.new_tensor(0.0)
    pipeline_gt_total_loss = logits_ce.new_tensor(0.0)

    if channel == "A":
        objective_specs_gt = _filter_channel_a_gt_specs(
            objective_specs,
            run_a1_text=run_a1_text,
        )
        token_ctx_gt = build_teacher_forcing_context(
            channel="A",
            registry_context="gt",
            input_ids=input_ids,
            logits=logits_ce,
            logits_ce=logits_ce,
            meta=meta,
            coord_token_ids=coord_token_ids,
            temperature=float(temperature),
            decode_mode=str(coord_decode_mode),
        )
        if objective_specs_gt:
            pipeline_gt = run_teacher_forcing_pipeline(
                context=token_ctx_gt,
                objective_specs=objective_specs_gt,
                diagnostics_specs=diagnostic_specs,
                initial_state=None,
                warn_once_cache=warn_once_cache,
            )
            pipeline_metrics_gt = dict(pipeline_gt.metrics)
            pipeline_gt_total_loss = pipeline_gt.total_loss

        objective_specs_a1 = _filter_channel_a_a1_coord_specs(
            objective_specs,
            run_a1_bbox_geo=run_a1_bbox_geo,
            run_a1_bbox_size_aux=run_a1_bbox_size_aux,
            run_a1_coord_reg=run_a1_coord_reg,
        )
        if objective_specs_a1:
            ctx_a1_obj = build_teacher_forcing_context(
                channel="A",
                registry_context="a1",
                input_ids=input_ids,
                logits=logits_ce,
                logits_ce=logits_ce,
                meta=meta,
                coord_token_ids=coord_token_ids,
                temperature=float(temperature),
                decode_mode=str(coord_decode_mode),
                token_type_masks=token_type_masks,
                rollout_subset_masks=rollout_subset_masks,
            )
            pipeline_a1 = run_teacher_forcing_pipeline(
                context=ctx_a1_obj,
                objective_specs=objective_specs_a1,
                diagnostics_specs=[],
                initial_state=None,
                warn_once_cache=warn_once_cache,
            )
            pipeline_metrics_a1 = dict(pipeline_a1.metrics)
            pipeline_a1_total_loss = pipeline_a1.total_loss
            a1_bbox_state = dict(pipeline_a1.state or {})
            a1_coord_state = dict(pipeline_a1.state or {})

        total_loss = (
            pipeline_gt_total_loss + pipeline_ctx_total_loss + pipeline_a1_total_loss
        )
    else:
        total_loss = pipeline_ctx_total_loss

    return Stage2ObjectiveRunResult(
        objective_specs_ctx=objective_specs_ctx,
        pipeline_ctx_result=pipeline_ctx_result,
        pipeline_metrics_ctx=pipeline_metrics_ctx,
        pipeline_ctx_total_loss=pipeline_ctx_total_loss,
        pipeline_metrics_gt=pipeline_metrics_gt,
        pipeline_gt_total_loss=pipeline_gt_total_loss,
        pipeline_metrics_a1=pipeline_metrics_a1,
        pipeline_a1_total_loss=pipeline_a1_total_loss,
        a1_bbox_state=a1_bbox_state,
        a1_coord_state=a1_coord_state,
        total_loss=total_loss,
    )


__all__ = [
    "Stage2ObjectiveRunResult",
    "build_teacher_forcing_context",
    "run_channel_a_a1_coord_diagnostics",
    "run_stage2_objective_pipelines",
]


def run_channel_a_a1_coord_diagnostics(
    *,
    input_ids: torch.Tensor,
    logits_ce: torch.Tensor,
    meta: Sequence[Mapping[str, Any]],
    coord_token_ids: Sequence[int],
    temperature: float,
    coord_decode_mode: str,
    token_type_masks: Mapping[str, torch.Tensor],
    rollout_subset_masks: Mapping[str, torch.Tensor],
    bbox_cfg: Mapping[str, Any],
) -> Dict[str, float]:
    from ..teacher_forcing.modules import run_bbox_geo_module, run_coord_diag_module

    ctx_a1 = build_teacher_forcing_context(
        channel="A",
        registry_context="self_context",
        input_ids=input_ids,
        logits=logits_ce,
        logits_ce=logits_ce,
        meta=meta,
        coord_token_ids=coord_token_ids,
        temperature=float(temperature),
        decode_mode=str(coord_decode_mode),
        token_type_masks=token_type_masks,
        rollout_subset_masks=rollout_subset_masks,
    )
    bbox_spec = PipelineModuleSpec(
        name="bbox_geo",
        enabled=True,
        weight=0.0,
        channels=("A",),
        config=dict(bbox_cfg),
    )
    diag_spec = PipelineModuleSpec(
        name="coord_diag",
        enabled=True,
        weight=0.0,
        channels=("A",),
        config={},
    )

    bbox_out = run_bbox_geo_module(
        context=ctx_a1,
        spec=bbox_spec,
    )
    diag_out = run_coord_diag_module(
        context=ctx_a1,
        spec=diag_spec,
        state=bbox_out.state,
    )
    return {str(k): float(v) for k, v in dict(diag_out.metrics).items()}
