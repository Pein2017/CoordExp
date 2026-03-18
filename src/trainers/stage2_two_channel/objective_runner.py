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
    "build_stage2_core_loss_logs",
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


def build_stage2_core_loss_logs(
    *,
    channel: str,
    pipeline_metrics_ctx: Mapping[str, float],
    pipeline_metrics_gt: Mapping[str, float],
    pipeline_metrics_a1: Mapping[str, float],
    pipeline_a1_total_loss: torch.Tensor,
    token_ce_module_w: float,
    bbox_geo_module_w: float,
    bbox_size_aux_module_w: float,
    coord_reg_module_w: float,
    dead_anchor_suppression_module_w: float,
    run_a1_text: bool,
    run_a1_bbox_geo: bool,
    run_a1_bbox_size_aux: bool,
    run_a1_coord_reg: bool,
    run_a2_text: bool,
    run_a2_bbox_geo: bool,
    run_a2_bbox_size_aux: bool,
    run_a2_coord_reg: bool,
    n_softctx_iter: int,
    self_context_struct_ce_weight: float,
    token_desc_ce_weight: float,
    fn_desc_ce_weight: float,
    bbox_smoothl1_w: float,
    bbox_ciou_w: float,
    bbox_log_wh_w: float,
    bbox_oversize_w: float,
    coord_ce_w: float,
    coord_soft_ce_w: float,
    coord_w1_w: float,
    coord_el1_w: float,
    coord_ehuber_w: float,
    coord_entropy_w: float,
    coord_gate_w: float,
    text_gate_w: float,
) -> Dict[str, float]:
    stage2_logs: Dict[str, float] = {}

    if channel == "A":
        if float(token_ce_module_w) != 0.0 and run_a1_text:
            token_struct = float(
                pipeline_metrics_gt.get("loss/token_ce_struct", 0.0) or 0.0
            )
            token_desc = float(
                pipeline_metrics_gt.get("loss/token_ce_desc", 0.0) or 0.0
            )

            stage2_logs["loss/A1_text/struct_ce"] = float(
                float(token_ce_module_w) * float(token_struct)
            )
            if float(token_desc_ce_weight) != 0.0:
                stage2_logs["loss/A1_text/desc_ce"] = float(
                    float(token_ce_module_w) * float(token_desc)
                )

            fmt_weight = (
                float(self_context_struct_ce_weight)
                if int(n_softctx_iter) > 1 and run_a2_text
                else 0.0
            )
            if float(fmt_weight) != 0.0:
                token_self_struct = float(
                    pipeline_metrics_ctx.get("loss/token_ce_struct", 0.0) or 0.0
                )
                stage2_logs["loss/A2_text/struct_ce"] = float(
                    float(token_ce_module_w)
                    * float(fmt_weight)
                    * float(token_self_struct)
                )

        if pipeline_metrics_a1:
            if float(bbox_geo_module_w) != 0.0 and run_a1_bbox_geo:
                smoothl1_a1 = float(
                    pipeline_metrics_a1.get("loss/bbox_smoothl1", 0.0) or 0.0
                )
                ciou_a1 = float(
                    pipeline_metrics_a1.get("loss/bbox_ciou", 0.0) or 0.0
                )
                if float(bbox_smoothl1_w) != 0.0:
                    stage2_logs["loss/A1_coord/bbox_smoothl1"] = float(
                        float(bbox_geo_module_w)
                        * float(bbox_smoothl1_w)
                        * float(smoothl1_a1)
                    )
                if float(bbox_ciou_w) != 0.0:
                    stage2_logs["loss/A1_coord/bbox_ciou"] = float(
                        float(bbox_geo_module_w)
                        * float(bbox_ciou_w)
                        * float(ciou_a1)
                    )

            if float(bbox_size_aux_module_w) != 0.0 and run_a1_bbox_size_aux:
                def _emit_a1_bbox_size(term: str, weight: float, raw_key: str) -> None:
                    if float(weight) == 0.0:
                        return
                    value = float(pipeline_metrics_a1.get(raw_key, 0.0) or 0.0)
                    stage2_logs[f"loss/A1_coord/{term}"] = float(
                        float(bbox_size_aux_module_w) * float(weight) * float(value)
                    )

                _emit_a1_bbox_size("bbox_log_wh", bbox_log_wh_w, "loss/bbox_log_wh")
                _emit_a1_bbox_size("bbox_oversize", bbox_oversize_w, "loss/bbox_oversize")

            if float(coord_reg_module_w) != 0.0 and run_a1_coord_reg:
                def _emit_a1(term: str, weight: float, raw_key: str) -> None:
                    if float(weight) == 0.0:
                        return
                    value = float(pipeline_metrics_a1.get(raw_key, 0.0) or 0.0)
                    stage2_logs[f"loss/A1_coord/{term}"] = float(
                        float(coord_reg_module_w) * float(weight) * float(value)
                    )

                _emit_a1("coord_token_ce", coord_ce_w, "loss/coord_token_ce")
                _emit_a1("coord_soft_ce", coord_soft_ce_w, "loss/coord_soft_ce")
                _emit_a1("coord_w1", coord_w1_w, "loss/coord_w1")
                _emit_a1("coord_el1", coord_el1_w, "loss/coord_el1")
                _emit_a1("coord_ehuber", coord_ehuber_w, "loss/coord_ehuber")
                _emit_a1("coord_entropy", coord_entropy_w, "loss/coord_entropy")
                _emit_a1("coord_gate", coord_gate_w, "loss/coord_gate")
                _emit_a1("text_gate", text_gate_w, "loss/text_gate")

            stage2_logs["loss/A1_coord/total"] = float(
                pipeline_a1_total_loss.detach().cpu().item()
            )

        if float(bbox_geo_module_w) != 0.0 and run_a2_bbox_geo:
            smoothl1 = float(
                pipeline_metrics_ctx.get("loss/bbox_smoothl1", 0.0) or 0.0
            )
            ciou = float(
                pipeline_metrics_ctx.get("loss/bbox_ciou", 0.0) or 0.0
            )
            if float(bbox_smoothl1_w) != 0.0:
                stage2_logs["loss/A2_coord/bbox_smoothl1"] = float(
                    float(bbox_geo_module_w) * float(bbox_smoothl1_w) * float(smoothl1)
                )
            if float(bbox_ciou_w) != 0.0:
                stage2_logs["loss/A2_coord/bbox_ciou"] = float(
                    float(bbox_geo_module_w) * float(bbox_ciou_w) * float(ciou)
                )

        if float(bbox_size_aux_module_w) != 0.0 and run_a2_bbox_size_aux:
            def _emit_a2_bbox_size(term: str, weight: float, raw_key: str) -> None:
                if float(weight) == 0.0:
                    return
                value = float(pipeline_metrics_ctx.get(raw_key, 0.0) or 0.0)
                stage2_logs[f"loss/A2_coord/{term}"] = float(
                    float(bbox_size_aux_module_w) * float(weight) * float(value)
                )

            _emit_a2_bbox_size("bbox_log_wh", bbox_log_wh_w, "loss/bbox_log_wh")
            _emit_a2_bbox_size("bbox_oversize", bbox_oversize_w, "loss/bbox_oversize")

        if float(coord_reg_module_w) != 0.0 and run_a2_coord_reg:
            def _emit_a2(term: str, weight: float, raw_key: str) -> None:
                if float(weight) == 0.0:
                    return
                value = float(pipeline_metrics_ctx.get(raw_key, 0.0) or 0.0)
                stage2_logs[f"loss/A2_coord/{term}"] = float(
                    float(coord_reg_module_w) * float(weight) * float(value)
                )

            _emit_a2("coord_token_ce", coord_ce_w, "loss/coord_token_ce")
            _emit_a2("coord_soft_ce", coord_soft_ce_w, "loss/coord_soft_ce")
            _emit_a2("coord_w1", coord_w1_w, "loss/coord_w1")
            _emit_a2("coord_el1", coord_el1_w, "loss/coord_el1")
            _emit_a2("coord_ehuber", coord_ehuber_w, "loss/coord_ehuber")
            _emit_a2("coord_entropy", coord_entropy_w, "loss/coord_entropy")
            _emit_a2("coord_gate", coord_gate_w, "loss/coord_gate")
            _emit_a2("text_gate", text_gate_w, "loss/text_gate")
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

        if float(dead_anchor_suppression_module_w) != 0.0:
            dead_anchor_loss = float(
                pipeline_metrics_ctx.get(
                    "train/optimization/loss_dead_anchor_suppression", 0.0
                )
                or 0.0
            )
            dead_anchor_num_terms = float(
                pipeline_metrics_ctx.get(
                    "train/triage/dead_anchor_suppression_target_count", 0.0
                )
                or 0.0
            )
            dead_anchor_num_boundaries = float(
                pipeline_metrics_ctx.get(
                    "train/triage/dead_anchor_suppression_boundary_count", 0.0
                )
                or 0.0
            )
            stage2_logs["train/optimization/loss_dead_anchor_suppression"] = float(
                dead_anchor_loss
            )
            stage2_logs["loss/B_rollout_text/dead_anchor_suppression"] = float(
                dead_anchor_loss
            )
            stage2_logs["diag/dead_anchor/num_terms"] = float(dead_anchor_num_terms)
            stage2_logs["diag/dead_anchor/num_ul_boundaries"] = float(
                dead_anchor_num_boundaries
            )
            stage2_logs["diag/dead_anchor/loss_per_term"] = float(
                dead_anchor_loss if dead_anchor_num_terms > 0.0 else 0.0
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
            _emit_b("coord_el1", coord_el1_w, "loss/coord_el1")
            _emit_b("coord_ehuber", coord_ehuber_w, "loss/coord_ehuber")
            _emit_b("coord_entropy", coord_entropy_w, "loss/coord_entropy")
            _emit_b("coord_gate", coord_gate_w, "loss/coord_gate")
            _emit_b("text_gate", text_gate_w, "loss/text_gate")

    return stage2_logs
