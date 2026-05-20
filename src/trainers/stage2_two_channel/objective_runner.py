from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

import torch

from src.training.teacher_forcing.vocab import RoleVocab

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
) -> list[Mapping[str, Any]]:
    out: list[Mapping[str, Any]] = []
    for spec in list(specs or []):
        if not isinstance(spec, Mapping):
            continue
        name = _spec_name(spec)
        if name == "token_ce" and run_a_text:
            out.append(spec)
            continue
        if name == "hard_sft":
            out.append(spec)
            continue
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
    role_vocab: RoleVocab | None = None,
) -> TeacherForcingContext:
    extra = {"role_vocab": role_vocab} if isinstance(role_vocab, RoleVocab) else {}
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
        extra=extra,
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
    warn_once_cache: set[str],
    role_vocab: RoleVocab | None = None,
) -> Stage2ObjectiveRunResult:
    objective_specs_ctx = (
        _filter_channel_a_specs(
            objective_specs,
            run_a_text=run_a_text,
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
        role_vocab=role_vocab,
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
    run_a_text: bool,
    token_desc_ce_weight: float,
    fn_desc_ce_weight: float,
) -> Dict[str, float]:
    stage2_logs: Dict[str, float] = {}

    def _emit_passthrough_metrics() -> None:
        for key, value in pipeline_metrics_ctx.items():
            key_s = str(key)
            if not (
                key_s.startswith("stage2_trie/")
                or key_s.startswith("stage2_ab/channel_b/residual_set/")
                or key_s in {"loss/B/stage2_trie_ce", "loss/stage2_trie_ce"}
            ):
                continue
            stage2_logs[key_s] = float(value or 0.0)

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

    _emit_passthrough_metrics()

    return stage2_logs
