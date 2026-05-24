from __future__ import annotations

from typing import Any, Mapping

import torch

from ..contracts import ModuleResult, PipelineModuleSpec, TeacherForcingContext
from .token_ce import run_token_ce_module


def _coerce_nonnegative_float(value: Any, *, default: float, key: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"schema_format_ce config {key} must be float-compatible") from exc
    if result < 0.0:
        raise ValueError(f"schema_format_ce config {key} must be nonnegative")
    return float(result)


def run_schema_format_ce_module(
    *,
    context: TeacherForcingContext,
    spec: PipelineModuleSpec,
) -> ModuleResult:
    cfg = spec.config if isinstance(spec.config, Mapping) else {}
    schema_ce_weight = _coerce_nonnegative_float(
        cfg.get("schema_ce_weight", 1.0),
        default=1.0,
        key="schema_ce_weight",
    )
    token_spec = PipelineModuleSpec(
        name="token_ce",
        enabled=spec.enabled,
        weight=1.0,
        channels=spec.channels,
        application={"preset": "rollout_text_only"},
        config={
            "desc_ce_weight": 0.0,
            "rollout_fn_desc_weight": 0.0,
            "rollout_global_prefix_struct_ce_weight": 1.0,
            "schema_ce_weight": float(schema_ce_weight),
        },
    )
    out = run_token_ce_module(context=context, spec=token_spec)
    schema_loss = out.loss
    metrics = {
        "loss/schema_format_ce": float(schema_loss.detach().cpu().item()),
        "loss/schema_format_struct_ce": float(out.metrics.get("loss/struct_ce", 0.0)),
        "loss/schema_format_desc_ce": float(out.metrics.get("loss/desc_ce", 0.0)),
        "loss/schema_format_token_ce_struct": float(
            out.metrics.get("loss/token_ce_struct", 0.0)
        ),
        "loss/schema_format_token_ce_desc": float(
            out.metrics.get("loss/token_ce_desc", 0.0)
        ),
    }
    state = dict(out.state)
    labels_masked = state.get("labels_masked")
    weights_masked = state.get("weights_masked")
    if isinstance(labels_masked, torch.Tensor) and isinstance(weights_masked, torch.Tensor):
        labels_clean = labels_masked.clone()
        labels_clean = labels_clean.masked_fill(weights_masked <= 0.0, -100)
        state["labels_masked"] = labels_clean
    state["schema_format_ce"] = schema_loss
    state["schema_format_ce_contrib"] = schema_loss
    return ModuleResult(loss=schema_loss, metrics=metrics, state=state)
