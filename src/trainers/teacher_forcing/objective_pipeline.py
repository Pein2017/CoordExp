from __future__ import annotations

import logging
from typing import Any, Mapping, MutableMapping, Sequence

import torch

from .contracts import ModuleResult, PipelineModuleSpec, PipelineResult, TeacherForcingContext
from .modules import (
    run_bbox_geo_module,
    run_coord_diag_module,
    run_coord_reg_module,
    run_token_ce_module,
)

logger = logging.getLogger(__name__)


def _coerce_specs(specs: Sequence[Mapping[str, Any]] | None) -> list[PipelineModuleSpec]:
    out: list[PipelineModuleSpec] = []
    for spec in list(specs or []):
        if not isinstance(spec, Mapping):
            continue
        parsed = PipelineModuleSpec.from_mapping(spec)
        if not parsed.name:
            continue
        out.append(parsed)
    return out


def run_teacher_forcing_pipeline(
    *,
    context: TeacherForcingContext,
    objective_specs: Sequence[Mapping[str, Any]] | None,
    diagnostics_specs: Sequence[Mapping[str, Any]] | None,
    initial_state: Mapping[str, Any] | None = None,
    warn_once_cache: set[str] | None = None,
) -> PipelineResult:
    obj_specs = _coerce_specs(objective_specs)
    diag_specs = _coerce_specs(diagnostics_specs)

    total = context.logits.new_tensor(0.0)
    module_losses: dict[str, torch.Tensor] = {}
    metrics: dict[str, float] = {}
    state: dict[str, Any] = dict(initial_state or {})

    precomputed_module_outputs: dict[str, ModuleResult] = {}
    raw_precomputed_obj = state.pop("_precomputed_module_outputs", {})
    if isinstance(raw_precomputed_obj, Mapping):
        for name, out in raw_precomputed_obj.items():
            if isinstance(out, ModuleResult):
                precomputed_module_outputs[str(name)] = out

    precomputed_diag_outputs: dict[str, ModuleResult] = {}
    raw_precomputed_diag = state.pop("_precomputed_diagnostics", {})
    if isinstance(raw_precomputed_diag, Mapping):
        for name, out in raw_precomputed_diag.items():
            if isinstance(out, ModuleResult):
                precomputed_diag_outputs[str(name)] = out

    objective_registry = {
        "token_ce": lambda spec: run_token_ce_module(context=context, spec=spec),
        "bbox_geo": lambda spec: run_bbox_geo_module(context=context, spec=spec),
        "coord_reg": lambda spec: run_coord_reg_module(context=context, spec=spec, state=state),
    }

    diag_registry = {
        "coord_diag": lambda spec: run_coord_diag_module(context=context, spec=spec, state=state),
    }

    for spec in obj_specs:
        if not spec.enabled_for_channel(context.channel):
            continue
        out = precomputed_module_outputs.get(spec.name)
        if out is None:
            module_fn = objective_registry.get(spec.name)
            if module_fn is None:
                raise ValueError(f"unknown objective module: {spec.name}")
            out = module_fn(spec)
        weighted_loss = out.loss * float(spec.weight)
        total = total + weighted_loss

        module_losses[spec.name] = weighted_loss
        if out.metrics:
            for k, v in out.metrics.items():
                metrics[str(k)] = float(v)
        metrics[f"loss/{spec.name}_obj"] = float(weighted_loss.detach().cpu().item())

        if out.state:
            state.update(dict(out.state))

    warn_cache = warn_once_cache if warn_once_cache is not None else set()
    for spec in diag_specs:
        if not spec.enabled_for_channel(context.channel):
            continue
        out = precomputed_diag_outputs.get(spec.name)
        if out is None:
            module_fn = diag_registry.get(spec.name)
            if module_fn is None:
                raise ValueError(f"unknown diagnostics module: {spec.name}")

            try:
                out = module_fn(spec)
            except Exception as exc:
                key = f"{spec.name}:{type(exc).__name__}:{str(exc)}"
                if key not in warn_cache:
                    logger.warning(
                        "teacher-forcing diagnostics module %s disabled for current run after failure: %s",
                        spec.name,
                        exc,
                    )
                    warn_cache.add(key)
                metrics[f"diag/{spec.name}_failed"] = 1.0
                state[f"diag/{spec.name}_failed"] = True
                out = None

        if out is None:
            continue

        if out.metrics:
            for k, v in out.metrics.items():
                metrics[str(k)] = float(v)
        if out.state:
            state.update(dict(out.state))

    return PipelineResult(
        total_loss=total,
        module_losses=module_losses,
        metrics=metrics,
        state=state,
    )
