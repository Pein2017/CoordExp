from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import torch

from .contracts import PipelineModuleSpec, PipelineResult, TeacherForcingContext
from .module_registry import DIAGNOSTIC_MODULE_CATALOG, OBJECTIVE_MODULE_CATALOG
from .modules import (
    run_stage2_trie_ce_module,
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


def _validate_registry_coverage(
    registry: Mapping[str, object],
    *,
    allowed: set[str],
    kind: str,
) -> None:
    missing = allowed.difference(registry)
    unexpected = set(registry).difference(allowed)
    if missing or unexpected:
        raise RuntimeError(
            f"{kind} registry is out of sync with loss catalog: "
            f"missing={sorted(missing)} unexpected={sorted(unexpected)}"
        )


def _validate_module_config_keys(
    spec: PipelineModuleSpec,
    *,
    catalog: Mapping[str, object],
    kind: str,
) -> None:
    definition = catalog.get(spec.name)
    if definition is None:
        return
    allowed = {str(key) for key in getattr(definition, "config_keys", frozenset())}
    allowed.update(
        str(key) for key in getattr(definition, "optional_config_keys", frozenset())
    )
    unknown = sorted(str(key) for key in spec.config if str(key) not in allowed)
    if unknown:
        raise ValueError(
            f"{kind} module {spec.name!r} config contains unsupported key(s): "
            + ", ".join(unknown)
        )


def _run_residual_set_correction_module(
    *,
    context: TeacherForcingContext,
    spec: PipelineModuleSpec,
) -> PipelineResult:
    try:
        from .modules.residual_set_correction import run_residual_set_correction_module
    except ModuleNotFoundError as exc:
        expected_missing_modules = {
            f"{__package__}.modules.residual_set_correction",
            "src.trainers.teacher_forcing.modules.residual_set_correction",
        }
        if exc.name not in expected_missing_modules:
            raise
        raise NotImplementedError(
            "residual_set_correction objective is registered for config/import "
            "sanity, but its loss module is implemented by Task 5."
        ) from exc

    return run_residual_set_correction_module(context=context, spec=spec)


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

    total = context.logits.new_tensor(0.0, dtype=torch.float32)
    module_losses: dict[str, torch.Tensor] = {}
    metrics: dict[str, float] = {}
    state: dict[str, Any] = dict(initial_state or {})

    objective_registry = {
        "token_ce": lambda spec: run_token_ce_module(context=context, spec=spec),
        "hard_sft": lambda spec: run_token_ce_module(
            context=context,
            spec=spec,
        ),
        "stage2_trie_ce": lambda spec: run_stage2_trie_ce_module(
            context=context,
            spec=spec,
        ),
        "residual_set_correction": lambda spec: _run_residual_set_correction_module(
            context=context,
            spec=spec,
        ),
    }
    _validate_registry_coverage(
        objective_registry,
        allowed=set(OBJECTIVE_MODULE_CATALOG),
        kind="objective",
    )

    diag_registry = {}
    _validate_registry_coverage(
        diag_registry,
        allowed=set(DIAGNOSTIC_MODULE_CATALOG),
        kind="diagnostic",
    )

    for spec in obj_specs:
        if not spec.enabled_for_channel(context.channel):
            continue
        module_fn = objective_registry.get(spec.name)
        if module_fn is None:
            raise ValueError(f"unknown objective module: {spec.name}")
        _validate_module_config_keys(
            spec,
            catalog=OBJECTIVE_MODULE_CATALOG,
            kind="objective",
        )
        out = module_fn(spec)
        weighted_loss = out.loss * float(spec.weight)
        total = total + weighted_loss

        module_losses[spec.name] = weighted_loss
        if out.metrics:
            for k, v in out.metrics.items():
                metrics[str(k)] = float(v)
        metrics[f"loss/{spec.name}"] = float(weighted_loss.detach().cpu().item())

        if out.state:
            state.update(dict(out.state))

    warn_cache = warn_once_cache if warn_once_cache is not None else set()
    for spec in diag_specs:
        if not spec.enabled_for_channel(context.channel):
            continue
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
