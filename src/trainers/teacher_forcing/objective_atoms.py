"""Projection of teacher-forcing pipeline outputs into Stage-2 objective atoms.

This helper converts the generic `PipelineResult` emitted by the shared teacher-forcing
objective pipeline into the canonical Stage-2 atom keys that trainers log, such as:

- `loss/text/struct_ce`
- `loss/B_rollout_text/desc_ce`
- `loss/coord/bbox_smoothl1`
- `loss/B_coord/coord_soft_ce`

The function is intentionally strict by default (`require_additive=True`): it verifies
that projected atom sums reconstruct both per-module weighted losses and the overall
pipeline total, which guards against drift between objective implementation and trainer
logging semantics.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence

import torch

from .contracts import PipelineModuleSpec, PipelineResult
from .module_registry import OBJECTIVE_MODULE_CATALOG, ObjectiveModuleDefinition


def _isfinite_scalar(t: torch.Tensor) -> bool:
    if not isinstance(t, torch.Tensor) or t.numel() != 1:
        return False
    return bool(torch.isfinite(t).all().item())


def _as_scalar_tensor(value: Any) -> Optional[torch.Tensor]:
    if not isinstance(value, torch.Tensor):
        return None
    if value.numel() != 1:
        return None
    return value


def _assert_allclose(
    *,
    where: str,
    got: torch.Tensor,
    expected: torch.Tensor,
    rtol: float,
    atol: float,
) -> None:
    if not _isfinite_scalar(got) or not _isfinite_scalar(expected):
        raise ValueError(
            f"Stage2 atom projection expected finite scalar tensors at {where}; "
            f"got={type(got)} shape={getattr(got, 'shape', None)} "
            f"expected={type(expected)} shape={getattr(expected, 'shape', None)}"
        )
    got_f = got.detach().float()
    exp_f = expected.detach().float()
    if not torch.allclose(got_f, exp_f, rtol=rtol, atol=atol):
        diff = float((got_f - exp_f).abs().max().item())
        exp_abs = float(exp_f.abs().max().item())
        raise ValueError(
            f"Stage2 atom projection mismatch ({where}): "
            f"diff={diff:.6g} expected_abs={exp_abs:.6g} got={float(got_f.item()):.6g} expected={float(exp_f.item()):.6g}"
        )


def _parse_objective_specs(
    objective_specs: Optional[Sequence[Mapping[str, Any]]],
) -> Dict[str, PipelineModuleSpec]:
    specs: Dict[str, PipelineModuleSpec] = {}
    for raw in list(objective_specs or []):
        if not isinstance(raw, Mapping):
            continue
        parsed = PipelineModuleSpec.from_mapping(raw)
        if not parsed.name:
            continue
        specs.setdefault(parsed.name, parsed)
    return specs


def _resolve_emission_context(
    definition: ObjectiveModuleDefinition,
    *,
    text_provenance: Optional[str],
    coord_provenance: Optional[str],
    emit_text: bool,
    emit_coord: bool,
) -> tuple[bool, Optional[str]]:
    if definition.emission_group == "text":
        return bool(emit_text and text_provenance), text_provenance
    if definition.emission_group == "coord":
        return bool(emit_coord and coord_provenance), coord_provenance
    return True, None


def project_stage2_objective_atoms(
    *,
    pipeline_result: PipelineResult,
    objective_specs: Optional[Sequence[Mapping[str, Any]]],
    text_provenance: Optional[str],
    coord_provenance: Optional[str],
    emit_text: bool = True,
    emit_coord: bool = True,
    require_additive: bool = True,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> Dict[str, float]:
    """Project a teacher-forcing PipelineResult into stage2 objective atoms.

    The returned dict is ready to be merged into the trainer logs.

    Args:
      pipeline_result: Output of `run_teacher_forcing_pipeline`.
      objective_specs: The exact objective spec list passed to the pipeline.
      text_provenance: Provenance key for token-CE atoms (e.g. "B_rollout_text").
      coord_provenance: Provenance key for bbox/coord atoms (e.g. "B_coord").
      emit_text: If False, asserts token-CE weighted loss is zero.
      emit_coord: If False, asserts bbox/coord weighted losses are zero.
      require_additive: If True, validates strict additivity vs pipeline losses.

    Returns:
      Mapping from `loss/<prov>/<atom>` to float contribution.
    """

    specs = _parse_objective_specs(objective_specs)
    state: MutableMapping[str, Any] = dict(pipeline_result.state or {})

    atoms_t: Dict[str, torch.Tensor] = {}

    def _spec(name: str) -> PipelineModuleSpec:
        spec = specs.get(name)
        if spec is None:
            raise ValueError(
                f"stage2 atom projection requires objective spec for module '{name}'"
            )
        return spec

    def _weighted(atom: torch.Tensor, *, module_weight: float) -> torch.Tensor:
        return atom * float(module_weight)

    def _maybe_add(key: str, t: torch.Tensor) -> None:
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"Expected tensor objective atom for {key}; got {type(t)}")
        if t.numel() != 1:
            raise ValueError(f"Expected scalar objective atom for {key}; got shape={tuple(t.shape)}")
        if key in atoms_t:
            raise ValueError(
                f"Stage2 atom projection encountered duplicate projected atom key: {key}"
            )
        atoms_t[key] = t

    module_losses = dict(pipeline_result.module_losses or {})

    for module_name, weighted_loss in module_losses.items():
        definition = OBJECTIVE_MODULE_CATALOG.get(module_name)
        if definition is None:
            raise ValueError(
                f"stage2 atom projection does not know how to project module '{module_name}'"
            )

        spec = _spec(module_name)
        module_weight = float(spec.weight)
        scalar_weighted_loss = _as_scalar_tensor(weighted_loss)
        if scalar_weighted_loss is None:
            raise ValueError(
                f"pipeline_result.module_losses['{module_name}'] must be a scalar tensor"
            )

        emission_enabled, provenance = _resolve_emission_context(
            definition,
            text_provenance=text_provenance,
            coord_provenance=coord_provenance,
            emit_text=emit_text,
            emit_coord=emit_coord,
        )

        if not emission_enabled:
            if require_additive:
                _assert_allclose(
                    where=f"{module_name} disabled emission",
                    got=scalar_weighted_loss,
                    expected=scalar_weighted_loss.new_tensor(0.0),
                    rtol=rtol,
                    atol=atol,
                )
            continue

        if module_weight == 0.0 or provenance is None:
            continue

        terms: Dict[str, torch.Tensor] = {}
        for atom in definition.projected_atoms:
            t = _as_scalar_tensor(state.get(atom.state_key))
            if t is None:
                if require_additive and atom.required_state:
                    raise ValueError(
                        f"{module_name} module did not expose '{atom.state_key}' tensor in pipeline state"
                    )
                continue
            terms[atom.atom_name] = t

        for atom_name, t in terms.items():
            _maybe_add(
                f"loss/{str(provenance)}/{atom_name}",
                _weighted(t, module_weight=module_weight),
            )

        if require_additive:
            total = scalar_weighted_loss.new_tensor(0.0)
            for t in terms.values():
                total = total + _weighted(t, module_weight=module_weight)
            _assert_allclose(
                where=module_name,
                got=total,
                expected=scalar_weighted_loss,
                rtol=rtol,
                atol=atol,
            )

    if require_additive:
        total_atoms = None
        for t in atoms_t.values():
            total_atoms = t if total_atoms is None else (total_atoms + t)
        if total_atoms is None:
            total_atoms = pipeline_result.total_loss.new_tensor(0.0)
        _assert_allclose(
            where="pipeline_total",
            got=total_atoms,
            expected=pipeline_result.total_loss,
            rtol=rtol,
            atol=atol,
        )

    out: Dict[str, float] = {}
    for k, t in atoms_t.items():
        v = float(t.detach().cpu().item())
        if v == 0.0:
            continue
        out[str(k)] = v

    return out
