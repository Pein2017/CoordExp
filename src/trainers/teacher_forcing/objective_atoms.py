"""Stage-2 objective atom projection (strictly additive).

This module centralizes the mapping from teacher-forcing pipeline outputs into the
Stage-2 logging contract (provenance-keyed post-weighting objective atoms).

Key property (enforced when requested):
- The projected atoms are *strictly additive* and sum to the pipeline total loss.

Implementation note:
- To avoid duplicated config/weight interpretation in trainers, the teacher-forcing
  objective modules expose per-atom loss contributions as tensors in
  `PipelineResult.state`.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence

import torch

from .contracts import PipelineModuleSpec, PipelineResult


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
            f"Non-finite scalar in stage2 atom projection check ({where}): "
            f"got={got} expected={expected}"
        )

    got_f = got.detach().float().cpu()
    exp_f = expected.detach().float().cpu()
    if not torch.allclose(got_f, exp_f, rtol=float(rtol), atol=float(atol)):
        diff = float((got_f - exp_f).abs().item())
        exp_abs = float(exp_f.abs().item())
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
        atoms_t[key] = t

    module_losses = dict(pipeline_result.module_losses or {})

    # token_ce -> {struct_ce, desc_ce}
    if "token_ce" in module_losses:
        token_spec = _spec("token_ce")
        token_w = float(token_spec.weight)
        weighted_loss = module_losses.get("token_ce")
        if weighted_loss is None:
            raise ValueError("pipeline_result.module_losses missing token_ce")

        if (not emit_text) or not text_provenance:
            if require_additive:
                got = _as_scalar_tensor(weighted_loss)
                if got is None:
                    raise ValueError(
                        "pipeline_result.module_losses['token_ce'] must be a scalar tensor"
                    )
                _assert_allclose(
                    where="token_ce disabled emission",
                    got=got,
                    expected=weighted_loss.new_tensor(0.0),
                    rtol=rtol,
                    atol=atol,
                )
        elif token_w != 0.0:
            struct = _as_scalar_tensor(state.get("token_ce_struct_contrib"))
            desc = _as_scalar_tensor(state.get("token_ce_desc_contrib"))
            if struct is None or desc is None:
                if require_additive:
                    raise ValueError(
                        "token_ce module did not expose token_ce_*_contrib tensors in pipeline state"
                    )
            else:
                _maybe_add(
                    f"loss/{str(text_provenance)}/struct_ce",
                    _weighted(struct, module_weight=token_w),
                )
                _maybe_add(
                    f"loss/{str(text_provenance)}/desc_ce",
                    _weighted(desc, module_weight=token_w),
                )

                if require_additive:
                    _assert_allclose(
                        where="token_ce",
                        got=_weighted(struct + desc, module_weight=token_w),
                        expected=weighted_loss,
                        rtol=rtol,
                        atol=atol,
                    )

    # bbox_geo -> {bbox_smoothl1, bbox_ciou}
    if "bbox_geo" in module_losses:
        bbox_spec = _spec("bbox_geo")
        bbox_w = float(bbox_spec.weight)
        weighted_loss = module_losses.get("bbox_geo")
        if weighted_loss is None:
            raise ValueError("pipeline_result.module_losses missing bbox_geo")

        if (not emit_coord) or not coord_provenance:
            if require_additive:
                got = _as_scalar_tensor(weighted_loss)
                if got is None:
                    raise ValueError(
                        "pipeline_result.module_losses['bbox_geo'] must be a scalar tensor"
                    )
                _assert_allclose(
                    where="bbox_geo disabled emission",
                    got=got,
                    expected=weighted_loss.new_tensor(0.0),
                    rtol=rtol,
                    atol=atol,
                )
        elif bbox_w != 0.0:
            smoothl1 = _as_scalar_tensor(state.get("bbox_smoothl1_contrib"))
            ciou = _as_scalar_tensor(state.get("bbox_ciou_contrib"))
            if smoothl1 is None or ciou is None:
                if require_additive:
                    raise ValueError(
                        "bbox_geo module did not expose bbox_*_contrib tensors in pipeline state"
                    )
            else:
                _maybe_add(
                    f"loss/{str(coord_provenance)}/bbox_smoothl1",
                    _weighted(smoothl1, module_weight=bbox_w),
                )
                _maybe_add(
                    f"loss/{str(coord_provenance)}/bbox_ciou",
                    _weighted(ciou, module_weight=bbox_w),
                )

                if require_additive:
                    _assert_allclose(
                        where="bbox_geo",
                        got=_weighted(smoothl1 + ciou, module_weight=bbox_w),
                        expected=weighted_loss,
                        rtol=rtol,
                        atol=atol,
                    )

    # coord_reg -> {coord_*, *_gate}
    if "coord_reg" in module_losses:
        coord_spec = _spec("coord_reg")
        coord_w = float(coord_spec.weight)
        weighted_loss = module_losses.get("coord_reg")
        if weighted_loss is None:
            raise ValueError("pipeline_result.module_losses missing coord_reg")

        if (not emit_coord) or not coord_provenance:
            if require_additive:
                got = _as_scalar_tensor(weighted_loss)
                if got is None:
                    raise ValueError(
                        "pipeline_result.module_losses['coord_reg'] must be a scalar tensor"
                    )
                _assert_allclose(
                    where="coord_reg disabled emission",
                    got=got,
                    expected=weighted_loss.new_tensor(0.0),
                    rtol=rtol,
                    atol=atol,
                )
        elif coord_w != 0.0:
            contrib_keys = {
                "coord_token_ce": "coord_token_ce_contrib",
                "coord_soft_ce": "coord_soft_ce_contrib",
                "coord_w1": "coord_w1_contrib",
                "coord_el1": "coord_el1_contrib",
                "coord_ehuber": "coord_ehuber_contrib",
                "coord_entropy": "coord_entropy_contrib",
                "coord_gate": "coord_gate_contrib",
                "text_gate": "text_gate_contrib",
            }

            terms: Dict[str, torch.Tensor] = {}
            for atom_name, state_key in contrib_keys.items():
                t = _as_scalar_tensor(state.get(state_key))
                if t is None:
                    if require_additive:
                        raise ValueError(
                            f"coord_reg module did not expose '{state_key}' tensor in pipeline state"
                        )
                    continue
                terms[atom_name] = t

            if terms:
                for atom_name, t in terms.items():
                    _maybe_add(
                        f"loss/{str(coord_provenance)}/{atom_name}",
                        _weighted(t, module_weight=coord_w),
                    )

                if require_additive:
                    total_terms = None
                    for t in terms.values():
                        total_terms = t if total_terms is None else (total_terms + t)
                    if total_terms is None:
                        total_terms = weighted_loss.new_tensor(0.0)
                    _assert_allclose(
                        where="coord_reg",
                        got=_weighted(total_terms, module_weight=coord_w),
                        expected=weighted_loss,
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
