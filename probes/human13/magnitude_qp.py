#!/usr/bin/env python3
"""Exact magnitude-only Human13 tangent operator mechanics.

This experiment-local runner deliberately stops before the dual QP and finite
candidate replay.  It reuses the existing Human13 request/materialization
path, freezes vision inputs with one ordinary Source forward, and applies
exact forward AD only to the language model.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
from typing import Any

import torch
from torch import Tensor, nn
from src.qwen.inspection import CaptureInputs, resolve_text_stack
from src.adapters.dora import select_dora_parameters


from probes.human13 import output_qp as same_panel


EXPECTED_MAGNITUDE_VECTOR_COUNT = 196
EXPECTED_MAGNITUDE_SCALAR_COUNT = 573_440
EXACT_DERIVATIVE_METHOD = "torch.func.vjp_jvp"
PASS_DISPOSITION = "OPERATOR_PASS"
FAIL_DISPOSITIONS = frozenset({"MECHANICAL_INVALID", "JVP_ORACLE_HOLD"})


class MechanicalInvalid(RuntimeError):
    """The bound Source or operator violates a mechanics contract."""


class JVPOracleHold(RuntimeError):
    """The production seam cannot provide an admitted exact forward product."""


def _tensor_sha256(tensor: Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _maximum_abs_difference(left: Tensor, right: Tensor) -> float:
    if left.shape != right.shape:
        raise MechanicalInvalid(
            f"tensor shape mismatch: {tuple(left.shape)} != {tuple(right.shape)}"
        )
    if left.numel() == 0:
        return 0.0
    return float(
        (left.detach().to(torch.float64) - right.detach().to(torch.float64)).abs().max()
    )


def _effective_output_head(
    head: nn.Module,
) -> tuple[Tensor, Tensor | None, dict[str, Any]]:
    """Snapshot the actual frozen linear rows, including selected-token deltas."""

    weight = getattr(head, "weight", None)
    bias = getattr(head, "bias", None)
    if not isinstance(weight, Tensor) or weight.ndim != 2:
        raise MechanicalInvalid("output head has no matrix weight")
    if bias is not None and (
        not isinstance(bias, Tensor) or bias.shape != weight.shape[:1]
    ):
        raise MechanicalInvalid("output head bias layout differs")
    selected_ids = getattr(head, "selected_token_ids", None)
    selected_delta = getattr(head, "shared_embed_delta", None)
    if selected_ids is None and selected_delta is None:
        return (
            weight.detach(),
            None if bias is None else bias.detach(),
            {
                "semantics": "direct_linear",
                "selected_token_count": 0,
            },
        )
    if not (
        isinstance(selected_ids, Tensor)
        and isinstance(selected_delta, Tensor)
        and selected_ids.ndim == 1
        and selected_delta.ndim == 2
        and selected_ids.numel() == selected_delta.shape[0]
        and selected_delta.shape[1] == weight.shape[1]
        and bool(torch.isfinite(selected_delta).all())
    ):
        raise MechanicalInvalid("selected-token output-head delta layout differs")
    ids = selected_ids.to(device=weight.device, dtype=torch.long)
    if (
        len(set(int(value) for value in ids.cpu().tolist())) != ids.numel()
        or bool(torch.any(ids < 0))
        or bool(torch.any(ids >= weight.shape[0]))
    ):
        raise MechanicalInvalid("selected-token output-head ids are invalid")
    effective = weight.detach().clone()
    effective.index_add_(
        0, ids, selected_delta.detach().to(device=weight.device, dtype=weight.dtype)
    )
    return (
        effective,
        None if bias is None else bias.detach(),
        {
            "semantics": "selected_delta_output_head",
            "selected_token_count": int(ids.numel()),
            "selected_token_ids_sha256": same_panel.sha256_json(
                tuple(int(value) for value in ids.cpu().tolist())
            ),
            "selected_delta_sha256": _tensor_sha256(selected_delta),
        },
    )


@dataclass(frozen=True)
class MagnitudeSurface:
    model: nn.Module = field(repr=False, compare=False)
    language_module_name: str
    names: tuple[str, ...]
    functional_names: tuple[str, ...]
    parameters: tuple[nn.Parameter, ...] = field(repr=False, compare=False)
    shapes: tuple[tuple[int, ...], ...]
    tensor_hashes: tuple[str, ...]
    scalar_count: int

    def receipt(self) -> dict[str, Any]:
        return {
            "language_module_name": self.language_module_name,
            "magnitude_vector_count": len(self.names),
            "magnitude_scalar_count": self.scalar_count,
            "ordered_names": list(self.names),
            "ordered_shapes": [list(shape) for shape in self.shapes],
            "ordered_tensor_sha256": list(self.tensor_hashes),
            "all_non_magnitude_parameters_frozen": True,
            "unmerged": True,
        }


def _language_module(
    model: nn.Module, language_module_name: str | None = None
) -> tuple[str, nn.Module]:
    if language_module_name is not None:
        try:
            module = model.get_submodule(language_module_name)
        except AttributeError as exc:
            raise MechanicalInvalid(
                f"missing language module {language_module_name!r}"
            ) from exc
        return language_module_name, module
    language = resolve_text_stack(model).language_model
    matches = tuple((name, module) for name, module in model.named_modules() if module is language)
    if len(matches) != 1:
        raise MechanicalInvalid("resolved language module has no unique model name")
    return matches[0]


def bind_magnitude_surface(
    model: nn.Module,
    *,
    expected_vector_count: int = EXPECTED_MAGNITUDE_VECTOR_COUNT,
    expected_scalar_count: int = EXPECTED_MAGNITUDE_SCALAR_COUNT,
    language_module_name: str | None = None,
    adapter_name: str = "default",
) -> MagnitudeSurface:
    """Freeze the model, then admit exactly the registered language magnitudes."""

    language_name, _ = _language_module(model, language_module_name)
    model.requires_grad_(False)
    suffix = f".lora_magnitude_vector.{adapter_name}.weight"
    named = tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if "lora_magnitude_vector" in name
    )
    if len(named) != expected_vector_count:
        raise MechanicalInvalid(
            f"magnitude-vector count differs: {len(named)} != {expected_vector_count}"
        )
    invalid_names = tuple(
        name
        for name, _ in named
        if not name.startswith(f"{language_name}.") or not name.endswith(suffix)
    )
    if invalid_names:
        raise MechanicalInvalid(
            f"non-language or unsupported magnitude parameters: {invalid_names[:4]}"
        )

    selected = tuple(
        (name, parameter)
        for name, parameter in select_dora_parameters(model, towers=("language",), adapter_name=adapter_name)
        if name.endswith(suffix)
    )
    if tuple(name for name, _ in selected) != tuple(name for name, _ in named):
        raise MechanicalInvalid("public parameter selection differs from frozen magnitude order")
    named = selected

    for name, parameter in named:
        if parameter.ndim != 1 or not bool(torch.isfinite(parameter).all()):
            raise MechanicalInvalid(f"invalid finite magnitude vector: {name}")
        owner_name = name.split(".lora_magnitude_vector.", 1)[0]
        owner = model.get_submodule(owner_name)
        if bool(getattr(owner, "merged", False)) or bool(
            getattr(owner, "merged_adapters", ())
        ):
            raise MechanicalInvalid(f"merged DoRA target: {owner_name}")

    scalar_count = sum(parameter.numel() for _, parameter in named)
    if scalar_count != expected_scalar_count:
        raise MechanicalInvalid(
            f"magnitude scalar count differs: {scalar_count} != {expected_scalar_count}"
        )
    for _, parameter in named:
        parameter.requires_grad_(True)
    names = tuple(name for name, _ in named)
    parameters = tuple(parameter for _, parameter in named)
    surface = MagnitudeSurface(
        model=model,
        language_module_name=language_name,
        names=names,
        functional_names=tuple(
            name.removeprefix(f"{language_name}.") for name in names
        ),
        parameters=parameters,
        shapes=tuple(tuple(parameter.shape) for parameter in parameters),
        tensor_hashes=tuple(_tensor_sha256(parameter) for parameter in parameters),
        scalar_count=scalar_count,
    )
    validate_trainable_surface(model, surface)
    return surface


def validate_trainable_surface(
    model: nn.Module, surface: MagnitudeSurface
) -> dict[str, Any]:
    named_parameters = dict(model.named_parameters())
    missing = tuple(name for name in surface.names if name not in named_parameters)
    if missing:
        raise MechanicalInvalid(
            f"registered magnitude parameters are missing: {missing[:4]}"
        )
    trainable = tuple(
        name for name, parameter in named_parameters.items() if parameter.requires_grad
    )
    if trainable != surface.names:
        unexpected = tuple(name for name in trainable if name not in surface.names)
        absent = tuple(name for name in surface.names if name not in trainable)
        raise MechanicalInvalid(
            "trainable surface contains non-magnitude or missing magnitude parameters: "
            f"unexpected={unexpected[:4]} missing={absent[:4]}"
        )
    for name, expected_parameter, expected_shape in zip(
        surface.names, surface.parameters, surface.shapes, strict=True
    ):
        parameter = named_parameters[name]
        if (
            parameter is not expected_parameter
            or tuple(parameter.shape) != expected_shape
        ):
            raise MechanicalInvalid(f"magnitude tensor layout drift: {name}")
        if not bool(torch.isfinite(parameter).all()):
            raise MechanicalInvalid(f"nonfinite magnitude parameter: {name}")
    return {
        "trainable_count": len(trainable),
        "trainable_scalar_count": sum(
            named_parameters[name].numel() for name in trainable
        ),
        "trainable_names": list(trainable),
    }


def magnitude_isometry(
    magnitude_deltas: Sequence[Tensor],
    frozen_unit_rows: Sequence[Tensor],
    *,
    atol: float = 1e-10,
    rtol: float = 1e-10,
) -> dict[str, float]:
    """Verify ||delta m||_2 equals the realized direct-sum Frobenius norm."""

    if not magnitude_deltas or len(magnitude_deltas) != len(frozen_unit_rows):
        raise MechanicalInvalid("magnitude deltas and frozen unit rows differ")
    magnitude_norm = torch.zeros((), dtype=torch.float64)
    realized_norm = torch.zeros((), dtype=torch.float64)
    maximum_row_norm_error = 0.0
    for delta, unit_rows in zip(magnitude_deltas, frozen_unit_rows, strict=True):
        flat_delta = delta.detach().to(torch.float64).reshape(-1)
        rows = unit_rows.detach().to(torch.float64)
        if rows.ndim != 2 or rows.shape[0] != flat_delta.numel():
            raise MechanicalInvalid("unit-row layout differs from its magnitude vector")
        if not bool(torch.isfinite(flat_delta).all() and torch.isfinite(rows).all()):
            raise MechanicalInvalid("nonfinite magnitude-isometry input")
        row_norms = torch.linalg.vector_norm(rows, dim=1)
        maximum_row_norm_error = max(
            maximum_row_norm_error, float((row_norms - 1).abs().max())
        )
        if not torch.allclose(
            row_norms, torch.ones_like(row_norms), atol=atol, rtol=rtol
        ):
            raise MechanicalInvalid("frozen realized-weight rows are not unit norm")
        magnitude_norm = magnitude_norm + torch.dot(flat_delta, flat_delta)
        realized = flat_delta[:, None] * rows
        realized_norm = realized_norm + torch.sum(realized * realized)
    if not torch.allclose(magnitude_norm, realized_norm, atol=atol, rtol=rtol):
        raise MechanicalInvalid("magnitude/realized-weight isometry differs")
    return {
        "magnitude_norm_squared": float(magnitude_norm),
        "realized_weight_frobenius_squared": float(realized_norm),
        "maximum_unit_row_norm_error": maximum_row_norm_error,
    }


@dataclass(frozen=True)
class OperatorEvaluation:
    states: Tensor = field(repr=False)
    magnitude_direction: tuple[Tensor, ...] = field(repr=False)
    state_tangent: Tensor = field(repr=False)
    derivative_method: str
    source_max_abs_difference: float
    jvp_source_max_abs_difference: float
    covector_dot_state_tangent: float
    direction_norm_squared: float
    relative_adjoint_error: float
    adjoint_atol: float
    adjoint_rtol: float

    def receipt(self) -> dict[str, Any]:
        return {
            "derivative_method": self.derivative_method,
            "exact_forward_ad": True,
            "state_shape": list(self.states.shape),
            "magnitude_direction_shapes": [
                list(item.shape) for item in self.magnitude_direction
            ],
            "source_max_abs_difference": self.source_max_abs_difference,
            "jvp_source_max_abs_difference": self.jvp_source_max_abs_difference,
            "covector_dot_state_tangent": self.covector_dot_state_tangent,
            "direction_norm_squared": self.direction_norm_squared,
            "relative_adjoint_error": self.relative_adjoint_error,
            "adjoint_atol": self.adjoint_atol,
            "adjoint_rtol": self.adjoint_rtol,
            "model_forward_count": 2,
            "reverse_product_count": 1,
            "forward_product_count": 1,
        }


def exact_vjp_jvp(
    state_function: Callable[[tuple[Tensor, ...]], Tensor],
    source_magnitudes: Sequence[Tensor],
    state_covector: Tensor,
    *,
    source_states: Tensor | None = None,
    source_atol: float = 0.0,
    source_rtol: float = 0.0,
    adjoint_atol: float = 2e-5,
    adjoint_rtol: float = 2e-5,
) -> OperatorEvaluation:
    """Evaluate d=J^Tq and exact Jd without admitting finite differences."""

    primals = tuple(source_magnitudes)
    if not primals or any(not bool(torch.isfinite(item).all()) for item in primals):
        raise MechanicalInvalid("source magnitude coordinates are empty or nonfinite")

    def wrapped(*values: Tensor) -> Tensor:
        return state_function(tuple(values))

    try:
        states, pullback = torch.func.vjp(wrapped, *primals)
    except (NotImplementedError, RuntimeError) as exc:
        raise MechanicalInvalid(f"exact VJP unavailable: {exc}") from exc
    if states.shape != state_covector.shape:
        raise MechanicalInvalid(
            "state covector shape differs: "
            f"{tuple(state_covector.shape)} != {tuple(states.shape)}"
        )
    if not bool(torch.isfinite(states).all() and torch.isfinite(state_covector).all()):
        raise MechanicalInvalid("nonfinite Source states or state covector")
    source_difference = 0.0
    if source_states is not None:
        source_difference = _maximum_abs_difference(states, source_states)
        if not torch.allclose(
            states, source_states, atol=source_atol, rtol=source_rtol
        ):
            raise MechanicalInvalid(
                f"full-vs-language Source hidden parity differs: {source_difference}"
            )
    magnitude_direction = tuple(pullback(state_covector))
    if len(magnitude_direction) != len(primals) or any(
        not bool(torch.isfinite(item).all()) for item in magnitude_direction
    ):
        raise MechanicalInvalid("VJP returned invalid magnitude coordinates")
    try:
        jvp_states, state_tangent = torch.func.jvp(
            wrapped, primals, magnitude_direction, strict=True
        )
    except (NotImplementedError, RuntimeError) as exc:
        raise JVPOracleHold(f"exact JVP unavailable: {exc}") from exc
    if state_tangent.shape != states.shape or not bool(
        torch.isfinite(state_tangent).all()
    ):
        raise JVPOracleHold("exact JVP returned invalid pre-head states")
    jvp_source_difference = _maximum_abs_difference(states, jvp_states)
    if not torch.allclose(states, jvp_states, atol=source_atol, rtol=source_rtol):
        raise JVPOracleHold(f"exact JVP Source parity differs: {jvp_source_difference}")

    lhs = torch.sum(
        state_covector.detach().to(torch.float64)
        * state_tangent.detach().to(torch.float64)
    )
    rhs = sum(
        torch.sum(item.detach().to(torch.float64).square())
        for item in magnitude_direction
    )
    denominator = max(abs(float(lhs)), abs(float(rhs)), torch.finfo(torch.float64).tiny)
    relative_error = abs(float(lhs - rhs)) / denominator
    if not torch.allclose(lhs, rhs, atol=adjoint_atol, rtol=adjoint_rtol):
        raise JVPOracleHold(
            "VJP/JVP adjoint identity differs: "
            f"q_dot_jd={float(lhs)} norm_squared={float(rhs)} "
            f"relative_error={relative_error}"
        )
    return OperatorEvaluation(
        states=states.detach(),
        magnitude_direction=tuple(item.detach() for item in magnitude_direction),
        state_tangent=state_tangent.detach(),
        derivative_method=EXACT_DERIVATIVE_METHOD,
        source_max_abs_difference=source_difference,
        jvp_source_max_abs_difference=jvp_source_difference,
        covector_dot_state_tangent=float(lhs),
        direction_norm_squared=float(rhs),
        relative_adjoint_error=relative_error,
        adjoint_atol=float(adjoint_atol),
        adjoint_rtol=float(adjoint_rtol),
    )


def active_hidden_covector(
    position_ids: Tensor,
    target_token_ids: Tensor,
    competitor_token_ids: Tensor,
    dual_weights: Tensor,
    head_weight: Tensor,
    *,
    position_count: int,
) -> Tensor:
    """Build q_p=sum lambda(E_t-E_c) for one restricted active set."""

    position_ids = position_ids.to(device=head_weight.device, dtype=torch.long)
    targets = target_token_ids.to(device=head_weight.device, dtype=torch.long)
    competitors = competitor_token_ids.to(device=head_weight.device, dtype=torch.long)
    weights = dual_weights.to(device=head_weight.device, dtype=head_weight.dtype)
    if not (
        position_ids.ndim == targets.ndim == competitors.ndim == weights.ndim == 1
        and position_ids.numel()
        == targets.numel()
        == competitors.numel()
        == weights.numel()
    ):
        raise MechanicalInvalid("active-constraint arrays differ")
    vocab_width = int(head_weight.shape[0])
    if (
        position_count <= 0
        or bool(torch.any(position_ids < 0))
        or bool(torch.any(position_ids >= position_count))
        or bool(torch.any(targets < 0))
        or bool(torch.any(targets >= vocab_width))
        or bool(torch.any(competitors < 0))
        or bool(torch.any(competitors >= vocab_width))
        or bool(torch.any(targets == competitors))
        or bool(torch.any(weights < 0))
        or not bool(torch.isfinite(weights).all())
    ):
        raise MechanicalInvalid("invalid active constraint identity or weight")
    rows = head_weight.index_select(0, targets) - head_weight.index_select(
        0, competitors
    )
    covector = torch.zeros(
        (position_count, head_weight.shape[1]),
        dtype=head_weight.dtype,
        device=head_weight.device,
    )
    covector.index_add_(0, position_ids, weights[:, None] * rows)
    return covector


@dataclass(frozen=True)
class VocabularySeparation:
    competitor_token_ids: tuple[int, ...]
    competitor_logits: tuple[float, ...]
    predicted_margins: tuple[float, ...]
    vocabulary_width: int
    scanned_row_count: int
    chunk_count: int
    certificate_eligible: bool

    def receipt(self) -> dict[str, Any]:
        return {
            "scan_mode": (
                "exhaustive_full_vocabulary"
                if self.certificate_eligible
                else "candidate_only_diagnostic"
            ),
            "certificate_eligible": self.certificate_eligible,
            "vocabulary_width": self.vocabulary_width,
            "scanned_row_count": self.scanned_row_count,
            "chunk_count": self.chunk_count,
            "competitor_token_ids": list(self.competitor_token_ids),
            "competitor_logits": list(self.competitor_logits),
            "predicted_margins": list(self.predicted_margins),
        }


def separate_vocabulary(
    source_states: Tensor,
    state_tangent: Tensor,
    head_weight: Tensor,
    target_token_ids: Tensor,
    *,
    source_logits: Tensor | None = None,
    head_bias: Tensor | None = None,
    chunk_size: int,
    candidate_token_ids: Sequence[int] | None = None,
    expected_vocabulary_width: int | None = None,
) -> VocabularySeparation:
    """Scan frozen head rows; candidate-only mode is never certificate eligible."""

    if source_states.ndim != 2 or source_states.shape != state_tangent.shape:
        raise MechanicalInvalid("Source/tangent pre-head states differ")
    if head_weight.ndim != 2 or head_weight.shape[1] != source_states.shape[1]:
        raise MechanicalInvalid("frozen output-head layout differs")
    if chunk_size <= 0:
        raise MechanicalInvalid("vocabulary chunk size must be positive")
    vocab_width = int(head_weight.shape[0])
    if (
        expected_vocabulary_width is not None
        and vocab_width != expected_vocabulary_width
    ):
        raise MechanicalInvalid("tokenizer/output-head vocabulary width differs")
    targets = target_token_ids.to(device=head_weight.device, dtype=torch.long)
    if targets.ndim != 1 or targets.numel() != source_states.shape[0]:
        raise MechanicalInvalid("target-token layout differs from decision states")
    if bool(torch.any(targets < 0)) or bool(torch.any(targets >= vocab_width)):
        raise MechanicalInvalid("target token is outside the bound vocabulary")
    if head_bias is not None and tuple(head_bias.shape) != (vocab_width,):
        raise MechanicalInvalid("output-head bias layout differs")
    if source_logits is not None and tuple(source_logits.shape) != (
        source_states.shape[0],
        vocab_width,
    ):
        raise MechanicalInvalid("Source full-vocabulary logits layout differs")
    if not bool(
        torch.isfinite(source_states).all()
        and torch.isfinite(state_tangent).all()
        and torch.isfinite(head_weight).all()
        and (head_bias is None or torch.isfinite(head_bias).all())
        and (source_logits is None or torch.isfinite(source_logits).all())
    ):
        raise MechanicalInvalid("nonfinite separator input")

    source_states = source_states.to(head_weight.dtype)
    state_tangent = state_tangent.to(head_weight.dtype)
    predicted_states = source_states + state_tangent
    target_rows = head_weight.index_select(0, targets)
    if source_logits is None:
        target_logits = torch.sum(predicted_states * target_rows, dim=1)
        if head_bias is not None:
            target_logits = target_logits + head_bias.index_select(0, targets)
    else:
        target_logits = source_logits.to(head_weight.dtype).gather(1, targets[:, None])[
            :, 0
        ] + torch.sum(state_tangent * target_rows, dim=1)

    exhaustive = candidate_token_ids is None
    if exhaustive:
        candidates = None
        scanned_row_count = vocab_width
    else:
        raw = tuple(int(value) for value in candidate_token_ids)
        if (
            not raw
            or len(set(raw)) != len(raw)
            or min(raw) < 0
            or max(raw) >= vocab_width
        ):
            raise MechanicalInvalid(
                "candidate vocabulary rows are empty, duplicated, or invalid"
            )
        candidates = torch.tensor(
            sorted(raw), dtype=torch.long, device=head_weight.device
        )
        scanned_row_count = len(raw)

    best_logits = torch.full(
        (source_states.shape[0],),
        -torch.inf,
        dtype=predicted_states.dtype,
        device=predicted_states.device,
    )
    best_ids = torch.full_like(targets, vocab_width)
    chunk_count = 0
    for start in range(0, scanned_row_count, chunk_size):
        if candidates is None:
            ids = torch.arange(
                start,
                min(start + chunk_size, vocab_width),
                dtype=torch.long,
                device=head_weight.device,
            )
        else:
            ids = candidates[start : start + chunk_size]
        rows = head_weight.index_select(0, ids)
        if source_logits is None:
            logits = predicted_states @ rows.transpose(0, 1)
            if head_bias is not None:
                logits = logits + head_bias.index_select(0, ids)
        else:
            logits = source_logits.to(head_weight.dtype).index_select(
                1, ids
            ) + state_tangent @ rows.transpose(0, 1)
        if not bool(torch.isfinite(logits).all()):
            raise MechanicalInvalid("nonfinite predicted vocabulary logit")
        logits = logits.masked_fill(targets[:, None] == ids[None, :], -torch.inf)
        chunk_logits, offsets = torch.max(logits, dim=1)
        chunk_ids = ids.index_select(0, offsets)
        replace = (chunk_logits > best_logits) | (
            (chunk_logits == best_logits) & (chunk_ids < best_ids)
        )
        best_logits = torch.where(replace, chunk_logits, best_logits)
        best_ids = torch.where(replace, chunk_ids, best_ids)
        chunk_count += 1
    if not bool(torch.isfinite(best_logits).all()) or bool(
        torch.any(best_ids >= vocab_width)
    ):
        raise MechanicalInvalid("vocabulary scan found no non-target competitor")
    margins = target_logits - best_logits
    return VocabularySeparation(
        competitor_token_ids=tuple(int(value) for value in best_ids.cpu().tolist()),
        competitor_logits=tuple(
            float(value) for value in best_logits.float().cpu().tolist()
        ),
        predicted_margins=tuple(
            float(value) for value in margins.float().cpu().tolist()
        ),
        vocabulary_width=vocab_width,
        scanned_row_count=scanned_row_count,
        chunk_count=chunk_count,
        certificate_eligible=exhaustive,
    )


class MagnitudeTransaction:
    """Apply one delta and restore the exact Source magnitudes on every exit."""

    def __init__(
        self,
        surface: MagnitudeSurface,
        deltas: Sequence[Tensor],
        *,
        protected_parameters: Mapping[str, Tensor] | None = None,
    ) -> None:
        self.surface = surface
        self.deltas = tuple(deltas)
        self.protected_parameters = dict(protected_parameters or {})
        self.source_restored = False
        self._source: tuple[Tensor, ...] = ()
        self._source_hashes: tuple[str, ...] = ()
        self._protected_hashes: dict[str, str] = {}
        self._active = False

    def __enter__(self) -> "MagnitudeTransaction":
        if self._active or len(self.deltas) != len(self.surface.parameters):
            raise MechanicalInvalid("magnitude transaction layout differs")
        for parameter, delta in zip(self.surface.parameters, self.deltas, strict=True):
            if parameter.shape != delta.shape or not bool(torch.isfinite(delta).all()):
                raise MechanicalInvalid("invalid finite magnitude transaction delta")
        magnitude_ids = {id(item) for item in self.surface.parameters}
        if any(
            id(item) in magnitude_ids for item in self.protected_parameters.values()
        ):
            raise MechanicalInvalid("protected non-magnitude set contains a magnitude")
        self._source = tuple(item.detach().clone() for item in self.surface.parameters)
        self._source_hashes = tuple(_tensor_sha256(item) for item in self._source)
        self._protected_hashes = {
            name: _tensor_sha256(parameter)
            for name, parameter in self.protected_parameters.items()
        }
        try:
            with torch.no_grad():
                for parameter, source, delta in zip(
                    self.surface.parameters, self._source, self.deltas, strict=True
                ):
                    parameter.copy_(
                        source + delta.to(device=source.device, dtype=source.dtype)
                    )
        except BaseException:
            self._restore()
            raise
        self._active = True
        return self

    def _restore(self) -> None:
        with torch.no_grad():
            for parameter, source in zip(
                self.surface.parameters, self._source, strict=True
            ):
                parameter.copy_(source)
        magnitude_hashes = tuple(
            _tensor_sha256(parameter) for parameter in self.surface.parameters
        )
        protected_hashes = {
            name: _tensor_sha256(parameter)
            for name, parameter in self.protected_parameters.items()
        }
        self.source_restored = (
            magnitude_hashes == self._source_hashes
            and protected_hashes == self._protected_hashes
        )
        self._active = False
        if not self.source_restored:
            raise MechanicalInvalid(
                "Source restoration or non-magnitude identity differs"
            )

    def __exit__(self, *_exc: object) -> bool:
        self._restore()
        return False


def mechanics_disposition(error: BaseException) -> str:
    return (
        "JVP_ORACLE_HOLD" if isinstance(error, JVPOracleHold) else "MECHANICAL_INVALID"
    )


def publish_mechanics_receipt(
    path: Path,
    *,
    disposition: str,
    bindings: Mapping[str, Any],
    surface: Mapping[str, Any],
    operator: Mapping[str, Any] | None,
    separation: Mapping[str, Any] | None,
    source_restored: bool,
    measurements: Mapping[str, Any] | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    """Publish one immutable receipt; only exact exhaustive evidence can pass."""

    if disposition not in FAIL_DISPOSITIONS | {PASS_DISPOSITION}:
        raise MechanicalInvalid(f"unknown mechanics disposition: {disposition}")
    certificate_eligible = disposition == PASS_DISPOSITION
    if certificate_eligible:
        if not bindings:
            raise MechanicalInvalid("bound Source identity is required for PASS")
        if (
            operator is None
            or operator.get("derivative_method") != EXACT_DERIVATIVE_METHOD
        ):
            raise MechanicalInvalid(
                "finite differences cannot produce exact-certificate status"
            )
        if separation is None or separation.get("certificate_eligible") is not True:
            raise MechanicalInvalid(
                "an exhaustive vocabulary scan is required for PASS"
            )
        if not source_restored:
            raise MechanicalInvalid("Source restoration is required for PASS")
    body: dict[str, Any] = {
        "schema_version": "human13_dora_magnitude_operator.v1",
        "disposition": disposition,
        "certificate_eligible": certificate_eligible,
        "bindings": dict(bindings),
        "surface": dict(surface),
        "operator": None if operator is None else dict(operator),
        "separation": None if separation is None else dict(separation),
        "source_restored": bool(source_restored),
        "measurements": dict(measurements or {}),
        "reason": reason,
    }
    body["content_sha256"] = same_panel.sha256_json(body)
    same_panel.immutable_json(path, body)
    return body




@dataclass
class ProductionOperatorBinding:
    context: Any = field(repr=False)
    session: Any = field(repr=False)
    surface: MagnitudeSurface
    state_function: Callable[[tuple[Tensor, ...]], Tensor] = field(repr=False)
    source_magnitudes: tuple[Tensor, ...] = field(repr=False)
    source_states: Tensor = field(repr=False)
    source_logits: Tensor = field(repr=False)
    head_weight: Tensor = field(repr=False)
    head_bias: Tensor | None = field(repr=False)
    head_receipt: Mapping[str, Any]
    target_token_ids: Tensor = field(repr=False)
    image_id: int
    vocabulary_width: int
    route_receipt: Mapping[str, Any]
    binding_receipt_sha256: str
    runtime_identity: Mapping[str, Any]
    executed_prompt_token_ids_sha256: str
    full_language_source_max_abs_difference: float
    actual_head_replay_max_abs_difference: float
    collapsed_effective_head_max_abs_difference: float
    _closed: bool = False

    def __enter__(self) -> "ProductionOperatorBinding":
        return self

    def close(self) -> None:
        if not self._closed:
            self.context.__exit__(None, None, None)
            self._closed = True

    def __exit__(self, *_exc: object) -> bool:
        self.close()
        return False


def open_production_operator(image_id: int) -> ProductionOperatorBinding:
    """Capture fixed vision inputs, then expose the exact language-only state map."""

    binding_receipt = same_panel.check_bindings(load_tokenizer=False)
    context, session, (_example, request), runtime_identity, components = same_panel._runtime_setup(
        image_id
    )
    try:
        panel = {int(row["image_id"]): row for row in same_panel.load_panel()}
        if image_id not in panel:
            raise MechanicalInvalid(f"image {image_id} is not in the Human13 panel")
        route_ids, route_receipt = same_panel.tokenize_canonical_route(
            components.tokenizer, panel[image_id]
        )
        model_inputs, executed_ids, decision_positions = same_panel.prepare_decision_history(
            components, request, route_ids
        )
        model = components.model
        language_name, language = _language_module(model)
        surface = bind_magnitude_surface(model, language_module_name=language_name)
        head = same_panel._output_head(model)
        if any(parameter.requires_grad for parameter in head.parameters()):
            raise MechanicalInvalid("output head is not frozen")

        with CaptureInputs(language) as language_capture, CaptureInputs(head) as head_capture:
            with torch.no_grad():
                output = model(**model_inputs)
        if len(head_capture.args) != 1 or not isinstance(head_capture.args[0], Tensor):
            raise MechanicalInvalid("output head input capture differs")
        source_states = head_capture.args[0]
        if source_states.ndim != 3 or tuple(source_states.shape[:2]) != (
            1,
            len(route_ids),
        ):
            raise MechanicalInvalid("captured pre-head decision-state layout differs")
        source_states = source_states[0]
        source_logits = output.logits[0].detach()
        if source_logits.shape[0] != len(route_ids):
            raise MechanicalInvalid("Source logits do not cover canonical decisions")
        if not bool(
            torch.isfinite(source_states).all() and torch.isfinite(source_logits).all()
        ):
            raise MechanicalInvalid("Source hidden states or logits are nonfinite")
        source_magnitudes = tuple(
            parameter.detach().clone() for parameter in surface.parameters
        )
        language_args = language_capture.args
        language_kwargs = language_capture.kwargs

        def state_function(magnitudes: tuple[Tensor, ...]) -> Tensor:
            if len(magnitudes) != len(surface.functional_names):
                raise MechanicalInvalid("functional magnitude layout differs")
            replacements = dict(zip(surface.functional_names, magnitudes, strict=True))
            language_output = torch.func.functional_call(
                language,
                replacements,
                language_args,
                language_kwargs,
                strict=False,
            )
            hidden = getattr(language_output, "last_hidden_state", None)
            if (
                not isinstance(hidden, Tensor)
                or hidden.ndim != 3
                or hidden.shape[0] != 1
            ):
                raise MechanicalInvalid(
                    "language model has no canonical last hidden state"
                )
            return hidden[0].index_select(0, decision_positions)

        with torch.no_grad():
            functional_source = state_function(source_magnitudes)
        full_language_difference = _maximum_abs_difference(
            functional_source, source_states
        )
        if full_language_difference != 0.0:
            raise MechanicalInvalid(
                f"full-vs-language hidden parity differs: {full_language_difference}"
            )
        head_weight, head_bias, head_receipt = _effective_output_head(head)
        with torch.no_grad():
            replay_logits = head(source_states.unsqueeze(0))[0]
        actual_head_replay_difference = _maximum_abs_difference(
            replay_logits, source_logits
        )
        if not torch.equal(replay_logits, source_logits):
            raise MechanicalInvalid(
                f"actual output-head replay differs: {actual_head_replay_difference}"
            )
        collapsed_logits = torch.nn.functional.linear(
            source_states, head_weight, head_bias
        )
        collapsed_effective_head_difference = _maximum_abs_difference(
            collapsed_logits.float(), source_logits.float()
        )
        vocabulary_width = int(head_weight.shape[0])
        if vocabulary_width != len(components.tokenizer):
            raise MechanicalInvalid("tokenizer/output-head vocabulary width differs")
        return ProductionOperatorBinding(
            context=context,
            session=session,
            surface=surface,
            state_function=state_function,
            source_magnitudes=source_magnitudes,
            source_states=source_states,
            source_logits=source_logits,
            head_weight=head_weight,
            head_bias=head_bias,
            head_receipt=head_receipt,
            target_token_ids=torch.tensor(
                route_ids, dtype=torch.long, device=head_weight.device
            ),
            image_id=image_id,
            vocabulary_width=vocabulary_width,
            route_receipt=route_receipt,
            binding_receipt_sha256=same_panel.sha256_json(binding_receipt),
            runtime_identity=runtime_identity,
            executed_prompt_token_ids_sha256=same_panel.sha256_json(
                tuple(int(value) for value in executed_ids)
            ),
            full_language_source_max_abs_difference=full_language_difference,
            actual_head_replay_max_abs_difference=actual_head_replay_difference,
            collapsed_effective_head_max_abs_difference=(
                collapsed_effective_head_difference
            ),
        )
    except BaseException:
        context.__exit__(*sys.exc_info())
        raise


def run_operator_image(
    *,
    image_id: int,
    receipt_path: Path,
    chunk_size: int,
    adjoint_atol: float,
    adjoint_rtol: float,
) -> dict[str, Any]:
    """Run one image operator evaluation; the complete N2 smoke is a later gate."""

    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    binding: ProductionOperatorBinding | None = None
    try:
        binding = open_production_operator(image_id)
        with binding:
            positions = torch.arange(
                binding.target_token_ids.numel(), device=binding.head_weight.device
            )
            competitors = (binding.target_token_ids + 1) % binding.vocabulary_width
            dual = torch.linspace(
                1.0,
                2.0,
                positions.numel(),
                dtype=binding.head_weight.dtype,
                device=binding.head_weight.device,
            )
            dual = dual / torch.linalg.vector_norm(dual)
            covector = active_hidden_covector(
                positions,
                binding.target_token_ids,
                competitors,
                dual,
                binding.head_weight,
                position_count=positions.numel(),
            )
            evaluation = exact_vjp_jvp(
                binding.state_function,
                binding.source_magnitudes,
                covector,
                source_states=binding.source_states,
                adjoint_atol=adjoint_atol,
                adjoint_rtol=adjoint_rtol,
            )
            separation = separate_vocabulary(
                evaluation.states,
                evaluation.state_tangent,
                binding.head_weight,
                binding.target_token_ids,
                source_logits=binding.source_logits,
                head_bias=binding.head_bias,
                chunk_size=chunk_size,
                expected_vocabulary_width=binding.vocabulary_width,
            )
            restored = (
                tuple(
                    _tensor_sha256(parameter)
                    for parameter in binding.surface.parameters
                )
                == binding.surface.tensor_hashes
            )
            measurements = {
                "elapsed_seconds": time.perf_counter() - started,
                "peak_host_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "peak_gpu_reserved_bytes": (
                    int(torch.cuda.max_memory_reserved())
                    if torch.cuda.is_available()
                    else 0
                ),
                "full_language_source_max_abs_difference": (
                    binding.full_language_source_max_abs_difference
                ),
                "actual_head_replay_max_abs_difference": (
                    binding.actual_head_replay_max_abs_difference
                ),
                "collapsed_effective_head_max_abs_difference": (
                    binding.collapsed_effective_head_max_abs_difference
                ),
                "vocabulary_chunk_size": chunk_size,
                "source_capture_forward_count": 1,
                "operator_model_forward_count": 2,
                "total_model_forward_count": 3,
                "reverse_product_count": 1,
                "forward_product_count": 1,
                "full_vocabulary_separator_count": 1,
            }
            return publish_mechanics_receipt(
                receipt_path,
                disposition=PASS_DISPOSITION,
                bindings={
                    "image_id": binding.image_id,
                    "route": dict(binding.route_receipt),
                    "binding_receipt_sha256": binding.binding_receipt_sha256,
                    "runtime_identity": dict(binding.runtime_identity),
                    "output_head": dict(binding.head_receipt),
                    "executed_prompt_token_ids_sha256": (
                        binding.executed_prompt_token_ids_sha256
                    ),
                    "vocabulary_width": binding.vocabulary_width,
                },
                surface=binding.surface.receipt(),
                operator=evaluation.receipt(),
                separation=separation.receipt(),
                source_restored=restored,
                measurements=measurements,
            )
    except (MechanicalInvalid, JVPOracleHold) as exc:
        surface = {} if binding is None else binding.surface.receipt()
        return publish_mechanics_receipt(
            receipt_path,
            disposition=mechanics_disposition(exc),
            bindings=(
                {}
                if binding is None
                else {
                    "image_id": binding.image_id,
                    "route": dict(binding.route_receipt),
                    "binding_receipt_sha256": binding.binding_receipt_sha256,
                    "runtime_identity": dict(binding.runtime_identity),
                    "output_head": dict(binding.head_receipt),
                    "executed_prompt_token_ids_sha256": (
                        binding.executed_prompt_token_ids_sha256
                    ),
                    "vocabulary_width": binding.vocabulary_width,
                }
            ),
            surface=surface,
            operator=None,
            separation=None,
            source_restored=(
                False
                if binding is None
                else tuple(
                    _tensor_sha256(parameter)
                    for parameter in binding.surface.parameters
                )
                == binding.surface.tensor_hashes
            ),
            measurements={"elapsed_seconds": time.perf_counter() - started},
            reason=str(exc),
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-bindings", action="store_true")
    subparsers = parser.add_subparsers(dest="command")
    image = subparsers.add_parser("operator-image")
    image.add_argument("--image-id", type=int, required=True)
    image.add_argument("--receipt", type=Path, required=True)
    image.add_argument("--chunk-size", type=int, default=4096)
    image.add_argument("--adjoint-atol", type=float, default=2e-5)
    image.add_argument("--adjoint-rtol", type=float, default=2e-5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.check_bindings:
            if args.command is not None:
                raise MechanicalInvalid(
                    "--check-bindings cannot be combined with a command"
                )
            result = same_panel.check_bindings()
        elif args.command == "operator-image":
            result = run_operator_image(
                image_id=args.image_id,
                receipt_path=args.receipt,
                chunk_size=args.chunk_size,
                adjoint_atol=args.adjoint_atol,
                adjoint_rtol=args.adjoint_rtol,
            )
        else:
            _parser().print_help()
            return 2
        print(json.dumps(result, sort_keys=True))
        return (
            0 if result.get("disposition", PASS_DISPOSITION) == PASS_DISPOSITION else 1
        )
    except (MechanicalInvalid, JVPOracleHold, FileExistsError, ValueError) as exc:
        print(
            json.dumps(
                {"disposition": mechanics_disposition(exc), "reason": str(exc)},
                sort_keys=True,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
