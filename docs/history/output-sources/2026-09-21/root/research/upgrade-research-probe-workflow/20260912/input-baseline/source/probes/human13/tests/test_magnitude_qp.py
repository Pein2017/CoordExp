from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from probes.human13.magnitude_qp import (
    EXACT_DERIVATIVE_METHOD,
    JVPOracleHold,
    MagnitudeTransaction,
    MechanicalInvalid,
    _effective_output_head,
    bind_magnitude_surface,
    exact_vjp_jvp,
    magnitude_isometry,
    mechanics_disposition,
    publish_mechanics_receipt,
    separate_vocabulary,
    validate_trainable_surface,
)
from src.qwen.special_token_embeddings import (
    SelectedDeltaOutputHead,
    SpecialTokenSelection,
)


class _Magnitude(nn.Module):
    def __init__(self, values: tuple[float, ...]) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(values, dtype=torch.float64))


class _DoraTarget(nn.Module):
    def __init__(self, values: tuple[float, ...]) -> None:
        super().__init__()
        self.lora_magnitude_vector = nn.ModuleDict({"default": _Magnitude(values)})
        self.merged = False
        self.merged_adapters: tuple[str, ...] = ()


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.left = _DoraTarget((2.0, 3.0))
        self.model.language_model.right = _DoraTarget((4.0,))
        self.model.language_model.frozen = nn.Parameter(
            torch.tensor([7.0], dtype=torch.float64)
        )
        self.vision = nn.Parameter(torch.tensor([11.0], dtype=torch.float64))


def _state_map(magnitudes: tuple[torch.Tensor, ...]) -> torch.Tensor:
    left, right = magnitudes
    return torch.stack(
        (
            torch.stack((left[0].square() + right[0], left[1] * right[0])),
            torch.stack((left.sum(), right[0].square() - left[0])),
        )
    )


def test_exact_magnitude_operator_tracer_bullet(tmp_path: Path) -> None:
    model = _TinyModel()
    model.model.language_model.right.lora_magnitude_vector[
        "default"
    ].weight.data.zero_()
    surface = bind_magnitude_surface(
        model,
        expected_vector_count=2,
        expected_scalar_count=3,
        language_module_name="model.language_model",
    )
    assert surface.scalar_count == 3
    assert validate_trainable_surface(model, surface)["trainable_count"] == 2

    isometry = magnitude_isometry(
        (
            torch.tensor([0.2, -0.3], dtype=torch.float64),
            torch.tensor([0.4], dtype=torch.float64),
        ),
        (
            torch.tensor([[1.0, 0.0], [0.6, 0.8]], dtype=torch.float64),
            torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64),
        ),
    )
    assert isometry["magnitude_norm_squared"] == pytest.approx(
        isometry["realized_weight_frobenius_squared"]
    )

    model.vision.requires_grad_(True)
    with pytest.raises(MechanicalInvalid, match="non-magnitude"):
        validate_trainable_surface(model, surface)
    model.vision.requires_grad_(False)

    source = tuple(parameter.detach().clone() for parameter in surface.parameters)
    source_states = _state_map(source)
    covector = torch.tensor([[0.2, -0.4], [0.7, 0.1]], dtype=torch.float64)
    evaluation = exact_vjp_jvp(
        _state_map,
        source,
        covector,
        source_states=source_states,
        adjoint_rtol=1e-10,
        adjoint_atol=1e-12,
    )
    assert evaluation.derivative_method == EXACT_DERIVATIVE_METHOD
    assert evaluation.source_max_abs_difference == 0.0
    assert evaluation.relative_adjoint_error < 1e-10

    head_weight = torch.tensor(
        [[0.0, 0.0], [0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [4.0, 5.0]],
        dtype=torch.float64,
    )
    targets = torch.tensor([0, 1])
    seeded_only = separate_vocabulary(
        evaluation.states,
        evaluation.state_tangent,
        head_weight,
        targets,
        chunk_size=2,
        candidate_token_ids=(0, 1, 2),
    )
    exhaustive = separate_vocabulary(
        evaluation.states,
        evaluation.state_tangent,
        head_weight,
        targets,
        chunk_size=2,
    )
    assert seeded_only.certificate_eligible is False
    assert exhaustive.certificate_eligible is True
    assert seeded_only.competitor_token_ids != exhaustive.competitor_token_ids
    assert exhaustive.competitor_token_ids == (4, 4)

    base_head = nn.Linear(2, 5, bias=False, dtype=torch.float64)
    base_head.weight.data.zero_()
    selected_head = SelectedDeltaOutputHead(
        base_head,
        SpecialTokenSelection(token_strings=("<selected>",), token_ids=(4,)),
        nn.Parameter(torch.tensor([[4.0, 5.0]], dtype=torch.float64)),
    )
    effective_head, effective_bias, head_receipt = _effective_output_head(selected_head)
    assert effective_bias is None
    assert head_receipt["semantics"] == "selected_delta_output_head"
    assert torch.equal(effective_head[4], torch.tensor([4.0, 5.0], dtype=torch.float64))

    anchored = separate_vocabulary(
        torch.zeros((1, 2), dtype=torch.float64),
        torch.tensor([[1.0, 1.0]], dtype=torch.float64),
        effective_head,
        torch.tensor([0]),
        source_logits=torch.tensor([[0.0, 0.0, 20.0, 0.0, -20.0]], dtype=torch.float64),
        chunk_size=2,
    )
    assert anchored.competitor_token_ids == (2,)

    before_magnitudes = tuple(item.detach().clone() for item in surface.parameters)
    before_frozen = model.model.language_model.frozen.detach().clone()
    transaction: MagnitudeTransaction
    with pytest.raises(RuntimeError, match="toy failure"):
        with MagnitudeTransaction(
            surface,
            tuple(torch.full_like(item, 0.25) for item in source),
            protected_parameters={
                "language_frozen": model.model.language_model.frozen,
                "vision": model.vision,
            },
        ) as transaction:
            assert all(
                torch.equal(observed, expected + 0.25)
                for observed, expected in zip(
                    surface.parameters, before_magnitudes, strict=True
                )
            )
            raise RuntimeError("toy failure")
    assert transaction.source_restored is True
    assert all(
        torch.equal(observed, expected)
        for observed, expected in zip(
            surface.parameters, before_magnitudes, strict=True
        )
    )
    assert torch.equal(model.model.language_model.frozen, before_frozen)

    receipt_path = tmp_path / "operator.json"
    receipt = publish_mechanics_receipt(
        receipt_path,
        disposition="OPERATOR_PASS",
        bindings={"source": "tiny"},
        surface=surface.receipt(),
        operator=evaluation.receipt(),
        separation=exhaustive.receipt(),
        source_restored=transaction.source_restored,
        measurements={"elapsed_seconds": 0.1, "peak_gpu_reserved_bytes": 0},
    )
    assert receipt["certificate_eligible"] is True
    with pytest.raises(FileExistsError, match="overwrite"):
        publish_mechanics_receipt(
            receipt_path,
            disposition="OPERATOR_PASS",
            bindings={"source": "tiny"},
            surface=surface.receipt(),
            operator=evaluation.receipt(),
            separation=exhaustive.receipt(),
            source_restored=True,
        )
    with pytest.raises(MechanicalInvalid, match="exhaustive"):
        publish_mechanics_receipt(
            tmp_path / "partial.json",
            disposition="OPERATOR_PASS",
            bindings={"source": "tiny"},
            surface=surface.receipt(),
            operator=evaluation.receipt(),
            separation=seeded_only.receipt(),
            source_restored=True,
        )


def test_exact_jvp_failure_is_hold_and_never_finite_difference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = (torch.tensor([1.0], dtype=torch.float64),)

    def unavailable(*_args: object, **_kwargs: object) -> object:
        raise NotImplementedError("forward AD unavailable")

    monkeypatch.setattr(torch.func, "jvp", unavailable)
    with pytest.raises(JVPOracleHold, match="exact JVP") as exc_info:
        exact_vjp_jvp(
            lambda values: values[0].square(),
            source,
            torch.ones(1, dtype=torch.float64),
        )
    assert mechanics_disposition(exc_info.value) == "JVP_ORACLE_HOLD"
    assert mechanics_disposition(RuntimeError("central finite difference")) == (
        "MECHANICAL_INVALID"
    )
