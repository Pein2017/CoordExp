from __future__ import annotations

import pytest
import torch

from src.training.coverage_ledger.head import CoverageLedgerHead
from src.training.coverage_ledger.loss import (
    CoverageLedgerLossConfig,
    build_coverage_ledger_targets,
    compute_coverage_ledger_loss,
)
from src.training.coverage_ledger.sidecars import (
    CoverageLedgerObjectEntry,
    CoverageLedgerSidecar,
)


def _entry(index: int, *, box_start: int, box_end: int) -> CoverageLedgerObjectEntry:
    return CoverageLedgerObjectEntry(
        object_instance_id=f"object-{index}",
        source_object_index=index,
        emitted_order_index=index,
        image_index=0,
        bbox_norm1000_xyxy=(10, 20, 110 + index, 220 + index),
        box_start_position=box_start,
        coord_label_positions=(box_start + 1, box_start + 2, box_start + 3, box_start + 4),
        object_ref_end_position=box_start - 1,
        box_end_position=box_end,
    )


def _sidecar() -> CoverageLedgerSidecar:
    return CoverageLedgerSidecar(
        sample_id="sample-1",
        prompt_end_position=2,
        object_entries=(
            _entry(0, box_start=4, box_end=9),
            _entry(1, box_start=11, box_end=16),
            _entry(2, box_start=18, box_end=23),
        ),
        image_grid_thw=(1, 16, 16),
        processed_width=640,
        processed_height=480,
        image_identity="image.jpg",
    )


def _identity_head(*, dim: int = 3, normalize_eps: float = 1.0e-6) -> CoverageLedgerHead:
    head = CoverageLedgerHead(
        hidden_size=dim,
        visual_dim=dim,
        ledger_projection_dim=dim,
        normalize_eps=normalize_eps,
    )
    with torch.no_grad():
        eye = torch.eye(dim)
        head.state_projection.weight.copy_(eye)
        head.region_anchor_state_projection.weight.copy_(eye)
        head.object_projection.weight.copy_(eye)
    return head


def _hidden_states() -> torch.Tensor:
    hidden = torch.zeros((24, 3), dtype=torch.float32)
    hidden[2] = torch.tensor([-1.0, -1.0, -1.0])
    hidden[4] = torch.tensor([1.0, 0.0, 0.0])
    hidden[9] = torch.tensor([1.0, 0.0, 0.0])
    hidden[11] = torch.tensor([0.0, 1.0, 0.0])
    hidden[16] = torch.tensor([1.0, 1.0, 0.0])
    hidden[18] = torch.tensor([0.0, 0.0, 1.0])
    hidden[23] = torch.tensor([1.0, 1.0, 1.0])
    return hidden


def _visual_objects() -> torch.Tensor:
    return torch.eye(3, dtype=torch.float32)


def test_build_coverage_ledger_targets_for_three_objects() -> None:
    targets = build_coverage_ledger_targets(_sidecar(), device=torch.device("cpu"))

    assert targets.coverage_state_positions == (2, 9, 16, 23)
    assert targets.region_anchor_positions == (4, 11, 18)
    assert targets.region_anchor_object_indices == (0, 1, 2)
    assert targets.coverage_targets.tolist() == [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
    ]


def test_region_anchor_uses_only_current_object_positive_pairs() -> None:
    sidecar = _sidecar()
    head = _identity_head(dim=4)
    hidden_states = torch.zeros((24, 4), dtype=torch.float32)
    hidden_states[4] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    hidden_states[11] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    hidden_states[18] = torch.tensor([0.0, 0.0, 1.0, 0.0])
    current_only_visuals = torch.tensor(
        [
            [0.6, 0.0, 0.0, 0.8],
            [0.0, 0.6, 0.0, 0.8],
            [0.0, 0.0, 0.6, 0.8],
        ],
        dtype=torch.float32,
    )
    config = CoverageLedgerLossConfig(
        coverage_weight=0.0,
        region_anchor_weight=1.0,
        temperature=0.5,
        pos_weight=1.0,
    )

    baseline = compute_coverage_ledger_loss(
        head=head,
        final_hidden_states=hidden_states,
        pooled_visual_object_embeddings=current_only_visuals,
        sidecar=sidecar,
        config=config,
    )

    assert baseline.debug_rows.region_anchor_object_indices == (0, 1, 2)
    assert baseline.debug_rows.region_anchor_positive_logits.tolist() == pytest.approx(
        [1.2, 1.2, 1.2]
    )

    non_current_perturbed_visuals = torch.tensor(
        [
            [0.6, 0.8, 0.0, 0.0],
            [0.0, 0.6, 0.8, 0.0],
            [0.8, 0.0, 0.6, 0.0],
        ],
        dtype=torch.float32,
    )
    perturbed = compute_coverage_ledger_loss(
        head=head,
        final_hidden_states=hidden_states,
        pooled_visual_object_embeddings=non_current_perturbed_visuals,
        sidecar=sidecar,
        config=config,
    )

    assert perturbed.debug_rows.region_anchor_positive_logits.tolist() == pytest.approx(
        baseline.debug_rows.region_anchor_positive_logits.tolist()
    )
    assert perturbed.region_anchor_loss.item() == pytest.approx(
        baseline.region_anchor_loss.item()
    )


def test_lower_temperature_increases_abs_logits() -> None:
    kwargs = {
        "head": _identity_head(),
        "final_hidden_states": _hidden_states(),
        "pooled_visual_object_embeddings": _visual_objects(),
        "sidecar": _sidecar(),
    }

    warm = compute_coverage_ledger_loss(
        **kwargs,
        config=CoverageLedgerLossConfig(
            coverage_weight=1.0,
            region_anchor_weight=1.0,
            temperature=1.0,
            pos_weight=1.0,
        ),
    )
    cool = compute_coverage_ledger_loss(
        **kwargs,
        config=CoverageLedgerLossConfig(
            coverage_weight=1.0,
            region_anchor_weight=1.0,
            temperature=0.25,
            pos_weight=1.0,
        ),
    )

    assert cool.debug_rows.coverage_logits.abs().max() > warm.debug_rows.coverage_logits.abs().max()
    assert (
        cool.debug_rows.region_anchor_positive_logits.abs().max()
        > warm.debug_rows.region_anchor_positive_logits.abs().max()
    )


def test_zero_vectors_produce_finite_float32_losses() -> None:
    result = compute_coverage_ledger_loss(
        head=_identity_head(normalize_eps=1.0e-6),
        final_hidden_states=torch.zeros((24, 3), dtype=torch.bfloat16),
        pooled_visual_object_embeddings=torch.zeros((3, 3), dtype=torch.bfloat16),
        sidecar=_sidecar(),
        config=CoverageLedgerLossConfig(
            coverage_weight=1.0,
            region_anchor_weight=1.0,
            temperature=0.5,
            pos_weight=2.0,
        ),
    )

    assert result.total_loss.dtype == torch.float32
    assert result.coverage_loss.dtype == torch.float32
    assert result.region_anchor_loss.dtype == torch.float32
    assert torch.isfinite(result.total_loss)
    assert torch.isfinite(result.debug_rows.coverage_logits).all()
    assert torch.isfinite(result.debug_rows.region_anchor_positive_logits).all()


def test_non_finite_logits_or_losses_raise_floating_point_error() -> None:
    hidden_states = _hidden_states()
    hidden_states[9, 0] = float("nan")

    with pytest.raises(FloatingPointError, match="coverage ledger.*non-finite"):
        compute_coverage_ledger_loss(
            head=_identity_head(),
            final_hidden_states=hidden_states,
            pooled_visual_object_embeddings=_visual_objects(),
            sidecar=_sidecar(),
            config=CoverageLedgerLossConfig(
                coverage_weight=1.0,
                region_anchor_weight=1.0,
                temperature=0.5,
                pos_weight=1.0,
            ),
        )


def test_zero_component_weights_disable_contributions_but_keep_diagnostic_counts() -> None:
    coverage_disabled = compute_coverage_ledger_loss(
        head=_identity_head(),
        final_hidden_states=_hidden_states(),
        pooled_visual_object_embeddings=_visual_objects(),
        sidecar=_sidecar(),
        config=CoverageLedgerLossConfig(
            coverage_weight=0.0,
            region_anchor_weight=1.0,
            temperature=0.5,
            pos_weight=1.0,
        ),
    )
    anchor_disabled = compute_coverage_ledger_loss(
        head=_identity_head(),
        final_hidden_states=_hidden_states(),
        pooled_visual_object_embeddings=_visual_objects(),
        sidecar=_sidecar(),
        config=CoverageLedgerLossConfig(
            coverage_weight=1.0,
            region_anchor_weight=0.0,
            temperature=0.5,
            pos_weight=1.0,
        ),
    )

    assert coverage_disabled.weighted_loss.item() == pytest.approx(
        coverage_disabled.region_anchor_loss.item()
    )
    assert anchor_disabled.weighted_loss.item() == pytest.approx(
        anchor_disabled.coverage_loss.item()
    )
    assert coverage_disabled.coverage_weight == pytest.approx(0.0)
    assert coverage_disabled.region_anchor_weight == pytest.approx(1.0)
    assert anchor_disabled.coverage_weight == pytest.approx(1.0)
    assert anchor_disabled.region_anchor_weight == pytest.approx(0.0)
    assert coverage_disabled.debug_rows.object_count == 3
    assert coverage_disabled.debug_rows.coverage_state_count == 4
    assert coverage_disabled.debug_rows.coverage_pair_count == 12
    assert coverage_disabled.debug_rows.region_anchor_pair_count == 3
    assert {event.key: event.value for event in coverage_disabled.metric_events} == {
        "training/objectives/coverage_ledger/object_count": 3.0,
        "training/objectives/coverage_ledger/coverage_state_count": 4.0,
        "training/objectives/coverage_ledger/coverage_pair_count": 12.0,
        "training/objectives/coverage_ledger/region_anchor_pair_count": 3.0,
    }


def test_backward_reaches_head_and_hidden_states_but_not_detached_visual_embeddings() -> None:
    head = _identity_head()
    hidden_states = _hidden_states()
    hidden_states[4] = torch.tensor([1.0, 0.25, 0.0])
    hidden_states = hidden_states.requires_grad_()
    pooled_visuals = _visual_objects().requires_grad_()

    result = compute_coverage_ledger_loss(
        head=head,
        final_hidden_states=hidden_states,
        pooled_visual_object_embeddings=pooled_visuals,
        sidecar=_sidecar(),
        config=CoverageLedgerLossConfig(
            coverage_weight=1.0,
            region_anchor_weight=1.0,
            temperature=0.5,
            pos_weight=1.0,
        ),
    )
    result.total_loss.backward()

    assert head.state_projection.weight.grad is not None
    assert head.region_anchor_state_projection.weight.grad is not None
    assert head.object_projection.weight.grad is not None
    assert torch.isfinite(head.state_projection.weight.grad).all()
    assert torch.isfinite(head.region_anchor_state_projection.weight.grad).all()
    assert torch.isfinite(head.object_projection.weight.grad).all()
    assert hidden_states.grad is not None
    assert torch.isfinite(hidden_states.grad).all()
    assert hidden_states.grad[2].abs().sum() > 0
    assert hidden_states.grad[4].abs().sum() > 0
    assert hidden_states.grad[9].abs().sum() > 0
    assert pooled_visuals.grad is None
