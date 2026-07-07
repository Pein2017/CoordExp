from __future__ import annotations

import pytest
import torch

from src.common.errors import LossContractError
from src.coordinate_targets import CoordinateLossTarget
from src.losses.coord_gaussian_rps import (
    CoordGaussianRPSLoss,
    gaussian_soft_targets_from_r95,
    ranked_probability_score,
)
from src.losses.context import LossContext
from src.losses.vocab import TokenVocabularyGroups
from src.packing.planner import PackedSegment
from src.supervision import TokenAtom, TokenSequence


def test_gaussian_targets_are_gt_centered_and_shape_aware() -> None:
    targets = gaussian_soft_targets_from_r95(
        torch.tensor([4, 4], dtype=torch.long),
        torch.tensor([0, 2], dtype=torch.float32),
        num_bins=9,
    )

    assert torch.argmax(targets[0]).item() == 4
    assert targets[0, 4] == pytest.approx(1.0)
    assert targets[0].sum().item() == pytest.approx(1.0)
    assert torch.argmax(targets[1]).item() == 4
    assert targets[1].sum().item() == pytest.approx(1.0)
    assert targets[1, 3].item() > targets[0, 3].item()
    assert targets[1, 5].item() == pytest.approx(targets[1, 3].item())


def test_ranked_probability_score_zero_for_identical_distribution() -> None:
    probs = torch.tensor([[0.0, 0.25, 0.5, 0.25]], dtype=torch.float32)

    rps = ranked_probability_score(probs, probs)

    assert rps.shape == (1,)
    assert rps.item() == pytest.approx(0.0)


def test_coord_gaussian_rps_loss_uses_bbox_axis_radii_and_coord_vocab_only() -> None:
    context = _context(
        _logits(
            (
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 4.0, 0.0, 0.0, -5.0, -5.0, -5.0),
                (0.0, 0.0, 0.0, 4.0, 0.0, -5.0, -5.0, -5.0),
                (0.0, 0.0, 0.0, 0.0, 4.0, -5.0, -5.0, -5.0),
                (0.0, 0.0, 0.0, 0.0, 5.0, -5.0, -5.0, -5.0),
            ),
            requires_grad=True,
        ),
        (
            _coord_atom(target_position=1, token_id=2, slot_index=0, bbox=(2, 3, 8, 13)),
            _coord_atom(target_position=2, token_id=3, slot_index=1, bbox=(2, 3, 8, 13)),
            _coord_atom(target_position=3, token_id=4, slot_index=2, bbox=(2, 3, 8, 13)),
            _coord_atom(target_position=4, token_id=4, slot_index=3, bbox=(2, 3, 8, 13)),
        ),
    )
    loss = CoordGaussianRPSLoss(
        gaussian_weight=0.5,
        rps_weight=0.2,
        temperature=1.0,
        gaussian_r95_axis_fraction=0.5,
        gaussian_r95_cap_bins=4,
        gaussian_r95_min_bins=1,
        gaussian_r95_fallback_bins=4,
    )

    result = loss.per_atom_loss(context)

    assert result.shape == (4,)
    assert torch.isfinite(result).all()
    assert loss.last_diagnostics is not None
    assert loss.last_diagnostics["selected_count"] == 4
    assert loss.last_diagnostics["target_r95_radius_mean"] == pytest.approx(3.5)
    assert loss.last_diagnostics["target_r95_radius_max"] == pytest.approx(4.0)
    result.sum().backward()
    assert context.logits.grad is not None
    assert torch.isfinite(context.logits.grad).all()


def test_coord_gaussian_rps_loss_requires_coordinate_target_metadata() -> None:
    context = _context(
        _logits(
            (
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                (0.0, 2.0, 0.0, 0.0, 0.0, 0.0),
            )
        ),
        (
            TokenAtom(
                pack_index=0,
                segment_index=0,
                example_index=0,
                example_id="ex-0",
                target_position=1,
                token_id=1,
                token_type="coordinate",
                text="<|coord_0|>",
                logical_target_position=1,
                object_id="obj-1",
                field="bbox[0]",
            ),
        ),
        coordinate_ids=(1, 2),
    )
    loss = CoordGaussianRPSLoss(
        gaussian_weight=1.0,
        rps_weight=0.0,
        temperature=1.0,
        gaussian_r95_axis_fraction=0.04,
        gaussian_r95_cap_bins=8,
        gaussian_r95_min_bins=1,
        gaussian_r95_fallback_bins=8,
    )

    with pytest.raises(LossContractError) as exc_info:
        loss.per_atom_loss(context)

    assert exc_info.value.code == "loss.coord_gaussian_rps_target_missing"


def test_coord_gaussian_rps_loss_rejects_tiny_temperature() -> None:
    with pytest.raises(LossContractError) as exc_info:
        CoordGaussianRPSLoss(
            gaussian_weight=1.0,
            rps_weight=0.0,
            temperature=1e-45,
            gaussian_r95_axis_fraction=0.04,
            gaussian_r95_cap_bins=8,
            gaussian_r95_min_bins=1,
            gaussian_r95_fallback_bins=8,
        )

    assert exc_info.value.code == "loss.coord_gaussian_rps_temperature"


def _context(
    logits: torch.Tensor,
    atoms: tuple[TokenAtom, ...],
    *,
    coordinate_ids: tuple[int, ...] = (1, 2, 3, 4),
) -> LossContext:
    return LossContext(
        logits=logits,
        token_sequence=TokenSequence(
            pack_index=0,
            input_ids=tuple(0 for _ in range(int(logits.shape[1]))),
            segments=(
                PackedSegment(
                    pack_index=0,
                    segment_index=0,
                    example_index=0,
                    example_id="ex-0",
                    start=0,
                    end=int(logits.shape[1]),
                ),
            ),
            atoms=atoms,
            spans=(),
        ),
        vocab_groups=TokenVocabularyGroups(
            vocab_size=int(logits.shape[2]),
            desc_text=(0,),
            schema=(int(logits.shape[2]) - 3,),
            coordinate=coordinate_ids,
            eos=(int(logits.shape[2]) - 2,),
            blocked=tuple(
                index
                for index in range(int(logits.shape[2]))
                if index
                not in {0, int(logits.shape[2]) - 3, int(logits.shape[2]) - 2, *coordinate_ids}
            ),
        ),
    )


def _coord_atom(
    *,
    target_position: int,
    token_id: int,
    slot_index: int,
    bbox: tuple[int, int, int, int],
) -> TokenAtom:
    return TokenAtom(
        pack_index=0,
        segment_index=0,
        example_index=0,
        example_id="ex-0",
        target_position=target_position,
        token_id=token_id,
        token_type="coordinate",
        text=f"<|coord_{token_id - 1}|>",
        logical_target_position=target_position,
        object_id="obj-1",
        field=f"bbox[{slot_index}]",
        coordinate_target=CoordinateLossTarget(bbox=bbox, slot_index=slot_index),
    )


def _logits(
    rows: tuple[tuple[float, ...], ...],
    *,
    requires_grad: bool = False,
) -> torch.Tensor:
    logits = torch.tensor((rows,), dtype=torch.float32)
    if requires_grad:
        logits.requires_grad_()
    return logits
