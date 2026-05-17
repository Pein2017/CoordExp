from __future__ import annotations

import torch

from src.training.objectives.runner import ObjectiveRunner
from src.training.objectives.types import DEFAULT_PRECISION_POLICY, ObjectiveSpec
from src.training.supervision.batch import SupervisionBatch
from src.training.supervision.distributions import (
    BoxRegressionDistribution,
    CoordinateSoftTokenDistribution,
    HardTokenDistribution,
    MultiPositiveTokenDistribution,
)
from src.training.supervision.spans import SupervisionSpan


def _span(
    *,
    label_positions: tuple[int, ...] = (1,),
    distribution: object,
    role: str = "schema",
) -> SupervisionSpan:
    return SupervisionSpan(
        sample_id="sample-1",
        role=role,
        label_positions=label_positions,
        distribution=distribution,
    )


def _run(
    logits: torch.Tensor,
    supervision: SupervisionBatch,
    *objectives: ObjectiveSpec,
):
    return ObjectiveRunner().run(
        logits=logits,
        supervision=supervision,
        objectives=objectives,
    )


def _coord_vocab(size: int = 1000, *, start: int = 8) -> list[int]:
    return list(range(start, start + size))


def test_bf16_autocast_hard_ce_and_trie_ce_upcast_and_backprop() -> None:
    logits = torch.tensor(
        [
            [0.25, -0.75, 1.5, -2.0],
            [1.25, -3.0, 0.0, 0.5],
        ],
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    supervision = SupervisionBatch(
        spans=(
            _span(
                label_positions=(1,),
                distribution=HardTokenDistribution(token_id=2),
            ),
            _span(
                label_positions=(2,),
                distribution=MultiPositiveTokenDistribution(token_ids=(0, 3)),
            ),
        )
    )

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        result = _run(
            logits,
            supervision,
            ObjectiveSpec("token_ce"),
            ObjectiveSpec("trie_ce"),
        )

    assert result.loss.dtype == torch.float32
    assert result.objectives["token_ce"].loss.dtype == torch.float32
    assert result.objectives["trie_ce"].loss.dtype == torch.float32
    assert result.objectives["token_ce"].precision_policy == DEFAULT_PRECISION_POLICY
    assert result.objectives["trie_ce"].precision_policy.math_dtype == torch.float32
    assert torch.isfinite(result.loss)
    result.loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_coord_soft_ce_upcasts_and_backprops() -> None:
    logits = torch.tensor(
        [[0.25, 1.5, -0.75, 0.5]],
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    supervision = SupervisionBatch(
        spans=(
            _span(
                role="coordinate",
                distribution=CoordinateSoftTokenDistribution(
                    token_weights=((1, 2.0), (3, 1.0)),
                    loss_mode="full_vocab_ce",
                ),
            ),
        )
    )

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        result = _run(logits, supervision, ObjectiveSpec("coord_soft_ce", weight=2.5))

    assert result.loss.dtype == torch.float32
    assert result.objectives["coord_soft_ce"].loss.dtype == torch.float32
    assert result.objectives["coord_soft_ce"].weight == 2.5
    assert result.objectives["coord_soft_ce"].precision_policy.math_dtype == torch.float32
    assert torch.isfinite(result.loss)
    result.loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_box_regression_bf16_logits_return_float32_and_backprop() -> None:
    coord_token_ids = _coord_vocab()
    vocab = max(coord_token_ids) + 1
    logits = torch.zeros((4, vocab), dtype=torch.bfloat16, requires_grad=True)
    target_bins = (120, 180, 640, 720)
    with torch.no_grad():
        for slot_index, target_bin in enumerate(target_bins):
            logits[slot_index, coord_token_ids[target_bin]] = 8.0

    target_box = tuple(float(value) / 999.0 for value in target_bins)
    supervision = SupervisionBatch(
        spans=(
            _span(
                label_positions=(1, 2, 3, 4),
                role="coordinate",
                distribution=BoxRegressionDistribution(target_bbox=target_box),
            ),
        )
    )

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        result = _run(
            logits,
            supervision,
            ObjectiveSpec(
                "box_regression",
                config={
                    "coord_token_ids": coord_token_ids,
                    "temperature": 0.25,
                    "smoothl1_weight": 1.0,
                    "ciou_weight": 1.0,
                },
            ),
        )

    assert result.loss.dtype == torch.float32
    assert result.objectives["box_regression"].loss.dtype == torch.float32
    assert result.objectives["box_regression"].weight == 1.0
    assert result.objectives["box_regression"].precision_policy == DEFAULT_PRECISION_POLICY
    assert torch.isfinite(result.loss)
    result.loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_zero_objective_result_exposes_weight_and_precision_policy() -> None:
    result = _run(
        torch.zeros((1, 3), dtype=torch.float32),
        SupervisionBatch(),
        ObjectiveSpec("token_ce", weight=4.0),
    )

    objective = result.objectives["token_ce"]
    assert objective.weight == 4.0
    assert objective.precision_policy == DEFAULT_PRECISION_POLICY
    assert objective.precision_policy.math_dtype == torch.float32
