from __future__ import annotations

import pytest
import torch

from src.training.bridge import PredictionCoordinateMapper
from src.training.objectives.types import LabelLogitRowMap
from src.training.supervision.batch import SupervisionBatch
from src.training.supervision.distributions import HardTokenDistribution
from src.training.supervision.spans import SupervisionSpan


def _span(
    *,
    sample_id: str = "sample-1",
    label_positions: tuple[int, ...] = (1,),
) -> SupervisionSpan:
    return SupervisionSpan(
        sample_id=sample_id,
        role="schema",
        label_positions=label_positions,
        distribution=HardTokenDistribution(token_id=2),
    )


def _batch(*spans: SupervisionSpan) -> SupervisionBatch:
    return SupervisionBatch(spans=spans, batch_id="batch-1")


def test_prediction_coordinate_mapper_wraps_2d_causal_row_map() -> None:
    logits = torch.zeros((4, 7), dtype=torch.float32)

    mapper = PredictionCoordinateMapper.from_logits(
        logits,
        supervision=_batch(_span(label_positions=(1, 3))),
    )

    assert mapper.constructed is True
    assert mapper.logits_shape == (4, 7)
    assert mapper.time_steps == 4
    assert mapper.vocab_size == 7
    assert mapper.batch_size is None
    assert type(mapper.label_rows) is LabelLogitRowMap
    assert [row.row_index for row in mapper.resolved_rows] == [0, 2]
    assert [row.label_position for row in mapper.resolved_rows] == [1, 3]


def test_prediction_coordinate_mapper_rejects_label_position_zero() -> None:
    logits = torch.zeros((4, 7), dtype=torch.float32)

    with pytest.raises(ValueError, match="label position 0.*causal"):
        PredictionCoordinateMapper.from_logits(
            logits,
            supervision=_batch(_span(label_positions=(0,))),
        )


def test_prediction_coordinate_mapper_rejects_positions_outside_logits_rows() -> None:
    logits = torch.zeros((2, 7), dtype=torch.float32)

    with pytest.raises(ValueError, match="outside logits rows"):
        PredictionCoordinateMapper.from_logits(
            logits,
            supervision=_batch(_span(label_positions=(4,))),
        )


def test_prediction_coordinate_mapper_routes_3d_logits_by_sample_id() -> None:
    logits = torch.zeros((2, 4, 7), dtype=torch.float32)

    mapper = PredictionCoordinateMapper.from_logits(
        logits,
        supervision=_batch(
            _span(sample_id="sample-a", label_positions=(2,)),
            _span(sample_id="sample-b", label_positions=(4,)),
        ),
        sample_id_to_batch_index={"sample-a": 1, "sample-b": 0},
    )

    assert mapper.logits_shape == (2, 4, 7)
    assert mapper.time_steps == 4
    assert mapper.vocab_size == 7
    assert mapper.batch_size == 2
    assert [(row.batch_index, row.row_index) for row in mapper.resolved_rows] == [
        (1, 1),
        (0, 3),
    ]


def test_prediction_coordinate_mapper_requires_batch_map_for_non_empty_3d() -> None:
    logits = torch.zeros((2, 4, 7), dtype=torch.float32)

    with pytest.raises(ValueError, match="sample_id_to_batch_index.*3D"):
        PredictionCoordinateMapper.from_logits(
            logits,
            supervision=_batch(_span(label_positions=(1,))),
        )


def test_prediction_coordinate_mapper_allows_empty_3d_without_batch_map() -> None:
    logits = torch.zeros((2, 4, 7), dtype=torch.float32)

    mapper = PredictionCoordinateMapper.from_logits(
        logits,
        supervision=SupervisionBatch(),
    )

    assert mapper.constructed is True
    assert mapper.logits_shape == (2, 4, 7)
    assert mapper.batch_size == 2
    assert mapper.label_rows is None
    assert mapper.resolved_rows == ()
