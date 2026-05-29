from __future__ import annotations

import pytest

from src.training.span_adapters.compact_projector import CompactFullSpanProjector
from src.training.span_adapters.stage1_compact import (
    CompactCoordinateSoftTargetSpec,
    CompactCoordinateTokenWeightSpec,
    CompactTrieTargetSpec,
    Stage1CompactSpanAdapter,
)
from src.training.supervision.distributions import (
    CoordinateSoftTokenDistribution,
    MultiPositiveTokenDistribution,
)

from test_compact_span_projector import (
    _compact_view,
    _compact_view_with_unsupervised_coordinate,
)


def test_stage1_adapter_builds_trie_and_coordinate_spans() -> None:
    projection = CompactFullSpanProjector().project(_compact_view())
    batch = Stage1CompactSpanAdapter().build_batch(
        sample_id="sample-1",
        projection=projection,
        trie_targets=(
            CompactTrieTargetSpec(
                label_position=2,
                token_ids=(12, 13),
                token_weights=(2.0, 1.0),
            ),
            CompactTrieTargetSpec(label_position=3, token_ids=(21, 22)),
        ),
        coordinate_targets=(
            CompactCoordinateSoftTargetSpec(
                label_position=6,
                token_weights=(
                    CompactCoordinateTokenWeightSpec(token_id=1006, weight=0.7),
                    CompactCoordinateTokenWeightSpec(token_id=1007, weight=0.3),
                ),
                target_family="ciou_gibbs_v0",
                loss_mode="coord_vocab_ce",
            ),
        ),
        context_id="ctx-1",
        provenance="stage1-fixture",
    )

    assert len(batch.spans) == 3
    schema_span, desc_span, coord_span = batch.spans
    assert schema_span.sample_id == "sample-1"
    assert schema_span.context_id == "ctx-1"
    assert schema_span.provenance == "stage1-fixture"
    assert schema_span.role == "schema"
    assert schema_span.label_positions == (2,)
    assert isinstance(schema_span.distribution, MultiPositiveTokenDistribution)
    assert schema_span.distribution.token_ids == (12, 13)
    assert schema_span.distribution.token_weights == (2.0, 1.0)

    assert desc_span.role == "free_text"
    assert desc_span.label_positions == (3,)
    assert desc_span.distribution.token_weights is None

    assert coord_span.role == "coordinate"
    assert coord_span.label_positions == (6,)
    assert isinstance(coord_span.distribution, CoordinateSoftTokenDistribution)
    assert tuple(coord_span.distribution.token_weights) == (
        CompactCoordinateTokenWeightSpec(token_id=1006, weight=0.7).to_distribution_weight(),
        CompactCoordinateTokenWeightSpec(token_id=1007, weight=0.3).to_distribution_weight(),
    )
    assert coord_span.distribution.target_family == "ciou_gibbs_v0"
    assert coord_span.distribution.loss_mode == "coord_vocab_ce"


def test_stage1_adapter_accepts_encoded_view_and_does_not_shift_positions() -> None:
    batch = Stage1CompactSpanAdapter().build_batch(
        sample_id="sample-1",
        projection=_compact_view(),
        trie_targets=(CompactTrieTargetSpec(label_position=6, token_ids=(106,)),),
    )

    span = batch.spans[0]
    assert span.role == "coordinate"
    assert span.label_positions == (6,)
    assert not hasattr(span, "logit_positions")


def test_stage1_target_specs_do_not_accept_role_override() -> None:
    with pytest.raises(TypeError, match="role"):
        CompactTrieTargetSpec(
            label_position=3,
            token_ids=(103,),
            role="coordinate",
        )


def test_stage1_adapter_rejects_invalid_target_positions() -> None:
    adapter = Stage1CompactSpanAdapter()
    projection = CompactFullSpanProjector().project(_compact_view())

    with pytest.raises(ValueError, match="label position"):
        adapter.build_batch(
            sample_id="sample-1",
            projection=projection,
            trie_targets=(CompactTrieTargetSpec(label_position=10, token_ids=(1,)),),
        )

    with pytest.raises(ValueError, match="coordinate"):
        adapter.build_batch(
            sample_id="sample-1",
            projection=projection,
            coordinate_targets=(
                CompactCoordinateSoftTargetSpec(
                    label_position=3,
                    token_weights=(CompactCoordinateTokenWeightSpec(token_id=1, weight=1.0),),
                ),
            ),
        )


def test_stage1_adapter_rejects_unsupervised_coordinate_soft_target() -> None:
    adapter = Stage1CompactSpanAdapter()
    projection = CompactFullSpanProjector().project(
        _compact_view_with_unsupervised_coordinate()
    )

    with pytest.raises(ValueError, match="label position"):
        adapter.build_batch(
            sample_id="sample-1",
            projection=projection,
            coordinate_targets=(
                CompactCoordinateSoftTargetSpec(
                    label_position=8,
                    token_weights=(
                        CompactCoordinateTokenWeightSpec(
                            token_id=1008,
                            weight=1.0,
                        ),
                    ),
                ),
            ),
        )
