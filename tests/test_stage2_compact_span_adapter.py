from __future__ import annotations

import pytest

from src.training.span_adapters.compact_projector import CompactFullSpanProjector
from src.training.span_adapters.stage2_compact import (
    Stage2CompactSpanAdapter,
    Stage2CompactTokenTargetSpec,
    Stage2SpanProvenance,
)
from src.training.supervision.distributions import MultiPositiveTokenDistribution

from test_compact_span_projector import _compact_view


def test_stage2_adapter_preserves_channel_assignment_and_duplicate_provenance() -> None:
    projection = CompactFullSpanProjector().project(_compact_view())
    batch = Stage2CompactSpanAdapter().build_batch(
        sample_id="sample-1",
        projection=projection,
        targets=(
            Stage2CompactTokenTargetSpec(
                label_position=6,
                token_ids=(106, 1006),
                channel="channel_a",
                provenance=Stage2SpanProvenance(
                    source_role="matched_gt",
                    assignment_id="assign-1",
                    duplicate_filter_id="dup-keep-1",
                    false_negative=False,
                    source_object_instance_id="obj-1",
                ),
                span_provenance="channel-a-target",
                token_weights=(3.0, 1.0),
            ),
            Stage2CompactTokenTargetSpec(
                label_position=2,
                token_ids=(202,),
                channel="channel_b",
                provenance=Stage2SpanProvenance(
                    source_role="false_negative_append",
                    assignment_id=None,
                    duplicate_filter_id="fn-1",
                    false_negative=True,
                    source_object_instance_id="obj-2",
                ),
                span_provenance="channel-b-target",
            ),
        ),
        context_id="ctx-2",
    )

    channel_a, channel_b = batch.spans
    assert channel_a.sample_id == "sample-1"
    assert channel_a.context_id == "ctx-2"
    assert channel_a.provenance == "channel-a-target"
    assert channel_a.role == "coordinate"
    assert channel_a.label_positions == (6,)
    assert isinstance(channel_a.distribution, MultiPositiveTokenDistribution)
    assert channel_a.distribution.token_ids == (106, 1006)
    assert channel_a.distribution.token_weights == (3.0, 1.0)
    assert channel_a.metadata["channel"] == "channel_a"
    assert channel_a.metadata["source_role"] == "matched_gt"
    assert channel_a.metadata["assignment_id"] == "assign-1"
    assert channel_a.metadata["duplicate_filter_id"] == "dup-keep-1"
    assert channel_a.metadata["false_negative"] is False
    assert channel_a.metadata["source_object_instance_id"] == "obj-1"

    assert channel_b.role == "schema"
    assert channel_b.distribution.token_weights is None
    assert channel_b.metadata["channel"] == "channel_b"
    assert channel_b.metadata["source_role"] == "false_negative_append"
    assert channel_b.metadata["assignment_id"] is None
    assert channel_b.metadata["duplicate_filter_id"] == "fn-1"
    assert channel_b.metadata["false_negative"] is True
    assert channel_b.metadata["source_object_instance_id"] == "obj-2"


def test_stage2_adapter_propagates_objective_weight_metadata_without_mutation() -> None:
    projection = CompactFullSpanProjector().project(_compact_view())
    target = Stage2CompactTokenTargetSpec(
        label_position=6,
        token_ids=(106, 1006),
        channel="channel_a",
        provenance=Stage2SpanProvenance(
            source_role="matched_gt",
            state_weight=2.0,
            loss_weight=3.0,
            support_weight=0.5,
            balance_weight=4.0,
        ),
    )

    batch = Stage2CompactSpanAdapter().build_batch(
        sample_id="sample-1",
        projection=projection,
        targets=(target,),
    )

    span = batch.spans[0]
    assert span.metadata["state_weight"] == 2.0
    assert span.metadata["loss_weight"] == 3.0
    assert span.metadata["support_weight"] == 0.5
    assert span.metadata["balance_weight"] == 4.0
    assert target.provenance.state_weight == 2.0


def test_stage2_target_specs_do_not_accept_role_override() -> None:
    with pytest.raises(TypeError, match="role"):
        Stage2CompactTokenTargetSpec(
            label_position=3,
            token_ids=(103,),
            channel="channel_a",
            provenance=Stage2SpanProvenance(source_role="matched_gt"),
            role="coordinate",
        )


def test_stage2_adapter_validates_channel_and_positions() -> None:
    adapter = Stage2CompactSpanAdapter()
    projection = CompactFullSpanProjector().project(_compact_view())
    provenance = Stage2SpanProvenance(source_role="matched_gt")

    with pytest.raises(ValueError, match="channel_a"):
        adapter.build_batch(
            sample_id="sample-1",
            projection=projection,
            targets=(
                Stage2CompactTokenTargetSpec(
                    label_position=2,
                    token_ids=(1,),
                    channel="primary",
                    provenance=provenance,
                ),
            ),
        )

    with pytest.raises(ValueError, match="label position"):
        adapter.build_batch(
            sample_id="sample-1",
            projection=projection,
            targets=(
                Stage2CompactTokenTargetSpec(
                    label_position=10,
                    token_ids=(1,),
                    channel="channel_a",
                    provenance=provenance,
                ),
            ),
        )
