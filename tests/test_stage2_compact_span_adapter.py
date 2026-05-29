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


def test_stage2_adapter_preserves_rollout_correction_and_duplicate_provenance() -> None:
    projection = CompactFullSpanProjector().project(_compact_view())
    batch = Stage2CompactSpanAdapter().build_batch(
        sample_id="sample-1",
        projection=projection,
        targets=(
            Stage2CompactTokenTargetSpec(
                label_position=6,
                token_ids=(106, 1006),
                channel="rollout_correction",
                provenance=Stage2SpanProvenance(
                    source_role="matched_gt",
                    assignment_id="assign-1",
                    duplicate_filter_id="dup-keep-1",
                    false_negative=False,
                    source_object_instance_id="obj-1",
                ),
                span_provenance="rollout-correction-coordinate-target",
                token_weights=(3.0, 1.0),
            ),
            Stage2CompactTokenTargetSpec(
                label_position=2,
                token_ids=(202,),
                channel="rollout_correction",
                provenance=Stage2SpanProvenance(
                    source_role="false_negative_append",
                    assignment_id=None,
                    duplicate_filter_id="fn-1",
                    false_negative=True,
                    source_object_instance_id="obj-2",
                ),
                span_provenance="rollout-correction-schema-target",
            ),
        ),
        context_id="ctx-2",
    )

    coordinate_span, schema_span = batch.spans
    assert coordinate_span.sample_id == "sample-1"
    assert coordinate_span.context_id == "ctx-2"
    assert coordinate_span.provenance == "rollout-correction-coordinate-target"
    assert coordinate_span.role == "coordinate"
    assert coordinate_span.label_positions == (6,)
    assert isinstance(coordinate_span.distribution, MultiPositiveTokenDistribution)
    assert coordinate_span.distribution.token_ids == (106, 1006)
    assert coordinate_span.distribution.token_weights == (3.0, 1.0)
    assert coordinate_span.metadata["channel"] == "rollout_correction"
    assert coordinate_span.metadata["source_role"] == "matched_gt"
    assert coordinate_span.metadata["assignment_id"] == "assign-1"
    assert coordinate_span.metadata["duplicate_filter_id"] == "dup-keep-1"
    assert coordinate_span.metadata["false_negative"] is False
    assert coordinate_span.metadata["source_object_instance_id"] == "obj-1"

    assert schema_span.role == "schema"
    assert schema_span.distribution.token_weights is None
    assert schema_span.metadata["channel"] == "rollout_correction"
    assert schema_span.metadata["source_role"] == "false_negative_append"
    assert schema_span.metadata["assignment_id"] is None
    assert schema_span.metadata["duplicate_filter_id"] == "fn-1"
    assert schema_span.metadata["false_negative"] is True
    assert schema_span.metadata["source_object_instance_id"] == "obj-2"


def test_stage2_adapter_propagates_objective_weight_metadata_without_mutation() -> None:
    projection = CompactFullSpanProjector().project(_compact_view())
    target = Stage2CompactTokenTargetSpec(
        label_position=6,
        token_ids=(106, 1006),
        channel="rollout_correction",
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
            channel="rollout_correction",
            provenance=Stage2SpanProvenance(source_role="matched_gt"),
            role="coordinate",
        )


def test_stage2_adapter_validates_rollout_correction_surface_and_positions() -> None:
    adapter = Stage2CompactSpanAdapter()
    projection = CompactFullSpanProjector().project(_compact_view())
    provenance = Stage2SpanProvenance(source_role="matched_gt")

    with pytest.raises(ValueError, match="rollout_correction"):
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
                    channel="rollout_correction",
                    provenance=provenance,
                ),
            ),
        )
