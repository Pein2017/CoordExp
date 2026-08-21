"""Zero-eligible-segment failure must be a collective, post-gather decision.

Spec (`coordexp-swift-supervision-losses`, scenario "Zero eligible protected
atoms on one rank"): the rank-local eligible count MUST enter the same
planned-step all-rank denominator decision before any rank raises, without
distributed deadlock. Which runs fail does NOT change: a rank observing zero
eligible segments for a composed term still fails the planned step.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from src.common.errors import LossContractError
from src.losses import LossRunner
from src.losses.normalizers import SegmentBalancedDenominator
from src.losses.runner import _build_denominator_from_token_sequences
from src.packing.planner import PackedSegment
from src.supervision import TokenAtom, TokenSequence


class _RecordingGatherer:
    """Records the local payload it was handed and returns a fixed world."""

    def __init__(self, peer_payload: Mapping[str, Mapping[str, Any]]) -> None:
        self._peer_payload = {
            str(name): dict(payload) for name, payload in peer_payload.items()
        }
        self.calls: list[dict[str, dict[str, Any]]] = []

    @property
    def called(self) -> bool:
        return bool(self.calls)

    def __call__(
        self, local_payload: Mapping[str, Mapping[str, Any]]
    ) -> tuple[Mapping[str, Mapping[str, Any]], ...]:
        self.calls.append(
            {str(name): dict(payload) for name, payload in local_payload.items()}
        )
        return (
            {str(name): dict(payload) for name, payload in local_payload.items()},
            dict(self._peer_payload),
        )


def test_zero_eligible_rank_enters_the_gather_before_it_raises() -> None:
    """The rank-local zero count reaches the all-rank gather, then fails."""

    runner = _coordinate_gate_runner()
    gatherer = _RecordingGatherer(_nonzero_rank_payload(runner))

    with pytest.raises(LossContractError) as excinfo:
        runner.prepare_planned_step(
            (_sequence_without_coordinates(),),
            denominator_gatherer=gatherer,
            world_size=2,
            rank=0,
        )

    assert gatherer.called is True, (
        "the zero-eligible rank must enter the collective gather before raising; "
        "raising locally desyncs peers that are already inside the gather"
    )
    assert len(gatherer.calls) == 1
    local_payload = gatherer.calls[0]
    # The zero count is CARRIED, not suppressed: it is what the all-rank
    # decision is made from.
    assert local_payload["token_type_gate"]["eligible_segment_count"] == 0
    assert local_payload["base_ce"]["eligible_segment_count"] == 2
    assert excinfo.value.code == "loss.segment_balanced_zero_eligible"
    assert excinfo.value.context["term"] == "token_type_gate"
    assert excinfo.value.context["zero_eligible_ranks"] == [0]


def test_local_zero_with_global_nonzero_still_fails_the_planned_step() -> None:
    """Pinned semantics: which runs fail does not change."""

    runner = _coordinate_gate_runner()
    peer_payload = _nonzero_rank_payload(runner)
    # The peer's shard alone makes the GLOBAL sum non-zero; the planned step
    # must still fail (rescuing it would be a loss-semantics change).
    assert peer_payload["token_type_gate"]["eligible_segment_count"] > 0
    gatherer = _RecordingGatherer(peer_payload)

    with pytest.raises(LossContractError) as excinfo:
        runner.prepare_planned_step(
            (_sequence_without_coordinates(),),
            denominator_gatherer=gatherer,
            world_size=2,
            rank=0,
        )

    assert gatherer.called is True
    assert excinfo.value.code == "loss.segment_balanced_zero_eligible"
    assert excinfo.value.context["eligible_segment_count"] > 0


def test_every_rank_converges_the_identical_zero_eligible_failure() -> None:
    """Collective cleanliness: the failure is rank-symmetric after the gather."""

    runner = _coordinate_gate_runner()
    zero_shard = (_sequence_without_coordinates(),)
    nonzero_shard = (_sequence_with_coordinates(),)
    world = (
        _rank_payload(runner, zero_shard),
        _rank_payload(runner, nonzero_shard),
    )

    def _gatherer(
        _local_payload: Mapping[str, Mapping[str, Any]],
    ) -> tuple[Mapping[str, Mapping[str, Any]], ...]:
        return tuple(dict(payload) for payload in world)

    errors = []
    for rank, shard in ((0, zero_shard), (1, nonzero_shard)):
        with pytest.raises(LossContractError) as excinfo:
            runner.prepare_planned_step(
                shard,
                denominator_gatherer=_gatherer,
                world_size=2,
                rank=rank,
            )
        errors.append(excinfo.value)

    zero_rank_error, nonzero_rank_error = errors
    assert zero_rank_error.code == nonzero_rank_error.code
    assert str(zero_rank_error) == str(nonzero_rank_error)
    # Built only from gathered facts, so it is identical on every rank.
    assert zero_rank_error.context == nonzero_rank_error.context
    assert zero_rank_error.context["zero_eligible_ranks"] == [0]


def test_world_size_one_retains_the_local_zero_eligible_raise() -> None:
    """No collective exists at world size one: the local raise is unchanged."""

    runner = _coordinate_gate_runner()

    with pytest.raises(LossContractError) as excinfo:
        runner.prepare_planned_step((_sequence_without_coordinates(),))

    error = excinfo.value
    assert error.code == "loss.segment_balanced_zero_eligible"
    assert set(error.context) == {
        "term",
        "context_count",
        "selected_atom_count",
        "skipped_segment_count",
    }
    assert error.context["term"] == "token_type_gate"
    assert error.context["context_count"] == 1
    assert error.context["selected_atom_count"] == 0
    assert error.context["skipped_segment_count"] == 2


def test_world_size_one_never_invokes_the_gatherer() -> None:
    runner = _coordinate_gate_runner()
    gatherer = _RecordingGatherer(_nonzero_rank_payload(runner))

    with pytest.raises(LossContractError) as excinfo:
        runner.prepare_planned_step(
            (_sequence_without_coordinates(),),
            denominator_gatherer=gatherer,
            world_size=1,
            rank=0,
        )

    assert gatherer.called is False
    assert excinfo.value.code == "loss.segment_balanced_zero_eligible"


def test_multi_rank_without_a_gatherer_retains_the_local_raise() -> None:
    """No gatherer means no collective to desync: precedence is preserved."""

    runner = _coordinate_gate_runner()

    with pytest.raises(LossContractError) as excinfo:
        runner.prepare_planned_step(
            (_sequence_without_coordinates(),),
            world_size=2,
            rank=0,
        )

    assert excinfo.value.code == "loss.segment_balanced_zero_eligible"
    assert excinfo.value.context["term"] == "token_type_gate"


def test_all_ranks_nonzero_planned_step_is_unchanged() -> None:
    runner = _coordinate_gate_runner()
    gatherer = _RecordingGatherer(_nonzero_rank_payload(runner))

    plan = runner.prepare_planned_step(
        (_sequence_with_coordinates(),),
        denominator_gatherer=gatherer,
        world_size=2,
        rank=1,
    )

    assert gatherer.called is True
    assert plan.denominator_scope == "planned_step_global"
    assert plan.backend_gradient_scale == 2.0
    for name in ("base_ce", "token_type_gate"):
        denominator = plan.denominators[name]
        assert denominator.denominator_scope == "planned_step_global"
        assert denominator.eligible_segment_count == 4
        assert denominator.selected_atom_count == 4
        assert denominator.context_count == 2


def _coordinate_gate_runner() -> LossRunner:
    return LossRunner(
        base_ce_weight=1.0,
        token_type_gate_weight=0.1,
        token_type_gate_groups=("coordinate",),
    )


def _nonzero_rank_payload(runner: LossRunner) -> dict[str, dict[str, Any]]:
    """A schema-complete peer payload built from a real nonzero shard."""

    return _rank_payload(runner, (_sequence_with_coordinates(),))


def _rank_payload(
    runner: LossRunner, shard: tuple[TokenSequence, ...]
) -> dict[str, dict[str, Any]]:
    """One rank's gather payload, produced by the production denominator path."""

    return {
        name: dict(denominator.to_artifact_dict())
        for name, denominator in _local_denominators(runner, shard).items()
    }


def _local_denominators(
    runner: LossRunner, shard: tuple[TokenSequence, ...]
) -> dict[str, SegmentBalancedDenominator]:
    return {
        active.binding.name: _build_denominator_from_token_sequences(
            active.binding.name,
            shard,
            token_types=active.token_types,
        )
        for active in runner._active_token_losses()
    }


def _sequence_without_coordinates() -> TokenSequence:
    return _token_sequence(
        segments=(_segment(0, 0, 2), _segment(1, 2, 4)),
        atoms=(
            _atom(segment_index=0, target_position=1, token_id=7),
            _atom(segment_index=1, target_position=3, token_id=7),
        ),
        pack_length=4,
    )


def _sequence_with_coordinates() -> TokenSequence:
    return _token_sequence(
        segments=(_segment(0, 0, 2), _segment(1, 2, 4)),
        atoms=(
            _atom(
                segment_index=0, target_position=1, token_id=3, token_type="coordinate"
            ),
            _atom(
                segment_index=1, target_position=3, token_id=4, token_type="coordinate"
            ),
        ),
        pack_length=4,
    )


def _token_sequence(
    *,
    segments: tuple[PackedSegment, ...],
    atoms: tuple[TokenAtom, ...],
    pack_length: int,
) -> TokenSequence:
    return TokenSequence(
        pack_index=0,
        input_ids=tuple(0 for _ in range(pack_length)),
        segments=segments,
        atoms=atoms,
        spans=(),
    )


def _segment(segment_index: int, start: int, end: int) -> PackedSegment:
    return PackedSegment(
        pack_index=0,
        segment_index=segment_index,
        example_index=segment_index,
        example_id=f"ex-{segment_index}",
        start=start,
        end=end,
    )


def _atom(
    *,
    segment_index: int,
    target_position: int,
    token_id: int,
    token_type: str = "desc_text",
) -> TokenAtom:
    return TokenAtom(
        pack_index=0,
        segment_index=segment_index,
        example_index=segment_index,
        example_id=f"ex-{segment_index}",
        target_position=target_position,
        token_id=token_id,
        token_type=token_type,
        text="x",
        logical_target_position=target_position,
        object_id=None,
        field=None,
        source="unit",
        coordinate_target=None,
    )
