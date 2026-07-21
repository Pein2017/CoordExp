"""Atomic event planning over existing isolated packed Qwen segments."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from src.common.errors import PackingContractError
from src.config.models import ProcessorConfig
from src.packing.planner import PackedSequence, PackedSegment, plan_packed_sequences
from src.qwen.positions import QwenPositionInputs, build_qwen_position_inputs
from src.rollout_calibration.replay import (
    ExactReplaySegment,
    build_exact_replay_segment,
)
from src.rollout_calibration.state_bank import (
    CoordinateDecision,
    LoadedStateBank,
    SelectedSite,
    StateBankCandidate,
    StateBankEvent,
)


@dataclass(frozen=True)
class CalibrationSelectedSite:
    candidate_id: str
    segment_index: int
    candidate_token_offset: int
    intended_token_type: str
    physical_target_position: int
    physical_logits_position: int


@dataclass(frozen=True)
class CalibrationCandidateMetadata:
    candidate_id: str
    segment_index: int
    role: str
    harmful_kind: str | None
    physical_owner_id: str | None
    coverage_status: str
    entity_review_status: str
    geometry_review_status: str
    entity_eligible: bool
    geometry_eligible: bool
    owner_resolution_candidate_interval: tuple[int, int] | None
    owner_resolution_physical_target_interval: tuple[int, int] | None
    coordinate_decision: CoordinateDecision | None
    coordinate_physical_target_position: int | None
    coordinate_physical_logits_position: int | None
    selected_sites: tuple[CalibrationSelectedSite, ...]


@dataclass(frozen=True)
class CalibrationEventMetadata:
    event_id: str
    image_id: int
    split: str
    entity_transition_eligible: bool
    coordinate_boundary_eligible: bool
    candidates: tuple[CalibrationCandidateMetadata, ...]
    selected_logits_positions: tuple[int, ...]
    positive_path_imitation_eligible: bool = False
    image_balanced_event_weight: float = 1.0

    @property
    def selected_causal_logits_positions(self) -> tuple[int, ...]:
        """Compatibility name used by the trainer's calibration boundary."""

        return self.selected_logits_positions

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "image_id": self.image_id,
            "split": self.split,
            "entity_transition_eligible": self.entity_transition_eligible,
            "coordinate_boundary_eligible": self.coordinate_boundary_eligible,
            "positive_path_imitation_eligible": self.positive_path_imitation_eligible,
            "image_balanced_event_weight": self.image_balanced_event_weight,
            "candidate_count": len(self.candidates),
            "selected_logits_positions": list(self.selected_logits_positions),
        }


@dataclass(frozen=True)
class CalibrationPlannedMicroStep:
    pack: PackedSequence
    replay_segments: tuple[ExactReplaySegment, ...]
    position_inputs: QwenPositionInputs
    calibration_metadata: CalibrationEventMetadata


def plan_calibration_micro_steps(
    bank: LoadedStateBank,
    *,
    split: str,
    components: Any,
    processor_config: ProcessorConfig,
    global_max_length: int,
    image_token_id: int,
) -> tuple[CalibrationPlannedMicroStep, ...]:
    """Build one atomic pack/micro-step per event.

    This deliberately does not attempt multi-pack event placement.  Requiring
    the full candidate group to fit one physical packed row makes the existing
    rank/cache stride incapable of separating a grouped objective.
    """

    planned: list[CalibrationPlannedMicroStep] = []
    for pack_index, event in enumerate(bank.records_for_split(split)):
        segments = tuple(
            build_exact_replay_segment(
                event,
                candidate,
                components=components,
                processor_config=processor_config,
                global_max_length=global_max_length,
                image_token_id=image_token_id,
            )
            for candidate in event.candidates
        )
        required_length = sum(segment.input_length for segment in segments)
        if required_length > global_max_length:
            raise PackingContractError(
                "complete calibration event does not fit one atomic packed micro-step",
                code="rollout_calibration.event_capacity",
                context={
                    "event_id": event.event_id,
                    "candidate_lengths": {
                        segment.candidate_id: segment.input_length
                        for segment in segments
                    },
                    "required_length": required_length,
                    "global_max_length": global_max_length,
                },
            )
        packs = plan_packed_sequences(segments, global_max_length=global_max_length)
        if len(packs) != 1 or len(packs[0].segments) != len(event.candidates):
            raise PackingContractError(
                "atomic calibration planner produced a partial candidate group",
                code="rollout_calibration.partial_event",
                context={
                    "event_id": event.event_id,
                    "pack_count": len(packs),
                    "candidate_count": len(event.candidates),
                },
            )
        pack = _reindex_pack(packs[0], pack_index=pack_index)
        positions = build_qwen_position_inputs(
            pack,
            segments,
            image_token_id=image_token_id,
        )
        metadata = _materialize_event_metadata(event, pack, segments)
        planned.append(
            CalibrationPlannedMicroStep(
                pack=pack,
                replay_segments=segments,
                position_inputs=positions,
                calibration_metadata=metadata,
            )
        )
    if not planned:
        raise PackingContractError(
            "calibration state-bank split contains no events",
            code="rollout_calibration.empty_split",
            context={"split": split, "bank_id": bank.manifest.bank_id},
        )
    return tuple(planned)


def _reindex_pack(pack: PackedSequence, *, pack_index: int) -> PackedSequence:
    segments = tuple(
        replace(segment, pack_index=pack_index, segment_index=index)
        for index, segment in enumerate(pack.segments)
    )
    return replace(pack, pack_index=pack_index, segments=segments)


def _materialize_event_metadata(
    event: StateBankEvent,
    pack: PackedSequence,
    replay_segments: tuple[ExactReplaySegment, ...],
) -> CalibrationEventMetadata:
    segment_by_example = {segment.example_id: segment for segment in pack.segments}
    replay_by_candidate = {segment.candidate_id: segment for segment in replay_segments}
    candidates: list[CalibrationCandidateMetadata] = []
    selected_logits_positions: set[int] = set()
    for candidate in event.candidates:
        replay = replay_by_candidate[candidate.candidate_id]
        segment = segment_by_example[replay.example_id]
        metadata = _candidate_metadata(candidate, segment=segment, replay=replay)
        candidates.append(metadata)
        selected_logits_positions.update(
            site.physical_logits_position for site in metadata.selected_sites
        )
        if metadata.owner_resolution_physical_target_interval is not None:
            start, end = metadata.owner_resolution_physical_target_interval
            selected_logits_positions.update(range(start - 1, end - 1))
        if metadata.coordinate_physical_logits_position is not None:
            selected_logits_positions.add(metadata.coordinate_physical_logits_position)
    return CalibrationEventMetadata(
        event_id=event.event_id,
        image_id=event.image.image_id,
        split=event.split,
        entity_transition_eligible=event.entity_transition_eligible,
        coordinate_boundary_eligible=event.coordinate_boundary_eligible,
        candidates=tuple(candidates),
        selected_logits_positions=tuple(sorted(selected_logits_positions)),
        positive_path_imitation_eligible=event.positive_path_imitation_eligible,
        image_balanced_event_weight=event.image_balanced_event_weight,
    )


def _candidate_metadata(
    candidate: StateBankCandidate,
    *,
    segment: PackedSegment,
    replay: ExactReplaySegment,
) -> CalibrationCandidateMetadata:
    selected_sites = tuple(
        _selected_site(candidate, site, segment=segment, replay=replay)
        for site in candidate.selected_sites
    )
    owner_targets = None
    if candidate.owner_resolution_interval is not None:
        start, end = candidate.owner_resolution_interval
        owner_targets = (
            segment.start + replay.candidate_token_start + start,
            segment.start + replay.candidate_token_start + end,
        )
    coordinate_target = None
    coordinate_logits = None
    if candidate.coordinate_decision is not None:
        coordinate_target = (
            segment.start
            + replay.candidate_token_start
            + candidate.coordinate_decision.candidate_token_offset
        )
        coordinate_logits = coordinate_target - 1
    return CalibrationCandidateMetadata(
        candidate_id=candidate.candidate_id,
        segment_index=segment.segment_index,
        role=candidate.role,
        harmful_kind=candidate.harmful_kind,
        physical_owner_id=candidate.physical_owner_id,
        coverage_status=candidate.coverage_status,
        entity_review_status=candidate.entity_review_status,
        geometry_review_status=candidate.geometry_review_status,
        entity_eligible=candidate.entity_eligible,
        geometry_eligible=candidate.geometry_eligible,
        owner_resolution_candidate_interval=candidate.owner_resolution_interval,
        owner_resolution_physical_target_interval=owner_targets,
        coordinate_decision=candidate.coordinate_decision,
        coordinate_physical_target_position=coordinate_target,
        coordinate_physical_logits_position=coordinate_logits,
        selected_sites=selected_sites,
    )


def _selected_site(
    candidate: StateBankCandidate,
    site: SelectedSite,
    *,
    segment: PackedSegment,
    replay: ExactReplaySegment,
) -> CalibrationSelectedSite:
    target = segment.start + replay.candidate_token_start + site.candidate_token_offset
    return CalibrationSelectedSite(
        candidate_id=candidate.candidate_id,
        segment_index=segment.segment_index,
        candidate_token_offset=site.candidate_token_offset,
        intended_token_type=site.intended_token_type,
        physical_target_position=target,
        physical_logits_position=target - 1,
    )


__all__ = [
    "CalibrationCandidateMetadata",
    "CalibrationEventMetadata",
    "CalibrationPlannedMicroStep",
    "CalibrationSelectedSite",
    "plan_calibration_micro_steps",
]
