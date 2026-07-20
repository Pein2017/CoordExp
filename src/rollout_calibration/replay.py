"""Exact-token candidate replay without decode-and-retokenize reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.common.errors import EncodingContractError
from src.config.models import ProcessorConfig
from src.qwen.images import QwenImageEncoding, plan_qwen_image
from src.rollout_calibration.state_bank import StateBankCandidate, StateBankEvent


@dataclass(frozen=True)
class ExactReplaySegment:
    """One isolated causal segment for one reviewed event candidate."""

    example_id: str
    event_id: str
    candidate_id: str
    input_ids: tuple[int, ...]
    image_pad_physical_start: int
    image_pad_physical_end: int
    candidate_token_start: int
    candidate_token_end: int
    image_encoding: QwenImageEncoding

    @property
    def input_length(self) -> int:
        return len(self.input_ids)

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "event_id": self.event_id,
            "candidate_id": self.candidate_id,
            "input_length": self.input_length,
            "image_pad_physical_start": self.image_pad_physical_start,
            "image_pad_physical_end": self.image_pad_physical_end,
            "candidate_token_start": self.candidate_token_start,
            "candidate_token_end": self.candidate_token_end,
            "candidate_token_count": self.candidate_token_end
            - self.candidate_token_start,
            "image_encoding": self.image_encoding.to_artifact_dict(),
        }


def build_exact_replay_segment(
    event: StateBankEvent,
    candidate: StateBankCandidate,
    *,
    components: Any,
    processor_config: ProcessorConfig,
    global_max_length: int,
) -> ExactReplaySegment:
    """Materialize exact stored ids and the existing no-resize image plan.

    The function intentionally never accesses ``components.tokenizer`` or a
    human-readable evidence string.
    """

    if candidate not in event.candidates:
        raise EncodingContractError(
            "exact replay candidate does not belong to the requested event",
            code="rollout_calibration.candidate_event_mismatch",
            context={
                "event_id": event.event_id,
                "candidate_id": candidate.candidate_id,
            },
        )
    example_id = f"{event.event_id}::{candidate.candidate_id}"
    candidate_start = len(event.executed_prompt_token_ids) + len(event.prefix_token_ids)
    input_ids = (
        *event.executed_prompt_token_ids,
        *event.prefix_token_ids,
        *candidate.token_ids,
    )
    if len(input_ids) > global_max_length:
        raise EncodingContractError(
            "exact replay candidate segment exceeds packing.global_max_length",
            code="rollout_calibration.segment_capacity",
            context={
                "event_id": event.event_id,
                "candidate_id": candidate.candidate_id,
                "input_length": len(input_ids),
                "global_max_length": global_max_length,
            },
        )
    image_encoding = plan_qwen_image(
        event.to_raw_example(example_id=example_id),
        components=components,
        processor_config=processor_config,
    )
    return ExactReplaySegment(
        example_id=example_id,
        event_id=event.event_id,
        candidate_id=candidate.candidate_id,
        input_ids=tuple(int(token_id) for token_id in input_ids),
        image_pad_physical_start=event.image_pad_interval[0],
        image_pad_physical_end=event.image_pad_interval[1],
        candidate_token_start=candidate_start,
        candidate_token_end=candidate_start + len(candidate.token_ids),
        image_encoding=image_encoding,
    )


__all__ = ["ExactReplaySegment", "build_exact_replay_segment"]
