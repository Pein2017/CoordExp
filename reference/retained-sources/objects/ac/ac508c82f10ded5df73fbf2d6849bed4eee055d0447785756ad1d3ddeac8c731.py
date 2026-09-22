"""Token-chronology comparisons for sampled-rescue candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .artifacts import CallRecord, GeometryMode, box_iou, cluster_geometry_modes


OBJECT_REF_START = 151646
OBJECT_REF_END = 151647
BOX_START = 151648
BOX_END = 151649


def common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    index = 0
    while index < min(len(left), len(right)) and int(left[index]) == int(right[index]):
        index += 1
    return index


def trajectory_rows(tokens: Sequence[int]) -> tuple[tuple[int, int], ...]:
    """Return token spans for complete object rows, preserving generation order."""

    rows: list[tuple[int, int]] = []
    start: int | None = None
    for index, token in enumerate(tokens):
        if int(token) == OBJECT_REF_START:
            start = index
        elif int(token) == BOX_END and start is not None:
            rows.append((start, index + 1))
            start = None
    return tuple(rows)


@dataclass(frozen=True)
class BoundaryComparison:
    greedy_request_id: str
    sampled_request_id: str
    common_prompt: bool
    prompt_token_count: int
    common_generated_prefix_tokens: int
    first_divergence_token: int | None
    greedy_row_count: int
    sampled_row_count: int
    sampled_rescue_object_ids: tuple[str, ...]
    sampled_geometry_only_mode_ids: tuple[str, ...]
    legal_prefix_evidence: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "greedy_request_id": self.greedy_request_id,
            "sampled_request_id": self.sampled_request_id,
            "common_prompt": self.common_prompt,
            "prompt_token_count": self.prompt_token_count,
            "common_generated_prefix_tokens": self.common_generated_prefix_tokens,
            "first_divergence_token": self.first_divergence_token,
            "greedy_row_count": self.greedy_row_count,
            "sampled_row_count": self.sampled_row_count,
            "sampled_rescue_object_ids": list(self.sampled_rescue_object_ids),
            "sampled_geometry_only_mode_ids": list(self.sampled_geometry_only_mode_ids),
            "legal_prefix_evidence": self.legal_prefix_evidence,
        }


def _mode_contains(mode: GeometryMode, pred: Any) -> bool:
    return box_iou(mode.representative_box, pred.box) >= 0.50


def compare_greedy_to_samples(
    greedy: CallRecord,
    sampled: Sequence[CallRecord],
    *,
    iou_threshold: float = 0.50,
    accepted_ledger_rows: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[BoundaryComparison, ...]:
    """Compare one exact greedy anchor against sampled calls at the same image."""

    greedy_modes = cluster_geometry_modes((greedy,), iou_threshold=iou_threshold)
    greedy_prompt = tuple(greedy.prompt_token_ids)
    comparisons: list[BoundaryComparison] = []
    for call in sampled:
        if call.image_id != greedy.image_id:
            continue
        sampled_modes = cluster_geometry_modes((call,), iou_threshold=iou_threshold)
        greedy_boxes = [pred.box for pred in greedy.predictions]
        geometry_only = tuple(
            mode.mode_id
            for mode in sampled_modes
            if not any(box_iou(mode.representative_box, box) >= iou_threshold for box in greedy_boxes)
        )
        # A geometry-only difference is deliberately not called a rescue.  A
        # rescue requires a reviewed, accepted ledger object that is absent
        # from greedy and present in this sampled call; callers may supply
        # accepted ledger rows for that stronger claim.
        rescue: tuple[str, ...] = ()
        if accepted_ledger_rows:
            from .artifacts import match_mode_to_ledger

            greedy_object_ids = {
                match_mode_to_ledger(mode, accepted_ledger_rows)["object_identifier"]
                for mode in greedy_modes
                if match_mode_to_ledger(mode, accepted_ledger_rows)["match_status"] == "reviewed_match"
            }
            sampled_rescues: list[str] = []
            for mode in sampled_modes:
                match = match_mode_to_ledger(mode, accepted_ledger_rows)
                if (
                    match["match_status"] == "reviewed_match"
                    and match["object_identifier"] not in greedy_object_ids
                ):
                    object_id = str(match["object_identifier"])
                    if object_id not in sampled_rescues:
                        sampled_rescues.append(object_id)
            rescue = tuple(sampled_rescues)
        divergence = common_prefix_length(greedy.generated_token_ids, call.generated_token_ids)
        prompt_equal = greedy_prompt == tuple(call.prompt_token_ids)
        comparisons.append(
            BoundaryComparison(
                greedy_request_id=greedy.request_id,
                sampled_request_id=call.request_id,
                common_prompt=prompt_equal,
                prompt_token_count=len(greedy_prompt) if prompt_equal else min(len(greedy_prompt), len(call.prompt_token_ids)),
                common_generated_prefix_tokens=divergence,
                first_divergence_token=(divergence if divergence < min(len(greedy.generated_token_ids), len(call.generated_token_ids)) else None),
                greedy_row_count=len(trajectory_rows(greedy.generated_token_ids)),
                sampled_row_count=len(trajectory_rows(call.generated_token_ids)),
                sampled_rescue_object_ids=rescue,
                sampled_geometry_only_mode_ids=geometry_only,
                legal_prefix_evidence={
                    "prompt_token_ids_equal": prompt_equal,
                    "prompt_token_hash_equal": greedy.prompt_token_hash == call.prompt_token_hash,
                    "generated_prefix_token_ids_equal_through": divergence,
                    "greedy_prompt_token_hash": greedy.prompt_token_hash,
                    "sampled_prompt_token_hash": call.prompt_token_hash,
                },
            )
        )
    return tuple(comparisons)
