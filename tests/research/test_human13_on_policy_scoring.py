from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from scripts.research.build_human13_on_policy_frontier import (
    FrontierCandidateAlias,
    FrontierDuplicateEvent,
    FrontierImage,
    FrontierRow,
)
from scripts.research.human13_on_policy_scoring import (
    prepare_on_policy_candidate_scoring,
    score_on_policy_frontier_candidates,
)


IMAGE_TOKEN_ID = 151655
PROMPT_IDS = (10, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID)
VOCAB_SIZE = 512


@dataclass(frozen=True)
class FakeImagePlan:
    merge_size: int = 2


@dataclass(frozen=True)
class FakeImageEncoding:
    image_grid_thw: tuple[int, int, int] = (1, 4, 4)
    merged_visual_tokens: int = 4
    plan: FakeImagePlan = field(default_factory=FakeImagePlan)
    pixel_values: torch.Tensor = field(default_factory=lambda: torch.zeros((16, 2)))


@dataclass(frozen=True)
class Skeleton:
    example_id: str
    input_ids: tuple[int, ...]
    prompt_token_count: int
    image_pad_physical_start: int = 1
    image_pad_physical_end: int = 5
    image_grid_thw: tuple[int, int, int] = (1, 4, 4)
    merge_size: int = 2
    image_token_id: int = IMAGE_TOKEN_ID
    image_encoding: FakeImageEncoding = field(default_factory=FakeImageEncoding)


class FakeTokenizer:
    def __len__(self) -> int:
        return VOCAB_SIZE


def _frontier() -> FrontierImage:
    rows = (
        FrontierRow(0, "person", (0, 0, 10, 10), 0, 2, (30, 31)),
        FrontierRow(1, "cat", (20, 0, 30, 10), 3, 5, (40, 41)),
        FrontierRow(2, "dog", (20.1, 0, 30.1, 10), 5, 7, (40, 49)),
    )
    return FrontierImage(
        image_id=7,
        trajectory_id="accepted:3",
        generated_token_ids=(30, 31, 777, 40, 41, 40, 49, 999),
        parser="compact_object_box_closed_only",
        parser_status="complete",
        stop_reason="im_end",
        rows=rows,
        canonical_owner_ids=("g0",),
        constrained_protected_owner_ids=("g0",),
        covered_h_owner_ids=(),
        uncovered_h_owner_ids=("h0", "h1"),
        candidate_aliases=(
            FrontierCandidateAlias("h0", "a", "k0", 1, 0.91, (100, 101)),
            FrontierCandidateAlias("h0", "b", "k1", 2, 0.92, (110, 111)),
            FrontierCandidateAlias("h0", "c", "k2", 3, 0.93, (120, 121)),
            FrontierCandidateAlias("h1", "d", "k3", 4, 0.94, (130, 131)),
        ),
        duplicate_events=(FrontierDuplicateEvent(2, 1, 0.98),),
    )


class FakePackedForward:
    def __init__(
        self,
        deficits: dict[str, float],
        *,
        omit_last: bool = False,
        vocab_size: int = VOCAB_SIZE,
        nonfinite: bool = False,
    ) -> None:
        self.deficits = deficits
        self.omit_last = omit_last
        self.vocab_size = vocab_size
        self.nonfinite = nonfinite
        self.calls: list[tuple[Any, tuple[int, ...]]] = []

    def __call__(
        self,
        model: Any,
        runtime: Any,
        tokenizer: Any,
        packed: Any,
        positions: tuple[int, ...],
    ) -> Any:
        del model, runtime, tokenizer
        self.calls.append((packed, positions))
        returned = positions[:-1] if self.omit_last else tuple(reversed(positions))
        rows = []
        for position in returned:
            segment = next(
                item for item in packed.pack.segments if item.start <= position < item.end
            )
            logical = next(
                item
                for item in packed.logical_segments
                if item.segment_id == segment.example_id
            )
            local_position = position - segment.start
            target = logical.encoded_example.input_ids[local_position + 1]
            alias_id = logical.encoded_example.human13_candidate_path.alias_id
            row = torch.zeros(self.vocab_size, dtype=torch.bfloat16)
            row[target] = 5.0
            if local_position == logical.encoded_example.human13_candidate_positions[0]:
                row[target] = -self.deficits[alias_id]
            if self.nonfinite:
                row[0] = torch.nan
            rows.append(row)
        return SimpleNamespace(
            logits=torch.stack(rows).unsqueeze(0),
            logits_position_ids=returned,
        )


class FakeHFScorer:
    model_dtype = "torch.float32"
    attention_implementation = "sdpa"

    def __init__(
        self,
        deficits: dict[str, float],
        *,
        omit_last: bool = False,
        vocab_size: int = VOCAB_SIZE,
    ) -> None:
        self.deficits = deficits
        self.omit_last = omit_last
        self.vocab_size = vocab_size
        self.calls: list[tuple[Any, tuple[int, ...]]] = []

    def score_causal_logits(
        self, encoded_example: Any, causal_positions: tuple[int, ...]
    ) -> Any:
        self.calls.append((encoded_example, causal_positions))
        returned = causal_positions[:-1] if self.omit_last else tuple(
            reversed(causal_positions)
        )
        rows = []
        first = encoded_example.human13_candidate_positions[0]
        deficit = self.deficits[encoded_example.human13_candidate_path.alias_id]
        for position in returned:
            target = encoded_example.input_ids[position + 1]
            row = torch.zeros(self.vocab_size, dtype=torch.float32)
            row[target] = -deficit if position == first else 5.0
            rows.append(row)
        return SimpleNamespace(
            logits=torch.stack(rows).unsqueeze(0),
            logits_position_ids=returned,
        )


def _kwargs() -> dict[str, Any]:
    return {
        "frontier_images": {7: _frontier()},
        "prompt_skeletons": {
            7: Skeleton("image:7", PROMPT_IDS, len(PROMPT_IDS))
        },
        "packed_model": object(),
        "packed_runtime": SimpleNamespace(
            accelerator=SimpleNamespace(device="cpu")
        ),
        "tokenizer": FakeTokenizer(),
    }


def test_prepare_materializes_every_alias_after_the_deduplicated_current_prefix() -> None:
    prepared = prepare_on_policy_candidate_scoring(
        frontier_images={7: _frontier()},
        prompt_skeletons={7: Skeleton("image:7", PROMPT_IDS, len(PROMPT_IDS))},
    )

    assert len(prepared.bindings) == 4
    assert len(prepared.packed_plan.logical_segments) == 4
    assert all(pack.pack.to_artifact_dict()["padding_tokens"] == 0 for pack in prepared.packed_plan.packs)
    by_alias = {binding.path.alias_id: binding for binding in prepared.bindings}
    for alias_id, row in {
        "a": (100, 101),
        "b": (110, 111),
        "c": (120, 121),
        "d": (130, 131),
    }.items():
        binding = by_alias[alias_id]
        segment = next(
            item
            for item in prepared.packed_plan.logical_segments
            if item.segment_id == binding.segment_id
        )
        # Duplicate row (40, 49) and terminal 999 are absent; the natural boundary stays.
        assert segment.encoded_example.input_ids == (
            *PROMPT_IDS,
            30,
            31,
            777,
            40,
            41,
            *row,
        )
        assert binding.local_causal_positions == (9, 10)
        assert segment.encoded_example.human13_candidate_positions == (9, 10)
    prefix = prepared.prefix_receipts[0]
    assert prefix.raw_natural_token_count == 8
    assert prefix.training_prefix_token_count == 5
    assert prefix.removed_duplicate_row_orders == (2,)


def test_packed_prefilter_hf_rescore_and_owner_shortlist_are_surface_aligned() -> None:
    packed = FakePackedForward({"a": 4.0, "b": 1.0, "c": 2.0, "d": 3.0})
    hf = FakeHFScorer({"a": 0.1, "b": 4.0, "c": 1.0, "d": 2.0})

    result = score_on_policy_frontier_candidates(
        **_kwargs(), hf_scorer=hf, packed_forward=packed
    )

    assert len(result.packed_receipts) == 4
    # Packed prefilter keeps b/c for h0 and d for h1, so HF never sees filtered a.
    assert [call[0].human13_candidate_path.alias_id for call in hf.calls] == [
        "b",
        "c",
        "d",
    ]
    assert [score.path.alias_id for score in result.shortlist_by_image[7]] == [
        "c",
        "d",
    ]
    assert [score.path.owner_id for score in result.shortlist_by_image[7]] == [
        "h0",
        "h1",
    ]
    cross = {item.path.alias_id: item for item in result.cross_surface_receipts}
    assert set(cross) == {"b", "c", "d"}
    assert cross["c"].packed_causal_positions != cross["c"].hf_causal_positions
    assert cross["c"].hf_causal_positions == (9, 10)
    assert len(cross["c"].packed_evidence.logits[0]) == VOCAB_SIZE
    assert len(cross["c"].hf_evidence.logits[0]) == VOCAB_SIZE
    assert cross["c"].score.hf_barrier == pytest.approx(1.0)
    assert cross["c"].score.packed_barrier == pytest.approx(2.0)
    assert result.receipt["packed_candidate_count"] == 4
    assert result.receipt["hf_candidate_count"] == 3
    assert result.receipt["shortlisted_owner_count"] == 2
    with pytest.raises(TypeError):
        result.receipt["hf_candidate_count"] = 99  # type: ignore[index]
    with pytest.raises(TypeError):
        result.shortlist_by_image[7] = ()  # type: ignore[index]


@pytest.mark.parametrize(
    ("packed", "hf", "message"),
    (
        (
            FakePackedForward({"a": 4, "b": 1, "c": 2, "d": 3}, omit_last=True),
            FakeHFScorer({"b": 1, "c": 1, "d": 1}),
            "packed.*position coverage",
        ),
        (
            FakePackedForward({"a": 4, "b": 1, "c": 2, "d": 3}),
            FakeHFScorer({"b": 1, "c": 1, "d": 1}, omit_last=True),
            "HF.*position coverage",
        ),
        (
            FakePackedForward(
                {"a": 4, "b": 1, "c": 2, "d": 3}, vocab_size=VOCAB_SIZE - 1
            ),
            FakeHFScorer({"b": 1, "c": 1, "d": 1}),
            "full vocabulary",
        ),
        (
            FakePackedForward(
                {"a": 4, "b": 1, "c": 2, "d": 3}, nonfinite=True
            ),
            FakeHFScorer({"b": 1, "c": 1, "d": 1}),
            "finite",
        ),
    ),
)
def test_scoring_fails_closed_on_position_vocab_and_finite_evidence(
    packed: FakePackedForward, hf: FakeHFScorer, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        score_on_policy_frontier_candidates(
            **_kwargs(), hf_scorer=hf, packed_forward=packed
        )


def test_wrong_hf_surface_and_nonfrontier_alias_fail_before_forward() -> None:
    packed = FakePackedForward({"a": 4, "b": 1, "c": 2, "d": 3})
    hf = FakeHFScorer({"b": 1, "c": 1, "d": 1})
    hf.attention_implementation = "flash_attention_2"
    with pytest.raises(ValueError, match="SDPA"):
        score_on_policy_frontier_candidates(
            **_kwargs(), hf_scorer=hf, packed_forward=packed
        )
    assert not packed.calls

    frontier = _frontier()
    invalid = replace(frontier, uncovered_h_owner_ids=("h1",))
    with pytest.raises(ValueError, match="uncovered"):
        prepare_on_policy_candidate_scoring(
            frontier_images={7: invalid},
            prompt_skeletons={7: Skeleton("image:7", PROMPT_IDS, len(PROMPT_IDS))},
        )


def test_candidate_token_outside_full_tokenizer_vocabulary_fails_before_forward() -> None:
    frontier = _frontier()
    aliases = (
        replace(frontier.candidate_aliases[0], token_ids=(VOCAB_SIZE, 101)),
        *frontier.candidate_aliases[1:],
    )
    packed = FakePackedForward({"a": 4, "b": 1, "c": 2, "d": 3})
    with pytest.raises(ValueError, match="outside.*vocabulary"):
        score_on_policy_frontier_candidates(
            **{
                **_kwargs(),
                "frontier_images": {7: replace(frontier, candidate_aliases=aliases)},
            },
            hf_scorer=FakeHFScorer({"b": 1, "c": 1, "d": 1}),
            packed_forward=packed,
        )
    assert not packed.calls
