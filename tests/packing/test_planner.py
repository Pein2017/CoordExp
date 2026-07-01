from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.common.errors import PackingContractError
from src.packing.planner import plan_packed_sequences


def test_pack_planner_appends_until_overflow_then_commits() -> None:
    packs = plan_packed_sequences(
        [
            FakeEncodedExample("ex-0", (10, 11, 12, 13)),
            FakeEncodedExample("ex-1", (20, 21, 22)),
            FakeEncodedExample("ex-2", (30, 31, 32, 33, 34)),
        ],
        global_max_length=7,
    )

    assert len(packs) == 2
    assert packs[0].input_ids == (10, 11, 12, 13, 20, 21, 22)
    assert packs[0].length == 7
    assert [(seg.example_id, seg.start, seg.end) for seg in packs[0].segments] == [
        ("ex-0", 0, 4),
        ("ex-1", 4, 7),
    ]
    assert packs[1].input_ids == (30, 31, 32, 33, 34)
    assert [(seg.example_id, seg.start, seg.end) for seg in packs[1].segments] == [
        ("ex-2", 0, 5),
    ]


def test_pack_planner_exact_fit_stays_in_current_pack() -> None:
    packs = plan_packed_sequences(
        [
            FakeEncodedExample("ex-0", (1, 2)),
            FakeEncodedExample("ex-1", (3, 4, 5)),
        ],
        global_max_length=5,
    )

    assert len(packs) == 1
    assert packs[0].length == 5
    assert [segment.length for segment in packs[0].segments] == [2, 3]
    assert packs[0].to_artifact_dict()["padding_tokens"] == 0


def test_pack_planner_rejects_single_example_over_global_max_length() -> None:
    with pytest.raises(PackingContractError) as exc_info:
        plan_packed_sequences(
            [FakeEncodedExample("too-long", (1, 2, 3, 4))],
            global_max_length=3,
        )

    assert exc_info.value.code == "packing.example_too_long"
    assert exc_info.value.context["example_id"] == "too-long"
    assert exc_info.value.context["input_length"] == 4


def test_pack_planner_rejects_empty_encoded_example() -> None:
    with pytest.raises(PackingContractError) as exc_info:
        plan_packed_sequences(
            [FakeEncodedExample("empty", ())],
            global_max_length=3,
        )

    assert exc_info.value.code == "packing.example_empty"


def test_pack_planner_empty_input_returns_no_packs() -> None:
    assert plan_packed_sequences([], global_max_length=10) == ()


@dataclass(frozen=True)
class FakeEncodedExample:
    example_id: str
    input_ids: tuple[int, ...]

    @property
    def input_length(self) -> int:
        return len(self.input_ids)
