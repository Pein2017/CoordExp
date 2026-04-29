from __future__ import annotations

import math

import pytest

from src.trainers.stage1_set_continuation.entry_trie import (
    EntryTrieCandidate,
    build_entry_trie_target_steps,
)


def _probs(step) -> dict[int, float]:
    return {int(item.token_id): float(item.probability) for item in step.targets}


def test_entry_trie_branches_at_desc_tokens_with_object_uniform_targets() -> None:
    candidates = [
        EntryTrieCandidate(
            object_index=0, tokens=(10, 20), token_types=("text", "coord")
        ),
        EntryTrieCandidate(
            object_index=1, tokens=(10, 21), token_types=("text", "coord")
        ),
        EntryTrieCandidate(
            object_index=2, tokens=(11, 22), token_types=("text", "coord")
        ),
        EntryTrieCandidate(
            object_index=3, tokens=(12, 23), token_types=("text", "coord")
        ),
    ]

    steps = build_entry_trie_target_steps(
        candidates=candidates,
        teacher_tokens=(10, 20),
    )

    assert steps[0].is_branch
    assert steps[0].token_type == "text"
    assert steps[0].active_object_count == 4
    assert steps[0].valid_child_count == 3
    assert _probs(steps[0]) == {
        10: pytest.approx(0.5),
        11: pytest.approx(0.25),
        12: pytest.approx(0.25),
    }
    assert steps[0].target_entropy == pytest.approx(
        -(0.5 * math.log(0.5) + 0.25 * math.log(0.25) + 0.25 * math.log(0.25))
    )


def test_entry_trie_continues_to_branch_at_same_desc_bbox_tokens() -> None:
    candidates = [
        EntryTrieCandidate(
            object_index=0,
            tokens=(10, 30, 40),
            token_types=("text", "coord", "coord"),
        ),
        EntryTrieCandidate(
            object_index=1,
            tokens=(10, 31, 41),
            token_types=("text", "coord", "coord"),
        ),
        EntryTrieCandidate(
            object_index=2,
            tokens=(11, 32, 42),
            token_types=("text", "coord", "coord"),
        ),
    ]

    steps = build_entry_trie_target_steps(
        candidates=candidates,
        teacher_tokens=(10, 30, 40),
    )

    assert steps[0].is_branch
    assert _probs(steps[0]) == {10: pytest.approx(2 / 3), 11: pytest.approx(1 / 3)}
    assert steps[1].is_branch
    assert steps[1].token_type == "coord"
    assert steps[1].active_object_count == 2
    assert _probs(steps[1]) == {30: pytest.approx(0.5), 31: pytest.approx(0.5)}
    assert not steps[2].is_branch
    assert _probs(steps[2]) == {40: pytest.approx(1.0)}


def test_entry_trie_uses_unique_path_when_coordinates_are_shared_until_later() -> None:
    candidates = [
        EntryTrieCandidate(
            object_index=0,
            tokens=(10, 30, 40, 50),
            token_types=("text", "coord", "coord", "coord"),
        ),
        EntryTrieCandidate(
            object_index=1,
            tokens=(10, 30, 41, 51),
            token_types=("text", "coord", "coord", "coord"),
        ),
    ]

    steps = build_entry_trie_target_steps(
        candidates=candidates,
        teacher_tokens=(10, 30, 40, 50),
    )

    assert not steps[0].is_branch
    assert not steps[1].is_branch
    assert steps[2].is_branch
    assert steps[2].token_type == "coord"
    assert _probs(steps[2]) == {40: pytest.approx(0.5), 41: pytest.approx(0.5)}


def test_entry_trie_treats_exact_duplicate_entries_as_multiplicity() -> None:
    candidates = [
        EntryTrieCandidate(
            object_index=0, tokens=(10, 20), token_types=("text", "coord")
        ),
        EntryTrieCandidate(
            object_index=1, tokens=(10, 20), token_types=("text", "coord")
        ),
        EntryTrieCandidate(
            object_index=2, tokens=(11, 21), token_types=("text", "coord")
        ),
    ]

    steps = build_entry_trie_target_steps(
        candidates=candidates,
        teacher_tokens=(10, 20),
    )

    assert steps[0].is_branch
    assert _probs(steps[0]) == {10: pytest.approx(2 / 3), 11: pytest.approx(1 / 3)}
    assert not steps[1].is_branch
    assert steps[1].active_object_count == 2
    assert _probs(steps[1]) == {20: pytest.approx(1.0)}


def test_entry_trie_rejects_teacher_path_not_in_remaining_candidates() -> None:
    candidates = [
        EntryTrieCandidate(
            object_index=0, tokens=(10, 20), token_types=("text", "coord")
        ),
    ]

    with pytest.raises(ValueError, match="teacher token"):
        build_entry_trie_target_steps(
            candidates=candidates,
            teacher_tokens=(11, 20),
        )


def test_entry_trie_rejects_empty_candidates_and_mismatched_token_types() -> None:
    with pytest.raises(ValueError, match="at least one"):
        build_entry_trie_target_steps(candidates=[], teacher_tokens=(1,))

    with pytest.raises(ValueError, match="token_types"):
        build_entry_trie_target_steps(
            candidates=[
                EntryTrieCandidate(
                    object_index=0,
                    tokens=(10, 20),
                    token_types=("text",),
                )
            ],
            teacher_tokens=(10, 20),
        )
