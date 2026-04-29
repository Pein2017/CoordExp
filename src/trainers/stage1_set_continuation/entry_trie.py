from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True)
class EntryTrieCandidate:
    object_index: int
    tokens: tuple[int, ...]
    token_types: tuple[str, ...]


@dataclass(frozen=True)
class EntryTrieTarget:
    token_id: int
    object_count: int
    probability: float


@dataclass(frozen=True)
class EntryTrieTargetStep:
    position: int
    teacher_token_id: int
    token_type: str
    active_object_count: int
    targets: tuple[EntryTrieTarget, ...]

    @property
    def is_branch(self) -> bool:
        return len(self.targets) > 1

    @property
    def valid_child_count(self) -> int:
        return len(self.targets)

    @property
    def target_entropy(self) -> float:
        return float(
            -sum(
                target.probability * math.log(target.probability)
                for target in self.targets
                if target.probability > 0.0
            )
        )


@dataclass
class _EntryTrieNode:
    terminal_count: int = 0
    children: dict[int, "_EntryTrieNode"] = field(default_factory=dict)
    edge_token_types: dict[int, str] = field(default_factory=dict)

    def descendant_count(self) -> int:
        return int(
            self.terminal_count
            + sum(child.descendant_count() for child in self.children.values())
        )


def _normalize_token_type(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"desc", "text", "description"}:
        return "text"
    if normalized in {"coord", "coordinate"}:
        return "coord"
    if normalized in {"struct", "structure", "structural", "json"}:
        return "structural"
    return "other"


def _coerce_candidate(candidate: EntryTrieCandidate) -> EntryTrieCandidate:
    tokens = tuple(int(token) for token in candidate.tokens)
    token_types = tuple(_normalize_token_type(value) for value in candidate.token_types)

    if not tokens:
        raise ValueError("entry-trie candidates must contain at least one token")
    if len(tokens) != len(token_types):
        raise ValueError("entry-trie candidate token_types must match tokens")

    return EntryTrieCandidate(
        object_index=int(candidate.object_index),
        tokens=tokens,
        token_types=token_types,
    )


def _insert_candidate(root: _EntryTrieNode, candidate: EntryTrieCandidate) -> None:
    node = root
    for token_id, token_type in zip(
        candidate.tokens,
        candidate.token_types,
        strict=True,
    ):
        child = node.children.setdefault(int(token_id), _EntryTrieNode())
        previous_type = node.edge_token_types.setdefault(int(token_id), token_type)
        if previous_type != token_type:
            node.edge_token_types[int(token_id)] = "other"
        node = child
    node.terminal_count += 1


def _validate_no_prefix_entries(node: _EntryTrieNode) -> None:
    if node.terminal_count and node.children:
        raise ValueError(
            "entry-trie candidates may not contain one serialized entry that is "
            "a strict prefix of another"
        )
    for child in node.children.values():
        _validate_no_prefix_entries(child)


def build_entry_trie_target_steps(
    *,
    candidates: Sequence[EntryTrieCandidate],
    teacher_tokens: Sequence[int],
) -> tuple[EntryTrieTargetStep, ...]:
    if not candidates:
        raise ValueError("entry-trie requires at least one candidate")

    normalized_candidates = tuple(
        _coerce_candidate(candidate) for candidate in candidates
    )
    teacher_path = tuple(int(token) for token in teacher_tokens)
    if not teacher_path:
        raise ValueError("entry-trie teacher_tokens must not be empty")

    root = _EntryTrieNode()
    for candidate in normalized_candidates:
        _insert_candidate(root, candidate)
    _validate_no_prefix_entries(root)

    steps: list[EntryTrieTargetStep] = []
    node = root
    for position, teacher_token_id in enumerate(teacher_path):
        active_count = node.descendant_count()
        if int(teacher_token_id) not in node.children:
            raise ValueError(
                f"entry-trie teacher token {int(teacher_token_id)} at position "
                f"{position} is not a valid remaining-candidate child"
            )

        targets = []
        for child_token_id in sorted(node.children):
            child = node.children[child_token_id]
            object_count = child.descendant_count()
            targets.append(
                EntryTrieTarget(
                    token_id=int(child_token_id),
                    object_count=object_count,
                    probability=float(object_count / max(active_count, 1)),
                )
            )

        steps.append(
            EntryTrieTargetStep(
                position=position,
                teacher_token_id=int(teacher_token_id),
                token_type=node.edge_token_types.get(int(teacher_token_id), "other"),
                active_object_count=active_count,
                targets=tuple(targets),
            )
        )
        node = node.children[int(teacher_token_id)]

    if not node.terminal_count:
        raise ValueError("entry-trie teacher path did not reach a serialized entry")

    return tuple(steps)


__all__ = [
    "EntryTrieCandidate",
    "EntryTrieTarget",
    "EntryTrieTargetStep",
    "build_entry_trie_target_steps",
]
