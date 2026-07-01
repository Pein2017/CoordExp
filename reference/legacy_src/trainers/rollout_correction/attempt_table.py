"""Lightweight rollout-correction rollout attempt wrappers.

This module deliberately wraps existing rollout view dictionaries instead of
owning parsing, UL consensus, or target construction.  The seam exists so the
trainer can reason in terms of current/peer attempts while preserving the
current view/meta contract consumed downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, MutableMapping, Sequence


RolloutAttemptRole = Literal["primary_attempt", "peer_attempt"]
RolloutAttemptSource = Literal["live", "missing_peer"]
RolloutResultTuple = tuple[list[int], str, str, list[int]]


@dataclass(frozen=True)
class RolloutAttemptRecord:
    sample_id: str
    sample_index: int
    role: RolloutAttemptRole
    source: RolloutAttemptSource
    rollout_index: int
    view: MutableMapping[str, Any]
    rollout_result: RolloutResultTuple | None = None

    @property
    def explicit_rollout_id(self) -> str | None:
        rollout_id = self.view.get("rollout_id")
        if rollout_id is None or not str(rollout_id):
            return None
        return str(rollout_id)

    @property
    def sequence_id(self) -> str:
        explicit = self.explicit_rollout_id
        if explicit:
            return str(explicit)
        return f"{self.sample_id}:rollout_index={int(self.rollout_index)}"

    def residual_event_identity(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"rollout_index": int(self.rollout_index)}
        explicit = self.explicit_rollout_id
        if explicit:
            payload["rollout_id"] = str(explicit)
        return payload


@dataclass(frozen=True)
class AttemptTable:
    sample_id: str
    sample_index: int
    primary: RolloutAttemptRecord
    peers: tuple[RolloutAttemptRecord, ...]

    @property
    def current_view(self) -> MutableMapping[str, Any]:
        return self.primary.view

    @property
    def peer_views(self) -> tuple[MutableMapping[str, Any], ...]:
        return tuple(peer.view for peer in self.peers)

    @property
    def views_for_ul(self) -> tuple[MutableMapping[str, Any], ...]:
        return (self.primary.view, *(peer.view for peer in self.peers))


def wrap_rollout_attempt_view(
    *,
    sample_id: str,
    sample_index: int,
    role: RolloutAttemptRole,
    source: RolloutAttemptSource,
    rollout_index: int,
    view: MutableMapping[str, Any],
    rollout_result: RolloutResultTuple | None = None,
) -> RolloutAttemptRecord:
    """Annotate an existing view and return its attempt wrapper.

    The view object is mutated in place to preserve the historical downstream
    contract.  Live attempts intentionally do not receive a synthetic
    ``rollout_id``; residual-set IR still derives the fallback sequence id from
    sample id plus rollout index.
    """

    view["rollout_index"] = int(rollout_index)
    view["rollout_role"] = str(role)

    return RolloutAttemptRecord(
        sample_id=str(sample_id),
        sample_index=int(sample_index),
        role=role,
        source=source,
        rollout_index=int(rollout_index),
        view=view,
        rollout_result=rollout_result,
    )


def build_attempt_table(
    *,
    sample_id: str,
    sample_index: int,
    primary: RolloutAttemptRecord,
    peers: Sequence[RolloutAttemptRecord],
) -> AttemptTable:
    return AttemptTable(
        sample_id=str(sample_id),
        sample_index=int(sample_index),
        primary=primary,
        peers=tuple(peers),
    )


__all__ = [
    "AttemptTable",
    "RolloutAttemptRecord",
    "RolloutAttemptRole",
    "RolloutAttemptSource",
    "RolloutResultTuple",
    "build_attempt_table",
    "wrap_rollout_attempt_view",
]
