"""Shared TypedDict/Literal schemas for CoordExp."""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping, Sequence, TypedDict, Literal

# Coordinate normalization spaces used across datasets/eval/prompting.
CoordSpace = Literal["pixel", "norm100", "norm1000"]

# Canonical geometry kinds.
GeometryKind = Literal["bbox_2d", "poly"]


class GeometryDict(TypedDict, total=False):
    """Canonical geometry representation used in JSONL and runtime objects."""

    bbox_2d: Sequence[float]
    poly: Sequence[float]
    desc: str
    score: float
    object_id: str
    attributes: Mapping[str, Any]


class ImageRecord(TypedDict, total=False):
    """Canonical dataset record for detection/grounding."""

    images: Sequence[str]
    width: int
    height: int
    objects: Sequence[GeometryDict]
    metadata: Mapping[str, Any]
    dataset: str
    dataset_name: str
    _fusion_source: str


class MessageContent(TypedDict, total=False):
    type: str
    text: str
    image: Any


class MessageDict(TypedDict, total=False):
    role: str
    content: Sequence[MessageContent]


class ConversationRecord(TypedDict, total=False):
    """Conversation-style record used by chat-style datasets."""

    messages: Sequence[MessageDict]
    metadata: Mapping[str, Any]


# Mutable clone helpers are often used in preprocessing; keep alias for clarity.
MutableRecord = MutableMapping[str, Any]

__all__ = [
    "CoordSpace",
    "GeometryKind",
    "GeometryDict",
    "ImageRecord",
    "MessageContent",
    "MessageDict",
    "ConversationRecord",
    "MutableRecord",
]
