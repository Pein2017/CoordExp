from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.common.object_field_order import (
    ObjectFieldOrder,
    build_object_payload,
    normalize_object_field_order,
)
from src.utils.assistant_json import dumps_coordjson

_EMPTY_OBJECTS_TEXT = dumps_coordjson({"objects": []})
if not _EMPTY_OBJECTS_TEXT.endswith("]}"):
    raise ValueError("Unexpected CoordJSON empty-object rendering contract")

_OBJECTS_PREFIX = _EMPTY_OBJECTS_TEXT[:-2]
_GLOBAL_CLOSE_TEXT = "]}"


@dataclass(frozen=True)
class ObjectIndexSpan:
    """A character span for a concrete rendered object entry."""

    object_index: int
    start: int
    end: int


@dataclass(frozen=True)
class StructuralSpan:
    """A character span for global container-close text."""

    start: int
    end: int


@dataclass(frozen=True)
class IndexedObjectList:
    """Full rendered object list with deterministic per-index spans."""

    text: str
    entry_texts_by_index: tuple[str, ...]
    object_spans: tuple[ObjectIndexSpan, ...]
    global_close_start_span: StructuralSpan
    global_full_close_span: StructuralSpan

    @property
    def object_spans_by_index(self) -> dict[int, ObjectIndexSpan]:
        return {span.object_index: span for span in self.object_spans}


@dataclass(frozen=True)
class PrefixText:
    """Append-ready prefix text for a chosen ordered subset of objects."""

    text: str
    object_spans: tuple[ObjectIndexSpan, ...]


@dataclass(frozen=True)
class CandidateEntryText:
    """Candidate entry text followed by the global container close."""

    text: str
    candidate_span: ObjectIndexSpan
    global_close_start_span: StructuralSpan
    global_full_close_span: StructuralSpan


@dataclass(frozen=True)
class GlobalCloseText:
    """The global object-list close continuation for prefixes with no candidate."""

    text: str
    global_close_start_span: StructuralSpan
    global_full_close_span: StructuralSpan


def _resolve_object_geometry(
    obj: Mapping[str, Any],
) -> tuple[str, Any]:
    if "bbox_2d" in obj:
        return "bbox_2d", obj["bbox_2d"]
    if "poly" in obj:
        return "poly", obj["poly"]
    raise ValueError("Each object must define either 'bbox_2d' or 'poly'")


def _render_object_entry(
    obj: Mapping[str, Any], *, object_field_order: ObjectFieldOrder
) -> str:
    if "desc" not in obj or obj["desc"] is None:
        raise ValueError("Each object must define a non-null 'desc'")
    geometry_key, geometry_value = _resolve_object_geometry(obj)
    payload = build_object_payload(
        desc=str(obj["desc"]),
        geometry_key=geometry_key,
        geometry_value=geometry_value,
        object_field_order=object_field_order,
    )
    return dumps_coordjson(payload)


def _validate_indices(
    entry_texts_by_index: Sequence[str], indices: Sequence[int]
) -> tuple[int, ...]:
    seen: set[int] = set()
    resolved: list[int] = []
    for raw_index in indices:
        index = int(raw_index)
        if index < 0 or index >= len(entry_texts_by_index):
            raise IndexError(f"Object index out of range: {index}")
        if index in seen:
            raise ValueError(f"Duplicate object index in serialization request: {index}")
        seen.add(index)
        resolved.append(index)
    return tuple(resolved)


def render_indexed_object_list(
    objects: Sequence[Mapping[str, Any]],
    *,
    object_field_order: str = "desc_first",
) -> IndexedObjectList:
    """Render a full CoordJSON object list with stable per-index spans."""

    resolved_field_order = normalize_object_field_order(
        object_field_order,
        path="custom.object_field_order",
    )
    entry_texts = tuple(
        _render_object_entry(obj, object_field_order=resolved_field_order)
        for obj in objects
    )

    text = _OBJECTS_PREFIX
    object_spans: list[ObjectIndexSpan] = []
    for index, entry_text in enumerate(entry_texts):
        if index > 0:
            text += ", "
        start = len(text)
        text += entry_text
        object_spans.append(
            ObjectIndexSpan(object_index=index, start=start, end=len(text))
        )

    close_start = len(text)
    text += _GLOBAL_CLOSE_TEXT
    return IndexedObjectList(
        text=text,
        entry_texts_by_index=entry_texts,
        object_spans=tuple(object_spans),
        global_close_start_span=StructuralSpan(start=close_start, end=close_start + 1),
        global_full_close_span=StructuralSpan(start=close_start, end=len(text)),
    )


def build_prefix_text(
    rendered: IndexedObjectList, selected_indices: Sequence[int]
) -> PrefixText:
    """Build an append-ready prefix from a chosen ordered subset of indices."""

    indices = _validate_indices(rendered.entry_texts_by_index, selected_indices)
    text = _OBJECTS_PREFIX
    object_spans: list[ObjectIndexSpan] = []
    for position, index in enumerate(indices):
        if position > 0:
            text += ", "
        start = len(text)
        text += rendered.entry_texts_by_index[index]
        object_spans.append(
            ObjectIndexSpan(object_index=index, start=start, end=len(text))
        )
    if object_spans and len(indices) < len(rendered.entry_texts_by_index):
        text += ", "
    return PrefixText(text=text, object_spans=tuple(object_spans))


def build_candidate_entry_text(
    rendered: IndexedObjectList,
    *,
    prefix_indices: Sequence[int],
    candidate_index: int,
) -> CandidateEntryText:
    """Build the candidate entry and final global close for a given prefix."""

    prefix_index_set = set(
        _validate_indices(rendered.entry_texts_by_index, prefix_indices)
    )
    if int(candidate_index) in prefix_index_set:
        raise ValueError(
            f"Candidate index {int(candidate_index)} is already present in prefix_indices"
        )
    candidate = int(candidate_index)
    if candidate < 0 or candidate >= len(rendered.entry_texts_by_index):
        raise IndexError(f"Object index out of range: {candidate}")

    entry_text = rendered.entry_texts_by_index[candidate]
    text = entry_text + _GLOBAL_CLOSE_TEXT
    entry_end = len(entry_text)
    return CandidateEntryText(
        text=text,
        candidate_span=ObjectIndexSpan(
            object_index=candidate,
            start=0,
            end=entry_end,
        ),
        global_close_start_span=StructuralSpan(start=entry_end, end=entry_end + 1),
        global_full_close_span=StructuralSpan(start=entry_end, end=len(text)),
    )


def build_global_close_text() -> GlobalCloseText:
    """Build the global close continuation without any object-entry close target."""

    return GlobalCloseText(
        text=_GLOBAL_CLOSE_TEXT,
        global_close_start_span=StructuralSpan(start=0, end=1),
        global_full_close_span=StructuralSpan(start=0, end=len(_GLOBAL_CLOSE_TEXT)),
    )


__all__ = [
    "CandidateEntryText",
    "GlobalCloseText",
    "IndexedObjectList",
    "ObjectIndexSpan",
    "PrefixText",
    "StructuralSpan",
    "build_candidate_entry_text",
    "build_global_close_text",
    "build_prefix_text",
    "render_indexed_object_list",
]
