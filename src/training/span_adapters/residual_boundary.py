"""Residual-set boundary slicing over compact detection span owners."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from src.detection.data import (
    CoordinateTokenBox,
    DetectionMetadata,
    NormalizedDetectionObject,
    NormalizedDetectionSample,
    ObjectOrderingPlan,
)
from src.detection.template import RenderedAssistantSequence, get_detection_template
from src.detection.tokenization import TokenizedDetectionExample
from src.detection.tokenization import tokenize_rendered_detection_conversation
from src.training.encoding.view import EncodedDetectionView


@dataclass(frozen=True, slots=True)
class ResidualBoundaryObjectSpan:
    """Assistant-local compact-full object span owned by detection tokenization."""

    object_index: int
    object_start: int
    desc_start: int
    desc_end: int
    box_start: int
    coord_positions: tuple[int, int, int, int]
    separator_span: tuple[int, int] | None = None


@dataclass(frozen=True, slots=True)
class ResidualBoundarySlice:
    """Assistant-local residual suffix split projected from existing spans."""

    tokenized: TokenizedDetectionExample
    encoded_view: EncodedDetectionView
    suffix_start: int
    suffix_input_ids: tuple[int, ...]
    retained_prefix_input_ids: tuple[int, ...]
    object_spans: tuple[ResidualBoundaryObjectSpan, ...]
    boundary_object_span: ResidualBoundaryObjectSpan | None = None


class ResidualBoundaryAdapter:
    """Render, tokenize, and slice compact-full residual suffix boundaries."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        template_mode: str = "compact_full",
        system_prompt: str | None = None,
        user_content: str = "<image>",
        messages: Sequence[Mapping[str, Any]] | None = None,
        assistant_stop_markers: Sequence[str] | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.template_mode = str(template_mode)
        self.template = get_detection_template(self.template_mode)
        self.system_prompt = system_prompt
        self.user_content = str(user_content)
        self.messages = messages
        self.assistant_stop_markers = assistant_stop_markers

    def render_objects(
        self,
        objects: Sequence[Mapping[str, Any] | Any],
    ) -> RenderedAssistantSequence:
        """Render objects through the existing detection template."""

        normalized_objects = tuple(
            _normalized_object_from_mapping_or_attr(obj, normalized_index=index)
            for index, obj in enumerate(objects)
        )
        sample = NormalizedDetectionSample(
            images=("residual-boundary.jpg",),
            objects=normalized_objects,
            width=1000,
            height=1000,
            image_id=0,
            file_name="residual-boundary.jpg",
            metadata=DetectionMetadata(source="residual_boundary", split="train"),
            object_ordering=ObjectOrderingPlan.sorted().with_realized(
                tuple(obj.source_object_index for obj in normalized_objects)
            ),
        )
        return self.template.render_assistant(sample)

    def tokenize_rendered(
        self,
        rendered: RenderedAssistantSequence,
    ) -> TokenizedDetectionExample:
        """Tokenize rendered compact-full assistant text through the owner."""

        return tokenize_rendered_detection_conversation(
            rendered,
            tokenizer=self.tokenizer,
            system_prompt=self.system_prompt,
            user_content=self.user_content,
            messages=self.messages,
            assistant_stop_markers=self.assistant_stop_markers,
        )

    def object_spans(
        self,
        tokenized: TokenizedDetectionExample,
    ) -> tuple[ResidualBoundaryObjectSpan, ...]:
        """Return assistant-local object spans derived from tokenized entries."""

        assistant_start = int(tokenized.assistant_token_span.start)
        spans: list[ResidualBoundaryObjectSpan] = []
        for entry in tokenized.object_entries:
            if entry.object_ref_start_span is None:
                raise ValueError(
                    "residual boundary adapter requires object_ref_start spans"
                )
            if len(entry.coord_spans) != 4:
                raise ValueError(
                    "residual boundary adapter requires four coordinate spans"
                )
            separator_span = (
                (
                    int(entry.separator_span.start) - assistant_start,
                    int(entry.separator_span.end) - assistant_start,
                )
                if entry.separator_span is not None
                else None
            )
            spans.append(
                ResidualBoundaryObjectSpan(
                    object_index=int(entry.object_index),
                    object_start=int(entry.object_ref_start_span.start)
                    - assistant_start,
                    desc_start=int(entry.desc_span.start) - assistant_start,
                    desc_end=int(entry.desc_span.end) - assistant_start,
                    box_start=int(entry.bbox_start_span.start) - assistant_start,
                    coord_positions=tuple(
                        int(span.start) - assistant_start
                        for span in entry.coord_spans
                    ),  # type: ignore[arg-type]
                    separator_span=separator_span,
                )
            )
        return tuple(spans)

    def assistant_input_ids(
        self,
        tokenized: TokenizedDetectionExample,
    ) -> tuple[int, ...]:
        """Return assistant payload token ids without chat stop markers."""

        return tuple(
            int(token_id)
            for token_id in tokenized.input_ids[
                tokenized.assistant_token_span.start : tokenized.assistant_token_span.end
            ]
        )

    def slice_from_boundary(
        self,
        rendered: RenderedAssistantSequence,
        *,
        boundary: str,
        object_index: int | None = None,
    ) -> ResidualBoundarySlice:
        """Slice the assistant payload from a residual boundary."""

        tokenized = self.tokenize_rendered(rendered)
        encoded_view = EncodedDetectionView.from_tokenized(tokenized)
        assistant_ids = self.assistant_input_ids(tokenized)
        object_spans = self.object_spans(tokenized)
        boundary_object_span: ResidualBoundaryObjectSpan | None = None

        if boundary in {"assistant", "assistant_start", "start"}:
            suffix_start = 0
        elif boundary in {"assistant_end", "end"}:
            suffix_start = len(assistant_ids)
        elif boundary == "object":
            if object_index is None:
                raise ValueError("object boundary requires object_index")
            index = int(object_index)
            if index < 0 or index > len(object_spans):
                raise IndexError("object_index is outside rendered objects")
            if index == len(object_spans):
                suffix_start = len(assistant_ids)
            else:
                boundary_object_span = object_spans[index]
                suffix_start = int(boundary_object_span.object_start)
        else:
            raise ValueError(f"unsupported residual boundary: {boundary!r}")

        retained_prefix = tuple(assistant_ids[:suffix_start])
        suffix = tuple(assistant_ids[suffix_start:])
        self.validate_no_adjacent_duplicate_boundary_token(
            retained_prefix_input_ids=retained_prefix,
            suffix_input_ids=suffix,
        )
        return ResidualBoundarySlice(
            tokenized=tokenized,
            encoded_view=encoded_view,
            suffix_start=int(suffix_start),
            suffix_input_ids=suffix,
            retained_prefix_input_ids=retained_prefix,
            object_spans=object_spans,
            boundary_object_span=boundary_object_span,
        )

    def validate_no_adjacent_duplicate_boundary_token(
        self,
        *,
        retained_prefix_input_ids: Sequence[int],
        suffix_input_ids: Sequence[int],
    ) -> None:
        """Reject a boundary join that repeats the first suffix token."""

        if not retained_prefix_input_ids or not suffix_input_ids:
            return
        if int(retained_prefix_input_ids[-1]) == int(suffix_input_ids[0]):
            raise ValueError("token duplicated at residual boundary")


def _normalized_object_from_mapping_or_attr(
    obj: Mapping[str, Any] | Any,
    *,
    normalized_index: int,
) -> NormalizedDetectionObject:
    desc = str(_field(obj, "desc"))
    source_object_index = int(_field(obj, "source_object_index", normalized_index))
    object_id_raw = _field(obj, "object_id", None)
    object_id = None if object_id_raw is None else str(object_id_raw)
    instance_id = str(
        _field(
            obj,
            "object_instance_id",
            object_id or f"residual-boundary:{source_object_index}",
        )
    )
    return NormalizedDetectionObject(
        normalized_object_index=int(normalized_index),
        source_object_index=source_object_index,
        object_instance_id=instance_id,
        desc=desc,
        bbox_2d=_coordinate_box_from_object(obj),
        category_id=int(_field(obj, "category_id", 0)),
        category_name=str(_field(obj, "category_name", "")),
        coco_ann_id=int(_field(obj, "coco_ann_id", source_object_index)),
        object_id=object_id,
        source_role=_optional_str(_field(obj, "source_role", None)),
        relation_snapshot=_optional_mapping(_field(obj, "relation_snapshot", None)),
    )


def _coordinate_box_from_object(obj: Mapping[str, Any] | Any) -> CoordinateTokenBox:
    bbox = _field(obj, "bbox_2d", None)
    if type(bbox) is CoordinateTokenBox:
        return bbox
    if isinstance(bbox, Mapping):
        return CoordinateTokenBox(
            bbox["x1"],
            bbox["y1"],
            bbox["x2"],
            bbox["y2"],
        )
    if isinstance(bbox, Sequence) and not isinstance(bbox, (str, bytes)):
        if len(bbox) != 4:
            raise ValueError("bbox_2d must contain exactly four values")
        return CoordinateTokenBox(bbox[0], bbox[1], bbox[2], bbox[3])

    points = _field(obj, "points_norm1000", None)
    if isinstance(points, Sequence) and not isinstance(points, (str, bytes)):
        if len(points) != 4:
            raise ValueError("points_norm1000 must contain exactly four values")
        return CoordinateTokenBox(points[0], points[1], points[2], points[3])
    raise ValueError("residual object requires bbox_2d or points_norm1000")


def _field(obj: Mapping[str, Any] | Any, name: str, default: Any = ...) -> Any:
    if isinstance(obj, Mapping) and name in obj:
        return obj[name]
    if hasattr(obj, name):
        return getattr(obj, name)
    if default is not ...:
        return default
    raise ValueError(f"residual object missing required field: {name}")


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _optional_mapping(value: Any) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("relation_snapshot must be a mapping when provided")
    return value


__all__ = [
    "ResidualBoundaryAdapter",
    "ResidualBoundaryObjectSpan",
    "ResidualBoundarySlice",
]
