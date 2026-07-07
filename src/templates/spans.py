"""Rendered character-span records for template output."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.common.errors import TemplateContractError
from src.coordinate_targets import CoordinateLossTarget


RenderedSpanKind = Literal[
    "assistant_content",
    "object",
    "description",
    "schema_token",
    "coordinate_token",
    "eos_transition",
    "ignored_text",
]


LEAF_SPAN_KINDS = frozenset(
    {
        "description",
        "schema_token",
        "coordinate_token",
        "eos_transition",
        "ignored_text",
    }
)
COORD_TOKEN_RE = r"^<\|coord_(0|[1-9][0-9]{0,2})\|>$"
APPROVED_SCHEMA_TOKENS = frozenset(
    {
        "<|object_ref_start|>",
        "<|object_ref_end|>",
        "<|box_start|>",
        "<|box_end|>",
    }
)
APPROVED_EOS_TOKEN = "<|im_end|>"


@dataclass(frozen=True)
class RenderedSpan:
    kind: RenderedSpanKind
    char_start: int
    char_end: int
    text: str
    object_id: str | None = None
    field: str | None = None
    source: str | None = None
    coordinate_target: CoordinateLossTarget | None = None

    def __post_init__(self) -> None:
        if self.char_start < 0 or self.char_end <= self.char_start:
            raise TemplateContractError(
                "rendered span must use a non-empty half-open range",
                code="template.span_range",
                context={
                    "kind": self.kind,
                    "char_start": self.char_start,
                    "char_end": self.char_end,
                },
            )
        if not self.text:
            raise TemplateContractError(
                "rendered span text must not be empty",
                code="template.span_text_empty",
                context={"kind": self.kind},
            )

    def text_from(self, rendered_text: str) -> str:
        return rendered_text[self.char_start : self.char_end]


def assert_span_text(rendered_text: str, span: RenderedSpan) -> None:
    actual = span.text_from(rendered_text)
    if actual != span.text:
        raise TemplateContractError(
            "rendered span text does not match supervised response slice",
            code="template.span_text_mismatch",
            context={
                "kind": span.kind,
                "char_start": span.char_start,
                "char_end": span.char_end,
                "span_text": span.text,
                "actual_text": actual,
            },
        )


def validate_rendered_spans(rendered_text: str, spans: tuple[RenderedSpan, ...]) -> None:
    if not rendered_text:
        raise TemplateContractError(
            "supervised response text must not be empty",
            code="template.rendered_text_empty",
        )
    if not spans:
        raise TemplateContractError(
            "rendered spans must not be empty",
            code="template.spans_empty",
        )

    for span in spans:
        if span.char_end > len(rendered_text):
            raise TemplateContractError(
                "rendered span extends past supervised response text",
                code="template.span_out_of_bounds",
                context={
                    "kind": span.kind,
                    "char_start": span.char_start,
                    "char_end": span.char_end,
                    "text_length": len(rendered_text),
                },
            )
        assert_span_text(rendered_text, span)
        _validate_leaf_literal(span)
        _validate_coordinate_target_polarity(span)
    _validate_no_crossing_spans(spans)
    _validate_exact_leaf_coverage(rendered_text, spans)


def _validate_no_crossing_spans(spans: tuple[RenderedSpan, ...]) -> None:
    for left_index, left in enumerate(spans):
        for right in spans[left_index + 1 :]:
            if not _overlaps(left, right):
                continue
            if _contains(left, right) or _contains(right, left):
                continue
            raise TemplateContractError(
                "rendered spans must be nested or disjoint, not crossing",
                code="template.span_crossing",
                context={
                    "left_kind": left.kind,
                    "left_range": [left.char_start, left.char_end],
                    "right_kind": right.kind,
                    "right_range": [right.char_start, right.char_end],
                },
            )


def _validate_exact_leaf_coverage(
    rendered_text: str,
    spans: tuple[RenderedSpan, ...],
) -> None:
    coverage = [0] * len(rendered_text)
    for span in spans:
        if span.kind not in LEAF_SPAN_KINDS:
            continue
        for position in range(span.char_start, span.char_end):
            coverage[position] += 1

    bad_positions = [index for index, count in enumerate(coverage) if count != 1]
    if bad_positions:
        first = bad_positions[0]
        raise TemplateContractError(
            "each supervised-response character must have exactly one leaf span",
            code="template.leaf_coverage",
            context={
                "position": first,
                "coverage": coverage[first],
                "character": rendered_text[first],
            },
        )


def _validate_leaf_literal(span: RenderedSpan) -> None:
    import re

    if span.kind == "coordinate_token" and re.fullmatch(COORD_TOKEN_RE, span.text) is None:
        raise TemplateContractError(
            "coordinate leaf span must cover one canonical coordinate token",
            code="template.coordinate_span_literal",
            context={"text": span.text},
        )
    if span.kind == "schema_token" and span.text not in APPROVED_SCHEMA_TOKENS:
        raise TemplateContractError(
            "schema leaf span must cover one approved V1 wrapper token",
            code="template.special_span_literal",
            context={"kind": span.kind, "text": span.text},
        )
    if span.kind == "eos_transition" and span.text != APPROVED_EOS_TOKEN:
        raise TemplateContractError(
            "eos leaf span must cover the approved assistant transition token",
            code="template.special_span_literal",
            context={"kind": span.kind, "text": span.text},
        )


def _validate_coordinate_target_polarity(span: RenderedSpan) -> None:
    if span.kind == "coordinate_token":
        if span.coordinate_target is None:
            raise TemplateContractError(
                "coordinate leaf span must carry coordinate target metadata",
                code="template.coordinate_target_missing",
                context={
                    "kind": span.kind,
                    "text": span.text,
                    "object_id": span.object_id,
                    "field": span.field,
                },
            )
        return
    if span.coordinate_target is not None:
        raise TemplateContractError(
            "only coordinate leaf spans may carry coordinate target metadata",
            code="template.coordinate_target_unexpected",
            context={
                "kind": span.kind,
                "text": span.text,
                "object_id": span.object_id,
                "field": span.field,
            },
        )


def _overlaps(left: RenderedSpan, right: RenderedSpan) -> bool:
    return max(left.char_start, right.char_start) < min(left.char_end, right.char_end)


def _contains(left: RenderedSpan, right: RenderedSpan) -> bool:
    return left.char_start <= right.char_start and left.char_end >= right.char_end


__all__ = [
    "APPROVED_EOS_TOKEN",
    "APPROVED_SCHEMA_TOKENS",
    "LEAF_SPAN_KINDS",
    "RenderedSpan",
    "RenderedSpanKind",
    "assert_span_text",
    "validate_rendered_spans",
]
