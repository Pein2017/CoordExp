from __future__ import annotations

from typing import Literal, cast

from src.metrics.events import (
    MetricEvent,
    MetricScalar,
    MetricUnit,
    ratio_event,
    sum_event,
    weighted_mean_event,
)

CoordinateSlotName = Literal["x1", "y1", "x2", "y2"]
ParserMetricUnit = Literal["sample", "image"]

VALID_COORDINATE_SLOT_NAMES: tuple[CoordinateSlotName, ...] = (
    "x1",
    "y1",
    "x2",
    "y2",
)

DEFAULT_COORDINATE_SURFACE = "coord_token"
DEFAULT_GEOMETRY_TYPE = "bbox"
DEFAULT_METRIC_SURFACE = "training_logits"


def schema_token_accuracy_event(
    correct: MetricScalar,
    total: MetricScalar,
    *,
    top_k: int = 1,
    vocab_scope: str = "full_vocab",
    metric_surface: str = DEFAULT_METRIC_SURFACE,
    template_id: str | None = None,
    diagnostic_only: bool = False,
) -> MetricEvent:
    """Token-denominated schema/control accuracy."""

    return _token_accuracy_event(
        "schema",
        correct,
        total,
        top_k=top_k,
        vocab_scope=vocab_scope,
        token_role="schema",
        metric_surface=metric_surface,
        template_id=template_id,
        diagnostic_only=diagnostic_only,
    )


def schema_token_cross_entropy_event(
    value: MetricScalar,
    token_count: MetricScalar,
    *,
    vocab_scope: str = "full_vocab",
    metric_surface: str = DEFAULT_METRIC_SURFACE,
    template_id: str | None = None,
    diagnostic_only: bool = False,
) -> MetricEvent:
    """Token-denominated schema/control cross entropy."""

    return _token_cross_entropy_event(
        "schema",
        value,
        token_count,
        vocab_scope=vocab_scope,
        token_role="schema",
        metric_surface=metric_surface,
        template_id=template_id,
        diagnostic_only=diagnostic_only,
    )


def bbox_start_accuracy_event(
    correct: MetricScalar,
    total: MetricScalar,
    *,
    metric_surface: str = DEFAULT_METRIC_SURFACE,
    template_id: str | None = None,
    diagnostic_only: bool = False,
) -> MetricEvent:
    """Span-denominated bbox-start marker accuracy."""

    return ratio_event(
        "detection_sequence/schema/bbox_start_acc/span",
        correct,
        total,
        unit="span",
        semantic_role="schema",
        token_role="bbox_start",
        geometry_type=DEFAULT_GEOMETRY_TYPE,
        metric_surface=metric_surface,
        template_id=template_id,
        diagnostic_only=diagnostic_only,
    )


def terminal_accuracy_event(
    correct: MetricScalar,
    total: MetricScalar,
    *,
    metric_surface: str = DEFAULT_METRIC_SURFACE,
    template_id: str | None = None,
    diagnostic_only: bool = False,
) -> MetricEvent:
    """Span-denominated terminal-control accuracy."""

    return ratio_event(
        "detection_sequence/schema/terminal_acc/span",
        correct,
        total,
        unit="span",
        semantic_role="schema",
        token_role="terminal",
        metric_surface=metric_surface,
        template_id=template_id,
        diagnostic_only=diagnostic_only,
    )


def description_token_accuracy_event(
    correct: MetricScalar,
    total: MetricScalar,
    *,
    top_k: int = 1,
    vocab_scope: str = "full_vocab",
    metric_surface: str = DEFAULT_METRIC_SURFACE,
    template_id: str | None = None,
    diagnostic_only: bool = False,
) -> MetricEvent:
    """Token-denominated description accuracy."""

    return _token_accuracy_event(
        "description",
        correct,
        total,
        top_k=top_k,
        vocab_scope=vocab_scope,
        token_role="description",
        metric_surface=metric_surface,
        template_id=template_id,
        diagnostic_only=diagnostic_only,
    )


def description_token_cross_entropy_event(
    value: MetricScalar,
    token_count: MetricScalar,
    *,
    vocab_scope: str = "full_vocab",
    metric_surface: str = DEFAULT_METRIC_SURFACE,
    template_id: str | None = None,
    diagnostic_only: bool = False,
) -> MetricEvent:
    """Token-denominated description cross entropy."""

    return _token_cross_entropy_event(
        "description",
        value,
        token_count,
        vocab_scope=vocab_scope,
        token_role="description",
        metric_surface=metric_surface,
        template_id=template_id,
        diagnostic_only=diagnostic_only,
    )


def description_span_exact_match_event(
    correct: MetricScalar,
    total: MetricScalar,
    *,
    object_scope: str = "object_entry",
    metric_surface: str = DEFAULT_METRIC_SURFACE,
    template_id: str | None = None,
    diagnostic_only: bool = False,
) -> MetricEvent:
    """Span-denominated exact match for rendered description text."""

    return ratio_event(
        f"detection_sequence/description/exact_span_match/{object_scope}",
        correct,
        total,
        unit="span",
        semantic_role="description",
        token_role="description",
        object_scope=object_scope,
        metric_surface=metric_surface,
        template_id=template_id,
        diagnostic_only=diagnostic_only,
    )


def coordinate_token_accuracy_event(
    correct: MetricScalar,
    total: MetricScalar,
    *,
    top_k: int = 1,
    vocab_scope: str = "full_vocab",
    coordinate_surface: str = DEFAULT_COORDINATE_SURFACE,
    geometry_type: str = DEFAULT_GEOMETRY_TYPE,
    metric_surface: str = DEFAULT_METRIC_SURFACE,
    template_id: str | None = None,
    diagnostic_only: bool = False,
) -> MetricEvent:
    """Token-denominated coordinate accuracy.

    This helper intentionally stays token-denominated; per-slot metrics should use
    ``coordinate_slot_accuracy_event`` so their denominator is the slot count.
    """

    return _token_accuracy_event(
        "coordinate",
        correct,
        total,
        top_k=top_k,
        vocab_scope=vocab_scope,
        token_role="coord",
        coordinate_surface=coordinate_surface,
        geometry_type=geometry_type,
        metric_surface=metric_surface,
        template_id=template_id,
        diagnostic_only=diagnostic_only,
    )


def coordinate_token_cross_entropy_event(
    value: MetricScalar,
    token_count: MetricScalar,
    *,
    vocab_scope: str = "full_vocab",
    coordinate_surface: str = DEFAULT_COORDINATE_SURFACE,
    geometry_type: str = DEFAULT_GEOMETRY_TYPE,
    metric_surface: str = DEFAULT_METRIC_SURFACE,
    template_id: str | None = None,
    diagnostic_only: bool = False,
) -> MetricEvent:
    """Token-denominated coordinate cross entropy."""

    return _token_cross_entropy_event(
        "coordinate",
        value,
        token_count,
        vocab_scope=vocab_scope,
        token_role="coord",
        coordinate_surface=coordinate_surface,
        geometry_type=geometry_type,
        metric_surface=metric_surface,
        template_id=template_id,
        diagnostic_only=diagnostic_only,
    )


def coordinate_slot_accuracy_event(
    slot_name: str,
    correct: MetricScalar,
    total: MetricScalar,
    *,
    coordinate_surface: str = DEFAULT_COORDINATE_SURFACE,
    geometry_type: str = DEFAULT_GEOMETRY_TYPE,
    metric_surface: str = DEFAULT_METRIC_SURFACE,
    template_id: str | None = None,
    diagnostic_only: bool = False,
) -> MetricEvent:
    """Slot-denominated coordinate accuracy for one of x1/y1/x2/y2."""

    valid_slot_name = validate_coordinate_slot_name(slot_name)
    return ratio_event(
        (
            "detection_sequence/coordinate/slot_acc/"
            f"{geometry_type}/{coordinate_surface}/{valid_slot_name}"
        ),
        correct,
        total,
        unit="slot",
        semantic_role="bbox_coord",
        token_role="coord",
        coordinate_surface=coordinate_surface,
        geometry_type=geometry_type,
        slot_name=valid_slot_name,
        metric_surface=metric_surface,
        template_id=template_id,
        diagnostic_only=diagnostic_only,
    )


def object_entry_exact_match_event(
    correct: MetricScalar,
    total: MetricScalar,
    *,
    object_scope: str = "object_entry",
    metric_surface: str = DEFAULT_METRIC_SURFACE,
    template_id: str | None = None,
    diagnostic_only: bool = False,
) -> MetricEvent:
    """Object-denominated exact sequence match for rendered object entries."""

    return ratio_event(
        f"detection_sequence/object_entry/exact_sequence_match/{object_scope}",
        correct,
        total,
        unit="object",
        semantic_role="object_entry",
        object_scope=object_scope,
        metric_surface=metric_surface,
        template_id=template_id,
        diagnostic_only=diagnostic_only,
    )


def trie_decision_accuracy_event(
    correct: MetricScalar,
    total: MetricScalar,
    *,
    object_scope: str = "object_entry",
    metric_surface: str = DEFAULT_METRIC_SURFACE,
    template_id: str | None = None,
    diagnostic_only: bool = False,
) -> MetricEvent:
    """Span-denominated accuracy for trie-eligible object-entry decisions."""

    return ratio_event(
        f"detection_sequence/object_entry/trie_decision_acc/{object_scope}",
        correct,
        total,
        unit="span",
        semantic_role="object_entry",
        token_role="trie_decision",
        object_scope=object_scope,
        metric_surface=metric_surface,
        template_id=template_id,
        diagnostic_only=diagnostic_only,
    )


def desc_box_binding_success_event(
    correct: MetricScalar,
    total: MetricScalar,
    *,
    object_scope: str = "object_entry",
    metric_surface: str = DEFAULT_METRIC_SURFACE,
    template_id: str | None = None,
    diagnostic_only: bool = False,
) -> MetricEvent:
    """Object-denominated success for description-plus-box binding."""

    return ratio_event(
        f"detection_sequence/object_entry/desc_box_binding_success/{object_scope}",
        correct,
        total,
        unit="object",
        semantic_role="object_entry",
        token_role="description_bbox_binding",
        object_scope=object_scope,
        metric_surface=metric_surface,
        template_id=template_id,
        diagnostic_only=diagnostic_only,
    )


def parser_success_event(
    success_count: MetricScalar,
    generated_count: MetricScalar,
    *,
    parser_mode: str,
    metric_surface: str,
    unit: ParserMetricUnit = "sample",
    diagnostic_only: bool = False,
    template_id: str | None = None,
) -> MetricEvent:
    """Parser-surface success rate.

    The denominator is generated samples by default; callers may opt into image
    denominators only when the parser surface is explicitly image-denominated.
    """

    diagnostic_scope = "diagnostic" if diagnostic_only else "strict"
    return ratio_event(
        (
            "detection_sequence/parser/success/"
            f"{unit}/{parser_mode}/{metric_surface}/{diagnostic_scope}"
        ),
        success_count,
        generated_count,
        unit=unit,
        semantic_role="parser",
        template_id=template_id,
        parser_mode=parser_mode,
        metric_surface=metric_surface,
        diagnostic_only=diagnostic_only,
    )


def diagnostic_salvage_rate_event(
    salvaged_count: MetricScalar,
    generated_count: MetricScalar,
    *,
    parser_mode: str = "diagnostic_salvage",
    metric_surface: str = "diagnostic_only",
    unit: ParserMetricUnit = "sample",
    template_id: str | None = None,
) -> MetricEvent:
    """Diagnostic-only parser salvage rate."""

    return parser_success_event(
        salvaged_count,
        generated_count,
        parser_mode=parser_mode,
        metric_surface=metric_surface,
        unit=unit,
        diagnostic_only=True,
        template_id=template_id,
    )


def strict_metric_bearing_object_count_event(
    object_count: MetricScalar,
    *,
    parser_mode: str,
    metric_surface: str,
    unit: MetricUnit = "object",
    template_id: str | None = None,
    diagnostic_only: bool = False,
) -> MetricEvent:
    """Strict metric-bearing object count on a parser/eval surface."""

    diagnostic_scope = "diagnostic" if diagnostic_only else "strict"
    return sum_event(
        (
            "detection_sequence/parser/metric_bearing_object_count/"
            f"{parser_mode}/{metric_surface}/{diagnostic_scope}"
        ),
        object_count,
        unit=unit,
        semantic_role="parser",
        object_scope="metric_bearing",
        template_id=template_id,
        parser_mode=parser_mode,
        metric_surface=metric_surface,
        diagnostic_only=diagnostic_only,
    )


def validate_coordinate_slot_name(slot_name: str) -> CoordinateSlotName:
    if slot_name not in VALID_COORDINATE_SLOT_NAMES:
        valid = ", ".join(VALID_COORDINATE_SLOT_NAMES)
        raise ValueError(
            f"Invalid coordinate slot name {slot_name!r}; expected one of: {valid}"
        )
    return cast(CoordinateSlotName, slot_name)


def _token_accuracy_event(
    semantic_role: str,
    correct: MetricScalar,
    total: MetricScalar,
    *,
    top_k: int,
    vocab_scope: str,
    token_role: str,
    metric_surface: str,
    coordinate_surface: str | None = None,
    geometry_type: str | None = None,
    template_id: str | None = None,
    diagnostic_only: bool = False,
) -> MetricEvent:
    top_k = _validate_top_k(top_k)
    return ratio_event(
        (
            "detection_sequence/"
            f"{semantic_role}/token_acc/{vocab_scope}/top{top_k}"
        ),
        correct,
        total,
        unit="token",
        semantic_role=semantic_role,
        token_role=token_role,
        vocab_scope=vocab_scope,
        coordinate_surface=coordinate_surface,
        geometry_type=geometry_type,
        metric_surface=metric_surface,
        template_id=template_id,
        diagnostic_only=diagnostic_only,
    )


def _token_cross_entropy_event(
    semantic_role: str,
    value: MetricScalar,
    token_count: MetricScalar,
    *,
    vocab_scope: str,
    token_role: str,
    metric_surface: str,
    coordinate_surface: str | None = None,
    geometry_type: str | None = None,
    template_id: str | None = None,
    diagnostic_only: bool = False,
) -> MetricEvent:
    return weighted_mean_event(
        f"detection_sequence/{semantic_role}/token_ce/{vocab_scope}",
        value,
        token_count,
        unit="token",
        semantic_role=semantic_role,
        token_role=token_role,
        vocab_scope=vocab_scope,
        coordinate_surface=coordinate_surface,
        geometry_type=geometry_type,
        metric_surface=metric_surface,
        template_id=template_id,
        diagnostic_only=diagnostic_only,
    )


def _validate_top_k(top_k: int) -> int:
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k!r}")
    return top_k


__all__ = [
    "CoordinateSlotName",
    "DEFAULT_COORDINATE_SURFACE",
    "DEFAULT_GEOMETRY_TYPE",
    "DEFAULT_METRIC_SURFACE",
    "ParserMetricUnit",
    "VALID_COORDINATE_SLOT_NAMES",
    "bbox_start_accuracy_event",
    "coordinate_slot_accuracy_event",
    "coordinate_token_accuracy_event",
    "coordinate_token_cross_entropy_event",
    "desc_box_binding_success_event",
    "description_span_exact_match_event",
    "description_token_accuracy_event",
    "description_token_cross_entropy_event",
    "diagnostic_salvage_rate_event",
    "object_entry_exact_match_event",
    "parser_success_event",
    "schema_token_accuracy_event",
    "schema_token_cross_entropy_event",
    "strict_metric_bearing_object_count_event",
    "terminal_accuracy_event",
    "trie_decision_accuracy_event",
    "validate_coordinate_slot_name",
]
