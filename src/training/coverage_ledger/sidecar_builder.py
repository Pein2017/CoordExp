"""Build coverage-ledger sidecars from structured detection span metadata."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from src.detection.tokenization import (
    DetectionSupervisionView,
    TokenSpan,
    TokenizedObjectEntry,
)
from src.training.coverage_ledger.sidecars import (
    CoverageLedgerObjectEntry,
    CoverageLedgerSidecar,
)

_COORD_TOKEN_RE = re.compile(r"<\|coord_(\d{1,3})\|>")


def build_coverage_ledger_sidecar(
    tokenized: DetectionSupervisionView,
    *,
    sample_id: str,
    image_grid_thw: Sequence[Any],
    processed_width: int,
    processed_height: int,
    image_identity: str,
) -> CoverageLedgerSidecar:
    """Extract coverage-ledger supervision from tokenized detection metadata."""

    if type(tokenized) is not DetectionSupervisionView:
        raise TypeError("tokenized must be a DetectionSupervisionView")
    if not tokenized.object_entries:
        raise ValueError("coverage ledger requires at least one tokenized object entry")

    object_entries: list[CoverageLedgerObjectEntry] = []
    for emitted_order_index, tokenized_entry in enumerate(tokenized.object_entries):
        _require_one_token_span(
            tokenized_entry.bbox_start_span,
            label="bbox_start",
        )
        control_positions = _required_control_positions(
            tokenized_entry,
            labels=("object_ref_end", "box_end"),
        )
        coord_label_positions = tuple(span.start for span in tokenized_entry.coord_spans)
        if len(coord_label_positions) != 4:
            raise ValueError("coverage ledger requires exactly four coord_spans per object")
        bbox_norm1000_xyxy = tuple(
            _coord_value_from_span(tokenized, span)
            for span in tokenized_entry.coord_spans
        )

        object_entries.append(
            CoverageLedgerObjectEntry(
                object_instance_id=tokenized_entry.object_instance_id,
                source_object_index=tokenized_entry.source_object_index,
                emitted_order_index=emitted_order_index,
                image_index=0,
                bbox_norm1000_xyxy=bbox_norm1000_xyxy,
                box_start_position=tokenized_entry.bbox_start_span.start,
                coord_label_positions=coord_label_positions,
                object_ref_end_position=control_positions["object_ref_end"],
                box_end_position=control_positions["box_end"],
            )
        )

    return CoverageLedgerSidecar(
        sample_id=sample_id,
        prompt_end_position=tokenized.object_entries[0].entry_span.start - 1,
        object_entries=tuple(object_entries),
        image_grid_thw=_normalize_image_grid_thw(image_grid_thw),
        processed_width=processed_width,
        processed_height=processed_height,
        image_identity=image_identity,
    )


def _coord_value_from_span(tokenized: DetectionSupervisionView, span: TokenSpan) -> int:
    _require_one_token_span(span, label=span.label)
    try:
        char_start, char_end = tokenized.offset_mapping[span.start]
    except IndexError as exc:
        raise ValueError(
            f"coord span {span.label!r} starts outside offset_mapping"
        ) from exc
    token_text = tokenized.chat_text[char_start:char_end]
    match = _COORD_TOKEN_RE.fullmatch(token_text)
    if match is None:
        raise ValueError(
            f"coord span {span.label!r} must align to a <|coord_N|> token"
        )
    return int(match.group(1))


def _required_control_positions(
    entry: TokenizedObjectEntry,
    *,
    labels: tuple[str, ...],
) -> dict[str, int]:
    matches_by_label = {
        label: tuple(span for span in entry.control_spans if span.label == label)
        for label in labels
    }
    invalid = {
        label: len(matches)
        for label, matches in matches_by_label.items()
        if len(matches) != 1
    }
    if invalid:
        details = ", ".join(
            f"{label}={count}" for label, count in sorted(invalid.items())
        )
        raise ValueError(
            "coverage ledger requires exactly one control span per object for "
            f"{', '.join(labels)}; got {details} for "
            f"object_instance_id={entry.object_instance_id!r}"
        )
    positions: dict[str, int] = {}
    for label, matches in matches_by_label.items():
        _require_one_token_span(matches[0], label=label)
        positions[label] = matches[0].start
    return positions


def _require_one_token_span(span: TokenSpan, *, label: str) -> None:
    if span.end - span.start != 1:
        raise ValueError(
            f"coverage ledger requires {label!r} to cover exactly one token; "
            f"got token span [{span.start}, {span.end})"
        )


def _normalize_image_grid_thw(value: Sequence[Any]) -> tuple[int, int, int]:
    raw = value
    tolist = getattr(raw, "tolist", None)
    if callable(tolist):
        raw = tolist()
    if (
        isinstance(raw, Sequence)
        and not isinstance(raw, (str, bytes))
        and len(raw) == 1
        and isinstance(raw[0], Sequence)
        and not isinstance(raw[0], (str, bytes))
    ):
        raw = raw[0]
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise TypeError("image_grid_thw must be a sequence")
    return tuple(raw)  # type: ignore[return-value]


__all__ = ["build_coverage_ledger_sidecar"]
