"""Compact object-box-closed parser for V1 inference."""

from __future__ import annotations

import json
import re
import hashlib
from dataclasses import dataclass
from typing import Any

from src.common.errors import DataContractError
from src.data.geometry import coord_bins_to_pixel_xyxy, parse_coord_token
from src.templates.renderer import (
    BOX_END_TOKEN,
    BOX_START_TOKEN,
    IM_END_TOKEN,
    OBJECT_REF_END_TOKEN,
    OBJECT_REF_START_TOKEN,
)


PARSER_ID = "compact-object-box-closed-v1"
PARSER_POLICY = "compact_object_box_closed_only"
_OBJECT_RE = re.compile(
    r"^" + re.escape(OBJECT_REF_START_TOKEN)
    + r"(?P<description>.*?)"
    + re.escape(OBJECT_REF_END_TOKEN)
    + re.escape(BOX_START_TOKEN)
    + r"(?P<coords>(?:<\|coord_[^>]+\|>){4})"
    + re.escape(BOX_END_TOKEN),
    re.DOTALL,
)
_COORD_RE = re.compile(r"<\|coord_[^>]+\|>")


@dataclass(frozen=True)
class ParseRow:
    row_id: str
    row_index: int
    parser_id: str
    parser_policy: str
    metric_bearing: bool
    parse_status: str
    predictions: list[dict[str, Any]]
    dropped_predictions: list[dict[str, Any]]
    diagnostics: list[dict[str, Any]]

    @property
    def valid_prediction_count(self) -> int:
        return len(self.predictions)

    @property
    def dropped_prediction_count(self) -> int:
        return len(self.dropped_predictions)

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "row_index": self.row_index,
            "parser_id": self.parser_id,
            "parser_policy": self.parser_policy,
            "metric_bearing": self.metric_bearing,
            "parse_status": self.parse_status,
            "valid_prediction_count": self.valid_prediction_count,
            "dropped_prediction_count": self.dropped_prediction_count,
            "predictions": self.predictions,
            "dropped_predictions": self.dropped_predictions,
        }


def parse_compact_object_box_closed(
    text: str,
    *,
    row_id: str,
    row_index: int,
    image_width: int,
    image_height: int,
) -> ParseRow:
    stripped = text.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        return _row(
            row_id=row_id,
            row_index=row_index,
            status="unsupported_format",
            predictions=[],
            dropped=[
                _drop(
                    row_id=row_id,
                    row_index=row_index,
                    object_span_id=f"{row_id}:unsupported-json",
                    generated_order=None,
                    reason="unsupported_json_response",
                    raw_text=stripped,
                    char_start=0,
                    char_end=len(text),
                )
            ],
        )

    predictions: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for candidate in _object_candidates(text):
        if candidate["kind"] == "unmatched":
            if not _is_ignored_unmatched_text(candidate["text"]):
                dropped.append(
                    _drop(
                        row_id=row_id,
                        row_index=row_index,
                        object_span_id=f"{row_id}:unmatched-{len(dropped)}",
                        generated_order=None,
                        reason="unmatched_text",
                        raw_text=candidate["text"],
                        char_start=candidate["char_start"],
                        char_end=candidate["char_end"],
                    )
                )
            continue

        order = candidate["generated_order"]
        object_span_id = f"{row_id}:span-{order}"
        match = _OBJECT_RE.match(candidate["text"])
        if match is None:
            dropped.append(
                _drop(
                    row_id=row_id,
                    row_index=row_index,
                    object_span_id=object_span_id,
                    generated_order=order,
                    reason="malformed_object_span",
                    raw_text=candidate["text"],
                    char_start=candidate["char_start"],
                    char_end=candidate["char_end"],
                )
            )
            continue
        raw_span_text = match.group(0)
        span_char_start = candidate["char_start"]
        span_char_end = span_char_start + len(raw_span_text)
        trailing = candidate["text"][len(raw_span_text) :]
        description = match.group("description").strip()
        if not description:
            dropped.append(
                _drop(
                    row_id=row_id,
                    row_index=row_index,
                    object_span_id=object_span_id,
                    generated_order=order,
                    reason="empty_description",
                    raw_text=raw_span_text,
                    char_start=span_char_start,
                    char_end=span_char_end,
                    evidence=_span_evidence(raw_span_text, absolute_start=span_char_start),
                )
            )
            _append_unmatched_drop(
                dropped,
                row_id=row_id,
                row_index=row_index,
                raw_text=trailing,
                char_start=span_char_end,
            )
            continue
        try:
            coord_tokens = _COORD_RE.findall(match.group("coords"))
            coord_bins = [
                parse_coord_token(token, field=f"pred[{order}].bbox[{index}]")
                for index, token in enumerate(coord_tokens)
            ]
            bbox = coord_bins_to_pixel_xyxy(
                coord_bins,
                image_width=image_width,
                image_height=image_height,
                field=f"pred[{order}].bbox",
            )
        except DataContractError as exc:
            dropped.append(
                _drop(
                    row_id=row_id,
                    row_index=row_index,
                    object_span_id=object_span_id,
                    generated_order=order,
                    reason="geometry_invalid",
                    raw_text=raw_span_text,
                    char_start=span_char_start,
                    char_end=span_char_end,
                    code=exc.code,
                    context=exc.context,
                    evidence=_span_evidence(raw_span_text, absolute_start=span_char_start),
                )
            )
            _append_unmatched_drop(
                dropped,
                row_id=row_id,
                row_index=row_index,
                raw_text=trailing,
                char_start=span_char_end,
            )
            continue
        evidence = _span_evidence(raw_span_text, absolute_start=span_char_start)
        predictions.append(
            {
                "object_span_id": object_span_id,
                "description": description,
                "bbox": list(bbox),
                "bbox_format": "xyxy",
                "coord_bins": coord_bins,
                "generated_order": order,
                "char_start": span_char_start,
                "char_end": span_char_end,
                "raw_span_text": raw_span_text,
                "raw_span_sha256": _sha256_text(raw_span_text),
                **evidence,
            }
        )
        _append_unmatched_drop(
            dropped,
            row_id=row_id,
            row_index=row_index,
            raw_text=trailing,
            char_start=span_char_end,
        )

    if predictions and dropped:
        status = "accepted_with_drops"
    elif predictions:
        status = "accepted"
    elif dropped:
        status = "all_spans_dropped"
    else:
        status = "empty"
    return _row(
        row_id=row_id,
        row_index=row_index,
        status=status,
        predictions=predictions,
        dropped=dropped,
    )


def _object_candidates(text: str) -> list[dict[str, Any]]:
    starts = [match.start() for match in re.finditer(re.escape(OBJECT_REF_START_TOKEN), text)]
    candidates: list[dict[str, Any]] = []
    if not starts:
        if text:
            candidates.append(
                {
                    "kind": "unmatched",
                    "text": text,
                    "char_start": 0,
                    "char_end": len(text),
                }
            )
        return candidates

    if starts[0] > 0:
        candidates.append(
            {
                "kind": "unmatched",
                "text": text[: starts[0]],
                "char_start": 0,
                "char_end": starts[0],
            }
        )
    for order, start in enumerate(starts):
        end = starts[order + 1] if order + 1 < len(starts) else len(text)
        candidates.append(
            {
                "kind": "object",
                "generated_order": order,
                "text": text[start:end],
                "char_start": start,
                "char_end": end,
            }
        )
    return candidates


def _append_unmatched_drop(
    dropped: list[dict[str, Any]],
    *,
    row_id: str,
    row_index: int,
    raw_text: str,
    char_start: int,
) -> None:
    if _is_ignored_unmatched_text(raw_text):
        return
    dropped.append(
        _drop(
            row_id=row_id,
            row_index=row_index,
            object_span_id=f"{row_id}:unmatched-{len(dropped)}",
            generated_order=None,
            reason="unmatched_text",
            raw_text=raw_text,
            char_start=char_start,
            char_end=char_start + len(raw_text),
        )
    )


def _is_ignored_unmatched_text(text: str) -> bool:
    stripped = text.strip()
    return not stripped or stripped == IM_END_TOKEN


def _span_evidence(raw_span_text: str, *, absolute_start: int) -> dict[str, Any]:
    schema_spans: list[dict[str, Any]] = []
    for token in (
        OBJECT_REF_START_TOKEN,
        OBJECT_REF_END_TOKEN,
        BOX_START_TOKEN,
        BOX_END_TOKEN,
    ):
        relative_start = raw_span_text.find(token)
        if relative_start >= 0:
            schema_spans.append(
                _range_record(
                    token,
                    absolute_start=absolute_start,
                    relative_start=relative_start,
                )
            )
    coord_token_spans = [
        _range_record(
            match.group(0),
            absolute_start=absolute_start,
            relative_start=match.start(),
        )
        for match in _COORD_RE.finditer(raw_span_text)
    ]
    return {
        "schema_spans": schema_spans,
        "coord_token_spans": coord_token_spans,
    }


def _range_record(
    text: str,
    *,
    absolute_start: int,
    relative_start: int,
) -> dict[str, int | str]:
    char_start = absolute_start + relative_start
    return {
        "text": text,
        "char_start": char_start,
        "char_end": char_start + len(text),
    }


def _row(
    *,
    row_id: str,
    row_index: int,
    status: str,
    predictions: list[dict[str, Any]],
    dropped: list[dict[str, Any]],
) -> ParseRow:
    metric_bearing = bool(predictions)
    diagnostics = [
        {
            "row_id": row_id,
            "row_index": row_index,
            "parser_id": PARSER_ID,
            "parse_status": status,
            "valid_prediction_count": len(predictions),
            "dropped_prediction_count": len(dropped),
            "dropped_predictions": dropped,
        }
    ]
    return ParseRow(
        row_id=row_id,
        row_index=row_index,
        parser_id=PARSER_ID,
        parser_policy=PARSER_POLICY,
        metric_bearing=metric_bearing,
        parse_status=status,
        predictions=predictions,
        dropped_predictions=dropped,
        diagnostics=diagnostics,
    )


def _drop(
    *,
    row_id: str,
    row_index: int,
    object_span_id: str,
    generated_order: int | None,
    reason: str,
    raw_text: str,
    char_start: int,
    char_end: int,
    code: str | None = None,
    context: dict[str, Any] | None = None,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "row_id": row_id,
        "row_index": row_index,
        "object_span_id": object_span_id,
        "generated_order": generated_order,
        "reason": reason,
        "raw_text": raw_text,
        "char_start": char_start,
        "char_end": char_end,
        "raw_span_text": raw_text,
        "raw_span_sha256": _sha256_text(raw_text),
    }
    if evidence is not None:
        payload.update(evidence)
    if code is not None:
        payload["code"] = code
    if context is not None:
        payload["context"] = json.loads(json.dumps(context, default=str))
    return payload


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
