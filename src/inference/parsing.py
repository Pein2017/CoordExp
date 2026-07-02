"""Compact object-box-closed parser for V1 inference."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from src.common.errors import DataContractError
from src.data.geometry import coord_bins_to_pixel_xyxy, parse_coord_token
from src.templates.renderer import (
    BOX_END_TOKEN,
    BOX_START_TOKEN,
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
    + re.escape(BOX_END_TOKEN)
    + r"$",
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
                    generated_order=None,
                    reason="unsupported_json_response",
                    raw_text=stripped,
                )
            ],
        )

    predictions: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for order, candidate in enumerate(_object_candidates(text)):
        match = _OBJECT_RE.fullmatch(candidate)
        if match is None:
            dropped.append(
                _drop(
                    row_id=row_id,
                    row_index=row_index,
                    generated_order=order,
                    reason="malformed_object_span",
                    raw_text=candidate,
                )
            )
            continue
        description = match.group("description").strip()
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
                    generated_order=order,
                    reason="geometry_invalid",
                    raw_text=match.group(0),
                    code=exc.code,
                    context=exc.context,
                )
            )
            continue
        predictions.append(
            {
                "description": description,
                "bbox": list(bbox),
                "bbox_format": "xyxy",
                "coord_bins": coord_bins,
                "generated_order": order,
            }
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


def _object_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    parts = text.split(OBJECT_REF_START_TOKEN)
    for part in parts[1:]:
        candidate = f"{OBJECT_REF_START_TOKEN}{part}".strip()
        if candidate:
            candidates.append(candidate)
    return candidates


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
    generated_order: int | None,
    reason: str,
    raw_text: str,
    code: str | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "row_id": row_id,
        "row_index": row_index,
        "generated_order": generated_order,
        "reason": reason,
        "raw_text": raw_text,
    }
    if code is not None:
        payload["code"] = code
    if context is not None:
        payload["context"] = json.loads(json.dumps(context, default=str))
    return payload
