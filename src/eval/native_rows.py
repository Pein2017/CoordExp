"""Mechanical conversion from native decode text to the standard detection row shape."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from src.inference.parsing import parse_compact_object_box_closed


def native_detection_record(
    text: str,
    case: Mapping[str, Any],
    golden: Mapping[str, Any],
    stop_reason: str,
) -> dict[str, Any]:
    """Parse one native completion while preserving the caller's bound GT envelope."""

    parsed = parse_compact_object_box_closed(
        text,
        row_id=str(case["row_id"]),
        row_index=golden["row_index"],
        image_width=golden["image_width"],
        image_height=golden["image_height"],
    ).to_artifact_dict()
    result = {
        key: copy.deepcopy(golden[key])
        for key in (
            "example_id",
            "gt",
            "image_height",
            "image_path",
            "image_width",
            "row_id",
            "row_index",
        )
    }
    result.update({key: value for key, value in parsed.items() if key != "predictions"})
    result.update(
        pred=parsed["predictions"],
        raw_decode_text=text,
        decode_stop_reason=stop_reason,
    )
    return result
