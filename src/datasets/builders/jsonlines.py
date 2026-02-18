"""JSON conversation builder for dense captioning"""

import base64
import json
import os
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Tuple

from src.common.object_field_order import (
    ObjectFieldOrder,
    build_object_payload,
    normalize_object_field_order,
)
from src.coord_tokens.codec import sequence_has_coord_tokens, tokens_to_ints

from ..contracts import ConversationRecord, validate_conversation_record
from ..utils import extract_object_points
from .base import BaseBuilder

MAX_NORM = 999


def _assert_norm_range(points: Iterable[Any], geom_type: str) -> None:
    for v in points:
        try:
            fv = float(v)
        except Exception:
            raise ValueError(
                f"{geom_type} contains a non-numeric value: {v!r}; expected pre-normalized coords in [0, 999]."
            ) from None
        if fv < 0 or fv > MAX_NORM:
            raise ValueError(
                f"{geom_type} value {fv} out of expected [0, 999] range for pre-normalized data."
            )


class JSONLinesBuilder(BaseBuilder):
    """Builder for dense caption conversations.

    Produces a single-round chat where the user embeds the image, followed by the
    assistant emitting the minimal object hierarchy (no 图片_N wrapper).

    Modes:
    - ``dense``: assistant returns a JSON object mapping ``object_{n}`` keys to
      geometry/description payloads.
    - ``summary``: assistant returns the summary string stored in the record.
    """

    def __init__(
        self,
        *,
        user_prompt: str,
        emit_norm: Literal["none", "norm100", "norm1000"],
        mode: Literal["dense", "summary"] = "dense",
        json_format: Literal["standard"] = "standard",
        coord_tokens_enabled: bool = False,
        object_field_order: ObjectFieldOrder = "desc_first",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.user_prompt = user_prompt
        self.emit_norm = emit_norm
        self.mode = mode
        self.json_format = json_format
        self.coord_tokens_enabled = bool(coord_tokens_enabled)
        self.object_field_order = normalize_object_field_order(object_field_order)

    def _get_summary_text(self, record: ConversationRecord, record_index: int) -> str:
        """Extract and validate summary from record.

        Args:
            record: The data record
            record_index: Index of the record (for error reporting)

        Returns:
            The summary string

        Raises:
            ValueError: if summary is missing or invalid
        """
        summary = record.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError(
                f"Missing or invalid 'summary' for record index {record_index}; "
                f"expected non-empty string. Please ensure all records in JSONL have a 'summary' field "
                f"when using summary mode."
            )
        return summary

    def build(self, record: ConversationRecord) -> Dict[str, Any]:
        """Build a single-record conversation payload."""
        return self.build_many([record])

    def build_many(self, records: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
        """Build conversation messages from one record.

        Dynamic pairing is no longer supported; this method fails if more than one
        record is provided to highlight legacy call paths.
        """

        records_list = list(records)
        if len(records_list) != 1:
            raise ValueError(
                "Dynamic pairing is no longer supported; JSONLinesBuilder expects exactly one record."
            )

        record = validate_conversation_record(records_list[0])

        user_contents: List[Dict[str, Any]] = []
        objects_out: Dict[str, List[Any]] = {"ref": [], "bbox": [], "image_id": []}
        objects_payload: Dict[str, Any] = {}

        images = record.get("images", []) or []
        objects = record.get("objects", []) or []

        for image in images:
            user_contents.append({"type": "image", "image": self._to_url(image)})

        if self.mode == "summary":
            assistant_payload: Any = self._get_summary_text(record, 0)
        else:
            assistant_payload = self._build_group_entry(objects, record)
            self._update_objects_metadata(objects_out, objects, 0)
            objects_payload = assistant_payload

        user_contents.append({"type": "text", "text": self.user_prompt})

        if self.mode == "summary":
            assistant_text = (
                assistant_payload
                if isinstance(assistant_payload, str)
                else self._render_json_text(assistant_payload)
            )
        else:
            assistant_text = self._render_json_text(assistant_payload)

        messages = [
            {"role": "user", "content": user_contents},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_text}],
            },
        ]

        merged: Dict[str, Any] = {"messages": messages}
        if objects_payload:
            merged["assistant_payload"] = objects_payload
        if objects_out["bbox"]:
            merged["objects"] = objects_out
        return merged

    def _build_group_entry(
        self, objects: List[Dict[str, Any]], record: ConversationRecord
    ) -> Dict[str, Any]:
        width = float(record.get("width") or 1)
        height = float(record.get("height") or 1)

        grouped_objects: Dict[str, Any] = {}
        for idx, obj in enumerate(objects, start=1):
            geom_type, points = extract_object_points(obj)
            if not geom_type or not points:
                raise ValueError(
                    f"Object object_{idx} must contain exactly one valid geometry field"
                )

            desc = self._sanitize_desc(obj.get("desc"), idx)
            text_points = self._select_text_points(obj, geom_type, points)
            numeric_points = self._select_numeric_points(obj, geom_type, points)
            obj.setdefault("_coord_numeric_cache", {})[geom_type] = numeric_points
            payload = build_object_payload(
                desc=desc,
                geometry_key=geom_type,
                geometry_value=self._format_points(
                    text_points, width, height, geom_type
                ),
                object_field_order=self.object_field_order,
            )
            grouped_objects[f"object_{idx}"] = payload
        return grouped_objects

    def _sanitize_desc(self, value: Any, object_index: int) -> str:
        if not isinstance(value, str):
            raise ValueError(
                f"Object object_{object_index} must provide a string 'desc'; got {type(value)!r}"
            )

        desc = value.strip()
        if not desc:
            raise ValueError(
                f"Object object_{object_index} has empty 'desc' after stripping whitespace"
            )

        if any(char in desc for char in "\n\r\t"):
            raise ValueError(
                f"Object object_{object_index} desc contains forbidden control whitespace"
            )

        disallowed_patterns = (" ,", ", ", " /", "/ ", " :", ": ", "  ")
        for pattern in disallowed_patterns:
            if pattern in desc:
                raise ValueError(
                    "Object object_{} desc contains disallowed whitespace pattern '{}'".format(
                        object_index, pattern
                    )
                )

        return desc

    def _update_objects_metadata(
        self,
        objects_out: Dict[str, List[Any]],
        objects: List[Dict[str, Any]],
        image_id: int,
    ) -> None:
        for obj in objects:
            geom_type, points = extract_object_points(obj)
            if not geom_type or not points:
                continue
            cached_numeric = obj.get("_coord_numeric_cache", {}).get(geom_type)
            points_for_meta = (
                cached_numeric
                if cached_numeric is not None
                else self._select_numeric_points(obj, geom_type, points)
            )
            objects_out["bbox"].append(points_for_meta)
            objects_out["image_id"].append(image_id)
            desc = obj.get("desc")
            if desc:
                objects_out["ref"].append(desc.split("/")[0])

    def _format_points(
        self, points: List[float], width: float, height: float, geom_type: str
    ) -> List[int | float]:
        if self.coord_tokens_enabled:
            return list(points)
        _assert_norm_range(points, geom_type)
        return [int(float(v)) for v in points]

    def _select_text_points(
        self, obj: Mapping[str, Any], geom_type: str, points: List[Any]
    ) -> List[Any]:
        if not self.coord_tokens_enabled:
            return points
        token_map = obj.get("_coord_tokens") if isinstance(obj, Mapping) else None
        if isinstance(token_map, Mapping) and geom_type in token_map:
            return list(token_map[geom_type])
        if sequence_has_coord_tokens(points):
            return list(points)
        return points

    def _select_numeric_points(
        self, obj: Mapping[str, Any], geom_type: str, points: List[Any]
    ) -> List[Any]:
        if self.coord_tokens_enabled:
            numeric_map = obj.get("_coord_token_ints") if isinstance(obj, Mapping) else None
            if isinstance(numeric_map, Mapping) and geom_type in numeric_map:
                nums = list(numeric_map[geom_type])
                _assert_norm_range(nums, geom_type)
                return nums
            if sequence_has_coord_tokens(points):
                nums = tokens_to_ints(points, require_even=True)
                _assert_norm_range(nums, geom_type)
                return nums
        return points

    def _render_json_text(self, payload: Mapping[str, Any]) -> str:
        text_payload = self._prepare_text_payload(payload)
        indent, separators = self._json_style()
        assistant_text = json.dumps(
            text_payload,
            ensure_ascii=False,
            indent=indent,
            separators=separators,
        )
        return assistant_text

    def _prepare_text_payload(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        formatted: Dict[str, Any] = {}
        for key, entry in payload.items():
            if isinstance(entry, Mapping):
                formatted[key] = self._format_object_entry(entry)
            else:
                formatted[key] = entry
        return formatted

    def _format_object_entry(self, entry: Mapping[str, Any]) -> Dict[str, Any]:
        formatted_entry: Dict[str, Any] = {}
        for field, value in entry.items():
            if field in {"poly"} and isinstance(value, list):
                formatted_entry[field] = self._format_geometry_sequence(value)
            elif field == "bbox_2d" and isinstance(value, list):
                formatted_entry[field] = list(value)
            else:
                formatted_entry[field] = value
        return formatted_entry

    def _format_geometry_sequence(self, values: List[int | float]) -> List[Any]:
        if not values:
            return []
        if len(values) % 2 != 0:
            return list(values)
        grouped: List[Any] = []
        for idx in range(0, len(values), 2):
            x = values[idx]
            y = values[idx + 1]
            grouped.append([x, y])
        return grouped

    def _json_style(self) -> Tuple[Optional[int], Tuple[str, str]]:
        return None, (", ", ": ")

    def _to_url(self, image: Any) -> str:
        """Canonicalize an image entry to a URL string for the template.

        - If dict with bytes: produce a data URL (PNG)
        - If relative path: prefix with ROOT_IMAGE_DIR when available
        - If absolute path: pass through
        """
        if isinstance(image, dict) and "bytes" in image:
            b = image["bytes"]
            if not isinstance(b, (bytes, bytearray)):
                raise TypeError("image bytes must be bytes-like")
            b64 = base64.b64encode(b).decode("ascii")
            return f"data:image/png;base64,{b64}"
        if isinstance(image, str):
            if not os.path.isabs(image):
                root = os.environ.get("ROOT_IMAGE_DIR")
                if root:
                    return os.path.join(root, image)
            return image
        # Fallback: stringify
        return str(image)


__all__ = ["JSONLinesBuilder"]
