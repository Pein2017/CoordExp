from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

from src.utils.assistant_json import dumps_canonical_json

CoordJSONMode = Literal["strict", "salvage"]
ObjectFieldOrder = Literal["desc_first", "geometry_first"]

_COORD_LITERAL_RE = re.compile(r"<\|coord_(\d{1,4})\|>")
_GEOMETRY_KEYS = {"bbox_2d", "poly"}
_ALLOWED_RECORD_KEYS = {"bbox_2d", "poly", "desc"}
_ALLOWED_OBJECT_FIELD_ORDER: tuple[ObjectFieldOrder, ObjectFieldOrder] = (
    "desc_first",
    "geometry_first",
)


def _normalize_object_field_order(
    value: Any, *, path: str = "custom.object_field_order"
) -> ObjectFieldOrder:
    normalized = str(value).strip().lower()
    if normalized not in _ALLOWED_OBJECT_FIELD_ORDER:
        raise ValueError(
            f"{path} must be one of {{'desc_first', 'geometry_first'}}; got {value!r}"
        )
    return "geometry_first" if normalized == "geometry_first" else "desc_first"


def _build_object_payload(
    *,
    desc: str,
    geometry_key: str,
    geometry_value: Any,
    object_field_order: ObjectFieldOrder,
) -> Dict[str, Any]:
    if geometry_key not in {"bbox_2d", "poly"}:
        raise ValueError(
            f"geometry_key must be 'bbox_2d' or 'poly', got {geometry_key!r}"
        )
    if object_field_order == "geometry_first":
        return {
            geometry_key: geometry_value,
            "desc": desc,
        }
    return {
        "desc": desc,
        geometry_key: geometry_value,
    }


class CoordJSONValidationError(ValueError):
    """Raised when CoordJSON violates the strict contract."""


class _RecordValidationError(ValueError):
    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = str(reason)


@dataclass
class ParsedCoordJSONRecord:
    index: int
    geometry_key: Literal["bbox_2d", "poly"]
    geometry_values: List[int]
    desc: str
    key_order: Tuple[str, ...]
    record_span: Tuple[int, int]
    geometry_span: Tuple[int, int]


@dataclass
class CoordJSONParseResult:
    records: List[ParsedCoordJSONRecord] = field(default_factory=list)
    objects_array_open_cut: Optional[int] = None
    record_end_cuts: List[int] = field(default_factory=list)
    container_start: Optional[int] = None
    container_end: Optional[int] = None
    truncated: bool = False
    parse_failed: bool = False
    dropped_invalid_records: int = 0
    dropped_invalid_by_reason: Dict[str, int] = field(default_factory=dict)
    dropped_incomplete_tail: int = 0


@dataclass
class CoordJSONTranspileMeta:
    parse_failed: bool = False
    truncated: bool = False
    dropped_invalid_records: int = 0
    dropped_invalid_by_reason: Dict[str, int] = field(default_factory=dict)
    dropped_incomplete_tail: int = 0


def _skip_ws(text: str, idx: int, end: int) -> int:
    i = int(idx)
    while i < int(end) and text[i].isspace():
        i += 1
    return i


def _parse_coord_literal(text: str, idx: int, end: int) -> Tuple[int, int]:
    match = _COORD_LITERAL_RE.match(text, int(idx), int(end))
    if match is None:
        raise _RecordValidationError(
            "wrong_arity",
            f"Malformed coord literal at char {int(idx)}",
        )
    value = int(match.group(1))
    if value < 0 or value > 999:
        raise _RecordValidationError(
            "wrong_arity",
            f"Coord literal out of range at char {int(idx)}: {value}",
        )
    return value, int(match.end())


def _parse_json_string(text: str, idx: int, end: int) -> Tuple[str, int]:
    i = int(idx)
    if i >= int(end) or text[i] != '"':
        raise _RecordValidationError("other", f"Expected JSON string at char {i}")

    j = i + 1
    escaped = False
    while j < int(end):
        ch = text[j]
        if escaped:
            escaped = False
        elif ch == "\\":
            escaped = True
        elif ch == '"':
            raw = text[i : j + 1]
            try:
                value = json.loads(raw)
            except Exception as exc:
                raise _RecordValidationError(
                    "other",
                    f"Invalid JSON string literal at chars [{i}, {j + 1})",
                ) from exc
            if not isinstance(value, str):
                raise _RecordValidationError(
                    "other",
                    f"Expected decoded JSON string at chars [{i}, {j + 1})",
                )
            return value, int(j + 1)
        j += 1

    raise _RecordValidationError("other", f"Unterminated JSON string starting at char {i}")


def _scan_value_end(text: str, idx: int, end: int) -> int:
    i = int(idx)
    if i >= int(end):
        raise _RecordValidationError("other", "Missing value")

    ch0 = text[i]
    if ch0 == '"':
        _, j = _parse_json_string(text, i, int(end))
        return int(j)

    if ch0 == "<":
        _, j = _parse_coord_literal(text, i, int(end))
        return int(j)

    if ch0 in {"{", "["}:
        stack: List[str] = [ch0]
        j = i + 1
        in_string = False
        escaped = False
        while j < int(end):
            ch = text[j]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                j += 1
                continue

            if ch == '"':
                in_string = True
                j += 1
                continue
            if ch == "<":
                _, j = _parse_coord_literal(text, j, int(end))
                continue
            if ch in {"{", "["}:
                stack.append(ch)
                j += 1
                continue
            if ch in {"}", "]"}:
                if not stack:
                    raise _RecordValidationError("other", "Unexpected closing bracket")
                opening = stack.pop()
                if (opening, ch) not in {("{", "}"), ("[", "]")}:
                    raise _RecordValidationError(
                        "other",
                        f"Mismatched brackets: {opening!r} closed by {ch!r}",
                    )
                j += 1
                if not stack:
                    return int(j)
                continue
            j += 1

        raise _RecordValidationError("other", "Unterminated JSON value")

    j = i
    while j < int(end) and text[j] not in {",", "}", "]"}:
        j += 1
    if j == i:
        raise _RecordValidationError("other", f"Invalid value at char {i}")
    return int(j)


def _scan_balanced_object_end(text: str, start: int, end: int) -> Optional[int]:
    i = int(start)
    if i >= int(end) or text[i] != "{":
        return None

    depth = 1
    j = i + 1
    in_string = False
    escaped = False
    while j < int(end):
        ch = text[j]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            j += 1
            continue

        if ch == '"':
            in_string = True
            j += 1
            continue
        if ch == "{":
            depth += 1
            j += 1
            continue
        if ch == "}":
            depth -= 1
            j += 1
            if depth == 0:
                return int(j)
            continue
        j += 1

    return None


def _parse_coord_array(raw: str) -> List[int]:
    text = str(raw)
    n = len(text)
    i = _skip_ws(text, 0, n)
    if i >= n or text[i] != "[":
        raise _RecordValidationError("wrong_arity", "Geometry value must be an array")
    i += 1

    values: List[int] = []
    while True:
        i = _skip_ws(text, i, n)
        if i >= n:
            raise _RecordValidationError("wrong_arity", "Unterminated geometry array")
        if text[i] == "]":
            i += 1
            break

        if text[i] != "<":
            raise _RecordValidationError(
                "wrong_arity",
                "Geometry arrays must be CoordTok-only (bare <|coord_k|> literals)",
            )
        value, i = _parse_coord_literal(text, i, n)
        values.append(int(value))

        i = _skip_ws(text, i, n)
        if i >= n:
            raise _RecordValidationError("wrong_arity", "Unterminated geometry array")
        if text[i] == ",":
            i += 1
            continue
        if text[i] == "]":
            i += 1
            break
        raise _RecordValidationError("wrong_arity", "Invalid geometry array separator")

    i = _skip_ws(text, i, n)
    if i != n:
        raise _RecordValidationError("other", "Unexpected trailing text after geometry array")

    return values


def _expected_record_order(
    *, object_field_order: str, geometry_key: str
) -> Tuple[str, str]:
    if object_field_order == "geometry_first":
        return geometry_key, "desc"
    return "desc", geometry_key


def _parse_record(
    *,
    text: str,
    start: int,
    end: int,
    object_field_order: str,
    record_index: int,
) -> ParsedCoordJSONRecord:
    if int(start) < 0 or int(end) <= int(start):
        raise _RecordValidationError("other", "Invalid record span")

    i = int(start)
    if text[i] != "{":
        raise _RecordValidationError("other", f"Record at index {record_index} must start with '{{'")

    i += 1
    fields: List[Tuple[str, int, int]] = []

    while True:
        i = _skip_ws(text, i, int(end))
        if i >= int(end):
            raise _RecordValidationError("other", "Unterminated record object")
        if text[i] == "}":
            i += 1
            break

        key, i = _parse_json_string(text, i, int(end))
        i = _skip_ws(text, i, int(end))
        if i >= int(end) or text[i] != ":":
            raise _RecordValidationError("other", f"Missing ':' after key {key!r}")
        i += 1
        i = _skip_ws(text, i, int(end))
        value_start = i
        value_end = _scan_value_end(text, value_start, int(end))
        fields.append((str(key), int(value_start), int(value_end)))
        i = _skip_ws(text, value_end, int(end))

        if i < int(end) and text[i] == ",":
            i += 1
            continue
        if i < int(end) and text[i] == "}":
            i += 1
            break
        if i >= int(end):
            raise _RecordValidationError("other", "Unterminated record object")
        raise _RecordValidationError("other", "Invalid record separator")

    i = _skip_ws(text, i, int(end))
    if i != int(end):
        raise _RecordValidationError("other", "Unexpected trailing text in record")

    key_order = tuple(str(k) for k, _, _ in fields)
    if any(k not in _ALLOWED_RECORD_KEYS for k in key_order):
        raise _RecordValidationError("unexpected_keys", f"Record has unsupported keys: {key_order!r}")

    if key_order.count("desc") != 1:
        raise _RecordValidationError("missing_desc", "Record must contain exactly one non-empty desc")

    geometry_keys = [k for k in key_order if k in _GEOMETRY_KEYS]
    if len(geometry_keys) != 1:
        raise _RecordValidationError(
            "unexpected_keys",
            "Record must contain exactly one geometry key (bbox_2d or poly)",
        )

    if len(key_order) != 2:
        raise _RecordValidationError(
            "unexpected_keys",
            "Record schema is closed: only geometry + desc are allowed",
        )

    geometry_key = str(geometry_keys[0])
    expected_order = _expected_record_order(
        object_field_order=object_field_order,
        geometry_key=geometry_key,
    )
    if key_order != expected_order:
        raise _RecordValidationError(
            "order_violation",
            f"Record key order {key_order!r} violates required {expected_order!r}",
        )

    span_by_key: Dict[str, Tuple[int, int]] = {
        str(k): (int(vs), int(ve)) for k, vs, ve in fields
    }

    desc_start, desc_end = span_by_key["desc"]
    desc_raw = text[desc_start:desc_end]
    try:
        desc_value = json.loads(desc_raw)
    except Exception as exc:
        raise _RecordValidationError("missing_desc", "desc must be a valid JSON string") from exc
    if not isinstance(desc_value, str) or not str(desc_value).strip():
        raise _RecordValidationError("missing_desc", "desc must be a non-empty string")

    geom_start, geom_end = span_by_key[geometry_key]
    geom_raw = text[geom_start:geom_end]
    geometry_values = _parse_coord_array(geom_raw)

    if geometry_key == "bbox_2d":
        if len(geometry_values) != 4:
            raise _RecordValidationError("wrong_arity", "bbox_2d must contain exactly 4 CoordTok values")
    else:
        if len(geometry_values) < 6 or len(geometry_values) % 2 != 0:
            raise _RecordValidationError(
                "wrong_arity",
                "poly must contain an even number of CoordTok values and at least 6 entries",
            )

    return ParsedCoordJSONRecord(
        index=int(record_index),
        geometry_key=("bbox_2d" if geometry_key == "bbox_2d" else "poly"),
        geometry_values=[int(v) for v in geometry_values],
        desc=str(desc_value).strip(),
        key_order=tuple(key_order),
        record_span=(int(start), int(end)),
        geometry_span=(int(geom_start), int(geom_end)),
    )


def _increment_reason(counter: Dict[str, int], reason: str) -> None:
    key = str(reason)
    counter[key] = int(counter.get(key, 0)) + 1


def _parse_container_from_start(
    *,
    text: str,
    start: int,
    object_field_order: str,
    strict_mode: bool,
    allow_truncated: bool,
) -> Optional[CoordJSONParseResult]:
    n = len(text)
    i = int(start)
    if i >= n or text[i] != "{":
        return None

    result = CoordJSONParseResult(container_start=int(start))

    i += 1
    i = _skip_ws(text, i, n)
    if i >= n:
        if allow_truncated:
            result.truncated = True
            result.parse_failed = True
            return result
        return None

    try:
        key, i = _parse_json_string(text, i, n)
    except _RecordValidationError:
        return None

    if str(key) != "objects":
        return None

    i = _skip_ws(text, i, n)
    if i >= n or text[i] != ":":
        return None

    i += 1
    i = _skip_ws(text, i, n)
    if i >= n or text[i] != "[":
        return None

    result.objects_array_open_cut = int(i + 1)
    i += 1

    array_index = 0
    while True:
        i = _skip_ws(text, i, n)
        if i >= n:
            if allow_truncated:
                result.truncated = True
                break
            if strict_mode:
                raise CoordJSONValidationError("Top-level objects array is truncated")
            return None

        ch = text[i]
        if ch == "]":
            i += 1
            break

        if ch != "{":
            if strict_mode:
                raise CoordJSONValidationError(
                    f"Invalid objects array entry at char {i}; expected '{{'"
                )
            return None

        rec_start = int(i)
        rec_end = _scan_balanced_object_end(text, rec_start, n)
        if rec_end is None:
            if allow_truncated:
                result.truncated = True
                result.dropped_incomplete_tail += 1
                i = int(n)
                break
            if strict_mode:
                raise CoordJSONValidationError(
                    f"objects[{array_index}] is truncated and cannot be parsed"
                )
            return None

        result.record_end_cuts.append(int(rec_end))
        try:
            parsed_record = _parse_record(
                text=text,
                start=rec_start,
                end=int(rec_end),
                object_field_order=object_field_order,
                record_index=int(array_index),
            )
            result.records.append(parsed_record)
        except _RecordValidationError as rec_exc:
            if strict_mode:
                raise CoordJSONValidationError(
                    f"objects[{array_index}] invalid ({rec_exc.reason}): {str(rec_exc)}"
                ) from rec_exc
            result.dropped_invalid_records += 1
            _increment_reason(result.dropped_invalid_by_reason, rec_exc.reason)

        array_index += 1
        i = _skip_ws(text, int(rec_end), n)
        if i >= n:
            if allow_truncated:
                result.truncated = True
                break
            if strict_mode:
                raise CoordJSONValidationError("Top-level objects array is truncated")
            return None

        if text[i] == ",":
            i += 1
            continue
        if text[i] == "]":
            i += 1
            break

        if strict_mode:
            raise CoordJSONValidationError(
                f"Invalid objects array separator at char {i}: {text[i]!r}"
            )
        return None

    i = _skip_ws(text, i, n)
    if i < n and text[i] == ",":
        if strict_mode:
            raise CoordJSONValidationError(
                "Top-level CoordJSON object must contain exactly one key: 'objects'"
            )
        return None

    if i < n and text[i] == "}":
        i += 1
        result.container_end = int(i)
        return result

    if i >= n and allow_truncated:
        result.truncated = True
        result.container_end = int(n)
        return result

    if strict_mode:
        raise CoordJSONValidationError(
            "Top-level CoordJSON object is malformed; expected closing '}'"
        )
    return None


def _build_strict_payload(
    records: List[ParsedCoordJSONRecord], *, object_field_order: str
) -> Dict[str, Any]:
    objects: List[Dict[str, Any]] = []
    for record in records:
        payload = _build_object_payload(
            desc=str(record.desc),
            geometry_key=str(record.geometry_key),
            geometry_value=[int(v) for v in record.geometry_values],
            object_field_order=object_field_order,  # type: ignore[arg-type]
        )
        objects.append(payload)
    return {"objects": objects}


def parse_coordjson(
    text: str,
    *,
    mode: CoordJSONMode,
    object_field_order: str = "desc_first",
) -> CoordJSONParseResult:
    mode_norm = str(mode).strip().lower()
    if mode_norm not in {"strict", "salvage"}:
        raise ValueError(f"mode must be 'strict' or 'salvage'; got {mode!r}")

    field_order = _normalize_object_field_order(
        object_field_order,
        path="custom.object_field_order",
    )

    raw_text = str(text)
    if mode_norm == "strict":
        start = _skip_ws(raw_text, 0, len(raw_text))
        if start >= len(raw_text) or raw_text[start] != "{":
            raise CoordJSONValidationError("Strict CoordJSON must start with '{'")
        parsed = _parse_container_from_start(
            text=raw_text,
            start=int(start),
            object_field_order=field_order,
            strict_mode=True,
            allow_truncated=False,
        )
        if parsed is None:
            raise CoordJSONValidationError("Strict CoordJSON must contain top-level {'objects': [...]} structure")

        if parsed.container_end is None:
            raise CoordJSONValidationError("Strict CoordJSON parsing did not produce a closed top-level container")

        trailing = _skip_ws(raw_text, int(parsed.container_end), len(raw_text))
        if trailing != len(raw_text):
            raise CoordJSONValidationError(
                "Strict CoordJSON cannot contain leading/trailing junk outside the container"
            )

        parsed.parse_failed = False
        return parsed

    # salvage mode: scan for first valid container left-to-right.
    for pos, ch in enumerate(raw_text):
        if ch != "{":
            continue
        parsed = _parse_container_from_start(
            text=raw_text,
            start=int(pos),
            object_field_order=field_order,
            strict_mode=False,
            allow_truncated=True,
        )
        if parsed is None:
            continue
        parsed.parse_failed = False
        return parsed

    return CoordJSONParseResult(parse_failed=True)


def coordjson_to_strict_json_with_meta(
    text: str,
    *,
    mode: CoordJSONMode,
    object_field_order: str = "desc_first",
) -> Tuple[str, CoordJSONTranspileMeta]:
    parsed = parse_coordjson(
        text,
        mode=mode,
        object_field_order=object_field_order,
    )

    if bool(parsed.parse_failed):
        strict_text = dumps_canonical_json({"objects": []})
        return (
            strict_text,
            CoordJSONTranspileMeta(
                parse_failed=True,
                truncated=bool(parsed.truncated),
                dropped_invalid_records=0,
                dropped_invalid_by_reason={},
                dropped_incomplete_tail=0,
            ),
        )

    field_order = _normalize_object_field_order(
        object_field_order,
        path="custom.object_field_order",
    )
    strict_payload = _build_strict_payload(parsed.records, object_field_order=field_order)
    strict_text = dumps_canonical_json(strict_payload)
    return (
        strict_text,
        CoordJSONTranspileMeta(
            parse_failed=False,
            truncated=bool(parsed.truncated),
            dropped_invalid_records=int(parsed.dropped_invalid_records),
            dropped_invalid_by_reason=dict(parsed.dropped_invalid_by_reason),
            dropped_incomplete_tail=int(parsed.dropped_incomplete_tail),
        ),
    )


def coordjson_to_strict_json(
    text: str,
    *,
    mode: CoordJSONMode,
    object_field_order: str = "desc_first",
) -> str:
    strict_text, _ = coordjson_to_strict_json_with_meta(
        text,
        mode=mode,
        object_field_order=object_field_order,
    )
    return strict_text


__all__ = [
    "CoordJSONMode",
    "CoordJSONParseResult",
    "CoordJSONTranspileMeta",
    "CoordJSONValidationError",
    "ParsedCoordJSONRecord",
    "coordjson_to_strict_json",
    "coordjson_to_strict_json_with_meta",
    "parse_coordjson",
]
