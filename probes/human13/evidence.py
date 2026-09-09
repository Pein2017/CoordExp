"""Strict token-span evidence for finite-panel natural verification."""
from __future__ import annotations
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

@dataclass(frozen=True)
class RequestIdentity:
    backend: str
    backend_version: str
    mode: Literal["source_greedy", "k_sampled"]
    n: int
    seed: int | None
    physical_batch_index: int
    temperature: float
    top_p: float
    repetition_penalty: float
    max_new_tokens: int


@dataclass(frozen=True)
class PredictionRowInput:
    row_id: str
    row_index: int
    category: str
    bbox: tuple[float, float, float, float]
    token_start: int
    token_end: int
    final_coordinate_token_index: int
    parser_status: str = "complete"
    geometry_valid: bool = True
    row_terminated: bool = True


@dataclass(frozen=True)
class TrajectoryInput:
    trajectory_id: str
    request: RequestIdentity
    token_ids: tuple[int, ...]
    terminal_token_index: int | None
    stop_reason: str
    parser_status: str
    rows: tuple[PredictionRowInput, ...]

_PARSER_ID = "compact-object-box-closed-v1"


_PARSER_POLICY = "compact_object_box_closed_only"


_ALLOWED_PARSE_STATUSES = {
    "accepted",
    "accepted_with_drops",
    "empty",
    "all_spans_dropped",
}


_IM_END_TOKEN_ID = 151645


_SOURCE_POLICY = {
    "n": 1,
    "temperature": 0.0,
    "top_p": 1.0,
    "repetition_penalty": 1.0,
    "max_new_tokens": 3084,
}




def _field(value: Mapping[str, object] | object, name: str) -> object:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _normalized_trace(
    token_trace: Sequence[Mapping[str, object] | object],
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for expected_index, item in enumerate(token_trace):
        step_index = _field(item, "step_index")
        token_id = _field(item, "token_id")
        token_text = _field(item, "token_text")
        is_stop = _field(item, "is_stop")
        is_pad = _field(item, "is_pad")
        if step_index != expected_index:
            raise ValueError("token trace step indices are not contiguous")
        if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
            raise ValueError("token trace contains an invalid token id")
        if not isinstance(token_text, str) or not token_text:
            raise ValueError("token trace contains empty token text")
        if not isinstance(is_stop, bool) or not isinstance(is_pad, bool):
            raise ValueError("token trace stop/pad flags must be boolean")
        rows.append(
            {
                "step_index": step_index,
                "token_id": token_id,
                "token_text": token_text,
                "is_stop": is_stop,
                "is_pad": is_pad,
            }
        )
    if not rows:
        raise ValueError("token trace is empty")
    return tuple(rows)


def _trace_character_boundaries(
    *,
    parser_text: str,
    token_trace: Sequence[Mapping[str, object] | object],
) -> tuple[tuple[dict[str, object], ...], dict[int, int]]:
    trace = _normalized_trace(token_trace)
    non_pad = tuple(row for row in trace if not bool(row["is_pad"]))
    stop_indices = [index for index, row in enumerate(non_pad) if row["is_stop"]]
    if stop_indices and stop_indices != [len(non_pad) - 1]:
        raise ValueError("token trace has a non-terminal stop token")
    body = non_pad[:-1] if stop_indices else non_pad
    if "".join(str(row["token_text"]) for row in body) != parser_text:
        raise ValueError("token trace text does not exactly reconstruct parser text")
    boundaries = {0: 0}
    cursor = 0
    for token_index, row in enumerate(body):
        cursor += len(str(row["token_text"]))
        if cursor in boundaries:
            raise ValueError("token trace has ambiguous character boundaries")
        boundaries[cursor] = token_index + 1
    return non_pad, boundaries


def _text_occurrences(text: str, needle: str) -> tuple[int, ...]:
    starts: list[int] = []
    offset = 0
    while True:
        found = text.find(needle, offset)
        if found < 0:
            break
        starts.append(found)
        offset = found + 1
    return tuple(starts)


def align_char_span_to_token_span(
    *,
    parser_text: str,
    token_trace: Sequence[Mapping[str, object] | object],
    span_text: str,
    char_start: int | None,
    char_end: int | None,
) -> tuple[int, int]:
    """Map one parser span to exact generated-token boundaries.

    Absolute parser positions are authoritative.  A text-only fallback is
    accepted only when the text occurs exactly once, so repeated rows cannot be
    assigned by an arbitrary ``str.find`` result.
    """

    _, boundaries = _trace_character_boundaries(
        parser_text=parser_text,
        token_trace=token_trace,
    )
    if not span_text:
        raise ValueError("alignment span text must be non-empty")
    if char_start is None or char_end is None:
        occurrences = _text_occurrences(parser_text, span_text)
        if len(occurrences) != 1:
            raise ValueError("text-only token-span alignment is ambiguous or missing")
        char_start = occurrences[0]
        char_end = char_start + len(span_text)
    if (
        isinstance(char_start, bool)
        or not isinstance(char_start, int)
        or isinstance(char_end, bool)
        or not isinstance(char_end, int)
        or not 0 <= char_start < char_end <= len(parser_text)
    ):
        raise ValueError("parser character span is invalid")
    if parser_text[char_start:char_end] != span_text:
        raise ValueError("parser character span text does not match its evidence")
    if char_start not in boundaries or char_end not in boundaries:
        raise ValueError("parser character span does not align to token boundaries")
    return boundaries[char_start], boundaries[char_end]


def _terminal_index(
    *,
    token_ids: tuple[int, ...],
    stop_reason: object,
    trace: tuple[dict[str, object], ...],
) -> int | None:
    non_pad = tuple(row for row in trace if not bool(row["is_pad"]))
    traced_ids = tuple(int(row["token_id"]) for row in non_pad)
    if traced_ids != token_ids:
        raise ValueError("generated token ids differ from non-padding token trace")
    stop_indices = [index for index, row in enumerate(non_pad) if row["is_stop"]]
    if stop_reason == "im_end":
        if stop_indices != [len(token_ids) - 1]:
            raise ValueError("im_end result lacks one exact terminal token")
        return stop_indices[0]
    if stop_reason == "length":
        if stop_indices:
            raise ValueError("length result unexpectedly contains a terminal token")
        return None
    raise ValueError("result stop reason is outside the exact supported contract")


def trajectory_input_from_decode_result(
    *,
    image_id: int,
    trajectory_id: str,
    request: RequestIdentity,
    result: Mapping[str, object] | object,
    image_width: int,
    image_height: int,
) -> tuple[TrajectoryInput, dict[str, object]]:
    """Project one production DecodeResult through parser/token-span evidence."""

    from src.inference.parsing import parse_compact_object_box_closed

    token_ids_value = _field(result, "generated_token_ids")
    if not isinstance(token_ids_value, Sequence) or isinstance(
        token_ids_value, (str, bytes)
    ):
        raise ValueError("result generated token ids are missing")
    token_ids = tuple(int(value) for value in token_ids_value)
    if not token_ids or any(value < 0 for value in token_ids):
        raise ValueError("result generated token ids are invalid")
    parser_text = _field(result, "parser_text")
    if not isinstance(parser_text, str):
        raise ValueError("result parser text is missing")
    trace_value = _field(result, "token_trace")
    if not isinstance(trace_value, Sequence) or isinstance(trace_value, (str, bytes)):
        raise ValueError("result token trace is missing")
    trace = _normalized_trace(trace_value)
    stop_reason = _field(result, "stop_reason")
    terminal = _terminal_index(
        token_ids=token_ids,
        stop_reason=stop_reason,
        trace=trace,
    )
    _trace_character_boundaries(parser_text=parser_text, token_trace=trace)
    parsed = parse_compact_object_box_closed(
        parser_text,
        row_id=trajectory_id,
        row_index=0,
        image_width=image_width,
        image_height=image_height,
    )
    if parsed.parser_id != _PARSER_ID or parsed.parser_policy != _PARSER_POLICY:
        raise ValueError("parser identity differs from the frozen compact parser")
    if parsed.parse_status not in _ALLOWED_PARSE_STATUSES:
        raise ValueError("parser status is outside the strict discovery contract")
    rows: list[PredictionRowInput] = []
    previous_generated_order = -1
    for expected_order, prediction in enumerate(parsed.predictions):
        generated_order = prediction.get("generated_order")
        if (
            isinstance(generated_order, bool)
            or not isinstance(generated_order, int)
            or generated_order < 0
            or generated_order <= previous_generated_order
        ):
            raise ValueError(
                "accepted parser rows require strictly increasing generated order"
            )
        previous_generated_order = generated_order
        raw_span_text = prediction.get("raw_span_text")
        char_start = prediction.get("char_start")
        char_end = prediction.get("char_end")
        if not isinstance(raw_span_text, str):
            raise ValueError("complete row lacks raw parser span evidence")
        token_start, token_end = align_char_span_to_token_span(
            parser_text=parser_text,
            token_trace=trace,
            span_text=raw_span_text,
            char_start=char_start if isinstance(char_start, int) else None,
            char_end=char_end if isinstance(char_end, int) else None,
        )
        coord_spans = prediction.get("coord_token_spans")
        if not isinstance(coord_spans, list) or len(coord_spans) != 4:
            raise ValueError("complete row lacks four coordinate span records")
        final_coord = coord_spans[-1]
        if not isinstance(final_coord, Mapping):
            raise ValueError("final coordinate span is invalid")
        coordinate_start, coordinate_end = align_char_span_to_token_span(
            parser_text=parser_text,
            token_trace=trace,
            span_text=str(final_coord.get("text", "")),
            char_start=(
                int(final_coord["char_start"])
                if isinstance(final_coord.get("char_start"), int)
                else None
            ),
            char_end=(
                int(final_coord["char_end"])
                if isinstance(final_coord.get("char_end"), int)
                else None
            ),
        )
        if coordinate_end != coordinate_start + 1:
            raise ValueError("final coordinate does not align to exactly one token")
        bbox = prediction.get("bbox")
        category = prediction.get("description")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or not isinstance(category, str)
            or not category
        ):
            raise ValueError("complete row category/box evidence is invalid")
        rows.append(
            PredictionRowInput(
                row_id=f"{trajectory_id}:row:{expected_order:03d}",
                row_index=expected_order,
                category=category,
                bbox=tuple(float(value) for value in bbox),
                token_start=token_start,
                token_end=token_end,
                final_coordinate_token_index=coordinate_start,
                parser_status="complete",
                geometry_valid=True,
                row_terminated=True,
            )
        )
    trajectory = TrajectoryInput(
        trajectory_id=trajectory_id,
        request=request,
        token_ids=token_ids,
        terminal_token_index=terminal,
        stop_reason=str(stop_reason),
        parser_status=parsed.parse_status,
        rows=tuple(rows),
    )
    return trajectory, {
        "parser_id": parsed.parser_id,
        "parser_policy": parsed.parser_policy,
        "parse_status": parsed.parse_status,
        "dropped_predictions": parsed.dropped_predictions,
    }


def source_request(*, backend_version: str, repetition_penalty: float) -> RequestIdentity:
    return RequestIdentity(
        backend="hf", backend_version=backend_version, mode="source_greedy",
        seed=None, physical_batch_index=0,
        **{**_SOURCE_POLICY, "repetition_penalty": repetition_penalty},
    )
