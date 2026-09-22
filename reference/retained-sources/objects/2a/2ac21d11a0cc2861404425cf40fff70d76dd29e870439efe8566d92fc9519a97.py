"""Pure layout and loss helpers for the GT-free geometric-dedup pilot.

The layout is deliberately derived from the original generated token ids.  It
uses the native compact parser for discovery and maps its character evidence
back to those same token ids; it never encodes or retokenizes decoded text.
Parsing, geometry checks, and duplicate selection are detached bookkeeping.
Only the two loss helpers below touch tensors and therefore carry gradients.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import json
from typing import Any

import torch

from src.data.geometry import iou_xyxy
from src.inference.parsing import parse_compact_object_box_closed


DUPLICATE_IOU_THRESHOLD = 0.95
_UNLIKELIHOOD_EPSILON = 1e-8
_IM_END_TOKEN = "<|im_end|>"


def _decode(tokenizer: Any, token_ids: Sequence[int]) -> str:
    """Decode ids without changing their order or passing text through encode."""

    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        raise TypeError("tokenizer must expose decode(token_ids, ...) for exact history mapping")
    ids = [int(token_id) for token_id in token_ids]
    try:
        text = decode(
            ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        # A small fake tokenizer used by CPU tests may expose only the common
        # ``skip_special_tokens`` argument.  This fallback still decodes ids;
        # it never invokes tokenizer.encode/tokenize on generated text.
        text = decode(ids, skip_special_tokens=False)
    if not isinstance(text, str):
        text = str(text)
    return text


def _exact_token_text_frame(
    action_ids: Sequence[int], tokenizer: Any,
) -> tuple[str, list[tuple[int, int]]]:
    """Return decoded text and exact character ranges for each original id.

    The parser's character evidence is meaningful only in the same frame as
    the generated token sequence.  Requiring the per-token decode join to
    equal the full decode makes a tokenizer-frame mismatch fail closed rather
    than silently assigning spans to a retokenized/history-altered sequence.
    """

    ids = [int(token_id) for token_id in action_ids]
    full_text = _decode(tokenizer, ids)
    pieces = [_decode(tokenizer, [token_id]) for token_id in ids]
    joined = "".join(pieces)
    if joined != full_text:
        raise ValueError(
            "tokenizer per-token decode does not reconstruct full decode; "
            "cannot map parser spans to original action-token history"
        )
    spans: list[tuple[int, int]] = []
    cursor = 0
    for piece in pieces:
        end = cursor + len(piece)
        spans.append((cursor, end))
        cursor = end
    return full_text, spans


def _character_span_to_token_interval(
    char_start: Any,
    char_end: Any,
    *,
    text: str,
    token_spans: Sequence[tuple[int, int]],
) -> tuple[int, int]:
    """Map an exact parser character range to an original token interval."""

    if (
        isinstance(char_start, bool)
        or not isinstance(char_start, int)
        or isinstance(char_end, bool)
        or not isinstance(char_end, int)
        or char_start < 0
        or char_end <= char_start
        or char_end > len(text)
        or text[char_start:char_end] == ""
    ):
        raise ValueError("parser character span is invalid")

    # Empty decoded token pieces cannot provide a trustworthy boundary.  Do
    # not pick an arbitrary duplicate boundary if a tokenizer exposes one.
    start_matches = [
        index
        for index, (start, end) in enumerate(token_spans)
        if end > start and start == char_start
    ]
    end_matches = [
        index + 1
        for index, (start, end) in enumerate(token_spans)
        if end > start and end == char_end
    ]
    if len(start_matches) != 1 or len(end_matches) != 1:
        raise ValueError(
            "parser character span does not align to unique original token boundaries"
        )
    token_start, token_end = start_matches[0], end_matches[0]
    if token_end <= token_start:
        raise ValueError("parser character span maps to an empty token interval")
    if token_spans[token_start][0] != char_start or token_spans[token_end - 1][1] != char_end:
        raise ValueError("parser character span mapping is not contiguous")
    return token_start, token_end


def _copy_drop_with_optional_tokens(
    drop: Mapping[str, Any],
    *,
    text: str,
    token_spans: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    """Preserve native drop evidence and add best-effort original token spans."""

    value = copy.deepcopy(dict(drop))
    start, end = value.get("char_start"), value.get("char_end")
    try:
        token_start, token_end = _character_span_to_token_interval(
            start,
            end,
            text=text,
            token_spans=token_spans,
        )
    except ValueError:
        # A malformed/unmatched tail is not a negative and must not make an
        # otherwise usable row disappear.  Native character evidence remains
        # authoritative; token fields are included only when exact.
        return value
    value["token_start"] = token_start
    value["token_end"] = token_end
    value["token_positions"] = list(range(token_start, token_end))
    return value


def _terminal_im_end_id(tokenizer: Any) -> int | None:
    """Resolve the exact im_end token, never a generic EOS alias."""

    def checked(value: Any) -> int | None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        # Both tokenizers.Tokenizer and HF tokenizers can decode an id without
        # retokenizing.  Verify the id really decodes to im_end before using
        # it; an unknown-token id returned by a conversion API must not be
        # mistaken for the terminal marker.
        if _decode(tokenizer, [value]) != _IM_END_TOKEN:
            raise ValueError(
                "tokenizer im_end lookup did not resolve the exact <|im_end|> token"
            )
        return value

    token_to_id = getattr(tokenizer, "token_to_id", None)
    if callable(token_to_id):
        value = checked(token_to_id(_IM_END_TOKEN))
        if value is not None:
            return value

    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(convert):
        try:
            value = convert(_IM_END_TOKEN)
        except Exception:
            value = None
        value = checked(value)
        if value is not None:
            return value
    # Do not fall back to eos_token_id: it may be a provider-specific alias
    # and would silently add a non-im_end terminal state to the KL mask.
    return None


def _json_safe(value: Any) -> Any:
    """Normalize parser values so the public layout is JSON serializable."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    # Native parser diagnostics are already JSON-safe.  This guard keeps the
    # contract explicit if a tokenizer/parser implementation contributes an
    # enum-like scalar.
    return str(value)


def trajectory_layout(
    action_ids: Sequence[int],
    tokenizer: Any,
    *,
    image_width: int,
    image_height: int,
    row_id: str,
) -> dict[str, Any]:
    """Build GT-free duplicate and preservation positions for one trajectory.

    ``duplicate_row_indices`` are the parser's original ``generated_order``
    values, so malformed rows do not renumber the action history.  The
    corresponding ``duplicate_coordinate_positions`` entry always contains
    four original action-token indices.  ``valid_row_count`` counts all
    parser-accepted, geometry-valid rows, including duplicates, and is the
    denominator owned by :func:`duplicate_unlikelihood`.
    """

    if isinstance(action_ids, (str, bytes)) or not isinstance(action_ids, Sequence):
        raise TypeError("action_ids must be a sequence of original token ids")
    ids = []
    for index, token_id in enumerate(action_ids):
        if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
            raise ValueError(f"action_ids[{index}] must be a non-negative integer")
        ids.append(int(token_id))
    if (
        isinstance(image_width, bool)
        or not isinstance(image_width, int)
        or image_width <= 0
        or isinstance(image_height, bool)
        or not isinstance(image_height, int)
        or image_height <= 0
    ):
        raise ValueError("image dimensions must be positive integers")
    if not isinstance(row_id, str) or not row_id:
        raise ValueError("row_id must be a non-empty string")

    text, token_spans = _exact_token_text_frame(ids, tokenizer)
    parsed = parse_compact_object_box_closed(
        text,
        row_id=row_id,
        row_index=0,
        image_width=image_width,
        image_height=image_height,
    )

    rows: list[dict[str, Any]] = []
    for valid_row_index, prediction in enumerate(parsed.predictions):
        raw_span_text = prediction.get("raw_span_text")
        if not isinstance(raw_span_text, str) or not raw_span_text:
            raise ValueError("accepted parser row lacks raw span evidence")
        token_start, token_end = _character_span_to_token_interval(
            prediction.get("char_start"),
            prediction.get("char_end"),
            text=text,
            token_spans=token_spans,
        )
        coord_spans = prediction.get("coord_token_spans")
        if not isinstance(coord_spans, list) or len(coord_spans) != 4:
            raise ValueError("accepted parser row must expose four coordinate spans")
        coordinate_positions: list[int] = []
        for coord_span in coord_spans:
            if not isinstance(coord_span, Mapping):
                raise ValueError("coordinate parser evidence is not a mapping")
            coord_start, coord_end = _character_span_to_token_interval(
                coord_span.get("char_start"),
                coord_span.get("char_end"),
                text=text,
                token_spans=token_spans,
            )
            if coord_end != coord_start + 1:
                raise ValueError("each coordinate span must map to one original token")
            coordinate_positions.append(coord_start)
        bbox = prediction.get("bbox")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in bbox)
        ):
            raise ValueError("accepted parser row has invalid pixel bbox evidence")
        generated_order = prediction.get("generated_order")
        if isinstance(generated_order, bool) or not isinstance(generated_order, int) or generated_order < 0:
            raise ValueError("accepted parser row has invalid generated order")
        rows.append(
            {
                "row_index": int(generated_order),
                "valid_row_index": int(valid_row_index),
                "generated_order": int(generated_order),
                "object_span_id": str(prediction.get("object_span_id", "")),
                "char_start": int(prediction["char_start"]),
                "char_end": int(prediction["char_end"]),
                "token_start": int(token_start),
                "token_end": int(token_end),
                "token_positions": list(range(token_start, token_end)),
                "coordinate_positions": coordinate_positions,
                "bbox_pixel_xyxy": [float(value) if isinstance(value, float) else int(value) for value in bbox],
                "description": str(prediction.get("description", "")),
                "duplicate": False,
                "duplicate_reference_row_indices": [],
                "duplicate_reference_valid_row_indices": [],
                "duplicate_reference_ious": [],
            }
        )

    # Every accepted row enters the reference set, including rows already
    # flagged as duplicates.  Thus a later row can be a duplicate of an
    # earlier duplicate, and each later row is emitted exactly once.
    for current_index, row in enumerate(rows):
        current_box = tuple(row["bbox_pixel_xyxy"])
        references: list[tuple[int, int, float]] = []
        for previous in rows[:current_index]:
            previous_box = tuple(previous["bbox_pixel_xyxy"])
            overlap = float(iou_xyxy(current_box, previous_box))
            if overlap > DUPLICATE_IOU_THRESHOLD:
                references.append(
                    (
                        int(previous["row_index"]),
                        int(previous["valid_row_index"]),
                        overlap,
                    )
                )
        if references:
            row["duplicate"] = True
            row["duplicate_reference_row_indices"] = [item[0] for item in references]
            row["duplicate_reference_valid_row_indices"] = [item[1] for item in references]
            row["duplicate_reference_ious"] = [item[2] for item in references]

    duplicate_rows = [row for row in rows if row["duplicate"]]
    duplicate_row_indices = [int(row["row_index"]) for row in duplicate_rows]
    duplicate_coordinate_positions = [
        list(row["coordinate_positions"]) for row in duplicate_rows
    ]
    kl_positions = sorted(
        {
            position
            for row in rows
            if not row["duplicate"]
            for position in row["token_positions"]
        }
    )
    terminal_im_end_position: int | None = None
    im_end_id = _terminal_im_end_id(tokenizer)
    if ids and im_end_id is not None and ids[-1] == im_end_id:
        terminal_im_end_position = len(ids) - 1
        if terminal_im_end_position not in kl_positions:
            kl_positions.append(terminal_im_end_position)
            kl_positions.sort()

    parser_drop_rows = [
        _copy_drop_with_optional_tokens(drop, text=text, token_spans=token_spans)
        for drop in parsed.dropped_predictions
    ]
    invalid_geometry_rows = sorted(
        {
            int(drop["generated_order"])
            for drop in parsed.dropped_predictions
            if drop.get("reason") == "geometry_invalid"
            and isinstance(drop.get("generated_order"), int)
            and not isinstance(drop.get("generated_order"), bool)
        }
    )

    layout: dict[str, Any] = {
        "schema_version": "geometric_dedup.trajectory_layout.v1",
        "row_id": row_id,
        "parser_id": parsed.parser_id,
        "parser_policy": parsed.parser_policy,
        "parse_status": parsed.parse_status,
        "action_token_count": len(ids),
        "valid_row_count": len(rows),
        "duplicate_row_indices": duplicate_row_indices,
        "duplicate_valid_row_indices": [int(row["valid_row_index"]) for row in duplicate_rows],
        "duplicate_coordinate_positions": duplicate_coordinate_positions,
        "kl_positions": kl_positions,
        "terminal_im_end_position": terminal_im_end_position,
        # Keep the established probe convention: parser_drops is a count;
        # parser_drop_rows retains the native evidence for audit/debugging.
        "parser_drops": len(parser_drop_rows),
        "parser_drop_rows": parser_drop_rows,
        "invalid_geometry_rows": invalid_geometry_rows,
        "row_spans": rows,
    }
    # Fail at the layout boundary if a future parser/tokenizer contributes a
    # non-serializable object.  The returned value is therefore safe to put in
    # a launch packet without an implicit encoder or lossy stringification.
    try:
        json.dumps(layout, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("trajectory layout is not JSON serializable") from exc
    return _json_safe(layout)


def _checked_logits_and_targets(
    logits: torch.Tensor, targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(logits, torch.Tensor) or logits.ndim != 2 or logits.shape[1] <= 0:
        raise ValueError("logits must be a nonempty floating-point [T,V] tensor")
    if not logits.is_floating_point():
        raise ValueError("logits must be floating point")
    if not isinstance(targets, torch.Tensor) or targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
        raise ValueError("targets must be a [T] tensor aligned to logits")
    if targets.dtype != torch.long or targets.device != logits.device:
        raise ValueError("targets must be torch.long on the logits device")
    return logits, targets


def _layout_int(layout: Mapping[str, Any], key: str) -> int:
    value = layout.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"layout[{key!r}] must be a non-negative integer")
    return value


def _layout_positions(layout: Mapping[str, Any]) -> list[list[int]]:
    value = layout.get("duplicate_coordinate_positions", [])
    if not isinstance(value, list):
        raise ValueError("layout duplicate_coordinate_positions must be a list")
    result: list[list[int]] = []
    all_positions: set[int] = set()
    for row_index, positions in enumerate(value):
        if not isinstance(positions, list) or len(positions) != 4:
            raise ValueError(
                f"layout duplicate row {row_index} must expose four coordinate positions"
            )
        checked: list[int] = []
        for position in positions:
            if isinstance(position, bool) or not isinstance(position, int) or position < 0:
                raise ValueError("duplicate coordinate positions must be non-negative integers")
            checked.append(position)
        if len(set(checked)) != 4:
            raise ValueError("duplicate coordinate positions must identify four distinct tokens")
        if all_positions.intersection(checked):
            raise ValueError("duplicate coordinate rows must not reuse original token positions")
        all_positions.update(checked)
        result.append(checked)
    return result


def _differentiable_zero(logits: torch.Tensor) -> torch.Tensor:
    # Keep a graph edge for zero-eligible images so a DDP/trainer backward is
    # still valid without special-casing this image at the caller.
    return logits.sum() * 0.0


def duplicate_unlikelihood(
    logits: torch.Tensor,
    targets: torch.Tensor,
    layout: Mapping[str, Any],
) -> torch.Tensor:
    """Penalize sampled coordinate confidence in flagged duplicate rows.

    For each flagged row ``q = exp(mean(log p_i))`` over its four sampled
    coordinate tokens.  This is a geometric-mean confidence, not the joint
    probability of the four-token box.  The sum of row penalties is divided
    by all valid rows (including non-duplicates and duplicates); any outer
    image averaging remains the trainer's responsibility.
    """

    logits, targets = _checked_logits_and_targets(logits, targets)
    if not isinstance(layout, Mapping):
        raise ValueError("layout must be a mapping")
    valid_row_count = _layout_int(layout, "valid_row_count")
    duplicate_positions = _layout_positions(layout)
    if valid_row_count == 0 or not duplicate_positions:
        return _differentiable_zero(logits)

    token_count = int(logits.shape[0])
    vocab_size = int(logits.shape[1])
    flat_positions = [position for row in duplicate_positions for position in row]
    if any(position >= token_count for position in flat_positions):
        raise ValueError("duplicate row has a token position outside logits")
    position_tensor = torch.tensor(flat_positions, device=logits.device, dtype=torch.long)
    row_logits = logits.index_select(0, position_tensor).float()
    row_targets = targets.index_select(0, position_tensor)
    if bool(((row_targets < 0) | (row_targets >= vocab_size)).any()):
        raise ValueError("duplicate row has a target outside the vocabulary")
    sampled_logp = torch.log_softmax(row_logits, dim=-1).gather(
        1, row_targets[:, None]
    ).squeeze(1)
    q = sampled_logp.reshape(-1, 4).mean(dim=1).exp()
    row_losses = -torch.log(
        (1.0 - q + _UNLIKELIHOOD_EPSILON)
        / (1.0 + _UNLIKELIHOOD_EPSILON)
    )
    return row_losses.sum() / float(valid_row_count)


def preservation_kl(
    logits: torch.Tensor,
    reference_logp: torch.Tensor,
    positions: Sequence[int],
) -> torch.Tensor:
    """Mean detached-reference full-vocabulary forward KL at selected states."""

    if not isinstance(logits, torch.Tensor) or logits.ndim != 2 or logits.shape[1] <= 0:
        raise ValueError("logits must be a nonempty floating-point [T,V] tensor")
    if not logits.is_floating_point():
        raise ValueError("logits must be floating point")
    if isinstance(positions, torch.Tensor):
        if positions.ndim != 1:
            raise ValueError("positions must be a one-dimensional sequence")
        positions_list = positions.detach().cpu().tolist()
    elif isinstance(positions, Sequence) and not isinstance(positions, (str, bytes)):
        positions_list = list(positions)
    else:
        raise ValueError("positions must be a one-dimensional sequence")
    checked_positions: list[int] = []
    for position in positions_list:
        if isinstance(position, bool) or not isinstance(position, int) or position < 0:
            raise ValueError("positions must contain non-negative integers")
        if position >= logits.shape[0]:
            raise ValueError("a preservation position is outside logits")
        checked_positions.append(position)
    if len(set(checked_positions)) != len(checked_positions):
        raise ValueError("preservation positions must be unique")
    if not isinstance(reference_logp, torch.Tensor) or reference_logp.ndim != 2:
        raise ValueError("reference_logp must be a [selected_or_T,V] tensor")
    if reference_logp.shape[1] != logits.shape[1]:
        raise ValueError("reference_logp vocabulary dimension must match logits")
    if not checked_positions:
        return _differentiable_zero(logits)

    position_tensor = torch.tensor(checked_positions, device=logits.device, dtype=torch.long)
    if reference_logp.shape[0] == logits.shape[0]:
        reference = reference_logp.to(device=logits.device).index_select(0, position_tensor)
    elif reference_logp.shape[0] == len(checked_positions):
        reference = reference_logp.to(device=logits.device)
    else:
        raise ValueError(
            "reference_logp must cover all logits rows or exactly selected positions"
        )
    reference = reference.detach().float()
    student_logp = torch.log_softmax(
        logits.index_select(0, position_tensor).float(), dim=-1
    )
    per_state = (reference.exp() * (reference - student_logp)).sum(dim=-1)
    return per_state.mean()


__all__ = [
    "DUPLICATE_IOU_THRESHOLD",
    "duplicate_unlikelihood",
    "preservation_kl",
    "trajectory_layout",
]
