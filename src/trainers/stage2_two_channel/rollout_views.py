from dataclasses import dataclass, replace
import json
import re
from os import PathLike
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from src.common.duplicate_control import duplicate_control_object_from_bbox
from src.common.detection_sequence import BOX_START_TOKEN, OBJECT_REF_START_TOKEN
from src.common.semantic_desc import normalize_desc
from src.training.stage2.rollout_codec import (
    CompactFullRolloutCodec,
    Stage2RolloutTemplatePolicy,
)

from ..rollout_matching.contracts import GTObject
from ..rollout_matching.parsing import decode_pieces


_COMPACT_COORD_RE = re.compile(r"<\|coord_(0|[1-9]\d{0,2})\|>")


@dataclass(frozen=True)
class CompactFullObjectTokenSpan:
    object_index: int
    object_start: int
    desc_start: int
    desc_end: int
    box_start: int
    coord_positions: tuple[int, int, int, int]


@dataclass(frozen=True)
class PreparedRolloutAttempt:
    sample_id: str
    image_id: str
    image_path: str
    rollout_id: str
    response_token_ids: tuple[int, ...] | None
    raw_text: str
    decode_mode: str
    generation_config_hash: str
    sampling_seed: int | None = None
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class PreparedRolloutDedupStats:
    K_total: int
    K_after_dedup: int
    exact_duplicate_attempts: int
    dropped_reasons: Mapping[str, int]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "K_total": int(self.K_total),
            "K_after_dedup": int(self.K_after_dedup),
            "exact_duplicate_attempts": int(self.exact_duplicate_attempts),
            "dropped_reasons": dict(self.dropped_reasons),
        }

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[str(key)]


_PREPARED_ROLLOUT_REQUIRED_FIELDS = (
    "sample_id",
    "image_id",
    "image_path",
    "rollout_id",
    "raw_text",
    "decode_mode",
    "generation_config_hash",
)


def _require_prepared_string(record: Mapping[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"prepared rollout record missing required field: {field}")
    return str(value)


def _parse_response_token_ids(
    value: Any,
    *,
    strict_prepared_rollout_tokens: bool,
) -> tuple[int, ...] | None:
    if value is None:
        if strict_prepared_rollout_tokens:
            raise ValueError("prepared rollout record missing response_token_ids")
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("prepared rollout response_token_ids must be a sequence of integers")
    parsed: List[int] = []
    for token_id in value:
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise ValueError("prepared rollout response_token_ids must contain only integers")
        parsed.append(int(token_id))
    return tuple(parsed)


def parse_prepared_rollout_attempt(
    record: Mapping[str, Any],
    *,
    strict_prepared_rollout_tokens: bool,
) -> PreparedRolloutAttempt:
    if not isinstance(record, Mapping):
        raise TypeError("prepared rollout record must be a mapping")
    for field in _PREPARED_ROLLOUT_REQUIRED_FIELDS:
        _require_prepared_string(record, field)

    response_token_ids = _parse_response_token_ids(
        record.get("response_token_ids"),
        strict_prepared_rollout_tokens=bool(strict_prepared_rollout_tokens),
    )
    sampling_seed_raw = record.get("sampling_seed")
    sampling_seed: int | None = None
    if sampling_seed_raw is not None:
        if isinstance(sampling_seed_raw, bool) or not isinstance(sampling_seed_raw, int):
            raise ValueError("prepared rollout sampling_seed must be an integer")
        sampling_seed = int(sampling_seed_raw)

    preserved_keys = {
        key: value
        for key, value in record.items()
        if key
        not in {
            "sample_id",
            "image_id",
            "image_path",
            "rollout_id",
            "response_token_ids",
            "raw_text",
            "decode_mode",
            "generation_config_hash",
            "sampling_seed",
        }
    }
    return PreparedRolloutAttempt(
        sample_id=_require_prepared_string(record, "sample_id"),
        image_id=_require_prepared_string(record, "image_id"),
        image_path=_require_prepared_string(record, "image_path"),
        rollout_id=_require_prepared_string(record, "rollout_id"),
        response_token_ids=response_token_ids,
        raw_text=_require_prepared_string(record, "raw_text"),
        decode_mode=_require_prepared_string(record, "decode_mode"),
        generation_config_hash=_require_prepared_string(
            record, "generation_config_hash"
        ),
        sampling_seed=sampling_seed,
        metadata=preserved_keys,
    )


def load_prepared_rollout_jsonl(
    path: str | PathLike[str],
    *,
    strict_prepared_rollout_tokens: bool,
) -> List[PreparedRolloutAttempt]:
    attempts: List[PreparedRolloutAttempt] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid prepared rollout JSONL at line {int(line_no)}: {exc}"
                ) from exc
            try:
                attempts.append(
                    parse_prepared_rollout_attempt(
                        record,
                        strict_prepared_rollout_tokens=bool(
                            strict_prepared_rollout_tokens
                        ),
                    )
                )
            except Exception as exc:
                raise ValueError(
                    f"invalid prepared rollout record at line {int(line_no)}: {exc}"
                ) from exc
    return attempts


def validate_prepared_rollout_token_text(
    attempt: PreparedRolloutAttempt,
    *,
    tokenizer: Any,
) -> None:
    token_ids = attempt.response_token_ids
    if token_ids is None:
        return
    decoded = str(
        tokenizer.decode(
            [int(token_id) for token_id in token_ids],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    )
    if str(attempt.raw_text) != decoded:
        raise ValueError(
            "prepared rollout raw_text does not match tokenizer.decode("
            "response_token_ids) for rollout_id="
            f"{attempt.rollout_id!r}"
        )


def dedup_prepared_rollout_attempts(
    attempts: Sequence[PreparedRolloutAttempt],
    *,
    legacy_reencode_fallback: bool,
) -> tuple[List[PreparedRolloutAttempt], PreparedRolloutDedupStats]:
    kept: List[PreparedRolloutAttempt] = []
    seen: set[tuple[int, ...]] = set()
    dropped_reasons: Dict[str, int] = {}
    exact_duplicates = 0
    for attempt in attempts:
        token_ids = attempt.response_token_ids
        if token_ids is None:
            reason = (
                "missing_response_token_ids_legacy_reencode_unavailable"
                if bool(legacy_reencode_fallback)
                else "missing_response_token_ids"
            )
            dropped_reasons[reason] = int(dropped_reasons.get(reason, 0)) + 1
            continue
        key = tuple(int(token_id) for token_id in token_ids)
        if key in seen:
            exact_duplicates += 1
            dropped_reasons["exact_duplicate_response_token_ids"] = (
                int(dropped_reasons.get("exact_duplicate_response_token_ids", 0)) + 1
            )
            continue
        seen.add(key)
        kept.append(attempt)
    stats = PreparedRolloutDedupStats(
        K_total=int(len(attempts)),
        K_after_dedup=int(len(kept)),
        exact_duplicate_attempts=int(exact_duplicates),
        dropped_reasons=dropped_reasons,
    )
    return kept, stats


def _response_text_from_rollout(
    *,
    tokenizer: Any,
    response_token_ids: List[int],
    response_text: str,
) -> str:
    text = str(response_text or "")
    if text:
        return text

    return str(
        tokenizer.decode(
            [int(t) for t in response_token_ids],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    )


def _view_from_bbox_objects(
    *,
    parsed_objects: List[GTObject],
    source_label: str,
    duplicate_iou_threshold: float,
    center_radius_scale: float,
    duplicate_diagnostics_fn: Any,
) -> tuple[List[Any], Mapping[str, Any]]:
    duplicate_control_objects_raw = [
        duplicate_control_object_from_bbox(
            index=int(index),
            desc=str(obj.desc),
            bbox_norm1000=obj.points_norm1000,
            source=str(source_label),
        )
        for index, obj in enumerate(parsed_objects)
    ]
    duplicate_metrics = duplicate_diagnostics_fn(
        parsed_objects,
        duplicate_iou_threshold=float(duplicate_iou_threshold),
        center_radius_scale=float(center_radius_scale),
    )
    return duplicate_control_objects_raw, duplicate_metrics


def extract_compact_full_object_token_spans(
    *,
    tokenizer: Any,
    response_token_ids: List[int],
    parsed_objects: List[GTObject] | None = None,
) -> List[CompactFullObjectTokenSpan]:
    token_ids = [int(t) for t in response_token_ids]
    if not token_ids:
        return []

    pieces = decode_pieces(tokenizer, token_ids)
    token_spans: List[tuple[int, int]] = []
    cursor = 0
    for piece in pieces:
        start = int(cursor)
        cursor += int(len(piece))
        token_spans.append((start, int(cursor)))
    text = "".join(pieces)

    terminal_positions = [
        pos
        for marker in ("<|im_end|>", "<|endoftext|>")
        if (pos := text.find(marker)) >= 0
    ]
    parse_text = text[: min(terminal_positions)] if terminal_positions else text

    object_starts: List[int] = []
    search_start = 0
    while True:
        found = parse_text.find(OBJECT_REF_START_TOKEN, search_start)
        if found < 0:
            break
        object_starts.append(int(found))
        search_start = int(found + len(OBJECT_REF_START_TOKEN))

    if not object_starts:
        if parse_text.strip("\r\n"):
            raise ValueError("compact-full span extractor found row without object_start marker")
        return []
    if parse_text[: object_starts[0]].strip("\r\n"):
        raise ValueError("compact-full span extractor found row without object_start marker")

    spans_by_row_index: dict[int, CompactFullObjectTokenSpan] = {}
    payload_by_row_index: dict[int, tuple[str, tuple[int, int, int, int]]] = {}
    for row_index, row_start in enumerate(object_starts):
        next_row_start = (
            int(object_starts[int(row_index) + 1])
            if int(row_index) + 1 < len(object_starts)
            else len(parse_text)
        )
        row_end = int(next_row_start)
        while row_end > int(row_start) and parse_text[int(row_end) - 1] in "\r\n":
            row_end -= 1
        row = parse_text[int(row_start) : int(row_end)]
        if not row:
            continue
        if not row.startswith(OBJECT_REF_START_TOKEN):
            raise ValueError("compact-full span extractor found row without object_start marker")
        box_rel = row.rfind(BOX_START_TOKEN)
        if box_rel < len(OBJECT_REF_START_TOKEN):
            raise ValueError("compact-full span extractor found row without box_start marker")

        coord_tail_start = box_rel + len(BOX_START_TOKEN)
        coord_matches = list(_COMPACT_COORD_RE.finditer(row, coord_tail_start))
        if len(coord_matches) != 4:
            raise ValueError("compact-full span extractor requires exactly four coord tokens")
        if coord_matches[0].start() != coord_tail_start:
            raise ValueError("compact-full coord tail must start immediately after box_start")
        if any(
            left.end() != right.start()
            for left, right in zip(coord_matches, coord_matches[1:])
        ):
            raise ValueError("compact-full coord tokens must be contiguous")
        if coord_matches[-1].end() != len(row):
            raise ValueError("compact-full coord tail must end at row boundary")
        coord_values = tuple(int(match.group(1)) for match in coord_matches)
        if len(coord_values) != 4:
            raise ValueError("compact-full span extractor requires four coord values")

        object_start = _marker_token_for_char_span(
            token_spans,
            row_start,
            row_start + len(OBJECT_REF_START_TOKEN),
            label="object_start",
        )
        desc_start_char = row_start + len(OBJECT_REF_START_TOKEN)
        desc_end_char = row_start + box_rel
        desc_positions = _token_indices_overlapping_char_span(
            token_spans,
            desc_start_char,
            desc_end_char,
        )
        if not desc_positions:
            raise ValueError("compact-full span extractor found empty desc token span")
        box_start = _marker_token_for_char_span(
            token_spans,
            row_start + box_rel,
            row_start + box_rel + len(BOX_START_TOKEN),
            label="box_start",
        )
        coord_positions = tuple(
            _single_token_for_char_span(
                token_spans,
                row_start + match.start(),
                row_start + match.end(),
                label="coord",
            )
            for match in coord_matches
        )
        spans_by_row_index[int(row_index)] = CompactFullObjectTokenSpan(
            object_index=int(row_index),
            object_start=int(object_start),
            desc_start=int(min(desc_positions)),
            desc_end=int(max(desc_positions) + 1),
            box_start=int(box_start),
            coord_positions=coord_positions,  # type: ignore[arg-type]
        )
        payload_by_row_index[int(row_index)] = (
            row[len(OBJECT_REF_START_TOKEN) : box_rel],
            coord_values,  # type: ignore[arg-type]
        )

    if parsed_objects is None:
        return [spans_by_row_index[index] for index in sorted(spans_by_row_index)]

    aligned: List[CompactFullObjectTokenSpan] = []
    for obj in parsed_objects:
        obj_index = int(obj.index)
        span = spans_by_row_index.get(obj_index)
        if span is None:
            raise ValueError(
                "compact-full span extractor could not align parsed_bbox_objects_raw"
            )
        desc_text, coord_values = payload_by_row_index[obj_index]
        if normalize_desc(desc_text) != normalize_desc(str(obj.desc)):
            raise ValueError("compact-full span desc does not align parsed object")
        if list(coord_values) != [int(v) for v in obj.points_norm1000]:
            raise ValueError("compact-full span coords do not align parsed object")
        aligned.append(span)
    return aligned


def _token_indices_overlapping_char_span(
    token_spans: List[tuple[int, int]],
    char_start: int,
    char_end: int,
) -> List[int]:
    return [
        int(index)
        for index, (start, end) in enumerate(token_spans)
        if int(start) < int(char_end) and int(end) > int(char_start)
    ]


def _single_token_for_char_span(
    token_spans: List[tuple[int, int]],
    char_start: int,
    char_end: int,
    *,
    label: str,
) -> int:
    positions = _token_indices_overlapping_char_span(token_spans, char_start, char_end)
    if len(positions) != 1:
        raise ValueError(f"compact-full {label} span must align to one token")
    return int(positions[0])


def _marker_token_for_char_span(
    token_spans: List[tuple[int, int]],
    char_start: int,
    char_end: int,
    *,
    label: str,
) -> int:
    positions = _token_indices_overlapping_char_span(token_spans, char_start, char_end)
    if not positions:
        raise ValueError(f"compact-full {label} span did not align to tokens")
    return int(positions[0])


def _build_compact_full_rollout_view(
    *,
    tokenizer: Any,
    duplicate_iou_threshold: float,
    center_radius_scale: float,
    max_new_tokens: int,
    rollout_result: Tuple[List[int], str, str, List[int]],
    source_label: str,
    duplicate_diagnostics_fn: Any,
    rollout_template_policy: Stage2RolloutTemplatePolicy,
) -> Dict[str, Any]:
    resp_ids, resp_text, rollout_decode_mode, prompt_ids = rollout_result
    resp_ids_local = [int(t) for t in resp_ids]
    eos_id = getattr(tokenizer, "eos_token_id", None)
    truncated_by_budget = False
    if int(max_new_tokens) > 0 and int(len(resp_ids_local)) >= int(max_new_tokens):
        try:
            eos = int(eos_id) if eos_id is not None else -1
        except (TypeError, ValueError):
            eos = -1
        if eos >= 0 and (not resp_ids_local or int(resp_ids_local[-1]) != eos):
            resp_ids_local.append(int(eos))
            truncated_by_budget = True

    response_text = _response_text_from_rollout(
        tokenizer=tokenizer,
        response_token_ids=resp_ids_local,
        response_text=resp_text,
    )
    codec = CompactFullRolloutCodec(rollout_template_policy)
    parse = codec.parse(response_text)
    parse = replace(
        parse,
        response_token_ids=tuple(int(t) for t in resp_ids_local),
        truncated=bool(parse.truncated or truncated_by_budget),
    )

    drop_reasons: Dict[str, int] = {}
    drop_bbox_invalid = 0
    parsed_bbox_objects_raw: List[GTObject] = []
    for pobj in list(parse.valid_objects):
        if pobj.geom_type != "bbox_2d" or pobj.bbox_norm1000 is None:
            drop_bbox_invalid += 1
            continue
        try:
            x1, y1, x2, y2 = [int(x) for x in pobj.bbox_norm1000]
        except (TypeError, ValueError):
            drop_bbox_invalid += 1
            continue
        if x2 <= x1 or y2 <= y1:
            drop_bbox_invalid += 1
            continue

        parsed_bbox_objects_raw.append(
            GTObject(
                index=int(pobj.index),
                geom_type="bbox_2d",
                points_norm1000=[x1, y1, x2, y2],
                desc=str(pobj.desc),
            )
        )

    if drop_bbox_invalid:
        drop_reasons["bbox_invalid"] = int(drop_bbox_invalid)

    parse_empty_after_geometry_filter = bool(
        parse.valid_objects and not parsed_bbox_objects_raw
    )
    if parse_empty_after_geometry_filter:
        parse = replace(
            parse,
            empty_valid_object_set=True,
            fallback_reason="empty_valid_object_set",
            dropped_invalid=int(parse.dropped_invalid) + int(drop_bbox_invalid),
            dropped_invalid_by_reason={
                **dict(parse.dropped_invalid_by_reason),
                "bbox_invalid": int(drop_bbox_invalid),
            },
        )

    duplicate_control_objects_raw, duplicate_metrics = _view_from_bbox_objects(
        parsed_objects=parsed_bbox_objects_raw,
        source_label=source_label,
        duplicate_iou_threshold=duplicate_iou_threshold,
        center_radius_scale=center_radius_scale,
        duplicate_diagnostics_fn=duplicate_diagnostics_fn,
    )
    fallback_applies = bool(parse.invalid_rollout or parse.empty_valid_object_set)

    compact_full_object_spans: List[CompactFullObjectTokenSpan] = []
    if not fallback_applies:
        compact_full_object_spans = extract_compact_full_object_token_spans(
            tokenizer=tokenizer,
            response_token_ids=resp_ids_local,
            parsed_objects=parsed_bbox_objects_raw,
        )

    return {
        "prompt_ids": [int(t) for t in prompt_ids],
        "decode_mode": str(rollout_decode_mode),
        "pred_objects": int(len(parse.valid_objects)),
        "parse_truncated": int(1 if bool(parse.truncated) else 0),
        "gen_new_tokens": int(len(parse.response_token_ids)),
        "parse": parse,
        "invalid_rollout": int(1 if bool(parse.invalid_rollout) else 0),
        "empty_valid_object_set": int(
            1 if bool(parse.empty_valid_object_set) else 0
        ),
        "fallback_reason": parse.fallback_reason,
        "compact_fallback_applies": int(1 if fallback_applies else 0),
        "drop_reasons": drop_reasons,
        "drop_poly": int(0),
        "drop_unknown": int(0),
        "drop_bbox_invalid": int(drop_bbox_invalid),
        "parsed_bbox_objects_raw": parsed_bbox_objects_raw,
        "compact_full_object_spans": compact_full_object_spans,
        "duplicate_control_objects_raw": duplicate_control_objects_raw,
        "n_valid_pred": int(len(parsed_bbox_objects_raw)),
        "n_drop_invalid": int(drop_bbox_invalid),
        "duplicate_metrics": duplicate_metrics,
        "rollout_template_family": "compact_full",
        "rollout_parser_id": str(rollout_template_policy.parser_id),
        "rollout_append_policy_id": str(rollout_template_policy.append_policy_id),
        "rollout_counts_as_valid_rollout": int(0 if fallback_applies else 1),
        "rollout_fallback_loss_weight": (
            float(rollout_template_policy.fallback_loss_weight)
            if fallback_applies
            else 0.0
        ),
    }


def build_channel_b_rollout_view(
    *,
    tokenizer: Any,
    object_field_order: str,
    coord_id_to_bin: Mapping[int, int],
    duplicate_iou_threshold: float,
    center_radius_scale: float,
    max_new_tokens: int,
    rollout_result: Tuple[List[int], str, str, List[int]],
    source_label: str,
    parse_rollout_for_matching_fn: Any,
    points_from_coord_tokens_fn: Any,
    duplicate_diagnostics_fn: Any,
    rollout_template_policy: Stage2RolloutTemplatePolicy | None = None,
) -> Dict[str, Any]:
    if (
        rollout_template_policy is not None
        and rollout_template_policy.template_family == "compact_full"
    ):
        return _build_compact_full_rollout_view(
            tokenizer=tokenizer,
            duplicate_iou_threshold=duplicate_iou_threshold,
            center_radius_scale=center_radius_scale,
            max_new_tokens=max_new_tokens,
            rollout_result=rollout_result,
            source_label=source_label,
            duplicate_diagnostics_fn=duplicate_diagnostics_fn,
            rollout_template_policy=rollout_template_policy,
        )

    resp_ids, _resp_text, rollout_decode_mode, prompt_ids = rollout_result
    resp_ids_local = [int(t) for t in resp_ids]
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if int(max_new_tokens) > 0 and int(len(resp_ids_local)) >= int(max_new_tokens):
        try:
            eos = int(eos_id) if eos_id is not None else -1
        except (TypeError, ValueError):
            eos = -1
        if eos >= 0 and (not resp_ids_local or int(resp_ids_local[-1]) != eos):
            resp_ids_local.append(int(eos))

    parse = parse_rollout_for_matching_fn(
        tokenizer=tokenizer,
        response_token_ids=resp_ids_local,
        object_field_order=object_field_order,
    )
    invalid_rollout = int(1 if bool(getattr(parse, "invalid_rollout", False)) else 0)

    drop_reasons: Dict[str, int] = {}
    raw = getattr(parse, "dropped_invalid_by_reason", None)
    if isinstance(raw, Mapping):
        for k, v in raw.items():
            try:
                drop_reasons[str(k)] = int(v)
            except (TypeError, ValueError):
                continue

    drop_poly = 0
    drop_unknown = 0
    drop_bbox_invalid = 0
    parsed_bbox_objects_raw: List[GTObject] = []
    for pobj in list(parse.valid_objects):
        if pobj.geom_type != "bbox_2d":
            if pobj.geom_type == "poly":
                drop_poly += 1
            else:
                drop_unknown += 1
            continue

        pts = points_from_coord_tokens_fn(
            response_token_ids=parse.response_token_ids,
            coord_token_indices=pobj.coord_token_indices,
            coord_id_to_bin=coord_id_to_bin,
        )
        if pts is None or len(pts) != 4:
            drop_bbox_invalid += 1
            continue
        try:
            x1, y1, x2, y2 = [int(x) for x in pts]
        except (TypeError, ValueError):
            drop_bbox_invalid += 1
            continue
        # Keep the rollout view contract aligned with duplicate-control:
        # zero-width / zero-height boxes are invalid and should be dropped.
        if x2 <= x1 or y2 <= y1:
            drop_bbox_invalid += 1
            continue

        parsed_bbox_objects_raw.append(
            GTObject(
                index=int(pobj.index),
                geom_type="bbox_2d",
                points_norm1000=[x1, y1, x2, y2],
                desc=str(pobj.desc),
            )
        )

    if drop_poly:
        drop_reasons["poly_unsupported"] = int(
            drop_reasons.get("poly_unsupported", 0)
        ) + int(drop_poly)
    if drop_unknown:
        drop_reasons["unknown_geom"] = int(drop_reasons.get("unknown_geom", 0)) + int(
            drop_unknown
        )
    if drop_bbox_invalid:
        drop_reasons["bbox_invalid"] = int(drop_reasons.get("bbox_invalid", 0)) + int(
            drop_bbox_invalid
        )

    n_valid_pred = int(len(parsed_bbox_objects_raw))
    n_drop_invalid = (
        int(getattr(parse, "dropped_invalid", 0) or 0)
        + int(getattr(parse, "dropped_ambiguous", 0) or 0)
        + int(drop_poly)
        + int(drop_unknown)
        + int(drop_bbox_invalid)
    )

    duplicate_control_objects_raw, duplicate_metrics = _view_from_bbox_objects(
        parsed_objects=parsed_bbox_objects_raw,
        source_label=source_label,
        duplicate_iou_threshold=duplicate_iou_threshold,
        center_radius_scale=center_radius_scale,
        duplicate_diagnostics_fn=duplicate_diagnostics_fn,
    )

    return {
        "prompt_ids": [int(t) for t in prompt_ids],
        "decode_mode": str(rollout_decode_mode),
        "pred_objects": int(len(parse.valid_objects)),
        "parse_truncated": int(1 if bool(getattr(parse, "truncated", False)) else 0),
        "gen_new_tokens": int(len(parse.response_token_ids)),
        "parse": parse,
        "invalid_rollout": int(invalid_rollout),
        "empty_valid_object_set": int(0),
        "fallback_reason": None,
        "compact_fallback_applies": int(0),
        "drop_reasons": drop_reasons,
        "drop_poly": int(drop_poly),
        "drop_unknown": int(drop_unknown),
        "drop_bbox_invalid": int(drop_bbox_invalid),
        "parsed_bbox_objects_raw": parsed_bbox_objects_raw,
        "duplicate_control_objects_raw": duplicate_control_objects_raw,
        "n_valid_pred": int(n_valid_pred),
        "n_drop_invalid": int(n_drop_invalid),
        "duplicate_metrics": duplicate_metrics,
        "rollout_template_family": "coordjson",
        "rollout_parser_id": (
            str(rollout_template_policy.parser_id)
            if rollout_template_policy is not None
            else "coordjson_legacy"
        ),
        "rollout_append_policy_id": (
            str(rollout_template_policy.append_policy_id)
            if rollout_template_policy is not None
            else "coordjson_append"
        ),
        "rollout_counts_as_valid_rollout": int(0 if invalid_rollout else 1),
        "rollout_fallback_loss_weight": float(0.0),
    }


__all__ = [
    "CompactFullObjectTokenSpan",
    "PreparedRolloutAttempt",
    "PreparedRolloutDedupStats",
    "build_channel_b_rollout_view",
    "dedup_prepared_rollout_attempts",
    "extract_compact_full_object_token_spans",
    "load_prepared_rollout_jsonl",
    "parse_prepared_rollout_attempt",
    "validate_prepared_rollout_token_text",
]
