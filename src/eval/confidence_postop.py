"""Offline confidence post-operation for score-aware COCO evaluation.

This module joins:
- `gt_vs_pred.jsonl` (base inference artifact), and
- `pred_token_trace.jsonl` (generated token text + token logprobs)

to emit:
- `pred_confidence.jsonl` (per-object confidence sidecar),
- `gt_vs_pred_scored.jsonl` (derived scored artifact with dropped unscorable preds),
- `confidence_postop_summary.json` (run-level keep/drop summary).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.common.coord_standardizer import CoordinateStandardizer
from src.common.geometry import decode_coord, flatten_points
from src.common.geometry.object_geometry import extract_single_geometry
from src.eval.bbox_confidence import (
    EXPECTED_BBOX_COORD_TOKEN_COUNT,
    assign_spans_left_to_right,
    compute_bbox_confidence_from_logprobs,
    coord_bins_to_tokens,
    is_valid_confidence_score,
)

PRED_SCORE_SOURCE = "confidence_postop"
PRED_SCORE_VERSION = 1
CONFIDENCE_METHOD = "bbox_coord_mean_logprob_exp"

MISSING_TRACE = "missing_trace"
TRACE_LEN_MISMATCH = "trace_len_mismatch"
UNSUPPORTED_GEOMETRY_TYPE = "unsupported_geometry_type"
MISSING_COORD_BINS = "missing_coord_bins"
MISSING_SPAN = "missing_span"
NONFINITE_LOGPROB = "nonfinite_logprob"
PRED_ALIGNMENT_MISMATCH = "pred_alignment_mismatch"
OBJECT_IDX_OOB = "object_idx_oob"

FAILURE_REASONS = {
    MISSING_TRACE,
    TRACE_LEN_MISMATCH,
    UNSUPPORTED_GEOMETRY_TYPE,
    MISSING_COORD_BINS,
    MISSING_SPAN,
    NONFINITE_LOGPROB,
    PRED_ALIGNMENT_MISMATCH,
    OBJECT_IDX_OOB,
}


@dataclass(frozen=True)
class ConfidencePostOpPaths:
    gt_vs_pred_jsonl: Path
    pred_token_trace_jsonl: Path
    pred_confidence_jsonl: Path
    gt_vs_pred_scored_jsonl: Path
    confidence_postop_summary_json: Path


@dataclass(frozen=True)
class TraceRecord:
    line_idx: int
    generated_token_text: list[str]
    token_logprobs: list[float]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fin:
        for line_no, raw in enumerate(fin, start=1):
            line = raw.rstrip("\n")
            if not line.strip():
                raise ValueError(f"{path}:{line_no} is blank; JSONL records must be non-empty")
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is malformed JSON") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no} must be a JSON object")
            records.append(obj)
    return records


def _require_int(value: Any, *, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be int-compatible, got {value!r}") from exc


def _as_mode(value: Any) -> str:
    mode = str(value or "").strip().lower()
    return "coord" if mode == "coord" else "text"


def _normalize_desc(desc: Any) -> str:
    return str(desc or "").strip()


def _normalize_points(points: Any) -> list[int] | None:
    if not isinstance(points, list):
        return None
    out: list[int] = []
    for value in points:
        try:
            out.append(int(round(float(value))))
        except (TypeError, ValueError):
            return None
    return out


def _compact_object(obj: Mapping[str, Any]) -> dict[str, Any] | None:
    gtype = obj.get("type")
    points = obj.get("points")
    if gtype not in {"bbox_2d", "poly"}:
        return None
    if not isinstance(points, list):
        return None
    return {
        "type": gtype,
        "points": points,
        "desc": _normalize_desc(obj.get("desc", "")),
    }


def _extract_bbox_coord_bins(raw_object: Mapping[str, Any]) -> list[int] | None:
    try:
        gtype, points_raw = extract_single_geometry(
            raw_object,
            allow_type_and_points=True,
            allow_nested_points=True,
            path="raw_output_json.objects[*]",
        )
    except ValueError:
        return None

    if gtype != "bbox_2d":
        return None

    flat = flatten_points(points_raw)
    if flat is None or len(flat) != EXPECTED_BBOX_COORD_TOKEN_COUNT:
        return None

    bins: list[int] = []
    for value in flat:
        decoded = decode_coord(value)
        if decoded is None:
            return None
        bins.append(int(decoded))
    return bins


def _reconstruct_pred_alignment(
    *,
    raw_output_json: Any,
    width: int,
    height: int,
    mode: str,
) -> tuple[list[dict[str, Any]], list[list[int] | None]] | None:
    if not isinstance(raw_output_json, dict):
        return None
    objects = raw_output_json.get("objects")
    if not isinstance(objects, list):
        return None

    standardizer = CoordinateStandardizer(mode=mode, pred_coord_mode="auto")
    reconstructed: list[dict[str, Any]] = []
    bbox_bins_by_object: list[list[int] | None] = []

    for raw_obj in objects:
        if not isinstance(raw_obj, dict):
            continue
        local_errors: list[str] = []
        standardized = standardizer.process_objects(
            [dict(raw_obj)],
            width=width,
            height=height,
            is_gt=False,
            errors=local_errors,
        )
        if not standardized:
            continue
        compact = _compact_object(standardized[0])
        if compact is None:
            continue
        reconstructed.append(compact)
        if compact["type"] == "bbox_2d":
            bbox_bins_by_object.append(_extract_bbox_coord_bins(raw_obj))
        else:
            bbox_bins_by_object.append(None)

    return reconstructed, bbox_bins_by_object


def _pred_alignment_matches(
    emitted_pred: Sequence[Mapping[str, Any]],
    reconstructed_pred: Sequence[Mapping[str, Any]],
) -> bool:
    if len(emitted_pred) != len(reconstructed_pred):
        return False

    for emitted_obj, reconstructed_obj in zip(emitted_pred, reconstructed_pred):
        emitted_type = str(emitted_obj.get("type", ""))
        reconstructed_type = str(reconstructed_obj.get("type", ""))
        if emitted_type != reconstructed_type:
            return False

        emitted_desc = _normalize_desc(emitted_obj.get("desc", ""))
        reconstructed_desc = _normalize_desc(reconstructed_obj.get("desc", ""))
        if emitted_desc != reconstructed_desc:
            return False

        emitted_points = _normalize_points(emitted_obj.get("points"))
        reconstructed_points = _normalize_points(reconstructed_obj.get("points"))
        if emitted_points is None or reconstructed_points is None:
            return False
        if emitted_points != reconstructed_points:
            return False
    return True


def _confidence_details(
    *,
    coord_token_count: int,
    matched_token_indices: Sequence[int],
    ambiguous_matches: int,
    failure_reason: str | None,
) -> dict[str, Any]:
    return {
        "method": CONFIDENCE_METHOD,
        "coord_token_count": int(coord_token_count),
        "matched_token_indices": [int(i) for i in matched_token_indices],
        "ambiguous_matches": int(ambiguous_matches),
        "failure_reason": failure_reason,
    }


def _build_object_result(
    *,
    object_idx: int,
    pred_obj: Mapping[str, Any],
    confidence: float | None,
    score: float | None,
    kept: bool,
    failure_reason: str | None,
    coord_token_count: int,
    matched_token_indices: Sequence[int],
    ambiguous_matches: int,
) -> dict[str, Any]:
    return {
        "object_idx": int(object_idx),
        "type": str(pred_obj.get("type", "")),
        "desc": str(pred_obj.get("desc", "")),
        "points": list(pred_obj.get("points") or []),
        "confidence": confidence,
        "score": score,
        "kept": bool(kept),
        "confidence_details": _confidence_details(
            coord_token_count=coord_token_count,
            matched_token_indices=matched_token_indices,
            ambiguous_matches=ambiguous_matches,
            failure_reason=failure_reason,
        ),
    }


def _build_sample_failure_objects(
    pred_objs: Sequence[Mapping[str, Any]],
    *,
    failure_reason: str,
) -> list[dict[str, Any]]:
    return [
        _build_object_result(
            object_idx=object_idx,
            pred_obj=pred_obj,
            confidence=None,
            score=None,
            kept=False,
            failure_reason=failure_reason,
            coord_token_count=0,
            matched_token_indices=[],
            ambiguous_matches=0,
        )
        for object_idx, pred_obj in enumerate(pred_objs)
    ]


def _load_trace_index(path: Path) -> dict[int, TraceRecord]:
    traces: dict[int, TraceRecord] = {}
    for line_no, record in enumerate(_read_jsonl(path), start=1):
        line_idx = _require_int(record.get("line_idx"), name=f"{path}:{line_no}.line_idx")
        if line_idx < 0:
            raise ValueError(f"{path}:{line_no}.line_idx must be >= 0")
        if line_idx in traces:
            raise ValueError(f"{path}:{line_no} duplicates line_idx={line_idx}")

        generated_token_text_raw = record.get("generated_token_text")
        token_logprobs_raw = record.get("token_logprobs")
        if not isinstance(generated_token_text_raw, list):
            raise ValueError(f"{path}:{line_no}.generated_token_text must be a list[str]")
        if not isinstance(token_logprobs_raw, list):
            raise ValueError(f"{path}:{line_no}.token_logprobs must be a list[number]")

        generated_token_text: list[str] = []
        for idx, token in enumerate(generated_token_text_raw):
            if not isinstance(token, str):
                raise ValueError(
                    f"{path}:{line_no}.generated_token_text[{idx}] must be a string"
                )
            generated_token_text.append(token)

        token_logprobs: list[float] = []
        for idx, value in enumerate(token_logprobs_raw):
            try:
                token_logprobs.append(float(value))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{path}:{line_no}.token_logprobs[{idx}] must be numeric"
                ) from exc

        traces[line_idx] = TraceRecord(
            line_idx=line_idx,
            generated_token_text=generated_token_text,
            token_logprobs=token_logprobs,
        )
    return traces


def _compute_sample_confidence_objects(
    *,
    line_idx: int,
    record: Mapping[str, Any],
    trace: TraceRecord | None,
) -> list[dict[str, Any]]:
    emitted_pred = record.get("pred")
    pred_objs = list(emitted_pred) if isinstance(emitted_pred, list) else []
    if not pred_objs:
        return []

    if trace is None:
        return _build_sample_failure_objects(pred_objs, failure_reason=MISSING_TRACE)
    if len(trace.generated_token_text) != len(trace.token_logprobs):
        return _build_sample_failure_objects(pred_objs, failure_reason=TRACE_LEN_MISMATCH)

    raw_output_json = record.get("raw_output_json")
    if raw_output_json is None:
        reconstructed = None
    else:
        try:
            width = _require_int(record.get("width"), name=f"line[{line_idx}].width")
            height = _require_int(record.get("height"), name=f"line[{line_idx}].height")
        except ValueError:
            reconstructed = None
        else:
            reconstructed = _reconstruct_pred_alignment(
                raw_output_json=raw_output_json,
                width=width,
                height=height,
                mode=_as_mode(record.get("mode")),
            )

    bbox_bins_by_object: list[list[int] | None] = [None] * len(pred_objs)
    if reconstructed is not None:
        reconstructed_pred, reconstructed_bins = reconstructed
        if not _pred_alignment_matches(pred_objs, reconstructed_pred):
            return _build_sample_failure_objects(
                pred_objs,
                failure_reason=PRED_ALIGNMENT_MISMATCH,
            )
        if len(reconstructed_bins) != len(pred_objs):
            return _build_sample_failure_objects(
                pred_objs,
                failure_reason=PRED_ALIGNMENT_MISMATCH,
            )
        bbox_bins_by_object = reconstructed_bins

    expected_sequences: list[tuple[int, tuple[str, ...]]] = []
    for object_idx, pred_obj in enumerate(pred_objs):
        if str(pred_obj.get("type", "")) != "bbox_2d":
            continue
        bins = bbox_bins_by_object[object_idx]
        if bins is None:
            continue
        try:
            tokens = coord_bins_to_tokens(bins)
        except ValueError:
            continue
        expected_sequences.append((object_idx, tokens))

    span_matches_by_idx: dict[int, Any] = {}
    if expected_sequences:
        assignments = assign_spans_left_to_right(
            generated_token_text=trace.generated_token_text,
            expected_sequences=[tokens for _, tokens in expected_sequences],
        )
        for (object_idx, _tokens), match in zip(expected_sequences, assignments):
            span_matches_by_idx[object_idx] = match

    results: list[dict[str, Any]] = []
    for object_idx, pred_obj in enumerate(pred_objs):
        gtype = str(pred_obj.get("type", ""))
        if gtype != "bbox_2d":
            results.append(
                _build_object_result(
                    object_idx=object_idx,
                    pred_obj=pred_obj,
                    confidence=None,
                    score=None,
                    kept=False,
                    failure_reason=UNSUPPORTED_GEOMETRY_TYPE,
                    coord_token_count=0,
                    matched_token_indices=[],
                    ambiguous_matches=0,
                )
            )
            continue

        bins = bbox_bins_by_object[object_idx]
        if bins is None:
            results.append(
                _build_object_result(
                    object_idx=object_idx,
                    pred_obj=pred_obj,
                    confidence=None,
                    score=None,
                    kept=False,
                    failure_reason=MISSING_COORD_BINS,
                    coord_token_count=0,
                    matched_token_indices=[],
                    ambiguous_matches=0,
                )
            )
            continue

        span_match = span_matches_by_idx.get(object_idx)
        if span_match is None:
            results.append(
                _build_object_result(
                    object_idx=object_idx,
                    pred_obj=pred_obj,
                    confidence=None,
                    score=None,
                    kept=False,
                    failure_reason=MISSING_SPAN,
                    coord_token_count=EXPECTED_BBOX_COORD_TOKEN_COUNT,
                    matched_token_indices=[],
                    ambiguous_matches=0,
                )
            )
            continue

        matched_indices = list(span_match.matched_token_indices)
        ambiguous_matches = int(span_match.ambiguous_matches)
        matched_logprobs = [trace.token_logprobs[idx] for idx in matched_indices]
        if not all(math.isfinite(lp) for lp in matched_logprobs):
            results.append(
                _build_object_result(
                    object_idx=object_idx,
                    pred_obj=pred_obj,
                    confidence=None,
                    score=None,
                    kept=False,
                    failure_reason=NONFINITE_LOGPROB,
                    coord_token_count=EXPECTED_BBOX_COORD_TOKEN_COUNT,
                    matched_token_indices=matched_indices,
                    ambiguous_matches=ambiguous_matches,
                )
            )
            continue

        score = compute_bbox_confidence_from_logprobs(matched_logprobs)
        if not is_valid_confidence_score(score):
            results.append(
                _build_object_result(
                    object_idx=object_idx,
                    pred_obj=pred_obj,
                    confidence=None,
                    score=None,
                    kept=False,
                    failure_reason=NONFINITE_LOGPROB,
                    coord_token_count=EXPECTED_BBOX_COORD_TOKEN_COUNT,
                    matched_token_indices=matched_indices,
                    ambiguous_matches=ambiguous_matches,
                )
            )
            continue

        results.append(
            _build_object_result(
                object_idx=object_idx,
                pred_obj=pred_obj,
                confidence=float(score),
                score=float(score),
                kept=True,
                failure_reason=None,
                coord_token_count=EXPECTED_BBOX_COORD_TOKEN_COUNT,
                matched_token_indices=matched_indices,
                ambiguous_matches=ambiguous_matches,
            )
        )

    return results


def _enforce_object_idx_contract(
    pred_objs: Sequence[Mapping[str, Any]],
    confidence_objects: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if len(pred_objs) != len(confidence_objects):
        return _build_sample_failure_objects(pred_objs, failure_reason=OBJECT_IDX_OOB)

    out: list[dict[str, Any]] = []
    for expected_idx, (pred_obj, conf_obj_any) in enumerate(zip(pred_objs, confidence_objects)):
        conf_obj = dict(conf_obj_any)
        object_idx = conf_obj.get("object_idx")
        if not isinstance(object_idx, int) or object_idx != expected_idx:
            return _build_sample_failure_objects(pred_objs, failure_reason=OBJECT_IDX_OOB)

        kept = bool(conf_obj.get("kept", False))
        score = conf_obj.get("score")
        confidence = conf_obj.get("confidence")
        details = conf_obj.get("confidence_details")
        failure_reason = None
        if isinstance(details, Mapping):
            failure_reason = details.get("failure_reason")

        if kept:
            try:
                score_f = float(score)
            except (TypeError, ValueError):
                score_f = float("nan")
            if not is_valid_confidence_score(score_f):
                return _build_sample_failure_objects(pred_objs, failure_reason=OBJECT_IDX_OOB)
            conf_obj["score"] = score_f
            conf_obj["confidence"] = float(confidence)
            if isinstance(conf_obj.get("confidence_details"), dict):
                conf_obj["confidence_details"]["failure_reason"] = None
        else:
            if failure_reason not in FAILURE_REASONS:
                return _build_sample_failure_objects(pred_objs, failure_reason=OBJECT_IDX_OOB)
            conf_obj["confidence"] = None
            conf_obj["score"] = None
        out.append(conf_obj)
    return out


def _build_scored_record(
    *,
    record: Mapping[str, Any],
    confidence_objects: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    pred_objs = list(record.get("pred")) if isinstance(record.get("pred"), list) else []
    confidence_checked = _enforce_object_idx_contract(pred_objs, confidence_objects)

    scored_pred: list[dict[str, Any]] = []
    for pred_obj, conf_obj in zip(pred_objs, confidence_checked):
        kept = bool(conf_obj.get("kept", False))
        if not kept:
            continue
        score = float(conf_obj["score"])
        scored_obj = dict(pred_obj)
        scored_obj["score"] = score
        scored_pred.append(scored_obj)

    out = dict(record)
    out["pred"] = scored_pred
    out["pred_score_source"] = PRED_SCORE_SOURCE
    out["pred_score_version"] = PRED_SCORE_VERSION
    return out


def run_confidence_postop(paths: ConfidencePostOpPaths) -> dict[str, Any]:
    gt_records = _read_jsonl(paths.gt_vs_pred_jsonl)
    traces = _load_trace_index(paths.pred_token_trace_jsonl)

    paths.pred_confidence_jsonl.parent.mkdir(parents=True, exist_ok=True)
    paths.gt_vs_pred_scored_jsonl.parent.mkdir(parents=True, exist_ok=True)
    paths.confidence_postop_summary_json.parent.mkdir(parents=True, exist_ok=True)

    total_samples = 0
    total_pred_objects = 0
    kept_pred_objects = 0
    dropped_by_reason: dict[str, int] = {}

    with (
        paths.pred_confidence_jsonl.open("w", encoding="utf-8") as confidence_out,
        paths.gt_vs_pred_scored_jsonl.open("w", encoding="utf-8") as scored_out,
    ):
        for line_idx, record in enumerate(gt_records):
            trace = traces.get(line_idx)
            confidence_objects = _compute_sample_confidence_objects(
                line_idx=line_idx,
                record=record,
                trace=trace,
            )
            pred_objs = list(record.get("pred")) if isinstance(record.get("pred"), list) else []
            confidence_objects = _enforce_object_idx_contract(pred_objs, confidence_objects)

            confidence_record = {
                "line_idx": int(line_idx),
                "image": str(record.get("image", "")),
                "objects": confidence_objects,
            }
            confidence_out.write(json.dumps(confidence_record, ensure_ascii=False) + "\n")

            scored_record = _build_scored_record(
                record=record,
                confidence_objects=confidence_objects,
            )
            scored_out.write(json.dumps(scored_record, ensure_ascii=False) + "\n")

            total_samples += 1
            total_pred_objects += len(confidence_objects)
            for obj in confidence_objects:
                if bool(obj.get("kept", False)):
                    kept_pred_objects += 1
                    continue
                details = obj.get("confidence_details")
                reason = None
                if isinstance(details, Mapping):
                    reason = details.get("failure_reason")
                if reason not in FAILURE_REASONS:
                    reason = OBJECT_IDX_OOB
                dropped_by_reason[str(reason)] = dropped_by_reason.get(str(reason), 0) + 1

    dropped_pred_objects = total_pred_objects - kept_pred_objects
    kept_fraction = (
        float(kept_pred_objects) / float(total_pred_objects)
        if total_pred_objects > 0
        else 1.0
    )
    summary = {
        "total_samples": int(total_samples),
        "total_pred_objects": int(total_pred_objects),
        "kept_pred_objects": int(kept_pred_objects),
        "dropped_pred_objects": int(dropped_pred_objects),
        "kept_fraction": float(kept_fraction),
        "dropped_by_reason": dropped_by_reason,
        "pred_score_source": PRED_SCORE_SOURCE,
        "pred_score_version": PRED_SCORE_VERSION,
    }
    paths.confidence_postop_summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def _resolve_path(
    artifacts_cfg: Mapping[str, Any],
    *,
    key: str,
    run_dir: Path | None,
    default_name: str | None,
    required: bool,
) -> Path:
    raw = artifacts_cfg.get(key)
    if isinstance(raw, str) and raw.strip():
        return Path(raw)
    if run_dir is not None and default_name is not None:
        return run_dir / default_name
    if required:
        raise ValueError(
            f"artifacts.{key} is required (or provide artifacts.run_dir for defaults)"
        )
    raise ValueError(f"Could not resolve artifacts.{key}")


def paths_from_config(cfg: Mapping[str, Any]) -> ConfidencePostOpPaths:
    artifacts_cfg_raw = cfg.get("artifacts", {})
    if not isinstance(artifacts_cfg_raw, Mapping):
        raise ValueError("artifacts must be a mapping")
    artifacts_cfg: Mapping[str, Any] = artifacts_cfg_raw

    run_dir_raw = artifacts_cfg.get("run_dir")
    run_dir: Path | None = None
    if isinstance(run_dir_raw, str) and run_dir_raw.strip():
        run_dir = Path(run_dir_raw)

    return ConfidencePostOpPaths(
        gt_vs_pred_jsonl=_resolve_path(
            artifacts_cfg,
            key="gt_vs_pred_jsonl",
            run_dir=run_dir,
            default_name="gt_vs_pred.jsonl",
            required=True,
        ),
        pred_token_trace_jsonl=_resolve_path(
            artifacts_cfg,
            key="pred_token_trace_jsonl",
            run_dir=run_dir,
            default_name="pred_token_trace.jsonl",
            required=True,
        ),
        pred_confidence_jsonl=_resolve_path(
            artifacts_cfg,
            key="pred_confidence_jsonl",
            run_dir=run_dir,
            default_name="pred_confidence.jsonl",
            required=True,
        ),
        gt_vs_pred_scored_jsonl=_resolve_path(
            artifacts_cfg,
            key="gt_vs_pred_scored_jsonl",
            run_dir=run_dir,
            default_name="gt_vs_pred_scored.jsonl",
            required=True,
        ),
        confidence_postop_summary_json=_resolve_path(
            artifacts_cfg,
            key="confidence_postop_summary_json",
            run_dir=run_dir,
            default_name="confidence_postop_summary.json",
            required=True,
        ),
    )


def run_confidence_postop_from_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    return run_confidence_postop(paths_from_config(cfg))


__all__ = [
    "CONFIDENCE_METHOD",
    "ConfidencePostOpPaths",
    "FAILURE_REASONS",
    "PRED_SCORE_SOURCE",
    "PRED_SCORE_VERSION",
    "paths_from_config",
    "run_confidence_postop",
    "run_confidence_postop_from_config",
]
