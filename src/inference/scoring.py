"""Selected-token scoring for compact object-box inference artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

from src.common.errors import ArtifactContractError
from src.inference.backend import TokenTrace
from src.templates.renderer import (
    BOX_END_TOKEN,
    BOX_START_TOKEN,
    OBJECT_REF_END_TOKEN,
    OBJECT_REF_START_TOKEN,
)


PRED_SCORE_VERSION = 1
SCORE_POLICY: dict[str, Any] = {
    "id": "compact-object-selected-token-score-v1",
    "formula": "exp(sum(selected_token_logprobs) / n_selected)",
    "selected_schema_wrappers": [
        OBJECT_REF_START_TOKEN,
        OBJECT_REF_END_TOKEN,
        BOX_START_TOKEN,
        BOX_END_TOKEN,
    ],
    "selected_coordinate_tokens": 4,
    "selected_token_count": 8,
    "excluded": ["description", "category"],
}
SCORE_POLICY_FINGERPRINT = hashlib.sha256(
    json.dumps(SCORE_POLICY, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()


@dataclass(frozen=True)
class ScoredPrediction:
    prediction: dict[str, Any]
    replay: dict[str, Any]
    score: float


def fingerprint_score_policy(policy: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(policy, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def score_prediction(
    *,
    row_id: str,
    prediction: dict[str, Any],
    token_trace: list[TokenTrace],
) -> ScoredPrediction:
    if not token_trace:
        raise ArtifactContractError(
            "cannot score prediction without generated token trace evidence",
            code="scoring.trace_alignment_missing",
            context={"row_id": row_id, "object_span_id": prediction.get("object_span_id")},
        )
    interval = _locate_object_interval(row_id=row_id, prediction=prediction, token_trace=token_trace)
    selected = _select_tokens(prediction, token_trace, interval=interval)
    if len(selected) != SCORE_POLICY["selected_token_count"]:
        raise ArtifactContractError(
            "compact object selected-token count does not match V1 score policy",
            code="scoring.selected_count_mismatch",
            context={
                "row_id": row_id,
                "object_span_id": prediction.get("object_span_id"),
                "selected_count": len(selected),
                "expected": SCORE_POLICY["selected_token_count"],
            },
        )
    selected_logprobs = [item["logprob"] for item in selected]
    if not selected_logprobs:
        raise ArtifactContractError(
            "selected-token set is empty",
            code="scoring.empty_selected_set",
            context={"row_id": row_id, "object_span_id": prediction.get("object_span_id")},
        )
    for index, logprob in enumerate(selected_logprobs):
        if not math.isfinite(logprob):
            raise ArtifactContractError(
                "selected-token logprob must be finite",
                code="scoring.non_finite_logprob",
                context={
                    "row_id": row_id,
                    "object_span_id": prediction.get("object_span_id"),
                    "selected_index": index,
                    "logprob": logprob,
                },
            )
        if logprob > 0.0:
            raise ArtifactContractError(
                "selected-token logprob must be a natural-log probability",
                code="scoring.positive_logprob",
                context={
                    "row_id": row_id,
                    "object_span_id": prediction.get("object_span_id"),
                    "selected_index": index,
                    "logprob": logprob,
                },
            )
    score = math.exp(sum(selected_logprobs) / len(selected_logprobs))
    if not math.isfinite(score) or score < 0.0 or score > 1.0:
        raise ArtifactContractError(
            "selected-token score must be finite and in [0.0, 1.0]",
            code="scoring.invalid_score",
            context={
                "row_id": row_id,
                "object_span_id": prediction.get("object_span_id"),
                "score": score,
            },
        )

    replay = {
        "row_id": row_id,
        "object_span_id": prediction.get("object_span_id"),
        "generated_step_indices": [item["step_index"] for item in selected],
        "token_ids": [item["token_id"] for item in selected],
        "token_text": [item["token_text"] for item in selected],
        "selected_logprobs": selected_logprobs,
        "selected_count": len(selected),
        "selected_tokens": selected,
        "score_policy_fingerprint": SCORE_POLICY_FINGERPRINT,
        "score": score,
    }
    scored_prediction = _artifact_prediction(prediction)
    scored_prediction["score"] = score
    scored_prediction["pred_score_version"] = PRED_SCORE_VERSION
    scored_prediction["pred_score_source"] = {
        "kind": "token_trace_selected_logprob_mean",
        "row_id": row_id,
        "object_span_id": prediction.get("object_span_id"),
        "generated_step_indices": replay["generated_step_indices"],
        "token_ids": replay["token_ids"],
        "token_text": replay["token_text"],
        "selected_logprobs": selected_logprobs,
        "selected_count": len(selected),
        "score_policy_fingerprint": SCORE_POLICY_FINGERPRINT,
    }
    return ScoredPrediction(
        prediction=scored_prediction,
        replay=replay,
        score=score,
    )


def _locate_object_interval(
    *,
    row_id: str,
    prediction: dict[str, Any],
    token_trace: list[TokenTrace],
) -> tuple[int, int, int]:
    raw_span_text = str(prediction.get("raw_span_text") or "")
    if not raw_span_text:
        raise ArtifactContractError(
            "parser prediction lacks raw span evidence for scoring",
            code="scoring.missing_span_evidence",
            context={"row_id": row_id, "object_span_id": prediction.get("object_span_id")},
        )
    expected_sha = prediction.get("raw_span_sha256")
    if expected_sha is not None and expected_sha != _sha256_text(raw_span_text):
        raise ArtifactContractError(
            "parser raw span sha does not match raw span text",
            code="scoring.raw_span_sha_mismatch",
            context={"row_id": row_id, "object_span_id": prediction.get("object_span_id")},
        )
    matches: list[tuple[int, int, int]] = []
    token_count = len(token_trace)
    for start in range(token_count):
        joined = ""
        char_start = sum(len(item.token_text) for item in token_trace[:start])
        for end in range(start, token_count):
            joined += token_trace[end].token_text
            if joined == raw_span_text:
                matches.append((start, end + 1, char_start))
                break
            if not raw_span_text.startswith(joined):
                break
    matches = _filter_matches_by_absolute_span(prediction, matches, raw_span_text=raw_span_text)
    if not matches:
        generated_text = "".join(item.token_text for item in token_trace)
        if raw_span_text in generated_text or _span_tokens_present_in_order(
            prediction,
            generated_text=generated_text,
        ):
            code = "scoring.object_span_not_contiguous"
        else:
            code = "scoring.trace_alignment_missing"
        raise ArtifactContractError(
            "parser object span does not map to one contiguous generated-token interval",
            code=code,
            context={
                "object_span_id": prediction.get("object_span_id"),
                "row_id": row_id,
            },
        )
    if len(matches) != 1:
        raise ArtifactContractError(
            "parser object span maps to multiple generated-token intervals",
            code="scoring.trace_alignment_ambiguous",
            context={
                "row_id": row_id,
                "object_span_id": prediction.get("object_span_id"),
                "match_count": len(matches),
            },
        )
    return matches[0]


def _filter_matches_by_absolute_span(
    prediction: dict[str, Any],
    matches: list[tuple[int, int, int]],
    *,
    raw_span_text: str,
) -> list[tuple[int, int, int]]:
    if "char_start" not in prediction or "char_end" not in prediction:
        return matches
    expected_start = int(prediction["char_start"])
    expected_end = int(prediction["char_end"])
    if expected_end - expected_start != len(raw_span_text):
        return []
    return [
        match
        for match in matches
        if match[2] == expected_start and match[2] + len(raw_span_text) == expected_end
    ]


def _span_tokens_present_in_order(
    prediction: dict[str, Any],
    *,
    generated_text: str,
) -> bool:
    cursor = 0
    spans = sorted(
        list(prediction.get("schema_spans") or [])
        + list(prediction.get("coord_token_spans") or []),
        key=lambda item: int(item["char_start"]),
    )
    if not spans:
        return False
    for span in spans:
        token_text = str(span.get("text") or "")
        position = generated_text.find(token_text, cursor)
        if position < 0:
            return False
        cursor = position + len(token_text)
    return True


def _select_tokens(
    prediction: dict[str, Any],
    token_trace: list[TokenTrace],
    *,
    interval: tuple[int, int, int],
) -> list[dict[str, Any]]:
    start_token, end_token, span_char_start = interval
    token_ranges = _token_char_ranges(token_trace, start_token, end_token)
    selected_spans = sorted(
        list(prediction.get("schema_spans") or [])
        + list(prediction.get("coord_token_spans") or []),
        key=lambda item: int(item["char_start"]),
    )
    selected: list[dict[str, Any]] = []
    for span in selected_spans:
        trace = _trace_for_span(
            span,
            token_trace=token_trace,
            token_ranges=token_ranges,
            span_char_start=span_char_start,
            object_span_id=str(prediction.get("object_span_id")),
        )
        if trace.logprob is None:
            raise ArtifactContractError(
                "selected generated token lacks logprob evidence",
                code="scoring.missing_logprob",
                context={
                    "object_span_id": prediction.get("object_span_id"),
                    "token_text": trace.token_text,
                },
            )
        selected.append(
            {
                "generated_step_index": trace.step_index,
                "step_index": trace.step_index,
                "token_id": trace.token_id,
                "token_text": trace.token_text,
                "logprob": float(trace.logprob),
                "char_start": int(span["char_start"]),
                "char_end": int(span["char_end"]),
            }
        )
    return selected


def _token_char_ranges(
    token_trace: list[TokenTrace],
    start_token: int,
    end_token: int,
) -> list[tuple[int, int, int]]:
    ranges: list[tuple[int, int, int]] = []
    cursor = 0
    for index, trace in enumerate(token_trace):
        next_cursor = cursor + len(trace.token_text)
        if start_token <= index < end_token:
            ranges.append((index, cursor, next_cursor))
        cursor = next_cursor
    return ranges


def _trace_for_span(
    span: dict[str, Any],
    *,
    token_trace: list[TokenTrace],
    token_ranges: list[tuple[int, int, int]],
    span_char_start: int,
    object_span_id: str,
) -> TokenTrace:
    char_start = int(span["char_start"])
    char_end = int(span["char_end"])
    matches = [
        token_trace[index]
        for index, token_start, token_end in token_ranges
        if token_start == char_start and token_end == char_end
    ]
    if len(matches) != 1:
        raise ArtifactContractError(
            "selected parser span does not align to exactly one generated token",
            code="scoring.selected_span_alignment_failed",
            context={
                "object_span_id": object_span_id,
                "span_text": span.get("text"),
                "span_char_start": char_start,
                "span_char_end": char_end,
                "object_char_start": span_char_start,
                "match_count": len(matches),
            },
        )
    trace = matches[0]
    if trace.token_text != span.get("text"):
        raise ArtifactContractError(
            "selected parser span text disagrees with token trace text",
            code="scoring.selected_span_text_mismatch",
            context={
                "object_span_id": object_span_id,
                "span_text": span.get("text"),
                "token_text": trace.token_text,
            },
        )
    return trace


def _artifact_prediction(prediction: dict[str, Any]) -> dict[str, Any]:
    keep = {
        "object_span_id",
        "description",
        "bbox",
        "bbox_format",
        "coord_bins",
        "generated_order",
    }
    return {key: prediction[key] for key in sorted(keep) if key in prediction}


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
