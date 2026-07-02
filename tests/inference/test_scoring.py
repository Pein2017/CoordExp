from __future__ import annotations

import math

import pytest

from src.common.errors import ArtifactContractError
from src.inference.backend import TokenTrace


def _trace_for_text(text: str, *, logprob: float = math.log(0.25)) -> list[TokenTrace]:
    pieces = [
        "<|object_ref_start|>",
        "cat",
        "<|object_ref_end|>",
        "<|box_start|>",
        "<|coord_100|>",
        "<|coord_200|>",
        "<|coord_300|>",
        "<|coord_400|>",
        "<|box_end|>",
    ]
    assert "".join(pieces) == text
    return [
        TokenTrace(
            step_index=index,
            token_id=151646 + index,
            token_text=piece,
            logprob=logprob,
            is_stop=False,
            is_pad=False,
            backend="hf",
            backend_mode="generate",
            response_family="hf",
        )
        for index, piece in enumerate(pieces)
    ]


def _prediction(text: str) -> dict:
    from src.inference.parsing import parse_compact_object_box_closed

    return parse_compact_object_box_closed(
        text,
        row_id="row-1",
        row_index=0,
        image_width=1000,
        image_height=1000,
    ).predictions[0]


def test_score_formula_uses_exp_mean_selected_token_logprobs() -> None:
    from src.inference.scoring import score_prediction

    text = (
        "<|object_ref_start|>cat<|object_ref_end|>"
        "<|box_start|><|coord_100|><|coord_200|><|coord_300|><|coord_400|><|box_end|>"
    )
    logprob = math.log(0.2)

    scored = score_prediction(
        row_id="row-1",
        prediction=_prediction(text),
        token_trace=_trace_for_text(text, logprob=logprob),
    )

    assert scored.score == pytest.approx(0.2)
    assert scored.replay["selected_count"] == 8
    assert scored.prediction["score"] == pytest.approx(0.2)


def test_compact_object_policy_selects_exactly_four_wrappers_and_four_coordinates() -> None:
    from src.inference.scoring import score_prediction

    text = (
        "<|object_ref_start|>cat<|object_ref_end|>"
        "<|box_start|><|coord_100|><|coord_200|><|coord_300|><|coord_400|><|box_end|>"
    )

    scored = score_prediction(
        row_id="row-1",
        prediction=_prediction(text),
        token_trace=_trace_for_text(text),
    )

    assert [item["token_text"] for item in scored.replay["selected_tokens"]] == [
        "<|object_ref_start|>",
        "<|object_ref_end|>",
        "<|box_start|>",
        "<|coord_100|>",
        "<|coord_200|>",
        "<|coord_300|>",
        "<|coord_400|>",
        "<|box_end|>",
    ]
    assert scored.prediction["pred_score_source"]["selected_count"] == 8


def test_selected_token_replay_evidence_is_persisted() -> None:
    from src.inference.scoring import SCORE_POLICY_FINGERPRINT, score_prediction

    text = (
        "<|object_ref_start|>cat<|object_ref_end|>"
        "<|box_start|><|coord_100|><|coord_200|><|coord_300|><|coord_400|><|box_end|>"
    )

    scored = score_prediction(
        row_id="row-1",
        prediction=_prediction(text),
        token_trace=_trace_for_text(text),
    )

    assert scored.replay["row_id"] == "row-1"
    assert scored.replay["object_span_id"] == "row-1:span-0"
    assert scored.replay["generated_step_indices"] == [0, 2, 3, 4, 5, 6, 7, 8]
    assert scored.replay["token_ids"] == [151646, 151648, 151649, 151650, 151651, 151652, 151653, 151654]
    assert scored.replay["selected_logprobs"] == [pytest.approx(math.log(0.25))] * 8
    assert scored.replay["score_policy_fingerprint"] == SCORE_POLICY_FINGERPRINT
    assert scored.prediction["pred_score_source"]["object_span_id"] == "row-1:span-0"


def test_description_tokens_are_excluded_from_score() -> None:
    from src.inference.scoring import score_prediction

    text = (
        "<|object_ref_start|>cat<|object_ref_end|>"
        "<|box_start|><|coord_100|><|coord_200|><|coord_300|><|coord_400|><|box_end|>"
    )
    trace = _trace_for_text(text, logprob=math.log(0.5))
    trace[1] = TokenTrace(**{**trace[1].__dict__, "logprob": math.log(0.01)})

    scored = score_prediction(
        row_id="row-1",
        prediction=_prediction(text),
        token_trace=trace,
    )

    assert scored.score == pytest.approx(0.5)
    assert "cat" not in [item["token_text"] for item in scored.replay["selected_tokens"]]


def test_empty_selected_token_set_invalidates_prediction() -> None:
    from src.inference.scoring import score_prediction

    text = (
        "<|object_ref_start|>cat<|object_ref_end|>"
        "<|box_start|><|coord_100|><|coord_200|><|coord_300|><|coord_400|><|box_end|>"
    )

    with pytest.raises(ArtifactContractError) as exc_info:
        score_prediction(row_id="row-1", prediction=_prediction(text), token_trace=[])

    assert exc_info.value.code == "scoring.trace_alignment_missing"


def test_non_finite_selected_logprob_invalidates_prediction() -> None:
    from src.inference.scoring import score_prediction

    text = (
        "<|object_ref_start|>cat<|object_ref_end|>"
        "<|box_start|><|coord_100|><|coord_200|><|coord_300|><|coord_400|><|box_end|>"
    )
    trace = _trace_for_text(text)
    trace[4] = TokenTrace(**{**trace[4].__dict__, "logprob": float("nan")})

    with pytest.raises(ArtifactContractError) as exc_info:
        score_prediction(row_id="row-1", prediction=_prediction(text), token_trace=trace)

    assert exc_info.value.code == "scoring.non_finite_logprob"


def test_duplicate_span_alignment_is_ambiguous_without_unique_trace_interval() -> None:
    from src.inference.scoring import score_prediction

    object_text = (
        "<|object_ref_start|>cat<|object_ref_end|>"
        "<|box_start|><|coord_100|><|coord_200|><|coord_300|><|coord_400|><|box_end|>"
    )
    full_text = object_text + object_text
    prediction = _prediction(object_text)
    trace = _trace_for_text(object_text) + [
        TokenTrace(**{**item.__dict__, "step_index": item.step_index + 9})
        for item in _trace_for_text(object_text)
    ]

    assert "".join(item.token_text for item in trace) == full_text
    with pytest.raises(ArtifactContractError) as exc_info:
        score_prediction(row_id="row-1", prediction=prediction, token_trace=trace)

    assert exc_info.value.code == "scoring.trace_alignment_ambiguous"


def test_object_span_must_map_to_contiguous_generated_token_interval() -> None:
    from src.inference.scoring import score_prediction

    text = (
        "<|object_ref_start|>cat<|object_ref_end|>"
        "<|box_start|><|coord_100|><|coord_200|><|coord_300|><|coord_400|><|box_end|>"
    )
    trace = _trace_for_text(text)
    trace = trace[:4] + [
        TokenTrace(
            step_index=99,
            token_id=999,
            token_text="noise",
            logprob=math.log(0.5),
            is_stop=False,
            is_pad=False,
            backend="hf",
            backend_mode="generate",
            response_family="hf",
        )
    ] + trace[4:]

    with pytest.raises(ArtifactContractError) as exc_info:
        score_prediction(row_id="row-1", prediction=_prediction(text), token_trace=trace)

    assert exc_info.value.code == "scoring.object_span_not_contiguous"


def test_score_policy_fingerprint_is_stable_and_content_addressed() -> None:
    from src.inference.scoring import SCORE_POLICY, SCORE_POLICY_FINGERPRINT, fingerprint_score_policy

    assert SCORE_POLICY["selected_token_count"] == 8
    assert fingerprint_score_policy(SCORE_POLICY) == SCORE_POLICY_FINGERPRINT
    assert len(SCORE_POLICY_FINGERPRINT) == 64
