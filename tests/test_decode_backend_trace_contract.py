from __future__ import annotations

import math

import pytest


def test_decode_result_rejects_shape_mismatch_without_clipping() -> None:
    from src.infer.backend import DetectionDecodeResult, validate_decode_trace

    result = DetectionDecodeResult(
        text="abc",
        generated_token_ids=[1, 2, 3],
        generated_tokens=["a", "b", "c"],
        generated_logprobs=[-0.1, -0.2],
        stop_reason="length",
        backend="fake",
    )

    with pytest.raises(ValueError, match="trace shape"):
        validate_decode_trace(result, trace_logprobs=True)


def test_decode_result_rejects_missing_token_ids_for_nonempty_text() -> None:
    from src.infer.backend import DetectionDecodeResult, validate_decode_trace

    result = DetectionDecodeResult(
        text="abc",
        generated_token_ids=[],
        generated_tokens=[],
        generated_logprobs=[],
        stop_reason="length",
        backend="fake",
    )

    with pytest.raises(ValueError, match="generated_token_ids"):
        validate_decode_trace(result, trace_logprobs=True)


def test_decode_result_rejects_missing_token_text() -> None:
    from src.infer.backend import DetectionDecodeResult, validate_decode_trace

    result = DetectionDecodeResult(
        text="abc",
        generated_token_ids=[1, 2, 3],
        generated_tokens=None,
        generated_logprobs=[-0.1, -0.2, -0.3],
        stop_reason="length",
        backend="fake",
    )

    with pytest.raises(ValueError, match="generated_tokens"):
        validate_decode_trace(result, trace_logprobs=True)


def test_decode_result_rejects_missing_logprobs() -> None:
    from src.infer.backend import DetectionDecodeResult, validate_decode_trace

    result = DetectionDecodeResult(
        text="abc",
        generated_token_ids=[1, 2, 3],
        generated_tokens=["a", "b", "c"],
        generated_logprobs=None,
        stop_reason="length",
        backend="fake",
    )

    with pytest.raises(ValueError, match="generated_logprobs"):
        validate_decode_trace(result, trace_logprobs=True)


@pytest.mark.parametrize("bad_logprob", [math.inf, -math.inf, math.nan])
def test_decode_result_rejects_nonfinite_logprobs(bad_logprob: float) -> None:
    from src.infer.backend import DetectionDecodeResult, validate_decode_trace

    result = DetectionDecodeResult(
        text="abc",
        generated_token_ids=[1, 2, 3],
        generated_tokens=["a", "b", "c"],
        generated_logprobs=[-0.1, bad_logprob, -0.3],
        stop_reason="length",
        backend="fake",
    )

    with pytest.raises(ValueError, match="finite"):
        validate_decode_trace(result, trace_logprobs=True)


def test_decode_result_accepts_complete_aligned_trace() -> None:
    from src.infer.backend import DetectionDecodeResult, validate_decode_trace

    result = DetectionDecodeResult(
        text="abc",
        generated_token_ids=[1, 2, 3],
        generated_tokens=["a", "b", "c"],
        generated_logprobs=[-0.1, -0.2, -0.3],
        stop_reason="length",
        backend="fake",
    )

    assert validate_decode_trace(result, trace_logprobs=True) is result


def test_decode_result_accepts_empty_trace_for_empty_text() -> None:
    from src.infer.backend import DetectionDecodeResult, validate_decode_trace

    result = DetectionDecodeResult(
        text="",
        generated_token_ids=[],
        generated_tokens=[],
        generated_logprobs=[],
        stop_reason="stop",
        backend="fake",
    )

    assert validate_decode_trace(result, trace_logprobs=True) is result


def test_decode_result_does_not_require_trace_when_disabled() -> None:
    from src.infer.backend import DetectionDecodeResult, validate_decode_trace

    result = DetectionDecodeResult(
        text="abc",
        generated_token_ids=None,
        generated_tokens=None,
        generated_logprobs=None,
        stop_reason="length",
        backend="fake",
    )

    assert validate_decode_trace(result, trace_logprobs=False) is result


def test_openai_compatible_vllm_trace_without_token_ids_fails() -> None:
    from src.infer.backend import normalize_vllm_trace_response

    response = {
        "choices": [
            {
                "message": {"content": "abc"},
                "finish_reason": "stop",
                "logprobs": {"content": [{"token": "a", "logprob": -0.1}]},
            }
        ]
    }

    with pytest.raises(ValueError, match="generated_token_ids"):
        normalize_vllm_trace_response(
            response,
            trace_logprobs=True,
            backend_mode="openai-compatible",
        )


def test_openai_compatible_vllm_trace_with_token_ids_normalizes() -> None:
    from src.infer.backend import normalize_vllm_trace_response

    response = {
        "prompt_token_ids": [11, 12],
        "choices": [
            {
                "message": {"content": "ab"},
                "finish_reason": "stop",
                "token_ids": [101, 102],
                "logprobs": {
                    "content": [
                        {"token": "a", "logprob": -0.1},
                        {"token": "b", "logprob": -0.2},
                    ]
                },
            }
        ],
    }

    result = normalize_vllm_trace_response(
        response,
        trace_logprobs=True,
        backend_mode="openai-compatible",
    )

    assert result.backend == "vllm"
    assert result.backend_metadata["response_family"] == "openai-compatible"
    assert result.prompt_token_ids == [11, 12]
    assert result.generated_token_ids == [101, 102]
    assert result.generated_tokens == ["a", "b"]
    assert result.generated_logprobs == [-0.1, -0.2]


def test_openai_compatible_vllm_rejects_empty_choices() -> None:
    from src.infer.backend import normalize_vllm_trace_response

    with pytest.raises(ValueError, match="at least one choice"):
        normalize_vllm_trace_response(
            {"choices": []},
            trace_logprobs=False,
            backend_mode="openai-compatible",
        )


def test_openai_compatible_vllm_concatenates_text_content_parts() -> None:
    from src.infer.backend import normalize_vllm_trace_response

    result = normalize_vllm_trace_response(
        {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": "a"},
                            {"type": "image_url", "image_url": {"url": "ignored"}},
                            {"type": "text", "text": "b"},
                        ]
                    }
                }
            ]
        },
        trace_logprobs=False,
        backend_mode="openai-compatible",
    )

    assert result.text == "ab"


def test_ms_swift_vllm_details_normalize_complete_trace() -> None:
    from src.infer.backend import normalize_vllm_trace_response

    response = {
        "text": "abc",
        "details": {
            "prompt_token_ids": [11, 12],
            "token_ids": [1, 2, 3],
            "tokens": ["a", "b", "c"],
            "logprobs": [-0.1, -0.2, -0.3],
            "stop_reason": "length",
        },
    }

    result = normalize_vllm_trace_response(
        response,
        trace_logprobs=True,
        backend_mode="ms-swift",
    )

    assert result.backend == "vllm"
    assert result.prompt_token_ids == [11, 12]
    assert result.generated_token_ids == [1, 2, 3]
    assert result.generated_tokens == ["a", "b", "c"]
    assert result.generated_logprobs == [-0.1, -0.2, -0.3]


def test_ms_swift_vllm_choices_normalize_complete_trace() -> None:
    from src.infer.backend import normalize_vllm_trace_response

    response = {
        "prompt_token_ids": [11, 12],
        "choices": [
            {
                "message": {"content": "ok"},
                "token_ids": [101, 102],
                "logprobs": {
                    "content": [
                        {"token": "a", "logprob": -0.1},
                        {"token": "b", "logprob": -0.2},
                    ]
                },
            }
        ],
    }

    result = normalize_vllm_trace_response(
        response,
        trace_logprobs=True,
        backend_mode="ms-swift",
    )

    assert result.backend == "vllm"
    assert result.prompt_token_ids == [11, 12]
    assert result.generated_token_ids == [101, 102]
    assert result.generated_tokens == ["a", "b"]
    assert result.generated_logprobs == [-0.1, -0.2]


def test_ms_swift_vllm_details_missing_generated_logprobs_fail() -> None:
    from src.infer.backend import normalize_vllm_trace_response

    response = {
        "text": "abc",
        "details": {
            "prompt_token_ids": [11, 12],
            "token_ids": [1, 2, 3],
            "tokens": ["a", "b", "c"],
        },
    }

    with pytest.raises(ValueError, match="generated_logprobs"):
        normalize_vllm_trace_response(
            response,
            trace_logprobs=True,
            backend_mode="ms-swift",
        )


def test_ms_swift_vllm_details_missing_prompt_token_ids_fail() -> None:
    from src.infer.backend import normalize_vllm_trace_response

    response = {
        "text": "abc",
        "details": {
            "token_ids": [1, 2, 3],
            "tokens": ["a", "b", "c"],
            "logprobs": [-0.1, -0.2, -0.3],
        },
    }

    with pytest.raises(ValueError, match="prompt_token_ids"):
        normalize_vllm_trace_response(
            response,
            trace_logprobs=True,
            backend_mode="ms-swift",
        )


@pytest.mark.parametrize(
    "logprobs",
    [
        [-0.1, -0.2],
        [-0.1, -0.2, -0.3, -0.4],
    ],
)
def test_ms_swift_vllm_details_short_or_padded_logprobs_fail(
    logprobs: list[float],
) -> None:
    from src.infer.backend import normalize_vllm_trace_response

    response = {
        "text": "abc",
        "details": {
            "prompt_token_ids": [11, 12],
            "token_ids": [1, 2, 3],
            "tokens": ["a", "b", "c"],
            "logprobs": logprobs,
        },
    }

    with pytest.raises(ValueError, match="trace shape"):
        normalize_vllm_trace_response(
            response,
            trace_logprobs=True,
            backend_mode="ms-swift",
        )


def test_ms_swift_vllm_details_clipped_tokens_fail() -> None:
    from src.infer.backend import normalize_vllm_trace_response

    response = {
        "text": "abc",
        "details": {
            "prompt_token_ids": [11, 12],
            "token_ids": [1, 2, 3],
            "tokens": ["a", "b"],
            "logprobs": [-0.1, -0.2, -0.3],
        },
    }

    with pytest.raises(ValueError, match="trace shape"):
        normalize_vllm_trace_response(
            response,
            trace_logprobs=True,
            backend_mode="ms-swift",
        )
