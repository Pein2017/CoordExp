from types import SimpleNamespace

import pytest

from src.infer.backend import (
    extract_swift_choice_logprobs,
    normalize_vllm_trace_response,
    vllm_request_config_kwargs_from_decode_request,
)
from src.infer.backend_vllm_server import (
    effective_vllm_server_sync_mode,
    vllm_server_specs,
    vllm_server_timeouts,
)
from src.infer.runtime import build_decode_request_from_rollout_matching_config


class _NoopLogger:
    def warning(self, *_args, **_kwargs):
        return None


def test_vllm_server_backend_owns_config_normalization() -> None:
    owner = SimpleNamespace(
        _cfg=lambda key, default=None: {
            "server": {
                "servers": [
                    {
                        "base_url": "http://127.0.0.1:8000/",
                        "group_port": "51216",
                    }
                ],
                "timeout_s": "60",
                "infer_timeout_s": None,
            },
            "sync": {"mode": "adapter"},
        }
        if key == "vllm"
        else default
    )

    assert vllm_server_specs(owner) == [
        {"base_url": "http://127.0.0.1:8000", "group_port": 51216}
    ]
    assert vllm_server_timeouts(owner=owner, logger=_NoopLogger()) == (60.0, 60.0)
    assert effective_vllm_server_sync_mode(owner) == "adapter"


def test_vllm_request_config_enforces_return_details() -> None:
    request = build_decode_request_from_rollout_matching_config(
        {
            "rollout_backend": "vllm",
            "max_new_tokens": 16,
            "decoding": {"temperature": 0.0, "top_p": 1.0, "top_k": -1},
            "repetition_penalty": 1.0,
        }
    )
    cfg = vllm_request_config_kwargs_from_decode_request(request)

    assert cfg["return_details"] is True


def test_parse_vllm_server_output_requires_prompt_and_token_ids() -> None:
    raw = {
        "prompt_token_ids": [1, 2, 3],
        "choices": [{"message": {"content": "hi"}, "token_ids": [4, 5]}],
    }
    result = normalize_vllm_trace_response(
        raw,
        trace_logprobs=False,
        backend_mode="ms-swift",
    )
    assert result.generated_token_ids == [4, 5]
    assert result.text == "hi"
    assert result.prompt_token_ids == [1, 2, 3]


def test_parse_vllm_server_output_accepts_response_wrapper() -> None:
    raw = {
        "response": {
            "prompt_token_ids": [1],
            "choices": [{"message": {"content": "ok"}, "token_ids": [2]}],
        }
    }
    result = normalize_vllm_trace_response(
        raw,
        trace_logprobs=False,
        backend_mode="ms-swift",
    )
    assert result.generated_token_ids == [2]
    assert result.text == "ok"
    assert result.prompt_token_ids == [1]


def test_parse_vllm_server_output_raises_when_missing_prompt_token_ids() -> None:
    raw = {"choices": [{"message": {"content": "hi"}, "token_ids": [4, 5]}]}
    with pytest.raises(ValueError, match=r"prompt_token_ids"):
        normalize_vllm_trace_response(
            raw,
            trace_logprobs=False,
            backend_mode="ms-swift",
        )


def test_parse_vllm_server_output_rejects_openai_shape_without_return_details() -> None:
    raw = {"choices": [{"message": {"content": "hi"}}]}
    with pytest.raises(ValueError, match=r"prompt_token_ids|return_details"):
        normalize_vllm_trace_response(
            raw,
            trace_logprobs=False,
            backend_mode="ms-swift",
        )


def test_parse_vllm_server_output_raises_when_missing_token_ids() -> None:
    raw = {"prompt_token_ids": [1], "choices": [{"message": {"content": "hi"}}]}
    with pytest.raises(ValueError, match=r"token_ids"):
        normalize_vllm_trace_response(
            raw,
            trace_logprobs=False,
            backend_mode="ms-swift",
        )


def test_parse_vllm_server_output_traced_accepts_well_formed_trace() -> None:
    raw = {
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
        raw,
        trace_logprobs=True,
        backend_mode="ms-swift",
    )
    assert result.generated_token_ids == [101, 102]
    assert result.text == "ok"
    assert result.prompt_token_ids == [11, 12]
    assert result.generated_logprobs == pytest.approx([-0.1, -0.2])
    assert result.generated_tokens == ["a", "b"]


def test_parse_vllm_server_output_traced_rejects_longer_trace() -> None:
    raw = {
        "prompt_token_ids": [11],
        "choices": [
            {
                "message": {"content": "ok"},
                "token_ids": [101, 102],
                "logprobs": {
                    "content": [
                        {"token": "a", "logprob": -0.1},
                        {"token": "b", "logprob": -0.2},
                        {"token": "</s>", "logprob": -0.3},
                    ]
                },
            }
        ],
    }
    with pytest.raises(ValueError, match=r"trace shape"):
        normalize_vllm_trace_response(
            raw,
            trace_logprobs=True,
            backend_mode="ms-swift",
        )


def test_parse_vllm_server_output_traced_rejects_shorter_trace() -> None:
    raw = {
        "prompt_token_ids": [11],
        "choices": [
            {
                "message": {"content": "ok"},
                "token_ids": [101, 102],
                "logprobs": {
                    "content": [
                        {"token": "a", "logprob": -0.1},
                    ]
                },
            }
        ],
    }
    with pytest.raises(ValueError, match=r"trace shape"):
        normalize_vllm_trace_response(
            raw,
            trace_logprobs=True,
            backend_mode="ms-swift",
        )


def test_parse_vllm_server_output_traced_uses_token_id_frame_when_tokenizer_provided() -> None:
    class _ToyTokenizer:
        def decode(
            self,
            token_ids,
            *,
            skip_special_tokens: bool = False,
            clean_up_tokenization_spaces: bool = False,
        ) -> str:
            mapping = {
                101: "<|coord_1|>",
                102: "<|coord_2|>",
            }
            tid = int(token_ids[0])
            return mapping.get(tid, f"<|tok_{tid}|>")

    raw = {
        "prompt_token_ids": [11],
        "choices": [
            {
                "message": {"content": "ok"},
                "token_ids": [101, 102],
                "logprobs": {
                    "content": [
                        {"token": "not_coord_a", "logprob": -0.1},
                        {"token": "not_coord_b", "logprob": -0.2},
                    ]
                },
            }
        ],
    }
    result = normalize_vllm_trace_response(
        raw,
        trace_logprobs=True,
        backend_mode="ms-swift",
        tokenizer=_ToyTokenizer(),
    )
    assert result.generated_token_ids == [101, 102]
    assert result.text == "ok"
    assert result.prompt_token_ids == [11]
    assert result.generated_logprobs == pytest.approx([-0.1, -0.2])
    assert result.generated_tokens == ["<|coord_1|>", "<|coord_2|>"]


def test_extract_swift_choice_logprobs_rejects_non_finite_values() -> None:
    raw = {"content": [{"token": "a", "logprob": float("nan")}]}
    with pytest.raises(RuntimeError, match=r"non-finite"):
        extract_swift_choice_logprobs(raw)
