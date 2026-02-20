import pytest

from src.trainers.rollout_matching_sft import RolloutMatchingSFTTrainer


def test_vllm_request_config_enforces_return_details() -> None:
    cfg = RolloutMatchingSFTTrainer._rollout_vllm_request_config_kwargs(
        max_tokens=16,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        repetition_penalty=1.0,
    )
    assert cfg["return_details"] is True


def test_parse_vllm_server_output_requires_prompt_and_token_ids() -> None:
    raw = {
        "prompt_token_ids": [1, 2, 3],
        "choices": [{"message": {"content": "hi"}, "token_ids": [4, 5]}],
    }
    token_ids, text, prompt_ids = RolloutMatchingSFTTrainer._parse_vllm_server_output(raw)
    assert token_ids == [4, 5]
    assert text == "hi"
    assert prompt_ids == [1, 2, 3]


def test_parse_vllm_server_output_accepts_response_wrapper() -> None:
    raw = {
        "response": {
            "prompt_token_ids": [1],
            "choices": [{"message": {"content": "ok"}, "token_ids": [2]}],
        }
    }
    token_ids, text, prompt_ids = RolloutMatchingSFTTrainer._parse_vllm_server_output(raw)
    assert token_ids == [2]
    assert text == "ok"
    assert prompt_ids == [1]


def test_parse_vllm_server_output_raises_when_missing_prompt_token_ids() -> None:
    raw = {"choices": [{"message": {"content": "hi"}, "token_ids": [4, 5]}]}
    with pytest.raises(RuntimeError, match=r"prompt_token_ids"):
        RolloutMatchingSFTTrainer._parse_vllm_server_output(raw)


def test_parse_vllm_server_output_raises_when_missing_token_ids() -> None:
    raw = {"prompt_token_ids": [1], "choices": [{"message": {"content": "hi"}}]}
    with pytest.raises(RuntimeError, match=r"token_ids"):
        RolloutMatchingSFTTrainer._parse_vllm_server_output(raw)
