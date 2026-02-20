import json

import pytest

from src.trainers.rollout_matching_sft import RolloutMatchingSFTTrainer


def test_build_vllm_server_infer_requests_matches_swift_rollout_infer_request() -> None:
    # Import from ms-swift (CPU-only; no server required).
    from swift.llm.template.template_inputs import RolloutInferRequest

    t = object.__new__(RolloutMatchingSFTTrainer)

    samples = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "img.png"},
                        {"type": "text", "text": "describe"},
                    ],
                }
            ],
            # Accept tuple input; normalize to list[str].
            "images": ("img.png",),
        }
    ]

    infer_requests = t._build_vllm_server_infer_requests(samples)
    assert isinstance(infer_requests, list)
    assert infer_requests and isinstance(infer_requests[0], dict)

    assert infer_requests[0]["images"] == ["img.png"]
    json.dumps(infer_requests)

    parsed = RolloutInferRequest(**infer_requests[0])
    assert parsed.images == ["img.png"]


def test_build_vllm_server_infer_requests_rejects_non_string_images() -> None:
    t = object.__new__(RolloutMatchingSFTTrainer)

    with pytest.raises(ValueError, match=r"image entries"):
        t._build_vllm_server_infer_requests(
            [
                {
                    "messages": [{"role": "user", "content": "hi"}],
                    "images": [123],
                }
            ]
        )


def test_build_vllm_server_infer_requests_requires_messages_list() -> None:
    t = object.__new__(RolloutMatchingSFTTrainer)

    with pytest.raises(ValueError, match=r"messages"):
        t._build_vllm_server_infer_requests([{"messages": "not-a-list"}])
