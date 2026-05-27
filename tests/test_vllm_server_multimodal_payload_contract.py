import json

import pytest

from src.infer.backend_vllm_server import build_vllm_server_infer_requests


def test_build_vllm_server_infer_requests_matches_swift_rollout_infer_request() -> None:
    # Import from ms-swift (CPU-only; no server required).
    from swift.infer_engine.protocol import RolloutInferRequest

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

    infer_requests = build_vllm_server_infer_requests(samples=samples)
    assert isinstance(infer_requests, list)
    assert infer_requests and isinstance(infer_requests[0], dict)

    assert infer_requests[0]["images"] == ["img.png"]
    json.dumps(infer_requests)

    parsed = RolloutInferRequest(**infer_requests[0])
    assert parsed.images == ["img.png"]


def test_build_vllm_server_infer_requests_rejects_non_string_images() -> None:
    with pytest.raises(ValueError, match=r"image entries"):
        build_vllm_server_infer_requests(
            samples=[
                {"messages": [{"role": "user", "content": "hi"}], "images": [123]}
            ],
        )


def test_build_vllm_server_infer_requests_requires_messages_list() -> None:
    with pytest.raises(ValueError, match=r"messages"):
        build_vllm_server_infer_requests(samples=[{"messages": "not-a-list"}])
