from __future__ import annotations

from src.analysis.rollout_parity import build_stage2_vllm_sample


def test_build_stage2_vllm_sample_inserts_system_and_strips_assistant() -> None:
    record = {
        "images": ["images/demo.jpg"],
        "width": 1000,
        "height": 1000,
        "objects": [],
    }
    sample = build_stage2_vllm_sample(
        record,
        line_idx=7,
        prompt_variant="coco_80",
        object_field_order="desc_first",
    )
    assert sample.line_idx == 7
    assert sample.image == "images/demo.jpg"
    assert sample.messages[0]["role"] == "system"
    assert sample.messages[-1]["role"] == "user"
    assert all(message.get("role") != "assistant" for message in sample.messages)
