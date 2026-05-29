from __future__ import annotations

import unittest

from src.infer.prompt import (
    ensure_system_prompt_message,
    force_last_user_prompt_text,
    prepare_rollout_prompt_samples,
    require_verified_prompt_token_parity,
    rollout_visual_metadata_from_sample,
    strip_trailing_assistant_turns_for_rollout,
)


def test_prompt_token_parity_guard_accepts_exact_match() -> None:
    result = require_verified_prompt_token_parity(
        local_prompt_token_ids=[1, 2, 3],
        backend_prompt_token_ids=[1, 2, 3],
        expected_prompt_len=3,
        context="stage2",
    )

    assert result.verified is True
    assert result.prompt_token_parity == "verified"


def test_prompt_token_parity_guard_rejects_token_mismatch() -> None:
    with unittest.TestCase().assertRaisesRegex(
        ValueError,
        "prompt token IDs differ",
    ):
        require_verified_prompt_token_parity(
            local_prompt_token_ids=[1, 2, 3],
            backend_prompt_token_ids=[1, 9, 3],
            expected_prompt_len=3,
            context="stage2",
        )


def test_prompt_token_parity_guard_rejects_length_mismatch() -> None:
    with unittest.TestCase().assertRaisesRegex(
        ValueError,
        "prompt_len mismatch",
    ):
        require_verified_prompt_token_parity(
            local_prompt_token_ids=[1, 2, 3],
            backend_prompt_token_ids=[1, 2, 3],
            expected_prompt_len=4,
            context="stage2",
        )


def test_prompt_token_parity_guard_rejects_missing_backend_ids_when_required() -> None:
    with unittest.TestCase().assertRaisesRegex(
        ValueError,
        "backend prompt token IDs are missing",
    ):
        require_verified_prompt_token_parity(
            local_prompt_token_ids=[1, 2, 3],
            backend_prompt_token_ids=[],
            expected_prompt_len=3,
            context="stage2",
            require_backend_prompt_ids=True,
        )


def test_rollout_prompt_helpers_trim_teacher_forced_assistant_tail() -> None:
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "context"},
        {"role": "user", "content": "detect"},
        {"role": "assistant", "content": "{\"objects\": []}"},
    ]

    assert strip_trailing_assistant_turns_for_rollout(messages) == messages[:4]


def test_rollout_prompt_helpers_inject_plain_system_message_once() -> None:
    messages = [{"role": "user", "content": "detect"}]

    out = ensure_system_prompt_message(messages, "SYS")

    assert out == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "detect"},
    ]
    assert ensure_system_prompt_message(out, "OTHER") == out


def test_rollout_prompt_helpers_replace_user_text_without_dropping_image() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "a.png"},
                {"type": "text", "text": "old"},
                {"type": "text", "text": "duplicate"},
            ],
        }
    ]

    out = force_last_user_prompt_text(messages, "new")

    assert out == [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "a.png"},
                {"type": "text", "text": "new"},
            ],
        }
    ]
    assert messages[0]["content"][1]["text"] == "old"


def test_prepare_rollout_prompt_samples_rebuilds_eval_prompt_and_visual_metadata() -> None:
    sample = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "img.jpg"},
                    {"type": "text", "text": "old prompt"},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "{}"}]},
        ],
        "images": ("img.jpg",),
        "width": 12,
        "height": 8,
    }

    out = prepare_rollout_prompt_samples(
        [sample],
        rollout_backend="vllm",
        prompt_variant_override="coco_80",
        training_prompt_variant="default",
        object_ordering="sorted",
        object_field_order="geometry_first",
        template_system=None,
    )

    assert len(out) == 1
    prepared = out[0]
    assert prepared is not sample
    assert prepared["images"] == ["img.jpg"]
    assert prepared["_coordexp_prompt_visual_metadata"] == {
        "image_count": 1,
        "image_path": "img.jpg",
        "image_placement": "message_content",
        "do_resize": False,
        "original_width": 12,
        "original_height": 8,
        "post_preprocessing_width": 12,
        "post_preprocessing_height": 8,
    }
    messages = prepared["messages"]
    assert messages[0]["role"] == "system"
    assert messages[-1]["role"] == "user"
    assert all(message.get("role") != "assistant" for message in messages)
    user_texts = [
        part.get("text")
        for part in messages[-1]["content"]
        if isinstance(part, dict) and part.get("type") == "text"
    ]
    assert len(user_texts) == 1
    assert user_texts[0] != "old prompt"
    assert "COCO" in user_texts[0]


def test_prepare_rollout_prompt_samples_uses_default_variant_when_compact_full_train_variant_missing() -> None:
    sample = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "img.jpg"},
                    {"type": "text", "text": "old prompt"},
                ],
            }
        ],
        "images": ["img.jpg"],
        "width": 12,
        "height": 8,
    }

    out = prepare_rollout_prompt_samples(
        [sample],
        rollout_backend="hf",
        prompt_variant_override=None,
        detection_sequence_format="compact_full",
        training_prompt_variant=None,
        object_ordering="sorted",
        object_field_order="desc_first",
        template_system=None,
    )

    assert len(out) == 1
    user_content = out[0]["messages"][-1]["content"]
    user_texts = [
        part.get("text")
        for part in user_content
        if isinstance(part, dict) and part.get("type") == "text"
    ]
    assert len(user_texts) == 1
    assert user_texts[0] != "old prompt"


def test_rollout_visual_metadata_accepts_message_image_when_side_channel_empty() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "img.jpg"},
                {"type": "text", "text": "detect"},
            ],
        }
    ]
    sample = {
        "messages": messages,
        "images": [],
        "width": 12,
        "height": 8,
    }

    assert rollout_visual_metadata_from_sample(sample, messages=messages) == {
        "image_count": 1,
        "image_path": "img.jpg",
        "image_placement": "message_content",
        "do_resize": False,
        "original_width": 12,
        "original_height": 8,
        "post_preprocessing_width": 12,
        "post_preprocessing_height": 8,
    }


def test_prepare_rollout_prompt_samples_stamps_visual_metadata_from_message_image_when_side_channel_empty() -> None:
    sample = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "img.jpg"},
                    {"type": "text", "text": "old prompt"},
                ],
            }
        ],
        "images": [],
        "width": 12,
        "height": 8,
    }

    out = prepare_rollout_prompt_samples(
        [sample],
        rollout_backend="vllm",
        prompt_variant_override="coco_80",
        training_prompt_variant="default",
        detection_sequence_format="compact_full",
        object_ordering="sorted",
        object_field_order="desc_first",
        template_system=None,
    )

    assert out[0]["images"] == []
    assert out[0]["_coordexp_prompt_visual_metadata"] == {
        "image_count": 1,
        "image_path": "img.jpg",
        "image_placement": "message_content",
        "do_resize": False,
        "original_width": 12,
        "original_height": 8,
        "post_preprocessing_width": 12,
        "post_preprocessing_height": 8,
    }
