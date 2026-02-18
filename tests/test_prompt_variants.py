from __future__ import annotations

import pytest
from PIL import Image

from src.config.loader import ConfigLoader
from src.config.prompt_variants import (
    COCO_80_CLASS_LIST_COMPACT,
    COCO_80_CLASS_NAMES,
)
from src.config.prompts import (
    SYSTEM_PROMPT_SUMMARY,
    USER_PROMPT_SUMMARY,
    get_template_prompts,
)
from src.infer.engine import GenerationConfig, InferenceConfig, InferenceEngine


def test_prompt_variant_default_fallback_matches_explicit_default() -> None:
    system_implicit, user_implicit = get_template_prompts()
    system_explicit, user_explicit = get_template_prompts(prompt_variant="default")

    assert system_implicit == system_explicit
    assert user_implicit == user_explicit


def test_prompt_variant_resolution_is_deterministic_across_repeated_calls() -> None:
    first = get_template_prompts(
        ordering="random",
        coord_mode="numeric",
        prompt_variant="coco_80",
    )
    second = get_template_prompts(
        ordering="random",
        coord_mode="numeric",
        prompt_variant="coco_80",
    )

    assert first == second


def test_prompt_variant_cross_surface_parity_between_training_and_inference() -> None:
    train_prompts = ConfigLoader.resolve_prompts(
        {
            "custom": {
                "object_ordering": "sorted",
                "coord_tokens": {"enabled": True},
                "extra": {"prompt_variant": "coco_80"},
            }
        }
    )

    inf_cfg = InferenceConfig(
        gt_jsonl="dummy.jsonl",
        model_checkpoint="dummy",
        mode="text",
        prompt_variant="coco_80",
    )
    engine = InferenceEngine(inf_cfg, GenerationConfig())

    messages = engine._build_messages(Image.new("RGB", (16, 16), color=(0, 0, 0)))
    infer_system = messages[0]["content"][0]["text"]
    infer_user = messages[1]["content"][0]["text"]

    assert infer_system == train_prompts.system
    assert infer_user == train_prompts.user


def test_training_prompt_resolution_uses_fixed_base_plus_variant() -> None:
    sorted_tokens = ConfigLoader.resolve_prompts(
        {
            "custom": {
                "object_ordering": "sorted",
                "coord_tokens": {"enabled": True},
                "extra": {"prompt_variant": "coco_80"},
            }
        }
    )
    random_numeric = ConfigLoader.resolve_prompts(
        {
            "custom": {
                "object_ordering": "random",
                "coord_tokens": {"enabled": False},
                "extra": {"prompt_variant": "coco_80"},
            }
        }
    )

    expected_system, expected_user = get_template_prompts(prompt_variant="coco_80")

    assert sorted_tokens.system == expected_system
    assert sorted_tokens.user == expected_user
    assert random_numeric.system == expected_system
    assert random_numeric.user == expected_user


def test_coco_80_prompt_variant_has_compact_canonical_unique_list() -> None:
    assert len(COCO_80_CLASS_NAMES) == 80
    assert len(set(COCO_80_CLASS_NAMES)) == 80
    assert COCO_80_CLASS_LIST_COMPACT.count(",") == 79

    system_prompt, user_prompt = get_template_prompts(prompt_variant="coco_80")
    assert "COCO-80 closed-class policy" in system_prompt
    assert "Restrict `desc` to this COCO-80 class list" in user_prompt

    prefix = "Restrict `desc` to this COCO-80 class list: "
    class_list = user_prompt.split(prefix, maxsplit=1)[1].rstrip(".")
    parsed = [name.strip() for name in class_list.split(",")]

    assert parsed == list(COCO_80_CLASS_NAMES)
    assert len(parsed) == 80
    assert len(set(parsed)) == 80


def test_summary_prompts_are_unaffected_by_prompt_variant() -> None:
    prompts = ConfigLoader.resolve_prompts(
        {
            "custom": {
                "use_summary": True,
                "extra": {"prompt_variant": "coco_80"},
            }
        }
    )

    assert prompts.output_variant == "summary"
    assert prompts.system == SYSTEM_PROMPT_SUMMARY
    assert prompts.user == USER_PROMPT_SUMMARY


def test_unknown_prompt_variant_error_lists_unknown_and_available_keys() -> None:
    with pytest.raises(ValueError, match="Unknown prompt variant") as exc_info:
        get_template_prompts(prompt_variant="unknown_variant")

    message = str(exc_info.value)
    assert "unknown_variant" in message
    assert "default" in message
    assert "coco_80" in message

    with pytest.raises(ValueError, match="Unknown prompt variant"):
        ConfigLoader.resolve_prompts(
            {"custom": {"extra": {"prompt_variant": "unknown_variant"}}}
        )
