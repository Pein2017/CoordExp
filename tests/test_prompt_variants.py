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
    get_template_prompt_hash,
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
        coord_mode="coord_tokens",
        prompt_variant="coco_80",
    )
    second = get_template_prompts(
        ordering="random",
        coord_mode="coord_tokens",
        prompt_variant="coco_80",
    )

    assert first == second


def test_prompt_template_hash_changes_when_bbox_format_changes() -> None:
    xyxy_hash = get_template_prompt_hash(
        prompt_variant="lvis_stage1_federated",
        bbox_format="xyxy",
        object_field_order="desc_first",
    )
    cxcy_logw_logh_hash = get_template_prompt_hash(
        prompt_variant="lvis_stage1_federated",
        bbox_format="cxcy_logw_logh",
        object_field_order="desc_first",
    )

    assert xyxy_hash != cxcy_logw_logh_hash


@pytest.mark.parametrize("object_ordering", ["sorted", "random"])
def test_prompt_variant_cross_surface_parity_between_training_and_inference(
    object_ordering: str,
) -> None:
    train_prompts = ConfigLoader.resolve_prompts(
        {
            "custom": {
                "object_ordering": object_ordering,
                "object_field_order": "desc_first",
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
        object_ordering=object_ordering,
    )
    engine = InferenceEngine(inf_cfg, GenerationConfig())

    messages = engine._build_messages(Image.new("RGB", (16, 16), color=(0, 0, 0)))
    infer_system = messages[0]["content"][0]["text"]
    infer_user = messages[1]["content"][0]["text"]

    assert infer_system == train_prompts.system
    assert infer_user == train_prompts.user


@pytest.mark.parametrize("object_field_order", ["desc_first", "geometry_first"])
def test_prompt_variant_cross_surface_parity_for_cxcy_logw_logh(
    object_field_order: str,
) -> None:
    train_prompts = ConfigLoader.resolve_prompts(
        {
            "custom": {
                "bbox_format": "cxcy_logw_logh",
                "object_ordering": "sorted",
                "object_field_order": object_field_order,
                "coord_tokens": {"enabled": True},
                "extra": {"prompt_variant": "lvis_stage1_federated"},
            }
        }
    )

    inf_cfg = InferenceConfig(
        gt_jsonl="dummy.jsonl",
        model_checkpoint="dummy",
        mode="text",
        prompt_variant="lvis_stage1_federated",
        bbox_format="cxcy_logw_logh",
        object_field_order=object_field_order,
        object_ordering="sorted",
    )
    engine = InferenceEngine(inf_cfg, GenerationConfig())

    messages = engine._build_messages(Image.new("RGB", (16, 16), color=(0, 0, 0)))
    infer_system = messages[0]["content"][0]["text"]
    infer_user = messages[1]["content"][0]["text"]

    assert infer_system == train_prompts.system
    assert infer_user == train_prompts.user
    if object_field_order == "geometry_first":
        assert "geometry key (bbox_2d OR poly) before desc" in infer_system
        assert "geometry (bbox_2d or poly) before desc" in infer_user
    else:
        assert "desc before exactly one geometry key" in infer_system
        assert "desc before one geometry" in infer_user


def test_inference_object_ordering_defaults_to_sorted() -> None:
    engine = InferenceEngine(
        InferenceConfig(
            gt_jsonl="dummy.jsonl",
            model_checkpoint="dummy",
            mode="text",
        ),
        GenerationConfig(),
    )

    assert engine.object_ordering == "sorted"


def test_training_prompt_resolution_uses_ordering_plus_variant() -> None:
    sorted_tokens = ConfigLoader.resolve_prompts(
        {
            "custom": {
                "object_ordering": "sorted",
                "object_field_order": "desc_first",
                "coord_tokens": {"enabled": True},
                "extra": {"prompt_variant": "coco_80"},
            }
        }
    )
    random_tokens = ConfigLoader.resolve_prompts(
        {
            "custom": {
                "object_ordering": "random",
                "object_field_order": "desc_first",
                "coord_tokens": {"enabled": True},
                "extra": {"prompt_variant": "coco_80"},
            }
        }
    )

    expected_sorted_system, expected_sorted_user = get_template_prompts(
        ordering="sorted",
        prompt_variant="coco_80",
    )
    expected_random_system, expected_random_user = get_template_prompts(
        ordering="random",
        prompt_variant="coco_80",
    )

    assert sorted_tokens.system == expected_sorted_system
    assert sorted_tokens.user == expected_sorted_user
    assert random_tokens.system == expected_random_system
    assert random_tokens.user == expected_random_user
    assert sorted_tokens.system != random_tokens.system
    assert sorted_tokens.user != random_tokens.user


def test_coord_mode_numeric_is_rejected() -> None:
    with pytest.raises(ValueError, match="coord_mode must be 'coord_tokens'"):
        get_template_prompts(coord_mode="numeric")


def test_training_prompt_resolution_rejects_coord_tokens_disabled() -> None:
    with pytest.raises(
        ValueError,
        match="custom.coord_tokens.enabled must be true",
    ):
        ConfigLoader.resolve_prompts(
            {
                "custom": {
                    "object_field_order": "desc_first",
                    "coord_tokens": {"enabled": False},
                    "extra": {"prompt_variant": "default"},
                }
            }
        )


def test_training_prompt_resolution_rejects_skip_bbox_norm_false() -> None:
    with pytest.raises(
        ValueError,
        match="custom.coord_tokens.skip_bbox_norm must be true",
    ):
        ConfigLoader.resolve_prompts(
            {
                "custom": {
                    "object_field_order": "desc_first",
                    "coord_tokens": {"enabled": True, "skip_bbox_norm": False},
                    "extra": {"prompt_variant": "default"},
                }
            }
        )


def test_coco_80_prompt_variant_has_compact_canonical_unique_list() -> None:
    assert len(COCO_80_CLASS_NAMES) == 80
    assert len(set(COCO_80_CLASS_NAMES)) == 80
    assert COCO_80_CLASS_LIST_COMPACT.count(",") == 79

    system_prompt, user_prompt = get_template_prompts(prompt_variant="coco_80")
    assert "COCO-80 closed-class policy" in system_prompt
    assert "Restrict `desc` to this COCO-80 class list" in user_prompt

    prefix = "Restrict `desc` to this COCO-80 class list: "
    class_clause = user_prompt.split(prefix, maxsplit=1)[1]
    class_list = class_clause.split(
        ". Locate each clearly visible object instance",
        maxsplit=1,
    )[0].rstrip(".")
    parsed = [name.strip() for name in class_list.split(",")]

    assert parsed == list(COCO_80_CLASS_NAMES)
    assert len(parsed) == 80
    assert len(set(parsed)) == 80


def test_summary_prompts_are_unaffected_by_prompt_variant() -> None:
    prompts = ConfigLoader.resolve_prompts(
        {
            "custom": {
                "use_summary": True,
                "object_field_order": "desc_first",
                "extra": {"prompt_variant": "coco_80"},
            }
        }
    )

    assert prompts.output_variant == "summary"
    assert prompts.system == SYSTEM_PROMPT_SUMMARY
    assert prompts.user == USER_PROMPT_SUMMARY


def test_lvis_federated_prompt_variants_encode_partial_label_semantics() -> None:
    stage1_system, stage1_user = get_template_prompts(
        prompt_variant="lvis_stage1_federated"
    )
    stage2_system, stage2_user = get_template_prompts(
        prompt_variant="lvis_stage2_federated"
    )

    assert "verified annotation subset" in stage1_system
    assert "omission as absence" in stage1_system
    assert "verified subset rather than an exhaustive absence claim" in stage1_user
    assert "including small or partially occluded ones" in stage1_user

    assert "do not assume an omitted category is absent" in stage2_system
    assert "continue listing clearly visible" in stage2_system.lower()
    assert "continue with additional visible instances" in stage2_user
    assert "prefer inclusion over omission" in stage2_user

    assert "Detect every object in the image" not in stage1_system
    assert "Detect every object in the image" not in stage2_system


def test_cxcy_logw_logh_prompts_explain_u_of_s_and_render_variant_placeholders() -> None:
    default_system, default_user = get_template_prompts(bbox_format="cxcy_logw_logh")
    lvis_system, lvis_user = get_template_prompts(
        prompt_variant="lvis_stage1_federated",
        bbox_format="cxcy_logw_logh",
    )

    for prompt in (default_system, default_user, lvis_system, lvis_user):
        assert "[cx, cy, u(w), u(h)]" in prompt
        assert "__BBOX_" not in prompt
        assert "__USER_EXAMPLE_" not in prompt

    for prompt in (default_system, lvis_system):
        assert "log(max(s, 1/1024))" in prompt

    assert "bbox_2d is [x1, y1, x2, y2]" not in lvis_system
    assert '{"desc": "category", "bbox_2d": [<|coord_110|>' not in lvis_user


def test_unknown_prompt_variant_error_lists_unknown_and_available_keys() -> None:
    with pytest.raises(ValueError, match="Unknown prompt variant") as exc_info:
        get_template_prompts(prompt_variant="unknown_variant")

    message = str(exc_info.value)
    assert "unknown_variant" in message
    assert "default" in message
    assert "coco_80" in message

    with pytest.raises(ValueError, match="Unknown prompt variant"):
        ConfigLoader.resolve_prompts(
            {
                "custom": {
                    "object_field_order": "desc_first",
                    "extra": {"prompt_variant": "unknown_variant"},
                }
            }
        )
