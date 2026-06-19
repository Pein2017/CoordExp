from __future__ import annotations

import pytest
from PIL import Image

from src.common.detection_chat import build_detection_chat_messages
from src.config.loader import ConfigLoader
from src.config.prompt_variants import (
    COCO_80_CLASS_LIST_COMPACT,
    COCO_80_CLASS_NAMES,
)
from src.config.prompts import (
    SYSTEM_PROMPT_SUMMARY,
    USER_PROMPT_SUMMARY,
    coord_mode_from_coord_tokens_enabled,
    get_template_prompt_hash,
    get_template_prompts,
)


def _message_text(content: object) -> str:
    if isinstance(content, str):
        return content
    assert isinstance(content, list)
    text_parts = [
        str(item.get("text", ""))
        for item in content
        if isinstance(item, dict) and item.get("type") == "text"
    ]
    assert text_parts
    return "".join(text_parts)


def _build_inference_messages(
    *,
    mode: str = "coord",
    prompt_variant: str = "coco_80",
    object_ordering: str = "sorted",
    object_field_order: str = "desc_first",
    bbox_format: str = "xyxy",
    detection_sequence_format: str = "coordjson",
    detection_template_id: str | None = None,
) -> list[dict[str, object]]:
    system_prompt, user_prompt = get_template_prompts(
        ordering=object_ordering,
        coord_mode=coord_mode_from_coord_tokens_enabled(mode == "coord"),
        prompt_variant=prompt_variant,
        object_field_order=object_field_order,
        bbox_format=bbox_format,
        detection_sequence_format=detection_sequence_format,
        detection_template_id=detection_template_id,
    )
    return build_detection_chat_messages(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        images=[Image.new("RGB", (16, 16), color=(0, 0, 0))],
    )


def test_detection_chat_builder_keeps_text_roles_swift_compatible() -> None:
    messages = build_detection_chat_messages(
        system_prompt="system prompt",
        user_prompt="user prompt",
        images=["/tmp/image.png"],
        assistant_text="assistant response",
    )

    assert messages[0] == {"role": "system", "content": "system prompt"}
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == [
        {"type": "image", "image": "/tmp/image.png"},
        {"type": "text", "text": "user prompt"},
    ]
    assert messages[2] == {"role": "assistant", "content": "assistant response"}


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


def test_prompt_template_hash_changes_when_detection_template_changes() -> None:
    coordjson_hash = get_template_prompt_hash(
        prompt_variant="coco_80",
        detection_template_id="stage1_json_pretty",
    )
    compact_hash = get_template_prompt_hash(
        prompt_variant="coco_80",
        detection_template_id="compact",
    )

    assert coordjson_hash != compact_hash


def test_compact_prompt_template_hash_changes_when_object_field_order_changes() -> None:
    desc_first_pattern = (
        "<|object_ref_start|>{desc}<|object_ref_end|><|box_start|>"
        "<|coord_x1|><|coord_y1|><|coord_x2|><|coord_y2|><|box_end|>"
    )
    geometry_first_pattern = (
        "<|box_start|><|coord_x1|><|coord_y1|><|coord_x2|><|coord_y2|><|box_end|>"
        "<|object_ref_start|>{desc}<|object_ref_end|>"
    )

    desc_first_system, desc_first_user = get_template_prompts(
        prompt_variant="coco_80",
        detection_template_id="compact_object_box_closed",
        object_field_order="desc_first",
    )
    geometry_first_system, geometry_first_user = get_template_prompts(
        prompt_variant="coco_80",
        detection_template_id="compact_object_box_closed",
        object_field_order="geometry_first",
    )
    desc_first_hash = get_template_prompt_hash(
        prompt_variant="coco_80",
        detection_template_id="compact_object_box_closed",
        object_field_order="desc_first",
    )
    geometry_first_hash = get_template_prompt_hash(
        prompt_variant="coco_80",
        detection_template_id="compact_object_box_closed",
        object_field_order="geometry_first",
    )

    assert desc_first_pattern in desc_first_system
    assert desc_first_pattern in desc_first_user
    assert geometry_first_pattern in geometry_first_system
    assert geometry_first_pattern in geometry_first_user
    assert desc_first_hash != geometry_first_hash


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

    messages = _build_inference_messages(
        prompt_variant="coco_80",
        object_ordering=object_ordering,
    )
    infer_system = _message_text(messages[0]["content"])
    infer_user = _message_text(messages[1]["content"])

    assert infer_system == train_prompts.system
    assert infer_user == train_prompts.user


def test_compact_prompt_variant_cross_surface_parity_between_training_and_inference() -> None:
    train_prompts = ConfigLoader.resolve_prompts(
        {
            "custom": {
                "object_ordering": "sorted",
                "object_field_order": "desc_first",
                "coord_tokens": {"enabled": True},
                "detection_sequence_format": "compact",
                "extra": {"prompt_variant": "coco_80"},
            }
        }
    )

    messages = _build_inference_messages(
        prompt_variant="coco_80",
        object_ordering="sorted",
        detection_template_id="compact",
    )
    infer_system = _message_text(messages[0]["content"])
    infer_user = _message_text(messages[1]["content"])

    assert infer_system == train_prompts.system
    assert infer_user == train_prompts.user
    assert [item["type"] for item in messages[1]["content"]] == ["image", "text"]
    assert "<|object_ref_start|>{desc}<|box_start|>" in infer_system
    assert "<|object_ref_start|>{desc}<|box_start|>" in infer_user
    assert "CoordJSON" not in infer_system
    assert "CoordJSON" not in infer_user


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

    messages = _build_inference_messages(
        prompt_variant="lvis_stage1_federated",
        bbox_format="cxcy_logw_logh",
        object_field_order=object_field_order,
        object_ordering="sorted",
    )
    infer_system = _message_text(messages[0]["content"])
    infer_user = _message_text(messages[1]["content"])

    assert infer_system == train_prompts.system
    assert infer_user == train_prompts.user
    if object_field_order == "geometry_first":
        assert "geometry key (bbox_2d OR poly) before desc" in infer_system
        assert "geometry (bbox_2d or poly) before desc" in infer_user
    else:
        assert "desc before exactly one geometry key" in infer_system
        assert "desc before one geometry" in infer_user


def test_inference_object_ordering_defaults_to_sorted() -> None:
    default_system, default_user = get_template_prompts(
        coord_mode=coord_mode_from_coord_tokens_enabled(False)
    )
    sorted_system, sorted_user = get_template_prompts(
        ordering="sorted",
        coord_mode=coord_mode_from_coord_tokens_enabled(False),
    )

    assert default_system == sorted_system
    assert default_user == sorted_user


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


def test_training_prompt_resolution_uses_target_hierarchy_prompt_variant() -> None:
    prompts = ConfigLoader.resolve_prompts(
        {
            "prompt": {"variant": "coco_80"},
            "detection_template": {"id": "compact_object_box_closed"},
            "sample_factory": {
                "id": "detection_sequence",
                "target_sequence": {
                    "object_ordering": "random_permutation",
                    "object_field_order": "geometry_first",
                    "bbox_format": "xyxy",
                    "coordinate_surface": "coord_token",
                },
            },
        }
    )
    expected_system, expected_user = get_template_prompts(
        ordering="random",
        prompt_variant="coco_80",
        detection_template_id="compact_object_box_closed",
        object_field_order="geometry_first",
    )

    assert prompts.system == expected_system
    assert prompts.user == expected_user


def test_target_hierarchy_prompt_resolution_rejects_prompt_variant_enabled() -> None:
    with pytest.raises(
        ValueError,
        match=r"prompt\.prompt_variant_enabled.*prompt\.variant",
    ):
        ConfigLoader.resolve_prompts(
            {
                "prompt": {"prompt_variant_enabled": True},
                "detection_template": {"id": "compact"},
                "sample_factory": {
                    "id": "detection_sequence",
                    "target_sequence": {
                        "object_ordering": "random_permutation",
                        "object_field_order": "geometry_first",
                        "bbox_format": "xyxy",
                        "coordinate_surface": "coord_token",
                    },
                },
            }
        )


@pytest.mark.parametrize("custom_section", [{}, {"unrelated": True}])
def test_target_hierarchy_prompt_resolution_rejects_any_custom_mapping(
    custom_section: dict[str, object],
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            r"custom is obsolete for target-hierarchy detection prompt resolution.*"
            r"prompt\.variant.*sample_factory\.target_sequence.*detection_template\.id"
        ),
    ):
        ConfigLoader.resolve_prompts(
            {
                "prompt": {"variant": "coco_80"},
                "detection_template": {"id": "compact"},
                "sample_factory": {
                    "id": "detection_sequence",
                    "target_sequence": {
                        "object_ordering": "random_permutation",
                        "object_field_order": "geometry_first",
                        "bbox_format": "xyxy",
                        "coordinate_surface": "coord_token",
                    },
                },
                "custom": custom_section,
            }
        )


@pytest.mark.parametrize(
    ("legacy_custom", "match"),
    [
        (
            {"object_ordering": "sorted", "object_field_order": "desc_first"},
            r"custom\.object_ordering.*sample_factory\.target_sequence\.object_ordering",
        ),
        (
            {"object_field_order": "desc_first"},
            r"custom\.object_field_order.*sample_factory\.target_sequence\.object_field_order",
        ),
        (
            {"object_field_order": "desc_first", "detection_template_id": "compact"},
            r"custom\.detection_template_id.*detection_template\.id",
        ),
        (
            {
                "object_field_order": "desc_first",
                "detection_sequence_format": "compact",
            },
            r"custom\.detection_sequence_format.*sample_factory",
        ),
        (
            {
                "object_field_order": "desc_first",
                "extra": {"prompt_variant": "coco_80"},
            },
            r"custom\.extra\.prompt_variant.*prompt\.variant",
        ),
    ],
)
def test_target_hierarchy_prompt_resolution_rejects_legacy_custom_sequence_fields(
    legacy_custom: dict[str, object],
    match: str,
) -> None:
    config = {
        "prompt": {"variant": "coco_80"},
        "detection_template": {"id": "compact"},
        "sample_factory": {
            "id": "detection_sequence",
            "target_sequence": {
                "object_ordering": "random_permutation",
                "object_field_order": "geometry_first",
                "bbox_format": "xyxy",
                "coordinate_surface": "coord_token",
            },
        },
        "custom": legacy_custom,
    }

    with pytest.raises(ValueError, match=match):
        ConfigLoader.resolve_prompts(config)


def test_coord_mode_numeric_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be one of"):
        get_template_prompts(coord_mode="numeric")


def test_training_prompt_resolution_rejects_compact_without_coord_tokens() -> None:
    with pytest.raises(
        ValueError,
        match="compact detection rendering requires custom.coord_tokens.enabled=true",
    ):
        ConfigLoader.resolve_prompts(
            {
                "custom": {
                    "object_field_order": "desc_first",
                    "coord_tokens": {"enabled": False},
                    "detection_sequence_format": "compact",
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


@pytest.mark.parametrize(
    ("fmt", "expected_pattern"),
    [
        ("compact", "<|object_ref_start|>{desc}<|box_start|><|coord_x1|><|coord_y1|><|coord_x2|><|coord_y2|>"),
        ("compact_box_closed", "<|object_ref_start|>{desc}<|box_start|><|coord_x1|><|coord_y1|><|coord_x2|><|coord_y2|><|box_end|>"),
        ("compact_object_box_closed", "<|object_ref_start|>{desc}<|object_ref_end|><|box_start|><|coord_x1|><|coord_y1|><|coord_x2|><|coord_y2|><|box_end|>"),
        ("compact_object_box_closed_lines", "<|object_ref_start|>{desc}<|object_ref_end|><|box_start|><|coord_x1|><|coord_y1|><|coord_x2|><|coord_y2|><|box_end|>\n"),
    ],
)
def test_coco_80_prompt_variant_renders_each_compact_template(
    fmt: str, expected_pattern: str
) -> None:
    system_prompt, user_prompt = get_template_prompts(
        prompt_variant="coco_80",
        detection_template_id=fmt,
    )

    assert expected_pattern in system_prompt
    assert expected_pattern in user_prompt
    assert "JSON layout" not in system_prompt


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


def test_cxcywh_prompts_explain_center_and_linear_size_slots() -> None:
    default_system, default_user = get_template_prompts(bbox_format="cxcywh")
    lvis_system, lvis_user = get_template_prompts(
        prompt_variant="lvis_stage1_federated",
        bbox_format="cxcywh",
    )

    for prompt in (default_system, default_user, lvis_system, lvis_user):
        assert "[cx, cy, w, h]" in prompt
        assert "__BBOX_" not in prompt
        assert "__USER_EXAMPLE_" not in prompt

    for prompt in (default_system, lvis_system):
        assert "normalized box width and height" in prompt

    assert "bbox_2d is [x1, y1, x2, y2]" not in lvis_system


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
