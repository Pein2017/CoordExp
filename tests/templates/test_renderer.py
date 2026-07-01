from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.common.errors import ConfigContractError, TemplateContractError
from src.config import load_train_config
from src.data import ImageRef, RawExample, RawObject, SourceProvenance, load_raw_examples
from src.templates import (
    BOX_END_TOKEN,
    BOX_START_TOKEN,
    IM_END_SUFFIX,
    IM_END_TOKEN,
    RenderedSpan,
    rendered_examples_snapshot,
    OBJECT_REF_END_TOKEN,
    OBJECT_REF_START_TOKEN,
    render_example,
    validate_rendered_spans,
)


FIXTURE = Path("tests/fixtures/smoke/qwen3_vl_single_image_pack")


def test_render_source_order_object_box_closed_smoke_fixture() -> None:
    resolved = load_train_config(FIXTURE / "config.yaml")
    first = load_raw_examples(resolved.config.data.train)[0]

    rendered = render_example(first, resolved.config.template)

    expected_content = (
        f"{OBJECT_REF_START_TOKEN}potted plant{OBJECT_REF_END_TOKEN}"
        f"{BOX_START_TOKEN}<|coord_319|><|coord_72|><|coord_718|><|coord_830|>{BOX_END_TOKEN}"
        f"{OBJECT_REF_START_TOKEN}vase{OBJECT_REF_END_TOKEN}"
        f"{BOX_START_TOKEN}<|coord_370|><|coord_364|><|coord_632|><|coord_820|>{BOX_END_TOKEN}"
    )
    assert rendered.assistant_content_text == expected_content
    assert rendered.supervised_response_text == expected_content + IM_END_SUFFIX
    assert [item.object_id for item in rendered.realized_object_order] == [
        "291613",
        "1155486",
    ]
    assert rendered.object_order_seed is None
    assert rendered.messages[-1]["content"][0]["text"] == expected_content
    assert not rendered.messages[-1]["content"][0]["text"].endswith(IM_END_SUFFIX)
    assert rendered.messages[-2]["content"][0]["type"] == "image"
    assert rendered.messages[-2]["content"][1]["text"] == resolved.config.template.prompt.user


def test_geometry_first_renders_box_before_description() -> None:
    resolved = load_train_config(FIXTURE / "config.yaml")
    config = resolved.config.template.model_copy(update={"object_field_order": "geometry_first"})
    first = load_raw_examples(resolved.config.data.train)[0]

    rendered = render_example(first, config)

    assert rendered.assistant_content_text.startswith(
        f"{BOX_START_TOKEN}<|coord_319|><|coord_72|><|coord_718|><|coord_830|>{BOX_END_TOKEN}"
        f"{OBJECT_REF_START_TOKEN}potted plant{OBJECT_REF_END_TOKEN}"
    )


def test_no_inserted_separator_between_objects_or_coordinate_tokens() -> None:
    resolved = load_train_config(FIXTURE / "config.yaml")
    first = load_raw_examples(resolved.config.data.train)[0]

    rendered = render_example(first, resolved.config.template)

    assert f"{BOX_END_TOKEN}{OBJECT_REF_START_TOKEN}" in rendered.assistant_content_text
    assert "<|coord_319|><|coord_72|><|coord_718|><|coord_830|>" in rendered.assistant_content_text
    assert ",<|coord_" not in rendered.assistant_content_text
    assert " <|coord_" not in rendered.assistant_content_text


def test_rendered_spans_cover_wrappers_coordinates_eos_and_ignored_newline() -> None:
    resolved = load_train_config(FIXTURE / "config.yaml")
    first = load_raw_examples(resolved.config.data.train)[0]

    rendered = render_example(first, resolved.config.template)

    for span in rendered.spans:
        assert rendered.supervised_response_text[span.char_start : span.char_end] == span.text
    leaf = [
        span
        for span in rendered.spans
        if span.kind in {"description", "schema_token", "coordinate_token", "eos_transition", "ignored_text"}
    ]
    assert [span.text for span in leaf[:3]] == [
        OBJECT_REF_START_TOKEN,
        "potted plant",
        OBJECT_REF_END_TOKEN,
    ]
    assert [span.text for span in leaf if span.kind == "coordinate_token"][:4] == [
        "<|coord_319|>",
        "<|coord_72|>",
        "<|coord_718|>",
        "<|coord_830|>",
    ]
    assert leaf[-2].kind == "eos_transition"
    assert leaf[-2].text == IM_END_TOKEN
    assert leaf[-1].kind == "ignored_text"
    assert leaf[-1].text == "\n"


def test_rendered_span_validation_rejects_crossing_spans() -> None:
    text = "abcdef"
    spans = (
        RenderedSpan("assistant_content", 0, 4, "abcd"),
        RenderedSpan("object", 2, 6, "cdef"),
        RenderedSpan("description", 0, 6, text),
    )

    with pytest.raises(TemplateContractError) as exc_info:
        validate_rendered_spans(text, spans)

    assert exc_info.value.code == "template.span_crossing"


def test_rendered_span_validation_rejects_uncovered_characters() -> None:
    text = "abc"
    spans = (RenderedSpan("description", 0, 2, "ab"),)

    with pytest.raises(TemplateContractError) as exc_info:
        validate_rendered_spans(text, spans)

    assert exc_info.value.code == "template.leaf_coverage"


def test_rendered_span_validation_rejects_partial_or_loose_special_literals() -> None:
    with pytest.raises(TemplateContractError) as coord_exc:
        validate_rendered_spans(
            "<|coord_001|>",
            (RenderedSpan("coordinate_token", 0, len("<|coord_001|>"), "<|coord_001|>"),),
        )
    assert coord_exc.value.code == "template.coordinate_span_literal"

    with pytest.raises(TemplateContractError) as schema_exc:
        validate_rendered_spans(
            "<|box_start|><|box_end|>",
            (
                RenderedSpan(
                    "schema_token",
                    0,
                    len("<|box_start|><|box_end|>"),
                    "<|box_start|><|box_end|>",
                ),
            ),
        )
    assert schema_exc.value.code == "template.special_span_literal"

    with pytest.raises(TemplateContractError) as arbitrary_schema_exc:
        validate_rendered_spans(
            "<|vision_start|>",
            (RenderedSpan("schema_token", 0, len("<|vision_start|>"), "<|vision_start|>"),),
        )
    assert arbitrary_schema_exc.value.code == "template.special_span_literal"

    validate_rendered_spans(
        OBJECT_REF_START_TOKEN,
        (
            RenderedSpan(
                "schema_token",
                0,
                len(OBJECT_REF_START_TOKEN),
                OBJECT_REF_START_TOKEN,
            ),
        ),
    )
    validate_rendered_spans(
        IM_END_TOKEN,
        (RenderedSpan("eos_transition", 0, len(IM_END_TOKEN), IM_END_TOKEN),),
    )


def test_random_object_ordering_is_seeded_and_reproducible() -> None:
    resolved = load_train_config(FIXTURE / "config.yaml")
    config = resolved.config.template.model_copy(update={"object_ordering": "random"})
    second = load_raw_examples(resolved.config.data.train)[1]
    seed = resolved.config.runtime.seed

    first_render = render_example(second, config, object_order_seed=seed)
    second_render = render_example(second, config, object_order_seed=seed)

    assert first_render.assistant_content_text == second_render.assistant_content_text
    assert first_render.realized_object_order == second_render.realized_object_order
    assert first_render.object_order_seed == seed
    assert first_render.object_order_seed_source == f"{seed}:{second.example_id}"


def test_random_object_ordering_requires_explicit_seed() -> None:
    resolved = load_train_config(FIXTURE / "config.yaml")
    config = resolved.config.template.model_copy(update={"object_ordering": "random"})
    first = load_raw_examples(resolved.config.data.train)[0]

    with pytest.raises(TemplateContractError) as exc_info:
        render_example(first, config)

    assert exc_info.value.code == "template.random_seed_required"


@pytest.mark.parametrize(
    "unsafe_text",
    [
        OBJECT_REF_START_TOKEN,
        IM_END_TOKEN,
        "<|vision_start|>",
        "<|vision_end|>",
        "<|image_pad|>",
        "<|video_pad|>",
        "<|quad_start|>",
        "<|fim_prefix|>",
        "<tool_call>",
        "</tool_response>",
        "<think>",
    ],
)
def test_unsafe_description_token_fails_rendering(tmp_path: Path, unsafe_text: str) -> None:
    image = tmp_path / "image.jpg"
    image.write_bytes(b"bytes")
    example = RawExample(
        "example-1",
        ImageRef("image.jpg", image, 64, 64, {}),
        (
            RawObject("object-1", f"bad {unsafe_text}", [1, 2, 3, 4], {}),
        ),
        {},
        SourceProvenance(tmp_path / "examples.jsonl", 1, "abc", "manual"),
    )
    template = load_train_config(FIXTURE / "config.yaml").config.template

    with pytest.raises(TemplateContractError) as exc_info:
        render_example(example, template)

    assert exc_info.value.code == "template.description_unsafe_token"


@pytest.mark.parametrize(
    ("prompt_field", "alias"),
    [
        ("user", "<|object_start|>"),
        ("system", "<|object_end|>"),
    ],
)
def test_invalid_wrapper_alias_in_prompt_fails_rendering(
    prompt_field: str,
    alias: str,
) -> None:
    resolved = load_train_config(FIXTURE / "config.yaml")
    first = load_raw_examples(resolved.config.data.train)[0]
    prompt = resolved.config.template.prompt.model_copy(
        update={prompt_field: f"Use bad alias {alias}."}
    )
    template = resolved.config.template.model_copy(update={"prompt": prompt})

    with pytest.raises(TemplateContractError) as exc_info:
        render_example(first, template)

    assert exc_info.value.code == "template.prompt_unsafe_alias"


def test_expected_rendered_snapshot_matches_real_renderer() -> None:
    resolved = load_train_config(FIXTURE / "config.yaml")
    examples = load_raw_examples(resolved.config.data.train)
    rendered = [render_example(example, resolved.config.template) for example in examples]

    actual = rendered_examples_snapshot(rendered, image_root=FIXTURE)
    expected_path = FIXTURE / "expected_rendered.json"
    assert expected_path.exists()
    assert actual == json.loads(expected_path.read_text(encoding="utf-8"))


def test_legacy_sorted_object_ordering_rejected_by_config(tmp_path: Path) -> None:
    payload = _load_fixture_config_payload()
    payload["template"]["object_ordering"] = "sorted"
    path = tmp_path / "config.yaml"
    path.write_text(_to_yamlish_json(payload), encoding="utf-8")

    with pytest.raises(ConfigContractError):
        load_train_config(path)


def _load_fixture_config_payload() -> dict:
    import yaml

    return yaml.safe_load((FIXTURE / "config.yaml").read_text(encoding="utf-8"))


def _to_yamlish_json(payload: dict) -> str:
    copied = copy.deepcopy(payload)
    copied["data"]["train"]["path"] = str((FIXTURE / "examples.jsonl").resolve())
    copied["data"]["eval"]["path"] = str((FIXTURE / "examples.jsonl").resolve())
    return json.dumps(copied, indent=2, sort_keys=True)
