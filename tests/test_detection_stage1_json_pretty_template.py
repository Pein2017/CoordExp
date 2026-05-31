from dataclasses import replace

import pytest

from src.detection.data import (
    CoordinateTokenBox,
    DetectionMetadata,
    NormalizedDetectionObject,
    NormalizedDetectionSample,
    ObjectOrderingPlan,
)
from src.detection.scene import detection_scene_from_normalized_sample_bridge
from src.detection.template import Stage1JsonPrettyTemplate
from src.utils.assistant_json import dumps_coordjson


def _sample() -> NormalizedDetectionSample:
    return NormalizedDetectionSample(
        images=("image.jpg",),
        objects=(
            NormalizedDetectionObject(
                normalized_object_index=0,
                source_object_index=2,
                object_instance_id="img-17:ann-101:src-2",
                desc="cat",
                bbox_2d=CoordinateTokenBox(
                    "<|coord_1|>",
                    "<|coord_2|>",
                    "<|coord_3|>",
                    "<|coord_4|>",
                ),
                category_id=17,
                category_name="cat",
                coco_ann_id=101,
            ),
            NormalizedDetectionObject(
                normalized_object_index=1,
                source_object_index=4,
                object_instance_id="img-17:ann-102:src-4",
                desc="dog",
                bbox_2d=CoordinateTokenBox(
                    "<|coord_10|>",
                    "<|coord_20|>",
                    "<|coord_30|>",
                    "<|coord_40|>",
                ),
                category_id=18,
                category_name="dog",
                coco_ann_id=102,
            ),
        ),
        width=640,
        height=480,
        image_id=17,
        file_name="image.jpg",
        metadata=DetectionMetadata(source="unit", split="test"),
        object_ordering=ObjectOrderingPlan.sorted().with_realized((2, 4)),
    )


def _repeated_identical_sample() -> NormalizedDetectionSample:
    sample = _sample()
    first = sample.objects[0]
    repeated = replace(
        first,
        normalized_object_index=1,
        source_object_index=8,
        object_instance_id="img-17:ann-103:src-8",
        coco_ann_id=103,
    )
    return replace(
        sample,
        objects=(first, repeated),
        object_ordering=ObjectOrderingPlan.sorted().with_realized((2, 8)),
    )


def test_stage1_json_pretty_matches_canonical_coordjson_fixture() -> None:
    sample = _sample()
    rendered = Stage1JsonPrettyTemplate().render_assistant(sample)
    expected = dumps_coordjson(
        {
            "objects": [
                {
                    "desc": "cat",
                    "bbox_2d": [
                        "<|coord_1|>",
                        "<|coord_2|>",
                        "<|coord_3|>",
                        "<|coord_4|>",
                    ],
                },
                {
                    "desc": "dog",
                    "bbox_2d": [
                        "<|coord_10|>",
                        "<|coord_20|>",
                        "<|coord_30|>",
                        "<|coord_40|>",
                    ],
                },
            ]
        }
    )

    assert rendered.text == expected
    assert rendered.text == (
        '{"objects": [{"desc": "cat", "bbox_2d": '
        "[<|coord_1|>, <|coord_2|>, <|coord_3|>, <|coord_4|>]}, "
        '{"desc": "dog", "bbox_2d": '
        "[<|coord_10|>, <|coord_20|>, <|coord_30|>, <|coord_40|>]}]}"
    )
    assert rendered.terminal_close_span.text(rendered.text) == "]}"
    assert rendered.stop_marker_spans == ()


def test_stage1_json_pretty_exposes_authoritative_object_spans() -> None:
    rendered = Stage1JsonPrettyTemplate().render_assistant(_sample())
    first, second = rendered.object_entries

    assert first.object_instance_id == "img-17:ann-101:src-2"
    assert first.object_index == 0
    assert first.source_object_index == 2
    assert first.entry_span.text(rendered.text) == (
        '{"desc": "cat", "bbox_2d": '
        "[<|coord_1|>, <|coord_2|>, <|coord_3|>, <|coord_4|>]}"
    )
    assert first.desc_span.text(rendered.text) == "cat"
    assert first.object_ref_start_span is None
    assert first.bbox_start_span.text(rendered.text) == '"bbox_2d": ['
    assert first.bbox_opener_span.text(rendered.text) == '"bbox_2d": ['
    assert [span.text(rendered.text) for span in first.coord_spans] == [
        "<|coord_1|>",
        "<|coord_2|>",
        "<|coord_3|>",
        "<|coord_4|>",
    ]
    assert [span.text(rendered.text) for span in first.coordinate_spans] == [
        "<|coord_1|>",
        "<|coord_2|>",
        "<|coord_3|>",
        "<|coord_4|>",
    ]
    assert [span.label for span in first.control_spans] == [
        "object_open",
        "desc_key",
        "field_separator",
        "bbox_start",
        "coordinate_separator",
        "coordinate_separator",
        "coordinate_separator",
        "bbox_close",
        "object_close",
    ]
    assert first.trie_eligible_span == first.entry_span
    assert rendered.separator_spans[0].text(rendered.text) == ", "
    assert first.separator_span == rendered.separator_spans[0]

    assert second.object_instance_id == "img-17:ann-102:src-4"
    assert second.object_index == 1
    assert second.source_object_index == 4
    assert second.separator_span is None
    assert rendered.trie_eligible_spans == (
        first.trie_eligible_span,
        second.trie_eligible_span,
    )
    assert [span.label for span in rendered.structural_token_spans] == [
        "json_root_open",
        "objects_key",
        "json_array_open",
        "object_open",
        "desc_key",
        "field_separator",
        "bbox_start",
        "coordinate_separator",
        "coordinate_separator",
        "coordinate_separator",
        "bbox_close",
        "object_close",
        "object_separator",
        "object_open",
        "desc_key",
        "field_separator",
        "bbox_start",
        "coordinate_separator",
        "coordinate_separator",
        "coordinate_separator",
        "bbox_close",
        "object_close",
        "json_array_close",
        "json_root_close",
    ]


def test_stage1_json_pretty_strict_parser_round_trips_rendered_text() -> None:
    template = Stage1JsonPrettyTemplate()
    rendered = template.render_assistant(_sample())

    assert template.parse_assistant(rendered.text) == {
        "objects": [
            {
                "desc": "cat",
                "bbox_2d": [
                    "<|coord_1|>",
                    "<|coord_2|>",
                    "<|coord_3|>",
                    "<|coord_4|>",
                ],
            },
            {
                "desc": "dog",
                "bbox_2d": [
                    "<|coord_10|>",
                    "<|coord_20|>",
                    "<|coord_30|>",
                    "<|coord_40|>",
                ],
            },
        ]
    }

    with pytest.raises(ValueError, match="canonical stage1_json_pretty"):
        template.parse_assistant(rendered.text.replace('"desc"', '"bbox_2d"', 1))

    with pytest.raises(ValueError, match="strict terminal closure"):
        template.parse_assistant(rendered.text[:-2])


def test_stage1_json_pretty_renders_detection_scene_with_sample_semantic_parity() -> None:
    template = Stage1JsonPrettyTemplate()
    sample = _sample()
    scene = detection_scene_from_normalized_sample_bridge(
        sample,
        image_reference="/resolved/image.jpg",
    )

    rendered_from_sample = template.render_assistant(sample)
    rendered_from_scene = template.render_assistant(scene)

    assert rendered_from_scene.text == rendered_from_sample.text
    assert template.parse_assistant(rendered_from_scene.text) == template.parse_assistant(
        rendered_from_sample.text
    )
    assert rendered_from_scene.object_entries == rendered_from_sample.object_entries
    assert rendered_from_scene.render_span_events == rendered_from_sample.render_span_events


@pytest.mark.parametrize(
    ("bad_coord", "error_match"),
    [
        ("<|coord_1000|>", "coord tokens"),
        ("<|coord_12345|>", "coord tokens"),
        ("<|coord_001|>", "coord tokens"),
        ("<|coord_x|>", "canonical stage1_json_pretty"),
        ("<|coord_1|>junk", "canonical stage1_json_pretty"),
    ],
)
def test_stage1_json_pretty_strict_parser_rejects_non_canonical_coord_tokens(
    bad_coord: str, error_match: str
) -> None:
    template = Stage1JsonPrettyTemplate()
    rendered = template.render_assistant(_sample())

    with pytest.raises(ValueError, match=error_match):
        template.parse_assistant(rendered.text.replace("<|coord_1|>", bad_coord, 1))


@pytest.mark.parametrize(
    "bad_desc",
    [
        'bad " quote',
        "bad \\ slash",
        "bad\nrow",
        "bad\trow",
        "bad\x1fcontrol",
    ],
)
def test_stage1_json_pretty_rejects_desc_values_that_escape_json_spans(
    bad_desc: str,
) -> None:
    sample = _sample()
    sample = replace(sample, objects=(replace(sample.objects[0], desc=bad_desc),))

    with pytest.raises(ValueError, match="stage1_json_pretty desc"):
        Stage1JsonPrettyTemplate().render_assistant(sample)


def test_stage1_json_pretty_allows_literal_non_ascii_desc_span() -> None:
    sample = _sample()
    sample = replace(sample, objects=(replace(sample.objects[0], desc="café"),))
    rendered = Stage1JsonPrettyTemplate().render_assistant(sample)

    assert rendered.object_entries[0].desc_span.text(rendered.text) == "café"
    assert '"desc": "café"' in rendered.text


def test_stage1_json_pretty_preserves_distinct_repeated_object_instances() -> None:
    rendered = Stage1JsonPrettyTemplate().render_assistant(_repeated_identical_sample())
    first, second = rendered.object_entries

    assert first.entry_span.text(rendered.text) == second.entry_span.text(rendered.text)
    assert first.desc_span.text(rendered.text) == "cat"
    assert second.desc_span.text(rendered.text) == "cat"
    assert first.object_instance_id == "img-17:ann-101:src-2"
    assert second.object_instance_id == "img-17:ann-103:src-8"
    assert first.source_object_index == 2
    assert second.source_object_index == 8
    assert first.entry_span != second.entry_span
    assert first.desc_span != second.desc_span
