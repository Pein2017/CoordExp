from dataclasses import replace

import pytest

from src.common.detection_sequence import (
    BOX_START_TOKEN,
    END_OF_TEXT_TOKEN,
    OBJECT_REF_START_TOKEN,
)
from src.detection.data import (
    CoordinateTokenBox,
    DetectionMetadata,
    NormalizedDetectionObject,
    NormalizedDetectionSample,
    ObjectOrderingPlan,
)
from src.detection.template import CompactFullTemplate


def _sample(desc: str = "traffic light") -> NormalizedDetectionSample:
    return NormalizedDetectionSample(
        images=("image.jpg",),
        objects=(
            NormalizedDetectionObject(
                normalized_object_index=0,
                source_object_index=7,
                object_instance_id="img-9:ann-501:src-7",
                desc=desc,
                bbox_2d=CoordinateTokenBox(
                    "<|coord_10|>",
                    "<|coord_20|>",
                    "<|coord_30|>",
                    "<|coord_40|>",
                ),
                category_id=10,
                category_name="traffic light",
                coco_ann_id=501,
            ),
            NormalizedDetectionObject(
                normalized_object_index=1,
                source_object_index=3,
                object_instance_id="img-9:ann-502:src-3",
                desc="person",
                bbox_2d=CoordinateTokenBox(
                    "<|coord_100|>",
                    "<|coord_200|>",
                    "<|coord_300|>",
                    "<|coord_400|>",
                ),
                category_id=1,
                category_name="person",
                coco_ann_id=502,
            ),
        ),
        width=640,
        height=480,
        image_id=9,
        file_name="image.jpg",
        metadata=DetectionMetadata(source="unit", split="test"),
        object_ordering=ObjectOrderingPlan.sorted().with_realized((7, 3)),
    )


def _repeated_identical_sample() -> NormalizedDetectionSample:
    sample = _sample()
    first = sample.objects[0]
    repeated = replace(
        first,
        normalized_object_index=1,
        source_object_index=8,
        object_instance_id="img-9:ann-503:src-8",
        coco_ann_id=503,
    )
    return replace(
        sample,
        objects=(first, repeated),
        object_ordering=ObjectOrderingPlan.sorted().with_realized((7, 8)),
    )


def test_compact_full_renders_approved_token_grammar_without_json_closure() -> None:
    rendered = CompactFullTemplate().render_assistant(_sample())

    assert rendered.text == (
        f"{OBJECT_REF_START_TOKEN}traffic light{BOX_START_TOKEN}"
        "<|coord_10|><|coord_20|><|coord_30|><|coord_40|>"
        f"{OBJECT_REF_START_TOKEN}person{BOX_START_TOKEN}"
        "<|coord_100|><|coord_200|><|coord_300|><|coord_400|>"
    )
    assert "\n" not in rendered.text
    assert '{"objects"' not in rendered.text
    assert "]}" not in rendered.text
    assert rendered.separator_spans[0].text(rendered.text) == ""
    assert rendered.terminal_close_span.start == len(rendered.text)
    assert rendered.terminal_close_span.end == len(rendered.text)
    assert rendered.terminal_close_span.text(rendered.text) == ""
    assert rendered.stop_marker_spans == ()


def test_compact_full_exposes_entry_marker_coordinate_and_trie_spans() -> None:
    rendered = CompactFullTemplate().render_assistant(_sample())
    first, second = rendered.object_entries

    assert first.object_instance_id == "img-9:ann-501:src-7"
    assert first.object_index == 0
    assert first.source_object_index == 7
    assert first.entry_span.text(rendered.text) == (
        f"{OBJECT_REF_START_TOKEN}traffic light{BOX_START_TOKEN}"
        "<|coord_10|><|coord_20|><|coord_30|><|coord_40|>"
    )
    assert first.desc_span.text(rendered.text) == "traffic light"
    assert first.object_ref_start_span is not None
    assert first.object_ref_start_span.text(rendered.text) == OBJECT_REF_START_TOKEN
    assert first.bbox_start_span.text(rendered.text) == BOX_START_TOKEN
    assert first.bbox_opener_span.text(rendered.text) == BOX_START_TOKEN
    assert [span.text(rendered.text) for span in first.coord_spans] == [
        "<|coord_10|>",
        "<|coord_20|>",
        "<|coord_30|>",
        "<|coord_40|>",
    ]
    assert [span.text(rendered.text) for span in first.coordinate_spans] == [
        "<|coord_10|>",
        "<|coord_20|>",
        "<|coord_30|>",
        "<|coord_40|>",
    ]
    assert [span.text(rendered.text) for span in first.control_spans] == [
        OBJECT_REF_START_TOKEN,
        BOX_START_TOKEN,
    ]
    assert first.structural_token_spans == first.control_spans
    assert first.separator_span == rendered.separator_spans[0]
    assert first.trie_eligible_span == first.entry_span

    assert second.object_instance_id == "img-9:ann-502:src-3"
    assert second.object_index == 1
    assert second.source_object_index == 3
    assert second.separator_span is None
    assert rendered.trie_eligible_spans == (
        first.trie_eligible_span,
        second.trie_eligible_span,
    )
    assert [span.label for span in rendered.structural_token_spans] == [
        "object_ref_start",
        "bbox_start",
        "object_separator",
        "object_ref_start",
        "bbox_start",
    ]


def test_compact_full_strict_parser_round_trips_rendered_text() -> None:
    template = CompactFullTemplate()
    rendered = template.render_assistant(_sample())

    assert template.parse_assistant(rendered.text) == {
        "objects": [
            {
                "desc": "traffic light",
                "bbox_2d": [
                    "<|coord_10|>",
                    "<|coord_20|>",
                    "<|coord_30|>",
                    "<|coord_40|>",
                ],
            },
            {
                "desc": "person",
                "bbox_2d": [
                    "<|coord_100|>",
                    "<|coord_200|>",
                    "<|coord_300|>",
                    "<|coord_400|>",
                ],
            },
        ]
    }

    with pytest.raises(ValueError, match="strict compact_full"):
        template.parse_assistant(rendered.text + "\n")

    with pytest.raises(ValueError, match="strict compact_full"):
        template.parse_assistant(rendered.text.replace(OBJECT_REF_START_TOKEN, "", 1))

    for bad_coord in (
        "<|coord_1000|>",
        "<|coord_12345|>",
        "<|coord_001|>",
        "<|coord_x|>",
        "<|coord_10|>junk",
    ):
        with pytest.raises(ValueError, match="strict compact_full"):
            template.parse_assistant(
                rendered.text.replace("<|coord_10|>", bad_coord, 1)
            )


@pytest.mark.parametrize(
    "bad_desc",
    [
        "bad\nrow",
        "bad\trow",
        f"bad {OBJECT_REF_START_TOKEN}",
        f"bad {BOX_START_TOKEN}",
        "bad <|coord_1|>",
        "bad <|im_start|>",
        "bad <|im_end|>",
        f"bad {END_OF_TEXT_TOKEN}",
    ],
)
def test_compact_full_rejects_desc_values_that_collide_with_grammar(
    bad_desc: str,
) -> None:
    with pytest.raises(ValueError, match="compact_full desc"):
        CompactFullTemplate().render_assistant(_sample(desc=bad_desc))


@pytest.mark.parametrize(
    "bbox_2d",
    [
        CoordinateTokenBox(
            "<|coord_30|>",
            "<|coord_20|>",
            "<|coord_10|>",
            "<|coord_40|>",
        ),
        CoordinateTokenBox(
            "<|coord_10|>",
            "<|coord_20|>",
            "<|coord_10|>",
            "<|coord_40|>",
        ),
        CoordinateTokenBox(
            "<|coord_10|>",
            "<|coord_40|>",
            "<|coord_30|>",
            "<|coord_40|>",
        ),
    ],
)
def test_compact_full_render_rejects_non_positive_area_boxes(
    bbox_2d: CoordinateTokenBox,
) -> None:
    sample = _sample()
    bad_object = replace(sample.objects[0], bbox_2d=bbox_2d)
    bad_sample = replace(sample, objects=(bad_object,))

    with pytest.raises(ValueError, match="valid xyxy positive-area box"):
        CompactFullTemplate().render_assistant(bad_sample)


def test_compact_full_preserves_distinct_repeated_object_instances() -> None:
    rendered = CompactFullTemplate().render_assistant(_repeated_identical_sample())
    first, second = rendered.object_entries

    assert first.entry_span.text(rendered.text) == second.entry_span.text(rendered.text)
    assert first.desc_span.text(rendered.text) == "traffic light"
    assert second.desc_span.text(rendered.text) == "traffic light"
    assert first.object_instance_id == "img-9:ann-501:src-7"
    assert second.object_instance_id == "img-9:ann-503:src-8"
    assert first.source_object_index == 7
    assert second.source_object_index == 8
    assert first.entry_span != second.entry_span
    assert first.desc_span != second.desc_span
