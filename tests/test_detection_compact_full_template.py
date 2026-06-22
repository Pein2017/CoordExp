from dataclasses import replace

import pytest

from src.common.detection_sequence import (
    BOX_END_TOKEN,
    BOX_START_TOKEN,
    END_OF_TEXT_TOKEN,
    OBJECT_REF_END_TOKEN,
    OBJECT_REF_START_TOKEN,
)
from src.detection.data import (
    CoordinateTokenBox,
    DetectionMetadata,
    NormalizedDetectionObject,
    NormalizedDetectionSample,
    ObjectOrderingPlan,
)
from src.detection.scene import detection_scene_from_normalized_sample_bridge
from src.detection.template import CompactFullTemplate, get_detection_template


EXPECTED_COMPACT = (
    f"{OBJECT_REF_START_TOKEN}traffic light{BOX_START_TOKEN}"
    "<|coord_10|><|coord_20|><|coord_30|><|coord_40|>"
    f"{OBJECT_REF_START_TOKEN}person{BOX_START_TOKEN}"
    "<|coord_100|><|coord_200|><|coord_300|><|coord_400|>"
)
EXPECTED_BOX_CLOSED = (
    f"{OBJECT_REF_START_TOKEN}traffic light{BOX_START_TOKEN}"
    f"<|coord_10|><|coord_20|><|coord_30|><|coord_40|>{BOX_END_TOKEN}"
    f"{OBJECT_REF_START_TOKEN}person{BOX_START_TOKEN}"
    f"<|coord_100|><|coord_200|><|coord_300|><|coord_400|>{BOX_END_TOKEN}"
)
EXPECTED_OBJECT_CLOSED = (
    f"{OBJECT_REF_START_TOKEN}traffic light{OBJECT_REF_END_TOKEN}{BOX_START_TOKEN}"
    "<|coord_10|><|coord_20|><|coord_30|><|coord_40|>"
    f"{OBJECT_REF_START_TOKEN}person{OBJECT_REF_END_TOKEN}{BOX_START_TOKEN}"
    "<|coord_100|><|coord_200|><|coord_300|><|coord_400|>"
)
EXPECTED_OBJECT_BOX_CLOSED = (
    f"{OBJECT_REF_START_TOKEN}traffic light{OBJECT_REF_END_TOKEN}{BOX_START_TOKEN}"
    f"<|coord_10|><|coord_20|><|coord_30|><|coord_40|>{BOX_END_TOKEN}"
    f"{OBJECT_REF_START_TOKEN}person{OBJECT_REF_END_TOKEN}{BOX_START_TOKEN}"
    f"<|coord_100|><|coord_200|><|coord_300|><|coord_400|>{BOX_END_TOKEN}"
)
EXPECTED_OBJECT_BOX_CLOSED_LINES = (
    f"{OBJECT_REF_START_TOKEN}traffic light{OBJECT_REF_END_TOKEN}{BOX_START_TOKEN}"
    f"<|coord_10|><|coord_20|><|coord_30|><|coord_40|>{BOX_END_TOKEN}\n"
    f"{OBJECT_REF_START_TOKEN}person{OBJECT_REF_END_TOKEN}{BOX_START_TOKEN}"
    f"<|coord_100|><|coord_200|><|coord_300|><|coord_400|>{BOX_END_TOKEN}\n"
)
EXPECTED_DESC_FIRST_RICH_CAT = (
    f"{OBJECT_REF_START_TOKEN}cat{OBJECT_REF_END_TOKEN}"
    f"{BOX_START_TOKEN}<|coord_1|><|coord_2|><|coord_3|><|coord_4|>{BOX_END_TOKEN}"
)
EXPECTED_GEOMETRY_FIRST_RICH_CAT = (
    f"{BOX_START_TOKEN}<|coord_1|><|coord_2|><|coord_3|><|coord_4|>{BOX_END_TOKEN}"
    f"{OBJECT_REF_START_TOKEN}cat{OBJECT_REF_END_TOKEN}"
)
EXPECTED_DESC_FIRST_OBJECT_CLOSED_CAT = (
    f"{OBJECT_REF_START_TOKEN}cat{OBJECT_REF_END_TOKEN}"
    f"{BOX_START_TOKEN}<|coord_1|><|coord_2|><|coord_3|><|coord_4|>"
)
EXPECTED_GEOMETRY_FIRST_OBJECT_CLOSED_CAT = (
    f"{BOX_START_TOKEN}<|coord_1|><|coord_2|><|coord_3|><|coord_4|>"
    f"{OBJECT_REF_START_TOKEN}cat{OBJECT_REF_END_TOKEN}"
)
EXPECTED_GEOMETRY_FIRST_RICH_CAT_LINE = f"{EXPECTED_GEOMETRY_FIRST_RICH_CAT}\n"
EXPECTED_GEOMETRY_FIRST_OBJECT_BOX_CLOSED = (
    f"{BOX_START_TOKEN}<|coord_10|><|coord_20|><|coord_30|><|coord_40|>{BOX_END_TOKEN}"
    f"{OBJECT_REF_START_TOKEN}traffic light{OBJECT_REF_END_TOKEN}"
    f"{BOX_START_TOKEN}<|coord_100|><|coord_200|><|coord_300|><|coord_400|>{BOX_END_TOKEN}"
    f"{OBJECT_REF_START_TOKEN}person{OBJECT_REF_END_TOKEN}"
)
EXPECTED_GEOMETRY_FIRST_OBJECT_BOX_CLOSED_LINES = (
    f"{BOX_START_TOKEN}<|coord_10|><|coord_20|><|coord_30|><|coord_40|>{BOX_END_TOKEN}"
    f"{OBJECT_REF_START_TOKEN}traffic light{OBJECT_REF_END_TOKEN}\n"
    f"{BOX_START_TOKEN}<|coord_100|><|coord_200|><|coord_300|><|coord_400|>{BOX_END_TOKEN}"
    f"{OBJECT_REF_START_TOKEN}person{OBJECT_REF_END_TOKEN}\n"
)


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


def _cat_sample() -> NormalizedDetectionSample:
    return NormalizedDetectionSample(
        images=("image.jpg",),
        objects=(
            NormalizedDetectionObject(
                normalized_object_index=0,
                source_object_index=0,
                object_instance_id="img-10:ann-601:src-0",
                desc="cat",
                bbox_2d=CoordinateTokenBox(
                    "<|coord_1|>",
                    "<|coord_2|>",
                    "<|coord_3|>",
                    "<|coord_4|>",
                ),
                category_id=17,
                category_name="cat",
                coco_ann_id=601,
            ),
        ),
        width=640,
        height=480,
        image_id=10,
        file_name="image.jpg",
        metadata=DetectionMetadata(source="unit", split="test"),
        object_ordering=ObjectOrderingPlan.sorted().with_realized((0,)),
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


@pytest.mark.parametrize(
    ("template_id", "expected"),
    [
        ("compact", EXPECTED_COMPACT),
        ("compact_box_closed", EXPECTED_BOX_CLOSED),
        ("compact_object_closed", EXPECTED_OBJECT_CLOSED),
        ("compact_object_box_closed", EXPECTED_OBJECT_BOX_CLOSED),
        ("compact_object_box_closed_lines", EXPECTED_OBJECT_BOX_CLOSED_LINES),
    ],
)
def test_semantic_compact_templates_render_exact_bytes(
    template_id: str,
    expected: str,
) -> None:
    rendered = get_detection_template(template_id).render_assistant(_sample())

    assert rendered.text == expected
    assert rendered.template_id == template_id


@pytest.mark.parametrize(
    ("object_field_order", "expected"),
    [
        ("desc_first", EXPECTED_DESC_FIRST_RICH_CAT),
        ("geometry_first", EXPECTED_GEOMETRY_FIRST_RICH_CAT),
    ],
)
def test_compact_object_box_closed_renders_exact_bytes_for_field_order(
    object_field_order: str,
    expected: str,
) -> None:
    rendered = get_detection_template("compact_object_box_closed").render_assistant(
        _cat_sample(),
        object_field_order=object_field_order,
    )

    assert rendered.text == expected
    assert rendered.template_id == "compact_object_box_closed"


@pytest.mark.parametrize(
    ("object_field_order", "expected"),
    [
        ("desc_first", EXPECTED_DESC_FIRST_OBJECT_CLOSED_CAT),
        ("geometry_first", EXPECTED_GEOMETRY_FIRST_OBJECT_CLOSED_CAT),
    ],
)
def test_compact_object_closed_renders_exact_bytes_for_field_order(
    object_field_order: str,
    expected: str,
) -> None:
    rendered = get_detection_template("compact_object_closed").render_assistant(
        _cat_sample(),
        object_field_order=object_field_order,
    )

    assert rendered.text == expected
    assert rendered.template_id == "compact_object_closed"


@pytest.mark.parametrize(
    ("template_id", "expected"),
    [
        ("compact", EXPECTED_COMPACT),
        ("compact_box_closed", EXPECTED_BOX_CLOSED),
        ("compact_object_closed", EXPECTED_OBJECT_CLOSED),
        ("compact_object_box_closed", EXPECTED_OBJECT_BOX_CLOSED),
        ("compact_object_box_closed_lines", EXPECTED_OBJECT_BOX_CLOSED_LINES),
    ],
)
def test_semantic_compact_templates_strict_parse_roundtrip(
    template_id: str,
    expected: str,
) -> None:
    template = get_detection_template(template_id)

    assert template.parse_assistant(expected) == {
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


@pytest.mark.parametrize(
    ("template_id", "object_field_order", "expected"),
    [
        (
            "compact_object_closed",
            "desc_first",
            EXPECTED_DESC_FIRST_OBJECT_CLOSED_CAT,
        ),
        (
            "compact_object_closed",
            "geometry_first",
            EXPECTED_GEOMETRY_FIRST_OBJECT_CLOSED_CAT,
        ),
    ],
)
def test_compact_object_closed_strict_parse_respects_field_order(
    template_id: str,
    object_field_order: str,
    expected: str,
) -> None:
    template = get_detection_template(template_id)

    assert template.parse_assistant(
        expected,
        object_field_order=object_field_order,
    ) == {
        "objects": [
            {
                "desc": "cat",
                "bbox_2d": [
                    "<|coord_1|>",
                    "<|coord_2|>",
                    "<|coord_3|>",
                    "<|coord_4|>",
                ],
            }
        ]
    }


def test_semantic_compact_templates_reject_wrong_variant_structure() -> None:
    with pytest.raises(ValueError):
        get_detection_template("compact").parse_assistant(EXPECTED_BOX_CLOSED)

    with pytest.raises(ValueError):
        get_detection_template("compact_box_closed").parse_assistant(EXPECTED_COMPACT)

    with pytest.raises(ValueError):
        get_detection_template("compact_object_box_closed").parse_assistant(
            EXPECTED_OBJECT_BOX_CLOSED_LINES
        )


def test_compact_object_box_closed_strict_parse_splits_geometry_first_rows() -> None:
    template = get_detection_template("compact_object_box_closed")

    assert template.parse_assistant(
        EXPECTED_GEOMETRY_FIRST_OBJECT_BOX_CLOSED,
        object_field_order="geometry_first",
    ) == {
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

    with pytest.raises(ValueError, match="strict compact_object_box_closed"):
        template.parse_assistant(EXPECTED_GEOMETRY_FIRST_OBJECT_BOX_CLOSED)


def test_compact_object_box_closed_lines_strict_parse_geometry_first_rows() -> None:
    template = get_detection_template("compact_object_box_closed_lines")

    assert template.parse_assistant(
        EXPECTED_GEOMETRY_FIRST_OBJECT_BOX_CLOSED_LINES,
        object_field_order="geometry_first",
    )["objects"][1] == {
        "desc": "person",
        "bbox_2d": [
            "<|coord_100|>",
            "<|coord_200|>",
            "<|coord_300|>",
            "<|coord_400|>",
        ],
    }


def test_compact_geometry_first_strict_parse_rejects_malformed_second_row() -> None:
    malformed = EXPECTED_GEOMETRY_FIRST_OBJECT_BOX_CLOSED.replace(
        f"{BOX_START_TOKEN}<|coord_100|>",
        "<|coord_100|>",
        1,
    )

    with pytest.raises(ValueError, match="strict compact_object_box_closed"):
        get_detection_template("compact_object_box_closed").parse_assistant(
            malformed,
            object_field_order="geometry_first",
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
    assert rendered.separator_spans == ()
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
    assert first.separator_span is None
    assert rendered.separator_spans == ()
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
        "object_ref_start",
        "bbox_start",
    ]


def test_compact_object_box_closed_lines_geometry_first_exposes_spans() -> None:
    rendered = get_detection_template("compact_object_box_closed_lines").render_assistant(
        _cat_sample(),
        object_field_order="geometry_first",
    )
    (entry,) = rendered.object_entries

    assert rendered.text == EXPECTED_GEOMETRY_FIRST_RICH_CAT_LINE
    assert entry.entry_span.text(rendered.text) == EXPECTED_GEOMETRY_FIRST_RICH_CAT_LINE
    assert entry.desc_span.text(rendered.text) == "cat"
    assert entry.bbox_span.text(rendered.text) == (
        "<|coord_1|><|coord_2|><|coord_3|><|coord_4|>"
    )
    assert [span.text(rendered.text) for span in entry.coord_spans] == [
        "<|coord_1|>",
        "<|coord_2|>",
        "<|coord_3|>",
        "<|coord_4|>",
    ]
    assert [span.text(rendered.text) for span in entry.control_spans] == [
        BOX_START_TOKEN,
        BOX_END_TOKEN,
        OBJECT_REF_START_TOKEN,
        OBJECT_REF_END_TOKEN,
        "\n",
    ]
    assert entry.trie_eligible_span == entry.entry_span
    assert entry.separator_span is not None
    assert entry.separator_span.text(rendered.text) == "\n"
    assert rendered.separator_spans == ()


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

    with pytest.raises(ValueError, match="strict compact"):
        template.parse_assistant(rendered.text + "\n")

    with pytest.raises(ValueError, match="strict compact"):
        template.parse_assistant(rendered.text.replace(OBJECT_REF_START_TOKEN, "", 1))

    for bad_coord in (
        "<|coord_1000|>",
        "<|coord_12345|>",
        "<|coord_001|>",
        "<|coord_x|>",
        "<|coord_10|>junk",
    ):
        with pytest.raises(ValueError, match="strict compact"):
            template.parse_assistant(
                rendered.text.replace("<|coord_10|>", bad_coord, 1)
            )


def test_compact_full_renders_detection_scene_with_sample_semantic_parity() -> None:
    template = CompactFullTemplate()
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
