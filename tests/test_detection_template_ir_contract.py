from dataclasses import fields, replace
from typing import get_args

import pytest

from src.common.detection_sequence import (
    ALLOWED_DETECTION_SEQUENCE_FORMATS as COMMON_FORMATS,
    BOX_START_TOKEN,
    COMPACT_MIN_FORMAT,
    COMPACT_NO_BBOX_FORMAT,
    COMPACT_NO_DESC_FORMAT,
    OBJECT_REF_START_TOKEN,
    parse_compact_detection_sequence,
    render_compact_detection_sequence,
)
from src.detection.data import (
    CoordinateTokenBox,
    DetectionMetadata,
    NormalizedDetectionObject,
    NormalizedDetectionSample,
    ObjectOrderingPlan,
)
from src.detection.template import (
    CharSpan,
    CompactFullTemplate,
    RenderSpanEvent,
    Stage1JsonPrettyTemplate,
    TemplateId,
    _assert_no_equal_priority_classifying_overlaps,
    get_detection_template,
)
from src.utils.assistant_json import dumps_coordjson


def _sample() -> NormalizedDetectionSample:
    return NormalizedDetectionSample(
        images=("image.jpg",),
        objects=(
            NormalizedDetectionObject(
                normalized_object_index=0,
                source_object_index=7,
                object_instance_id="img-9:ann-501:src-7",
                desc="cat",
                bbox_2d=CoordinateTokenBox(
                    "<|coord_1|>",
                    "<|coord_2|>",
                    "<|coord_3|>",
                    "<|coord_4|>",
                ),
                category_id=17,
                category_name="cat",
                coco_ann_id=501,
            ),
        ),
        width=640,
        height=480,
        image_id=9,
        file_name="image.jpg",
        metadata=DetectionMetadata(source="unit", split="test"),
        object_ordering=ObjectOrderingPlan.sorted().with_realized((7,)),
    )


def _two_object_sample() -> NormalizedDetectionSample:
    sample = _sample()
    return NormalizedDetectionSample(
        images=sample.images,
        objects=(
            sample.objects[0],
            NormalizedDetectionObject(
                normalized_object_index=1,
                source_object_index=8,
                object_instance_id="img-9:ann-502:src-8",
                desc="dog",
                bbox_2d=CoordinateTokenBox(
                    "<|coord_10|>",
                    "<|coord_20|>",
                    "<|coord_30|>",
                    "<|coord_40|>",
                ),
                category_id=18,
                category_name="dog",
                coco_ann_id=502,
            ),
        ),
        width=sample.width,
        height=sample.height,
        image_id=sample.image_id,
        file_name=sample.file_name,
        metadata=sample.metadata,
        object_ordering=ObjectOrderingPlan.sorted().with_realized((7, 8)),
    )


def _payload() -> dict[str, list[dict[str, object]]]:
    return {
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


def _two_object_payload() -> dict[str, list[dict[str, object]]]:
    payload = _payload()
    return {
        "objects": [
            payload["objects"][0],
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


def _events(
    rendered,
    *,
    span_kind: str | None = None,
    primary_role: str | None = None,
) -> tuple[RenderSpanEvent, ...]:
    return tuple(
        event
        for event in rendered.render_span_events
        if (span_kind is None or event.span_kind == span_kind)
        and (primary_role is None or event.primary_role == primary_role)
    )


def _find_event(
    rendered,
    *,
    span_kind: str,
    primary_role: str | None = None,
) -> RenderSpanEvent:
    matches = _events(rendered, span_kind=span_kind, primary_role=primary_role)
    assert len(matches) == 1
    return matches[0]


def _has_primary_role(rendered, role: str) -> bool:
    return any(event.primary_role == role for event in rendered.render_span_events)


def _overlaps(left: RenderSpanEvent, right: RenderSpanEvent) -> bool:
    return (
        left.char_span.start < left.char_span.end
        and right.char_span.start < right.char_span.end
        and left.char_span.start < right.char_span.end
        and right.char_span.start < left.char_span.end
    )


def _equal_priority_classifying_overlaps(
    events: tuple[RenderSpanEvent, ...],
) -> list[tuple[RenderSpanEvent, RenderSpanEvent]]:
    conflicts: list[tuple[RenderSpanEvent, RenderSpanEvent]] = []
    for left_index, left in enumerate(events):
        if not left.classifying:
            continue
        for right in events[left_index + 1 :]:
            if (
                right.classifying
                and left.priority == right.priority
                and _overlaps(left, right)
            ):
                conflicts.append((left, right))
    return conflicts


def test_render_span_event_preserves_legacy_positional_field_order() -> None:
    field_names = tuple(field.name for field in fields(RenderSpanEvent))

    assert field_names[
        field_names.index("source_object_index") + 1 :
        field_names.index("provenance") + 1
    ] == ("geometry_kind", "slot_name", "provenance")
    assert all(
        field_names.index(field_name) > field_names.index("provenance")
        for field_name in (
            "object_id",
            "supervision_key",
            "span_family",
            "field_name",
            "source_role",
            "relation_snapshot",
            "coordinate_weight",
            "regression_weight",
            "hard_bbox_supervision",
        )
    )


def test_stage1_json_pretty_render_bytes_stay_canonical() -> None:
    rendered = Stage1JsonPrettyTemplate().render_assistant(_sample())

    assert rendered.template_id == "stage1_json_pretty"
    assert rendered.text == dumps_coordjson(_payload())
    assert rendered.text == (
        '{"objects": [{"desc": "cat", "bbox_2d": '
        "[<|coord_1|>, <|coord_2|>, <|coord_3|>, <|coord_4|>]}]}"
    )
    assert rendered.terminal_close_span.text(rendered.text) == "]}"
    assert rendered.stop_marker_spans == ()


def test_stage1_json_pretty_two_object_render_bytes_and_separator_stay_canonical() -> None:
    rendered = Stage1JsonPrettyTemplate().render_assistant(_two_object_sample())

    assert rendered.text == dumps_coordjson(_two_object_payload())
    assert rendered.text == (
        '{"objects": [{"desc": "cat", "bbox_2d": '
        "[<|coord_1|>, <|coord_2|>, <|coord_3|>, <|coord_4|>]}, "
        '{"desc": "dog", "bbox_2d": '
        "[<|coord_10|>, <|coord_20|>, <|coord_30|>, <|coord_40|>]}]}"
    )
    assert len(rendered.object_entries) == 2
    assert len(rendered.separator_spans) == 1
    assert rendered.separator_spans[0].text(rendered.text) == ", "
    assert rendered.object_entries[0].separator_span == rendered.separator_spans[0]
    assert rendered.object_entries[0].entry_span.text(rendered.text) == (
        '{"desc": "cat", "bbox_2d": '
        "[<|coord_1|>, <|coord_2|>, <|coord_3|>, <|coord_4|>]}"
    )
    assert rendered.object_entries[1].entry_span.text(rendered.text) == (
        '{"desc": "dog", "bbox_2d": '
        "[<|coord_10|>, <|coord_20|>, <|coord_30|>, <|coord_40|>]}"
    )


def test_compact_full_render_bytes_stay_exact() -> None:
    rendered = CompactFullTemplate().render_assistant(_sample())

    assert rendered.template_id == "compact_full"
    assert rendered.text == (
        f"{OBJECT_REF_START_TOKEN}cat{BOX_START_TOKEN}"
        "<|coord_1|><|coord_2|><|coord_3|><|coord_4|>"
    )
    assert rendered.separator_spans == ()
    assert rendered.terminal_close_span.start == len(rendered.text)
    assert rendered.terminal_close_span.end == len(rendered.text)
    assert rendered.terminal_close_span.text(rendered.text) == ""
    assert rendered.stop_marker_spans == ()


def test_compact_full_two_object_render_bytes_and_separator_stay_exact() -> None:
    rendered = CompactFullTemplate().render_assistant(_two_object_sample())

    assert rendered.text == (
        f"{OBJECT_REF_START_TOKEN}cat{BOX_START_TOKEN}"
        "<|coord_1|><|coord_2|><|coord_3|><|coord_4|>"
        f"{OBJECT_REF_START_TOKEN}dog{BOX_START_TOKEN}"
        "<|coord_10|><|coord_20|><|coord_30|><|coord_40|>"
    )
    assert len(rendered.object_entries) == 2
    assert len(rendered.separator_spans) == 1
    assert rendered.separator_spans[0].text(rendered.text) == ""
    assert rendered.object_entries[0].separator_span == rendered.separator_spans[0]
    assert rendered.object_entries[0].entry_span.text(rendered.text) == (
        f"{OBJECT_REF_START_TOKEN}cat{BOX_START_TOKEN}"
        "<|coord_1|><|coord_2|><|coord_3|><|coord_4|>"
    )
    assert rendered.object_entries[1].entry_span.text(rendered.text) == (
        f"{OBJECT_REF_START_TOKEN}dog{BOX_START_TOKEN}"
        "<|coord_10|><|coord_20|><|coord_30|><|coord_40|>"
    )
    assert rendered.terminal_close_span.start == len(rendered.text)
    assert rendered.terminal_close_span.end == len(rendered.text)


def test_object_separator_events_follow_previous_entry_projection_for_json_and_compact() -> None:
    for template in (Stage1JsonPrettyTemplate(), CompactFullTemplate()):
        rendered = template.render_assistant(_two_object_sample())
        previous_entry = rendered.object_entries[0]

        event = _find_event(
            rendered,
            span_kind="object_separator",
            primary_role="SEPARATOR",
        )

        assert event.char_span == rendered.separator_spans[0]
        assert event.char_span == previous_entry.separator_span
        assert event.object_instance_id == previous_entry.object_instance_id
        assert event.object_index == previous_entry.object_index
        assert event.source_object_index == previous_entry.source_object_index
        assert event.mask_groups == frozenset({"separator", "control"})
        assert event.provenance == "rendered_control"


def test_shared_render_projection_keeps_trie_and_object_container_fields_for_json_and_compact() -> None:
    for template in (Stage1JsonPrettyTemplate(), CompactFullTemplate()):
        rendered = template.render_assistant(_two_object_sample())
        object_container_events = _events(
            rendered,
            span_kind="object_entry_container",
        )

        assert rendered.trie_eligible_spans == tuple(
            entry.trie_eligible_span for entry in rendered.object_entries
        )
        assert tuple(event.char_span for event in object_container_events) == tuple(
            entry.entry_span for entry in rendered.object_entries
        )
        assert tuple(event.object_instance_id for event in object_container_events) == tuple(
            entry.object_instance_id for entry in rendered.object_entries
        )


def test_desc_and_coord_spans_are_not_shadowed_by_container_spans() -> None:
    rendered = CompactFullTemplate().render_assistant(_sample())

    assert _equal_priority_classifying_overlaps(rendered.render_span_events) == []
    assert _has_primary_role(rendered, "DESC")
    assert _has_primary_role(rendered, "COORD")

    container_kinds = {"assistant_container", "object_entry_container"}
    containers = [
        event
        for event in rendered.render_span_events
        if event.span_kind in container_kinds
    ]
    assert {event.span_kind for event in containers} == container_kinds
    assert all(event.classifying is False for event in containers)
    assert all(event.primary_role is None for event in containers)


def test_bbox_start_is_primary_role_and_schema_mask_group() -> None:
    rendered = CompactFullTemplate().render_assistant(_sample())

    event = _find_event(rendered, span_kind="bbox_start_marker")

    assert event.char_span.text(rendered.text) == BOX_START_TOKEN
    assert event.primary_role == "BBOX_START"
    assert "schema" in event.mask_groups
    assert "control" in event.mask_groups
    assert event.classifying is True


def test_coordinate_slot_events_carry_slot_names_and_coord_role() -> None:
    rendered = CompactFullTemplate().render_assistant(_sample())

    events = _events(rendered, span_kind="coordinate_slot", primary_role="COORD")

    assert tuple(event.slot_name for event in events) == ("x1", "y1", "x2", "y2")
    assert tuple(event.char_span.text(rendered.text) for event in events) == (
        "<|coord_1|>",
        "<|coord_2|>",
        "<|coord_3|>",
        "<|coord_4|>",
    )
    assert all(event.geometry_kind == "bbox_2d" for event in events)
    assert all(event.classifying is True for event in events)


def test_terminal_close_categories_cover_json_and_compact_zero_length() -> None:
    json_rendered = Stage1JsonPrettyTemplate().render_assistant(_sample())
    compact_rendered = CompactFullTemplate().render_assistant(_sample())

    json_terminal = _find_event(json_rendered, span_kind="terminal_close")
    compact_terminal = _find_event(compact_rendered, span_kind="terminal_close")

    assert json_terminal.primary_role == "TERMINAL"
    assert json_terminal.char_span.text(json_rendered.text) == "]}"
    assert json_terminal.char_span.start < json_terminal.char_span.end
    assert compact_terminal.primary_role == "TERMINAL"
    assert compact_terminal.char_span.text(compact_rendered.text) == ""
    assert compact_terminal.char_span.start == compact_terminal.char_span.end


def test_template_render_events_reserve_chat_stop_marker_for_encoded_projection() -> None:
    json_rendered = Stage1JsonPrettyTemplate().render_assistant(_sample())
    compact_rendered = CompactFullTemplate().render_assistant(_sample())

    assert _events(json_rendered, span_kind="terminal_close")
    assert _events(compact_rendered, span_kind="terminal_close")
    assert _events(json_rendered, span_kind="chat_stop_marker") == ()
    assert _events(compact_rendered, span_kind="chat_stop_marker") == ()


def test_equal_priority_classifying_overlap_invariant_reports_conflict() -> None:
    left = RenderSpanEvent(
        char_span=CharSpan(0, 3, "left_desc"),
        span_kind="description_text",
        primary_role="DESC",
        mask_groups=frozenset({"desc"}),
        classifying=True,
        priority=90,
    )
    right = RenderSpanEvent(
        char_span=CharSpan(1, 2, "right_desc"),
        span_kind="description_text",
        primary_role="DESC",
        mask_groups=frozenset({"desc"}),
        classifying=True,
        priority=90,
    )

    with pytest.raises(ValueError, match="equal-priority classifying overlap"):
        _assert_no_equal_priority_classifying_overlaps((left, right))


def test_json_desc_quotes_are_control_events() -> None:
    rendered = Stage1JsonPrettyTemplate().render_assistant(_sample())
    entry = rendered.object_entries[0]
    quote_spans = {
        (entry.desc_span.start - 1, entry.desc_span.start),
        (entry.desc_span.end, entry.desc_span.end + 1),
    }

    control_quote_spans = {
        (event.char_span.start, event.char_span.end)
        for event in rendered.render_span_events
        if event.span_kind == "json_punctuation"
        and event.primary_role == "CONTROL"
        and event.char_span.text(rendered.text) == '"'
    }

    assert quote_spans <= control_quote_spans


def test_json_render_span_events_cover_leaf_and_schema_roles() -> None:
    rendered = Stage1JsonPrettyTemplate().render_assistant(_sample())

    assert _equal_priority_classifying_overlaps(rendered.render_span_events) == []
    assert _has_primary_role(rendered, "DESC")
    assert _has_primary_role(rendered, "COORD")
    assert _has_primary_role(rendered, "BBOX_START")
    assert _events(rendered, span_kind="bbox_field_binding", primary_role="BBOX_START")
    assert _events(rendered, span_kind="json_key", primary_role="CONTROL")
    assert _events(rendered, span_kind="json_punctuation", primary_role="CONTROL")
    assert _events(rendered, span_kind="coordinate_separator", primary_role="SEPARATOR")


def test_compact_full_template_matches_common_render_helper() -> None:
    rendered = CompactFullTemplate().render_assistant(_two_object_sample())

    assert rendered.text == render_compact_detection_sequence(
        _two_object_payload(),
        detection_sequence_format="compact_full",
    )


def test_existing_parse_behavior_stays_strict_for_stage1_json_pretty() -> None:
    template = Stage1JsonPrettyTemplate()
    rendered = template.render_assistant(_sample())

    assert template.parse_assistant(rendered.text) == _payload()
    with pytest.raises(ValueError, match="strict terminal closure"):
        template.parse_assistant(rendered.text.removesuffix("]}"))
    with pytest.raises(ValueError, match="canonical stage1_json_pretty"):
        template.parse_assistant(rendered.text.replace('", "bbox_2d"', '",  "bbox_2d"'))


def test_existing_parse_behavior_stays_strict_for_compact_full() -> None:
    template = CompactFullTemplate()
    rendered = template.render_assistant(_sample())

    assert template.parse_assistant(rendered.text) == _payload()
    # Compact parser messages are intentionally not pinned here; only strict
    # rejection of malformed compact rows is part of this freeze.
    with pytest.raises(ValueError):
        template.parse_assistant(rendered.text.replace(BOX_START_TOKEN, ""))
    with pytest.raises(ValueError):
        template.parse_assistant(rendered.text.replace("<|coord_2|>", " <|coord_2|>"))


def test_compact_full_template_parse_matches_common_parse_for_fixture() -> None:
    template = CompactFullTemplate()
    rendered = template.render_assistant(_two_object_sample())

    assert template.parse_assistant(rendered.text) == parse_compact_detection_sequence(
        rendered.text,
        detection_sequence_format="compact_full",
    )


def test_common_compact_parser_keeps_suffix_stripping_compatibility() -> None:
    template = CompactFullTemplate()
    rendered = template.render_assistant(_sample())
    generated_text = f"{rendered.text}  <|im_end|>ignored suffix"

    assert parse_compact_detection_sequence(
        generated_text,
        detection_sequence_format="compact_full",
    ) == _payload()
    with pytest.raises(ValueError):
        template.parse_assistant(generated_text)


def test_common_compact_parser_returns_none_for_diagnostic_failures() -> None:
    template = CompactFullTemplate()
    rendered = template.render_assistant(_sample())
    malformed_row = rendered.text.replace("<|coord_2|>", " <|coord_2|>")

    assert (
        parse_compact_detection_sequence(
            malformed_row,
            detection_sequence_format="compact_full",
        )
        is None
    )
    with pytest.raises(ValueError):
        template.parse_assistant(malformed_row)


@pytest.mark.parametrize("bad_coord", ["<|coord_1000|>", "<|coord_01|>"])
def test_common_and_strict_compact_parsers_reject_invalid_coord_tokens(
    bad_coord: str,
) -> None:
    template = CompactFullTemplate()
    rendered = template.render_assistant(_sample())
    malformed_row = rendered.text.replace("<|coord_2|>", bad_coord)

    assert (
        parse_compact_detection_sequence(
            malformed_row,
            detection_sequence_format="compact_full",
        )
        is None
    )
    with pytest.raises(ValueError):
        template.parse_assistant(malformed_row)


@pytest.mark.parametrize("bad_control", ["\n", "\r", "\t"])
def test_common_and_strict_compact_renderers_reject_desc_control_characters(
    bad_control: str,
) -> None:
    sample = _sample()
    bad_object = replace(sample.objects[0], desc=f"bad{bad_control}desc")
    bad_sample = replace(sample, objects=(bad_object,))
    payload = {
        "objects": [
            {
                "desc": bad_object.desc,
                "bbox_2d": list(bad_object.bbox_2d.tokens),
            }
        ]
    }

    with pytest.raises(ValueError):
        render_compact_detection_sequence(
            payload,
            detection_sequence_format="compact_full",
        )
    with pytest.raises(ValueError):
        CompactFullTemplate().render_assistant(bad_sample)


def test_common_marker_omission_variants_are_not_training_templates() -> None:
    first_class_training_template_ids = set(get_args(TemplateId))
    common_helper_only_formats = (
        COMPACT_NO_DESC_FORMAT,
        COMPACT_NO_BBOX_FORMAT,
        COMPACT_MIN_FORMAT,
    )

    for detection_sequence_format in common_helper_only_formats:
        assert detection_sequence_format in COMMON_FORMATS
        assert detection_sequence_format not in first_class_training_template_ids
        with pytest.raises(ValueError):
            get_detection_template(detection_sequence_format)


def test_common_marker_omission_variants_remain_common_facade_only() -> None:
    common_helper_only_formats = (
        COMPACT_NO_DESC_FORMAT,
        COMPACT_NO_BBOX_FORMAT,
        COMPACT_MIN_FORMAT,
    )

    for detection_sequence_format in common_helper_only_formats:
        rendered = render_compact_detection_sequence(
            _payload(),
            detection_sequence_format=detection_sequence_format,
        )

        assert parse_compact_detection_sequence(
            rendered,
            detection_sequence_format=detection_sequence_format,
        ) == _payload()
