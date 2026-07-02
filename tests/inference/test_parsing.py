from __future__ import annotations


def test_valid_compact_object_emits_canonical_prediction() -> None:
    from src.inference.parsing import parse_compact_object_box_closed

    row = parse_compact_object_box_closed(
        "<|object_ref_start|>cat<|object_ref_end|>"
        "<|box_start|><|coord_100|><|coord_200|><|coord_300|><|coord_400|><|box_end|>",
        row_id="row-1",
        row_index=0,
        image_width=1000,
        image_height=500,
    )

    assert row.parse_status == "accepted"
    assert row.metric_bearing is True
    assert row.valid_prediction_count == 1
    assert row.dropped_prediction_count == 0
    prediction = row.predictions[0]
    assert prediction["description"] == "cat"
    assert prediction["bbox"] == [100, 100, 300, 200]
    assert prediction["bbox_format"] == "xyxy"
    assert prediction["coord_bins"] == [100, 200, 300, 400]
    assert prediction["generated_order"] == 0
    assert row.to_artifact_dict()["parser_id"] == "compact-object-box-closed-v1"


def test_json_assistant_response_is_unsupported() -> None:
    from src.inference.parsing import parse_compact_object_box_closed

    row = parse_compact_object_box_closed(
        '{"pred": [{"description": "cat", "bbox": [1, 2, 3, 4]}]}',
        row_id="row-1",
        row_index=0,
        image_width=1000,
        image_height=500,
    )

    assert row.parse_status == "unsupported_format"
    assert row.metric_bearing is False
    assert row.predictions == []
    assert row.dropped_prediction_count == 1
    assert row.dropped_predictions[0]["reason"] == "unsupported_json_response"


def test_generated_prediction_order_is_preserved_without_geo_sorting() -> None:
    from src.inference.parsing import parse_compact_object_box_closed

    row = parse_compact_object_box_closed(
        "<|object_ref_start|>bottom right<|object_ref_end|>"
        "<|box_start|><|coord_800|><|coord_800|><|coord_900|><|coord_900|><|box_end|>"
        "<|object_ref_start|>top left<|object_ref_end|>"
        "<|box_start|><|coord_100|><|coord_100|><|coord_200|><|coord_200|><|box_end|>",
        row_id="row-1",
        row_index=0,
        image_width=1000,
        image_height=1000,
    )

    assert [prediction["description"] for prediction in row.predictions] == [
        "bottom right",
        "top left",
    ]
    assert [prediction["generated_order"] for prediction in row.predictions] == [0, 1]


def test_accepted_with_drops_preserves_valid_objects_and_diagnostics() -> None:
    from src.inference.parsing import parse_compact_object_box_closed

    row = parse_compact_object_box_closed(
        "<|object_ref_start|>cat<|object_ref_end|>"
        "<|box_start|><|coord_100|><|coord_200|><|coord_300|><|coord_400|><|box_end|>"
        "<|object_ref_start|>bad<|object_ref_end|>"
        "<|box_start|><|coord_1|><|coord_2|><|coord_3|><|box_end|>",
        row_id="row-1",
        row_index=0,
        image_width=1000,
        image_height=1000,
    )

    assert row.parse_status == "accepted_with_drops"
    assert row.metric_bearing is True
    assert [prediction["description"] for prediction in row.predictions] == ["cat"]
    assert row.dropped_prediction_count == 1
    assert row.dropped_predictions[0]["reason"] == "malformed_object_span"
    assert row.diagnostics[0]["row_id"] == "row-1"


def test_malformed_span_before_valid_object_does_not_fuse_across_boundary() -> None:
    from src.inference.parsing import parse_compact_object_box_closed

    row = parse_compact_object_box_closed(
        "<|object_ref_start|>bad"
        "<|object_ref_start|>cat<|object_ref_end|>"
        "<|box_start|><|coord_100|><|coord_200|><|coord_300|><|coord_400|><|box_end|>",
        row_id="row-1",
        row_index=0,
        image_width=1000,
        image_height=1000,
    )

    assert row.parse_status == "accepted_with_drops"
    assert [prediction["description"] for prediction in row.predictions] == ["cat"]
    assert row.dropped_predictions[0]["reason"] == "malformed_object_span"
    assert row.dropped_predictions[0]["generated_order"] == 0
    assert row.predictions[0]["generated_order"] == 1


def test_leading_prose_is_preserved_as_drop_with_salvaged_valid_object() -> None:
    from src.inference.parsing import parse_compact_object_box_closed

    row = parse_compact_object_box_closed(
        "Here are the detections: "
        "<|object_ref_start|>cat<|object_ref_end|>"
        "<|box_start|><|coord_100|><|coord_200|><|coord_300|><|coord_400|><|box_end|>",
        row_id="row-1",
        row_index=0,
        image_width=1000,
        image_height=1000,
    )

    assert row.parse_status == "accepted_with_drops"
    assert [prediction["description"] for prediction in row.predictions] == ["cat"]
    assert row.dropped_predictions[0]["reason"] == "unmatched_text"
    assert row.dropped_predictions[0]["raw_text"] == "Here are the detections: "


def test_trailing_prose_is_preserved_as_drop_with_salvaged_valid_object() -> None:
    from src.inference.parsing import parse_compact_object_box_closed

    row = parse_compact_object_box_closed(
        "<|object_ref_start|>cat<|object_ref_end|>"
        "<|box_start|><|coord_100|><|coord_200|><|coord_300|><|coord_400|><|box_end|>"
        " done.",
        row_id="row-1",
        row_index=0,
        image_width=1000,
        image_height=1000,
    )

    assert row.parse_status == "accepted_with_drops"
    assert [prediction["description"] for prediction in row.predictions] == ["cat"]
    assert row.dropped_predictions[0]["reason"] == "unmatched_text"
    assert row.dropped_predictions[0]["raw_text"] == " done."


def test_whitespace_only_prefix_is_ignored() -> None:
    from src.inference.parsing import parse_compact_object_box_closed

    row = parse_compact_object_box_closed(
        "\n  "
        "<|object_ref_start|>cat<|object_ref_end|>"
        "<|box_start|><|coord_100|><|coord_200|><|coord_300|><|coord_400|><|box_end|>",
        row_id="row-1",
        row_index=0,
        image_width=1000,
        image_height=1000,
    )

    assert row.parse_status == "accepted"
    assert row.dropped_predictions == []


def test_terminal_stop_suffix_is_ignored_after_valid_object() -> None:
    from src.inference.parsing import parse_compact_object_box_closed

    row = parse_compact_object_box_closed(
        "<|object_ref_start|>cat<|object_ref_end|>"
        "<|box_start|><|coord_100|><|coord_200|><|coord_300|><|coord_400|><|box_end|>"
        "<|im_end|>\n",
        row_id="row-1",
        row_index=0,
        image_width=1000,
        image_height=1000,
    )

    assert row.parse_status == "accepted"
    assert row.dropped_predictions == []


def test_empty_description_span_is_dropped_with_typed_reason() -> None:
    from src.inference.parsing import parse_compact_object_box_closed

    row = parse_compact_object_box_closed(
        "<|object_ref_start|>   <|object_ref_end|>"
        "<|box_start|><|coord_100|><|coord_200|><|coord_300|><|coord_400|><|box_end|>"
        "<|object_ref_start|>cat<|object_ref_end|>"
        "<|box_start|><|coord_100|><|coord_200|><|coord_300|><|coord_400|><|box_end|>",
        row_id="row-1",
        row_index=0,
        image_width=1000,
        image_height=1000,
    )

    assert row.parse_status == "accepted_with_drops"
    assert [prediction["description"] for prediction in row.predictions] == ["cat"]
    assert row.dropped_predictions[0]["reason"] == "empty_description"
    assert row.dropped_predictions[0]["generated_order"] == 0
    assert row.predictions[0]["generated_order"] == 1


def test_span_handles_and_ranges_are_stable_for_scoring_replay() -> None:
    from src.inference.parsing import parse_compact_object_box_closed

    text = (
        "<|object_ref_start|>bad"
        "<|object_ref_start|>cat<|object_ref_end|>"
        "<|box_start|><|coord_100|><|coord_200|><|coord_300|><|coord_400|><|box_end|>"
        "<|object_ref_start|>cat<|object_ref_end|>"
        "<|box_start|><|coord_100|><|coord_200|><|coord_300|><|coord_400|><|box_end|>"
    )
    row = parse_compact_object_box_closed(
        text,
        row_id="row-1",
        row_index=0,
        image_width=1000,
        image_height=1000,
    )

    span_ids = [
        row.dropped_predictions[0]["object_span_id"],
        row.predictions[0]["object_span_id"],
        row.predictions[1]["object_span_id"],
    ]
    assert span_ids == ["row-1:span-0", "row-1:span-1", "row-1:span-2"]
    offsets = [
        (row.dropped_predictions[0]["char_start"], row.dropped_predictions[0]["char_end"]),
        (row.predictions[0]["char_start"], row.predictions[0]["char_end"]),
        (row.predictions[1]["char_start"], row.predictions[1]["char_end"]),
    ]
    assert offsets == sorted(offsets)
    assert len({prediction["raw_span_sha256"] for prediction in row.predictions}) == 1
    for prediction in row.predictions:
        assert prediction["raw_span_text"] == text[
            prediction["char_start"] : prediction["char_end"]
        ]
        assert [item["text"] for item in prediction["schema_spans"]] == [
            "<|object_ref_start|>",
            "<|object_ref_end|>",
            "<|box_start|>",
            "<|box_end|>",
        ]
        assert [item["text"] for item in prediction["coord_token_spans"]] == [
            "<|coord_100|>",
            "<|coord_200|>",
            "<|coord_300|>",
            "<|coord_400|>",
        ]


def test_all_spans_dropped_keeps_empty_row_and_non_metric_status() -> None:
    from src.inference.parsing import parse_compact_object_box_closed

    row = parse_compact_object_box_closed(
        "<|object_ref_start|>bad<|object_ref_end|>"
        "<|box_start|><|coord_1|><|coord_2|><|coord_3|><|box_end|>",
        row_id="row-1",
        row_index=0,
        image_width=1000,
        image_height=1000,
    )

    assert row.parse_status == "all_spans_dropped"
    assert row.metric_bearing is False
    assert row.predictions == []
    assert row.valid_prediction_count == 0
    assert row.dropped_prediction_count == 1


def test_degenerate_bbox_is_dropped_with_geometry_diagnostic() -> None:
    from src.inference.parsing import parse_compact_object_box_closed

    row = parse_compact_object_box_closed(
        "<|object_ref_start|>flat<|object_ref_end|>"
        "<|box_start|><|coord_300|><|coord_200|><|coord_300|><|coord_400|><|box_end|>",
        row_id="row-1",
        row_index=0,
        image_width=1000,
        image_height=1000,
    )

    assert row.parse_status == "all_spans_dropped"
    assert row.dropped_predictions[0]["reason"] == "geometry_invalid"
    assert row.dropped_predictions[0]["code"] == "data.bbox_order"


def test_out_of_range_coordinates_are_dropped_with_geometry_diagnostic() -> None:
    from src.inference.parsing import parse_compact_object_box_closed

    row = parse_compact_object_box_closed(
        "<|object_ref_start|>wide<|object_ref_end|>"
        "<|box_start|><|coord_0|><|coord_10|><|coord_1000|><|coord_200|><|box_end|>",
        row_id="row-1",
        row_index=0,
        image_width=1000,
        image_height=1000,
    )

    assert row.parse_status == "all_spans_dropped"
    assert row.dropped_predictions[0]["reason"] == "geometry_invalid"
    assert row.dropped_predictions[0]["code"] == "data.coord_token_format"
