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
    assert row.predictions == [
        {
            "description": "cat",
            "bbox": [100, 100, 300, 200],
            "bbox_format": "xyxy",
            "coord_bins": [100, 200, 300, 400],
            "generated_order": 0,
        }
    ]
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
