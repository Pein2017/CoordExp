import pytest

from src.common.detection_sequence import BOX_START_TOKEN, OBJECT_REF_START_TOKEN
from src.detection.evaluation import (
    DetectionTemplateEvalManifest,
    build_detection_template_eval_manifest,
    parse_compact_full_strict_expected,
    parse_detection_output_strict_expected,
    parse_stage1_json_pretty_strict_expected,
)
from src.detection.template import CompactFullTemplate, Stage1JsonPrettyTemplate


STAGE1_JSON_TEXT = (
    '{"objects": [{"desc": "cat", "bbox_2d": '
    "[<|coord_1|>, <|coord_2|>, <|coord_3|>, <|coord_4|>]}]}"
)
COMPACT_TEXT = (
    f"{OBJECT_REF_START_TOKEN}cat{BOX_START_TOKEN}"
    "<|coord_1|><|coord_2|><|coord_3|><|coord_4|>"
)
COMPACT_LEGACY_TEXT = (
    f"{OBJECT_REF_START_TOKEN}cat{BOX_START_TOKEN}"
    "<|coord_1|><|coord_2|><|coord_3|><|coord_4|>\n"
    f"{OBJECT_REF_START_TOKEN}dog{BOX_START_TOKEN}"
    "<|coord_10|><|coord_20|><|coord_30|><|coord_40|>"
)


def test_compact_full_strict_expected_parses_valid_compact_output() -> None:
    assert parse_compact_full_strict_expected(COMPACT_TEXT) == {
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
    assert parse_detection_output_strict_expected(
        COMPACT_TEXT,
        expected_template="compact_full",
    ) == CompactFullTemplate().parse_assistant(COMPACT_TEXT)


def test_compact_full_strict_expected_rejects_json_like_output() -> None:
    with pytest.raises(ValueError, match="expected compact_full"):
        parse_compact_full_strict_expected(STAGE1_JSON_TEXT)


def test_compact_full_eval_routes_marker_strict_and_legacy_compatible_modes() -> None:
    with pytest.raises(ValueError, match="legacy_separator_in_new_format"):
        parse_detection_output_strict_expected(
            COMPACT_LEGACY_TEXT,
            expected_template="compact_full",
            parser_mode="marker_delimited_strict",
        )

    assert parse_detection_output_strict_expected(
        COMPACT_LEGACY_TEXT,
        expected_template="compact_full",
        parser_mode="legacy_compatible",
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


def test_stage1_json_pretty_strict_expected_parses_valid_json_output() -> None:
    assert parse_stage1_json_pretty_strict_expected(STAGE1_JSON_TEXT) == {
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
    assert parse_detection_output_strict_expected(
        STAGE1_JSON_TEXT,
        expected_template="stage1_json_pretty",
    ) == Stage1JsonPrettyTemplate().parse_assistant(STAGE1_JSON_TEXT)


def test_stage1_json_pretty_strict_expected_rejects_compact_like_output() -> None:
    with pytest.raises(ValueError, match="expected stage1_json_pretty"):
        parse_stage1_json_pretty_strict_expected(COMPACT_TEXT)


def test_strict_expected_parse_rejects_unsupported_template_and_parser_mode() -> None:
    with pytest.raises(ValueError, match="Unsupported detection template"):
        parse_detection_output_strict_expected(
            STAGE1_JSON_TEXT,
            expected_template="unknown_template",
        )

    with pytest.raises(ValueError, match="Unsupported detection parser_mode"):
        parse_detection_output_strict_expected(
            STAGE1_JSON_TEXT,
            expected_template="stage1_json_pretty",
            parser_mode="diagnostic_auto_detect",
        )


@pytest.mark.parametrize("bad_text", [None, [], {}])
def test_strict_expected_parse_rejects_non_string_text(bad_text) -> None:
    with pytest.raises(TypeError, match="text must be a string"):
        parse_compact_full_strict_expected(bad_text)


@pytest.mark.parametrize("bad_parser_mode", [None, [], {}])
def test_strict_expected_parse_rejects_non_string_parser_mode(
    bad_parser_mode,
) -> None:
    with pytest.raises(TypeError, match="parser_mode must be a string"):
        parse_detection_output_strict_expected(
            STAGE1_JSON_TEXT,
            expected_template="stage1_json_pretty",
            parser_mode=bad_parser_mode,
        )


def test_eval_manifest_records_metric_surface_contract() -> None:
    manifest = build_detection_template_eval_manifest(
        expected_template="compact_full",
        parser_mode="marker_delimited_strict",
        coordinate_surface="coord_token",
        benchmark_scope="val200",
    )

    assert manifest == DetectionTemplateEvalManifest(
        expected_template="compact_full",
        parser_mode="marker_delimited_strict",
        coordinate_surface="coord_token",
        benchmark_scope="val200",
        metric_surface="strict_expected_template",
        diagnostic_only=False,
        metric_surface_break=(
            "strict_expected_template_metrics_are_not_directly_comparable_to_"
            "salvage_or_auto_detect_metrics_unless_parser_mode_is_reported"
        ),
    )
    assert manifest.to_manifest_dict() == {
        "expected_template": "compact_full",
        "parser_mode": "marker_delimited_strict",
        "coordinate_surface": "coord_token",
        "benchmark_scope": "val200",
        "metric_surface": "strict_expected_template",
        "diagnostic_only": False,
        "metric_surface_break": (
            "strict_expected_template_metrics_are_not_directly_comparable_to_"
            "salvage_or_auto_detect_metrics_unless_parser_mode_is_reported"
        ),
    }


def test_eval_manifest_marks_salvage_and_auto_detect_as_diagnostic_only() -> None:
    for parser_mode in ("diagnostic_auto_detect", "diagnostic_salvage"):
        manifest = build_detection_template_eval_manifest(
            expected_template="stage1_json_pretty",
            parser_mode=parser_mode,
            coordinate_surface="coord_token",
            benchmark_scope="tiny",
        )

        assert manifest.metric_surface == "diagnostic_only"
        assert manifest.diagnostic_only is True
        assert manifest.metric_surface_break == (
            "diagnostic_parser_outputs_are_not_metric_bearing"
        )


@pytest.mark.parametrize(
    ("field_name", "overrides", "error_type", "match"),
    [
        (
            "parser_mode",
            {"parser_mode": []},
            TypeError,
            "parser_mode must be a string",
        ),
        (
            "coordinate_surface",
            {"coordinate_surface": None},
            TypeError,
            "coordinate_surface must be a string",
        ),
        (
            "benchmark_scope",
            {"benchmark_scope": None},
            TypeError,
            "benchmark_scope must be a string",
        ),
        (
            "benchmark_scope",
            {"benchmark_scope": "   "},
            ValueError,
            "benchmark_scope must be a non-empty string",
        ),
    ],
)
def test_eval_manifest_rejects_accidental_types_and_empty_scope(
    field_name: str,
    overrides: dict[str, object],
    error_type: type[Exception],
    match: str,
) -> None:
    payload = {
        "expected_template": "compact_full",
        "parser_mode": "strict_expected",
        "coordinate_surface": "coord_token",
        "benchmark_scope": "val200",
    }
    payload.update(overrides)

    with pytest.raises(error_type, match=match):
        build_detection_template_eval_manifest(**payload)
