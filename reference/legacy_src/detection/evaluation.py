"""Strict detection-template parsing helpers for metric-bearing evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from src.common.object_field_order import normalize_object_field_order
from src.detection.template import TemplateId, get_detection_template
from src.detection.teacher_forcing.compact_full_policy import (
    END_OF_TEXT_TOKEN,
    IM_END_TOKEN,
    OBJECT_REF_START_TOKEN,
    parse_compact_full,
)


ParserMode = Literal[
    "strict_expected",
    "marker_delimited_strict",
    "legacy_compatible",
    "diagnostic_auto_detect",
    "diagnostic_salvage",
]
MetricSurface = Literal["strict_expected_template", "diagnostic_only"]

STRICT_EXPECTED_METRIC_SURFACE_BREAK = (
    "strict_expected_template_metrics_are_not_directly_comparable_to_"
    "salvage_or_auto_detect_metrics_unless_parser_mode_is_reported"
)
DIAGNOSTIC_ONLY_METRIC_SURFACE_BREAK = (
    "diagnostic_parser_outputs_are_not_metric_bearing"
)
_SUPPORTED_PARSER_MODES = frozenset(
    {
        "strict_expected",
        "marker_delimited_strict",
        "legacy_compatible",
        "diagnostic_auto_detect",
        "diagnostic_salvage",
    }
)


def _require_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    return value


def _require_non_empty_string(value: Any, *, field_name: str) -> str:
    text = _require_string(value, field_name=field_name).strip()
    if not text:
        raise ValueError(f"{field_name} must be a non-empty string")
    return text


@dataclass(frozen=True)
class DetectionTemplateEvalManifest:
    expected_template: TemplateId
    parser_mode: ParserMode
    coordinate_surface: Literal["coord_token"]
    benchmark_scope: str
    metric_surface: MetricSurface
    diagnostic_only: bool
    metric_surface_break: str

    def to_manifest_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_stage1_json_pretty_strict_expected(text: str) -> dict[str, Any]:
    return _parse_expected_template(
        text,
        expected_template="stage1_json_pretty",
    )


def parse_compact_full_strict_expected(text: str) -> dict[str, Any]:
    return _parse_compact_full_expected(
        text,
        parser_mode="marker_delimited_strict",
    )


def parse_compact_full_output_artifact(
    text: str,
    *,
    parse_mode: str = "marker_delimited_strict",
) -> dict[str, Any]:
    """Parse compact_full inference text and return payload plus policy metadata."""

    text = _require_string(text, field_name="text")
    parse_mode = _require_string(parse_mode, field_name="parse_mode")
    result = parse_compact_full(text, mode=parse_mode)
    serialization_policy, object_separator = _compact_full_output_policy(
        text,
        parse_mode=result.mode,
    )
    return {
        "raw_output_json": result.to_payload() if result.ok else None,
        "parse_mode": result.mode,
        "serialization_policy": serialization_policy,
        "object_separator": object_separator,
        "terminal_token": result.terminal_token,
        "parse_error_code": result.error_code,
        "parse_error_offset": result.error_offset,
    }


def parse_detection_output_strict_expected(
    text: str,
    *,
    expected_template: TemplateId | str,
    parser_mode: str = "strict_expected",
    object_field_order: str = "desc_first",
) -> dict[str, Any]:
    parser_mode = _require_string(parser_mode, field_name="parser_mode")
    field_order = normalize_object_field_order(
        object_field_order,
        path="detection_template.object_field_order",
    )
    template = get_detection_template(expected_template)
    if template.template_id == "compact":
        if parser_mode == "strict_expected":
            return _parse_expected_template(
                text,
                expected_template=template.template_id,
                object_field_order=field_order,
            )
        if parser_mode not in {"marker_delimited_strict", "legacy_compatible"}:
            raise ValueError(
                f"Unsupported detection parser_mode for metrics: {parser_mode!r}"
            )
        if field_order != "desc_first":
            return _parse_expected_template(
                text,
                expected_template=template.template_id,
                object_field_order=field_order,
            )
        return _parse_compact_full_expected(text, parser_mode=parser_mode)
    if template.capabilities.object_field_order == "compact_row":
        if parser_mode != "strict_expected":
            raise ValueError(
                f"{template.template_id} evaluation does not support "
                f"parser_mode={parser_mode!r}"
            )
        return _parse_expected_template(
            text,
            expected_template=template.template_id,
            object_field_order=field_order,
        )
    if parser_mode != "strict_expected":
        raise ValueError(f"Unsupported detection parser_mode for metrics: {parser_mode!r}")
    return _parse_expected_template(
        text,
        expected_template=template.template_id,
        object_field_order=field_order,
    )


def build_detection_template_eval_manifest(
    *,
    expected_template: TemplateId | str,
    parser_mode: str,
    coordinate_surface: str,
    benchmark_scope: str,
) -> DetectionTemplateEvalManifest:
    template = get_detection_template(expected_template)
    parser_mode = _require_string(parser_mode, field_name="parser_mode")
    coordinate_surface = _require_string(
        coordinate_surface,
        field_name="coordinate_surface",
    )
    benchmark_scope = _require_non_empty_string(
        benchmark_scope,
        field_name="benchmark_scope",
    )
    if parser_mode not in _SUPPORTED_PARSER_MODES:
        raise ValueError(f"Unsupported detection parser_mode: {parser_mode!r}")
    if coordinate_surface != template.capabilities.coordinate_surface:
        raise ValueError(
            f"{template.template_id} evaluation requires coordinate_surface="
            f"{template.capabilities.coordinate_surface}"
        )
    if template.template_id == "compact" and parser_mode == "strict_expected":
        parser_mode = "marker_delimited_strict"
    if parser_mode in {
        "strict_expected",
        "marker_delimited_strict",
        "legacy_compatible",
    }:
        if template.template_id != "compact" and parser_mode != "strict_expected":
            raise ValueError(
                f"{template.template_id} evaluation does not support "
                f"parser_mode={parser_mode!r}"
            )
        return DetectionTemplateEvalManifest(
            expected_template=template.template_id,
            parser_mode=_metric_parser_mode(parser_mode),
            coordinate_surface="coord_token",
            benchmark_scope=benchmark_scope,
            metric_surface="strict_expected_template",
            diagnostic_only=False,
            metric_surface_break=STRICT_EXPECTED_METRIC_SURFACE_BREAK,
        )

    return DetectionTemplateEvalManifest(
        expected_template=template.template_id,
        parser_mode=_diagnostic_parser_mode(parser_mode),
        coordinate_surface="coord_token",
        benchmark_scope=benchmark_scope,
        metric_surface="diagnostic_only",
        diagnostic_only=True,
        metric_surface_break=DIAGNOSTIC_ONLY_METRIC_SURFACE_BREAK,
    )


def _parse_expected_template(
    text: str,
    *,
    expected_template: TemplateId | str,
    object_field_order: str = "desc_first",
) -> dict[str, Any]:
    text = _require_string(text, field_name="text")
    template = get_detection_template(expected_template)
    try:
        return template.parse_assistant(text, object_field_order=object_field_order)
    except ValueError as exc:
        raise ValueError(
            f"output does not match expected {template.template_id} template"
        ) from exc


def _parse_compact_full_expected(text: str, *, parser_mode: str) -> dict[str, Any]:
    text = _require_string(text, field_name="text")
    result = parse_compact_full(text, mode=parser_mode)
    if not result.ok:
        raise ValueError(
            "output does not match expected compact_full template: "
            f"{result.error_code}"
        )
    return result.to_payload()


def _compact_full_output_policy(
    text: str,
    *,
    parse_mode: str,
) -> tuple[str, str]:
    if parse_mode == "marker_delimited_strict":
        return "marker_delimited", OBJECT_REF_START_TOKEN
    effective = _strip_compact_terminal_and_padding(text)
    if "\n" in effective:
        return "legacy_newline_delimited", "\n"
    return "marker_delimited", OBJECT_REF_START_TOKEN


def _strip_compact_terminal_and_padding(text: str) -> str:
    im_end_pos = text.find(IM_END_TOKEN)
    if im_end_pos >= 0:
        return text[:im_end_pos].rstrip()
    stripped = text.rstrip()
    while stripped.endswith(END_OF_TEXT_TOKEN):
        stripped = stripped[: -len(END_OF_TEXT_TOKEN)].rstrip()
    return stripped


def _metric_parser_mode(parser_mode: str) -> Literal[
    "strict_expected",
    "marker_delimited_strict",
    "legacy_compatible",
]:
    if parser_mode == "strict_expected":
        return "strict_expected"
    if parser_mode == "marker_delimited_strict":
        return "marker_delimited_strict"
    if parser_mode == "legacy_compatible":
        return "legacy_compatible"
    raise ValueError(f"Unsupported detection parser_mode: {parser_mode!r}")


def _diagnostic_parser_mode(parser_mode: str) -> Literal[
    "diagnostic_auto_detect",
    "diagnostic_salvage",
]:
    if parser_mode == "diagnostic_auto_detect":
        return "diagnostic_auto_detect"
    if parser_mode == "diagnostic_salvage":
        return "diagnostic_salvage"
    raise ValueError(f"Unsupported detection parser_mode: {parser_mode!r}")


__all__ = [
    "DIAGNOSTIC_ONLY_METRIC_SURFACE_BREAK",
    "DetectionTemplateEvalManifest",
    "MetricSurface",
    "ParserMode",
    "STRICT_EXPECTED_METRIC_SURFACE_BREAK",
    "build_detection_template_eval_manifest",
    "parse_compact_full_output_artifact",
    "parse_compact_full_strict_expected",
    "parse_detection_output_strict_expected",
    "parse_stage1_json_pretty_strict_expected",
]
