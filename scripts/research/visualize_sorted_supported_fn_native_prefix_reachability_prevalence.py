#!/usr/bin/env python3
"""Visualization for the sorted supported false-negative native-prefix
reachability prevalence unit
(``2026-08-03-sorted-supported-fn-native-prefix-reachability-prevalence``).

This module reads only ``analysis/report.json``, ``analysis/report.md``,
``analysis/owner-records.jsonl`` and ``analysis/receipt.json`` published by
``analyze_sorted_supported_fn_native_prefix_reachability_prevalence.py``.  It
never opens a predecessor artifact directly, never recomputes a rank, margin,
gate, or support decision, and fails closed if the analysis outputs do not
match the exact digests sealed in the receipt -- including ``report.md``,
whose bytes are re-hashed and compared, not merely assumed present.

Three products
--------------
``owner_matrix``
    One row per owner in the ``114``-owner supported false-negative cohort,
    grouped by image then category.  Columns are categorical read-outs only
    (support-context count/role, frontier timing, gate, category rank,
    owner rank, favorable-conjunction flags, competitor coverage): no cell
    encodes a raw or cross-image log probability.
``native_tp_matrix``
    The same columns for the ``141``-owner native true-positive due-boundary
    reference.
``summary_panel``
    A compact bar rendering of the FN cohort's U-bound headline Wilson
    proportions -- a summary statistic per metric, never a raw or
    cross-image log probability.

An optional ``--representative-visual-root`` names the predecessor's sealed
``visual/combined`` directory.  When supplied, this module resolves (never
copies or reinterprets) the exact ``owner_map__<image_id>.png`` file for one
deterministically selected owner in each of four representative categories,
and seals each referenced file's exact sha256/size into
``representative-case-references.json`` and the visual manifest.  It fails
closed if a selected case's source image is absent.

``--specs-only`` emits the pure, testable specs, the case-reference JSON, and
the manifest without importing PIL or rendering a local PNG; sealed external
image references are still resolved and hashed if a visual root is supplied,
since that is reading and hashing an existing file, not rendering one.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, NoReturn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import (  # noqa: E402
    analyze_sorted_supported_fn_native_prefix_reachability_prevalence as analyzer,
)

UNIT_ID = analyzer.UNIT_ID
VISUAL_SCHEMA_VERSION = "sorted-supported-fn-native-prefix-reachability-prevalence-visual.v1"
MANIFEST_SCHEMA_VERSION = (
    "sorted-supported-fn-native-prefix-reachability-prevalence-visual-manifest.v1"
)
MANIFEST_NAME = "visual-manifest.json"
OWNER_MATRIX_NAME = "owner-matrix.png"
NATIVE_TP_MATRIX_NAME = "native-tp-due-boundary-matrix.png"
SUMMARY_PANEL_NAME = "summary-panel.png"

FN_COHORT = analyzer.FN_COHORT
TP_COHORT = analyzer.TP_COHORT

#: Categorical, discrete cell colors.  No cell ever encodes a continuous or
#: cross-image raw log probability; every color below is a fixed label for a
#: discrete category (open/closed, rank tier, coverage state).
COLOR_GREEN = (76, 175, 80)
COLOR_RED = (229, 115, 115)
COLOR_YELLOW = (255, 213, 79)
COLOR_GREY = (189, 189, 189)
COLOR_BLUE = (100, 181, 246)

COLUMN_NAMES: tuple[str, ...] = (
    "owner",
    "support_count",
    "role_coverage",
    "frontier_timing",
    "gate",
    "category_rank",
    "owner_rank",
    "favorable_top3",
    "favorable_rank1",
    "competitor",
)


class VisualContractError(RuntimeError):
    """A precondition for rendering this unit's figures was not proven."""


def _fail(message: str) -> NoReturn:
    raise VisualContractError(message)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        _fail(f"{label} is missing at {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"{label} is unreadable at {path}: {exc}")
    if not isinstance(value, Mapping):
        _fail(f"{label} at {path} is not a JSON object")
    return dict(value)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_jsonl_rows(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        _fail(f"{label} is missing at {path}")
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            _fail(f"{label} line {number} is not valid JSON: {exc}")
        if not isinstance(value, Mapping):
            _fail(f"{label} line {number} is not a JSON object")
        rows.append(dict(value))
    return rows


class Artifacts:
    """The analyzer's three published files, digest-verified before use."""

    def __init__(self, report: Mapping[str, Any], owner_records: list[dict[str, Any]], receipt: Mapping[str, Any]):
        self.report = report
        self.owner_records = owner_records
        self.receipt = receipt


def load_artifacts(analysis_root: Path) -> Artifacts:
    """Load and digest-validate the analyzer's report, records, and receipt.

    Reads receipt.json first, re-verifies its own declared digest, then
    verifies that report.json and owner-records.jsonl reproduce exactly the
    bytes-level sha256 the receipt sealed.  Nothing downstream is built from
    an artifact whose bytes have drifted from the receipt.
    """

    analysis_root = Path(analysis_root)
    receipt_path = analysis_root / analyzer.RECEIPT_NAME
    report_path = analysis_root / analyzer.REPORT_JSON_NAME
    report_md_path = analysis_root / analyzer.REPORT_MD_NAME
    owner_records_path = analysis_root / analyzer.OWNER_RECORDS_NAME

    receipt = _read_json(receipt_path, "analysis/receipt.json")
    if receipt.get("schema_version") != analyzer.RECEIPT_SCHEMA_VERSION:
        _fail("analysis/receipt.json has an unexpected schema_version")
    if receipt.get("unit_id") != UNIT_ID:
        _fail("analysis/receipt.json belongs to another unit")
    reconstructed = analyzer.sha256_json(
        {key: value for key, value in receipt.items() if key != "receipt_content_sha256"}
    )
    if reconstructed != receipt.get("receipt_content_sha256"):
        _fail("analysis/receipt.json does not reconstruct its own digest; it was edited")

    if not report_path.is_file():
        _fail(f"analysis/report.json is missing at {report_path}")
    report_bytes = report_path.read_bytes()
    if _sha256_bytes(report_bytes) != receipt.get("report_json_sha256"):
        _fail("analysis/report.json bytes do not match the digest sealed in the receipt; tampered or stale")
    report = json.loads(report_bytes.decode("utf-8"))
    if report.get("schema_version") != analyzer.REPORT_SCHEMA_VERSION:
        _fail("analysis/report.json has an unexpected schema_version")
    if report.get("unit_id") != UNIT_ID:
        _fail("analysis/report.json belongs to another unit")

    if not report_md_path.is_file():
        _fail(f"analysis/report.md is missing at {report_md_path}")
    report_md_bytes = report_md_path.read_bytes()
    if _sha256_bytes(report_md_bytes) != receipt.get("report_md_sha256"):
        _fail("analysis/report.md bytes do not match the digest sealed in the receipt; tampered or stale")

    if not owner_records_path.is_file():
        _fail(f"analysis/owner-records.jsonl is missing at {owner_records_path}")
    owner_records_bytes = owner_records_path.read_bytes()
    if _sha256_bytes(owner_records_bytes) != receipt.get("owner_records_jsonl_sha256"):
        _fail(
            "analysis/owner-records.jsonl bytes do not match the digest sealed in the "
            "receipt; tampered or stale"
        )
    owner_records = _read_jsonl_rows(owner_records_path, "analysis/owner-records.jsonl")

    fn_records = [row for row in owner_records if row.get("cohort") == FN_COHORT]
    tp_records = [row for row in owner_records if row.get("cohort") == TP_COHORT]
    if len(fn_records) != report["false_negative_cohort"]["denominator"]:
        _fail(
            "owner-records.jsonl false-negative row count does not match "
            "report.json's false_negative_cohort.denominator"
        )
    if len(tp_records) != report["native_true_positive_reference"]["denominator"]:
        _fail(
            "owner-records.jsonl native true-positive row count does not match "
            "report.json's native_true_positive_reference.denominator"
        )

    return Artifacts(report=report, owner_records=owner_records, receipt=receipt)


# ---------------------------------------------------------------------------
# Pure spec builders (no PIL import; fully testable)
# ---------------------------------------------------------------------------


def _role_coverage_text(bound: Mapping[str, Any]) -> str:
    return "".join(
        [
            "R" if bound.get("any_root") else "-",
            "B" if bound.get("any_row_boundary") else "-",
            "T" if bound.get("any_terminal") else "-",
        ]
    )


def _frontier_timing_cell(bound: Mapping[str, Any]) -> dict[str, Any]:
    before_or_at = bool(bound.get("any_before_or_at_frontier"))
    after = bool(bound.get("any_after_frontier"))
    if before_or_at and after:
        return {"text": "mixed", "color": COLOR_YELLOW}
    if before_or_at:
        return {"text": "before/at", "color": COLOR_GREEN}
    if after:
        return {"text": "after", "color": COLOR_RED}
    return {"text": "n/a", "color": COLOR_GREY}


def _bool_cell(value: bool, *, true_text: str, false_text: str) -> dict[str, Any]:
    return {
        "text": true_text if value else false_text,
        "color": COLOR_GREEN if value else COLOR_RED,
    }


def _category_rank_cell(bound: Mapping[str, Any]) -> dict[str, Any]:
    if bound.get("any_category_rank_one"):
        return {"text": "rank1", "color": COLOR_GREEN}
    if bound.get("any_category_rank_top3"):
        return {"text": "top3", "color": COLOR_YELLOW}
    return {"text": "none", "color": COLOR_RED}


def _fn_competitor_cell(record: Mapping[str, Any]) -> dict[str, Any]:
    if not record["never_owner_rank_one"]:
        return {"text": "target", "color": COLOR_GREEN}
    bound = record["upper_bound_u"]
    covered = bool(bound.get("any_covered_other"))
    uncovered = bool(bound.get("any_uncovered_other"))
    if covered and uncovered:
        return {"text": "mixed", "color": COLOR_YELLOW}
    if covered:
        return {"text": "covered", "color": COLOR_BLUE}
    if uncovered:
        return {"text": "uncovered", "color": COLOR_RED}
    return {"text": "n/a", "color": COLOR_GREY}


def build_owner_matrix_spec(fn_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """One row per supported false-negative owner, U-bound columns only."""

    ordered = sorted(
        fn_records,
        key=lambda r: (str(r["image_id"]), str(r["normalized_description"]), str(r["gt_owner_id"])),
    )
    rows: list[dict[str, Any]] = []
    for record in ordered:
        bound = record["upper_bound_u"]
        rows.append(
            {
                "gt_owner_id": str(record["gt_owner_id"]),
                "image_id": str(record["image_id"]),
                "normalized_description": str(record["normalized_description"]),
                "cells": {
                    "owner": {"text": str(record["gt_owner_id"]), "color": None},
                    "support_count": {
                        "text": str(bound["support_context_count"]),
                        "color": COLOR_GREEN if bound["support_context_count"] > 0 else COLOR_GREY,
                    },
                    "role_coverage": {"text": _role_coverage_text(bound), "color": None},
                    "frontier_timing": _frontier_timing_cell(bound),
                    "gate": _bool_cell(bool(bound["any_gate_open"]), true_text="open", false_text="closed"),
                    "category_rank": _category_rank_cell(bound),
                    "owner_rank": _bool_cell(
                        bool(bound["any_owner_rank_one"]), true_text="rank1", false_text="none"
                    ),
                    "favorable_top3": _bool_cell(
                        bool(bound["any_favorable_top3_before_or_at_frontier"]),
                        true_text="yes",
                        false_text="no",
                    ),
                    "favorable_rank1": _bool_cell(
                        bool(bound["any_favorable_rank1_before_or_at_frontier"]),
                        true_text="yes",
                        false_text="no",
                    ),
                    "competitor": _fn_competitor_cell(record),
                },
            }
        )
    return {
        "schema_version": VISUAL_SCHEMA_VERSION,
        "product": "owner_matrix",
        "cohort": FN_COHORT,
        "bound": "upper_bound_u",
        "columns": list(COLUMN_NAMES),
        "row_count": len(rows),
        "rows": rows,
        "grouping": "image_then_category",
        "color_semantics": "categorical_discrete_readouts_never_raw_or_cross_image_logprob",
    }


def _tp_competitor_cell(channel: Mapping[str, Any]) -> dict[str, Any]:
    status = str(channel["competitor_status"])
    mapping = {
        analyzer.COMPETITOR_TARGET: ("target", COLOR_GREEN),
        analyzer.COMPETITOR_COVERED_OTHER: ("covered", COLOR_BLUE),
        analyzer.COMPETITOR_UNCOVERED_OTHER: ("uncovered", COLOR_RED),
        analyzer.COMPETITOR_NO_POPULATION: ("n/a", COLOR_GREY),
    }
    text, color = mapping.get(status, (status, COLOR_GREY))
    return {"text": text, "color": color}


def build_native_tp_matrix_spec(tp_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """One row per native true-positive owner, at its exact due boundary."""

    ordered = sorted(
        tp_records,
        key=lambda r: (str(r["image_id"]), str(r["normalized_description"]), str(r["gt_owner_id"])),
    )
    rows: list[dict[str, Any]] = []
    for record in ordered:
        channel = record["channel"]
        role_coverage = {
            "any_root": channel["context_role"] == "root",
            "any_row_boundary": channel["context_role"] == "row_boundary",
            "any_terminal": channel["context_role"] == "terminal",
        }
        frontier_bound = {
            "any_before_or_at_frontier": bool(channel["before_or_at_frontier"]),
            "any_after_frontier": not bool(channel["before_or_at_frontier"]),
        }
        category_rank_bound = {
            "any_category_rank_one": bool(channel["category_rank_one"]),
            "any_category_rank_top3": bool(channel["category_rank_top3"]),
        }
        rows.append(
            {
                "gt_owner_id": str(record["gt_owner_id"]),
                "image_id": str(record["image_id"]),
                "normalized_description": str(record["normalized_description"]),
                "due_context_id": str(record["due_context_id"]),
                "cells": {
                    "owner": {"text": str(record["gt_owner_id"]), "color": None},
                    "support_count": {
                        "text": "1" if record["due_context_support"] else "0",
                        "color": COLOR_GREEN if record["due_context_support"] else COLOR_GREY,
                    },
                    "role_coverage": {"text": _role_coverage_text(role_coverage), "color": None},
                    "frontier_timing": _frontier_timing_cell(frontier_bound),
                    "gate": _bool_cell(
                        bool(channel["gate_open"]), true_text="open", false_text="closed"
                    ),
                    "category_rank": _category_rank_cell(category_rank_bound),
                    "owner_rank": _bool_cell(
                        bool(channel["owner_rank_one"]), true_text="rank1", false_text="none"
                    ),
                    "favorable_top3": _bool_cell(
                        bool(record["favorable_top3_before_or_at_frontier"]),
                        true_text="yes",
                        false_text="no",
                    ),
                    "favorable_rank1": _bool_cell(
                        bool(record["favorable_rank1_before_or_at_frontier"]),
                        true_text="yes",
                        false_text="no",
                    ),
                    "competitor": _tp_competitor_cell(channel),
                },
            }
        )
    return {
        "schema_version": VISUAL_SCHEMA_VERSION,
        "product": "native_tp_matrix",
        "cohort": TP_COHORT,
        "columns": list(COLUMN_NAMES),
        "row_count": len(rows),
        "rows": rows,
        "grouping": "image_then_category",
        "color_semantics": "categorical_discrete_readouts_never_raw_or_cross_image_logprob",
        "role": "descriptive_positive_reference_never_a_matched_causal_control",
    }


#: The small-multiple counts drawn per image and per category, alongside
#: owner_count.  Every value here is an owner-level count already published
#: in report.json's per_image/per_category strata; nothing is recomputed and
#: nothing encodes a raw or cross-image log probability.
SMALL_MULTIPLE_METRICS: tuple[str, ...] = (
    "any_favorable_top3_before_or_at_frontier",
    "any_owner_rank_one",
    "any_gate_open",
)


def _small_multiple_panel(strata: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One row per stratum, reusing exactly the report's own strata counts."""

    return [
        {
            "stratum": stratum_key,
            "owner_count": int(stratum["owner_count"]),
            **{metric: int(stratum[metric]) for metric in SMALL_MULTIPLE_METRICS},
        }
        for stratum_key, stratum in sorted(strata.items())
    ]


def build_summary_panel_spec(report: Mapping[str, Any]) -> dict[str, Any]:
    """A compact bar-style summary of the U-bound headline metrics.

    Alongside the single overall bar chart, this emits two categorical
    small-multiple panels -- one row per image, one row per category -- each
    carrying ``owner_count`` plus the same three owner-level U-bound counts
    (favorable-top3-before/at, owner-rank-one, gate-open) already published in
    ``report.json``'s ``per_image``/``per_category`` strata.  These are
    counts over owners, never a raw or cross-image log probability.
    """

    upper = report["false_negative_cohort"]["headline"]["upper_bound_u"]
    metrics = upper["metrics"]
    bars = [
        {
            "metric": name,
            "successes": interval["successes"],
            "total": interval["total"],
            "proportion": interval["proportion"],
        }
        for name, interval in metrics.items()
    ]
    return {
        "schema_version": VISUAL_SCHEMA_VERSION,
        "product": "summary_panel",
        "cohort": FN_COHORT,
        "bound": "upper_bound_u",
        "bars": bars,
        "small_multiple_metrics": list(SMALL_MULTIPLE_METRICS),
        "per_image_small_multiples": _small_multiple_panel(upper["per_image"]),
        "per_category_small_multiples": _small_multiple_panel(upper["per_category"]),
        "small_multiple_semantics": (
            "owner_level_counts_from_report_strata_never_raw_or_cross_image_logprob"
        ),
    }


# ---------------------------------------------------------------------------
# Rendering (PIL import deferred so --specs-only needs no image backend)
# ---------------------------------------------------------------------------


def render_matrix_png(spec: Mapping[str, Any], output_path: Path) -> str:
    from PIL import Image, ImageDraw

    columns = list(spec["columns"])
    rows = spec["rows"]
    row_height = 16
    header_height = 20
    label_width = 220
    column_width = 78
    width = label_width + column_width * (len(columns) - 1)
    height = header_height + row_height * max(1, len(rows))

    figure = Image.new("RGB", (width, height), (18, 18, 20))
    draw = ImageDraw.Draw(figure)
    draw.text((4, 2), f"{spec['product']} ({spec['cohort']}, n={spec['row_count']})", fill=(230, 230, 230))
    for column_index, column_name in enumerate(columns[1:]):
        x = label_width + column_index * column_width
        draw.text((x + 4, header_height - 16), column_name[:11], fill=(190, 190, 200))

    for row_index, row in enumerate(rows):
        y = header_height + row_index * row_height
        label = f"{row['image_id']}/{row['normalized_description']}/{row['gt_owner_id']}"
        draw.text((4, y + 2), label[:44], fill=(220, 220, 225))
        for column_index, column_name in enumerate(columns[1:]):
            cell = row["cells"][column_name]
            x = label_width + column_index * column_width
            color = cell.get("color") or (60, 60, 64)
            draw.rectangle([x, y, x + column_width - 2, y + row_height - 2], fill=color)
            draw.text((x + 3, y + 2), str(cell["text"])[:11], fill=(20, 20, 20))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.save(output_path)
    return str(output_path)


def _draw_proportion_bar(
    draw: Any,
    *,
    y: int,
    row_height: int,
    label: str,
    label_width: int,
    bar_area_width: int,
    successes: int,
    total: int,
) -> None:
    draw.text((4, y + 2), label[:44], fill=(220, 220, 225))
    proportion = (successes / total) if total else 0.0
    bar_width = max(1, int(bar_area_width * proportion)) if total else 0
    draw.rectangle(
        [label_width, y, label_width + bar_area_width, y + row_height - 2],
        outline=(80, 80, 86),
    )
    if bar_width:
        draw.rectangle(
            [label_width, y, label_width + bar_width, y + row_height - 2],
            fill=COLOR_BLUE,
        )
    draw.text(
        (label_width + bar_area_width + 4, y + 2),
        f"{successes}/{total}",
        fill=(220, 220, 225),
    )


def render_summary_panel_png(spec: Mapping[str, Any], output_path: Path) -> str:
    """A compact bar rendering of the U-bound headline Wilson proportions.

    Bar length is the owner-level proportion (successes / total) for one
    named metric; this is a summary statistic over the whole FN cohort, never
    a raw or per-image log probability, and the same bar color is used for
    every bar.  Below the overall bars, two categorical small-multiple
    sections repeat ``owner_count`` plus three owner-level counts
    (favorable-top3-before/at, owner-rank-one, gate-open) once per image and
    once per category, each proportion taken over that stratum's own
    ``owner_count`` -- again never a raw or cross-image log probability.
    """

    from PIL import Image, ImageDraw

    bars = list(spec["bars"])
    row_height = 18
    header_height = 24
    label_width = 340
    bar_area_width = 260
    width = label_width + bar_area_width + 80

    small_multiple_metrics = list(spec["small_multiple_metrics"])
    per_image_rows = list(spec["per_image_small_multiples"])
    per_category_rows = list(spec["per_category_small_multiples"])
    section_gap = 20
    small_multiple_row_height = 16
    small_multiple_label_width = 90
    small_multiple_column_width = 150

    def _small_multiple_section_height(row_count: int) -> int:
        return section_gap + small_multiple_row_height * (row_count + 1)

    height = (
        header_height
        + row_height * max(1, len(bars))
        + _small_multiple_section_height(len(per_image_rows))
        + _small_multiple_section_height(len(per_category_rows))
    )

    figure = Image.new("RGB", (width, height), (18, 18, 20))
    draw = ImageDraw.Draw(figure)
    draw.text(
        (4, 2),
        f"{spec['product']} ({spec['cohort']}, {spec['bound']})",
        fill=(230, 230, 230),
    )
    for row_index, bar in enumerate(bars):
        y = header_height + row_index * row_height
        _draw_proportion_bar(
            draw,
            y=y,
            row_height=row_height,
            label=str(bar["metric"]),
            label_width=label_width,
            bar_area_width=bar_area_width,
            successes=int(bar["successes"]),
            total=int(bar["total"]),
        )

    def _draw_small_multiple_section(*, top: int, title: str, rows: list[dict[str, Any]]) -> int:
        draw.text((4, top), title, fill=(190, 190, 200))
        header_y = top + small_multiple_row_height
        draw.text((4, header_y), "owner_count", fill=(160, 160, 170))
        for column_index, metric in enumerate(small_multiple_metrics):
            x = small_multiple_label_width + column_index * small_multiple_column_width
            draw.text((x, header_y), metric[:20], fill=(160, 160, 170))
        for row_index, row in enumerate(rows):
            y = header_y + small_multiple_row_height * (row_index + 1)
            draw.text((4, y), f"{row['stratum']} (n={row['owner_count']})"[:16], fill=(220, 220, 225))
            for column_index, metric in enumerate(small_multiple_metrics):
                x = small_multiple_label_width + column_index * small_multiple_column_width
                count = int(row[metric])
                owner_count = int(row["owner_count"])
                proportion = (count / owner_count) if owner_count else 0.0
                bar_width = max(1, int(120 * proportion)) if count else 0
                draw.rectangle([x, y, x + 120, y + small_multiple_row_height - 2], outline=(80, 80, 86))
                if bar_width:
                    draw.rectangle([x, y, x + bar_width, y + small_multiple_row_height - 2], fill=COLOR_BLUE)
                draw.text((x + 124, y), f"{count}/{owner_count}", fill=(200, 200, 205))
        return header_y + small_multiple_row_height * (len(rows) + 1)

    section_top = header_height + row_height * max(1, len(bars)) + 4
    section_top = _draw_small_multiple_section(
        top=section_top, title="Per image", rows=per_image_rows
    )
    _draw_small_multiple_section(top=section_top, title="Per category", rows=per_category_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.save(output_path)
    return str(output_path)


# ---------------------------------------------------------------------------
# Representative-case selection and external image references
# ---------------------------------------------------------------------------

REPRESENTATIVE_CASE_SCHEMA_VERSION = (
    "sorted-supported-fn-native-prefix-reachability-prevalence-representative-cases.v1"
)
REPRESENTATIVE_CASE_REFERENCES_NAME = "representative-case-references.json"
OWNER_MAP_FILENAME_TEMPLATE = "owner_map__{image_id}.png"

CASE_ROBUST_FAVORABLE_PREFRONTIER_MISS = "robust_favorable_prefrontier_miss"
CASE_ONLY_UNCOVERED_OTHER = "only_uncovered_other_among_never_owner_rank_one"
CASE_ONLY_COVERED_OTHER = "only_covered_other_among_never_owner_rank_one"
CASE_SUPPORT_ONLY_AFTER_FRONTIER = "support_only_after_frontier"
REPRESENTATIVE_CASE_NAMES: tuple[str, ...] = (
    CASE_ROBUST_FAVORABLE_PREFRONTIER_MISS,
    CASE_ONLY_UNCOVERED_OTHER,
    CASE_ONLY_COVERED_OTHER,
    CASE_SUPPORT_ONLY_AFTER_FRONTIER,
)


def _shared_u_and_l_favorable_context_count(record: Mapping[str, Any]) -> int:
    """How many context IDs are favorable-top3-before/at under both bounds.

    Uses ``favorable_top3_before_or_at_frontier`` -- the unit's primary
    favorable-surface variant (``49``/``114`` under U in the frozen census;
    the stricter rank-one variant is a narrower ``22``/``114`` sub-cohort and
    must not silently become the selection pool here.  "Robust" means the
    favorable surface is not an artifact of one bound's context selection:
    the same context ID must appear in both the U and the L usable-support
    context lists and be favorable under both.
    """

    u_by_context = {c["context_id"]: c for c in record["upper_bound_u"]["contexts"]}
    l_by_context = {c["context_id"]: c for c in record["lower_bound_l"]["contexts"]}
    shared_ids = set(u_by_context) & set(l_by_context)
    return sum(
        1
        for context_id in shared_ids
        if u_by_context[context_id]["favorable_top3_before_or_at_frontier"]
        and l_by_context[context_id]["favorable_top3_before_or_at_frontier"]
    )


def select_representative_cases(
    fn_records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any] | None]:
    """Deterministically select one FN owner for each representative category.

    Selection never touches an image or a pixel; it reads only the already
    published owner-record fields.  Categories are kept mutually exclusive
    where the underlying predicates already are: ``only_uncovered_other`` and
    ``only_covered_other`` cannot both hold for one owner by construction, and
    the favorable case requires ``owner_rank_one`` to hold somewhere, which
    ``never_owner_rank_one`` (the precondition for the other two "never"
    cases) rules out for that same owner.
    """

    cases: dict[str, dict[str, Any] | None] = {}

    favorable_candidates = [
        (_shared_u_and_l_favorable_context_count(record), record) for record in fn_records
    ]
    favorable_candidates = [(count, record) for count, record in favorable_candidates if count > 0]
    if favorable_candidates:
        favorable_candidates.sort(key=lambda pair: (-pair[0], str(pair[1]["gt_owner_id"])))
        count, record = favorable_candidates[0]
        cases[CASE_ROBUST_FAVORABLE_PREFRONTIER_MISS] = {
            "record": record,
            "selection_rule": (
                "max_count_of_context_ids_favorable_top3_before_or_at_frontier_under_both_u_"
                "and_l_bounds_tie_break_gt_owner_id_ascending"
            ),
            "selection_metric": {"shared_u_and_l_favorable_context_count": count},
        }
    else:
        cases[CASE_ROBUST_FAVORABLE_PREFRONTIER_MISS] = None

    for case_name, predicate, rule in (
        (
            CASE_ONLY_UNCOVERED_OTHER,
            lambda r: bool(r["only_uncovered_other_among_never_rank_one"]),
            "first_by_gt_owner_id_ascending_among_only_uncovered_other_among_never_rank_one",
        ),
        (
            CASE_ONLY_COVERED_OTHER,
            lambda r: bool(r["only_covered_other_among_never_rank_one"]),
            "first_by_gt_owner_id_ascending_among_only_covered_other_among_never_rank_one",
        ),
        (
            CASE_SUPPORT_ONLY_AFTER_FRONTIER,
            lambda r: bool(r["obstruction_flags"]["support_only_after_frontier"]),
            "first_by_gt_owner_id_ascending_among_support_only_after_frontier",
        ),
    ):
        qualifying = sorted(
            (record for record in fn_records if predicate(record)),
            key=lambda record: str(record["gt_owner_id"]),
        )
        cases[case_name] = (
            None if not qualifying else {"record": qualifying[0], "selection_rule": rule, "selection_metric": {}}
        )

    return cases


def resolve_representative_case_references(
    cases: Mapping[str, Mapping[str, Any] | None],
    *,
    visual_root: Path | None,
) -> dict[str, Any]:
    """Attach an exact, sealed external image reference to each selected case.

    Never opens, decodes, copies, or modifies the referenced image: only its
    path and byte-level sha256/size are read.  Fails closed if a case has a
    selected owner and a visual root was supplied but that owner's exact
    ``owner_map__<image_id>.png`` file is absent.
    """

    references: dict[str, Any] = {}
    for case_name in REPRESENTATIVE_CASE_NAMES:
        selected = cases.get(case_name)
        if selected is None:
            references[case_name] = {
                "selected": False,
                "reason": "no_qualifying_owner",
                "source_image": None,
            }
            continue

        record = selected["record"]
        entry: dict[str, Any] = {
            "selected": True,
            "gt_owner_id": str(record["gt_owner_id"]),
            "image_id": str(record["image_id"]),
            "normalized_description": str(record["normalized_description"]),
            "selection_rule": selected["selection_rule"],
            "selection_metric": dict(selected["selection_metric"]),
        }
        if visual_root is None:
            entry["source_image"] = {"reason": "visual_root_not_supplied"}
        else:
            image_path = Path(visual_root) / OWNER_MAP_FILENAME_TEMPLATE.format(
                image_id=str(record["image_id"])
            )
            if not image_path.is_file():
                _fail(
                    f"representative case {case_name!r} selected owner "
                    f"{record['gt_owner_id']!r} (image {record['image_id']!r}), but its source "
                    f"image is absent at {image_path}"
                )
            data = image_path.read_bytes()
            entry["source_image"] = {
                "path": str(image_path),
                "sha256": analyzer.sha256_bytes(data),
                "size_bytes": len(data),
            }
        references[case_name] = entry
    return references


def build_representative_case_document(
    fn_records: Sequence[Mapping[str, Any]], *, visual_root: Path | None
) -> dict[str, Any]:
    cases = select_representative_cases(fn_records)
    references = resolve_representative_case_references(cases, visual_root=visual_root)
    return {
        "schema_version": REPRESENTATIVE_CASE_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "visual_root_supplied": visual_root is not None,
        "visual_root": None if visual_root is None else str(visual_root),
        "owner_map_filename_template": OWNER_MAP_FILENAME_TEMPLATE,
        "pixel_interpretation": "never_opened_or_decoded_path_and_byte_digest_only",
        "cases": references,
    }


# ---------------------------------------------------------------------------
# Manifest and CLI
# ---------------------------------------------------------------------------


def _artifact_descriptor(path: Path | None) -> dict[str, Any] | None:
    """Exact sha256 and byte size of one emitted file, read back from disk.

    A manifest that only records a path is not sealed: the file could be
    silently rewritten afterward.  Reading the bytes back (rather than
    trusting whatever was passed to the writer) is what lets a downstream
    reader detect that drift.
    """

    if path is None:
        return None
    data = path.read_bytes()
    return {
        "path": str(path),
        "sha256": analyzer.sha256_bytes(data),
        "size_bytes": len(data),
    }


def build_manifest(
    *,
    artifacts: Artifacts,
    output_dir: Path,
    rendered_paths: Mapping[str, Path | None],
    representative_case_document: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    external_images: dict[str, Any] = {}
    if representative_case_document is not None:
        for case_name, entry in representative_case_document["cases"].items():
            source_image = entry.get("source_image") or {}
            if "sha256" in source_image:
                external_images[case_name] = {
                    "path": source_image["path"],
                    "sha256": source_image["sha256"],
                    "size_bytes": source_image["size_bytes"],
                    "gt_owner_id": entry["gt_owner_id"],
                    "image_id": entry["image_id"],
                }

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "analysis_input_digests": {
            "report_json_sha256": artifacts.receipt.get("report_json_sha256"),
            "report_md_sha256": artifacts.receipt.get("report_md_sha256"),
            "owner_records_jsonl_sha256": artifacts.receipt.get("owner_records_jsonl_sha256"),
            "receipt_content_sha256": artifacts.receipt.get("receipt_content_sha256"),
            "predecessor_run_root": artifacts.report.get("predecessor_run_root"),
            "predecessor_input_file_sha256": dict(artifacts.report.get("input_file_sha256") or {}),
        },
        "output_dir": str(output_dir),
        "rendered_paths": {
            name: _artifact_descriptor(path) for name, path in rendered_paths.items()
        },
        # Sealed byte-level descriptors of the exact predecessor
        # ``owner_map__<image_id>.png`` files referenced by the
        # representative-case selection.  These files are never copied,
        # rewritten, or pixel-interpreted; only their path and digest are
        # recorded here, mirroring representative-case-references.json.
        "external_representative_image_references": external_images,
        "semantics": list(analyzer.SEMANTICS_NOTES),
        "color_semantics": "categorical_discrete_readouts_never_raw_or_cross_image_logprob",
        "visualizer_source_sha256": analyzer.sha256_bytes(Path(__file__).resolve().read_bytes()),
    }
    manifest["visual_manifest_sha256"] = analyzer.sha256_json(manifest)
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analysis-root", type=Path, required=True, help="Directory containing report.json, owner-records.jsonl, receipt.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--specs-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--representative-visual-root",
        type=Path,
        default=None,
        help=(
            "Predecessor's sealed visual/combined directory; when supplied, exact "
            "owner_map__<image_id>.png files for the selected representative cases are "
            "hashed and referenced (never copied or pixel-interpreted)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and not args.force and any(output_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty {output_dir}; pass --force")

    try:
        artifacts = load_artifacts(Path(args.analysis_root))
    except VisualContractError as exc:
        raise SystemExit(f"visualization contract violated: {exc}") from exc

    fn_records = [row for row in artifacts.owner_records if row.get("cohort") == FN_COHORT]
    tp_records = [row for row in artifacts.owner_records if row.get("cohort") == TP_COHORT]

    owner_matrix_spec = build_owner_matrix_spec(fn_records)
    native_tp_matrix_spec = build_native_tp_matrix_spec(tp_records)
    summary_panel_spec = build_summary_panel_spec(artifacts.report)

    output_dir.mkdir(parents=True, exist_ok=True)
    owner_matrix_spec_path = output_dir / "owner-matrix-spec.json"
    native_tp_matrix_spec_path = output_dir / "native-tp-matrix-spec.json"
    summary_panel_spec_path = output_dir / "summary-panel-spec.json"
    owner_matrix_spec_path.write_text(
        json.dumps(owner_matrix_spec, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    native_tp_matrix_spec_path.write_text(
        json.dumps(native_tp_matrix_spec, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    summary_panel_spec_path.write_text(
        json.dumps(summary_panel_spec, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    rendered_paths: dict[str, Path | None] = {
        "owner_matrix_spec": owner_matrix_spec_path,
        "native_tp_matrix_spec": native_tp_matrix_spec_path,
        "summary_panel_spec": summary_panel_spec_path,
        "owner_matrix_png": None,
        "native_tp_matrix_png": None,
        "summary_panel_png": None,
    }
    if not args.specs_only:
        owner_matrix_png_path = output_dir / OWNER_MATRIX_NAME
        native_tp_matrix_png_path = output_dir / NATIVE_TP_MATRIX_NAME
        summary_panel_png_path = output_dir / SUMMARY_PANEL_NAME
        render_matrix_png(owner_matrix_spec, owner_matrix_png_path)
        render_matrix_png(native_tp_matrix_spec, native_tp_matrix_png_path)
        render_summary_panel_png(summary_panel_spec, summary_panel_png_path)
        rendered_paths["owner_matrix_png"] = owner_matrix_png_path
        rendered_paths["native_tp_matrix_png"] = native_tp_matrix_png_path
        rendered_paths["summary_panel_png"] = summary_panel_png_path

    # Representative-case selection is pure data over owner-records and is
    # always emitted; only the local PNG renders above are skipped in
    # --specs-only mode.  Sealing an external image is reading and hashing a
    # file that already exists, not rendering a new one, so it still happens
    # here even under --specs-only.
    try:
        representative_case_document = build_representative_case_document(
            fn_records, visual_root=args.representative_visual_root
        )
    except VisualContractError as exc:
        raise SystemExit(f"visualization contract violated: {exc}") from exc
    representative_case_references_path = output_dir / REPRESENTATIVE_CASE_REFERENCES_NAME
    representative_case_references_path.write_text(
        json.dumps(representative_case_document, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    rendered_paths["representative_case_references"] = representative_case_references_path

    manifest = build_manifest(
        artifacts=artifacts,
        output_dir=output_dir,
        rendered_paths=rendered_paths,
        representative_case_document=representative_case_document,
    )
    (output_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
