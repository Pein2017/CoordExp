#!/usr/bin/env python3
"""Summarize the frozen 200-boundary natural-terminal diagnostic panel."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
import json
import math
from pathlib import Path
import sys
from typing import Any, cast

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research.summarize_continuation_locality_owner_compositionality import (  # noqa: E402
    _bootstrap_mean_interval,
    _describe,
    _load_locality,
    _read_json,
    _receipt_files,
)
from src.config.fingerprint import sha256_file  # noqa: E402


SCHEMA_VERSION = "untouched_terminal_boundary_statistics.summary.v1"
CASE_SCHEMA_VERSION = "untouched_terminal_boundary_statistics.case.v1"
UNIT_ID = "2026-07-25-untouched-terminal-boundary-statistical-analysis"
SOURCE_UNIT_ID = "2026-07-25-continuation-locality-and-exact-prefix-owner-compositionality"
COHORT = "untouched_terminal_with_remaining_owner"
ROLES = (
    "source",
    "transition-step36",
    "pairwise-lr3e6-step90",
    "owner-conditioned-lr1e5-step90",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            values.append(value)
    return values


def _wilson_interval(successes: int, count: int) -> list[float] | None:
    if count <= 0:
        return None
    z = 1.959963984540054
    proportion = successes / count
    denominator = 1.0 + z * z / count
    center = (proportion + z * z / (2.0 * count)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / count + z * z / (4.0 * count * count)
        )
        / denominator
    )
    return [max(0.0, center - half_width), min(1.0, center + half_width)]


def _fraction(successes: int, count: int) -> dict[str, Any]:
    return {
        "success_count": successes,
        "count": count,
        "fraction": None if count == 0 else successes / count,
        "wilson_95_interval": _wilson_interval(successes, count),
    }


def _threshold_summary(values: Sequence[float]) -> dict[str, Any]:
    data = [float(value) for value in values]
    count = len(data)
    return {
        "margin": _describe(data),
        "row_opener_over_terminal": _fraction(sum(value > 0.0 for value in data), count),
        "margin_greater_than_0_5": _fraction(sum(value > 0.5 for value in data), count),
        "margin_greater_than_1_0": _fraction(sum(value > 1.0 for value in data), count),
        "within_0_5_of_pairwise_boundary": _fraction(
            sum(abs(value) <= 0.5 for value in data), count
        ),
        "terminal_ahead_by_at_least_2_0": _fraction(sum(value <= -2.0 for value in data), count),
    }


def _paired_sign_summary(source: Sequence[float], other: Sequence[float]) -> dict[str, Any]:
    if len(source) != len(other):
        raise ValueError("paired sign inputs differ in length")
    counts: Counter[str] = Counter()
    for source_value, other_value in zip(source, other, strict=True):
        source_positive = source_value > 0.0
        other_positive = other_value > 0.0
        counts[f"{str(source_positive).lower()}_to_{str(other_positive).lower()}"] += 1
    return {
        "source_nonpositive_to_checkpoint_nonpositive": counts["false_to_false"],
        "source_nonpositive_to_checkpoint_positive": counts["false_to_true"],
        "source_positive_to_checkpoint_nonpositive": counts["true_to_false"],
        "source_positive_to_checkpoint_positive": counts["true_to_true"],
    }


def _spearman(first: Sequence[float], second: Sequence[float]) -> dict[str, Any]:
    if len(first) != len(second):
        raise ValueError("correlation inputs differ in length")
    from scipy.stats import spearmanr

    result = cast(Any, spearmanr(first, second))
    coefficient = float(result[0])
    pvalue = float(result[1])
    return {
        "count": len(first),
        "coefficient": None if math.isnan(coefficient) else coefficient,
        "two_sided_pvalue": None if math.isnan(pvalue) else pvalue,
        "claim_boundary": "descriptive within the deterministic frozen panel",
    }


def _depth_bin(value: int) -> str:
    if value <= 2:
        return "depth_1_to_2"
    if value <= 5:
        return "depth_3_to_5"
    if value <= 9:
        return "depth_6_to_9"
    return "depth_10_plus"


def _remaining_owner_bin(value: int) -> str:
    if value == 1:
        return "remaining_1"
    if value <= 3:
        return "remaining_2_to_3"
    if value <= 7:
        return "remaining_4_to_7"
    return "remaining_8_plus"


def _source_margin_bin(value: float) -> str:
    if value > 0.0:
        return "source_diagnostic_positive_despite_observed_stop"
    if value >= -0.5:
        return "source_terminal_ahead_by_less_than_0_5"
    if value > -2.0:
        return "source_terminal_ahead_by_0_5_to_2_0"
    return "source_terminal_ahead_by_at_least_2_0"


def _strata(
    cases: Sequence[Mapping[str, Any]],
    *,
    key: Callable[[Mapping[str, Any]], str],
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[key(case)].append(case)
    result: dict[str, Any] = {}
    for name, values in sorted(grouped.items()):
        result[name] = {
            "boundary_count": len(values),
            "verified_remaining_owner_count": sum(
                int(value["remaining_owner_count"]) for value in values
            ),
            "checkpoint_margins": {
                role: _threshold_summary(
                    [float(value["checkpoint_margins"][role]) for value in values]
                )
                for role in ROLES
            },
            "checkpoint_minus_source": {
                role: _describe(
                    [float(value["checkpoint_minus_source"][role]) for value in values]
                )
                for role in ROLES
            },
        }
    return result


def _candidate_category_index(path: Path) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    for row in _read_jsonl(path):
        image_id = str(int(row["image_id"]))
        by_owner: dict[str, str] = {}
        for value in row.get("objects", []):
            if not isinstance(value, Mapping):
                raise ValueError(f"image {image_id} contains a malformed object")
            owner_id = str(value["coco_ann_id"])
            category = str(value.get("category_name", value.get("desc", "")))
            if not category:
                raise ValueError(f"image {image_id} owner {owner_id} lacks a category")
            by_owner[owner_id] = category
        index[image_id] = by_owner
    return index


def _leave_one_out_influence(
    cases: Sequence[Mapping[str, Any]], *, role: str
) -> dict[str, Any]:
    deltas = [float(case["checkpoint_minus_source"][role]) for case in cases]
    total = sum(deltas)
    count = len(deltas)
    if count <= 1:
        raise ValueError("leave-one-out influence requires at least two cases")
    leave_one_out = [(total - value) / (count - 1) for value in deltas]
    ranked = sorted(
        zip(cases, deltas, strict=True), key=lambda item: abs(item[1]), reverse=True
    )
    return {
        "full_mean": total / count,
        "leave_one_out_mean_minimum": min(leave_one_out),
        "leave_one_out_mean_maximum": max(leave_one_out),
        "largest_absolute_changes": [
            {
                "boundary_id": str(case["boundary_id"]),
                "image_id": str(case["image_id"]),
                "checkpoint_minus_source": delta,
            }
            for case, delta in ranked[:10]
        ],
    }


def _set_overlap(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    positive = {
        role: {
            str(case["boundary_id"])
            for case in cases
            if float(case["checkpoint_margins"][role]) > 0.0
        }
        for role in ROLES
    }
    comparisons: dict[str, Any] = {}
    for left_index, left in enumerate(ROLES):
        for right in ROLES[left_index + 1 :]:
            intersection = positive[left] & positive[right]
            union = positive[left] | positive[right]
            comparisons[f"{left}__{right}"] = {
                "left_positive_count": len(positive[left]),
                "right_positive_count": len(positive[right]),
                "intersection_count": len(intersection),
                "left_only_count": len(positive[left] - positive[right]),
                "right_only_count": len(positive[right] - positive[left]),
                "jaccard": None if not union else len(intersection) / len(union),
            }
    return comparisons


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.manifest.expanduser().resolve(strict=True)
    locality_root = args.locality_root.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite immutable output root: {output_root}")

    manifest = _read_json(manifest_path)
    if manifest.get("unit_id") != SOURCE_UNIT_ID:
        raise ValueError(f"unexpected source unit: {manifest.get('unit_id')}")
    manifest_hash = sha256_file(manifest_path)
    roles, receipt_provenance = _load_locality(locality_root)
    for path in _receipt_files(locality_root):
        receipt = _read_json(path)
        if receipt["manifest"]["sha256"] != manifest_hash:
            raise ValueError(f"receipt manifest mismatch: {path}")

    manifest_values = [
        dict(value)
        for value in manifest.get("locality_boundaries", [])
        if value.get("cohort") == COHORT
    ]
    if len(manifest_values) != 200:
        raise ValueError(f"expected 200 terminal boundaries, found {len(manifest_values)}")
    manifest_by_id = {str(value["boundary_id"]): value for value in manifest_values}
    if len(manifest_by_id) != len(manifest_values):
        raise ValueError("duplicate terminal boundary IDs in manifest")

    terminal_by_role: dict[str, dict[str, Any]] = {}
    for role in ROLES:
        values = {
            boundary_id: record
            for boundary_id, record in roles[role].items()
            if record["cohort"] == COHORT
        }
        if set(values) != set(manifest_by_id):
            raise ValueError(f"terminal boundary coverage mismatch for {role}")
        terminal_by_role[role] = values

    candidate_input = manifest["inputs"]["candidate_pool"]
    candidate_path = Path(str(candidate_input["path"])).resolve(strict=True)
    if sha256_file(candidate_path) != candidate_input["sha256"]:
        raise ValueError("candidate pool hash mismatch")
    category_index = _candidate_category_index(candidate_path)

    cases: list[dict[str, Any]] = []
    for boundary_id in sorted(manifest_by_id):
        boundary = manifest_by_id[boundary_id]
        image_id = str(boundary["image_id"])
        owner_ids = [str(value) for value in boundary["census_uncovered_owner_ids_at_stop"]]
        owner_categories: list[str] = []
        for owner_id in owner_ids:
            suffix = owner_id.split(":", 1)[-1]
            try:
                owner_categories.append(category_index[image_id][suffix])
            except KeyError as error:
                raise ValueError(
                    f"missing candidate category for image {image_id} owner {owner_id}"
                ) from error
        remaining_count = int(boundary["remaining_annotation_owner_count"])
        if remaining_count != len(owner_ids):
            raise ValueError(f"remaining-owner count mismatch for {boundary_id}")
        margins = {
            role: float(
                terminal_by_role[role][boundary_id]["terminal_boundary"][
                    "row_entry_minus_terminal"
                ]
            )
            for role in ROLES
        }
        cases.append(
            {
                "schema_version": CASE_SCHEMA_VERSION,
                "boundary_id": boundary_id,
                "image_id": image_id,
                "prefix_depth": int(boundary["prefix_depth"]),
                "object_count_band": str(boundary["object_count_band"]),
                "annotation_object_count": int(boundary["annotation_object_count"]),
                "remaining_owner_count": remaining_count,
                "remaining_owner_ids": owner_ids,
                "remaining_owner_categories": owner_categories,
                "observed_source_action": str(boundary["observed_next_action"]),
                "source_natural_end": bool(boundary["natural_end"]),
                "prefix_token_ids_sha256": str(boundary["prefix_token_ids_sha256"]),
                "checkpoint_margins": margins,
                "checkpoint_minus_source": {
                    role: margins[role] - margins["source"] for role in ROLES
                },
            }
        )

    if any(
        case["observed_source_action"] != "terminal" or not case["source_natural_end"]
        for case in cases
    ):
        raise ValueError("panel contains a boundary that is not an observed Source natural stop")

    source_margins = [float(case["checkpoint_margins"]["source"]) for case in cases]
    checkpoint_summaries: dict[str, Any] = {}
    for role in ROLES:
        values = [float(case["checkpoint_margins"][role]) for case in cases]
        deltas = [float(case["checkpoint_minus_source"][role]) for case in cases]
        checkpoint_summaries[role] = {
            "absolute_pairwise_margin": _threshold_summary(values),
            "checkpoint_minus_source": _describe(deltas),
            "checkpoint_minus_source_bootstrap_mean": _bootstrap_mean_interval(
                deltas, seed=250725 + sum(ord(value) for value in role)
            ),
            "paired_signs_relative_to_source_diagnostic": _paired_sign_summary(
                source_margins, values
            ),
        }

    numeric_fields = {
        "source_pairwise_margin": source_margins,
        "prefix_depth": [float(case["prefix_depth"]) for case in cases],
        "annotation_object_count": [
            float(case["annotation_object_count"]) for case in cases
        ],
        "remaining_owner_count": [float(case["remaining_owner_count"]) for case in cases],
    }
    correlations: dict[str, Any] = {}
    for role in ROLES[1:]:
        delta = [float(case["checkpoint_minus_source"][role]) for case in cases]
        correlations[role] = {
            field: _spearman(values, delta) for field, values in numeric_fields.items()
        }
    correlations["between_checkpoint_deltas"] = {}
    for left_index, left in enumerate(ROLES[1:]):
        left_values = [float(case["checkpoint_minus_source"][left]) for case in cases]
        for right in ROLES[1:][left_index + 1 :]:
            right_values = [float(case["checkpoint_minus_source"][right]) for case in cases]
            correlations["between_checkpoint_deltas"][f"{left}__{right}"] = _spearman(
                left_values, right_values
            )

    category_counts = Counter(
        category for case in cases for category in case["remaining_owner_categories"]
    )
    category_image_counts: Counter[str] = Counter()
    for case in cases:
        category_image_counts.update(set(case["remaining_owner_categories"]))

    source_positive_cases = [
        {
            "boundary_id": case["boundary_id"],
            "image_id": case["image_id"],
            "source_pairwise_margin": case["checkpoint_margins"]["source"],
            "remaining_owner_count": case["remaining_owner_count"],
            "remaining_owner_categories": case["remaining_owner_categories"],
        }
        for case in cases
        if float(case["checkpoint_margins"]["source"]) > 0.0
    ]

    import scipy

    result = {
        "schema_version": SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "inputs": {
            "manifest": {"path": str(manifest_path), "sha256": manifest_hash},
            "locality_receipts": receipt_provenance,
            "candidate_pool": {
                "path": str(candidate_path),
                "sha256": sha256_file(candidate_path),
            },
        },
        "runtime": {"scipy_version": scipy.__version__},
        "panel": {
            "boundary_count": len(cases),
            "unique_image_count": len({case["image_id"] for case in cases}),
            "verified_remaining_owner_count": sum(
                int(case["remaining_owner_count"]) for case in cases
            ),
            "single_remaining_owner_boundary_count": sum(
                int(case["remaining_owner_count"]) == 1 for case in cases
            ),
            "multiple_remaining_owner_boundary_count": sum(
                int(case["remaining_owner_count"]) > 1 for case in cases
            ),
            "remaining_owner_count_per_boundary": _describe(
                [float(case["remaining_owner_count"]) for case in cases]
            ),
            "annotation_object_count": _describe(
                [float(case["annotation_object_count"]) for case in cases]
            ),
            "prefix_depth": _describe([float(case["prefix_depth"]) for case in cases]),
            "object_count_band_counts": dict(
                sorted(Counter(case["object_count_band"] for case in cases).items())
            ),
            "remaining_owner_category_count": len(category_counts),
            "top_4_category_owner_share": (
                sum(value for _, value in category_counts.most_common(4))
                / sum(category_counts.values())
            ),
            "top_10_category_owner_share": (
                sum(value for _, value in category_counts.most_common(10))
                / sum(category_counts.values())
            ),
            "remaining_owner_categories": [
                {
                    "category": category,
                    "owner_count": owner_count,
                    "image_count": category_image_counts[category],
                }
                for category, owner_count in category_counts.most_common()
            ],
            "selection_boundary": (
                "deterministic depth-first then stable-hash selection; not a random sample of all stops"
            ),
        },
        "checkpoint_summaries": checkpoint_summaries,
        "positive_margin_set_overlap": _set_overlap(cases),
        "strata": {
            "prefix_depth_bins": _strata(
                cases, key=lambda case: _depth_bin(int(case["prefix_depth"]))
            ),
            "object_count_bands": _strata(
                cases, key=lambda case: str(case["object_count_band"])
            ),
            "remaining_owner_count_bins": _strata(
                cases,
                key=lambda case: _remaining_owner_bin(int(case["remaining_owner_count"])),
            ),
            "source_margin_bins": _strata(
                cases,
                key=lambda case: _source_margin_bin(
                    float(case["checkpoint_margins"]["source"])
                ),
            ),
        },
        "spearman_correlations": correlations,
        "influence": {
            role: _leave_one_out_influence(cases, role=role) for role in ROLES[1:]
        },
        "diagnostic_inconsistencies": {
            "observed_source_terminal_but_fp32_pairwise_margin_positive_count": len(
                source_positive_cases
            ),
            "cases": source_positive_cases,
            "interpretation": (
                "the FP32 two-token margin is diagnostic and does not reconstruct the historical decode"
            ),
        },
        "claim_boundary": (
            "read-only statistics on a deterministic 200-boundary panel; no margin crossing is a decode flip or owner recovery"
        ),
    }

    output_root.mkdir(parents=True)
    (output_root / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_root / "cases.jsonl").open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize the frozen 200-boundary natural-terminal diagnostic panel."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--locality-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main() -> None:
    result = summarize(build_parser().parse_args())
    print(
        json.dumps(
            {
                "boundary_count": result["panel"]["boundary_count"],
                "verified_remaining_owner_count": result["panel"][
                    "verified_remaining_owner_count"
                ],
                "output_schema": result["schema_version"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
