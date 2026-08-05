#!/usr/bin/env python3
"""Summarize iterative forced-continue extreme-capacity artifacts."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
import glob
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


SCHEMA_VERSION = "iterative_forced_continue_extreme_capacity_summary.v1"
RUN_SCHEMA_VERSIONS = {
    "iterative_forced_continue_extreme_capacity.v1",
    "iterative_forced_continue_exact_native.v2",
}
TOKEN_BUDGETS = (128, 256, 512, 1024, 2048, 3084)
FORCE_BUDGETS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256)


def _paths(patterns: Iterable[str]) -> list[Path]:
    found: set[Path] = set()
    for pattern in patterns:
        matches = glob.glob(str(pattern))
        if not matches and Path(pattern).exists():
            matches = [pattern]
        found.update(Path(value).expanduser().resolve(strict=True) for value in matches)
    if not found:
        raise ValueError("no input artifacts matched")
    return sorted(found)


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    if not denominator:
        return None
    return float(numerator) / float(denominator)


def _coverage_at(
    curve: Sequence[Mapping[str, Any]],
    *,
    field: str,
    budget: int,
) -> int:
    eligible = [point for point in curve if int(point[field]) <= int(budget)]
    if not eligible:
        return 0
    return int(eligible[-1]["coverage"])


def _summarize_arm(payload: Mapping[str, Any], *, source: Path) -> dict[str, Any]:
    cases = list(payload.get("cases") or [])
    if int(payload.get("case_count", -1)) != len(cases):
        raise ValueError(f"case_count mismatch in {source}")
    aggregate = Counter()
    terminal_reasons: Counter[str] = Counter()
    marginal_force_gains: Counter[int] = Counter()
    interval_force_gains: Counter[int] = Counter()
    token_curve_totals = Counter()
    force_curve_totals = Counter()
    case_receipts: list[dict[str, Any]] = []

    for case in cases:
        native = case["native_snapshot"]
        final = case["final_snapshot"]
        executed_ids = list(case.get("executed_completion_token_ids") or [])
        if executed_ids and len(executed_ids) != int(case["executed_completion_token_count"]):
            raise ValueError(f"executed token count mismatch for image {case['image_id']}")
        gt_count = int(final["gt_owner_count"])
        if int(native["gt_owner_count"]) != gt_count:
            raise ValueError(f"native/final GT count mismatch for image {case['image_id']}")
        native_tp = int(native["coverage"])
        final_tp = int(final["coverage"])
        if final_tp < native_tp:
            raise ValueError(f"cumulative coverage decreased for image {case['image_id']}")
        native_ids = {str(value) for value in native["matched_owner_ids"]}
        final_ids = {str(value) for value in final["matched_owner_ids"]}
        force_rows = list(case.get("forces") or [])
        if "force_count" in case and int(case["force_count"]) != len(force_rows):
            raise ValueError(f"force_count mismatch for image {case['image_id']}")
        productive = sum(int(item["marginal_owner_gain"]) > 0 for item in force_rows)
        zero_yield = sum(int(item["marginal_owner_gain"]) == 0 for item in force_rows)
        interval_productive = sum(
            int(item.get("interval_owner_gain", item["marginal_owner_gain"])) > 0
            for item in force_rows
        )
        interval_zero_yield = len(force_rows) - interval_productive
        for item in force_rows:
            marginal_force_gains[int(item["marginal_owner_gain"])] += 1
            interval_force_gains[
                int(item.get("interval_owner_gain", item["marginal_owner_gain"]))
            ] += 1
        aggregate.update(
            gt=gt_count,
            native_tp=native_tp,
            native_fn=gt_count - native_tp,
            native_predictions=int(native["prediction_count"]),
            native_fp=int(native["false_positive_count"]),
            final_tp=final_tp,
            final_fn=gt_count - final_tp,
            final_predictions=int(final["prediction_count"]),
            final_fp=int(final["false_positive_count"]),
            coverage_gain=final_tp - native_tp,
            force_count=len(force_rows),
            productive_force_count=productive,
            zero_yield_force_count=zero_yield,
            interval_productive_force_count=interval_productive,
            interval_zero_yield_force_count=interval_zero_yield,
            executed_tokens=int(case["executed_completion_token_count"]),
            completed_rows=int(case["complete_row_count"]),
        )
        if case.get("native_boundary_observed"):
            aggregate["native_boundary_observed_cases"] += 1
        if (case.get("native_replay") or {}).get("status") == "exact_match":
            aggregate["exact_native_replay_cases"] += 1
        terminal_reason = str(case["terminal_reason"])
        terminal_reasons[terminal_reason] += 1
        if terminal_reason == "all_gt_matched":
            aggregate["all_gt_matched_cases"] += 1
            probe = case.get("post_complete_boundary_probe") or {}
            if probe.get("would_naturally_stop"):
                aggregate["all_gt_natural_stop_probe_cases"] += 1

        curve = list(case.get("coverage_curve") or [])
        previous_token = -1
        previous_force = -1
        previous_coverage = -1
        for point in curve:
            token_count = int(point["executed_token_count"])
            force_count = int(point["force_count"])
            point_coverage = int(point["coverage"])
            if token_count < previous_token or force_count < previous_force:
                raise ValueError(f"coverage curve regressed in budget for image {case['image_id']}")
            if point_coverage < previous_coverage:
                raise ValueError(f"coverage curve regressed in coverage for image {case['image_id']}")
            previous_token = token_count
            previous_force = force_count
            previous_coverage = point_coverage
        if not curve or int(curve[-1]["coverage"]) != final_tp:
            raise ValueError(f"coverage curve does not end at final coverage for image {case['image_id']}")
        for budget in TOKEN_BUDGETS:
            token_curve_totals[budget] += _coverage_at(
                curve,
                field="executed_token_count",
                budget=budget,
            )
        for budget in FORCE_BUDGETS:
            force_curve_totals[budget] += _coverage_at(
                curve,
                field="force_count",
                budget=budget,
            )
        case_receipts.append(
            {
                "image_id": str(case["image_id"]),
                "gt_owner_count": gt_count,
                "native_coverage": native_tp,
                "final_coverage": final_tp,
                "coverage_gain": final_tp - native_tp,
                "native_owner_ids": sorted(native_ids),
                "final_owner_ids": sorted(final_ids),
                "owner_ids_added_under_final_assignment": sorted(final_ids - native_ids),
                "owner_ids_removed_under_final_assignment": sorted(native_ids - final_ids),
                "force_count": len(force_rows),
                "productive_force_count": productive,
                "zero_yield_force_count": zero_yield,
                "interval_productive_force_count": interval_productive,
                "interval_zero_yield_force_count": interval_zero_yield,
                "executed_completion_token_count": int(case["executed_completion_token_count"]),
                "complete_row_count": int(case["complete_row_count"]),
                "terminal_reason": terminal_reason,
                "post_complete_would_naturally_stop": (
                    case.get("post_complete_boundary_probe") or {}
                ).get("would_naturally_stop"),
                "native_generated_token_ids_sha256": native.get(
                    "generated_token_ids_sha256"
                ),
                "native_replay_status": (case.get("native_replay") or {}).get("status"),
            }
        )

    gt = int(aggregate["gt"])
    native_tp = int(aggregate["native_tp"])
    final_tp = int(aggregate["final_tp"])
    native_predictions = int(aggregate["native_predictions"])
    final_predictions = int(aggregate["final_predictions"])
    native_precision = _safe_ratio(native_tp, native_predictions)
    final_precision = _safe_ratio(final_tp, final_predictions)
    native_recall = _safe_ratio(native_tp, gt)
    final_recall = _safe_ratio(final_tp, gt)

    def f1(precision: float | None, recall: float | None) -> float | None:
        if precision is None or recall is None or precision + recall == 0.0:
            return None
        return 2.0 * precision * recall / (precision + recall)

    config = dict(payload["config"])
    return {
        "source_artifact": str(source),
        "run_schema_version": str(payload["schema_version"]),
        "infer_config": config["infer_config"],
        "infer_config_sha256": config["infer_config_sha256"],
        "repetition_penalty": float(config["repetition_penalty"]),
        "case_count": len(cases),
        "aggregate": {
            **dict(aggregate),
            "native_precision": native_precision,
            "native_recall": native_recall,
            "native_f1": f1(native_precision, native_recall),
            "final_precision": final_precision,
            "final_recall": final_recall,
            "final_f1": f1(final_precision, final_recall),
            "force_productivity": _safe_ratio(
                aggregate["interval_productive_force_count"], aggregate["force_count"]
            ),
            "interval_force_productivity": _safe_ratio(
                aggregate["interval_productive_force_count"], aggregate["force_count"]
            ),
            "immediate_forced_row_productivity": _safe_ratio(
                aggregate["productive_force_count"], aggregate["force_count"]
            ),
            "owners_per_force": _safe_ratio(aggregate["coverage_gain"], aggregate["force_count"]),
            "all_gt_natural_stop_rate": _safe_ratio(
                aggregate["all_gt_natural_stop_probe_cases"], aggregate["all_gt_matched_cases"]
            ),
        },
        "terminal_reason_counts": dict(sorted(terminal_reasons.items())),
        "marginal_force_gain_histogram": {
            str(key): value for key, value in sorted(marginal_force_gains.items())
        },
        "interval_force_gain_histogram": {
            str(key): value for key, value in sorted(interval_force_gains.items())
        },
        "equal_token_budget_curve": [
            {
                "executed_token_budget": budget,
                "coverage": int(token_curve_totals[budget]),
                "recall": _safe_ratio(token_curve_totals[budget], gt),
            }
            for budget in TOKEN_BUDGETS
        ],
        "equal_force_count_curve": [
            {
                "force_budget": budget,
                "coverage": int(force_curve_totals[budget]),
                "recall": _safe_ratio(force_curve_totals[budget], gt),
            }
            for budget in FORCE_BUDGETS
        ],
        "cases": sorted(case_receipts, key=lambda item: int(item["image_id"])),
    }


def summarize(patterns: Iterable[str]) -> dict[str, Any]:
    arms: list[dict[str, Any]] = []
    seen: set[tuple[str, float]] = set()
    run_schema_versions: set[str] = set()
    for path in _paths(patterns):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") not in RUN_SCHEMA_VERSIONS:
            raise ValueError(f"unsupported schema in {path}: {payload.get('schema_version')!r}")
        run_schema_versions.add(str(payload["schema_version"]))
        arm = _summarize_arm(payload, source=path)
        identity = (str(arm["infer_config_sha256"]), float(arm["repetition_penalty"]))
        if identity in seen:
            raise ValueError(f"duplicate arm identity: {identity}")
        seen.add(identity)
        arms.append(arm)
    if len(run_schema_versions) != 1:
        raise ValueError(
            "refusing to mix rowwise-restart and exact-native run schemas in one summary"
        )
    arms.sort(key=lambda item: (str(item["infer_config"]), float(item["repetition_penalty"])))
    return {
        "schema_version": SCHEMA_VERSION,
        "claim_boundary": (
            "GT-aware iterative forced-continuation capacity; native and forced metrics must not be conflated"
        ),
        "run_schema_version": next(iter(run_schema_versions)),
        "arm_count": len(arms),
        "arms": arms,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite {output}; pass --force")
    payload = summarize(args.input)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
