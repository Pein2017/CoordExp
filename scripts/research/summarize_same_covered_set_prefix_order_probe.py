#!/usr/bin/env python3
"""Summarize same-covered-set prefix-order probe result artifacts.

This file intentionally contains only mechanical aggregation.  It does not
choose scientific thresholds or emit a prose verdict.  A run's physical owner
is derived from the runner's IoU match records; unmatched and failed runs are
kept as statuses rather than silently being treated as hallucinations.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research.run_same_covered_set_prefix_order_probe import (
    ARMS,
    SCHEMA_VERSION,
    SCHEMA_VERSION_V2,
    validate_artifact_payload,
)


def _as_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _run_key(case_id: str, arm_name: str, run: Mapping[str, Any]) -> tuple[str, str, str, str]:
    seed = run.get("seed")
    return (case_id, arm_name, str(run.get("mode")), "none" if seed is None else str(seed))


def _owner_labels(run: Mapping[str, Any]) -> tuple[str, ...]:
    """Return sorted unique matched physical entity ids for one free run."""

    if str(run.get("status")) != "success":
        return ()
    matches = run.get("entity_matches")
    if not isinstance(matches, list):
        return ()
    owners = {
        str(match["matched_entity_id"])
        for match in matches
        if isinstance(match, Mapping)
        and match.get("status") == "matched"
        and match.get("matched_entity_id") is not None
    }
    return tuple(sorted(owners))


def _permutation_distance(arm: Mapping[str, Any]) -> Any:
    """Read the explicit inversion-count field and older aliases.

    The case builder uses the long field name so durable artifacts explain the
    metric without requiring a separate abbreviation. Older hand-authored v2
    cases may use one of the shorter aliases, which remain supported here.
    """

    return arm.get(
        "permutation_inversion_count_relative_to_canonical_order",
        arm.get("permutation_distance", arm.get("permutation_distance_from_canonical")),
    )


def _primary_owner(run: Mapping[str, Any]) -> str:
    owners = _owner_labels(run)
    if len(owners) == 1:
        return owners[0]
    if len(owners) > 1:
        return "multiple"
    if str(run.get("status")) != "success":
        return "failed"
    return "none"


def _status_counts(run: Mapping[str, Any]) -> Counter[str]:
    counts: Counter[str] = Counter()
    if str(run.get("status")) != "success":
        counts[str(run.get("status"))] += 1
        return counts
    matches = run.get("entity_matches")
    if not isinstance(matches, list) or not matches:
        counts["no_entity_matches"] += 1
        return counts
    for match in matches:
        if isinstance(match, Mapping):
            counts[str(match.get("status", "missing_status"))] += 1
        else:
            counts["invalid_match_record"] += 1
    return counts


def _run_summary(arm_name: str, arm: Mapping[str, Any]) -> dict[str, Any]:
    covered = {str(value) for value in arm.get("entity_ids", [])}
    runs = arm.get("runs")
    if not isinstance(runs, list):
        raise ValueError(f"arm {arm_name}.runs must be a list")
    primary_counts: Counter[str] = Counter()
    owner_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    covered_counts: Counter[str] = Counter()
    uncovered_counts: Counter[str] = Counter()
    run_rows: list[dict[str, Any]] = []
    for index, raw_run in enumerate(runs):
        run = _as_mapping(raw_run, f"arm {arm_name} run {index}")
        owners = _owner_labels(run)
        primary = _primary_owner(run)
        primary_counts[primary] += 1
        status_counts.update(_status_counts(run))
        for owner in owners:
            owner_counts[owner] += 1
            (covered_counts if owner in covered else uncovered_counts)[owner] += 1
        run_rows.append(
            {
                "mode": run.get("mode"),
                "seed": run.get("seed"),
                "status": run.get("status"),
                "primary_owner": primary,
                "owner_ids": list(owners),
                "raw_generated_token_ids_sha256": run.get("raw_generated_token_ids_sha256"),
                "stop_reason": _as_mapping(run.get("row_stop", {}), "row_stop").get("stop_reason"),
            }
        )
    return {
        "entity_ids": sorted(covered),
        "permutation_distance": _permutation_distance(arm),
        "run_count": len(runs),
        "primary_owner_counts": dict(sorted(primary_counts.items())),
        "physical_owner_counts": dict(sorted(owner_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "covered_owner_counts": dict(sorted(covered_counts.items())),
        "uncovered_owner_counts": dict(sorted(uncovered_counts.items())),
        "runs": run_rows,
    }


def _horizon_owner_labels(run: Mapping[str, Any]) -> tuple[str, ...]:
    owners: set[str] = set()
    for row in run.get("rows", []):
        if not isinstance(row, Mapping):
            continue
        owners.update(str(value) for value in row.get("owner_entity_ids", []) if value is not None)
    return tuple(sorted(owners))


def _v2_run_summary(arm_name: str, arm: Mapping[str, Any], *, comparison_id: str | None = None) -> dict[str, Any]:
    covered = {str(value) for value in arm.get("entity_ids", [])}
    primary_counts: Counter[str] = Counter()
    owner_counts: Counter[str] = Counter()
    horizon_counts: Counter[str] = Counter()
    covered_owner_counts: Counter[str] = Counter()
    uncovered_owner_counts: Counter[str] = Counter()
    unmatched_or_ambiguous_count = 0
    rows_attempted = 0
    rows_accepted = 0
    run_rows: list[dict[str, Any]] = []
    for raw_run in arm.get("runs", []):
        run = _as_mapping(raw_run, f"v2 arm {arm_name} run")
        if comparison_id is not None and str(run.get("comparison_id")) != str(comparison_id):
            continue
        owners = _horizon_owner_labels(run)
        rows = run.get("rows", []) if isinstance(run.get("rows", []), list) else []
        first_row = _as_mapping(rows[0], "v2 first generated row") if rows else None
        first_owners = tuple(str(value) for value in (first_row or {}).get("owner_entity_ids", []) if value is not None)
        first_owner = first_owners[0] if len(first_owners) == 1 else "multiple" if len(first_owners) > 1 else ("failed" if run.get("status") != "success" else "none")
        primary = first_owner
        primary_counts[primary] += 1
        owner_counts.update(owners)
        rows_attempted += len(rows)
        rows_accepted += sum(bool(row.get("accepted_complete_row")) for row in rows if isinstance(row, Mapping))
        horizon_counts[str(run.get("horizon_rows_generated", 0))] += 1
        row_trajectory = []
        for row in rows:
            row_map = _as_mapping(row, "v2 generated row")
            row_trajectory.append({
                "row_index": row_map.get("row_index"),
                "owner_entity_ids": list(row_map.get("owner_entity_ids", [])),
                "strict_matched_owner_ids": list(row_map.get("strict_matched_owner_ids", row_map.get("owner_entity_ids", []))),
                "covered_prefix_owner_ids": list(row_map.get("covered_prefix_owner_ids", [])),
                "uncovered_ledger_owner_ids": list(row_map.get("uncovered_ledger_owner_ids", [])),
                "unmatched_or_ambiguous_prediction_indices": list(row_map.get("unmatched_or_ambiguous_prediction_indices", [])),
                "owner_recurrence": list(row_map.get("owner_recurrence", [])),
                "cumulative_owner_entity_ids": list(row_map.get("cumulative_owner_entity_ids", [])),
                "status": row_map.get("status", "success"),
                "stop_reason": _as_mapping(row_map.get("row_stop", {}), "v2 row_stop").get("stop_reason"),
                "input_prefix_token_ids_sha256": row_map.get("input_prefix_token_ids_sha256"),
                "cumulative_prefix_token_ids_sha256": row_map.get("cumulative_prefix_token_ids_sha256"),
            })
        for item in row_trajectory:
            covered_owner_counts.update(item["covered_prefix_owner_ids"])
            uncovered_owner_counts.update(item["uncovered_ledger_owner_ids"])
            unmatched_or_ambiguous_count += len(item["unmatched_or_ambiguous_prediction_indices"])
        run_rows.append({
            "comparison_id": run.get("comparison_id"),
            "mode": run.get("mode"),
            "seed": run.get("seed"),
            "status": run.get("status"),
            "first_row_status": None if first_row is None else first_row.get("status", "success"),
            "first_row_owner": first_owner,
            "horizon_complete": run.get("horizon_complete"),
            "horizon_rows_generated": run.get("horizon_rows_generated"),
            "owner_ids": list(owners),
            "cumulative_owner_entity_ids": list(run.get("cumulative_owner_entity_ids", [])),
            "initial_prefix_token_ids_sha256": run.get("initial_prefix_token_ids_sha256"),
            "final_prefix_token_ids_sha256": run.get("final_prefix_token_ids_sha256"),
            "ordered_owner_trajectory": row_trajectory,
            "per_row_stop_reasons": [item["stop_reason"] for item in row_trajectory],
            "cumulative_unique_owner_ids": list(run.get("cumulative_owner_entity_ids", [])),
            "covered_recurrence_by_row": [
                {
                    "row_index": item["row_index"],
                    "owner_entity_ids": item["owner_entity_ids"],
                    "strict_matched_owner_ids": item["strict_matched_owner_ids"],
                    "uncovered_ledger_owner_ids": item["uncovered_ledger_owner_ids"],
                    "covered_owner_entity_ids": [owner for owner in item["owner_entity_ids"] if owner in covered],
                    "recurrence": item["owner_recurrence"],
                }
                for item in row_trajectory
            ],
        })
    return {
        "entity_ids": sorted(covered),
        "permutation_distance": _permutation_distance(arm),
        "run_count": len(run_rows),
        # ``generated_row_count`` is retained for compatibility and now means
        # accepted complete rows.  The two explicit fields remove ambiguity
        # for horizon analyses that retain terminal/malformed attempts.
        "generated_row_count": rows_accepted,
        "accepted_row_count": rows_accepted,
        "attempted_row_count": rows_attempted,
        "horizon_rows_generated_counts": dict(sorted(horizon_counts.items())),
        "primary_owner_counts": dict(sorted(primary_counts.items())),
        "physical_owner_counts": dict(sorted(owner_counts.items())),
        "covered_prefix_owner_counts": dict(sorted(covered_owner_counts.items())),
        "uncovered_ledger_owner_counts": dict(sorted(uncovered_owner_counts.items())),
        "unmatched_or_ambiguous_prediction_count": unmatched_or_ambiguous_count,
        "runs": run_rows,
    }


def _v2_case_summary(case_id: str, case: Mapping[str, Any]) -> dict[str, Any]:
    arms = _as_mapping(case.get("arms"), f"case {case_id}.arms")
    comparisons: list[dict[str, Any]] = []
    for raw_comparison in case.get("comparisons", []):
        comparison = _as_mapping(raw_comparison, f"case {case_id} comparison")
        names = [str(value) for value in comparison.get("arm_names", [])]
        comparisons.append({
            "comparison_id": comparison.get("comparison_id"),
            "arm_names": names,
            "shared_suffix_length": comparison.get("shared_suffix_length"),
            "rollout_horizon_rows": comparison.get("rollout_horizon_rows"),
            "permutation_distance": comparison.get("permutation_distance"),
            "require_same_covered_set": comparison.get("require_same_covered_set", True),
            "invariants": _as_mapping(case.get("invariants_by_comparison", {}), "invariants_by_comparison").get(str(comparison.get("comparison_id")), {}),
            "arms": {name: _v2_run_summary(name, _as_mapping(arms.get(name), f"arm {name}"), comparison_id=str(comparison.get("comparison_id"))) for name in names},
        })
    return {
        "case_id": case_id,
        "image_id": case.get("image_id"),
        "comparisons": comparisons,
        "arms": {str(name): _v2_run_summary(str(name), _as_mapping(arm, f"arm {name}")) for name, arm in arms.items()},
    }


def _paired_seed_comparison(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    def by_seed(arm: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
        result: dict[int, Mapping[str, Any]] = {}
        for raw_run in arm.get("runs", []):
            run = _as_mapping(raw_run, "paired run")
            if run.get("mode") != "sample" or run.get("seed") is None:
                continue
            result[int(run["seed"])] = run
        return result

    left_runs, right_runs = by_seed(left), by_seed(right)
    seeds = sorted(set(left_runs) & set(right_runs))
    owner_equal = 0
    raw_equal = 0
    both_success = 0
    pairs: list[dict[str, Any]] = []
    for seed in seeds:
        lrun, rrun = left_runs[seed], right_runs[seed]
        lowner, rowner = _primary_owner(lrun), _primary_owner(rrun)
        same_owner = lowner == rowner
        if same_owner:
            owner_equal += 1
        raw_same: bool | None = None
        if lrun.get("status") == "success" and rrun.get("status") == "success":
            both_success += 1
            raw_same = lrun.get("raw_generated_token_ids") == rrun.get("raw_generated_token_ids")
            if raw_same:
                raw_equal += 1
        pairs.append(
            {
                "seed": seed,
                "left_owner": lowner,
                "right_owner": rowner,
                "owner_equal": same_owner,
                "both_success": lrun.get("status") == "success" and rrun.get("status") == "success",
                "raw_generated_row_equal": raw_same,
            }
        )
    return {
        "shared_seed_count": len(seeds),
        "owner_agreement_count": owner_equal,
        "owner_agreement_rate": owner_equal / len(seeds) if seeds else None,
        "both_success_count": both_success,
        "raw_generated_row_agreement_count": raw_equal,
        "raw_generated_row_agreement_rate": raw_equal / both_success if both_success else None,
        "pairs": pairs,
    }


def _case_summary(case_id: str, case: Mapping[str, Any]) -> dict[str, Any]:
    arms = _as_mapping(case.get("arms"), f"case {case_id}.arms")
    for arm_name in ARMS:
        if arm_name not in arms:
            raise ValueError(f"case {case_id} is missing arm {arm_name}")
    arm_summaries = {arm_name: _run_summary(arm_name, _as_mapping(arms[arm_name], f"arm {arm_name}")) for arm_name in ARMS}
    left, right, control = (arms[name] for name in ARMS)
    left_runs = _as_mapping(left, "left arm")
    right_runs = _as_mapping(right, "right arm")
    control_runs = _as_mapping(control, "coverage control arm")
    left_entity_ids = [str(value) for value in left_runs.get("entity_ids", [])]
    right_entity_ids = [str(value) for value in right_runs.get("entity_ids", [])]
    control_entity_ids = [str(value) for value in control_runs.get("entity_ids", [])]
    if len(left_entity_ids) < 3 or len(right_entity_ids) < 3 or len(control_entity_ids) < 2:
        raise ValueError(f"case {case_id} arms do not contain A/B/C entity ids")
    inferred_a = left_entity_ids[0]
    if right_entity_ids[1] != inferred_a:
        raise ValueError(f"case {case_id} A entity differs between order arms")
    if control_entity_ids[0] != right_entity_ids[0]:
        raise ValueError(f"case {case_id} B entity differs between order arms")
    if control_entity_ids[1] != left_entity_ids[2] or right_entity_ids[2] != left_entity_ids[2]:
        raise ValueError(f"case {case_id} C entity differs between order arms")
    greedy_left = next((run for run in left_runs.get("runs", []) if run.get("mode") == "greedy"), None)
    greedy_right = next((run for run in right_runs.get("runs", []) if run.get("mode") == "greedy"), None)
    greedy_owner = {
        "a_then_b_then_c": None if greedy_left is None else _primary_owner(greedy_left),
        "b_then_a_then_c": None if greedy_right is None else _primary_owner(greedy_right),
    }
    a_id = str(case.get("a_entity_id") or inferred_a)
    if a_id != inferred_a:
        raise ValueError(f"case {case_id} explicit A entity differs from arm entity ids")
    a_recurrence = {
        arm_name: {
            "a_entity_id": a_id,
            "a_owner_run_count": sum(a_id in _owner_labels(run) for run in _as_mapping(arms[arm_name], "arm").get("runs", [])),
            "run_count": len(_as_mapping(arms[arm_name], "arm").get("runs", [])),
        }
        for arm_name in ARMS
    }
    control_a_runs = sum(a_id in _owner_labels(run) for run in control_runs.get("runs", []))
    control_total = len(control_runs.get("runs", []))
    return {
        "case_id": case_id,
        "image_id": case.get("image_id"),
        "arms": arm_summaries,
        "paired_sample_comparison": _paired_seed_comparison(left_runs, right_runs),
        "a_recurrence": a_recurrence,
        "greedy_owner_comparison": {
            **greedy_owner,
            "equal": greedy_owner["a_then_b_then_c"] == greedy_owner["b_then_a_then_c"] if None not in greedy_owner.values() else None,
        },
        "b_then_c_activation_of_a": {
            "a_entity_id": a_id,
            "a_owner_run_count": control_a_runs,
            "run_count": control_total,
            "a_owner_run_rate": control_a_runs / control_total if control_total else None,
        },
    }


def _merge_payloads(paths: Sequence[Path]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not paths:
        raise ValueError("at least one result JSON path is required")
    merged: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    case_meta: dict[str, tuple[str, str]] = {}
    case_records: dict[str, dict[str, Any]] = {}
    source_paths: list[str] = []
    for path in paths:
        resolved = path.expanduser().resolve(strict=True)
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        validate_artifact_payload(payload)
        source_paths.append(str(resolved))
        for raw_case in payload["cases"]:
            case = dict(_as_mapping(raw_case, "artifact case"))
            case_id = str(case["case_id"])
            signature = (str(case.get("image_id")), json.dumps(case.get("entity_ledger"), sort_keys=True, separators=(",", ":")))
            if case_id in case_meta and case_meta[case_id] != signature:
                raise ValueError(f"case {case_id} metadata differs across result files")
            case_meta[case_id] = signature
            case_records.setdefault(
                case_id,
                {
                    "case_id": case_id,
                    "image_id": case.get("image_id"),
                    "a_entity_id": case.get("a_entity_id"),
                    "b_entity_id": case.get("b_entity_id"),
                    "c_entity_id": case.get("c_entity_id"),
                    "arms": {},
                },
            )
            for arm_name in ARMS:
                arm = _as_mapping(case["arms"][arm_name], f"case {case_id} arm {arm_name}")
                target = case_records[case_id]["arms"].setdefault(arm_name, {key: value for key, value in arm.items() if key != "runs"} | {"runs": []})
                for raw_run in arm["runs"]:
                    run = dict(_as_mapping(raw_run, "artifact run"))
                    key = _run_key(case_id, arm_name, run)
                    if key in merged:
                        raise ValueError(f"duplicate run key {key} across result files; refusing silent overwrite")
                    merged[key] = run
                    target["runs"].append(run)
    for case_id, record in case_records.items():
        for arm_name in ARMS:
            record["arms"][arm_name]["runs"].sort(key=lambda run: (str(run.get("mode")), -1 if run.get("seed") is None else int(run.get("seed"))))
    metadata = {"source_result_paths": source_paths, "case_count": len(case_records), "run_count": len(merged)}
    return metadata, [case_records[key] for key in sorted(case_records)]


def summarize(paths: Sequence[Path]) -> dict[str, Any]:
    payloads = []
    for path in paths:
        resolved = path.expanduser().resolve(strict=True)
        payloads.append(json.loads(resolved.read_text(encoding="utf-8")))
    if payloads and all(payload.get("schema_version") == SCHEMA_VERSION_V2 for payload in payloads):
        cases: dict[str, dict[str, Any]] = {}
        seen_runs: set[tuple[str, str, str, str, str]] = set()
        source_paths: list[str] = []
        global_signature: dict[str, Any] | None = None
        case_signatures: dict[str, dict[str, Any]] = {}
        for path, payload in zip(paths, payloads, strict=True):
            validate_artifact_payload(payload)
            source_paths.append(str(path.expanduser().resolve()))
            config = dict(_as_mapping(payload.get("config", {}), "v2 config"))
            config.pop("case_ids", None)
            signature = {
                "config": config,
                "model_identity": payload.get("model_identity"),
            }
            if global_signature is None:
                global_signature = signature
            elif signature != global_signature:
                raise ValueError("v2 result files disagree on model identity or inference configuration")
            for raw_case in payload["cases"]:
                case = dict(_as_mapping(raw_case, "v2 artifact case"))
                case_id = str(case["case_id"])
                image = payload.get("image")
                image_signature = None
                if isinstance(image, Mapping):
                    image_signature = {key: image.get(key) for key in ("image_id", "image_sha256", "executed_media_sha256", "width", "height")}
                arm_signature = {
                    str(name): {
                        "prefix_token_ids_sha256": _as_mapping(arm, f"v2 arm {name}").get("prefix_token_ids_sha256"),
                        "row_token_ids_sha256": _as_mapping(arm, f"v2 arm {name}").get("row_token_ids_sha256"),
                    }
                    for name, arm in _as_mapping(case["arms"], "v2 artifact arms").items()
                }
                case_signature = {
                    "image": image_signature,
                    "entity_ledger": case.get("entity_ledger"),
                    "comparisons": case.get("comparisons"),
                    "arm_signature": arm_signature,
                }
                if case_id in case_signatures and case_signatures[case_id] != case_signature:
                    raise ValueError(f"case {case_id} differs in image hash, entity ledger, comparison, or arm prefix hashes")
                case_signatures[case_id] = case_signature
                target = cases.setdefault(case_id, {
                    "case_id": case_id,
                    "image_id": case.get("image_id"),
                    "comparisons": case.get("comparisons", []),
                    "invariants_by_comparison": case.get("invariants_by_comparison", {}),
                    "arms": {},
                })
                if target["image_id"] != case.get("image_id"):
                    raise ValueError(f"case {case_id} metadata differs across v2 result files")
                for raw_comparison in case.get("comparisons", []):
                    comparison = dict(raw_comparison)
                    comparison_id = str(comparison.get("comparison_id"))
                    known = {str(item.get("comparison_id")): item for item in target["comparisons"]}
                    if comparison_id not in known:
                        target["comparisons"].append(comparison)
                target["invariants_by_comparison"].update(case.get("invariants_by_comparison", {}))
                for raw_name, raw_arm in _as_mapping(case["arms"], "v2 artifact arms").items():
                    name = str(raw_name)
                    arm = _as_mapping(raw_arm, f"v2 arm {name}")
                    arm_target = target["arms"].setdefault(name, {key: value for key, value in arm.items() if key != "runs"} | {"runs": []})
                    for raw_run in arm.get("runs", []):
                        run = dict(_as_mapping(raw_run, "v2 run"))
                        key = (case_id, name, str(run.get("comparison_id")), str(run.get("mode")), "none" if run.get("seed") is None else str(run.get("seed")))
                        if key in seen_runs:
                            raise ValueError(f"duplicate v2 run key {key} across result files")
                        seen_runs.add(key)
                        arm_target["runs"].append(run)
        return {
            "schema_version": "same_covered_set_prefix_order.summary.v2",
            "source": {"source_result_paths": source_paths, "case_count": len(cases), "run_count": len(seen_runs)},
            "cases": [_v2_case_summary(case_id, cases[case_id]) for case_id in sorted(cases)],
        }
    metadata, cases = _merge_payloads(paths)
    return {
        "schema_version": "same_covered_set_prefix_order.summary.v1",
        "source": metadata,
        "cases": [_case_summary(str(case["case_id"]), case) for case in cases],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path, help="one or more result.json artifacts")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists() and not args.force:
        parser.error(f"refusing to overwrite {output}; pass --force")
    result = summarize(args.results)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
