#!/usr/bin/env python3
"""Reduce continuation-locality and exact-prefix owner-composition receipts."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research.run_continuation_locality_boundary_scoring import (  # noqa: E402
    RECEIPT_SCHEMA_VERSION as LOCALITY_SCHEMA,
)
from scripts.research.run_exact_prefix_owner_compositionality import (  # noqa: E402
    RECEIPT_SCHEMA_VERSION as OWNER_SCHEMA,
)
from src.config.fingerprint import sha256_file  # noqa: E402


SCHEMA_VERSION = "continuation_locality_owner_compositionality.summary.v1"
ROLE_TO_TRAINED_ARM = {
    "source": "source",
    "transition-step36": "transition_step36",
    "pairwise-lr3e6-step90": "pairwise_lr3e6_step90",
    "owner-conditioned-lr1e5-step90": "owner_conditioned_lr1e5_step90",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _describe(values: Sequence[float]) -> dict[str, Any]:
    numbers = [float(value) for value in values]
    if not numbers:
        return {"count": 0}
    ordered = sorted(numbers)
    trim = int(len(ordered) * 0.1)
    trimmed = ordered[trim : len(ordered) - trim] if 2 * trim < len(ordered) else ordered
    return {
        "count": len(numbers),
        "mean": statistics.fmean(numbers),
        "median": statistics.median(numbers),
        "trimmed_mean_10_percent_each_tail": statistics.fmean(trimmed),
        "minimum": ordered[0],
        "q10": _quantile(ordered, 0.10),
        "q25": _quantile(ordered, 0.25),
        "q75": _quantile(ordered, 0.75),
        "q90": _quantile(ordered, 0.90),
        "maximum": ordered[-1],
        "positive_count": sum(value > 0.0 for value in numbers),
        "negative_count": sum(value < 0.0 for value in numbers),
        "zero_count": sum(value == 0.0 for value in numbers),
        "positive_fraction": sum(value > 0.0 for value in numbers) / len(numbers),
    }


def _bootstrap_mean_interval(
    values: Sequence[float], *, seed: int, replicates: int = 5000
) -> dict[str, Any]:
    numbers = [float(value) for value in values]
    if not numbers:
        return {"count": 0, "replicates": 0}
    generator = random.Random(seed)
    means = [
        statistics.fmean(generator.choice(numbers) for _ in range(len(numbers)))
        for _ in range(replicates)
    ]
    return {
        "count": len(numbers),
        "replicates": replicates,
        "seed": seed,
        "mean": statistics.fmean(numbers),
        "percentile_95_interval": [
            _quantile(means, 0.025),
            _quantile(means, 0.975),
        ],
    }


def _receipt_files(root: Path) -> list[Path]:
    files = sorted(root.glob("*/shard-*.json"))
    if not files:
        raise ValueError(f"no shard receipts under {root}")
    return files


def _load_locality(root: Path) -> tuple[dict[str, dict[str, dict[str, Any]]], list[dict[str, Any]]]:
    roles: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    provenance: list[dict[str, Any]] = []
    manifest_hashes: set[str] = set()
    for path in _receipt_files(root):
        receipt = _read_json(path)
        if receipt.get("schema_version") != LOCALITY_SCHEMA:
            raise ValueError(f"unexpected locality schema: {path}")
        role = str(receipt["checkpoint_role"])
        manifest_hashes.add(str(receipt["manifest"]["sha256"]))
        for record in receipt.get("records", []):
            boundary_id = str(record["boundary_id"])
            if boundary_id in roles[role]:
                raise ValueError(f"duplicate locality record {role}:{boundary_id}")
            roles[role][boundary_id] = dict(record)
        provenance.append({"path": str(path), "sha256": sha256_file(path)})
    if len(manifest_hashes) != 1:
        raise ValueError("locality receipts do not share one manifest")
    if set(roles) != set(ROLE_TO_TRAINED_ARM):
        raise ValueError(f"unexpected locality roles: {sorted(roles)}")
    source_ids = set(roles["source"])
    for role, records in roles.items():
        if set(records) != source_ids:
            raise ValueError(f"locality coverage mismatch for {role}")
    return dict(roles), provenance


def _new_margin(record: Mapping[str, Any]) -> float:
    return float(record["terminal_boundary"]["row_entry_minus_terminal"])


def _trained_events(reduction: Mapping[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    results: dict[str, dict[str, dict[str, Any]]] = {}
    for role, arm_name in ROLE_TO_TRAINED_ARM.items():
        events = reduction["arms"][arm_name]["events"]
        indexed = {str(item["event_id"]): dict(item) for item in events}
        if len(indexed) != 1440:
            raise ValueError(f"expected 1,440 trained events for {role}")
        results[role] = indexed
    return results


def _trained_margin(record: Mapping[str, Any]) -> float:
    return float(record["stop_boundary"]["row_entry_minus_stop_log_probability"])


def _locality_summary(
    *,
    roles: Mapping[str, Mapping[str, Mapping[str, Any]]],
    trained: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    source_new = roles["source"]
    source_trained = trained["source"]
    cohorts = sorted({str(item["cohort"]) for item in source_new.values()})
    result: dict[str, Any] = {"checkpoint_roles": {}}
    for role in ROLE_TO_TRAINED_ARM:
        new_deltas = {
            boundary_id: _new_margin(record) - _new_margin(source_new[boundary_id])
            for boundary_id, record in roles[role].items()
        }
        trained_deltas = {
            event_id: _trained_margin(record) - _trained_margin(source_trained[event_id])
            for event_id, record in trained[role].items()
        }
        cohort_values = {}
        for cohort in cohorts:
            values = [
                new_deltas[boundary_id]
                for boundary_id, record in source_new.items()
                if record["cohort"] == cohort
            ]
            cohort_values[cohort] = {
                "checkpoint_minus_source": _describe(values),
                "bootstrap_mean": _bootstrap_mean_interval(
                    values, seed=250725 + sum(ord(char) for char in role + cohort)
                ),
            }
        matched = {}
        for cohort in ("same_image_near_continue", "untouched_matched_continue"):
            contrasts = []
            for boundary_id, record in source_new.items():
                if record["cohort"] != cohort:
                    continue
                event_id = str(record["matched_training_event_id"])
                contrasts.append(trained_deltas[event_id] - new_deltas[boundary_id])
            matched[cohort] = {
                "trained_exact_delta_minus_control_delta": _describe(contrasts),
                "bootstrap_mean": _bootstrap_mean_interval(
                    contrasts, seed=250726 + sum(ord(char) for char in role + cohort)
                ),
            }
        result["checkpoint_roles"][role] = {
            "trained_exact_checkpoint_minus_source": _describe(list(trained_deltas.values())),
            "trained_exact_bootstrap_mean": _bootstrap_mean_interval(
                list(trained_deltas.values()), seed=250727 + sum(ord(char) for char in role)
            ),
            "control_cohorts": cohort_values,
            "matched_locality_contrasts": matched,
        }
    result["estimand"] = (
        "checkpoint minus Source change in row-opener-minus-terminal log probability; "
        "matched locality is trained-state change minus matched control-state change"
    )
    return result


def _load_owner(root: Path) -> tuple[dict[str, dict[str, dict[str, Any]]], list[dict[str, Any]]]:
    roles: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    provenance: list[dict[str, Any]] = []
    manifest_hashes: set[str] = set()
    for path in _receipt_files(root):
        receipt = _read_json(path)
        if receipt.get("schema_version") != OWNER_SCHEMA:
            raise ValueError(f"unexpected owner schema: {path}")
        role = str(receipt["checkpoint_role"])
        manifest_hashes.add(str(receipt["manifest"]["sha256"]))
        for case in receipt.get("cases", []):
            case_id = str(case["case_id"])
            if case_id in roles[role]:
                raise ValueError(f"duplicate owner case {role}:{case_id}")
            roles[role][case_id] = dict(case)
        provenance.append({"path": str(path), "sha256": sha256_file(path)})
    if len(manifest_hashes) != 1:
        raise ValueError("owner receipts do not share one manifest")
    if set(roles) != {"source", "transition-step36"}:
        raise ValueError(f"unexpected owner roles: {sorted(roles)}")
    if set(roles["source"]) != set(roles["transition-step36"]):
        raise ValueError("owner case coverage differs between checkpoints")
    if len(roles["source"]) != 400:
        raise ValueError(f"expected 400 owner cases, found {len(roles['source'])}")
    return dict(roles), provenance


def _realized(case: Mapping[str, Any], release: str) -> bool:
    return case["releases"][release].get("intended_owner_realized") is True


def _release_outcome(row: Mapping[str, Any]) -> str:
    if row.get("intended_owner_realized") is True:
        return "intended_owner"
    if row.get("covered_prefix_owner_ids"):
        return "covered_owner_repeat"
    if row.get("uncovered_ledger_owner_ids"):
        return "other_uncovered_owner"
    if row.get("status") != "success":
        return "invalid_or_incomplete_row"
    if row.get("parsed_predictions"):
        return "unmatched_or_ambiguous_valid_row"
    return "no_parsed_row"


def _paired_recovery_transitions(
    cases: Mapping[str, Mapping[str, Any]], *, before: str, after: str
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for case in cases.values():
        before_value = _realized(case, before)
        after_value = _realized(case, after)
        counts[f"{str(before_value).lower()}_to_{str(after_value).lower()}"] += 1
    return {
        "before": before,
        "after": after,
        "case_count": len(cases),
        "false_to_true": counts["false_to_true"],
        "true_to_false": counts["true_to_false"],
        "false_to_false": counts["false_to_false"],
        "true_to_true": counts["true_to_true"],
        "net_recovery_change": counts["false_to_true"] - counts["true_to_false"],
    }


def _binary_transition_counts(
    before_values: Sequence[float], after_values: Sequence[float]
) -> dict[str, int]:
    if len(before_values) != len(after_values):
        raise ValueError("paired binary transition inputs differ in length")
    counts: Counter[str] = Counter(
        f"{bool(before_value)}_to_{bool(after_value)}"
        for before_value, after_value in zip(before_values, after_values, strict=True)
    )
    return {
        "false_to_false": counts["False_to_False"],
        "false_to_true": counts["False_to_True"],
        "true_to_false": counts["True_to_False"],
        "true_to_true": counts["True_to_True"],
    }


def _owner_recovery_strata(
    cases: Mapping[str, Mapping[str, Any]], *, field: str
) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for case in cases.values():
        groups[str(case[field])].append(case)
    return {
        key: {
            "case_count": len(values),
            "unique_image_count": len({str(value["image_id"]) for value in values}),
            "continuation_margin": _describe(
                [
                    float(value["state_score"]["terminal_boundary"]["row_entry_minus_terminal"])
                    for value in values
                ]
            ),
            "intended_owner_recovery": {
                release: _describe([float(_realized(value, release)) for value in values])
                for release in ("native", "forced_opener", "forced_description")
            },
        }
        for key, values in sorted(groups.items())
    }


def _per_image_owner_recovery(
    cases: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for case in cases.values():
        groups[str(case["image_id"])].append(case)
    return {
        "unique_image_count": len(groups),
        "mean_case_recovery_within_image": {
            release: _describe(
                [
                    sum(float(_realized(case, release)) for case in values) / len(values)
                    for values in groups.values()
                ]
            )
            for release in ("native", "forced_opener", "forced_description")
        },
    }


def _owner_role_summary(cases: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    release_names = ("native", "forced_opener", "forced_description")
    releases = {}
    for name in release_names:
        outcomes = Counter(_release_outcome(case["releases"][name]) for case in cases.values())
        realized = [float(_realized(case, name)) for case in cases.values()]
        multi = [
            float(_realized(case, name))
            for case in cases.values()
            if int(case["target"]["same_category_annotation_count"]) > 1
        ]
        single = [
            float(_realized(case, name))
            for case in cases.values()
            if int(case["target"]["same_category_annotation_count"]) == 1
        ]
        releases[name] = {
            "intended_owner_recovery": _describe(realized),
            "same_category_multi_instance_recovery": _describe(multi),
            "single_category_instance_recovery": _describe(single),
            "outcome_counts": dict(sorted(outcomes.items())),
        }
    margins = [
        float(case["state_score"]["terminal_boundary"]["row_entry_minus_terminal"])
        for case in cases.values()
    ]
    full_row = [
        float(case["state_score"]["candidate_score"]["full_row"]["sum"])
        for case in cases.values()
    ]
    description = [
        float(case["state_score"]["candidate_score"]["description"]["mean"])
        for case in cases.values()
    ]
    geometry = [
        float(case["state_score"]["candidate_score"]["geometry"]["mean"])
        for case in cases.values()
    ]
    terminal_cases = {
        case_id: case
        for case_id, case in cases.items()
        if case["releases"]["native"].get("row_stop", {}).get("stop_reason") == "terminal"
    }
    return {
        "case_count": len(cases),
        "continuation_margin": _describe(margins),
        "verified_target_complete_row_log_probability_sum": _describe(full_row),
        "verified_target_description_mean_log_probability": _describe(description),
        "verified_target_geometry_mean_log_probability": _describe(geometry),
        "releases": releases,
        "paired_release_recovery": {
            "forced_opener_after_native": _paired_recovery_transitions(
                cases, before="native", after="forced_opener"
            ),
            "forced_description_after_forced_opener": _paired_recovery_transitions(
                cases, before="forced_opener", after="forced_description"
            ),
            "forced_description_after_native": _paired_recovery_transitions(
                cases, before="native", after="forced_description"
            ),
        },
        "native_terminal_cases": {
            "count": len(terminal_cases),
            "case_ids": sorted(terminal_cases),
            "forced_opener_intended_owner_recovery_count": sum(
                _realized(case, "forced_opener") for case in terminal_cases.values()
            ),
        },
        "per_image_macro_recovery": _per_image_owner_recovery(cases),
        "recovery_by_prefix_depth": _owner_recovery_strata(cases, field="prefix_depth"),
        "recovery_by_object_count_band": _owner_recovery_strata(
            cases, field="object_count_band"
        ),
    }


def _composition_summary(
    cases: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    baseline: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for case in cases.values():
        key = (
            str(case["image_id"]),
            str(case["state_score"]["prefix_token_ids_sha256"]),
            str(case["target"]["owner_id"]),
        )
        baseline[key] = case
    oracle_score_changes: list[float] = []
    oracle_margin_changes: list[float] = []
    oracle_opener_changes: list[float] = []
    oracle_description_changes: list[float] = []
    self_score_changes: list[float] = []
    self_margin_changes: list[float] = []
    self_opener_changes: list[float] = []
    self_description_changes: list[float] = []
    oracle_base_opener: list[float] = []
    oracle_after_opener: list[float] = []
    oracle_base_description: list[float] = []
    oracle_after_description: list[float] = []
    self_base_opener: list[float] = []
    self_after_opener: list[float] = []
    self_base_description: list[float] = []
    self_after_description: list[float] = []
    record_count = 0
    self_count = 0
    for case in cases.values():
        original_prefix_hash = str(case["state_score"]["prefix_token_ids_sha256"])
        for record in case.get("composition", []):
            record_count += 1
            key = (str(case["image_id"]), original_prefix_hash, str(record["secondary_owner_id"]))
            base = baseline.get(key)
            if base is None:
                raise ValueError(f"composition baseline is absent for {key}")
            base_score = float(base["state_score"]["candidate_score"]["full_row"]["sum"])
            base_margin = float(base["state_score"]["terminal_boundary"]["row_entry_minus_terminal"])
            base_opener = float(_realized(base, "forced_opener"))
            base_description = float(_realized(base, "forced_description"))
            oracle = record["oracle_sampled_target_row"]
            oracle_base_opener.append(base_opener)
            oracle_after_opener.append(
                float(oracle["forced_opener"].get("intended_owner_realized") is True)
            )
            oracle_base_description.append(base_description)
            oracle_after_description.append(
                float(oracle["forced_description"].get("intended_owner_realized") is True)
            )
            oracle_score_changes.append(
                float(oracle["score"]["candidate_score"]["full_row"]["sum"]) - base_score
            )
            oracle_margin_changes.append(
                float(oracle["score"]["terminal_boundary"]["row_entry_minus_terminal"])
                - base_margin
            )
            oracle_opener_changes.append(
                float(oracle["forced_opener"].get("intended_owner_realized") is True)
                - base_opener
            )
            oracle_description_changes.append(
                float(oracle["forced_description"].get("intended_owner_realized") is True)
                - base_description
            )
            self_probe = record.get("self_generated_target_row")
            if self_probe is not None:
                self_count += 1
                self_base_opener.append(base_opener)
                self_after_opener.append(
                    float(self_probe["forced_opener"].get("intended_owner_realized") is True)
                )
                self_base_description.append(base_description)
                self_after_description.append(
                    float(
                        self_probe["forced_description"].get("intended_owner_realized")
                        is True
                    )
                )
                self_score_changes.append(
                    float(self_probe["score"]["candidate_score"]["full_row"]["sum"])
                    - base_score
                )
                self_margin_changes.append(
                    float(self_probe["score"]["terminal_boundary"]["row_entry_minus_terminal"])
                    - base_margin
                )
                self_opener_changes.append(
                    float(self_probe["forced_opener"].get("intended_owner_realized") is True)
                    - base_opener
                )
                self_description_changes.append(
                    float(self_probe["forced_description"].get("intended_owner_realized") is True)
                    - base_description
                )
    return {
        "ordered_owner_pair_count": record_count,
        "model_realized_after_forced_description_first_owner_pair_count": self_count,
        "oracle_sampled_first_owner": {
            "secondary_complete_row_log_probability_change": _describe(oracle_score_changes),
            "secondary_continuation_margin_change": _describe(oracle_margin_changes),
            "secondary_forced_opener_recovery_change": _describe(oracle_opener_changes),
            "secondary_forced_description_recovery_change": _describe(
                oracle_description_changes
            ),
            "secondary_forced_opener_recovery_before_after": {
                "before": _describe(oracle_base_opener),
                "after": _describe(oracle_after_opener),
                "paired_transitions": _binary_transition_counts(
                    oracle_base_opener, oracle_after_opener
                ),
            },
            "secondary_forced_description_recovery_before_after": {
                "before": _describe(oracle_base_description),
                "after": _describe(oracle_after_description),
                "paired_transitions": _binary_transition_counts(
                    oracle_base_description, oracle_after_description
                ),
            },
        },
        "model_realized_after_forced_description_first_owner": {
            "secondary_complete_row_log_probability_change": _describe(self_score_changes),
            "secondary_continuation_margin_change": _describe(self_margin_changes),
            "secondary_forced_opener_recovery_change": _describe(self_opener_changes),
            "secondary_forced_description_recovery_change": _describe(
                self_description_changes
            ),
            "secondary_forced_opener_recovery_before_after": {
                "before": _describe(self_base_opener),
                "after": _describe(self_after_opener),
                "paired_transitions": _binary_transition_counts(
                    self_base_opener, self_after_opener
                ),
            },
            "secondary_forced_description_recovery_before_after": {
                "before": _describe(self_base_description),
                "after": _describe(self_after_description),
                "paired_transitions": _binary_transition_counts(
                    self_base_description, self_after_description
                ),
            },
        },
    }


def _paired_owner_summary(
    source: Mapping[str, Mapping[str, Any]],
    transition: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    margins = []
    rows = []
    description_scores = []
    geometry_scores = []
    release_changes: dict[str, list[float]] = defaultdict(list)
    for case_id, source_case in source.items():
        transition_case = transition[case_id]
        margins.append(
            float(transition_case["state_score"]["terminal_boundary"]["row_entry_minus_terminal"])
            - float(source_case["state_score"]["terminal_boundary"]["row_entry_minus_terminal"])
        )
        rows.append(
            float(transition_case["state_score"]["candidate_score"]["full_row"]["sum"])
            - float(source_case["state_score"]["candidate_score"]["full_row"]["sum"])
        )
        description_scores.append(
            float(transition_case["state_score"]["candidate_score"]["description"]["mean"])
            - float(source_case["state_score"]["candidate_score"]["description"]["mean"])
        )
        geometry_scores.append(
            float(transition_case["state_score"]["candidate_score"]["geometry"]["mean"])
            - float(source_case["state_score"]["candidate_score"]["geometry"]["mean"])
        )
        for name in ("native", "forced_opener", "forced_description"):
            release_changes[name].append(
                float(_realized(transition_case, name)) - float(_realized(source_case, name))
            )
    return {
        "transition_minus_source_continuation_margin": _describe(margins),
        "transition_minus_source_complete_row_log_probability_sum": _describe(rows),
        "transition_minus_source_description_mean_log_probability": _describe(
            description_scores
        ),
        "transition_minus_source_geometry_mean_log_probability": _describe(geometry_scores),
        "transition_minus_source_intended_owner_recovery": {
            name: _describe(values) for name, values in release_changes.items()
        },
    }


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    trained_path = args.trained_reduction.expanduser().resolve(strict=True)
    locality_root = args.locality_root.expanduser().resolve(strict=True)
    owner_root = args.owner_root.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite output root: {output_root}")
    trained_reduction = _read_json(trained_path)
    locality_roles, locality_receipts = _load_locality(locality_root)
    trained = _trained_events(trained_reduction)
    owner_roles, owner_receipts = _load_owner(owner_root)
    result = {
        "schema_version": SCHEMA_VERSION,
        "unit_id": "2026-07-25-continuation-locality-and-exact-prefix-owner-compositionality",
        "inputs": {
            "trained_exact_reduction": {
                "path": str(trained_path),
                "sha256": sha256_file(trained_path),
            },
            "locality_receipts": locality_receipts,
            "owner_receipts": owner_receipts,
        },
        "continuation_locality": _locality_summary(roles=locality_roles, trained=trained),
        "owner_compositionality": {
            "checkpoint_roles": {
                role: {
                    **_owner_role_summary(cases),
                    "composition": _composition_summary(cases),
                }
                for role, cases in owner_roles.items()
            },
            "paired_transition_minus_source": _paired_owner_summary(
                owner_roles["source"], owner_roles["transition-step36"]
            ),
        },
        "claim_boundary": (
            "fixed-prefix evidence identifies checkpoint- and intervention-conditional behavior; "
            "it is not a free-rollout final owner-set result or proof of an explicit owner ledger"
        ),
    }
    output_root.mkdir(parents=True)
    (output_root / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trained-reduction", type=Path, required=True)
    parser.add_argument("--locality-root", type=Path, required=True)
    parser.add_argument("--owner-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main() -> None:
    result = summarize(build_parser().parse_args())
    print(
        json.dumps(
            {
                "locality_roles": sorted(
                    result["continuation_locality"]["checkpoint_roles"]
                ),
                "owner_roles": sorted(
                    result["owner_compositionality"]["checkpoint_roles"]
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
