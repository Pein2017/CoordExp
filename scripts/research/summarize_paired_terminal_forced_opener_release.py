#!/usr/bin/env python3
"""Summarize paired native and opener-forced releases at 200 Source stops."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
import json
import math
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research.summarize_continuation_locality_owner_compositionality import (  # noqa: E402
    _bootstrap_mean_interval,
    _describe,
)
from src.config.fingerprint import sha256_file  # noqa: E402


SCHEMA_VERSION = "paired_terminal_forced_opener_release.summary.v1"
CASE_SCHEMA_VERSION = "paired_terminal_forced_opener_release.case.v1"
RECEIPT_SCHEMA_VERSION = "paired_terminal_forced_opener_release.receipt.v1"
UNIT_ID = "2026-07-25-paired-natural-terminal-forced-opener-release"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


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


def _diagnostic_positive_band(margins: Mapping[str, Any]) -> str:
    if float(margins["source"]) > 0.0:
        return "source_positive"
    if float(margins["transition-step36"]) > 0.0:
        return "transition_positive_beyond_source"
    if float(margins["pairwise-lr3e6-step90"]) > 0.0:
        return "pairwise_positive_beyond_transition"
    if float(margins["owner-conditioned-lr1e5-step90"]) > 0.0:
        return "owner_conditioned_positive_beyond_pairwise"
    return "all_checkpoints_nonpositive"


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


def _ledger_ids(values: Any) -> set[str]:
    if not isinstance(values, list):
        raise ValueError("owner IDs must be a list")
    return {str(value).split(":", 1)[-1] for value in values}


def _release_owner_sets(
    release: Mapping[str, Any], *, remaining: set[str], covered: set[str]
) -> dict[str, Any]:
    matched = _ledger_ids(release.get("strict_matched_owner_ids", []))
    parse = release.get("parse_evidence")
    parse_mapping = parse if isinstance(parse, Mapping) else {}
    row_stop = release.get("row_stop")
    stop_mapping = row_stop if isinstance(row_stop, Mapping) else {}
    valid = bool(parse_mapping.get("metric_bearing")) and int(
        parse_mapping.get("valid_prediction_count", 0)
    ) > 0
    unmatched_indices = release.get("unmatched_or_ambiguous_prediction_indices", [])
    if not isinstance(unmatched_indices, list):
        raise ValueError("unmatched_or_ambiguous_prediction_indices must be a list")
    immediate_terminal = stop_mapping.get("stop_reason") in {
        "terminal",
        "assistant_terminal",
        "end_of_turn",
    }
    uncovered = matched & remaining
    covered_repeat = matched & covered
    other_matched = matched - remaining - covered
    if uncovered:
        outcome = "verified_uncovered_owner"
    elif covered_repeat:
        outcome = "covered_owner_repeat"
    elif other_matched:
        outcome = "other_strict_matched_owner"
    elif valid and unmatched_indices:
        outcome = "valid_unmatched_or_ambiguous"
    elif immediate_terminal:
        outcome = "immediate_terminal"
    elif valid:
        outcome = "valid_without_strict_owner_match"
    else:
        outcome = "invalid_or_incomplete"
    return {
        "strict_matched_owner_ids": sorted(matched),
        "verified_uncovered_owner_ids": sorted(uncovered),
        "covered_repeat_owner_ids": sorted(covered_repeat),
        "other_strict_matched_owner_ids": sorted(other_matched),
        "valid_row": valid,
        "unmatched_or_ambiguous_prediction_count": len(unmatched_indices),
        "immediate_terminal": immediate_terminal,
        "outcome": outcome,
    }


def _strict_match_evidence(
    release: Mapping[str, Any], owner_ids: set[str]
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    values = release.get("entity_matches", [])
    if not isinstance(values, list):
        raise ValueError("entity_matches must be a list")
    for value in values:
        if not isinstance(value, Mapping):
            raise ValueError("entity match must be an object")
        matched_owner = value.get("matched_entity_id")
        if matched_owner is None or str(matched_owner) not in owner_ids:
            continue
        candidates = value.get("candidates", [])
        if not isinstance(candidates, list):
            raise ValueError("entity-match candidates must be a list")
        candidate = next(
            (
                item
                for item in candidates
                if isinstance(item, Mapping)
                and str(item.get("entity_id")) == str(matched_owner)
            ),
            None,
        )
        if candidate is None:
            raise ValueError(f"strict match {matched_owner} lacks candidate evidence")
        evidence.append(
            {
                "owner_id": str(matched_owner),
                "description": str(value.get("description", "")),
                "iou": float(candidate["iou"]),
                "center_distance_norm": float(candidate["center_distance_norm"]),
                "predicted_bbox": value.get("predicted_bbox"),
                "predicted_bbox_norm1000": value.get("predicted_bbox_norm1000"),
            }
        )
    if {item["owner_id"] for item in evidence} != owner_ids:
        raise ValueError("strict match evidence does not cover requested owner IDs")
    return evidence


def _classify_case(
    raw: Mapping[str, Any], *, statistics: Mapping[str, Any]
) -> dict[str, Any]:
    remaining = _ledger_ids(raw.get("remaining_owner_ids"))
    covered = _ledger_ids(raw.get("covered_owner_ids"))
    if remaining & covered:
        raise ValueError(f"{raw.get('boundary_id')} remaining and covered sets overlap")
    releases = raw.get("releases")
    if not isinstance(releases, Mapping):
        raise ValueError(f"{raw.get('boundary_id')} lacks releases")
    native_raw = releases.get("native")
    forced_raw = releases.get("forced_opener")
    if not isinstance(native_raw, Mapping) or not isinstance(forced_raw, Mapping):
        raise ValueError(f"{raw.get('boundary_id')} lacks paired release arms")
    native = _release_owner_sets(native_raw, remaining=remaining, covered=covered)
    forced = _release_owner_sets(forced_raw, remaining=remaining, covered=covered)
    native_uncovered = set(native["verified_uncovered_owner_ids"])
    forced_uncovered = set(forced["verified_uncovered_owner_ids"])
    gained = forced_uncovered - native_uncovered
    lost = native_uncovered - forced_uncovered
    retained = native_uncovered & forced_uncovered
    diagnostic_margins = {
        str(role): float(value)
        for role, value in statistics["checkpoint_margins"].items()
    }
    return {
        "schema_version": CASE_SCHEMA_VERSION,
        "boundary_id": str(raw["boundary_id"]),
        "image_id": str(raw["image_id"]),
        "prefix_depth": int(raw["prefix_depth"]),
        "object_count_band": str(raw["object_count_band"]),
        "annotation_object_count": int(raw["annotation_object_count"]),
        "remaining_owner_count": len(remaining),
        "remaining_ledger_owner_ids": sorted(remaining),
        "covered_ledger_owner_ids": sorted(covered),
        "diagnostic_margins": diagnostic_margins,
        "source_pairwise_margin": diagnostic_margins["source"],
        "native": native,
        "forced_opener": forced,
        "causal_gained_owner_ids": sorted(gained),
        "causal_gain_match_evidence": _strict_match_evidence(forced_raw, gained),
        "causal_lost_owner_ids": sorted(lost),
        "retained_uncovered_owner_ids": sorted(retained),
        "causal_success": bool(gained),
        "owner_net_change": len(gained) - len(lost),
        "raw_row_token_ids_equal": native_raw.get("raw_generated_token_ids")
        == forced_raw.get("raw_generated_token_ids"),
    }


def _fraction(successes: int, count: int) -> dict[str, Any]:
    return {
        "success_count": successes,
        "count": count,
        "fraction": None if count == 0 else successes / count,
        "wilson_95_interval": _wilson_interval(successes, count),
    }


def _aggregate(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(cases)
    causal_successes = sum(bool(case["causal_success"]) for case in cases)
    native_any = sum(bool(case["native"]["verified_uncovered_owner_ids"]) for case in cases)
    forced_any = sum(
        bool(case["forced_opener"]["verified_uncovered_owner_ids"]) for case in cases
    )
    gained_count = sum(len(case["causal_gained_owner_ids"]) for case in cases)
    lost_count = sum(len(case["causal_lost_owner_ids"]) for case in cases)
    retained_count = sum(len(case["retained_uncovered_owner_ids"]) for case in cases)
    paired_any = Counter()
    for case in cases:
        native_positive = bool(case["native"]["verified_uncovered_owner_ids"])
        forced_positive = bool(case["forced_opener"]["verified_uncovered_owner_ids"])
        paired_any[f"native_{native_positive}_forced_{forced_positive}"] += 1
    net_changes = [float(case["owner_net_change"]) for case in cases]
    gain_evidence = [
        evidence
        for case in cases
        for evidence in case.get("causal_gain_match_evidence", [])
    ]
    gain_ious = [float(value["iou"]) for value in gain_evidence]
    gain_distances = [float(value["center_distance_norm"]) for value in gain_evidence]
    return {
        "boundary_count": count,
        "primary_causal_success": _fraction(causal_successes, count),
        "native_any_verified_uncovered_owner": _fraction(native_any, count),
        "forced_any_verified_uncovered_owner": _fraction(forced_any, count),
        "causal_gained_owner_count": gained_count,
        "causal_lost_owner_count": lost_count,
        "retained_uncovered_owner_count": retained_count,
        "net_owner_change": gained_count - lost_count,
        "per_boundary_owner_net_change": _describe(net_changes),
        "per_boundary_owner_net_change_bootstrap_mean": _bootstrap_mean_interval(
            net_changes, seed=250725
        ),
        "paired_any_uncovered_owner_transitions": dict(sorted(paired_any.items())),
        "native_outcomes": dict(
            sorted(Counter(str(case["native"]["outcome"]) for case in cases).items())
        ),
        "forced_opener_outcomes": dict(
            sorted(
                Counter(str(case["forced_opener"]["outcome"]) for case in cases).items()
            )
        ),
        "raw_row_token_ids_equal": _fraction(
            sum(bool(case["raw_row_token_ids_equal"]) for case in cases), count
        ),
        "causal_gain_match_quality": {
            "iou": _describe(gain_ious),
            "center_distance_norm": _describe(gain_distances),
            "iou_at_least_0_50": _fraction(
                sum(value >= 0.50 for value in gain_ious), len(gain_ious)
            ),
            "iou_at_least_0_55": _fraction(
                sum(value >= 0.55 for value in gain_ious), len(gain_ious)
            ),
            "iou_at_least_0_60": _fraction(
                sum(value >= 0.60 for value in gain_ious), len(gain_ious)
            ),
        },
    }


def _strata(
    cases: Sequence[Mapping[str, Any]], *, key: Callable[[Mapping[str, Any]], str]
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[key(case)].append(case)
    return {name: _aggregate(values) for name, values in sorted(grouped.items())}


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.manifest.expanduser().resolve(strict=True)
    manifest = _read_json(manifest_path)
    manifest_hash = sha256_file(manifest_path)
    statistics_path = args.statistics_cases.expanduser().resolve(strict=True)
    statistics_rows = _read_jsonl(statistics_path)
    statistics_by_id = {str(row["boundary_id"]): row for row in statistics_rows}
    if len(statistics_by_id) != 200:
        raise ValueError("statistics ledger must contain 200 unique boundaries")

    receipt_paths = sorted(args.receipts_root.expanduser().resolve(strict=True).glob("shard-*.json"))
    if not receipt_paths:
        raise ValueError("no production shard receipts found")
    runtime_contract: dict[str, Any] | None = None
    raw_cases: list[dict[str, Any]] = []
    shard_indices: set[int] = set()
    receipt_provenance: list[dict[str, str]] = []
    contract_keys = (
        "shard_count",
        "physical_batch_size",
        "runtime_dtype_mode",
        "model_dtype",
        "max_new_tokens",
        "malformed_limit",
        "repetition_penalty",
        "temperature",
        "top_p",
        "authored_config_sha256",
        "effective_config_sha256",
        "source_jsonl_sha256",
        "attention_implementation",
        "forced_opener_token_id",
    )
    for path in receipt_paths:
        receipt = _read_json(path)
        if receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION:
            raise ValueError(f"unexpected receipt schema: {path}")
        if receipt.get("unit_id") != UNIT_ID:
            raise ValueError(f"unexpected receipt unit: {path}")
        if receipt.get("manifest", {}).get("sha256") != manifest_hash:
            raise ValueError(f"manifest hash mismatch: {path}")
        runtime = receipt.get("runtime")
        if not isinstance(runtime, Mapping):
            raise ValueError(f"receipt lacks runtime: {path}")
        current_contract = {key: runtime.get(key) for key in contract_keys}
        if runtime_contract is None:
            runtime_contract = current_contract
        elif current_contract != runtime_contract:
            raise ValueError(f"runtime contract mismatch: {path}")
        shard_index = int(runtime["shard_index"])
        if shard_index in shard_indices:
            raise ValueError(f"duplicate shard index {shard_index}")
        shard_indices.add(shard_index)
        values = receipt.get("cases")
        if not isinstance(values, list):
            raise ValueError(f"receipt cases must be a list: {path}")
        raw_cases.extend(dict(value) for value in values)
        receipt_provenance.append({"path": str(path), "sha256": sha256_file(path)})

    if runtime_contract is None:
        raise AssertionError("unreachable empty runtime contract")
    expected_shards = set(range(int(runtime_contract["shard_count"])))
    if shard_indices != expected_shards:
        raise ValueError(f"shard coverage mismatch: {sorted(shard_indices)}")
    raw_by_id = {str(case["boundary_id"]): case for case in raw_cases}
    if len(raw_cases) != 200 or len(raw_by_id) != 200:
        raise ValueError("production receipts must contain 200 unique boundaries")
    if set(raw_by_id) != set(statistics_by_id):
        raise ValueError("production and statistics boundary coverage differ")

    cases = [
        _classify_case(raw_by_id[boundary_id], statistics=statistics_by_id[boundary_id])
        for boundary_id in sorted(raw_by_id)
    ]
    unmatched_matches: list[Mapping[str, Any]] = []
    invalid_cases: list[dict[str, Any]] = []
    for case in cases:
        raw_case = raw_by_id[str(case["boundary_id"])]
        forced_raw = raw_case["releases"]["forced_opener"]
        if case["forced_opener"]["outcome"] == "valid_unmatched_or_ambiguous":
            values = forced_raw.get("entity_matches", [])
            if not isinstance(values, list) or any(
                not isinstance(value, Mapping) for value in values
            ):
                raise ValueError("forced unmatched entity evidence is malformed")
            unmatched_matches.extend(values)
        if not case["forced_opener"]["valid_row"]:
            parse = forced_raw.get("parse_evidence", {})
            dropped = parse.get("dropped_predictions", []) if isinstance(parse, Mapping) else []
            invalid_cases.append(
                {
                    "boundary_id": case["boundary_id"],
                    "image_id": case["image_id"],
                    "raw_generated_text": forced_raw.get("raw_generated_text"),
                    "raw_generated_token_ids": forced_raw.get("raw_generated_token_ids"),
                    "parse_status": parse.get("parse_status")
                    if isinstance(parse, Mapping)
                    else None,
                    "drop_codes": [
                        value.get("code")
                        for value in dropped
                        if isinstance(value, Mapping)
                    ],
                }
            )
    top_same_category_ious = [
        float(value["top_same_category_candidate_iou"])
        for value in unmatched_matches
        if value.get("top_same_category_candidate_iou") is not None
    ]
    candidate_input = manifest.get("inputs", {}).get("candidate_pool", {})
    candidate_path = Path(str(candidate_input.get("path"))).resolve(strict=True)
    if sha256_file(candidate_path) != candidate_input.get("sha256"):
        raise ValueError("candidate pool hash mismatch")
    categories = _candidate_category_index(candidate_path)

    gained_categories: Counter[str] = Counter()
    lost_categories: Counter[str] = Counter()
    for case in cases:
        image_categories = categories[str(case["image_id"])]
        for owner_id in case["causal_gained_owner_ids"]:
            gained_categories[image_categories[owner_id]] += 1
        for owner_id in case["causal_lost_owner_ids"]:
            lost_categories[image_categories[owner_id]] += 1

    result = {
        "schema_version": SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "inputs": {
            "manifest": {"path": str(manifest_path), "sha256": manifest_hash},
            "statistics_cases": {
                "path": str(statistics_path),
                "sha256": sha256_file(statistics_path),
            },
            "candidate_pool": {
                "path": str(candidate_path),
                "sha256": sha256_file(candidate_path),
            },
            "receipts": receipt_provenance,
        },
        "runtime_contract": runtime_contract,
        "aggregate": _aggregate(cases),
        "strata": {
            "source_margin": _strata(
                cases, key=lambda case: _source_margin_bin(float(case["source_pairwise_margin"]))
            ),
            "nested_checkpoint_positive_band": _strata(
                cases,
                key=lambda case: _diagnostic_positive_band(case["diagnostic_margins"]),
            ),
            "prefix_depth": _strata(
                cases, key=lambda case: _depth_bin(int(case["prefix_depth"]))
            ),
            "object_count_band": _strata(
                cases, key=lambda case: str(case["object_count_band"])
            ),
            "remaining_owner_count": _strata(
                cases,
                key=lambda case: _remaining_owner_bin(int(case["remaining_owner_count"])),
            ),
        },
        "causal_gain_categories": dict(gained_categories.most_common()),
        "causal_loss_categories": dict(lost_categories.most_common()),
        "audit": {
            "forced_valid_unmatched_or_ambiguous": {
                "entity_match_status_counts": dict(
                    sorted(
                        Counter(str(value.get("status")) for value in unmatched_matches).items()
                    )
                ),
                "top_same_category_iou_available_count": len(top_same_category_ious),
                "top_same_category_iou_at_least_0_30_count": sum(
                    value >= 0.30 for value in top_same_category_ious
                ),
                "top_same_category_iou_at_least_0_50_count": sum(
                    value >= 0.50 for value in top_same_category_ious
                ),
                "description_counts": dict(
                    Counter(str(value.get("description")) for value in unmatched_matches).most_common()
                ),
            },
            "forced_invalid_or_incomplete_cases": invalid_cases,
        },
        "audit_boundary_ids": [
            str(case["boundary_id"])
            for case in cases
            if case["causal_gained_owner_ids"]
            or case["causal_lost_owner_ids"]
            or case["native"]["unmatched_or_ambiguous_prediction_count"]
            or case["forced_opener"]["unmatched_or_ambiguous_prediction_count"]
            or not case["forced_opener"]["valid_row"]
        ],
        "claim_boundary": (
            "paired one-row causal outcomes on a deterministic 200-boundary panel; not free-rollout final-set gain"
        ),
    }

    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite immutable output root: {output_root}")
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--statistics-cases", type=Path, required=True)
    parser.add_argument("--receipts-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main() -> None:
    result = summarize(build_parser().parse_args())
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
