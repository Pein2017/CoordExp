#!/usr/bin/env python3
"""Compare Source and transition-step-36 one-row releases on the same boundaries."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
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
from scripts.research.summarize_paired_terminal_forced_opener_release import (  # noqa: E402
    CASE_SCHEMA_VERSION,
    RECEIPT_SCHEMA_VERSION,
    SCHEMA_VERSION as RELEASE_SUMMARY_SCHEMA_VERSION,
    _depth_bin,
    _diagnostic_positive_band,
    _read_json,
    _read_jsonl,
    _remaining_owner_bin,
    _wilson_interval,
)
from src.config.fingerprint import sha256_file  # noqa: E402


UNIT_ID = "2026-07-26-source-versus-transition-step36-forced-opener-owner-selection"
SOURCE_UNIT_ID = "2026-07-25-paired-natural-terminal-forced-opener-release"
SCHEMA_VERSION = "source_transition_forced_opener_comparison.summary.v1"
COMPARISON_CASE_SCHEMA_VERSION = "source_transition_forced_opener_comparison.case.v1"


def _owner_ids(release: Mapping[str, Any]) -> set[str]:
    values = release.get("verified_uncovered_owner_ids")
    if not isinstance(values, list):
        raise ValueError("release lacks verified uncovered owner IDs")
    return {str(value) for value in values}


def _load_receipts(
    root: Path, *, expected_unit_id: str, expected_role: str
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], list[dict[str, str]]]:
    paths = sorted(root.expanduser().resolve(strict=True).glob("shard-*.json"))
    if not paths:
        raise ValueError(f"no receipts under {root}")
    cases: list[dict[str, Any]] = []
    runtime: dict[str, Any] | None = None
    shard_indices: set[int] = set()
    provenance: list[dict[str, str]] = []
    for path in paths:
        receipt = _read_json(path)
        if receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION:
            raise ValueError(f"unexpected receipt schema: {path}")
        if receipt.get("unit_id") != expected_unit_id:
            raise ValueError(f"unexpected receipt unit: {path}")
        if receipt.get("checkpoint_role") != expected_role:
            raise ValueError(f"unexpected checkpoint role: {path}")
        current_runtime = receipt.get("runtime")
        if not isinstance(current_runtime, Mapping):
            raise ValueError(f"receipt lacks runtime: {path}")
        current_runtime = dict(current_runtime)
        if runtime is None:
            runtime = current_runtime
        shard_index = int(current_runtime["shard_index"])
        if shard_index in shard_indices:
            raise ValueError(f"duplicate shard index {shard_index}: {path}")
        shard_indices.add(shard_index)
        values = receipt.get("cases")
        if not isinstance(values, list):
            raise ValueError(f"receipt lacks cases: {path}")
        cases.extend(dict(value) for value in values)
        provenance.append({"path": str(path), "sha256": sha256_file(path)})
    if runtime is None:
        raise AssertionError("unreachable empty receipt set")
    if shard_indices != set(range(int(runtime["shard_count"]))):
        raise ValueError(f"incomplete shard set: {sorted(shard_indices)}")
    by_id = {str(case["boundary_id"]): case for case in cases}
    if len(cases) != 200 or len(by_id) != 200:
        raise ValueError("receipts must contain 200 unique boundaries")
    return by_id, runtime, provenance


def _runtime_comparison(source: Mapping[str, Any], transition: Mapping[str, Any]) -> dict[str, Any]:
    common_keys = (
        "physical_batch_size",
        "runtime_dtype_mode",
        "model_dtype",
        "max_new_tokens",
        "malformed_limit",
        "repetition_penalty",
        "temperature",
        "top_p",
        "source_jsonl_sha256",
        "attention_implementation",
        "forced_opener_token_id",
    )
    mismatches = {
        key: {"source": source.get(key), "transition": transition.get(key)}
        for key in common_keys
        if source.get(key) != transition.get(key)
    }
    if mismatches:
        raise ValueError(f"decision-owning runtime mismatch: {mismatches}")
    source_backend = source.get("backend_session")
    transition_backend = transition.get("backend_session")
    if not isinstance(source_backend, Mapping) or not isinstance(
        transition_backend, Mapping
    ):
        raise ValueError("runtime lacks backend-session identity")
    identity_keys = ("processor_identity", "tokenizer_identity", "response_family")
    frontend_mismatches = {
        key: {"source": source_backend.get(key), "transition": transition_backend.get(key)}
        for key in identity_keys
        if source_backend.get(key) != transition_backend.get(key)
    }
    source_model = source_backend.get("model_identity")
    transition_model = transition_backend.get("model_identity")
    if not isinstance(source_model, Mapping) or not isinstance(transition_model, Mapping):
        raise ValueError("runtime lacks model identity")
    source_base = source_model.get("base")
    transition_base = transition_model.get("base")
    if source_base != transition_base:
        frontend_mismatches["base_model"] = {
            "source": source_base,
            "transition": transition_base,
        }
    if source_model.get("family") != transition_model.get("family"):
        frontend_mismatches["model_family"] = {
            "source": source_model.get("family"),
            "transition": transition_model.get("family"),
        }
    if frontend_mismatches:
        raise ValueError(f"frontend identity mismatch: {frontend_mismatches}")
    return {
        "status": "matched_except_checkpoint_adapter_and_embedding_delta",
        "matched_runtime_keys": list(common_keys),
        "matched_frontend_identity_keys": list(identity_keys) + ["base_model", "model_family"],
        "source_config_path": source.get("config_path"),
        "source_authored_config_sha256": source.get("authored_config_sha256"),
        "transition_config_path": transition.get("config_path"),
        "transition_authored_config_sha256": transition.get("authored_config_sha256"),
        "source_model_identity": source_model,
        "transition_checkpoint_contract": transition.get("checkpoint_contract"),
    }


def _exact_two_sided_sign_test(positive: int, negative: int) -> dict[str, Any]:
    discordant = positive + negative
    if discordant == 0:
        return {"positive": positive, "negative": negative, "discordant": 0, "p_value": 1.0}
    tail = min(positive, negative)
    probability = sum(math.comb(discordant, k) for k in range(tail + 1)) / (2**discordant)
    return {
        "positive": positive,
        "negative": negative,
        "discordant": discordant,
        "p_value": min(1.0, 2.0 * probability),
    }


def _classify_case(
    *,
    source: Mapping[str, Any],
    transition: Mapping[str, Any],
    source_raw: Mapping[str, Any],
    transition_raw: Mapping[str, Any],
) -> dict[str, Any]:
    invariant_keys = (
        "boundary_id",
        "image_id",
        "prefix_depth",
        "object_count_band",
        "annotation_object_count",
        "remaining_owner_count",
        "remaining_ledger_owner_ids",
        "covered_ledger_owner_ids",
        "diagnostic_margins",
    )
    for key in invariant_keys:
        if source.get(key) != transition.get(key):
            raise ValueError(f"{source.get('boundary_id')} classified invariant mismatch: {key}")
    raw_invariants = (
        "image_id",
        "prefix_depth",
        "covered_owner_ids",
        "remaining_owner_ids",
        "base_prompt_token_ids_sha256",
        "prefix_token_ids_sha256",
    )
    for key in raw_invariants:
        if source_raw.get(key) != transition_raw.get(key):
            raise ValueError(f"{source.get('boundary_id')} raw invariant mismatch: {key}")

    source_forced = source["forced_opener"]
    transition_forced = transition["forced_opener"]
    source_native = source["native"]
    transition_native = transition["native"]
    source_owners = _owner_ids(source_forced)
    transition_owners = _owner_ids(transition_forced)
    gained = transition_owners - source_owners
    lost = source_owners - transition_owners
    retained = source_owners & transition_owners
    source_forced_raw = source_raw["releases"]["forced_opener"]
    transition_forced_raw = transition_raw["releases"]["forced_opener"]
    return {
        "schema_version": COMPARISON_CASE_SCHEMA_VERSION,
        "boundary_id": str(source["boundary_id"]),
        "image_id": str(source["image_id"]),
        "prefix_depth": int(source["prefix_depth"]),
        "object_count_band": str(source["object_count_band"]),
        "remaining_owner_count": int(source["remaining_owner_count"]),
        "diagnostic_margins": source["diagnostic_margins"],
        "remaining_ledger_owner_ids": source["remaining_ledger_owner_ids"],
        "covered_ledger_owner_ids": source["covered_ledger_owner_ids"],
        "source_native": source_native,
        "transition_native": transition_native,
        "source_forced": source_forced,
        "transition_forced": transition_forced,
        "checkpoint_gained_owner_ids": sorted(gained),
        "checkpoint_lost_owner_ids": sorted(lost),
        "checkpoint_retained_owner_ids": sorted(retained),
        "checkpoint_owner_net": len(gained) - len(lost),
        "forced_raw_row_token_ids_equal": source_forced_raw.get("raw_generated_token_ids")
        == transition_forced_raw.get("raw_generated_token_ids"),
        "source_forced_raw_text": source_forced_raw.get("raw_generated_text"),
        "transition_forced_raw_text": transition_forced_raw.get("raw_generated_text"),
        "source_forced_entity_matches": source_forced_raw.get("entity_matches"),
        "transition_forced_entity_matches": transition_forced_raw.get("entity_matches"),
    }


def _outcome_counts(cases: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(case[key]["outcome"]) for case in cases).items()))


def _aggregate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    source_any = sum(bool(_owner_ids(case["source_forced"])) for case in cases)
    transition_any = sum(bool(_owner_ids(case["transition_forced"])) for case in cases)
    gained = sum(len(case["checkpoint_gained_owner_ids"]) for case in cases)
    lost = sum(len(case["checkpoint_lost_owner_ids"]) for case in cases)
    retained = sum(len(case["checkpoint_retained_owner_ids"]) for case in cases)
    nets = [int(case["checkpoint_owner_net"]) for case in cases]
    positive = sum(value > 0 for value in nets)
    negative = sum(value < 0 for value in nets)
    exchanges = sum(
        bool(case["checkpoint_gained_owner_ids"])
        and bool(case["checkpoint_lost_owner_ids"])
        for case in cases
    )
    transitions = Counter(
        f"source_{bool(_owner_ids(case['source_forced']))}_transition_{bool(_owner_ids(case['transition_forced']))}"
        for case in cases
    )
    return {
        "boundary_count": len(cases),
        "source_forced_any_uncovered_owner": {
            "success_count": source_any,
            "count": len(cases),
            "fraction": source_any / len(cases),
            "wilson_95_interval": _wilson_interval(source_any, len(cases)),
        },
        "transition_forced_any_uncovered_owner": {
            "success_count": transition_any,
            "count": len(cases),
            "fraction": transition_any / len(cases),
            "wilson_95_interval": _wilson_interval(transition_any, len(cases)),
        },
        "forced_any_uncovered_owner_transitions": dict(sorted(transitions.items())),
        "checkpoint_gained_owner_count": gained,
        "checkpoint_lost_owner_count": lost,
        "checkpoint_retained_owner_count": retained,
        "checkpoint_owner_net": gained - lost,
        "positive_net_boundary_count": positive,
        "negative_net_boundary_count": negative,
        "owner_exchange_boundary_count": exchanges,
        "exact_two_sided_paired_sign_test": _exact_two_sided_sign_test(positive, negative),
        "per_boundary_owner_net": _describe(nets),
        "per_boundary_owner_net_bootstrap_mean": _bootstrap_mean_interval(
            nets, seed=260726, replicates=5000
        ),
        "forced_raw_row_token_ids_equal_count": sum(
            bool(case["forced_raw_row_token_ids_equal"]) for case in cases
        ),
        "source_native_outcomes": _outcome_counts(cases, "source_native"),
        "transition_native_outcomes": _outcome_counts(cases, "transition_native"),
        "source_forced_outcomes": _outcome_counts(cases, "source_forced"),
        "transition_forced_outcomes": _outcome_counts(cases, "transition_forced"),
    }


def _strata(
    cases: list[dict[str, Any]], *, key: Callable[[Mapping[str, Any]], str]
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[key(case)].append(case)
    return {name: _aggregate(values) for name, values in sorted(grouped.items())}


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    source_summary_path = args.source_summary.expanduser().resolve(strict=True)
    transition_summary_path = args.transition_summary.expanduser().resolve(strict=True)
    source_summary = _read_json(source_summary_path)
    transition_summary = _read_json(transition_summary_path)
    if source_summary.get("schema_version") != RELEASE_SUMMARY_SCHEMA_VERSION:
        raise ValueError("unexpected Source summary schema")
    if transition_summary.get("schema_version") != RELEASE_SUMMARY_SCHEMA_VERSION:
        raise ValueError("unexpected transition summary schema")
    if source_summary.get("unit_id") != SOURCE_UNIT_ID:
        raise ValueError("unexpected Source summary unit")
    if transition_summary.get("unit_id") != UNIT_ID:
        raise ValueError("unexpected transition summary unit")

    source_cases_path = args.source_cases.expanduser().resolve(strict=True)
    transition_cases_path = args.transition_cases.expanduser().resolve(strict=True)
    source_rows = _read_jsonl(source_cases_path)
    transition_rows = _read_jsonl(transition_cases_path)
    source_by_id = {str(case["boundary_id"]): case for case in source_rows}
    transition_by_id = {str(case["boundary_id"]): case for case in transition_rows}
    if len(source_rows) != 200 or len(source_by_id) != 200:
        raise ValueError("Source reduction must contain 200 unique cases")
    if len(transition_rows) != 200 or len(transition_by_id) != 200:
        raise ValueError("transition reduction must contain 200 unique cases")
    if set(source_by_id) != set(transition_by_id):
        raise ValueError("checkpoint reductions cover different boundaries")
    if any(case.get("schema_version") != CASE_SCHEMA_VERSION for case in source_rows):
        raise ValueError("unexpected Source case schema")
    if any(case.get("schema_version") != CASE_SCHEMA_VERSION for case in transition_rows):
        raise ValueError("unexpected transition case schema")

    source_raw, source_runtime, source_receipts = _load_receipts(
        args.source_receipts_root,
        expected_unit_id=SOURCE_UNIT_ID,
        expected_role="source",
    )
    transition_raw, transition_runtime, transition_receipts = _load_receipts(
        args.transition_receipts_root,
        expected_unit_id=UNIT_ID,
        expected_role="transition-step36",
    )
    if set(source_raw) != set(source_by_id) or set(transition_raw) != set(source_by_id):
        raise ValueError("raw receipts and reductions cover different boundaries")
    runtime_comparison = _runtime_comparison(source_runtime, transition_runtime)
    cases = [
        _classify_case(
            source=source_by_id[boundary_id],
            transition=transition_by_id[boundary_id],
            source_raw=source_raw[boundary_id],
            transition_raw=transition_raw[boundary_id],
        )
        for boundary_id in sorted(source_by_id)
    ]
    review_cases = [
        case
        for case in cases
        if not case["forced_raw_row_token_ids_equal"]
        and (
            case["source_forced"]["outcome"] == "valid_unmatched_or_ambiguous"
            or case["transition_forced"]["outcome"] == "valid_unmatched_or_ambiguous"
            or case["checkpoint_gained_owner_ids"]
            or case["checkpoint_lost_owner_ids"]
        )
    ]
    result = {
        "schema_version": SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "inputs": {
            "source_summary": {
                "path": str(source_summary_path),
                "sha256": sha256_file(source_summary_path),
            },
            "source_cases": {
                "path": str(source_cases_path),
                "sha256": sha256_file(source_cases_path),
            },
            "transition_summary": {
                "path": str(transition_summary_path),
                "sha256": sha256_file(transition_summary_path),
            },
            "transition_cases": {
                "path": str(transition_cases_path),
                "sha256": sha256_file(transition_cases_path),
            },
            "source_receipts": source_receipts,
            "transition_receipts": transition_receipts,
        },
        "runtime_comparison": runtime_comparison,
        "aggregate": _aggregate(cases),
        "strata": {
            "diagnostic_positive_band": _strata(
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
        "review_case_count": len(review_cases),
        "review_boundary_ids": [case["boundary_id"] for case in review_cases],
        "claim_boundary": (
            "paired one-row checkpoint effect with continuation fixed; not a free-rollout final-set effect"
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
    with (output_root / "review-cases.jsonl").open("w", encoding="utf-8") as handle:
        for case in review_cases:
            handle.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-summary", type=Path, required=True)
    parser.add_argument("--source-cases", type=Path, required=True)
    parser.add_argument("--source-receipts-root", type=Path, required=True)
    parser.add_argument("--transition-summary", type=Path, required=True)
    parser.add_argument("--transition-cases", type=Path, required=True)
    parser.add_argument("--transition-receipts-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main() -> None:
    result = summarize(build_parser().parse_args())
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
