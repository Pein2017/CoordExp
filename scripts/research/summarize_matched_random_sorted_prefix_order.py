#!/usr/bin/env python3
"""Select matched prefix-order effects across sorted and random checkpoints.

The input is the raw ``same_covered_set_prefix_order.v2`` receipt written by
the stage-one runner.  This is deliberately an experiment-local, mechanical
selector: it does not interpret a different prompt as an order effect.  It
first proves that the two checkpoint receipts describe the same intervention,
then applies the predeclared first-row promotion rule independently at both
checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "same_covered_set_prefix_order.v2"
SUMMARY_SCHEMA_VERSION = "matched_random_sorted_prefix_order.summary.v1"
CHECKPOINTS = ("sorted", "random")


def _json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _list(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _required_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _identity(value: Mapping[str, Any], fields: Sequence[str], label: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in fields:
        if field not in value:
            raise ValueError(f"{label} is missing {field}")
        result[field] = value[field]
    return result


def _case_key(case: Mapping[str, Any]) -> str:
    return _required_string(case.get("case_id"), "case_id")


def _comparison_key(comparison: Mapping[str, Any]) -> str:
    return _required_string(comparison.get("comparison_id"), "comparison_id")


def _run_key(run: Mapping[str, Any]) -> tuple[str, str]:
    mode = _required_string(run.get("mode"), "run mode")
    seed = run.get("seed")
    if mode == "greedy":
        if seed is not None:
            raise ValueError("greedy run seed must be null")
        return (mode, "")
    if mode == "sample":
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("sample run seed must be an integer")
        return (mode, str(seed))
    raise ValueError(f"unexpected run mode {mode!r}")


def _load(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON") from exc
    payload = _mapping(value, str(path))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path} has unexpected schema version {payload.get('schema_version')!r}")
    return payload


def _artifact_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    image = _mapping(payload.get("image"), "artifact image")
    case_spec = _mapping(payload.get("case_spec"), "artifact case_spec")
    base_prompt = _mapping(payload.get("base_prompt"), "artifact base_prompt")
    return {
        "image": _identity(image, ("image_id", "image_sha256"), "artifact image"),
        "case_spec": _identity(case_spec, ("schema_version", "sha256"), "artifact case_spec"),
        # Deliberately hash only the executed base prompt identity, rather than
        # text.  Prompt wording is not a selection signal in this comparison.
        "base_prompt": {
            "image_sha256": _required_string(base_prompt.get("image_sha256"), "base_prompt image_sha256"),
            "prompt_token_ids_sha256": _required_string(
                base_prompt.get("observed_prompt_token_ids_sha256", base_prompt.get("prompt_token_ids_sha256")),
                "base_prompt prompt token hash",
            ),
        },
    }


def _index_payload(payload: Mapping[str, Any], *, source: str) -> dict[str, Any]:
    """Validate the local declaration and return a comparison/run index."""

    if payload.get("experiment") != "same_covered_set_prefix_order":
        raise ValueError(f"{source} has unexpected experiment {payload.get('experiment')!r}")
    config = _mapping(payload.get("config"), f"{source} config")
    declared_seeds = _list(config.get("seeds"), f"{source} config.seeds")
    if len(declared_seeds) != 4 or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in declared_seeds):
        raise ValueError(f"{source} must declare exactly four integer sample seeds")
    if len(set(declared_seeds)) != 4 or config.get("include_greedy") is not True:
        raise ValueError(f"{source} must declare unique sample seeds and greedy decoding")

    cases = _list(payload.get("cases"), f"{source} cases")
    if not cases:
        raise ValueError(f"{source} cases must be non-empty")
    indexed: dict[str, Any] = {"identity": _artifact_identity(payload), "config": {"seeds": list(declared_seeds), "include_greedy": True}, "cases": {}}
    for raw_case in cases:
        case = _mapping(raw_case, f"{source} case")
        case_id = _case_key(case)
        if case_id in indexed["cases"]:
            raise ValueError(f"{source} has duplicate case_id {case_id!r}")
        image_id = _required_string(case.get("image_id"), f"{source} case image_id")
        if image_id != indexed["identity"]["image"]["image_id"]:
            raise ValueError(f"{source} case {case_id} image_id differs from artifact image")
        arms = _mapping(case.get("arms"), f"{source} case {case_id} arms")
        invariants = _mapping(case.get("invariants_by_comparison"), f"{source} case {case_id} invariants")
        comparisons = _list(case.get("comparisons"), f"{source} case {case_id} comparisons")
        case_out: dict[str, Any] = {"image_id": image_id, "comparisons": {}}
        for raw_comparison in comparisons:
            comparison = _mapping(raw_comparison, f"{source} comparison")
            comparison_id = _comparison_key(comparison)
            if comparison_id in case_out["comparisons"]:
                raise ValueError(f"{source} case {case_id} has duplicate comparison_id {comparison_id!r}")
            arm_names = _list(comparison.get("arm_names"), f"{source} comparison {comparison_id} arm_names")
            if len(arm_names) != 2 or len(set(arm_names)) != 2 or any(not isinstance(name, str) for name in arm_names):
                raise ValueError(f"{source} comparison {comparison_id} must declare exactly two unique arms")
            if comparison_id not in invariants:
                raise ValueError(f"{source} comparison {comparison_id} has no declared invariants")
            arm_out: dict[str, Any] = {}
            for name in arm_names:
                arm = _mapping(arms.get(name), f"{source} comparison {comparison_id} arm {name}")
                prefix_hash = _required_string(arm.get("prefix_token_ids_sha256"), f"{source} arm {name} prefix hash")
                prefix_token_ids = _list(arm.get("prefix_token_ids"), f"{source} arm {name} prefix token ids")
                if _json_hash(prefix_token_ids) != prefix_hash:
                    raise ValueError(f"{source} arm {name} prefix token hash is inconsistent")
                runs: dict[tuple[str, str], Mapping[str, Any]] = {}
                for raw_run in _list(arm.get("runs"), f"{source} arm {name} runs"):
                    run = _mapping(raw_run, f"{source} arm {name} run")
                    if str(run.get("comparison_id")) != comparison_id:
                        continue
                    initial_prefix = _list(run.get("initial_prefix_token_ids"), f"{source} arm {name} initial prefix token ids")
                    initial_hash = _required_string(run.get("initial_prefix_token_ids_sha256"), f"{source} arm {name} initial prefix hash")
                    if _json_hash(initial_prefix) != prefix_hash or initial_hash != prefix_hash:
                        raise ValueError(f"{source} arm {name} run initial prefix differs from declared arm prefix")
                    key = _run_key(run)
                    if key in runs:
                        raise ValueError(f"{source} case {case_id} comparison {comparison_id} arm {name} has duplicate run {key}")
                    runs[key] = run
                expected = {("greedy", "")} | {("sample", str(seed)) for seed in declared_seeds}
                if set(runs) != expected:
                    raise ValueError(f"{source} case {case_id} comparison {comparison_id} arm {name} does not have one greedy plus declared samples")
                arm_out[name] = {"prefix_hash": prefix_hash, "runs": runs}
            case_out["comparisons"][comparison_id] = {
                "declaration": dict(comparison),
                "invariants": invariants[comparison_id],
                "arms": arm_out,
            }
        indexed["cases"][case_id] = case_out
    return indexed


def _assert_same(left: Any, right: Any, label: str) -> None:
    if left != right:
        raise ValueError(f"sorted/random mismatch for {label}")


def _validate_pairing(sorted_index: Mapping[str, Any], random_index: Mapping[str, Any], *, stem: str) -> None:
    _assert_same(sorted_index["identity"], random_index["identity"], f"{stem} artifact identity")
    _assert_same(sorted_index["config"], random_index["config"], f"{stem} decode seeds/modes")
    _assert_same(set(sorted_index["cases"]), set(random_index["cases"]), f"{stem} case identities")
    for case_id in sorted(sorted_index["cases"]):
        left_case = sorted_index["cases"][case_id]
        right_case = random_index["cases"][case_id]
        _assert_same(left_case["image_id"], right_case["image_id"], f"{stem}/{case_id} image")
        _assert_same(set(left_case["comparisons"]), set(right_case["comparisons"]), f"{stem}/{case_id} comparison identities")
        for comparison_id in sorted(left_case["comparisons"]):
            left = left_case["comparisons"][comparison_id]
            right = right_case["comparisons"][comparison_id]
            _assert_same(left["declaration"], right["declaration"], f"{stem}/{case_id}/{comparison_id} declaration")
            _assert_same(left["invariants"], right["invariants"], f"{stem}/{case_id}/{comparison_id} invariants")
            _assert_same(set(left["arms"]), set(right["arms"]), f"{stem}/{case_id}/{comparison_id} arm identities")
            for arm_name in left["arms"]:
                _assert_same(left["arms"][arm_name]["prefix_hash"], right["arms"][arm_name]["prefix_hash"], f"{stem}/{case_id}/{comparison_id}/{arm_name} prefix hash")
                _assert_same(set(left["arms"][arm_name]["runs"]), set(right["arms"][arm_name]["runs"]), f"{stem}/{case_id}/{comparison_id}/{arm_name} mode/seed identities")


def _first_row(run: Mapping[str, Any]) -> Mapping[str, Any] | None:
    rows = run.get("rows")
    if not isinstance(rows, list) or not rows:
        return None
    return _mapping(rows[0], "first generated row")


def _row_observation(run: Mapping[str, Any]) -> dict[str, Any]:
    row = _first_row(run)
    if row is None:
        # This is retained in the receipt but never activates promotion.
        return {"strict_owner": None, "outcome": "generic_corruption", "covered_recurrence": None, "exact_generated_row_sha256": None}
    accepted = bool(row.get("accepted_complete_row"))
    parse = _mapping(row.get("parse_evidence", {}), "first row parse_evidence")
    stop = _mapping(row.get("row_stop", {}), "first row row_stop")
    parse_status = str(parse.get("parse_status", "missing"))
    stop_reason = str(stop.get("stop_reason", "missing"))
    if accepted:
        outcome = "accepted"
    elif parse_status not in {"accepted", "missing"}:
        outcome = "malformed"
    elif stop_reason not in {"complete_row", "missing"} or bool(row.get("terminal", False)):
        outcome = "terminal"
    else:
        outcome = "generic_corruption"
    owners = row.get("strict_matched_owner_ids", row.get("owner_entity_ids", []))
    owners = _list(owners, "first row strict owners")
    strict_owner = str(owners[0]) if accepted and len(owners) == 1 else None
    # ``owner_recurrence`` is recurrence *within newly generated horizon rows*.
    # Stage One has a one-row horizon, so it is necessarily false at row zero.
    # The causal question here is instead whether the first generated owner was
    # already present in the frozen prefix; the runner records that explicitly.
    covered_prefix_owner_ids = _list(row.get("covered_prefix_owner_ids", []), "first row covered prefix owners")
    covered_recurrence = bool(covered_prefix_owner_ids) if accepted else None
    return {
        "strict_owner": strict_owner,
        "outcome": outcome,
        "covered_recurrence": covered_recurrence,
        "covered_prefix_owner_ids": list(covered_prefix_owner_ids),
        "exact_generated_row_sha256": row.get("raw_generated_token_ids_sha256"),
    }


def _pair_observations(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    left_observation = _row_observation(left)
    right_observation = _row_observation(right)
    owner_switch = (
        left_observation["strict_owner"] is not None
        and right_observation["strict_owner"] is not None
        and left_observation["strict_owner"] != right_observation["strict_owner"]
    )
    outcome_difference = left_observation["outcome"] != right_observation["outcome"]
    recurrence_difference = (
        left_observation["covered_recurrence"] is not None
        and right_observation["covered_recurrence"] is not None
        and left_observation["covered_recurrence"] != right_observation["covered_recurrence"]
    )
    return {
        "left": left_observation,
        "right": right_observation,
        "strict_owner_switch": owner_switch,
        "outcome_difference": outcome_difference,
        "covered_recurrence_difference": recurrence_difference,
        "exact_generated_row_difference": left_observation["exact_generated_row_sha256"] != right_observation["exact_generated_row_sha256"],
    }


def _checkpoint_summary(comparison: Mapping[str, Any]) -> dict[str, Any]:
    names = list(comparison["arms"])
    left_name, right_name = names
    left_runs = comparison["arms"][left_name]["runs"]
    right_runs = comparison["arms"][right_name]["runs"]
    paired = {key: _pair_observations(left_runs[key], right_runs[key]) for key in sorted(left_runs)}
    greedy = paired[("greedy", "")]
    sampled = [paired[("sample", str(seed))] for seed in sorted(int(key[1]) for key in paired if key[0] == "sample")]
    if len(sampled) != 4:
        raise AssertionError("local run validation must provide four samples")
    owner_switches = sum(item["strict_owner_switch"] for item in sampled)
    status_directions = Counter(
        f"{item['left']['outcome']}->{item['right']['outcome']}"
        for item in sampled
        if item["outcome_difference"] and "generic_corruption" not in {item["left"]["outcome"], item["right"]["outcome"]}
    )
    recurrence_directions = Counter(
        f"{str(item['left']['covered_recurrence']).lower()}->{str(item['right']['covered_recurrence']).lower()}"
        for item in sampled if item["covered_recurrence_difference"]
    )
    reasons: list[str] = []
    if greedy["strict_owner_switch"]:
        reasons.append("greedy_strict_owner_switch")
    if owner_switches >= 2:
        reasons.append("sampled_strict_owner_switches_at_least_2_of_4")
    greedy_outcomes = {greedy["left"]["outcome"], greedy["right"]["outcome"]}
    if "accepted" in greedy_outcomes and "terminal" in greedy_outcomes:
        reasons.append("greedy_valid_vs_terminal")
    if "accepted" in greedy_outcomes and "malformed" in greedy_outcomes:
        reasons.append("greedy_valid_vs_malformed")
    if greedy["covered_recurrence_difference"]:
        reasons.append("greedy_covered_recurrence_difference")
    for direction, count in sorted(status_directions.items()):
        if count >= 2:
            reasons.append(f"sampled_same_direction_status_difference_{direction}_at_least_2_of_4")
    for direction, count in sorted(recurrence_directions.items()):
        if count >= 2:
            reasons.append(f"sampled_same_direction_covered_recurrence_difference_{direction}_at_least_2_of_4")
    any_noncorruption_signal = any(
        item["strict_owner_switch"] or item["covered_recurrence_difference"]
        or (item["outcome_difference"] and "generic_corruption" not in {item["left"]["outcome"], item["right"]["outcome"]})
        for item in paired.values()
    )
    return {
        "arms": names,
        "arm_prefix_token_ids_sha256": {name: comparison["arms"][name]["prefix_hash"] for name in names},
        "paired_runs": [
            {"mode": key[0], "seed": None if key[0] == "greedy" else int(key[1]), **paired[key]}
            for key in sorted(paired)
        ],
        "greedy": greedy,
        "sampled_strict_owner_switch_count_out_of_4": owner_switches,
        "sampled_same_direction_status_difference_counts": dict(sorted(status_directions.items())),
        "sampled_same_direction_covered_recurrence_difference_counts": dict(sorted(recurrence_directions.items())),
        "promotion_reasons": reasons,
        "activated": bool(reasons),
        "generic_corruption_only": not any_noncorruption_signal,
    }


def summarize(sorted_dir: Path, random_dir: Path) -> dict[str, Any]:
    """Return deterministic, validated cross-checkpoint selection receipt."""

    sorted_paths = {path.name: path for path in sorted_dir.glob("*.json")}
    random_paths = {path.name: path for path in random_dir.glob("*.json")}
    if not sorted_paths or not random_paths:
        raise ValueError("both checkpoint directories must contain JSON artifacts")
    if set(sorted_paths) != set(random_paths):
        raise ValueError("sorted/random artifact file names do not match")
    pairs: list[dict[str, Any]] = []
    for filename in sorted(sorted_paths):
        sorted_index = _index_payload(_load(sorted_paths[filename]), source=f"sorted/{filename}")
        random_index = _index_payload(_load(random_paths[filename]), source=f"random/{filename}")
        _validate_pairing(sorted_index, random_index, stem=filename)
        for case_id in sorted(sorted_index["cases"]):
            for comparison_id in sorted(sorted_index["cases"][case_id]["comparisons"]):
                sorted_summary = _checkpoint_summary(sorted_index["cases"][case_id]["comparisons"][comparison_id])
                random_summary = _checkpoint_summary(random_index["cases"][case_id]["comparisons"][comparison_id])
                activated = sorted_summary["activated"] or random_summary["activated"]
                pairs.append({
                    "artifact_file": filename,
                    "case_id": case_id,
                    "comparison_id": comparison_id,
                    "image_id": sorted_index["cases"][case_id]["image_id"],
                    "declared_invariants": sorted_index["cases"][case_id]["comparisons"][comparison_id]["invariants"],
                    "checkpoint_results": {"sorted": sorted_summary, "random": random_summary},
                    "promoted_for_identical_scoring_under_both_checkpoints": activated,
                    "promotion_checkpoints": [name for name, summary in (("sorted", sorted_summary), ("random", random_summary)) if summary["activated"]],
                })
    selected = [
        {key: pair[key] for key in ("artifact_file", "case_id", "comparison_id", "image_id", "promotion_checkpoints")}
        for pair in pairs if pair["promoted_for_identical_scoring_under_both_checkpoints"]
    ]
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "checkpoint_directories": {"sorted": str(sorted_dir.resolve()), "random": str(random_dir.resolve())},
        "selection_rule": {
            "sample_count": 4,
            "promote_when": [
                "greedy strict-owner switch",
                "at least two sampled strict-owner switches out of four",
                "greedy valid-versus-terminal, valid-versus-malformed, or covered-recurrence difference",
                "same-direction sampled status or covered-recurrence difference in at least two of four samples",
            ],
            "reject": "generic-corruption-only differences",
            "cross_checkpoint": "an activation at either checkpoint selects the pair for identical scoring under both checkpoints",
        },
        "pairs": pairs,
        "selected_pairs": selected,
        "selected_pair_count": len(selected),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sorted-dir", type=Path, required=True)
    parser.add_argument("--random-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Deterministic JSON selection receipt")
    args = parser.parse_args()
    summary = summarize(args.sorted_dir.expanduser().resolve(), args.random_dir.expanduser().resolve())
    args.output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.expanduser().resolve().write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
