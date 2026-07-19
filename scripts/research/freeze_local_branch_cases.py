#!/usr/bin/env python3
"""Freeze outcome-blind primary cases from local branch discovery shards.

The discovery shards are model-run artifacts.  This module is deliberately a
pure post-processor: it never runs inference, never infers negative labels
from an absent annotation, and never looks at downstream intervention
results.  A case is admitted only when both natural rows have an explicit,
unambiguous positive owner under the frozen discovery contract.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
import glob
import hashlib
import json
from pathlib import Path
from typing import Any


DISCOVERY_SCHEMA_VERSION = "local_branch_causal_value.v2"
FROZEN_CASE_SCHEMA_VERSION = "local_branch_causal_cases.v1"
FROZEN_DISCOVERY_SEEDS = tuple(range(11, 19))
VERIFIED_STATUSES = {"verified", "approved", "human_verified"}
DIAGNOSTIC_OUTCOMES = {
    "verified_benign_uncovered_owner",
    "verified_unsupported",
    "terminal_diagnostic",
    "malformed_diagnostic",
    "unresolved_refusal",
}


class FreezeValidationError(ValueError):
    """Raised when a frozen input contract cannot be trusted."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FreezeValidationError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FreezeValidationError(f"JSON artifact must be an object: {path}")
    return value


def _as_image_id(value: Any) -> str:
    if isinstance(value, bool) or value is None:
        raise FreezeValidationError(f"invalid image identifier: {value!r}")
    text = str(value).strip()
    if not text:
        raise FreezeValidationError("empty image identifier")
    return text


def _token_ids(value: Any, *, field: str) -> list[int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise FreezeValidationError(f"{field} must be a token-id sequence")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise FreezeValidationError(f"{field} contains an invalid token id")
        result.append(int(item))
    return result


def token_ids_hash(value: Sequence[int]) -> str:
    return sha256_bytes([int(item) for item in value])


def normalized_category(value: Any) -> str:
    """Normalize only spelling/whitespace; do not merge semantic labels."""

    return " ".join(str(value).strip().casefold().split())


def longest_common_prefix_and_difference(
    left: Sequence[int], right: Sequence[int]
) -> dict[str, Any]:
    """Return the exact raw-token branch point, without decoding text."""

    left_ids = _token_ids(left, field="left row token ids")
    right_ids = _token_ids(right, field="right row token ids")
    if not left_ids or not right_ids:
        raise FreezeValidationError("empty_complete_row_token_ids")
    if len(left_ids) != len(right_ids):
        raise FreezeValidationError("row_length_mismatch")
    index = 0
    while index < len(left_ids) and left_ids[index] == right_ids[index]:
        index += 1
    if index == len(left_ids):
        raise FreezeValidationError("identical_rows")
    shared = left_ids[:index]
    return {
        "shared_row_prefix_token_ids": shared,
        "shared_row_prefix_token_ids_sha256": token_ids_hash(shared),
        "shared_row_prefix_length": len(shared),
        "first_difference_index": index,
        "native_branch_token_id": left_ids[index],
        "sampled_branch_token_id": right_ids[index],
    }


def _owner_category(owner: Mapping[str, Any]) -> str:
    for key in ("category", "description", "label", "name", "class"):
        if key in owner and str(owner[key]).strip():
            return normalized_category(owner[key])
    return ""


def _owner_status(owner: Mapping[str, Any]) -> str:
    return normalized_category(owner.get("verification", owner.get("status", "")))


def _ledger_by_id(ledger: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for owner in ledger:
        if not isinstance(owner, Mapping) or owner.get("entity_id") is None:
            continue
        result[str(owner["entity_id"])] = owner
    return result


def _explicitly_unsupported(row: Mapping[str, Any]) -> bool:
    """Accept unsupported only from an explicit human-review field."""

    for key in ("review_outcome", "human_review_outcome", "outcome"):
        value = normalized_category(row.get(key, ""))
        if value in {"verified_unsupported", "unsupported"}:
            return bool(str(row.get("review_reason", row.get("comment", ""))).strip())
    return False


def clean_single_owner(
    row: Mapping[str, Any],
    *,
    covered_entity_ids: Iterable[str],
    entity_ledger: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate one complete row with exactly one verified physical owner.

    Every parsed prediction must be matched to the same owner.  A row with one
    matched owner plus another unmatched or ambiguous prediction is rejected;
    it must not become a seemingly clean positive case.
    """

    covered = {str(value) for value in covered_entity_ids}
    ledger = _ledger_by_id(entity_ledger)
    if row.get("status") != "success":
        return {"accepted": False, "refusal_code": "row_not_success"}
    # Natural sampled actions carry this explicit acceptance bit.  Native
    # greedy replays are stored directly from row generation and therefore do
    # not.  When present it must be true; when absent, the exact success,
    # complete-row stop, and raw-token checks below remain authoritative.
    if "accepted_complete_row" in row and row.get("accepted_complete_row") is not True:
        return {"accepted": False, "refusal_code": "row_not_complete"}
    stop = row.get("row_stop") if isinstance(row.get("row_stop"), Mapping) else {}
    if str(stop.get("stop_reason", "")) != "complete_row":
        return {"accepted": False, "refusal_code": "row_not_complete"}
    try:
        raw_ids = _token_ids(row.get("raw_generated_token_ids", []), field="raw_generated_token_ids")
    except FreezeValidationError as exc:
        return {"accepted": False, "refusal_code": str(exc)}
    if not raw_ids:
        return {"accepted": False, "refusal_code": "empty_complete_row_token_ids"}
    matches = row.get("entity_matches")
    if not isinstance(matches, list) or not matches:
        return {"accepted": False, "refusal_code": "owner_matches_missing"}
    if any(not isinstance(match, Mapping) for match in matches):
        return {"accepted": False, "refusal_code": "owner_match_record_invalid"}
    if any(match.get("status") != "matched" for match in matches):
        return {"accepted": False, "refusal_code": "row_has_unmatched_or_ambiguous"}
    matched_ids = {
        str(match.get("matched_entity_id"))
        for match in matches
        if match.get("matched_entity_id") is not None
    }
    strict_ids = {str(value) for value in row.get("strict_matched_owner_ids", [])}
    if matched_ids != strict_ids or len(matched_ids) != 1:
        return {"accepted": False, "refusal_code": "owner_count_not_one"}
    owner_id = next(iter(matched_ids))
    owner = ledger.get(owner_id)
    if owner is None:
        return {"accepted": False, "refusal_code": "owner_missing_from_ledger"}
    if _owner_status(owner) not in VERIFIED_STATUSES:
        return {"accepted": False, "refusal_code": "owner_not_verified"}
    category = _owner_category(owner)
    if not category:
        return {"accepted": False, "refusal_code": "category_missing", "owner_id": owner_id}
    return {
        "accepted": True,
        "owner_id": owner_id,
        "category": category,
        "covered": owner_id in covered,
        "raw_token_ids": raw_ids,
        "raw_token_length": len(raw_ids),
    }


def _prefix_hash(prefix: Mapping[str, Any]) -> str:
    ids = _token_ids(prefix.get("prefix_token_ids", []), field="prefix_token_ids")
    observed = str(prefix.get("prefix_token_ids_sha256", ""))
    expected = token_ids_hash(ids)
    if observed and observed != expected:
        raise FreezeValidationError("prefix_hash_mismatch")
    return expected


def _native_root_reachable(prefix: Mapping[str, Any]) -> bool:
    provenance = prefix.get("trajectory_provenance", [])
    if not isinstance(provenance, list):
        return False
    return any(
        isinstance(item, Mapping) and int(item.get("trajectory_index", -1)) == 0
        for item in provenance
    )


def _action_matches_provenance(
    action: Mapping[str, Any], prefix: Mapping[str, Any]
) -> bool:
    provenance = prefix.get("trajectory_provenance", [])
    if not isinstance(provenance, list):
        return False
    action_tuple = (
        action.get("trajectory_index"),
        action.get("row_index"),
        action.get("mode"),
        action.get("seed"),
    )
    return any(
        isinstance(item, Mapping)
        and (
            item.get("trajectory_index"),
            item.get("row_index"),
            item.get("mode"),
            item.get("seed"),
        )
        == action_tuple
        for item in provenance
    )


def _native_diagnostic(
    native_row: Mapping[str, Any],
    *,
    classification: Mapping[str, Any],
    covered_entity_ids: Sequence[str],
    ledger: Sequence[Mapping[str, Any]],
) -> str:
    native = clean_single_owner(
        native_row,
        covered_entity_ids=covered_entity_ids,
        entity_ledger=ledger,
    )
    if native.get("accepted") and native.get("covered"):
        return "verified_covered_duplicate"
    label = normalized_category(classification.get("classification", ""))
    if label == "terminal_with_verified_remaining_owner":
        return "terminal_diagnostic"
    if label == "malformed":
        return "malformed_diagnostic"
    return "unresolved_refusal"


def derive_prefix_candidates(
    evaluation: Mapping[str, Any],
    *,
    entity_ledger: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Derive all admissible pairs and refusals for one exact prefix."""

    prefix = evaluation.get("prefix")
    if not isinstance(prefix, Mapping):
        return [], [{"outcome": "unresolved_refusal", "refusal_code": "prefix_record_missing"}]
    try:
        parent_hash = _prefix_hash(prefix)
    except FreezeValidationError as exc:
        return [], [{"outcome": "unresolved_refusal", "refusal_code": str(exc)}]
    if prefix.get("coverage_valid") is not True:
        return [], [{"outcome": "unresolved_refusal", "refusal_code": "prior_prefix_unresolved", "prefix_hash": parent_hash}]
    covered = [str(value) for value in prefix.get("covered_entity_ids", [])]
    native_row = evaluation.get("native_greedy_row")
    if not isinstance(native_row, Mapping):
        return [], [{"outcome": "unresolved_refusal", "refusal_code": "native_row_missing", "prefix_hash": parent_hash}]
    native_check = clean_single_owner(native_row, covered_entity_ids=covered, entity_ledger=entity_ledger)
    classification = evaluation.get("classification") if isinstance(evaluation.get("classification"), Mapping) else {}
    native_outcome = _native_diagnostic(native_row, classification=classification, covered_entity_ids=covered, ledger=entity_ledger)
    actions = prefix.get("natural_actions", [])
    if not isinstance(actions, list):
        return [], [{"outcome": "unresolved_refusal", "refusal_code": "natural_actions_missing", "prefix_hash": parent_hash}]
    candidates: list[dict[str, Any]] = []
    refusals: list[dict[str, Any]] = []
    for action in actions:
        if not isinstance(action, Mapping):
            refusals.append({"outcome": "unresolved_refusal", "refusal_code": "natural_action_invalid", "prefix_hash": parent_hash})
            continue
        if action.get("mode") != "sample":
            continue
        seed = action.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed not in FROZEN_DISCOVERY_SEEDS:
            refusals.append({"outcome": "unresolved_refusal", "refusal_code": "sample_seed_outside_frozen_set", "prefix_hash": parent_hash})
            continue
        if not _action_matches_provenance(action, prefix):
            refusals.append({"outcome": "unresolved_refusal", "refusal_code": "sampled_action_not_natural_or_wrong_prefix", "prefix_hash": parent_hash, "seed": seed})
            continue
        sampled_check = clean_single_owner(action, covered_entity_ids=covered, entity_ledger=entity_ledger)
        if not sampled_check.get("accepted"):
            outcome = "verified_unsupported" if _explicitly_unsupported(action) else "unresolved_refusal"
            refusals.append({"outcome": outcome, "refusal_code": sampled_check.get("refusal_code"), "prefix_hash": parent_hash, "seed": seed})
            continue
        if sampled_check.get("covered"):
            refusals.append({"outcome": "unresolved_refusal", "refusal_code": "sampled_owner_already_covered", "prefix_hash": parent_hash, "seed": seed})
            continue
        if native_outcome != "verified_covered_duplicate":
            diagnostic = native_outcome
            if diagnostic == "unresolved_refusal" and sampled_check.get("owner_id"):
                diagnostic = "verified_benign_uncovered_owner"
            refusals.append({"outcome": diagnostic, "refusal_code": "native_row_not_verified_covered_duplicate", "prefix_hash": parent_hash, "seed": seed})
            continue
        if not native_check.get("accepted") or not native_check.get("covered"):
            refusals.append({"outcome": "unresolved_refusal", "refusal_code": "native_row_not_verified_covered_duplicate", "prefix_hash": parent_hash, "seed": seed})
            continue
        if native_check.get("category") != sampled_check.get("category"):
            refusals.append({"outcome": "unresolved_refusal", "refusal_code": "category_mismatch", "prefix_hash": parent_hash, "seed": seed})
            continue
        try:
            branch = longest_common_prefix_and_difference(native_check["raw_token_ids"], sampled_check["raw_token_ids"])
        except FreezeValidationError as exc:
            refusals.append({"outcome": "unresolved_refusal", "refusal_code": str(exc), "prefix_hash": parent_hash, "seed": seed})
            continue
        candidates.append({
            "image_id": _as_image_id(evaluation.get("image_id", prefix.get("image_id", ""))),
            "parent_prefix_token_ids": _token_ids(prefix.get("prefix_token_ids", []), field="prefix_token_ids"),
            "parent_prefix_token_ids_sha256": parent_hash,
            "covered_entity_ids": sorted(set(covered)),
            "native_root_greedy_reachable": _native_root_reachable(prefix),
            "natural_sample_seed": int(seed),
            "natural_trajectory_index": int(action.get("trajectory_index", -1)),
            "natural_row_index": int(action.get("row_index", -1)),
            "native_owner_id": native_check["owner_id"],
            "sampled_owner_id": sampled_check["owner_id"],
            "native_category": native_check["category"],
            "sampled_category": sampled_check["category"],
            "native_row_token_ids": native_check["raw_token_ids"],
            "sampled_row_token_ids": sampled_check["raw_token_ids"],
            "native_row_token_ids_sha256": token_ids_hash(native_check["raw_token_ids"]),
            "sampled_row_token_ids_sha256": token_ids_hash(sampled_check["raw_token_ids"]),
            "row_token_length": int(native_check["raw_token_length"]),
            **branch,
            "case_outcome": "admitted_object_vs_object",
            "source_prefix_provenance": prefix.get("trajectory_provenance", []),
            "source_sample_action_provenance": {
                "trajectory_index": int(action.get("trajectory_index", -1)),
                "row_index": int(action.get("row_index", -1)),
                "mode": "sample",
                "seed": int(seed),
            },
        })
    return candidates, refusals


def select_one_case_per_image(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    ordered = sorted(
        candidates,
        key=lambda item: (
            int(item["natural_row_index"]),
            int(item["natural_sample_seed"]),
            str(item["parent_prefix_token_ids_sha256"]),
        ),
    )
    return dict(ordered[0])


def _contract_from_manifest(manifest: Mapping[str, Any], manifest_path: Path) -> dict[str, Any]:
    if manifest.get("schema_version") != 2:
        raise FreezeValidationError("frozen_manifest_schema_version_mismatch")
    images = manifest.get("images")
    if not isinstance(images, list) or len(images) != 8:
        raise FreezeValidationError("frozen_manifest_must_contain_exactly_eight_images")
    image_ids = [_as_image_id(image.get("image_id")) for image in images if isinstance(image, Mapping)]
    if len(image_ids) != 8 or len(set(image_ids)) != 8:
        raise FreezeValidationError("frozen_manifest_image_ids_invalid")
    inference = manifest.get("inference_contract")
    budget = manifest.get("discovery_budget")
    if not isinstance(inference, Mapping) or not isinstance(budget, Mapping):
        raise FreezeValidationError("frozen_manifest_contract_missing")
    seeds = tuple(int(value) for value in budget.get("sampled_root_seeds", []))
    if seeds != FROZEN_DISCOVERY_SEEDS:
        raise FreezeValidationError("frozen_discovery_seed_contract_mismatch")
    if bool(budget.get("additional_same_prefix_sampling")):
        raise FreezeValidationError("supplementary_same_prefix_sampling_enabled")
    if inference.get("backend") != "Hugging Face" or inference.get("floating_point") != "full float32":
        raise FreezeValidationError("unsupported_decode_contract")
    if int(inference.get("physical_batch_size", -1)) != 1 or float(inference.get("repetition_penalty", -1)) != 1.0:
        raise FreezeValidationError("unsupported_decode_contract")
    return {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "image_ids": tuple(sorted(image_ids, key=lambda value: int(value) if value.isdigit() else value)),
        "inference_contract": dict(inference),
        "discovery_budget": dict(budget),
        "primary_checkpoint": dict(manifest.get("primary_checkpoint", {})),
    }


def _validate_shard_identity(
    shard: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
) -> None:
    if shard.get("schema_version") != DISCOVERY_SCHEMA_VERSION:
        raise FreezeValidationError("schema_version_mismatch")
    if shard.get("phase") != "discovery":
        raise FreezeValidationError("phase_mismatch")
    if shard.get("manifest_sha256") != contract["manifest_sha256"]:
        raise FreezeValidationError("manifest_sha256_mismatch")
    config = shard.get("config")
    if not isinstance(config, Mapping):
        raise FreezeValidationError("shard_config_missing")
    inference = contract["inference_contract"]
    budget = contract["discovery_budget"]
    expected = {
        "infer_config": str(inference.get("config_path")),
        "temperature": float(inference.get("temperature")),
        "top_p": float(inference.get("top_p")),
        "repetition_penalty": 1.0,
        "max_new_tokens": int(budget.get("maximum_new_tokens_per_row")),
        "horizon_rows": int(budget.get("maximum_complete_rows_per_trajectory")),
        "seeds": list(FROZEN_DISCOVERY_SEEDS),
        "physical_batch_size": 1,
        "model_dtype": "fp32",
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise FreezeValidationError(f"shard_decode_contract_mismatch:{key}")
    if str(config.get("manifest")) != contract["manifest_path"]:
        raise FreezeValidationError("shard_manifest_path_mismatch")
    if not isinstance(shard.get("frozen_file_identity"), Mapping):
        raise FreezeValidationError("frozen_file_identity_missing")
    if not isinstance(shard.get("source_identity"), Mapping):
        raise FreezeValidationError("source_identity_missing")


def resolve_shard_paths(
    shard_paths: Sequence[Path], shard_globs: Sequence[str]
) -> list[Path]:
    resolved: dict[str, Path] = {}
    for path in shard_paths:
        value = path.expanduser().resolve()
        if not value.is_file():
            raise FreezeValidationError(f"discovery shard is not a file: {value}")
        resolved[str(value)] = value
    for pattern in shard_globs:
        matches = [Path(item).expanduser().resolve() for item in glob.glob(str(Path(pattern).expanduser()))]
        if not matches:
            raise FreezeValidationError(f"discovery shard glob matched no files: {pattern}")
        for value in matches:
            if not value.is_file():
                raise FreezeValidationError(f"discovery glob result is not a file: {value}")
            resolved[str(value)] = value
    if not resolved:
        raise FreezeValidationError("at least one --shard or --shard-glob is required")
    return [resolved[key] for key in sorted(resolved)]


def freeze_from_shards(
    *,
    manifest_path: Path,
    shard_paths: Sequence[Path],
) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve(strict=True)
    manifest = load_json(manifest_path)
    contract = _contract_from_manifest(manifest, manifest_path)
    expected_images = set(contract["image_ids"])
    shard_receipts: list[dict[str, Any]] = []
    merged_images: dict[str, Mapping[str, Any]] = {}
    frozen_file_identity: Mapping[str, Any] | None = None
    source_identity: Mapping[str, Any] | None = None
    for path in shard_paths:
        shard = load_json(path)
        _validate_shard_identity(shard, contract=contract)
        if frozen_file_identity is None:
            frozen_file_identity = shard["frozen_file_identity"]
        elif shard["frozen_file_identity"] != frozen_file_identity:
            raise FreezeValidationError("frozen_identity_mismatch")
        current_source = shard["source_identity"]
        if source_identity is None:
            source_identity = current_source
        else:
            stable_keys = ("commit", "branch", "runner_sha256")
            if any(current_source.get(key) != source_identity.get(key) for key in stable_keys):
                raise FreezeValidationError("source_identity_mismatch")
        images = shard.get("images")
        if not isinstance(images, list) or not images:
            raise FreezeValidationError("shard_images_missing")
        shard_image_ids: list[str] = []
        for image in images:
            if not isinstance(image, Mapping):
                raise FreezeValidationError("shard_image_record_invalid")
            image_id = _as_image_id(image.get("image_id"))
            if image_id not in expected_images:
                raise FreezeValidationError("shard_contains_image_outside_frozen_pool")
            if image_id in merged_images:
                raise FreezeValidationError("shard_duplicate_image")
            merged_images[image_id] = image
            shard_image_ids.append(image_id)
        selected_ids = {_as_image_id(value) for value in config_selected_image_ids(shard)}
        if selected_ids != set(shard_image_ids):
            raise FreezeValidationError("shard_selected_image_ids_mismatch")
        shard_receipts.append({
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "image_ids": sorted(shard_image_ids, key=lambda value: int(value) if value.isdigit() else value),
        })
    if set(merged_images) != expected_images:
        missing = sorted(expected_images - set(merged_images))
        raise FreezeValidationError(f"shard_missing_image:{','.join(missing)}")
    all_candidates: list[dict[str, Any]] = []
    refusal_counts: Counter[str] = Counter()
    outcome_counts: Counter[str] = Counter()
    image_summaries: list[dict[str, Any]] = []
    for image_id in sorted(merged_images, key=lambda value: int(value) if value.isdigit() else value):
        image = merged_images[image_id]
        ledger = image.get("entity_ledger")
        if not isinstance(ledger, list):
            raise FreezeValidationError("image_entity_ledger_missing")
        evaluations = image.get("prefix_evaluations")
        if not isinstance(evaluations, list):
            raise FreezeValidationError("image_prefix_evaluations_missing")
        candidates: list[dict[str, Any]] = []
        refusals: list[dict[str, Any]] = []
        seen_prefixes: set[str] = set()
        for evaluation in evaluations:
            if not isinstance(evaluation, Mapping):
                refusals.append({"outcome": "unresolved_refusal", "refusal_code": "prefix_evaluation_invalid"})
                continue
            prefix = evaluation.get("prefix")
            if isinstance(prefix, Mapping):
                try:
                    prefix_hash = _prefix_hash(prefix)
                except FreezeValidationError as exc:
                    prefix_hash = f"invalid:{exc}"
                if prefix_hash in seen_prefixes:
                    raise FreezeValidationError("duplicate_exact_prefix_evaluation")
                seen_prefixes.add(prefix_hash)
            evaluation_with_image = dict(evaluation)
            evaluation_with_image["image_id"] = image_id
            found, rejected = derive_prefix_candidates(evaluation_with_image, entity_ledger=ledger)
            candidates.extend(found)
            refusals.extend(rejected)
        for rejection in refusals:
            outcome_counts[str(rejection.get("outcome", "unresolved_refusal"))] += 1
            refusal_code = rejection.get("refusal_code")
            if refusal_code:
                refusal_counts[str(refusal_code)] += 1
        selected = select_one_case_per_image(candidates)
        for candidate in candidates:
            all_candidates.append(candidate)
        if selected is not None:
            outcome_counts["admitted_object_vs_object"] += 1
        image_summaries.append({
            "image_id": image_id,
            "prefix_evaluation_count": len(evaluations),
            "eligible_pair_count": len(candidates),
            "selected_case_parent_prefix_sha256": None if selected is None else selected["parent_prefix_token_ids_sha256"],
            "native_root_greedy_reachable": None if selected is None else selected["native_root_greedy_reachable"],
            "refusal_counts": dict(sorted(Counter(str(item.get("refusal_code")) for item in refusals if item.get("refusal_code")).items())),
            "diagnostic_outcome_counts": dict(sorted(Counter(str(item.get("outcome", "unresolved_refusal")) for item in refusals).items())),
        })
    selected_cases: list[dict[str, Any]] = []
    by_image: dict[str, list[dict[str, Any]]] = {}
    for candidate in all_candidates:
        by_image.setdefault(str(candidate["image_id"]), []).append(candidate)
    for image_id in sorted(by_image, key=lambda value: int(value) if value.isdigit() else value):
        selected = select_one_case_per_image(by_image[image_id])
        if selected is not None:
            selected_cases.append(selected)
    selected_cases = sorted(selected_cases, key=lambda item: str(item["image_id"]))
    selected_hash = sha256_bytes(selected_cases)
    merged_shard_paths_hash = sha256_bytes(shard_receipts)
    minimum_required_images = 4
    promotion_status = (
        "ready_for_causal_intervention"
        if len(selected_cases) >= minimum_required_images
        else "insufficient_cohort"
    )
    return {
        "schema_version": FROZEN_CASE_SCHEMA_VERSION,
        "experiment": "local_branch_causal_value",
        "selection_phase": "discovery_only",
        "selection_rule": ["lowest natural row index", "lowest discovery seed", "lexicographically smallest exact-prefix SHA-256"],
        "annotation_policy": "positive-only ledger; annotation absence is never negative evidence",
        "frozen_pool_manifest": str(manifest_path),
        "frozen_pool_manifest_sha256": contract["manifest_sha256"],
        "source_discovery_schema_version": DISCOVERY_SCHEMA_VERSION,
        "input_shards": shard_receipts,
        "input_shards_sha256": merged_shard_paths_hash,
        "frozen_file_identity": frozen_file_identity,
        "source_identity": source_identity,
        "frozen_image_ids": list(contract["image_ids"]),
        "discovery_contract": {
            "backend": contract["inference_contract"].get("backend"),
            "floating_point": contract["inference_contract"].get("floating_point"),
            "physical_batch_size": contract["inference_contract"].get("physical_batch_size"),
            "repetition_penalty": contract["inference_contract"].get("repetition_penalty"),
            "temperature": contract["inference_contract"].get("temperature"),
            "top_p": contract["inference_contract"].get("top_p"),
            "sampled_root_seeds": list(FROZEN_DISCOVERY_SEEDS),
            "maximum_complete_rows_per_trajectory": contract["discovery_budget"].get("maximum_complete_rows_per_trajectory"),
            "additional_same_prefix_sampling": False,
        },
        "image_summaries": image_summaries,
        "diagnostic_counts": {
            "outcomes": dict(sorted(outcome_counts.items())),
            "refusals": dict(sorted(refusal_counts.items())),
        },
        "eligible_pair_count": len(all_candidates),
        "minimum_required_images": minimum_required_images,
        "promotion_status": promotion_status,
        "selected_case_count": len(selected_cases),
        "selected_cases_sha256": selected_hash,
        "selected_cases": selected_cases,
    }


def config_selected_image_ids(shard: Mapping[str, Any]) -> list[str]:
    config = shard.get("config")
    if not isinstance(config, Mapping) or not isinstance(config.get("selected_image_ids"), list):
        raise FreezeValidationError("shard_selected_image_ids_missing")
    return [_as_image_id(value) for value in config["selected_image_ids"]]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Frozen discovery-pool manifest")
    parser.add_argument("--shard", action="append", type=Path, default=[], help="Discovery shard JSON; repeatable")
    parser.add_argument("--shard-glob", action="append", default=[], help="Glob for discovery shards; repeatable")
    parser.add_argument("--output", type=Path, required=True, help="Immutable admitted-cases JSON")
    parser.add_argument("--force", action="store_true", help="Allow replacing an existing output")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output = args.output.expanduser().resolve()
    if output.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite {output}; pass --force")
    try:
        paths = resolve_shard_paths(args.shard, args.shard_glob)
        payload = freeze_from_shards(manifest_path=args.manifest, shard_paths=paths)
    except FreezeValidationError as exc:
        raise SystemExit(f"freeze validation failed: {exc}") from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_bytes(payload) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
