#!/usr/bin/env python3
"""Sample one exact complete row from an explicitly frozen prefix.

The runner is a narrow source builder. It accepts a legacy Stage 6 arm, a
local branch prefix-evaluation artifact, or one exact row from a Stage-1 root
trace. It re-materializes the frozen image through the ordinary inference
frontend, proves exact greedy parity at the selected prefix, and only then
draws one row for each declared seed. Prefix and generated rows stay as token
identifiers throughout; decoded text is retained only as parser evidence from
the existing row helper.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from scripts.research.run_local_branch_causal_value import (  # noqa: E402
    _annotate_owner_matches,
    _generate_row,
    _single_native_inputs,
    build_positive_entity_ledger,
    git_execution_identity,
    hash_prefix_token_ids,
    select_verified_uncovered_owners,
    sha256_file,
)
from scripts.research.run_row_four_coordinate_factorial import (  # noqa: E402
    PHASE as STAGE6_PHASE,
    SCHEMA_VERSION as STAGE6_SCHEMA_VERSION,
    compare_execution_identity,
    compare_stable_model_identity,
)


SCHEMA_VERSION = "exact_prefix_sampled_rescue.v1"
ROOT_TRACE_SCHEMA_VERSION = "sampled_history_target_reachability.v1"
ROOT_TRACE_PHASE = "stage_one_extended_root_greedy_screen"
TEMPERATURE = 0.4
TOP_P = 0.95
REPETITION_PENALTY = 1.0
MALFORMED_LIMIT = 2
STATIC_IDENTITY_LABELS = (
    "infer_config",
    "checkpoint_json",
    "adapter_model",
    "special_token_embeddings",
)
BLIND_IMAGE_IDS = frozenset(
    {
        "1584",
        "2685",
        "4134",
        "5001",
        "6040",
        "7511",
        "10707",
        "13348",
        "13923",
        "14038",
        "14439",
        "16228",
    }
)


class RescueValidationError(ValueError):
    """Raised when the frozen source cannot support exact-prefix sampling."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RescueValidationError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RescueValidationError(f"JSON artifact must be an object: {path}")
    return value


def _ids(value: Any, *, label: str, allow_empty: bool = False) -> list[int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RescueValidationError(f"{label} must be a token-id sequence")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise RescueValidationError(f"{label} contains an invalid token id")
        result.append(int(item))
    if not allow_empty and not result:
        raise RescueValidationError(f"{label} must not be empty")
    return result


def parse_seeds(value: str) -> tuple[int, ...]:
    """Parse a non-empty, ordered, duplicate-free comma-separated seed list."""

    pieces = [piece.strip() for piece in str(value).split(",")]
    if not pieces or any(not piece for piece in pieces):
        raise RescueValidationError(
            "--seeds requires non-empty comma-separated integers"
        )
    try:
        seeds = tuple(int(piece) for piece in pieces)
    except ValueError as exc:
        raise RescueValidationError("--seeds contains a non-integer value") from exc
    if any(seed < 0 for seed in seeds):
        raise RescueValidationError("--seeds must contain non-negative integers")
    if len(set(seeds)) != len(seeds):
        raise RescueValidationError("--seeds must not contain duplicates")
    return seeds


def parse_seed_range(start: int | str, end: int | str) -> tuple[int, ...]:
    """Return every non-negative seed in the inclusive ``start..end`` range."""

    try:
        first, last = int(start), int(end)
    except (TypeError, ValueError) as exc:
        raise RescueValidationError(
            "seed range bounds must be integer values"
        ) from exc
    if first < 0 or last < 0:
        raise RescueValidationError("seed range bounds must be non-negative")
    if last < first:
        raise RescueValidationError("seed range end must be >= start")
    return tuple(range(first, last + 1))


def parse_seed_range_spec(value: str) -> tuple[int, ...]:
    """Parse an inclusive ``START-END`` (or ``START:END``) seed range."""

    text = str(value).strip()
    for delimiter in ("-", ":"):
        if delimiter in text:
            pieces = [piece.strip() for piece in text.split(delimiter)]
            if len(pieces) != 2 or any(not piece for piece in pieces):
                break
            return parse_seed_range(pieces[0], pieces[1])
    raise RescueValidationError(
        "--seed-range requires an inclusive START-END or START:END range"
    )


def resolve_seeds(
    *,
    seeds: str | None = None,
    seed_start: int | str | None = None,
    seed_end: int | str | None = None,
    seed_range: str | None = None,
) -> tuple[int, ...]:
    """Resolve exactly one explicit seed-list or seed-range specification."""

    supplied = int(seeds is not None) + int(seed_range is not None) + int(
        seed_start is not None or seed_end is not None
    )
    if supplied != 1:
        raise RescueValidationError(
            "provide exactly one of --seeds, --seed-range, or --seed-start/--seed-end"
        )
    if seeds is not None:
        return parse_seeds(seeds)
    if seed_range is not None:
        return parse_seed_range_spec(seed_range)
    if seed_start is None or seed_end is None:
        raise RescueValidationError(
            "--seed-start and --seed-end must be provided together"
        )
    return parse_seed_range(seed_start, seed_end)


def resolve_row_token_budget(config: Mapping[str, Any]) -> int:
    """Resolve the positive row-generation budget across frozen source schemas."""

    for key in (
        "post_prefix_generated_token_budget",
        "max_new_tokens",
        "total_generated_token_budget",
    ):
        value = config.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    raise RescueValidationError(
        "frozen source config lacks a positive row-generation token budget"
    )


def validate_prefix_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and copy an explicitly frozen prefix record.

    The accepted record is intentionally small and JSON-native.  It may put
    the prefix fields at the root or under a ``prefix`` object, which keeps
    this helper usable with both standalone records and larger run artifacts.
    """

    if not isinstance(record, Mapping):
        raise RescueValidationError("frozen prefix record must be an object")
    nested = record.get("prefix")
    source = nested if isinstance(nested, Mapping) else record
    raw_ids = source.get("token_ids", source.get("prefix_token_ids"))
    prefix_ids = _ids(raw_ids, label="frozen prefix record")
    declared_hash = source.get(
        "token_ids_sha256", source.get("prefix_token_ids_sha256", "")
    )
    observed_hash = hash_prefix_token_ids(prefix_ids)
    if observed_hash != str(declared_hash):
        raise RescueValidationError("frozen prefix record token hash mismatch")
    raw_owners = source.get(
        "covered_owner_ids",
        source.get("prefix_owner_ids", source.get("covered_entity_ids", [])),
    )
    if isinstance(raw_owners, (str, bytes)) or not isinstance(raw_owners, Sequence):
        raise RescueValidationError("frozen prefix record owners must be a sequence")
    owner_ids = sorted({str(value) for value in raw_owners})
    if not owner_ids or any(not value for value in owner_ids):
        raise RescueValidationError("frozen prefix record lacks covered owners")
    frozen = source.get("frozen_greedy_row")
    if frozen is None and isinstance(source.get("greedy"), Mapping):
        frozen = source["greedy"].get("row")
    frozen_copy: dict[str, Any] | None = None
    if frozen is not None:
        if not isinstance(frozen, Mapping):
            raise RescueValidationError("frozen prefix record greedy row must be an object")
        frozen_copy = deepcopy(dict(frozen))
        frozen_ids = _ids(
            frozen_copy.get("raw_generated_token_ids"),
            label="frozen prefix record greedy row",
        )
        if hash_prefix_token_ids(frozen_ids) != str(
            frozen_copy.get("raw_generated_token_ids_sha256", "")
        ):
            raise RescueValidationError(
                "frozen prefix record greedy row token hash mismatch"
            )
    return {
        "prefix_token_ids": list(prefix_ids),
        "prefix_token_ids_sha256": observed_hash,
        "covered_owner_ids": owner_ids,
        "frozen_greedy_row": frozen_copy,
        "image_id": str(record.get("image_id", source.get("image_id", "7816"))),
        "record": deepcopy(dict(record)),
    }


def select_local_prefix_evaluation(
    artifact: Mapping[str, Any], *, prefix_hash: str
) -> dict[str, Any]:
    """Select one exact prefix evaluation from a local branch artifact."""

    images = artifact.get("images")
    if not isinstance(images, list):
        raise RescueValidationError("prefix artifact lacks an images list")
    matches: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for image in images:
        if not isinstance(image, Mapping):
            continue
        evaluations = image.get("prefix_evaluations")
        if not isinstance(evaluations, list):
            continue
        for evaluation in evaluations:
            if not isinstance(evaluation, Mapping):
                continue
            prefix = evaluation.get("prefix")
            if isinstance(prefix, Mapping) and str(
                prefix.get("prefix_token_ids_sha256", "")
            ) == str(prefix_hash):
                matches.append((image, evaluation))
    if not matches:
        raise RescueValidationError(f"prefix hash not found in artifact: {prefix_hash}")
    if len(matches) != 1:
        raise RescueValidationError(
            f"prefix hash is not unique in artifact: {prefix_hash} ({len(matches)} matches)"
        )
    image, evaluation = matches[0]
    prefix = evaluation.get("prefix")
    if not isinstance(prefix, Mapping):
        raise RescueValidationError("selected prefix evaluation lacks prefix")
    selected = validate_prefix_record(
        {
            "image_id": image.get("image_id"),
            "prefix_token_ids": prefix.get("prefix_token_ids"),
            "prefix_token_ids_sha256": prefix.get("prefix_token_ids_sha256"),
            "covered_entity_ids": prefix.get("covered_entity_ids"),
            "trajectory_provenance": prefix.get("trajectory_provenance"),
        }
    )
    native = evaluation.get("native_greedy_row")
    if not isinstance(native, Mapping):
        raise RescueValidationError("selected prefix evaluation lacks native_greedy_row")
    native_copy = deepcopy(dict(native))
    native_ids = _ids(native_copy.get("raw_generated_token_ids"), label="native greedy row")
    if hash_prefix_token_ids(native_ids) != str(
        native_copy.get("raw_generated_token_ids_sha256", "")
    ):
        raise RescueValidationError("native greedy row token hash mismatch")
    native_prefix = _ids(native_copy.get("prefix_token_ids"), label="native greedy prefix")
    if native_prefix != selected["prefix_token_ids"] or str(
        native_copy.get("prefix_token_ids_sha256", "")
    ) != selected["prefix_token_ids_sha256"]:
        raise RescueValidationError("native greedy row prefix disagrees with selected prefix")
    selected["frozen_greedy_row"] = native_copy
    selected["trajectory_provenance"] = deepcopy(prefix.get("trajectory_provenance"))
    selected["classification"] = deepcopy(evaluation.get("classification"))
    selected["prompt"] = deepcopy(image.get("prompt"))
    selected["runtime"] = deepcopy(image.get("runtime"))
    return selected


def _verified_ledger_owner_ids(entity_ledger: Sequence[Mapping[str, Any]]) -> set[str]:
    """Return owner identifiers explicitly marked as verified in a ledger."""

    return {
        str(item.get("entity_id"))
        for item in entity_ledger
        if item.get("entity_id") is not None
        and str(item.get("verification", item.get("status", ""))).lower()
        in {"verified", "approved", "human_verified"}
    }


def select_root_trace_row(
    artifact: Mapping[str, Any],
    *,
    row_index: int,
    harmful_kind: str,
) -> dict[str, Any]:
    """Select one exact row from a Stage-1 root-greedy trace.

    The selected input prefix is copied from ``input_prefix_token_ids`` rather
    than reconstructed from decoded text.  Both the input-prefix and the
    legacy ``prefix_token_ids`` fields must agree byte-for-byte.  This keeps
    root-trace replay on the same exact-token contract as the existing arms.
    """

    if artifact.get("schema_version") != ROOT_TRACE_SCHEMA_VERSION:
        raise RescueValidationError("root trace schema mismatch")
    if artifact.get("phase") != ROOT_TRACE_PHASE:
        raise RescueValidationError("root trace phase mismatch")
    if harmful_kind not in {"duplicate", "premature_terminal"}:
        raise RescueValidationError(f"unsupported root harmful kind {harmful_kind!r}")
    if isinstance(row_index, bool) or not isinstance(row_index, int) or row_index < 0:
        raise RescueValidationError("root trace row index must be a non-negative integer")
    images = artifact.get("images")
    if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], Mapping):
        raise RescueValidationError("root trace must contain exactly one image")
    image = images[0]
    image_id = str(image.get("image_id", ""))
    if not image_id:
        raise RescueValidationError("root trace image lacks image_id")
    if image_id in BLIND_IMAGE_IDS:
        raise RescueValidationError(f"blind image {image_id} is not allowed")
    extended = image.get("extended_root_greedy")
    if not isinstance(extended, Mapping) or not isinstance(extended.get("rows"), list):
        raise RescueValidationError("root trace lacks extended_root_greedy.rows")
    matches = [
        row
        for row in extended["rows"]
        if isinstance(row, Mapping) and row.get("row_index") == row_index
    ]
    if len(matches) != 1:
        raise RescueValidationError(
            f"root trace row index {row_index} is not unique ({len(matches)} matches)"
        )
    row = deepcopy(dict(matches[0]))
    input_ids = _ids(
        row.get("input_prefix_token_ids"),
        label="root trace input prefix",
        allow_empty=True,
    )
    input_hash = hash_prefix_token_ids(input_ids)
    if input_hash != str(row.get("input_prefix_token_ids_sha256", "")):
        raise RescueValidationError("root trace input prefix token hash mismatch")
    prefix_ids = _ids(
        row.get("prefix_token_ids"),
        label="root trace prefix",
        allow_empty=True,
    )
    prefix_hash = hash_prefix_token_ids(prefix_ids)
    if prefix_hash != str(row.get("prefix_token_ids_sha256", "")):
        raise RescueValidationError("root trace prefix token hash mismatch")
    if input_ids != prefix_ids or input_hash != prefix_hash:
        raise RescueValidationError(
            "root trace input prefix disagrees with frozen row prefix"
        )
    generated_ids = _ids(
        row.get("raw_generated_token_ids"), label="root trace frozen greedy row"
    )
    generated_hash = hash_prefix_token_ids(generated_ids)
    if generated_hash != str(row.get("raw_generated_token_ids_sha256", "")):
        raise RescueValidationError("root trace frozen greedy row token hash mismatch")
    raw_covered = row.get("covered_owner_ids_before_row", [])
    if isinstance(raw_covered, (str, bytes)) or not isinstance(raw_covered, Sequence):
        raise RescueValidationError("root trace covered-owner set must be a sequence")
    covered_owner_ids = sorted({str(value) for value in raw_covered})
    if any(not value for value in covered_owner_ids):
        raise RescueValidationError("root trace covered-owner set contains an empty id")
    ledger = image.get("entity_ledger")
    if not isinstance(ledger, list) or not ledger:
        raise RescueValidationError("root trace lacks a non-empty entity ledger")
    ledger_copy = [deepcopy(dict(item)) for item in ledger if isinstance(item, Mapping)]
    if len(ledger_copy) != len(ledger):
        raise RescueValidationError("root trace entity ledger contains a non-object")
    if not _verified_ledger_owner_ids(ledger_copy):
        raise RescueValidationError("root trace entity ledger has no verified owner")
    if harmful_kind == "duplicate":
        # Do this before model loading so an accidentally selected normal row
        # cannot consume a GPU run.  Replay performs the same exact checks.
        validate_greedy_duplicate(
            row, frozen=row, covered_owner_ids=covered_owner_ids
        )
    else:
        validate_greedy_terminal(
            row,
            frozen=row,
            covered_owner_ids=covered_owner_ids,
            entity_ledger=ledger_copy,
        )
    return {
        "mode": "root_trace",
        "harmful_kind": harmful_kind,
        "row_index": int(row_index),
        "image_id": image_id,
        "prefix_token_ids": input_ids,
        "prefix_token_ids_sha256": input_hash,
        "covered_owner_ids": covered_owner_ids,
        "frozen_greedy_row": row,
        "entity_ledger": ledger_copy,
        "prompt": deepcopy(image.get("prompt")),
        "runtime": deepcopy(image.get("runtime")),
        "root_trace_schema_version": artifact.get("schema_version"),
        "root_trace_phase": artifact.get("phase"),
    }


def validate_stage6_arm(stage6: Mapping[str, Any], *, arm_name: str) -> dict[str, Any]:
    """Extract one arm and its frozen first greedy row after strict checks."""

    if (
        stage6.get("schema_version") != STAGE6_SCHEMA_VERSION
        or stage6.get("phase") != STAGE6_PHASE
    ):
        raise RescueValidationError("Stage 6 artifact schema or phase mismatch")
    config = stage6.get("config")
    if not isinstance(config, Mapping) or config.get("model_dtype") != "fp32":
        raise RescueValidationError("Stage 6 source must be the fp32 artifact")
    arms = stage6.get("arms")
    if not isinstance(arms, Mapping) or arm_name not in arms:
        raise RescueValidationError(f"unknown Stage 6 arm {arm_name!r}")
    raw_arm = arms[arm_name]
    if not isinstance(raw_arm, Mapping):
        raise RescueValidationError(f"Stage 6 arm {arm_name!r} is not an object")
    prefix_ids = _ids(raw_arm.get("prefix_token_ids"), label="Stage 6 prefix")
    prefix_hash = hash_prefix_token_ids(prefix_ids)
    if prefix_hash != str(raw_arm.get("prefix_token_ids_sha256", "")):
        raise RescueValidationError("Stage 6 prefix token hash mismatch")
    covered = sorted({str(value) for value in raw_arm.get("prefix_owner_ids", [])})
    if not covered or any(not value for value in covered):
        raise RescueValidationError("Stage 6 arm lacks a valid covered-owner set")
    continuation = raw_arm.get("continuation")
    rows = continuation.get("rows") if isinstance(continuation, Mapping) else None
    if not isinstance(rows, list) or not rows or not isinstance(rows[0], Mapping):
        raise RescueValidationError("Stage 6 arm lacks its frozen first greedy row")
    frozen_row = dict(rows[0])
    frozen_ids = _ids(
        frozen_row.get("raw_generated_token_ids"), label="frozen greedy row"
    )
    if hash_prefix_token_ids(frozen_ids) != str(
        frozen_row.get("raw_generated_token_ids_sha256", "")
    ):
        raise RescueValidationError("frozen greedy row token hash mismatch")
    return {
        "arm_name": arm_name,
        "prefix_token_ids": prefix_ids,
        "prefix_token_ids_sha256": prefix_hash,
        "covered_owner_ids": covered,
        "frozen_greedy_row": frozen_row,
    }


def validate_static_identity(
    stage6: Mapping[str, Any], *, infer_config_path: Path
) -> dict[str, Any]:
    """Rehash immutable model/config files, intentionally excluding source JSONL."""

    frozen = stage6.get("frozen_file_identity")
    if not isinstance(frozen, Mapping):
        frozen_inputs = stage6.get("frozen_inputs")
        if isinstance(frozen_inputs, Mapping):
            frozen = frozen_inputs.get("checkpoint_config_source_identity")
    if not isinstance(frozen, Mapping):
        raise RescueValidationError("Stage 6 artifact lacks frozen file identity")
    receipt: dict[str, Any] = {}
    for label in STATIC_IDENTITY_LABELS:
        expected = frozen.get(label)
        if not isinstance(expected, Mapping):
            raise RescueValidationError(f"Stage 6 artifact lacks {label} identity")
        expected_path = (
            Path(str(expected.get("path", ""))).expanduser().resolve(strict=True)
        )
        observed_path = (
            infer_config_path.expanduser().resolve(strict=True)
            if label == "infer_config"
            else expected_path
        )
        if observed_path != expected_path:
            raise RescueValidationError(
                f"{label} path drift: observed {observed_path}, frozen {expected_path}"
            )
        observed_sha = sha256_file(observed_path)
        if observed_sha != str(expected.get("sha256", "")):
            raise RescueValidationError(f"{label} SHA-256 drift")
        receipt[label] = {"path": str(observed_path), "sha256": observed_sha}
    source = frozen.get("source_jsonl")
    receipt["source_jsonl_policy"] = {
        "status": "selected_identity_only",
        "reason": "known_full_source_jsonl_hash_drift",
        "frozen_path": source.get("path") if isinstance(source, Mapping) else None,
        "frozen_sha256_not_enforced": (
            source.get("sha256") if isinstance(source, Mapping) else None
        ),
    }
    return receipt


def validate_greedy_duplicate(
    observed: Mapping[str, Any],
    *,
    frozen: Mapping[str, Any],
    covered_owner_ids: Sequence[str],
) -> dict[str, Any]:
    """Require exact frozen-row parity and one strict already-covered owner."""

    observed_ids = _ids(
        observed.get("raw_generated_token_ids"), label="observed greedy row"
    )
    frozen_ids = _ids(frozen.get("raw_generated_token_ids"), label="frozen greedy row")
    observed_stop = observed.get("row_stop")
    frozen_stop = frozen.get("row_stop")
    checks = {
        "token_ids_equal": observed_ids == frozen_ids,
        "observed_token_hash_valid": hash_prefix_token_ids(observed_ids)
        == str(observed.get("raw_generated_token_ids_sha256", "")),
        "frozen_token_hash_valid": hash_prefix_token_ids(frozen_ids)
        == str(frozen.get("raw_generated_token_ids_sha256", "")),
        "token_hash_equal": hash_prefix_token_ids(observed_ids)
        == hash_prefix_token_ids(frozen_ids),
        "status_equal": observed.get("status") == frozen.get("status") == "success",
        "complete_row_equal": (
            isinstance(observed_stop, Mapping)
            and isinstance(frozen_stop, Mapping)
            and observed_stop.get("stop_reason")
            == frozen_stop.get("stop_reason")
            == "complete_row"
        ),
    }
    observed_owners = sorted(
        {str(value) for value in observed.get("strict_matched_owner_ids", [])}
    )
    frozen_owners = sorted(
        {str(value) for value in frozen.get("strict_matched_owner_ids", [])}
    )
    unresolved = list(observed.get("unmatched_or_ambiguous_prediction_indices", []))
    covered = {str(value) for value in covered_owner_ids}
    checks.update(
        {
            "strict_owner_equal": observed_owners == frozen_owners,
            "exactly_one_strict_owner": len(observed_owners) == 1,
            "owner_is_covered": len(observed_owners) == 1
            and observed_owners[0] in covered,
            "no_unresolved_prediction": not unresolved,
        }
    )
    if not all(checks.values()):
        raise RescueValidationError(
            f"greedy row is not the exact strict covered duplicate: {checks}"
        )
    return {
        "passed": True,
        "checks": checks,
        "duplicate_owner_id": observed_owners[0],
        "raw_generated_token_ids_sha256": hash_prefix_token_ids(observed_ids),
    }


def validate_greedy_terminal(
    observed: Mapping[str, Any],
    *,
    frozen: Mapping[str, Any],
    covered_owner_ids: Sequence[str],
    entity_ledger: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Require an exact terminal replay with verified owners still uncovered."""

    observed_ids = _ids(
        observed.get("raw_generated_token_ids"), label="observed terminal row"
    )
    frozen_ids = _ids(
        frozen.get("raw_generated_token_ids"), label="frozen terminal row"
    )
    observed_stop = observed.get("row_stop")
    frozen_stop = frozen.get("row_stop")
    observed_parsed = observed.get("parsed_predictions")
    frozen_parsed = frozen.get("parsed_predictions")
    observed_unresolved = list(
        observed.get("unmatched_or_ambiguous_prediction_indices", [])
    )
    frozen_unresolved = list(
        frozen.get("unmatched_or_ambiguous_prediction_indices", [])
    )
    observed_owners = sorted(
        {str(value) for value in observed.get("strict_matched_owner_ids", [])}
    )
    frozen_owners = sorted(
        {str(value) for value in frozen.get("strict_matched_owner_ids", [])}
    )
    verified = _verified_ledger_owner_ids(entity_ledger)
    covered = {str(value) for value in covered_owner_ids}
    verified_uncovered = sorted(verified - covered)
    checks = {
        "token_ids_equal": observed_ids == frozen_ids,
        "observed_token_hash_valid": hash_prefix_token_ids(observed_ids)
        == str(observed.get("raw_generated_token_ids_sha256", "")),
        "frozen_token_hash_valid": hash_prefix_token_ids(frozen_ids)
        == str(frozen.get("raw_generated_token_ids_sha256", "")),
        "token_hash_equal": hash_prefix_token_ids(observed_ids)
        == hash_prefix_token_ids(frozen_ids),
        "status_equal": observed.get("status") == frozen.get("status") == "success",
        "terminal_stop_equal": (
            isinstance(observed_stop, Mapping)
            and isinstance(frozen_stop, Mapping)
            and observed_stop.get("stop_reason")
            == frozen_stop.get("stop_reason")
            == "terminal"
        ),
        "null_strict_owner": not observed_owners and not frozen_owners,
        "parsed_predictions_empty": (
            isinstance(observed_parsed, list)
            and isinstance(frozen_parsed, list)
            and not observed_parsed
            and not frozen_parsed
        ),
        "no_unresolved_prediction": not observed_unresolved and not frozen_unresolved,
        "verified_uncovered_owner_exists": bool(verified_uncovered),
    }
    if not all(checks.values()):
        raise RescueValidationError(
            f"greedy row is not an exact premature terminal: {checks}"
        )
    return {
        "passed": True,
        "checks": checks,
        "verified_uncovered_owner_ids": verified_uncovered,
        "raw_generated_token_ids_sha256": hash_prefix_token_ids(observed_ids),
    }


def build_producer_policy(
    *,
    mode: str,
    seed: int | None,
    checkpoint_id: str,
    prompt_token_ids_sha256: str,
    prefix_token_ids_sha256: str,
) -> dict[str, Any]:
    """Build the fixed typed producer policy persisted beside every row."""

    if mode not in {"greedy", "sampled"}:
        raise RescueValidationError(f"unsupported producer mode {mode!r}")
    if mode == "sampled" and seed is None:
        raise RescueValidationError("sampled producer policy requires a seed")
    if mode == "greedy" and seed is not None:
        raise RescueValidationError("greedy producer policy must not carry a seed")
    return {
        "mode": mode,
        "seed": seed,
        "temperature": 0.0 if mode == "greedy" else TEMPERATURE,
        "top_p": 1.0 if mode == "greedy" else TOP_P,
        "repetition_penalty": REPETITION_PENALTY,
        "checkpoint_id": str(checkpoint_id),
        "prompt_token_ids_sha256": str(prompt_token_ids_sha256),
        "prefix_token_ids_sha256": str(prefix_token_ids_sha256),
    }


def build_sample_record(
    row: Mapping[str, Any],
    *,
    seed: int,
    producer_policy: Mapping[str, Any],
    prompt_token_ids: Sequence[int],
    prefix_token_ids: Sequence[int],
    covered_owner_ids: Sequence[str],
    uncovered_owner_ids: Sequence[str],
    entity_ledger: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Normalize one sampled attempt without retrying or dropping its seed."""

    stop = row.get("row_stop")
    complete = bool(
        row.get("status") == "success"
        and isinstance(stop, Mapping)
        and stop.get("stop_reason") == "complete_row"
    )
    candidate_ids = _ids(
        row.get("raw_generated_token_ids"),
        label=f"seed {seed} candidate row",
        allow_empty=True,
    )
    candidate_hash = hash_prefix_token_ids(candidate_ids)
    if candidate_hash != str(row.get("raw_generated_token_ids_sha256", "")):
        raise RescueValidationError(f"seed {seed} candidate token hash mismatch")
    if row.get("mode") != "sample" or row.get("seed") != seed:
        raise RescueValidationError(f"seed {seed} producer metadata mismatch")
    prompt_hash = hash_prefix_token_ids(prompt_token_ids)
    prefix_hash = hash_prefix_token_ids(prefix_token_ids)
    expected_policy = {
        "mode": "sampled",
        "seed": seed,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "repetition_penalty": REPETITION_PENALTY,
        "prompt_token_ids_sha256": prompt_hash,
        "prefix_token_ids_sha256": prefix_hash,
    }
    policy_checks = {
        field: producer_policy.get(field) == expected
        for field, expected in expected_policy.items()
    }
    policy_checks["checkpoint_id"] = bool(producer_policy.get("checkpoint_id"))
    if not all(policy_checks.values()):
        raise RescueValidationError(
            f"seed {seed} typed producer policy mismatch: {policy_checks}"
        )
    covered = sorted({str(value) for value in covered_owner_ids})
    uncovered = sorted({str(value) for value in uncovered_owner_ids})
    verified_rescues = select_verified_uncovered_owners(
        row,
        covered_entity_ids=covered,
        entity_ledger=entity_ledger,
    )
    strict_owners = sorted(
        {str(value) for value in row.get("strict_matched_owner_ids", [])}
    )
    unresolved = list(row.get("unmatched_or_ambiguous_prediction_indices", []))
    selected = (
        complete
        and len(strict_owners) == 1
        and not unresolved
        and strict_owners == verified_rescues
        and strict_owners[0] in set(uncovered)
    )
    return {
        "seed": int(seed),
        "completion_status": "complete" if complete else "incomplete",
        "generation_status": row.get("status"),
        "failure": row.get("failure"),
        "producer_policy": dict(producer_policy),
        "prompt_token_ids": [int(value) for value in prompt_token_ids],
        "prompt_token_ids_sha256": prompt_hash,
        "prefix_token_ids": [int(value) for value in prefix_token_ids],
        "prefix_token_ids_sha256": prefix_hash,
        "candidate_token_ids": candidate_ids,
        "candidate_token_ids_sha256": candidate_hash,
        "row_stop": dict(stop) if isinstance(stop, Mapping) else {},
        "parse_evidence": row.get("parse_evidence"),
        "owner_matches": list(row.get("entity_matches", [])),
        "strict_matched_owner_ids": strict_owners,
        "covered_owner_ids": covered,
        "uncovered_owner_ids": uncovered,
        "verified_uncovered_owner_ids": verified_rescues,
        "selected_verified_uncovered_rescue": selected,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--stage6-artifact", type=Path)
    source_group.add_argument(
        "--prefix-artifact",
        "--prefix-record",
        dest="prefix_artifact",
        type=Path,
        help="local branch artifact containing prefix_evaluations",
    )
    source_group.add_argument(
        "--root-trace",
        type=Path,
        help="Stage-1 sampled_history_target_reachability.v1 root trace",
    )
    parser.add_argument(
        "--prefix-hash",
        "--prefix-token-ids-sha256",
        dest="prefix_hash",
        help="exact prefix_token_ids_sha256 for --prefix-artifact",
    )
    parser.add_argument("--infer-config", type=Path)
    parser.add_argument("--arm-name")
    parser.add_argument("--row-index", type=int)
    parser.add_argument(
        "--harmful-kind",
        choices=("duplicate", "premature_terminal"),
        help="root-trace validation arm",
    )
    seed_group = parser.add_mutually_exclusive_group(required=True)
    seed_group.add_argument(
        "--seeds",
        help="explicit ordered comma-separated seed list (for example 11,17,19)",
    )
    seed_group.add_argument(
        "--seed-range",
        help="inclusive seed range, written START-END or START:END",
    )
    seed_group.add_argument("--seed-start", "--seed-range-start", dest="seed_start", type=int)
    parser.add_argument("--seed-end", "--seed-range-end", dest="seed_end", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output = args.output.expanduser().resolve()
    if output.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite {output}; pass --force")
    try:
        seeds = resolve_seeds(
            seeds=args.seeds,
            seed_range=args.seed_range,
            seed_start=args.seed_start,
            seed_end=args.seed_end,
        )
        if args.stage6_artifact is None and (
            args.prefix_artifact is None or not args.prefix_hash
        ) and args.root_trace is None:
            raise RescueValidationError(
                "--prefix-artifact requires an exact --prefix-hash"
            )
        if args.root_trace is not None:
            if args.row_index is None or args.harmful_kind is None:
                raise RescueValidationError(
                    "--root-trace requires --row-index and --harmful-kind"
                )
        elif args.row_index is not None or args.harmful_kind is not None:
            raise RescueValidationError(
                "--row-index and --harmful-kind are only valid with --root-trace"
            )
        stage6_path: Path | None = None
        stage6: dict[str, Any] | None = None
        prefix_artifact_path: Path | None = None
        prefix_artifact: dict[str, Any] | None = None
        root_trace_path: Path | None = None
        root_trace: dict[str, Any] | None = None
        if args.stage6_artifact is not None:
            if args.infer_config is None or not args.arm_name:
                raise RescueValidationError(
                    "Stage 6 mode requires --infer-config and --arm-name"
                )
            stage6_path = args.stage6_artifact.expanduser().resolve(strict=True)
            stage6 = _load_json(stage6_path)
            arm = validate_stage6_arm(stage6, arm_name=str(args.arm_name))
        elif args.prefix_artifact is not None:
            prefix_artifact_path = args.prefix_artifact.expanduser().resolve(strict=True)
            prefix_artifact = _load_json(prefix_artifact_path)
            arm = select_local_prefix_evaluation(
                prefix_artifact, prefix_hash=str(args.prefix_hash)
            )
        else:
            root_trace_path = args.root_trace.expanduser().resolve(strict=True)
            root_trace = _load_json(root_trace_path)
            arm = select_root_trace_row(
                root_trace,
                row_index=int(args.row_index),
                harmful_kind=str(args.harmful_kind),
            )
        prefix_record: dict[str, Any] | None = None
        prefix_record_path: Path | None = None
        source_artifact = (
            stage6
            if stage6 is not None
            else prefix_artifact
            if prefix_artifact is not None
            else root_trace
        )
        assert source_artifact is not None
        infer_config_path = (
            args.infer_config.expanduser().resolve(strict=True)
            if args.infer_config is not None
            else Path(str(source_artifact["config"]["infer_config"])).expanduser().resolve(strict=True)
        )
        static_identity = validate_static_identity(
            source_artifact, infer_config_path=infer_config_path
        )
    except (OSError, RescueValidationError) as exc:
        raise SystemExit(
            f"frozen source validation failed before runtime: {exc}"
        ) from exc

    try:
        import torch
        from src.config.fingerprint import sha256_json
        from src.config.inference import load_infer_config
        from src.data import load_raw_examples
        from src.inference.backend import open_backend_session
        from src.inference.runtime import assemble_frontend
        from scripts.research.run_sampled_history_prefix_sufficiency_ladder import (
            _build_request,
            _select_example,
        )
    except Exception as exc:
        raise SystemExit(
            f"runtime import failed; no artifact was written: {type(exc).__name__}: {exc}"
        ) from exc

    resolved_config_path = infer_config_path
    resolved = load_infer_config(resolved_config_path)
    config = resolved.config
    frozen_config = source_artifact["config"]
    try:
        row_token_budget = resolve_row_token_budget(frozen_config)
    except RescueValidationError as exc:
        raise SystemExit(str(exc)) from exc
    if str(config.model.dtype) != "fp32":
        raise SystemExit("exact-prefix rescue source requires model.dtype=fp32")
    if resolved.fingerprint != str(
        frozen_config.get("resolved_config_fingerprint", "")
    ):
        raise SystemExit("resolved config fingerprint disagrees with frozen source")
    source_jsonl = Path(config.data.input_jsonl).expanduser().resolve(strict=True)
    examples = list(load_raw_examples(source_jsonl))
    frontend = assemble_frontend(
        config,
        generation_config_fingerprint=sha256_json(
            config.generation.model_dump(mode="json")
        ),
    )
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))

    frozen_execution = source_artifact.get("execution_identity_check")
    expected_execution = frozen_execution.get("expected") if isinstance(frozen_execution, Mapping) else None
    if prefix_artifact is not None:
        discovery_prompt = arm.get("prompt")
        discovery_runtime = arm.get("runtime")
        if not isinstance(discovery_prompt, Mapping) or not isinstance(discovery_runtime, Mapping):
            raise SystemExit("prefix artifact lacks frozen prompt/runtime identity")
    elif root_trace is not None:
        discovery_prompt = arm.get("prompt")
        discovery_runtime = arm.get("runtime")
        if not isinstance(discovery_prompt, Mapping) or not isinstance(discovery_runtime, Mapping):
            raise SystemExit("root trace lacks frozen prompt/runtime identity")
    else:
        if not isinstance(expected_execution, Mapping):
            raise SystemExit("frozen Stage 6 source lacks execution identity")
        discovery_prompt = expected_execution
        discovery_runtime = expected_execution
    image_id = str(arm.get("image_id", "7816"))
    example = _select_example(examples, image_id)
    request, plan, prompt_meta = _build_request(config, frontend, example)
    if root_trace is not None:
        frozen_ledger = arm.get("entity_ledger")
        if not isinstance(frozen_ledger, list):
            raise SystemExit("root trace lacks its frozen entity ledger")
        ledger = [dict(item) for item in frozen_ledger]
    else:
        ledger = build_positive_entity_ledger(example)
    ledger_owner_ids = {str(item["entity_id"]) for item in ledger}
    covered_owner_ids = arm["covered_owner_ids"]
    unknown_covered = sorted(set(covered_owner_ids) - ledger_owner_ids)
    if unknown_covered:
        raise SystemExit(f"frozen prefix references unknown owners: {unknown_covered}")
    uncovered_owner_ids = sorted(ledger_owner_ids - set(covered_owner_ids))
    if not uncovered_owner_ids:
        raise SystemExit("selected frozen prefix has no verified uncovered owners")

    checkpoint_id = static_identity["checkpoint_json"]["sha256"]
    with open_backend_session(frontend.launch) as session:
        model_receipt = session.receipt.to_artifact_dict()
        model_identity = compare_stable_model_identity(
            model_receipt, source_artifact.get("model_identity") or {}
        )
        if not model_identity["passed"]:
            raise SystemExit("checkpoint/model identity disagrees with frozen source")
        native_inputs, executed_ids, observed_grids, media_sha = (
            session._materialize_native_inputs((request,))
        )
        if len(executed_ids) != 1 or len(media_sha) != 1 or len(observed_grids) != 1:
            raise SystemExit("exact-prefix rescue requires one materialized image")
        prompt_token_ids = [int(value) for value in executed_ids[0]]
        prompt_hash = hash_prefix_token_ids(prompt_token_ids)
        execution_identity = compare_execution_identity(
            observed_prompt=prompt_meta,
            observed_runtime={
                "executed_media_sha256": media_sha[0],
                "executed_prompt_token_ids_sha256": prompt_hash,
                "observed_image_grid_thw": (
                    None
                    if observed_grids[0] is None
                    else [int(value) for value in observed_grids[0]]
                ),
            },
            discovery_prompt=discovery_prompt,
            discovery_runtime=discovery_runtime,
        )
        if not execution_identity["passed"]:
            raise SystemExit(
                "selected image/prompt/media identity disagrees with frozen source"
            )
        one_native = _single_native_inputs(native_inputs)
        greedy_row = _generate_row(
            session=session,
            native_inputs=one_native,
            prefix_token_ids=arm["prefix_token_ids"],
            tokenizer=session._tokenizer,
            image_width=int(plan.decoded_width),
            image_height=int(plan.decoded_height),
            mode="greedy",
            seed=None,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=REPETITION_PENALTY,
            max_new_tokens=row_token_budget,
            malformed_limit=MALFORMED_LIMIT,
            row_index=int(arm.get("row_index", 0)),
        )
        _annotate_owner_matches(
            greedy_row,
            entity_ledger=ledger,
            image_width=int(plan.decoded_width),
            image_height=int(plan.decoded_height),
            covered_entity_ids=covered_owner_ids,
        )
        if root_trace is not None and arm.get("harmful_kind") == "premature_terminal":
            greedy_parity = validate_greedy_terminal(
                greedy_row,
                frozen=arm["frozen_greedy_row"],
                covered_owner_ids=covered_owner_ids,
                entity_ledger=ledger,
            )
        else:
            greedy_parity = validate_greedy_duplicate(
                greedy_row,
                frozen=arm["frozen_greedy_row"],
                covered_owner_ids=covered_owner_ids,
            )
        samples: list[dict[str, Any]] = []
        for seed in seeds:
            row = _generate_row(
                session=session,
                native_inputs=one_native,
                prefix_token_ids=arm["prefix_token_ids"],
                tokenizer=session._tokenizer,
                image_width=int(plan.decoded_width),
                image_height=int(plan.decoded_height),
                mode="sample",
                seed=seed,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                repetition_penalty=REPETITION_PENALTY,
                max_new_tokens=row_token_budget,
                malformed_limit=MALFORMED_LIMIT,
                row_index=int(arm.get("row_index", 0)),
            )
            _annotate_owner_matches(
                row,
                entity_ledger=ledger,
                image_width=int(plan.decoded_width),
                image_height=int(plan.decoded_height),
                covered_entity_ids=covered_owner_ids,
            )
            policy = build_producer_policy(
                mode="sampled",
                seed=seed,
                checkpoint_id=checkpoint_id,
                prompt_token_ids_sha256=prompt_hash,
                prefix_token_ids_sha256=arm["prefix_token_ids_sha256"],
            )
            samples.append(
                build_sample_record(
                    row,
                    seed=seed,
                    producer_policy=policy,
                    prompt_token_ids=prompt_token_ids,
                    prefix_token_ids=arm["prefix_token_ids"],
                    covered_owner_ids=covered_owner_ids,
                    uncovered_owner_ids=uncovered_owner_ids,
                    entity_ledger=ledger,
                )
            )

    selected_rescues = [
        {
            "seed": row["seed"],
            "candidate_token_ids": row["candidate_token_ids"],
            "candidate_token_ids_sha256": row["candidate_token_ids_sha256"],
            "owner_ids": row["verified_uncovered_owner_ids"],
        }
        for row in samples
        if row["selected_verified_uncovered_rescue"]
    ]
    source_identity = git_execution_identity(Path(__file__).resolve().parents[2])
    source_identity["runner_sha256"] = sha256_file(Path(__file__).resolve())
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source_identity": source_identity,
        "frozen_source": {
            "mode": (
                "stage6"
                if stage6 is not None
                else "local_prefix_artifact"
                if prefix_artifact is not None
                else "root_trace"
            ),
            "path": str(
                stage6_path
                if stage6_path is not None
                else prefix_artifact_path
                if prefix_artifact_path is not None
                else root_trace_path
            ),
            "sha256": sha256_file(
                stage6_path
                if stage6_path is not None
                else prefix_artifact_path
                if prefix_artifact_path is not None
                else root_trace_path
            ),
            "arm_name": arm.get("arm_name"),
            "row_index": arm.get("row_index"),
            "harmful_kind": arm.get("harmful_kind"),
            "prefix_token_ids_sha256": arm["prefix_token_ids_sha256"],
            "trajectory_provenance": arm.get("trajectory_provenance"),
        },
        "config": {
            "infer_config": str(resolved_config_path),
            "resolved_config_fingerprint": resolved.fingerprint,
            "device": str(args.device),
            "model_dtype": "fp32",
            "seeds": list(seeds),
            "seed_selection": (
                {"kind": "explicit_list", "value": list(seeds)}
                if args.seeds is not None
                else {
                    "kind": "inclusive_range",
                    "start": int(seeds[0]),
                    "end": int(seeds[-1]),
                }
            ),
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "repetition_penalty": REPETITION_PENALTY,
        },
        "static_identity": static_identity,
        "source_checkpoint_identity": {
            "checkpoint_id": checkpoint_id,
            "checkpoint_json": static_identity["checkpoint_json"],
        },
        "source_config_identity": {
            "infer_config": static_identity["infer_config"],
            "resolved_config_fingerprint": resolved.fingerprint,
        },
        "generation_policy": {
            "mode": "sampled",
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "repetition_penalty": REPETITION_PENALTY,
        },
        "frozen_prefix_record": (
            None
            if prefix_record is None
            else {
                "path": str(prefix_record_path),
                "sha256": sha256_file(prefix_record_path),
                "token_ids": list(prefix_record["prefix_token_ids"]),
                "token_ids_sha256": prefix_record["prefix_token_ids_sha256"],
            }
        ),
        "selected_source_identity": {
            "source_jsonl": str(source_jsonl),
            "source_jsonl_current_sha256": sha256_file(source_jsonl),
            "image_id": image_id,
            "prompt_token_ids": prompt_token_ids,
            "prompt_token_ids_sha256": prompt_hash,
            "image_sha256": prompt_meta["image_sha256"],
            "width": int(plan.decoded_width),
            "height": int(plan.decoded_height),
            "observed_image_grid_thw": execution_identity["observed"][
                "observed_image_grid_thw"
            ],
            "executed_media_sha256": media_sha[0],
            "positive_owner_ledger": ledger,
        },
        "model_identity": model_receipt,
        "model_identity_check": model_identity,
        "execution_identity_check": execution_identity,
        "prefix": {
            "token_ids": arm["prefix_token_ids"],
            "token_ids_sha256": arm["prefix_token_ids_sha256"],
            "covered_owner_ids": covered_owner_ids,
            "uncovered_owner_ids": uncovered_owner_ids,
            "row_index": arm.get("row_index"),
            "harmful_kind": arm.get("harmful_kind"),
        },
        "greedy": {
            "producer_policy": build_producer_policy(
                mode="greedy",
                seed=None,
                checkpoint_id=checkpoint_id,
                prompt_token_ids_sha256=prompt_hash,
                prefix_token_ids_sha256=arm["prefix_token_ids_sha256"],
            ),
            "candidate_token_ids": greedy_row["raw_generated_token_ids"],
            "candidate_token_ids_sha256": greedy_row["raw_generated_token_ids_sha256"],
            "owner_matches": greedy_row["entity_matches"],
            "strict_matched_owner_ids": greedy_row["strict_matched_owner_ids"],
            "parity": greedy_parity,
        },
        "samples": samples,
        "selected_verified_uncovered_rescues": selected_rescues,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
