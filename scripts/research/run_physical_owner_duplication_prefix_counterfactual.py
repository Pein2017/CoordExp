#!/usr/bin/env python3
"""Run complete-row prefix counterfactuals before a reviewed duplicate owner.

The input is a small JSON case document with literal, hash-declared complete
row token IDs.  It holds the image prompt fixed and changes only the generated
row history immediately before an impending physical-owner duplicate.  The
five paired arms are:

* ``exact_self_prefix``;
* ``same_covered_set_row_shuffle``;
* ``earlier_repeated_owner_row_removed``;
* ``length_matched_covered_owner_replacement``; and
* ``description_preserved_coordinate_corruption``.

The runner deliberately reuses the existing exact-prefix replay helper for
short free continuations and the existing candidate-row scorer for raw,
teacher-forced field and complete-row scores.  It is an experiment-local
research tool: candidate rows are never normalized into a shared probability
distribution, and a changed prefix is not represented as a naturally sampled
trajectory.

Case shape (abbreviated)::

    {
      "schema_version": "physical_owner_duplication_prefix_counterfactual.case.v1",
      "cases": [{
        "case_id": "455649-bottle",
        "image_id": "455649",
        "entity_ledger": [{"owner_id": "bottle-a", "description": "bottle",
                           "bbox_norm1000": [1, 2, 3, 4]}],
        "prefix_rows": [{"owner_id": "bottle-a", "row_token_ids": [...],
                         "row_token_ids_sha256": "..."}],
        "impending_duplicate_owner_id": "bottle-a",
        "candidates": [{"candidate_id": "duplicate", "owner_id": "bottle-a",
                        "role": "impending_duplicate", "row_token_ids": [...],
                        "row_token_ids_sha256": "..."}],
        "interventions": {
          "shuffle_owner_order": ["other-owner", "bottle-a"],
          "replacement_owner_id": "other-owner",
          "coordinate_donor_owner_id": "other-owner"
        }
      }]
    }

All row token IDs must be canonical object-description-plus-four-coordinate
rows.  Coordinate corruption either copies the four coordinate tokens from a
declared donor owner or accepts a separately hash-declared
``coordinate_token_ids`` list.  In both cases it preserves every non-coordinate
token of the earlier repeated-owner row exactly.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from scripts.research import run_complete_candidate_row_scoring as candidate_scoring  # noqa: E402
from scripts.research import run_same_covered_set_prefix_order_probe as prefix_replay  # noqa: E402


CASE_SCHEMA_VERSION = "physical_owner_duplication_prefix_counterfactual.case.v1"
RECEIPT_SCHEMA_VERSION = "physical_owner_duplication_prefix_counterfactual.receipt.v1"
EXACT_ARM = "exact_self_prefix"
SHUFFLE_ARM = "same_covered_set_row_shuffle"
REMOVAL_ARM = "earlier_repeated_owner_row_removed"
REPLACEMENT_ARM = "length_matched_covered_owner_replacement"
CORRUPTION_ARM = "description_preserved_coordinate_corruption"
REQUIRED_ARMS = (EXACT_ARM, SHUFFLE_ARM, REMOVAL_ARM, REPLACEMENT_ARM, CORRUPTION_ARM)


class PrefixCounterfactualValidationError(ValueError):
    """Raised before model loading when a paired counterfactual is not honest."""


def sha256_json(value: Any) -> str:
    """Use the same canonical JSON hashing convention as the row scorer."""

    return candidate_scoring.sha256_json(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve(strict=True).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PrefixCounterfactualValidationError(f"{context} must be an object")
    return value


def _nonempty_text(value: Any, context: str) -> str:
    text = str(value if value is not None else "").strip()
    if not text:
        raise PrefixCounterfactualValidationError(f"{context} must be non-empty")
    return text


def _owner_id(value: Mapping[str, Any], context: str) -> str:
    for key in ("owner_id", "entity_id", "owner"):
        if value.get(key) is not None:
            return _nonempty_text(value[key], f"{context}.{key}")
    raise PrefixCounterfactualValidationError(f"{context} requires owner_id")


def _integer_tokens(value: Any, context: str, *, allow_empty: bool = False) -> list[int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PrefixCounterfactualValidationError(f"{context} must be a token-id sequence")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise PrefixCounterfactualValidationError(f"{context} contains an invalid token id")
        result.append(int(item))
    if not allow_empty and not result:
        raise PrefixCounterfactualValidationError(f"{context} must not be empty")
    return result


def _row_reference(record: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = record.get("row")
    return _mapping(nested, "row") if nested is not None else record


def _read_hashed_row(record: Mapping[str, Any], context: str) -> tuple[list[int], str]:
    source = _row_reference(record)
    tokens: Any = None
    for key in ("row_token_ids", "token_ids", "row_ids"):
        if source.get(key) is not None:
            tokens = source[key]
            break
    if tokens is None:
        raise PrefixCounterfactualValidationError(f"{context} requires row_token_ids")
    row = _integer_tokens(tokens, f"{context}.row_token_ids")
    declared: Any = None
    for key in ("row_token_ids_sha256", "token_ids_sha256", "sha256"):
        if source.get(key) is not None:
            declared = source[key]
            break
        if record.get(key) is not None:
            declared = record[key]
            break
    if not isinstance(declared, str) or not declared:
        raise PrefixCounterfactualValidationError(f"{context} requires a row token hash")
    actual = sha256_json(row)
    if declared != actual:
        raise PrefixCounterfactualValidationError(f"{context} row token hash mismatch")
    try:
        candidate_scoring._canonical_row_phases(row)  # noqa: SLF001 - established scorer contract
    except ValueError as exc:
        raise PrefixCounterfactualValidationError(f"{context} is not a canonical complete row") from exc
    return row, actual


def _row_coordinates(row_tokens: Sequence[int], context: str) -> list[int]:
    try:
        phases = candidate_scoring._canonical_row_phases(row_tokens)  # noqa: SLF001
    except ValueError as exc:
        raise PrefixCounterfactualValidationError(f"{context} is not a canonical complete row") from exc
    positions = [*phases["x1"], *phases["y1"], *phases["x2"], *phases["y2"]]
    return [int(row_tokens[index]) for index in positions]


def _validate_coordinate_address(coordinates: Sequence[int], context: str) -> list[int]:
    values = _integer_tokens(coordinates, context)
    if len(values) != 4:
        raise PrefixCounterfactualValidationError(f"{context} must contain four coordinate tokens")
    low = int(candidate_scoring.COORDINATE_TOKEN_START)
    high = int(candidate_scoring.COORDINATE_TOKEN_END_EXCLUSIVE)
    if any(token < low or token >= high for token in values):
        raise PrefixCounterfactualValidationError(f"{context} contains a non-coordinate token")
    x1, y1, x2, y2 = (token - low for token in values)
    if x2 <= x1 or y2 <= y1:
        raise PrefixCounterfactualValidationError(f"{context} is not a positive-area coordinate address")
    return values


def _replace_row_coordinates(row_tokens: Sequence[int], coordinates: Sequence[int], context: str) -> list[int]:
    row = [int(value) for value in row_tokens]
    address = _validate_coordinate_address(coordinates, context)
    phases = candidate_scoring._canonical_row_phases(row)  # noqa: SLF001
    positions = [*phases["x1"], *phases["y1"], *phases["x2"], *phases["y2"]]
    original = [row[index] for index in positions]
    if address == original:
        raise PrefixCounterfactualValidationError(
            f"{context} must differ from the original spatial address"
        )
    for index, token in zip(positions, address, strict=True):
        row[index] = token
    candidate_scoring._canonical_row_phases(row)  # noqa: SLF001
    if [row[index] for index in positions] != address:
        raise AssertionError("coordinate corruption did not replace the requested tokens")
    return row


def _normalise_prefix_row(value: Any, context: str) -> dict[str, Any]:
    raw = _mapping(value, context)
    owner = _owner_id(raw, context)
    tokens, token_hash = _read_hashed_row(raw, context)
    result = {
        "owner_id": owner,
        "row_token_ids": tokens,
        "row_token_ids_sha256": token_hash,
    }
    for key in ("description", "source", "notes", "row_id"):
        if raw.get(key) is not None:
            result[key] = raw[key]
    return result


def _normalise_ledger(value: Any, context: str) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if not isinstance(value, list) or not value:
        raise PrefixCounterfactualValidationError(f"{context} requires a non-empty entity_ledger")
    result: list[dict[str, Any]] = []
    by_owner: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(value):
        raw = _mapping(item, f"{context}[{index}]")
        owner = _owner_id(raw, f"{context}[{index}]")
        if owner in by_owner:
            raise PrefixCounterfactualValidationError(f"{context} has duplicate owner_id={owner!r}")
        description = _nonempty_text(raw.get("description"), f"{context}[{index}].description")
        bbox = raw.get("bbox_norm1000", raw.get("bbox"))
        if isinstance(bbox, (str, bytes)) or not isinstance(bbox, Sequence) or len(bbox) != 4:
            raise PrefixCounterfactualValidationError(f"{context}[{index}].bbox_norm1000 must contain four values")
        try:
            normalized_bbox = [float(component) for component in bbox]
        except (TypeError, ValueError) as exc:
            raise PrefixCounterfactualValidationError(f"{context}[{index}].bbox_norm1000 must be numeric") from exc
        if not all(0.0 <= component <= 1000.0 for component in normalized_bbox):
            raise PrefixCounterfactualValidationError(f"{context}[{index}].bbox_norm1000 must lie in [0,1000]")
        if normalized_bbox[2] <= normalized_bbox[0] or normalized_bbox[3] <= normalized_bbox[1]:
            raise PrefixCounterfactualValidationError(f"{context}[{index}].bbox_norm1000 must have positive area")
        normalized = {**dict(raw), "owner_id": owner, "entity_id": owner, "description": description, "bbox_norm1000": normalized_bbox}
        result.append(normalized)
        by_owner[owner] = normalized
    return result, by_owner


def _normalise_candidates(value: Any, *, duplicate_owner_id: str, context: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise PrefixCounterfactualValidationError(f"{context} requires a non-empty candidates list")
    candidates: list[dict[str, Any]] = []
    ids: set[str] = set()
    for index, item in enumerate(value):
        raw = _mapping(item, f"{context}[{index}]")
        candidate_id = _nonempty_text(raw.get("candidate_id", f"candidate-{index}"), f"{context}[{index}].candidate_id")
        if candidate_id in ids:
            raise PrefixCounterfactualValidationError(f"{context} has duplicate candidate_id={candidate_id!r}")
        ids.add(candidate_id)
        owner = _owner_id(raw, f"{context}[{index}]")
        tokens, token_hash = _read_hashed_row(raw, f"{context}[{index}]")
        role = str(raw.get("role", "")).strip() or (
            "impending_duplicate" if owner == duplicate_owner_id else "reviewed_candidate"
        )
        normalized = {
            "candidate_id": candidate_id,
            "owner_id": owner,
            "role": role,
            "row_token_ids": tokens,
            "row_token_ids_sha256": token_hash,
        }
        for key in ("category", "source", "truth_status", "notes", "covered"):
            if raw.get(key) is not None:
                normalized[key] = raw[key]
        # ``description`` is a numeric field-score key emitted by the reused
        # candidate scorer.  Keep the human-readable candidate label under a
        # non-score key so it cannot overwrite that score in a receipt.
        if raw.get("description") is not None:
            normalized["candidate_description"] = raw["description"]
        candidates.append(normalized)
    duplicate_candidates = [candidate for candidate in candidates if candidate["owner_id"] == duplicate_owner_id]
    if len(duplicate_candidates) != 1:
        raise PrefixCounterfactualValidationError(
            f"{context} must contain exactly one candidate for impending duplicate owner {duplicate_owner_id!r}"
        )
    return candidates


def _controls(case: Mapping[str, Any]) -> Mapping[str, Any]:
    value = case.get("interventions", {})
    return _mapping(value, "interventions")


def _control_value(controls: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if controls.get(name) is not None:
            return controls[name]
    return None


def validate_case_document(document: Mapping[str, Any]) -> dict[str, Any]:
    """Validate literal case input before constructing or replaying an arm."""

    raw = _mapping(document, "case document")
    if raw.get("schema_version") != CASE_SCHEMA_VERSION:
        raise PrefixCounterfactualValidationError(f"expected schema_version {CASE_SCHEMA_VERSION!r}")
    cases_value = raw.get("cases")
    if not isinstance(cases_value, list) or not cases_value:
        raise PrefixCounterfactualValidationError("case document requires a non-empty cases list")
    global_image_id = raw.get("image_id")
    cases: list[dict[str, Any]] = []
    case_ids: set[str] = set()
    for index, item in enumerate(cases_value):
        case = _mapping(item, f"cases[{index}]")
        case_id = _nonempty_text(case.get("case_id", f"case-{index}"), f"cases[{index}].case_id")
        if case_id in case_ids:
            raise PrefixCounterfactualValidationError(f"case document has duplicate case_id={case_id!r}")
        case_ids.add(case_id)
        image_id = _nonempty_text(case.get("image_id", global_image_id), f"cases[{index}].image_id")
        entity_ledger, ledger_by_owner = _normalise_ledger(case.get("entity_ledger"), f"cases[{index}]")
        rows_value = case.get("prefix_rows", case.get("self_prefix_rows"))
        if not isinstance(rows_value, list) or len(rows_value) < 2:
            raise PrefixCounterfactualValidationError(f"cases[{index}] requires at least two prefix_rows")
        prefix_rows = [_normalise_prefix_row(row, f"cases[{index}].prefix_rows[{row_index}]") for row_index, row in enumerate(rows_value)]
        owner_ids = [row["owner_id"] for row in prefix_rows]
        if len(set(owner_ids)) != len(owner_ids):
            raise PrefixCounterfactualValidationError(
                f"cases[{index}] prefix_rows must be before the first duplicate and therefore have distinct owners"
            )
        missing_ledger = [owner for owner in owner_ids if owner not in ledger_by_owner]
        if missing_ledger:
            raise PrefixCounterfactualValidationError(f"cases[{index}] prefix owners absent from entity_ledger: {missing_ledger}")
        duplicate_owner = _nonempty_text(case.get("impending_duplicate_owner_id"), f"cases[{index}].impending_duplicate_owner_id")
        if duplicate_owner not in owner_ids:
            raise PrefixCounterfactualValidationError(
                f"cases[{index}] impending duplicate owner is absent from the self prefix"
            )
        candidates = _normalise_candidates(case.get("candidates"), duplicate_owner_id=duplicate_owner, context=f"cases[{index}].candidates")
        candidate_owners_missing = [candidate["owner_id"] for candidate in candidates if candidate["owner_id"] not in ledger_by_owner]
        if candidate_owners_missing:
            raise PrefixCounterfactualValidationError(
                f"cases[{index}] candidate owners absent from entity_ledger: {candidate_owners_missing}"
            )
        cases.append(
            {
                "case_id": case_id,
                "image_id": image_id,
                "entity_ledger": entity_ledger,
                "entity_ledger_by_owner": ledger_by_owner,
                "prefix_rows": prefix_rows,
                "baseline_covered_owner_ids": owner_ids,
                "impending_duplicate_owner_id": duplicate_owner,
                "candidates": candidates,
                "interventions": dict(_controls(case)),
            }
        )
    return {**dict(raw), "cases": cases}


def _materialize_arm(
    name: str,
    *,
    intervention: str,
    rows: Sequence[Mapping[str, Any]],
    baseline_covered_owner_ids: Sequence[str],
) -> dict[str, Any]:
    copied_rows = [
        {
            **{key: deepcopy(value) for key, value in row.items() if key not in {"row_token_ids"}},
            "row_token_ids": [int(value) for value in row["row_token_ids"]],
            "row_token_ids_sha256": sha256_json([int(value) for value in row["row_token_ids"]]),
        }
        for row in rows
    ]
    tokens = [token for row in copied_rows for token in row["row_token_ids"]]
    return {
        "arm_name": name,
        "intervention": intervention,
        "prefix_rows": copied_rows,
        "prefix_owner_ids": [str(row["owner_id"]) for row in copied_rows],
        "baseline_covered_owner_ids": [str(value) for value in baseline_covered_owner_ids],
        "prefix_token_ids": tokens,
        "prefix_token_ids_sha256": sha256_json(tokens),
        "prefix_token_count": len(tokens),
    }


def _row_by_owner(rows: Sequence[Mapping[str, Any]], owner_id: str) -> Mapping[str, Any]:
    matches = [row for row in rows if str(row["owner_id"]) == owner_id]
    if len(matches) != 1:
        raise PrefixCounterfactualValidationError(f"expected one self-prefix row for owner {owner_id!r}")
    return matches[0]


def first_prefix_divergence(
    exact_arm: Mapping[str, Any], modified_arm: Mapping[str, Any]
) -> dict[str, Any]:
    """Locate the first token and row difference without hiding length changes."""

    exact = _integer_tokens(exact_arm.get("prefix_token_ids"), "exact prefix")
    modified = _integer_tokens(modified_arm.get("prefix_token_ids"), "modified prefix")
    index = next(
        (position for position, (left, right) in enumerate(zip(exact, modified, strict=False)) if left != right),
        None,
    )
    if index is None and len(exact) != len(modified):
        index = min(len(exact), len(modified))

    def row_index(rows: Any, token_index: int | None) -> int | None:
        if token_index is None or not isinstance(rows, list):
            return None
        start = 0
        for row_number, row in enumerate(rows):
            if not isinstance(row, Mapping):
                continue
            length = len(row.get("row_token_ids", []))
            if start <= token_index < start + length:
                return row_number
            start += length
        return None

    return {
        "relative_to": EXACT_ARM,
        "has_divergence": index is not None,
        "first_differing_token_index": index,
        "common_prefix_token_count": len(exact) if index is None else index,
        "exact_token_id": None if index is None or index >= len(exact) else exact[index],
        "modified_token_id": None if index is None or index >= len(modified) else modified[index],
        "exact_row_index": row_index(exact_arm.get("prefix_rows"), index),
        "modified_row_index": row_index(modified_arm.get("prefix_rows"), index),
        "exact_prefix_token_count": len(exact),
        "modified_prefix_token_count": len(modified),
    }


def _coordinate_corruption_address(
    controls: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], target_owner: str
) -> tuple[list[int], str]:
    nested = controls.get("coordinate_corruption")
    corruption = _mapping(nested, "coordinate_corruption") if nested is not None else controls
    coordinates = _control_value(corruption, "coordinate_token_ids", "coordinates")
    if coordinates is not None:
        if isinstance(coordinates, Mapping):
            values = coordinates.get("token_ids", coordinates.get("coordinate_token_ids"))
            declared = coordinates.get("token_ids_sha256", coordinates.get("coordinate_token_ids_sha256", coordinates.get("sha256")))
        else:
            values = coordinates
            declared = _control_value(corruption, "coordinate_token_ids_sha256", "coordinates_sha256")
        address = _validate_coordinate_address(values, "coordinate_corruption.coordinate_token_ids")
        actual = sha256_json(address)
        if not isinstance(declared, str) or not declared:
            raise PrefixCounterfactualValidationError("coordinate corruption requires coordinate_token_ids_sha256")
        if declared != actual:
            raise PrefixCounterfactualValidationError("coordinate corruption coordinate token hash mismatch")
        return address, "literal_coordinate_token_ids"
    donor = _control_value(corruption, "coordinate_donor_owner_id", "donor_owner_id")
    owners = [str(row["owner_id"]) for row in rows]
    if donor is None:
        donor = next(owner for owner in owners if owner != target_owner)
    donor_id = _nonempty_text(donor, "coordinate corruption donor owner")
    if donor_id == target_owner:
        raise PrefixCounterfactualValidationError("coordinate corruption donor must be a different covered owner")
    donor_row = _row_by_owner(rows, donor_id)
    return _validate_coordinate_address(_row_coordinates(donor_row["row_token_ids"], "coordinate donor row"), "coordinate donor row"), f"owner:{donor_id}"


def build_prefix_arms(case: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Construct and prove the five requested single-factor prefix arms."""

    rows = [dict(row) for row in case["prefix_rows"]]
    owners = [str(row["owner_id"]) for row in rows]
    target_owner = str(case["impending_duplicate_owner_id"])
    controls = _controls(case)
    target_index = owners.index(target_owner)
    target_row = rows[target_index]
    baseline = [str(value) for value in case["baseline_covered_owner_ids"]]

    requested_order = _control_value(controls, "shuffle_owner_order", "same_covered_set_shuffle_order")
    shuffled_owners = list(reversed(owners)) if requested_order is None else [str(value) for value in requested_order]
    if len(shuffled_owners) != len(owners) or set(shuffled_owners) != set(owners):
        raise PrefixCounterfactualValidationError("same-covered-set shuffle must contain every covered owner exactly once")
    if shuffled_owners == owners:
        raise PrefixCounterfactualValidationError("same-covered-set shuffle must change complete-row order")
    shuffled_rows = [_row_by_owner(rows, owner) for owner in shuffled_owners]

    removed_rows = [row for row_index, row in enumerate(rows) if row_index != target_index]

    replacement_value = _control_value(controls, "replacement_owner_id", "length_matched_replacement_owner_id")
    if replacement_value is None:
        replacement_owner = next(owner for owner in owners if owner != target_owner)
    else:
        replacement_owner = _nonempty_text(replacement_value, "replacement_owner_id")
    if replacement_owner == target_owner or replacement_owner not in owners:
        raise PrefixCounterfactualValidationError("replacement owner must be a different already covered owner")
    replacement_rows = list(rows)
    replacement_rows[target_index] = _row_by_owner(rows, replacement_owner)

    corrupted_coordinates, corruption_source = _coordinate_corruption_address(controls, rows, target_owner)
    corrupted_row = {
        **target_row,
        "row_token_ids": _replace_row_coordinates(
            target_row["row_token_ids"], corrupted_coordinates, "description-preserved coordinate corruption"
        ),
        "coordinate_corruption_source": corruption_source,
    }
    corrupted_rows = list(rows)
    corrupted_rows[target_index] = corrupted_row

    arms = {
        EXACT_ARM: _materialize_arm(EXACT_ARM, intervention="unmodified_exact_self_rollout", rows=rows, baseline_covered_owner_ids=baseline),
        SHUFFLE_ARM: _materialize_arm(SHUFFLE_ARM, intervention="same_covered_set_complete_row_shuffle", rows=shuffled_rows, baseline_covered_owner_ids=baseline),
        REMOVAL_ARM: _materialize_arm(REMOVAL_ARM, intervention="earlier_repeated_owner_row_removed", rows=removed_rows, baseline_covered_owner_ids=baseline),
        REPLACEMENT_ARM: _materialize_arm(REPLACEMENT_ARM, intervention="length_matched_different_covered_owner_replacement", rows=replacement_rows, baseline_covered_owner_ids=baseline),
        CORRUPTION_ARM: _materialize_arm(CORRUPTION_ARM, intervention="description_preserved_wrong_coordinate_address", rows=corrupted_rows, baseline_covered_owner_ids=baseline),
    }
    exact = arms[EXACT_ARM]
    for name, arm in arms.items():
        arm["first_divergence_from_exact"] = first_prefix_divergence(exact, arm)
    validate_prefix_arm_invariants(arms, impending_duplicate_owner_id=target_owner)
    return arms


def validate_prefix_arm_invariants(
    arms: Mapping[str, Mapping[str, Any]], *, impending_duplicate_owner_id: str
) -> dict[str, Any]:
    """Reject accidental multi-factor constructions before loading a model."""

    missing = [name for name in REQUIRED_ARMS if name not in arms]
    if missing:
        raise PrefixCounterfactualValidationError(f"missing required prefix arms: {missing}")
    exact = arms[EXACT_ARM]
    exact_rows = exact.get("prefix_rows")
    if not isinstance(exact_rows, list) or not exact_rows:
        raise PrefixCounterfactualValidationError("exact self prefix has no rows")
    exact_owner_ids = [str(row["owner_id"]) for row in exact_rows]
    exact_hashes = [str(row["row_token_ids_sha256"]) for row in exact_rows]
    if str(impending_duplicate_owner_id) not in exact_owner_ids:
        raise PrefixCounterfactualValidationError("exact self prefix does not cover impending duplicate owner")
    target_index = exact_owner_ids.index(str(impending_duplicate_owner_id))
    target_tokens = list(exact_rows[target_index]["row_token_ids"])

    shuffled = arms[SHUFFLE_ARM]
    shuffled_rows = shuffled.get("prefix_rows")
    if not isinstance(shuffled_rows, list) or Counter(str(row["row_token_ids_sha256"]) for row in shuffled_rows) != Counter(exact_hashes):
        raise PrefixCounterfactualValidationError("shuffle must preserve the complete-row token multiset")
    if [str(row["owner_id"]) for row in shuffled_rows] == exact_owner_ids:
        raise PrefixCounterfactualValidationError("shuffle arm is identical to exact self prefix")

    removed_rows = arms[REMOVAL_ARM].get("prefix_rows")
    if not isinstance(removed_rows, list) or len(removed_rows) != len(exact_rows) - 1:
        raise PrefixCounterfactualValidationError("removal arm must remove exactly one complete row")
    if str(impending_duplicate_owner_id) in [str(row["owner_id"]) for row in removed_rows]:
        raise PrefixCounterfactualValidationError("removal arm still contains the earlier repeated owner")

    replaced_rows = arms[REPLACEMENT_ARM].get("prefix_rows")
    if not isinstance(replaced_rows, list) or len(replaced_rows) != len(exact_rows):
        raise PrefixCounterfactualValidationError("replacement arm must preserve row count")
    if any(
        replaced_rows[index]["row_token_ids"] != exact_rows[index]["row_token_ids"]
        for index in range(len(exact_rows))
        if index != target_index
    ):
        raise PrefixCounterfactualValidationError("replacement arm changed a row other than the earlier repeated owner")
    if replaced_rows[target_index]["row_token_ids"] == target_tokens:
        raise PrefixCounterfactualValidationError("replacement arm did not replace the earlier repeated-owner row")
    replacement_owner = str(replaced_rows[target_index]["owner_id"])
    if replacement_owner == str(impending_duplicate_owner_id) or replacement_owner not in exact_owner_ids:
        raise PrefixCounterfactualValidationError("replacement arm does not use another already covered owner")

    corrupted_rows = arms[CORRUPTION_ARM].get("prefix_rows")
    if not isinstance(corrupted_rows, list) or len(corrupted_rows) != len(exact_rows):
        raise PrefixCounterfactualValidationError("coordinate-corruption arm must preserve row count")
    if any(
        corrupted_rows[index]["row_token_ids"] != exact_rows[index]["row_token_ids"]
        for index in range(len(exact_rows))
        if index != target_index
    ):
        raise PrefixCounterfactualValidationError("coordinate corruption changed a non-target prefix row")
    original_phases = candidate_scoring._canonical_row_phases(target_tokens)  # noqa: SLF001
    coordinate_positions = set(original_phases["x1"] + original_phases["y1"] + original_phases["x2"] + original_phases["y2"])
    corrupted_tokens = list(corrupted_rows[target_index]["row_token_ids"])
    if any(
        token != target_tokens[index]
        for index, token in enumerate(corrupted_tokens)
        if index not in coordinate_positions
    ):
        raise PrefixCounterfactualValidationError("coordinate corruption did not preserve description and wrappers")
    if _row_coordinates(corrupted_tokens, "coordinate corruption row") == _row_coordinates(target_tokens, "exact target row"):
        raise PrefixCounterfactualValidationError("coordinate corruption did not alter the spatial address")
    return {
        "same_covered_set_shuffle_preserves_row_multiset": True,
        "repeated_owner_removal_changes_one_row_count": True,
        "length_matched_replacement_changes_one_target_row": True,
        "coordinate_corruption_preserves_non_coordinate_tokens": True,
    }


def select_cases(document: Mapping[str, Any], requested_case_ids: Sequence[str] | None = None) -> list[Mapping[str, Any]]:
    cases = document.get("cases")
    if not isinstance(cases, list):
        raise PrefixCounterfactualValidationError("validated case document lacks cases")
    requested = [str(value) for value in (requested_case_ids or [])]
    if not requested:
        return list(cases)
    available = {str(case["case_id"]): case for case in cases}
    missing = [case_id for case_id in requested if case_id not in available]
    if missing:
        raise PrefixCounterfactualValidationError(f"requested case_id(s) absent from case document: {missing}")
    if len(set(requested)) != len(requested):
        raise PrefixCounterfactualValidationError("requested case ids must be unique")
    wanted = set(requested)
    return [case for case in cases if str(case["case_id"]) in wanted]


def summarize_short_continuation(
    run: Mapping[str, Any], *, baseline_covered_owner_ids: Sequence[str]
) -> dict[str, Any]:
    """Make the downstream destination explicit without reinterpreting rows."""

    rows = run.get("rows")
    typed_rows = [row for row in rows if isinstance(row, Mapping)] if isinstance(rows, list) else []
    baseline = {str(owner) for owner in baseline_covered_owner_ids}
    first = typed_rows[0] if typed_rows else None
    first_owners = [] if first is None else sorted({str(value) for value in first.get("strict_matched_owner_ids", [])})
    first_covered = sorted(owner for owner in first_owners if owner in baseline)
    first_new = sorted(owner for owner in first_owners if owner not in baseline)
    downstream_owners = sorted(
        {
            str(owner)
            for row in typed_rows
            for owner in row.get("strict_matched_owner_ids", [])
        }
    )
    downstream_new = sorted(owner for owner in downstream_owners if owner not in baseline)
    downstream_covered = sorted(owner for owner in downstream_owners if owner in baseline)
    complete_rows = [row for row in typed_rows if bool(row.get("accepted_complete_row"))]
    unresolved_rows = [
        row
        for row in typed_rows
        if row.get("unmatched_or_ambiguous_prediction_indices")
    ]
    first_stop = None if first is None else _mapping(first.get("row_stop", {}), "continuation row_stop").get("stop_reason")
    return {
        "continuation_status": run.get("status"),
        "horizon_rows_requested": run.get("horizon_rows_requested"),
        "horizon_rows_generated": run.get("horizon_rows_generated"),
        "horizon_complete": run.get("horizon_complete"),
        "first_row_stop_reason": first_stop,
        "first_owner_ids": first_owners,
        "first_covered_owner_ids": first_covered,
        "first_new_owner_ids": first_new,
        "first_row_is_duplicate": bool(first_covered),
        "first_row_is_new_owner": bool(first_new),
        "first_row_is_terminal": first_stop == "terminal",
        "first_row_invalid_or_incomplete": first is not None and not bool(first.get("accepted_complete_row")),
        "downstream_owner_ids": downstream_owners,
        "downstream_new_owner_ids": downstream_new,
        "downstream_covered_owner_ids": downstream_covered,
        "complete_row_count": len(complete_rows),
        "duplicate_row_count": sum(bool(row.get("covered_prefix_owner_ids")) for row in complete_rows),
        "unresolved_or_ambiguous_row_count": len(unresolved_rows),
        "termination_observed": any(
            _mapping(row.get("row_stop", {}), "continuation row_stop").get("stop_reason") == "terminal"
            for row in typed_rows
        ),
    }


def _case_artifact_skeleton(case: Mapping[str, Any]) -> dict[str, Any]:
    arms = build_prefix_arms(case)
    return {
        "case_id": str(case["case_id"]),
        "image_id": str(case["image_id"]),
        "impending_duplicate_owner_id": str(case["impending_duplicate_owner_id"]),
        "baseline_covered_owner_ids": [str(value) for value in case["baseline_covered_owner_ids"]],
        "entity_ledger": [dict(entity) for entity in case["entity_ledger"]],
        "candidates": [dict(candidate) for candidate in case["candidates"]],
        "prefix_arm_invariants": validate_prefix_arm_invariants(
            arms, impending_duplicate_owner_id=str(case["impending_duplicate_owner_id"])
        ),
        "original_self_prefix_token_ids": list(arms[EXACT_ARM]["prefix_token_ids"]),
        "original_self_prefix_token_ids_sha256": str(arms[EXACT_ARM]["prefix_token_ids_sha256"]),
        "arms": arms,
    }


_CANDIDATE_SCORE_FIELDS = (
    "row_entry",
    "description",
    "geometry",
    "x1",
    "y1",
    "x2",
    "y2",
    "closure",
    "full_row",
)
_TERMINAL_SCORE_FIELDS = (
    "row_entry_log_probability",
    "terminal_log_probability",
    "row_entry_minus_terminal",
)


def _finite_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PrefixCounterfactualValidationError(f"{context} must be a finite numeric value")
    result = float(value)
    if not math.isfinite(result):
        raise PrefixCounterfactualValidationError(f"{context} must be finite")
    return result


def _positive_count(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PrefixCounterfactualValidationError(f"{context} must be a positive integer")
    return int(value)


def _validate_field_score(value: Any, context: str) -> dict[str, Any]:
    score = _mapping(value, f"{context} score")
    return {
        "sum": _finite_number(score.get("sum"), f"{context} score.sum"),
        "mean": _finite_number(score.get("mean"), f"{context} score.mean"),
        "count": _positive_count(score.get("count"), f"{context} score.count"),
    }


def _validate_candidate_score(value: Any, context: str) -> None:
    score = _mapping(value, context)
    _nonempty_text(score.get("candidate_id"), f"{context}.candidate_id")
    _nonempty_text(score.get("owner_id", score.get("owner")), f"{context}.owner_id")
    _nonempty_text(score.get("candidate_role", score.get("role")), f"{context}.candidate_role")
    if score.get("candidate_description") is not None and not isinstance(score["candidate_description"], str):
        raise PrefixCounterfactualValidationError(f"{context}.candidate_description must be text when present")
    token_count = _positive_count(score.get("token_count"), f"{context}.token_count")
    token_log_probabilities = score.get("token_log_probabilities")
    if not isinstance(token_log_probabilities, list) or len(token_log_probabilities) != token_count:
        raise PrefixCounterfactualValidationError(f"{context}.token_log_probabilities must match token_count")
    for index, probability in enumerate(token_log_probabilities):
        _finite_number(probability, f"{context}.token_log_probabilities[{index}]")
    fields = {field: _validate_field_score(score.get(field), f"{context}.{field}") for field in _CANDIDATE_SCORE_FIELDS}
    if fields["full_row"]["count"] != token_count:
        raise PrefixCounterfactualValidationError(f"{context}.full_row score count must match token_count")


def _validate_terminal_score(value: Any, context: str) -> None:
    terminal = _mapping(value, context)
    for field in _TERMINAL_SCORE_FIELDS:
        _finite_number(terminal.get(field), f"{context}.{field}")


def _validate_short_continuations(value: Any, context: str) -> None:
    if not isinstance(value, list) or not value:
        raise PrefixCounterfactualValidationError(f"{context} requires a non-empty continuation result list")
    for index, item in enumerate(value):
        continuation = _mapping(item, f"{context}[{index}]")
        _nonempty_text(continuation.get("mode"), f"{context}[{index}].mode")
        _nonempty_text(continuation.get("status"), f"{context}[{index}].status")
        outcome = _mapping(continuation.get("short_continuation_outcome"), f"{context}[{index}].short_continuation_outcome")
        _nonempty_text(outcome.get("continuation_status"), f"{context}[{index}].short_continuation_outcome.continuation_status")


def validate_receipt_payload(payload: Mapping[str, Any]) -> None:
    """Cheap structural and hash check for a completed or synthetic receipt."""

    if payload.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise PrefixCounterfactualValidationError("receipt schema_version is invalid")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise PrefixCounterfactualValidationError("receipt requires non-empty cases")
    for case in cases:
        raw_case = _mapping(case, "receipt case")
        arms = _mapping(raw_case.get("arms"), "receipt case arms")
        validate_prefix_arm_invariants(arms, impending_duplicate_owner_id=str(raw_case.get("impending_duplicate_owner_id", "")))
        exact = arms[EXACT_ARM]
        if raw_case.get("original_self_prefix_token_ids_sha256") != exact.get("prefix_token_ids_sha256"):
            raise PrefixCounterfactualValidationError("receipt original prefix hash does not match exact arm")
        for name in REQUIRED_ARMS:
            arm = _mapping(arms[name], f"receipt arm {name}")
            tokens = _integer_tokens(arm.get("prefix_token_ids"), f"receipt arm {name}.prefix_token_ids")
            if arm.get("prefix_token_ids_sha256") != sha256_json(tokens):
                raise PrefixCounterfactualValidationError(f"receipt arm {name} prefix hash mismatch")
            if "first_divergence_from_exact" not in arm:
                raise PrefixCounterfactualValidationError(f"receipt arm {name} lacks first-divergence evidence")
            scores = arm.get("raw_candidate_field_row_scores")
            if not isinstance(scores, list) or not scores:
                raise PrefixCounterfactualValidationError(f"receipt arm {name} requires a non-empty candidate score list")
            for index, score in enumerate(scores):
                _validate_candidate_score(score, f"receipt arm {name}.raw_candidate_field_row_scores[{index}]")
            _validate_terminal_score(arm.get("terminal_boundary_score"), f"receipt arm {name}.terminal_boundary_score")
            _validate_short_continuations(arm.get("short_continuations"), f"receipt arm {name}.short_continuations")


def _case_by_image(raw_examples: Sequence[Any], image_id: str) -> Any:
    return prefix_replay._select_example(raw_examples, image_id)  # noqa: SLF001 - exact replay owner


def _runtime_model_inputs(native_inputs: Mapping[str, Any]) -> tuple[Mapping[str, Any], Any]:
    import torch

    image_grid_thw = native_inputs.get("image_grid_thw")
    if not isinstance(image_grid_thw, torch.Tensor):
        raise RuntimeError("native inputs lack image_grid_thw for candidate scoring")
    excluded = {"input_ids", "attention_mask", "position_ids", "token_type_ids"}
    return ({key: value for key, value in native_inputs.items() if key not in excluded}, image_grid_thw)


def _score_arm_candidates(
    *,
    model: Any,
    model_inputs: Mapping[str, Any],
    image_grid_thw: Any,
    full_prefix_token_ids: Sequence[int],
    candidates: Sequence[Mapping[str, Any]],
    terminal_token_id: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    boundary_logits = candidate_scoring._forward_logits(  # noqa: SLF001 - established scorer implementation
        model, model_inputs, full_prefix_token_ids, image_grid_thw
    )
    terminal = candidate_scoring.terminal_boundary_score(
        boundary_logits,
        boundary_length=len(full_prefix_token_ids),
        row_entry_token_id=int(candidate_scoring.OBJECT_REF_START),
        terminal_token_id=int(terminal_token_id),
    )
    scores: list[dict[str, Any]] = []
    for candidate in candidates:
        row = [int(value) for value in candidate["row_token_ids"]]
        logits = candidate_scoring._forward_logits(  # noqa: SLF001
            model, model_inputs, [*full_prefix_token_ids, *row], image_grid_thw
        )
        metadata = {
            "candidate_id": candidate["candidate_id"],
            "owner": candidate["owner_id"],
            "category": candidate.get("category", candidate.get("candidate_description")),
            "role": candidate["role"],
            "covered": candidate.get("covered"),
            **{key: candidate[key] for key in ("source", "truth_status", "notes") if key in candidate},
        }
        score = candidate_scoring._score_candidate(  # noqa: SLF001
            logits,
            boundary_length=len(full_prefix_token_ids),
            row_tokens=row,
            metadata=metadata,
        )
        score["owner_id"] = str(candidate["owner_id"])
        score["candidate_role"] = str(candidate["role"])
        score["row_token_ids_sha256"] = str(candidate["row_token_ids_sha256"])
        if candidate.get("candidate_description") is not None:
            score["candidate_description"] = str(candidate["candidate_description"])
        scores.append(score)
    return terminal, scores


def _failed_continuation(
    *,
    mode: str,
    seed: int | None,
    horizon_rows: int,
    error: BaseException,
) -> dict[str, Any]:
    return {
        "mode": mode,
        "seed": seed,
        "status": "failed",
        "horizon_complete": False,
        "horizon_rows_requested": int(horizon_rows),
        "horizon_rows_generated": 0,
        "rows": [],
        "failure": {"type": type(error).__name__, "message": str(error)},
    }


def _parse_seed_list(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        pieces = [piece.strip() for piece in value.split(",") if piece.strip()]
    else:
        pieces = list(value)
    try:
        seeds = tuple(int(value) for value in pieces)
    except (TypeError, ValueError) as exc:
        raise PrefixCounterfactualValidationError("seeds must be integer values") from exc
    if any(seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise PrefixCounterfactualValidationError("seeds must be unique non-negative integers")
    return seeds


def run_probe(
    *,
    cases_path: Path,
    output: Path,
    infer_config: Path,
    case_ids: Sequence[str] | None = None,
    seeds: Sequence[int] = (11, 12, 13),
    include_greedy: bool = True,
    horizon_rows: int = 2,
    temperature: float = 0.2,
    top_p: float = 0.95,
    repetition_penalty: float = 1.0,
    max_new_tokens: int = 256,
    malformed_limit: int = 2,
    device: str = "cuda:0",
    force: bool = False,
) -> Path:
    """Execute the paired analysis with one model session and frozen conditions."""

    import torch

    from src.config.fingerprint import sha256_json as config_sha256_json
    from src.config.inference import load_infer_config
    from src.data import load_raw_examples
    from src.inference.backend import open_backend_session
    from src.inference.runtime import assemble_frontend

    cases_path = cases_path.expanduser().resolve(strict=True)
    output = output.expanduser().resolve()
    infer_config = infer_config.expanduser().resolve(strict=True)
    if output.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {output}; pass --force")
    seeds = _parse_seed_list(seeds)
    if not include_greedy and not seeds:
        raise PrefixCounterfactualValidationError("at least greedy or one sampled continuation seed is required")
    if not 1 <= int(horizon_rows) <= 4:
        raise PrefixCounterfactualValidationError("horizon_rows must be in [1,4]")
    if not 0 < float(temperature) or not 0 < float(top_p) <= 1:
        raise PrefixCounterfactualValidationError("temperature must be positive and top_p must be in (0,1]")
    if float(repetition_penalty) <= 0 or int(max_new_tokens) <= 0 or int(malformed_limit) <= 0:
        raise PrefixCounterfactualValidationError("decode values must be positive")

    source_document = json.loads(cases_path.read_text(encoding="utf-8"))
    document = validate_case_document(source_document)
    selected_cases = select_cases(document, case_ids)
    resolved = load_infer_config(infer_config)
    config = resolved.config
    raw_examples = list(load_raw_examples(config.data.input_jsonl))
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.set_device(torch.device(device))
    frontend = assemble_frontend(
        config,
        generation_config_fingerprint=config_sha256_json(config.generation.model_dump(mode="json")),
    )
    case_artifacts = [_case_artifact_skeleton(case) for case in selected_cases]
    output.parent.mkdir(parents=True, exist_ok=True)
    with open_backend_session(frontend.launch) as session:
        model = getattr(session, "_model", None)
        tokenizer = getattr(session, "_tokenizer", None)
        if model is None or tokenizer is None:
            raise RuntimeError("prefix counterfactual requires the native Hugging Face backend session")
        model.eval()
        terminal_token_id = getattr(tokenizer, "eos_token_id", None)
        if terminal_token_id is None or int(terminal_token_id) < 0:
            raise RuntimeError("tokenizer does not expose an eos_token_id for terminal scoring")
        for case, artifact in zip(selected_cases, case_artifacts, strict=True):
            example = _case_by_image(raw_examples, str(case["image_id"]))
            request, plan, prompt_meta = prefix_replay._build_request(config, frontend, example)  # noqa: SLF001
            native_inputs, executed_ids, observed_grids, media_sha = session._materialize_native_inputs((request,))  # noqa: SLF001
            if tuple(executed_ids[0]) != tuple(request.expected_executed_prompt_token_ids):
                raise RuntimeError(f"{case['case_id']} base image prompt token parity failed")
            model_inputs, image_grid_thw = _runtime_model_inputs(native_inputs)
            base_prompt = [int(value) for value in executed_ids[0]]
            artifact["base_prompt"] = {
                **prompt_meta,
                "observed_prompt_token_ids": base_prompt,
                "observed_prompt_token_ids_sha256": sha256_json(base_prompt),
            }
            artifact["source_image"] = {
                "path": str(plan.image_path),
                "sha256": plan.image_content_sha256,
                "width": int(plan.decoded_width),
                "height": int(plan.decoded_height),
                "executed_media_sha256": media_sha[0],
                "observed_image_grid_thw": None if observed_grids[0] is None else list(observed_grids[0]),
            }
            for arm_name in REQUIRED_ARMS:
                arm = artifact["arms"][arm_name]
                full_prefix = [*base_prompt, *[int(value) for value in arm["prefix_token_ids"]]]
                terminal, candidate_scores = _score_arm_candidates(
                    model=model,
                    model_inputs=model_inputs,
                    image_grid_thw=image_grid_thw,
                    full_prefix_token_ids=full_prefix,
                    candidates=case["candidates"],
                    terminal_token_id=int(terminal_token_id),
                )
                arm["model_input_prefix_token_ids_sha256"] = sha256_json(full_prefix)
                arm["model_input_prefix_token_count"] = len(full_prefix)
                arm["terminal_boundary_score"] = terminal
                arm["raw_candidate_field_row_scores"] = candidate_scores
                continuation_conditions: list[tuple[str, int | None]] = (
                    [("greedy", None)] if include_greedy else []
                )
                continuation_conditions.extend(("sample", int(seed)) for seed in seeds)
                continuations: list[dict[str, Any]] = []
                for mode, seed in continuation_conditions:
                    try:
                        run = prefix_replay._generate_horizon(  # noqa: SLF001 - exact prefix replay owner
                            session=session,
                            native_inputs=native_inputs,
                            prefix_token_ids=arm["prefix_token_ids"],
                            tokenizer=tokenizer,
                            image_width=int(plan.decoded_width),
                            image_height=int(plan.decoded_height),
                            mode=mode,
                            seed=seed,
                            temperature=float(temperature),
                            top_p=float(top_p),
                            repetition_penalty=float(repetition_penalty),
                            max_new_tokens=int(max_new_tokens),
                            malformed_limit=int(malformed_limit),
                            horizon_rows=int(horizon_rows),
                            entity_rows=case["entity_ledger_by_owner"],
                            # This remains the original covered-owner set for
                            # every arm, including removal/replacement.  It
                            # makes a reselection of the removed surface row a
                            # duplicate relative to the same physical state.
                            covered_entity_ids=case["baseline_covered_owner_ids"],
                        )
                    except Exception as exc:  # preserve an arm/seed failure in the receipt
                        run = _failed_continuation(mode=mode, seed=seed, horizon_rows=int(horizon_rows), error=exc)
                    run["short_continuation_outcome"] = summarize_short_continuation(
                        run, baseline_covered_owner_ids=case["baseline_covered_owner_ids"]
                    )
                    continuations.append(run)
                arm["short_continuations"] = continuations
        backend_receipt = session.receipt.to_artifact_dict()

    payload = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "experiment": "physical_owner_duplication_prefix_counterfactual",
        "case_document": {
            "path": str(cases_path),
            "sha256": sha256_file(cases_path),
            "schema_version": CASE_SCHEMA_VERSION,
            "selected_case_ids": [str(case["case_id"]) for case in selected_cases],
        },
        "paired_conditions": {
            "image_encoding": "fixed per image across all prefix arms",
            "prompt": "fixed canonical source prompt per image",
            "model_and_adapter": "one backend session",
            "candidate_scoring": "raw float32 teacher-forced field and full-row scores; no shared candidate distribution",
            "continuation_horizon_rows": int(horizon_rows),
            "continuation_modes": (["greedy"] if include_greedy else []) + (["sample"] if seeds else []),
            "sample_seeds": list(seeds),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "repetition_penalty": float(repetition_penalty),
            "max_new_tokens": int(max_new_tokens),
            "malformed_limit": int(malformed_limit),
            "device": str(device),
        },
        "runtime": {
            "infer_config_path": str(infer_config),
            "infer_config_sha256": sha256_file(infer_config),
            "resolved_config_fingerprint": resolved.fingerprint,
            "backend_session": backend_receipt,
            "eos_token_id": int(terminal_token_id),
            "row_entry_token_id": int(candidate_scoring.OBJECT_REF_START),
        },
        "compact_receipt": {
            "case_count": len(case_artifacts),
            "arm_names": list(REQUIRED_ARMS),
            "per_case_outputs": [
                "original_and_modified_prefix_tokens_and_hashes",
                "first_divergence_from_exact",
                "raw_candidate_field_row_scores",
                "terminal_boundary_score",
                "short_continuation_outcomes",
            ],
        },
        "cases": case_artifacts,
    }
    validate_receipt_payload(payload)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True, help="hash-declared prefix counterfactual case JSON")
    parser.add_argument("--output", type=Path, required=True, help="immutable JSON receipt output")
    parser.add_argument("--infer-config", type=Path, required=True)
    parser.add_argument("--case-id", action="append", default=None, help="case id to execute; repeat for a bounded subset")
    parser.add_argument("--seeds", default="11,12,13", help="optional comma-separated sampled continuation seeds")
    parser.add_argument("--no-greedy", action="store_true", help="omit the deterministic short continuation")
    parser.add_argument("--horizon-rows", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--malformed-limit", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        args.seeds = _parse_seed_list(args.seeds)
        if args.case_id is not None:
            selected = [str(value).strip() for value in args.case_id if str(value).strip()]
            if len(selected) != len(set(selected)):
                raise PrefixCounterfactualValidationError("--case-id values must be unique")
            args.case_id = selected
    except PrefixCounterfactualValidationError as exc:
        parser.error(str(exc))
    return args


def main() -> int:
    args = _parse_args()
    run_probe(
        cases_path=args.cases,
        output=args.output,
        infer_config=args.infer_config,
        case_ids=args.case_id,
        seeds=args.seeds,
        include_greedy=not args.no_greedy,
        horizon_rows=args.horizon_rows,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_new_tokens,
        malformed_limit=args.malformed_limit,
        device=args.device,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
