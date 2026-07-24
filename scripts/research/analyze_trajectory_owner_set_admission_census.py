#!/usr/bin/env python3
"""Read-only root-state trajectory owner-set admission census.

This analyzer consumes the frozen v2 Source@B16 plus sixteen-sample panel
through ``load_v2_b16_panel_adapter``.  It never materializes a StateBank and
keeps annotation-anchored entity identity separate from the stricter geometry
gate used for primary admission.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any
import uuid

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config.fingerprint import sha256_file, sha256_json  # noqa: E402
from src.inference.backend import token_ids_sha256  # noqa: E402

from scripts.research.assemble_constant_dose_breadth_state_banks import (  # noqa: E402
    AssemblyError,
    _candidate_pool,
    _split_membership,
    load_v2_b16_panel_adapter,
)
from scripts.research.assemble_positive_path_imitation_state_bank import (  # noqa: E402
    GEOMETRY_IOU_THRESHOLD,
    geometry_eligibility_receipt,
)


SCHEMA_VERSION = "trajectory_owner_set_admission_census.v1"
EXPECTED_CANDIDATE_POOL_SHA256 = (
    "133afcf6659b78d71893e9f79e568dca676515a05c7fe23f3c237de01350b7e2"
)
EXPECTED_SPLIT_RECEIPT_SHA256 = (
    "478afe838c63dab7e55cad4ad16d8d1d0aa490231de8f25a0766768bda3678bf"
)
EXPECTED_TRAIN_CANDIDATE_SHA256 = (
    "c0efce5806e0ee487298f7b3b7697f4ebe663d1fa210b2e0e10e08de08ab5746"
)
FROZEN_CANDIDATE_POOL_PATH = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-22-constant-dose-image-breadth-treatment-screen/"
    "candidate-pool-v1/candidate-pool-2432.coord.jsonl"
)
FROZEN_SPLIT_RECEIPT_PATH = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-22-constant-dose-image-breadth-treatment-screen/"
    "candidate-pool-v1/split-receipt.json"
)
FROZEN_SPLIT_OUTPUT_PATHS = {
    "train_candidate": Path(
        "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
        "2026-07-22-constant-dose-image-breadth-treatment-screen/"
        "candidate-pool-v1/train-candidate.jsonl"
    ),
    "development": Path(
        "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
        "2026-07-22-constant-dose-image-breadth-treatment-screen/"
        "candidate-pool-v1/development.jsonl"
    ),
    "heldout": Path(
        "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
        "2026-07-22-constant-dose-image-breadth-treatment-screen/"
        "candidate-pool-v1/heldout.jsonl"
    ),
}
FROZEN_OUTPUT_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-23-trajectory-owner-set-admission-census/production-v1"
)
EXPECTED_POOL_IMAGE_COUNT = 2432
EXPECTED_TRAIN_IMAGE_COUNT = 2048
EXPECTED_ELIGIBLE_TRAIN_IMAGE_COUNT = 2004
EXPECTED_CANDIDATES_PER_IMAGE = 17
EXPECTED_MANIFEST_COUNT_PER_ROOT = 8
EXPECTED_BATCH_COUNT_PER_ROOT = 152
FROZEN_SAMPLED_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-22-constant-dose-image-breadth-treatment-screen/"
    "trajectory-panel-2432-vllm/production-v2"
)
FROZEN_SOURCE_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-22-constant-dose-image-breadth-treatment-screen/"
    "source-b16-vllm/production-v1"
)
FROZEN_SAMPLED_MANIFEST_SET_SHA256 = (
    "28304194b0b5e2aa7a718b2ee080253f616a8d9cf8fde8653e335589b3179b60"
)
FROZEN_SOURCE_MANIFEST_SET_SHA256 = (
    "4e979f920798dcd338c0620500d36b007f90404bb4a2852014702de93b9f521e"
)
FROZEN_EXECUTION_MODEL_IDENTITY_SHA256 = (
    "8e0cc5c679df56cd55f965b56a4e9f4a661c0277bc93248100a0d143190918ac"
)
FROZEN_TOKENIZER_IDENTITY_SHA256 = (
    "878bc75fd27e4668788cb864bf93dee4d90dbbbbceec0f2037ba8eab75247fd1"
)
SOURCE_SNAPSHOT_PATHS = (
    "research/investigations/qwen3-vl-dense-enumeration/experiments/"
    "2026-07-23-trajectory-owner-set-admission-census/unit.md",
    "scripts/research/analyze_trajectory_owner_set_admission_census.py",
    "scripts/research/assemble_constant_dose_breadth_state_banks.py",
    "scripts/research/analyze_individual_trajectory_union_support.py",
    "scripts/research/assemble_positive_path_imitation_state_bank.py",
    "src/inference/backend.py",
    "src/config/fingerprint.py",
)
FEASIBILITY_IMAGE_COUNT = 256
SAFETY_FIELDS = (
    "duplicate",
    "malformed",
    "confirmed_false",
    "semantic_error",
    "unknown",
    "premature_stop",
)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AssemblyError(f"{label} must be an object")
    return value


def _int_count(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise AssemblyError(f"{label} must be a nonnegative integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise AssemblyError(f"{label} must be a nonnegative integer") from exc
    if result < 0:
        raise AssemblyError(f"{label} must be a nonnegative integer")
    return result


def _route_sort_key(route_id: str) -> tuple[int, str]:
    return (0 if route_id == "source-b16" else 1, route_id)


def _row_owner(row: Mapping[str, Any]) -> tuple[str | None, float | None]:
    status = str(row.get("entity_status", ""))
    if status == "verified_owner":
        owner = row.get("owner_id")
        iou = row.get("intersection_over_union")
    elif status in {"duplicate", "duplicate_owner"}:
        owner = row.get("owner_id", row.get("candidate_owner_id"))
        iou = row.get("intersection_over_union", row.get("candidate_owner_iou"))
    else:
        return None, None
    if owner is None or iou is None:
        return None, None
    try:
        return str(owner), float(iou)
    except (TypeError, ValueError):
        return None, None


def _candidate_receipt(
    *,
    route_id: str,
    route_row: Mapping[str, Any],
    route_evidence: Mapping[str, Any],
    assignment: Mapping[str, Any],
    owners: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Project one route to the frozen owner, safety, and geometry semantics."""

    parser = _mapping(route_evidence.get("parser"), f"{route_id}.parser")
    raw_receipts = assignment.get("row_assignment_receipts", [])
    if not isinstance(raw_receipts, list) or any(
        not isinstance(item, Mapping) for item in raw_receipts
    ):
        raise AssemblyError(f"{route_id}.row_assignment_receipts must be objects")
    receipts = sorted(
        (dict(item) for item in raw_receipts),
        key=lambda item: (
            _int_count(item.get("generated_row_index", 0), "generated_row_index"),
            str(item.get("prediction_id", "")),
        ),
    )
    row_counts = _mapping(assignment.get("row_counts", {}), f"{route_id}.row_counts")
    statuses = [str(item.get("entity_status", "")) for item in receipts]
    duplicate_count = sum(
        status in {"duplicate", "duplicate_owner"} for status in statuses
    )
    malformed_count = max(
        _int_count(assignment.get("malformed_row_count", 0), "malformed_row_count"),
        _int_count(row_counts.get("malformed", 0), "row_counts.malformed"),
    )
    confirmed_false_count = _int_count(
        row_counts.get("confirmed_false", row_counts.get("unsupported_hallucination", 0)),
        "row_counts.confirmed_false",
    ) + sum(
        status in {"confirmed_false", "confirmed_harmful", "unsupported_hallucination"}
        for status in statuses
    )
    semantic_error_count = _int_count(
        row_counts.get("semantic_error", 0), "row_counts.semantic_error"
    )
    unresolved_statuses = {
        "ambiguous_matched_review",
        "semantic_mismatch_unresolved",
        "unresolved_pending_crop_review",
        "uncertain",
        "",
    }
    unknown_count = sum(status in unresolved_statuses for status in statuses)
    unknown_count = max(
        unknown_count,
        _int_count(row_counts.get("unresolved", 0), "row_counts.unresolved"),
    )
    parser_drop_count = _int_count(
        parser.get("dropped_prediction_count", 0), "parser.dropped_prediction_count"
    )
    parse_status = str(parser.get("parse_status", ""))
    reported_valid_count = parser.get("valid_prediction_count", len(receipts))
    valid_count = _int_count(reported_valid_count, "parser.valid_prediction_count")
    evaluated_count = min(valid_count, 16)

    owner_ids = {str(owner.get("owner_id")) for owner in owners}
    ordered_owner_sequence: list[str] = []
    row_geometry: list[dict[str, Any]] = []
    unresolved_identity_count = 0
    geometry_untrusted_count = 0
    ambiguity_count = sum(status == "ambiguous_matched_review" for status in statuses)
    for row in receipts:
        status = str(row.get("entity_status", ""))
        owner_id, iou = _row_owner(row)
        resolved = (
            status in {"verified_owner", "duplicate", "duplicate_owner"}
            and owner_id in owner_ids
            and iou is not None
            and iou >= 0.5
        )
        if not resolved:
            unresolved_identity_count += 1
        elif owner_id is not None and owner_id not in ordered_owner_sequence:
            ordered_owner_sequence.append(owner_id)

        if status == "verified_owner":
            existing = geometry_eligibility_receipt(row, owners)
            geometry_trusted = bool(existing["geometry_trusted"])
            geometry_reason = str(existing["reason"])
        elif status in {"duplicate", "duplicate_owner"}:
            geometry_trusted = bool(resolved and iou is not None and iou >= GEOMETRY_IOU_THRESHOLD)
            geometry_reason = (
                "confirmed_duplicate_iou_at_least_0.75"
                if geometry_trusted
                else "confirmed_duplicate_iou_below_0.75_or_unresolved"
            )
        else:
            geometry_trusted = False
            geometry_reason = f"entity_status:{status or 'missing'}"
        if not geometry_trusted:
            geometry_untrusted_count += 1
        row_geometry.append(
            {
                "generated_row_index": row.get("generated_row_index"),
                "prediction_id": row.get("prediction_id"),
                "entity_status": status,
                "owner_id": owner_id,
                "intersection_over_union": iou,
                "entity_owner_iou_at_least_0_50": resolved,
                "trusted_exact_geometry": geometry_trusted,
                "geometry_reason": geometry_reason,
            }
        )

    if len(receipts) != evaluated_count:
        unresolved_identity_count += abs(evaluated_count - len(receipts))
    unknown_count = max(unknown_count, unresolved_identity_count)
    uncovered_owner_ids = sorted(owner_ids - set(ordered_owner_sequence))
    source_provenance = route_row.get("_source_b16_provenance")
    source_status = (
        str(source_provenance.get("status", ""))
        if isinstance(source_provenance, Mapping)
        else None
    )
    stop_reason = str(
        route_evidence.get("stop_reason", route_row.get("stop_reason", ""))
    )
    decode_mode = str(
        route_evidence.get("decode_mode", route_row.get("decode_mode", ""))
    )
    sampled_provenance = route_row.get("_sampled_b16_provenance")
    sampled_status = (
        str(sampled_provenance.get("status", ""))
        if isinstance(sampled_provenance, Mapping)
        else None
    )
    if route_id == "source-b16":
        natural_semantic_stop = source_status == "accepted_natural_end"
    else:
        natural_semantic_stop = sampled_status == "accepted_natural_end"
    safety_counts = {
        "duplicate": duplicate_count,
        "malformed": malformed_count,
        "confirmed_false": confirmed_false_count,
        "semantic_error": semantic_error_count,
        "unknown": unknown_count,
        "premature_stop": int(natural_semantic_stop and bool(uncovered_owner_ids)),
    }
    exclusion_reasons: list[str] = []
    if route_id != "source-b16" and sampled_status not in {
        "accepted_budget",
        "accepted_natural_end",
    }:
        exclusion_reasons.append(
            f"sampled_b16_projection_status:{sampled_status or 'missing'}"
        )
    if parse_status not in {"accepted", "accepted_with_drops"}:
        exclusion_reasons.append(f"parser_status:{parse_status or 'missing'}")
    if malformed_count:
        exclusion_reasons.append(
            "parser_drop_or_malformed_before_b16"
            if parser_drop_count
            else "malformed_row_before_b16"
        )
    if ambiguity_count:
        exclusion_reasons.append("ambiguous_owner")
    if semantic_error_count:
        exclusion_reasons.append("semantic_error")
    if confirmed_false_count:
        exclusion_reasons.append("confirmed_false")
    if unknown_count:
        exclusion_reasons.append("unknown_row")
    if geometry_untrusted_count:
        exclusion_reasons.append("geometry_untrusted")

    projection_provenance = (
        source_provenance if route_id == "source-b16" else sampled_provenance
    )
    if not isinstance(projection_provenance, Mapping):
        raise AssemblyError(f"{route_id} lacks B16 projection provenance")
    projected_ids = route_row.get("generated_token_ids")
    if not isinstance(projected_ids, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in projected_ids
    ):
        raise AssemblyError(f"{route_id} lacks projected generated token IDs")
    token_hash = str(route_row.get("generated_token_ids_sha256", ""))
    provenance_projected_hash = str(
        projection_provenance.get("projected_token_ids_sha256", "")
    )
    provenance_projected_count = _int_count(
        projection_provenance.get("projected_token_count"),
        f"{route_id}.projected_token_count",
    )
    raw_hash = str(
        projection_provenance.get("raw_generated_token_ids_sha256", "")
    )
    raw_count = _int_count(
        projection_provenance.get("raw_generated_token_count"),
        f"{route_id}.raw_generated_token_count",
    )
    if (
        len(token_hash) != 64
        or token_ids_sha256(projected_ids) != token_hash
        or provenance_projected_hash != token_hash
        or provenance_projected_count != len(projected_ids)
        or len(raw_hash) != 64
        or raw_count < provenance_projected_count
    ):
        raise AssemblyError(f"{route_id} B16 raw/projected token provenance mismatch")
    return {
        "candidate_id": route_id,
        "trajectory_id": route_id,
        "decode_mode": decode_mode,
        "sample_index": route_row.get("sample_index"),
        "seed": route_evidence.get("seed", route_row.get("seed")),
        "stop_reason": stop_reason,
        "source_b16_projection_status": source_status,
        "sampled_b16_projection_status": sampled_status,
        "natural_semantic_stop": natural_semantic_stop,
        "uncovered_trusted_owner_ids_at_stop": uncovered_owner_ids,
        "generated_token_ids_sha256": token_hash,
        "generated_token_count": len(projected_ids),
        "projected_generated_token_ids_sha256": token_hash,
        "projected_generated_token_count": len(projected_ids),
        "raw_generated_token_ids_sha256": raw_hash,
        "raw_generated_token_count": raw_count,
        "parse_status": parse_status,
        "parser_drop_count": parser_drop_count,
        "parser_drop_or_malformed_before_b16_count": malformed_count,
        "evaluated_complete_row_count": evaluated_count,
        "annotation_owner_set_iou_threshold": 0.5,
        "final_owner_ids": sorted(ordered_owner_sequence),
        "ordered_unique_owner_sequence": ordered_owner_sequence,
        "first_owner_id": ordered_owner_sequence[0] if ordered_owner_sequence else None,
        "safety_counts": safety_counts,
        "duplicate_count": duplicate_count,
        "malformed_count": malformed_count,
        "confirmed_false_count": confirmed_false_count,
        "semantic_error_count": semantic_error_count,
        "unknown_count": unknown_count,
        "premature_stop_count": safety_counts["premature_stop"],
        "ambiguity_count": ambiguity_count,
        "geometry": {
            "trusted_exact_geometry_iou_threshold": GEOMETRY_IOU_THRESHOLD,
            "all_evaluated_rows_trusted": geometry_untrusted_count == 0,
            "untrusted_row_count": geometry_untrusted_count,
            "rows": row_geometry,
        },
        "eligible": not exclusion_reasons,
        "exclusion_reasons": exclusion_reasons,
        "exact_token_representative": True,
        "exact_token_duplicate_of": None,
    }


def _safety_no_worse(higher: Mapping[str, int], lower: Mapping[str, int]) -> bool:
    return all(int(higher[field]) <= int(lower[field]) for field in SAFETY_FIELDS)


def _class_id(owner_ids: Sequence[str]) -> str:
    return f"owner-set-{sha256_json(list(owner_ids))[:16]}"


def _exact_token_semantic_signature(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Remove route-local identifiers from exact-token semantic comparison."""

    geometry = _mapping(candidate.get("geometry", {}), "candidate.geometry")
    rows = geometry.get("rows", [])
    if not isinstance(rows, list):
        raise AssemblyError("candidate.geometry.rows must be a list")
    semantic_rows = []
    for raw in rows:
        row = _mapping(raw, "candidate.geometry.rows[]")
        semantic_rows.append(
            {
                key: copy.deepcopy(row.get(key))
                for key in (
                    "generated_row_index",
                    "entity_status",
                    "owner_id",
                    "intersection_over_union",
                    "entity_owner_iou_at_least_0_50",
                    "trusted_exact_geometry",
                    "geometry_reason",
                )
            }
        )
    return {
        "eligible": bool(candidate["eligible"]),
        "exclusion_reasons": copy.deepcopy(candidate["exclusion_reasons"]),
        "final_owner_ids": copy.deepcopy(candidate["final_owner_ids"]),
        "ordered_unique_owner_sequence": copy.deepcopy(
            candidate["ordered_unique_owner_sequence"]
        ),
        "first_owner_id": candidate["first_owner_id"],
        "safety_counts": copy.deepcopy(candidate["safety_counts"]),
        "geometry": {
            "trusted_exact_geometry_iou_threshold": geometry.get(
                "trusted_exact_geometry_iou_threshold"
            ),
            "all_evaluated_rows_trusted": geometry.get(
                "all_evaluated_rows_trusted"
            ),
            "untrusted_row_count": geometry.get("untrusted_row_count"),
            "rows": semantic_rows,
        },
    }


def _analyze_image_candidates(
    image_id: str, candidates: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Group candidates, derive every set edge, and expose both alias views."""

    normalized = [copy.deepcopy(dict(item)) for item in candidates]
    if len({str(item["candidate_id"]) for item in normalized}) != len(normalized):
        raise AssemblyError(f"image {image_id} duplicates candidate_id")
    by_hash: dict[str, list[dict[str, Any]]] = {}
    for candidate in normalized:
        by_hash.setdefault(str(candidate["generated_token_ids_sha256"]), []).append(candidate)
    representatives: list[dict[str, Any]] = []
    exact_token_duplicate_count = 0
    for token_hash in sorted(by_hash):
        aliases = sorted(
            by_hash[token_hash], key=lambda item: _route_sort_key(str(item["candidate_id"]))
        )
        reference_signature = sha256_json(
            _exact_token_semantic_signature(aliases[0])
        )
        if any(
            sha256_json(_exact_token_semantic_signature(item))
            != reference_signature
            for item in aliases[1:]
        ):
            raise AssemblyError(
                f"image {image_id} exact-token aliases disagree semantically: {token_hash}"
            )
        representative = aliases[0]
        representative["exact_token_alias_candidate_ids"] = [
            str(item["candidate_id"]) for item in aliases
        ]
        representative["exact_token_alias_count"] = len(aliases)
        representatives.append(representative)
        for duplicate in aliases[1:]:
            duplicate["exact_token_representative"] = False
            duplicate["exact_token_duplicate_of"] = str(representative["candidate_id"])
        exact_token_duplicate_count += len(aliases) - 1

    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for candidate in representatives:
        if candidate["eligible"]:
            key = tuple(str(item) for item in candidate["final_owner_ids"])
            grouped.setdefault(key, []).append(candidate)
    classes: list[dict[str, Any]] = []
    for owner_key in sorted(grouped, key=lambda value: (len(value), value)):
        aliases = sorted(
            grouped[owner_key], key=lambda item: _route_sort_key(str(item["candidate_id"]))
        )
        first_owners = sorted(
            {str(item["first_owner_id"]) for item in aliases if item["first_owner_id"] is not None}
        )
        orders = sorted(
            {tuple(str(value) for value in item["ordered_unique_owner_sequence"]) for item in aliases}
        )
        minimum_safety = {
            field: min(int(item["safety_counts"][field]) for item in aliases)
            for field in SAFETY_FIELDS
        }
        maximum_safety = {
            field: max(int(item["safety_counts"][field]) for item in aliases)
            for field in SAFETY_FIELDS
        }
        raw_alias_count = sum(int(item["exact_token_alias_count"]) for item in aliases)
        classes.append(
            {
                "class_id": _class_id(owner_key),
                "final_owner_ids": list(owner_key),
                "owner_count": len(owner_key),
                "minimum_alias_safety_counts": minimum_safety,
                "maximum_alias_safety_counts": maximum_safety,
                "alias_safety_vectors": [
                    {
                        "candidate_id": str(item["candidate_id"]),
                        "safety_counts": copy.deepcopy(item["safety_counts"]),
                    }
                    for item in aliases
                ],
                "exact_serialization_count": len(aliases),
                "exact_token_alias_count": raw_alias_count - len(aliases),
                "candidate_ids": [str(item["candidate_id"]) for item in aliases],
                "distinct_row_order_alias_count": len(orders),
                "ordered_unique_owner_sequences": [list(item) for item in orders],
                "distinct_first_owner_count": len(first_owners),
                "first_owner_ids": first_owners,
                "natural_alias_multiple_first_owner": len(aliases) >= 2
                and len(first_owners) >= 2,
            }
        )

    edges: list[dict[str, Any]] = []
    inclusion_diagnostics: list[dict[str, Any]] = []
    incomparable: list[dict[str, Any]] = []
    for left_index, left in enumerate(classes):
        left_set = set(left["final_owner_ids"])
        for right in classes[left_index + 1 :]:
            right_set = set(right["final_owner_ids"])
            if left_set > right_set:
                higher, lower = left, right
            elif right_set > left_set:
                higher, lower = right, left
            else:
                incomparable.append(
                    {
                        "left_class_id": left["class_id"],
                        "right_class_id": right["class_id"],
                        "left_only_owner_ids": sorted(left_set - right_set),
                        "right_only_owner_ids": sorted(right_set - left_set),
                    }
                )
                continue
            existential_safe_alias_pair_count = sum(
                _safety_no_worse(
                    high_alias["safety_counts"], low_alias["safety_counts"]
                )
                for high_alias in higher["alias_safety_vectors"]
                for low_alias in lower["alias_safety_vectors"]
            )
            alias_pair_count = (
                len(higher["alias_safety_vectors"])
                * len(lower["alias_safety_vectors"])
            )
            universal_safety = _safety_no_worse(
                higher["maximum_alias_safety_counts"],
                lower["minimum_alias_safety_counts"],
            )
            diagnostic = {
                "higher_class_id": higher["class_id"],
                "lower_class_id": lower["class_id"],
                "alias_pair_count": alias_pair_count,
                "existential_safe_alias_pair_count": existential_safe_alias_pair_count,
                "universal_safety": universal_safety,
            }
            inclusion_diagnostics.append(diagnostic)
            if not universal_safety:
                continue
            lower_first = set(lower["first_owner_ids"])
            higher_first = set(higher["first_owner_ids"])
            orphaned = sorted(lower_first - higher_first)
            edges.append(
                {
                    "higher_class_id": higher["class_id"],
                    "lower_class_id": lower["class_id"],
                    "higher_owner_ids": higher["final_owner_ids"],
                    "lower_owner_ids": lower["final_owner_ids"],
                    "safety_no_worse": True,
                    "universal_safety": True,
                    "alias_pair_count": alias_pair_count,
                    "existential_safe_alias_pair_count": existential_safe_alias_pair_count,
                    "admission_eligible": not orphaned,
                    "orphaned_lower_first_owner_ids": orphaned,
                }
            )
    edges.sort(key=lambda item: (item["higher_class_id"], item["lower_class_id"]))
    inclusion_diagnostics.sort(
        key=lambda item: (item["higher_class_id"], item["lower_class_id"])
    )
    incomparable.sort(key=lambda item: (item["left_class_id"], item["right_class_id"]))

    dominated = {str(edge["lower_class_id"]) for edge in edges}
    frontier = [item for item in classes if str(item["class_id"]) not in dominated]
    frontier_ids = [str(item["class_id"]) for item in frontier]
    frontier_serializations = sum(int(item["exact_serialization_count"]) for item in frontier)
    frontier_first = sorted(
        {str(owner) for item in frontier for owner in item["first_owner_ids"]}
    )
    frontier_orders = {
        tuple(str(owner) for owner in order)
        for item in frontier
        for order in item["ordered_unique_owner_sequences"]
    }
    alias_qualified_frontier_classes = [
        str(item["class_id"])
        for item in frontier
        if item["natural_alias_multiple_first_owner"]
    ]
    admissible_higher_classes = {
        str(item["higher_class_id"])
        for item in edges
        if item["admission_eligible"]
    }
    primary_qualifying_classes = sorted(
        set(alias_qualified_frontier_classes) & admissible_higher_classes
    )
    has_edge = bool(admissible_higher_classes)
    image_level_alias = frontier_serializations >= 2 and len(frontier_first) >= 2
    per_class_alias = bool(alias_qualified_frontier_classes)
    normalized.sort(key=lambda item: _route_sort_key(str(item["candidate_id"])))
    return {
        "schema_version": SCHEMA_VERSION,
        "image_id": str(image_id),
        "candidate_count": len(normalized),
        "eligible_candidate_count": sum(bool(item["eligible"]) for item in normalized),
        "excluded_candidate_count": sum(not bool(item["eligible"]) for item in normalized),
        "exact_token_duplicate_count": exact_token_duplicate_count,
        "candidates": normalized,
        "outcome_class_count": len(classes),
        "outcome_classes": classes,
        "strict_semantic_edge_count": len(edges),
        "semantic_edges": edges,
        "strict_owner_set_inclusion_comparison_count": len(inclusion_diagnostics),
        "strict_owner_set_inclusion_diagnostics": inclusion_diagnostics,
        "existential_safe_alias_pair_count": sum(
            int(item["existential_safe_alias_pair_count"])
            for item in inclusion_diagnostics
        ),
        "admissible_strict_edge_count": sum(bool(item["admission_eligible"]) for item in edges),
        "first_owner_orphaned_edge_count": sum(not bool(item["admission_eligible"]) for item in edges),
        "incomparable_owner_exchange_pair_count": len(incomparable),
        "incomparable_owner_exchange_pairs": incomparable,
        "nondominated_outcome_class_ids": frontier_ids,
        "positive_frontier_class_ids": frontier_ids,
        "positive_frontier_alias_metrics": {
            "class_count": len(frontier),
            "exact_serialization_count": frontier_serializations,
            "distinct_row_order_alias_count": len(frontier_orders),
            "distinct_first_owner_count": len(frontier_first),
            "first_owner_ids": frontier_first,
            "per_class": [
                {
                    "class_id": item["class_id"],
                    "exact_serialization_count": item["exact_serialization_count"],
                    "exact_token_alias_count": item["exact_token_alias_count"],
                    "distinct_row_order_alias_count": item["distinct_row_order_alias_count"],
                    "distinct_first_owner_count": item["distinct_first_owner_count"],
                    "first_owner_ids": item["first_owner_ids"],
                }
                for item in frontier
            ],
        },
        "singleton_maximal_exact_serialization": frontier_serializations == 1,
        "fully_adjudicable": len(normalized) == EXPECTED_CANDIDATES_PER_IMAGE
        and all(bool(item["eligible"]) for item in normalized),
        "has_at_least_two_eligible_candidates": sum(
            bool(item["eligible"]) for item in normalized
        )
        >= 2,
        "admission": {
            "has_admissible_strict_edge": has_edge,
            "image_level_frontier_natural_alias": image_level_alias,
            "per_frontier_class_natural_alias": per_class_alias,
            "alias_qualified_frontier_class_ids": alias_qualified_frontier_classes,
            "primary_qualifying_frontier_class_ids": primary_qualifying_classes,
            "image_level_admitted": has_edge and image_level_alias,
            "per_frontier_class_alias_only": per_class_alias,
            "per_frontier_class_admitted": bool(primary_qualifying_classes),
            "primary_natural_alias_admitted": bool(primary_qualifying_classes),
        },
    }


def _image_candidates(
    image_id: str, adapter: Mapping[str, Any], *, reverse_input: bool
) -> list[dict[str, Any]]:
    result = _mapping(adapter["image_results"][image_id], f"image_results[{image_id}]")
    evidence = _mapping(result.get("trajectory_evidence"), "trajectory_evidence")
    budgets = result.get("budgets")
    if not isinstance(budgets, list) or len(budgets) != 1 or budgets[0].get("budget") != 16:
        raise AssemblyError(f"image {image_id} lacks its exact B16 assignment")
    assignments = _mapping(budgets[0].get("trajectory_assignments"), "trajectory_assignments")
    owners = result.get("owners")
    if not isinstance(owners, list):
        raise AssemblyError(f"image {image_id} lacks owner records")
    route_ids = ["source-b16", *(f"sample-{index:02d}" for index in range(16))]
    if reverse_input:
        route_ids.reverse()
    candidates: list[dict[str, Any]] = []
    for route_id in route_ids:
        if route_id == "source-b16":
            row = adapter["source_rows"].get((image_id, 0))
        else:
            row = adapter["sampled_rows"].get((image_id, int(route_id[-2:])))
        if not isinstance(row, Mapping):
            raise AssemblyError(f"image {image_id} lacks {route_id}")
        candidates.append(
            _candidate_receipt(
                route_id=route_id,
                route_row=row,
                route_evidence=_mapping(evidence.get(route_id), f"evidence[{route_id}]"),
                assignment=_mapping(assignments.get(route_id), f"assignment[{route_id}]"),
                owners=owners,
            )
        )
    return candidates


def _histogram(values: Iterable[int]) -> dict[str, int]:
    return {
        str(key): value
        for key, value in sorted(Counter(int(item) for item in values).items())
    }


def _summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    reason_counts: Counter[str] = Counter()
    for record in records:
        for candidate in record["candidates"]:
            reason_counts.update(str(item) for item in candidate["exclusion_reasons"])
    image_level_ids = sorted(
        (str(item["image_id"]) for item in records if item["admission"]["image_level_admitted"]),
        key=int,
    )
    per_class_ids = sorted(
        (
            str(item["image_id"])
            for item in records
            if item["admission"]["per_frontier_class_admitted"]
        ),
        key=int,
    )
    fully_adjudicable = [item for item in records if item["fully_adjudicable"]]
    at_least_two = [
        item for item in records if item["has_at_least_two_eligible_candidates"]
    ]
    censored = [item for item in records if not item["fully_adjudicable"]]
    censored_image_reasons: Counter[str] = Counter()
    for record in censored:
        censored_image_reasons.update(
            {
                str(reason)
                for candidate in record["candidates"]
                for reason in candidate["exclusion_reasons"]
            }
        )

    def structural_scope(values: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        return {
            "image_count": len(values),
            "primary_admitted_image_count": sum(
                bool(item["admission"]["primary_natural_alias_admitted"])
                for item in values
            ),
            "no_admissible_universal_edge_image_count": sum(
                not bool(item["admission"]["has_admissible_strict_edge"])
                for item in values
            ),
            "no_same_class_first_owner_diverse_alias_image_count": sum(
                not bool(item["admission"]["per_frontier_class_alias_only"])
                for item in values
            ),
            "alias_class_not_higher_side_of_admissible_edge_image_count": sum(
                bool(item["admission"]["per_frontier_class_alias_only"])
                and not bool(item["admission"]["primary_natural_alias_admitted"])
                for item in values
            ),
        }
    candidate_count = sum(int(item["candidate_count"]) for item in records)
    eligible_count = sum(int(item["eligible_candidate_count"]) for item in records)
    excluded_count = sum(int(item["excluded_candidate_count"]) for item in records)
    edge_count = sum(int(item["strict_semantic_edge_count"]) for item in records)
    admitted_edges = sum(int(item["admissible_strict_edge_count"]) for item in records)
    orphan_edges = sum(int(item["first_owner_orphaned_edge_count"]) for item in records)
    reconciliation = {
        "candidate_partition": eligible_count + excluded_count == candidate_count,
        "edge_partition": admitted_edges + orphan_edges == edge_count,
        "unique_image_ids": len({str(item["image_id"]) for item in records}) == len(records),
        "record_candidate_partitions": all(
            int(item["eligible_candidate_count"]) + int(item["excluded_candidate_count"])
            == int(item["candidate_count"])
            for item in records
        ),
        "record_edge_partitions": all(
            int(item["admissible_strict_edge_count"])
            + int(item["first_owner_orphaned_edge_count"])
            == int(item["strict_semantic_edge_count"])
            for item in records
        ),
    }
    reconciliation["passed"] = all(reconciliation.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "terminal_status": "completed",
        "image_count": len(records),
        "candidate_count": candidate_count,
        "eligible_candidate_count": eligible_count,
        "excluded_candidate_count": excluded_count,
        "exact_token_duplicate_count": sum(int(item["exact_token_duplicate_count"]) for item in records),
        "outcome_class_count": sum(int(item["outcome_class_count"]) for item in records),
        "strict_semantic_edge_count": edge_count,
        "admissible_strict_edge_count": admitted_edges,
        "first_owner_orphaned_edge_count": orphan_edges,
        "incomparable_owner_exchange_pair_count": sum(
            int(item["incomparable_owner_exchange_pair_count"]) for item in records
        ),
        "fully_adjudicable_image_count": len(fully_adjudicable),
        "at_least_two_eligible_candidate_image_count": len(at_least_two),
        "censored_image_count": len(censored),
        "singleton_maximal_exact_serialization_image_count": sum(
            bool(item["singleton_maximal_exact_serialization"]) for item in records
        ),
        "histograms": {
            "eligible_candidates_per_image": _histogram(item["eligible_candidate_count"] for item in records),
            "outcome_classes_per_image": _histogram(item["outcome_class_count"] for item in records),
            "strict_semantic_edges_per_image": _histogram(item["strict_semantic_edge_count"] for item in records),
            "frontier_classes_per_image": _histogram(
                len(item["positive_frontier_class_ids"]) for item in records
            ),
        },
        "excluded_candidate_reason_counts": dict(sorted(reason_counts.items())),
        "natural_alias_admission_variants": {
            "image_level_frontier": {
                "image_count": len(image_level_ids),
                "image_ids": image_level_ids,
            },
            "per_frontier_class": {
                "image_count": len(per_class_ids),
                "image_ids": per_class_ids,
                "predicate": (
                    "same nondominated owner-set class has at least two unique token "
                    "serializations, at least two first owners, and is the higher side "
                    "of an admissible universal-safety strict edge"
                ),
            },
        },
        "failure_cause_split": {
            "censoring": {
                "image_count": len(censored),
                "candidate_exclusion_reason_image_counts": dict(
                    sorted(censored_image_reasons.items())
                ),
                "interpretation": "not evidence of structural signal absence",
            },
            "structural_scarcity": {
                "fully_adjudicable_images": structural_scope(fully_adjudicable),
                "images_with_at_least_two_eligible_candidates": structural_scope(
                    at_least_two
                ),
            },
        },
        "strongest_natural_alias_view": "per_frontier_class",
        "feasibility": {
            "required_image_count": FEASIBILITY_IMAGE_COUNT,
            "observed_image_count": len(per_class_ids),
            "supports_256_image_screen": len(per_class_ids) >= FEASIBILITY_IMAGE_COUNT,
            "training_promotion_authorized": False,
            "bounded_conclusion": (
                "supports_separate_training_design"
                if len(per_class_ids) >= FEASIBILITY_IMAGE_COUNT
                else "does_not_support_256_image_screen_under_frozen_admission"
            ),
            "causal_failure_diagnosis_chosen": False,
        },
        "aggregate_reconciliation": reconciliation,
    }


def analyze_admission_census(
    adapter: Mapping[str, Any], train_ids: Iterable[str], *, reverse_input: bool = False
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ids = sorted((str(item) for item in train_ids), key=int, reverse=reverse_input)
    records = [
        _analyze_image_candidates(
            image_id,
            _image_candidates(image_id, adapter, reverse_input=reverse_input),
        )
        for image_id in ids
    ]
    records.sort(key=lambda item: int(item["image_id"]))
    return records, _summarize_records(records)


def _validate_summary_reconciliation(
    records: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]
) -> None:
    expected = _summarize_records(records)
    if dict(summary) != expected:
        raise AssemblyError("summary does not exactly reconcile to image census records")
    image_ids = [str(item["image_id"]) for item in records]
    if len(image_ids) != len(set(image_ids)):
        raise AssemblyError("image census record IDs are not unique")
    image_set = set(image_ids)
    variants = _mapping(
        summary.get("natural_alias_admission_variants"), "admission variants"
    )
    for name in ("image_level_frontier", "per_frontier_class"):
        variant = _mapping(variants.get(name), f"admission variant {name}")
        ids = [str(item) for item in variant.get("image_ids", [])]
        if (
            len(ids) != len(set(ids))
            or not set(ids) <= image_set
            or variant.get("image_count") != len(ids)
        ):
            raise AssemblyError(f"admission variant {name} ID/count reconciliation failed")
    histograms = _mapping(summary.get("histograms"), "summary histograms")
    if any(
        sum(_int_count(value, f"histogram {name}") for value in _mapping(histogram, name).values())
        != len(records)
        for name, histogram in histograms.items()
    ):
        raise AssemblyError("summary histogram totals do not equal image count")
    if (
        _int_count(summary.get("fully_adjudicable_image_count"), "fully adjudicable")
        + _int_count(summary.get("censored_image_count"), "censored")
        != len(records)
        or _mapping(summary.get("feasibility"), "feasibility").get(
            "observed_image_count"
        )
        != _mapping(variants.get("per_frontier_class"), "per frontier class").get(
            "image_count"
        )
    ):
        raise AssemblyError("summary population or feasibility reconciliation failed")


def _validate_serialized_output_readback(
    *,
    census_path: Path,
    summary_path: Path,
    records: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    raw_lines = census_path.read_text(encoding="utf-8").splitlines()
    if any(not line.strip() for line in raw_lines):
        raise AssemblyError("serialized image census contains blank rows")
    read_records = [json.loads(line) for line in raw_lines]
    read_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if read_records != list(records) or read_summary != dict(summary):
        raise AssemblyError("serialized census/summary readback differs from memory")
    _validate_summary_reconciliation(read_records, read_summary)


def _validate_success_receipt_readback(
    *,
    receipt_path: Path,
    expected_receipt: Mapping[str, Any],
    output_root: Path,
) -> str:
    read_receipt = _mapping(
        json.loads(receipt_path.read_text(encoding="utf-8")), "success receipt"
    )
    if dict(read_receipt) != dict(expected_receipt):
        raise AssemblyError("serialized success receipt differs from memory")
    if (
        read_receipt.get("schema_version") != SCHEMA_VERSION
        or read_receipt.get("terminal_status") != "completed"
    ):
        raise AssemblyError("serialized success receipt has invalid schema or status")
    integrity = _mapping(
        read_receipt.get("artifact_integrity_checks"), "receipt integrity checks"
    )
    if integrity.get("serialized_success_receipt_readback") is not True:
        raise AssemblyError("success receipt lacks its readback integrity declaration")
    output_hashes = _mapping(
        read_receipt.get("output_hashes"), "receipt output hashes"
    )
    expected_artifacts = {
        "image-census.jsonl",
        "summary.json",
        "source-snapshot/manifest.json",
    }
    if set(output_hashes) != expected_artifacts:
        raise AssemblyError("success receipt output-hash inventory is not exact")
    for relative in sorted(expected_artifacts):
        digest = output_hashes.get(relative)
        if not isinstance(digest, str) or len(digest) != 64:
            raise AssemblyError(f"success receipt has invalid artifact hash: {relative}")
        try:
            int(digest, 16)
        except ValueError as exc:
            raise AssemblyError(
                f"success receipt has invalid artifact hash: {relative}"
            ) from exc
        if sha256_file(output_root / relative) != digest:
            raise AssemblyError(f"success receipt artifact hash does not reproduce: {relative}")
    snapshot = _mapping(read_receipt.get("source_snapshot"), "source snapshot receipt")
    if snapshot.get("manifest_sha256") != output_hashes[
        "source-snapshot/manifest.json"
    ]:
        raise AssemblyError("source snapshot manifest hash is not reconciled")
    census_lines = (output_root / "image-census.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    records = [json.loads(line) for line in census_lines if line.strip()]
    summary = _mapping(
        json.loads((output_root / "summary.json").read_text(encoding="utf-8")),
        "serialized summary",
    )
    row_counts = _mapping(read_receipt.get("row_counts"), "receipt row counts")
    if row_counts != {
        "image_census": len(records),
        "candidate": summary.get("candidate_count"),
        "eligible_candidate": summary.get("eligible_candidate_count"),
        "excluded_candidate": summary.get("excluded_candidate_count"),
    }:
        raise AssemblyError("success receipt row counts do not reconcile")
    determinism = _mapping(read_receipt.get("determinism"), "receipt determinism")
    records_hash = sha256_json(records)
    summary_hash = sha256_json(summary)
    if determinism != {
        "forward_records_sha256": records_hash,
        "reversed_records_sha256": records_hash,
        "forward_summary_sha256": summary_hash,
        "reversed_summary_sha256": summary_hash,
    }:
        raise AssemblyError("success receipt determinism hashes do not reconcile")
    return sha256_file(receipt_path)


def _git_identity(repo: Path, snapshotted_paths: Sequence[str]) -> dict[str, Any]:
    """Record commit and dirty identity only for the task-scoped source set."""

    def run(*arguments: str) -> bytes:
        return subprocess.run(
            ["git", "-C", str(repo), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout

    scope = sorted(str(item) for item in snapshotted_paths)
    status = run("status", "--porcelain=v1", "--untracked-files=all", "--", *scope)
    tracked_diff = run("diff", "--binary", "HEAD", "--", *scope)
    return {
        "commit": run("rev-parse", "HEAD").decode("ascii").strip(),
        "dirty": bool(status),
        "scope": scope,
        "scoped_status_sha256": hashlib.sha256(status).hexdigest(),
        "scoped_tracked_dirty_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        "scoped_status_lines": status.decode("utf-8").splitlines(),
    }


def _validate_panel_manifests(
    root: Path, *, mode: str, expected_pool_ids: Iterable[str]
) -> dict[str, Any]:
    """Bind every worker manifest and batch without inspecting route semantics."""

    root = root.expanduser().resolve(strict=True)
    expected_ids = {str(item) for item in expected_pool_ids}
    manifests = sorted(root.glob("worker-*-of-*/manifest.json"))
    if len(manifests) != EXPECTED_MANIFEST_COUNT_PER_ROOT:
        raise AssemblyError(
            f"{mode} root has {len(manifests)} manifests, expected 8"
        )
    manifest_receipts: list[dict[str, Any]] = []
    batch_receipts: list[dict[str, Any]] = []
    worker_indices: set[int] = set()
    inventory_image_ids: set[str] = set()
    bound_paths: set[Path] = set()
    for manifest_path in manifests:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = _mapping(payload, f"manifest {manifest_path}")
        worker_index = _int_count(manifest.get("worker_index"), "worker_index")
        if (
            worker_index in worker_indices
            or manifest.get("worker_count") != EXPECTED_MANIFEST_COUNT_PER_ROOT
            or manifest.get("status")
            not in {"completed", "completed_with_source_b16_ineligible"}
        ):
            raise AssemblyError(f"invalid {mode} worker manifest: {manifest_path}")
        worker_indices.add(worker_index)
        batches = manifest.get("batches")
        if not isinstance(batches, list):
            raise AssemblyError(f"{manifest_path} lacks batches")
        manifest_receipts.append(
            {"path": str(manifest_path), "sha256": sha256_file(manifest_path)}
        )
        for raw_batch in batches:
            batch = _mapping(raw_batch, f"{manifest_path}.batch")
            image_ids = batch.get("image_ids")
            if not isinstance(image_ids, list) or len(image_ids) != 16:
                raise AssemblyError(f"{manifest_path} batch lacks 16 image IDs")
            canonical_ids = [str(item) for item in image_ids]
            if inventory_image_ids & set(canonical_ids):
                raise AssemblyError(f"{mode} manifest image inventory overlaps")
            inventory_image_ids.update(canonical_ids)
            artifacts = _mapping(batch.get("artifacts"), "batch.artifacts")
            entry = _mapping(artifacts.get(mode), f"batch.artifacts.{mode}")
            relative_path = entry.get("path")
            expected_hash = entry.get("sha256")
            if not isinstance(relative_path, str) or not isinstance(expected_hash, str):
                raise AssemblyError(f"{manifest_path} has an invalid batch artifact entry")
            artifact_path = (manifest_path.parent / relative_path).resolve(strict=True)
            if artifact_path in bound_paths or sha256_file(artifact_path) != expected_hash:
                raise AssemblyError(f"{mode} batch hash mismatch or duplicate: {artifact_path}")
            bound_paths.add(artifact_path)
            batch_receipts.append(
                {
                    "path": str(artifact_path),
                    "sha256": expected_hash,
                    "worker_index": worker_index,
                    "batch_index": batch.get("batch_index"),
                }
            )
    if worker_indices != set(range(EXPECTED_MANIFEST_COUNT_PER_ROOT)):
        raise AssemblyError(f"{mode} worker indices are not 0 through 7")
    if len(batch_receipts) != EXPECTED_BATCH_COUNT_PER_ROOT:
        raise AssemblyError(
            f"{mode} root has {len(batch_receipts)} bound batches, expected 152"
        )
    if inventory_image_ids != expected_ids:
        missing = sorted(expected_ids - inventory_image_ids, key=int)
        alien = sorted(inventory_image_ids - expected_ids, key=int)
        raise AssemblyError(
            f"{mode} manifest image inventory differs from the frozen pool: "
            f"missing={missing[:5]}, alien={alien[:5]}"
        )
    discovered = {
        path.resolve()
        for path in root.glob(f"worker-*-of-*/{mode}-batch-*.json")
    }
    if discovered != bound_paths:
        raise AssemblyError(f"{mode} manifest/file inventory differs")
    ordered_manifest_entries = [
        {
            "relative_path": str(Path(item["path"]).relative_to(root)),
            "sha256": item["sha256"],
        }
        for item in sorted(manifest_receipts, key=lambda item: item["path"])
    ]
    ordered_inventory = sorted(inventory_image_ids, key=int)
    return {
        "root": str(root),
        "manifest_count": len(manifest_receipts),
        "batch_count": len(batch_receipts),
        "image_inventory_count": len(inventory_image_ids),
        "ordered_image_inventory_sha256": sha256_json(ordered_inventory),
        "manifest_set_sha256": sha256_json(ordered_manifest_entries),
        "manifests": manifest_receipts,
        "batches": sorted(batch_receipts, key=lambda item: item["path"]),
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _input_identity_receipt(
    *,
    candidate_pool_ids: Sequence[str],
    sampled_manifests: Mapping[str, Any],
    source_manifests: Mapping[str, Any],
    adapter: Mapping[str, Any],
) -> dict[str, Any]:
    ordered_pool = [str(item) for item in candidate_pool_ids]
    canonical_pool = sorted(set(ordered_pool), key=int)
    if len(canonical_pool) != len(ordered_pool):
        raise AssemblyError("candidate pool ordered inventory contains duplicates")
    canonical_hash = sha256_json(canonical_pool)
    sampled_hash = str(sampled_manifests.get("ordered_image_inventory_sha256", ""))
    source_hash = str(source_manifests.get("ordered_image_inventory_sha256", ""))
    sampled_count = _int_count(
        sampled_manifests.get("image_inventory_count"), "sampled image inventory count"
    )
    source_count = _int_count(
        source_manifests.get("image_inventory_count"), "Source image inventory count"
    )
    equality = (
        len(canonical_pool) == sampled_count == source_count
        and canonical_hash == sampled_hash == source_hash
    )
    if not equality:
        raise AssemblyError("candidate/sampled/Source image inventories do not agree")
    execution = copy.deepcopy(
        _mapping(adapter.get("execution_model_identity"), "execution model identity")
    )
    tokenizer = copy.deepcopy(
        _mapping(adapter.get("tokenizer_identity"), "tokenizer identity")
    )
    execution_hash = str(adapter.get("execution_model_identity_sha256", ""))
    tokenizer_hash = str(adapter.get("tokenizer_identity_sha256", ""))
    if execution_hash != sha256_json(execution):
        raise AssemblyError("execution model identity digest does not match its object")
    if tokenizer_hash != sha256_json(tokenizer):
        raise AssemblyError("tokenizer identity digest does not match its object")
    return {
        "candidate_pool_inventory": {
            "count": len(ordered_pool),
            "ordered_image_ids_sha256": sha256_json(ordered_pool),
            "canonical_image_ids_sha256": canonical_hash,
        },
        "sampled_manifest_inventory": {
            "count": sampled_count,
            "ordered_image_ids_sha256": sampled_hash,
        },
        "source_manifest_inventory": {
            "count": source_count,
            "ordered_image_ids_sha256": source_hash,
        },
        "three_way_inventory_equality": {
            "candidate_pool_equals_sampled_equals_source": equality,
            "canonical_image_ids_sha256": canonical_hash,
        },
        "shared_execution_model_identity": {
            "sha256": execution_hash,
            "identity": execution,
        },
        "shared_tokenizer_identity": {
            "sha256": tokenizer_hash,
            "identity": tokenizer,
        },
    }


def _validate_frozen_panel_bindings(
    *,
    sampled_root: Path,
    source_root: Path,
    sampled_manifests: Mapping[str, Any],
    source_manifests: Mapping[str, Any],
    adapter: Mapping[str, Any],
) -> None:
    if sampled_root != FROZEN_SAMPLED_ROOT.expanduser().resolve(strict=True):
        raise AssemblyError("sampled root is not the frozen production-v2 root")
    if source_root != FROZEN_SOURCE_ROOT.expanduser().resolve(strict=True):
        raise AssemblyError("Source root is not the frozen production-v1 root")
    if sampled_manifests.get("manifest_set_sha256") != FROZEN_SAMPLED_MANIFEST_SET_SHA256:
        raise AssemblyError("sampled manifest-set digest differs from the frozen input")
    if source_manifests.get("manifest_set_sha256") != FROZEN_SOURCE_MANIFEST_SET_SHA256:
        raise AssemblyError("Source manifest-set digest differs from the frozen input")
    if adapter.get("execution_model_identity_sha256") != FROZEN_EXECUTION_MODEL_IDENTITY_SHA256:
        raise AssemblyError("execution-model identity differs from the frozen checkpoint")
    if adapter.get("tokenizer_identity_sha256") != FROZEN_TOKENIZER_IDENTITY_SHA256:
        raise AssemblyError("tokenizer identity differs from the frozen tokenizer")


def _validate_frozen_paths(
    *,
    candidate_pool: Path,
    split_receipt: Path,
    split_output_paths: Mapping[str, Path],
    sampled_root: Path,
    source_root: Path,
    output_root: Path,
) -> None:
    expected = {
        "candidate_pool": FROZEN_CANDIDATE_POOL_PATH.resolve(strict=True),
        "split_receipt": FROZEN_SPLIT_RECEIPT_PATH.resolve(strict=True),
        "sampled_root": FROZEN_SAMPLED_ROOT.resolve(strict=True),
        "source_root": FROZEN_SOURCE_ROOT.resolve(strict=True),
        "output_root": FROZEN_OUTPUT_ROOT.resolve(strict=False),
    }
    observed = {
        "candidate_pool": candidate_pool,
        "split_receipt": split_receipt,
        "sampled_root": sampled_root,
        "source_root": source_root,
        "output_root": output_root,
    }
    for name, expected_path in expected.items():
        if observed[name] != expected_path:
            raise AssemblyError(f"{name} is not the frozen path")
    if set(split_output_paths) != set(FROZEN_SPLIT_OUTPUT_PATHS):
        raise AssemblyError("split output path names differ from the frozen split")
    for name, expected_path in FROZEN_SPLIT_OUTPUT_PATHS.items():
        if split_output_paths[name] != expected_path.resolve(strict=True):
            raise AssemblyError(f"split output {name} is not the frozen path")


def _repo_loaded_source_paths(repo: Path) -> list[str]:
    repo = repo.resolve(strict=True)
    result: set[str] = set()
    for module in tuple(sys.modules.values()):
        raw = getattr(module, "__file__", None)
        if not isinstance(raw, str) or not raw:
            continue
        path = Path(raw)
        if path.suffix in {".pyc", ".pyo"}:
            try:
                path = Path(importlib.util.source_from_cache(str(path)))
            except ValueError:
                continue
        try:
            resolved = path.resolve(strict=True)
            relative = resolved.relative_to(repo)
        except (FileNotFoundError, ValueError):
            continue
        if resolved.suffix == ".py":
            result.add(str(relative))
    return sorted(result)


def _materialize_source_snapshot(repo: Path, output_root: Path) -> dict[str, Any]:
    snapshot_root = output_root / "source-snapshot"
    entries: list[dict[str, Any]] = []
    loaded_closure = _repo_loaded_source_paths(repo)
    snapshot_paths = sorted(set(SOURCE_SNAPSHOT_PATHS) | set(loaded_closure))
    for relative in snapshot_paths:
        source = (repo / relative).resolve(strict=True)
        destination = snapshot_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        source_hash = sha256_file(source)
        snapshot_hash = sha256_file(destination)
        if source_hash != snapshot_hash:
            raise AssemblyError(f"source snapshot hash mismatch: {relative}")
        entries.append(
            {
                "repo_relative_path": relative,
                "source_sha256": source_hash,
                "snapshot_relative_path": str(destination.relative_to(output_root)),
                "snapshot_sha256": snapshot_hash,
            }
        )
    manifest = {
        "schema_version": f"{SCHEMA_VERSION}.source_snapshot.v1",
        "scope": "task_relevant_source_only",
        "entries": entries,
        "source_set_sha256": sha256_json(
            [[item["repo_relative_path"], item["source_sha256"]] for item in entries]
        ),
    }
    manifest_path = snapshot_root / "manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "manifest_relative_path": str(manifest_path.relative_to(output_root)),
        "manifest_sha256": sha256_file(manifest_path),
        "source_set_sha256": manifest["source_set_sha256"],
        "loaded_repo_source_paths": loaded_closure,
        "snapshotted_repo_paths": snapshot_paths,
        "entries": entries,
    }


def _verify_source_snapshot(
    repo: Path, output_root: Path, snapshot_receipt: Mapping[str, Any]
) -> None:
    entries = snapshot_receipt.get("entries")
    if not isinstance(entries, list):
        raise AssemblyError("source snapshot receipt lacks entries")
    expected_paths = sorted(
        str(item) for item in snapshot_receipt.get("snapshotted_repo_paths", [])
    )
    observed_paths = [
        str(_mapping(item, "source snapshot entry").get("repo_relative_path", ""))
        for item in entries
    ]
    if observed_paths != expected_paths:
        raise AssemblyError("source snapshot entry ordering or scope drifted")
    if not set(SOURCE_SNAPSHOT_PATHS) <= set(expected_paths):
        raise AssemblyError("source snapshot lost a required direct source")
    current_closure = _repo_loaded_source_paths(repo)
    absent = sorted(set(current_closure) - set(expected_paths))
    if absent:
        raise AssemblyError(f"new repo-local execution source is absent from snapshot: {absent[:5]}")
    source_pairs: list[list[str]] = []
    for raw in entries:
        entry = _mapping(raw, "source snapshot entry")
        relative = str(entry["repo_relative_path"])
        live = (repo / relative).resolve(strict=True)
        snapshot = (output_root / str(entry["snapshot_relative_path"])).resolve(
            strict=True
        )
        live_hash = sha256_file(live)
        snapshot_hash = sha256_file(snapshot)
        if (
            live_hash != entry.get("source_sha256")
            or snapshot_hash != entry.get("snapshot_sha256")
            or live_hash != snapshot_hash
        ):
            raise AssemblyError(f"source or snapshot drift detected: {relative}")
        source_pairs.append([relative, live_hash])
    if sha256_json(source_pairs) != snapshot_receipt.get("source_set_sha256"):
        raise AssemblyError("source snapshot set hash drifted")
    manifest_path = (
        output_root / str(snapshot_receipt.get("manifest_relative_path", ""))
    ).resolve(strict=True)
    if sha256_file(manifest_path) != snapshot_receipt.get("manifest_sha256"):
        raise AssemblyError("source snapshot manifest drifted")


def _discard_staging(path: Path) -> None:
    if path.exists():
        for candidate in sorted(path.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            try:
                os.chmod(candidate, 0o755 if candidate.is_dir() else 0o644)
            except OSError:
                pass
        try:
            os.chmod(path, 0o755)
        except OSError:
            pass
        shutil.rmtree(path)


def _publish_failed_output(
    output_dir: Path,
    *,
    analyzer: Path,
    failed_validation_stage: str,
    failure: BaseException,
    failures: Sequence[str],
) -> None:
    if output_dir.exists():
        raise AssemblyError(f"refusing to overwrite existing final output: {output_dir}")
    failed_staging = output_dir.parent / (
        f".{output_dir.name}.failed-{uuid.uuid4().hex}"
    )
    failed_staging.mkdir()
    receipt_path = failed_staging / "receipt.json"
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "terminal_status": "failed",
        "failed_validation_stage": failed_validation_stage,
        "failure_class": type(failure).__name__,
        "failure_message": str(failure),
        "failure_output_immutable": True,
        "immutability_error": None,
        "immutable_output_directory": str(output_dir),
        "analyzer": {"path": str(analyzer), "sha256": sha256_file(analyzer)},
        "failures": list(failures),
    }
    _write_json(receipt_path, receipt)
    try:
        os.chmod(receipt_path, 0o444)
        os.chmod(failed_staging, 0o555)
    except OSError as exc:
        receipt["failure_output_immutable"] = False
        receipt["immutability_error"] = f"{type(exc).__name__}: {exc}"
        replacement = failed_staging / ".receipt-replacement.json"
        _write_json(replacement, receipt)
        os.replace(replacement, receipt_path)
        try:
            os.chmod(receipt_path, 0o444)
            os.chmod(failed_staging, 0o555)
        except OSError:
            pass
    os.replace(failed_staging, output_dir)


def _finalize_staging(staging: Path, output_dir: Path, *, analyzer: Path) -> None:
    """Freeze and atomically expose staging, or publish only a failed receipt."""

    try:
        if output_dir.exists():
            raise AssemblyError(f"final output appeared during staging: {output_dir}")
        files = sorted((path for path in staging.rglob("*") if path.is_file()), key=str)
        directories = sorted(
            (path for path in staging.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        )
        for path in files:
            os.chmod(path, 0o444)
        for path in directories:
            os.chmod(path, 0o555)
        os.chmod(staging, 0o555)
        os.replace(staging, output_dir)
    except Exception as exc:
        failures = [f"{type(exc).__name__}: {exc}"]
        try:
            _discard_staging(staging)
        except Exception as cleanup_exc:
            failures.append(f"staging cleanup failed: {type(cleanup_exc).__name__}: {cleanup_exc}")
        _publish_failed_output(
            output_dir,
            analyzer=analyzer,
            failed_validation_stage="finalize_staging",
            failure=exc,
            failures=failures,
        )
        raise


def _materialize(
    *,
    candidate_pool: Path,
    split_receipt: Path,
    sampled_root: Path,
    source_root: Path,
    output_dir: Path,
) -> None:
    final_output_dir = output_dir.expanduser().resolve(strict=False)
    if final_output_dir != FROZEN_OUTPUT_ROOT.expanduser().resolve(strict=False):
        raise AssemblyError("output_root is not the frozen path")
    output_dir = final_output_dir
    if output_dir.exists():
        raise AssemblyError(f"immutable output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    output_dir = staging
    analyzer = Path(__file__).resolve()
    repo = analyzer.parents[2]
    failures: list[str] = []
    current_stage = "resolve_and_validate_inputs"
    try:
        candidate_pool = candidate_pool.expanduser().resolve(strict=True)
        split_receipt = split_receipt.expanduser().resolve(strict=True)
        sampled_root = sampled_root.expanduser().resolve(strict=True)
        source_root = source_root.expanduser().resolve(strict=True)
        split_payload = _mapping(
            json.loads(split_receipt.read_text(encoding="utf-8")), "split receipt"
        )
        split_outputs = _mapping(
            split_payload.get("outputs"), "split receipt outputs"
        )
        split_output_paths = {
            name: Path(
                str(
                    _mapping(
                        split_outputs.get(name), f"split receipt {name}"
                    ).get("path", "")
                )
            )
            .expanduser()
            .resolve(strict=True)
            for name in ("train_candidate", "development", "heldout")
        }
        _validate_frozen_paths(
            candidate_pool=candidate_pool,
            split_receipt=split_receipt,
            split_output_paths=split_output_paths,
            sampled_root=sampled_root,
            source_root=source_root,
            output_root=final_output_dir,
        )
        current_stage = "snapshot_source"
        source_snapshot = _materialize_source_snapshot(repo, output_dir)
        current_stage = "resolve_and_validate_inputs"
        if sha256_file(candidate_pool) != EXPECTED_CANDIDATE_POOL_SHA256:
            raise AssemblyError("candidate pool does not match the frozen SHA-256")
        if sha256_file(split_receipt) != EXPECTED_SPLIT_RECEIPT_SHA256:
            raise AssemblyError("split receipt does not match the frozen SHA-256")
        membership = _split_membership(
            candidate_pool=candidate_pool, split_receipt=split_receipt
        )
        candidate_pool_ids = list(_candidate_pool(candidate_pool))
        train_path = split_output_paths["train_candidate"]
        if sha256_file(train_path) != EXPECTED_TRAIN_CANDIDATE_SHA256:
            raise AssemblyError("training-candidate split does not match the frozen SHA-256")
        train = set(membership["train_candidate"])
        frozen_pool_ids = set().union(*membership.values())
        current_stage = "validate_manifests"
        sampled_manifests = _validate_panel_manifests(
            sampled_root, mode="sampled", expected_pool_ids=frozen_pool_ids
        )
        source_manifests = _validate_panel_manifests(
            source_root, mode="source_b16", expected_pool_ids=frozen_pool_ids
        )
        current_stage = "load_train_semantics"
        adapter = load_v2_b16_panel_adapter(
            sampled_panel_root=sampled_root,
            source_b16_root=source_root,
            candidate_pool=candidate_pool,
            semantic_image_ids=sorted(train, key=int),
        )
        _validate_frozen_panel_bindings(
            sampled_root=sampled_root,
            source_root=source_root,
            sampled_manifests=sampled_manifests,
            source_manifests=source_manifests,
            adapter=adapter,
        )
        input_identity = _input_identity_receipt(
            candidate_pool_ids=candidate_pool_ids,
            sampled_manifests=sampled_manifests,
            source_manifests=source_manifests,
            adapter=adapter,
        )
        census = _mapping(adapter.get("census"), "adapter.census")
        if (
            census.get("sampled_image_count") != EXPECTED_TRAIN_IMAGE_COUNT
            or census.get("sampled_trajectory_count")
            != EXPECTED_TRAIN_IMAGE_COUNT * 16
            or census.get("source_image_count") != EXPECTED_TRAIN_IMAGE_COUNT
            or census.get("source_accepted_image_count")
            != EXPECTED_ELIGIBLE_TRAIN_IMAGE_COUNT
            or census.get("source_ineligible_image_count") != 44
        ):
            raise AssemblyError("train-only v2 adapter census differs from the frozen counts")
        eligible_train = train & set(adapter["image_results"])
        if len(train) != EXPECTED_TRAIN_IMAGE_COUNT or len(eligible_train) != EXPECTED_ELIGIBLE_TRAIN_IMAGE_COUNT:
            raise AssemblyError("training join does not reproduce 2,004 Source-eligible images")
        current_stage = "analyze_census"
        records, summary = analyze_admission_census(adapter, eligible_train)
        reversed_records, reversed_summary = analyze_admission_census(
            adapter, eligible_train, reverse_input=True
        )
        deterministic = (
            sha256_json(records) == sha256_json(reversed_records)
            and sha256_json(summary) == sha256_json(reversed_summary)
        )
        if not deterministic:
            raise AssemblyError("census differs under reversed input order")
        if (
            len(records) != EXPECTED_ELIGIBLE_TRAIN_IMAGE_COUNT
            or any(item["candidate_count"] != EXPECTED_CANDIDATES_PER_IMAGE for item in records)
            or not summary["aggregate_reconciliation"]["passed"]
        ):
            raise AssemblyError("image records do not reconcile to the frozen census")

        current_stage = "write_outputs"
        census_path = output_dir / "image-census.jsonl"
        census_path.write_text(
            "".join(
                json.dumps(item, sort_keys=True, ensure_ascii=False) + "\n"
                for item in records
            ),
            encoding="utf-8",
        )
        summary_path = output_dir / "summary.json"
        _write_json(summary_path, summary)
        current_stage = "readback_outputs"
        _validate_serialized_output_readback(
            census_path=census_path,
            summary_path=summary_path,
            records=records,
            summary=summary,
        )
        current_stage = "revalidate_source_snapshot"
        _verify_source_snapshot(repo, output_dir, source_snapshot)
        current_stage = "write_success_receipt"
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "terminal_status": "completed",
            "immutable_output_directory": str(final_output_dir),
            "source_identity": _git_identity(
                repo, source_snapshot["snapshotted_repo_paths"]
            ),
            "source_snapshot": source_snapshot,
            "analyzer": {
                "path": str(analyzer),
                "sha256": sha256_file(analyzer),
            },
            "inputs": {
                "candidate_pool": {"path": str(candidate_pool), "sha256": sha256_file(candidate_pool)},
                "split_receipt": {"path": str(split_receipt), "sha256": sha256_file(split_receipt)},
                "train_candidate": {"path": str(train_path), "sha256": sha256_file(train_path)},
                "sampled_root": str(sampled_root),
                "source_root": str(source_root),
                "sampled_manifest_binding": sampled_manifests,
                "source_manifest_binding": source_manifests,
                "identity_receipt": input_identity,
            },
            "row_counts": {
                "image_census": len(records),
                "candidate": summary["candidate_count"],
                "eligible_candidate": summary["eligible_candidate_count"],
                "excluded_candidate": summary["excluded_candidate_count"],
            },
            "artifact_integrity_checks": {
                "frozen_input_hashes": True,
                "split_membership": True,
                "development_and_heldout_filtered_before_semantic_inspection": True,
                "all_manifests_and_batch_hashes": True,
                "frozen_roots_manifest_sets_execution_model_and_tokenizer": True,
                "task_scoped_source_snapshot": True,
                "sample_indices_0_through_15": True,
                "sampled_natural_closure": True,
                "source_status_counts": True,
                "training_join": True,
                "aggregate_reconciliation": True,
                "serialized_output_readback": True,
                "serialized_success_receipt_readback": True,
                "reversed_input_order_determinism": deterministic,
            },
            "determinism": {
                "forward_records_sha256": sha256_json(records),
                "reversed_records_sha256": sha256_json(reversed_records),
                "forward_summary_sha256": sha256_json(summary),
                "reversed_summary_sha256": sha256_json(reversed_summary),
            },
            "output_hashes": {
                "image-census.jsonl": sha256_file(census_path),
                "summary.json": sha256_file(summary_path),
                "source-snapshot/manifest.json": source_snapshot[
                    "manifest_sha256"
                ],
            },
            "failures": failures,
        }
        receipt_path = output_dir / "receipt.json"
        _write_json(receipt_path, receipt)
        current_stage = "readback_success_receipt"
        _validate_success_receipt_readback(
            receipt_path=receipt_path,
            expected_receipt=receipt,
            output_root=output_dir,
        )
        current_stage = "finalize_staging"
        _finalize_staging(output_dir, final_output_dir, analyzer=analyzer)
    except Exception as exc:
        failures.append(f"{type(exc).__name__}: {exc}")
        try:
            _discard_staging(output_dir)
        except Exception as cleanup_exc:
            failures.append(
                f"staging cleanup failed: {type(cleanup_exc).__name__}: {cleanup_exc}"
            )
        if not final_output_dir.exists():
            _publish_failed_output(
                final_output_dir,
                analyzer=analyzer,
                failed_validation_stage=current_stage,
                failure=exc,
                failures=failures,
            )
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--split-receipt", type=Path, required=True)
    parser.add_argument("--sampled-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _materialize(
        candidate_pool=args.candidate_pool,
        split_receipt=args.split_receipt,
        sampled_root=args.sampled_root,
        source_root=args.source_root,
        output_dir=args.output_dir.expanduser().resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
