#!/usr/bin/env python3
"""Assemble matched Source-preservation and multi-route StateBanks.

This is an experiment-local assembler for the 2026-07-22 preservation screen.
It deliberately reuses the exact-token helpers and geometry policy from
``assemble_positive_path_imitation_state_bank``.  The only new policy is the
selection of up to three complementary safe routes and a matched set of
greedy Source anchors.  No text is decoded and no token is re-tokenized.

The command writes two immutable, canonical banks:

* ``single-route-plus-source-preservation`` selects 496 unique sampled events
  from the frozen route pool;
* ``multi-route-plus-source-preservation`` selects 496 rows from up to three
  safe sampled routes per image.

Both banks contain the identical 496 Source anchors, for 992 events per arm.
Counts, image/family coverage, and the twelve blind development images are
checked before any output is written.  An existing output path is always
rejected.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import copy
import json
import hashlib
import math
from pathlib import Path
import re
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config.fingerprint import sha256_file, sha256_json  # noqa: E402
from src.inference.backend import token_ids_sha256  # noqa: E402
from src.rollout_calibration import (  # noqa: E402
    CheckpointIdentity,
    assemble_state_bank,
    load_state_bank,
    load_state_bank_manifest_binding,
)
from src.adapters.dora import inspect_dora_adapter_payload  # noqa: E402
from src.qwen.special_token_embeddings import (  # noqa: E402
    SpecialTokenSelection,
    inspect_special_token_embedding_delta_payload,
)
from src.rollout_calibration.state_bank import BLIND_IMAGE_IDS  # noqa: E402

from scripts.research.assemble_positive_path_imitation_state_bank import (  # noqa: E402
    AssemblyError,
    BOX_END,
    COORDINATE_TOKEN_END,
    COORDINATE_TOKEN_START,
    _annotation_entities,
    _image_id,
    _image_pad_interval,
    _mapping,
    _norm_sha,
    _prefix_status,
    _read_json,
    _route_row_counts,
    _route_rollout_index,
    _route_seed,
    _row_geometry_receipt,
    _string,
    _token_ids,
    exact_row_slices,  # noqa: F401 - re-exported for focused exact-token tests
    exact_row_site_types,
    load_jsonl,
    _load_rollout_rows,
    build_pre_state_bank,
)


SCHEMA_VERSION = "source_preservation_multi_route_state_bank_assembler.v1"
PRE_STATE_BANK_SCHEMA_VERSION = "source_preservation_multi_route_pre_state_bank.v1"
ROUTE_ANALYSIS_SCHEMA_VERSION = "individual_trajectory_union_support.v1"
ROLLOUT_SCHEMA_VERSION = "current_seeded_sampled_rollouts.v1"
TARGET_EVENT_COUNT = 512
ARM_EVENT_COUNT = 496
TARGET_IMAGE_COUNT = 118
MAX_ROUTES_PER_IMAGE = 3
GEOMETRY_IOU_THRESHOLD = 0.75
BREADTH_EVENT_COUNT = 496
BREADTH_BROAD_IMAGE_COUNT = 496
BREADTH_CONCENTRATED_IMAGE_COUNT = 118
_BREADTH_BANDS = (
    "sparse_1_to_3",
    "medium_4_to_7",
    "dense_8_to_15",
    "very_dense_16_plus",
)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise AssemblyError(f"output already exists: {path}")
    path.write_bytes(_canonical(value) + b"\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise AssemblyError(f"output already exists: {path}")
    with path.open("wb") as handle:
        for row in rows:
            handle.write(_canonical(dict(row)) + b"\n")


_CHECKPOINT_IDENTITY_FIELDS = frozenset(
    {
        "adapter_fingerprint",
        "embedding_delta_fingerprint",
        "base_config_sha256",
        "tokenizer_sha256",
        "token_identity_sha256",
        "special_token_identity_sha256",
        "processor_identity_sha256",
    }
)


def _checkpoint_identity_from_mapping(
    value: Mapping[str, Any], *, field: str
) -> CheckpointIdentity:
    """Parse one canonical ``CheckpointIdentity`` mapping with local errors."""

    try:
        return CheckpointIdentity.from_mapping(value)
    except Exception as exc:  # state-bank validators use several error classes
        raise AssemblyError(f"{field} is not a valid CheckpointIdentity: {exc}") from exc


def _explicit_rollout_checkpoint_identity(
    payload: Mapping[str, Any], *, field: str
) -> CheckpointIdentity | None:
    """Read an explicitly serialized checkpoint identity when one is present.

    A few older research artifacts carry the state-bank identity directly,
    while current seeded-rollout artifacts carry a runtime model receipt.  An
    explicit identity always wins over derivation, but it is still validated
    through the canonical ``CheckpointIdentity`` type.
    """

    locations: list[tuple[str, Any]] = []
    for key in ("checkpoint_identity", "source_checkpoint_identity", "source_checkpoint"):
        if key in payload:
            locations.append((f"{field}.{key}", payload.get(key)))
    model_identity = payload.get("model_identity")
    if isinstance(model_identity, Mapping):
        for key in ("checkpoint_identity", "source_checkpoint_identity", "source_checkpoint"):
            if key in model_identity:
                locations.append((f"{field}.model_identity.{key}", model_identity.get(key)))
        if _CHECKPOINT_IDENTITY_FIELDS <= set(model_identity):
            locations.append((f"{field}.model_identity", model_identity))
    for location, value in locations:
        if not isinstance(value, Mapping):
            raise AssemblyError(f"{location} must be an object")
        if _CHECKPOINT_IDENTITY_FIELDS <= set(value):
            return _checkpoint_identity_from_mapping(value, field=location)
    return None


def _derive_runtime_checkpoint_identity_from_rollout_artifact(
    payload: Mapping[str, Any], *, artifact_path: Path | None = None
) -> CheckpointIdentity:
    """Derive the executable checkpoint identity recorded by one rollout.

    Current seeded-rollout artifacts retain the backend model receipt rather
    than a state-bank ``CheckpointIdentity``.  We content-address the exact
    adapter and embedding payloads referenced by that receipt and hash the
    recorded tokenizer, processor, and selected-special-token identities using
    the same utilities used by StateBank training/evaluation.
    """

    field = f"rollout artifact {artifact_path}" if artifact_path is not None else "rollout artifact"
    runtime = payload.get("model_identity")
    if not isinstance(runtime, Mapping):
        raise AssemblyError(f"{field} lacks model_identity checkpoint evidence")
    runtime_model = runtime.get("model_identity")
    if not isinstance(runtime_model, Mapping):
        raise AssemblyError(f"{field}.model_identity lacks nested model_identity payload")
    adapter = runtime_model.get("adapter")
    if not isinstance(adapter, Mapping):
        raise AssemblyError(f"{field}.model_identity.model_identity lacks adapter payload")
    embedding = runtime_model.get("embedding_delta")
    if not isinstance(embedding, Mapping):
        raise AssemblyError(f"{field}.model_identity.model_identity lacks embedding_delta payload")
    embedding_reference = embedding.get("identity")
    if not isinstance(embedding_reference, Mapping):
        raise AssemblyError(f"{field}.model_identity.model_identity.embedding_delta lacks identity")
    adapter_path_value = adapter.get("adapter_path")
    embedding_path_value = embedding_reference.get("delta_path")
    if not isinstance(adapter_path_value, str) or not adapter_path_value:
        raise AssemblyError(f"{field} adapter identity lacks adapter_path")
    if not isinstance(embedding_path_value, str) or not embedding_path_value:
        raise AssemblyError(f"{field} embedding identity lacks delta_path")
    base_reference = runtime_model.get("base")
    base_path_value = (
        base_reference.get("path")
        if isinstance(base_reference, Mapping)
        else adapter.get("base_model_path") or embedding_reference.get("base_model_path")
    )
    if not isinstance(base_path_value, str) or not base_path_value:
        raise AssemblyError(f"{field} checkpoint identity lacks base model path")
    base_path = Path(base_path_value).expanduser().resolve()
    try:
        adapter_identity = inspect_dora_adapter_payload(
            Path(adapter_path_value).expanduser().resolve(strict=True),
            expected_base_model_path=base_path,
        )
        embedding_identity = inspect_special_token_embedding_delta_payload(
            Path(embedding_path_value).expanduser().resolve(strict=True),
            expected_base_model_path=base_path,
        )
    except Exception as exc:
        raise AssemblyError(f"{field} checkpoint payload identity inspection failed: {exc}") from exc
    token_identity = runtime.get("tokenizer_identity")
    processor_identity = runtime.get("processor_identity")
    if not isinstance(token_identity, Mapping):
        raise AssemblyError(f"{field} lacks tokenizer_identity")
    if not isinstance(processor_identity, Mapping):
        raise AssemblyError(f"{field} lacks processor_identity")
    semantic = embedding_identity.get("semantic_identity")
    if not isinstance(semantic, Mapping):
        raise AssemblyError(f"{field} embedding identity lacks semantic_identity")
    token_strings = semantic.get("token_strings")
    token_ids = semantic.get("token_ids")
    if not isinstance(token_strings, list) or not isinstance(token_ids, list):
        raise AssemblyError(f"{field} embedding identity lacks selected token strings/IDs")
    selection = SpecialTokenSelection(token_strings=token_strings, token_ids=token_ids)
    try:
        return CheckpointIdentity(
            adapter_fingerprint=str(adapter_identity["fingerprint"]),
            embedding_delta_fingerprint=str(embedding_identity["fingerprint"]),
            base_config_sha256=str(semantic.get("base_config_sha256", "")),
            tokenizer_sha256=str(semantic.get("tokenizer_sha256", "")),
            token_identity_sha256=sha256_json(dict(token_identity)),
            special_token_identity_sha256=sha256_json(selection.to_artifact_dict()),
            processor_identity_sha256=sha256_json(dict(processor_identity)),
        )
    except Exception as exc:  # canonical identity validation has multiple error types
        raise AssemblyError(f"{field} runtime checkpoint identity is invalid: {exc}") from exc


def derive_checkpoint_identity_from_rollout_artifact(
    payload: Mapping[str, Any], *, artifact_path: Path | None = None
) -> CheckpointIdentity:
    """Derive runtime identity, then cross-check any explicit serialization.

    Explicit state-bank identity fields are evidence to compare, never a
    substitute for independently inspecting the runtime model receipt and its
    referenced payloads.  In particular, an artifact with only an explicit
    identity and no ``model_identity`` proof is rejected.
    """

    field = f"rollout artifact {artifact_path}" if artifact_path is not None else "rollout artifact"
    runtime_identity = _derive_runtime_checkpoint_identity_from_rollout_artifact(
        payload, artifact_path=artifact_path
    )
    explicit = _explicit_rollout_checkpoint_identity(payload, field=field)
    if explicit is not None and explicit != runtime_identity:
        raise AssemblyError(
            f"{field} explicit checkpoint identity contradicts independently derived runtime identity"
        )
    return runtime_identity


def verify_rollout_checkpoint_identities(
    paths: Sequence[Path], *, reference_checkpoint: CheckpointIdentity
) -> tuple[CheckpointIdentity, list[dict[str, Any]]]:
    """Verify every input rollout artifact against one exact reference identity."""

    if not paths:
        raise AssemblyError("at least one rollout artifact is required for checkpoint verification")
    verified: list[dict[str, Any]] = []
    expected: CheckpointIdentity | None = None
    for path in paths:
        payload = _read_json(path)
        if not isinstance(payload, Mapping):
            raise AssemblyError(f"rollout artifact {path} must be an object")
        observed = derive_checkpoint_identity_from_rollout_artifact(payload, artifact_path=path)
        if expected is None:
            expected = observed
        elif observed != expected:
            raise AssemblyError(
                "rollout checkpoint identity mismatch across input artifacts: "
                f"{path} differs from {verified[0]['artifact_path']}"
            )
        if observed != reference_checkpoint:
            raise AssemblyError(
                "rollout checkpoint identity differs from reference binding: "
                f"{path} does not match source_checkpoint"
            )
        verified.append(
            {
                "artifact_path": str(path.resolve()),
                "checkpoint_id": sha256_json(observed.to_artifact_dict()),
                "checkpoint_identity": observed.to_artifact_dict(),
            }
        )
    assert expected is not None
    return expected, verified


def _load_source_anchor_review(path: Path) -> dict[str, Any]:
    """Load and validate the explicit crop-review decisions for Source rows.

    The trajectory assignment deliberately leaves uncertain rows unresolved.
    This sidecar is the only authority allowed to promote one of those rows or
    to remove an image from the matched cohort.  Keeping it as an input (and
    hashing it in the receipt) avoids embedding image-specific review choices
    in the assembler.
    """

    payload = _read_json(path)
    if not isinstance(payload, Mapping):
        raise AssemblyError("source-anchor manual review must be an object")
    if payload.get("schema_version") != "source_route_anchor_manual_review.v1":
        raise AssemblyError("unsupported source-anchor manual review schema")
    if payload.get("status") != "review_only_manual_overrides":
        raise AssemblyError("source-anchor manual review must be review-only")
    raw_decisions = payload.get("decisions")
    if not isinstance(raw_decisions, list) or not raw_decisions:
        raise AssemblyError("source-anchor manual review requires decisions")
    decisions: list[dict[str, Any]] = []
    seen_images: set[str] = set()
    for index, raw in enumerate(raw_decisions):
        item = _require_mapping(raw, f"source-anchor manual review decisions[{index}]")
        image = _image_id(item.get("image_id"), f"manual review decision {index}.image_id")
        if image in seen_images:
            raise AssemblyError(f"manual review has duplicate decision for image {image}")
        seen_images.add(image)
        decision = str(item.get("decision", ""))
        if decision not in {"admit", "exclude"}:
            raise AssemblyError(f"manual review decision {image} must be admit or exclude")
        normalized: dict[str, Any] = {
            "image_id": image,
            "decision": decision,
            "reviewer": str(item.get("reviewer", "")),
            "review_source": str(item.get("review_source", "")),
            "comment": str(item.get("comment", "")),
        }
        if decision == "admit":
            if item.get("row_index") is None:
                raise AssemblyError(f"manual admit decision {image} lacks row_index")
            row_index = int(item["row_index"])
            if row_index < 0:
                raise AssemblyError(f"manual admit decision {image} has negative row_index")
            owner_id = str(item.get("owner_id", ""))
            if not owner_id or not owner_id.startswith(f"{image}:"):
                raise AssemblyError(f"manual admit decision {image} has invalid owner_id")
            category = str(item.get("category", "")).strip()
            if not category:
                raise AssemblyError(f"manual admit decision {image} lacks category")
            if item.get("iou") is None:
                raise AssemblyError(f"manual admit decision {image} lacks iou")
            iou = float(item["iou"])
            if not 0.0 <= iou <= 1.0:
                raise AssemblyError(f"manual admit decision {image} has invalid iou {iou}")
            normalized.update({"row_index": row_index, "owner_id": owner_id, "category": category, "iou": iou})
        elif any(key in item for key in ("row_index", "owner_id", "category", "iou")):
            raise AssemblyError(f"manual exclude decision {image} must not carry row admission fields")
        decisions.append(normalized)
    raw_replacements = payload.get("replacement_image_ids", [])
    if not isinstance(raw_replacements, list):
        raise AssemblyError("source-anchor replacement_image_ids must be a list")
    replacements: list[str] = []
    for index, value in enumerate(raw_replacements):
        image = _image_id(value, f"replacement_image_ids[{index}]")
        if image in replacements:
            raise AssemblyError(f"source-anchor replacements contain duplicate image {image}")
        if image in seen_images:
            raise AssemblyError(f"replacement image {image} also has a manual decision")
        replacements.append(image)
    raw_geometry_allowlist = payload.get("treatment_geometry_untrusted_allowlist", [])
    if not isinstance(raw_geometry_allowlist, list):
        raise AssemblyError("treatment_geometry_untrusted_allowlist must be a list")
    geometry_allowlist: list[dict[str, Any]] = []
    seen_geometry_keys: set[tuple[str, str, int, str]] = set()
    for index, raw in enumerate(raw_geometry_allowlist):
        item = _require_mapping(raw, f"treatment geometry allowlist[{index}]")
        image = _image_id(item.get("image_id"), f"treatment geometry allowlist[{index}].image_id")
        route_id = str(item.get("route_id", ""))
        if not route_id:
            raise AssemblyError(f"treatment geometry allowlist {image} lacks route_id")
        row_index = int(item.get("row_index", -1))
        if row_index < 0:
            raise AssemblyError(f"treatment geometry allowlist {image} has invalid row_index")
        owner_id = str(item.get("owner_id", ""))
        if not owner_id.startswith(f"{image}:"):
            raise AssemblyError(f"treatment geometry allowlist {image} has invalid owner_id")
        iou = float(item.get("iou", -1.0))
        if not 0.0 <= iou < GEOMETRY_IOU_THRESHOLD:
            raise AssemblyError(
                f"treatment geometry allowlist {image}:{row_index} must be below the geometry threshold"
            )
        key = (image, route_id, row_index, owner_id)
        if key in seen_geometry_keys:
            raise AssemblyError(f"duplicate treatment geometry allowlist key: {key}")
        seen_geometry_keys.add(key)
        geometry_allowlist.append(
            {
                "image_id": image,
                "route_id": route_id,
                "row_index": row_index,
                "owner_id": owner_id,
                "iou": iou,
                "reviewer": str(item.get("reviewer", "")),
                "comment": str(item.get("comment", "")),
            }
        )
    return {
        "schema_version": str(payload["schema_version"]),
        "status": str(payload["status"]),
        "review_packet": str(payload.get("review_packet", "")),
        "treatment_geometry_review_packet": str(payload.get("treatment_geometry_review_packet", "")),
        "decisions": decisions,
        "replacement_image_ids": replacements,
        "treatment_geometry_untrusted_allowlist": geometry_allowlist,
    }


def _source_review_by_image(review: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    decisions = review.get("decisions")
    if not isinstance(decisions, list):
        raise AssemblyError("source-anchor manual review decisions must be a list")
    return {str(item["image_id"]): dict(item) for item in decisions if isinstance(item, Mapping)}


def _admitted_source_image_ids(
    prior_image_ids: Sequence[str], review: Mapping[str, Any]
) -> tuple[list[str], list[str]]:
    """Derive the matched cohort from prior IDs and sidecar exclusions."""

    prior = {str(item) for item in prior_image_ids}
    decisions = _source_review_by_image(review)
    replacements = {str(item) for item in review.get("replacement_image_ids", [])}
    if replacements & prior:
        raise AssemblyError(
            f"replacement images already belong to the prior cohort: {sorted(replacements & prior, key=int)}"
        )
    unknown = sorted((set(decisions) - prior) - replacements, key=int)
    if unknown:
        raise AssemblyError(f"manual review references images outside prior cohort: {unknown}")
    excluded = sorted(
        (
            image
            for image, item in decisions.items()
            if str(item.get("decision")) == "exclude"
        ),
        key=int,
    )
    admitted = sorted((prior - set(excluded)) | replacements, key=int)
    if not admitted:
        raise AssemblyError("manual review excludes the complete prior cohort")
    reject_blind_images(admitted)
    return admitted, excluded


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    return _mapping(value, name)


def _safe_route_candidates(
    image_result: Mapping[str, Any], *, budget: int = 16
) -> tuple[dict[str, Any], ...]:
    """Return all safe sampled routes for one image in deterministic order.

    Route quality remains the inherited analysis authority.  We do not infer
    owners from text or geometry here: assignment receipts and owner sets are
    consumed exactly as recorded by the frozen trajectory analysis.
    """

    image = _image_id(image_result.get("image_id"))
    greedy_id = _string(image_result.get("greedy_trajectory_id"), "greedy_trajectory_id")
    evidence = _require_mapping(image_result.get("trajectory_evidence"), "trajectory_evidence")
    sampled_ids = [str(item) for item in image_result.get("sampled_trajectory_ids", [])]
    budgets = image_result.get("budgets")
    if not isinstance(budgets, list):
        raise AssemblyError(f"image {image} budgets must be a list")
    selected_budget = next(
        (
            _require_mapping(item, "budget")
            for item in budgets
            if isinstance(item, Mapping) and int(item.get("budget", -1)) == int(budget)
        ),
        None,
    )
    if selected_budget is None:
        raise AssemblyError(f"image {image} lacks budget {budget}")
    assignments = _require_mapping(selected_budget.get("trajectory_assignments"), "trajectory_assignments")
    owner_sets = _require_mapping(selected_budget.get("owner_sets"), "owner_sets")
    greedy_assignment = _require_mapping(assignments.get(greedy_id), f"trajectory_assignments[{greedy_id}]")
    greedy_owners = frozenset(str(item) for item in owner_sets.get(greedy_id, []))
    greedy_counts = _route_row_counts(greedy_assignment)
    greedy_duplicate = int(greedy_counts.get("duplicate", 0))
    greedy_malformed = int(greedy_counts.get("malformed", greedy_assignment.get("malformed_row_count", 0) or 0))
    candidates: list[dict[str, Any]] = []
    for route_id in sorted(sampled_ids):
        route_evidence = _require_mapping(evidence.get(route_id), f"trajectory_evidence[{route_id}]")
        assignment = _require_mapping(assignments.get(route_id), f"trajectory_assignments[{route_id}]")
        route_owners = frozenset(str(item) for item in owner_sets.get(route_id, []))
        added = route_owners - greedy_owners
        parser = _require_mapping(route_evidence.get("parser"), f"trajectory_evidence[{route_id}].parser")
        status = str(parser.get("parse_status"))
        stop_reason = str(route_evidence.get("stop_reason"))
        counts = _route_row_counts(assignment)
        duplicate = int(counts.get("duplicate", 0))
        malformed = int(counts.get("malformed", assignment.get("malformed_row_count", 0) or 0))
        unresolved = int(counts.get("unresolved", 0))
        seed = _route_seed(route_evidence, assignment)
        reasons: list[str] = []
        if status != "accepted":
            reasons.append(f"parser_status:{status}")
        if stop_reason != "im_end":
            reasons.append(f"stop_reason:{stop_reason}")
        if not route_owners > greedy_owners:
            reasons.append("owner_set_not_strict_superset")
        if duplicate > greedy_duplicate:
            reasons.append("duplicate_count_worse_than_greedy")
        if malformed > greedy_malformed:
            reasons.append("malformed_count_worse_than_greedy")
        candidates.append(
            {
                "route_id": route_id,
                "seed": seed,
                "owner_ids": sorted(route_owners),
                "greedy_owner_ids": sorted(greedy_owners),
                "added_owner_ids": sorted(added),
                "duplicate_count": duplicate,
                "malformed_count": malformed,
                "duplicate_delta": duplicate - greedy_duplicate,
                "malformed_delta": malformed - greedy_malformed,
                "unresolved_rows": unresolved,
                "admissible": not reasons,
                "rejection_reasons": reasons,
            }
        )
    return tuple(candidates)


def _last_owner_row_index(
    assignment: Mapping[str, Any], owner_ids: Sequence[str], *, route_id: str
) -> int:
    wanted = {str(item) for item in owner_ids}
    rows = assignment.get("row_assignment_receipts")
    if not isinstance(rows, list):
        raise AssemblyError(f"route {route_id} row_assignment_receipts must be a list")
    indices = [
        int(item["generated_row_index"])
        for item in rows
        if isinstance(item, Mapping) and str(item.get("owner_id")) in wanted
    ]
    if not indices:
        raise AssemblyError(f"route {route_id} has marginal owners but no row receipts")
    return max(indices)


def select_complementary_routes(
    image_result: Mapping[str, Any], *, max_routes: int = MAX_ROUTES_PER_IMAGE, budget: int = 16
) -> dict[str, Any]:
    """Select up to ``max_routes`` safe routes by marginal owner coverage.

    The rank is exactly ``marginal added-owner count``, ``total added-owner
    count``, fewer unresolved rows, lower seed, then route id.  Selection
    stops when no remaining safe route contributes a new owner.
    """

    if max_routes < 1:
        raise AssemblyError("max_routes must be positive")
    image = _image_id(image_result.get("image_id"))
    candidates = list(_safe_route_candidates(image_result, budget=budget))
    selected_budget = next(
        item
        for item in image_result["budgets"]
        if isinstance(item, Mapping) and int(item.get("budget", -1)) == int(budget)
    )
    assignments = _require_mapping(selected_budget.get("trajectory_assignments"), "trajectory_assignments")
    union: set[str] = set()
    selected: list[dict[str, Any]] = []
    remaining = [item for item in candidates if item["admissible"]]
    while remaining and len(selected) < max_routes:
        ranked: list[tuple[tuple[Any, ...], dict[str, Any], set[str]]] = []
        for item in remaining:
            marginal = set(item["added_owner_ids"]) - union
            if not marginal:
                continue
            key = (
                -len(marginal),
                -len(item["added_owner_ids"]),
                int(item["unresolved_rows"]),
                int(item["seed"]),
                str(item["route_id"]),
            )
            ranked.append((key, item, marginal))
        if not ranked:
            break
        _, choice, marginal = min(ranked, key=lambda value: value[0])
        assignment = _require_mapping(assignments.get(choice["route_id"]), f"trajectory_assignments[{choice['route_id']}]" )
        last_index = _last_owner_row_index(assignment, sorted(marginal), route_id=str(choice["route_id"]))
        selected.append({**choice, "marginal_added_owner_ids": sorted(marginal), "last_marginal_owner_row_index": last_index})
        union.update(marginal)
        remaining = [item for item in remaining if item["route_id"] != choice["route_id"]]
    return {
        "image_id": image,
        "budget": int(budget),
        "max_routes": int(max_routes),
        "candidate_routes": candidates,
        "selected_routes": selected,
        "selected_route_ids": [str(item["route_id"]) for item in selected],
        "selected_added_owner_union": sorted(union),
        "selected_added_owner_count": len(union),
        "admissible_route_count": sum(1 for item in candidates if item["admissible"]),
    }


def select_multi_route_routes(
    image_result: Mapping[str, Any], *, max_routes: int = MAX_ROUTES_PER_IMAGE, budget: int = 16
) -> dict[str, Any]:
    """Alias with an explicit treatment-oriented name for focused tests."""

    return select_complementary_routes(image_result, max_routes=max_routes, budget=budget)


def select_source_anchor_rows(
    assignment: Mapping[str, Any],
    owners: Sequence[Mapping[str, Any]],
    *,
    allow_geometry_unknown: bool = False,
) -> list[dict[str, Any]]:
    """Keep first trusted physical-owner occurrences from a greedy route.

    ``allow_geometry_unknown`` is used only by Source preservation.  A reviewed
    owner identity can remain an eligible complete-row target when IoU is below
    the coordinate trust threshold; the event retains exact coordinates but
    marks them geometry-ineligible so the trainer masks those sites.
    """

    rows = assignment.get("row_assignment_receipts")
    if not isinstance(rows, list):
        raise AssemblyError("greedy assignment row_assignment_receipts must be a list")
    owner_ids = {str(item.get("owner_id")) for item in owners}
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for raw in sorted(rows, key=lambda item: int(item.get("generated_row_index", -1))):
        if not isinstance(raw, Mapping):
            raise AssemblyError("row assignment receipt must be an object")
        owner = raw.get("owner_id")
        if owner is None:
            continue
        owner_id = str(owner)
        if owner_id in seen or owner_id not in owner_ids:
            continue
        decision = _row_geometry_receipt(raw, owners)
        owner_known = owner_id in owner_ids
        if not decision["gradient_eligible"] and not (
            allow_geometry_unknown
            and owner_known
            and decision["entity_status"] == "verified_owner"
        ):
            continue
        seen.add(owner_id)
        result.append(
            {
                "generated_row_index": int(raw["generated_row_index"]),
                "owner_id": owner_id,
                "geometry": decision,
                "receipt": dict(raw),
            }
        )
    return result


def source_first_owner_rows(
    assignment: Mapping[str, Any], owners: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Focused-test alias for :func:`select_source_anchor_rows`."""

    return select_source_anchor_rows(assignment, owners)


def deduplicate_route_events(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate exact route rows by image/prefix/row/owner identity."""

    seen: set[tuple[str, str, str, str]] = set()
    result: list[dict[str, Any]] = []
    for raw in events:
        item = dict(raw)
        image = str(item.get("image_id", item.get("image", {}).get("image_id", "")))
        prefix_hash = str(item.get("prefix_token_ids_sha256", ""))
        row_hash = str(item.get("candidate_token_ids_sha256", item.get("token_ids_sha256", "")))
        owner = str(item.get("owner_id", item.get("physical_owner_id", "")))
        key = (image, prefix_hash, row_hash, owner)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def select_image_diverse_events(
    candidates_by_image: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    image_ids: Sequence[str],
    event_count: int = TARGET_EVENT_COUNT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select a fixed event count while retaining one diverse row per image."""

    ids = tuple(sorted({str(item) for item in image_ids}, key=int))
    if not ids:
        raise AssemblyError("image-diverse selection requires images")
    if event_count < len(ids):
        raise AssemblyError("event_count must be at least one per image")
    pools: dict[str, list[dict[str, Any]]] = {}
    for image in ids:
        pool = deduplicate_route_events(candidates_by_image.get(image, []))
        if not pool:
            raise AssemblyError(f"image {image} has no admitted treatment/source candidate")
        pools[image] = pool
    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[str, str, str, str]] = set()
    covered_owners: set[str] = set()

    def key_for(item: Mapping[str, Any]) -> tuple[str, str, str, str]:
        return (
            str(item.get("image_id", item.get("image", {}).get("image_id", ""))),
            str(item.get("prefix_token_ids_sha256", "")),
            str(item.get("candidate_token_ids_sha256", item.get("token_ids_sha256", ""))),
            str(item.get("owner_id", item.get("physical_owner_id", ""))),
        )

    def rank(item: Mapping[str, Any]) -> tuple[Any, ...]:
        owner = str(item.get("owner_id", item.get("physical_owner_id", "")))
        return (
            0 if owner not in covered_owners else 1,
            -int(item.get("marginal_owner_count", item.get("added_owner_count", 0)) or 0),
            int(item.get("route_seed", item.get("seed", 0)) or 0),
            int(item.get("generated_row_index", 0) or 0),
            owner,
            str(item.get("event_id", "")),
        )

    # Round-robin admission is the image-diversity guarantee.  A fresh owner
    # wins over a repeated owner within each round, then route/row ties are
    # stable and independent of JSONL input order.
    while len(selected) < event_count:
        progressed = False
        for image in ids:
            if len(selected) >= event_count:
                break
            available = [item for item in pools[image] if key_for(item) not in selected_keys]
            if not available:
                continue
            chosen = min(available, key=rank)
            selected.append(dict(chosen))
            selected_keys.add(key_for(chosen))
            owner = str(chosen.get("owner_id", chosen.get("physical_owner_id", "")))
            if owner:
                covered_owners.add(owner)
            progressed = True
        if not progressed:
            break
    if len(selected) != event_count:
        raise AssemblyError(f"cannot select exactly {event_count} events; only {len(selected)} candidates available")
    counts = Counter(str(item.get("image_id", item.get("image", {}).get("image_id", ""))) for item in selected)
    if set(counts) != set(ids):
        raise AssemblyError("image-diverse selection dropped an image")
    return selected, {
        "event_count": len(selected),
        "image_ids": list(ids),
        "image_count": len(ids),
        "image_event_counts": [
            {"image_id": image, "event_count": int(counts[image])} for image in ids
        ],
        "unique_owner_count": len(covered_owners),
    }


def image_family_event_weights(
    family_event_counts: Mapping[str, Mapping[str, int]], *, event_count: int = TARGET_EVENT_COUNT
) -> dict[tuple[str, str], float]:
    """Return equal-image/equal-family weights with global event mean one."""

    images = tuple(sorted({str(item) for item in family_event_counts}, key=int))
    if not images:
        raise AssemblyError("family weighting requires at least one image")
    for image in images:
        for family in ("source_preservation", "treatment"):
            count = family_event_counts[image].get(family)
            if not isinstance(count, int) or count < 1:
                raise AssemblyError(f"image {image} lacks positive {family} count")
    actual = sum(sum(int(value) for value in family_event_counts[image].values()) for image in images)
    if actual != int(event_count):
        raise AssemblyError(f"family event count {actual} differs from required {event_count}")
    factor = float(event_count) / float(2 * len(images))
    result: dict[tuple[str, str], float] = {}
    for image in images:
        for family in ("source_preservation", "treatment"):
            result[(image, family)] = factor / float(family_event_counts[image][family])
    mean = sum(result[(image, family)] * family_event_counts[image][family] for image in images for family in ("source_preservation", "treatment")) / float(event_count)
    if abs(mean - 1.0) > 1e-12:
        raise AssemblyError(f"family weights do not have mean one: {mean}")
    return result


def source_treatment_family_weights(
    family_event_counts: Mapping[str, Mapping[str, int]], *, event_count: int = TARGET_EVENT_COUNT
) -> dict[tuple[str, str], float]:
    """Focused-test alias for :func:`image_family_event_weights`."""

    return image_family_event_weights(family_event_counts, event_count=event_count)


def _breadth_quota(total: int) -> dict[str, int]:
    """Return the closest deterministic four-band allocation for ``total``."""

    quotient, remainder = divmod(total, len(_BREADTH_BANDS))
    return {
        band: quotient + int(index < remainder)
        for index, band in enumerate(_BREADTH_BANDS)
    }


def _breadth_image_order(image_id: str) -> tuple[str, str, str]:
    """Stable image-hash order, with the ID as an audit-friendly final tie-break."""

    return (hashlib.sha256(image_id.encode("utf-8")).hexdigest(), image_id, image_id)


def _breadth_identity(item: Mapping[str, Any]) -> tuple[str, str, str, str]:
    """Return the exact-row identity used to forbid cloned sampled/Source rows."""

    image = str(item.get("image_id", ""))
    prefix = str(item.get("prefix_token_ids_sha256", ""))
    row = str(item.get("candidate_token_ids_sha256", ""))
    owner = str(item.get("owner_id", ""))
    if not all((image, prefix, row, owner)):
        raise AssemblyError(
            "breadth candidate lacks image_id, prefix hash, row hash, or physical owner ID"
        )
    return image, prefix, row, owner


def _breadth_int(item: Mapping[str, Any], field: str) -> int:
    value = item.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AssemblyError(f"breadth candidate {item.get('event_id', '<unknown>')} lacks non-negative {field}")
    return value


def _breadth_sampled_rank(item: Mapping[str, Any]) -> tuple[Any, ...]:
    """The unit's fixed sampled-event ranking rule, independent of row scores."""

    return (
        -_breadth_int(item, "marginal_route_added_owner_count"),
        -_breadth_int(item, "route_added_owner_count"),
        _breadth_int(item, "unresolved_row_count"),
        _breadth_int(item, "route_seed"),
        _breadth_int(item, "generated_row_index"),
        str(item.get("event_id", "")),
    )


def _breadth_source_rank(item: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        _breadth_int(item, "trusted_source_anchor_order"),
        _breadth_int(item, "generated_row_index"),
        str(item.get("event_id", "")),
    )


def _breadth_complete_row_coordinate_token_supervision(
    item: Mapping[str, Any],
) -> str:
    """Describe coordinate-token inclusion in complete-row imitation.

    This is intentionally distinct from ``geometry_eligible``, which gates
    the separate coordinate-boundary objective.  Complete-row imitation uses
    trusted geometry review to include exact coordinate tokens.
    """

    geometry = item.get("geometry")
    if not isinstance(geometry, Mapping) or not isinstance(geometry.get("geometry_trusted"), bool):
        raise AssemblyError(
            f"breadth candidate {item.get('event_id', '<unknown>')} lacks geometry trust evidence"
        )
    return "enabled" if geometry["geometry_trusted"] else "masked"


def _breadth_pair_pools(
    *,
    sampled_candidates: Mapping[str, Sequence[Mapping[str, Any]]],
    source_candidates: Mapping[str, Sequence[Mapping[str, Any]]],
    image_bands: Mapping[str, str],
) -> dict[str, list[dict[str, Any]]]:
    """Build non-cloned, ranked sampled/Source pairs for each eligible image.

    Candidate rows are already admitted exact-token rows.  This helper does
    not infer owner trust or route quality; it only applies the frozen
    constant-dose selection policy and therefore stays reusable for a larger
    trajectory panel.
    """

    result: dict[str, list[dict[str, Any]]] = {}
    all_images = set(sampled_candidates) | set(source_candidates)
    unknown = sorted(all_images - set(image_bands))
    if unknown:
        raise AssemblyError(f"breadth candidates fall outside the frozen training reservoir: {unknown}")
    invalid_bands = sorted(
        image for image, band in image_bands.items() if band not in _BREADTH_BANDS
    )
    if invalid_bands:
        raise AssemblyError(f"breadth training reservoir has unsupported object-count bands: {invalid_bands}")
    for image in sorted(all_images, key=_breadth_image_order):
        sampled = [dict(item) for item in sampled_candidates.get(image, [])]
        source = [dict(item) for item in source_candidates.get(image, [])]
        if not sampled or not source:
            continue
        if any(str(item.get("image_id")) != image for item in [*sampled, *source]):
            raise AssemblyError(f"breadth candidate image ID disagrees with its candidate-pool key: {image}")
        sampled_identities = [_breadth_identity(item) for item in sampled]
        source_identities = [_breadth_identity(item) for item in source]
        if len(set(sampled_identities)) != len(sampled_identities):
            raise AssemblyError(f"breadth sampled candidates contain an exact-row clone for image {image}")
        if len(set(source_identities)) != len(source_identities):
            raise AssemblyError(f"breadth Source candidates contain an exact-row clone for image {image}")
        source_identity_set = set(source_identities)
        sampled = [
            item for item in sampled if _breadth_identity(item) not in source_identity_set
        ]
        if not sampled:
            continue
        sampled.sort(key=_breadth_sampled_rank)
        source.sort(key=_breadth_source_rank)
        pair_count = min(len(sampled), len(source))
        result[image] = [
            {
                "image_id": image,
                "object_count_band": image_bands[image],
                "selection_rank": rank,
                "sampled": sampled[rank - 1],
                "source": source[rank - 1],
            }
            for rank in range(1, pair_count + 1)
        ]
    return result


def _breadth_distribution(events: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    fields = (
        "selection_rank",
        "route_count",
        "row_depth",
        "complete_row_coordinate_token_supervision",
    )
    result: dict[str, dict[str, int]] = {field: {} for field in fields}
    for item in events:
        for field in fields:
            value = str(item[field])
            result[field][value] = result[field].get(value, 0) + 1
    return result


def _breadth_pair_rank_histogram(
    pairs: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    """Count selected pair ranks independently inside each object-count band."""

    result = {band: {} for band in _BREADTH_BANDS}
    for pair in pairs:
        band = str(pair.get("object_count_band", ""))
        if band not in result:
            raise AssemblyError(f"selected breadth pair has unsupported band: {band!r}")
        rank = str(_breadth_int(pair, "selection_rank"))
        result[band][rank] = result[band].get(rank, 0) + 1
    return result


def _breadth_rank_group(rank: int) -> str:
    if rank <= 0:
        raise AssemblyError(f"selection rank must be positive, got {rank}")
    return str(rank) if rank <= 3 else "4_plus"


def _breadth_pair_rank_group_histogram(
    pairs: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    """Count the one predeclared coarse rank grouping by object-count band."""

    result = {band: {} for band in _BREADTH_BANDS}
    for pair in pairs:
        band = str(pair.get("object_count_band", ""))
        if band not in result:
            raise AssemblyError(f"selected breadth pair has unsupported band: {band!r}")
        group = _breadth_rank_group(_breadth_int(pair, "selection_rank"))
        result[band][group] = result[band].get(group, 0) + 1
    return result


def _select_concentrated_pairs_by_band_round_robin(
    *,
    pairs_by_image: Mapping[str, Sequence[Mapping[str, Any]]],
    concentrated_images: Sequence[str],
    image_bands: Mapping[str, str],
    pair_quota_by_band: Mapping[str, int],
) -> list[dict[str, Any]]:
    """Select each band's concentrated pairs by an independent round robin."""

    selected: list[dict[str, Any]] = []
    for band in _BREADTH_BANDS:
        images = sorted(
            (image for image in concentrated_images if image_bands[image] == band),
            key=_breadth_image_order,
        )
        target = int(pair_quota_by_band[band])
        band_selected: list[dict[str, Any]] = []
        next_rank = {image: 0 for image in images}
        while len(band_selected) < target:
            progressed = False
            for image in images:
                rank_index = next_rank[image]
                if rank_index >= len(pairs_by_image[image]):
                    continue
                band_selected.append(dict(pairs_by_image[image][rank_index]))
                next_rank[image] = rank_index + 1
                progressed = True
                if len(band_selected) == target:
                    break
            if not progressed:
                raise AssemblyError(
                    "concentrated trusted pair supply is insufficient for band quota: "
                    f"band={band}, selected={len(band_selected)}, required={target}"
                )
        selected.extend(band_selected)
    if len(selected) != BREADTH_EVENT_COUNT:
        raise AssemblyError(
            "concentrated band-wise round robin did not select exactly 496 pairs: "
            f"selected={len(selected)}"
        )
    return selected


def _breadth_capacity_feasibility_receipt(
    *,
    broad_images: Sequence[str],
    pairs_by_image: Mapping[str, Sequence[Mapping[str, Any]]],
    image_bands: Mapping[str, str],
    target_histogram: Mapping[str, Mapping[str, int]],
    mode: str,
) -> dict[str, Any]:
    """Prove whether interval capacities can realize a requested histogram.

    Every broad image supplies a contiguous interval of ranks ``1..capacity``.
    For such intervals, the descending threshold inequalities are necessary
    and sufficient.  The receipt records those inequalities rather than only
    reporting the outcome of the later greedy construction.
    """

    if mode not in {"exact", "coarse"}:
        raise AssemblyError(f"unsupported rank matching feasibility mode: {mode}")
    minimum_rank = (
        (lambda key: int(key))
        if mode == "exact"
        else (lambda key: 4 if key == "4_plus" else int(key))
    )
    band_receipts: dict[str, Any] = {}
    overall_feasible = True
    for band in _BREADTH_BANDS:
        images = [image for image in broad_images if image_bands[image] == band]
        capacities = {image: len(pairs_by_image[image]) for image in images}
        capacity_histogram = Counter(capacities.values())
        target = {str(key): int(value) for key, value in target_histogram[band].items()}
        target_total = sum(target.values())
        thresholds = sorted({minimum_rank(key) for key in target}, reverse=True)
        checks: list[dict[str, Any]] = []
        band_feasible = target_total == len(images)
        for threshold in thresholds:
            demand = sum(
                count for key, count in target.items() if minimum_rank(key) >= threshold
            )
            supply = sum(capacity >= threshold for capacity in capacities.values())
            passed = supply >= demand
            band_feasible = band_feasible and passed
            checks.append(
                {
                    "minimum_rank": threshold,
                    "required_distinct_images": demand,
                    "available_distinct_images": supply,
                    "passed": passed,
                }
            )
        overall_feasible = overall_feasible and band_feasible
        band_receipts[band] = {
            "image_count": len(images),
            "target_pair_count": target_total,
            "target_histogram": target,
            "capacity_histogram": {
                str(capacity): count for capacity, count in sorted(capacity_histogram.items())
            },
            "threshold_checks": checks,
            "feasible": band_feasible,
        }
    return {
        "mode": mode,
        "interval_capacity_rule": "one distinct broad image supplies one rank from 1 through its trusted pair capacity",
        "bands": band_receipts,
        "feasible": overall_feasible,
    }


def _assign_broad_pairs_to_rank_histogram(
    *,
    broad_images: Sequence[str],
    pairs_by_image: Mapping[str, Sequence[Mapping[str, Any]]],
    image_bands: Mapping[str, str],
    target_histogram: Mapping[str, Mapping[str, int]],
    mode: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Assign one pair per broad image under exact or predeclared coarse ranks."""

    receipt = _breadth_capacity_feasibility_receipt(
        broad_images=broad_images,
        pairs_by_image=pairs_by_image,
        image_bands=image_bands,
        target_histogram=target_histogram,
        mode=mode,
    )
    if not receipt["feasible"]:
        return [], receipt
    minimum_rank = (
        (lambda key: int(key))
        if mode == "exact"
        else (lambda key: 4 if key == "4_plus" else int(key))
    )
    selected: list[dict[str, Any]] = []
    assignment_by_band: dict[str, list[dict[str, Any]]] = {}
    for band in _BREADTH_BANDS:
        images = [image for image in broad_images if image_bands[image] == band]
        available = set(images)
        assignments: list[dict[str, Any]] = []
        target = {str(key): int(value) for key, value in target_histogram[band].items()}
        for key in sorted(target, key=lambda value: (-minimum_rank(value), value)):
            required_rank = minimum_rank(key)
            for _ in range(target[key]):
                candidates = [
                    image for image in available if len(pairs_by_image[image]) >= required_rank
                ]
                if not candidates:  # guarded by the interval feasibility proof
                    raise AssemblyError(
                        "feasible breadth rank assignment construction unexpectedly exhausted supply"
                    )
                image = min(
                    candidates,
                    key=lambda value: (len(pairs_by_image[value]), _breadth_image_order(value)),
                )
                available.remove(image)
                pair = dict(pairs_by_image[image][required_rank - 1])
                selected.append(pair)
                assignments.append(
                    {
                        "image_id": image,
                        "requested_rank_or_group": key,
                        "selected_rank": required_rank,
                        "trusted_pair_capacity": len(pairs_by_image[image]),
                    }
                )
        if available:
            raise AssemblyError(
                f"breadth rank assignment left images unassigned in band {band}: {len(available)}"
            )
        assignment_by_band[band] = assignments
    receipt["assignment_by_band"] = assignment_by_band
    receipt["selected_exact_rank_histogram"] = _breadth_pair_rank_histogram(selected)
    receipt["selected_coarse_rank_histogram"] = _breadth_pair_rank_group_histogram(selected)
    return selected, receipt


def validate_constant_dose_panel_execution_contract(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize the validated request-scoped batch-one execution metadata.

    Old/new trajectory-union provenance is validated upstream.  This selector
    retains only the canonical execution metadata that binds the frozen event
    reservoir, plus any additional producer evidence unchanged.
    """

    batch_size = contract.get("physical_batch_size")
    if batch_size != 1:
        raise AssemblyError(
            "constant-dose trajectory panel requires physical_batch_size=1"
        )
    if contract.get("sampling_order") != "request_major":
        raise AssemblyError(
            "constant-dose trajectory panel requires sampling_order='request_major'"
        )
    if contract.get("rng_reset") != "per_image_seed":
        raise AssemblyError(
            "constant-dose trajectory panel requires rng_reset='per_image_seed'"
        )
    return copy.deepcopy(dict(contract))


def select_constant_dose_breadth_arms(
    *,
    sampled_candidates: Mapping[str, Sequence[Mapping[str, Any]]],
    source_candidates: Mapping[str, Sequence[Mapping[str, Any]]],
    training_image_bands: Mapping[str, str],
    trajectory_panel_execution_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze the matched 496-image broad and nested 118-image arms.

    Each selected pair has one sampled treatment row and one trusted Source
    row.  Concentrated round-robins 124 whole pairs independently inside each
    object-count band.  Broad uses one pair from each of 496 distinct images
    and first attempts to match the concentrated band-by-exact-rank histogram.
    It may then try only the unit's declared rank-1/rank-2/rank-3/rank-4-plus
    coarsening.  If neither is feasible, rank one is retained as an explicitly
    policy-only comparison.  All weights are assigned locally so both arms
    have exactly 992 events, mean weight one, and total weight 992.
    """

    execution_contract = validate_constant_dose_panel_execution_contract(
        trajectory_panel_execution_metadata
    )
    image_bands = {str(image): str(band) for image, band in training_image_bands.items()}
    pairs_by_image = _breadth_pair_pools(
        sampled_candidates=sampled_candidates,
        source_candidates=source_candidates,
        image_bands=image_bands,
    )
    eligible_by_band = {
        band: sorted(
            (image for image, pairs in pairs_by_image.items() if pairs and image_bands[image] == band),
            key=_breadth_image_order,
        )
        for band in _BREADTH_BANDS
    }
    eligible_image_count = sum(len(images) for images in eligible_by_band.values())
    if eligible_image_count < BREADTH_BROAD_IMAGE_COUNT:
        raise AssemblyError(
            "broad unique-image feasibility is below 496: "
            f"eligible_unique_training_images={eligible_image_count}"
        )
    broad_quota = _breadth_quota(BREADTH_BROAD_IMAGE_COUNT)
    insufficient = {
        band: {"available": len(eligible_by_band[band]), "required": count}
        for band, count in broad_quota.items()
        if len(eligible_by_band[band]) < count
    }
    if insufficient:
        raise AssemblyError(f"broad band-balanced image selection is infeasible: {insufficient}")
    broad_images = [
        image
        for band in _BREADTH_BANDS
        for image in eligible_by_band[band][: broad_quota[band]]
    ]
    concentrated_quota = _breadth_quota(BREADTH_CONCENTRATED_IMAGE_COUNT)
    concentrated_images = [
        image
        for band in _BREADTH_BANDS
        for image in broad_images
        if image_bands[image] == band
    ]
    concentrated_images = [
        image
        for band in _BREADTH_BANDS
        for image in sorted(
            (image for image in concentrated_images if image_bands[image] == band),
            key=_breadth_image_order,
        )[: concentrated_quota[band]]
    ]
    if len(concentrated_images) != BREADTH_CONCENTRATED_IMAGE_COUNT:
        raise AssemblyError("nested concentrated image selection did not reach 118 images")

    def events_for_pairs(pairs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for pair in pairs:
            for family, candidate in (("treatment", pair["sampled"]), ("source_preservation", pair["source"])):
                candidate = dict(candidate)
                route_count = _breadth_int(candidate, "route_count")
                events.append(
                    {
                        **candidate,
                        "image_id": str(pair["image_id"]),
                        "object_count_band": str(pair["object_count_band"]),
                        "event_family": family,
                        "selection_rank": int(pair["selection_rank"]),
                        "route_count": route_count,
                        "row_depth": _breadth_int(candidate, "generated_row_index") + 1,
                        "complete_row_coordinate_token_supervision": (
                            _breadth_complete_row_coordinate_token_supervision(candidate)
                        ),
                    }
                )
        return events

    pair_quota_by_band = _breadth_quota(BREADTH_EVENT_COUNT)
    concentrated_pairs = _select_concentrated_pairs_by_band_round_robin(
        pairs_by_image=pairs_by_image,
        concentrated_images=concentrated_images,
        image_bands=image_bands,
        pair_quota_by_band=pair_quota_by_band,
    )
    concentrated_exact_histogram = _breadth_pair_rank_histogram(concentrated_pairs)
    concentrated_coarse_histogram = _breadth_pair_rank_group_histogram(
        concentrated_pairs
    )
    broad_pairs, exact_feasibility = _assign_broad_pairs_to_rank_histogram(
        broad_images=broad_images,
        pairs_by_image=pairs_by_image,
        image_bands=image_bands,
        target_histogram=concentrated_exact_histogram,
        mode="exact",
    )
    coarse_feasibility: dict[str, Any] = {
        "mode": "coarse",
        "attempted": False,
        "feasible": None,
    }
    if broad_pairs:
        matching_mode = "exact_band_by_selection_rank"
        interpretation_scope = "image_breadth_with_exact_band_by_selection_rank_matching"
    else:
        broad_pairs, coarse_feasibility = _assign_broad_pairs_to_rank_histogram(
            broad_images=broad_images,
            pairs_by_image=pairs_by_image,
            image_bands=image_bands,
            target_histogram=concentrated_coarse_histogram,
            mode="coarse",
        )
        coarse_feasibility["attempted"] = True
        if broad_pairs:
            matching_mode = "coarse_band_by_rank_1_2_3_4_plus"
            interpretation_scope = "image_breadth_with_predeclared_coarse_band_by_rank_matching"
        else:
            broad_pairs = [dict(pairs_by_image[image][0]) for image in broad_images]
            matching_mode = "policy_only_broad_rank_one"
            interpretation_scope = (
                "data_allocation_policy_comparison_only_not_an_image_breadth_alone_claim"
            )
    for arm_name, pairs in (
        ("broad", broad_pairs),
        ("concentrated", concentrated_pairs),
    ):
        observed = Counter(str(pair["object_count_band"]) for pair in pairs)
        expected = Counter(pair_quota_by_band)
        if observed != expected:
            raise AssemblyError(
                f"{arm_name} selected-pair band counts differ from the frozen quota: "
                f"observed={dict(observed)}, expected={dict(expected)}"
            )
    broad_events = events_for_pairs(broad_pairs)
    concentrated_events = events_for_pairs(concentrated_pairs)

    def arm(name: str, events: list[dict[str, Any]], image_ids: Sequence[str]) -> dict[str, Any]:
        if len(events) != 2 * BREADTH_EVENT_COUNT:
            raise AssemblyError(f"{name} does not contain exactly 992 events")
        family_counts: dict[str, dict[str, int]] = defaultdict(
            lambda: {"treatment": 0, "source_preservation": 0}
        )
        identities: set[tuple[str, str, str, str]] = set()
        for event in events:
            identity = _breadth_identity(event)
            if identity in identities:
                raise AssemblyError(f"{name} contains a cloned exact event: {identity}")
            identities.add(identity)
            family_counts[str(event["image_id"])][str(event["event_family"])] += 1
        if set(family_counts) != set(image_ids):
            raise AssemblyError(f"{name} event images do not match its frozen image cohort")
        weights = image_family_event_weights(family_counts, event_count=len(events))
        weighted = [
            {
                **event,
                "image_balanced_event_weight": weights[
                    (str(event["image_id"]), str(event["event_family"]))
                ],
            }
            for event in events
        ]
        total = math.fsum(float(event["image_balanced_event_weight"]) for event in weighted)
        if not math.isclose(total, float(2 * BREADTH_EVENT_COUNT), rel_tol=0.0, abs_tol=1e-9):
            raise AssemblyError(f"{name} total event weight is not 992: {total}")
        if not math.isclose(total / len(weighted), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise AssemblyError(f"{name} mean event weight is not one")
        return {
            "arm": name,
            "image_ids": list(image_ids),
            "events": weighted,
            "event_count": len(weighted),
            "sampled_event_count": sum(event["event_family"] == "treatment" for event in weighted),
            "source_event_count": sum(event["event_family"] == "source_preservation" for event in weighted),
            "mean_event_weight": total / len(weighted),
            "total_event_weight": total,
            "image_family_event_counts": {image: family_counts[image] for image in sorted(family_counts, key=_breadth_image_order)},
            "selection_distributions": _breadth_distribution(weighted),
        }

    broad_arm = arm("broad", broad_events, broad_images)
    concentrated_arm = arm("concentrated", concentrated_events, concentrated_images)
    matching_receipt = {
        "matching_mode": matching_mode,
        "interpretation_scope": interpretation_scope,
        "pair_quota_by_object_count_band": pair_quota_by_band,
        "concentrated_exact_rank_histogram_by_band": concentrated_exact_histogram,
        "concentrated_coarse_rank_histogram_by_band": concentrated_coarse_histogram,
        "broad_exact_rank_histogram_by_band": _breadth_pair_rank_histogram(broad_pairs),
        "broad_coarse_rank_histogram_by_band": _breadth_pair_rank_group_histogram(
            broad_pairs
        ),
        "exact_rank_feasibility": exact_feasibility,
        "coarse_rank_feasibility": coarse_feasibility,
    }
    broad_arm["rank_matching_receipt"] = copy.deepcopy(matching_receipt)
    concentrated_arm["rank_matching_receipt"] = copy.deepcopy(matching_receipt)
    broad_arm["trajectory_panel_execution_metadata"] = copy.deepcopy(execution_contract)
    concentrated_arm["trajectory_panel_execution_metadata"] = copy.deepcopy(execution_contract)
    return {
        "trajectory_panel_execution_metadata": execution_contract,
        "eligible_unique_training_image_count": eligible_image_count,
        "eligible_images_by_band": {band: len(images) for band, images in eligible_by_band.items()},
        "broad_band_quota": broad_quota,
        "concentrated_band_quota": concentrated_quota,
        "pair_quota_by_object_count_band": pair_quota_by_band,
        "rank_matching_receipt": matching_receipt,
        "broad": broad_arm,
        "concentrated": concentrated_arm,
    }


def validate_constant_dose_training_reservoir(
    *,
    training_image_bands: Mapping[str, str],
    development_image_ids: Sequence[str],
    heldout_image_ids: Sequence[str],
    sampled_candidates: Mapping[str, Sequence[Mapping[str, Any]]],
    source_candidates: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    """Fail closed when route admission candidates cross the frozen split.

    The candidate maps must contain training-candidate images only.  Passing
    the frozen development and held-out identities here makes accidental
    admission leakage visible before either arm is selected or written.
    """

    training = {str(image) for image in training_image_bands}
    development = {str(image) for image in development_image_ids}
    heldout = {str(image) for image in heldout_image_ids}
    overlaps = {
        "training_development": sorted(training & development),
        "training_heldout": sorted(training & heldout),
        "development_heldout": sorted(development & heldout),
    }
    if any(overlaps.values()):
        raise AssemblyError(f"frozen candidate-pool split leakage: {overlaps}")
    admitted = set(sampled_candidates) | set(source_candidates)
    escaped = sorted(admitted - training)
    if escaped:
        raise AssemblyError(
            "route admission candidates are not confined to the frozen training reservoir: "
            f"{escaped}"
        )


def materialize_constant_dose_breadth_arm(
    selection: Mapping[str, Any], *, checkpoint_id: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Turn a frozen breadth selection into normal StateBank input rows.

    This is deliberately a narrow bridge: selected candidates must still
    carry the exact rollout/reference/entity inputs produced by
    ``_build_multi_candidates`` and ``_build_source_candidates``.  It keeps
    StateBank serialization and validation in the established assembler while
    attaching the breadth-specific difficulty receipt to every selected row.
    """

    arm_name = str(selection.get("arm", ""))
    events = selection.get("events")
    image_ids = selection.get("image_ids")
    if arm_name not in {"broad", "concentrated"}:
        raise AssemblyError("breadth selection lacks arm name")
    if not isinstance(events, list) or not isinstance(image_ids, list):
        raise AssemblyError("breadth selection lacks event/image lists")
    treatment_rollouts: list[dict[str, Any]] = []
    treatment_reviews: list[dict[str, Any]] = []
    treatment_receipts: list[dict[str, Any]] = []
    source_rollouts: list[dict[str, Any]] = []
    source_reviews: list[dict[str, Any]] = []
    source_receipts: list[dict[str, Any]] = []
    for selected in events:
        if not isinstance(selected, Mapping):
            raise AssemblyError("breadth selection event is not an object")
        family = str(selected.get("event_family", ""))
        inputs = selected.get("_event_inputs")
        if not isinstance(inputs, Mapping):
            raise AssemblyError(
                f"breadth selection event {selected.get('event_id', '<unknown>')} lacks exact event inputs"
            )
        route_id = str(selected.get("route_id", ""))
        if not route_id:
            raise AssemblyError("breadth selection event lacks route_id")
        route_seed = _breadth_int(selected, "route_seed")
        event_family = "source_preservation" if family == "source_preservation" else "multi_route_treatment"
        if family not in {"treatment", "source_preservation"}:
            raise AssemblyError(f"unsupported breadth event family: {family!r}")
        event = _make_route_event(
            image_result=_require_mapping(inputs.get("image_result"), "breadth image_result"),
            route_id=route_id,
            seed=route_seed,
            row=selected,
            rollout=_require_mapping(inputs.get("rollout"), "breadth rollout"),
            reference=_require_mapping(inputs.get("reference"), "breadth reference"),
            entities=inputs.get("entities", []),
            checkpoint_id=checkpoint_id,
            family=event_family,
            event_weight=float(selected.get("image_balanced_event_weight", 1.0)),
        )
        selection_metadata = {
            field: selected[field]
            for field in (
                "selection_rank",
                "route_count",
                "row_depth",
                "complete_row_coordinate_token_supervision",
                "object_count_band",
            )
        }
        review = copy.deepcopy(event[1])
        provenance = dict(review["review_provenance"])
        provenance["constant_dose_breadth_selection"] = selection_metadata
        review["review_provenance"] = provenance
        receipt = dict(event[2])
        receipt["constant_dose_breadth_selection"] = selection_metadata
        if family == "treatment":
            treatment_rollouts.append(event[0])
            treatment_reviews.append(review)
            treatment_receipts.append(receipt)
        else:
            source_rollouts.append(event[0])
            source_reviews.append(review)
            source_receipts.append(receipt)
    assembled = _arm_from_events(
        treatment_rollouts=treatment_rollouts,
        treatment_reviews=treatment_reviews,
        treatment_receipts=treatment_receipts,
        source_rollouts=source_rollouts,
        source_reviews=source_reviews,
        source_receipts=source_receipts,
        family=f"constant_dose_{arm_name}_plus_source_preservation",
        image_ids=[str(image) for image in image_ids],
    )
    assembled[3]["constant_dose_breadth_selection"] = {
        "arm": arm_name,
        "image_ids": [str(image) for image in image_ids],
        "trajectory_panel_execution_metadata": copy.deepcopy(
            selection.get("trajectory_panel_execution_metadata", {})
        ),
        "selection_distributions": copy.deepcopy(selection.get("selection_distributions", {})),
        "rank_matching_receipt": copy.deepcopy(selection.get("rank_matching_receipt", {})),
        "total_event_weight": float(selection.get("total_event_weight", 0.0)),
        "mean_event_weight": float(selection.get("mean_event_weight", 0.0)),
    }
    return assembled


def reject_blind_images(image_ids: Sequence[str]) -> None:
    """Reject the canonical twelve-image blind development cohort."""

    overlap = sorted({int(str(item)) for item in image_ids} & set(BLIND_IMAGE_IDS))
    if overlap:
        raise AssemblyError(f"blind evaluation images are forbidden in StateBanks: {overlap}")


def _prediction_receipt(
    rollout: Mapping[str, Any], image_id: str, row_index: int
) -> dict[str, Any]:
    """Materialize one unresolved receipt from a full greedy rollout.

    The trajectory analysis caps its owner-assignment budget at sixteen rows,
    while a greedy rollout can naturally contain more rows.  Manual crop
    review may therefore promote a later row.  We copy the parser's exact raw
    span and coordinates into a receipt without assigning an owner here; the
    sidecar override below supplies only the reviewed owner and IoU.
    """

    predictions = _require_mapping(rollout.get("predictions"), f"greedy rollout {image_id}.predictions")
    rows = predictions.get("predictions")
    if not isinstance(rows, list):
        raise AssemblyError(f"greedy rollout {image_id}.predictions.predictions must be a list")
    if row_index < 0 or row_index >= len(rows):
        raise AssemblyError(
            f"manual Source row {image_id}:{row_index} exceeds parsed greedy row count {len(rows)}"
        )
    parsed = _require_mapping(rows[row_index], f"greedy rollout {image_id} parsed row {row_index}")
    raw = {
        "bbox": list(parsed.get("bbox", [])),
        "bbox_format": str(parsed.get("bbox_format", "xyxy")),
        "char_end": parsed.get("char_end"),
        "char_start": parsed.get("char_start"),
        "coord_bins": [int(item) for item in parsed.get("coord_bins", [])],
        "coord_token_spans": copy.deepcopy(parsed.get("coord_token_spans", [])),
        "description": str(parsed.get("description", "")),
        "generated_order": int(parsed.get("generated_order", row_index)),
        "object_span_id": str(parsed.get("object_span_id", "")),
        "raw_span_sha256": str(parsed.get("raw_span_sha256", "")),
        "raw_span_text": str(parsed.get("raw_span_text", "")),
        "schema_spans": copy.deepcopy(parsed.get("schema_spans", [])),
    }
    if len(raw["coord_bins"]) != 4:
        raise AssemblyError(f"manual Source row {image_id}:{row_index} lacks four coordinate bins")
    return {
        "bbox": list(parsed.get("bbox", [])),
        "candidate_owner_id": None,
        "candidate_owner_iou": None,
        "category": str(parsed.get("description", "")),
        "decode_mode": str(rollout.get("decode_mode", "greedy")),
        "entity_status": "unresolved_pending_crop_review",
        "generated_row_index": int(row_index),
        "geometry_status": "unresolved",
        "image_id": image_id,
        "intersection_over_union": None,
        "owner_id": None,
        "prediction_id": str(parsed.get("object_span_id", "")),
        "raw": raw,
        "review_required": True,
        "seed": int(rollout.get("seed", 0)),
        "trajectory_id": "greedy",
    }


def _manual_assignment_override(
    assignment: Mapping[str, Any],
    decision: Mapping[str, Any],
    *,
    rollout: Mapping[str, Any],
    image_id: str,
) -> Mapping[str, Any]:
    """Apply one sidecar owner decision to an assignment copy."""

    row_index = int(decision["row_index"])
    override = copy.deepcopy(dict(assignment))
    raw_rows = override.get("row_assignment_receipts")
    if not isinstance(raw_rows, list):
        raise AssemblyError(f"greedy assignment {image_id} row receipts must be a list")
    rows = [dict(item) for item in raw_rows if isinstance(item, Mapping)]
    by_index = {int(item["generated_row_index"]): item for item in rows}
    target = by_index.get(row_index)
    if target is None:
        target = _prediction_receipt(rollout, image_id, row_index)
        rows.append(target)
    existing_owner = target.get("owner_id")
    requested_owner = str(decision["owner_id"])
    if existing_owner is not None and str(existing_owner) != requested_owner:
        raise AssemblyError(
            f"manual Source row {image_id}:{row_index} conflicts with existing owner {existing_owner}"
        )
    target.update(
        {
            "owner_id": requested_owner,
            "candidate_owner_id": requested_owner,
            "candidate_owner_iou": float(decision["iou"]),
            "category": str(decision["category"]),
            "entity_status": "verified_owner",
            "geometry_status": "trusted",
            "intersection_over_union": float(decision["iou"]),
            "review_required": False,
            "manual_review": {
                "decision": "admit",
                "reviewer": str(decision.get("reviewer", "")),
                "review_source": str(decision.get("review_source", "")),
                "comment": str(decision.get("comment", "")),
            },
        }
    )
    override["row_assignment_receipts"] = sorted(rows, key=lambda item: int(item["generated_row_index"]))
    return override


def _exact_complete_row_slices_allow_trailing_partial(
    generated_ids: Sequence[int], row_index: int
) -> tuple[list[int], list[int]]:
    """Slice a complete row while ignoring an unparsed trailing suffix.

    Some accepted greedy artifacts stop at the generation cap after their last
    complete object row.  The assignment receipt still certifies earlier rows;
    a trailing partial span must not prevent preserving those exact tokens.
    """

    tokens = _token_ids(list(generated_ids), "generated_token_ids")
    complete: list[list[int]] = []
    start = 0
    for index, token in enumerate(tokens):
        if int(token) != BOX_END:
            continue
        row = tokens[start : index + 1]
        exact_row_site_types(row)
        complete.append(row)
        start = index + 1
    if int(row_index) < 0 or int(row_index) >= len(complete):
        raise AssemblyError(
            f"row index {row_index} exceeds complete generated row count {len(complete)}"
        )
    prefix = [token for row in complete[: int(row_index)] for token in row]
    return prefix, list(complete[int(row_index)])


def _candidate_rows_for_route(
    *,
    image_result: Mapping[str, Any],
    route_id: str,
    cutoff: int,
    owners: Sequence[Mapping[str, Any]],
    generated_ids: Sequence[int],
    assignment_override: Mapping[str, Any] | None = None,
    allow_geometry_unknown: bool = False,
    geometry_untrusted_allowlist: set[tuple[str, str, int, str]] | None = None,
) -> list[dict[str, Any]]:
    selected_budget = next(
        item for item in image_result["budgets"] if isinstance(item, Mapping) and int(item.get("budget", -1)) == 16
    )
    assignment = (
        assignment_override
        if assignment_override is not None
        else _require_mapping(
            _require_mapping(selected_budget.get("trajectory_assignments"), "trajectory_assignments").get(route_id),
            f"trajectory_assignments[{route_id}]",
        )
    )
    receipts = [
        _require_mapping(item, f"trajectory_assignments[{route_id}].row_assignment_receipts[{i}]")
        for i, item in enumerate(assignment.get("row_assignment_receipts", []))
    ]
    receipts_by_index = {int(item["generated_row_index"]): item for item in receipts}
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for row_index in range(int(cutoff) + 1):
        receipt = receipts_by_index.get(row_index)
        if receipt is None:
            continue
        owner = receipt.get("owner_id")
        if owner is None or str(owner) in seen:
            continue
        geometry = _row_geometry_receipt(receipt, owners)
        owner_known = str(geometry.get("owner_id")) in {str(item.get("owner_id")) for item in owners}
        allowlisted_geometry = (
            str(image_result.get("image_id")),
            str(route_id),
            int(row_index),
            str(geometry.get("owner_id", "")),
        ) in (geometry_untrusted_allowlist or set()) and owner_known and geometry["entity_status"] == "verified_owner"
        if not geometry["gradient_eligible"] and not (
            (allow_geometry_unknown and owner_known and geometry["entity_status"] == "verified_owner")
            or allowlisted_geometry
        ):
            continue
        prefix, row = _exact_complete_row_slices_allow_trailing_partial(generated_ids, row_index)
        raw = _require_mapping(receipt.get("raw"), f"route {route_id} row {row_index}.raw")
        bins = [int(item) for item in raw.get("coord_bins", [])]
        actual = [int(token) - COORDINATE_TOKEN_START for token in row if COORDINATE_TOKEN_START <= int(token) < COORDINATE_TOKEN_END]
        if actual != bins:
            raise AssemblyError(f"route {route_id} row {row_index} coordinate tokens disagree with parser bins")
        seen.add(str(owner))
        result.append(
            {
                "route_id": route_id,
                "generated_row_index": row_index,
                "owner_id": str(owner),
                "geometry": geometry,
                "prefix_token_ids": prefix,
                "candidate_token_ids": row,
                "prefix_token_ids_sha256": token_ids_sha256(prefix),
                "candidate_token_ids_sha256": token_ids_sha256(row),
                "receipt": dict(receipt),
            }
        )
    return result


def _route_assignment(
    image_result: Mapping[str, Any], route_id: str, *, budget: int = 16
) -> Mapping[str, Any]:
    selected_budget = next(
        item for item in image_result["budgets"] if isinstance(item, Mapping) and int(item.get("budget", -1)) == int(budget)
    )
    assignments = _require_mapping(selected_budget.get("trajectory_assignments"), "trajectory_assignments")
    return _require_mapping(assignments.get(route_id), f"trajectory_assignments[{route_id}]")


def _prefix_receipt(
    *, row_index: int, assignment: Mapping[str, Any]
) -> tuple[str, list[dict[str, Any]], int]:
    receipts = {
        int(item["generated_row_index"]): item
        for item in assignment.get("row_assignment_receipts", [])
        if isinstance(item, Mapping)
    }
    malformed = int(_route_row_counts(assignment).get("malformed", assignment.get("malformed_row_count", 0) or 0))
    status, proofs = _prefix_status(row_index=row_index, receipts_by_index=receipts, malformed_count=malformed)
    unresolved = sum(
        1
        for index in range(row_index)
        if index in receipts and str(receipts[index].get("entity_status")) != "verified_owner"
    )
    return status, proofs, unresolved


def _make_route_event(
    *,
    image_result: Mapping[str, Any],
    route_id: str,
    seed: int,
    row: Mapping[str, Any],
    rollout: Mapping[str, Any],
    reference: Mapping[str, Any],
    entities: Sequence[Mapping[str, Any]],
    checkpoint_id: str,
    family: str,
    event_weight: float,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    image = _image_id(image_result.get("image_id"))
    generated_ids = _token_ids(rollout.get("generated_token_ids"), f"rollout {image}.generated_token_ids")
    generated_hash = _norm_sha(rollout.get("generated_token_ids_sha256"), f"rollout {image}.generated_token_ids_sha256")
    if generated_hash != token_ids_sha256(generated_ids):
        raise AssemblyError(f"rollout {image}, route {route_id} generated-token hash mismatch")
    prompt_ids = _token_ids(reference.get("executed_prompt_token_ids"), f"reference {image}.executed_prompt_token_ids")
    prompt_hash = _norm_sha(reference.get("executed_prompt_token_ids_sha256"), f"reference {image}.executed_prompt_token_ids_sha256")
    if prompt_hash != token_ids_sha256(prompt_ids):
        raise AssemblyError(f"reference {image} prompt-token hash mismatch")
    if rollout.get("prompt_token_ids") != prompt_ids:
        raise AssemblyError(f"route {route_id} prompt token IDs differ from reference")
    assignment = _route_assignment(image_result, route_id)
    row_index = int(row["generated_row_index"])
    prefix_ids = list(row["prefix_token_ids"])
    candidate_ids = list(row["candidate_token_ids"])
    prefix_status, prefix_proofs, unresolved_prefix = _prefix_receipt(row_index=row_index, assignment=assignment)
    raw = _require_mapping(row["receipt"].get("raw"), f"route {route_id} row {row_index}.raw")
    source_family = family == "source_preservation"
    mode = "greedy" if source_family else "sampled"
    temperature = float(rollout.get("_temperature", 0.0 if mode == "greedy" else 0.4))
    top_p = float(rollout.get("_top_p", 1.0 if mode == "greedy" else 0.95))
    repetition_penalty = float(rollout.get("_repetition_penalty", 1.0))
    event_id = f"{family.replace('_', '-')}-image-{image}-route-{route_id}-row-{row_index}"
    candidate_id = f"{event_id}-candidate"
    generation = {
        "mode": mode,
        "seed": int(seed),
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "checkpoint_id": checkpoint_id,
        "prompt_token_ids_sha256": prompt_hash,
        "prefix_token_ids_sha256": token_ids_sha256(prefix_ids),
    }
    candidate = {
        "candidate_id": candidate_id,
        "token_ids": candidate_ids,
        "token_ids_sha256": token_ids_sha256(candidate_ids),
        "generation_provenance": generation,
        "evidence_text": str(raw.get("raw_span_text", "")),
    }
    review_candidate = {
        "candidate_id": candidate_id,
        "role": "positive",
        "harmful_kind": None,
        "physical_owner_id": str(row["owner_id"]),
        "coverage_status": "uncovered",
        "entity_review_status": "trusted",
        "geometry_review_status": "trusted" if row["geometry"]["geometry_trusted"] else "unknown",
        "entity_eligible": True,
        "geometry_eligible": False,
        "owner_resolution_interval": [0, len(candidate_ids)],
        "coordinate_decision": None,
        "selected_sites": exact_row_site_types(candidate_ids),
    }
    rollout_row = {
        "event_id": event_id,
        "image": dict(_require_mapping(reference.get("image"), f"reference {image}.image")),
        "split": str(reference.get("split")),
        "split_group_id": f"image:{image}",
        "executed_prompt_token_ids": prompt_ids,
        "executed_prompt_token_ids_sha256": prompt_hash,
        "image_pad_interval": list(reference.get("image_pad_interval") or _image_pad_interval(prompt_ids)),
        "prefix_token_ids": prefix_ids,
        "prefix_token_ids_sha256": token_ids_sha256(prefix_ids),
        "candidates": [candidate],
    }
    review_row = {
        "event_id": event_id,
        "admission_status": "accepted",
        "rejection_reason": None,
        "physical_entities": [dict(item) for item in entities],
        "prefix_object_row_count": row_index,
        "prefix_coverage_status": prefix_status,
        "prefix_covered_owner_proofs": prefix_proofs,
        "entity_transition_eligible": False,
        "coordinate_boundary_eligible": False,
        "positive_path_imitation_eligible": not source_family,
        "source_route_imitation_eligible": source_family,
        "image_balanced_event_weight": float(event_weight),
        "candidates": [review_candidate],
        "review_provenance": {
            "schema_version": SCHEMA_VERSION,
            "policy": "matched-source-preservation-and-multi-route-positive-row-imitation",
            "event_family": family,
            "decode_mode": mode,
            "route_id": route_id,
            "seed": int(seed),
            "generated_row_index": row_index,
            "geometry_owner_rule": {
                "entity_status": "verified_owner",
                "unique_gt_category_or_iou_at_least": GEOMETRY_IOU_THRESHOLD,
                "category_owner_count": int(row["geometry"]["category_owner_count"]),
                "intersection_over_union": float(row["geometry"]["intersection_over_union"]),
                "reason": str(row["geometry"]["reason"]),
            },
            "prefix_context": {
                "coverage_status": prefix_status,
                "unresolved_rows": unresolved_prefix,
                "duplicate_count": int(_route_row_counts(assignment).get("duplicate", 0)),
                "malformed_count": int(_route_row_counts(assignment).get("malformed", 0)),
            },
        },
    }
    if row.get("manual_review") is not None:
        review_row["review_provenance"]["source_anchor_manual_review"] = copy.deepcopy(row["manual_review"])
    selection_metadata = {
        field: row[field]
        for field in (
            "marginal_route_added_owner_count",
            "route_added_owner_count",
            "unresolved_row_count",
            "route_count",
            "trusted_source_anchor_order",
        )
        if field in row
    }
    if selection_metadata:
        review_row["review_provenance"]["selection_metadata"] = selection_metadata
    receipt = {
        "event_id": event_id,
        "image_id": image,
        "event_family": family,
        "route_id": route_id,
        "seed": int(seed),
        "generated_row_index": row_index,
        "owner_id": str(row["owner_id"]),
        "geometry": dict(row["geometry"]),
        "candidate_token_count": len(candidate_ids),
        "candidate_token_ids_sha256": token_ids_sha256(candidate_ids),
        "prefix_token_ids_sha256": token_ids_sha256(prefix_ids),
        "image_balanced_event_weight": float(event_weight),
    }
    if selection_metadata:
        receipt["selection_metadata"] = selection_metadata
    return rollout_row, review_row, receipt


def _set_event_weight(
    rollout: Mapping[str, Any], review: Mapping[str, Any], receipt: Mapping[str, Any], weight: float
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    review_copy = copy.deepcopy(dict(review))
    review_copy["image_balanced_event_weight"] = float(weight)
    provenance = dict(review_copy["review_provenance"])
    provenance["image_balanced_event_weight"] = float(weight)
    review_copy["review_provenance"] = provenance
    receipt_copy = dict(receipt)
    receipt_copy["image_balanced_event_weight"] = float(weight)
    return dict(rollout), review_copy, receipt_copy


def _prior_rows(
    prior_manifest: Path,
    prior_receipt_path: Path,
    *,
    admitted_image_ids: Sequence[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], set[str]]:
    receipt = _read_json(prior_receipt_path)
    if not isinstance(receipt, Mapping):
        raise AssemblyError("prior assembly receipt must be an object")
    if receipt.get("gradient_event_count") != TARGET_EVENT_COUNT or receipt.get("gradient_bearing_image_count") != TARGET_IMAGE_COUNT:
        raise AssemblyError("prior StateBank receipt is not the exact 512-event/118-image source screen")
    root = prior_manifest.parent
    if root.name != "state-bank":
        raise AssemblyError("prior manifest must be the canonical state-bank/manifest.json")
    pre_root = root.parent / "pre-state-bank"
    rollout_rows = load_jsonl(pre_root / "rollout_rows.jsonl")
    review_rows = load_jsonl(pre_root / "review_rows.jsonl")
    if len(rollout_rows) != TARGET_EVENT_COUNT or len(review_rows) != TARGET_EVENT_COUNT:
        raise AssemblyError("prior pre-StateBank rows are not exactly 512 events")
    image_ids = {
        str(_mapping(row.get("image"), "prior rollout image").get("image_id")) for row in rollout_rows
    }
    if len(image_ids) != TARGET_IMAGE_COUNT:
        raise AssemblyError("prior pre-StateBank rows are not exactly 118 images")
    reject_blind_images(sorted(image_ids, key=int))
    receipt_images = {str(item["image_id"]) for item in receipt.get("image_event_counts", []) if isinstance(item, Mapping) and "image_id" in item}
    if receipt_images != image_ids:
        raise AssemblyError("prior receipt and pre-StateBank image sets differ")
    event_receipts = receipt.get("event_receipts")
    if not isinstance(event_receipts, list) or len(event_receipts) != TARGET_EVENT_COUNT:
        raise AssemblyError("prior receipt lacks exact best-route event receipts")
    # Preserve the prior event tokens and route IDs byte-for-byte; only the
    # new family flag/provenance is added in memory for the matched arm.
    normalized_reviews: list[dict[str, Any]] = []
    for row in review_rows:
        item = copy.deepcopy(row)
        item["positive_path_imitation_eligible"] = True
        item["source_route_imitation_eligible"] = False
        provenance = dict(item.get("review_provenance", {}))
        provenance["event_family"] = "single_route_treatment"
        provenance["decode_mode"] = "sampled"
        item["review_provenance"] = provenance
        normalized_reviews.append(item)
    if admitted_image_ids is None:
        return rollout_rows, normalized_reviews, dict(receipt), image_ids
    admitted = {str(item) for item in admitted_image_ids}
    present = admitted & image_ids
    filtered_rollouts: list[dict[str, Any]] = []
    filtered_reviews: list[dict[str, Any]] = []
    filtered_receipts: list[dict[str, Any]] = []
    for rollout, review, event_receipt in zip(
        rollout_rows,
        normalized_reviews,
        receipt["event_receipts"],
        strict=True,
    ):
        image = str(_mapping(rollout.get("image"), "prior rollout image").get("image_id"))
        if image in present:
            filtered_rollouts.append(dict(rollout))
            filtered_reviews.append(copy.deepcopy(review))
            filtered_receipts.append(dict(event_receipt))
    if not filtered_rollouts:
        raise AssemblyError("no prior treatment rows remain after source cohort filtering")
    replacement_ids = sorted(admitted - image_ids, key=int)
    filtered_receipt = copy.deepcopy(dict(receipt))
    filtered_receipt["event_receipts"] = filtered_receipts
    filtered_receipt["image_event_counts"] = [
        {"image_id": image, "event_count": sum(
            1 for row in filtered_rollouts
            if str(_mapping(row.get("image"), "prior rollout image").get("image_id")) == image
        )}
        for image in sorted(present, key=int)
    ]
    filtered_receipt["gradient_bearing_image_count"] = len(present)
    filtered_receipt["gradient_event_count"] = len(filtered_rollouts)
    filtered_receipt["source_cohort_filtering"] = {
        "prior_event_count": TARGET_EVENT_COUNT,
        "retained_event_count": len(filtered_rollouts),
        "removed_event_count": TARGET_EVENT_COUNT - len(filtered_rollouts),
        "replacement_image_ids": replacement_ids,
        "synthetic_event_count": 0,
        "route_supplementation_event_count": 0,
        "duplicate_exact_event_count": 0,
    }
    return filtered_rollouts, filtered_reviews, filtered_receipt, present


def _validate_source_anchor_cohort(
    image_ids: Sequence[str], source_candidates_by_image: Mapping[str, Sequence[Mapping[str, Any]]]
) -> None:
    missing = sorted((set(str(item) for item in image_ids) - set(source_candidates_by_image)), key=int)
    if missing:
        raise AssemblyError(
            "Source preservation cannot satisfy one-event-per-image for the admitted cohort; "
            f"images have no trusted greedy owner occurrence: {missing}"
        )


def _event_identity(
    rollout: Mapping[str, Any], review: Mapping[str, Any]
) -> tuple[str, str, str, str]:
    image = _image_id(_mapping(rollout.get("image"), "event image").get("image_id"))
    candidates = review.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 1:
        raise AssemblyError("treatment event must contain exactly one review candidate")
    candidate = _require_mapping(candidates[0], "treatment review candidate")
    rollout_candidates = rollout.get("candidates")
    if not isinstance(rollout_candidates, list) or len(rollout_candidates) != 1:
        raise AssemblyError("treatment rollout must contain exactly one candidate")
    rollout_candidate = _require_mapping(rollout_candidates[0], "treatment rollout candidate")
    return (
        image,
        str(rollout.get("prefix_token_ids_sha256", "")),
        str(rollout_candidate.get("token_ids_sha256", "")),
        str(candidate.get("physical_owner_id", "")),
    )


def _candidate_identity(item: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(item.get("image_id", "")),
        str(item.get("prefix_token_ids_sha256", "")),
        str(item.get("candidate_token_ids_sha256", item.get("token_ids_sha256", ""))),
        str(item.get("owner_id", item.get("physical_owner_id", ""))),
    )


def exclude_sampled_source_collisions(
    candidates_by_image: Mapping[str, Sequence[Mapping[str, Any]]],
    source_identities: set[tuple[str, str, str, str]] | Sequence[tuple[str, str, str, str]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Remove sampled candidates colliding with frozen Source identities.

    This filter is intentionally applied before image-diverse selection.  It
    preserves the route candidate pools and their family balance; a later
    global deduplication pass would silently reduce the required 496 sampled
    events.
    """

    blocked = set(source_identities)
    filtered: dict[str, list[dict[str, Any]]] = {}
    collisions: list[dict[str, Any]] = []
    for image in sorted((str(item) for item in candidates_by_image), key=lambda value: int(value)):
        keep: list[dict[str, Any]] = []
        for item in candidates_by_image[image]:
            identity = _candidate_identity(item)
            if identity in blocked:
                collisions.append(
                    {
                        "image_id": identity[0],
                        "route_id": str(item.get("route_id", "")),
                        "row_index": int(item.get("generated_row_index", -1)),
                        "owner_id": identity[3],
                        "prefix_token_ids_sha256": identity[1],
                        "candidate_token_ids_sha256": identity[2],
                    }
                )
                continue
            keep.append(dict(item))
        filtered[image] = keep
    return filtered, {
        "source_identity_count": len(blocked),
        "collision_event_count": len(collisions),
        "collision_events": collisions,
        "candidate_event_count_before": sum(len(items) for items in candidates_by_image.values()),
        "candidate_event_count_after": sum(len(items) for items in filtered.values()),
    }


def _validate_route_event_identity_uniqueness(
    *,
    treatment_rollouts: Sequence[Mapping[str, Any]],
    treatment_reviews: Sequence[Mapping[str, Any]],
    source_rollouts: Sequence[Mapping[str, Any]],
    source_reviews: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate exact semantic identities across both families in one arm."""

    if len(treatment_rollouts) != len(treatment_reviews):
        raise AssemblyError("treatment rollout/review counts differ during identity validation")
    if len(source_rollouts) != len(source_reviews):
        raise AssemblyError("Source rollout/review counts differ during identity validation")
    treatment_keys = [
        _event_identity(rollout, review)
        for rollout, review in zip(treatment_rollouts, treatment_reviews, strict=True)
    ]
    source_keys = [
        _event_identity(rollout, review)
        for rollout, review in zip(source_rollouts, source_reviews, strict=True)
    ]
    treatment_unique = len(set(treatment_keys))
    source_unique = len(set(source_keys))
    cross_family = set(treatment_keys) & set(source_keys)
    duplicate_count = len(treatment_keys) + len(source_keys) - len(set(treatment_keys + source_keys))
    receipt = {
        "treatment_identity_count": len(treatment_keys),
        "treatment_unique_identity_count": treatment_unique,
        "treatment_duplicate_identity_count": len(treatment_keys) - treatment_unique,
        "source_preservation_identity_count": len(source_keys),
        "source_preservation_unique_identity_count": source_unique,
        "source_preservation_duplicate_identity_count": len(source_keys) - source_unique,
        "cross_family_collision_count": len(cross_family),
        "combined_identity_count": len(treatment_keys) + len(source_keys),
        "combined_unique_identity_count": len(set(treatment_keys + source_keys)),
        "combined_duplicate_identity_count": duplicate_count,
        "combined_identity_unique": duplicate_count == 0,
    }
    if duplicate_count:
        raise AssemblyError(
            "final combined treatment/Source semantic identities are not unique: "
            f"duplicate_count={duplicate_count}, cross_family_collision_count={len(cross_family)}"
        )
    return receipt


def _validate_single_route_purity(
    reviews: Sequence[Mapping[str, Any]],
    *,
    expected_event_count: int = ARM_EVENT_COUNT,
    expected_image_count: int | None = None,
    expected_image_ids: Sequence[str] | None = None,
    rollouts: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Ensure every single-route treatment row has one parseable route/image.

    ``expected_image_count`` and ``expected_image_ids`` are optional so the
    helper remains useful for small focused tests.  The production rebuild
    supplies both and therefore fails closed on a partial cohort.
    """

    by_image: dict[str, set[str]] = defaultdict(set)
    if len(reviews) != int(expected_event_count):
        raise AssemblyError(
            "single-route treatment cannot reach the required event count: "
            f"{len(reviews)} != {expected_event_count}"
        )
    if rollouts is not None and len(rollouts) != len(reviews):
        raise AssemblyError("single-route treatment rollout/review counts differ during purity validation")
    parsed_images: list[str] = []
    for index, review in enumerate(reviews):
        provenance = review.get("review_provenance")
        if not isinstance(provenance, Mapping):
            raise AssemblyError(f"single-route treatment review {index} lacks review_provenance")
        event_id = str(review.get("event_id", ""))
        image_match = re.search(
            r"(?:^|-)image-(?P<image>[A-Za-z0-9_]+)(?:-route-|-seed-|-row-)",
            event_id,
        )
        if image_match is None:
            raise AssemblyError(
                f"single-route treatment review {index} has no parseable image identity: {event_id!r}"
            )
        image = str(image_match.group("image"))
        route_id = str(provenance.get("route_id", ""))
        if not route_id:
            raise AssemblyError(f"single-route treatment review {index} lacks non-empty route_id")
        if rollouts is not None:
            rollout_image = _mapping(rollouts[index].get("image"), f"treatment rollout {index}.image")
            observed_image = _image_id(rollout_image.get("image_id"), f"treatment rollout {index}.image_id")
            if str(observed_image) != image:
                raise AssemblyError(
                    f"single-route treatment review {index} image identity disagrees with rollout: "
                    f"{image} != {observed_image}"
                )
        parsed_images.append(image)
        by_image[image].add(route_id)
    offenders = {
        image: sorted(routes)
        for image, routes in by_image.items()
        if len(routes) > 1
    }
    if offenders:
        raise AssemblyError(
            "single-route treatment selected multiple sampled routes per image: "
            f"{offenders}"
        )
    represented = set(parsed_images)
    if expected_image_count is not None and len(represented) != int(expected_image_count):
        raise AssemblyError(
            "single-route treatment represented image count differs from the required cohort: "
            f"{len(represented)} != {expected_image_count}"
        )
    if expected_image_ids is not None:
        expected_images = {str(item) for item in expected_image_ids}
        if represented != expected_images:
            raise AssemblyError(
                "single-route treatment represented image IDs differ from the required cohort: "
                f"missing={sorted(expected_images - represented, key=str)}, "
                f"extra={sorted(represented - expected_images, key=str)}"
            )
    return {
        "event_count": len(reviews),
        "expected_event_count": int(expected_event_count),
        "image_count": len(represented),
        "expected_image_count": None if expected_image_count is None else int(expected_image_count),
        "multi_route_image_count": len(offenders),
        "route_ids_by_image": {image: sorted(routes) for image, routes in sorted(by_image.items(), key=lambda item: str(item[0]))},
        "route_supplementation_event_count": 0,
        "synthetic_event_count": 0,
    }


def _normalize_sampled_treatment_review(review: Mapping[str, Any]) -> dict[str, Any]:
    item = copy.deepcopy(dict(review))
    item["positive_path_imitation_eligible"] = True
    item["source_route_imitation_eligible"] = False
    provenance = dict(item.get("review_provenance", {}))
    provenance["event_family"] = "single_route_treatment"
    provenance["decode_mode"] = "sampled"
    item["review_provenance"] = provenance
    return item


def _rebuild_single_route_treatment_events(
    *,
    prior_rollouts: Sequence[Mapping[str, Any]],
    prior_reviews: Sequence[Mapping[str, Any]],
    prior_receipts: Sequence[Mapping[str, Any]],
    frozen_pool_rollouts: Sequence[Mapping[str, Any]],
    frozen_pool_reviews: Sequence[Mapping[str, Any]],
    frozen_pool_receipts: Sequence[Mapping[str, Any]],
    replacement_image_ids: Sequence[str],
    extra_candidates_by_image: Mapping[str, Sequence[Mapping[str, Any]]],
    image_ids: Sequence[str],
    checkpoint_id: str,
    excluded_identities: set[tuple[str, str, str, str]] | Sequence[tuple[str, str, str, str]] = (),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Rebuild 496 unique sampled events without synthetic events.

    The historical 512-row bank contributes its admitted events verbatim.  A
    removed image is replaced by the exact rows from the no-batch-fit frozen
    pool.  The resulting 511-row replacement pool is deterministically trimmed
    to the 496-event arm budget while retaining one event per image. Every
    selected event has a distinct exact
    ``(image,prefix,row,owner)`` identity.
    """

    if not (
        len(frozen_pool_rollouts)
        == len(frozen_pool_reviews)
        == len(frozen_pool_receipts)
    ):
        raise AssemblyError("frozen treatment pool triplets differ in length")
    blocked = set(excluded_identities)
    prior_triplets = list(
        zip(
            prior_rollouts,
            prior_reviews,
            prior_receipts,
            strict=True,
        )
    )
    retained_prior_triplets: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    source_collision_count = 0
    for rollout, review, receipt in prior_triplets:
        normalized_review = _normalize_sampled_treatment_review(review)
        identity = _event_identity(rollout, normalized_review)
        if identity in blocked:
            source_collision_count += 1
            continue
        retained_prior_triplets.append((dict(rollout), normalized_review, dict(receipt)))
    if not retained_prior_triplets:
        raise AssemblyError("filtered prior treatment rows are empty")
    pool = list(
        zip(
            frozen_pool_rollouts,
            frozen_pool_reviews,
            frozen_pool_receipts,
            strict=True,
        )
    )
    replacements = {
        str(item)
        for item in replacement_image_ids
    }
    replacement_triplets = [
        (dict(rollout), _normalize_sampled_treatment_review(review), dict(receipt))
        for rollout, review, receipt in pool
        if _image_id(_mapping(rollout.get("image"), "pool event image").get("image_id")) in replacements
    ]
    replacement_triplets.sort(key=lambda item: str(item[0].get("event_id", "")))
    if not replacement_triplets:
        raise AssemblyError(f"frozen treatment pool has no replacement events: {sorted(replacements, key=int)}")
    final_triplets = list(retained_prior_triplets)
    identities = {_event_identity(rollout, review) for rollout, review, _ in final_triplets}
    retained_replacement_triplets: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for rollout, review, receipt in replacement_triplets:
        identity = _event_identity(rollout, review)
        if identity in blocked:
            source_collision_count += 1
            continue
        if identity in identities:
            continue
        final_triplets.append((rollout, review, receipt))
        retained_replacement_triplets.append((rollout, review, receipt))
        identities.add(identity)

    if len(final_triplets) < ARM_EVENT_COUNT:
        raise AssemblyError(
            f"single-route sampled pool has only {len(final_triplets)} events after excluding "
            f"{source_collision_count} Source collisions; cannot reach {ARM_EVENT_COUNT} "
            "under the one-route-per-image constraint"
        )
    pools: dict[str, list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for triplet in final_triplets:
        image = _image_id(_mapping(triplet[0].get("image"), "treatment image").get("image_id"))
        pools[image].append(triplet)
    selected: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    selected_keys: set[tuple[str, str, str, str]] = set()
    expected_images = sorted((str(item) for item in image_ids), key=int)
    if any(image not in pools for image in expected_images):
        raise AssemblyError("replacement treatment pool dropped an image from the matched cohort")
    while len(selected) < ARM_EVENT_COUNT:
        progressed = False
        for image in expected_images:
            if len(selected) >= ARM_EVENT_COUNT:
                break
            available = [
                triplet
                for triplet in pools[image]
                if _event_identity(triplet[0], triplet[1]) not in selected_keys
            ]
            if not available:
                continue
            chosen = min(available, key=lambda triplet: str(triplet[0].get("event_id", "")))
            selected.append(chosen)
            selected_keys.add(_event_identity(chosen[0], chosen[1]))
            progressed = True
        if not progressed:
            break
    if len(selected) != ARM_EVENT_COUNT:
        raise AssemblyError(f"single-route rebuild selected {len(selected)} events")
    final_rollouts = [triplet[0] for triplet in selected]
    final_reviews = [triplet[1] for triplet in selected]
    final_receipts = [triplet[2] for triplet in selected]
    if len(selected_keys) != ARM_EVENT_COUNT:
        raise AssemblyError("single-route rebuild contains duplicate exact treatment identities")
    cohort = {
        _image_id(_mapping(item.get("image"), "treatment image").get("image_id"))
        for item in final_rollouts
    }
    expected = {str(item) for item in image_ids}
    if not cohort <= expected:
        raise AssemblyError(f"single-route rebuild contains out-of-cohort images: {sorted(cohort - expected, key=int)}")
    purity = _validate_single_route_purity(
        final_reviews,
        expected_event_count=ARM_EVENT_COUNT,
        expected_image_count=TARGET_IMAGE_COUNT,
        expected_image_ids=image_ids,
        rollouts=final_rollouts,
    )
    return final_rollouts, final_reviews, final_receipts, {
        "frozen_pool_event_count": len(pool),
        "retained_prior_event_count": len(retained_prior_triplets),
        "replacement_image_ids": sorted(replacements, key=int),
        "replacement_event_count": len(retained_replacement_triplets),
        "trimmed_event_count": len(final_triplets) - ARM_EVENT_COUNT,
        "cross_route_extra_event_count": 0,
        "synthetic_event_count": 0,
        "route_supplementation_event_count": 0,
        "source_collision_exclusion_count": source_collision_count,
        "source_collision_exclusion_identities": len(blocked),
        "duplicate_exact_event_count": 0,
        "single_route_purity": purity,
        "final_event_count": len(final_rollouts),
    }


def _build_multi_candidates(
    *,
    image_results: Mapping[str, Mapping[str, Any]],
    sampled_rows: Mapping[tuple[str, int], Mapping[str, Any]],
    reference_records: Mapping[str, Mapping[str, Any]],
    annotations: Mapping[str, Mapping[str, Any]],
    image_ids: Sequence[str],
    checkpoint_id: str,
    geometry_allowlist: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    route_receipts: list[dict[str, Any]] = []
    allowlist_by_key = {
        (
            str(item.get("image_id")),
            str(item.get("route_id")),
            int(item.get("row_index", -1)),
            str(item.get("owner_id", "")),
        ): float(item.get("iou", -1.0))
        for item in (geometry_allowlist or [])
    }
    for image in sorted((str(item) for item in image_ids), key=int):
        image_result = image_results.get(image)
        if image_result is None:
            raise AssemblyError(f"trajectory analysis lacks admitted image {image}")
        selection = select_complementary_routes(image_result)
        route_receipts.append(selection)
        ref = _require_mapping(reference_records.get(image), f"reference_records[{image}]")
        annotation = _require_mapping(annotations.get(image), f"annotations[{image}]")
        entities = _annotation_entities(annotation, source_path=Path("annotations.jsonl").resolve())
        for selected in selection["selected_routes"]:
            route_id = str(selected["route_id"])
            seed = int(selected["seed"])
            sampled = dict(_route_rollout_index(sampled_rows, image, seed))
            rows = _candidate_rows_for_route(
                image_result=image_result,
                route_id=route_id,
                cutoff=int(selected["last_marginal_owner_row_index"]),
                owners=image_result.get("owners", []),
                generated_ids=_token_ids(sampled.get("generated_token_ids"), f"sampled rollout {image}.generated_token_ids"),
                geometry_untrusted_allowlist=set(allowlist_by_key),
            )
            marginal = set(str(item) for item in selected["marginal_added_owner_ids"])
            for row in rows:
                allow_key = (image, route_id, int(row["generated_row_index"]), str(row["owner_id"]))
                if allow_key in allowlist_by_key:
                    actual_iou = float(row["geometry"]["intersection_over_union"])
                    if abs(actual_iou - allowlist_by_key[allow_key]) > 1e-9:
                        raise AssemblyError(
                            f"treatment geometry allowlist IoU mismatch for {allow_key}: "
                            f"{actual_iou} != {allowlist_by_key[allow_key]}"
                        )
                    row["geometry_untrusted_allowlisted"] = True
                row["marginal_owner_count"] = 1 if row["owner_id"] in marginal else 0
                row["route_added_owner_ids"] = sorted(marginal)
                row["marginal_route_added_owner_count"] = len(marginal)
                row["route_added_owner_count"] = len(selected["added_owner_ids"])
                row["unresolved_row_count"] = int(selected["unresolved_rows"])
                row["route_count"] = len(selection["selected_routes"])
                row["route_seed"] = seed
                row["image_id"] = image
                row["_event_inputs"] = {
                    "image_result": image_result,
                    "rollout": sampled,
                    "reference": ref,
                    "entities": entities,
                    "checkpoint_id": checkpoint_id,
                }
                candidates[image].append(row)
    return {key: list(value) for key, value in candidates.items()}, {
        "image_count": len(route_receipts),
        "route_receipts": route_receipts,
        "selected_route_count": sum(len(item["selected_routes"]) for item in route_receipts),
        "selected_added_owner_union_count": sum(int(item["selected_added_owner_count"]) for item in route_receipts),
        "geometry_untrusted_allowlist": [dict(item) for item in (geometry_allowlist or [])],
        "geometry_untrusted_allowlisted_candidate_count": sum(
            1 for values in candidates.values() for item in values if item.get("geometry_untrusted_allowlisted")
        ),
    }


def _build_source_candidates(
    *,
    image_results: Mapping[str, Mapping[str, Any]],
    greedy_rows: Mapping[tuple[str, int], Mapping[str, Any]],
    reference_records: Mapping[str, Mapping[str, Any]],
    annotations: Mapping[str, Mapping[str, Any]],
    image_ids: Sequence[str],
    manual_review: Mapping[str, Any] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    candidates: dict[str, list[dict[str, Any]]] = {}
    census: list[dict[str, Any]] = []
    review_by_image = _source_review_by_image(manual_review or {"decisions": []})
    for image in sorted((str(item) for item in image_ids), key=int):
        image_result = _require_mapping(image_results.get(image), f"trajectory_analysis.image_results[{image}]")
        greedy_id = _string(image_result.get("greedy_trajectory_id"), "greedy_trajectory_id")
        assignment = _route_assignment(image_result, greedy_id)
        ref = _require_mapping(reference_records.get(image), f"reference_records[{image}]")
        annotation = _require_mapping(annotations.get(image), f"annotations[{image}]")
        entities = _annotation_entities(annotation, source_path=Path("annotations.jsonl").resolve())
        owners = image_result.get("owners", [])
        greedy = dict(greedy_rows.get((image, int(_route_seed(_require_mapping(image_result.get("trajectory_evidence"), "trajectory_evidence").get(greedy_id), assignment))), {}))
        if not greedy:
            raise AssemblyError(f"greedy rollout missing admitted image {image}")
        source_rows = select_source_anchor_rows(
            assignment,
            owners,
            allow_geometry_unknown=True,
        )
        manual = review_by_image.get(image)
        if manual is not None and str(manual.get("decision")) == "exclude":
            raise AssemblyError(f"excluded image {image} was passed to Source candidate assembly")
        manual_rows: list[dict[str, Any]] = []
        if manual is not None and str(manual.get("decision")) == "admit":
            override_assignment = _manual_assignment_override(
                assignment,
                manual,
                rollout=greedy,
                image_id=image,
            )
            manual_index = int(manual["row_index"])
            override_rows = _candidate_rows_for_route(
                image_result=image_result,
                route_id=greedy_id,
                cutoff=manual_index,
                owners=owners,
                generated_ids=_token_ids(greedy.get("generated_token_ids"), f"greedy rollout {image}.generated_token_ids"),
                assignment_override=override_assignment,
            )
            manual_row = next(
                (candidate for candidate in override_rows if candidate["generated_row_index"] == manual_index),
                None,
            )
            if manual_row is None:
                raise AssemblyError(f"manual Source row {image}:{manual_index} disappeared during exact slicing")
            manual_row["manual_review"] = {
                "decision": "admit",
                "reviewer": str(manual.get("reviewer", "")),
                "review_source": str(manual.get("review_source", "")),
                "comment": str(manual.get("comment", "")),
            }
            manual_row["manual_review_owner_id"] = str(manual["owner_id"])
            manual_row.update(
                {
                    "image_id": image,
                    "marginal_owner_count": 0,
                    "marginal_route_added_owner_count": 0,
                    "route_added_owner_count": 0,
                    "unresolved_row_count": int(_route_row_counts(assignment).get("unresolved", 0)),
                    "route_count": 1,
                    "route_seed": int(greedy.get("seed", 0)),
                    "_event_inputs": {
                        "image_result": image_result,
                        "rollout": greedy,
                        "reference": ref,
                        "entities": entities,
                        "checkpoint_id": str(greedy.get("_checkpoint_id", "")),
                    },
                }
            )
            manual_rows.append(manual_row)
        rows: list[dict[str, Any]] = []
        generated_ids = _token_ids(greedy.get("generated_token_ids"), f"greedy rollout {image}.generated_token_ids")
        for item in source_rows:
            row = _candidate_rows_for_route(
                image_result=image_result,
                route_id=greedy_id,
                cutoff=int(item["generated_row_index"]),
                owners=owners,
                generated_ids=generated_ids,
                allow_geometry_unknown=True,
            )
            row = next((candidate for candidate in row if candidate["generated_row_index"] == int(item["generated_row_index"])), None)
            if row is None:
                raise AssemblyError(f"source row {image}:{item['generated_row_index']} disappeared during exact slicing")
            row.update({
                "image_id": image,
                "marginal_owner_count": 0,
                "marginal_route_added_owner_count": 0,
                "route_added_owner_count": 0,
                "unresolved_row_count": int(_route_row_counts(assignment).get("unresolved", 0)),
                "route_count": 1,
                "route_seed": int(greedy.get("seed", 0)),
                "_event_inputs": {
                    "image_result": image_result,
                    "rollout": greedy,
                    "reference": ref,
                    "entities": entities,
                    "checkpoint_id": str(greedy.get("_checkpoint_id", "")),
                },
            })
            rows.append(row)
        existing_owners = {str(row["owner_id"]) for row in rows}
        for manual_row in manual_rows:
            if str(manual_row["owner_id"]) not in existing_owners:
                rows.append(manual_row)
                existing_owners.add(str(manual_row["owner_id"]))
        rows.sort(key=lambda item: int(item["generated_row_index"]))
        for anchor_order, row in enumerate(rows, start=1):
            row["trusted_source_anchor_order"] = anchor_order
        candidates[image] = rows
        census.append({
            "image_id": image,
            "trusted_source_anchor_count": len(rows),
            "greedy_route_id": greedy_id,
            "manual_review_decision": None if manual is None else str(manual.get("decision")),
            "manual_review_owner_id": None if manual is None else manual.get("owner_id"),
            "manual_review_row_index": None if manual is None else manual.get("row_index"),
        })
    return candidates, {"image_count": len(census), "rows_by_image": census, "trusted_anchor_count": sum(int(item["trusted_source_anchor_count"]) for item in census)}


def _arm_from_events(
    *,
    treatment_rollouts: Sequence[Mapping[str, Any]],
    treatment_reviews: Sequence[Mapping[str, Any]],
    treatment_receipts: Sequence[Mapping[str, Any]],
    source_rollouts: Sequence[Mapping[str, Any]],
    source_reviews: Sequence[Mapping[str, Any]],
    source_receipts: Sequence[Mapping[str, Any]],
    family: str,
    image_ids: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if len(treatment_rollouts) != ARM_EVENT_COUNT or len(source_rollouts) != ARM_EVENT_COUNT:
        raise AssemblyError(
            f"each arm requires exactly {ARM_EVENT_COUNT} treatment and {ARM_EVENT_COUNT} Source events"
        )
    if len(treatment_reviews) != len(treatment_rollouts) or len(source_reviews) != len(source_rollouts):
        raise AssemblyError("rollout/review family counts differ")
    identity_receipt = _validate_route_event_identity_uniqueness(
        treatment_rollouts=treatment_rollouts,
        treatment_reviews=treatment_reviews,
        source_rollouts=source_rollouts,
        source_reviews=source_reviews,
    )
    all_rollouts = [dict(item) for item in treatment_rollouts] + [dict(item) for item in source_rollouts]
    all_reviews = [copy.deepcopy(dict(item)) for item in treatment_reviews] + [copy.deepcopy(dict(item)) for item in source_reviews]
    all_receipts = [dict(item) for item in treatment_receipts] + [dict(item) for item in source_receipts]
    family_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"treatment": 0, "source_preservation": 0})
    for rollout, review in zip(all_rollouts, all_reviews, strict=True):
        image = _image_id(_mapping(rollout.get("image"), "event image").get("image_id"))
        event_family = str(_mapping(review.get("review_provenance"), "review_provenance").get("event_family"))
        if event_family == "source_preservation":
            family_counts[image]["source_preservation"] += 1
        elif event_family in {"single_route_treatment", "multi_route_treatment"}:
            family_counts[image]["treatment"] += 1
        else:
            raise AssemblyError(f"unknown event family {event_family!r}")
    if set(family_counts) != {str(item) for item in image_ids}:
        missing = sorted({str(item) for item in image_ids} - set(family_counts), key=int)
        extra = sorted(set(family_counts) - {str(item) for item in image_ids}, key=int)
        raise AssemblyError(f"arm image cohort mismatch; missing={missing}, extra={extra}")
    weights = image_family_event_weights(family_counts, event_count=2 * ARM_EVENT_COUNT)
    updated_rollouts: list[dict[str, Any]] = []
    updated_reviews: list[dict[str, Any]] = []
    updated_receipts: list[dict[str, Any]] = []
    for rollout, review, receipt in zip(all_rollouts, all_reviews, all_receipts, strict=True):
        image = str(_mapping(rollout.get("image"), "event image").get("image_id"))
        event_family = str(_mapping(review.get("review_provenance"), "review_provenance").get("event_family"))
        weight = weights[(image, "source_preservation" if event_family == "source_preservation" else "treatment")]
        updated_rollout, updated_review, updated_receipt = _set_event_weight(rollout, review, receipt, weight)
        updated_rollouts.append(updated_rollout)
        updated_reviews.append(updated_review)
        updated_receipts.append(updated_receipt)
    total = sum(float(item["image_balanced_event_weight"]) for item in updated_receipts)
    if abs(total / len(updated_receipts) - 1.0) > 1e-9:
        raise AssemblyError("combined event weights do not have global mean one")
    return updated_rollouts, updated_reviews, updated_receipts, {
        "arm": family,
        "event_count": len(updated_rollouts),
        "treatment_event_count": len(treatment_rollouts),
        "source_preservation_event_count": len(source_rollouts),
        "image_count": len(image_ids),
        "image_family_event_counts": {image: family_counts[image] for image in sorted(family_counts, key=int)},
        "weights": {f"{image}:{fam}": value for (image, fam), value in sorted(weights.items())},
        "mean_event_weight": total / len(updated_receipts),
        "identity_validation": identity_receipt,
    }


def _source_artifacts(paths: Sequence[tuple[str, Path]]) -> list[dict[str, str]]:
    return [{"artifact_id": name, "sha256": sha256_file(path.resolve(strict=True))} for name, path in sorted(paths, key=lambda item: item[0])]


def _write_arm(
    *,
    root: Path,
    rollouts: Sequence[Mapping[str, Any]],
    reviews: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
    census: Mapping[str, Any],
    arm_receipt: Mapping[str, Any],
    source_checkpoint: Any,
    prompt_identity_sha256: str,
    source_artifacts: Sequence[Mapping[str, Any]],
    verified_rollout_checkpoint_identities: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if root.exists():
        raise AssemblyError(f"arm output already exists: {root}")
    root.mkdir(parents=True)
    pre = root / "pre-state-bank"
    _write_jsonl(pre / "rollout_rows.jsonl", rollouts)
    _write_jsonl(pre / "review_rows.jsonl", reviews)
    _write_json(pre / "selection-census.json", census)
    manifest = assemble_state_bank(
        output_dir=root / "state-bank",
        rollout_rows=rollouts,
        review_rows=reviews,
        source_checkpoint=source_checkpoint,
        prompt_identity_sha256=prompt_identity_sha256,
        source_artifacts=source_artifacts,
    )
    loaded = load_state_bank(
        root / "state-bank" / "manifest.json",
        expected_source_checkpoint=source_checkpoint,
        expected_prompt_identity_sha256=prompt_identity_sha256,
    )
    receipt = {
        **dict(arm_receipt),
        "schema_version": SCHEMA_VERSION,
        "status": "assembled",
        "state_bank_manifest": manifest.to_artifact_dict(),
        "state_bank_manifest_path": str((root / "state-bank" / "manifest.json").resolve()),
        "state_bank_validation_receipt": loaded.validation_receipt.to_artifact_dict(),
        "verified_rollout_checkpoint_identities": [
            dict(item) for item in (verified_rollout_checkpoint_identities or [])
        ],
        "counts": {
            "rollout_rows": len(rollouts),
            "review_rows": len(reviews),
            "event_receipts": len(receipts),
        },
        "event_receipts": [dict(item) for item in receipts],
    }
    _write_json(root / "assembly-receipt.json", receipt)
    return receipt


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-analysis", type=Path, required=True)
    parser.add_argument("--greedy-rollout", type=Path, required=True)
    parser.add_argument("--sampled-rollout", type=Path, action="append", required=True, dest="sampled_rollouts")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--prior-state-bank-manifest", type=Path, required=True)
    parser.add_argument("--prior-assembly-receipt", type=Path, required=True)
    parser.add_argument(
        "--source-anchor-review",
        type=Path,
        required=True,
        help="review-only sidecar admitting later greedy rows or excluding unresolved images",
    )
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"output path already exists and will not be overwritten: {output}")
    analysis_path = args.trajectory_analysis.expanduser().resolve(strict=True)
    greedy_path = args.greedy_rollout.expanduser().resolve(strict=True)
    sampled_paths = [Path(item).expanduser().resolve(strict=True) for item in args.sampled_rollouts]
    if len(sampled_paths) != 8:
        raise SystemExit("exactly eight sampled-shard artifacts are required")
    annotations_path = args.annotations.expanduser().resolve(strict=True)
    prior_manifest_path = args.prior_state_bank_manifest.expanduser().resolve(strict=True)
    prior_receipt_path = args.prior_assembly_receipt.expanduser().resolve(strict=True)
    source_anchor_review_path = args.source_anchor_review.expanduser().resolve(strict=True)
    reference_manifest_path = args.reference_manifest.expanduser().resolve(strict=True)
    analysis = _read_json(analysis_path)
    if not isinstance(analysis, Mapping) or analysis.get("schema_version") != ROUTE_ANALYSIS_SCHEMA_VERSION:
        raise SystemExit("unsupported trajectory analysis schema")
    greedy_rows, greedy_configs = _load_rollout_rows([greedy_path], sampled_only=False)
    sampled_rows, sampled_configs = _load_rollout_rows(sampled_paths, sampled_only=True)
    for key, row in sampled_rows.items():
        config = sampled_configs.get(str(Path(str(row.get("_source_path"))).resolve()), {})
        row["_temperature"] = config.get("temperature", 0.4)
        row["_top_p"] = config.get("top_p", 0.95)
        row["_repetition_penalty"] = config.get("repetition_penalty", 1.0)
    greedy_config = next(iter(greedy_configs.values()), {})
    for row in greedy_rows.values():
        row["_temperature"] = greedy_config.get("temperature", 0.0)
        row["_top_p"] = greedy_config.get("top_p", 1.0)
        row["_repetition_penalty"] = greedy_config.get("repetition_penalty", 1.0)
    annotation_rows = load_jsonl(annotations_path)
    annotations = {_image_id(item.get("image_id")): item for item in annotation_rows}
    if len(annotations) != len(annotation_rows):
        raise SystemExit("annotations contain duplicate image IDs")
    reference_binding = load_state_bank_manifest_binding(reference_manifest_path)
    verified_checkpoint, verified_rollout_identities = verify_rollout_checkpoint_identities(
        [greedy_path, *sorted(sampled_paths)],
        reference_checkpoint=reference_binding.source_checkpoint,
    )
    identity_by_path = {
        str(item["artifact_path"]): item for item in verified_rollout_identities
    }
    for row in (*greedy_rows.values(), *sampled_rows.values()):
        source_path = str(Path(str(row.get("_source_path"))).resolve())
        identity_record = identity_by_path.get(source_path)
        if identity_record is None:
            raise AssemblyError(f"rollout row source path was not checkpoint-verified: {source_path}")
        row["_checkpoint_id"] = str(identity_record["checkpoint_id"])
        row["_checkpoint_identity"] = dict(identity_record["checkpoint_identity"])
    reference_rows = load_jsonl(reference_manifest_path.parent / "records.jsonl")
    reference_records = {_image_id(item.get("image", {}).get("image_id")): item for item in reference_rows}
    if len(reference_records) != len(reference_rows):
        raise SystemExit("reference records contain duplicate image IDs")
    source_anchor_review = _load_source_anchor_review(source_anchor_review_path)
    source_gap_review_packet_path: Path | None = None
    if source_anchor_review.get("review_packet"):
        source_gap_review_packet_path = (
            source_anchor_review_path.parent
            / str(source_anchor_review["review_packet"])
        ).resolve(strict=True)
    geometry_review_packet_path: Path | None = None
    if source_anchor_review.get("treatment_geometry_review_packet"):
        geometry_review_packet_path = (
            source_anchor_review_path.parent
            / str(source_anchor_review["treatment_geometry_review_packet"])
        ).resolve(strict=True)
    prior_rollouts, prior_reviews, prior_receipt, prior_image_ids = _prior_rows(
        prior_manifest_path,
        prior_receipt_path,
    )
    image_ids, excluded_image_ids = _admitted_source_image_ids(
        sorted(prior_image_ids, key=int),
        source_anchor_review,
    )
    prior_rollouts, prior_reviews, prior_receipt, _ = _prior_rows(
        prior_manifest_path,
        prior_receipt_path,
        admitted_image_ids=image_ids,
    )
    image_results = {_image_id(item.get("image_id")): item for item in analysis.get("image_results", [])}
    if len(image_results) != len(analysis.get("image_results", [])):
        raise SystemExit("trajectory analysis contains duplicate image IDs")
    source_candidates, source_census = _build_source_candidates(
        image_results=image_results,
        greedy_rows=greedy_rows,
        reference_records=reference_records,
        annotations=annotations,
        image_ids=sorted(image_ids, key=int),
        manual_review=source_anchor_review,
    )
    _validate_source_anchor_cohort(sorted(image_ids, key=int), source_candidates)
    source_selected, source_selection_receipt = select_image_diverse_events(
        source_candidates,
        image_ids=sorted(image_ids, key=int),
        event_count=ARM_EVENT_COUNT,
    )
    source_identities = {_candidate_identity(item) for item in source_selected}
    source_selection_receipt["semantic_identity_count"] = len(source_identities)
    source_selection_receipt["semantic_identity_unique"] = len(source_identities) == len(source_selected)
    if len(source_identities) != len(source_selected):
        raise AssemblyError("Source preservation selection contains duplicate semantic identities")
    source_rollouts: list[dict[str, Any]] = []
    source_reviews: list[dict[str, Any]] = []
    source_receipts: list[dict[str, Any]] = []
    for item in source_selected:
        inputs = item.pop("_event_inputs")
        route_id = str(item["route_id"])
        event = _make_route_event(
            image_result=inputs["image_result"], route_id=route_id, seed=int(item["route_seed"]), row=item,
            rollout=inputs["rollout"], reference=inputs["reference"], entities=inputs["entities"], checkpoint_id=sha256_json(verified_checkpoint.to_artifact_dict()),
            family="source_preservation", event_weight=1.0,
        )
        source_rollouts.append(event[0])
        source_reviews.append(event[1])
        source_receipts.append(event[2])
    multi_candidates_before_source_exclusion, multi_census = _build_multi_candidates(
        image_results=image_results, sampled_rows=sampled_rows, reference_records=reference_records, annotations=annotations,
        image_ids=sorted(image_ids, key=int), checkpoint_id=sha256_json(verified_checkpoint.to_artifact_dict()),
        geometry_allowlist=source_anchor_review.get("treatment_geometry_untrusted_allowlist", []),
    )
    multi_candidates, multi_collision_receipt = exclude_sampled_source_collisions(
        multi_candidates_before_source_exclusion,
        source_identities,
    )
    try:
        multi_selected, multi_selection_receipt = select_image_diverse_events(
            multi_candidates,
            image_ids=sorted(image_ids, key=int),
            event_count=ARM_EVENT_COUNT,
        )
    except AssemblyError as exc:
        raise AssemblyError(
            "multi-route sampled pool cannot reach exactly "
            f"{ARM_EVENT_COUNT} events after Source-collision exclusion and "
            f"the at-most-{MAX_ROUTES_PER_IMAGE}-routes-per-image constraint: {exc}"
        ) from exc
    multi_selection_receipt["source_collision_exclusion"] = multi_collision_receipt
    multi_geometry_untrusted_selected = [
        {
            "image_id": str(item["image_id"]),
            "route_id": str(item["route_id"]),
            "row_index": int(item["generated_row_index"]),
            "owner_id": str(item["owner_id"]),
            "iou": float(item["geometry"]["intersection_over_union"]),
            "geometry_review_status": "unknown",
            "coordinate_gradient": "masked",
        }
        for item in multi_selected
        if item.get("geometry_untrusted_allowlisted")
    ]
    expected_geometry_allowlist = {
        (
            str(item.get("image_id")),
            str(item.get("route_id")),
            int(item.get("row_index", -1)),
            str(item.get("owner_id", "")),
        )
        for item in source_anchor_review.get("treatment_geometry_untrusted_allowlist", [])
    }
    selected_geometry_allowlist = {
        (item["image_id"], item["route_id"], item["row_index"], item["owner_id"])
        for item in multi_geometry_untrusted_selected
    }
    if selected_geometry_allowlist != expected_geometry_allowlist:
        raise AssemblyError(
            "treatment geometry allowlist was not selected exactly: "
            f"expected={sorted(expected_geometry_allowlist)}, selected={sorted(selected_geometry_allowlist)}"
        )
    multi_selection_receipt["geometry_untrusted_allowlisted_selected"] = multi_geometry_untrusted_selected
    multi_selection_receipt["geometry_untrusted_allowlisted_selected_count"] = len(multi_geometry_untrusted_selected)
    multi_rollouts: list[dict[str, Any]] = []
    multi_reviews: list[dict[str, Any]] = []
    multi_receipts: list[dict[str, Any]] = []
    for item in multi_selected:
        inputs = item.pop("_event_inputs")
        event = _make_route_event(
            image_result=inputs["image_result"], route_id=str(item["route_id"]), seed=int(item["route_seed"]), row=item,
            rollout=inputs["rollout"], reference=inputs["reference"], entities=inputs["entities"], checkpoint_id=sha256_json(verified_checkpoint.to_artifact_dict()),
            family="multi_route_treatment", event_weight=1.0,
        )
        multi_rollouts.append(event[0])
        multi_reviews.append(event[1])
        multi_receipts.append(event[2])
    # Reconstruct the single-route treatment pool from admitted historical
    # rows and exact no-batch-fit replacement rows, then trim deterministically
    # to 496 without cross-route supplementation.
    frozen_pool_rollouts, frozen_pool_reviews, frozen_pool_receipt = build_pre_state_bank(
        trajectory_analysis=analysis,
        greedy_rows=greedy_rows,
        sampled_rows=sampled_rows,
        annotations=annotations,
        reference_records=reference_records,
        checkpoint_id=sha256_json(verified_checkpoint.to_artifact_dict()),
        required_event_multiple=None,
        annotation_source_path=annotations_path,
    )
    frozen_pool_receipts = frozen_pool_receipt.get("event_receipts")
    if not isinstance(frozen_pool_receipts, list):
        raise AssemblyError("frozen treatment pool receipt lacks event_receipts")
    single_rollouts, single_reviews, single_receipts, single_rebuild = _rebuild_single_route_treatment_events(
        prior_rollouts=prior_rollouts,
        prior_reviews=prior_reviews,
        prior_receipts=[dict(item) for item in prior_receipt["event_receipts"]],
        frozen_pool_rollouts=frozen_pool_rollouts,
        frozen_pool_reviews=frozen_pool_reviews,
        frozen_pool_receipts=frozen_pool_receipts,
        replacement_image_ids=source_anchor_review.get("replacement_image_ids", []),
        extra_candidates_by_image=multi_candidates,
        image_ids=sorted(image_ids, key=int),
        checkpoint_id=sha256_json(verified_checkpoint.to_artifact_dict()),
        excluded_identities=source_identities,
    )
    single_arm = _arm_from_events(
        treatment_rollouts=single_rollouts, treatment_reviews=single_reviews, treatment_receipts=single_receipts,
        source_rollouts=source_rollouts, source_reviews=source_reviews, source_receipts=source_receipts,
        family="single_route_plus_source_preservation", image_ids=sorted(image_ids, key=int),
    )
    multi_arm = _arm_from_events(
        treatment_rollouts=multi_rollouts, treatment_reviews=multi_reviews, treatment_receipts=multi_receipts,
        source_rollouts=source_rollouts, source_reviews=source_reviews, source_receipts=source_receipts,
        family="multi_route_plus_source_preservation", image_ids=sorted(image_ids, key=int),
    )
    single_arm[3]["source_collision_exclusion"] = {
        "source_identity_count": len(source_identities),
        "excluded_event_count": int(single_rebuild.get("source_collision_exclusion_count", 0)),
    }
    multi_arm[3]["source_collision_exclusion"] = multi_collision_receipt
    source_paths = [
        ("trajectory-analysis", analysis_path), ("greedy-rollout", greedy_path), ("annotations", annotations_path),
        ("prior-state-bank-manifest", prior_manifest_path), ("prior-assembly-receipt", prior_receipt_path),
        ("source-anchor-review", source_anchor_review_path),
        ("reference-state-bank-manifest", reference_manifest_path), ("reference-state-bank-records", reference_manifest_path.parent / "records.jsonl"),
    ] + [(f"sampled-rollout-{index:02d}", path) for index, path in enumerate(sorted(sampled_paths))]
    if geometry_review_packet_path is not None:
        source_paths.append(("treatment-geometry-review", geometry_review_packet_path))
    if source_gap_review_packet_path is not None:
        source_paths.append(("source-anchor-gap-review", source_gap_review_packet_path))
    artifacts = _source_artifacts(source_paths)
    root_receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "assembled",
        "source_checkpoint_id": reference_binding.source_checkpoint_id,
        "verified_source_checkpoint": verified_checkpoint.to_artifact_dict(),
        "verified_rollout_checkpoint_identities": verified_rollout_identities,
        "prompt_identity_sha256": reference_binding.prompt_identity_sha256,
        "image_ids": sorted(image_ids, key=int),
        "image_count": len(image_ids),
        "prior_image_ids": sorted(prior_image_ids, key=int),
        "excluded_image_ids": excluded_image_ids,
        "source_anchor_review": {
            "artifact_path": str(source_anchor_review_path),
            "sha256": sha256_file(source_anchor_review_path),
            "schema_version": source_anchor_review["schema_version"],
            "status": source_anchor_review["status"],
            "decisions": source_anchor_review["decisions"],
            "replacement_image_ids": source_anchor_review.get("replacement_image_ids", []),
            "treatment_geometry_untrusted_allowlist": source_anchor_review.get(
                "treatment_geometry_untrusted_allowlist", []
            ),
            "treatment_geometry_review_packet": (
                None if geometry_review_packet_path is None else str(geometry_review_packet_path)
            ),
            "review_packet": (
                None if source_gap_review_packet_path is None else str(source_gap_review_packet_path)
            ),
        },
        "prior_source_cohort_filtering": prior_receipt.get("source_cohort_filtering"),
        "single_route_treatment_rebuild": single_rebuild,
        "source_anchor_selection": source_selection_receipt,
        "source_anchor_census": source_census,
        "multi_route_census": multi_census,
        "multi_route_selection": multi_selection_receipt,
        "source_collision_exclusion": {
            "source_identity_count": len(source_identities),
            "single_route": {
                "excluded_event_count": int(single_rebuild.get("source_collision_exclusion_count", 0)),
            },
            "multi_route": multi_collision_receipt,
        },
        "multi_route_geometry_untrusted_treatment_rows": multi_geometry_untrusted_selected,
        "source_artifacts": artifacts,
    }
    output.mkdir(parents=True)
    _write_json(output / "input-census.json", root_receipt)
    single_root = output / "single-route-plus-source-preservation"
    multi_root = output / "multi-route-plus-source-preservation"
    _write_arm(root=single_root, rollouts=single_arm[0], reviews=single_arm[1], receipts=single_arm[2], census={"source": source_census, "selection": source_selection_receipt, "arm": single_arm[3]}, arm_receipt=single_arm[3], source_checkpoint=reference_binding.source_checkpoint, prompt_identity_sha256=reference_binding.prompt_identity_sha256, source_artifacts=artifacts, verified_rollout_checkpoint_identities=verified_rollout_identities)
    _write_arm(root=multi_root, rollouts=multi_arm[0], reviews=multi_arm[1], receipts=multi_arm[2], census={"source": source_census, "multi_route": multi_census, "selection": multi_selection_receipt, "arm": multi_arm[3]}, arm_receipt=multi_arm[3], source_checkpoint=reference_binding.source_checkpoint, prompt_identity_sha256=reference_binding.prompt_identity_sha256, source_artifacts=artifacts, verified_rollout_checkpoint_identities=verified_rollout_identities)
    _write_json(output / "assembly-receipt.json", {**root_receipt, "arms": {"single_route_plus_source_preservation": str((single_root / "assembly-receipt.json").resolve()), "multi_route_plus_source_preservation": str((multi_root / "assembly-receipt.json").resolve())}})
    print(json.dumps({"status": "assembled", "output_dir": str(output), "event_count_per_arm": 2 * ARM_EVENT_COUNT, "image_count": len(image_ids)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
