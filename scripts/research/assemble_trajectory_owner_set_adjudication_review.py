#!/usr/bin/env python3
"""Assemble the blinded owner-ledger review and exact frozen-panel replay.

This experiment-local module intentionally does not extend the stable spatial-
scope review framework.  Its pure functions expose every pre-seed contract,
role-seal, adjudication, ledger, replay, and outcome boundary used by the
trajectory owner-set salvage gate.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import sys
from typing import Any
import uuid

from PIL import Image

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import scripts.research.analyze_trajectory_owner_set_admission_census as census  # noqa: E402
from scripts.research.analyze_individual_trajectory_union_support import (  # noqa: E402
    load_generation7_annotations,
    match_prefix,
)
from scripts.research.assemble_constant_dose_breadth_state_banks import (  # noqa: E402
    AssemblyError,
    load_v2_b16_panel_adapter,
)
from src.eval.detection_categories import (  # noqa: E402
    COCO_80_CATEGORY_NAMESPACE_SHA256,
    COCO_80_EVALUATOR_LOCAL_CATEGORY_ID_BY_NAME,
    COCO_80_OFFICIAL_CATEGORY_ID_BY_NAME,
    normalize_coco_category_name,
)


PACKET_ID = "trajectory-owner-set-image-only-review-v1"
ONTOLOGY_ID = "coco-80-review-ontology-v1"
SELECTION_SCHEMA_VERSION = "trajectory_owner_set_adjudication_selection.v1"
ENTROPY_CLAIM_SCHEMA_VERSION = (
    "trajectory_owner_set_adjudication_selection.v1.entropy_journal.claim.v2"
)
ENTROPY_TERMINAL_SCHEMA_VERSION = (
    "trajectory_owner_set_adjudication_selection.v1.entropy_journal.terminal.v2"
)
FROZEN_SOURCE_MANIFEST_SCHEMA_VERSION = (
    "trajectory_owner_set_adjudication_selection.v1.frozen_sources.v2"
)
SELECTOR_MEMBER_MANIFEST_SCHEMA_VERSION = (
    "trajectory_owner_set_adjudication_selection.v1.possible_pool_members.v2"
)
SELECTION_RECEIPT_SCHEMA_VERSION = (
    "trajectory_owner_set_adjudication_selection.v1.receipt.v2"
)
MEMBER_MANIFEST_SCHEMA_VERSION = "trajectory_owner_set_review.preentropy-member.v1"
MEMBER_MANIFEST_RECEIPT_SCHEMA_VERSION = (
    "trajectory_owner_set_review.preentropy-member-receipt.v1"
)
CONTRACT_SEAL_SCHEMA_VERSION = "trajectory_owner_set_review.contract-seal.v1"
REVIEW_QUEUE_SCHEMA_VERSION = "trajectory_owner_set_review.queue.v1"
REVIEWER_SCHEMA_BY_ROLE = {
    "reviewer-one": "trajectory_owner_set_review.reviewer-one.v1",
    "reviewer-two": "trajectory_owner_set_review.reviewer-two.v1",
}
ROLE_SEAL_SCHEMA_VERSION = "trajectory_owner_set_review.role-seal.v1"
OFFICIAL_OWNER_SCHEMA_VERSION = "trajectory_owner_set_review.owner-ledger.v1"
ADJUDICATION_QUEUE_SCHEMA_VERSION = "trajectory_owner_set_review.adjudication-queue.v1"
ADJUDICATION_DECISION_SCHEMA_VERSION = (
    "trajectory_owner_set_review.adjudication-decision.v1"
)
UNCERTAINTY_LEDGER_SCHEMA_VERSION = "trajectory_owner_set_review.uncertainty-ledger.v1"
OWNER_LEDGER_SEAL_SCHEMA_VERSION = "trajectory_owner_set_review.owner-ledger-seal.v1"
REPLAY_RECORD_SCHEMA_VERSION = "trajectory_owner_set_review.global-replay.v1"
OUTCOME_SCHEMA_VERSION = "trajectory_owner_set_review.outcome.v1"
SEQUENTIAL_DECISION_SCHEMA_VERSION = (
    "trajectory_owner_set_review.sequential-decision.v1"
)
FINAL_LOOK_RECEIPT_SCHEMA_VERSION = "trajectory_owner_set_review.look-receipt.v1"
QUEUE_MANIFEST_SCHEMA_VERSION = "trajectory_owner_set_review.queue-manifest.v2"
REPLAY_RECEIPT_SCHEMA_VERSION = "trajectory_owner_set_review.global-replay-receipt.v2"
SOURCE_SNAPSHOT_SCHEMA_VERSION = "trajectory_owner_set_review.source-snapshot.v1"
STAGE_ZERO_SCHEMA_VERSION = (
    "trajectory_owner_set_adjudication_salvage_gate.stage_zero.v1"
)
STAGE_ZERO_AUDIT_SCHEMA_VERSION = (
    "trajectory_owner_set_adjudication_salvage_gate.stage_zero_audit.v2"
)
STAGE_ZERO_POPULATION_SCOPE = "train_only_censored_nonpassing_U"
STAGE_ZERO_AUDIT_REVIEW_SCOPE = "full_witness_and_certificate_replay"
SELECTION_POPULATION_SCOPE = "train_only"
FROZEN_SELECTOR_PATH = (
    REPOSITORY_ROOT
    / "scripts/research/select_trajectory_owner_set_adjudication_review_sample.py"
)
FROZEN_SELECTOR_SHA256 = (
    "5473d370a06e74304d27e8fa34b4ff0a2268be2946ffe0f28076efd40a1d5d1d"
)
REVIEWER_ROLES = ("reviewer-one", "reviewer-two")
LOOK_IDS = ("look_one", "look_two_additional")
FINAL_LOOK_ARTIFACT_NAMES = frozenset(
    {
        "selection.json",
        "entropy-journal.json",
        "member-manifest.jsonl",
        "member-manifest-receipt.json",
        "review-queue.jsonl",
        "review-queue-manifest.json",
        "official-owner-ledger.jsonl",
        "reviewer-one.jsonl",
        "reviewer-one-seal.json",
        "reviewer-two.jsonl",
        "reviewer-two-seal.json",
        "adjudication-queue.jsonl",
        "adjudication.jsonl",
        "owner-ledger.jsonl",
        "uncertainty-ledger.jsonl",
        "owner-ledger-seal.json",
        "replay-records.jsonl",
        "replay-receipt.json",
        "outcomes.jsonl",
        "sequential-decision.json",
    }
)
EXPECTED_ROUTE_IDS = ("source-b16", *(f"sample-{index:02d}" for index in range(16)))
EXPECTED_NAMED_FROZEN_SOURCE_LABELS = (
    "b16_drop_chronology_helper",
    "b16_drop_chronology_impact_review",
    "census_analyzer",
    "detection_categories",
    "fingerprint",
    "geometry_semantics",
    "global_matcher",
    "inference_backend",
    "ontology_state_artifact",
    "outcome_classifier_contract",
    "panel_adapter",
    "review_adjudication_replay_implementation",
    "review_implementation_approval_receipt",
    "reviewer_instruction_packet",
)
AUTO_FROZEN_SOURCE_LABELS = (
    "selector",
    "unit_contract",
    "contract_review",
    "stage_zero_producer",
)
ENTROPY_BYTE_COUNT = 64
SELECTION_COUNT = 32
LOOK_ONE_COUNT = 16
NULL_SUCCESS_COUNT = 248
PER_LOOK_ALPHA = Fraction(1, 40)
FROZEN_CENSUS_PATH = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-23-trajectory-owner-set-admission-census/production-v1/"
    "image-census.jsonl"
)
FROZEN_CENSUS_SHA256 = (
    "c375e09df621714da82118a2cdf3d80205fd7429424b1a853f38ec60e68e1805"
)
FROZEN_CANDIDATE_POOL_SHA256 = (
    "133afcf6659b78d71893e9f79e568dca676515a05c7fe23f3c237de01350b7e2"
)
OUTCOMES = (
    "actual_admission",
    "potential_admission_unresolved",
    "definitive_non_admission",
)
UNCERTAINTY_AXES = (
    "owner_universe",
    "owner_assignment",
    "owner_identity",
    "first_owner",
    "trusted_geometry",
    "safety_counts",
    "outcome_class_construction",
    "semantic_edge_construction",
    "frontier_status",
    "admission",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVIEW_REASON_BY_STATE = {
    "accepted": {"none", "occluded_but_boxable", "truncated_but_boxable"},
    "ambiguous": {"category_not_unique", "instance_not_separable"},
    "partial": {"boundary_not_reproducible", "instance_not_separable"},
    "crowd": {"instance_not_separable"},
    "out-of-scope": {"non_coco80"},
}
_STATE_ORDER = {
    "accepted": 0,
    "ambiguous": 1,
    "partial": 2,
    "crowd": 3,
    "out-of-scope": 4,
}
_ADJUDICATION_REASON_BY_STATE = {
    "official_duplicate": {"official_duplicate"},
    "accepted_missing_owner": {
        "none",
        "reviewer_agreement",
        "adjudicator_override",
    },
    "ambiguous": {
        "category_not_unique",
        "instance_not_separable",
        "reviewer_disagreement",
    },
    "partial": {
        "boundary_not_reproducible",
        "instance_not_separable",
        "reviewer_disagreement",
    },
    "crowd": {"instance_not_separable", "reviewer_disagreement"},
    "out_of_scope": {"non_coco80"},
    "unresolved_disagreement": {"reviewer_disagreement"},
}
_OWNER_UNIVERSE_REASONS = {
    "reviewer_disagreement",
    "ambiguous_visible_evidence",
    "partial_visible_evidence",
    "crowd_unindividuated",
    "reviewer_coverage_disagreement",
    "adjudicator_unresolved",
    "ledger_completeness_not_established",
}
_FORBIDDEN_REVIEW_EVIDENCE_FIELDS = {
    "admission",
    "candidate_id",
    "category_capacity_witness",
    "census_label",
    "class_id",
    "cutoff",
    "decode_mode",
    "edge",
    "frontier",
    "generated_token_ids",
    "look_id",
    "look_two_authorized",
    "model_prediction",
    "official_annotation",
    "official_box",
    "official_count",
    "owner_id",
    "owner_set",
    "parser",
    "route_id",
    "sample_cutoff",
    "sample_order",
    "sample_position",
    "seed",
    "sequential_decision",
    "prior_look_decision",
    "prior_look_outcome",
    "cumulative_success_count",
    "token_hash",
}

_QUEUE_FIELDS = frozenset(
    {
        "schema_version",
        "review_identifier",
        "reviewer_role_identifier",
        "image_id",
        "image_path",
        "image_sha256",
        "source_image_width",
        "source_image_height",
        "packet_id",
        "packet_path",
        "packet_sha256",
        "ontology_id",
        "ontology_path",
        "ontology_sha256",
    }
)
_ROLE_FIELDS = frozenset(
    {
        "schema_version",
        "packet_id",
        "packet_sha256",
        "ontology_sha256",
        "review_queue_sha256",
        "review_identifier",
        "reviewer_role_identifier",
        "image_id",
        "image_sha256",
        "source_image_width",
        "source_image_height",
        "image_disposition",
        "labels",
    }
)
_LABEL_FIELDS = frozenset(
    {
        "reviewer_local_object_identifier",
        "normalized_category_name",
        "official_coco_category_id",
        "candidate_categories",
        "source_canvas_box_xyxy",
        "reviewer_state",
        "reason_code",
    }
)
_CATEGORY_FIELDS = frozenset({"normalized_category_name", "official_coco_category_id"})
_OWNER_FIELDS = frozenset(
    {
        "schema_version",
        "image_id",
        "image_sha256",
        "owner_id",
        "owner_origin",
        "normalized_category_name",
        "official_coco_category_id",
        "source_canvas_box_xyxy",
        "linked_reviewer_label_identifiers",
    }
)
_DISPOSITION_FIELDS = frozenset(
    {
        "linked_reviewer_label_identifiers",
        "adjudication_state",
        "reason_code",
        "linked_official_owner_ids",
        "final_normalized_category_name",
        "final_official_coco_category_id",
        "final_source_canvas_box_xyxy",
    }
)
_ADJUDICATION_QUEUE_FIELDS = frozenset(
    {
        "schema_version",
        "adjudication_identifier",
        "image_id",
        "image_sha256",
        "source_image_width",
        "source_image_height",
        "review_queue_sha256",
        "packet_sha256",
        "ontology_sha256",
        "reviewer_role_artifact_sha256",
        "reviewer_role_seal_sha256",
        "reviewer_dispositions",
        "official_owner_ledger_sha256",
        "official_owners",
    }
)
_UNCERTAINTY_FIELDS = frozenset(
    {
        "schema_version",
        "image_id",
        "owner_universe_complete",
        "owner_universe_uncertainty_reasons",
        "retained_uncertainty_axes",
        "unresolved_proposal_dispositions",
    }
)
_OWNER_LEDGER_SEAL_FIELDS = frozenset(
    {
        "schema_version",
        "look_id",
        "queue_manifest_sha256",
        "selection_sha256",
        "member_manifest_sha256",
        "candidate_pool_sha256",
        "census_sha256",
        "current_image_ids_sha256",
        "review_queue_sha256",
        "official_owner_ledger_sha256",
        "adjudication_queue_sha256",
        "adjudication_sha256",
        "owner_ledger_sha256",
        "uncertainty_ledger_sha256",
        "selected_image_count",
        "owner_count",
        "official_owner_count",
        "added_owner_count",
        "owner_universe_uncertain_image_count",
        "add_only",
        "official_owner_identity_and_box_immutable",
        "route_blind",
    }
)
_QUEUE_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "look_id",
        "population_scope",
        "selected_image_count",
        "private_sample_order_image_ids",
        "reviewer_numeric_order_image_ids",
        "current_image_ids_sha256",
        "selection_sha256",
        "member_manifest_sha256",
        "member_manifest_receipt_sha256",
        "candidate_pool_sha256",
        "census_sha256",
        "prior_look_receipt_sha256",
        "packet_sha256",
        "ontology_sha256",
        "review_queue_sha256",
        "official_owner_ledger_sha256",
        "reviewer_visible_fields",
        "official_owner_ledger_reviewer_visible",
        "private_manifest_reviewer_visible",
        "sample_order_reviewer_visible",
        "look_identity_reviewer_visible",
        "cutoff_or_prior_outcome_reviewer_visible",
    }
)
_REPLAY_FIELDS = frozenset(
    {
        "schema_version",
        "look_id",
        "image_id",
        "selection_sha256",
        "member_manifest_sha256",
        "candidate_pool_sha256",
        "census_sha256",
        "current_image_ids_sha256",
        "owner_ledger_sha256",
        "owner_ledger_seal_sha256",
        "official_owner_count",
        "added_owner_count",
        "route_count",
        "route_ids",
        "route_identity",
        "no_op_exact_reproduction",
        "census_record",
    }
)
_OUTCOME_FIELDS = frozenset(
    {
        "schema_version",
        "look_id",
        "image_id",
        "selection_sha256",
        "member_manifest_sha256",
        "current_image_ids_sha256",
        "owner_ledger_seal_sha256",
        "replay_records_sha256",
        "replay_receipt_sha256",
        "uncertainty_ledger_sha256",
        "outcome",
        "statistical_success",
        "primary_natural_alias_admitted",
        "owner_universe_complete",
        "review_uncertainty_axes",
        "replay_uncertainty_axes",
        "retained_uncertainty_axes",
        "uncertain_candidate_ids",
    }
)
_REPLAY_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "terminal_status",
        "look_id",
        "selection_sha256",
        "member_manifest_sha256",
        "member_manifest_receipt_sha256",
        "candidate_pool_sha256",
        "census_path",
        "census_sha256",
        "current_image_ids_sha256",
        "queue_manifest_sha256",
        "adjudication_queue_sha256",
        "adjudication_sha256",
        "uncertainty_ledger_sha256",
        "owner_ledger_sha256",
        "owner_ledger_seal_sha256",
        "ordered_image_ids",
        "image_count",
        "route_count_per_image",
        "replay_records_sha256",
        "no_op_exact_reproduction_count",
        "original_candidate_receipt_reused",
        "original_image_analyzer_reused",
        "global_match_prefix_reused",
        "original_row_identity_and_order_preserved",
    }
)
_SEQUENTIAL_DECISION_FIELDS = frozenset(
    {
        "schema_version",
        "terminal_status",
        "look_id",
        "selection_sha256",
        "member_manifest_sha256",
        "member_manifest_receipt_sha256",
        "candidate_pool_sha256",
        "census_sha256",
        "current_image_ids_sha256",
        "current_outcomes_sha256",
        "prior_look_receipt_sha256",
        "cumulative_image_ids_sha256",
        "cumulative_outcomes_sha256",
        "cumulative_sample_size",
        "statistical_success_count",
        "rejection_success_count_max",
        "boundary_probability_numerator",
        "boundary_probability_denominator",
        "decision",
        "look_two_authorized",
        "closed_test_complete",
    }
)
_MEMBER_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "image_id",
        "source_image_path",
        "source_image_size_bytes",
        "source_image_sha256",
        "source_image_width",
        "source_image_height",
        "candidate_pool_record_sha256",
        "census_record_sha256",
        "official_owner_record_sha256",
        "replay_input_sha256",
        "route_count",
        "route_inventory_sha256",
    }
)
_MEMBER_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "terminal_status",
        "built_before_entropy",
        "member_manifest_path",
        "member_manifest_sha256",
        "member_count",
        "ordered_member_image_ids_sha256",
        "member_semantic_sha256",
        "stage_zero_root",
        "stage_zero_root_inventory_sha256",
        "stage_zero_receipt_path",
        "stage_zero_receipt_sha256",
        "stage_zero_audit_path",
        "stage_zero_audit_sha256",
        "stage_zero_audit_verdict",
        "stage_zero_possible_pool_path",
        "stage_zero_possible_pool_sha256",
        "stage_zero_possible_pool_count",
        "stage_zero_ordered_possible_image_ids_sha256",
        "candidate_pool_path",
        "candidate_pool_sha256",
        "census_path",
        "census_sha256",
        "frozen_contract_path",
        "frozen_contract_sha256",
        "source_panel_root",
        "source_panel_manifest_set_sha256",
        "sampled_panel_root",
        "sampled_panel_manifest_set_sha256",
        "route_count",
        "route_inventory_sha256",
        "adapter_inventory_sha256",
    }
)
_SELECTION_FIELDS = frozenset(
    {
        "schema_version",
        "terminal_status",
        "population_scope",
        "population_size_N",
        "null_success_count_K",
        "per_look_alpha",
        "cumulative_sample_sizes",
        "sample_size",
        "ordered_possible_pool_sha256",
        "ordered_sample_sha256",
        "look_one_image_ids",
        "look_two_additional_image_ids",
        "look_two_cumulative_image_ids",
        "entropy",
        "hypergeometric_design",
        "frozen_source_manifest",
        "frozen_source_manifest_sha256",
        "stage_zero_binding",
        "stage_zero_binding_sha256",
        "possible_pool_member_manifest",
        "possible_pool_member_manifest_sha256",
        "possible_pool_member_records_sha256",
        "stage_zero_replay_validation",
    }
)
_FINAL_LOOK_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "terminal_status",
        "look_id",
        "final_root",
        "selection_sha256",
        "member_manifest_sha256",
        "member_manifest_receipt_sha256",
        "candidate_pool_sha256",
        "census_sha256",
        "current_image_ids_sha256",
        "prior_look_receipt_sha256",
        "source_snapshot_sha256",
        "sequential_decision_sha256",
        "artifact_inventory",
        "artifact_inventory_sha256",
        "filesystem_immutable",
        "finalized_atomically",
    }
)

SCHEMA_REGISTRY = {
    "selection": sorted(_SELECTION_FIELDS),
    "member_manifest": sorted(_MEMBER_MANIFEST_FIELDS),
    "member_manifest_receipt": sorted(_MEMBER_RECEIPT_FIELDS),
    "review_queue": sorted(_QUEUE_FIELDS),
    "review_queue_manifest": sorted(_QUEUE_MANIFEST_FIELDS),
    "reviewer_role": sorted(_ROLE_FIELDS),
    "reviewer_label": sorted(_LABEL_FIELDS),
    "category": sorted(_CATEGORY_FIELDS),
    "owner_ledger": sorted(_OWNER_FIELDS),
    "proposal_disposition": sorted(_DISPOSITION_FIELDS),
    "adjudication_queue": sorted(_ADJUDICATION_QUEUE_FIELDS),
    "uncertainty_ledger": sorted(_UNCERTAINTY_FIELDS),
    "owner_ledger_seal": sorted(_OWNER_LEDGER_SEAL_FIELDS),
    "global_replay": sorted(_REPLAY_FIELDS),
    "global_replay_receipt": sorted(_REPLAY_RECEIPT_FIELDS),
    "outcome": sorted(_OUTCOME_FIELDS),
    "sequential_decision": sorted(_SEQUENTIAL_DECISION_FIELDS),
    "final_look_receipt": sorted(_FINAL_LOOK_RECEIPT_FIELDS),
    "reviewer_schema_by_role": REVIEWER_SCHEMA_BY_ROLE,
    "reviewer_reason_by_state": {
        key: sorted(value) for key, value in _REVIEW_REASON_BY_STATE.items()
    },
    "adjudication_reason_by_state": {
        key: sorted(value) for key, value in _ADJUDICATION_REASON_BY_STATE.items()
    },
    "uncertainty_axes": list(UNCERTAINTY_AXES),
}


@dataclass(frozen=True)
class ReviewQueueArtifacts:
    review_queue_jsonl: bytes
    official_owner_ledger_jsonl: bytes
    manifest_json: bytes


@dataclass(frozen=True)
class MemberManifestArtifacts:
    member_manifest_jsonl: bytes
    member_manifest_receipt_json: bytes


@dataclass(frozen=True)
class FrozenCensusSelection:
    records: Mapping[str, Mapping[str, Any]]
    source_path: str
    source_sha256: str
    selected_image_ids_sha256: str


@dataclass(frozen=True)
class ValidatedEntropyJournal:
    root: Path
    terminal_path: Path
    claim: Mapping[str, Any]
    claim_semantic_sha256: str
    claim_file_sha256: str
    entropy_bytes: bytes
    entropy_sha256: str


@dataclass(frozen=True)
class ValidatedSelection:
    document: Mapping[str, Any]
    selection_sha256: str
    selection_root: Path
    journal_terminal_path: Path
    ordered_pool_image_ids: tuple[str, ...]
    ordered_selected_image_ids: tuple[str, ...]
    look_one_image_ids: tuple[str, ...]
    look_two_additional_image_ids: tuple[str, ...]
    look_two_cumulative_image_ids: tuple[str, ...]
    cutoff_by_look: Mapping[str, int]
    member_manifest_path: Path
    member_manifest_sha256: str
    source_artifacts: Mapping[str, Mapping[str, Any]]
    frozen_contract_path: Path
    frozen_contract_sha256: str


@dataclass(frozen=True)
class OwnerLedgerArtifacts:
    adjudication_jsonl: bytes
    owner_ledger_jsonl: bytes
    uncertainty_ledger_jsonl: bytes
    owner_ledger_seal_json: bytes


@dataclass(frozen=True)
class ReplayArtifacts:
    replay_records_jsonl: bytes
    replay_receipt_json: bytes


@dataclass(frozen=True)
class ValidatedMemberManifest:
    rows: tuple[Mapping[str, Any], ...]
    ordered_image_ids: tuple[str, ...]
    semantic_sha256: str
    receipt: Mapping[str, Any]


@dataclass(frozen=True)
class ValidatedReplayChain:
    rows: tuple[Mapping[str, Any], ...]
    receipt: Mapping[str, Any]
    ordered_image_ids: tuple[str, ...]


def _canonical_json_value(value: Any, *, path: str = "$") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"canonical JSON contains a non-finite number at {path}")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"canonical JSON object key is not a string at {path}")
            result[key] = _canonical_json_value(item, path=f"{path}.{key}")
        return result
    if isinstance(value, (tuple, list)):
        return [
            _canonical_json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise ValueError(
        f"canonical JSON contains unsupported {type(value).__name__} at {path}"
    )


def canonical_json_text(value: Any) -> str:
    """Return the experiment's single canonical JSON representation."""

    return json.dumps(
        _canonical_json_value(value),
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_manifest(path: Path) -> tuple[list[dict[str, Any]], str]:
    """Return an exact, content-addressed inventory for one source tree."""

    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"manifest-set source is not a directory: {resolved}")
    rows: list[dict[str, Any]] = []
    for entry in sorted(resolved.rglob("*"), key=lambda item: item.as_posix()):
        if entry.is_symlink():
            raise ValueError(f"manifest-set source contains a symlink: {entry}")
        if entry.is_dir():
            continue
        if not entry.is_file():
            raise ValueError(f"manifest-set source contains a non-file: {entry}")
        rows.append(
            {
                "relative_path": entry.relative_to(resolved).as_posix(),
                "size_bytes": entry.stat().st_size,
                "sha256": _sha256_file(entry),
            }
        )
    return rows, _sha256_bytes(canonical_json_text(rows).encode("utf-8"))


def _root_inventory(path: Path) -> tuple[dict[str, dict[str, Any]], str]:
    """Return the Stage Zero inventory mapping used by its audit contract."""

    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"inventory root is not a directory: {resolved}")
    inventory: dict[str, dict[str, Any]] = {}
    for entry in sorted(resolved.rglob("*"), key=lambda item: item.as_posix()):
        if entry.is_symlink():
            raise ValueError(f"inventory root contains a symlink: {entry}")
        if entry.is_dir():
            continue
        if not entry.is_file():
            raise ValueError(f"inventory root contains a non-file: {entry}")
        relative = entry.relative_to(resolved).as_posix()
        inventory[relative] = {
            "relative_path": relative,
            "sha256": _sha256_file(entry),
            "size_bytes": entry.stat().st_size,
        }
    return inventory, _sha256_bytes(
        canonical_json_text(inventory).encode("utf-8")
    )


def _ordered_image_ids_sha256(image_ids: Sequence[str]) -> str:
    return _sha256_bytes(canonical_json_text(list(image_ids)).encode("utf-8"))


def _route_inventory_sha256() -> str:
    return _sha256_bytes(canonical_json_text(list(EXPECTED_ROUTE_IDS)).encode("utf-8"))


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (canonical_json_text(value) + "\n").encode("utf-8")


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(canonical_json_text(row) + "\n" for row in rows).encode("utf-8")


def _parse_json(payload: bytes, *, artifact: str) -> Mapping[str, Any]:
    if not payload.endswith(b"\n"):
        raise ValueError(f"{artifact} must end with a newline")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{artifact} is not valid UTF-8 JSON") from exc
    if not isinstance(value, Mapping) or payload != _json_bytes(value):
        raise ValueError(f"{artifact} must be one canonical JSON object")
    return value


def _parse_pretty_json(payload: bytes, *, artifact: str) -> Mapping[str, Any]:
    """Parse the selector's frozen indented, sorted-key JSON serialization."""

    if not payload.endswith(b"\n"):
        raise ValueError(f"{artifact} must end with a newline")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{artifact} is not valid UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{artifact} must contain one JSON object")
    expected = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    if payload != expected:
        raise ValueError(f"{artifact} does not use the frozen selector serialization")
    return value


def _parse_jsonl(
    payload: bytes, *, artifact: str, allow_empty: bool = False
) -> list[Mapping[str, Any]]:
    if (not payload and not allow_empty) or (payload and not payload.endswith(b"\n")):
        raise ValueError(f"{artifact} must be canonical JSONL with a final newline")
    rows: list[Mapping[str, Any]] = []
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{artifact} must use UTF-8") from exc
    for line in text.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{artifact} contains invalid JSON") from exc
        if not isinstance(value, Mapping) or line != canonical_json_text(value):
            raise ValueError(f"{artifact} rows must be canonical JSON objects")
        rows.append(value)
    return rows


def _parse_frozen_input_jsonl(
    payload: bytes, *, artifact: str
) -> list[Mapping[str, Any]]:
    """Parse a digest-bound legacy JSONL input without rewriting its whitespace."""

    if not payload or not payload.endswith(b"\n"):
        raise ValueError(f"{artifact} must be nonempty JSONL with a final newline")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{artifact} must use UTF-8") from exc
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise ValueError(f"{artifact} contains blank row {line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{artifact} contains invalid row {line_number}") from exc
        if not isinstance(value, Mapping):
            raise ValueError(f"{artifact} row {line_number} is not an object")
        rows.append(value)
    return rows


def load_frozen_census_records(
    *,
    census_path: Path,
    expected_census_path: Path,
    expected_census_sha256: str,
    selected_image_ids: Sequence[str],
) -> FrozenCensusSelection:
    """Read the frozen spaced JSONL exactly, then return the exact selected set.

    The byte digest is checked before UTF-8 or JSON decoding.  Serialization is
    deliberately not compared with this assembler's canonical JSON because the
    already-frozen census uses its own spaced JSONL representation.
    """

    requested = tuple(
        _canonical_image_id(value, field="selected census image_id")
        for value in selected_image_ids
    )
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("selected census image IDs must be nonempty and unique")
    expected_path = expected_census_path.expanduser().resolve(strict=True)
    resolved = census_path.expanduser().resolve(strict=True)
    if resolved != expected_path:
        raise ValueError("frozen census path drift")
    payload = resolved.read_bytes()
    observed_sha256 = _sha256_bytes(payload)
    if observed_sha256 != _require_sha256(
        expected_census_sha256, field="expected_census_sha256"
    ):
        raise ValueError("frozen census byte digest drift")
    rows = _parse_frozen_input_jsonl(payload, artifact="frozen census")
    by_image: dict[str, Mapping[str, Any]] = {}
    for line_number, row in enumerate(rows, start=1):
        image_id = _canonical_image_id(
            row.get("image_id"), field=f"frozen census row {line_number} image_id"
        )
        if image_id in by_image:
            raise ValueError(f"frozen census duplicates image_id {image_id}")
        by_image[image_id] = row
    requested_set = set(requested)
    missing = sorted(requested_set - set(by_image), key=int)
    if missing:
        raise ValueError(f"frozen census is missing selected image IDs: {missing}")
    selected = {image_id: by_image[image_id] for image_id in requested}
    if set(selected) != requested_set or len(selected) != len(requested):
        raise ValueError("frozen census selected-set proof failed")
    return FrozenCensusSelection(
        records=selected,
        source_path=str(resolved),
        source_sha256=observed_sha256,
        selected_image_ids_sha256=_sha256_bytes(
            canonical_json_text(list(requested)).encode("utf-8")
        ),
    )


def _require_fields(
    value: Mapping[str, Any], expected: set[str] | frozenset[str], *, artifact: str
) -> None:
    if set(value) != set(expected):
        difference = sorted(set(value) ^ set(expected))
        forbidden = sorted(set(value) & _FORBIDDEN_REVIEW_EVIDENCE_FIELDS)
        if forbidden:
            raise ValueError(
                f"{artifact} contains forbidden evidence fields: {forbidden}"
            )
        raise ValueError(f"{artifact} fields differ: {difference}")


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _canonical_image_id(value: Any, *, field: str) -> str:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a canonical nonnegative decimal image ID")
    text = str(value)
    if not text.isdigit() or str(int(text)) != text:
        raise ValueError(f"{field} must be a canonical nonnegative decimal image ID")
    return text


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _require_category(name: Any, official_id: Any) -> tuple[str, int]:
    if not isinstance(name, str) or normalize_coco_category_name(name) != name:
        raise ValueError("category name must be canonical COCO-80")
    expected = COCO_80_OFFICIAL_CATEGORY_ID_BY_NAME.get(name)
    if expected is None or isinstance(official_id, bool) or official_id != expected:
        raise ValueError("category name and official COCO identifier disagree")
    return name, expected


def _require_box(
    value: Any, *, width: int, height: int, integer: bool
) -> list[int | float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("source-canvas box must have four coordinates")
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float)) for item in value
    ):
        raise ValueError("source-canvas box coordinates must be numeric")
    if integer and any(not isinstance(item, int) for item in value):
        raise ValueError("review boxes must use integer pixel edges")
    box = list(value)
    if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
        raise ValueError("source-canvas box is outside the source image")
    return box


def _load_candidate_pool(
    path: Path, *, expected_sha256: str
) -> dict[str, Mapping[str, Any]]:
    _require_sha256(expected_sha256, field="expected_candidate_pool_sha256")
    resolved = path.expanduser().resolve(strict=True)
    if _sha256_file(resolved) != expected_sha256:
        raise ValueError("candidate-pool digest drift")
    rows = _parse_frozen_input_jsonl(resolved.read_bytes(), artifact="candidate pool")
    result: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows, start=1):
        image_id = _canonical_image_id(
            row.get("image_id"), field=f"candidate pool row {index} image_id"
        )
        if image_id in result:
            raise ValueError(f"candidate pool duplicates image {image_id}")
        result[image_id] = row
    return result


def _validate_ontology(path: Path, *, expected_sha256: str) -> None:
    if _sha256_file(path) != _require_sha256(
        expected_sha256, field="expected_ontology_sha256"
    ):
        raise ValueError("review ontology digest drift")
    try:
        value = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("review ontology is not valid UTF-8 JSON") from exc
    if not isinstance(value, list) or len(value) != len(
        COCO_80_OFFICIAL_CATEGORY_ID_BY_NAME
    ):
        raise ValueError("review ontology must contain the exact COCO-80 inventory")
    observed_names: set[str] = set()
    for expected_evaluator_id, raw in enumerate(value, start=1):
        if not isinstance(raw, Mapping):
            raise ValueError("review ontology entry must be an object")
        _require_fields(
            raw,
            {
                "evaluator_category_id",
                "normalized_category_name",
                "official_coco_category_id",
            },
            artifact="review ontology entry",
        )
        name, _ = _require_category(
            raw["normalized_category_name"], raw["official_coco_category_id"]
        )
        if (
            raw["evaluator_category_id"] != expected_evaluator_id
            or COCO_80_EVALUATOR_LOCAL_CATEGORY_ID_BY_NAME[name]
            != expected_evaluator_id
            or name in observed_names
        ):
            raise ValueError("review ontology order or evaluator identifier drift")
        observed_names.add(name)
    if observed_names != set(COCO_80_OFFICIAL_CATEGORY_ID_BY_NAME):
        raise ValueError("review ontology category inventory drift")


def _falling_factorial(value: int, count: int) -> int:
    result = 1
    for offset in range(count):
        result *= value - offset
    return result


def _hypergeometric_lower_tail(
    *, population_size: int, success_count: int, sample_size: int, observed: int
) -> Fraction:
    denominator = math.comb(population_size, sample_size)
    numerator = sum(
        math.comb(success_count, successes)
        * math.comb(population_size - success_count, sample_size - successes)
        for successes in range(
            max(0, sample_size - (population_size - success_count)), observed + 1
        )
        if successes <= success_count
        and 0 <= sample_size - successes <= population_size - success_count
    )
    return Fraction(numerator, denominator)


def _exact_hypergeometric_cutoff(
    *, population_size: int, success_count: int, sample_size: int
) -> tuple[int, Fraction]:
    selected = -1
    probability = Fraction(0, 1)
    for observed in range(sample_size + 1):
        current = _hypergeometric_lower_tail(
            population_size=population_size,
            success_count=success_count,
            sample_size=sample_size,
            observed=observed,
        )
        if current <= PER_LOOK_ALPHA:
            selected, probability = observed, current
        else:
            break
    return selected, probability


def _decimal_integer(value: Any, *, field: str) -> int:
    if (
        not isinstance(value, str)
        or not value
        or not value.isascii()
        or not value.isdecimal()
    ):
        raise ValueError(
            f"{field} must be a canonical nonnegative decimal integer string"
        )
    if value != "0" and value.startswith("0"):
        raise ValueError(f"{field} has a leading zero")
    return int(value)


def _validate_file_binding(value: Any, *, label: str) -> tuple[Path, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} binding must be an object")
    _require_fields(value, {"path", "sha256"}, artifact=f"{label} binding")
    path = Path(str(value["path"])).expanduser().resolve(strict=True)
    digest = _require_sha256(value["sha256"], field=f"{label}.sha256")
    if not path.is_file() or _sha256_file(path) != digest:
        raise ValueError(f"{label} path/hash drift")
    return path, digest


def _validate_member_manifest(
    payload: bytes, *, expected_sha256: str
) -> tuple[tuple[Mapping[str, Any], ...], tuple[str, ...], str]:
    if _sha256_bytes(payload) != _require_sha256(
        expected_sha256, field="expected_member_manifest_sha256"
    ):
        raise ValueError("pre-entropy member-manifest digest drift")
    rows = tuple(_parse_jsonl(payload, artifact="pre-entropy member manifest"))
    image_ids: list[str] = []
    prior: int | None = None
    for row in rows:
        _require_fields(row, _MEMBER_MANIFEST_FIELDS, artifact="pre-entropy member row")
        if row["schema_version"] != MEMBER_MANIFEST_SCHEMA_VERSION:
            raise ValueError("pre-entropy member schema drift")
        image_id = _canonical_image_id(row["image_id"], field="member image_id")
        if image_id in image_ids:
            raise ValueError("pre-entropy member manifest duplicates an image")
        if prior is not None and int(image_id) <= prior:
            raise ValueError("pre-entropy member manifest must use numeric image order")
        prior = int(image_id)
        image_ids.append(image_id)
        for field in (
            "source_image_sha256",
            "candidate_pool_record_sha256",
            "census_record_sha256",
            "official_owner_record_sha256",
            "replay_input_sha256",
            "route_inventory_sha256",
        ):
            _require_sha256(row[field], field=f"member.{field}")
        if row["route_count"] != len(EXPECTED_ROUTE_IDS):
            raise ValueError("pre-entropy member route count drift")
        if row["route_inventory_sha256"] != _sha256_bytes(
            canonical_json_text(list(EXPECTED_ROUTE_IDS)).encode("utf-8")
        ):
            raise ValueError("pre-entropy member route inventory drift")
        _positive_int(row["source_image_size_bytes"], field="member source image size")
        width = _positive_int(
            row["source_image_width"], field="member source image width"
        )
        height = _positive_int(
            row["source_image_height"], field="member source image height"
        )
        source_path = (
            Path(str(row["source_image_path"])).expanduser().resolve(strict=True)
        )
        if (
            not source_path.is_file()
            or source_path.stat().st_size != row["source_image_size_bytes"]
            or _sha256_file(source_path) != row["source_image_sha256"]
        ):
            raise ValueError("pre-entropy member source path/size/hash drift")
        with Image.open(source_path) as opened:
            if opened.size != (width, height):
                raise ValueError("pre-entropy member source dimensions drift")
    semantic_hash = _sha256_bytes(
        canonical_json_text(
            [
                [
                    row["image_id"],
                    row["source_image_sha256"],
                    row["official_owner_record_sha256"],
                    row["replay_input_sha256"],
                ]
                for row in rows
            ]
        ).encode("utf-8")
    )
    return rows, tuple(image_ids), semantic_hash


def _canonical_official_owner_semantics(
    owners: Any, *, image_id: str
) -> list[dict[str, Any]]:
    if not isinstance(owners, list) or any(
        not isinstance(owner, Mapping) for owner in owners
    ):
        raise AssemblyError(f"image {image_id} official owners are invalid")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    expected_keys = {
        "owner_id",
        "category",
        "bbox",
        "category_id",
        "annotation_index",
        "image_id",
    }
    for raw_owner in owners:
        owner = census._mapping(raw_owner, f"official owner {image_id}")
        _require_fields(owner, expected_keys, artifact=f"official owner {image_id}")
        owner_id = owner.get("owner_id")
        if (
            not isinstance(owner_id, str)
            or not owner_id.startswith(f"{image_id}:")
            or owner_id in seen
        ):
            raise AssemblyError(f"image {image_id} official owner IDs drift")
        seen.add(owner_id)
        source_category = owner.get("category")
        if not isinstance(source_category, str) or not source_category:
            raise AssemblyError(f"image {image_id} official category drift")
        name = normalize_coco_category_name(source_category)
        official_id = COCO_80_OFFICIAL_CATEGORY_ID_BY_NAME.get(name)
        _require_category(name, official_id)
        source_category_id = owner.get("category_id")
        if source_category_id is not None and (
            isinstance(source_category_id, bool)
            or not isinstance(source_category_id, int)
            or source_category_id != official_id
        ):
            raise AssemblyError(f"image {image_id} official category ID drift")
        annotation_index = owner.get("annotation_index")
        if (
            isinstance(annotation_index, bool)
            or not isinstance(annotation_index, int)
            or annotation_index < 0
        ):
            raise AssemblyError(f"image {image_id} annotation index drift")
        owner_image_id = _canonical_image_id(
            owner.get("image_id"), field="official owner image_id"
        )
        if owner_image_id != image_id:
            raise AssemblyError(f"image {image_id} owner image binding drift")
        bbox = owner.get("bbox")
        if (
            not isinstance(bbox, (list, tuple))
            or len(bbox) != 4
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in bbox
            )
        ):
            raise AssemblyError(f"image {image_id} official owner box is invalid")
        result.append(
            {
                "owner_id": owner_id,
                "source_category_name": source_category,
                "normalized_category_name": name,
                "source_category_id": source_category_id,
                "official_coco_category_id": official_id,
                "source_canvas_box_xyxy": [float(value) for value in bbox],
                "annotation_index": annotation_index,
                "image_id": owner_image_id,
            }
        )
    return sorted(result, key=lambda row: row["owner_id"])


def _validate_adapter_inventory(
    adapter: Mapping[str, Any], *, ordered_image_ids: Sequence[str]
) -> None:
    expected_images = set(ordered_image_ids)
    for field in ("image_results", "reference_records"):
        value = adapter.get(field)
        if not isinstance(value, Mapping) or set(value) != expected_images:
            raise AssemblyError(f"adapter {field} is not the exact image inventory")
    source_rows = adapter.get("source_rows")
    sampled_rows = adapter.get("sampled_rows")
    if not isinstance(source_rows, Mapping) or set(source_rows) != {
        (image_id, 0) for image_id in ordered_image_ids
    }:
        raise AssemblyError("adapter source rows are not the exact B16 inventory")
    if not isinstance(sampled_rows, Mapping) or set(sampled_rows) != {
        (image_id, index) for image_id in ordered_image_ids for index in range(16)
    }:
        raise AssemblyError("adapter sampled rows are not the exact 16-route inventory")
    for image_id in ordered_image_ids:
        result = census._mapping(
            adapter["image_results"][image_id], f"image_results[{image_id}]"
        )
        evidence = census._mapping(
            result.get("trajectory_evidence"), f"trajectory_evidence[{image_id}]"
        )
        if set(evidence) != set(EXPECTED_ROUTE_IDS):
            raise AssemblyError(f"image {image_id} trajectory evidence inventory drift")
        budgets = result.get("budgets")
        if (
            not isinstance(budgets, list)
            or len(budgets) != 1
            or not isinstance(budgets[0], Mapping)
            or budgets[0].get("budget") != 16
        ):
            raise AssemblyError(f"image {image_id} lacks its unique B16 budget")
        assignments = census._mapping(
            budgets[0].get("trajectory_assignments"),
            f"trajectory_assignments[{image_id}]",
        )
        if set(assignments) != set(EXPECTED_ROUTE_IDS):
            raise AssemblyError(f"image {image_id} assignment inventory drift")
        candidates = census._image_candidates(image_id, adapter, reverse_input=False)
        if tuple(row.get("candidate_id") for row in candidates) != EXPECTED_ROUTE_IDS:
            raise AssemblyError(f"image {image_id} candidate inventory/order drift")


def _member_replay_input_sha256(
    *, adapter: Mapping[str, Any], image_id: str, census_record: Mapping[str, Any]
) -> str:
    result = census._mapping(
        adapter["image_results"][image_id], f"image_results[{image_id}]"
    )
    evidence = census._mapping(
        result.get("trajectory_evidence"), f"trajectory_evidence[{image_id}]"
    )
    budgets = result.get("budgets")
    if not isinstance(budgets, list) or len(budgets) != 1:
        raise AssemblyError(f"image {image_id} B16 budget inventory drift")
    assignments = census._mapping(
        budgets[0].get("trajectory_assignments"),
        f"trajectory_assignments[{image_id}]",
    )
    routes: list[dict[str, Any]] = []
    for route_id in EXPECTED_ROUTE_IDS:
        route_row = (
            adapter["source_rows"].get((image_id, 0))
            if route_id == "source-b16"
            else adapter["sampled_rows"].get((image_id, int(route_id[-2:])))
        )
        if not isinstance(route_row, Mapping):
            raise AssemblyError(f"image {image_id} lacks exact route {route_id}")
        routes.append(
            {
                "route_id": route_id,
                "route_row_sha256": _sha256_bytes(
                    canonical_json_text(route_row).encode("utf-8")
                ),
                "trajectory_evidence_sha256": _sha256_bytes(
                    canonical_json_text(evidence[route_id]).encode("utf-8")
                ),
                "assignment_sha256": _sha256_bytes(
                    canonical_json_text(assignments[route_id]).encode("utf-8")
                ),
            }
        )
    return _sha256_bytes(
        canonical_json_text(
            {
                "image_id": image_id,
                "route_inventory": routes,
                "census_record_sha256": _sha256_bytes(
                    canonical_json_text(census_record).encode("utf-8")
                ),
            }
        ).encode("utf-8")
    )


def _adapter_inventory_sha256(
    adapter: Mapping[str, Any], *, ordered_image_ids: Sequence[str]
) -> str:
    return _sha256_bytes(
        canonical_json_text(
            {
                "image_results": list(ordered_image_ids),
                "reference_records": list(ordered_image_ids),
                "source_rows": [[image_id, 0] for image_id in ordered_image_ids],
                "sampled_rows": [
                    [image_id, index]
                    for image_id in ordered_image_ids
                    for index in range(16)
                ],
                "route_ids": list(EXPECTED_ROUTE_IDS),
            }
        ).encode("utf-8")
    )


def _load_stage_zero_possible_pool(
    *,
    stage_zero_root: Path,
    expected_stage_zero_root_inventory_sha256: str,
    stage_zero_receipt_path: Path,
    expected_stage_zero_receipt_sha256: str,
    stage_zero_audit_path: Path,
    expected_stage_zero_audit_sha256: str,
    possible_pool_path: Path,
    expected_possible_pool_sha256: str,
) -> tuple[tuple[str, ...], Mapping[str, Any]]:
    """Read and bind the exact approved Stage Zero possible-pool inventory."""

    root = stage_zero_root.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Stage Zero root is not a directory")
    root_inventory_sha256 = _require_sha256(
        expected_stage_zero_root_inventory_sha256,
        field="expected_stage_zero_root_inventory_sha256",
    )
    root_inventory, observed_root_inventory_sha256 = _root_inventory(root)
    if observed_root_inventory_sha256 != root_inventory_sha256:
        raise ValueError("Stage Zero complete root inventory drift")

    receipt_path = stage_zero_receipt_path.expanduser().resolve(strict=True)
    pool_path = possible_pool_path.expanduser().resolve(strict=True)
    if receipt_path != root / "receipt.json" or pool_path != root / "possible-pool.json":
        raise ValueError("Stage Zero receipt/possible-pool canonical path drift")
    receipt_sha256 = _require_sha256(
        expected_stage_zero_receipt_sha256,
        field="expected_stage_zero_receipt_sha256",
    )
    possible_pool_sha256 = _require_sha256(
        expected_possible_pool_sha256,
        field="expected_possible_pool_sha256",
    )
    receipt_payload = receipt_path.read_bytes()
    possible_pool_payload = pool_path.read_bytes()
    if _sha256_bytes(receipt_payload) != receipt_sha256:
        raise ValueError("Stage Zero receipt digest drift")
    if _sha256_bytes(possible_pool_payload) != possible_pool_sha256:
        raise ValueError("Stage Zero possible-pool digest drift")
    receipt = _parse_pretty_json(receipt_payload, artifact="Stage Zero receipt")
    possible_pool = _parse_pretty_json(
        possible_pool_payload, artifact="Stage Zero possible pool"
    )
    _require_fields(
        possible_pool,
        {
            "schema_version",
            "terminal_status",
            "population_scope",
            "pool_role",
            "count",
            "ordered_image_ids",
            "ordered_image_ids_sha256",
        },
        artifact="Stage Zero possible pool",
    )
    raw_ids = possible_pool["ordered_image_ids"]
    if not isinstance(raw_ids, list):
        raise ValueError("Stage Zero possible-pool image IDs are invalid")
    image_ids = tuple(
        _canonical_image_id(value, field="Stage Zero possible-pool image_id")
        for value in raw_ids
    )
    ordered_image_ids_sha256 = _ordered_image_ids_sha256(image_ids)
    if (
        possible_pool["schema_version"] != STAGE_ZERO_SCHEMA_VERSION
        or possible_pool["terminal_status"] != "completed"
        or possible_pool["population_scope"] != STAGE_ZERO_POPULATION_SCOPE
        or possible_pool["pool_role"] != "possible"
        or not image_ids
        or len(image_ids) != len(set(image_ids))
        or tuple(sorted(image_ids, key=int)) != image_ids
        or possible_pool["count"] != len(image_ids)
        or possible_pool["ordered_image_ids_sha256"] != ordered_image_ids_sha256
    ):
        raise ValueError("Stage Zero possible-pool schema/order binding drift")
    outputs = receipt.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("Stage Zero receipt output inventory is invalid")
    receipt_inventory: dict[str, dict[str, Any]] = {}
    for relative, raw_identity in outputs.items():
        if not isinstance(relative, str) or not isinstance(raw_identity, Mapping):
            raise ValueError("Stage Zero receipt output inventory is invalid")
        _require_fields(
            raw_identity,
            {"relative_path", "sha256", "size_bytes"},
            artifact=f"Stage Zero output {relative}",
        )
        identity = dict(raw_identity)
        if (
            identity["relative_path"] != relative
            or _require_sha256(identity["sha256"], field=f"output {relative}")
            != identity["sha256"]
            or isinstance(identity["size_bytes"], bool)
            or not isinstance(identity["size_bytes"], int)
            or identity["size_bytes"] < 0
        ):
            raise ValueError("Stage Zero receipt output identity is invalid")
        receipt_inventory[relative] = identity
    receipt_inventory["receipt.json"] = {
        "relative_path": "receipt.json",
        "sha256": receipt_sha256,
        "size_bytes": len(receipt_payload),
    }
    if receipt_inventory != root_inventory:
        raise ValueError("Stage Zero receipt and complete root inventory differ")
    required_semantic_outputs = {
        "category-state.jsonl",
        "possibility-census.jsonl",
        "possible-pool.json",
        "impossible-pool.json",
        "summary.json",
    }
    if not required_semantic_outputs.issubset(outputs):
        raise ValueError("Stage Zero semantic output inventory is incomplete")
    possible_output = outputs["possible-pool.json"]
    if (
        receipt.get("schema_version") != STAGE_ZERO_SCHEMA_VERSION
        or receipt.get("terminal_status") != "completed"
        or receipt.get("population_scope") != STAGE_ZERO_POPULATION_SCOPE
        or receipt.get("possible_pool_count") != len(image_ids)
        or receipt.get("ordered_possible_image_ids_sha256")
        != ordered_image_ids_sha256
        or receipt.get("possible_pool_artifact_sha256") != possible_pool_sha256
        or not isinstance(possible_output, Mapping)
        or possible_output.get("relative_path") != "possible-pool.json"
        or possible_output.get("sha256") != possible_pool_sha256
        or possible_output.get("size_bytes") != len(possible_pool_payload)
    ):
        raise ValueError("Stage Zero receipt/possible-pool binding drift")

    impossible_pool_path = root / "impossible-pool.json"
    impossible_pool_payload = impossible_pool_path.read_bytes()
    impossible_pool = _parse_pretty_json(
        impossible_pool_payload, artifact="Stage Zero impossible pool"
    )
    _require_fields(
        impossible_pool,
        {
            "schema_version",
            "terminal_status",
            "population_scope",
            "pool_role",
            "count",
            "ordered_image_ids",
            "ordered_image_ids_sha256",
        },
        artifact="Stage Zero impossible pool",
    )
    raw_impossible_ids = impossible_pool["ordered_image_ids"]
    if not isinstance(raw_impossible_ids, list):
        raise ValueError("Stage Zero impossible-pool image IDs are invalid")
    impossible_ids = tuple(
        _canonical_image_id(value, field="Stage Zero impossible-pool image_id")
        for value in raw_impossible_ids
    )
    ordered_impossible_sha256 = _ordered_image_ids_sha256(impossible_ids)
    if (
        impossible_pool["schema_version"] != STAGE_ZERO_SCHEMA_VERSION
        or impossible_pool["terminal_status"] != "completed"
        or impossible_pool["population_scope"] != STAGE_ZERO_POPULATION_SCOPE
        or impossible_pool["pool_role"] != "certified_impossible"
        or len(impossible_ids) != len(set(impossible_ids))
        or tuple(sorted(impossible_ids, key=int)) != impossible_ids
        or impossible_pool["count"] != len(impossible_ids)
        or impossible_pool["ordered_image_ids_sha256"]
        != ordered_impossible_sha256
        or set(image_ids) & set(impossible_ids)
    ):
        raise ValueError("Stage Zero impossible-pool schema/order binding drift")
    summary_payload = (root / "summary.json").read_bytes()
    summary = _parse_pretty_json(summary_payload, artifact="Stage Zero summary")
    category_rows = _parse_jsonl(
        (root / "category-state.jsonl").read_bytes(),
        artifact="Stage Zero category state",
    )
    possibility_rows = _parse_jsonl(
        (root / "possibility-census.jsonl").read_bytes(),
        artifact="Stage Zero possibility census",
    )
    category_ids = tuple(
        _canonical_image_id(row.get("image_id"), field="category-state image_id")
        for row in category_rows
    )
    possibility_ids = tuple(
        _canonical_image_id(
            row.get("image_id"), field="possibility-census image_id"
        )
        for row in possibility_rows
    )
    possible_from_census = tuple(
        image_id
        for image_id, row in zip(possibility_ids, possibility_rows, strict=True)
        if row.get("possible") is True
    )
    impossible_from_census = tuple(
        image_id
        for image_id, row in zip(possibility_ids, possibility_rows, strict=True)
        if row.get("possible") is False
    )
    category_state_sha256 = _sha256_bytes(
        canonical_json_text(category_rows).encode("utf-8")
    )
    possibility_census_sha256 = _sha256_bytes(
        canonical_json_text(possibility_rows).encode("utf-8")
    )
    if (
        category_ids != possibility_ids
        or len(category_ids) != len(set(category_ids))
        or tuple(sorted(category_ids, key=int)) != category_ids
        or possible_from_census != image_ids
        or impossible_from_census != impossible_ids
        or summary.get("population_count") != len(category_ids)
        or summary.get("possible_pool_count") != len(image_ids)
        or summary.get("impossible_pool_count") != len(impossible_ids)
        or summary.get("category_state_sha256") != category_state_sha256
        or summary.get("possibility_census_sha256")
        != possibility_census_sha256
    ):
        raise ValueError("Stage Zero population/summary partition drift")

    audit_path = stage_zero_audit_path.expanduser().resolve(strict=True)
    audit_sha256 = _require_sha256(
        expected_stage_zero_audit_sha256,
        field="expected_stage_zero_audit_sha256",
    )
    audit_payload = audit_path.read_bytes()
    if _sha256_bytes(audit_payload) != audit_sha256:
        raise ValueError("Stage Zero independent-audit digest drift")
    audit = _parse_pretty_json(audit_payload, artifact="Stage Zero independent audit")
    _require_fields(
        audit,
        {
            "schema_version",
            "terminal_status",
            "population_scope",
            "review_scope",
            "verdict",
            "stage_zero_receipt_sha256",
            "stage_zero_root_inventory_sha256",
            "population_count",
            "possible_pool_count",
            "impossible_pool_count",
            "ordered_possible_image_ids_sha256",
            "ordered_impossible_image_ids_sha256",
            "category_state_sha256",
            "possibility_census_sha256",
            "replayed_possible_witness_count",
            "replayed_impossibility_certificate_count",
            "full_certificate_replay",
        },
        artifact="Stage Zero independent audit",
    )
    expected_audit = {
        "schema_version": STAGE_ZERO_AUDIT_SCHEMA_VERSION,
        "terminal_status": "completed",
        "population_scope": STAGE_ZERO_POPULATION_SCOPE,
        "review_scope": STAGE_ZERO_AUDIT_REVIEW_SCOPE,
        "verdict": "approved",
        "stage_zero_receipt_sha256": receipt_sha256,
        "stage_zero_root_inventory_sha256": root_inventory_sha256,
        "population_count": len(category_ids),
        "possible_pool_count": len(image_ids),
        "impossible_pool_count": len(impossible_ids),
        "ordered_possible_image_ids_sha256": ordered_image_ids_sha256,
        "ordered_impossible_image_ids_sha256": ordered_impossible_sha256,
        "category_state_sha256": category_state_sha256,
        "possibility_census_sha256": possibility_census_sha256,
        "replayed_possible_witness_count": len(image_ids),
        "replayed_impossibility_certificate_count": len(impossible_ids),
        "full_certificate_replay": True,
    }
    if dict(audit) != expected_audit:
        raise ValueError("Stage Zero independent audit binding drift")
    return image_ids, {
        "stage_zero_root": str(root),
        "stage_zero_root_inventory_sha256": root_inventory_sha256,
        "stage_zero_receipt_path": str(receipt_path),
        "stage_zero_receipt_sha256": receipt_sha256,
        "stage_zero_audit_path": str(audit_path),
        "stage_zero_audit_sha256": audit_sha256,
        "stage_zero_audit_verdict": "approved",
        "stage_zero_possible_pool_path": str(pool_path),
        "stage_zero_possible_pool_sha256": possible_pool_sha256,
        "stage_zero_possible_pool_count": len(image_ids),
        "stage_zero_ordered_possible_image_ids_sha256": ordered_image_ids_sha256,
    }


def build_preentropy_member_manifest(
    *,
    adapter: Mapping[str, Any],
    ordered_image_ids: Sequence[str],
    stage_zero_root: Path,
    expected_stage_zero_root_inventory_sha256: str,
    stage_zero_receipt_path: Path,
    expected_stage_zero_receipt_sha256: str,
    stage_zero_audit_path: Path,
    expected_stage_zero_audit_sha256: str,
    possible_pool_path: Path,
    expected_possible_pool_sha256: str,
    candidate_pool_path: Path,
    expected_candidate_pool_sha256: str,
    census_path: Path,
    expected_census_path: Path,
    expected_census_sha256: str,
    intended_member_manifest_path: Path,
    frozen_contract_path: Path,
    expected_frozen_contract_sha256: str,
    source_panel_root: Path,
    sampled_panel_root: Path,
) -> MemberManifestArtifacts:
    """Build the complete source/replay member census before entropy exists."""

    image_ids = tuple(
        _canonical_image_id(value, field="member population image_id")
        for value in ordered_image_ids
    )
    if not image_ids or len(image_ids) != len(set(image_ids)):
        raise ValueError("member population image IDs must be nonempty and unique")
    if tuple(sorted(image_ids, key=int)) != image_ids:
        raise ValueError("member population must use numeric image order")
    possible_image_ids, stage_zero_binding = _load_stage_zero_possible_pool(
        stage_zero_root=stage_zero_root,
        expected_stage_zero_root_inventory_sha256=(
            expected_stage_zero_root_inventory_sha256
        ),
        stage_zero_receipt_path=stage_zero_receipt_path,
        expected_stage_zero_receipt_sha256=expected_stage_zero_receipt_sha256,
        stage_zero_audit_path=stage_zero_audit_path,
        expected_stage_zero_audit_sha256=expected_stage_zero_audit_sha256,
        possible_pool_path=possible_pool_path,
        expected_possible_pool_sha256=expected_possible_pool_sha256,
    )
    if image_ids != possible_image_ids:
        raise ValueError("member population differs from Stage Zero possible pool")
    pool = _load_candidate_pool(
        candidate_pool_path, expected_sha256=expected_candidate_pool_sha256
    )
    if not set(image_ids).issubset(pool):
        raise ValueError("Stage Zero possible pool is not a candidate-pool subset")
    census_selection = load_frozen_census_records(
        census_path=census_path,
        expected_census_path=expected_census_path,
        expected_census_sha256=expected_census_sha256,
        selected_image_ids=image_ids,
    )
    _validate_adapter_inventory(adapter, ordered_image_ids=image_ids)
    frozen_contract = frozen_contract_path.expanduser().resolve(strict=True)
    if _sha256_file(frozen_contract) != _require_sha256(
        expected_frozen_contract_sha256,
        field="expected_frozen_contract_sha256",
    ):
        raise ValueError("frozen contract path/hash drift")
    official_by_image = load_generation7_annotations(
        candidate_pool_path, image_ids=image_ids
    )
    if set(official_by_image) - set(image_ids):
        raise ValueError("official owner loader returned an extra image")
    rows: list[dict[str, Any]] = []
    for image_id in image_ids:
        pool_row = pool[image_id]
        metadata = pool_row.get("metadata")
        images = pool_row.get("images")
        if (
            not isinstance(metadata, Mapping)
            or metadata.get("split") != "train"
            or not isinstance(images, list)
            or len(images) != 1
            or not isinstance(images[0], str)
        ):
            raise ValueError(f"member {image_id} is not one train source image")
        raw_path = Path(images[0]).expanduser()
        source_path = (
            raw_path.resolve(strict=True)
            if raw_path.is_absolute()
            else (candidate_pool_path.resolve().parent / raw_path).resolve(strict=True)
        )
        width = _positive_int(pool_row.get("width"), field="member width")
        height = _positive_int(pool_row.get("height"), field="member height")
        source_sha256 = _sha256_file(source_path)
        with Image.open(source_path) as opened:
            if opened.size != (width, height):
                raise ValueError(f"member {image_id} source dimensions drift")
        reference = census._mapping(
            adapter["reference_records"][image_id], f"reference_records[{image_id}]"
        )
        image_reference = census._mapping(
            reference.get("image"), f"reference image[{image_id}]"
        )
        if (
            Path(str(image_reference.get("path"))).expanduser().resolve(strict=True)
            != source_path
            or image_reference.get("content_sha256") != source_sha256
            or image_reference.get("width") != width
            or image_reference.get("height") != height
        ):
            raise AssemblyError(f"member {image_id} adapter source binding drift")
        adapter_owners = _canonical_official_owner_semantics(
            adapter["image_results"][image_id].get("owners"), image_id=image_id
        )
        loaded_owners = _canonical_official_owner_semantics(
            official_by_image.get(image_id, []), image_id=image_id
        )
        if adapter_owners != loaded_owners:
            raise AssemblyError(f"member {image_id} official owner semantics drift")
        census_record = census_selection.records[image_id]
        rows.append(
            {
                "schema_version": MEMBER_MANIFEST_SCHEMA_VERSION,
                "image_id": image_id,
                "source_image_path": str(source_path),
                "source_image_size_bytes": source_path.stat().st_size,
                "source_image_sha256": source_sha256,
                "source_image_width": width,
                "source_image_height": height,
                "candidate_pool_record_sha256": _sha256_bytes(
                    canonical_json_text(pool_row).encode("utf-8")
                ),
                "census_record_sha256": _sha256_bytes(
                    canonical_json_text(census_record).encode("utf-8")
                ),
                "official_owner_record_sha256": _sha256_bytes(
                    canonical_json_text(adapter_owners).encode("utf-8")
                ),
                "replay_input_sha256": _member_replay_input_sha256(
                    adapter=adapter,
                    image_id=image_id,
                    census_record=census_record,
                ),
                "route_count": len(EXPECTED_ROUTE_IDS),
                "route_inventory_sha256": _route_inventory_sha256(),
            }
        )
    member_payload = _jsonl_bytes(rows)
    _, validated_ids, semantic_hash = _validate_member_manifest(
        member_payload, expected_sha256=_sha256_bytes(member_payload)
    )
    source_inventory, source_tree_sha = _directory_manifest(source_panel_root)
    sampled_inventory, sampled_tree_sha = _directory_manifest(sampled_panel_root)
    del source_inventory, sampled_inventory
    receipt = {
        "schema_version": MEMBER_MANIFEST_RECEIPT_SCHEMA_VERSION,
        "terminal_status": "completed_before_entropy",
        "built_before_entropy": True,
        "member_manifest_path": str(
            intended_member_manifest_path.expanduser().resolve()
        ),
        "member_manifest_sha256": _sha256_bytes(member_payload),
        "member_count": len(rows),
        "ordered_member_image_ids_sha256": _ordered_image_ids_sha256(validated_ids),
        "member_semantic_sha256": semantic_hash,
        **stage_zero_binding,
        "candidate_pool_path": str(candidate_pool_path.expanduser().resolve()),
        "candidate_pool_sha256": expected_candidate_pool_sha256,
        "census_path": census_selection.source_path,
        "census_sha256": census_selection.source_sha256,
        "frozen_contract_path": str(frozen_contract),
        "frozen_contract_sha256": expected_frozen_contract_sha256,
        "source_panel_root": str(source_panel_root.expanduser().resolve(strict=True)),
        "source_panel_manifest_set_sha256": source_tree_sha,
        "sampled_panel_root": str(sampled_panel_root.expanduser().resolve(strict=True)),
        "sampled_panel_manifest_set_sha256": sampled_tree_sha,
        "route_count": len(EXPECTED_ROUTE_IDS),
        "route_inventory_sha256": _route_inventory_sha256(),
        "adapter_inventory_sha256": _adapter_inventory_sha256(
            adapter, ordered_image_ids=image_ids
        ),
    }
    return MemberManifestArtifacts(member_payload, _json_bytes(receipt))


def _validate_member_manifest_receipt(
    payload: bytes,
    *,
    expected_sha256: str,
    member_manifest_jsonl: bytes,
    expected_member_manifest_sha256: str,
    expected_member_manifest_path: Path,
) -> Mapping[str, Any]:
    if _sha256_bytes(payload) != _require_sha256(
        expected_sha256, field="expected_member_manifest_receipt_sha256"
    ):
        raise ValueError("pre-entropy member receipt digest drift")
    receipt = _parse_json(payload, artifact="pre-entropy member receipt")
    _require_fields(
        receipt, _MEMBER_RECEIPT_FIELDS, artifact="pre-entropy member receipt"
    )
    rows, image_ids, semantic_hash = _validate_member_manifest(
        member_manifest_jsonl, expected_sha256=expected_member_manifest_sha256
    )
    manifest_path = expected_member_manifest_path.expanduser().resolve(strict=True)
    if (
        receipt["schema_version"] != MEMBER_MANIFEST_RECEIPT_SCHEMA_VERSION
        or receipt["terminal_status"] != "completed_before_entropy"
        or receipt["built_before_entropy"] is not True
        or Path(str(receipt["member_manifest_path"])).expanduser().resolve(strict=True)
        != manifest_path
        or manifest_path.read_bytes() != member_manifest_jsonl
        or receipt["member_manifest_sha256"] != expected_member_manifest_sha256
        or receipt["member_count"] != len(rows)
        or receipt["ordered_member_image_ids_sha256"]
        != _ordered_image_ids_sha256(image_ids)
        or receipt["member_semantic_sha256"] != semantic_hash
        or receipt["route_count"] != len(EXPECTED_ROUTE_IDS)
        or receipt["route_inventory_sha256"] != _route_inventory_sha256()
    ):
        raise ValueError("pre-entropy member receipt terminal binding drift")
    possible_image_ids, stage_zero_binding = _load_stage_zero_possible_pool(
        stage_zero_root=Path(str(receipt["stage_zero_root"])),
        expected_stage_zero_root_inventory_sha256=str(
            receipt["stage_zero_root_inventory_sha256"]
        ),
        stage_zero_receipt_path=Path(str(receipt["stage_zero_receipt_path"])),
        expected_stage_zero_receipt_sha256=str(
            receipt["stage_zero_receipt_sha256"]
        ),
        stage_zero_audit_path=Path(str(receipt["stage_zero_audit_path"])),
        expected_stage_zero_audit_sha256=str(receipt["stage_zero_audit_sha256"]),
        possible_pool_path=Path(str(receipt["stage_zero_possible_pool_path"])),
        expected_possible_pool_sha256=str(
            receipt["stage_zero_possible_pool_sha256"]
        ),
    )
    if possible_image_ids != image_ids or any(
        receipt[field] != expected
        for field, expected in stage_zero_binding.items()
    ):
        raise ValueError("pre-entropy member Stage Zero binding drift")
    for path_field, hash_field, kind in (
        ("candidate_pool_path", "candidate_pool_sha256", "file"),
        ("census_path", "census_sha256", "file"),
        ("frozen_contract_path", "frozen_contract_sha256", "file"),
        ("source_panel_root", "source_panel_manifest_set_sha256", "manifest_set"),
        ("sampled_panel_root", "sampled_panel_manifest_set_sha256", "manifest_set"),
    ):
        path = Path(str(receipt[path_field])).expanduser().resolve(strict=True)
        digest = _require_sha256(receipt[hash_field], field=hash_field)
        observed = (
            _sha256_file(path) if kind == "file" else _directory_manifest(path)[1]
        )
        if observed != digest:
            raise ValueError(f"pre-entropy member receipt {path_field} drift")
    _require_sha256(
        receipt["adapter_inventory_sha256"], field="adapter_inventory_sha256"
    )
    return receipt


def _fraction_decimal(value: Fraction) -> str:
    with localcontext() as context:
        context.prec = 80
        return format(Decimal(value.numerator) / Decimal(value.denominator), "f")


def _expected_hypergeometric_design(population_size: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample_size in (LOOK_ONE_COUNT, SELECTION_COUNT):
        cutoff, probability = _exact_hypergeometric_cutoff(
            population_size=population_size,
            success_count=NULL_SUCCESS_COUNT,
            sample_size=sample_size,
        )
        rows.append(
            {
                "cumulative_sample_size": sample_size,
                "largest_rejection_success_count": cutoff,
                "boundary_lower_tail_probability": {
                    "numerator": probability.numerator,
                    "denominator": probability.denominator,
                    "decimal": _fraction_decimal(probability),
                },
            }
        )
    return rows


def _validate_frozen_source_manifest(
    value: Any, *, selection_root: Path
) -> str:
    if not isinstance(value, Mapping):
        raise ValueError("journal frozen-source manifest is invalid")
    _require_fields(
        value,
        {
            "schema_version",
            "expected_named_labels",
            "ordered_labels",
            "entries",
            "entries_sha256",
        },
        artifact="journal frozen-source manifest",
    )
    expected_named = list(EXPECTED_NAMED_FROZEN_SOURCE_LABELS)
    ordered_labels = sorted(
        [*EXPECTED_NAMED_FROZEN_SOURCE_LABELS, *AUTO_FROZEN_SOURCE_LABELS]
    )
    entries = value["entries"]
    if (
        value["schema_version"] != FROZEN_SOURCE_MANIFEST_SCHEMA_VERSION
        or value["expected_named_labels"] != expected_named
        or value["ordered_labels"] != ordered_labels
        or not isinstance(entries, list)
        or len(entries) != len(ordered_labels)
    ):
        raise ValueError("journal frozen-source manifest schema/labels drift")
    observed_labels: list[str] = []
    observed_paths: list[Path] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("journal frozen-source entry is invalid")
        _require_fields(
            entry,
            {"label", "path", "bytes", "sha256", "snapshot_relative_path"},
            artifact="journal frozen-source entry",
        )
        label = entry["label"]
        if not isinstance(label, str):
            raise ValueError("journal frozen-source label is invalid")
        path = Path(str(entry["path"])).expanduser().resolve(strict=True)
        snapshot_relative = entry["snapshot_relative_path"]
        if snapshot_relative != f"source-snapshots/{label}":
            raise ValueError("journal frozen-source snapshot path drift")
        snapshot = (selection_root / str(snapshot_relative)).resolve(strict=True)
        size = entry["bytes"]
        digest = _require_sha256(
            entry["sha256"], field=f"journal frozen source {label}"
        )
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not path.is_file()
            or not snapshot.is_file()
            or path.stat().st_size != size
            or snapshot.stat().st_size != size
            or _sha256_file(path) != digest
            or _sha256_file(snapshot) != digest
        ):
            raise ValueError("journal frozen-source path/bytes/hash drift")
        observed_labels.append(label)
        observed_paths.append(path)
    entries_sha256 = _sha256_bytes(canonical_json_text(entries).encode("utf-8"))
    if (
        observed_labels != ordered_labels
        or len(set(observed_paths)) != len(observed_paths)
        or value["entries_sha256"] != entries_sha256
    ):
        raise ValueError("journal frozen-source inventory/order drift")
    return _sha256_bytes(canonical_json_text(value).encode("utf-8"))


def _validate_entropy_journal(*, selection_root: Path) -> ValidatedEntropyJournal:
    journal_root = selection_root.with_name(
        f"{selection_root.name}.entropy-journal-v1"
    ).resolve(strict=True)
    canonical_root = selection_root.with_name(
        f"{selection_root.name}.entropy-journal-v1"
    )
    if journal_root != canonical_root or journal_root.is_symlink():
        raise ValueError("selection entropy journal root is not canonical")
    expected_names = {
        "claim.json",
        "entropy.bin",
        "entropy-receipt.json",
        "terminal.json",
    }
    if (journal_root.stat().st_mode & 0o777) != 0o555:
        raise ValueError("selection entropy journal root is not immutable")
    observed_names: set[str] = set()
    for path in journal_root.iterdir():
        if path.is_symlink() or not path.is_file():
            raise ValueError("selection entropy journal contains a non-file")
        if (path.stat().st_mode & 0o777) != 0o444:
            raise ValueError("selection entropy journal file is not immutable")
        observed_names.add(path.name)
    if observed_names != expected_names:
        raise ValueError("selection entropy journal inventory is not exact")

    claim_path = journal_root / "claim.json"
    claim_payload = claim_path.read_bytes()
    claim_file_sha256 = _sha256_bytes(claim_payload)
    claim = _parse_pretty_json(claim_payload, artifact="entropy journal claim")
    claim_semantic_sha256 = _sha256_bytes(
        canonical_json_text(claim).encode("utf-8")
    )

    entropy_bytes = (journal_root / "entropy.bin").read_bytes()
    entropy_sha256 = _sha256_bytes(entropy_bytes)
    if len(entropy_bytes) != ENTROPY_BYTE_COUNT:
        raise ValueError("selection entropy journal does not contain exactly 64 bytes")
    entropy_receipt = _parse_pretty_json(
        (journal_root / "entropy-receipt.json").read_bytes(),
        artifact="entropy journal byte receipt",
    )
    _require_fields(
        entropy_receipt,
        {"byte_count", "sha256"},
        artifact="entropy journal byte receipt",
    )
    if entropy_receipt != {
        "byte_count": ENTROPY_BYTE_COUNT,
        "sha256": entropy_sha256,
    }:
        raise ValueError("selection entropy byte receipt drift")
    terminal_path = journal_root / "terminal.json"
    terminal = _parse_pretty_json(
        terminal_path.read_bytes(), artifact="entropy journal terminal"
    )
    _require_fields(
        terminal,
        {"schema_version", "terminal_status", "claim_sha256", "entropy_sha256", "reason"},
        artifact="entropy journal terminal",
    )
    if terminal != {
        "schema_version": ENTROPY_TERMINAL_SCHEMA_VERSION,
        "terminal_status": "completed",
        "claim_sha256": claim_semantic_sha256,
        "entropy_sha256": entropy_sha256,
        "reason": None,
    }:
        raise ValueError("selection entropy terminal binding drift")
    return ValidatedEntropyJournal(
        root=journal_root,
        terminal_path=terminal_path,
        claim=claim,
        claim_semantic_sha256=claim_semantic_sha256,
        claim_file_sha256=claim_file_sha256,
        entropy_bytes=entropy_bytes,
        entropy_sha256=entropy_sha256,
    )


def _selector_source_entry(
    manifest: Mapping[str, Any], *, label: str
) -> Mapping[str, Any]:
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise ValueError("selector frozen-source entries are invalid")
    matches = [
        entry
        for entry in entries
        if isinstance(entry, Mapping) and entry.get("label") == label
    ]
    if len(matches) != 1:
        raise ValueError(f"selector frozen-source manifest lacks one {label}")
    return matches[0]


def _validate_selector_stage_binding(
    value: Any,
    *,
    member_receipt: Mapping[str, Any],
) -> tuple[tuple[str, ...], Mapping[str, Any]]:
    if not isinstance(value, Mapping):
        raise ValueError("selector Stage Zero binding is invalid")
    fields = {
        "root",
        "receipt_path",
        "receipt_sha256",
        "root_inventory",
        "root_inventory_sha256",
        "possible_pool_path",
        "possible_pool_sha256",
        "possible_pool_count",
        "impossible_pool_count",
        "ordered_possible_image_ids_sha256",
        "audit_path",
        "audit_sha256",
        "audit_verdict",
        "audit_receipt",
    }
    _require_fields(value, fields, artifact="selector Stage Zero binding")
    possible_ids, receipt_binding = _load_stage_zero_possible_pool(
        stage_zero_root=Path(str(value["root"])),
        expected_stage_zero_root_inventory_sha256=str(
            value["root_inventory_sha256"]
        ),
        stage_zero_receipt_path=Path(str(value["receipt_path"])),
        expected_stage_zero_receipt_sha256=str(value["receipt_sha256"]),
        stage_zero_audit_path=Path(str(value["audit_path"])),
        expected_stage_zero_audit_sha256=str(value["audit_sha256"]),
        possible_pool_path=Path(str(value["possible_pool_path"])),
        expected_possible_pool_sha256=str(value["possible_pool_sha256"]),
    )
    root = Path(str(value["root"])).expanduser().resolve(strict=True)
    inventory, inventory_sha256 = _root_inventory(root)
    impossible_pool = _parse_pretty_json(
        (root / "impossible-pool.json").read_bytes(),
        artifact="Stage Zero impossible pool",
    )
    audit_path = Path(str(value["audit_path"])).expanduser().resolve(strict=True)
    audit = _parse_pretty_json(
        audit_path.read_bytes(), artifact="Stage Zero independent audit"
    )
    expected = {
        "root": str(root),
        "receipt_path": str(root / "receipt.json"),
        "receipt_sha256": receipt_binding["stage_zero_receipt_sha256"],
        "root_inventory": inventory,
        "root_inventory_sha256": inventory_sha256,
        "possible_pool_path": str(root / "possible-pool.json"),
        "possible_pool_sha256": receipt_binding[
            "stage_zero_possible_pool_sha256"
        ],
        "possible_pool_count": len(possible_ids),
        "impossible_pool_count": impossible_pool["count"],
        "ordered_possible_image_ids_sha256": _ordered_image_ids_sha256(
            possible_ids
        ),
        "audit_path": str(audit_path),
        "audit_sha256": receipt_binding["stage_zero_audit_sha256"],
        "audit_verdict": "approved",
        "audit_receipt": audit,
    }
    if dict(value) != expected:
        raise ValueError("selector Stage Zero exact binding drift")
    for selector_field, receipt_field in (
        ("root", "stage_zero_root"),
        ("root_inventory_sha256", "stage_zero_root_inventory_sha256"),
        ("receipt_path", "stage_zero_receipt_path"),
        ("receipt_sha256", "stage_zero_receipt_sha256"),
        ("audit_path", "stage_zero_audit_path"),
        ("audit_sha256", "stage_zero_audit_sha256"),
        ("audit_verdict", "stage_zero_audit_verdict"),
        ("possible_pool_path", "stage_zero_possible_pool_path"),
        ("possible_pool_sha256", "stage_zero_possible_pool_sha256"),
        ("possible_pool_count", "stage_zero_possible_pool_count"),
        (
            "ordered_possible_image_ids_sha256",
            "stage_zero_ordered_possible_image_ids_sha256",
        ),
    ):
        if value[selector_field] != member_receipt[receipt_field]:
            raise ValueError("selector and pre-entropy Stage Zero bindings differ")
    return possible_ids, expected


def _validate_selector_member_manifest(
    value: Any,
    *,
    ordered_pool: Sequence[str],
    preentropy_rows: Sequence[Mapping[str, Any]],
    member_receipt: Mapping[str, Any],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("selector possible-pool member manifest is invalid")
    _require_fields(
        value,
        {
            "schema_version",
            "population_scope",
            "count",
            "ordered_image_ids",
            "ordered_image_ids_sha256",
            "members",
            "members_sha256",
            "input_bindings",
            "input_bindings_sha256",
        },
        artifact="selector possible-pool member manifest",
    )
    expected_ids = list(ordered_pool)
    members = value["members"]
    member_fields = {
        "image_id",
        "canonical_source_path",
        "source_image_bytes",
        "source_image_sha256",
        "source_image_width",
        "source_image_height",
        "candidate_pool_record_sha256",
        "census_record_sha256",
        "official_owner_record_sha256",
        "route_replay_input_record_sha256",
        "route_count",
        "route_inventory_sha256",
    }
    if (
        value["schema_version"] != SELECTOR_MEMBER_MANIFEST_SCHEMA_VERSION
        or value["population_scope"] != STAGE_ZERO_POPULATION_SCOPE
        or value["count"] != len(expected_ids)
        or value["ordered_image_ids"] != expected_ids
        or value["ordered_image_ids_sha256"]
        != _ordered_image_ids_sha256(expected_ids)
        or not isinstance(members, list)
        or len(members) != len(preentropy_rows)
    ):
        raise ValueError("selector possible-pool member header drift")
    for expected_id, raw_member, preentropy in zip(
        expected_ids, members, preentropy_rows, strict=True
    ):
        if not isinstance(raw_member, Mapping):
            raise ValueError("selector possible-pool member row is invalid")
        _require_fields(
            raw_member,
            member_fields,
            artifact=f"selector possible-pool member {expected_id}",
        )
        expected_member = {
            "image_id": expected_id,
            "canonical_source_path": preentropy["source_image_path"],
            "source_image_bytes": preentropy["source_image_size_bytes"],
            "source_image_sha256": preentropy["source_image_sha256"],
            "source_image_width": preentropy["source_image_width"],
            "source_image_height": preentropy["source_image_height"],
            "candidate_pool_record_sha256": preentropy[
                "candidate_pool_record_sha256"
            ],
            "census_record_sha256": preentropy["census_record_sha256"],
            "official_owner_record_sha256": preentropy[
                "official_owner_record_sha256"
            ],
            "route_replay_input_record_sha256": preentropy["replay_input_sha256"],
            "route_count": len(EXPECTED_ROUTE_IDS),
            "route_inventory_sha256": _route_inventory_sha256(),
        }
        if dict(raw_member) != expected_member:
            raise ValueError("selector and pre-entropy member rows differ")
    if value["members_sha256"] != _sha256_bytes(
        canonical_json_text(members).encode("utf-8")
    ):
        raise ValueError("selector possible-pool member record hash drift")

    bindings = value["input_bindings"]
    if not isinstance(bindings, Mapping):
        raise ValueError("selector member input bindings are invalid")
    binding_fields = {
        "candidate_pool_path",
        "candidate_pool_bytes",
        "candidate_pool_sha256",
        "census_path",
        "census_bytes",
        "census_sha256",
        "sampled_manifest_binding_sha256",
        "source_manifest_binding_sha256",
        "execution_model_identity_sha256",
        "tokenizer_identity_sha256",
        "stage_zero_receipt_sha256",
    }
    _require_fields(bindings, binding_fields, artifact="selector member inputs")
    stage_receipt = _parse_pretty_json(
        Path(str(member_receipt["stage_zero_receipt_path"])).read_bytes(),
        artifact="Stage Zero receipt",
    )
    stage_inputs = stage_receipt.get("inputs")
    if not isinstance(stage_inputs, Mapping):
        raise ValueError("Stage Zero receipt lacks member input bindings")
    sampled_binding = stage_inputs.get("sampled_manifest_binding")
    source_binding = stage_inputs.get("source_manifest_binding")
    if not isinstance(sampled_binding, Mapping) or not isinstance(
        source_binding, Mapping
    ):
        raise ValueError("Stage Zero panel manifest bindings are invalid")
    execution_identity = _require_sha256(
        stage_inputs.get("execution_model_identity_sha256"),
        field="Stage Zero execution model identity",
    )
    tokenizer_identity = _require_sha256(
        stage_inputs.get("tokenizer_identity_sha256"),
        field="Stage Zero tokenizer identity",
    )
    candidate_path = Path(str(member_receipt["candidate_pool_path"])).resolve(
        strict=True
    )
    census_path = Path(str(member_receipt["census_path"])).resolve(strict=True)
    expected_bindings = {
        "candidate_pool_path": str(candidate_path),
        "candidate_pool_bytes": candidate_path.stat().st_size,
        "candidate_pool_sha256": member_receipt["candidate_pool_sha256"],
        "census_path": str(census_path),
        "census_bytes": census_path.stat().st_size,
        "census_sha256": member_receipt["census_sha256"],
        "sampled_manifest_binding_sha256": _sha256_bytes(
            canonical_json_text(sampled_binding).encode("utf-8")
        ),
        "source_manifest_binding_sha256": _sha256_bytes(
            canonical_json_text(source_binding).encode("utf-8")
        ),
        "execution_model_identity_sha256": execution_identity,
        "tokenizer_identity_sha256": tokenizer_identity,
        "stage_zero_receipt_sha256": member_receipt[
            "stage_zero_receipt_sha256"
        ],
    }
    if dict(bindings) != expected_bindings:
        raise ValueError("selector member input binding drift")
    if value["input_bindings_sha256"] != _sha256_bytes(
        canonical_json_text(bindings).encode("utf-8")
    ):
        raise ValueError("selector member input binding hash drift")
    return dict(value)


def _validate_entropy_claim(
    claim: Mapping[str, Any],
    *,
    selection_root: Path,
    source_manifest: Mapping[str, Any],
    source_manifest_sha256: str,
    stage_binding: Mapping[str, Any],
    population_size: int,
    ordered_pool_sha256: str,
    selector_member_manifest: Mapping[str, Any],
) -> None:
    fields = {
        "schema_version",
        "output_root",
        "selector",
        "unit_contract",
        "contract_review",
        "frozen_source_manifest",
        "frozen_source_manifest_sha256",
        "stage_zero",
        "stage_zero_binding_sha256",
        "population_scope",
        "population_size_N",
        "null_success_count_K",
        "per_look_alpha",
        "cumulative_sample_sizes",
        "hypergeometric_design",
        "ordered_possible_pool_sha256",
        "sample_size",
        "member_manifest_sha256",
        "member_records_sha256",
        "member_input_bindings_sha256",
    }
    _require_fields(claim, fields, artifact="entropy journal claim")
    if _sha256_file(FROZEN_SELECTOR_PATH) != FROZEN_SELECTOR_SHA256:
        raise ValueError("frozen selector implementation digest drift")
    selector_entry = _selector_source_entry(source_manifest, label="selector")
    unit_entry = _selector_source_entry(source_manifest, label="unit_contract")
    contract_entry = _selector_source_entry(source_manifest, label="contract_review")
    selector_binding = {
        "path": str(FROZEN_SELECTOR_PATH.resolve(strict=True)),
        "sha256": FROZEN_SELECTOR_SHA256,
    }
    if selector_entry["path"] != selector_binding["path"] or selector_entry[
        "sha256"
    ] != selector_binding["sha256"]:
        raise ValueError("frozen selector source-manifest binding drift")
    expected = {
        "schema_version": ENTROPY_CLAIM_SCHEMA_VERSION,
        "output_root": str(selection_root),
        "selector": selector_binding,
        "unit_contract": {
            "path": unit_entry["path"],
            "sha256": unit_entry["sha256"],
        },
        "contract_review": {
            "path": contract_entry["path"],
            "sha256": contract_entry["sha256"],
        },
        "frozen_source_manifest": copy.deepcopy(dict(source_manifest)),
        "frozen_source_manifest_sha256": source_manifest_sha256,
        "stage_zero": copy.deepcopy(dict(stage_binding)),
        "stage_zero_binding_sha256": _sha256_bytes(
            canonical_json_text(stage_binding).encode("utf-8")
        ),
        "population_scope": SELECTION_POPULATION_SCOPE,
        "population_size_N": population_size,
        "null_success_count_K": NULL_SUCCESS_COUNT,
        "per_look_alpha": {
            "numerator": PER_LOOK_ALPHA.numerator,
            "denominator": PER_LOOK_ALPHA.denominator,
            "decimal": _fraction_decimal(PER_LOOK_ALPHA),
        },
        "cumulative_sample_sizes": [LOOK_ONE_COUNT, SELECTION_COUNT],
        "hypergeometric_design": _expected_hypergeometric_design(population_size),
        "ordered_possible_pool_sha256": ordered_pool_sha256,
        "sample_size": SELECTION_COUNT,
        "member_manifest_sha256": _sha256_bytes(
            canonical_json_text(selector_member_manifest).encode("utf-8")
        ),
        "member_records_sha256": selector_member_manifest["members_sha256"],
        "member_input_bindings_sha256": selector_member_manifest[
            "input_bindings_sha256"
        ],
    }
    if dict(claim) != expected:
        raise ValueError("entropy journal pre-entropy claim binding drift")


def _validate_completed_selection_root(
    *,
    selection_root: Path,
    selection: Mapping[str, Any],
    selection_payload: bytes,
    journal: ValidatedEntropyJournal,
    source_manifest: Mapping[str, Any],
    selector_member_manifest: Mapping[str, Any],
) -> None:
    if selection_root.is_symlink() or (selection_root.stat().st_mode & 0o777) != 0o555:
        raise ValueError("completed selector root is not immutable")
    for path in selection_root.rglob("*"):
        if path.is_symlink():
            raise ValueError("completed selector root contains a symlink")
        expected_mode = 0o555 if path.is_dir() else 0o444
        if (path.stat().st_mode & 0o777) != expected_mode:
            raise ValueError("completed selector tree is not immutable")
    if _parse_pretty_json(
        (selection_root / "selection.json").read_bytes(), artifact="selection"
    ) != selection or (selection_root / "selection.json").read_bytes() != selection_payload:
        raise ValueError("completed selector selection snapshot drift")
    if _parse_pretty_json(
        (selection_root / "frozen-source-manifest.json").read_bytes(),
        artifact="frozen source manifest",
    ) != source_manifest:
        raise ValueError("completed selector source manifest snapshot drift")
    if _parse_pretty_json(
        (selection_root / "possible-pool-member-manifest.json").read_bytes(),
        artifact="selector member manifest",
    ) != selector_member_manifest:
        raise ValueError("completed selector member manifest snapshot drift")
    inventory, _ = _root_inventory(selection_root)
    receipt_identity = inventory.pop("receipt.json", None)
    expected_paths = {
        "selection.json",
        "frozen-source-manifest.json",
        "possible-pool-member-manifest.json",
        *(
            str(entry["snapshot_relative_path"])
            for entry in source_manifest["entries"]
        ),
    }
    if receipt_identity is None or set(inventory) != expected_paths:
        raise ValueError("completed selector root inventory drift")
    receipt = _parse_pretty_json(
        (selection_root / "receipt.json").read_bytes(), artifact="selection receipt"
    )
    source_manifest_sha256 = _sha256_bytes(
        canonical_json_text(source_manifest).encode("utf-8")
    )
    member_manifest_sha256 = _sha256_bytes(
        canonical_json_text(selector_member_manifest).encode("utf-8")
    )
    stage_binding = selection["stage_zero_binding"]
    replay_validation = selection["stage_zero_replay_validation"]
    randomization = {
        "population_size_N": selection["population_size_N"],
        "null_success_count_K": selection["null_success_count_K"],
        "per_look_alpha": copy.deepcopy(selection["per_look_alpha"]),
        "cumulative_sample_sizes": copy.deepcopy(
            selection["cumulative_sample_sizes"]
        ),
        "sample_size": selection["sample_size"],
        "ordered_possible_pool_sha256": selection[
            "ordered_possible_pool_sha256"
        ],
        "ordered_sample_sha256": selection["ordered_sample_sha256"],
        "look_one_image_ids": copy.deepcopy(selection["look_one_image_ids"]),
        "look_two_additional_image_ids": copy.deepcopy(
            selection["look_two_additional_image_ids"]
        ),
        "look_two_cumulative_image_ids": copy.deepcopy(
            selection["look_two_cumulative_image_ids"]
        ),
        "hypergeometric_design": copy.deepcopy(selection["hypergeometric_design"]),
    }
    bindings = {
        "selector_sha256": FROZEN_SELECTOR_SHA256,
        "unit_contract_sha256": _selector_source_entry(
            source_manifest, label="unit_contract"
        )["sha256"],
        "contract_review_sha256": _selector_source_entry(
            source_manifest, label="contract_review"
        )["sha256"],
        "stage_zero_producer_sha256": _selector_source_entry(
            source_manifest, label="stage_zero_producer"
        )["sha256"],
        "frozen_source_manifest_sha256": source_manifest_sha256,
        "stage_zero_binding_sha256": selection["stage_zero_binding_sha256"],
        "stage_zero_receipt_sha256": stage_binding["receipt_sha256"],
        "stage_zero_root_inventory_sha256": stage_binding[
            "root_inventory_sha256"
        ],
        "stage_zero_independent_audit_sha256": stage_binding["audit_sha256"],
        "possible_pool_member_manifest_sha256": member_manifest_sha256,
        "possible_pool_member_records_sha256": selector_member_manifest[
            "members_sha256"
        ],
        "member_input_bindings_sha256": selector_member_manifest[
            "input_bindings_sha256"
        ],
    }
    expected_receipt = {
        "schema_version": SELECTION_RECEIPT_SCHEMA_VERSION,
        "terminal_status": "completed",
        "output_root": str(selection_root),
        "journal_root": str(journal.root),
        "journal_claim_file_sha256": journal.claim_file_sha256,
        "journal_claim_sha256": journal.claim_semantic_sha256,
        "entropy": copy.deepcopy(selection["entropy"]),
        "randomization": randomization,
        "bindings": bindings,
        "frozen_source_manifest": copy.deepcopy(dict(source_manifest)),
        "stage_zero_binding": copy.deepcopy(dict(stage_binding)),
        "possible_pool_member_manifest": copy.deepcopy(
            dict(selector_member_manifest)
        ),
        "replay_validation": copy.deepcopy(dict(replay_validation)),
        "outputs": inventory,
        "publication_contract": {
            "method": "same_parent_hidden_staging_then_os_replace",
            "completed_file_mode": "0444",
            "completed_directory_mode": "0555",
            "hidden_until_terminal_readback": True,
            "invalid_post_rename_root_quarantined": True,
            "exact_inventory_hash_and_size_checked": True,
        },
    }
    if receipt != expected_receipt:
        raise ValueError("completed selector receipt binding drift")


def _validate_selection(
    payload: bytes,
    *,
    expected_sha256: str,
    selection_path: Path,
    member_manifest_jsonl: bytes,
    expected_member_manifest_sha256: str,
    member_manifest_receipt_json: bytes,
    expected_member_manifest_receipt_sha256: str,
    expected_packet_sha256: str | None = None,
    expected_ontology_sha256: str | None = None,
) -> ValidatedSelection:
    """Validate exact frozen-selector output and its pre-entropy chronology."""

    selection_sha256 = _require_sha256(
        expected_sha256, field="expected_selection_sha256"
    )
    if _sha256_bytes(payload) != selection_sha256:
        raise ValueError("selection artifact digest drift")
    resolved_selection = selection_path.expanduser().resolve(strict=True)
    if (
        resolved_selection.name != "selection.json"
        or resolved_selection.read_bytes() != payload
    ):
        raise ValueError("selection path/readback drift")
    selection_root = resolved_selection.parent
    value = _parse_pretty_json(payload, artifact="selection artifact")
    _require_fields(value, _SELECTION_FIELDS, artifact="selection artifact")

    population_size = value["population_size_N"]
    expected_alpha = {
        "numerator": PER_LOOK_ALPHA.numerator,
        "denominator": PER_LOOK_ALPHA.denominator,
        "decimal": _fraction_decimal(PER_LOOK_ALPHA),
    }
    if (
        value["schema_version"] != SELECTION_SCHEMA_VERSION
        or value["terminal_status"] != "completed"
        or value["population_scope"] != SELECTION_POPULATION_SCOPE
        or isinstance(population_size, bool)
        or not isinstance(population_size, int)
        or population_size < NULL_SUCCESS_COUNT
        or value["null_success_count_K"] != NULL_SUCCESS_COUNT
        or value["per_look_alpha"] != expected_alpha
        or value["cumulative_sample_sizes"] != [LOOK_ONE_COUNT, SELECTION_COUNT]
        or value["sample_size"] != SELECTION_COUNT
        or value["hypergeometric_design"]
        != _expected_hypergeometric_design(population_size)
    ):
        raise ValueError("selection terminal design contract drift")

    preentropy_rows, ordered_pool, member_semantic_hash = _validate_member_manifest(
        member_manifest_jsonl, expected_sha256=expected_member_manifest_sha256
    )
    receipt_preview = _parse_json(
        member_manifest_receipt_json, artifact="pre-entropy member receipt"
    )
    member_path = Path(
        str(receipt_preview.get("member_manifest_path"))
    ).expanduser().resolve(strict=True)
    member_receipt = _validate_member_manifest_receipt(
        member_manifest_receipt_json,
        expected_sha256=expected_member_manifest_receipt_sha256,
        member_manifest_jsonl=member_manifest_jsonl,
        expected_member_manifest_sha256=expected_member_manifest_sha256,
        expected_member_manifest_path=member_path,
    )
    ordered_pool_sha256 = _ordered_image_ids_sha256(ordered_pool)
    if (
        len(ordered_pool) != population_size
        or value["ordered_possible_pool_sha256"] != ordered_pool_sha256
        or member_receipt["member_semantic_sha256"] != member_semantic_hash
    ):
        raise ValueError("selection population differs from pre-entropy members")

    stage_ids, stage_binding = _validate_selector_stage_binding(
        value["stage_zero_binding"], member_receipt=member_receipt
    )
    stage_binding_sha256 = _sha256_bytes(
        canonical_json_text(stage_binding).encode("utf-8")
    )
    if (
        stage_ids != ordered_pool
        or value["stage_zero_binding_sha256"] != stage_binding_sha256
    ):
        raise ValueError("selection Stage Zero semantic hash drift")

    selector_member_manifest = _validate_selector_member_manifest(
        value["possible_pool_member_manifest"],
        ordered_pool=ordered_pool,
        preentropy_rows=preentropy_rows,
        member_receipt=member_receipt,
    )
    selector_member_manifest_sha256 = _sha256_bytes(
        canonical_json_text(selector_member_manifest).encode("utf-8")
    )
    if (
        value["possible_pool_member_manifest_sha256"]
        != selector_member_manifest_sha256
        or value["possible_pool_member_records_sha256"]
        != selector_member_manifest["members_sha256"]
    ):
        raise ValueError("selection possible-pool member manifest hash drift")

    source_manifest = value["frozen_source_manifest"]
    source_manifest_sha256 = _validate_frozen_source_manifest(
        source_manifest, selection_root=selection_root
    )
    if value["frozen_source_manifest_sha256"] != source_manifest_sha256:
        raise ValueError("selection frozen-source manifest hash drift")

    journal = _validate_entropy_journal(selection_root=selection_root)
    _validate_entropy_claim(
        journal.claim,
        selection_root=selection_root,
        source_manifest=source_manifest,
        source_manifest_sha256=source_manifest_sha256,
        stage_binding=stage_binding,
        population_size=population_size,
        ordered_pool_sha256=ordered_pool_sha256,
        selector_member_manifest=selector_member_manifest,
    )

    entropy = value["entropy"]
    entropy_fields = {
        "byte_count",
        "bytes_hex",
        "sha256",
        "permutation_count_M",
        "entropy_integer_R",
        "acceptance_limit_L",
        "accepted",
        "accepted_initial_rank",
        "unranking_trace",
    }
    if not isinstance(entropy, Mapping):
        raise ValueError("selection entropy record is invalid")
    _require_fields(entropy, entropy_fields, artifact="selection entropy record")
    entropy_hex = entropy["bytes_hex"]
    if not isinstance(entropy_hex, str):
        raise ValueError("selection entropy hex is invalid")
    try:
        selected_entropy = bytes.fromhex(entropy_hex)
    except ValueError as exc:
        raise ValueError("selection entropy hex is invalid") from exc
    if entropy_hex != selected_entropy.hex():
        raise ValueError("selection entropy hex is not canonical")
    entropy_domain_size = 1 << (8 * ENTROPY_BYTE_COUNT)
    permutation_count = _falling_factorial(population_size, SELECTION_COUNT)
    entropy_integer = int.from_bytes(selected_entropy, "big", signed=False)
    acceptance_limit = (entropy_domain_size // permutation_count) * permutation_count
    initial_rank = entropy_integer % permutation_count
    for field in (
        "byte_count",
        "permutation_count_M",
        "entropy_integer_R",
        "acceptance_limit_L",
        "accepted_initial_rank",
    ):
        if isinstance(entropy[field], bool) or not isinstance(entropy[field], int):
            raise ValueError(f"selection entropy {field} is not an integer")
    if (
        entropy["byte_count"] != ENTROPY_BYTE_COUNT
        or selected_entropy != journal.entropy_bytes
        or entropy["sha256"] != journal.entropy_sha256
        or entropy["permutation_count_M"] != permutation_count
        or entropy["entropy_integer_R"] != entropy_integer
        or entropy["acceptance_limit_L"] != acceptance_limit
        or entropy_integer >= acceptance_limit
        or entropy["accepted"] is not True
        or entropy["accepted_initial_rank"] != initial_rank
    ):
        raise ValueError("selection accepted entropy/rejection binding drift")

    trace = entropy["unranking_trace"]
    if not isinstance(trace, list) or len(trace) != SELECTION_COUNT:
        raise ValueError("selection unranking trace must contain exactly 32 steps")
    trace_fields = {
        "position",
        "remaining_item_count_before",
        "remaining_positions_after",
        "suffix_count",
        "rank_before",
        "choice_index",
        "selected_image_id",
        "rank_after",
    }
    remaining = list(ordered_pool)
    rank = initial_rank
    unranked: list[str] = []
    for position, raw_step in enumerate(trace):
        if not isinstance(raw_step, Mapping):
            raise ValueError("selection unranking step is invalid")
        _require_fields(
            raw_step, trace_fields, artifact="selection unranking step"
        )
        remaining_positions = SELECTION_COUNT - position - 1
        suffix_count = _falling_factorial(
            len(remaining) - 1, remaining_positions
        )
        choice_index, next_rank = divmod(rank, suffix_count)
        if choice_index >= len(remaining):
            raise ValueError("selection unranking index exceeds remaining pool")
        selected_image_id = remaining.pop(choice_index)
        expected_step = {
            "position": position,
            "remaining_item_count_before": len(remaining) + 1,
            "remaining_positions_after": remaining_positions,
            "suffix_count": suffix_count,
            "rank_before": rank,
            "choice_index": choice_index,
            "selected_image_id": selected_image_id,
            "rank_after": next_rank,
        }
        if dict(raw_step) != expected_step:
            raise ValueError("selection lexicographic unranking trace drift")
        unranked.append(selected_image_id)
        rank = next_rank
    if rank != 0:
        raise ValueError("selection unranking did not terminate at zero")

    def image_list(field: str) -> tuple[str, ...]:
        raw = value[field]
        if not isinstance(raw, list):
            raise ValueError(f"selection {field} must be a list")
        return tuple(_canonical_image_id(item, field=field) for item in raw)

    look_one = image_list("look_one_image_ids")
    look_two_additional = image_list("look_two_additional_image_ids")
    look_two_cumulative = image_list("look_two_cumulative_image_ids")
    ordered = tuple(unranked)
    if (
        len(ordered) != SELECTION_COUNT
        or len(set(ordered)) != SELECTION_COUNT
        or look_one != ordered[:LOOK_ONE_COUNT]
        or look_two_additional != ordered[LOOK_ONE_COUNT:]
        or look_two_cumulative != ordered
        or value["ordered_sample_sha256"] != _ordered_image_ids_sha256(ordered)
    ):
        raise ValueError("selection ordered/look image partitions drift")

    design = value["hypergeometric_design"]
    if not isinstance(design, list) or design != _expected_hypergeometric_design(
        population_size
    ):
        raise ValueError("selection exact hypergeometric design drift")
    cutoff_by_look = {
        "look_one": int(design[0]["largest_rejection_success_count"]),
        "look_two_cumulative": int(
            design[1]["largest_rejection_success_count"]
        ),
    }
    replay_validation = value["stage_zero_replay_validation"]
    expected_replay_validation = {
        "complete_category_witness_and_certificate_replay": True,
        "independent_audit_sha256": stage_binding["audit_sha256"],
        "stage_zero_root_inventory_sha256": stage_binding[
            "root_inventory_sha256"
        ],
    }
    if (
        not isinstance(replay_validation, Mapping)
        or set(replay_validation) != set(expected_replay_validation)
        or dict(replay_validation) != expected_replay_validation
    ):
        raise ValueError("selection Stage Zero replay validation drift")

    _validate_completed_selection_root(
        selection_root=selection_root,
        selection=value,
        selection_payload=payload,
        journal=journal,
        source_manifest=source_manifest,
        selector_member_manifest=selector_member_manifest,
    )

    frozen_contract_path = Path(
        str(member_receipt["frozen_contract_path"])
    ).expanduser().resolve(strict=True)
    frozen_contract_sha256 = _require_sha256(
        member_receipt["frozen_contract_sha256"],
        field="frozen contract sha256",
    )
    contract = _parse_json(
        frozen_contract_path.read_bytes(), artifact="frozen pre-entropy contract"
    )
    packet_entry = _selector_source_entry(
        source_manifest, label="reviewer_instruction_packet"
    )
    ontology_entry = _selector_source_entry(
        source_manifest, label="ontology_state_artifact"
    )
    if (
        expected_packet_sha256 is not None
        and (
            contract.get("packet_sha256") != expected_packet_sha256
            or packet_entry["sha256"] != expected_packet_sha256
        )
    ) or (
        expected_ontology_sha256 is not None
        and (
            contract.get("ontology_sha256") != expected_ontology_sha256
            or ontology_entry["sha256"] != expected_ontology_sha256
        )
    ):
        raise ValueError("selection packet/ontology pre-entropy binding drift")

    source_artifacts = {
        "candidate_pool": {
            "path": member_receipt["candidate_pool_path"],
            "sha256": member_receipt["candidate_pool_sha256"],
            "kind": "file",
        },
        "census": {
            "path": member_receipt["census_path"],
            "sha256": member_receipt["census_sha256"],
            "kind": "file",
        },
        "source_panel": {
            "path": member_receipt["source_panel_root"],
            "sha256": member_receipt["source_panel_manifest_set_sha256"],
            "kind": "manifest_set",
        },
        "sampled_panel": {
            "path": member_receipt["sampled_panel_root"],
            "sha256": member_receipt["sampled_panel_manifest_set_sha256"],
            "kind": "manifest_set",
        },
    }
    return ValidatedSelection(
        document=value,
        selection_sha256=selection_sha256,
        selection_root=selection_root,
        journal_terminal_path=journal.terminal_path,
        ordered_pool_image_ids=ordered_pool,
        ordered_selected_image_ids=ordered,
        look_one_image_ids=look_one,
        look_two_additional_image_ids=look_two_additional,
        look_two_cumulative_image_ids=look_two_cumulative,
        cutoff_by_look=cutoff_by_look,
        member_manifest_path=member_path,
        member_manifest_sha256=expected_member_manifest_sha256,
        source_artifacts=source_artifacts,
        frozen_contract_path=frozen_contract_path,
        frozen_contract_sha256=frozen_contract_sha256,
    )


def freeze_preseed_contract(
    *,
    packet_path: Path,
    expected_packet_sha256: str,
    ontology_path: Path,
    expected_ontology_sha256: str,
    candidate_pool_path: Path,
    expected_candidate_pool_sha256: str,
    census_path: Path,
    expected_census_sha256: str,
    source_panel_root: Path,
    sampled_panel_root: Path,
    source_paths: Sequence[Path] = (),
) -> bytes:
    """Freeze packet, schema registry, immutable data, and implementation hashes."""

    paths = {
        "packet": packet_path.expanduser().resolve(strict=True),
        "ontology": ontology_path.expanduser().resolve(strict=True),
        "candidate_pool": candidate_pool_path.expanduser().resolve(strict=True),
        "census": census_path.expanduser().resolve(strict=True),
        "review_assembler": Path(__file__).resolve(strict=True),
        "census_analyzer": Path(census.__file__).resolve(strict=True),
        "global_matcher": Path(match_prefix.__code__.co_filename).resolve(strict=True),
    }
    for index, path in enumerate(source_paths):
        paths[f"additional_source_{index:02d}"] = path.expanduser().resolve(strict=True)
    expected = {
        "packet": _require_sha256(
            expected_packet_sha256, field="expected_packet_sha256"
        ),
        "ontology": _require_sha256(
            expected_ontology_sha256, field="expected_ontology_sha256"
        ),
        "candidate_pool": _require_sha256(
            expected_candidate_pool_sha256, field="expected_candidate_pool_sha256"
        ),
        "census": _require_sha256(
            expected_census_sha256, field="expected_census_sha256"
        ),
    }
    observed = {name: _sha256_file(path) for name, path in paths.items()}
    for name, digest in expected.items():
        if observed[name] != digest:
            raise ValueError(f"{name} digest drift")
    _validate_ontology(paths["ontology"], expected_sha256=expected_ontology_sha256)
    source_root = source_panel_root.expanduser().resolve(strict=True)
    sampled_root = sampled_panel_root.expanduser().resolve(strict=True)
    source_tree_sha = _directory_manifest(source_root)[1]
    sampled_tree_sha = _directory_manifest(sampled_root)[1]
    source_bindings = {
        name: {"path": str(paths[name]), "sha256": observed[name], "kind": "file"}
        for name in sorted(paths)
    }
    source_bindings.update(
        {
            "source_panel": {
                "path": str(source_root),
                "sha256": source_tree_sha,
                "kind": "manifest_set",
            },
            "sampled_panel": {
                "path": str(sampled_root),
                "sha256": sampled_tree_sha,
                "kind": "manifest_set",
            },
        }
    )
    seal = {
        "schema_version": CONTRACT_SEAL_SCHEMA_VERSION,
        "packet_id": PACKET_ID,
        "packet_sha256": observed["packet"],
        "ontology_id": ONTOLOGY_ID,
        "ontology_sha256": observed["ontology"],
        "category_namespace_sha256": COCO_80_CATEGORY_NAMESPACE_SHA256,
        "schema_registry": copy.deepcopy(SCHEMA_REGISTRY),
        "schema_registry_sha256": _sha256_bytes(
            canonical_json_text(SCHEMA_REGISTRY).encode()
        ),
        "source_bindings": source_bindings,
        "seed_generated": False,
        "selection_materialized": False,
    }
    return _json_bytes(seal)


def build_review_queue(
    *,
    selection_json: bytes,
    expected_selection_sha256: str,
    selection_path: Path,
    member_manifest_jsonl: bytes,
    expected_member_manifest_sha256: str,
    member_manifest_receipt_json: bytes,
    expected_member_manifest_receipt_sha256: str,
    adapter: Mapping[str, Any],
    candidate_pool_path: Path,
    expected_candidate_pool_sha256: str,
    census_path: Path,
    expected_census_path: Path,
    expected_census_sha256: str,
    packet_path: Path,
    expected_packet_sha256: str,
    ontology_path: Path,
    expected_ontology_sha256: str,
    look_id: str,
    prior_look_decision_json: bytes | None = None,
    prior_look_receipt_json: bytes | None = None,
) -> ReviewQueueArtifacts:
    """Build the reviewer-visible queue and private immutable official ledger."""

    if look_id not in LOOK_IDS:
        raise ValueError("unknown review look")
    selection = _validate_selection(
        selection_json,
        expected_sha256=expected_selection_sha256,
        selection_path=selection_path,
        member_manifest_jsonl=member_manifest_jsonl,
        expected_member_manifest_sha256=expected_member_manifest_sha256,
        member_manifest_receipt_json=member_manifest_receipt_json,
        expected_member_manifest_receipt_sha256=expected_member_manifest_receipt_sha256,
        expected_packet_sha256=expected_packet_sha256,
        expected_ontology_sha256=expected_ontology_sha256,
    )
    selected_in_sample_order = (
        selection.look_one_image_ids
        if look_id == "look_one"
        else selection.look_two_additional_image_ids
    )
    if len(selected_in_sample_order) != LOOK_ONE_COUNT:
        raise ValueError("review look must contain exactly sixteen images")
    prior_receipt_sha256: str | None = None
    if look_id == "look_one":
        if prior_look_decision_json is not None or prior_look_receipt_json is not None:
            raise ValueError("look one cannot carry a prior-look artifact")
    else:
        prior_receipt_sha256 = _validate_look_two_authorization(
            prior_look_decision_json=prior_look_decision_json,
            prior_look_receipt_json=prior_look_receipt_json,
            expected_selection_sha256=expected_selection_sha256,
            expected_member_manifest_sha256=expected_member_manifest_sha256,
        )
    pool = _load_candidate_pool(
        candidate_pool_path, expected_sha256=expected_candidate_pool_sha256
    )
    if (
        expected_candidate_pool_sha256
        != selection.source_artifacts["candidate_pool"]["sha256"]
    ):
        raise ValueError("queue candidate pool differs from selection binding")
    if not set(selection.ordered_pool_image_ids).issubset(pool):
        raise ValueError("queue member set is not a candidate-pool subset")
    census_selection = load_frozen_census_records(
        census_path=census_path,
        expected_census_path=expected_census_path,
        expected_census_sha256=expected_census_sha256,
        selected_image_ids=selected_in_sample_order,
    )
    if (
        census_selection.source_sha256
        != selection.source_artifacts["census"]["sha256"]
    ):
        raise ValueError("queue census differs from selection binding")
    _validate_adapter_inventory(adapter, ordered_image_ids=selected_in_sample_order)
    member_rows, member_ids, _ = _validate_member_manifest(
        member_manifest_jsonl, expected_sha256=expected_member_manifest_sha256
    )
    if member_ids != selection.ordered_pool_image_ids:
        raise ValueError("queue member manifest order differs from selection")
    member_by_image = {str(row["image_id"]): row for row in member_rows}
    packet = packet_path.expanduser().resolve(strict=True)
    ontology = ontology_path.expanduser().resolve(strict=True)
    if _sha256_file(packet) != expected_packet_sha256:
        raise ValueError("review packet digest drift")
    _validate_ontology(ontology, expected_sha256=expected_ontology_sha256)
    sources: dict[str, dict[str, Any]] = {}
    official_semantics_by_image: dict[str, list[dict[str, Any]]] = {}
    official_loaded = load_generation7_annotations(
        candidate_pool_path, image_ids=selected_in_sample_order
    )
    for image_id in selected_in_sample_order:
        row = pool.get(image_id)
        if row is None:
            raise ValueError(f"selected image {image_id} is absent from candidate pool")
        metadata = row.get("metadata")
        if not isinstance(metadata, Mapping) or metadata.get("split") != "train":
            raise ValueError(f"selected image {image_id} is not train-only")
        images = row.get("images")
        if (
            not isinstance(images, list)
            or len(images) != 1
            or not isinstance(images[0], str)
        ):
            raise ValueError(f"selected image {image_id} lacks one source image")
        width = _positive_int(row.get("width"), field=f"image {image_id} width")
        height = _positive_int(row.get("height"), field=f"image {image_id} height")
        raw_path = Path(images[0]).expanduser()
        image_path = (
            raw_path.resolve(strict=True)
            if raw_path.is_absolute()
            else (candidate_pool_path.resolve().parent / raw_path).resolve(strict=True)
        )
        with Image.open(image_path) as opened:
            if opened.size != (width, height):
                raise ValueError(f"source-image dimension drift for image {image_id}")
        member = member_by_image[image_id]
        census_record = census_selection.records[image_id]
        reference = census._mapping(
            adapter["reference_records"][image_id], f"reference_records[{image_id}]"
        )
        reference_image = census._mapping(
            reference.get("image"), f"reference image[{image_id}]"
        )
        source_sha256 = _sha256_file(image_path)
        if (
            member["source_image_path"] != str(image_path)
            or member["source_image_size_bytes"] != image_path.stat().st_size
            or member["source_image_sha256"] != source_sha256
            or member["source_image_width"] != width
            or member["source_image_height"] != height
            or member["candidate_pool_record_sha256"]
            != _sha256_bytes(canonical_json_text(row).encode("utf-8"))
            or member["census_record_sha256"]
            != _sha256_bytes(canonical_json_text(census_record).encode("utf-8"))
            or Path(str(reference_image.get("path"))).expanduser().resolve(strict=True)
            != image_path
            or reference_image.get("content_sha256") != source_sha256
            or reference_image.get("width") != width
            or reference_image.get("height") != height
        ):
            raise ValueError(f"selected member {image_id} source/input binding drift")
        adapter_owners = _canonical_official_owner_semantics(
            adapter["image_results"][image_id].get("owners"), image_id=image_id
        )
        loaded_owners = _canonical_official_owner_semantics(
            official_loaded.get(image_id, []), image_id=image_id
        )
        if (
            adapter_owners != loaded_owners
            or member["official_owner_record_sha256"]
            != _sha256_bytes(canonical_json_text(adapter_owners).encode("utf-8"))
            or member["replay_input_sha256"]
            != _member_replay_input_sha256(
                adapter=adapter,
                image_id=image_id,
                census_record=census_record,
            )
        ):
            raise ValueError(f"selected member {image_id} owner/replay binding drift")
        official_semantics_by_image[image_id] = adapter_owners
        sources[image_id] = {
            "image_id": image_id,
            "image_path": str(image_path),
            "image_sha256": source_sha256,
            "source_image_width": width,
            "source_image_height": height,
        }
    private_image_order = list(selected_in_sample_order)
    reviewer_image_order = sorted(selected_in_sample_order, key=int)
    owner_rows: list[dict[str, Any]] = []
    for image_id in reviewer_image_order:
        for owner in official_semantics_by_image[image_id]:
            owner_id = str(owner["owner_id"])
            name = str(owner["normalized_category_name"])
            official_id = int(owner["official_coco_category_id"])
            box = list(owner["source_canvas_box_xyxy"])
            _require_box(
                box,
                width=sources[image_id]["source_image_width"],
                height=sources[image_id]["source_image_height"],
                integer=False,
            )
            owner_rows.append(
                {
                    "schema_version": OFFICIAL_OWNER_SCHEMA_VERSION,
                    "image_id": image_id,
                    "image_sha256": sources[image_id]["image_sha256"],
                    "owner_id": owner_id,
                    "owner_origin": "official_annotation",
                    "normalized_category_name": name,
                    "official_coco_category_id": official_id,
                    "source_canvas_box_xyxy": box,
                    "linked_reviewer_label_identifiers": [],
                }
            )
    queue_rows = [
        {
            "schema_version": REVIEW_QUEUE_SCHEMA_VERSION,
            "review_identifier": f"trajectory-owner-set-review:{image_id}:{role}",
            "reviewer_role_identifier": role,
            **sources[image_id],
            "packet_id": PACKET_ID,
            "packet_path": str(packet),
            "packet_sha256": expected_packet_sha256,
            "ontology_id": ONTOLOGY_ID,
            "ontology_path": str(ontology),
            "ontology_sha256": expected_ontology_sha256,
        }
        for image_id in reviewer_image_order
        for role in REVIEWER_ROLES
    ]
    queue_payload = _jsonl_bytes(queue_rows)
    owner_payload = _jsonl_bytes(owner_rows)
    manifest = {
        "schema_version": QUEUE_MANIFEST_SCHEMA_VERSION,
        "look_id": look_id,
        "population_scope": "train_only",
        "selected_image_count": len(reviewer_image_order),
        "private_sample_order_image_ids": private_image_order,
        "reviewer_numeric_order_image_ids": reviewer_image_order,
        "current_image_ids_sha256": _ordered_image_ids_sha256(private_image_order),
        "selection_sha256": expected_selection_sha256,
        "member_manifest_sha256": expected_member_manifest_sha256,
        "member_manifest_receipt_sha256": expected_member_manifest_receipt_sha256,
        "candidate_pool_sha256": expected_candidate_pool_sha256,
        "census_sha256": expected_census_sha256,
        "prior_look_receipt_sha256": prior_receipt_sha256,
        "packet_sha256": expected_packet_sha256,
        "ontology_sha256": expected_ontology_sha256,
        "review_queue_sha256": _sha256_bytes(queue_payload),
        "official_owner_ledger_sha256": _sha256_bytes(owner_payload),
        "reviewer_visible_fields": sorted(_QUEUE_FIELDS),
        "official_owner_ledger_reviewer_visible": False,
        "private_manifest_reviewer_visible": False,
        "sample_order_reviewer_visible": False,
        "look_identity_reviewer_visible": False,
        "cutoff_or_prior_outcome_reviewer_visible": False,
    }
    return ReviewQueueArtifacts(queue_payload, owner_payload, _json_bytes(manifest))


def _validate_queue(
    payload: bytes,
    *,
    expected_sha256: str,
    expected_packet_sha256: str,
    expected_ontology_sha256: str,
) -> tuple[list[Mapping[str, Any]], dict[str, dict[str, Mapping[str, Any]]], list[str]]:
    if _sha256_bytes(payload) != _require_sha256(
        expected_sha256, field="expected_review_queue_sha256"
    ):
        raise ValueError("review queue digest drift")
    rows = _parse_jsonl(payload, artifact="review queue")
    by_role: dict[str, dict[str, Mapping[str, Any]]] = {
        role: {} for role in REVIEWER_ROLES
    }
    image_order: list[str] = []
    image_facts: dict[str, tuple[Any, ...]] = {}
    verified_contract_paths: set[tuple[str, str]] = set()
    for index, row in enumerate(rows):
        _require_fields(row, _QUEUE_FIELDS, artifact="review queue row")
        if row["schema_version"] != REVIEW_QUEUE_SCHEMA_VERSION:
            raise ValueError("review queue schema drift")
        role = row["reviewer_role_identifier"]
        if role != REVIEWER_ROLES[index % 2]:
            raise ValueError("review queue role order is not canonical")
        image_id = _canonical_image_id(row["image_id"], field="review queue image_id")
        expected_identifier = f"trajectory-owner-set-review:{image_id}:{role}"
        if row["review_identifier"] != expected_identifier:
            raise ValueError("review identifier is not canonical")
        if (
            row["packet_id"] != PACKET_ID
            or row["packet_sha256"] != expected_packet_sha256
        ):
            raise ValueError("review queue packet drift")
        if (
            row["ontology_id"] != ONTOLOGY_ID
            or row["ontology_sha256"] != expected_ontology_sha256
        ):
            raise ValueError("review queue ontology drift")
        for path_field, digest_field, expected_digest in (
            ("packet_path", "packet_sha256", expected_packet_sha256),
            ("ontology_path", "ontology_sha256", expected_ontology_sha256),
        ):
            contract_path = Path(str(row[path_field]))
            cache_key = (str(contract_path), expected_digest)
            if cache_key not in verified_contract_paths:
                if (
                    not contract_path.is_file()
                    or row[digest_field] != expected_digest
                    or _sha256_file(contract_path) != expected_digest
                ):
                    raise ValueError(f"review queue {path_field} path/hash drift")
                verified_contract_paths.add(cache_key)
        width = _positive_int(row["source_image_width"], field="source_image_width")
        height = _positive_int(row["source_image_height"], field="source_image_height")
        image_path = Path(str(row["image_path"]))
        if not image_path.is_file() or _sha256_file(image_path) != row["image_sha256"]:
            raise ValueError("review queue source-image path/hash drift")
        with Image.open(image_path) as opened:
            if opened.size != (width, height):
                raise ValueError("review queue source-image dimension drift")
        facts = (row["image_sha256"], width, height, str(image_path))
        if image_id in image_facts and image_facts[image_id] != facts:
            raise ValueError("cross-role image facts disagree")
        image_facts[image_id] = facts
        if role == "reviewer-one":
            image_order.append(image_id)
        elif not image_order or image_order[-1] != image_id:
            raise ValueError("review role rows for one image must be adjacent")
        if row["review_identifier"] in by_role[role]:
            raise ValueError("duplicate review identifier")
        by_role[role][str(row["review_identifier"])] = row
    if (
        not rows
        or len(rows) != 2 * len(image_order)
        or len(image_order) != len(set(image_order))
    ):
        raise ValueError("review queue must assign both roles exactly once per image")
    return rows, by_role, image_order


def _validate_queue_manifest(
    payload: bytes,
    *,
    review_queue_jsonl: bytes,
    official_owner_ledger_jsonl: bytes,
) -> Mapping[str, Any]:
    manifest = _parse_json(payload, artifact="private review queue manifest")
    _require_fields(
        manifest, _QUEUE_MANIFEST_FIELDS, artifact="private review queue manifest"
    )
    private_order = manifest["private_sample_order_image_ids"]
    reviewer_order = manifest["reviewer_numeric_order_image_ids"]
    if (
        manifest["schema_version"] != QUEUE_MANIFEST_SCHEMA_VERSION
        or manifest["look_id"] not in LOOK_IDS
        or manifest["population_scope"] != "train_only"
        or not isinstance(private_order, list)
        or not isinstance(reviewer_order, list)
    ):
        raise ValueError("private queue manifest schema/scope/order drift")
    private_ids = [
        _canonical_image_id(value, field="private queue sample image_id")
        for value in private_order
    ]
    reviewer_ids = [
        _canonical_image_id(value, field="private queue reviewer image_id")
        for value in reviewer_order
    ]
    if (
        len(private_ids) != LOOK_ONE_COUNT
        or len(set(private_ids)) != LOOK_ONE_COUNT
        or reviewer_ids != sorted(private_ids, key=int)
        or manifest["selected_image_count"] != LOOK_ONE_COUNT
        or manifest["current_image_ids_sha256"]
        != _ordered_image_ids_sha256(private_ids)
        or manifest["review_queue_sha256"] != _sha256_bytes(review_queue_jsonl)
        or manifest["official_owner_ledger_sha256"]
        != _sha256_bytes(official_owner_ledger_jsonl)
        or manifest["reviewer_visible_fields"] != sorted(_QUEUE_FIELDS)
        or manifest["official_owner_ledger_reviewer_visible"] is not False
        or manifest["private_manifest_reviewer_visible"] is not False
        or manifest["sample_order_reviewer_visible"] is not False
        or manifest["look_identity_reviewer_visible"] is not False
        or manifest["cutoff_or_prior_outcome_reviewer_visible"] is not False
    ):
        raise ValueError("private queue manifest binding/blinding drift")
    for field in (
        "selection_sha256",
        "member_manifest_sha256",
        "member_manifest_receipt_sha256",
        "candidate_pool_sha256",
        "census_sha256",
        "packet_sha256",
        "ontology_sha256",
        "review_queue_sha256",
        "official_owner_ledger_sha256",
    ):
        _require_sha256(manifest[field], field=f"queue manifest {field}")
    prior = manifest["prior_look_receipt_sha256"]
    if manifest["look_id"] == "look_one":
        if prior is not None:
            raise ValueError("look-one queue manifest carries a prior receipt")
    else:
        _require_sha256(prior, field="queue manifest prior look receipt")
    return manifest


def _candidate_categories(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("candidate_categories must be a list")
    result: list[Mapping[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("candidate category must be an object")
        _require_fields(item, _CATEGORY_FIELDS, artifact="candidate category")
        _require_category(
            item["normalized_category_name"], item["official_coco_category_id"]
        )
        result.append(item)
    canonical = sorted(
        result,
        key=lambda item: (
            item["official_coco_category_id"],
            item["normalized_category_name"],
        ),
    )
    if result != canonical or len(
        {canonical_json_text(item) for item in result}
    ) != len(result):
        raise ValueError("candidate_categories must be sorted and unique")
    return result


def _label_sort_key(label: Mapping[str, Any]) -> tuple[Any, ...]:
    box = label["source_canvas_box_xyxy"]
    spatial = tuple(box) if box is not None else (float("inf"),) * 4
    official_id = label["official_coco_category_id"]
    return (
        0 if box is not None else 1,
        spatial[1],
        spatial[0],
        spatial[3],
        spatial[2],
        _STATE_ORDER[label["reviewer_state"]],
        official_id if official_id is not None else float("inf"),
        canonical_json_text(label["candidate_categories"]),
    )


def _validate_label(
    label: Mapping[str, Any],
    *,
    role: str,
    image_id: str,
    ordinal: int,
    width: int,
    height: int,
) -> None:
    _require_fields(label, _LABEL_FIELDS, artifact="reviewer label")
    state = label["reviewer_state"]
    reason = label["reason_code"]
    if (
        state not in _REVIEW_REASON_BY_STATE
        or reason not in _REVIEW_REASON_BY_STATE[state]
    ):
        raise ValueError(f"reviewer state/reason mismatch: {state}/{reason}")
    candidates = _candidate_categories(label["candidate_categories"])
    name = label["normalized_category_name"]
    official_id = label["official_coco_category_id"]
    if state in {"accepted", "partial", "crowd"}:
        _require_category(name, official_id)
        if candidates:
            raise ValueError(f"{state} label cannot retain candidate_categories")
    elif state == "ambiguous":
        if not candidates:
            raise ValueError("ambiguous label requires candidate_categories")
        if len(candidates) == 1:
            if (name, official_id) != (
                candidates[0]["normalized_category_name"],
                candidates[0]["official_coco_category_id"],
            ):
                raise ValueError("single-candidate ambiguous scalar category mismatch")
        elif name is not None or official_id is not None:
            raise ValueError("multi-candidate ambiguous scalar category must be null")
    elif name is not None or official_id is not None or candidates:
        raise ValueError("out-of-scope label cannot carry a category")
    box = label["source_canvas_box_xyxy"]
    if box is None:
        if state != "partial" or reason != "boundary_not_reproducible":
            raise ValueError(
                "null box is allowed only for boundary-unreproducible partial"
            )
    else:
        _require_box(box, width=width, height=height, integer=True)
    expected_id = f"{role}:{image_id}:{ordinal:04d}"
    if label["reviewer_local_object_identifier"] != expected_id:
        raise ValueError("reviewer local object identifier is not canonical")


def validate_role_artifact(
    *,
    role_artifact_jsonl: bytes,
    reviewer_role_identifier: str,
    review_queue_jsonl: bytes,
    expected_review_queue_sha256: str,
    expected_packet_sha256: str,
    expected_ontology_sha256: str,
) -> tuple[Mapping[str, Any], ...]:
    if reviewer_role_identifier not in REVIEWER_ROLES:
        raise ValueError("unknown reviewer role")
    _, by_role, image_order = _validate_queue(
        review_queue_jsonl,
        expected_sha256=expected_review_queue_sha256,
        expected_packet_sha256=expected_packet_sha256,
        expected_ontology_sha256=expected_ontology_sha256,
    )
    rows = _parse_jsonl(
        role_artifact_jsonl, artifact=f"{reviewer_role_identifier} artifact"
    )
    if len(rows) != len(image_order):
        raise ValueError("role artifact lacks a complete image disposition")
    seen_reviews: set[str] = set()
    for position, row in enumerate(rows):
        _require_fields(row, _ROLE_FIELDS, artifact="role artifact row")
        if row["schema_version"] != REVIEWER_SCHEMA_BY_ROLE[reviewer_role_identifier]:
            raise ValueError("role artifact schema drift")
        if row["reviewer_role_identifier"] != reviewer_role_identifier:
            raise ValueError("cross-role artifact leakage")
        review_id = str(row["review_identifier"])
        queue_row = by_role[reviewer_role_identifier].get(review_id)
        if queue_row is None:
            raise ValueError("role disposition is absent from its assigned queue")
        image_id = _canonical_image_id(row["image_id"], field="role image_id")
        if image_id != image_order[position] or image_id != queue_row["image_id"]:
            raise ValueError("role artifact image order drift")
        for field in (
            "packet_id",
            "packet_sha256",
            "ontology_sha256",
            "image_sha256",
            "source_image_width",
            "source_image_height",
        ):
            expected = PACKET_ID if field == "packet_id" else queue_row[field]
            if row[field] != expected:
                raise ValueError(f"role artifact {field} drift")
        if row["review_queue_sha256"] != expected_review_queue_sha256:
            raise ValueError("role artifact review-queue digest drift")
        if row["image_disposition"] != "complete":
            raise ValueError("every image disposition must be complete")
        labels = row["labels"]
        if not isinstance(labels, list) or any(
            not isinstance(item, Mapping) for item in labels
        ):
            raise ValueError("role labels must be a list of objects")
        sorted_labels = sorted(labels, key=_label_sort_key)
        keys = [_label_sort_key(item) for item in sorted_labels]
        if labels != sorted_labels or len(keys) != len(set(keys)):
            raise ValueError("role labels are not canonical or contain duplicates")
        for ordinal, label in enumerate(labels, start=1):
            _validate_label(
                label,
                role=reviewer_role_identifier,
                image_id=image_id,
                ordinal=ordinal,
                width=int(row["source_image_width"]),
                height=int(row["source_image_height"]),
            )
        if review_id in seen_reviews:
            raise ValueError("duplicate image disposition")
        seen_reviews.add(review_id)
    if seen_reviews != set(by_role[reviewer_role_identifier]):
        raise ValueError("role artifact does not dispose every assigned image")
    return tuple(rows)


def seal_role_artifact(**inputs: Any) -> bytes:
    """Validate one role independently and return its deterministic digest seal."""

    rows = validate_role_artifact(**inputs)
    role = str(inputs["reviewer_role_identifier"])
    payload = inputs["role_artifact_jsonl"]
    seal = {
        "schema_version": ROLE_SEAL_SCHEMA_VERSION,
        "reviewer_role_identifier": role,
        "role_schema_version": REVIEWER_SCHEMA_BY_ROLE[role],
        "role_artifact_sha256": _sha256_bytes(payload),
        "review_queue_sha256": inputs["expected_review_queue_sha256"],
        "packet_sha256": inputs["expected_packet_sha256"],
        "ontology_sha256": inputs["expected_ontology_sha256"],
        "image_disposition_count": len(rows),
        "label_count": sum(len(row["labels"]) for row in rows),
        "complete": True,
    }
    return _json_bytes(seal)


def _validate_role_seal(
    seal_payload: bytes,
    *,
    role: str,
    role_payload: bytes,
    queue_sha256: str,
    packet_sha256: str,
    ontology_sha256: str,
) -> Mapping[str, Any]:
    seal = _parse_json(seal_payload, artifact=f"{role} seal")
    expected_fields = {
        "schema_version",
        "reviewer_role_identifier",
        "role_schema_version",
        "role_artifact_sha256",
        "review_queue_sha256",
        "packet_sha256",
        "ontology_sha256",
        "image_disposition_count",
        "label_count",
        "complete",
    }
    _require_fields(seal, expected_fields, artifact="role seal")
    expected = {
        "schema_version": ROLE_SEAL_SCHEMA_VERSION,
        "reviewer_role_identifier": role,
        "role_schema_version": REVIEWER_SCHEMA_BY_ROLE[role],
        "role_artifact_sha256": _sha256_bytes(role_payload),
        "review_queue_sha256": queue_sha256,
        "packet_sha256": packet_sha256,
        "ontology_sha256": ontology_sha256,
        "complete": True,
    }
    if any(seal.get(key) != value for key, value in expected.items()):
        raise ValueError(f"{role} seal binding drift")
    return seal


def _validate_owner_rows(
    payload: bytes,
    *,
    image_facts: Mapping[str, tuple[str, int, int]],
    allow_additions: bool,
) -> list[Mapping[str, Any]]:
    rows = _parse_jsonl(payload, artifact="owner ledger", allow_empty=True)
    seen: set[str] = set()
    prior: tuple[int, str] | None = None
    order = {image_id: index for index, image_id in enumerate(image_facts)}
    for row in rows:
        _require_fields(row, _OWNER_FIELDS, artifact="owner ledger row")
        if row["schema_version"] != OFFICIAL_OWNER_SCHEMA_VERSION:
            raise ValueError("owner ledger schema drift")
        image_id = _canonical_image_id(row["image_id"], field="owner image_id")
        if image_id not in image_facts:
            raise ValueError("owner ledger contains an unselected image")
        image_sha, width, height = image_facts[image_id]
        if row["image_sha256"] != image_sha:
            raise ValueError("owner ledger image digest drift")
        owner_id = str(row["owner_id"])
        if not owner_id or owner_id in seen:
            raise ValueError("owner identifier is empty or duplicated")
        seen.add(owner_id)
        origin = row["owner_origin"]
        if origin not in {"official_annotation", "review_addition"}:
            raise ValueError("unknown owner origin")
        if not allow_additions and origin != "official_annotation":
            raise ValueError("private official ledger contains a review addition")
        linked = row["linked_reviewer_label_identifiers"]
        if not isinstance(linked, list) or linked != sorted(set(linked)):
            raise ValueError("owner linked reviewer labels must be sorted and unique")
        if (origin == "official_annotation" and linked) or (
            origin == "review_addition" and not linked
        ):
            raise ValueError("owner origin and reviewer-label provenance disagree")
        _require_category(
            row["normalized_category_name"], row["official_coco_category_id"]
        )
        _require_box(
            row["source_canvas_box_xyxy"], width=width, height=height, integer=False
        )
        key = (order[image_id], owner_id)
        if prior is not None and key <= prior:
            raise ValueError("owner ledger order is not canonical")
        prior = key
    return rows


def build_adjudication_queue(
    *,
    review_queue_jsonl: bytes,
    expected_review_queue_sha256: str,
    reviewer_one_jsonl: bytes,
    reviewer_one_seal_json: bytes,
    reviewer_two_jsonl: bytes,
    reviewer_two_seal_json: bytes,
    official_owner_ledger_jsonl: bytes,
    expected_official_owner_ledger_sha256: str,
    expected_packet_sha256: str,
    expected_ontology_sha256: str,
) -> bytes:
    """Join both sealed role artifacts with official owners, still route-blind."""

    queue_rows, _, image_order = _validate_queue(
        review_queue_jsonl,
        expected_sha256=expected_review_queue_sha256,
        expected_packet_sha256=expected_packet_sha256,
        expected_ontology_sha256=expected_ontology_sha256,
    )
    role_payloads = {
        "reviewer-one": reviewer_one_jsonl,
        "reviewer-two": reviewer_two_jsonl,
    }
    role_seals = {
        "reviewer-one": reviewer_one_seal_json,
        "reviewer-two": reviewer_two_seal_json,
    }
    role_rows: dict[str, tuple[Mapping[str, Any], ...]] = {}
    for role in REVIEWER_ROLES:
        role_rows[role] = validate_role_artifact(
            role_artifact_jsonl=role_payloads[role],
            reviewer_role_identifier=role,
            review_queue_jsonl=review_queue_jsonl,
            expected_review_queue_sha256=expected_review_queue_sha256,
            expected_packet_sha256=expected_packet_sha256,
            expected_ontology_sha256=expected_ontology_sha256,
        )
        seal = _validate_role_seal(
            role_seals[role],
            role=role,
            role_payload=role_payloads[role],
            queue_sha256=expected_review_queue_sha256,
            packet_sha256=expected_packet_sha256,
            ontology_sha256=expected_ontology_sha256,
        )
        if seal["image_disposition_count"] != len(role_rows[role]) or seal[
            "label_count"
        ] != sum(len(row["labels"]) for row in role_rows[role]):
            raise ValueError(f"{role} seal counts drift")
    if _sha256_bytes(official_owner_ledger_jsonl) != _require_sha256(
        expected_official_owner_ledger_sha256,
        field="expected_official_owner_ledger_sha256",
    ):
        raise ValueError("official owner ledger digest drift")
    queue_by_image = {
        str(row["image_id"]): row
        for row in queue_rows
        if row["reviewer_role_identifier"] == "reviewer-one"
    }
    image_facts = {
        image_id: (
            str(row["image_sha256"]),
            int(row["source_image_width"]),
            int(row["source_image_height"]),
        )
        for image_id, row in queue_by_image.items()
    }
    owners = _validate_owner_rows(
        official_owner_ledger_jsonl,
        image_facts=image_facts,
        allow_additions=False,
    )
    owners_by_image: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for owner in owners:
        owners_by_image[str(owner["image_id"])].append(owner)
    dispositions_by_role_image = {
        role: {str(row["image_id"]): row for row in rows}
        for role, rows in role_rows.items()
    }
    role_digests = {role: _sha256_bytes(role_payloads[role]) for role in REVIEWER_ROLES}
    seal_digests = {role: _sha256_bytes(role_seals[role]) for role in REVIEWER_ROLES}
    result = []
    for image_id in image_order:
        result.append(
            {
                "schema_version": ADJUDICATION_QUEUE_SCHEMA_VERSION,
                "adjudication_identifier": f"trajectory-owner-set-adjudication:{image_id}",
                "image_id": image_id,
                "image_sha256": image_facts[image_id][0],
                "source_image_width": image_facts[image_id][1],
                "source_image_height": image_facts[image_id][2],
                "review_queue_sha256": expected_review_queue_sha256,
                "packet_sha256": expected_packet_sha256,
                "ontology_sha256": expected_ontology_sha256,
                "reviewer_role_artifact_sha256": role_digests,
                "reviewer_role_seal_sha256": seal_digests,
                "reviewer_dispositions": {
                    role: dispositions_by_role_image[role][image_id]
                    for role in REVIEWER_ROLES
                },
                "official_owner_ledger_sha256": expected_official_owner_ledger_sha256,
                "official_owners": owners_by_image[image_id],
            }
        )
    return _jsonl_bytes(result)


def _validate_disposition(
    disposition: Mapping[str, Any],
    *,
    labels_by_id: Mapping[str, Mapping[str, Any]],
    official_owner_ids: set[str],
    width: int,
    height: int,
) -> None:
    _require_fields(disposition, _DISPOSITION_FIELDS, artifact="proposal disposition")
    state = disposition["adjudication_state"]
    reason = disposition["reason_code"]
    if (
        state not in _ADJUDICATION_REASON_BY_STATE
        or reason not in _ADJUDICATION_REASON_BY_STATE[state]
    ):
        raise ValueError(f"adjudication state/reason mismatch: {state}/{reason}")
    linked_labels = disposition["linked_reviewer_label_identifiers"]
    linked_official = disposition["linked_official_owner_ids"]
    if (
        not isinstance(linked_labels, list)
        or not linked_labels
        or linked_labels != sorted(set(linked_labels))
        or not set(linked_labels) <= set(labels_by_id)
    ):
        raise ValueError("proposal disposition label links are invalid")
    if (
        not isinstance(linked_official, list)
        or linked_official != sorted(set(linked_official))
        or not set(linked_official) <= official_owner_ids
    ):
        raise ValueError("proposal disposition official links are invalid")
    name = disposition["final_normalized_category_name"]
    official_id = disposition["final_official_coco_category_id"]
    box = disposition["final_source_canvas_box_xyxy"]
    if state == "official_duplicate":
        if (
            not linked_official
            or name is not None
            or official_id is not None
            or box is not None
        ):
            raise ValueError(
                "official_duplicate must link an official owner and add nothing"
            )
    elif state == "accepted_missing_owner":
        if linked_official:
            raise ValueError(
                "accepted missing owner cannot also be an official duplicate"
            )
        _require_category(name, official_id)
        _require_box(box, width=width, height=height, integer=True)
        linked_rows = [labels_by_id[str(identifier)] for identifier in linked_labels]
        if reason == "none" and (
            len(linked_rows) != 1
            or linked_rows[0]["reviewer_state"] != "accepted"
            or linked_rows[0]["normalized_category_name"] != name
            or linked_rows[0]["official_coco_category_id"] != official_id
            or linked_rows[0]["source_canvas_box_xyxy"] != box
        ):
            raise ValueError(
                "unqualified missing-owner acceptance must copy one accepted proposal"
            )
        if reason == "reviewer_agreement":
            linked_roles = {
                str(identifier).split(":", 1)[0] for identifier in linked_labels
            }
            if linked_roles != set(REVIEWER_ROLES) or any(
                row["reviewer_state"] != "accepted"
                or row["normalized_category_name"] != name
                or row["official_coco_category_id"] != official_id
                for row in linked_rows
            ):
                raise ValueError(
                    "reviewer-agreement addition lacks two-role category agreement"
                )
    elif state == "out_of_scope" and any(
        labels_by_id[str(identifier)]["reviewer_state"] != "out-of-scope"
        for identifier in linked_labels
    ):
        raise ValueError(
            "out-of-scope disposition must originate from out-of-scope labels"
        )
    elif (
        name is not None
        or official_id is not None
        or box is not None
        or linked_official
    ):
        raise ValueError("unresolved or excluded disposition cannot add an owner")


def _validate_adjudication_queue_row(row: Mapping[str, Any]) -> None:
    _require_fields(row, _ADJUDICATION_QUEUE_FIELDS, artifact="adjudication queue row")
    if row["schema_version"] != ADJUDICATION_QUEUE_SCHEMA_VERSION:
        raise ValueError("adjudication queue schema drift")
    image_id = _canonical_image_id(row["image_id"], field="adjudication image_id")
    if (
        row["adjudication_identifier"]
        != f"trajectory-owner-set-adjudication:{image_id}"
    ):
        raise ValueError("adjudication identifier is not canonical")
    width = _positive_int(row["source_image_width"], field="adjudication source width")
    height = _positive_int(
        row["source_image_height"], field="adjudication source height"
    )
    for field in (
        "image_sha256",
        "review_queue_sha256",
        "packet_sha256",
        "ontology_sha256",
        "official_owner_ledger_sha256",
    ):
        _require_sha256(row[field], field=f"adjudication.{field}")
    for field in ("reviewer_role_artifact_sha256", "reviewer_role_seal_sha256"):
        digests = row[field]
        if not isinstance(digests, Mapping) or set(digests) != set(REVIEWER_ROLES):
            raise ValueError(f"adjudication {field} role inventory drift")
        for role, digest in digests.items():
            _require_sha256(digest, field=f"adjudication.{field}.{role}")
    dispositions = row["reviewer_dispositions"]
    if not isinstance(dispositions, Mapping) or set(dispositions) != set(
        REVIEWER_ROLES
    ):
        raise ValueError("adjudication reviewer disposition role inventory drift")
    for role in REVIEWER_ROLES:
        role_row = dispositions[role]
        if not isinstance(role_row, Mapping):
            raise ValueError("adjudication reviewer disposition is not an object")
        _require_fields(
            role_row, _ROLE_FIELDS, artifact="adjudication reviewer disposition"
        )
        if (
            role_row["schema_version"] != REVIEWER_SCHEMA_BY_ROLE[role]
            or role_row["reviewer_role_identifier"] != role
            or role_row["image_id"] != image_id
            or role_row["image_sha256"] != row["image_sha256"]
            or role_row["packet_sha256"] != row["packet_sha256"]
            or role_row["ontology_sha256"] != row["ontology_sha256"]
            or role_row["review_queue_sha256"] != row["review_queue_sha256"]
            or role_row["image_disposition"] != "complete"
        ):
            raise ValueError("adjudication reviewer disposition binding drift")
        labels = role_row["labels"]
        if not isinstance(labels, list) or any(
            not isinstance(item, Mapping) for item in labels
        ):
            raise ValueError("adjudication reviewer labels are invalid")
        if labels != sorted(labels, key=_label_sort_key):
            raise ValueError("adjudication reviewer label order drift")
        for ordinal, label in enumerate(labels, start=1):
            _validate_label(
                label,
                role=role,
                image_id=image_id,
                ordinal=ordinal,
                width=width,
                height=height,
            )
    official = row["official_owners"]
    if not isinstance(official, list) or any(
        not isinstance(item, Mapping) for item in official
    ):
        raise ValueError("adjudication official owners are invalid")
    seen_official: set[str] = set()
    for owner in official:
        _require_fields(owner, _OWNER_FIELDS, artifact="adjudication official owner")
        if (
            owner["schema_version"] != OFFICIAL_OWNER_SCHEMA_VERSION
            or owner["image_id"] != image_id
            or owner["image_sha256"] != row["image_sha256"]
            or owner["owner_origin"] != "official_annotation"
            or owner["linked_reviewer_label_identifiers"] != []
        ):
            raise ValueError("adjudication official owner binding drift")
        owner_id = str(owner["owner_id"])
        if not owner_id or owner_id in seen_official:
            raise ValueError("adjudication official owner identifier drift")
        seen_official.add(owner_id)
        _require_category(
            owner["normalized_category_name"], owner["official_coco_category_id"]
        )
        _require_box(
            owner["source_canvas_box_xyxy"], width=width, height=height, integer=False
        )


def assemble_owner_ledger(
    *,
    adjudication_queue_jsonl: bytes,
    expected_adjudication_queue_sha256: str,
    adjudicator_decisions_jsonl: bytes,
    review_queue_manifest_json: bytes,
    expected_review_queue_manifest_sha256: str,
) -> OwnerLedgerArtifacts:
    """Compile complete route-blind decisions to an add-only owner ledger seal."""

    if _sha256_bytes(adjudication_queue_jsonl) != _require_sha256(
        expected_adjudication_queue_sha256,
        field="expected_adjudication_queue_sha256",
    ):
        raise ValueError("adjudication queue digest drift")
    queue_rows = _parse_jsonl(adjudication_queue_jsonl, artifact="adjudication queue")
    for queue_row in queue_rows:
        _validate_adjudication_queue_row(queue_row)
    if _sha256_bytes(review_queue_manifest_json) != _require_sha256(
        expected_review_queue_manifest_sha256,
        field="expected_review_queue_manifest_sha256",
    ):
        raise ValueError("private queue manifest digest drift")
    queue_manifest = _parse_json(
        review_queue_manifest_json, artifact="private review queue manifest"
    )
    _require_fields(
        queue_manifest,
        _QUEUE_MANIFEST_FIELDS,
        artifact="private review queue manifest",
    )
    queue_image_ids = [str(row["image_id"]) for row in queue_rows]
    private_image_ids = queue_manifest["private_sample_order_image_ids"]
    if (
        queue_manifest["schema_version"] != QUEUE_MANIFEST_SCHEMA_VERSION
        or queue_manifest["look_id"] not in LOOK_IDS
        or not isinstance(private_image_ids, list)
        or len(private_image_ids) != LOOK_ONE_COUNT
        or len(set(private_image_ids)) != LOOK_ONE_COUNT
        or set(private_image_ids) != set(queue_image_ids)
        or queue_manifest["current_image_ids_sha256"]
        != _ordered_image_ids_sha256(private_image_ids)
        or queue_manifest["reviewer_numeric_order_image_ids"] != queue_image_ids
        or len(queue_rows) != LOOK_ONE_COUNT
        or any(
            row["review_queue_sha256"] != queue_manifest["review_queue_sha256"]
            or row["official_owner_ledger_sha256"]
            != queue_manifest["official_owner_ledger_sha256"]
            or row["packet_sha256"] != queue_manifest["packet_sha256"]
            or row["ontology_sha256"] != queue_manifest["ontology_sha256"]
            for row in queue_rows
        )
    ):
        raise ValueError("adjudication queue and private manifest chain drift")
    for field in (
        "selection_sha256",
        "member_manifest_sha256",
        "candidate_pool_sha256",
        "census_sha256",
        "review_queue_sha256",
        "official_owner_ledger_sha256",
    ):
        _require_sha256(queue_manifest[field], field=f"queue manifest {field}")
    decisions = _parse_jsonl(
        adjudicator_decisions_jsonl, artifact="adjudicator decisions"
    )
    if len(decisions) != len(queue_rows):
        raise ValueError("adjudication decisions are incomplete")
    decision_fields = {
        "schema_version",
        "adjudication_identifier",
        "adjudication_queue_sha256",
        "source_binding_sha256",
        "confirmed_official_owner_ids",
        "proposal_dispositions",
        "owner_universe_complete",
        "owner_universe_uncertainty_reasons",
    }
    all_owner_rows: list[dict[str, Any]] = []
    uncertainty_rows: list[dict[str, Any]] = []
    for queue, decision in zip(queue_rows, decisions, strict=True):
        _require_fields(decision, decision_fields, artifact="adjudicator decision")
        if decision["schema_version"] != ADJUDICATION_DECISION_SCHEMA_VERSION:
            raise ValueError("adjudicator decision schema drift")
        if decision["adjudication_identifier"] != queue["adjudication_identifier"]:
            raise ValueError("adjudicator decision order or identifier drift")
        if decision["adjudication_queue_sha256"] != expected_adjudication_queue_sha256:
            raise ValueError("adjudicator decision queue digest drift")
        source_hash = _sha256_bytes(canonical_json_text(queue).encode())
        if decision["source_binding_sha256"] != source_hash:
            raise ValueError("adjudicator decision source binding drift")
        official = queue["official_owners"]
        if not isinstance(official, list):
            raise ValueError("adjudication queue official owners are invalid")
        official_ids = {str(row["owner_id"]) for row in official}
        if decision["confirmed_official_owner_ids"] != sorted(official_ids):
            raise ValueError("adjudicator must confirm every immutable official owner")
        labels_by_id = {
            str(label["reviewer_local_object_identifier"]): label
            for role in REVIEWER_ROLES
            for label in queue["reviewer_dispositions"][role]["labels"]
        }
        label_ids = set(labels_by_id)
        raw_dispositions = decision["proposal_dispositions"]
        if not isinstance(raw_dispositions, list) or any(
            not isinstance(item, Mapping) for item in raw_dispositions
        ):
            raise ValueError("proposal_dispositions must be a list of objects")
        seen_labels: set[str] = set()
        additions: list[Mapping[str, Any]] = []
        unresolved: list[Mapping[str, Any]] = []
        for disposition in raw_dispositions:
            _validate_disposition(
                disposition,
                labels_by_id=labels_by_id,
                official_owner_ids=official_ids,
                width=int(queue["source_image_width"]),
                height=int(queue["source_image_height"]),
            )
            linked = set(disposition["linked_reviewer_label_identifiers"])
            if seen_labels & linked:
                raise ValueError(
                    "reviewer label receives multiple adjudication dispositions"
                )
            seen_labels |= linked
            if disposition["adjudication_state"] == "accepted_missing_owner":
                additions.append(disposition)
            elif disposition["adjudication_state"] not in {
                "official_duplicate",
                "out_of_scope",
            }:
                unresolved.append(disposition)
        if seen_labels != label_ids:
            raise ValueError(
                "every reviewer label requires one adjudication disposition"
            )
        reasons = decision["owner_universe_uncertainty_reasons"]
        if (
            not isinstance(reasons, list)
            or reasons != sorted(set(reasons))
            or not set(reasons) <= _OWNER_UNIVERSE_REASONS
        ):
            raise ValueError("owner-universe uncertainty reasons are invalid")
        complete = decision["owner_universe_complete"]
        if not isinstance(complete, bool):
            raise ValueError("owner_universe_complete must be boolean")
        if complete and (reasons or unresolved):
            raise ValueError("retained uncertainty forbids a complete owner universe")
        if not complete and not reasons:
            raise ValueError("incomplete owner universe requires an explicit reason")
        all_owner_rows.extend(copy.deepcopy(official))
        additions.sort(
            key=lambda item: (
                item["final_official_coco_category_id"],
                item["final_source_canvas_box_xyxy"][1],
                item["final_source_canvas_box_xyxy"][0],
                item["final_source_canvas_box_xyxy"][3],
                item["final_source_canvas_box_xyxy"][2],
                tuple(item["linked_reviewer_label_identifiers"]),
            )
        )
        image_id = str(queue["image_id"])
        for ordinal, addition in enumerate(additions, start=1):
            all_owner_rows.append(
                {
                    "schema_version": OFFICIAL_OWNER_SCHEMA_VERSION,
                    "image_id": image_id,
                    "image_sha256": queue["image_sha256"],
                    "owner_id": f"review-owner:{image_id}:{ordinal:04d}",
                    "owner_origin": "review_addition",
                    "normalized_category_name": addition[
                        "final_normalized_category_name"
                    ],
                    "official_coco_category_id": addition[
                        "final_official_coco_category_id"
                    ],
                    "source_canvas_box_xyxy": addition["final_source_canvas_box_xyxy"],
                    "linked_reviewer_label_identifiers": addition[
                        "linked_reviewer_label_identifiers"
                    ],
                }
            )
        retained_axes = list(UNCERTAINTY_AXES) if (not complete or unresolved) else []
        uncertainty_rows.append(
            {
                "schema_version": UNCERTAINTY_LEDGER_SCHEMA_VERSION,
                "image_id": image_id,
                "owner_universe_complete": complete,
                "owner_universe_uncertainty_reasons": reasons,
                "retained_uncertainty_axes": retained_axes,
                "unresolved_proposal_dispositions": unresolved,
            }
        )
    image_order = {str(row["image_id"]): index for index, row in enumerate(queue_rows)}
    all_owner_rows.sort(
        key=lambda row: (image_order[str(row["image_id"])], str(row["owner_id"]))
    )
    owner_payload = _jsonl_bytes(all_owner_rows)
    uncertainty_payload = _jsonl_bytes(uncertainty_rows)
    seal = {
        "schema_version": OWNER_LEDGER_SEAL_SCHEMA_VERSION,
        "look_id": queue_manifest["look_id"],
        "queue_manifest_sha256": expected_review_queue_manifest_sha256,
        "selection_sha256": queue_manifest["selection_sha256"],
        "member_manifest_sha256": queue_manifest["member_manifest_sha256"],
        "candidate_pool_sha256": queue_manifest["candidate_pool_sha256"],
        "census_sha256": queue_manifest["census_sha256"],
        "current_image_ids_sha256": queue_manifest["current_image_ids_sha256"],
        "review_queue_sha256": queue_manifest["review_queue_sha256"],
        "official_owner_ledger_sha256": queue_manifest["official_owner_ledger_sha256"],
        "adjudication_queue_sha256": expected_adjudication_queue_sha256,
        "adjudication_sha256": _sha256_bytes(adjudicator_decisions_jsonl),
        "owner_ledger_sha256": _sha256_bytes(owner_payload),
        "uncertainty_ledger_sha256": _sha256_bytes(uncertainty_payload),
        "selected_image_count": len(queue_rows),
        "owner_count": len(all_owner_rows),
        "official_owner_count": sum(
            row["owner_origin"] == "official_annotation" for row in all_owner_rows
        ),
        "added_owner_count": sum(
            row["owner_origin"] == "review_addition" for row in all_owner_rows
        ),
        "owner_universe_uncertain_image_count": sum(
            not row["owner_universe_complete"] for row in uncertainty_rows
        ),
        "add_only": True,
        "official_owner_identity_and_box_immutable": True,
        "route_blind": True,
    }
    return OwnerLedgerArtifacts(
        adjudicator_decisions_jsonl,
        owner_payload,
        uncertainty_payload,
        _json_bytes(seal),
    )


def _prediction_from_assignment_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "image_id",
        "trajectory_id",
        "decode_mode",
        "seed",
        "generated_row_index",
        "prediction_id",
        "category",
        "bbox",
        "raw",
    )
    prediction = {
        field: copy.deepcopy(receipt[field]) for field in fields if field in receipt
    }
    required = {"generated_row_index", "prediction_id", "category", "bbox"}
    if not required <= set(prediction):
        raise AssemblyError(
            "existing assignment receipt lacks exact parsed-row evidence"
        )
    return prediction


def _rematch_assignment(
    original: Mapping[str, Any], owners: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    raw = original.get("row_assignment_receipts")
    if not isinstance(raw, list) or any(not isinstance(item, Mapping) for item in raw):
        raise AssemblyError("original assignment receipts are invalid")
    predictions = [_prediction_from_assignment_receipt(item) for item in raw]
    assignment = match_prefix(predictions, owners, 16)
    malformed = census._int_count(
        original.get("malformed_row_count", 0), "malformed_row_count"
    )
    duplicate = sum(
        item.get("entity_status") in {"duplicate", "duplicate_owner"}
        for item in assignment["row_assignment_receipts"]
    )
    unresolved = sum(
        item.get("entity_status")
        in {
            "semantic_mismatch_unresolved",
            "unresolved_pending_crop_review",
            "ambiguous_matched_review",
            "uncertain",
        }
        for item in assignment["row_assignment_receipts"]
    )
    assignment.update(
        {
            "malformed_row_count": malformed,
            "harmful_row_count": malformed + duplicate,
            "row_counts": {
                "duplicate": duplicate,
                "malformed": malformed,
                "unsupported_hallucination": 0,
                "semantic_error": 0,
                "unresolved": unresolved,
            },
        }
    )
    return assignment


def _assignment_row_identity_sha256(assignment: Mapping[str, Any]) -> str:
    raw = assignment.get("row_assignment_receipts")
    if not isinstance(raw, list) or any(not isinstance(item, Mapping) for item in raw):
        raise AssemblyError("assignment row receipts are invalid")
    identities = [_prediction_from_assignment_receipt(item) for item in raw]
    return _sha256_bytes(canonical_json_text(identities).encode("utf-8"))


def replay_owner_ledger(
    *,
    adapter: Mapping[str, Any],
    look_id: str,
    selection_json: bytes,
    expected_selection_sha256: str,
    selection_path: Path,
    member_manifest_jsonl: bytes,
    expected_member_manifest_sha256: str,
    member_manifest_receipt_json: bytes,
    expected_member_manifest_receipt_sha256: str,
    candidate_pool_path: Path,
    expected_candidate_pool_sha256: str,
    census_path: Path,
    expected_census_path: Path,
    expected_census_sha256: str,
    owner_ledger_jsonl: bytes,
    owner_ledger_seal_json: bytes,
) -> ReplayArtifacts:
    """Globally rematch all seventeen exact routes and reuse census semantics."""

    if look_id not in LOOK_IDS:
        raise ValueError("unknown replay look")
    selection = _validate_selection(
        selection_json,
        expected_sha256=expected_selection_sha256,
        selection_path=selection_path,
        member_manifest_jsonl=member_manifest_jsonl,
        expected_member_manifest_sha256=expected_member_manifest_sha256,
        member_manifest_receipt_json=member_manifest_receipt_json,
        expected_member_manifest_receipt_sha256=expected_member_manifest_receipt_sha256,
    )
    image_ids = list(
        selection.look_one_image_ids
        if look_id == "look_one"
        else selection.look_two_additional_image_ids
    )
    if len(image_ids) != LOOK_ONE_COUNT or len(image_ids) != len(set(image_ids)):
        raise ValueError("replay look image inventory drift")
    selection_sources = selection.source_artifacts
    if (
        selection_sources["candidate_pool"]["sha256"] != expected_candidate_pool_sha256
        or selection_sources["census"]["sha256"] != expected_census_sha256
    ):
        raise ValueError("replay source digests differ from selection")
    _validate_adapter_inventory(adapter, ordered_image_ids=image_ids)
    pool = _load_candidate_pool(
        candidate_pool_path, expected_sha256=expected_candidate_pool_sha256
    )
    if not set(selection.ordered_pool_image_ids).issubset(pool):
        raise ValueError("replay member set is not a candidate-pool subset")
    census_selection = load_frozen_census_records(
        census_path=census_path,
        expected_census_path=expected_census_path,
        expected_census_sha256=expected_census_sha256,
        selected_image_ids=image_ids,
    )
    member_rows, member_image_ids, _ = _validate_member_manifest(
        member_manifest_jsonl, expected_sha256=expected_member_manifest_sha256
    )
    if member_image_ids != selection.ordered_pool_image_ids:
        raise ValueError("replay member manifest order differs from selection")
    member_by_image = {str(row["image_id"]): row for row in member_rows}
    official_loaded = load_generation7_annotations(
        candidate_pool_path, image_ids=image_ids
    )
    references = adapter.get("reference_records")
    if not isinstance(references, Mapping):
        raise AssemblyError("adapter lacks source-image reference records")
    image_facts: dict[str, tuple[str, int, int]] = {}
    for image_id in image_ids:
        pool_row = pool[image_id]
        member = member_by_image[image_id]
        reference = census._mapping(
            references[image_id], f"reference_records[{image_id}]"
        )
        image = census._mapping(reference.get("image"), f"image[{image_id}]")
        image_path = Path(str(image.get("path"))).expanduser().resolve(strict=True)
        width = _positive_int(image.get("width"), field="adapter image width")
        height = _positive_int(image.get("height"), field="adapter image height")
        image_sha = _sha256_file(image_path)
        adapter_owners = _canonical_official_owner_semantics(
            adapter["image_results"][image_id].get("owners"), image_id=image_id
        )
        loaded_owners = _canonical_official_owner_semantics(
            official_loaded.get(image_id, []), image_id=image_id
        )
        census_record = census_selection.records[image_id]
        if (
            image.get("content_sha256") != image_sha
            or image_path.stat().st_size != member["source_image_size_bytes"]
            or str(image_path) != member["source_image_path"]
            or image_sha != member["source_image_sha256"]
            or width != member["source_image_width"]
            or height != member["source_image_height"]
            or member["candidate_pool_record_sha256"]
            != _sha256_bytes(canonical_json_text(pool_row).encode("utf-8"))
            or member["census_record_sha256"]
            != _sha256_bytes(canonical_json_text(census_record).encode("utf-8"))
            or adapter_owners != loaded_owners
            or member["official_owner_record_sha256"]
            != _sha256_bytes(canonical_json_text(adapter_owners).encode("utf-8"))
            or member["replay_input_sha256"]
            != _member_replay_input_sha256(
                adapter=adapter,
                image_id=image_id,
                census_record=census_record,
            )
        ):
            raise AssemblyError(f"replay selected member {image_id} binding drift")
        with Image.open(image_path) as opened:
            if opened.size != (width, height):
                raise AssemblyError(f"replay image {image_id} dimensions drift")
        image_facts[image_id] = (image_sha, width, height)

    seal = _parse_json(owner_ledger_seal_json, artifact="owner ledger seal")
    _require_fields(seal, _OWNER_LEDGER_SEAL_FIELDS, artifact="owner ledger seal")
    if (
        seal.get("schema_version") != OWNER_LEDGER_SEAL_SCHEMA_VERSION
        or seal.get("look_id") != look_id
        or seal.get("selection_sha256") != expected_selection_sha256
        or seal.get("member_manifest_sha256") != expected_member_manifest_sha256
        or seal.get("candidate_pool_sha256") != expected_candidate_pool_sha256
        or seal.get("census_sha256") != expected_census_sha256
        or seal.get("current_image_ids_sha256") != _ordered_image_ids_sha256(image_ids)
    ):
        raise ValueError("owner ledger seal schema drift")
    for field in (
        "queue_manifest_sha256",
        "selection_sha256",
        "member_manifest_sha256",
        "candidate_pool_sha256",
        "census_sha256",
        "current_image_ids_sha256",
        "review_queue_sha256",
        "official_owner_ledger_sha256",
        "adjudication_queue_sha256",
        "adjudication_sha256",
        "owner_ledger_sha256",
        "uncertainty_ledger_sha256",
    ):
        _require_sha256(seal[field], field=f"owner ledger seal {field}")
    for field in (
        "selected_image_count",
        "owner_count",
        "official_owner_count",
        "added_owner_count",
        "owner_universe_uncertain_image_count",
    ):
        value = seal[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"owner ledger seal {field} must be nonnegative integer")
    if seal.get("owner_ledger_sha256") != _sha256_bytes(owner_ledger_jsonl):
        raise ValueError("owner ledger is not bound by its seal")
    results = adapter.get("image_results")
    if not isinstance(results, Mapping) or set(results) != set(image_ids):
        raise AssemblyError("adapter does not cover the exact replay image set")
    ledger_rows = _validate_owner_rows(
        owner_ledger_jsonl,
        image_facts={
            image_id: image_facts[image_id] for image_id in sorted(image_ids, key=int)
        },
        allow_additions=True,
    )
    if (
        seal["selected_image_count"] != len(image_ids)
        or seal["owner_count"] != len(ledger_rows)
        or seal["official_owner_count"]
        != sum(row["owner_origin"] == "official_annotation" for row in ledger_rows)
        or seal["added_owner_count"]
        != sum(row["owner_origin"] == "review_addition" for row in ledger_rows)
        or seal["add_only"] is not True
        or seal["official_owner_identity_and_box_immutable"] is not True
        or seal["route_blind"] is not True
    ):
        raise ValueError("owner ledger seal counts or safety claims drift")
    owners_by_image: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in ledger_rows:
        owners_by_image[str(row["image_id"])].append(row)
    replay_rows: list[dict[str, Any]] = []
    invariant_fields = (
        "candidate_id",
        "trajectory_id",
        "decode_mode",
        "sample_index",
        "seed",
        "stop_reason",
        "source_b16_projection_status",
        "sampled_b16_projection_status",
        "generated_token_ids_sha256",
        "generated_token_count",
        "projected_generated_token_ids_sha256",
        "projected_generated_token_count",
        "raw_generated_token_ids_sha256",
        "raw_generated_token_count",
        "parse_status",
        "parser_drop_count",
        "evaluated_complete_row_count",
    )
    for image_id in image_ids:
        image_result = census._mapping(results[image_id], f"image_results[{image_id}]")
        original_owners = image_result.get("owners")
        if not isinstance(original_owners, list):
            raise AssemblyError(f"image {image_id} lacks official owners")
        official_rows = [
            row
            for row in owners_by_image[image_id]
            if row["owner_origin"] == "official_annotation"
        ]
        expected_official = sorted(
            (
                str(owner["owner_id"]),
                str(owner["category"]),
                [float(value) for value in owner["bbox"]],
            )
            for owner in original_owners
        )
        observed_official = sorted(
            (
                str(row["owner_id"]),
                str(row["normalized_category_name"]),
                [float(value) for value in row["source_canvas_box_xyxy"]],
            )
            for row in official_rows
        )
        if observed_official != expected_official:
            raise AssemblyError(
                "official owner identity/category/box immutability failed"
            )
        replay_owners = [
            {
                "owner_id": row["owner_id"],
                "category": row["normalized_category_name"],
                "bbox": row["source_canvas_box_xyxy"],
            }
            for row in owners_by_image[image_id]
        ]
        result = census._mapping(
            adapter["image_results"][image_id], f"image_results[{image_id}]"
        )
        evidence = census._mapping(
            result.get("trajectory_evidence"), "trajectory_evidence"
        )
        budgets = result.get("budgets")
        if (
            not isinstance(budgets, list)
            or len(budgets) != 1
            or budgets[0].get("budget") != 16
        ):
            raise AssemblyError("replay adapter lacks exact B16 assignments")
        assignments = census._mapping(
            budgets[0].get("trajectory_assignments"), "trajectory_assignments"
        )
        original_candidates = census._image_candidates(
            image_id, adapter, reverse_input=False
        )
        if (
            tuple(item["candidate_id"] for item in original_candidates)
            != EXPECTED_ROUTE_IDS
        ):
            raise AssemblyError(
                "replay must contain exactly the frozen seventeen routes"
            )
        replay_candidates: list[dict[str, Any]] = []
        route_row_identity: dict[str, tuple[str, str]] = {}
        for route_id in EXPECTED_ROUTE_IDS:
            route_row = (
                adapter["source_rows"].get((image_id, 0))
                if route_id == "source-b16"
                else adapter["sampled_rows"].get((image_id, int(route_id[-2:])))
            )
            if not isinstance(route_row, Mapping):
                raise AssemblyError(f"image {image_id} lacks exact route {route_id}")
            original_assignment = census._mapping(
                assignments.get(route_id), f"assignment[{route_id}]"
            )
            assignment = _rematch_assignment(original_assignment, replay_owners)
            original_row_identity = _assignment_row_identity_sha256(original_assignment)
            replayed_row_identity = _assignment_row_identity_sha256(assignment)
            if original_row_identity != replayed_row_identity:
                raise AssemblyError(
                    f"replay changed parsed row identity/order for {route_id}"
                )
            route_row_identity[route_id] = (
                original_row_identity,
                replayed_row_identity,
            )
            candidate = census._candidate_receipt(
                route_id=route_id,
                route_row=route_row,
                route_evidence=census._mapping(
                    evidence.get(route_id), f"evidence[{route_id}]"
                ),
                assignment=assignment,
                owners=replay_owners,
            )
            original = next(
                item for item in original_candidates if item["candidate_id"] == route_id
            )
            if any(
                candidate.get(field) != original.get(field)
                for field in invariant_fields
            ):
                raise AssemblyError(
                    f"replay changed token/decode/parser identity for {route_id}"
                )
            replay_candidates.append(candidate)
        replay_record = census._analyze_image_candidates(image_id, replay_candidates)
        original_record = census_selection.records.get(image_id)
        if not isinstance(original_record, Mapping):
            raise AssemblyError(
                f"original census record is missing for image {image_id}"
            )
        additions = [
            row
            for row in owners_by_image[image_id]
            if row["owner_origin"] == "review_addition"
        ]
        if not additions and replay_record != original_record:
            raise AssemblyError(
                f"no-op ledger does not exactly reproduce census image {image_id}"
            )
        route_identity = {
            route_id: {
                "candidate_identity_sha256": _sha256_bytes(
                    canonical_json_text(
                        {
                            field: next(
                                item
                                for item in replay_candidates
                                if item["candidate_id"] == route_id
                            ).get(field)
                            for field in invariant_fields
                        }
                    ).encode()
                ),
                "parser_evidence_sha256": _sha256_bytes(
                    canonical_json_text(evidence[route_id]["parser"]).encode()
                ),
                "original_row_identity_and_order_sha256": route_row_identity[route_id][
                    0
                ],
                "replayed_row_identity_and_order_sha256": route_row_identity[route_id][
                    1
                ],
            }
            for route_id in EXPECTED_ROUTE_IDS
        }
        replay_rows.append(
            {
                "schema_version": REPLAY_RECORD_SCHEMA_VERSION,
                "look_id": look_id,
                "image_id": image_id,
                "selection_sha256": expected_selection_sha256,
                "member_manifest_sha256": expected_member_manifest_sha256,
                "candidate_pool_sha256": expected_candidate_pool_sha256,
                "census_sha256": expected_census_sha256,
                "current_image_ids_sha256": _ordered_image_ids_sha256(image_ids),
                "owner_ledger_sha256": _sha256_bytes(owner_ledger_jsonl),
                "owner_ledger_seal_sha256": _sha256_bytes(owner_ledger_seal_json),
                "official_owner_count": len(official_rows),
                "added_owner_count": len(additions),
                "route_count": len(EXPECTED_ROUTE_IDS),
                "route_ids": list(EXPECTED_ROUTE_IDS),
                "route_identity": route_identity,
                "no_op_exact_reproduction": not additions,
                "census_record": replay_record,
            }
        )
    replay_payload = _jsonl_bytes(replay_rows)
    receipt = {
        "schema_version": REPLAY_RECEIPT_SCHEMA_VERSION,
        "terminal_status": "completed",
        "look_id": look_id,
        "selection_sha256": expected_selection_sha256,
        "member_manifest_sha256": expected_member_manifest_sha256,
        "member_manifest_receipt_sha256": expected_member_manifest_receipt_sha256,
        "candidate_pool_sha256": expected_candidate_pool_sha256,
        "census_path": census_selection.source_path,
        "census_sha256": expected_census_sha256,
        "current_image_ids_sha256": _ordered_image_ids_sha256(image_ids),
        "queue_manifest_sha256": seal["queue_manifest_sha256"],
        "adjudication_queue_sha256": seal["adjudication_queue_sha256"],
        "adjudication_sha256": seal["adjudication_sha256"],
        "uncertainty_ledger_sha256": seal["uncertainty_ledger_sha256"],
        "owner_ledger_sha256": _sha256_bytes(owner_ledger_jsonl),
        "owner_ledger_seal_sha256": _sha256_bytes(owner_ledger_seal_json),
        "ordered_image_ids": image_ids,
        "image_count": len(image_ids),
        "route_count_per_image": len(EXPECTED_ROUTE_IDS),
        "replay_records_sha256": _sha256_bytes(replay_payload),
        "no_op_exact_reproduction_count": sum(
            bool(row["no_op_exact_reproduction"]) for row in replay_rows
        ),
        "original_candidate_receipt_reused": True,
        "original_image_analyzer_reused": True,
        "global_match_prefix_reused": True,
        "original_row_identity_and_order_preserved": True,
    }
    return ReplayArtifacts(replay_payload, _json_bytes(receipt))


def _validate_replay_chain(
    *,
    replay_records_jsonl: bytes,
    replay_receipt_json: bytes,
    uncertainty_ledger_jsonl: bytes,
    owner_ledger_seal_json: bytes,
    look_id: str,
    ordered_image_ids: Sequence[str],
    expected_selection_sha256: str,
    expected_member_manifest_sha256: str,
    expected_member_manifest_receipt_sha256: str,
    expected_candidate_pool_sha256: str,
    expected_census_sha256: str,
) -> tuple[ValidatedReplayChain, dict[str, Mapping[str, Any]]]:
    for field, digest in (
        ("expected_selection_sha256", expected_selection_sha256),
        ("expected_member_manifest_sha256", expected_member_manifest_sha256),
        (
            "expected_member_manifest_receipt_sha256",
            expected_member_manifest_receipt_sha256,
        ),
        ("expected_candidate_pool_sha256", expected_candidate_pool_sha256),
        ("expected_census_sha256", expected_census_sha256),
    ):
        _require_sha256(digest, field=field)
    expected_ids = tuple(
        _canonical_image_id(value, field="classifier image_id")
        for value in ordered_image_ids
    )
    if (
        look_id not in LOOK_IDS
        or len(expected_ids) != LOOK_ONE_COUNT
        or len(set(expected_ids)) != LOOK_ONE_COUNT
    ):
        raise ValueError("classifier look image inventory drift")
    replay_rows = tuple(_parse_jsonl(replay_records_jsonl, artifact="replay records"))
    replay_ids = tuple(
        _canonical_image_id(row.get("image_id"), field="replay image_id")
        for row in replay_rows
    )
    if len(replay_ids) != len(set(replay_ids)):
        raise ValueError("replay records duplicate an image")
    if replay_ids != expected_ids:
        raise ValueError("replay records image set/order differs from the look")
    uncertainty_rows = tuple(
        _parse_jsonl(uncertainty_ledger_jsonl, artifact="uncertainty ledger")
    )
    uncertainty_ids = [
        _canonical_image_id(row.get("image_id"), field="uncertainty image_id")
        for row in uncertainty_rows
    ]
    if len(uncertainty_ids) != len(set(uncertainty_ids)):
        raise ValueError("uncertainty ledger duplicates an image")
    if set(uncertainty_ids) != set(expected_ids):
        raise ValueError("uncertainty ledger and replay image sets differ")
    uncertainty_by_image = {
        image_id: row for image_id, row in zip(uncertainty_ids, uncertainty_rows)
    }
    seal = _parse_json(owner_ledger_seal_json, artifact="owner ledger seal")
    _require_fields(seal, _OWNER_LEDGER_SEAL_FIELDS, artifact="owner ledger seal")
    for field in (
        "queue_manifest_sha256",
        "selection_sha256",
        "member_manifest_sha256",
        "candidate_pool_sha256",
        "census_sha256",
        "current_image_ids_sha256",
        "review_queue_sha256",
        "official_owner_ledger_sha256",
        "adjudication_queue_sha256",
        "adjudication_sha256",
        "owner_ledger_sha256",
        "uncertainty_ledger_sha256",
    ):
        _require_sha256(seal[field], field=f"classifier owner seal {field}")
    for field in (
        "selected_image_count",
        "owner_count",
        "official_owner_count",
        "added_owner_count",
        "owner_universe_uncertain_image_count",
    ):
        count = seal[field]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"classifier owner seal {field} is invalid")
    if seal["official_owner_count"] + seal["added_owner_count"] != seal["owner_count"]:
        raise ValueError("classifier owner seal owner counts disagree")
    owner_seal_sha = _sha256_bytes(owner_ledger_seal_json)
    expected_common = {
        "look_id": look_id,
        "selection_sha256": expected_selection_sha256,
        "member_manifest_sha256": expected_member_manifest_sha256,
        "candidate_pool_sha256": expected_candidate_pool_sha256,
        "census_sha256": expected_census_sha256,
        "current_image_ids_sha256": _ordered_image_ids_sha256(expected_ids),
    }
    if (
        seal["schema_version"] != OWNER_LEDGER_SEAL_SCHEMA_VERSION
        or any(seal.get(field) != value for field, value in expected_common.items())
        or seal["uncertainty_ledger_sha256"] != _sha256_bytes(uncertainty_ledger_jsonl)
        or seal["selected_image_count"] != LOOK_ONE_COUNT
        or seal["add_only"] is not True
        or seal["official_owner_identity_and_box_immutable"] is not True
        or seal["route_blind"] is not True
    ):
        raise ValueError("owner seal and classifier chain drift")
    receipt = _parse_json(replay_receipt_json, artifact="replay receipt")
    _require_fields(receipt, _REPLAY_RECEIPT_FIELDS, artifact="replay receipt")
    for field in (
        "selection_sha256",
        "member_manifest_sha256",
        "member_manifest_receipt_sha256",
        "candidate_pool_sha256",
        "census_sha256",
        "current_image_ids_sha256",
        "queue_manifest_sha256",
        "adjudication_queue_sha256",
        "adjudication_sha256",
        "uncertainty_ledger_sha256",
        "owner_ledger_sha256",
        "owner_ledger_seal_sha256",
        "replay_records_sha256",
    ):
        _require_sha256(receipt[field], field=f"classifier replay receipt {field}")
    expected_receipt = {
        **expected_common,
        "schema_version": REPLAY_RECEIPT_SCHEMA_VERSION,
        "terminal_status": "completed",
        "member_manifest_receipt_sha256": expected_member_manifest_receipt_sha256,
        "queue_manifest_sha256": seal["queue_manifest_sha256"],
        "adjudication_queue_sha256": seal["adjudication_queue_sha256"],
        "adjudication_sha256": seal["adjudication_sha256"],
        "uncertainty_ledger_sha256": seal["uncertainty_ledger_sha256"],
        "owner_ledger_sha256": seal["owner_ledger_sha256"],
        "owner_ledger_seal_sha256": owner_seal_sha,
        "ordered_image_ids": list(expected_ids),
        "image_count": LOOK_ONE_COUNT,
        "route_count_per_image": len(EXPECTED_ROUTE_IDS),
        "replay_records_sha256": _sha256_bytes(replay_records_jsonl),
        "original_candidate_receipt_reused": True,
        "original_image_analyzer_reused": True,
        "global_match_prefix_reused": True,
        "original_row_identity_and_order_preserved": True,
    }
    if any(receipt.get(field) != value for field, value in expected_receipt.items()):
        raise ValueError("replay receipt chain drift")
    census_path = Path(str(receipt["census_path"])).expanduser().resolve(strict=True)
    if _sha256_file(census_path) != expected_census_sha256:
        raise ValueError("replay receipt census path/hash drift")
    frozen_census = load_frozen_census_records(
        census_path=census_path,
        expected_census_path=census_path,
        expected_census_sha256=expected_census_sha256,
        selected_image_ids=expected_ids,
    )
    no_op_count = receipt["no_op_exact_reproduction_count"]
    if (
        isinstance(no_op_count, bool)
        or not isinstance(no_op_count, int)
        or no_op_count < 0
        or no_op_count > LOOK_ONE_COUNT
    ):
        raise ValueError("replay receipt no-op count drift")
    common_digest_fields = (
        "selection_sha256",
        "member_manifest_sha256",
        "candidate_pool_sha256",
        "census_sha256",
        "current_image_ids_sha256",
        "owner_ledger_sha256",
        "owner_ledger_seal_sha256",
    )
    for replay in replay_rows:
        _require_fields(replay, _REPLAY_FIELDS, artifact="replay record")
        if (
            replay["schema_version"] != REPLAY_RECORD_SCHEMA_VERSION
            or replay["look_id"] != look_id
            or any(
                replay[field]
                != (
                    owner_seal_sha
                    if field == "owner_ledger_seal_sha256"
                    else seal["owner_ledger_sha256"]
                    if field == "owner_ledger_sha256"
                    else expected_common[field]
                )
                for field in common_digest_fields
            )
            or replay["route_count"] != len(EXPECTED_ROUTE_IDS)
            or replay["route_ids"] != list(EXPECTED_ROUTE_IDS)
            or not isinstance(replay["no_op_exact_reproduction"], bool)
        ):
            raise ValueError("replay record common chain/route inventory drift")
        for count_field in ("official_owner_count", "added_owner_count"):
            count = replay[count_field]
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError(f"replay record {count_field} is invalid")
        no_addition = replay["added_owner_count"] == 0
        if replay["no_op_exact_reproduction"] is not no_addition:
            raise ValueError("replay no-op claim differs from added-owner count")
        if no_addition and replay["census_record"] != frozen_census.records[
            str(replay["image_id"])
        ]:
            raise ValueError("no-addition replay differs from frozen census")
        route_identity = replay["route_identity"]
        if not isinstance(route_identity, Mapping) or set(route_identity) != set(
            EXPECTED_ROUTE_IDS
        ):
            raise ValueError("replay route identity inventory drift")
        identity_fields = {
            "candidate_identity_sha256",
            "parser_evidence_sha256",
            "original_row_identity_and_order_sha256",
            "replayed_row_identity_and_order_sha256",
        }
        for route_id, identity in route_identity.items():
            if not isinstance(identity, Mapping):
                raise ValueError(f"replay route identity is invalid: {route_id}")
            _require_fields(
                identity,
                identity_fields,
                artifact=f"replay route identity {route_id}",
            )
            for field in identity_fields:
                _require_sha256(identity[field], field=f"replay {route_id} {field}")
            if (
                identity["original_row_identity_and_order_sha256"]
                != identity["replayed_row_identity_and_order_sha256"]
            ):
                raise ValueError("replay row identity/order drift")
    if no_op_count != sum(bool(row["no_op_exact_reproduction"]) for row in replay_rows):
        raise ValueError("replay receipt no-op count differs from records")
    return (
        ValidatedReplayChain(replay_rows, receipt, expected_ids),
        uncertainty_by_image,
    )


def classify_outcomes(
    *,
    replay_records_jsonl: bytes,
    replay_receipt_json: bytes,
    uncertainty_ledger_jsonl: bytes,
    owner_ledger_seal_json: bytes,
    look_id: str,
    ordered_image_ids: Sequence[str],
    expected_selection_sha256: str,
    expected_member_manifest_sha256: str,
    expected_member_manifest_receipt_sha256: str,
    expected_candidate_pool_sha256: str,
    expected_census_sha256: str,
) -> bytes:
    """Apply the conservative unresolved-as-statistical-success classifier."""

    chain, uncertainty_by_image = _validate_replay_chain(
        replay_records_jsonl=replay_records_jsonl,
        replay_receipt_json=replay_receipt_json,
        uncertainty_ledger_jsonl=uncertainty_ledger_jsonl,
        owner_ledger_seal_json=owner_ledger_seal_json,
        look_id=look_id,
        ordered_image_ids=ordered_image_ids,
        expected_selection_sha256=expected_selection_sha256,
        expected_member_manifest_sha256=expected_member_manifest_sha256,
        expected_member_manifest_receipt_sha256=expected_member_manifest_receipt_sha256,
        expected_candidate_pool_sha256=expected_candidate_pool_sha256,
        expected_census_sha256=expected_census_sha256,
    )
    owner_seal_sha = _sha256_bytes(owner_ledger_seal_json)
    replay_receipt_sha = _sha256_bytes(replay_receipt_json)
    uncertainty_sha = _sha256_bytes(uncertainty_ledger_jsonl)
    current_ids_sha = _ordered_image_ids_sha256(chain.ordered_image_ids)
    outcomes: list[dict[str, Any]] = []
    for replay in chain.rows:
        image_id = str(replay["image_id"])
        uncertainty = uncertainty_by_image[image_id]
        _require_fields(
            uncertainty, _UNCERTAINTY_FIELDS, artifact="uncertainty ledger row"
        )
        if uncertainty["schema_version"] != UNCERTAINTY_LEDGER_SCHEMA_VERSION:
            raise ValueError("uncertainty ledger schema drift")
        axes = uncertainty.get("retained_uncertainty_axes")
        if not isinstance(axes, list) or axes not in ([], list(UNCERTAINTY_AXES)):
            raise ValueError("retained uncertainty axes are noncanonical")
        reasons = uncertainty["owner_universe_uncertainty_reasons"]
        unresolved = uncertainty["unresolved_proposal_dispositions"]
        complete = uncertainty["owner_universe_complete"]
        if (
            not isinstance(complete, bool)
            or not isinstance(reasons, list)
            or reasons != sorted(set(reasons))
            or not set(reasons) <= _OWNER_UNIVERSE_REASONS
            or not isinstance(unresolved, list)
            or any(not isinstance(item, Mapping) for item in unresolved)
        ):
            raise ValueError("uncertainty ledger state is invalid")
        for disposition in unresolved:
            _require_fields(
                disposition,
                _DISPOSITION_FIELDS,
                artifact="unresolved proposal disposition",
            )
            if disposition["adjudication_state"] in {
                "official_duplicate",
                "accepted_missing_owner",
                "out_of_scope",
            }:
                raise ValueError("resolved disposition appears in uncertainty ledger")
        if complete and (axes or reasons or unresolved):
            raise ValueError("complete owner universe retains uncertainty")
        if not complete and (axes != list(UNCERTAINTY_AXES) or not reasons):
            raise ValueError(
                "incomplete owner universe is not conservatively unresolved"
            )
        census_record = replay["census_record"]
        if not isinstance(census_record, Mapping):
            raise ValueError("replay census record is invalid")
        admission = census_record.get("admission")
        if not isinstance(admission, Mapping) or not isinstance(
            admission.get("primary_natural_alias_admitted"), bool
        ):
            raise ValueError("replay census admission state is invalid")
        admitted = admission["primary_natural_alias_admitted"]
        candidates = census_record.get("candidates", [])
        if not isinstance(candidates, list) or any(
            not isinstance(item, Mapping) for item in candidates
        ):
            raise ValueError("replay census candidate inventory is invalid")
        candidate_ids = [str(candidate.get("candidate_id")) for candidate in candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("replay census candidates duplicate an identifier")
        uncertain_candidate_ids = sorted(
            {
                candidate_id
                for candidate_id, candidate in zip(candidate_ids, candidates)
                if census._int_count(candidate.get("unknown_count", 0), "unknown_count")
                > 0
                or census._int_count(
                    candidate.get("ambiguity_count", 0), "ambiguity_count"
                )
                > 0
            },
            key=census._route_sort_key,
        )
        replay_axes = list(UNCERTAINTY_AXES) if uncertain_candidate_ids else []
        combined_axes = list(UNCERTAINTY_AXES) if (axes or replay_axes) else []
        if admitted:
            outcome = "actual_admission"
        elif combined_axes or not complete:
            outcome = "potential_admission_unresolved"
        else:
            outcome = "definitive_non_admission"
        outcomes.append(
            {
                "schema_version": OUTCOME_SCHEMA_VERSION,
                "look_id": look_id,
                "image_id": image_id,
                "selection_sha256": expected_selection_sha256,
                "member_manifest_sha256": expected_member_manifest_sha256,
                "current_image_ids_sha256": current_ids_sha,
                "owner_ledger_seal_sha256": owner_seal_sha,
                "replay_records_sha256": _sha256_bytes(replay_records_jsonl),
                "replay_receipt_sha256": replay_receipt_sha,
                "uncertainty_ledger_sha256": uncertainty_sha,
                "outcome": outcome,
                "statistical_success": outcome != "definitive_non_admission",
                "primary_natural_alias_admitted": admitted,
                "owner_universe_complete": complete,
                "review_uncertainty_axes": axes,
                "replay_uncertainty_axes": replay_axes,
                "retained_uncertainty_axes": combined_axes,
                "uncertain_candidate_ids": uncertain_candidate_ids,
            }
        )
    return _jsonl_bytes(outcomes)


def _validate_outcome_rows(
    payload: bytes,
    *,
    look_id: str,
    ordered_image_ids: Sequence[str],
    expected_selection_sha256: str,
    expected_member_manifest_sha256: str,
) -> tuple[Mapping[str, Any], ...]:
    rows = tuple(_parse_jsonl(payload, artifact=f"{look_id} outcomes"))
    expected_ids = tuple(ordered_image_ids)
    observed_ids = tuple(
        _canonical_image_id(row.get("image_id"), field="outcome image_id")
        for row in rows
    )
    if len(observed_ids) != len(set(observed_ids)):
        raise ValueError("outcomes duplicate an image")
    if observed_ids != expected_ids:
        raise ValueError("outcomes image set/order differs from the look")
    current_hash = _ordered_image_ids_sha256(expected_ids)
    common: dict[str, Any] | None = None
    for row in rows:
        _require_fields(row, _OUTCOME_FIELDS, artifact="outcome row")
        if (
            row["schema_version"] != OUTCOME_SCHEMA_VERSION
            or row["look_id"] != look_id
            or row["selection_sha256"] != expected_selection_sha256
            or row["member_manifest_sha256"] != expected_member_manifest_sha256
            or row["current_image_ids_sha256"] != current_hash
            or row["outcome"] not in OUTCOMES
            or row["statistical_success"]
            is not (row["outcome"] != "definitive_non_admission")
        ):
            raise ValueError("outcome schema/common chain drift")
        row_common = {
            field: row[field]
            for field in (
                "owner_ledger_seal_sha256",
                "replay_records_sha256",
                "replay_receipt_sha256",
                "uncertainty_ledger_sha256",
            )
        }
        for field, digest in row_common.items():
            _require_sha256(digest, field=f"outcome {field}")
        if common is None:
            common = row_common
        elif common != row_common:
            raise ValueError("outcomes mix incompatible evidence chains")
    return rows


def _validate_final_look_receipt(
    payload: bytes,
    *,
    expected_look_id: str,
    expected_selection_sha256: str,
    expected_member_manifest_sha256: str,
) -> Mapping[str, Any]:
    receipt = _parse_json(payload, artifact="final look receipt")
    _require_fields(receipt, _FINAL_LOOK_RECEIPT_FIELDS, artifact="final look receipt")
    if (
        receipt["schema_version"] != FINAL_LOOK_RECEIPT_SCHEMA_VERSION
        or receipt["terminal_status"] != "finalized_immutable"
        or receipt["look_id"] != expected_look_id
        or receipt["selection_sha256"] != expected_selection_sha256
        or receipt["member_manifest_sha256"] != expected_member_manifest_sha256
        or receipt["filesystem_immutable"] is not True
        or receipt["finalized_atomically"] is not True
    ):
        raise ValueError("final look receipt terminal/common binding drift")
    root = Path(str(receipt["final_root"])).expanduser().resolve(strict=True)
    if not root.is_dir() or stat.S_IMODE(root.stat().st_mode) != 0o555:
        raise ValueError("final look root is absent or mutable")
    receipt_path = root / "look-receipt.json"
    if (
        receipt_path.read_bytes() != payload
        or stat.S_IMODE(receipt_path.stat().st_mode) != 0o444
    ):
        raise ValueError("final look receipt readback/mode drift")
    inventory = receipt["artifact_inventory"]
    if not isinstance(inventory, Mapping) or not inventory:
        raise ValueError("final look artifact inventory is invalid")
    expected_inventory_names = set(FINAL_LOOK_ARTIFACT_NAMES) | {"source-snapshot.json"}
    if set(inventory) != expected_inventory_names:
        raise ValueError("final look receipt artifact-name inventory drift")
    if receipt["artifact_inventory_sha256"] != _sha256_bytes(
        canonical_json_text(inventory).encode("utf-8")
    ):
        raise ValueError("final look artifact inventory digest drift")
    if set(path.name for path in root.iterdir()) != set(inventory) | {
        "look-receipt.json"
    }:
        raise ValueError("final look directory inventory drift")
    for name, record in inventory.items():
        if (
            not isinstance(name, str)
            or "/" in name
            or name.startswith(".")
            or not isinstance(record, Mapping)
        ):
            raise ValueError("final look inventory entry is invalid")
        _require_fields(
            record, {"size_bytes", "sha256"}, artifact="final look inventory entry"
        )
        path = root / name
        if (
            not path.is_file()
            or stat.S_IMODE(path.stat().st_mode) != 0o444
            or path.stat().st_size != record["size_bytes"]
            or _sha256_file(path) != record["sha256"]
        ):
            raise ValueError(f"final look artifact readback/mode drift: {name}")
    snapshot_record = inventory.get("source-snapshot.json")
    decision_record = inventory.get("sequential-decision.json")
    if (
        not isinstance(snapshot_record, Mapping)
        or not isinstance(decision_record, Mapping)
        or snapshot_record.get("sha256") != receipt["source_snapshot_sha256"]
        or decision_record.get("sha256") != receipt["sequential_decision_sha256"]
    ):
        raise ValueError("final look receipt terminal artifact binding drift")
    direct_hash_bindings = {
        "selection.json": "selection_sha256",
        "member-manifest.jsonl": "member_manifest_sha256",
        "member-manifest-receipt.json": "member_manifest_receipt_sha256",
    }
    for artifact_name, receipt_field in direct_hash_bindings.items():
        record = inventory[artifact_name]
        if (
            not isinstance(record, Mapping)
            or record.get("sha256") != receipt[receipt_field]
        ):
            raise ValueError("final look receipt selection/member snapshot drift")
    source_snapshot = _parse_json(
        (root / "source-snapshot.json").read_bytes(),
        artifact="finalized source snapshot",
    )
    _require_fields(
        source_snapshot,
        {"schema_version", "look_id", "source_bindings", "source_bindings_sha256"},
        artifact="finalized source snapshot",
    )
    source_bindings = source_snapshot["source_bindings"]
    if (
        source_snapshot["schema_version"] != SOURCE_SNAPSHOT_SCHEMA_VERSION
        or source_snapshot["look_id"] != expected_look_id
        or not isinstance(source_bindings, Mapping)
        or source_snapshot["source_bindings_sha256"]
        != _sha256_bytes(canonical_json_text(source_bindings).encode("utf-8"))
    ):
        raise ValueError("finalized source snapshot terminal binding drift")
    for name, source in source_bindings.items():
        if not isinstance(name, str) or not isinstance(source, Mapping):
            raise ValueError("finalized source snapshot entry is invalid")
        kind = source.get("kind")
        if kind == "file":
            _require_fields(
                source,
                {"kind", "path", "size_bytes", "sha256"},
                artifact=f"finalized file source {name}",
            )
        elif kind == "manifest_set":
            _require_fields(
                source,
                {"kind", "path", "file_count", "inventory", "sha256"},
                artifact=f"finalized manifest-set source {name}",
            )
            source_inventory = source["inventory"]
            if (
                not isinstance(source_inventory, list)
                or source["file_count"] != len(source_inventory)
                or source["sha256"]
                != _sha256_bytes(canonical_json_text(source_inventory).encode("utf-8"))
            ):
                raise ValueError("finalized manifest-set source inventory drift")
        else:
            raise ValueError("finalized source snapshot kind drift")
        _require_sha256(source["sha256"], field=f"finalized source {name} sha256")
    for source_name, receipt_field in (
        ("candidate_pool", "candidate_pool_sha256"),
        ("census", "census_sha256"),
    ):
        source = source_bindings.get(source_name)
        if (
            not isinstance(source, Mapping)
            or source.get("sha256") != receipt[receipt_field]
        ):
            raise ValueError("finalized source snapshot candidate/census drift")
    decision = _parse_json(
        (root / "sequential-decision.json").read_bytes(),
        artifact="finalized sequential decision",
    )
    _require_fields(
        decision,
        _SEQUENTIAL_DECISION_FIELDS,
        artifact="finalized sequential decision",
    )
    if (
        decision["schema_version"] != SEQUENTIAL_DECISION_SCHEMA_VERSION
        or decision["look_id"] != expected_look_id
        or decision["selection_sha256"] != receipt["selection_sha256"]
        or decision["member_manifest_sha256"] != receipt["member_manifest_sha256"]
        or decision["member_manifest_receipt_sha256"]
        != receipt["member_manifest_receipt_sha256"]
        or decision["candidate_pool_sha256"] != receipt["candidate_pool_sha256"]
        or decision["census_sha256"] != receipt["census_sha256"]
        or decision["current_image_ids_sha256"] != receipt["current_image_ids_sha256"]
        or decision["prior_look_receipt_sha256"] != receipt["prior_look_receipt_sha256"]
    ):
        raise ValueError("finalized sequential decision common chain drift")
    if expected_look_id == "look_one":
        valid_decision_state = decision[
            "terminal_status"
        ] == "sealed_before_next_look" and (
            (
                decision["decision"] == "continue_to_look_two"
                and decision["look_two_authorized"] is True
                and decision["closed_test_complete"] is False
            )
            or (
                decision["decision"] == "reject_null"
                and decision["look_two_authorized"] is False
                and decision["closed_test_complete"] is True
            )
        )
    else:
        valid_decision_state = (
            decision["terminal_status"] == "sealed_terminal"
            and decision["decision"] in {"reject_null", "do_not_reject_null"}
            and decision["look_two_authorized"] is False
            and decision["closed_test_complete"] is True
        )
    if not valid_decision_state:
        raise ValueError("finalized sequential decision state drift")
    queue_manifest = _parse_json(
        (root / "review-queue-manifest.json").read_bytes(),
        artifact="finalized queue manifest",
    )
    _require_fields(
        queue_manifest,
        _QUEUE_MANIFEST_FIELDS,
        artifact="finalized queue manifest",
    )
    if any(
        queue_manifest[field] != receipt[field]
        for field in (
            "look_id",
            "selection_sha256",
            "member_manifest_sha256",
            "candidate_pool_sha256",
            "census_sha256",
            "current_image_ids_sha256",
            "prior_look_receipt_sha256",
        )
    ):
        raise ValueError("finalized queue manifest common chain drift")
    for field in (
        "selection_sha256",
        "member_manifest_sha256",
        "member_manifest_receipt_sha256",
        "candidate_pool_sha256",
        "census_sha256",
        "current_image_ids_sha256",
        "source_snapshot_sha256",
        "sequential_decision_sha256",
        "artifact_inventory_sha256",
    ):
        _require_sha256(receipt[field], field=f"final look receipt {field}")
    prior = receipt["prior_look_receipt_sha256"]
    if expected_look_id == "look_one":
        if prior is not None:
            raise ValueError("finalized look one carries a prior receipt")
    else:
        _require_sha256(prior, field="final look prior receipt")
    return receipt


def _validate_look_two_authorization(
    *,
    prior_look_decision_json: bytes | None,
    prior_look_receipt_json: bytes | None,
    expected_selection_sha256: str,
    expected_member_manifest_sha256: str,
) -> str:
    if prior_look_decision_json is None or prior_look_receipt_json is None:
        raise ValueError("look two requires finalized look-one authorization")
    receipt = _validate_final_look_receipt(
        prior_look_receipt_json,
        expected_look_id="look_one",
        expected_selection_sha256=expected_selection_sha256,
        expected_member_manifest_sha256=expected_member_manifest_sha256,
    )
    root = Path(str(receipt["final_root"])).expanduser().resolve(strict=True)
    decision_path = root / "sequential-decision.json"
    if (
        decision_path.read_bytes() != prior_look_decision_json
        or _sha256_bytes(prior_look_decision_json)
        != receipt["sequential_decision_sha256"]
    ):
        raise ValueError("look-one decision differs from finalized receipt")
    decision = _parse_json(
        prior_look_decision_json, artifact="look-one sequential decision"
    )
    _require_fields(
        decision,
        _SEQUENTIAL_DECISION_FIELDS,
        artifact="look-one sequential decision",
    )
    if (
        decision["schema_version"] != SEQUENTIAL_DECISION_SCHEMA_VERSION
        or decision["terminal_status"] != "sealed_before_next_look"
        or decision["look_id"] != "look_one"
        or decision["selection_sha256"] != expected_selection_sha256
        or decision["member_manifest_sha256"] != expected_member_manifest_sha256
        or decision["decision"] != "continue_to_look_two"
        or decision["look_two_authorized"] is not True
        or decision["closed_test_complete"] is not False
    ):
        raise ValueError("look-one decision does not authorize look two")
    return _sha256_bytes(prior_look_receipt_json)


def build_sequential_decision(
    *,
    look_id: str,
    outcomes_jsonl: bytes,
    selection_json: bytes,
    expected_selection_sha256: str,
    selection_path: Path,
    member_manifest_jsonl: bytes,
    expected_member_manifest_sha256: str,
    member_manifest_receipt_json: bytes,
    expected_member_manifest_receipt_sha256: str,
    expected_candidate_pool_sha256: str,
    expected_census_sha256: str,
    prior_look_decision_json: bytes | None = None,
    prior_look_receipt_json: bytes | None = None,
) -> bytes:
    """Seal one exact closed-test decision before any later look may start."""

    if look_id not in LOOK_IDS:
        raise ValueError("unknown sequential look")
    selection = _validate_selection(
        selection_json,
        expected_sha256=expected_selection_sha256,
        selection_path=selection_path,
        member_manifest_jsonl=member_manifest_jsonl,
        expected_member_manifest_sha256=expected_member_manifest_sha256,
        member_manifest_receipt_json=member_manifest_receipt_json,
        expected_member_manifest_receipt_sha256=expected_member_manifest_receipt_sha256,
    )
    current_ids = (
        selection.look_one_image_ids
        if look_id == "look_one"
        else selection.look_two_additional_image_ids
    )
    selection_sources = selection.source_artifacts
    if (
        selection_sources["candidate_pool"]["sha256"] != expected_candidate_pool_sha256
        or selection_sources["census"]["sha256"] != expected_census_sha256
    ):
        raise ValueError("sequential source digests differ from selection")
    current_rows = _validate_outcome_rows(
        outcomes_jsonl,
        look_id=look_id,
        ordered_image_ids=current_ids,
        expected_selection_sha256=expected_selection_sha256,
        expected_member_manifest_sha256=expected_member_manifest_sha256,
    )
    if look_id == "look_one":
        if prior_look_decision_json is not None or prior_look_receipt_json is not None:
            raise ValueError("look one cannot carry prior-look artifacts")
        prior_receipt_sha: str | None = None
        cumulative_rows = list(current_rows)
        cumulative_ids = selection.look_one_image_ids
        terminal_status = "sealed_before_next_look"
        sample_size = LOOK_ONE_COUNT
    else:
        prior_receipt_sha = _validate_look_two_authorization(
            prior_look_decision_json=prior_look_decision_json,
            prior_look_receipt_json=prior_look_receipt_json,
            expected_selection_sha256=expected_selection_sha256,
            expected_member_manifest_sha256=expected_member_manifest_sha256,
        )
        if prior_look_receipt_json is None:
            raise AssertionError("validated look-two receipt unexpectedly absent")
        prior_receipt = _parse_json(
            prior_look_receipt_json, artifact="look-one receipt"
        )
        prior_root = Path(str(prior_receipt["final_root"])).resolve(strict=True)
        prior_outcomes = (prior_root / "outcomes.jsonl").read_bytes()
        prior_rows = _validate_outcome_rows(
            prior_outcomes,
            look_id="look_one",
            ordered_image_ids=selection.look_one_image_ids,
            expected_selection_sha256=expected_selection_sha256,
            expected_member_manifest_sha256=expected_member_manifest_sha256,
        )
        cumulative_rows = [*prior_rows, *current_rows]
        cumulative_ids = selection.look_two_cumulative_image_ids
        if tuple(str(row["image_id"]) for row in cumulative_rows) != cumulative_ids:
            raise ValueError("look-two cumulative outcome order drift")
        terminal_status = "sealed_terminal"
        sample_size = SELECTION_COUNT
    success_count = sum(bool(row["statistical_success"]) for row in cumulative_rows)
    cutoff = selection.cutoff_by_look[
        "look_one" if look_id == "look_one" else "look_two_cumulative"
    ]
    exact_cutoff, boundary = _exact_hypergeometric_cutoff(
        population_size=len(selection.ordered_pool_image_ids),
        success_count=NULL_SUCCESS_COUNT,
        sample_size=sample_size,
    )
    if cutoff != exact_cutoff:
        raise ValueError("selection cutoff differs at sequential decision")
    rejected = success_count <= cutoff
    if look_id == "look_one":
        decision = "reject_null" if rejected else "continue_to_look_two"
        look_two_authorized = not rejected
        closed_test_complete = rejected
    else:
        decision = "reject_null" if rejected else "do_not_reject_null"
        look_two_authorized = False
        closed_test_complete = True
    cumulative_payload = _jsonl_bytes(cumulative_rows)
    record = {
        "schema_version": SEQUENTIAL_DECISION_SCHEMA_VERSION,
        "terminal_status": terminal_status,
        "look_id": look_id,
        "selection_sha256": expected_selection_sha256,
        "member_manifest_sha256": expected_member_manifest_sha256,
        "member_manifest_receipt_sha256": expected_member_manifest_receipt_sha256,
        "candidate_pool_sha256": expected_candidate_pool_sha256,
        "census_sha256": expected_census_sha256,
        "current_image_ids_sha256": _ordered_image_ids_sha256(current_ids),
        "current_outcomes_sha256": _sha256_bytes(outcomes_jsonl),
        "prior_look_receipt_sha256": prior_receipt_sha,
        "cumulative_image_ids_sha256": _ordered_image_ids_sha256(cumulative_ids),
        "cumulative_outcomes_sha256": _sha256_bytes(cumulative_payload),
        "cumulative_sample_size": sample_size,
        "statistical_success_count": success_count,
        "rejection_success_count_max": cutoff,
        "boundary_probability_numerator": boundary.numerator,
        "boundary_probability_denominator": boundary.denominator,
        "decision": decision,
        "look_two_authorized": look_two_authorized,
        "closed_test_complete": closed_test_complete,
    }
    return _json_bytes(record)


def _source_snapshot_bytes(
    *, look_id: str, source_bindings: Mapping[str, Path]
) -> bytes:
    expected_names = {
        "packet",
        "ontology",
        "candidate_pool",
        "census",
        "frozen_contract",
        "source_panel",
        "sampled_panel",
    }
    if set(source_bindings) != expected_names:
        raise ValueError("final source-binding inventory drift")
    bindings: dict[str, Any] = {}
    for name in sorted(source_bindings):
        path = source_bindings[name].expanduser().resolve(strict=True)
        if path.is_file():
            bindings[name] = {
                "kind": "file",
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        elif path.is_dir():
            inventory, digest = _directory_manifest(path)
            bindings[name] = {
                "kind": "manifest_set",
                "path": str(path),
                "file_count": len(inventory),
                "inventory": inventory,
                "sha256": digest,
            }
        else:
            raise ValueError(f"final source binding is unsupported: {name}")
    return _json_bytes(
        {
            "schema_version": SOURCE_SNAPSHOT_SCHEMA_VERSION,
            "look_id": look_id,
            "source_bindings": bindings,
            "source_bindings_sha256": _sha256_bytes(
                canonical_json_text(bindings).encode("utf-8")
            ),
        }
    )


def _validate_final_bundle(
    *,
    adapter: Mapping[str, Any],
    look_id: str,
    artifact_payloads: Mapping[str, bytes],
    source_snapshot_json: bytes,
    selection_path: Path,
    member_manifest_path: Path,
    member_manifest_receipt_path: Path,
    expected_selection_sha256: str,
    expected_member_manifest_sha256: str,
    expected_member_manifest_receipt_sha256: str,
    expected_candidate_pool_sha256: str,
    expected_census_sha256: str,
    prior_look_decision_json: bytes | None,
    prior_look_receipt_json: bytes | None,
) -> ValidatedSelection:
    if set(artifact_payloads) != FINAL_LOOK_ARTIFACT_NAMES:
        difference = sorted(set(artifact_payloads) ^ set(FINAL_LOOK_ARTIFACT_NAMES))
        raise ValueError(f"final look artifact inventory differs: {difference}")
    if (
        selection_path.expanduser().resolve(strict=True).read_bytes()
        != artifact_payloads["selection.json"]
        or member_manifest_path.expanduser().resolve(strict=True).read_bytes()
        != artifact_payloads["member-manifest.jsonl"]
        or member_manifest_receipt_path.expanduser().resolve(strict=True).read_bytes()
        != artifact_payloads["member-manifest-receipt.json"]
    ):
        raise ValueError("final bundle upstream selection/member snapshot drift")
    selection = _validate_selection(
        artifact_payloads["selection.json"],
        expected_sha256=expected_selection_sha256,
        selection_path=selection_path,
        member_manifest_jsonl=artifact_payloads["member-manifest.jsonl"],
        expected_member_manifest_sha256=expected_member_manifest_sha256,
        member_manifest_receipt_json=artifact_payloads["member-manifest-receipt.json"],
        expected_member_manifest_receipt_sha256=expected_member_manifest_receipt_sha256,
    )
    if (
        selection.journal_terminal_path.read_bytes()
        != artifact_payloads["entropy-journal.json"]
    ):
        raise ValueError("final bundle entropy-journal snapshot drift")
    source_snapshot = _parse_json(
        source_snapshot_json, artifact="final source snapshot"
    )
    _require_fields(
        source_snapshot,
        {"schema_version", "look_id", "source_bindings", "source_bindings_sha256"},
        artifact="final source snapshot",
    )
    if (
        source_snapshot["schema_version"] != SOURCE_SNAPSHOT_SCHEMA_VERSION
        or source_snapshot["look_id"] != look_id
        or source_snapshot["source_bindings_sha256"]
        != _sha256_bytes(
            canonical_json_text(source_snapshot["source_bindings"]).encode("utf-8")
        )
    ):
        raise ValueError("final source snapshot schema/digest drift")
    source_bindings = source_snapshot["source_bindings"]
    if not isinstance(source_bindings, Mapping):
        raise ValueError("final source snapshot bindings are invalid")
    queue_manifest = _validate_queue_manifest(
        artifact_payloads["review-queue-manifest.json"],
        review_queue_jsonl=artifact_payloads["review-queue.jsonl"],
        official_owner_ledger_jsonl=artifact_payloads["official-owner-ledger.jsonl"],
    )
    current_ids = (
        selection.look_one_image_ids
        if look_id == "look_one"
        else selection.look_two_additional_image_ids
    )
    expected_prior_sha: str | None
    if look_id == "look_one":
        if prior_look_decision_json is not None or prior_look_receipt_json is not None:
            raise ValueError("look-one final bundle carries prior artifacts")
        expected_prior_sha = None
    else:
        expected_prior_sha = _validate_look_two_authorization(
            prior_look_decision_json=prior_look_decision_json,
            prior_look_receipt_json=prior_look_receipt_json,
            expected_selection_sha256=expected_selection_sha256,
            expected_member_manifest_sha256=expected_member_manifest_sha256,
        )
    if (
        queue_manifest["look_id"] != look_id
        or queue_manifest["private_sample_order_image_ids"] != list(current_ids)
        or queue_manifest["selection_sha256"] != expected_selection_sha256
        or queue_manifest["member_manifest_sha256"] != expected_member_manifest_sha256
        or queue_manifest["member_manifest_receipt_sha256"]
        != expected_member_manifest_receipt_sha256
        or queue_manifest["candidate_pool_sha256"] != expected_candidate_pool_sha256
        or queue_manifest["census_sha256"] != expected_census_sha256
        or queue_manifest["prior_look_receipt_sha256"] != expected_prior_sha
    ):
        raise ValueError("final bundle queue private chain drift")
    source_expected = {
        "packet": queue_manifest["packet_sha256"],
        "ontology": queue_manifest["ontology_sha256"],
        "candidate_pool": expected_candidate_pool_sha256,
        "census": expected_census_sha256,
        "frozen_contract": selection.frozen_contract_sha256,
        "source_panel": selection.source_artifacts["source_panel"]["sha256"],
        "sampled_panel": selection.source_artifacts["sampled_panel"]["sha256"],
    }
    if set(source_bindings) != set(source_expected):
        raise ValueError("final source snapshot differs from evidence chain")
    for name, digest in source_expected.items():
        source_record = source_bindings[name]
        if (
            not isinstance(source_record, Mapping)
            or source_record.get("sha256") != digest
        ):
            raise ValueError("final source snapshot differs from evidence chain")
    packet_sha = queue_manifest["packet_sha256"]
    ontology_sha = queue_manifest["ontology_sha256"]
    queue_sha = _sha256_bytes(artifact_payloads["review-queue.jsonl"])
    for role in REVIEWER_ROLES:
        role_payload = artifact_payloads[f"{role}.jsonl"]
        role_seal = artifact_payloads[f"{role}-seal.json"]
        validate_role_artifact(
            role_artifact_jsonl=role_payload,
            reviewer_role_identifier=role,
            review_queue_jsonl=artifact_payloads["review-queue.jsonl"],
            expected_review_queue_sha256=queue_sha,
            expected_packet_sha256=packet_sha,
            expected_ontology_sha256=ontology_sha,
        )
        _validate_role_seal(
            role_seal,
            role=role,
            role_payload=role_payload,
            queue_sha256=queue_sha,
            packet_sha256=packet_sha,
            ontology_sha256=ontology_sha,
        )
    expected_adjudication_queue = build_adjudication_queue(
        review_queue_jsonl=artifact_payloads["review-queue.jsonl"],
        expected_review_queue_sha256=queue_sha,
        reviewer_one_jsonl=artifact_payloads["reviewer-one.jsonl"],
        reviewer_one_seal_json=artifact_payloads["reviewer-one-seal.json"],
        reviewer_two_jsonl=artifact_payloads["reviewer-two.jsonl"],
        reviewer_two_seal_json=artifact_payloads["reviewer-two-seal.json"],
        official_owner_ledger_jsonl=artifact_payloads["official-owner-ledger.jsonl"],
        expected_official_owner_ledger_sha256=_sha256_bytes(
            artifact_payloads["official-owner-ledger.jsonl"]
        ),
        expected_packet_sha256=packet_sha,
        expected_ontology_sha256=ontology_sha,
    )
    if expected_adjudication_queue != artifact_payloads["adjudication-queue.jsonl"]:
        raise ValueError("final adjudication queue does not reproduce exactly")
    expected_ledger = assemble_owner_ledger(
        adjudication_queue_jsonl=artifact_payloads["adjudication-queue.jsonl"],
        expected_adjudication_queue_sha256=_sha256_bytes(
            artifact_payloads["adjudication-queue.jsonl"]
        ),
        adjudicator_decisions_jsonl=artifact_payloads["adjudication.jsonl"],
        review_queue_manifest_json=artifact_payloads["review-queue-manifest.json"],
        expected_review_queue_manifest_sha256=_sha256_bytes(
            artifact_payloads["review-queue-manifest.json"]
        ),
    )
    for name, expected in (
        ("adjudication.jsonl", expected_ledger.adjudication_jsonl),
        ("owner-ledger.jsonl", expected_ledger.owner_ledger_jsonl),
        ("uncertainty-ledger.jsonl", expected_ledger.uncertainty_ledger_jsonl),
        ("owner-ledger-seal.json", expected_ledger.owner_ledger_seal_json),
    ):
        if artifact_payloads[name] != expected:
            raise ValueError(f"final owner-ledger chain differs: {name}")
    candidate_binding = source_bindings["candidate_pool"]
    census_binding = source_bindings["census"]
    if (
        not isinstance(candidate_binding, Mapping)
        or candidate_binding.get("kind") != "file"
        or not isinstance(census_binding, Mapping)
        or census_binding.get("kind") != "file"
    ):
        raise ValueError("final replay sources must be exact files")
    candidate_pool_path = Path(str(candidate_binding.get("path")))
    census_path = Path(str(census_binding.get("path")))
    expected_replay = replay_owner_ledger(
        adapter=adapter,
        look_id=look_id,
        selection_json=artifact_payloads["selection.json"],
        expected_selection_sha256=expected_selection_sha256,
        selection_path=selection_path,
        member_manifest_jsonl=artifact_payloads["member-manifest.jsonl"],
        expected_member_manifest_sha256=expected_member_manifest_sha256,
        member_manifest_receipt_json=artifact_payloads[
            "member-manifest-receipt.json"
        ],
        expected_member_manifest_receipt_sha256=(
            expected_member_manifest_receipt_sha256
        ),
        candidate_pool_path=candidate_pool_path,
        expected_candidate_pool_sha256=expected_candidate_pool_sha256,
        census_path=census_path,
        expected_census_path=census_path,
        expected_census_sha256=expected_census_sha256,
        owner_ledger_jsonl=artifact_payloads["owner-ledger.jsonl"],
        owner_ledger_seal_json=artifact_payloads["owner-ledger-seal.json"],
    )
    for name, expected in (
        ("replay-records.jsonl", expected_replay.replay_records_jsonl),
        ("replay-receipt.json", expected_replay.replay_receipt_json),
    ):
        if artifact_payloads[name] != expected:
            raise ValueError(f"final global replay does not reproduce exactly: {name}")
    _validate_replay_chain(
        replay_records_jsonl=artifact_payloads["replay-records.jsonl"],
        replay_receipt_json=artifact_payloads["replay-receipt.json"],
        uncertainty_ledger_jsonl=artifact_payloads["uncertainty-ledger.jsonl"],
        owner_ledger_seal_json=artifact_payloads["owner-ledger-seal.json"],
        look_id=look_id,
        ordered_image_ids=current_ids,
        expected_selection_sha256=expected_selection_sha256,
        expected_member_manifest_sha256=expected_member_manifest_sha256,
        expected_member_manifest_receipt_sha256=expected_member_manifest_receipt_sha256,
        expected_candidate_pool_sha256=expected_candidate_pool_sha256,
        expected_census_sha256=expected_census_sha256,
    )
    expected_outcomes = classify_outcomes(
        replay_records_jsonl=artifact_payloads["replay-records.jsonl"],
        replay_receipt_json=artifact_payloads["replay-receipt.json"],
        uncertainty_ledger_jsonl=artifact_payloads["uncertainty-ledger.jsonl"],
        owner_ledger_seal_json=artifact_payloads["owner-ledger-seal.json"],
        look_id=look_id,
        ordered_image_ids=current_ids,
        expected_selection_sha256=expected_selection_sha256,
        expected_member_manifest_sha256=expected_member_manifest_sha256,
        expected_member_manifest_receipt_sha256=expected_member_manifest_receipt_sha256,
        expected_candidate_pool_sha256=expected_candidate_pool_sha256,
        expected_census_sha256=expected_census_sha256,
    )
    if expected_outcomes != artifact_payloads["outcomes.jsonl"]:
        raise ValueError("final outcome classifier does not reproduce exactly")
    expected_decision = build_sequential_decision(
        look_id=look_id,
        outcomes_jsonl=artifact_payloads["outcomes.jsonl"],
        selection_json=artifact_payloads["selection.json"],
        expected_selection_sha256=expected_selection_sha256,
        selection_path=selection_path,
        member_manifest_jsonl=artifact_payloads["member-manifest.jsonl"],
        expected_member_manifest_sha256=expected_member_manifest_sha256,
        member_manifest_receipt_json=artifact_payloads["member-manifest-receipt.json"],
        expected_member_manifest_receipt_sha256=expected_member_manifest_receipt_sha256,
        expected_candidate_pool_sha256=expected_candidate_pool_sha256,
        expected_census_sha256=expected_census_sha256,
        prior_look_decision_json=prior_look_decision_json,
        prior_look_receipt_json=prior_look_receipt_json,
    )
    if expected_decision != artifact_payloads["sequential-decision.json"]:
        raise ValueError("final sequential decision does not reproduce exactly")
    return selection


def _remove_created_tree(path: Path) -> None:
    if not path.exists():
        return
    for entry in path.rglob("*"):
        if entry.is_file():
            entry.chmod(0o600)
        elif entry.is_dir():
            entry.chmod(0o700)
    path.chmod(0o700)
    shutil.rmtree(path)


def finalize_look(
    *,
    adapter: Mapping[str, Any],
    final_root: Path,
    look_id: str,
    artifact_payloads: Mapping[str, bytes],
    source_bindings: Mapping[str, Path],
    selection_path: Path,
    member_manifest_path: Path,
    member_manifest_receipt_path: Path,
    expected_selection_sha256: str,
    expected_member_manifest_sha256: str,
    expected_member_manifest_receipt_sha256: str,
    expected_candidate_pool_sha256: str,
    expected_census_sha256: str,
    prior_look_decision_json: bytes | None = None,
    prior_look_receipt_json: bytes | None = None,
) -> bytes:
    """Validate, snapshot, fsync, and atomically publish one immutable look."""

    if look_id not in LOOK_IDS:
        raise ValueError("unknown finalization look")
    parent = final_root.expanduser().parent.resolve(strict=True)
    target = parent / final_root.name
    if target.exists():
        raise FileExistsError(f"refusing to replace finalized look root: {target}")
    source_snapshot = _source_snapshot_bytes(
        look_id=look_id, source_bindings=source_bindings
    )
    selection = _validate_final_bundle(
        adapter=adapter,
        look_id=look_id,
        artifact_payloads=artifact_payloads,
        source_snapshot_json=source_snapshot,
        selection_path=selection_path,
        member_manifest_path=member_manifest_path,
        member_manifest_receipt_path=member_manifest_receipt_path,
        expected_selection_sha256=expected_selection_sha256,
        expected_member_manifest_sha256=expected_member_manifest_sha256,
        expected_member_manifest_receipt_sha256=expected_member_manifest_receipt_sha256,
        expected_candidate_pool_sha256=expected_candidate_pool_sha256,
        expected_census_sha256=expected_census_sha256,
        prior_look_decision_json=prior_look_decision_json,
        prior_look_receipt_json=prior_look_receipt_json,
    )
    current_ids = (
        selection.look_one_image_ids
        if look_id == "look_one"
        else selection.look_two_additional_image_ids
    )
    prior_receipt_sha = (
        None
        if prior_look_receipt_json is None
        else _sha256_bytes(prior_look_receipt_json)
    )
    publish_payloads = dict(artifact_payloads)
    publish_payloads["source-snapshot.json"] = source_snapshot
    inventory = {
        name: {"size_bytes": len(payload), "sha256": _sha256_bytes(payload)}
        for name, payload in sorted(publish_payloads.items())
    }
    receipt = _json_bytes(
        {
            "schema_version": FINAL_LOOK_RECEIPT_SCHEMA_VERSION,
            "terminal_status": "finalized_immutable",
            "look_id": look_id,
            "final_root": str(target),
            "selection_sha256": expected_selection_sha256,
            "member_manifest_sha256": expected_member_manifest_sha256,
            "member_manifest_receipt_sha256": expected_member_manifest_receipt_sha256,
            "candidate_pool_sha256": expected_candidate_pool_sha256,
            "census_sha256": expected_census_sha256,
            "current_image_ids_sha256": _ordered_image_ids_sha256(current_ids),
            "prior_look_receipt_sha256": prior_receipt_sha,
            "source_snapshot_sha256": _sha256_bytes(source_snapshot),
            "sequential_decision_sha256": _sha256_bytes(
                artifact_payloads["sequential-decision.json"]
            ),
            "artifact_inventory": inventory,
            "artifact_inventory_sha256": _sha256_bytes(
                canonical_json_text(inventory).encode("utf-8")
            ),
            "filesystem_immutable": True,
            "finalized_atomically": True,
        }
    )
    staging = parent / f".{target.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(mode=0o700)
    renamed = False
    try:
        for name, payload in {**publish_payloads, "look-receipt.json": receipt}.items():
            path = staging / name
            with path.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            if path.read_bytes() != payload:
                raise OSError(f"final look staging readback drift: {name}")
        directory = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        for path in staging.iterdir():
            path.chmod(0o444)
        staging.chmod(0o555)
        os.rename(staging, target)
        renamed = True
        parent_directory = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(parent_directory)
        finally:
            os.close(parent_directory)
        _validate_final_look_receipt(
            receipt,
            expected_look_id=look_id,
            expected_selection_sha256=expected_selection_sha256,
            expected_member_manifest_sha256=expected_member_manifest_sha256,
        )
    except BaseException:
        if renamed and target.exists():
            quarantine = parent / f".{target.name}.quarantine-{uuid.uuid4().hex}"
            target.chmod(0o755)
            for path in target.iterdir():
                path.chmod(0o644)
            os.rename(target, quarantine)
            parent_directory = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(parent_directory)
            finally:
                os.close(parent_directory)
        elif staging.exists():
            _remove_created_tree(staging)
        raise
    return receipt


def _read_census_records(
    path: Path,
    image_ids: Sequence[str],
    *,
    expected_path: Path,
    expected_sha256: str,
) -> dict[str, Mapping[str, Any]]:
    return dict(
        load_frozen_census_records(
            census_path=path,
            expected_census_path=expected_path,
            expected_census_sha256=expected_sha256,
            selected_image_ids=image_ids,
        ).records
    )


def _publish_once(root: Path, artifacts: Mapping[str, bytes]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    targets = {name: root / name for name in artifacts}
    existing = [str(path) for path in targets.values() if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to replace artifact(s): {existing}")
    temporary: list[Path] = []
    published: list[Path] = []
    try:
        for name, payload in artifacts.items():
            path = root / f".{name}.{os.getpid()}.tmp"
            with path.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.append(path)
        for path, target in zip(temporary, targets.values(), strict=True):
            os.link(path, target)
            published.append(target)
        directory = os.open(root, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        for path in published:
            path.unlink(missing_ok=True)
        raise
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)


def _add_hash_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--expected-packet-sha256", required=True)
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument("--expected-ontology-sha256", required=True)


def _add_selection_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--expected-selection-sha256", required=True)
    parser.add_argument("--member-manifest", type=Path, required=True)
    parser.add_argument("--expected-member-manifest-sha256", required=True)
    parser.add_argument("--member-manifest-receipt", type=Path, required=True)
    parser.add_argument("--expected-member-manifest-receipt-sha256", required=True)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--expected-candidate-pool-sha256", required=True)
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--expected-census-path", type=Path, required=True)
    parser.add_argument("--expected-census-sha256", required=True)
    parser.add_argument("--sampled-panel-root", type=Path, required=True)
    parser.add_argument("--source-b16-root", type=Path, required=True)
    parser.add_argument("--look-id", choices=LOOK_IDS, required=True)
    parser.add_argument("--prior-look-decision", type=Path)
    parser.add_argument("--prior-look-receipt", type=Path)


def _validated_selection_from_args(args: argparse.Namespace) -> ValidatedSelection:
    return _validate_selection(
        args.selection.read_bytes(),
        expected_sha256=args.expected_selection_sha256,
        selection_path=args.selection,
        member_manifest_jsonl=args.member_manifest.read_bytes(),
        expected_member_manifest_sha256=args.expected_member_manifest_sha256,
        member_manifest_receipt_json=args.member_manifest_receipt.read_bytes(),
        expected_member_manifest_receipt_sha256=args.expected_member_manifest_receipt_sha256,
    )


def _current_image_ids(
    selection: ValidatedSelection, *, look_id: str
) -> tuple[str, ...]:
    return (
        selection.look_one_image_ids
        if look_id == "look_one"
        else selection.look_two_additional_image_ids
    )


def _optional_bytes(path: Path | None) -> bytes | None:
    return None if path is None else path.read_bytes()


def _require_production_frozen_sources(args: argparse.Namespace) -> None:
    census_path = args.census.expanduser().resolve(strict=True)
    frozen_census_path = FROZEN_CENSUS_PATH.expanduser().resolve(strict=True)
    if (
        census_path != frozen_census_path
        or args.expected_census_sha256 != FROZEN_CENSUS_SHA256
        or (
            hasattr(args, "expected_census_path")
            and args.expected_census_path.expanduser().resolve(strict=True)
            != frozen_census_path
        )
        or args.expected_candidate_pool_sha256 != FROZEN_CANDIDATE_POOL_SHA256
    ):
        raise ValueError("production frozen census/candidate binding drift")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    freeze = subparsers.add_parser("freeze-contract")
    _add_hash_inputs(freeze)
    freeze.add_argument("--candidate-pool", type=Path, required=True)
    freeze.add_argument("--expected-candidate-pool-sha256", required=True)
    freeze.add_argument("--census", type=Path, required=True)
    freeze.add_argument("--expected-census-sha256", required=True)
    freeze.add_argument("--source-b16-root", type=Path, required=True)
    freeze.add_argument("--sampled-panel-root", type=Path, required=True)
    freeze.add_argument("--additional-source", type=Path, action="append", default=[])
    freeze.add_argument("--output-root", type=Path, required=True)
    member = subparsers.add_parser("build-member-manifest")
    member.add_argument("--candidate-pool", type=Path, required=True)
    member.add_argument("--expected-candidate-pool-sha256", required=True)
    member.add_argument("--stage-zero-root", type=Path, required=True)
    member.add_argument(
        "--expected-stage-zero-root-inventory-sha256", required=True
    )
    member.add_argument("--stage-zero-receipt", type=Path, required=True)
    member.add_argument("--expected-stage-zero-receipt-sha256", required=True)
    member.add_argument("--stage-zero-audit", type=Path, required=True)
    member.add_argument("--expected-stage-zero-audit-sha256", required=True)
    member.add_argument("--possible-pool", type=Path, required=True)
    member.add_argument("--expected-possible-pool-sha256", required=True)
    member.add_argument("--census", type=Path, required=True)
    member.add_argument("--expected-census-path", type=Path, required=True)
    member.add_argument("--expected-census-sha256", required=True)
    member.add_argument("--source-b16-root", type=Path, required=True)
    member.add_argument("--sampled-panel-root", type=Path, required=True)
    member.add_argument("--frozen-contract", type=Path, required=True)
    member.add_argument("--expected-frozen-contract-sha256", required=True)
    member.add_argument("--output-root", type=Path, required=True)
    queue = subparsers.add_parser("build-queue")
    _add_hash_inputs(queue)
    _add_selection_inputs(queue)
    queue.add_argument("--output-root", type=Path, required=True)
    seal = subparsers.add_parser("seal-role")
    seal.add_argument("--review-queue", type=Path, required=True)
    seal.add_argument("--expected-review-queue-sha256", required=True)
    seal.add_argument("--role-artifact", type=Path, required=True)
    seal.add_argument(
        "--reviewer-role-identifier", choices=REVIEWER_ROLES, required=True
    )
    seal.add_argument("--expected-packet-sha256", required=True)
    seal.add_argument("--expected-ontology-sha256", required=True)
    seal.add_argument("--output-root", type=Path, required=True)
    adjudication = subparsers.add_parser("build-adjudication-queue")
    for name in (
        "review-queue",
        "reviewer-one",
        "reviewer-one-seal",
        "reviewer-two",
        "reviewer-two-seal",
        "official-owner-ledger",
    ):
        adjudication.add_argument(f"--{name}", type=Path, required=True)
    for name in (
        "expected-review-queue-sha256",
        "expected-official-owner-ledger-sha256",
        "expected-packet-sha256",
        "expected-ontology-sha256",
    ):
        adjudication.add_argument(f"--{name}", required=True)
    adjudication.add_argument("--output-root", type=Path, required=True)
    ledger = subparsers.add_parser("assemble-ledger")
    ledger.add_argument("--adjudication-queue", type=Path, required=True)
    ledger.add_argument("--expected-adjudication-queue-sha256", required=True)
    ledger.add_argument("--adjudicator-decisions", type=Path, required=True)
    ledger.add_argument("--review-queue-manifest", type=Path, required=True)
    ledger.add_argument("--expected-review-queue-manifest-sha256", required=True)
    ledger.add_argument("--output-root", type=Path, required=True)
    replay = subparsers.add_parser("replay")
    _add_selection_inputs(replay)
    replay.add_argument("--owner-ledger", type=Path, required=True)
    replay.add_argument("--owner-ledger-seal", type=Path, required=True)
    replay.add_argument("--output-root", type=Path, required=True)
    classify = subparsers.add_parser("classify")
    _add_selection_inputs(classify)
    classify.add_argument("--replay-records", type=Path, required=True)
    classify.add_argument("--replay-receipt", type=Path, required=True)
    classify.add_argument("--uncertainty-ledger", type=Path, required=True)
    classify.add_argument("--owner-ledger-seal", type=Path, required=True)
    classify.add_argument("--output-root", type=Path, required=True)
    sequential = subparsers.add_parser("sequential-decision")
    _add_selection_inputs(sequential)
    sequential.add_argument("--outcomes", type=Path, required=True)
    sequential.add_argument("--output-root", type=Path, required=True)
    finalize = subparsers.add_parser("finalize-look")
    _add_selection_inputs(finalize)
    finalize.add_argument("--artifact-root", type=Path, required=True)
    finalize.add_argument("--packet", type=Path, required=True)
    finalize.add_argument("--ontology", type=Path, required=True)
    finalize.add_argument("--frozen-contract", type=Path, required=True)
    finalize.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode in {
        "freeze-contract",
        "build-member-manifest",
        "build-queue",
        "replay",
        "classify",
        "sequential-decision",
        "finalize-look",
    }:
        _require_production_frozen_sources(args)
    if args.mode == "freeze-contract":
        payload = freeze_preseed_contract(
            packet_path=args.packet,
            expected_packet_sha256=args.expected_packet_sha256,
            ontology_path=args.ontology,
            expected_ontology_sha256=args.expected_ontology_sha256,
            candidate_pool_path=args.candidate_pool,
            expected_candidate_pool_sha256=args.expected_candidate_pool_sha256,
            census_path=args.census,
            expected_census_sha256=args.expected_census_sha256,
            source_panel_root=args.source_b16_root,
            sampled_panel_root=args.sampled_panel_root,
            source_paths=args.additional_source,
        )
        _publish_once(args.output_root, {"preseed-contract-seal.json": payload})
    elif args.mode == "build-member-manifest":
        image_ids, _ = _load_stage_zero_possible_pool(
            stage_zero_root=args.stage_zero_root,
            expected_stage_zero_root_inventory_sha256=(
                args.expected_stage_zero_root_inventory_sha256
            ),
            stage_zero_receipt_path=args.stage_zero_receipt,
            expected_stage_zero_receipt_sha256=(
                args.expected_stage_zero_receipt_sha256
            ),
            stage_zero_audit_path=args.stage_zero_audit,
            expected_stage_zero_audit_sha256=args.expected_stage_zero_audit_sha256,
            possible_pool_path=args.possible_pool,
            expected_possible_pool_sha256=args.expected_possible_pool_sha256,
        )
        adapter = load_v2_b16_panel_adapter(
            sampled_panel_root=args.sampled_panel_root,
            source_b16_root=args.source_b16_root,
            candidate_pool=args.candidate_pool,
            semantic_image_ids=image_ids,
        )
        artifacts = build_preentropy_member_manifest(
            adapter=adapter,
            ordered_image_ids=image_ids,
            stage_zero_root=args.stage_zero_root,
            expected_stage_zero_root_inventory_sha256=(
                args.expected_stage_zero_root_inventory_sha256
            ),
            stage_zero_receipt_path=args.stage_zero_receipt,
            expected_stage_zero_receipt_sha256=(
                args.expected_stage_zero_receipt_sha256
            ),
            stage_zero_audit_path=args.stage_zero_audit,
            expected_stage_zero_audit_sha256=args.expected_stage_zero_audit_sha256,
            possible_pool_path=args.possible_pool,
            expected_possible_pool_sha256=args.expected_possible_pool_sha256,
            candidate_pool_path=args.candidate_pool,
            expected_candidate_pool_sha256=args.expected_candidate_pool_sha256,
            census_path=args.census,
            expected_census_path=args.expected_census_path,
            expected_census_sha256=args.expected_census_sha256,
            intended_member_manifest_path=args.output_root / "member-manifest.jsonl",
            frozen_contract_path=args.frozen_contract,
            expected_frozen_contract_sha256=args.expected_frozen_contract_sha256,
            source_panel_root=args.source_b16_root,
            sampled_panel_root=args.sampled_panel_root,
        )
        _publish_once(
            args.output_root,
            {
                "member-manifest.jsonl": artifacts.member_manifest_jsonl,
                "member-manifest-receipt.json": artifacts.member_manifest_receipt_json,
            },
        )
    elif args.mode == "build-queue":
        selection = _validated_selection_from_args(args)
        image_ids = _current_image_ids(selection, look_id=args.look_id)
        adapter = load_v2_b16_panel_adapter(
            sampled_panel_root=args.sampled_panel_root,
            source_b16_root=args.source_b16_root,
            candidate_pool=args.candidate_pool,
            semantic_image_ids=image_ids,
        )
        artifacts = build_review_queue(
            selection_json=args.selection.read_bytes(),
            expected_selection_sha256=args.expected_selection_sha256,
            selection_path=args.selection,
            member_manifest_jsonl=args.member_manifest.read_bytes(),
            expected_member_manifest_sha256=args.expected_member_manifest_sha256,
            member_manifest_receipt_json=args.member_manifest_receipt.read_bytes(),
            expected_member_manifest_receipt_sha256=args.expected_member_manifest_receipt_sha256,
            adapter=adapter,
            candidate_pool_path=args.candidate_pool,
            expected_candidate_pool_sha256=args.expected_candidate_pool_sha256,
            census_path=args.census,
            expected_census_path=args.expected_census_path,
            expected_census_sha256=args.expected_census_sha256,
            packet_path=args.packet,
            expected_packet_sha256=args.expected_packet_sha256,
            ontology_path=args.ontology,
            expected_ontology_sha256=args.expected_ontology_sha256,
            look_id=args.look_id,
            prior_look_decision_json=_optional_bytes(args.prior_look_decision),
            prior_look_receipt_json=_optional_bytes(args.prior_look_receipt),
        )
        _publish_once(
            args.output_root,
            {
                "review-queue.jsonl": artifacts.review_queue_jsonl,
                "official-owner-ledger.jsonl": artifacts.official_owner_ledger_jsonl,
                "review-queue-manifest.json": artifacts.manifest_json,
            },
        )
    elif args.mode == "seal-role":
        payload = seal_role_artifact(
            role_artifact_jsonl=args.role_artifact.read_bytes(),
            reviewer_role_identifier=args.reviewer_role_identifier,
            review_queue_jsonl=args.review_queue.read_bytes(),
            expected_review_queue_sha256=args.expected_review_queue_sha256,
            expected_packet_sha256=args.expected_packet_sha256,
            expected_ontology_sha256=args.expected_ontology_sha256,
        )
        _publish_once(
            args.output_root, {f"{args.reviewer_role_identifier}-seal.json": payload}
        )
    elif args.mode == "build-adjudication-queue":
        payload = build_adjudication_queue(
            review_queue_jsonl=args.review_queue.read_bytes(),
            expected_review_queue_sha256=args.expected_review_queue_sha256,
            reviewer_one_jsonl=args.reviewer_one.read_bytes(),
            reviewer_one_seal_json=args.reviewer_one_seal.read_bytes(),
            reviewer_two_jsonl=args.reviewer_two.read_bytes(),
            reviewer_two_seal_json=args.reviewer_two_seal.read_bytes(),
            official_owner_ledger_jsonl=args.official_owner_ledger.read_bytes(),
            expected_official_owner_ledger_sha256=args.expected_official_owner_ledger_sha256,
            expected_packet_sha256=args.expected_packet_sha256,
            expected_ontology_sha256=args.expected_ontology_sha256,
        )
        _publish_once(args.output_root, {"adjudication-queue.jsonl": payload})
    elif args.mode == "assemble-ledger":
        artifacts = assemble_owner_ledger(
            adjudication_queue_jsonl=args.adjudication_queue.read_bytes(),
            expected_adjudication_queue_sha256=args.expected_adjudication_queue_sha256,
            adjudicator_decisions_jsonl=args.adjudicator_decisions.read_bytes(),
            review_queue_manifest_json=args.review_queue_manifest.read_bytes(),
            expected_review_queue_manifest_sha256=args.expected_review_queue_manifest_sha256,
        )
        _publish_once(
            args.output_root,
            {
                "adjudication.jsonl": artifacts.adjudication_jsonl,
                "owner-ledger.jsonl": artifacts.owner_ledger_jsonl,
                "uncertainty-ledger.jsonl": artifacts.uncertainty_ledger_jsonl,
                "owner-ledger-seal.json": artifacts.owner_ledger_seal_json,
            },
        )
    elif args.mode == "replay":
        selection = _validated_selection_from_args(args)
        image_ids = _current_image_ids(selection, look_id=args.look_id)
        adapter = load_v2_b16_panel_adapter(
            sampled_panel_root=args.sampled_panel_root,
            source_b16_root=args.source_b16_root,
            candidate_pool=args.candidate_pool,
            semantic_image_ids=image_ids,
        )
        artifacts = replay_owner_ledger(
            adapter=adapter,
            look_id=args.look_id,
            selection_json=args.selection.read_bytes(),
            expected_selection_sha256=args.expected_selection_sha256,
            selection_path=args.selection,
            member_manifest_jsonl=args.member_manifest.read_bytes(),
            expected_member_manifest_sha256=args.expected_member_manifest_sha256,
            member_manifest_receipt_json=args.member_manifest_receipt.read_bytes(),
            expected_member_manifest_receipt_sha256=args.expected_member_manifest_receipt_sha256,
            candidate_pool_path=args.candidate_pool,
            expected_candidate_pool_sha256=args.expected_candidate_pool_sha256,
            census_path=args.census,
            expected_census_path=args.expected_census_path,
            expected_census_sha256=args.expected_census_sha256,
            owner_ledger_jsonl=args.owner_ledger.read_bytes(),
            owner_ledger_seal_json=args.owner_ledger_seal.read_bytes(),
        )
        _publish_once(
            args.output_root,
            {
                "replay-records.jsonl": artifacts.replay_records_jsonl,
                "replay-receipt.json": artifacts.replay_receipt_json,
            },
        )
    elif args.mode == "classify":
        selection = _validated_selection_from_args(args)
        image_ids = _current_image_ids(selection, look_id=args.look_id)
        payload = classify_outcomes(
            replay_records_jsonl=args.replay_records.read_bytes(),
            replay_receipt_json=args.replay_receipt.read_bytes(),
            uncertainty_ledger_jsonl=args.uncertainty_ledger.read_bytes(),
            owner_ledger_seal_json=args.owner_ledger_seal.read_bytes(),
            look_id=args.look_id,
            ordered_image_ids=image_ids,
            expected_selection_sha256=args.expected_selection_sha256,
            expected_member_manifest_sha256=args.expected_member_manifest_sha256,
            expected_member_manifest_receipt_sha256=args.expected_member_manifest_receipt_sha256,
            expected_candidate_pool_sha256=args.expected_candidate_pool_sha256,
            expected_census_sha256=args.expected_census_sha256,
        )
        _publish_once(args.output_root, {"outcomes.jsonl": payload})
    elif args.mode == "sequential-decision":
        payload = build_sequential_decision(
            look_id=args.look_id,
            outcomes_jsonl=args.outcomes.read_bytes(),
            selection_json=args.selection.read_bytes(),
            expected_selection_sha256=args.expected_selection_sha256,
            selection_path=args.selection,
            member_manifest_jsonl=args.member_manifest.read_bytes(),
            expected_member_manifest_sha256=args.expected_member_manifest_sha256,
            member_manifest_receipt_json=args.member_manifest_receipt.read_bytes(),
            expected_member_manifest_receipt_sha256=args.expected_member_manifest_receipt_sha256,
            expected_candidate_pool_sha256=args.expected_candidate_pool_sha256,
            expected_census_sha256=args.expected_census_sha256,
            prior_look_decision_json=_optional_bytes(args.prior_look_decision),
            prior_look_receipt_json=_optional_bytes(args.prior_look_receipt),
        )
        _publish_once(args.output_root, {"sequential-decision.json": payload})
    elif args.mode == "finalize-look":
        selection = _validated_selection_from_args(args)
        image_ids = _current_image_ids(selection, look_id=args.look_id)
        adapter = load_v2_b16_panel_adapter(
            sampled_panel_root=args.sampled_panel_root,
            source_b16_root=args.source_b16_root,
            candidate_pool=args.candidate_pool,
            semantic_image_ids=image_ids,
        )
        artifact_payloads = {
            name: (args.artifact_root / name).read_bytes()
            for name in FINAL_LOOK_ARTIFACT_NAMES
        }
        finalize_look(
            adapter=adapter,
            final_root=args.output_root,
            look_id=args.look_id,
            artifact_payloads=artifact_payloads,
            source_bindings={
                "packet": args.packet,
                "ontology": args.ontology,
                "candidate_pool": args.candidate_pool,
                "census": args.census,
                "frozen_contract": args.frozen_contract,
                "source_panel": args.source_b16_root,
                "sampled_panel": args.sampled_panel_root,
            },
            selection_path=args.selection,
            member_manifest_path=args.member_manifest,
            member_manifest_receipt_path=args.member_manifest_receipt,
            expected_selection_sha256=args.expected_selection_sha256,
            expected_member_manifest_sha256=args.expected_member_manifest_sha256,
            expected_member_manifest_receipt_sha256=args.expected_member_manifest_receipt_sha256,
            expected_candidate_pool_sha256=args.expected_candidate_pool_sha256,
            expected_census_sha256=args.expected_census_sha256,
            prior_look_decision_json=_optional_bytes(args.prior_look_decision),
            prior_look_receipt_json=_optional_bytes(args.prior_look_receipt),
        )
    else:
        raise AssertionError(f"unhandled mode: {args.mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
