"""Contracts for the successor-local sorted-FN score-run attestor.

Fixtures reuse the real, unchanged planner (``prepare_sorted_fn_successor_
inputs``) and the successor's own merger (``merge_sorted_fn_successor_score_
shards``) to produce a genuine merge receipt against real successor-scorer-
shaped shards, then attest it. See ``test_merge_sorted_fn_successor_score_
shards.py`` for the shared fixture rationale (deliberately duplicated per
this codebase's self-contained-test-file convention).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.research.build_sorted_owner_basin_candidates import sha256_file, sha256_json
from scripts.research.sorted_owner_basin_landscape import GEOMETRY_IDENTITY_SCHEMA, RULES_SCHEMA_VERSION
from scripts.research import score_sorted_owner_basin_landscape as scorer
from scripts.research.build_sorted_fn_mechanism_registry import (
    SCHEMA_VERSION as REGISTRY_SCHEMA_VERSION,
    UNIT_ID as SUCCESSOR_UNIT_ID,
)
from scripts.research.prepare_sorted_fn_successor_inputs import prepare_sorted_fn_successor_inputs
import scripts.research.score_sorted_fn_fixed_budget as fixed_budget_scorer
from scripts.research.merge_sorted_fn_successor_score_shards import (
    PREDECESSOR_PRIMITIVES_FILE,
    SUCCESSOR_LIVE_SCORING_SEALED_STATUS,
    SUCCESSOR_SCORE_RECEIPT_SCHEMA_VERSION,
    SUCCESSOR_SCORE_ROW_SCHEMA_VERSION,
    SUCCESSOR_SCORER_UNIT_ID,
    ShardInput,
    merge_shards,
)
from scripts.research.attest_sorted_fn_successor_score_run import (
    ATTESTATION_SCHEMA_VERSION,
    RunAttestationError,
    attest,
)


# --------------------------------------------------------------------------
# Minimal, self-consistent rules-template fixture (see
# test_merge_sorted_fn_successor_score_shards.py for the same, deliberately
# duplicated per this codebase's self-contained-test-file convention).
# --------------------------------------------------------------------------

MODEL_IDENTITY = {"kind": "fake-model", "name": "test-model"}
TOKENIZER_IDENTITY = {"kind": "fake-tokenizer", "name": "test-tokenizer"}
MODEL_IDENTITY_SHA256 = sha256_json(MODEL_IDENTITY)
TOKENIZER_IDENTITY_SHA256 = sha256_json(TOKENIZER_IDENTITY)

FAKE_DIGEST_C = "c" * 64
FAKE_IDENTITY_RECEIPT_DIGEST = "d" * 64

SCHEMA_TOKENS = {
    "object_ref_start_token_id": 100000,
    "object_ref_end_token_id": 100001,
    "box_start_token_id": 100002,
    "box_end_token_id": 100003,
}
COORD_TOKEN_START = 200000
COORD_TOKEN_END = COORD_TOKEN_START + 1000
COORD_BIN_TOKEN_IDS = list(range(COORD_TOKEN_START, COORD_TOKEN_END))
COORD_BIN_TOKEN_IDS_SHA256 = sha256_json(COORD_BIN_TOKEN_IDS)
MODEL_VOCAB_SIZE = 300000
VOCAB_ATTESTATION = {
    "tokenizer_identity_sha256": TOKENIZER_IDENTITY_SHA256,
    "model_identity_sha256": MODEL_IDENTITY_SHA256,
    "runtime_identity_sha256": FAKE_DIGEST_C,
}
_TOKEN_REGISTRY_PAYLOAD = {
    "schema_tokens": SCHEMA_TOKENS,
    "coordinate_bin_to_token_id": {
        "bin_min": 0,
        "bin_max": 999,
        "token_id_start": COORD_TOKEN_START,
        "token_id_end_exclusive": COORD_TOKEN_END,
        "mapping": "linear_offset",
        "coordinate_bin_token_ids": COORD_BIN_TOKEN_IDS,
        "coordinate_bin_token_ids_sha256": COORD_BIN_TOKEN_IDS_SHA256,
    },
    "model_vocab_size": MODEL_VOCAB_SIZE,
    "vocabulary_attestation": VOCAB_ATTESTATION,
    "identity_receipt_digest": FAKE_IDENTITY_RECEIPT_DIGEST,
}
TOKEN_REGISTRY = {**_TOKEN_REGISTRY_PAYLOAD, "registry_sha256": sha256_json(_TOKEN_REGISTRY_PAYLOAD)}
STRUCTURAL_ROW_WRAPPER = {
    "composition": [
        "object_ref_start_token_id",
        "canonical_description_token_ids",
        "object_ref_end_token_id",
        "box_start_token_id",
        "x1_coordinate_token_id",
        "y1_coordinate_token_id",
        "x2_coordinate_token_id",
        "y2_coordinate_token_id",
        "box_end_token_id",
    ],
    "schema_tokens": SCHEMA_TOKENS,
    "coordinate_bin_token_ids_sha256": COORD_BIN_TOKEN_IDS_SHA256,
}
FOIL_SET_ID = "foil-set:test:v1"
TARGET_ROLE_ID = "basin-role:test:target"
BACKGROUND_ROLE_ID = "basin-role:test:background"
SCAN_ROLE_ID = "basin-role:test:scan"
_PLACEHOLDER_FOIL_MEMBER = {
    "foil_member_id": "foil-member:test:placeholder",
    "diagnostic_owner_id": "diagnostic:gt:placeholder:0",
    "context_id": "ctx:placeholder",
    "bank_name": "background",
    "source_id": "candidate-source:test:placeholder",
    "identity_kind": "registered_geometry",
    "identity_id": "foil-geometry:gt:placeholder:0:background",
    "provenance": {
        "artifact_path": "placeholder-review.json",
        "artifact_sha256": "a" * 64,
        "source_row_id": "foil-review:placeholder:background",
    },
}
_PLACEHOLDER_DESCRIPTION = {
    "text": "placeholder",
    "text_sha256": sha256_json("placeholder"),
    "token_ids": [999999],
    "token_ids_sha256": sha256_json([999999]),
    "forced_row_prefix_through_box_start_token_ids": [
        SCHEMA_TOKENS["object_ref_start_token_id"],
        999999,
        SCHEMA_TOKENS["object_ref_end_token_id"],
        SCHEMA_TOKENS["box_start_token_id"],
    ],
    "forced_row_prefix_through_box_start_sha256": sha256_json(
        [
            SCHEMA_TOKENS["object_ref_start_token_id"],
            999999,
            SCHEMA_TOKENS["object_ref_end_token_id"],
            SCHEMA_TOKENS["box_start_token_id"],
        ]
    ),
}
RULES_TEMPLATE: dict[str, Any] = {
    "schema_version": RULES_SCHEMA_VERSION,
    "contract_mode": "test_fixture",
    "geometry_identity": {"schema": GEOMETRY_IDENTITY_SCHEMA, "coordinate_denominator": 1000},
    "coordinate_bins": {"min": 0, "max": 999},
    "target_anchor": {"margin_fraction": 0.10, "min_margin_bins": 2, "max_margin_bins": 32},
    "bank_order": ["target", "background", "scan"],
    "proposal_measures": {
        "primary": {
            "comparability_group": "primary",
            "normalization": "full_domain_normalized_weighted_sum",
            "bank_weights": {"target": 1.0, "background": 1.0, "scan": 1.0},
        }
    },
    "bank_proposal_measure": {"target": "primary", "background": "primary", "scan": "primary"},
    "spatial_clustering": {
        "owner_link_iou_min": 0.5,
        "owner_link_center_distance_max": 50.0,
        "extent_submode_iou_min": 0.5,
    },
    "shape": {
        "near_peak_logprob_delta": 0.5,
        "wide_ridge_min_candidates": 2,
        "multi_submode_min": 2,
        "merged_extent_submodes": [],
        "scan_bank_names": ["scan"],
    },
    "declared_extent_submodes": [
        "gt_whole",
        "scale_0p80",
        "scale_1p20",
        "aspect_0p80",
        "aspect_1p20",
        "anchor_validity_floor",
        "equal_size_background",
        "sorted_scan",
    ],
    "registered_basin_roles": {
        TARGET_ROLE_ID: {
            "kind": "target",
            "foil_set_id": FOIL_SET_ID,
            "identity_kind": "reviewed_physical_owner",
            "allowed_bank_names": ["target"],
        },
        BACKGROUND_ROLE_ID: {
            "kind": "foil",
            "foil_set_id": FOIL_SET_ID,
            "identity_kind": "registered_geometry",
            "allowed_bank_names": ["background"],
        },
        SCAN_ROLE_ID: {
            "kind": "foil",
            "foil_set_id": FOIL_SET_ID,
            "identity_kind": "registered_geometry",
            "allowed_bank_names": ["scan"],
        },
    },
    "prominence": {"functional": "peak_height_difference"},
    "token_registry": TOKEN_REGISTRY,
    "schema_tokens": {
        **SCHEMA_TOKENS,
        "coordinate_token_id_start": COORD_TOKEN_START,
        "coordinate_token_id_end_exclusive": COORD_TOKEN_END,
    },
    "model_vocab_size": MODEL_VOCAB_SIZE,
    "structural_row_wrapper": STRUCTURAL_ROW_WRAPPER,
    "score_channels": {
        "raw_fp32": {"role": "primary_model_likelihood", "repetition_penalty": None},
        "rp_1_00": {"role": "auxiliary_policy_score", "repetition_penalty": 1.0},
        "rp_1_10": {"role": "auxiliary_policy_score", "repetition_penalty": 1.10},
        "shared_forward_rule": "derive both policy views from one raw forward only at byte-identical prefixes",
    },
    "owner_canonical_descriptions": {"gt:placeholder:0": _PLACEHOLDER_DESCRIPTION},
    "candidate_materializer": {
        "schema_version": "sorted_owner_basin_candidate_materializer_rules.v2",
        "status": "sealed",
        "coordinate_space": {
            "name": "qwen_coordinate_bins",
            "min": 0,
            "max": 999,
            "contract_kind": "test_fixture",
            "non_production_reason": "attest-run-attestation test fixture; not a decision-bearing run",
            "bin_to_extent_conversion": "round(value*extent/1000)",
        },
        "coco_namespace": {"name": "coco_2017_official_gapped", "id_space": "official_gapped"},
        "target_bank_name": "target",
        "bank_roles": {"target": TARGET_ROLE_ID, "background": BACKGROUND_ROLE_ID, "scan": SCAN_ROLE_ID},
        "foil_set": {
            "foil_set_id": FOIL_SET_ID,
            "members": [_PLACEHOLDER_FOIL_MEMBER],
            "members_sha256": sha256_json([_PLACEHOLDER_FOIL_MEMBER]),
        },
        "p_x1_y1_pruning": {
            "mode": "disabled",
            "declaration": "absence-neutral: no anchor is pruned before complete conditional-y1 scoring",
        },
        "free_search_budget": {
            "x1_branch_budget": 64,
            "y1_branch_budget_per_x1": 32,
            "extent_branch_budget_per_anchor": 16,
            "spatial_diversification": {
                "algorithm": "deterministic_farthest_point_xy_anchor_selection",
                "tie_break": "ascending_x1_then_y1",
                "minimum_center_distance_bins": 24,
            },
        },
        "vocabulary_attestation": VOCAB_ATTESTATION,
        "upstream_digests": {"placeholder": "0" * 64},
    },
}

_WIDGET_DESCRIPTION = {
    "text": "widget",
    "text_sha256": sha256_json("widget"),
    "token_ids": [777],
    "token_ids_sha256": sha256_json([777]),
    "forced_row_prefix_through_box_start_token_ids": [100000, 777, 100001, 100002],
    "forced_row_prefix_through_box_start_sha256": sha256_json([100000, 777, 100001, 100002]),
}
_DESCRIPTION_SUPPLEMENT = {"widget": _WIDGET_DESCRIPTION}
_OBJECT_REF_START = 151646
_PROMPT = list(range(1, 21))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _owner_ledger_row(gt_owner_id: str, image_id: str, annotation_index: int, description: str) -> dict[str, Any]:
    return {
        "schema_version": "sorted-owner-basin-owner-ledger.v2",
        "gt_owner_id": gt_owner_id,
        "image_id": str(image_id),
        "original_annotation_index": annotation_index,
        "normalized_description": description,
        "official_coco_category_id": 1,
        "diagnostic_owner_id": f"diagnostic:{gt_owner_id}",
        "source_digests": {"panel": "b" * 64},
    }


def _panel_line(image_id: str, width: int, height: int, boxes: list[list[int]]) -> dict[str, Any]:
    return {"image_id": str(image_id), "width": width, "height": height, "objects": [{"bbox_2d": list(box)} for box in boxes]}


def _rollout_document(image_id: str, decode_mode: str, seed: int, prompt_token_ids: list[int], row_chunks: list[list[int]]) -> dict[str, Any]:
    generated = [token for chunk in row_chunks for token in chunk]
    predictions = [{"generated_order": idx, "raw_span_sha256": "e" * 64} for idx in range(len(row_chunks))]
    return {
        "rollouts": [
            {
                "image_id": image_id,
                "decode_mode": decode_mode,
                "seed": seed,
                "stop_reason": "im_end",
                "prompt_token_ids": prompt_token_ids,
                "generated_token_ids": generated,
                "predictions": {"predictions": predictions, "dropped_predictions": []},
            }
        ]
    }


def _row_chunk(marker: int) -> list[int]:
    return [_OBJECT_REF_START, marker, marker + 1, marker + 2]


def _mechanism_registry(roles: list[dict[str, Any]], *, targets: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "unit_id": SUCCESSOR_UNIT_ID,
        "registry_digest": "f" * 64,
        "mechanism_cohort": {"targets": targets or []},
        "context_control_registry": {"bound_non_targets": []},
        "smoke": {"roles": roles, "null_pair_envelope": {"pairs": []}},
    }


def _context_role(role_id: str, gt_owner_id: str, *, role_kind: str = "root_context", image_id: str = "img1", cut_marker: int | None = None) -> dict[str, Any]:
    token_ids = list(_PROMPT)
    if cut_marker is not None:
        token_ids = token_ids + _row_chunk(cut_marker)
    return {
        "role_id": role_id,
        "role_kind": role_kind,
        "gt_owner_id": gt_owner_id,
        "trajectory": {"image_id": image_id, "decode_mode": "greedy", "seed": 0},
        "prefix": {"token_ids": token_ids},
        "provenance": {"source_artifact_path": "rollout.json", "source_artifact_sha256": "0" * 64},
    }


def _build_planned(tmp_path: Path, *, boxes: list[list[int]]) -> dict[str, Any]:
    fixtures_dir = tmp_path / "fixtures"
    owner_ledger_path = fixtures_dir / "owner-ledger.jsonl"
    _write_jsonl(owner_ledger_path, [_owner_ledger_row("gt:img1:0", "img1", 0, "widget"), _owner_ledger_row("gt:img1:1", "img1", 1, "widget")])
    panel_path = fixtures_dir / "panel.jsonl"
    _write_jsonl(panel_path, [_panel_line("img1", 1024, 1024, boxes)])
    rollout_path = fixtures_dir / "rollout.json"
    _write_json(rollout_path, _rollout_document("img1", "greedy", 0, _PROMPT, [_row_chunk(1), _row_chunk(10)]))
    rules_template_path = fixtures_dir / "rules-template.json"
    _write_json(rules_template_path, RULES_TEMPLATE)
    descriptions_path = fixtures_dir / "descriptions.json"
    _write_json(descriptions_path, _DESCRIPTION_SUPPLEMENT)

    roles = [_context_role("root:gt:img1:0", "gt:img1:0"), _context_role("root:gt:img1:1", "gt:img1:1")]
    registry_path = tmp_path / "registry.json"
    # gt:img1:0 is strict_rescued so its context materializes rung="scalar_smoke"
    # (30 target + 30 decoy + up to 7 reference) rows for the scalar_smoke
    # attestation tests below; both owners share one normalized_description
    # in one image, so a same-description zero-overlap reference is bound.
    _write_json(registry_path, _mechanism_registry(roles, targets=[{"gt_owner_id": "gt:img1:0", "cohort": "strict_rescued"}]))

    out_dir = tmp_path / "plan"
    prepare_sorted_fn_successor_inputs(
        registry=registry_path,
        owner_ledger=owner_ledger_path,
        panel=panel_path,
        rules_template=rules_template_path,
        rollouts=[rollout_path],
        canonical_description_registry=descriptions_path,
        predecessor_score_rows=None,
        out_dir=out_dir,
    )
    ledger_path = out_dir / "owner-context-ledger.jsonl"
    rules_path = out_dir / "landscape-decision-rules.json"
    mechanism_rules_path = out_dir / "mechanism-decision-rules.json"
    fixed_budget_path = out_dir / "fixed-budget-candidates.jsonl"
    planner_receipt_path = out_dir / "receipt.json"

    ledger_rows = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    fixed_budget_rows = [json.loads(line) for line in fixed_budget_path.read_text().splitlines()]
    rules = scorer.load_decision_rules(rules_path)
    mechanism_rules_document = json.loads(mechanism_rules_path.read_text())
    return {
        "tmp_path": tmp_path,
        "registry_path": registry_path,
        "ledger_path": ledger_path,
        "rules_path": rules_path,
        "mechanism_rules_path": mechanism_rules_path,
        "mechanism_rules_document": mechanism_rules_document,
        "fixed_budget_path": fixed_budget_path,
        "planner_receipt_path": planner_receipt_path,
        "ledger_rows": ledger_rows,
        "ledger_by_context": {row["context_id"]: row for row in ledger_rows},
        "fixed_budget_rows": fixed_budget_rows,
        "rules": rules,
    }


@pytest.fixture()
def planned(tmp_path: Path) -> dict[str, Any]:
    """Small (6x6) boxes: keeps coverage regardless of GT-box size for the
    near_gt_micro F3 exact-GT-singleton collapse (see ``planned_large_box``
    below for the large-box proof)."""

    return _build_planned(tmp_path, boxes=[[100, 100, 106, 106], [300, 300, 306, 306]])


@pytest.fixture()
def planned_large_box(tmp_path: Path) -> dict[str, Any]:
    """Large (100x100+) boxes, proving F3 exact-singleton carry-through
    through attestation is not an artifact of small-box geometry."""

    return _build_planned(tmp_path, boxes=[[100, 100, 200, 200], [300, 300, 380, 360]])


def _deterministic_slot_value(coord_token_ids: list[int], slot: str) -> float:
    payload = {"coord": tuple(coord_token_ids), "slot": slot}
    return -1.0 - (int.from_bytes(sha256_json(payload).encode()[:4], "big") % 5000) / 100.0


def _raw_and_policy(coord_token_ids: list[int]) -> tuple[dict[str, Any], dict[str, Any]]:
    slots = ["x1_logprob", "y1_logprob", "x2_logprob", "y2_logprob"]
    raw = {slot: _deterministic_slot_value(coord_token_ids, slot) for slot in slots}
    raw["complete_box_logprob_sum"] = sum(raw.values())
    raw["vocab_attestation"] = {"x1": {}, "y1": {}, "x2": {}, "y2": {}}
    policy: dict[str, Any] = {}
    for view_key in (fixed_budget_scorer.scorer._policy_view_key(1.0), fixed_budget_scorer.scorer._policy_view_key(1.10)):
        view_slots = {slot: _deterministic_slot_value(coord_token_ids, f"{view_key}:{slot}") for slot in slots}
        view_slots["complete_box_logprob_sum"] = sum(view_slots.values())
        view_slots["vocab_attestation"] = {"x1": {}, "y1": {}, "x2": {}, "y2": {}}
        policy[view_key] = view_slots
    return raw, policy


def _implementation_provenance(*, predecessor_digest: str = "1" * 64) -> dict[str, Any]:
    return {
        "git_head": "0" * 40,
        "git_dirty": False,
        "git_dirty_diff_sha256": "2" * 64,
        "relevant_file_digests": {
            "scripts/research/score_sorted_fn_fixed_budget.py": "3" * 64,
            PREDECESSOR_PRIMITIVES_FILE: predecessor_digest,
            "scripts/research/merge_sorted_fn_successor_score_shards.py": "4" * 64,
            "scripts/research/prepare_sorted_fn_successor_inputs.py": "5" * 64,
        },
    }


def _batch_admission_not_requested() -> dict[str, Any]:
    return {"schema_version": "batched_full_reforward_parity.v1", "status": "not_requested", "requested_batch_size": 1, "effective_batch_size": 1}


def _batch_admission_passed(*, requested: int = 8) -> dict[str, Any]:
    return {
        "schema_version": "batched_full_reforward_parity.v1",
        "status": "passed",
        "requested_batch_size": requested,
        "effective_batch_size": requested,
        "coordinate_logprob_max_abs_diff_tolerance": 1e-3,
        "argmax_requirement": "identical_within_coordinate_token_domain",
        "comparisons": [],
        "sha256": "6" * 64,
    }


def _accounting_entry(context_id: str) -> dict[str, Any]:
    return {
        "scoring_backend": scorer.FULL_REFORWARD_SCORING_BACKEND,
        "context_id": context_id,
        "group_id": sha256_json({"context_id": context_id}),
        "root_prefix_length": 3,
        "root_calls": 1,
        "logical_token_step_requests": 4,
        "logical_token_step_requests_by_depth": {"0": 1, "1": 1, "2": 1, "3": 1},
        "actual_forward_calls": 4,
        "physical_model_forward_calls": 4,
        "configured_full_reforward_batch_size": 1,
        "maximum_observed_model_forward_batch_size": 1,
        "physical_forward_batch_histogram": {"1": 4},
    }


def _admission(*, context_ids: list[str], batch_admission: dict[str, Any] | None = None) -> dict[str, Any]:
    content: dict[str, Any] = {
        "parity_status": "failed",
        "cache_admission_policy": None,
        "selected_backend": scorer.FULL_REFORWARD_SCORING_BACKEND,
        "cache_enabled": False,
        "use_cache": False,
        "atol": scorer.CACHE_PARITY_ATOL,
        "rtol": scorer.CACHE_PARITY_RTOL,
        "all_score_rows_backend": scorer.FULL_REFORWARD_SCORING_BACKEND,
        "backend_mixing_detected": False,
        "fallback_trigger": scorer.PARITY_FAILURE_FALLBACK_TRIGGER,
        "cache_path_note": "this successor scorer never attempts the KV-cache path",
        "batched_reforward_admission": batch_admission or _batch_admission_not_requested(),
        "per_context_accounting": [_accounting_entry(cid) for cid in context_ids],
    }
    return {**content, "sha256": sha256_json(content)}


#: run_sorted_fn_successor_behavior.validate_live_runtime_identity's own
#: frozen sealed check-key set (that function raises before ever returning
#: unless every check already passed, so a genuinely live-sealed receipt's
#: postload can only ever carry these keys, all literally True).
_POSTLOAD_CHECK_KEYS = (
    "model_identity",
    "model_identity_fingerprint",
    "tokenizer_identity",
    "runtime_identity",
    "effective_settings_extra_fields",
    "observed_attn_implementation",
    "observed_model_dtype",
    "processor_identity_fingerprint",
    "resolved_config_fingerprint",
    "generation_config_fingerprint",
    "precision",
)


def _postload_admission(
    *, projection_sha256: str = "5" * 64, checks_override: dict[str, Any] | None = None
) -> dict[str, Any]:
    checks = {key: True for key in _POSTLOAD_CHECK_KEYS}
    if checks_override:
        checks.update(checks_override)
    return {
        "status": "passed_before_generation",
        "checks": checks,
        "observed_projection_sha256": projection_sha256,
        "frozen_projection_sha256": projection_sha256,
    }


def _runtime_identity_admission(*, postload: dict[str, Any] | None = None) -> dict[str, Any]:
    """preload is the ledger/runtime-identity.json sha256 attestation domain;
    postload (a separate digest domain -- the actually-opened HF session
    compared against runtime-identity.json's own raw identity dicts) is
    never derived from the ledger sha256s here, matching the real producer."""

    return {
        "preload": {"status": "passed", "tokenizer_identity_sha256": TOKENIZER_IDENTITY_SHA256, "model_identity_sha256": MODEL_IDENTITY_SHA256},
        "postload": postload if postload is not None else _postload_admission(),
    }


def _build_selection(
    rows: list[dict[str, Any]],
    *,
    requested_context_ids: list[str] | None = None,
    selected_rungs_override: list[str] | None = None,
) -> dict[str, Any]:
    """Mirror ``score_sorted_fn_fixed_budget.build_execution_selection``'s shape.

    Only the fields the merger actually cross-checks are given real,
    self-consistent values (derived from ``rows`` unless overridden).
    """

    selected_rungs = (
        selected_rungs_override if selected_rungs_override is not None else sorted({str(row["rung"]) for row in rows})
    )
    selected_context_ids = sorted({str(row["context_id"]) for row in rows})
    candidate_keyset = sorted([[str(row["context_id"]), str(row["candidate_id"])] for row in rows])
    content = {
        "context_selection_sha256": "0" * 64,
        "rung_selection_sha256": "0" * 64,
        "requested_context_ids": requested_context_ids if requested_context_ids is not None else selected_context_ids,
        "selected_context_ids": selected_context_ids,
        "selected_rungs": selected_rungs,
        "candidate_count": len(rows),
        "candidate_keyset_sha256": sha256_json(candidate_keyset),
    }
    return {**content, "selection_sha256": sha256_json(content)}


def _shard_receipt(
    *,
    planned: dict[str, Any],
    context_ids: list[str],
    rows: list[dict[str, Any]],
    batch_admission: dict[str, Any] | None = None,
    selected_rungs_override: list[str] | None = None,
) -> dict[str, Any]:
    rules_path: Path = planned["rules_path"]
    rules = planned["rules"]
    selection = _build_selection(rows, requested_context_ids=context_ids, selected_rungs_override=selected_rungs_override)
    return {
        "schema_version": SUCCESSOR_SCORE_RECEIPT_SCHEMA_VERSION,
        "unit_id": SUCCESSOR_SCORER_UNIT_ID,
        "generated_at_unix": 0.0,
        "command": ["score_sorted_fn_fixed_budget.py"],
        "runtime_execution_status": SUCCESSOR_LIVE_SCORING_SEALED_STATUS,
        "runtime_receipt_id": sha256_json({"contexts": context_ids}),
        "source_digests": {
            "registry": {"path": str(planned["registry_path"]), "sha256": sha256_file(planned["registry_path"])},
            "owner_context_ledger": {"path": str(planned["ledger_path"]), "sha256": sha256_file(planned["ledger_path"])},
            "decision_rules": {"path": str(rules_path), "sha256": sha256_file(rules_path)},
            "mechanism_decision_rules": {
                "path": str(planned["mechanism_rules_path"]),
                "sha256": sha256_file(planned["mechanism_rules_path"]),
            },
            "fixed_budget_candidates": {"path": str(planned["fixed_budget_path"]), "sha256": sha256_file(planned["fixed_budget_path"])},
            "planner_receipt": {"path": str(planned["planner_receipt_path"]), "sha256": sha256_file(planned["planner_receipt_path"])},
            "runtime_identity": {"path": "runtime-identity.json", "sha256": "9" * 64},
        },
        "decision_rules": {"path": str(rules_path), "file_sha256": sha256_file(rules_path), "core_rule_digest": rules.rules_digest},
        "model_identity": MODEL_IDENTITY,
        "tokenizer_identity": TOKENIZER_IDENTITY,
        "backend_session": {"precision": "float32", "model_identity": MODEL_IDENTITY, "tokenizer_identity": TOKENIZER_IDENTITY},
        "environment": {"python_version": "3.12", "torch_version": "2.0", "cuda_available": True, "device_name": "fake-gpu", "tf32": False},
        "live_config_admission": {"status": "passed"},
        "runtime_identity_admission": _runtime_identity_admission(),
        "implementation_provenance": _implementation_provenance(),
        "numeric_reproduction_tolerance": rules.numeric_tolerance,
        "context_selection": {"included_context_ids": context_ids},
        "selection": selection,
        "candidate_count": 0,
        "owners_covered": 0,
        "likelihood_channels": {
            "raw": "fp32_log_softmax_over_the_complete_unfiltered_lm_head_vocabulary",
            "auxiliary_policy": [fixed_budget_scorer.scorer._policy_view_key(1.0), fixed_budget_scorer.scorer._policy_view_key(1.10)],
            "policy_is_not_a_model_likelihood": True,
            "subset_normalization_forbidden": True,
        },
        "scoring_backend_admission": _admission(context_ids=context_ids, batch_admission=batch_admission),
        "execution_architecture": {"kv_cache_ever_used": False, "strategy": "..."},
        "not_reused": {"free_coordinate_tree": "...", "conditional_x1_domain": "..."},
        "output_artifacts": {
            "fixed_budget_scores": {
                "path": "fn-fixed-budget-scores.jsonl",
                "sha256": "8" * 64,
                "row_count": len(rows),
                "selection_sha256": selection["selection_sha256"],
            }
        },
    }


def _fixed_budget_for_context(planned: dict[str, Any], context_id: str) -> list[dict[str, Any]]:
    return [row for row in planned["fixed_budget_rows"] if row["owner_context_id"] == context_id]


def _score_rows_for_contexts(planned: dict[str, Any], context_ids: list[str]) -> list[dict[str, Any]]:
    rules = planned["rules"]
    rows: list[dict[str, Any]] = []
    for context_id in context_ids:
        ledger_row = planned["ledger_by_context"][context_id]
        for candidate in _fixed_budget_for_context(planned, context_id):
            raw, policy = _raw_and_policy(candidate["coord_token_ids"])
            rows.append(
                {
                    "schema_version": SUCCESSOR_SCORE_ROW_SCHEMA_VERSION,
                    "unit_id": SUCCESSOR_SCORER_UNIT_ID,
                    "candidate_id": candidate["candidate_id"],
                    "context_id": context_id,
                    "owner_context_id": context_id,
                    "diagnostic_owner_id": ledger_row["diagnostic_owner_id"],
                    "gt_owner_id": ledger_row["gt_owner_id"],
                    "image_id": ledger_row["image_id"],
                    "rung": candidate["rung"],
                    "region": candidate["region"],
                    "population": candidate["population"],
                    "is_control": bool(candidate.get("is_control", False)),
                    "control_kind": candidate.get("control_kind"),
                    "matched_control_group": candidate["matched_control_group"],
                    "family_id": candidate["family_id"],
                    "iou_to_target": candidate.get("iou_to_target"),
                    "source_digest": candidate["source_digest"],
                    "coord_token_ids": list(candidate["coord_token_ids"]),
                    "candidate_neighborhood_member": bool(candidate.get("candidate_neighborhood_member", False)),
                    "candidate_neighborhood_id": candidate.get("candidate_neighborhood_id"),
                    "mechanism_decision_rules_sha256": candidate["mechanism_decision_rules_sha256"],
                    "other_owner_gt_owner_id": candidate.get("other_owner_gt_owner_id"),
                    "other_owner_selection_trace": candidate.get("other_owner_selection_trace"),
                    "exact_gt_singleton_member": bool(candidate.get("exact_gt_singleton_member", False)),
                    "exact_gt_singleton_id": candidate.get("exact_gt_singleton_id"),
                    "native_repetition_penalty_stratum": 1.0,
                    "rule_digest": rules.rules_digest,
                    "ledger_context_tokens_sha256": ledger_row["context_tokens"]["token_ids_sha256"],
                    "raw_model_logprob": raw,
                    "auxiliary_policy_scores": policy,
                    "likelihood_channel_note": "raw_model_logprob is the unmodified fp32 lm-head channel",
                }
            )
    return rows


def _write_shard(tmp_path: Path, name: str, rows: list[dict[str, Any]], receipt: dict[str, Any]) -> ShardInput:
    scores_path = tmp_path / f"{name}.jsonl"
    receipt_path = tmp_path / f"{name}-receipt.json"
    _write_jsonl(scores_path, rows)
    _write_json(receipt_path, receipt)
    return ShardInput(scores_path, receipt_path)


def _real_merge(
    planned: dict[str, Any],
    *,
    batch_admission: dict[str, Any] | None = None,
    only_rungs: set[str] | None = None,
) -> dict[str, Any]:
    """Build one real merge over an explicit, self-consistent rung selection.

    Defaults to the ``scalar_smoke`` rung only: this fixture's
    ``strict_rescued`` owner (gt:img1:0) is the sole source of
    ``scalar_smoke`` candidates and every non-rung-selection-specific test in
    this file attests under ``run_mode="scalar_smoke"``.  ``score_sorted_fn_
    fixed_budget.py`` never scores an implicit all-rungs domain (its
    ``--include-rung`` is fail-closed and required), so this fixture must not
    silently mix rungs either.
    """

    selected_rungs = only_rungs if only_rungs is not None else {"scalar_smoke"}
    all_context_ids = sorted(planned["ledger_by_context"])
    rows = [row for row in _score_rows_for_contexts(planned, all_context_ids) if row["rung"] in selected_rungs]
    context_ids = sorted({row["context_id"] for row in rows})
    receipt = _shard_receipt(planned=planned, context_ids=context_ids, rows=rows, batch_admission=batch_admission)
    shard = _write_shard(planned["tmp_path"], "shard0", rows, receipt)
    return merge_shards(
        shards=[shard],
        owner_context_ledger=planned["ledger_path"],
        decision_rules=planned["rules_path"],
        mechanism_decision_rules=planned["mechanism_rules_path"],
        fixed_budget_candidates=planned["fixed_budget_path"],
        planner_receipt=planned["planner_receipt_path"],
        fn_mechanism_registry=planned["registry_path"],
        output_dir=planned["tmp_path"] / "merged",
    )


def _write_merge_receipt(planned: dict[str, Any], name: str, receipt: dict[str, Any]) -> Path:
    path = planned["tmp_path"] / f"{name}.json"
    _write_json(path, receipt)
    return path


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def test_scalar_smoke_acceptance_passes_for_effective_batch_size_one(planned: dict[str, Any]) -> None:
    _real_merge(planned)
    merge_receipt_path = planned["tmp_path"] / "merged" / "fn-successor-merged-scores-receipt.json"

    document = attest(
        merge_receipt_path=merge_receipt_path,
        decision_rules_path=planned["rules_path"],
        mechanism_decision_rules_path=planned["mechanism_rules_path"],
        run_mode="scalar_smoke",
        output_dir=planned["tmp_path"] / "attest",
    )

    assert document["schema_version"] == ATTESTATION_SCHEMA_VERSION
    assert document["unit_id"] == SUCCESSOR_UNIT_ID
    assert document["run_mode"] == "scalar_smoke"
    assert document["selected_rungs"] == ["scalar_smoke"]
    assert document["disposition"] == "accepted"
    assert document["scalar_acceptance_backend"]["status"] == "passed"
    assert document["scalar_acceptance_backend"]["effective_batch_sizes"] == [1]
    assert document["scalar_acceptance_backend"]["selected_backend"] == scorer.FULL_REFORWARD_SCORING_BACKEND
    assert document["decision_channel_attestation"]["repetition_penalty_stratum"] == {"value": 1.0, "status": "passed"}
    assert document["decision_channel_attestation"]["decision_bearing_channel"] == "raw_model_logprob.complete_box_logprob_sum"
    assert document["decision_channel_attestation"]["successor_scorer_schema"] == {
        "row_schema_version": SUCCESSOR_SCORE_ROW_SCHEMA_VERSION,
        "receipt_schema_version": SUCCESSOR_SCORE_RECEIPT_SCHEMA_VERSION,
    }
    assert document["successor_scorer_provenance"]["unit_id"] == SUCCESSOR_SCORER_UNIT_ID
    assert document["successor_scorer_provenance"]["unit_id"] != scorer.UNIT_ID
    assert document["imported_predecessor_primitive_provenance"]["predecessor_primitives_file"] == PREDECESSOR_PRIMITIVES_FILE
    assert document["imported_predecessor_primitive_provenance"]["is_not_a_unit_ownership_claim"] is True


# --------------------------------------------------------------------------
# run_mode <-> selected_rungs fail-closed binding (score_sorted_fn_fixed_
# budget.py's --include-rung is always explicit and required; run_mode must
# be selection-derived, never a caller-chosen bypass around the scalar
# population/batch gates).
# --------------------------------------------------------------------------


def test_scalar_smoke_run_mode_rejects_a_merge_that_also_selected_l1(planned: dict[str, Any]) -> None:
    """A self-consistent (non-silent) mixed rung selection is still refused under run_mode=scalar_smoke.

    ``--include-rung scalar_smoke --include-rung L1`` is a legitimate,
    honestly-declared merge (the merger accepts it: one shard's own selected
    rungs exactly equal its own observed row rungs), but scalar_smoke
    acceptance's frozen 30/30/{0,7} population lattice and required
    effective-batch-size-1 gate must never be entered with a non-scalar row
    quietly along for the ride.
    """

    _real_merge(planned, only_rungs={"scalar_smoke", "L1"})
    merge_receipt_path = planned["tmp_path"] / "merged" / "fn-successor-merged-scores-receipt.json"

    with pytest.raises(RunAttestationError, match=r"requires the merge receipt's selected_rungs to be exactly \['scalar_smoke'\]"):
        attest(
            merge_receipt_path=merge_receipt_path,
            decision_rules_path=planned["rules_path"],
            mechanism_decision_rules_path=planned["mechanism_rules_path"],
            run_mode="scalar_smoke",
            output_dir=planned["tmp_path"] / "attest",
        )


def test_scale_mode_rejects_a_merge_that_also_selected_scalar_smoke(planned: dict[str, Any]) -> None:
    """The symmetric bypass this gate closes: --include-rung scalar_smoke plus
    a batched (effective_batch_size > 1) run attested under --run-mode scale
    would otherwise skip the scalar_smoke population/batch=1 gates entirely
    (``_scalar_acceptance_backend`` treats non-scalar_smoke run_mode as
    ``not_applicable_scale_mode``). Refusing any scalar_smoke rung presence
    under run_mode=scale closes that bypass outright.
    """

    _real_merge(planned, batch_admission=_batch_admission_passed(requested=4), only_rungs={"scalar_smoke", "L1"})
    merge_receipt_path = planned["tmp_path"] / "merged" / "fn-successor-merged-scores-receipt.json"

    with pytest.raises(RunAttestationError, match="forbids any scalar_smoke"):
        attest(
            merge_receipt_path=merge_receipt_path,
            decision_rules_path=planned["rules_path"],
            mechanism_decision_rules_path=planned["mechanism_rules_path"],
            run_mode="scale",
            output_dir=planned["tmp_path"] / "attest",
        )


def test_merge_receipt_selected_rungs_tampered_to_claim_scalar_smoke_only_fails(planned: dict[str, Any]) -> None:
    """Defense in depth: even a merge receipt that (post-merge) falsely
    claims ``selected_rungs == ["scalar_smoke"]`` while its rows are actually
    a different rung is independently re-derived and rejected here, not
    trusted from the merger's already-sealed declaration."""

    _real_merge(planned, only_rungs={"L1"})
    merge_receipt_path = planned["tmp_path"] / "merged" / "fn-successor-merged-scores-receipt.json"
    merge_receipt = json.loads(merge_receipt_path.read_text(encoding="utf-8"))
    assert merge_receipt["selected_rungs"] == ["L1"]
    merge_receipt["selected_rungs"] = ["scalar_smoke"]
    merge_receipt_path.write_text(json.dumps(merge_receipt), encoding="utf-8")

    with pytest.raises(RunAttestationError, match="does not exactly equal the rungs actually present"):
        attest(
            merge_receipt_path=merge_receipt_path,
            decision_rules_path=planned["rules_path"],
            mechanism_decision_rules_path=planned["mechanism_rules_path"],
            run_mode="scalar_smoke",
            output_dir=planned["tmp_path"] / "attest",
        )


def test_scalar_smoke_acceptance_rejects_batched_reforward(planned: dict[str, Any]) -> None:
    _real_merge(planned, batch_admission=_batch_admission_passed(requested=4))
    merge_receipt_path = planned["tmp_path"] / "merged" / "fn-successor-merged-scores-receipt.json"

    with pytest.raises(RunAttestationError, match="scalar_smoke acceptance requires"):
        attest(
            merge_receipt_path=merge_receipt_path,
            decision_rules_path=planned["rules_path"],
        mechanism_decision_rules_path=planned["mechanism_rules_path"],
            run_mode="scalar_smoke",
            output_dir=planned["tmp_path"] / "attest",
        )


def test_scale_mode_accepts_batched_reforward(planned: dict[str, Any]) -> None:
    _real_merge(planned, batch_admission=_batch_admission_passed(requested=4), only_rungs={"L0", "L1"})
    merge_receipt_path = planned["tmp_path"] / "merged" / "fn-successor-merged-scores-receipt.json"

    document = attest(
        merge_receipt_path=merge_receipt_path,
        decision_rules_path=planned["rules_path"],
        mechanism_decision_rules_path=planned["mechanism_rules_path"],
        run_mode="scale",
        output_dir=planned["tmp_path"] / "attest",
    )

    assert document["disposition"] == "accepted"
    assert document["scalar_acceptance_backend"]["status"] == "not_applicable_scale_mode"
    assert document["scalar_acceptance_backend"]["required"] is False
    assert document["scalar_acceptance_backend"]["effective_batch_sizes"] == [4]


def test_merge_receipt_claiming_predecessor_unit_id_is_rejected(planned: dict[str, Any]) -> None:
    merge_receipt = _real_merge(planned)
    tampered = copy.deepcopy(merge_receipt)
    tampered["unit_id"] = scorer.UNIT_ID
    tampered_path = _write_merge_receipt(planned, "tampered-unit-id", tampered)

    with pytest.raises(RunAttestationError, match="predecessor scorer's unit_id"):
        attest(
            merge_receipt_path=tampered_path,
            decision_rules_path=planned["rules_path"],
        mechanism_decision_rules_path=planned["mechanism_rules_path"],
            run_mode="scalar_smoke",
            output_dir=planned["tmp_path"] / "attest",
        )


def test_merge_receipt_with_predecessor_scorer_provenance_is_rejected(planned: dict[str, Any]) -> None:
    merge_receipt = _real_merge(planned)
    tampered = copy.deepcopy(merge_receipt)
    tampered["successor_scorer_provenance"]["unit_id"] = scorer.UNIT_ID
    tampered_path = _write_merge_receipt(planned, "tampered-scorer-provenance", tampered)

    with pytest.raises(RunAttestationError, match="successor_scorer_provenance.unit_id"):
        attest(
            merge_receipt_path=tampered_path,
            decision_rules_path=planned["rules_path"],
        mechanism_decision_rules_path=planned["mechanism_rules_path"],
            run_mode="scalar_smoke",
            output_dir=planned["tmp_path"] / "attest",
        )


def test_merge_receipt_with_wrong_schema_version_is_rejected(planned: dict[str, Any]) -> None:
    merge_receipt = _real_merge(planned)
    tampered = copy.deepcopy(merge_receipt)
    tampered["schema_version"] = "not-the-real-schema"
    tampered_path = _write_merge_receipt(planned, "tampered-schema", tampered)

    with pytest.raises(RunAttestationError, match="schema_version"):
        attest(
            merge_receipt_path=tampered_path,
            decision_rules_path=planned["rules_path"],
        mechanism_decision_rules_path=planned["mechanism_rules_path"],
            run_mode="scalar_smoke",
            output_dir=planned["tmp_path"] / "attest",
        )


def test_merge_receipt_with_wrong_row_schema_provenance_is_rejected(planned: dict[str, Any]) -> None:
    merge_receipt = _real_merge(planned)
    tampered = copy.deepcopy(merge_receipt)
    tampered["successor_scorer_provenance"]["row_schema_version"] = "not-a-real-row-schema"
    tampered_path = _write_merge_receipt(planned, "tampered-row-schema", tampered)

    with pytest.raises(RunAttestationError, match="row_schema_version"):
        attest(
            merge_receipt_path=tampered_path,
            decision_rules_path=planned["rules_path"],
        mechanism_decision_rules_path=planned["mechanism_rules_path"],
            run_mode="scalar_smoke",
            output_dir=planned["tmp_path"] / "attest",
        )


def test_stale_merged_scores_on_disk_is_rejected(planned: dict[str, Any]) -> None:
    _real_merge(planned)
    scores_path = planned["tmp_path"] / "merged" / "fn-successor-merged-scores.jsonl"
    scores_path.write_text(scores_path.read_text() + "\n", encoding="utf-8")
    merge_receipt_path = planned["tmp_path"] / "merged" / "fn-successor-merged-scores-receipt.json"

    with pytest.raises(RunAttestationError, match="stale merge output"):
        attest(
            merge_receipt_path=merge_receipt_path,
            decision_rules_path=planned["rules_path"],
        mechanism_decision_rules_path=planned["mechanism_rules_path"],
            run_mode="scalar_smoke",
            output_dir=planned["tmp_path"] / "attest",
        )


def test_invalid_run_mode_is_rejected(planned: dict[str, Any]) -> None:
    _real_merge(planned)
    merge_receipt_path = planned["tmp_path"] / "merged" / "fn-successor-merged-scores-receipt.json"

    with pytest.raises(RunAttestationError, match="run-mode"):
        attest(
            merge_receipt_path=merge_receipt_path,
            decision_rules_path=planned["rules_path"],
        mechanism_decision_rules_path=planned["mechanism_rules_path"],
            run_mode="not_a_real_mode",
            output_dir=planned["tmp_path"] / "attest",
        )


def test_exact_create_identical_rerun(planned: dict[str, Any]) -> None:
    _real_merge(planned)
    merge_receipt_path = planned["tmp_path"] / "merged" / "fn-successor-merged-scores-receipt.json"
    output_dir = planned["tmp_path"] / "attest"

    first = attest(merge_receipt_path=merge_receipt_path, decision_rules_path=planned["rules_path"], mechanism_decision_rules_path=planned["mechanism_rules_path"], run_mode="scalar_smoke", output_dir=output_dir)
    second = attest(merge_receipt_path=merge_receipt_path, decision_rules_path=planned["rules_path"], mechanism_decision_rules_path=planned["mechanism_rules_path"], run_mode="scalar_smoke", output_dir=output_dir)

    assert first == second
    attestation_path = output_dir / "fn-successor-run-attestation.json"
    before = attestation_path.read_bytes()
    attest(merge_receipt_path=merge_receipt_path, decision_rules_path=planned["rules_path"], mechanism_decision_rules_path=planned["mechanism_rules_path"], run_mode="scalar_smoke", output_dir=output_dir)
    assert attestation_path.read_bytes() == before


def test_attesting_a_scalar_smoke_only_merge_under_run_mode_scale_fails(planned: dict[str, Any]) -> None:
    """A merge whose selected_rungs is exactly ['scalar_smoke'] can only ever
    be attested under --run-mode scalar_smoke: scale mode forbids any
    scalar_smoke rung selection or row outright, so it can never reach (and
    thus never silently bypass) the scalar acceptance gates by reattesting
    the same merge under a different run_mode."""

    _real_merge(planned)
    merge_receipt_path = planned["tmp_path"] / "merged" / "fn-successor-merged-scores-receipt.json"
    output_dir = planned["tmp_path"] / "attest"
    attest(merge_receipt_path=merge_receipt_path, decision_rules_path=planned["rules_path"], mechanism_decision_rules_path=planned["mechanism_rules_path"], run_mode="scalar_smoke", output_dir=output_dir)

    with pytest.raises(RunAttestationError, match="forbids any scalar_smoke"):
        attest(merge_receipt_path=merge_receipt_path, decision_rules_path=planned["rules_path"], mechanism_decision_rules_path=planned["mechanism_rules_path"], run_mode="scale", output_dir=output_dir)


# --------------------------------------------------------------------------
# Mechanism-decision-rules parent/self-digest and reference/neighborhood
# field-validation negative tests.
# --------------------------------------------------------------------------


def _tamper_mechanism_rules_file(planned: dict[str, Any], document: dict[str, Any], *, merge_receipt_path: Path) -> None:
    """Directly rewrite mechanism-decision-rules.json on disk (post-merge)
    and re-seal only the merge receipt's own binding to it, so the attestor's
    outer exact-join gate does not fire before its own deeper self-digest/
    parent-binding validation gets a chance to run.
    """

    planned["mechanism_rules_path"].write_text(json.dumps(document), encoding="utf-8")
    merge_receipt = json.loads(merge_receipt_path.read_text(encoding="utf-8"))
    merge_receipt["source_digests"]["mechanism_decision_rules"]["sha256"] = sha256_file(planned["mechanism_rules_path"])
    merge_receipt_path.write_text(json.dumps(merge_receipt), encoding="utf-8")


def _tamper_merged_scores(merge_receipt_path: Path, *, transform: Any) -> None:
    """Directly rewrite the merged score rows and re-seal only the merge
    receipt's own ``output_artifacts.merged_scores`` binding (post-merge),
    to exercise the attestor's independent re-validation over the final
    surface rather than relying on the merger having already rejected it."""

    merge_receipt = json.loads(merge_receipt_path.read_text(encoding="utf-8"))
    scores_path = Path(merge_receipt["output_artifacts"]["merged_scores"]["path"])
    rows = [json.loads(line) for line in scores_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = transform(rows)
    scores_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    merge_receipt["output_artifacts"]["merged_scores"]["sha256"] = sha256_file(scores_path)
    merge_receipt["output_artifacts"]["merged_scores"]["row_count"] = len(rows)
    merge_receipt_path.write_text(json.dumps(merge_receipt), encoding="utf-8")


def test_mechanism_decision_rules_self_digest_tamper_fails(planned: dict[str, Any]) -> None:
    _real_merge(planned)
    merge_receipt_path = planned["tmp_path"] / "merged" / "fn-successor-merged-scores-receipt.json"
    document = dict(planned["mechanism_rules_document"])
    document["geometry"] = {**document["geometry"], "tampered": True}
    _tamper_mechanism_rules_file(planned, document, merge_receipt_path=merge_receipt_path)

    with pytest.raises(RunAttestationError, match="self_digest does not reconstruct"):
        attest(
            merge_receipt_path=merge_receipt_path,
            decision_rules_path=planned["rules_path"],
            mechanism_decision_rules_path=planned["mechanism_rules_path"],
            run_mode="scale",
            output_dir=planned["tmp_path"] / "attest",
        )


def test_mechanism_decision_rules_parent_execution_digest_mismatch_fails(planned: dict[str, Any]) -> None:
    _real_merge(planned)
    merge_receipt_path = planned["tmp_path"] / "merged" / "fn-successor-merged-scores-receipt.json"
    document = dict(planned["mechanism_rules_document"])
    document["upstream_digests"] = {**document["upstream_digests"], "execution_landscape_decision_rules_sha256": "0" * 64}
    document["self_digest"] = sha256_json({k: v for k, v in document.items() if k != "self_digest"})
    _tamper_mechanism_rules_file(planned, document, merge_receipt_path=merge_receipt_path)

    with pytest.raises(RunAttestationError, match="parent execution digest"):
        attest(
            merge_receipt_path=merge_receipt_path,
            decision_rules_path=planned["rules_path"],
            mechanism_decision_rules_path=planned["mechanism_rules_path"],
            run_mode="scale",
            output_dir=planned["tmp_path"] / "attest",
        )


def test_mechanism_decision_rules_registry_binding_mismatch_fails(planned: dict[str, Any]) -> None:
    _real_merge(planned)
    merge_receipt_path = planned["tmp_path"] / "merged" / "fn-successor-merged-scores-receipt.json"
    document = dict(planned["mechanism_rules_document"])
    document["upstream_digests"] = {**document["upstream_digests"], "fn_mechanism_registry_sha256": "0" * 64}
    document["self_digest"] = sha256_json({k: v for k, v in document.items() if k != "self_digest"})
    _tamper_mechanism_rules_file(planned, document, merge_receipt_path=merge_receipt_path)

    with pytest.raises(RunAttestationError, match="registry binding"):
        attest(
            merge_receipt_path=merge_receipt_path,
            decision_rules_path=planned["rules_path"],
            mechanism_decision_rules_path=planned["mechanism_rules_path"],
            run_mode="scale",
            output_dir=planned["tmp_path"] / "attest",
        )


def test_stale_mechanism_decision_rules_supplied_to_attest_fails(planned: dict[str, Any]) -> None:
    _real_merge(planned)
    merge_receipt_path = planned["tmp_path"] / "merged" / "fn-successor-merged-scores-receipt.json"
    tampered_path = planned["tmp_path"] / "tampered-mechanism-decision-rules.json"
    document = dict(planned["mechanism_rules_document"])
    document["geometry"] = {**document["geometry"], "unbound_local_edit": True}
    tampered_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(RunAttestationError, match="does not match the merge receipt's bound digest"):
        attest(
            merge_receipt_path=merge_receipt_path,
            decision_rules_path=planned["rules_path"],
            mechanism_decision_rules_path=tampered_path,
            run_mode="scale",
            output_dir=planned["tmp_path"] / "attest",
        )


def test_merged_row_mechanism_binding_mismatch_fails(planned: dict[str, Any]) -> None:
    _real_merge(planned, only_rungs={"L0", "L1"})
    merge_receipt_path = planned["tmp_path"] / "merged" / "fn-successor-merged-scores-receipt.json"
    _tamper_merged_scores(
        merge_receipt_path, transform=lambda rows: [{**rows[0], "mechanism_decision_rules_sha256": "0" * 64}, *rows[1:]]
    )

    with pytest.raises(RunAttestationError, match="mechanism_decision_rules_sha256"):
        attest(
            merge_receipt_path=merge_receipt_path,
            decision_rules_path=planned["rules_path"],
            mechanism_decision_rules_path=planned["mechanism_rules_path"],
            run_mode="scale",
            output_dir=planned["tmp_path"] / "attest",
        )


def test_merged_reference_row_missing_selection_trace_fails(planned: dict[str, Any]) -> None:
    _real_merge(planned, only_rungs={"L0", "L1"})
    merge_receipt_path = planned["tmp_path"] / "merged" / "fn-successor-merged-scores-receipt.json"

    def transform(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        reference_index = next(i for i, row in enumerate(rows) if row["population"] == "reference")
        rows[reference_index] = {**rows[reference_index], "other_owner_selection_trace": {}}
        return rows

    _tamper_merged_scores(merge_receipt_path, transform=transform)

    with pytest.raises(RunAttestationError, match="other_owner_selection_trace"):
        attest(
            merge_receipt_path=merge_receipt_path,
            decision_rules_path=planned["rules_path"],
            mechanism_decision_rules_path=planned["mechanism_rules_path"],
            run_mode="scale",
            output_dir=planned["tmp_path"] / "attest",
        )


def test_merged_reference_row_wrong_region_fails(planned: dict[str, Any]) -> None:
    _real_merge(planned, only_rungs={"L0", "L1"})
    merge_receipt_path = planned["tmp_path"] / "merged" / "fn-successor-merged-scores-receipt.json"

    def transform(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        reference_index = next(i for i, row in enumerate(rows) if row["population"] == "reference")
        rows[reference_index] = {**rows[reference_index], "region": "background"}
        return rows

    _tamper_merged_scores(merge_receipt_path, transform=transform)

    with pytest.raises(RunAttestationError, match="region"):
        attest(
            merge_receipt_path=merge_receipt_path,
            decision_rules_path=planned["rules_path"],
            mechanism_decision_rules_path=planned["mechanism_rules_path"],
            run_mode="scale",
            output_dir=planned["tmp_path"] / "attest",
        )


def test_merged_neighborhood_member_without_id_fails(planned: dict[str, Any]) -> None:
    _real_merge(planned, only_rungs={"L0", "L1"})
    merge_receipt_path = planned["tmp_path"] / "merged" / "fn-successor-merged-scores-receipt.json"

    def transform(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        member_index = next(i for i, row in enumerate(rows) if row["candidate_neighborhood_member"] is True)
        rows[member_index] = {**rows[member_index], "candidate_neighborhood_id": None}
        return rows

    _tamper_merged_scores(merge_receipt_path, transform=transform)

    with pytest.raises(RunAttestationError, match="candidate_neighborhood_id"):
        attest(
            merge_receipt_path=merge_receipt_path,
            decision_rules_path=planned["rules_path"],
            mechanism_decision_rules_path=planned["mechanism_rules_path"],
            run_mode="scale",
            output_dir=planned["tmp_path"] / "attest",
        )


def test_scalar_smoke_population_admission_reports_primary_counts(planned: dict[str, Any]) -> None:
    _real_merge(planned)
    merge_receipt_path = planned["tmp_path"] / "merged" / "fn-successor-merged-scores-receipt.json"

    document = attest(
        merge_receipt_path=merge_receipt_path,
        decision_rules_path=planned["rules_path"],
        mechanism_decision_rules_path=planned["mechanism_rules_path"],
        run_mode="scalar_smoke",
        output_dir=planned["tmp_path"] / "attest",
    )

    admission = document["scalar_smoke_population_admission"]
    assert admission["status"] == "passed"
    assert admission["target_count"] == 30
    assert admission["decoy_count"] == 30
    for counts in admission["per_context_counts"].values():
        assert counts["target"] == 30
        assert counts["decoy"] == 30
        assert counts["reference"] in {0, 7}


def test_scalar_smoke_population_admission_rejects_drifted_target_count(planned: dict[str, Any]) -> None:
    _real_merge(planned)
    merge_receipt_path = planned["tmp_path"] / "merged" / "fn-successor-merged-scores-receipt.json"

    def transform(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        target_index = next(i for i, row in enumerate(rows) if row["rung"] == "scalar_smoke" and row["population"] == "target")
        return [row for i, row in enumerate(rows) if i != target_index]

    _tamper_merged_scores(merge_receipt_path, transform=transform)

    with pytest.raises(RunAttestationError, match="primary population counts drifted"):
        attest(
            merge_receipt_path=merge_receipt_path,
            decision_rules_path=planned["rules_path"],
            mechanism_decision_rules_path=planned["mechanism_rules_path"],
            run_mode="scalar_smoke",
            output_dir=planned["tmp_path"] / "attest",
        )


def test_end_to_end_producer_shaped_attest_binds_both_rules_and_reference_provenance(planned: dict[str, Any]) -> None:
    """One realistic scalar_smoke attestation, asserting both rules
    documents are bound and reference/neighborhood provenance survives."""

    _real_merge(planned)
    merge_receipt_path = planned["tmp_path"] / "merged" / "fn-successor-merged-scores-receipt.json"

    document = attest(
        merge_receipt_path=merge_receipt_path,
        decision_rules_path=planned["rules_path"],
        mechanism_decision_rules_path=planned["mechanism_rules_path"],
        run_mode="scalar_smoke",
        output_dir=planned["tmp_path"] / "attest",
    )

    mechanism_attestation = document["mechanism_decision_rules_attestation"]
    assert mechanism_attestation["self_digest"] == planned["mechanism_rules_document"]["self_digest"]
    assert mechanism_attestation["parent_execution_rules_sha256"] == sha256_file(planned["rules_path"])
    assert mechanism_attestation["population_counts"]["reference"] > 0
    assert document["disposition"] == "accepted"


def test_end_to_end_large_box_scalar_smoke_attestation_carries_exact_singleton(planned_large_box: dict[str, Any]) -> None:
    """F3 exact-GT-singleton carry-through all the way through scalar_smoke
    attestation for genuinely large (100x100+) GT boxes, not just the small
    boxes used elsewhere in this file -- proving this attestor's
    exact_gt_singleton validation is not merely passing by accident of
    small-box geometry."""

    planned = planned_large_box
    # _real_merge defaults to the scalar_smoke rung only (a fail-closed,
    # single-rung selection), so the expected singleton set is scoped to
    # that same rung -- not every rung's singleton across the whole lattice.
    singleton_candidates = [
        row
        for row in planned["fixed_budget_rows"]
        if row["exact_gt_singleton_member"] is True and row["rung"] == "scalar_smoke"
    ]
    assert singleton_candidates

    _real_merge(planned)
    merge_receipt_path = planned["tmp_path"] / "merged" / "fn-successor-merged-scores-receipt.json"

    document = attest(
        merge_receipt_path=merge_receipt_path,
        decision_rules_path=planned["rules_path"],
        mechanism_decision_rules_path=planned["mechanism_rules_path"],
        run_mode="scalar_smoke",
        output_dir=planned["tmp_path"] / "attest",
    )

    assert document["disposition"] == "accepted"
    merged_scores_path = Path(document["merged_scores"]["path"])
    merged_rows = [json.loads(line) for line in merged_scores_path.read_text().splitlines()]
    merged_singletons = [row for row in merged_rows if row["exact_gt_singleton_member"] is True]
    assert merged_singletons
    assert {row["exact_gt_singleton_id"] for row in merged_singletons} == {row["exact_gt_singleton_id"] for row in singleton_candidates}
    admission = document["scalar_smoke_population_admission"]
    assert admission["status"] == "passed"
