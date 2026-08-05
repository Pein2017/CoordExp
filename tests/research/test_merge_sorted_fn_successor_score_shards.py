"""Contracts for the successor-local, generic sorted-FN score-shard merger.

All fixtures run the real, unchanged ``prepare_sorted_fn_successor_inputs``
planner to produce a genuine owner-context-ledger, decision rules,
fixed-budget candidate lattice, and planner receipt (mirroring
``test_score_sorted_fn_fixed_budget.py``'s own fixture, since both consume
the exact same five upstream artifacts). Score shards are then hand-built
against that real candidate/context identity space, matching the real
``score_sorted_fn_fixed_budget.py`` row/receipt schema field-for-field, so
the merger's validation is exercised against its true successor-scorer
contract rather than an approximation.
"""

from __future__ import annotations

import copy
import json
import math
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
    MERGE_SCHEMA_VERSION,
    PREDECESSOR_PRIMITIVES_FILE,
    SUCCESSOR_LIVE_SCORING_SEALED_STATUS,
    SUCCESSOR_OUTPUT_JSONL_NAME,
    SUCCESSOR_RECEIPT_NAME,
    SUCCESSOR_SCORE_RECEIPT_SCHEMA_VERSION,
    SUCCESSOR_SCORE_ROW_SCHEMA_VERSION,
    SUCCESSOR_SCORER_UNIT_ID,
    MergeContractError,
    ShardInput,
    merge_shards,
)


# --------------------------------------------------------------------------
# Drift guard: the four successor-scorer constants this module must hardcode
# (see its own module docstring for why it cannot import
# score_sorted_fn_fixed_budget.py directly -- that would be circular).
# --------------------------------------------------------------------------


def test_hardcoded_successor_scorer_constants_match_the_real_module() -> None:
    assert SUCCESSOR_SCORE_ROW_SCHEMA_VERSION == fixed_budget_scorer.SCHEMA_VERSION
    assert SUCCESSOR_SCORE_RECEIPT_SCHEMA_VERSION == fixed_budget_scorer.RECEIPT_SCHEMA_VERSION
    assert SUCCESSOR_LIVE_SCORING_SEALED_STATUS == fixed_budget_scorer.LIVE_SCORING_SEALED_STATUS
    assert SUCCESSOR_OUTPUT_JSONL_NAME == fixed_budget_scorer.OUTPUT_JSONL_NAME
    assert SUCCESSOR_RECEIPT_NAME == fixed_budget_scorer.RECEIPT_NAME
    assert SUCCESSOR_SCORER_UNIT_ID == fixed_budget_scorer.UNIT_ID
    assert SUCCESSOR_SCORER_UNIT_ID != scorer.UNIT_ID  # never the predecessor's


# --------------------------------------------------------------------------
# Minimal, self-consistent rules-template fixture (mirrors
# test_score_sorted_fn_fixed_budget.py's own fixture: same five upstream
# artifacts, same real planner). Model/tokenizer identity dicts are chosen
# so their own sha256_json digest is exactly what the ledger declares --
# needed so hand-built shard receipts below can honestly self-reconstruct.
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
            "non_production_reason": "merge-shard-contract test fixture; not a decision-bearing run",
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
    """Run the real, unchanged planner and return every produced artifact path."""

    fixtures_dir = tmp_path / "fixtures"
    owner_ledger_path = fixtures_dir / "owner-ledger.jsonl"
    _write_jsonl(
        owner_ledger_path,
        [
            _owner_ledger_row("gt:img1:0", "img1", 0, "widget"),
            _owner_ledger_row("gt:img1:1", "img1", 1, "widget"),
            _owner_ledger_row("gt:img1:2", "img1", 2, "widget"),
        ],
    )
    panel_path = fixtures_dir / "panel.jsonl"
    _write_jsonl(panel_path, [_panel_line("img1", 1024, 1024, boxes)])
    rollout_path = fixtures_dir / "rollout.json"
    _write_json(rollout_path, _rollout_document("img1", "greedy", 0, _PROMPT, [_row_chunk(1), _row_chunk(10)]))
    rules_template_path = fixtures_dir / "rules-template.json"
    _write_json(rules_template_path, RULES_TEMPLATE)
    descriptions_path = fixtures_dir / "descriptions.json"
    _write_json(descriptions_path, _DESCRIPTION_SUPPLEMENT)

    roles = [
        _context_role("root:gt:img1:0", "gt:img1:0"),
        _context_role("root:gt:img1:1", "gt:img1:1"),
        _context_role("root:gt:img1:2", "gt:img1:2"),
        _context_role("due_turn:gt:img1:0", "gt:img1:0", role_kind="due_turn_context", cut_marker=1),
        _context_role("due_turn:gt:img1:2", "gt:img1:2", role_kind="due_turn_context", cut_marker=10),
    ]
    registry_path = tmp_path / "registry.json"
    _write_json(registry_path, _mechanism_registry(roles, targets=[{"gt_owner_id": "gt:img1:2", "cohort": "strict_rescued"}]))

    out_dir = tmp_path / "plan"
    receipt = prepare_sorted_fn_successor_inputs(
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
        "planner_receipt": receipt,
        "ledger_rows": ledger_rows,
        "ledger_by_context": {row["context_id"]: row for row in ledger_rows},
        "fixed_budget_rows": fixed_budget_rows,
        "rules": rules,
    }


@pytest.fixture()
def planned(tmp_path: Path) -> dict[str, Any]:
    """Small (6x6) boxes.

    Owners 0 and 1 share one exact ground-truth box on purpose: their "root"
    contexts below both share the empty self-prefix (execution_dedup group),
    and now their fixed-budget lattices are geometrically coincident too,
    giving a genuine cross-role identical-candidate pair for the dedup
    tests. Small boxes keep coverage regardless of GT-box size for the
    near_gt_micro F3 exact-GT-singleton collapse (see
    ``planned_large_box`` below for the large-box proof).
    """

    return _build_planned(tmp_path, boxes=[[100, 100, 106, 106], [100, 100, 106, 106], [500, 500, 506, 506]])


@pytest.fixture()
def planned_large_box(tmp_path: Path) -> dict[str, Any]:
    """Large (100x100+) boxes, proving F3 exact-singleton carry-through is
    not an artifact of small-box geometry: local index 0 of the near_gt_micro
    family is now literally the GT box for boxes of this size too (see
    prepare_sorted_fn_successor_inputs._near_micro_boxes)."""

    return _build_planned(tmp_path, boxes=[[100, 100, 200, 200], [300, 300, 380, 360], [500, 500, 560, 540]])


# --------------------------------------------------------------------------
# Synthetic score-shard construction, matching score_sorted_fn_fixed_budget.py's
# real row/receipt schema field-for-field, built around the real ledger/rules/
# fixed-budget candidates produced above.
# --------------------------------------------------------------------------


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


def _batch_admission_failed_fallback(*, requested: int = 8) -> dict[str, Any]:
    return {
        "schema_version": "batched_full_reforward_parity.v1",
        "status": "failed_scalar_fallback_required",
        "requested_batch_size": requested,
        "effective_batch_size": 1,
        "coordinate_logprob_max_abs_diff_tolerance": 1e-3,
        "argmax_requirement": "identical_within_coordinate_token_domain",
        "comparisons": [],
        "sha256": "7" * 64,
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


def _admission(*, context_ids: list[str], batch_admission: dict[str, Any] | None = None, cache_enabled: bool = False) -> dict[str, Any]:
    content: dict[str, Any] = {
        "parity_status": "failed",
        "cache_admission_policy": None,
        "selected_backend": scorer.KV_CACHE_SCORING_BACKEND if cache_enabled else scorer.FULL_REFORWARD_SCORING_BACKEND,
        "cache_enabled": cache_enabled,
        "use_cache": cache_enabled,
        "atol": scorer.CACHE_PARITY_ATOL,
        "rtol": scorer.CACHE_PARITY_RTOL,
        "all_score_rows_backend": scorer.KV_CACHE_SCORING_BACKEND if cache_enabled else scorer.FULL_REFORWARD_SCORING_BACKEND,
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


def _runtime_identity_admission(
    *,
    model_identity_sha256: str = MODEL_IDENTITY_SHA256,
    tokenizer_identity_sha256: str = TOKENIZER_IDENTITY_SHA256,
    postload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """preload is the ledger/runtime-identity.json sha256 attestation domain;
    postload (a separate digest domain -- the actually-opened HF session
    compared against runtime-identity.json's own raw identity dicts) is
    never derived from the ledger sha256s here, matching the real producer."""

    return {
        "preload": {"status": "passed", "tokenizer_identity_sha256": tokenizer_identity_sha256, "model_identity_sha256": model_identity_sha256},
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
    self-consistent values (derived from ``rows`` unless overridden); the
    upstream sub-selection digests (``context_selection_sha256``/
    ``rung_selection_sha256``) are opaque placeholders here since the merger
    never re-derives them independently.
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
    cache_enabled: bool = False,
    schema_version: str = SUCCESSOR_SCORE_RECEIPT_SCHEMA_VERSION,
    unit_id: str = SUCCESSOR_SCORER_UNIT_ID,
    runtime_execution_status: str = SUCCESSOR_LIVE_SCORING_SEALED_STATUS,
    model_identity: dict[str, Any] | None = None,
    tokenizer_identity: dict[str, Any] | None = None,
    runtime_identity_admission: dict[str, Any] | None = None,
    predecessor_digest: str = "1" * 64,
    selected_rungs_override: list[str] | None = None,
) -> dict[str, Any]:
    rules_path: Path = planned["rules_path"]
    rules = planned["rules"]
    model_identity = MODEL_IDENTITY if model_identity is None else model_identity
    tokenizer_identity = TOKENIZER_IDENTITY if tokenizer_identity is None else tokenizer_identity
    selection = _build_selection(rows, requested_context_ids=context_ids, selected_rungs_override=selected_rungs_override)
    return {
        "schema_version": schema_version,
        "unit_id": unit_id,
        "generated_at_unix": 0.0,
        "command": ["score_sorted_fn_fixed_budget.py"],
        "runtime_execution_status": runtime_execution_status,
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
        "model_identity": model_identity,
        "tokenizer_identity": tokenizer_identity,
        "backend_session": {"precision": "float32", "model_identity": model_identity, "tokenizer_identity": tokenizer_identity},
        "environment": {"python_version": "3.12", "torch_version": "2.0", "cuda_available": True, "device_name": "fake-gpu", "tf32": False},
        "live_config_admission": {"status": "passed"},
        "runtime_identity_admission": runtime_identity_admission or _runtime_identity_admission(),
        "implementation_provenance": _implementation_provenance(predecessor_digest=predecessor_digest),
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
        "scoring_backend_admission": _admission(context_ids=context_ids, batch_admission=batch_admission, cache_enabled=cache_enabled),
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


def _score_rows_for_contexts(planned: dict[str, Any], context_ids: list[str], *, unit_id: str = SUCCESSOR_SCORER_UNIT_ID, schema_version: str = SUCCESSOR_SCORE_ROW_SCHEMA_VERSION) -> list[dict[str, Any]]:
    rules = planned["rules"]
    rows: list[dict[str, Any]] = []
    for context_id in context_ids:
        ledger_row = planned["ledger_by_context"][context_id]
        for candidate in _fixed_budget_for_context(planned, context_id):
            raw, policy = _raw_and_policy(candidate["coord_token_ids"])
            rows.append(
                {
                    "schema_version": schema_version,
                    "unit_id": unit_id,
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


def _all_context_ids(planned: dict[str, Any]) -> list[str]:
    return sorted(planned["ledger_by_context"])


def _merge(planned: dict[str, Any], shards: list[ShardInput], output_dir: Path) -> dict[str, Any]:
    return merge_shards(
        shards=shards,
        owner_context_ledger=planned["ledger_path"],
        decision_rules=planned["rules_path"],
        mechanism_decision_rules=planned["mechanism_rules_path"],
        fixed_budget_candidates=planned["fixed_budget_path"],
        planner_receipt=planned["planner_receipt_path"],
        fn_mechanism_registry=planned["registry_path"],
        output_dir=output_dir,
    )


def _shard_for_contexts(planned: dict[str, Any], name: str, context_ids: list[str], **receipt_overrides: Any) -> ShardInput:
    rows = _score_rows_for_contexts(planned, context_ids)
    receipt = _shard_receipt(planned=planned, context_ids=context_ids, rows=rows, **receipt_overrides)
    return _write_shard(planned["tmp_path"], name, rows, receipt)


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def test_planner_produced_at_least_five_contexts(planned: dict[str, Any]) -> None:
    assert len(planned["ledger_by_context"]) > 4


def test_successor_shard_merge_accepted_over_more_than_four_contexts(planned: dict[str, Any]) -> None:
    context_ids = _all_context_ids(planned)
    shard = _shard_for_contexts(planned, "shard0", context_ids)

    result = _merge(planned, [shard], planned["tmp_path"] / "out")

    assert result["schema_version"] == MERGE_SCHEMA_VERSION
    assert result["unit_id"] == SUCCESSOR_UNIT_ID
    assert result["candidate_count"] == sum(len(_fixed_budget_for_context(planned, cid)) for cid in context_ids)
    assert sorted(result["context_selection"]["selected_context_ids"]) == context_ids
    assert result["successor_scorer_provenance"]["unit_id"] == SUCCESSOR_SCORER_UNIT_ID
    assert result["successor_scorer_provenance"]["unit_id"] != scorer.UNIT_ID
    assert (
        result["imported_predecessor_primitive_provenance"]["predecessor_primitives_file"]
        == PREDECESSOR_PRIMITIVES_FILE
    )
    assert result["imported_predecessor_primitive_provenance"]["predecessor_primitives_file_sha256"] == "1" * 64
    assert result["decision_channel"]["name"] == "raw_model_logprob.complete_box_logprob_sum"
    assert result["decision_channel"]["primary_repetition_penalty_stratum"] == 1.0


def test_successor_shard_merge_accepted_when_split_across_arbitrary_context_subsets(planned: dict[str, Any]) -> None:
    context_ids = _all_context_ids(planned)
    left, right = context_ids[:2], context_ids[2:]
    shard_left = _shard_for_contexts(planned, "left", left)
    shard_right = _shard_for_contexts(planned, "right", right)

    result = _merge(planned, [shard_left, shard_right], planned["tmp_path"] / "out")

    assert sorted(result["context_selection"]["selected_context_ids"]) == context_ids


def test_execution_dedup_group_with_consistent_scores_passes(planned: dict[str, Any]) -> None:
    ctx0, ctx1 = "ctx:fn:root:gt:img1:0", "ctx:fn:root:gt:img1:1"
    assert planned["ledger_by_context"][ctx0]["execution_dedup_key"] == planned["ledger_by_context"][ctx1]["execution_dedup_key"]
    fb0 = {tuple(row["coord_token_ids"]): row for row in _fixed_budget_for_context(planned, ctx0)}
    fb1 = {tuple(row["coord_token_ids"]): row for row in _fixed_budget_for_context(planned, ctx1)}
    assert set(fb0) & set(fb1), "fixture must produce at least one coincident geometry across the two owners"

    # _deterministic_slot_value is keyed on coord_token_ids, not candidate_id,
    # so this shared geometry naturally scores identically across both roles.
    shard = _shard_for_contexts(planned, "dedup_ok", [ctx0, ctx1])

    result = _merge(planned, [shard], planned["tmp_path"] / "out")

    assert result["execution_dedup"]["groups_with_multiple_roles"] >= 1
    assert result["execution_dedup"]["status"] == "consistent"
    assert {ctx0, ctx1} <= set(result["context_selection"]["selected_context_ids"])


def test_execution_dedup_group_with_inconsistent_scores_fails(planned: dict[str, Any]) -> None:
    ctx0, ctx1 = "ctx:fn:root:gt:img1:0", "ctx:fn:root:gt:img1:1"
    fb0 = {tuple(row["coord_token_ids"]): row for row in _fixed_budget_for_context(planned, ctx0)}
    fb1 = {tuple(row["coord_token_ids"]): row for row in _fixed_budget_for_context(planned, ctx1)}
    shared_geometry = next(iter(set(fb0) & set(fb1)))

    rows = _score_rows_for_contexts(planned, [ctx0, ctx1])
    mirror_candidate_id = fb1[shared_geometry]["candidate_id"]
    mirror_row = next(row for row in rows if row["candidate_id"] == mirror_candidate_id)
    # Internally-consistent disagreement (slot and sum both shift together)
    # so this exercises the execution-dedup check specifically, not the
    # separate sum-of-slots internal-consistency check.
    mirror_row["raw_model_logprob"]["x1_logprob"] += 5.0
    mirror_row["raw_model_logprob"]["complete_box_logprob_sum"] += 5.0

    receipt = _shard_receipt(planned=planned, context_ids=[ctx0, ctx1], rows=rows)
    shard = _write_shard(planned["tmp_path"], "dedup_bad", rows, receipt)

    with pytest.raises(MergeContractError, match="execution-deduplicated rows disagree"):
        _merge(planned, [shard], planned["tmp_path"] / "out")


def test_same_context_duplicated_byte_identical_across_shards_dedups(planned: dict[str, Any]) -> None:
    context_ids = _all_context_ids(planned)
    rows = _score_rows_for_contexts(planned, context_ids)
    receipt = _shard_receipt(planned=planned, context_ids=context_ids, rows=rows)
    shard_a = _write_shard(planned["tmp_path"], "a", rows, receipt)
    shard_b = _write_shard(planned["tmp_path"], "b", rows, receipt)  # exact duplicate: same context, byte-identical

    result = _merge(planned, [shard_a, shard_b], planned["tmp_path"] / "out")

    assert result["candidate_count"] == len(rows)  # deduplicated, not doubled


def test_same_context_duplicated_with_disagreeing_content_fails(planned: dict[str, Any]) -> None:
    context_ids = _all_context_ids(planned)
    rows_a = _score_rows_for_contexts(planned, context_ids)
    rows_b = copy.deepcopy(rows_a)
    rows_b[0]["raw_model_logprob"]["complete_box_logprob_sum"] += 1.0
    rows_b[0]["raw_model_logprob"]["x1_logprob"] += 1.0
    shard_a = _write_shard(planned["tmp_path"], "a", rows_a, _shard_receipt(planned=planned, context_ids=context_ids, rows=rows_a))
    shard_b = _write_shard(planned["tmp_path"], "b", rows_b, _shard_receipt(planned=planned, context_ids=context_ids, rows=rows_b))

    with pytest.raises(MergeContractError, match="shards disagree on identical candidate"):
        _merge(planned, [shard_a, shard_b], planned["tmp_path"] / "out")


def test_partial_overlap_of_one_context_across_shards_fails(planned: dict[str, Any]) -> None:
    context_ids = _all_context_ids(planned)
    target_context = context_ids[0]
    rows_all = _score_rows_for_contexts(planned, [target_context])
    rows_partial = rows_all[:-1]  # missing exactly one candidate from the same context
    shard_a = _write_shard(
        planned["tmp_path"], "a", rows_all, _shard_receipt(planned=planned, context_ids=[target_context], rows=rows_all)
    )
    shard_b = _write_shard(
        planned["tmp_path"], "b", rows_partial, _shard_receipt(planned=planned, context_ids=[target_context], rows=rows_partial)
    )

    with pytest.raises(MergeContractError, match="partially overlap"):
        _merge(planned, [shard_a, shard_b], planned["tmp_path"] / "out")


def test_missing_candidate_fails(planned: dict[str, Any]) -> None:
    context_ids = _all_context_ids(planned)
    rows = _score_rows_for_contexts(planned, context_ids)
    rows.pop()
    receipt = _shard_receipt(planned=planned, context_ids=context_ids, rows=rows)
    shard = _write_shard(planned["tmp_path"], "missing", rows, receipt)

    with pytest.raises(MergeContractError, match="missing"):
        _merge(planned, [shard], planned["tmp_path"] / "out")


# --------------------------------------------------------------------------
# Selection-aware completeness and rung-selection integrity (score_sorted_
# fn_fixed_budget.py's fail-closed, repeatable --include-rung): a shard is
# only required to be complete against the exact (context, rung) domain its
# own receipt selected -- never against every rung the fixed-budget lattice
# ever declared for that context.
# --------------------------------------------------------------------------


def test_single_rung_shard_is_not_rejected_against_the_full_multi_rung_context_candidate_set(
    planned: dict[str, Any],
) -> None:
    """Regression: a shard scoring only one rung of a multi-rung context must be accepted whole.

    Before selection-aware reconciliation, a shard scoring e.g. only
    ``scalar_smoke`` for a ``strict_rescued`` context (which also has ``L1``
    candidates) was wrongly rejected as "missing" the L1 candidates it never
    selected in the first place.
    """

    strict_context = "ctx:fn:root:gt:img1:2"
    all_context_candidates = _fixed_budget_for_context(planned, strict_context)
    assert {row["rung"] for row in all_context_candidates} == {"scalar_smoke", "L1"}
    rows = [row for row in _score_rows_for_contexts(planned, [strict_context]) if row["rung"] == "scalar_smoke"]
    assert 0 < len(rows) < len(all_context_candidates)

    receipt = _shard_receipt(planned=planned, context_ids=[strict_context], rows=rows)
    shard = _write_shard(planned["tmp_path"], "scalar_only", rows, receipt)

    result = _merge(planned, [shard], planned["tmp_path"] / "out")

    assert result["candidate_count"] == len(rows)
    assert result["selected_rungs"] == ["scalar_smoke"]


def test_shard_receipt_claiming_scalar_smoke_selection_but_carrying_l1_rows_fails(planned: dict[str, Any]) -> None:
    strict_context = "ctx:fn:root:gt:img1:2"
    context_rows = _score_rows_for_contexts(planned, [strict_context])
    scalar_rows = [row for row in context_rows if row["rung"] == "scalar_smoke"]
    l1_row = next(row for row in context_rows if row["rung"] == "L1")
    rows = [*scalar_rows, l1_row]

    receipt = _shard_receipt(
        planned=planned, context_ids=[strict_context], rows=rows, selected_rungs_override=["scalar_smoke"]
    )
    shard = _write_shard(planned["tmp_path"], "claims_scalar_has_l1", rows, receipt)

    with pytest.raises(MergeContractError, match="does not exactly equal the rungs actually observed"):
        _merge(planned, [shard], planned["tmp_path"] / "out")


def test_overlapping_shards_disagreeing_on_selected_rungs_is_a_silent_mixed_rung_merge_and_fails(
    planned: dict[str, Any],
) -> None:
    """Two individually rung-pure shards must not jointly deposit a silently mixed rung set.

    Shard A honestly declares/observes only ``scalar_smoke`` for the shared
    context; shard B honestly declares/observes ``scalar_smoke`` (the exact
    same rows) plus ``L1`` for that same context. Both receipts are
    individually self-consistent, but they disagree on the rung selection
    for a context they both cover -- refused outright, never silently
    unioned.
    """

    strict_context = "ctx:fn:root:gt:img1:2"
    context_rows = _score_rows_for_contexts(planned, [strict_context])
    scalar_rows = [row for row in context_rows if row["rung"] == "scalar_smoke"]
    l1_rows = [row for row in context_rows if row["rung"] == "L1"]
    assert scalar_rows and l1_rows

    shard_a_rows = scalar_rows
    shard_b_rows = scalar_rows + l1_rows

    shard_a = _write_shard(
        planned["tmp_path"], "a", shard_a_rows, _shard_receipt(planned=planned, context_ids=[strict_context], rows=shard_a_rows)
    )
    shard_b = _write_shard(
        planned["tmp_path"], "b", shard_b_rows, _shard_receipt(planned=planned, context_ids=[strict_context], rows=shard_b_rows)
    )

    with pytest.raises(MergeContractError, match="disagree on their selected rungs"):
        _merge(planned, [shard_a, shard_b], planned["tmp_path"] / "out")


def test_extra_foreign_candidate_fails(planned: dict[str, Any]) -> None:
    context_ids = _all_context_ids(planned)
    rows = _score_rows_for_contexts(planned, context_ids)
    foreign = copy.deepcopy(rows[0])
    foreign["candidate_id"] = "cand:not-a-declared-fixed-budget-candidate"
    rows.append(foreign)
    receipt = _shard_receipt(planned=planned, context_ids=context_ids, rows=rows)
    shard = _write_shard(planned["tmp_path"], "extra", rows, receipt)

    with pytest.raises(MergeContractError, match="not a declared fixed-budget candidate"):
        _merge(planned, [shard], planned["tmp_path"] / "out")


def test_wrong_owner_for_context_fails(planned: dict[str, Any]) -> None:
    context_ids = _all_context_ids(planned)
    rows = _score_rows_for_contexts(planned, context_ids)
    rows[0]["gt_owner_id"] = "gt:img1:999-not-the-real-owner"
    receipt = _shard_receipt(planned=planned, context_ids=context_ids, rows=rows)
    shard = _write_shard(planned["tmp_path"], "wrong_owner", rows, receipt)

    with pytest.raises(MergeContractError, match="wrong owner"):
        _merge(planned, [shard], planned["tmp_path"] / "out")


def test_non_finite_score_fails(planned: dict[str, Any]) -> None:
    context_ids = _all_context_ids(planned)
    rows = _score_rows_for_contexts(planned, context_ids)
    rows[0]["raw_model_logprob"]["complete_box_logprob_sum"] = math.nan
    receipt = _shard_receipt(planned=planned, context_ids=context_ids, rows=rows)
    encoded_rows = "\n".join(json.dumps(row, allow_nan=True) for row in rows) + "\n"
    scores_path = planned["tmp_path"] / "nonfinite.jsonl"
    scores_path.write_text(encoded_rows, encoding="utf-8")
    receipt_path = planned["tmp_path"] / "nonfinite-receipt.json"
    _write_json(receipt_path, receipt)

    with pytest.raises(MergeContractError, match="finite"):
        _merge(planned, [ShardInput(scores_path, receipt_path)], planned["tmp_path"] / "out")


def test_predecessor_scorer_receipt_is_rejected(planned: dict[str, Any]) -> None:
    context_ids = _all_context_ids(planned)
    rows = _score_rows_for_contexts(planned, context_ids)
    receipt = _shard_receipt(
        planned=planned,
        context_ids=context_ids,
        rows=rows,
        schema_version=scorer.RECEIPT_SCHEMA_VERSION,
        unit_id=scorer.UNIT_ID,
    )
    shard = _write_shard(planned["tmp_path"], "predecessor_receipt", rows, receipt)

    with pytest.raises(MergeContractError, match="predecessor scorer artifacts are not accepted"):
        _merge(planned, [shard], planned["tmp_path"] / "out")


def test_predecessor_scorer_row_is_rejected(planned: dict[str, Any]) -> None:
    context_ids = _all_context_ids(planned)
    rows = _score_rows_for_contexts(planned, context_ids, unit_id=scorer.UNIT_ID, schema_version=scorer.SCHEMA_VERSION)
    receipt = _shard_receipt(planned=planned, context_ids=context_ids, rows=rows)
    shard = _write_shard(planned["tmp_path"], "predecessor_row", rows, receipt)

    with pytest.raises(MergeContractError, match="predecessor scorer artifacts are not accepted"):
        _merge(planned, [shard], planned["tmp_path"] / "out")


def test_unrecognized_schema_version_on_receipt_is_rejected(planned: dict[str, Any]) -> None:
    context_ids = _all_context_ids(planned)
    rows = _score_rows_for_contexts(planned, context_ids)
    receipt = _shard_receipt(planned=planned, context_ids=context_ids, rows=rows, schema_version="not-a-real-schema.v99")
    shard = _write_shard(planned["tmp_path"], "bad_schema", rows, receipt)

    with pytest.raises(MergeContractError, match="stale or foreign"):
        _merge(planned, [shard], planned["tmp_path"] / "out")


def test_unrecognized_schema_version_on_row_is_rejected(planned: dict[str, Any]) -> None:
    context_ids = _all_context_ids(planned)
    rows = _score_rows_for_contexts(planned, context_ids, schema_version="not-a-real-row-schema.v99")
    receipt = _shard_receipt(planned=planned, context_ids=context_ids, rows=rows)
    shard = _write_shard(planned["tmp_path"], "bad_row_schema", rows, receipt)

    with pytest.raises(MergeContractError, match="stale or foreign"):
        _merge(planned, [shard], planned["tmp_path"] / "out")


def test_stale_prefix_context_digest_fails(planned: dict[str, Any]) -> None:
    context_ids = _all_context_ids(planned)
    rows = _score_rows_for_contexts(planned, context_ids)
    rows[0]["ledger_context_tokens_sha256"] = "0" * 64
    receipt = _shard_receipt(planned=planned, context_ids=context_ids, rows=rows)
    shard = _write_shard(planned["tmp_path"], "stale_prefix", rows, receipt)

    with pytest.raises(MergeContractError, match="prefix digest"):
        _merge(planned, [shard], planned["tmp_path"] / "out")


def test_stale_source_digest_on_receipt_fails(planned: dict[str, Any]) -> None:
    context_ids = _all_context_ids(planned)
    rows = _score_rows_for_contexts(planned, context_ids)
    receipt = _shard_receipt(planned=planned, context_ids=context_ids, rows=rows)
    receipt["source_digests"]["owner_context_ledger"]["sha256"] = "0" * 64
    shard = _write_shard(planned["tmp_path"], "stale_source", rows, receipt)

    with pytest.raises(MergeContractError, match="source_digests.owner_context_ledger"):
        _merge(planned, [shard], planned["tmp_path"] / "out")


def test_fresh_score_rejects_predecessor_candidate_id_field(planned: dict[str, Any]) -> None:
    context_ids = _all_context_ids(planned)
    rows = _score_rows_for_contexts(planned, context_ids)
    rows[0]["predecessor_candidate_id"] = "cand:reused:0"
    receipt = _shard_receipt(planned=planned, context_ids=context_ids, rows=rows)
    shard = _write_shard(planned["tmp_path"], "reused", rows, receipt)

    with pytest.raises(MergeContractError, match="fresh live score"):
        _merge(planned, [shard], planned["tmp_path"] / "out")


def test_non_sealed_receipt_is_rejected(planned: dict[str, Any]) -> None:
    context_ids = _all_context_ids(planned)
    rows = _score_rows_for_contexts(planned, context_ids)
    receipt = _shard_receipt(
        planned=planned, context_ids=context_ids, rows=rows, runtime_execution_status="contract_validated_no_model_loaded"
    )
    shard = _write_shard(planned["tmp_path"], "not_sealed", rows, receipt)

    with pytest.raises(MergeContractError, match="not a sealed live"):
        _merge(planned, [shard], planned["tmp_path"] / "out")


def test_cache_enabled_backend_is_rejected(planned: dict[str, Any]) -> None:
    context_ids = _all_context_ids(planned)
    rows = _score_rows_for_contexts(planned, context_ids)
    receipt = _shard_receipt(planned=planned, context_ids=context_ids, rows=rows, cache_enabled=True)
    shard = _write_shard(planned["tmp_path"], "cached", rows, receipt)

    with pytest.raises(MergeContractError, match="raw fp32 uncached full-reforward"):
        _merge(planned, [shard], planned["tmp_path"] / "out")


@pytest.mark.parametrize(
    "batch_admission",
    [
        {"schema_version": "batched_full_reforward_parity.v1", "status": "not_requested", "requested_batch_size": 4, "effective_batch_size": 1},
        {"schema_version": "batched_full_reforward_parity.v1", "status": "passed", "requested_batch_size": 8, "effective_batch_size": 4},
        {"schema_version": "batched_full_reforward_parity.v1", "status": "failed_scalar_fallback_required", "requested_batch_size": 1, "effective_batch_size": 1},
    ],
)
def test_inconsistent_batch_parity_fallback_semantics_fails(planned: dict[str, Any], batch_admission: dict[str, Any]) -> None:
    context_ids = _all_context_ids(planned)
    rows = _score_rows_for_contexts(planned, context_ids)
    receipt = _shard_receipt(planned=planned, context_ids=context_ids, rows=rows, batch_admission=batch_admission)
    shard = _write_shard(planned["tmp_path"], "bad_batch", rows, receipt)

    with pytest.raises(MergeContractError, match="batched_reforward_admission"):
        _merge(planned, [shard], planned["tmp_path"] / "out")


def test_batch_admission_passed_with_consistent_semantics_accepts(planned: dict[str, Any]) -> None:
    context_ids = _all_context_ids(planned)
    rows = _score_rows_for_contexts(planned, context_ids)
    receipt = _shard_receipt(
        planned=planned, context_ids=context_ids, rows=rows, batch_admission=_batch_admission_passed(requested=4)
    )
    shard = _write_shard(planned["tmp_path"], "good_batch", rows, receipt)

    result = _merge(planned, [shard], planned["tmp_path"] / "out")
    assert result["candidate_count"] == len(rows)


def test_batch_admission_failed_fallback_with_consistent_semantics_accepts(planned: dict[str, Any]) -> None:
    context_ids = _all_context_ids(planned)
    rows = _score_rows_for_contexts(planned, context_ids)
    receipt = _shard_receipt(
        planned=planned, context_ids=context_ids, rows=rows, batch_admission=_batch_admission_failed_fallback(requested=4)
    )
    shard = _write_shard(planned["tmp_path"], "fallback_batch", rows, receipt)

    result = _merge(planned, [shard], planned["tmp_path"] / "out")
    assert result["candidate_count"] == len(rows)


def test_mixed_runtime_identity_model_identity_across_shards_fails(planned: dict[str, Any]) -> None:
    context_ids = _all_context_ids(planned)
    left, right = context_ids[:2], context_ids[2:]
    shard_left = _shard_for_contexts(planned, "left", left)
    foreign_model = {"kind": "fake-model", "name": "a-different-model"}
    foreign_sha = sha256_json(foreign_model)
    shard_right = _shard_for_contexts(
        planned,
        "right",
        right,
        model_identity=foreign_model,
        runtime_identity_admission=_runtime_identity_admission(model_identity_sha256=foreign_sha),
    )

    with pytest.raises(MergeContractError, match="model_identity"):
        _merge(planned, [shard_left, shard_right], planned["tmp_path"] / "out")


def test_ledger_declared_identity_mismatch_fails(planned: dict[str, Any]) -> None:
    context_ids = _all_context_ids(planned)
    rows = _score_rows_for_contexts(planned, context_ids)
    foreign_model = {"kind": "fake-model", "name": "not-the-ledger-model"}
    foreign_sha = sha256_json(foreign_model)
    receipt = _shard_receipt(
        planned=planned,
        context_ids=context_ids,
        rows=rows,
        model_identity=foreign_model,
        runtime_identity_admission=_runtime_identity_admission(model_identity_sha256=foreign_sha),
    )
    shard = _write_shard(planned["tmp_path"], "foreign_identity", rows, receipt)

    with pytest.raises(MergeContractError, match="ledger's declared identity"):
        _merge(planned, [shard], planned["tmp_path"] / "out")


# --------------------------------------------------------------------------
# Current live-sealed postload contract (run_sorted_fn_successor_behavior.
# validate_live_runtime_identity): status='passed_before_generation', an
# exact check-key set all literally True, and equal observed/frozen
# projection sha256s -- distinct from the ledger/preload sha256 domain.
# --------------------------------------------------------------------------


def test_exact_current_producer_contract_accepted_over_a_small_shard(planned: dict[str, Any]) -> None:
    """Regression: the exact live-sealed shape must merge cleanly end to end
    (the reported bug: a genuinely successful score run was rejected by a
    stale merger expecting the legacy postload shape)."""

    context_ids = _all_context_ids(planned)[:1]
    rows = _score_rows_for_contexts(planned, context_ids)
    receipt = _shard_receipt(planned=planned, context_ids=context_ids, rows=rows)
    shard = _write_shard(planned["tmp_path"], "current_contract", rows, receipt)

    result = _merge(planned, [shard], planned["tmp_path"] / "out")

    assert result["candidate_count"] == len(rows)


def test_legacy_postload_shape_with_status_passed_is_rejected(planned: dict[str, Any]) -> None:
    context_ids = _all_context_ids(planned)
    rows = _score_rows_for_contexts(planned, context_ids)
    legacy_postload = {
        "status": "passed",
        "observed_model_identity_sha256": MODEL_IDENTITY_SHA256,
        "observed_tokenizer_identity_sha256": TOKENIZER_IDENTITY_SHA256,
    }
    receipt = _shard_receipt(
        planned=planned,
        context_ids=context_ids,
        rows=rows,
        runtime_identity_admission=_runtime_identity_admission(postload=legacy_postload),
    )
    shard = _write_shard(planned["tmp_path"], "legacy_postload", rows, receipt)

    with pytest.raises(MergeContractError, match=r"postload\.status must be 'passed_before_generation'"):
        _merge(planned, [shard], planned["tmp_path"] / "out")


def test_postload_missing_a_check_key_is_rejected(planned: dict[str, Any]) -> None:
    context_ids = _all_context_ids(planned)
    rows = _score_rows_for_contexts(planned, context_ids)
    postload = _postload_admission()
    del postload["checks"]["precision"]
    receipt = _shard_receipt(
        planned=planned,
        context_ids=context_ids,
        rows=rows,
        runtime_identity_admission=_runtime_identity_admission(postload=postload),
    )
    shard = _write_shard(planned["tmp_path"], "missing_check_key", rows, receipt)

    with pytest.raises(MergeContractError, match="exactly the current check-key set"):
        _merge(planned, [shard], planned["tmp_path"] / "out")


def test_postload_extra_check_key_is_rejected(planned: dict[str, Any]) -> None:
    context_ids = _all_context_ids(planned)
    rows = _score_rows_for_contexts(planned, context_ids)
    postload = _postload_admission()
    postload["checks"]["an_unrecognized_extra_check"] = True
    receipt = _shard_receipt(
        planned=planned,
        context_ids=context_ids,
        rows=rows,
        runtime_identity_admission=_runtime_identity_admission(postload=postload),
    )
    shard = _write_shard(planned["tmp_path"], "extra_check_key", rows, receipt)

    with pytest.raises(MergeContractError, match="exactly the current check-key set"):
        _merge(planned, [shard], planned["tmp_path"] / "out")


def test_postload_one_false_check_is_rejected(planned: dict[str, Any]) -> None:
    context_ids = _all_context_ids(planned)
    rows = _score_rows_for_contexts(planned, context_ids)
    postload = _postload_admission(checks_override={"observed_model_dtype": False})
    receipt = _shard_receipt(
        planned=planned,
        context_ids=context_ids,
        rows=rows,
        runtime_identity_admission=_runtime_identity_admission(postload=postload),
    )
    shard = _write_shard(planned["tmp_path"], "false_check", rows, receipt)

    with pytest.raises(MergeContractError, match="non-passing check"):
        _merge(planned, [shard], planned["tmp_path"] / "out")


def test_postload_projection_sha256_mismatch_is_rejected(planned: dict[str, Any]) -> None:
    context_ids = _all_context_ids(planned)
    rows = _score_rows_for_contexts(planned, context_ids)
    postload = _postload_admission()
    postload["observed_projection_sha256"] = "6" * 64
    receipt = _shard_receipt(
        planned=planned,
        context_ids=context_ids,
        rows=rows,
        runtime_identity_admission=_runtime_identity_admission(postload=postload),
    )
    shard = _write_shard(planned["tmp_path"], "projection_mismatch", rows, receipt)

    with pytest.raises(MergeContractError, match="observed_projection_sha256 does not match"):
        _merge(planned, [shard], planned["tmp_path"] / "out")


@pytest.mark.parametrize(
    ("field_name", "malformed_value"),
    [
        ("observed_projection_sha256", "5" * 63),  # too short
        ("observed_projection_sha256", "5" * 65),  # too long
        ("observed_projection_sha256", "G" + "5" * 63),  # non-hex character
        ("observed_projection_sha256", "A" * 64),  # uppercase hex is rejected, not normalized
        ("frozen_projection_sha256", "5" * 63),
        ("frozen_projection_sha256", "A" * 64),
    ],
)
def test_postload_projection_sha256_malformed_is_rejected(planned: dict[str, Any], field_name: str, malformed_value: str) -> None:
    context_ids = _all_context_ids(planned)
    rows = _score_rows_for_contexts(planned, context_ids)
    postload = _postload_admission()
    postload[field_name] = malformed_value
    receipt = _shard_receipt(
        planned=planned,
        context_ids=context_ids,
        rows=rows,
        runtime_identity_admission=_runtime_identity_admission(postload=postload),
    )
    shard = _write_shard(planned["tmp_path"], f"malformed_{field_name}_{malformed_value[:8]}", rows, receipt)

    with pytest.raises(MergeContractError, match="must be a lowercase 64-character hex sha256 digest"):
        _merge(planned, [shard], planned["tmp_path"] / "out")


def test_backend_session_model_identity_drift_from_top_level_is_rejected(planned: dict[str, Any]) -> None:
    context_ids = _all_context_ids(planned)
    rows = _score_rows_for_contexts(planned, context_ids)
    receipt = _shard_receipt(planned=planned, context_ids=context_ids, rows=rows)
    receipt["backend_session"] = {**receipt["backend_session"], "model_identity": {"kind": "fake-model", "name": "drifted"}}
    shard = _write_shard(planned["tmp_path"], "backend_session_drift", rows, receipt)

    with pytest.raises(MergeContractError, match="backend_session.model_identity does not match"):
        _merge(planned, [shard], planned["tmp_path"] / "out")


def test_backend_session_tokenizer_identity_drift_from_top_level_is_rejected(planned: dict[str, Any]) -> None:
    context_ids = _all_context_ids(planned)
    rows = _score_rows_for_contexts(planned, context_ids)
    receipt = _shard_receipt(planned=planned, context_ids=context_ids, rows=rows)
    receipt["backend_session"] = {
        **receipt["backend_session"],
        "tokenizer_identity": {"kind": "fake-tokenizer", "name": "drifted"},
    }
    shard = _write_shard(planned["tmp_path"], "backend_session_tokenizer_drift", rows, receipt)

    with pytest.raises(MergeContractError, match="backend_session.tokenizer_identity does not match"):
        _merge(planned, [shard], planned["tmp_path"] / "out")


def test_implementation_provenance_mismatch_across_shards_fails(planned: dict[str, Any]) -> None:
    context_ids = _all_context_ids(planned)
    left, right = context_ids[:2], context_ids[2:]
    shard_left = _shard_for_contexts(planned, "left", left, predecessor_digest="1" * 64)
    shard_right = _shard_for_contexts(planned, "right", right, predecessor_digest="2" * 64)

    with pytest.raises(MergeContractError, match="identity"):
        _merge(planned, [shard_left, shard_right], planned["tmp_path"] / "out")


def test_missing_predecessor_primitives_digest_fails(planned: dict[str, Any]) -> None:
    context_ids = _all_context_ids(planned)
    rows = _score_rows_for_contexts(planned, context_ids)
    receipt = _shard_receipt(planned=planned, context_ids=context_ids, rows=rows)
    del receipt["implementation_provenance"]["relevant_file_digests"][PREDECESSOR_PRIMITIVES_FILE]
    shard = _write_shard(planned["tmp_path"], "missing_predecessor_digest", rows, receipt)

    with pytest.raises(MergeContractError, match="predecessor primitives file"):
        _merge(planned, [shard], planned["tmp_path"] / "out")


def test_stale_decision_rules_digest_fails(planned: dict[str, Any]) -> None:
    context_ids = _all_context_ids(planned)
    rows = _score_rows_for_contexts(planned, context_ids)
    receipt = _shard_receipt(planned=planned, context_ids=context_ids, rows=rows)
    receipt["decision_rules"]["core_rule_digest"] = "not-the-real-digest"
    shard = _write_shard(planned["tmp_path"], "stale_rules", rows, receipt)

    with pytest.raises(MergeContractError, match="rule"):
        _merge(planned, [shard], planned["tmp_path"] / "out")


def test_tampered_ledger_after_planner_seal_fails(planned: dict[str, Any]) -> None:
    context_ids = _all_context_ids(planned)
    shard = _shard_for_contexts(planned, "ok", context_ids)
    planned["ledger_path"].write_text(planned["ledger_path"].read_text() + "\n", encoding="utf-8")

    with pytest.raises(MergeContractError, match="planner receipt"):
        _merge(planned, [shard], planned["tmp_path"] / "out")


def test_exact_create_identical_rerun(planned: dict[str, Any]) -> None:
    context_ids = _all_context_ids(planned)
    shard = _shard_for_contexts(planned, "rerun", context_ids)
    output_dir = planned["tmp_path"] / "out"

    first = _merge(planned, [shard], output_dir)
    second = _merge(planned, [shard], output_dir)

    assert first == second
    scores_path = output_dir / "fn-successor-merged-scores.jsonl"
    receipt_path = output_dir / "fn-successor-merged-scores-receipt.json"
    before = (scores_path.read_bytes(), receipt_path.read_bytes())
    third = _merge(planned, [shard], output_dir)
    after = (scores_path.read_bytes(), receipt_path.read_bytes())
    assert before == after
    assert third == first


def test_rerun_with_different_content_at_same_output_dir_fails(planned: dict[str, Any]) -> None:
    context_ids = _all_context_ids(planned)
    shard = _shard_for_contexts(planned, "first", context_ids)
    output_dir = planned["tmp_path"] / "out"
    _merge(planned, [shard], output_dir)

    rows2 = _score_rows_for_contexts(planned, context_ids)
    rows2[0]["raw_model_logprob"]["complete_box_logprob_sum"] += 1.0
    rows2[0]["raw_model_logprob"]["x1_logprob"] += 1.0
    shard2 = _write_shard(
        planned["tmp_path"], "second", rows2, _shard_receipt(planned=planned, context_ids=context_ids, rows=rows2)
    )

    with pytest.raises(MergeContractError, match="already exists with different content"):
        _merge(planned, [shard2], output_dir)


def test_empty_shard_list_fails(planned: dict[str, Any]) -> None:
    with pytest.raises(MergeContractError, match="at least one"):
        _merge(planned, [], planned["tmp_path"] / "out")


# --------------------------------------------------------------------------
# Mechanism-decision-rules parent/self-digest and reference/neighborhood
# field-validation negative tests.
# --------------------------------------------------------------------------


def _reseal_planner_receipt_binding(planner_receipt_path: Path, *, section: str, key: str, mutated_path: Path) -> None:
    """After hand-mutating one downstream artifact, re-bind the planner
    receipt's own digest to it so the generic "stale input" gate does not
    fire before the merger's own semantic validation gets a chance to run.
    """

    receipt = json.loads(planner_receipt_path.read_text(encoding="utf-8"))
    entry = receipt[section][key]
    entry["sha256"] = sha256_file(mutated_path)
    if "row_count" in entry:
        entry["row_count"] = sum(1 for line in mutated_path.read_text(encoding="utf-8").splitlines() if line.strip())
    content = {k: v for k, v in receipt.items() if k != "receipt_digest"}
    receipt["receipt_digest"] = sha256_json(content)
    planner_receipt_path.write_text(json.dumps(receipt), encoding="utf-8")


def _rewrite_mechanism_rules(planned: dict[str, Any], document: dict[str, Any]) -> None:
    mechanism_rules_path: Path = planned["mechanism_rules_path"]
    mechanism_rules_path.write_text(json.dumps(document), encoding="utf-8")
    _reseal_planner_receipt_binding(
        planned["planner_receipt_path"], section="outputs", key="mechanism_decision_rules", mutated_path=mechanism_rules_path
    )


def _rewrite_fixed_budget_rows(planned: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    fixed_budget_path: Path = planned["fixed_budget_path"]
    fixed_budget_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    _reseal_planner_receipt_binding(
        planned["planner_receipt_path"], section="outputs", key="fixed_budget_candidates", mutated_path=fixed_budget_path
    )


def _reference_candidate(planned: dict[str, Any]) -> dict[str, Any]:
    return next(row for row in planned["fixed_budget_rows"] if row["population"] == "reference")


def _neighborhood_member_candidate(planned: dict[str, Any]) -> dict[str, Any]:
    return next(row for row in planned["fixed_budget_rows"] if row["candidate_neighborhood_member"] is True)


def test_planner_produced_reference_and_neighborhood_candidates(planned: dict[str, Any]) -> None:
    # Fixture self-check: this unit's three owners share one normalized
    # description in one image, so every context binds a same-description,
    # zero-overlap reference owner, and every near_gt_micro target row is a
    # candidate-neighborhood member.
    assert any(row["population"] == "reference" for row in planned["fixed_budget_rows"])
    assert any(row["candidate_neighborhood_member"] is True for row in planned["fixed_budget_rows"])


def test_mechanism_decision_rules_self_digest_tamper_fails(planned: dict[str, Any]) -> None:
    document = dict(planned["mechanism_rules_document"])
    document["geometry"] = {**document["geometry"], "tampered": True}
    _rewrite_mechanism_rules(planned, document)

    context_ids = _all_context_ids(planned)
    shard = _shard_for_contexts(planned, "shard0", context_ids)
    with pytest.raises(MergeContractError, match="self_digest does not reconstruct"):
        _merge(planned, [shard], planned["tmp_path"] / "out")


def test_mechanism_decision_rules_parent_execution_digest_mismatch_fails(planned: dict[str, Any]) -> None:
    document = dict(planned["mechanism_rules_document"])
    document["upstream_digests"] = {**document["upstream_digests"], "execution_landscape_decision_rules_sha256": "0" * 64}
    document["self_digest"] = sha256_json({k: v for k, v in document.items() if k != "self_digest"})
    _rewrite_mechanism_rules(planned, document)

    context_ids = _all_context_ids(planned)
    shard = _shard_for_contexts(planned, "shard0", context_ids)
    with pytest.raises(MergeContractError, match="parent binding"):
        _merge(planned, [shard], planned["tmp_path"] / "out")


def test_mechanism_decision_rules_registry_binding_mismatch_fails(planned: dict[str, Any]) -> None:
    document = dict(planned["mechanism_rules_document"])
    document["upstream_digests"] = {**document["upstream_digests"], "fn_mechanism_registry_sha256": "0" * 64}
    document["self_digest"] = sha256_json({k: v for k, v in document.items() if k != "self_digest"})
    _rewrite_mechanism_rules(planned, document)

    context_ids = _all_context_ids(planned)
    shard = _shard_for_contexts(planned, "shard0", context_ids)
    with pytest.raises(MergeContractError, match="registry binding"):
        _merge(planned, [shard], planned["tmp_path"] / "out")


def test_fixed_budget_row_mechanism_binding_mismatch_fails(planned: dict[str, Any]) -> None:
    rows = list(planned["fixed_budget_rows"])
    rows[0] = {**rows[0], "mechanism_decision_rules_sha256": "0" * 64}
    _rewrite_fixed_budget_rows(planned, rows)

    context_ids = _all_context_ids(planned)
    shard = _shard_for_contexts(planned, "shard0", context_ids)
    with pytest.raises(MergeContractError, match="does not match the loaded mechanism-decision-rules.json"):
        _merge(planned, [shard], planned["tmp_path"] / "out")


def test_reference_candidate_missing_selection_trace_fails(planned: dict[str, Any]) -> None:
    reference = _reference_candidate(planned)
    rows = [
        {**row, "other_owner_selection_trace": {}} if row["candidate_id"] == reference["candidate_id"] else row
        for row in planned["fixed_budget_rows"]
    ]
    _rewrite_fixed_budget_rows(planned, rows)

    context_ids = _all_context_ids(planned)
    shard = _shard_for_contexts(planned, "shard0", context_ids)
    with pytest.raises(MergeContractError, match="other_owner_selection_trace"):
        _merge(planned, [shard], planned["tmp_path"] / "out")


def test_reference_candidate_wrong_region_fails(planned: dict[str, Any]) -> None:
    reference = _reference_candidate(planned)
    rows = [
        {**row, "region": "background"} if row["candidate_id"] == reference["candidate_id"] else row
        for row in planned["fixed_budget_rows"]
    ]
    _rewrite_fixed_budget_rows(planned, rows)

    context_ids = _all_context_ids(planned)
    shard = _shard_for_contexts(planned, "shard0", context_ids)
    with pytest.raises(MergeContractError, match="region"):
        _merge(planned, [shard], planned["tmp_path"] / "out")


def test_non_reference_candidate_with_fabricated_other_owner_field_fails(planned: dict[str, Any]) -> None:
    target = next(row for row in planned["fixed_budget_rows"] if row["population"] == "target")
    rows = [
        {**row, "other_owner_gt_owner_id": "gt:img1:999"} if row["candidate_id"] == target["candidate_id"] else row
        for row in planned["fixed_budget_rows"]
    ]
    _rewrite_fixed_budget_rows(planned, rows)

    context_ids = _all_context_ids(planned)
    shard = _shard_for_contexts(planned, "shard0", context_ids)
    with pytest.raises(MergeContractError, match="not reference"):
        _merge(planned, [shard], planned["tmp_path"] / "out")


def test_neighborhood_member_without_id_fails(planned: dict[str, Any]) -> None:
    member = _neighborhood_member_candidate(planned)
    rows = [
        {**row, "candidate_neighborhood_id": None} if row["candidate_id"] == member["candidate_id"] else row
        for row in planned["fixed_budget_rows"]
    ]
    _rewrite_fixed_budget_rows(planned, rows)

    context_ids = _all_context_ids(planned)
    shard = _shard_for_contexts(planned, "shard0", context_ids)
    with pytest.raises(MergeContractError, match="candidate_neighborhood_member.*without.*candidate_neighborhood_id|candidate_neighborhood_id"):
        _merge(planned, [shard], planned["tmp_path"] / "out")


def test_ledger_reference_status_conflicts_with_absent_reference_candidates_fails(planned: dict[str, Any]) -> None:
    # Pick a context that is genuinely bound (has reference candidates), then
    # remove every one of its reference-population rows: the ledger's own
    # frozen "bound:<owner>" status must still demand at least one.
    reference = _reference_candidate(planned)
    context_id = reference["owner_context_id"]
    rows = [row for row in planned["fixed_budget_rows"] if not (row["owner_context_id"] == context_id and row["population"] == "reference")]
    _rewrite_fixed_budget_rows(planned, rows)

    context_ids = _all_context_ids(planned)
    shard = _shard_for_contexts(planned, "shard0", context_ids)
    with pytest.raises(MergeContractError, match="bound to owner"):
        _merge(planned, [shard], planned["tmp_path"] / "out")


def test_ledger_reference_status_conflicts_with_wrong_reference_owner_fails(planned: dict[str, Any]) -> None:
    reference = _reference_candidate(planned)
    rows = [
        {**row, "other_owner_gt_owner_id": "gt:img1:not-the-bound-owner"} if row["candidate_id"] == reference["candidate_id"] else row
        for row in planned["fixed_budget_rows"]
    ]
    _rewrite_fixed_budget_rows(planned, rows)

    context_ids = _all_context_ids(planned)
    shard = _shard_for_contexts(planned, "shard0", context_ids)
    with pytest.raises(MergeContractError, match="bound to owner"):
        _merge(planned, [shard], planned["tmp_path"] / "out")


def test_stale_mechanism_decision_rules_supplied_to_merge_fails(planned: dict[str, Any]) -> None:
    """Passing a byte-different (but schema-valid) file at --mechanism-decision-rules
    than the one the planner receipt bound must fail the exact join, not silently
    substitute it."""

    context_ids = _all_context_ids(planned)
    shard = _shard_for_contexts(planned, "shard0", context_ids)
    tampered_path = planned["tmp_path"] / "tampered-mechanism-decision-rules.json"
    document = dict(planned["mechanism_rules_document"])
    document["geometry"] = {**document["geometry"], "unbound_local_edit": True}
    tampered_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(MergeContractError, match="stale input"):
        merge_shards(
            shards=[shard],
            owner_context_ledger=planned["ledger_path"],
            decision_rules=planned["rules_path"],
            mechanism_decision_rules=tampered_path,
            fixed_budget_candidates=planned["fixed_budget_path"],
            planner_receipt=planned["planner_receipt_path"],
            fn_mechanism_registry=planned["registry_path"],
            output_dir=planned["tmp_path"] / "out",
        )


def test_end_to_end_producer_shaped_merge_carries_reference_and_neighborhood_fields(planned: dict[str, Any]) -> None:
    """One realistic multi-shard merge, asserting every reference/neighborhood
    field survives into the merged output exactly as the producer declared it."""

    context_ids = _all_context_ids(planned)
    shard = _shard_for_contexts(planned, "shard0", context_ids)

    result = _merge(planned, [shard], planned["tmp_path"] / "out")

    merged_rows = [json.loads(line) for line in Path(result["output_artifacts"]["merged_scores"]["path"]).read_text().splitlines()]
    reference_rows = [row for row in merged_rows if row["population"] == "reference"]
    neighborhood_rows = [row for row in merged_rows if row["candidate_neighborhood_member"] is True]
    assert reference_rows
    assert neighborhood_rows
    for row in reference_rows:
        assert row["region"] == "other_owner"
        assert row["other_owner_gt_owner_id"]
        assert row["other_owner_selection_trace"]
        assert row["mechanism_decision_rules_sha256"] == sha256_file(planned["mechanism_rules_path"])
    for row in neighborhood_rows:
        assert row["candidate_neighborhood_id"]
        assert row["population"] == "target"
    source_digests = result["source_digests"]
    assert source_digests["mechanism_decision_rules"]["self_digest"] == planned["mechanism_rules_document"]["self_digest"]
    assert (
        source_digests["mechanism_decision_rules"]["parent_execution_rules_sha256"]
        == sha256_file(planned["rules_path"])
    )


def test_end_to_end_large_box_exact_singleton_carry_through(planned_large_box: dict[str, Any]) -> None:
    """F3 exact-GT-singleton carry-through for genuinely large (100x100+)
    GT boxes, not just the small boxes used elsewhere in this file for
    execution-dedup/scalar_smoke coverage -- proving this merger's
    exact_gt_singleton_member/_id validation is not merely passing by
    accident of small-box geometry."""

    planned = planned_large_box
    singleton_candidates = [row for row in planned["fixed_budget_rows"] if row["exact_gt_singleton_member"] is True]
    assert singleton_candidates
    assert all(row["family_id"] == "near_gt_micro" and row["population"] == "target" for row in singleton_candidates)
    assert all(row["exact_gt_singleton_id"].startswith("exact-gt:") for row in singleton_candidates)

    context_ids = _all_context_ids(planned)
    shard = _shard_for_contexts(planned, "shard0", context_ids)

    result = _merge(planned, [shard], planned["tmp_path"] / "out")

    merged_rows = [json.loads(line) for line in Path(result["output_artifacts"]["merged_scores"]["path"]).read_text().splitlines()]
    merged_singletons = [row for row in merged_rows if row["exact_gt_singleton_member"] is True]
    assert merged_singletons
    assert {row["exact_gt_singleton_id"] for row in merged_singletons} == {row["exact_gt_singleton_id"] for row in singleton_candidates}
    # Exactly one singleton per (owner_context_id, rung) group, per F3.
    groups: dict[tuple[str, str], int] = {}
    for row in merged_rows:
        if row["population"] == "target" and row["family_id"] == "near_gt_micro" and row["exact_gt_singleton_member"]:
            key = (row["owner_context_id"], row["rung"])
            groups[key] = groups.get(key, 0) + 1
    assert groups
    assert set(groups.values()) == {1}


def test_large_box_exact_singleton_member_without_id_fails(planned_large_box: dict[str, Any]) -> None:
    planned = planned_large_box
    singleton = next(row for row in planned["fixed_budget_rows"] if row["exact_gt_singleton_member"] is True)
    rows = [{**row, "exact_gt_singleton_id": None} if row["candidate_id"] == singleton["candidate_id"] else row for row in planned["fixed_budget_rows"]]
    _rewrite_fixed_budget_rows(planned, rows)

    context_ids = _all_context_ids(planned)
    shard = _shard_for_contexts(planned, "shard0", context_ids)
    with pytest.raises(MergeContractError, match="exact_gt_singleton_id"):
        _merge(planned, [shard], planned["tmp_path"] / "out")


def test_large_box_two_exact_singletons_in_one_group_fails(planned_large_box: dict[str, Any]) -> None:
    planned = planned_large_box
    singleton = next(row for row in planned["fixed_budget_rows"] if row["exact_gt_singleton_member"] is True)
    other_member = next(
        row
        for row in planned["fixed_budget_rows"]
        if row["owner_context_id"] == singleton["owner_context_id"]
        and row["rung"] == singleton["rung"]
        and row["family_id"] == "near_gt_micro"
        and row["population"] == "target"
        and row["candidate_id"] != singleton["candidate_id"]
    )
    rows = [
        {**row, "exact_gt_singleton_member": True, "exact_gt_singleton_id": singleton["exact_gt_singleton_id"]}
        if row["candidate_id"] == other_member["candidate_id"]
        else row
        for row in planned["fixed_budget_rows"]
    ]
    _rewrite_fixed_budget_rows(planned, rows)

    context_ids = _all_context_ids(planned)
    shard = _shard_for_contexts(planned, "shard0", context_ids)
    with pytest.raises(MergeContractError, match="exactly one exact_gt_singleton_member"):
        _merge(planned, [shard], planned["tmp_path"] / "out")
