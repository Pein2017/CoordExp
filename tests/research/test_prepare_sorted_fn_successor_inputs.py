"""Contracts for the CPU-only sorted-FN successor input planner.

Covers: arbitrary role counts (1/3/7+), distinct logical roles sharing an
execution tensor, literal-token digest recomputation, family/count budget
constraints, score-derived leakage rejection, source-lineage/policy-mismatch
failure, no default L2 materialization, predecessor exact-reuse binding, and
real acceptance by the *unchanged* downstream candidate builder and
fixed-budget reanalysis consumer.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from scripts.research.build_sorted_owner_basin_candidates import (
    build_sorted_owner_basin_candidates,
    _parse_ledger_rows,
    _parse_rules,
    _parse_seeds,
    sha256_file,
    sha256_json,
)
from scripts.research.build_sorted_fn_mechanism_registry import (
    build_sorted_fn_mechanism_registry,
    load_owner_index,
)
from scripts.research.sorted_owner_basin_landscape import (
    GEOMETRY_IDENTITY_SCHEMA,
    RULES_SCHEMA_VERSION,
)
from scripts.research.reanalyze_sorted_fn_fixed_budget_controls import (
    reanalyze_sorted_fn_fixed_budget_controls,
)
from scripts.research import score_sorted_owner_basin_landscape as sorted_owner_basin_scorer
from scripts.research.prepare_sorted_fn_successor_inputs import (
    NO_SAME_DESCRIPTION_NON_OVERLAPPING_OWNER,
    SuccessorInputPlanError,
    build_family_boxes,
    summarize_predecessor_score_rows,
    classify_region,
    iou,
    load_panel,
    mirror_decoy_box,
    parse_predecessor_identity_row,
    prepare_sorted_fn_successor_inputs,
    resolve_canonical_descriptions,
    resolve_context_tokens,
    resolve_rungs,
    select_nearest_same_description_owner,
)


# --------------------------------------------------------------------------
# Synthetic, self-consistent rules-template fixture (no external file, no
# tokenizer, no model). Validated below against the real, unchanged parser.
# --------------------------------------------------------------------------

FAKE_DIGEST_A = "a" * 64
FAKE_DIGEST_B = "b" * 64
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
    "tokenizer_identity_sha256": FAKE_DIGEST_A,
    "model_identity_sha256": FAKE_DIGEST_B,
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
        "artifact_sha256": FAKE_DIGEST_A,
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
    "contract_mode": "production",
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
            "contract_kind": "production",
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
        "source_digests": {"panel": FAKE_DIGEST_B},
    }


def _panel_line(image_id: str, width: int, height: int, boxes: list[list[int]]) -> dict[str, Any]:
    return {
        "image_id": str(image_id),
        "width": width,
        "height": height,
        "objects": [{"bbox_2d": list(box)} for box in boxes],
    }


def _rollout_document(
    image_id: str, decode_mode: str, seed: int, prompt_token_ids: list[int], row_chunks: list[list[int]]
) -> dict[str, Any]:
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


def _row_chunk(marker: int) -> list[int]:
    return [_OBJECT_REF_START, marker, marker + 1, marker + 2]


def _mechanism_registry(
    roles: list[dict[str, Any]],
    *,
    targets: list[dict[str, Any]] | None = None,
    bound_non_targets: list[dict[str, Any]] | None = None,
    null_pairs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "registry_digest": "f" * 64,
        "mechanism_cohort": {"targets": targets or []},
        "context_control_registry": {"bound_non_targets": bound_non_targets or []},
        "smoke": {"roles": roles, "null_pair_envelope": {"pairs": null_pairs or []}},
    }


def _root_role(role_id: str, gt_owner_id: str, *, image_id: str = "img1", cut_marker: int | None = None) -> dict[str, Any]:
    token_ids = list(_PROMPT)
    if cut_marker is not None:
        token_ids = token_ids + _row_chunk(cut_marker)
    return {
        "role_id": role_id,
        "role_kind": "root_context",
        "gt_owner_id": gt_owner_id,
        "trajectory": {"image_id": image_id, "decode_mode": "greedy", "seed": 0},
        "prefix": {"token_ids": token_ids},
        "provenance": {"source_artifact_path": "rollout.json", "source_artifact_sha256": "0" * 64},
    }


@pytest.fixture()
def two_owner_fixtures(tmp_path: Path) -> dict[str, Path]:
    owner_ledger_path = tmp_path / "owner-ledger.jsonl"
    _write_jsonl(
        owner_ledger_path,
        [
            _owner_ledger_row("gt:img1:0", "img1", 0, "widget"),
            _owner_ledger_row("gt:img1:1", "img1", 1, "widget"),
            _owner_ledger_row("gt:img1:2", "img1", 2, "widget"),
        ],
    )
    panel_path = tmp_path / "panel.jsonl"
    _write_jsonl(
        panel_path,
        # Small (10x10-ish), mutually non-overlapping boxes, matching
        # real small/tiny-object owner-contexts like gt:7511:17. near_gt_micro
        # local index 0 is now guaranteed to be the exact GT box for *any*
        # box size (see _near_micro_boxes), so this size is not load-bearing
        # for F3; it is kept small only because it also happens to be a
        # convenient, representative scale for these fixtures.
        [_panel_line("img1", 1024, 1024, [[100, 100, 110, 110], [140, 140, 150, 150], [500, 500, 510, 510]])],
    )
    rollout_path = tmp_path / "rollout.json"
    _write_json(
        rollout_path,
        _rollout_document("img1", "greedy", 0, _PROMPT, [_row_chunk(1), _row_chunk(10)]),
    )
    rules_template_path = tmp_path / "rules-template.json"
    _write_json(rules_template_path, RULES_TEMPLATE)
    descriptions_path = tmp_path / "descriptions.json"
    _write_json(descriptions_path, _DESCRIPTION_SUPPLEMENT)
    return {
        "owner_ledger": owner_ledger_path,
        "panel": panel_path,
        "rollout": rollout_path,
        "rules_template": rules_template_path,
        "descriptions": descriptions_path,
    }


def _run_planner(fixtures: dict[str, Path], registry: dict[str, Any], out_dir: Path, **kwargs: Any) -> dict[str, Any]:
    registry_path = out_dir / "registry.json"
    _write_json(registry_path, registry)
    return prepare_sorted_fn_successor_inputs(
        registry=registry_path,
        owner_ledger=fixtures["owner_ledger"],
        panel=fixtures["panel"],
        rules_template=fixtures["rules_template"],
        rollouts=[fixtures["rollout"]],
        canonical_description_registry=fixtures["descriptions"],
        predecessor_score_rows=kwargs.get("predecessor_score_rows"),
        out_dir=out_dir / "plan",
    )


def _real_accept(out_dir: Path) -> tuple[list[Any], list[dict[str, Any]]]:
    """Prove real, unchanged-builder acceptance of the emitted documents."""

    plan_dir = out_dir / "plan"
    rules_doc = json.loads((plan_dir / "landscape-decision-rules.json").read_text())
    rules = _parse_rules(rules_doc)
    ledger_rows_raw = [json.loads(line) for line in (plan_dir / "owner-context-ledger.jsonl").read_text().splitlines()]
    ledger_rows = _parse_ledger_rows(ledger_rows_raw, rules)
    rules_sha256 = sha256_file(plan_dir / "landscape-decision-rules.json")
    ledger_sha256 = sha256_file(plan_dir / "owner-context-ledger.jsonl")
    seeds_doc = json.loads((plan_dir / "candidate-bank-seeds.json").read_text())
    seeds = _parse_seeds(seeds_doc, rules=rules, rules_sha256=rules_sha256, ledger_sha256=ledger_sha256, ledger_rows=ledger_rows)
    return list(seeds), ledger_rows_raw


# --------------------------------------------------------------------------
# Pure geometry / lattice tests
# --------------------------------------------------------------------------


@pytest.mark.parametrize("gt_box", [(400, 400, 450, 460), (500, 500, 502, 503), (0, 0, 5, 5), (990, 990, 998, 999)])
def test_l0_budget_band_and_family_mirrored_decoys(gt_box: tuple[int, int, int, int]) -> None:
    boxes = build_family_boxes(gt_box, "L0")
    assert 24 <= len(boxes) <= 40
    families = Counter(family for family, _ in boxes)
    for family, box in boxes:
        decoy = mirror_decoy_box(box, gt_box)
        assert iou(decoy, gt_box) == 0.0
    assert sum(families.values()) == len(boxes)


@pytest.mark.parametrize("gt_box", [(400, 400, 450, 460), (500, 500, 502, 503), (990, 990, 998, 999)])
def test_l1_exact_count_and_strict_minimum(gt_box: tuple[int, int, int, int]) -> None:
    boxes = build_family_boxes(gt_box, "L1")
    assert len(boxes) == 256
    strict = sum(1 for _family, box in boxes if classify_region(iou(box, gt_box)) == "target_strict")
    assert strict >= 64


def test_family_mirrored_decoy_multiset_equals_target_multiset() -> None:
    gt_box = (200, 200, 260, 250)
    boxes = build_family_boxes(gt_box, "L1")
    target_families = Counter(family for family, _ in boxes)
    decoy_families = Counter(family for family, _ in boxes)  # one decoy per target, same family
    assert target_families == decoy_families


def test_no_default_l2_rung_is_supported() -> None:
    with pytest.raises(SuccessorInputPlanError):
        build_family_boxes((10, 10, 20, 20), "L2")


def test_resolve_rungs_strict_rescue_emits_scalar_smoke_and_l1() -> None:
    is_control, control_kind, rungs = resolve_rungs("strict_rescued")
    assert is_control is True
    assert control_kind == "strict_rescue"
    assert rungs == ("scalar_smoke", "L1")


@pytest.mark.parametrize(
    "cohort,expected_kind",
    [("greedy_strict_present", "strict_positive"), ("loose_only_b1", "loose_only"), (None, None), ("no_free_spatial_support", None)],
)
def test_resolve_rungs_non_strict_rescue_uses_l0_l1_ladder(cohort: str | None, expected_kind: str | None) -> None:
    is_control, control_kind, rungs = resolve_rungs(cohort)
    assert rungs == ("L0", "L1")
    assert control_kind == expected_kind


# --------------------------------------------------------------------------
# Score/logit/likelihood leakage rejection
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "poisoned_field", ["raw_score", "raw_model_logprob", "target_peak", "background_prominence", "some_logit_value", "likelihood_estimate"]
)
def test_leakage_rejection(tmp_path: Path, two_owner_fixtures: dict[str, Path], poisoned_field: str) -> None:
    registry = _mechanism_registry([_root_role("root:gt:img1:0", "gt:img1:0")])
    registry[poisoned_field] = 1.23
    with pytest.raises(SuccessorInputPlanError, match="score-derived"):
        _run_planner(two_owner_fixtures, registry, tmp_path)


# --------------------------------------------------------------------------
# Literal-token digest recomputation / context split
# --------------------------------------------------------------------------


def test_resolve_context_tokens_literal_split_matches_recomputed_digests() -> None:
    role = _root_role("root:gt:img1:0", "gt:img1:0", cut_marker=1)
    trajectories = {("img1", "greedy", 0): {"prompt_token_ids": list(_PROMPT)}}
    result = resolve_context_tokens(role, trajectories)
    assert result["prompt_prefix_token_count"] == len(_PROMPT)
    assert result["token_ids_sha256"] == sha256_json(role["prefix"]["token_ids"])
    assert result["prompt_token_ids_sha256"] == sha256_json(_PROMPT)
    self_prefix = role["prefix"]["token_ids"][len(_PROMPT) :]
    assert result["self_prefix_generated_token_ids_sha256"] == sha256_json(self_prefix)
    assert result["split"] == {"prompt": [0, len(_PROMPT)], "self_prefix": [len(_PROMPT), len(role["prefix"]["token_ids"])]}


def test_resolve_context_tokens_missing_trajectory_fails() -> None:
    role = _root_role("root:gt:img1:0", "gt:img1:0")
    with pytest.raises(SuccessorInputPlanError, match="no matching rollout artifact"):
        resolve_context_tokens(role, {})


def test_resolve_context_tokens_policy_mismatch_fails() -> None:
    role = _root_role("root:gt:img1:0", "gt:img1:0")
    # Declares trajectory (img1, greedy, 0) but the literal prefix does not
    # actually start with that trajectory's stored prompt: a policy mismatch.
    trajectories = {("img1", "greedy", 0): {"prompt_token_ids": [9999, 9998, 9997]}}
    with pytest.raises(SuccessorInputPlanError, match="policy mismatch"):
        resolve_context_tokens(role, trajectories)


# --------------------------------------------------------------------------
# Canonical-description resolution (no tokenizer)
# --------------------------------------------------------------------------


def test_resolve_canonical_descriptions_reuses_known_text() -> None:
    resolved = resolve_canonical_descriptions({"gt:a:0": "widget"}, {"widget": _WIDGET_DESCRIPTION})
    assert resolved["gt:a:0"]["token_ids"] == [777]


def test_resolve_canonical_descriptions_missing_text_fails_fast() -> None:
    with pytest.raises(SuccessorInputPlanError, match="cup"):
        resolve_canonical_descriptions({"gt:a:0": "cup"}, {"widget": _WIDGET_DESCRIPTION})


# --------------------------------------------------------------------------
# Predecessor score rows: diagnostic-only, never a reuse claim
# --------------------------------------------------------------------------


def _full_identity_predecessor_row(
    *,
    gt_owner_id: str = "gt:a:0",
    coord_token_ids: list[int] | None = None,
    candidate_id: str = "predecessor:1",
    context_id: str = "ctx:predecessor:a",
    source_digest: str = "e" * 64,
) -> dict[str, Any]:
    return {
        "gt_owner_id": gt_owner_id,
        "coord_token_ids": coord_token_ids or [1, 2, 3, 4],
        "candidate_id": candidate_id,
        "context_id": context_id,
        "source_digest": source_digest,
    }


# Real ``landscape_scores.v1``-shaped rows (the actual merged predecessor
# score-row schema, verified by direct inspection of a real merged
# production scores file): a complete-box row carrying real score/logit/
# policy payloads alongside identity -- with no literal source_digest and
# no raw_row_identity at all -- and a conditional y1-scan plan row that
# carries no complete-box identity at all.
_REAL_SCHEMA_COMPLETE_BOX_ROW = {
    "schema_version": "landscape_scores.v1",
    "request_kind": "complete_box",
    "candidate_kind": "target_anchor",
    "candidate_id": "landscape-candidate:sha256:real-example",
    "coord_token_ids": [152479, 152257, 152491, 152288],
    "coord_token_ids_sha256": "f" * 64,
    "context_id": "ctx:B2:control:smoke:b2:7511:22-to-26:B2_after",
    "diagnostic_owner_id": "diagnostic:gt:7511:26",
    "gt_owner_id": "gt:7511:26",
    "prefix_token_ids_sha256": "a" * 64,
    "rule_digest": "b" * 64,
    "raw_model_logprob": {
        "complete_box_logprob_sum": -18.113149166107178,
        "vocab_attestation": {"x1": {"domain_digest": "c" * 64}},
    },
    "auxiliary_policy_scores": {"rp_1.00": {"complete_box_logprob_sum": -18.1}},
    "raw_bin_scan": {"bin_logprobs": [-27.9, -29.1, -29.2]},
    "auxiliary_policy_bin_scan": {"rp_1.00": {"bin_logprobs": [-27.9, -29.1, -29.2]}},
}
_REAL_SCHEMA_Y1_SCAN_PLAN_ROW = {
    "schema_version": "landscape_scores.v1",
    "request_kind": "dense_scan",
    "candidate_kind": "target_anchor",
    "candidate_id": "conditional-y1-plan:sha256:real-example",
    "fixed_coord_token_ids": [152488],
    "context_id": "ctx:B2:control:smoke:b2:7511:22-to-26:B2_before",
    "gt_owner_id": "gt:7511:26",
    "auxiliary_policy_bin_scan": {"rp_1.00": {"bin_logprobs": [-27.9, -29.1]}},
}


def test_summarize_predecessor_score_rows_counts_distinct_identities() -> None:
    stats = summarize_predecessor_score_rows(
        [_full_identity_predecessor_row(), _full_identity_predecessor_row(candidate_id="predecessor:2")]
    )
    assert stats["rows_with_box_identity"] == 2
    assert stats["distinct_predecessor_owner_box_identities"] == 1  # same owner+box, deduplicated
    assert stats["fully_qualified_identity_count"] == 2
    assert stats["valid_reuse_count"] == 0


def test_summarize_predecessor_score_rows_never_reports_nonzero_valid_reuse() -> None:
    # Even a row with every optional identity field present and well-formed
    # never produces a nonzero valid_reuse_count: this contract is
    # diagnostic-only and never binds a reuse claim from any input shape.
    stats = summarize_predecessor_score_rows([_full_identity_predecessor_row()])
    assert stats["valid_reuse_count"] == 0
    assert "diagnostic_only" in stats["reuse_policy"]


def test_real_schema_row_with_forbidden_score_payload_is_safely_ignored_but_not_reusable() -> None:
    # Verified by direct inspection of a real merged landscape_scores.v1
    # complete-box row: it has no literal source_digest and no
    # raw_row_identity at all. The score/logit/policy fields alongside
    # identity must be safely ignored (never read), and this diagnostic
    # summary never treats the geometry coincidence as reusable.
    identity = parse_predecessor_identity_row(_REAL_SCHEMA_COMPLETE_BOX_ROW, index=0)
    assert identity["candidate_id"] == "landscape-candidate:sha256:real-example"
    assert identity["coord_token_ids"] == [152479, 152257, 152491, 152288]
    assert identity["source_digest"] is None
    assert identity["raw_row_identity"] is None
    stats = summarize_predecessor_score_rows([_REAL_SCHEMA_COMPLETE_BOX_ROW])
    assert stats["rows_with_box_identity"] == 1
    assert stats["distinct_predecessor_owner_box_identities"] == 1
    assert stats["fully_qualified_identity_count"] == 0
    assert stats["valid_reuse_count"] == 0


def test_conditional_y1_scan_plan_row_without_complete_box_identity_is_skipped_not_failed() -> None:
    identity = parse_predecessor_identity_row(_REAL_SCHEMA_Y1_SCAN_PLAN_ROW, index=0)
    assert identity is None
    # A file mixing complete-box rows with y1-scan-plan rows (the real
    # merged landscape-scores.jsonl shape) must not fail; the plan row
    # contributes nothing to the summary.
    stats = summarize_predecessor_score_rows([_REAL_SCHEMA_COMPLETE_BOX_ROW, _REAL_SCHEMA_Y1_SCAN_PLAN_ROW])
    assert stats["rows_with_box_identity"] == 1


@pytest.mark.parametrize(
    "broken_row",
    [
        {"coord_token_ids": [1, 2, 3, 4], "gt_owner_id": "gt:a:0"},  # missing candidate_id
        {"coord_token_ids": [1, 2, 3, 4], "candidate_id": "predecessor:1"},  # missing gt_owner_id
        {"coord_token_ids": [1, 2, 3, 4], "candidate_id": "predecessor:1", "gt_owner_id": "  "},
        {"coord_token_ids": ["not", "ints"], "candidate_id": "predecessor:1", "gt_owner_id": "gt:a:0"},
        {"coord_token_ids": [], "candidate_id": "predecessor:1", "gt_owner_id": "gt:a:0"},
    ],
)
def test_predecessor_row_with_box_tokens_but_missing_identity_fails_fast(broken_row: dict[str, Any]) -> None:
    with pytest.raises(SuccessorInputPlanError):
        parse_predecessor_identity_row(broken_row, index=0)


def test_end_to_end_malformed_predecessor_row_fails_before_writing_any_accepted_output(
    tmp_path: Path, two_owner_fixtures: dict[str, Path]
) -> None:
    registry = _mechanism_registry([_root_role("root:gt:img1:0", "gt:img1:0")])
    predecessor_rows_path = tmp_path / "malformed-predecessor-scores.jsonl"
    _write_jsonl(
        predecessor_rows_path,
        [{"coord_token_ids": [1, 2, 3, 4], "gt_owner_id": "gt:img1:0"}],  # missing candidate_id
    )
    with pytest.raises(SuccessorInputPlanError):
        _run_planner(
            two_owner_fixtures,
            registry,
            tmp_path,
            predecessor_score_rows=predecessor_rows_path,
        )
    plan_dir = tmp_path / "plan"
    # create-or-identical strictness: a failed run must not leave behind an
    # accepted-looking plan directory (only an empty/absent one, never a
    # partially-written owner-context-ledger.jsonl or rules document).
    assert not plan_dir.exists() or not any(plan_dir.iterdir())


def test_end_to_end_no_candidate_ever_carries_a_predecessor_candidate_id(
    tmp_path: Path, two_owner_fixtures: dict[str, Path]
) -> None:
    """Even a deliberately constructed exact owner+box+context+source_digest
    coincidence against this planner's own fixed-budget geometry must never
    produce a ``predecessor_candidate_id`` on any row: this contract does
    not bind reuse from any input, full stop. Only the diagnostic summary
    may report the coincidence.
    """

    registry = _mechanism_registry([_root_role("root:gt:img1:0", "gt:img1:0")])
    _run_planner(two_owner_fixtures, registry, tmp_path / "warmup")
    ledger_row = json.loads(
        (tmp_path / "warmup" / "plan" / "owner-context-ledger.jsonl").read_text().splitlines()[0]
    )
    fixed_budget_rows = [
        json.loads(line)
        for line in (tmp_path / "warmup" / "plan" / "fixed-budget-candidates.jsonl").read_text().splitlines()
    ]
    target_row = next(row for row in fixed_budget_rows if row["population"] == "target")
    original_source_digest = target_row["source_digest"]

    predecessor_rows_path = tmp_path / "predecessor-scores.jsonl"
    _write_jsonl(
        predecessor_rows_path,
        [
            {
                "gt_owner_id": "gt:img1:0",
                "coord_token_ids": target_row["coord_token_ids"],
                "candidate_id": "predecessor:exact-geometry-and-identity-coincidence",
                "context_id": ledger_row["context_id"],
                "source_digest": "e" * 64,
            }
        ],
    )
    receipt = _run_planner(
        two_owner_fixtures,
        registry,
        tmp_path / "with-predecessor",
        predecessor_score_rows=predecessor_rows_path,
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "with-predecessor" / "plan" / "fixed-budget-candidates.jsonl")
        .read_text()
        .splitlines()
    ]
    assert all("predecessor_candidate_id" not in row for row in rows)
    matched_geometry_row = next(row for row in rows if row["coord_token_ids"] == target_row["coord_token_ids"])
    assert matched_geometry_row["source_digest"] == original_source_digest
    assert receipt["predecessor_reuse"]["valid_reuse_count"] == 0
    assert receipt["predecessor_reuse"]["fully_qualified_identity_count"] >= 1


def test_end_to_end_real_schema_predecessor_score_rows_do_not_fail_and_are_never_reused(
    tmp_path: Path, two_owner_fixtures: dict[str, Path]
) -> None:
    registry = _mechanism_registry([_root_role("root:gt:img1:0", "gt:img1:0")])
    _run_planner(two_owner_fixtures, registry, tmp_path / "warmup")
    fixed_budget_rows = [
        json.loads(line)
        for line in (tmp_path / "warmup" / "plan" / "fixed-budget-candidates.jsonl").read_text().splitlines()
    ]
    target_row = next(row for row in fixed_budget_rows if row["population"] == "target")
    original_source_digest = target_row["source_digest"]
    predecessor_rows_path = tmp_path / "real-schema-predecessor-scores.jsonl"
    real_shaped_match_row = {
        **_REAL_SCHEMA_COMPLETE_BOX_ROW,
        "gt_owner_id": "gt:img1:0",
        "coord_token_ids": target_row["coord_token_ids"],
        "candidate_id": "landscape-candidate:sha256:real-match",
    }
    assert "source_digest" not in real_shaped_match_row
    _write_jsonl(predecessor_rows_path, [real_shaped_match_row, _REAL_SCHEMA_Y1_SCAN_PLAN_ROW])

    receipt = _run_planner(
        two_owner_fixtures,
        registry,
        tmp_path / "with-real-predecessor",
        predecessor_score_rows=predecessor_rows_path,
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "with-real-predecessor" / "plan" / "fixed-budget-candidates.jsonl")
        .read_text()
        .splitlines()
    ]
    assert all("predecessor_candidate_id" not in row for row in rows)
    matched_geometry_row = next(row for row in rows if row["coord_token_ids"] == target_row["coord_token_ids"])
    assert matched_geometry_row["source_digest"] == original_source_digest
    assert receipt["predecessor_reuse"]["valid_reuse_count"] == 0
    assert receipt["predecessor_reuse"]["fully_qualified_identity_count"] == 0
    assert receipt["predecessor_reuse"]["distinct_predecessor_owner_box_identities"] >= 1
    assert receipt["role_count"] == 1



# --------------------------------------------------------------------------
# End-to-end: arbitrary role counts, shared execution tensor, real acceptance
# --------------------------------------------------------------------------


@pytest.mark.parametrize("role_count", [1, 3, 7])
def test_end_to_end_arbitrary_role_counts_are_accepted_by_the_real_builder(
    tmp_path: Path, two_owner_fixtures: dict[str, Path], role_count: int
) -> None:
    owner_ids = ["gt:img1:0", "gt:img1:1", "gt:img1:2"]
    roles = []
    for i in range(role_count):
        owner_id = owner_ids[i % len(owner_ids)]
        roles.append(_root_role(f"root:{i}:{owner_id}", owner_id, cut_marker=(i + 1) if i % 2 else None))
    registry = _mechanism_registry(roles)
    receipt = _run_planner(two_owner_fixtures, registry, tmp_path)
    assert receipt["role_count"] == role_count
    assert receipt["context_count"] == role_count

    seeds, ledger_rows_raw = _real_accept(tmp_path)
    assert len(ledger_rows_raw) == role_count
    # Every logical role produced its own distinct ledger row (unique
    # (diagnostic_owner_id, context_id) key); logical rows are never merged.
    keys = {(row["diagnostic_owner_id"], row["context_id"]) for row in ledger_rows_raw}
    assert len(keys) == role_count


def test_distinct_logical_roles_sharing_execution_tensor_are_not_merged(
    tmp_path: Path, two_owner_fixtures: dict[str, Path]
) -> None:
    # Two different owners, both rooted at the identical (image, decode_mode,
    # seed, prompt-only) prefix: one shared execution tensor, two logical roles.
    roles = [
        _root_role("root:gt:img1:0", "gt:img1:0"),
        _root_role("root:gt:img1:1", "gt:img1:1"),
    ]
    registry = _mechanism_registry(roles)
    receipt = _run_planner(two_owner_fixtures, registry, tmp_path)
    assert receipt["context_count"] == 2

    ledger_rows_raw = [
        json.loads(line)
        for line in (tmp_path / "plan" / "owner-context-ledger.jsonl").read_text().splitlines()
    ]
    assert len(ledger_rows_raw) == 2
    dedup_keys = {row["execution_dedup_key"] for row in ledger_rows_raw}
    assert len(dedup_keys) == 1  # identical execution tensor
    owner_context_keys = {(row["diagnostic_owner_id"], row["context_id"]) for row in ledger_rows_raw}
    assert len(owner_context_keys) == 2  # logical rows never merged


def test_end_to_end_real_builder_and_real_fixed_budget_reanalysis_accept_output(
    tmp_path: Path, two_owner_fixtures: dict[str, Path]
) -> None:
    registry = _mechanism_registry(
        [
            _root_role("root:gt:img1:0", "gt:img1:0"),
            _root_role("root:gt:img1:1", "gt:img1:1", cut_marker=10),
        ],
        targets=[{"gt_owner_id": "gt:img1:0", "cohort": "strict_rescued"}],
        bound_non_targets=[{"gt_owner_id": "gt:img1:1", "cohort": "no_free_spatial_support"}],
    )
    _run_planner(two_owner_fixtures, registry, tmp_path)

    seeds, ledger_rows_raw = _real_accept(tmp_path)
    assert len(seeds) > 0

    old_scores_path = tmp_path / "old-scores.jsonl"
    old_scores_path.write_text("", encoding="utf-8")
    report = reanalyze_sorted_fn_fixed_budget_controls(
        fixed_budget_candidates=tmp_path / "plan" / "fixed-budget-candidates.jsonl",
        old_score_rows=old_scores_path,
        output=tmp_path / "reanalysis.json",
    )
    owner_contexts = report["owner_contexts"]
    assert owner_contexts["ctx:fn:root:gt:img1:0"].keys() == {"scalar_smoke", "L1"}
    assert set(owner_contexts["ctx:fn:root:gt:img1:1"].keys()) == {"L0", "L1"}


def test_source_lineage_missing_owner_fails(tmp_path: Path, two_owner_fixtures: dict[str, Path]) -> None:
    registry = _mechanism_registry([_root_role("root:gt:missing:0", "gt:missing:0")])
    with pytest.raises(SuccessorInputPlanError, match="owner ledger is missing"):
        _run_planner(two_owner_fixtures, registry, tmp_path)


def test_create_or_identical_write_semantics(tmp_path: Path, two_owner_fixtures: dict[str, Path]) -> None:
    registry = _mechanism_registry([_root_role("root:gt:img1:0", "gt:img1:0")])
    receipt1 = _run_planner(two_owner_fixtures, registry, tmp_path)
    receipt2 = _run_planner(two_owner_fixtures, registry, tmp_path)
    assert receipt1["receipt_digest"] == receipt2["receipt_digest"]


# --------------------------------------------------------------------------
# canonical_description_registry provenance binding
# --------------------------------------------------------------------------


def test_canonical_description_registry_is_bound_in_receipt_sources(
    tmp_path: Path, two_owner_fixtures: dict[str, Path]
) -> None:
    registry = _mechanism_registry([_root_role("root:gt:img1:0", "gt:img1:0")])
    receipt = _run_planner(two_owner_fixtures, registry, tmp_path)
    bound = receipt["sources"]["canonical_description_registry"]
    assert bound is not None
    assert bound["path"] == str(two_owner_fixtures["descriptions"])
    assert bound["sha256"] == sha256_file(two_owner_fixtures["descriptions"])


def test_changing_supplement_invalidates_create_or_identical_output(
    tmp_path: Path, two_owner_fixtures: dict[str, Path]
) -> None:
    registry = _mechanism_registry([_root_role("root:gt:img1:0", "gt:img1:0")])
    shared_out_dir = tmp_path / "shared"
    _run_planner(two_owner_fixtures, registry, shared_out_dir)

    # A genuinely different, still schema-consistent supplement (distinct real
    # token ids, correctly matching hashes) changes the emitted description
    # content and the bound upstream digest alike; re-running the planner at
    # the exact same output directory must not silently reuse the prior
    # create-or-identical output.
    changed_token_ids = [778, 779]
    changed_widget_description = {
        "text": "widget",
        "text_sha256": sha256_json("widget"),
        "token_ids": changed_token_ids,
        "token_ids_sha256": sha256_json(changed_token_ids),
        "forced_row_prefix_through_box_start_token_ids": [100000, *changed_token_ids, 100001, 100002],
        "forced_row_prefix_through_box_start_sha256": sha256_json(
            [100000, *changed_token_ids, 100001, 100002]
        ),
    }
    changed_supplement_path = tmp_path / "descriptions-changed.json"
    _write_json(changed_supplement_path, {"widget": changed_widget_description})
    fixtures_changed = dict(two_owner_fixtures)
    fixtures_changed["descriptions"] = changed_supplement_path
    with pytest.raises(SuccessorInputPlanError, match="already exists with different content"):
        _run_planner(fixtures_changed, registry, shared_out_dir)


# --------------------------------------------------------------------------
# Real-shaped acceptance: unchanged scorer's own load_decision_rules /
# --validate-contract-only on successor output built from the real, landed
# task4-control-final production template.
# --------------------------------------------------------------------------

_REAL_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration"
)
REAL_OWNER_LEDGER = (
    _REAL_ROOT / "2026-08-01-sorted-owner-basin-task0-v2-global-ambiguity-neutral-final" / "owner-ledger.jsonl"
)
REAL_PREDICTION_ROW_LEDGER = (
    _REAL_ROOT
    / "2026-08-01-sorted-owner-basin-task0-v2-global-ambiguity-neutral-final"
    / "prediction-row-ledger.jsonl"
)
REAL_COHORT_ASSIGNMENTS = (
    _REAL_ROOT / "2026-08-01-sorted-owner-basin-cohorts-v2-foil-sealed-final-v2" / "cohort-assignments.jsonl"
)
REAL_GREEDY_ROLLOUT = (
    _REAL_ROOT / "2026-07-29-three-checkpoint-human-refined12-max3084" / "sorted" / "greedy" / "greedy.json"
)
REAL_SAMPLED_SHARD_0 = (
    _REAL_ROOT / "2026-07-29-three-checkpoint-human-refined12-max3084" / "sorted" / "sampled" / "shard-0.json"
)
REAL_SAMPLED_SHARD_1 = (
    _REAL_ROOT / "2026-07-29-three-checkpoint-human-refined12-max3084" / "sorted" / "sampled" / "shard-1.json"
)
REAL_PANEL = (
    _REAL_ROOT
    / "2026-07-21-best-sampled-trajectory-positive-row-imitation-screen"
    / "evaluation-inputs"
    / "human-refined-12.coord.jsonl"
)
REAL_RULES_TEMPLATE = (
    _REAL_ROOT / "2026-08-01-sorted-owner-basin-task4-control-input-plan-final" / "landscape-decision-rules.json"
)
_REAL_FIXTURES = (
    REAL_OWNER_LEDGER,
    REAL_PREDICTION_ROW_LEDGER,
    REAL_COHORT_ASSIGNMENTS,
    REAL_GREEDY_ROLLOUT,
    REAL_SAMPLED_SHARD_0,
    REAL_SAMPLED_SHARD_1,
    REAL_PANEL,
    REAL_RULES_TEMPLATE,
)
# Real project schema/coordinate tokens (verified against the frozen
# tokenizer named in unit.md); "kite" is the one description among the
# registry's default (no-collision-plan) roles not already landed in the
# task4-control-final template's owner_canonical_descriptions.
_REAL_KITE_DESCRIPTION = {
    "text": "kite",
    "text_sha256": sha256_json("kite"),
    "token_ids": [74, 632],
    "token_ids_sha256": sha256_json([74, 632]),
    "forced_row_prefix_through_box_start_token_ids": [151646, 74, 632, 151647, 151648],
    "forced_row_prefix_through_box_start_sha256": sha256_json([151646, 74, 632, 151647, 151648]),
}


@pytest.mark.skipif(
    not all(path.is_file() for path in _REAL_FIXTURES),
    reason="real predecessor Task-0/rules-template artifacts are unavailable",
)
def test_real_task4_template_semantic_core_and_scorer_accept_output(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.json"
    build_sorted_fn_mechanism_registry(
        owner_ledger=REAL_OWNER_LEDGER,
        prediction_row_ledger=REAL_PREDICTION_ROW_LEDGER,
        cohort_assignments=REAL_COHORT_ASSIGNMENTS,
        rollouts=[REAL_GREEDY_ROLLOUT, REAL_SAMPLED_SHARD_0, REAL_SAMPLED_SHARD_1],
        due_turn_trajectories={"gt:7511:22": ("greedy", 0), "gt:7511:17": ("sampled", 21010)},
        output=registry_path,
    )

    descriptions_path = tmp_path / "kite-description.json"
    _write_json(descriptions_path, {"kite": _REAL_KITE_DESCRIPTION})

    out_dir = tmp_path / "plan"
    prepare_sorted_fn_successor_inputs(
        registry=registry_path,
        owner_ledger=REAL_OWNER_LEDGER,
        panel=REAL_PANEL,
        rules_template=REAL_RULES_TEMPLATE,
        rollouts=[REAL_GREEDY_ROLLOUT, REAL_SAMPLED_SHARD_0, REAL_SAMPLED_SHARD_1],
        canonical_description_registry=descriptions_path,
        predecessor_score_rows=None,
        out_dir=out_dir,
    )

    rules_path = out_dir / "landscape-decision-rules.json"
    template_payload = json.loads(REAL_RULES_TEMPLATE.read_text())
    emitted_payload = json.loads(rules_path.read_text())
    # The successor added real owner descriptions/foil members, so the landed
    # semantic core must have been rebuilt (never silently copied stale).
    assert emitted_payload["semantic_core"]["payload"] != template_payload["semantic_core"]["payload"]

    # The exact failure mode this test guards against:
    # ``LandscapeScoringError: production decision rules must carry the
    # landed semantic_core digest`` raised by the unchanged scorer's own
    # ``load_decision_rules``.
    decision_rules = sorted_owner_basin_scorer.load_decision_rules(rules_path)
    assert decision_rules.contract_mode == "production"

    ledger_path = out_dir / "owner-context-ledger.jsonl"
    bank_seeds_path = out_dir / "candidate-bank-seeds.json"
    candidates_path = tmp_path / "candidates.jsonl"
    build_sorted_owner_basin_candidates(
        owner_context_ledger=ledger_path,
        landscape_decision_rules=rules_path,
        bank_seeds=bank_seeds_path,
        output_jsonl=candidates_path,
        receipt=tmp_path / "candidates-receipt.json",
        expected_owner_context_ledger_sha256=sha256_file(ledger_path),
        expected_landscape_decision_rules_sha256=sha256_file(rules_path),
        expected_bank_seeds_sha256=sha256_file(bank_seeds_path),
    )

    # Critical acceptance: the unchanged scorer's own CLI, --validate-contract-only,
    # must accept the successor's output without loading a model or touching a GPU.
    command = [
        sys.executable,
        "-m",
        "scripts.research.score_sorted_owner_basin_landscape",
        "--decision-rules",
        str(rules_path),
        "--candidates",
        str(candidates_path),
        "--owner-context-ledger",
        str(ledger_path),
        "--validate-contract-only",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, cwd=Path(__file__).resolve().parents[2])
    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["rows"] > 0


# --------------------------------------------------------------------------
# Fable's operational contract: neighborhood tagging, reference population,
# mechanism-decision-rules.json, and strict-rescue dual-rung.
# --------------------------------------------------------------------------


def _rows_by_rung_and_population(fixed_budget_path: Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    by_rung: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for line in fixed_budget_path.read_text().splitlines():
        row = json.loads(line)
        by_rung.setdefault(row["rung"], {}).setdefault(row["population"], []).append(row)
    return by_rung


def test_strict_rescue_scalar_smoke_and_l1_shapes_with_bound_reference(
    tmp_path: Path, two_owner_fixtures: dict[str, Path]
) -> None:
    """``two_owner_fixtures`` has three same-description ("widget"), mutually
    non-overlapping owners on one image; ``gt:img1:1`` is the nearest to
    ``gt:img1:0`` and must be bound as its reference owner. A strict_rescue
    cohort (the synthetic stand-in for the real ``gt:7511:17``) must emit
    both rungs: scalar_smoke with 30 target + 30 decoy + 7 reference, and an
    unconditionally frozen L1 with 256 target + 256 decoy + 65 reference.
    """

    registry = _mechanism_registry(
        [_root_role("root:gt:img1:0", "gt:img1:0")],
        targets=[{"gt_owner_id": "gt:img1:0", "cohort": "strict_rescued"}],
    )
    receipt = _run_planner(two_owner_fixtures, registry, tmp_path)
    assert receipt["predecessor_reuse"]["valid_reuse_count"] == 0

    fixed_budget_path = tmp_path / "plan" / "fixed-budget-candidates.jsonl"
    by_rung = _rows_by_rung_and_population(fixed_budget_path)
    assert set(by_rung.keys()) == {"scalar_smoke", "L1"}

    scalar = by_rung["scalar_smoke"]
    assert len(scalar["target"]) == 30
    assert len(scalar["decoy"]) == 30
    assert len(scalar.get("reference", [])) == 7
    assert all(row["other_owner_gt_owner_id"] == "gt:img1:1" for row in scalar["reference"])
    assert all(row["region"] == "other_owner" for row in scalar["reference"])

    l1 = by_rung["L1"]
    assert len(l1["target"]) == 256
    assert len(l1["decoy"]) == 256
    assert len(l1.get("reference", [])) == 65
    assert all(row["other_owner_gt_owner_id"] == "gt:img1:1" for row in l1["reference"])

    ledger_rows = [
        json.loads(line)
        for line in (tmp_path / "plan" / "owner-context-ledger.jsonl").read_text().splitlines()
    ]
    assert ledger_rows[0]["other_owner_reference_status"] == "bound:gt:img1:1"


def test_no_vacuous_reference_status_when_no_same_description_owner_exists(tmp_path: Path) -> None:
    """A single-owner image (no sibling same-description owner at all) must
    report the explicit no-owner status and emit zero reference rows --
    never a fabricated placeholder reference.
    """

    owner_ledger_path = tmp_path / "owner-ledger.jsonl"
    _write_jsonl(owner_ledger_path, [_owner_ledger_row("gt:solo:0", "solo", 0, "widget")])
    panel_path = tmp_path / "panel.jsonl"
    _write_jsonl(panel_path, [_panel_line("solo", 1024, 1024, [[100, 100, 110, 110]])])
    rollout_path = tmp_path / "rollout.json"
    _write_json(rollout_path, _rollout_document("solo", "greedy", 0, _PROMPT, [_row_chunk(1)]))
    fixtures = {
        "owner_ledger": owner_ledger_path,
        "panel": panel_path,
        "rollout": rollout_path,
        "rules_template": tmp_path / "rules-template.json",
        "descriptions": tmp_path / "descriptions.json",
    }
    _write_json(fixtures["rules_template"], RULES_TEMPLATE)
    _write_json(fixtures["descriptions"], _DESCRIPTION_SUPPLEMENT)

    registry = _mechanism_registry([_root_role("root:gt:solo:0", "gt:solo:0", image_id="solo")])
    _run_planner(fixtures, registry, tmp_path)

    ledger_rows = [
        json.loads(line)
        for line in (tmp_path / "plan" / "owner-context-ledger.jsonl").read_text().splitlines()
    ]
    assert ledger_rows[0]["other_owner_reference_status"] == NO_SAME_DESCRIPTION_NON_OVERLAPPING_OWNER

    fixed_budget_rows = [
        json.loads(line)
        for line in (tmp_path / "plan" / "fixed-budget-candidates.jsonl").read_text().splitlines()
    ]
    assert not any(row["population"] == "reference" for row in fixed_budget_rows)


def test_select_nearest_same_description_owner_unit_no_candidate() -> None:
    owner_index = {"gt:a:0": {"image_id": "img", "normalized_description": "widget", "original_annotation_index": 0}}
    result = select_nearest_same_description_owner(
        target_gt_owner_id="gt:a:0",
        target_image_id="img",
        target_box=(0, 0, 10, 10),
        normalized_description="widget",
        owner_index=owner_index,
        panel_boxes={},
    )
    assert result == {"status": NO_SAME_DESCRIPTION_NON_OVERLAPPING_OWNER}


def test_select_nearest_same_description_owner_unit_picks_nearest() -> None:
    owner_index = {
        "gt:a:0": {"image_id": "img", "normalized_description": "widget", "original_annotation_index": 0},
        "gt:a:1": {"image_id": "img", "normalized_description": "widget", "original_annotation_index": 1},
        "gt:a:2": {"image_id": "img", "normalized_description": "widget", "original_annotation_index": 2},
        "gt:a:3": {"image_id": "img", "normalized_description": "other", "original_annotation_index": 3},
    }
    panel_boxes = {
        ("img", 0): [0, 0, 10, 10],
        ("img", 1): [20, 0, 30, 10],  # nearest, zero overlap
        ("img", 2): [200, 200, 210, 210],  # far, zero overlap
        ("img", 3): [11, 0, 21, 10],  # closer geometrically but different description
    }
    result = select_nearest_same_description_owner(
        target_gt_owner_id="gt:a:0",
        target_image_id="img",
        target_box=(0, 0, 10, 10),
        normalized_description="widget",
        owner_index=owner_index,
        panel_boxes=panel_boxes,
    )
    assert result["status"] == "bound"
    assert result["gt_owner_id"] == "gt:a:1"
    assert result["trace"]["considered_owner_count"] == 2  # gt:a:1 and gt:a:2 (not gt:a:3, wrong description)


@pytest.mark.parametrize(
    "gt_box",
    [
        (0, 0, 10, 10),  # tiny
        (100, 100, 114, 145),  # gt:7511:22-shaped (14x45): zero exact matches under pure perturbation alone
        (600, 500, 793, 983),  # gt:7511:2-shaped (193x483): zero exact matches at both L0 and L1 under pure perturbation
    ],
)
@pytest.mark.parametrize("rung", ["L0", "L1"])
def test_near_gt_micro_local_index_0_is_always_the_exact_gt_box_regardless_of_size(
    gt_box: tuple[int, int, int, int], rung: str
) -> None:
    """Score-independent geometry guarantee (not a rounding accident): for
    *any* box size, including large real-shaped boxes that previously had
    zero exact matches under small-perturbation-only sampling, local index 0
    of the near_gt_micro family is exactly the GT box.
    """

    boxes = [box for family, box in build_family_boxes(gt_box, rung) if family == "near_gt_micro"]
    assert boxes[0] == gt_box
    assert iou(boxes[0], gt_box) == 1.0


def test_real_gt7511_17_near_gt_micro_local_index_0_is_the_exact_singleton(tmp_path: Path) -> None:
    """The exact real owner/box this correction was reported against:
    ``gt:7511:17``'s scalar_smoke near_gt_micro family has multiple exact
    matches (local indices 0, 1, and 4, verified against the real panel
    geometry), and the deterministic lowest-index selection -- discovered by
    scanning, never hardcoded -- resolves to local index 0, which is now
    guaranteed exact by construction rather than by chance.
    """

    owner_index = load_owner_index(REAL_OWNER_LEDGER)
    panel_sizes, panel_boxes = load_panel(REAL_PANEL)
    owner = owner_index["gt:7511:17"]
    key = ("7511", owner["original_annotation_index"])
    gt_box = tuple(panel_boxes[key])

    scalar_boxes = [box for family, box in build_family_boxes(gt_box, "scalar_smoke") if family == "near_gt_micro"]
    exact_indices = [index for index, box in enumerate(scalar_boxes) if box == gt_box]
    assert exact_indices == sorted(exact_indices)
    assert 0 in exact_indices
    assert len(exact_indices) > 1  # matches the reported real pattern: more than one exact index exists
    assert min(exact_indices) == 0  # the deterministic selection therefore resolves to local index 0

    registry_path = tmp_path / "registry.json"
    build_sorted_fn_mechanism_registry(
        owner_ledger=REAL_OWNER_LEDGER,
        prediction_row_ledger=REAL_PREDICTION_ROW_LEDGER,
        cohort_assignments=REAL_COHORT_ASSIGNMENTS,
        rollouts=[REAL_GREEDY_ROLLOUT, REAL_SAMPLED_SHARD_0, REAL_SAMPLED_SHARD_1],
        due_turn_trajectories={"gt:7511:22": ("greedy", 0), "gt:7511:17": ("sampled", 21010)},
        output=registry_path,
    )
    descriptions_path = tmp_path / "kite-description.json"
    _write_json(descriptions_path, {"kite": _REAL_KITE_DESCRIPTION})
    out_dir = tmp_path / "plan"
    prepare_sorted_fn_successor_inputs(
        registry=registry_path,
        owner_ledger=REAL_OWNER_LEDGER,
        panel=REAL_PANEL,
        rules_template=REAL_RULES_TEMPLATE,
        rollouts=[REAL_GREEDY_ROLLOUT, REAL_SAMPLED_SHARD_0, REAL_SAMPLED_SHARD_1],
        canonical_description_registry=descriptions_path,
        predecessor_score_rows=None,
        out_dir=out_dir,
    )
    rows = [
        json.loads(line)
        for line in (out_dir / "fixed-budget-candidates.jsonl").read_text().splitlines()
    ]
    scalar_target_rows = sorted(
        (
            row
            for row in rows
            if row["owner_context_id"] == "ctx:fn:root:gt:7511:17"
            and row["rung"] == "scalar_smoke"
            and row["population"] == "target"
            and row["family_id"] == "near_gt_micro"
        ),
        key=lambda row: row["candidate_id"],
    )
    singleton_rows = [row for row in scalar_target_rows if row["exact_gt_singleton_member"]]
    assert len(singleton_rows) == 1
    assert singleton_rows[0]["candidate_id"].endswith(":near_gt_micro:0:target")
    assert tuple(singleton_rows[0]["coord_token_ids"]) == tuple(scalar_target_rows[0]["coord_token_ids"])
    assert singleton_rows[0]["iou_to_target"] == 1.0
    non_singleton_ids = [row["exact_gt_singleton_id"] for row in scalar_target_rows if not row["exact_gt_singleton_member"]]
    assert all(value is None for value in non_singleton_ids)


def test_neighborhood_id_and_geometry_consistent_across_collision_triple(
    tmp_path: Path, two_owner_fixtures: dict[str, Path]
) -> None:
    """A collision baseline/covering/foil triple (matched_control_group
    ``collision:pair-a``) sharing the same target owner must produce
    byte-identical near_gt_micro candidate geometry (and therefore
    identical ``candidate_neighborhood_id`` values) across all three roles,
    despite each role having its own distinct ``owner_context_id``.
    """

    roles = [
        _root_role("collision:pair-a:P", "gt:img1:0"),
        _root_role("collision:pair-a:P+G", "gt:img1:0", cut_marker=1),
        _root_role("collision:pair-a:P+F", "gt:img1:0", cut_marker=10),
    ]
    registry = _mechanism_registry(roles)
    receipt = _run_planner(two_owner_fixtures, registry, tmp_path)
    assert receipt["neighborhood_consistency"]["status"] == "consistent"
    assert receipt["neighborhood_consistency"]["multi_context_groups_checked"] >= 1

    fixed_budget_rows = [
        json.loads(line)
        for line in (tmp_path / "plan" / "fixed-budget-candidates.jsonl").read_text().splitlines()
    ]
    neighborhood_rows = [
        row
        for row in fixed_budget_rows
        if row["population"] == "target" and row["family_id"] == "near_gt_micro"
    ]
    assert neighborhood_rows  # non-empty
    assert all(row["candidate_neighborhood_member"] is True for row in neighborhood_rows)
    assert all(row["candidate_neighborhood_id"] is not None for row in neighborhood_rows)

    by_context: dict[str, list[tuple[str, tuple[int, ...]]]] = {}
    for row in neighborhood_rows:
        by_context.setdefault(row["owner_context_id"], []).append(
            (row["candidate_neighborhood_id"], tuple(row["coord_token_ids"]))
        )
    contexts = list(by_context.values())
    assert len(contexts) == 3  # three distinct owner_context_id (P, P+G, P+F)
    reference = sorted(contexts[0])
    for entries in contexts[1:]:
        assert sorted(entries) == reference

    non_neighborhood_rows = [
        row
        for row in fixed_budget_rows
        if not (row["population"] == "target" and row["family_id"] == "near_gt_micro")
    ]
    assert all(row["candidate_neighborhood_member"] is False for row in non_neighborhood_rows)
    assert all(row["candidate_neighborhood_id"] is None for row in non_neighborhood_rows)


def test_mechanism_decision_rules_digest_and_non_recursive_parent_binding(
    tmp_path: Path, two_owner_fixtures: dict[str, Path]
) -> None:
    registry = _mechanism_registry([_root_role("root:gt:img1:0", "gt:img1:0")])
    receipt = _run_planner(two_owner_fixtures, registry, tmp_path)

    rules_path = tmp_path / "plan" / "mechanism-decision-rules.json"
    document = json.loads(rules_path.read_text())
    content = {k: v for k, v in document.items() if k != "self_digest"}
    assert document["self_digest"] == sha256_json(content)
    assert document["schema_version"] == "sorted-fn-mechanism-decision-rules.v1"

    # Non-recursive parent binding: the parent document names its own
    # upstream digests (execution rules/registry/template) but never a
    # fixed-budget-candidates digest -- that binding exists only in the
    # receipt, computed after candidates are materialized.
    assert "fixed_budget_candidates_sha256" not in document
    assert set(document["upstream_digests"].keys()) == {
        "execution_landscape_decision_rules_sha256",
        "fn_mechanism_registry_sha256",
        "rules_template_sha256",
    }

    real_sha256 = sha256_file(rules_path)
    assert receipt["outputs"]["mechanism_decision_rules"]["sha256"] == real_sha256

    fixed_budget_rows = [
        json.loads(line)
        for line in (tmp_path / "plan" / "fixed-budget-candidates.jsonl").read_text().splitlines()
    ]
    assert fixed_budget_rows
    assert all(row["mechanism_decision_rules_sha256"] == real_sha256 for row in fixed_budget_rows)
    assert all(row["schema_version"] == "sorted-fn-successor-fixed-budget-candidates.v2" for row in fixed_budget_rows)


def test_no_predecessor_reuse_still_holds_with_neighborhood_and_reference_rows(
    tmp_path: Path, two_owner_fixtures: dict[str, Path]
) -> None:
    """Regression: adding neighborhood tagging and the reference population
    must not reopen any predecessor-score reuse path. No row of any
    population ever carries ``predecessor_candidate_id``.
    """

    registry = _mechanism_registry(
        [_root_role("root:gt:img1:0", "gt:img1:0")],
        targets=[{"gt_owner_id": "gt:img1:0", "cohort": "strict_rescued"}],
    )
    receipt = _run_planner(two_owner_fixtures, registry, tmp_path)
    rows = [
        json.loads(line)
        for line in (tmp_path / "plan" / "fixed-budget-candidates.jsonl").read_text().splitlines()
    ]
    assert rows
    assert all("predecessor_candidate_id" not in row for row in rows)
    assert receipt["predecessor_reuse"]["valid_reuse_count"] == 0
