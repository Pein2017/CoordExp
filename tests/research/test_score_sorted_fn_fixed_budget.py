"""Contracts for the successor-local restricted-complete-box fixed-budget scorer.

Static-contract fixtures run the real, unchanged
``prepare_sorted_fn_successor_inputs`` planner to produce a genuine
owner-context-ledger, decision rules, and fixed-budget candidate lattice
(mirroring ``test_merge_sorted_fn_successor_score_shards.py``'s own
fixture, since both consume the exact same five upstream artifacts).
Live-math tests exercise the reused predecessor scoring primitives
(``FullReforwardBackend``, ``score_complete_box_candidate``,
``run_batched_reforward_parity_gate``) against deterministic fake forward
closures -- never a real model, tokenizer, or GPU.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from scripts.research.build_sorted_owner_basin_candidates import (
    sha256_file,
    sha256_json,
)
from scripts.research.sorted_owner_basin_landscape import (
    GEOMETRY_IDENTITY_SCHEMA,
    RULES_SCHEMA_VERSION,
)
from scripts.research import score_sorted_owner_basin_landscape as scorer
from scripts.research.build_sorted_fn_mechanism_registry import (
    SCHEMA_VERSION as REGISTRY_SCHEMA_VERSION,
    UNIT_ID as SUCCESSOR_UNIT_ID,
)
from scripts.research.prepare_sorted_fn_successor_inputs import (
    prepare_sorted_fn_successor_inputs,
)
import scripts.research.score_sorted_fn_fixed_budget as sut

_REAL_STATIC_LIVE_CONFIG_BINDING = (
    sut.behavior_runner.validate_static_live_config_binding
)

# ---------------------------------------------------------------------------
# Minimal, self-consistent rules-template fixture (mirrors
# test_merge_sorted_fn_successor_score_shards.py's own fixture: same five
# upstream artifacts, same real planner).
# ---------------------------------------------------------------------------

FAKE_DIGEST_A = "a" * 64
FAKE_DIGEST_B = "b" * 64
FAKE_DIGEST_C = "c" * 64
FAKE_IDENTITY_RECEIPT_DIGEST = "d" * 64


@pytest.fixture(autouse=True)
def _mock_production_runtime_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep scorer tests CPU/local while proving delegation to the live owner.

    The behavior runner's production identity/config validators have their own
    focused source-rebuild/config tests.  This suite supplies their accepted
    return contract so the fixed-budget scorer can exercise its orchestration
    without manufacturing a Task-0 manifest and model-component tree in every
    planner fixture.
    """

    def validate_runtime_identity(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        return document, {
            "path": str(Path(path).resolve()),
            "sha256": sha256_file(Path(path)),
            "receipt_digest": document["receipt_digest"],
            "recomputation_status": "exact_source_rebuild_passed",
        }

    def validate_static_live_config_binding(
        *,
        infer_config_path: Path,
        source_jsonl_path: Path,
        runtime_identity: dict[str, Any],
        runtime_receipt: dict[str, Any],
    ) -> dict[str, Any]:
        assert infer_config_path.is_file()
        assert runtime_identity["receipt_digest"] == runtime_receipt["receipt_digest"]
        return {
            "status": "passed_cpu_before_live_load",
            "resolved_config_fingerprint": FAKE_DIGEST_A,
            "generation_config_fingerprint": FAKE_DIGEST_B,
            "configured_components": {},
            "source_jsonl": {
                "path": str(source_jsonl_path),
                "sha256": sha256_file(source_jsonl_path),
            },
        }

    monkeypatch.setattr(
        sut.behavior_runner, "validate_runtime_identity", validate_runtime_identity
    )
    monkeypatch.setattr(
        sut.behavior_runner,
        "validate_static_live_config_binding",
        validate_static_live_config_binding,
    )


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
TOKEN_REGISTRY = {
    **_TOKEN_REGISTRY_PAYLOAD,
    "registry_sha256": sha256_json(_TOKEN_REGISTRY_PAYLOAD),
}
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
    "contract_mode": "test_fixture",
    "geometry_identity": {
        "schema": GEOMETRY_IDENTITY_SCHEMA,
        "coordinate_denominator": 1000,
    },
    "coordinate_bins": {"min": 0, "max": 999},
    "target_anchor": {
        "margin_fraction": 0.10,
        "min_margin_bins": 2,
        "max_margin_bins": 32,
    },
    "bank_order": ["target", "background", "scan"],
    "proposal_measures": {
        "primary": {
            "comparability_group": "primary",
            "normalization": "full_domain_normalized_weighted_sum",
            "bank_weights": {"target": 1.0, "background": 1.0, "scan": 1.0},
        }
    },
    "bank_proposal_measure": {
        "target": "primary",
        "background": "primary",
        "scan": "primary",
    },
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
            "non_production_reason": "fixed-budget scorer contract test fixture; not a decision-bearing run",
            "bin_to_extent_conversion": "round(value*extent/1000)",
        },
        "coco_namespace": {
            "name": "coco_2017_official_gapped",
            "id_space": "official_gapped",
        },
        "target_bank_name": "target",
        "bank_roles": {
            "target": TARGET_ROLE_ID,
            "background": BACKGROUND_ROLE_ID,
            "scan": SCAN_ROLE_ID,
        },
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
    "forced_row_prefix_through_box_start_sha256": sha256_json(
        [100000, 777, 100001, 100002]
    ),
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


def _owner_ledger_row(
    gt_owner_id: str, image_id: str, annotation_index: int, description: str
) -> dict[str, Any]:
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


def _panel_line(
    image_id: str, width: int, height: int, boxes: list[list[int]]
) -> dict[str, Any]:
    return {
        "image_id": str(image_id),
        "width": width,
        "height": height,
        "objects": [{"bbox_2d": list(box)} for box in boxes],
    }


def _rollout_document(
    image_id: str,
    decode_mode: str,
    seed: int,
    prompt_token_ids: list[int],
    row_chunks: list[list[int]],
) -> dict[str, Any]:
    generated = [token for chunk in row_chunks for token in chunk]
    predictions = [
        {"generated_order": idx, "raw_span_sha256": "e" * 64}
        for idx in range(len(row_chunks))
    ]
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


def _mechanism_registry(
    roles: list[dict[str, Any]], *, targets: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    return {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "unit_id": SUCCESSOR_UNIT_ID,
        "registry_digest": "f" * 64,
        "mechanism_cohort": {"targets": targets or []},
        "context_control_registry": {"bound_non_targets": []},
        "smoke": {"roles": roles, "null_pair_envelope": {"pairs": []}},
    }


def _context_role(
    role_id: str,
    gt_owner_id: str,
    *,
    role_kind: str = "root_context",
    image_id: str = "img1",
    cut_marker: int | None = None,
) -> dict[str, Any]:
    token_ids = list(_PROMPT)
    if cut_marker is not None:
        token_ids = token_ids + _row_chunk(cut_marker)
    return {
        "role_id": role_id,
        "role_kind": role_kind,
        "gt_owner_id": gt_owner_id,
        "trajectory": {"image_id": image_id, "decode_mode": "greedy", "seed": 0},
        "prefix": {"token_ids": token_ids},
        "provenance": {
            "source_artifact_path": "rollout.json",
            "source_artifact_sha256": "0" * 64,
        },
    }


@pytest.fixture()
def planned(tmp_path: Path) -> dict[str, Any]:
    """Run the real, unchanged planner and return every produced artifact path.

    Five roles across three owners (more than the predecessor's closed
    four-context world) so "arbitrary contexts" is meaningfully exercised.
    """

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
    _write_jsonl(
        panel_path,
        [
            _panel_line(
                "img1",
                1024,
                1024,
                # Small (6x6) boxes: the current planner's F3 exact-GT-singleton
                # micro-perturbation collapse is only guaranteed for small boxes
                # at every rung (including scalar_smoke/L0's 6-member family);
                # see build_reference_family_boxes/exact_gt_singleton_member.
                [[100, 100, 106, 106], [300, 300, 306, 306], [500, 500, 506, 506]],
            )
        ],
    )
    infer_config_path = fixtures_dir / "infer.yaml"
    infer_config_path.write_text("fixture: true\n", encoding="utf-8")
    rollout_path = fixtures_dir / "rollout.json"
    _write_json(
        rollout_path,
        _rollout_document(
            "img1", "greedy", 0, _PROMPT, [_row_chunk(1), _row_chunk(10)]
        ),
    )
    rules_template_path = fixtures_dir / "rules-template.json"
    _write_json(rules_template_path, RULES_TEMPLATE)
    descriptions_path = fixtures_dir / "descriptions.json"
    _write_json(descriptions_path, _DESCRIPTION_SUPPLEMENT)

    roles = [
        _context_role("root:gt:img1:0", "gt:img1:0"),
        _context_role("root:gt:img1:1", "gt:img1:1"),
        _context_role("root:gt:img1:2", "gt:img1:2"),
        _context_role(
            "due_turn:gt:img1:0",
            "gt:img1:0",
            role_kind="due_turn_context",
            cut_marker=1,
        ),
        _context_role(
            "due_turn:gt:img1:2",
            "gt:img1:2",
            role_kind="due_turn_context",
            cut_marker=10,
        ),
    ]
    registry_path = tmp_path / "registry.json"
    _write_json(
        registry_path,
        _mechanism_registry(
            roles, targets=[{"gt_owner_id": "gt:img1:2", "cohort": "strict_rescued"}]
        ),
    )

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
    fixed_budget_rows = [
        json.loads(line) for line in fixed_budget_path.read_text().splitlines()
    ]
    rules = scorer.load_decision_rules(rules_path)
    return {
        "tmp_path": tmp_path,
        "registry_path": registry_path,
        "ledger_path": ledger_path,
        "rules_path": rules_path,
        "mechanism_rules_path": mechanism_rules_path,
        "fixed_budget_path": fixed_budget_path,
        "planner_receipt_path": planner_receipt_path,
        "panel_path": panel_path,
        "infer_config_path": infer_config_path,
        "planner_receipt": receipt,
        "ledger_rows": ledger_rows,
        "ledger_by_context": {row["context_id"]: row for row in ledger_rows},
        "fixed_budget_rows": fixed_budget_rows,
        "rules": rules,
    }


def _runtime_identity_path(
    planned: dict[str, Any],
    tmp_path: Path,
    *,
    tokenizer_digest: str | None = None,
    model_digest: str | None = None,
) -> Path:
    content: dict[str, Any] = {
        "schema_version": sut._RUNTIME_IDENTITY_SCHEMA_VERSION,
        "status": "frozen",
        "tokenizer": {
            "identity_sha256": tokenizer_digest
            or VOCAB_ATTESTATION["tokenizer_identity_sha256"]
        },
        "model": {
            "identity_sha256": model_digest
            or VOCAB_ATTESTATION["model_identity_sha256"]
        },
        "model_vocab_size": MODEL_VOCAB_SIZE,
        "coordinate_vocabulary": {
            "token_id_start": COORD_TOKEN_START,
            "token_id_end_exclusive": COORD_TOKEN_END,
        },
    }
    content["receipt_digest"] = sha256_json(content)
    path = tmp_path / "runtime-identity.json"
    _write_json(path, content)
    return path


_DERIVE_EXPLICIT_RUNGS = object()


def _contract_args(
    planned: dict[str, Any],
    tmp_path: Path,
    *,
    runtime_identity_path: Path | None = None,
    include_context_id: list[str] | None = None,
    include_rung: list[str] | None | object = _DERIVE_EXPLICIT_RUNGS,
) -> argparse.Namespace:
    if include_rung is _DERIVE_EXPLICIT_RUNGS:
        # Every fixture context materializes L1, so tests that are not about
        # rung selection still make one explicit, scientifically attestable
        # choice rather than exercising a forbidden mixed/default route.
        explicit_rungs = ["L1"]
    else:
        explicit_rungs = list(include_rung or [])
    return sut.build_parser().parse_args(
        [
            "--planner-receipt",
            str(planned["planner_receipt_path"]),
            "--fn-mechanism-registry",
            str(planned["registry_path"]),
            "--owner-context-ledger",
            str(planned["ledger_path"]),
            "--decision-rules",
            str(planned["rules_path"]),
            "--mechanism-decision-rules",
            str(planned["mechanism_rules_path"]),
            "--fixed-budget-candidates",
            str(planned["fixed_budget_path"]),
            "--runtime-identity",
            str(runtime_identity_path or _runtime_identity_path(planned, tmp_path)),
            "--infer-config",
            str(planned["infer_config_path"]),
            "--source-jsonl",
            str(planned["panel_path"]),
            "--validate-contract-only",
            *(
                [
                    arg
                    for cid in (include_context_id or [])
                    for arg in ("--include-context-id", cid)
                ]
            ),
            *([arg for rung in explicit_rungs for arg in ("--include-rung", rung)]),
        ]
    )


# ---------------------------------------------------------------------------
# Fixture self-check
# ---------------------------------------------------------------------------


def test_planner_produced_more_than_four_contexts(planned: dict[str, Any]) -> None:
    assert len(planned["ledger_by_context"]) > 4


def test_planner_produced_exact_gt_singleton_and_reference_candidates(
    planned: dict[str, Any],
) -> None:
    # F3 (exact-GT-box singleton) and population=reference candidates are
    # both real producer-side outputs this scorer must carry through
    # build_score_row unmodified; see test_merge_sorted_fn_successor_score_
    # shards.py for large-box (100x100+) end-to-end carry-through coverage.
    singleton_rows = [
        row
        for row in planned["fixed_budget_rows"]
        if row["exact_gt_singleton_member"] is True
    ]
    reference_rows = [
        row for row in planned["fixed_budget_rows"] if row["population"] == "reference"
    ]
    assert singleton_rows
    assert reference_rows
    assert all(
        row["exact_gt_singleton_id"].startswith("exact-gt:") for row in singleton_rows
    )


# ---------------------------------------------------------------------------
# --validate-contract-only: exact joins, digests, identity, rejection paths
# ---------------------------------------------------------------------------


def test_contract_only_passes_with_exact_joins_and_no_model_import_side_effect(
    planned: dict[str, Any], tmp_path: Path
) -> None:
    args = _contract_args(planned, tmp_path)
    receipt = sut.run(args)
    assert receipt["schema_version"] == sut.RECEIPT_SCHEMA_VERSION
    assert receipt["unit_id"] == sut.UNIT_ID
    assert (
        receipt["unit_id"] != scorer.UNIT_ID
    )  # never impersonates the predecessor's unit
    assert receipt["runtime_execution_status"] == sut.CONTRACT_VALIDATED_STATUS
    assert receipt["candidate_count"] == sum(
        row["rung"] == "L1" for row in planned["fixed_budget_rows"]
    )
    assert set(receipt["context_selection"]["included_context_ids"]) == set(
        planned["ledger_by_context"]
    )
    assert receipt["live_config_admission"]["status"] == "passed_cpu_before_live_load"
    assert receipt["source_digests"]["infer_config"]["path"] == str(
        planned["infer_config_path"].resolve()
    )
    assert receipt["source_digests"]["source_jsonl"] == {
        "path": str(planned["panel_path"].resolve()),
        "sha256": sha256_file(planned["panel_path"]),
    }


def test_contract_only_preserves_sixty_scalar_context_rows(
    planned: dict[str, Any], tmp_path: Path
) -> None:
    scalar_rows = [
        row for row in planned["fixed_budget_rows"] if row["rung"] == "scalar_smoke"
    ]
    scalar_context_ids = {row["owner_context_id"] for row in scalar_rows}
    assert scalar_context_ids
    # 30 target + 30 decoy (unchanged primary counts) + 7 reference (this
    # fixture's owners share one normalized_description in one image, so a
    # same-description zero-overlap owner is always bound).
    counts = {
        context_id: sum(
            1 for row in scalar_rows if row["owner_context_id"] == context_id
        )
        for context_id in scalar_context_ids
    }
    assert set(counts.values()) == {67}
    for context_id in scalar_context_ids:
        context_rows = [
            row for row in scalar_rows if row["owner_context_id"] == context_id
        ]
        by_population = {
            population: sum(
                1 for row in context_rows if row["population"] == population
            )
            for population in ("target", "decoy", "reference")
        }
        assert by_population == {"target": 30, "decoy": 30, "reference": 7}
    receipt = sut.run(_contract_args(planned, tmp_path))
    assert receipt["candidate_count"] == sum(
        row["rung"] == "L1" for row in planned["fixed_budget_rows"]
    )


def test_contract_only_invokes_full_production_binding_before_return(
    planned: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def validate_runtime_identity(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
        calls.append("runtime_identity")
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        return document, {"receipt_digest": document["receipt_digest"]}

    def validate_static_live_config_binding(**kwargs: Any) -> dict[str, Any]:
        calls.append("static_live_config")
        assert kwargs["infer_config_path"] == planned["infer_config_path"].resolve()
        assert kwargs["source_jsonl_path"] == planned["panel_path"].resolve()
        return {
            "status": "passed_cpu_before_live_load",
            "resolved_config_fingerprint": FAKE_DIGEST_A,
            "generation_config_fingerprint": FAKE_DIGEST_B,
            "configured_components": {},
            "source_jsonl": {
                "path": str(planned["panel_path"].resolve()),
                "sha256": sha256_file(planned["panel_path"]),
            },
        }

    monkeypatch.setattr(
        sut.behavior_runner, "validate_runtime_identity", validate_runtime_identity
    )
    monkeypatch.setattr(
        sut.behavior_runner,
        "validate_static_live_config_binding",
        validate_static_live_config_binding,
    )
    receipt = sut.run(_contract_args(planned, tmp_path))
    assert calls == ["runtime_identity", "static_live_config"]
    assert receipt["runtime_execution_status"] == sut.CONTRACT_VALIDATED_STATUS


@pytest.mark.parametrize(
    "message",
    [
        "resolved infer config fingerprint differs from frozen runtime identity",
        "generation config fingerprint differs from frozen runtime identity",
        "live behavior config must be HF FP32",
        "infer config source JSONL differs from the explicitly bound source",
        "source JSONL differs from the panel frozen by runtime identity",
        "opened HF runtime processor identity mismatch",
    ],
)
def test_contract_only_fails_closed_on_production_live_binding_mismatch(
    planned: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    message: str,
) -> None:
    def reject(**_kwargs: Any) -> dict[str, Any]:
        raise sut.behavior_runner.BehaviorContractError(message)

    monkeypatch.setattr(
        sut.behavior_runner, "validate_static_live_config_binding", reject
    )
    with pytest.raises(sut.FixedBudgetScoringError, match=message):
        sut.run(_contract_args(planned, tmp_path))


def _production_config_binding_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    from src.config.fingerprint import sha256_json as config_sha256_json
    import src.config.inference as inference_config_module

    source_path = tmp_path / "source.jsonl"
    source_path.write_text('{"image_id":"img1"}\n', encoding="utf-8")
    other_source_path = tmp_path / "other-source.jsonl"
    other_source_path.write_text('{"image_id":"other"}\n', encoding="utf-8")
    infer_config_path = tmp_path / "infer.yaml"
    infer_config_path.write_text("production: fixture\n", encoding="utf-8")
    base_path = tmp_path / "base"
    adapter_path = tmp_path / "adapter"
    delta_path = tmp_path / "delta"
    generation_payload = {"max_new_tokens": 128, "temperature": 0.0}
    generation_fingerprint = config_sha256_json(generation_payload)
    resolved_fingerprint = "5" * 64
    config = SimpleNamespace(
        backend=SimpleNamespace(type="hf"),
        model=SimpleNamespace(dtype="fp32", base_model=str(base_path)),
        adapter=SimpleNamespace(path=str(adapter_path)),
        embedding_delta=SimpleNamespace(path=str(delta_path)),
        data=SimpleNamespace(input_jsonl=str(source_path)),
        generation=SimpleNamespace(model_dump=lambda **_kwargs: generation_payload),
    )
    resolved = SimpleNamespace(config=config, fingerprint=resolved_fingerprint)
    monkeypatch.setattr(
        inference_config_module, "load_infer_config", lambda _path: resolved
    )
    runtime_identity = {
        "runtime": {
            "identity_source": {
                "resolved_config_fingerprints": {
                    "infer_config": resolved_fingerprint,
                },
                "generation_config_fingerprint": generation_fingerprint,
            }
        },
        "model": {
            "identity_source": {
                "model_identity": {
                    "base": {"path": str(base_path.resolve())},
                    "adapter": {"adapter_path": str(adapter_path.resolve())},
                    "embedding_delta": {
                        "identity": {"delta_path": str(delta_path.resolve())}
                    },
                }
            }
        },
    }
    runtime_receipt = {
        "frozen_source_panel": {
            "path": str(source_path.resolve()),
            "sha256": sha256_file(source_path),
        }
    }
    return {
        "infer_config_path": infer_config_path,
        "source_path": source_path,
        "other_source_path": other_source_path,
        "config": config,
        "resolved": resolved,
        "runtime_identity": runtime_identity,
        "runtime_receipt": runtime_receipt,
    }


def test_exact_production_static_live_config_binding_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _production_config_binding_fixture(tmp_path, monkeypatch)
    admission = _REAL_STATIC_LIVE_CONFIG_BINDING(
        infer_config_path=fixture["infer_config_path"],
        source_jsonl_path=fixture["source_path"].resolve(),
        runtime_identity=fixture["runtime_identity"],
        runtime_receipt=fixture["runtime_receipt"],
    )
    assert admission["status"] == "passed_cpu_before_live_load"
    assert admission["source_jsonl"]["sha256"] == sha256_file(fixture["source_path"])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("resolved_fingerprint", "resolved infer config fingerprint"),
        ("generation_fingerprint", "generation config fingerprint"),
        ("backend", "HF FP32"),
        ("dtype", "HF FP32"),
        ("configured_source", "explicitly bound source"),
        ("frozen_panel_digest", "panel frozen by runtime identity"),
        ("model_component", "model components differ"),
    ],
)
def test_exact_production_static_live_config_binding_rejects_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    fixture = _production_config_binding_fixture(tmp_path, monkeypatch)
    if mutation == "resolved_fingerprint":
        fixture["resolved"].fingerprint = "6" * 64
    elif mutation == "generation_fingerprint":
        fixture["runtime_identity"]["runtime"]["identity_source"][
            "generation_config_fingerprint"
        ] = "6" * 64
    elif mutation == "backend":
        fixture["config"].backend.type = "vllm"
    elif mutation == "dtype":
        fixture["config"].model.dtype = "bf16"
    elif mutation == "configured_source":
        fixture["config"].data.input_jsonl = str(fixture["other_source_path"])
    elif mutation == "frozen_panel_digest":
        fixture["runtime_receipt"]["frozen_source_panel"]["sha256"] = "6" * 64
    elif mutation == "model_component":
        fixture["runtime_identity"]["model"]["identity_source"]["model_identity"][
            "base"
        ]["path"] = str((tmp_path / "wrong-base").resolve())
    with pytest.raises(sut.behavior_runner.BehaviorContractError, match=message):
        _REAL_STATIC_LIVE_CONFIG_BINDING(
            infer_config_path=fixture["infer_config_path"],
            source_jsonl_path=fixture["source_path"].resolve(),
            runtime_identity=fixture["runtime_identity"],
            runtime_receipt=fixture["runtime_receipt"],
        )


def test_contract_only_arbitrary_context_subset_selection(
    planned: dict[str, Any], tmp_path: Path
) -> None:
    chosen = sorted(planned["ledger_by_context"])[:2]
    args = _contract_args(planned, tmp_path, include_context_id=chosen)
    receipt = sut.run(args)
    assert receipt["context_selection"]["included_context_ids"] == chosen
    expected_count = sum(
        1
        for row in planned["fixed_budget_rows"]
        if row["owner_context_id"] in chosen and row["rung"] == "L1"
    )
    assert receipt["candidate_count"] == expected_count


def test_contract_only_strict_rescue_scalar_rung_selects_exactly_67_without_broadening(
    planned: dict[str, Any], tmp_path: Path
) -> None:
    context_id = "ctx:fn:root:gt:img1:2"
    with pytest.raises(sut.FixedBudgetScoringError, match="cannot be combined"):
        sut.run(
            _contract_args(
                planned,
                tmp_path,
                include_context_id=[context_id],
                include_rung=["scalar_smoke", "L1"],
            )
        )

    scalar_only = sut.run(
        _contract_args(
            planned,
            tmp_path,
            include_context_id=[context_id],
            include_rung=["scalar_smoke"],
        )
    )
    assert scalar_only["candidate_count"] == 67
    assert scalar_only["selection"]["selected_context_ids"] == [context_id]
    assert scalar_only["selection"]["selected_rungs"] == ["scalar_smoke"]
    assert scalar_only["selection"]["candidate_count"] == 67
    expected_keys = sorted(
        [row["owner_context_id"], row["candidate_id"]]
        for row in planned["fixed_budget_rows"]
        if row["owner_context_id"] == context_id and row["rung"] == "scalar_smoke"
    )
    assert scalar_only["selection"]["candidate_keyset_sha256"] == sha256_json(
        expected_keys
    )
    assert scalar_only["rung_selection"]["selected_candidate_counts_by_rung"] == {
        "scalar_smoke": 67
    }
    assert scalar_only["rung_selection"]["excluded_rungs_for_selected_contexts"] == [
        "L1"
    ]


def test_contract_only_selected_rungs_have_distinct_output_identities(
    planned: dict[str, Any], tmp_path: Path
) -> None:
    context_id = "ctx:fn:root:gt:img1:2"
    scalar = sut.run(
        _contract_args(
            planned,
            tmp_path,
            include_context_id=[context_id],
            include_rung=["scalar_smoke"],
        )
    )
    l1 = sut.run(
        _contract_args(
            planned,
            tmp_path,
            include_context_id=[context_id],
            include_rung=["L1"],
        )
    )
    assert l1["candidate_count"] == 577
    assert l1["selection"]["selected_rungs"] == ["L1"]
    assert (
        scalar["selection"]["selection_sha256"] != l1["selection"]["selection_sha256"]
    )
    for receipt in (scalar, l1):
        selection = receipt["selection"]
        content = {
            key: value for key, value in selection.items() if key != "selection_sha256"
        }
        assert selection["selection_sha256"] == sha256_json(content)


def test_contract_only_rung_selection_is_canonical_across_cli_order(
    planned: dict[str, Any], tmp_path: Path
) -> None:
    context_id = "ctx:fn:root:gt:img1:0"
    forward = sut.run(
        _contract_args(
            planned,
            tmp_path,
            include_context_id=[context_id],
            include_rung=["L0", "L1"],
        )
    )
    reverse = sut.run(
        _contract_args(
            planned,
            tmp_path,
            include_context_id=[context_id],
            include_rung=["L1", "L0"],
        )
    )
    assert forward["selection"] == reverse["selection"]


@pytest.mark.parametrize(
    ("rungs", "message"),
    [
        (["scalar_smoke", "scalar_smoke"], "duplicates"),
        (["not-a-rung"], "absent from mechanism decision rules"),
        (["L2"], "no candidates in the selected contexts"),
    ],
)
def test_contract_only_rejects_invalid_rung_allowlists(
    planned: dict[str, Any],
    tmp_path: Path,
    rungs: list[str],
    message: str,
) -> None:
    context_id = "ctx:fn:root:gt:img1:2"
    with pytest.raises(sut.FixedBudgetScoringError, match=message):
        sut.run(
            _contract_args(
                planned,
                tmp_path,
                include_context_id=[context_id],
                include_rung=rungs,
            )
        )


def test_contract_only_rejects_empty_context_rung_intersection(
    planned: dict[str, Any], tmp_path: Path
) -> None:
    with pytest.raises(sut.FixedBudgetScoringError, match="no candidates"):
        sut.run(
            _contract_args(
                planned,
                tmp_path,
                include_context_id=["ctx:fn:root:gt:img1:0"],
                include_rung=["scalar_smoke"],
            )
        )


def test_contract_only_rejects_omitted_rung_after_static_validation(
    planned: dict[str, Any], tmp_path: Path
) -> None:
    with pytest.raises(sut.FixedBudgetScoringError, match="explicit --include-rung"):
        sut.run(_contract_args(planned, tmp_path, include_rung=None))


def test_contract_only_rejects_unknown_include_context_id(
    planned: dict[str, Any], tmp_path: Path
) -> None:
    args = _contract_args(planned, tmp_path, include_context_id=["ctx:does-not-exist"])
    with pytest.raises(sut.FixedBudgetScoringError, match="unknown"):
        sut.run(args)


def _reseal_planner_receipt_binding(
    planner_receipt_path: Path, *, section: str, key: str, mutated_path: Path
) -> None:
    """After hand-mutating one downstream artifact, re-bind the planner
    receipt's own digest to it so the generic "stale input" gate does not
    fire before the scorer's own semantic validation gets a chance to run.
    Mirrors ``_write_create_or_identical``'s create-or-identical contract:
    this rewrites the receipt file directly, exactly like the planner would
    if it had produced this (deliberately invalid) content itself.
    """

    receipt = json.loads(planner_receipt_path.read_text(encoding="utf-8"))
    entry = receipt[section][key]
    entry["sha256"] = sha256_file(mutated_path)
    if "row_count" in entry:
        entry["row_count"] = sum(
            1
            for line in mutated_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    content = {k: v for k, v in receipt.items() if k != "receipt_digest"}
    receipt["receipt_digest"] = sha256_json(content)
    planner_receipt_path.write_text(json.dumps(receipt), encoding="utf-8")


def test_contract_only_rejects_unknown_stale_registry_context(
    planned: dict[str, Any], tmp_path: Path
) -> None:
    ledger_path: Path = planned["ledger_path"]
    rows = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    rows[0]["context_provenance"] = {
        **rows[0]["context_provenance"],
        "registry_id": "role-that-was-never-registered",
    }
    ledger_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    _reseal_planner_receipt_binding(
        planned["planner_receipt_path"],
        section="outputs",
        key="owner_context_ledger",
        mutated_path=ledger_path,
    )

    args = _contract_args(planned, tmp_path)
    with pytest.raises(sut.FixedBudgetScoringError, match="unknown or stale"):
        sut.run(args)


def test_contract_only_rejects_score_derived_fixed_budget_field(
    planned: dict[str, Any], tmp_path: Path
) -> None:
    fixed_budget_path: Path = planned["fixed_budget_path"]
    rows = [json.loads(line) for line in fixed_budget_path.read_text().splitlines()]
    rows[0]["raw_score"] = -1.23
    fixed_budget_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    _reseal_planner_receipt_binding(
        planned["planner_receipt_path"],
        section="outputs",
        key="fixed_budget_candidates",
        mutated_path=fixed_budget_path,
    )

    args = _contract_args(planned, tmp_path)
    with pytest.raises(
        sut.FixedBudgetScoringError, match="forbidden score-derived field"
    ):
        sut.run(args)


def test_contract_only_rejects_stale_runtime_identity(
    planned: dict[str, Any], tmp_path: Path
) -> None:
    bad_identity_path = _runtime_identity_path(
        planned, tmp_path, tokenizer_digest="f" * 64
    )
    args = _contract_args(planned, tmp_path, runtime_identity_path=bad_identity_path)
    with pytest.raises(sut.FixedBudgetScoringError, match="tokenizer identity"):
        sut.run(args)


def test_contract_only_rejects_planner_panel_byte_mismatch(
    planned: dict[str, Any], tmp_path: Path
) -> None:
    planned["panel_path"].write_text('{"image_id":"changed"}\n', encoding="utf-8")
    with pytest.raises(sut.FixedBudgetScoringError, match="planner receipt.*digest"):
        sut.run(_contract_args(planned, tmp_path))


def test_contract_only_rejects_incompatible_coordinate_vocabulary(
    planned: dict[str, Any], tmp_path: Path
) -> None:
    content = {
        "schema_version": sut._RUNTIME_IDENTITY_SCHEMA_VERSION,
        "status": "frozen",
        "tokenizer": {
            "identity_sha256": VOCAB_ATTESTATION["tokenizer_identity_sha256"]
        },
        "model": {"identity_sha256": VOCAB_ATTESTATION["model_identity_sha256"]},
        "model_vocab_size": MODEL_VOCAB_SIZE,
        "coordinate_vocabulary": {"token_id_start": 0, "token_id_end_exclusive": 999},
    }
    content["receipt_digest"] = sha256_json(content)
    path = tmp_path / "bad-coord-runtime-identity.json"
    _write_json(path, content)
    args = _contract_args(planned, tmp_path, runtime_identity_path=path)
    with pytest.raises(
        sut.FixedBudgetScoringError, match="coordinate/model vocabulary"
    ):
        sut.run(args)


def test_semantic_core_round_trip_is_validated_only_when_present() -> None:
    # RULES_TEMPLATE (this suite's test_fixture-mode document) carries no
    # semantic_core at all, so the specialized rules document produced by
    # every `planned` fixture above never exercises this check -- proving
    # here, directly, that it fires when the field *is* present and stale.
    document = {
        "semantic_core": {"schema_version": "bogus", "payload": {}, "sha256": "0" * 64}
    }
    with pytest.raises(ValueError):
        sut._validate_semantic_core_payload(document)


def test_admit_output_dir_empty_is_a_no_op(tmp_path: Path) -> None:
    sut._admit_output_dir(
        jsonl_path=tmp_path / "a.jsonl",
        receipt_path=tmp_path / "a-receipt.json",
        force=False,
    )


def test_admit_output_dir_sealed_refuses_even_with_force(tmp_path: Path) -> None:
    jsonl_path = tmp_path / sut.OUTPUT_JSONL_NAME
    receipt_path = tmp_path / sut.RECEIPT_NAME
    jsonl_path.write_text('{"a": 1}\n', encoding="utf-8")
    receipt_document = {
        "schema_version": sut.RECEIPT_SCHEMA_VERSION,
        "runtime_execution_status": sut.LIVE_SCORING_SEALED_STATUS,
        "output_artifacts": {
            "fixed_budget_scores": {"sha256": sha256_file(jsonl_path)}
        },
    }
    receipt_path.write_text(json.dumps(receipt_document), encoding="utf-8")

    with pytest.raises(sut.FixedBudgetScoringError, match="sealed"):
        sut._admit_output_dir(
            jsonl_path=jsonl_path, receipt_path=receipt_path, force=False
        )
    with pytest.raises(sut.FixedBudgetScoringError, match="sealed"):
        sut._admit_output_dir(
            jsonl_path=jsonl_path, receipt_path=receipt_path, force=True
        )


def test_admit_output_dir_incomplete_requires_force_then_clears(tmp_path: Path) -> None:
    jsonl_path = tmp_path / sut.OUTPUT_JSONL_NAME
    receipt_path = tmp_path / sut.RECEIPT_NAME
    jsonl_path.write_text("stray partial output\n", encoding="utf-8")

    with pytest.raises(sut.FixedBudgetScoringError, match="incomplete"):
        sut._admit_output_dir(
            jsonl_path=jsonl_path, receipt_path=receipt_path, force=False
        )
    assert jsonl_path.exists()

    sut._admit_output_dir(jsonl_path=jsonl_path, receipt_path=receipt_path, force=True)
    assert not jsonl_path.exists()
    assert not receipt_path.exists()


# ---------------------------------------------------------------------------
# Live-math primitives: deterministic fake forward closures, no model/GPU.
# ---------------------------------------------------------------------------


def _deterministic_logits_row(
    context: tuple[int, ...], vocab_size: int
) -> torch.Tensor:
    import zlib

    seed = zlib.crc32(json.dumps(list(context)).encode("utf-8")) & 0xFFFFFFFF
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(vocab_size, generator=generator)


def test_materialized_image_identity_uses_project_panel_semantics() -> None:
    panel_sha256 = "1" * 64
    image_id = "img-fixture"
    ledger_identity = sha256_json({"image_id": image_id, "panel_sha256": panel_sha256})
    admission = sut.validate_materialized_image_identity(
        image_id=image_id,
        ledger_rows=[{"image_identity": ledger_identity}],
        source_panel_sha256=panel_sha256,
        planned_image_content_sha256="2" * 64,
        executed_media_sha256=["3" * 64],
        expected_materialization_count=1,
    )
    assert admission["status"] == "passed_before_coordinate_score"
    assert admission["ledger_image_identity"] == ledger_identity
    assert admission["planned_image_content_sha256"] == "2" * 64
    assert admission["planned_image_content_hash_domain"] == "raw_file_bytes_sha256"
    assert admission["canonical_executed_media_sha256"] == "3" * 64
    assert (
        admission["executed_media_hash_domain"]
        == "coordexp_rgb8_pixels_v1_sha256"
    )
    assert admission["materialization_count"] == 1
    assert admission["expected_materialization_count"] == 1


def test_materialized_image_identity_rejects_ledger_mismatch() -> None:
    panel_sha256 = "1" * 64
    image_id = "img-fixture"
    coordinate_score_started = False
    with pytest.raises(
        sut.FixedBudgetScoringError,
        match="ledger image_identity",
    ):
        sut.validate_materialized_image_identity(
            image_id=image_id,
            ledger_rows=[{"image_identity": "3" * 64}],
            source_panel_sha256=panel_sha256,
            planned_image_content_sha256="2" * 64,
            executed_media_sha256=["4" * 64],
            expected_materialization_count=1,
        )
        coordinate_score_started = True
    assert coordinate_score_started is False


@pytest.mark.parametrize(
    ("executed", "expected_count", "error_match"),
    [
        ([], 1, "executed RGB identity is missing"),
        (["A" * 64], 1, "lowercase SHA-256"),
        (["not-a-digest"], 1, "lowercase SHA-256"),
        (["3" * 64, "4" * 64], 2, "not unanimous"),
    ],
)
def test_materialized_image_identity_mismatch_rejected_before_coordinate_score(
    executed: list[str],
    expected_count: int,
    error_match: str,
) -> None:
    panel_sha256 = "1" * 64
    image_id = "img-fixture"
    ledger_identity = sha256_json(
        {"image_id": image_id, "panel_sha256": panel_sha256}
    )
    coordinate_score_started = False
    with pytest.raises(
        sut.FixedBudgetScoringError,
        match=error_match,
    ):
        sut.validate_materialized_image_identity(
            image_id=image_id,
            ledger_rows=[{"image_identity": ledger_identity}],
            source_panel_sha256=panel_sha256,
            planned_image_content_sha256="2" * 64,
            executed_media_sha256=executed,
            expected_materialization_count=expected_count,
        )
        coordinate_score_started = True
    assert coordinate_score_started is False


def test_repeated_materialized_image_identity_must_match_initial_rgb() -> None:
    panel_sha256 = "1" * 64
    image_id = "img-fixture"
    ledger_identity = sha256_json({"image_id": image_id, "panel_sha256": panel_sha256})

    with pytest.raises(
        sut.FixedBudgetScoringError,
        match="repeated materialization changed canonical executed RGB identity",
    ):
        sut.validate_materialized_image_identity(
            image_id=image_id,
            ledger_rows=[{"image_identity": ledger_identity}],
            source_panel_sha256=panel_sha256,
            planned_image_content_sha256="2" * 64,
            executed_media_sha256=["4" * 64, "4" * 64],
            expected_materialization_count=2,
            expected_executed_media_sha256="3" * 64,
        )


@pytest.mark.parametrize(
    ("executed", "expected_count"),
    [
        (["3" * 64, "3" * 64], 1),
        (["3" * 64, "3" * 64], 3),
    ],
)
def test_materialized_image_identity_rejects_wrong_materialization_count(
    executed: list[str],
    expected_count: int,
) -> None:
    panel_sha256 = "1" * 64
    image_id = "img-fixture"
    ledger_identity = sha256_json({"image_id": image_id, "panel_sha256": panel_sha256})

    with pytest.raises(
        sut.FixedBudgetScoringError,
        match="materialization count",
    ):
        sut.validate_materialized_image_identity(
            image_id=image_id,
            ledger_rows=[{"image_identity": ledger_identity}],
            source_panel_sha256=panel_sha256,
            planned_image_content_sha256="2" * 64,
            executed_media_sha256=executed,
            expected_materialization_count=expected_count,
        )


# Small, self-contained coordinate domain for the live-math fixtures below
# (distinct from the big COORD_TOKEN_START/END used by the planner rules
# fixture above): must be a subset of the fake 64-entry vocabulary.
_FAKE_COORD_START = 10
_FAKE_COORD_END = 20


def _fake_ledger_row(context_id: str = "ctx:fixture") -> dict[str, Any]:
    return {
        "diagnostic_owner_id": "diagnostic:gt:fixture:0",
        "gt_owner_id": "gt:fixture:0",
        "image_id": "img-fixture",
        "context_tokens": {"token_ids": [1, 2, 3], "token_ids_sha256": "e" * 64},
    }


def _fake_rules(*, rules_digest: str = "r" * 64) -> Any:
    return scorer.DecisionRules(
        rules_digest=rules_digest,
        schema_tokens={
            "coordinate_token_id_start": _FAKE_COORD_START,
            "coordinate_token_id_end_exclusive": _FAKE_COORD_END,
        },
        owner_canonical_description={},
        foil_set_digests={},
        model_vocab_size=64,
        numeric_tolerance=1e-6,
    )


def _fake_candidate(
    candidate_id: str, coord_token_ids: list[int], **overrides: Any
) -> dict[str, Any]:
    base = {
        "candidate_id": candidate_id,
        "owner_context_id": "ctx:fixture",
        "rung": "L0",
        "region": "target_strict",
        "population": "target",
        "is_control": False,
        "control_kind": None,
        "matched_control_group": "gt:fixture:0",
        "family_id": "near_gt_micro",
        "iou_to_target": 1.0,
        "source_digest": "s" * 64,
        "coord_token_ids": coord_token_ids,
        "candidate_neighborhood_member": False,
        "candidate_neighborhood_id": None,
        "mechanism_decision_rules_sha256": "m" * 64,
        "other_owner_gt_owner_id": None,
        "other_owner_selection_trace": None,
    }
    base.update(overrides)
    return base


def _fake_full_reforward_backend(
    prefix: list[int], *, vocab_size: int = 64
) -> tuple[scorer.FullReforwardBackend, list[tuple[int, ...]]]:
    calls: list[tuple[int, ...]] = []

    def full_reforward(token_ids: Sequence[int]) -> torch.Tensor:
        literal = tuple(int(v) for v in token_ids)
        calls.append(literal)
        return _deterministic_logits_row(literal, vocab_size)

    backend = scorer.FullReforwardBackend(
        root_prefix_token_ids=prefix,
        full_reforward=full_reforward,
        context_id="fixture-context",
        group_id="fixture-group",
        progress_every_actual_forwards=0,
    )
    return backend, calls


def _fake_attestation(rules: Any) -> scorer.AttestationContext:
    return scorer.build_attestation_context(
        expected_vocab_size=rules.model_vocab_size,
        tokenizer_identity={"name": "fake"},
        model_identity={"name": "fake"},
        rule_digest=rules.rules_digest,
        runtime_receipt_id="runtime-receipt",
    )


def _score_one_candidate(
    candidate: dict[str, Any], *, prefix: list[int], vocab_size: int = 64
) -> tuple[dict[str, Any], scorer.FullReforwardBackend, list[tuple[int, ...]]]:
    rules = _fake_rules()
    backend, calls = _fake_full_reforward_backend(prefix, vocab_size=vocab_size)
    scorer._prefetch_backend_suffixes(
        backend, sut._candidate_probe_suffixes(candidate["coord_token_ids"])
    )  # noqa: SLF001
    combined = scorer.score_complete_box_candidate(
        backend=backend,
        prefill_logits=backend.root_logits,
        coord_token_ids=candidate["coord_token_ids"],
        attestation=_fake_attestation(rules),
        running_context_token_ids=prefix,
    )
    row = sut.build_score_row(
        context_id="ctx:fixture",
        candidate=candidate,
        ledger_row=_fake_ledger_row(),
        rules=rules,
        combined=combined,
    )
    return row, backend, calls


def test_scalar_semantics_root_plus_three_literal_reforwards_per_candidate() -> None:
    coord = [10, 11, 12, 13]
    candidate = _fake_candidate("cand-1", coord)
    row, backend, calls = _score_one_candidate(candidate, prefix=[7, 8], vocab_size=64)
    # depth-0 root (x1) + three literal continuations (y1, x2, y2): exactly
    # the four-forward decision channel unit.md requires, never a fifth.
    assert calls == [
        (7, 8),
        (7, 8, coord[0]),
        (7, 8, coord[0], coord[1]),
        (7, 8, coord[0], coord[1], coord[2]),
    ]
    assert row["schema_version"] == sut.SCHEMA_VERSION
    assert row["unit_id"] == sut.UNIT_ID
    assert row["coord_token_ids"] == coord
    assert row["context_id"] == row["owner_context_id"] == "ctx:fixture"
    assert math.isfinite(row["raw_model_logprob"]["complete_box_logprob_sum"])


def test_raw_and_auxiliary_policy_views_are_separated_and_never_mixed() -> None:
    coord = [14, 15, 16, 17]
    candidate = _fake_candidate("cand-2", coord)
    row, _backend, _calls = _score_one_candidate(candidate, prefix=[5])
    raw = row["raw_model_logprob"]
    policy = row["auxiliary_policy_scores"]
    assert set(raw) >= {
        "x1_logprob",
        "y1_logprob",
        "x2_logprob",
        "y2_logprob",
        "complete_box_logprob_sum",
    }
    assert set(policy) == {"rp_1.00", "rp_1.10"}
    for view in policy.values():
        assert set(view) >= {
            "x1_logprob",
            "y1_logprob",
            "x2_logprob",
            "y2_logprob",
            "complete_box_logprob_sum",
        }
    assert row["native_repetition_penalty_stratum"] == 1.0
    # The raw channel and the two policy channels are independently derived
    # from the same logits (repetition penalty != 1.0 nudges the value), so
    # they must not be numerically identical for a non-degenerate context.
    assert (
        raw["complete_box_logprob_sum"] != policy["rp_1.10"]["complete_box_logprob_sum"]
    )


def test_build_score_row_rejects_nonfinite_scores() -> None:
    candidate = _fake_candidate("cand-nan", [_FAKE_COORD_START] * 4)
    combined = {
        "raw": {
            "x1_logprob": float("nan"),
            "y1_logprob": -1.0,
            "x2_logprob": -1.0,
            "y2_logprob": -1.0,
            "complete_box_logprob_sum": float("nan"),
        },
        "auxiliary_policy": {},
    }
    with pytest.raises(sut.FixedBudgetScoringError, match="non-finite"):
        sut.build_score_row(
            context_id="ctx:fixture",
            candidate=candidate,
            ledger_row=_fake_ledger_row(),
            rules=_fake_rules(),
            combined=combined,
        )


def test_scorer_never_reuses_a_predecessor_score_every_candidate_is_scored_fresh() -> (
    None
):
    """Two candidates sharing one predecessor_candidate_id but distinct coord
    tokens/prefixes must still be scored independently from their own
    literal continuations -- never short-circuited by that shared field."""

    shared_predecessor_id = "cand:predecessor:shared:0:target"
    coord_a = [10, 11, 12, 13]
    coord_b = [14, 15, 16, 17]
    candidate_a = _fake_candidate(
        "cand-a", coord_a, predecessor_candidate_id=shared_predecessor_id
    )
    candidate_b = _fake_candidate(
        "cand-b", coord_b, predecessor_candidate_id=shared_predecessor_id
    )

    row_a, _backend_a, calls_a = _score_one_candidate(candidate_a, prefix=[9, 10])
    row_b, _backend_b, calls_b = _score_one_candidate(candidate_b, prefix=[9, 10])

    assert calls_a != calls_b
    assert (
        row_a["raw_model_logprob"]["complete_box_logprob_sum"]
        != row_b["raw_model_logprob"]["complete_box_logprob_sum"]
    )
    assert "predecessor_candidate_id" not in row_a
    assert "predecessor_candidate_id" not in row_b
    # Reproducing the exact same literal continuation independently must
    # reproduce the exact same score: proof this is a pure function of
    # (prefix, coord tokens), never of any cached/reused predecessor value.
    row_a_again, _backend, _calls = _score_one_candidate(candidate_a, prefix=[9, 10])
    assert (
        row_a_again["raw_model_logprob"]["complete_box_logprob_sum"]
        == row_a["raw_model_logprob"]["complete_box_logprob_sum"]
    )


def test_candidate_probe_suffixes_matches_predecessor_convention() -> None:
    coord = [11, 22, 33, 44]
    assert sut._candidate_probe_suffixes(coord) == [[11], [11, 22], [11, 22, 33]]


# ---------------------------------------------------------------------------
# Batch parity: pass admits the requested batch; drift falls back to scalar.
# ---------------------------------------------------------------------------


def test_batch_parity_pass_and_fallback_reuses_predecessor_gate() -> None:
    prefix = [1, 2]
    coord = [
        COORD_TOKEN_START + 5,
        COORD_TOKEN_START + 6,
        COORD_TOKEN_START + 7,
        COORD_TOKEN_START + 8,
    ]

    def scalar(token_ids: Sequence[int]) -> torch.Tensor:
        return _deterministic_logits_row(tuple(token_ids), 64)

    def matching(token_rows: Sequence[Sequence[int]]) -> torch.Tensor:
        return torch.stack([scalar(row) for row in token_rows])

    passed = scorer.run_batched_reforward_parity_gate(
        prefix_token_ids=prefix,
        coordinate_token_ids=coord,
        coordinate_token_id_start=0,
        coordinate_token_id_end_exclusive=64,
        full_reforward=scalar,
        batched_full_reforward=matching,
        requested_batch_size=8,
    )
    assert passed["status"] == "passed"
    assert passed["effective_batch_size"] == 8
    assert passed["coordinate_logprob_max_abs_diff_tolerance"] == pytest.approx(1e-3)

    def drifted(token_rows: Sequence[Sequence[int]]) -> torch.Tensor:
        rows = matching(token_rows)
        rows[:, 0] += 5.0
        return rows

    failed = scorer.run_batched_reforward_parity_gate(
        prefix_token_ids=prefix,
        coordinate_token_ids=coord,
        coordinate_token_id_start=0,
        coordinate_token_id_end_exclusive=64,
        full_reforward=scalar,
        batched_full_reforward=drifted,
        requested_batch_size=8,
    )
    assert failed["status"] == "failed_scalar_fallback_required"
    assert failed["effective_batch_size"] == 1


def test_admission_content_reconstructs_via_predecessor_vocabulary() -> None:
    """The receipt's scoring_backend_admission is expressed honestly in the
    predecessor's own status vocabulary (never claiming a cache path this
    scorer never attempts) and must reconstruct through the predecessor's
    own selector, exactly like a real full-reforward-only run would."""

    selection = scorer.select_scoring_backend_from_parity(
        {"status": "failed", "admission_policy": None}
    )
    assert selection["selected_backend"] == scorer.FULL_REFORWARD_SCORING_BACKEND
    assert selection["cache_enabled"] is False
    assert selection["use_cache"] is False
    assert selection["fallback_trigger"] == scorer.PARITY_FAILURE_FALLBACK_TRIGGER


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_full_reforward_batch_size_requires_positive_integer() -> None:
    parser = sut.build_parser()
    base = [
        "--planner-receipt",
        "p.json",
        "--fn-mechanism-registry",
        "r.json",
        "--owner-context-ledger",
        "l.jsonl",
        "--decision-rules",
        "d.json",
        "--mechanism-decision-rules",
        "m.json",
        "--fixed-budget-candidates",
        "c.jsonl",
        "--runtime-identity",
        "i.json",
        "--infer-config",
        "infer.yaml",
        "--source-jsonl",
        "source.jsonl",
    ]
    with pytest.raises(SystemExit):
        parser.parse_args([*base, "--full-reforward-batch-size", "0"])
    parsed = parser.parse_args([*base, "--full-reforward-batch-size", "4"])
    assert parsed.full_reforward_batch_size == 4
    assert parsed.force is False
    assert parsed.validate_contract_only is False
    assert parsed.include_rung is None
    selected = parser.parse_args(
        [
            *base,
            "--include-rung",
            "L0",
            "--include-rung",
            "L1",
        ]
    )
    assert selected.include_rung == ["L0", "L1"]


def test_cli_requires_all_five_static_contract_inputs() -> None:
    parser = sut.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--decision-rules", "d.json"])
