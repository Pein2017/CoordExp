"""Research-surface contract smoke tests for the support completion hand-off."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.research import merge_natural_boundary_support_completion as merge
from scripts.research import run_natural_boundary_support_completion as execution
from scripts.research.materialize_natural_boundary_census_v3 import (
    EXPECTED_BASE_CENSUS_SHA256,
    EXPECTED_SUPPORT_LEDGER_SHA256,
    CensusV3ContractError,
    _load_support,
    _old_replay,
    _update_row,
    _geometry_supersession,
    materialize,
    sha256_json,
)
from scripts.research.run_s_natural_boundary_k_n_h_cohort import ARM_ORDER, validate_manifest


SUPPORT_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-08-06-natural-boundary-routing-history-replication"
)
SUPPORT_PLAN_PATH = SUPPORT_ROOT / "support-completion-plan-v1/plan.json"
SUPPORT_CENSUS_PATH = SUPPORT_ROOT / "cpu-census-v2/admission-census.json"
MERGE_V6_PATH = SUPPORT_ROOT / "support-merge-v6/s-step2444-complete-support.json"
GEOMETRY_ROOT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration")
PRIOR_SUPPORT_PATH = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-08-05-static-dynamic-owner-interface-crossover/ledgers/"
    "s-step2444-final-support.json"
)


@pytest.fixture(scope="module")
def support_plan() -> dict:
    return json.loads(SUPPORT_PLAN_PATH.read_text())


def _complete_cpu_shards(plan: dict) -> dict[int, dict]:
    shards: dict[int, dict] = {}
    for shard_index in range(execution.NUM_SHARDS):
        assigned = execution.shard_contexts(plan, shard_index=shard_index)
        observations = []
        for context in assigned:
            scores = {str(candidate): 0.0 for candidate in context["candidate_ids"]}
            observations.append(
                {
                    "context_id": context["context_id"],
                    "status": "measured",
                    "support_features": {"assessed": True, "peak_lift": 4.0, "local_concentration": 5.0},
                    "candidate_score_count": int(context["scalar_equivalent_forward_count"]),
                    "candidate_scores": scores,
                    "candidate_scores_sha256": execution.sha256_json(scores),
                }
            )
        scalar = sum(int(context["scalar_equivalent_forward_count"]) for context in assigned)
        shards[shard_index] = {
            "schema_version": execution.RECEIPT_SCHEMA_VERSION,
            "status": "completed",
            "unit_id": execution.UNIT_ID,
            "plan_content_sha256": plan["plan_content_sha256"],
            "shard_index": shard_index,
            "num_shards": execution.NUM_SHARDS,
            "assigned_context_ids_sha256": execution.sha256_json([str(context["context_id"]) for context in assigned]),
            "assigned_context_count": len(assigned),
            "expected_scalar_forward_count": scalar,
            "realized_scalar_forward_count": scalar,
            "complete_assigned_observations": True,
            "failure_count": 0,
            "failure_log": [],
            "failure_log_content_sha256": execution.sha256_bytes(b""),
            "observations": observations,
        }
    return shards


def _merge_with_prior(plan: dict, prior_support: dict | Path) -> dict:
    return merge.merge_support_receipts(
        plan,
        _complete_cpu_shards(plan),
        input_plan_sha256=merge.EXPECTED_PLAN_SHA256,
        prior_support=prior_support,
        test_only=True,
    )


def test_production_census_binding_and_arm_contract_are_frozen() -> None:
    assert EXPECTED_BASE_CENSUS_SHA256 == "dd1c61abb9acff7f4fc42380365439ee931db604525749f297bbc3e191a26a2e"
    assert ARM_ORDER == (
        "K00", "K01", "K10", "K11", "K12", "K13", "K14T", "K14B",
        "N00", "N01", "N10", "N20", "H00", "H10", "H20",
    )


def test_consumer_rejects_missing_arm_order() -> None:
    malformed = {
        "schema_version": "s_natural_boundary_admitted_event_manifest.v3",
        "status": "sealed",
        "unit_id": "2026-08-06-natural-boundary-routing-history-replication",
        "primary": {"checkpoint": "S", "step": 2444, "substrate": "four-coordinate geo_sorted_xy"},
    }
    try:
        validate_manifest(malformed)
    except ValueError:
        return
    raise AssertionError("consumer accepted a manifest without arm_order")


def test_merge_normalizes_legacy_prior_rule_and_preserves_strict_hashes(support_plan: dict) -> None:
    result = _merge_with_prior(support_plan, PRIOR_SUPPORT_PATH)
    ledger = result["ledger"]
    support_rule = support_plan["support_lineage"]["support_rule"]

    assert ledger["record_count"] == 220
    assert all(record["support_rule"] == support_rule for record in ledger["records"])
    assert ledger["records_sha256"] == merge.sha256_json(ledger["records"])
    assert ledger["content_sha256"] == merge.sha256_json(
        {key: value for key, value in ledger.items() if key != "content_sha256"}
    )
    assert result["receipt"]["ledger_content_sha256"] == ledger["content_sha256"]
    assert result["receipt"]["self_sha256"] == merge.sha256_json(
        {key: value for key, value in result["receipt"].items() if key != "self_sha256"}
    )

    strict_ledger, _ = _load_support(
        ledger,
        calibration=support_plan["calibration_reuse"]["calibration"],
        support_rule=support_rule,
    )
    assert strict_ledger == ledger

    materialized = materialize(
        SUPPORT_CENSUS_PATH,
        ledger,
        plan_source=SUPPORT_PLAN_PATH,
        test_only=True,
    )
    assert len(materialized["census"]["rows"]) == 784


def test_merge_rejects_conflicting_legacy_prior_rule(support_plan: dict) -> None:
    prior_support = json.loads(PRIOR_SUPPORT_PATH.read_text())
    conflicting = next(record for record in prior_support["records"] if record.get("native_fn") is True)
    conflicting["support_rule"] = {"criterion_id": "conflicting_legacy_rule"}

    plan = copy.deepcopy(support_plan)
    plan["support_lineage"]["file_sha256"] = merge.sha256_json(prior_support)
    plan_body = dict(plan)
    plan_body.pop("plan_content_sha256", None)
    plan["plan_content_sha256"] = merge.sha256_json(plan_body)

    with pytest.raises(merge.MergeContractError, match="support rule differs from sealed plan"):
        _merge_with_prior(plan, prior_support)


def _geometry_regions() -> dict:
    def receipt(cells: list[int], *, zero: bool = False) -> dict:
        return {
            "cell_indices": cells,
            "fractional_weights": [
                {"cell_index": cell, "overlap_fraction": 0.0 if zero else 1.0}
                for cell in cells
            ],
        }

    cells = {"a_exclusive": [1], "b_exclusive": [2], "background": [3]}
    return {
        "image_cell_regions": cells,
        "image_cell_region_receipts": {
            "a_exclusive": receipt(cells["a_exclusive"]),
            "b_exclusive": receipt(cells["b_exclusive"]),
            "background": receipt(cells["background"], zero=True),
        },
        "covered_owner_ids_at_b_boundary": ["gt:1:0"],
        "owner_region_owner_ids": ["gt:1:0", "gt:1:1"],
        "geometry_sha256": "a" * 64,
    }


@pytest.mark.parametrize("mutation", ["unexplained_added", "removed_still_present"])
def test_geometry_supersession_rejects_unexplained_owner_drift(mutation: str) -> None:
    old = _geometry_regions()
    new = copy.deepcopy(old)
    old_verified = {"gt:1:1": {"verified_support": True}}
    new_verified = {"gt:1:1": {"image_id": 1, "verified_support": True, "native_fn": True}}
    if mutation == "unexplained_added":
        new["owner_region_owner_ids"].append("gt:1:2")
    else:
        old["owner_region_owner_ids"].append("gt:1:2")
        old_verified["gt:1:2"] = {"verified_support": True}
        new_verified["gt:1:2"] = {"image_id": 1, "verified_support": True, "native_fn": True}
    with pytest.raises(CensusV3ContractError, match="unexplained"):
        _geometry_supersession(
            {"gt_owner_id": "gt:1:1", "image_id": 1, "covered_A_owner_id": "gt:1:0", "covered_owner_ids": ["gt:1:0"]},
            stored_old=old,
            recomputed_old=old,
            new=new,
            old_bank={},
            new_bank={},
            old_verified=old_verified,
            new_verified_native_fn=new_verified,
            old_replay={},
        )


def test_geometry_supersession_rejects_old_recompute_mismatch() -> None:
    old = _geometry_regions()
    recomputed = copy.deepcopy(old)
    recomputed["image_cell_regions"]["a_exclusive"] = []
    recomputed["image_cell_region_receipts"]["a_exclusive"] = {
        "cell_indices": [], "fractional_weights": []
    }
    with pytest.raises(CensusV3ContractError, match="old geometry recomputation"):
        _geometry_supersession(
            {"gt_owner_id": "gt:1:1", "image_id": 1, "covered_A_owner_id": "gt:1:0", "covered_owner_ids": ["gt:1:0"]},
            stored_old=old,
            recomputed_old=recomputed,
            new=old,
            old_bank={},
            new_bank={},
            old_verified={"gt:1:1": {"verified_support": True}},
            new_verified_native_fn={"gt:1:1": {"image_id": 1, "verified_support": True, "native_fn": True}},
            old_replay={},
        )


def test_old_replay_rejects_unbound_fixed_target() -> None:
    row = {"gt_owner_id": "gt:1:1", "image_id": 1, "exact_prefix_sha256": "a" * 64}
    with pytest.raises(CensusV3ContractError, match="identity drifted"):
        _old_replay(
            row,
            target_record={"gt_owner_id": "gt:1:2", "image_id": 1, "exact_prefix_sha256": "a" * 64},
            target_source={"sha256": "b" * 64},
            target_population="complete_native_fn_bank",
            injection_required=True,
            old_bank={"raw_sha256": "c" * 64},
            recomputed_old={},
        )


def test_old_replay_rejects_mutated_adapted_b_binding() -> None:
    row = {"gt_owner_id": "gt:1:1", "image_id": 1, "exact_prefix_sha256": "a" * 64}
    with pytest.raises(CensusV3ContractError, match="B binding drifted"):
        _old_replay(
            row,
            target_record=dict(row),
            target_source={"sha256": "b" * 64},
            target_population="complete_native_fn_bank",
            injection_required=True,
            old_bank={"raw_sha256": "c" * 64},
            recomputed_old={"b_support_binding": {"gt_owner_id": "gt:1:2"}},
        )


def test_production_rejects_support_merge_raw_sha_drift(support_plan: dict, tmp_path: Path) -> None:
    copied = tmp_path / "complete-support.json"
    copied.write_bytes(MERGE_V6_PATH.read_bytes() + b"\n")
    with pytest.raises(CensusV3ContractError, match="raw SHA-256"):
        materialize(SUPPORT_CENSUS_PATH, copied, plan_source=SUPPORT_PLAN_PATH)


def test_real_merge_v6_geometry_supersession_contract() -> None:
    paths = (
        MERGE_V6_PATH,
        SUPPORT_PLAN_PATH,
        SUPPORT_CENSUS_PATH,
        GEOMETRY_ROOT / "2026-08-04-sorted-prospective-13-image-panel-admission/evaluation-inputs/human-refined-13.coord.jsonl",
        GEOMETRY_ROOT / "2026-08-05-static-dynamic-owner-interface-crossover/inputs/human-refined-13.geo_sorted_xy.coord.jsonl",
        GEOMETRY_ROOT / "2026-08-05-static-dynamic-owner-interface-crossover/inputs/human-refined-13.geo_sorted_xy.coord.receipt.json",
        GEOMETRY_ROOT / "2026-08-05-static-dynamic-owner-interface-crossover/ledgers/s-step2444-native-h0.json",
    )
    if not all(path.is_file() for path in paths):
        pytest.skip("sealed merge-v6 geometry inputs are unavailable")
    result = materialize(
        SUPPORT_CENSUS_PATH,
        MERGE_V6_PATH,
        plan_source=SUPPORT_PLAN_PATH,
        panel_source=paths[3],
        derived_panel_source=paths[4],
        derived_receipt_source=paths[5],
        h0_source=paths[6],
        test_only=True,
    )
    supersession = result["receipt"]["geometry_supersession"]
    assert EXPECTED_SUPPORT_LEDGER_SHA256 == hashlib.sha256(MERGE_V6_PATH.read_bytes()).hexdigest()
    assert (result["manifest"]["event_count"], result["manifest"]["image_count"]) == (11, 8)
    assert result["manifest"]["dynamic_only"]["count"] == 0
    assert supersession["evaluated_event_count"] == 11
    assert len(supersession["evaluated_event_image_ids"]) == 8
    assert supersession["effective_owner_bank_changed_event_count"] == 8
    assert supersession["core_region_changed_event_count"] == 7
    assert supersession["competitor_changed_event_count"] == 6
    receipts = result["receipt"]["geometry_supersession_events"]
    assert result["receipt"]["geometry_supersession_event_count"] == 11
    assert result["receipt"]["geometry_supersession_events_sha256"] == sha256_json(receipts)
    assert sum(item["old_replay"]["fixed_target_operand"]["injection_required"] for item in receipts) == 10
    assert sum(not item["old_replay"]["fixed_target_operand"]["injection_required"] for item in receipts) == 1
    merge_records = {record["gt_owner_id"]: record for record in json.loads(MERGE_V6_PATH.read_text())["records"]}
    for item in receipts:
        fixed_target = item["old_replay"]["fixed_target_operand"]
        assert fixed_target["role"] == "fixed_pair_target_B"
        assert fixed_target["included_in_old_competitor_population"] is False
        assert item["competitor"]["after_owner_id"] != fixed_target["target_owner_id"]
        if fixed_target["injection_required"]:
            target = merge_records[fixed_target["target_owner_id"]]
            assert fixed_target["source_ledger"]["raw_sha256"] == EXPECTED_SUPPORT_LEDGER_SHA256
            assert fixed_target["source_record_semantic_sha256"] == sha256_json(target)
            assert len(fixed_target["adapted_b_binding_record_sha256"]) == 64
        else:
            assert fixed_target["source_ledger"]["raw_sha256"] == item["old_bank"]["raw_sha256"]
    assert all("geometry_superseded" in row and "geometry_supersession" in row for row in result["census"]["rows"] if row.get("geometry_supersession"))
    k13 = next(event for event in result["manifest"]["events"] if event["owner_refs"]["gt_owner_id"] == "gt:5001:15")
    assert k13["same_class_competitor_owner_id"] is None
    assert k13["geometry_supersession"]["competitor"]["before_owner_id"] is not None
    assert k13["geometry_supersession"]["competitor"]["before_geometry_applicability"] == "applicable"
    assert k13["geometry_supersession"]["competitor"]["after_geometry_applicability"] == "not_applicable"
    assert k13["geometry_supersession"]["competitor"]["historical_execution_status"] == "retired_historical_diagnostic"
    sampled = next(event for event in result["manifest"]["events"] if event["owner_refs"]["gt_owner_id"] == "gt:2299:23")
    assert sampled["geometry_supersession"]["competitor"]["before_owner_id"] == "gt:2299:2"
    assert sampled["geometry_supersession"]["competitor"]["before_geometry_applicability"] == "applicable"
    assert sampled["geometry_supersession"]["competitor"]["historical_execution_status"] == "not_executed"


def _valid_row_record() -> tuple[dict, dict]:
    prefix = [151646, 151647, 151648, 151649]
    import scripts.research.materialize_natural_boundary_census_v3 as m

    digest = m.sha256_json(prefix)
    row = {"checkpoint": "S", "gt_owner_id": "gt:1584:0", "image_id": 1584, "native_fn": True, "strict_complete_row": False, "natural_boundary": 1, "natural_boundary_valid": True, "covered_owner_ids": ["gt:1584:1"], "exact_prefix_sha256": digest}
    record = {**row, "exact_prefix_token_ids": prefix, "verified_support": True, "support_features": {"assessed": True}}
    return row, record


@pytest.mark.parametrize("field", ["checkpoint", "natural_boundary", "covered_owner_ids", "exact_prefix_sha256"])
def test_update_row_rejects_malformed_identity(field: str) -> None:
    row, record = _valid_row_record()
    record[field] = "drift" if field != "natural_boundary" else 99
    with pytest.raises(CensusV3ContractError):
        _update_row(row, record)


def test_update_row_rejects_unassessed_support() -> None:
    row, record = _valid_row_record()
    record["support_features"] = {"assessed": False}
    with pytest.raises(CensusV3ContractError):
        _update_row(row, record)
