from __future__ import annotations

import copy
from pathlib import Path

import pytest

from scripts.research import plan_natural_boundary_owner_support_completion as planner


@pytest.fixture(scope="module")
def sealed_plan() -> dict[str, object]:
    """Use the immutable local S lineage; this remains CPU-only."""

    return planner.build_support_completion_plan(
        source_panel=planner.DEFAULT_SOURCE_PANEL,
        derived_panel=planner.DEFAULT_DERIVED_PANEL,
        derived_receipt=planner.DEFAULT_DERIVED_RECEIPT,
        h0_ledger=planner.DEFAULT_H0_LEDGER,
        support_ledger=planner.DEFAULT_SUPPORT_LEDGER,
        candidate_batch_size=16,
        num_shards=8,
        strict_contract=True,
    )


def test_missing_native_fn_scope_and_calibration_reuse_are_exact(sealed_plan: dict[str, object]) -> None:
    scope = sealed_plan["scope"]
    assert scope["native_fn_denominator"] == 220
    assert scope["support_measured_fn_retained"] == 20
    assert scope["support_unassessed_fn"] == 200
    assert scope["support_completion_candidates"] == 200
    assert scope["native_tp_calibration_complete"] == 172
    assert scope["native_tp_other_not_scored"] == 160

    reuse = sealed_plan["calibration_reuse"]
    assert reuse["reused"] is True
    assert reuse["recomputed"] is False
    assert reuse["observation_count"] == 172
    assert reuse["excluded"] == 0
    assert reuse["calibration_sha256"] == sealed_plan["support_lineage"]["calibration_sha256"]

    contexts = sealed_plan["contexts"]
    assert len(contexts) == 200
    assert all(row["native_fn"] is True for row in contexts)
    assert all(row["native_tp"] is False for row in contexts)
    assert len({row["gt_owner_id"] for row in contexts}) == 200


def test_content_stable_eight_shards_and_safe_batch_estimate(sealed_plan: dict[str, object]) -> None:
    work = sealed_plan["work"]
    assert work["shard_count"] == 8
    assert tuple(row["context_count"] for row in work["per_shard"]) == planner.EXPECTED_SHARD_CONTEXTS
    assert tuple(row["scalar_equivalent_forward_count"] for row in work["per_shard"]) == planner.EXPECTED_SHARD_FORWARDS
    assert work["scalar_equivalent_forward_count"] == planner.EXPECTED_SCALAR_FORWARDS
    assert work["batching_admitted"] is False
    assert work["safe_batched_forward_estimate"] == 4840
    assert tuple(row["safe_batched_forward_estimate"] for row in work["per_shard"]) == (
        423,
        439,
        739,
        513,
        616,
        532,
        867,
        714,
    )

    contexts = sealed_plan["contexts"]
    assert [row["context_plan_position"] for row in contexts] == list(range(200))
    for row in contexts:
        assert row["context_id"] == planner.context_id_for_stable_key(row["stable_key"])
        assert row["shard_index"] == planner.shard_index_for_stable_key(row["stable_key"], 8)


def test_plan_validation_rejects_candidate_identity_drift(sealed_plan: dict[str, object]) -> None:
    tampered = copy.deepcopy(sealed_plan)
    tampered["contexts"][0]["candidate_ids"][0] = "cand:tampered"
    tampered.pop("plan_content_sha256")
    tampered["plan_content_sha256"] = planner.sha256_json(tampered)
    with pytest.raises(planner.SupportCompletionPlanError, match="candidate/target-row identity"):
        planner.validate_plan(tampered)


def test_materialization_is_canonical_and_immutable(
    sealed_plan: dict[str, object], tmp_path: Path
) -> None:
    plan_path = tmp_path / "plan.json"
    receipt_path = tmp_path / "receipt.json"
    first = planner.materialize_plan_and_receipt(
        sealed_plan,
        output_path=plan_path,
        receipt_path=receipt_path,
    )
    second = planner.materialize_plan_and_receipt(
        sealed_plan,
        output_path=plan_path,
        receipt_path=receipt_path,
    )
    assert first["plan_sha256"] == second["plan_sha256"]
    assert first["receipt_sha256"] == second["receipt_sha256"]
    assert first["receipt"]["measured_wall_time_seconds"] is None
    assert first["receipt"]["realized_scalar_forward_count"] is None

    tampered = copy.deepcopy(sealed_plan)
    tampered["contexts"][0]["status"] = "executed"
    with pytest.raises(planner.SupportCompletionPlanError, match="plan content hash mismatch"):
        planner.materialize_plan_and_receipt(
            tampered,
            output_path=plan_path,
            receipt_path=tmp_path / "tampered-receipt.json",
        )
