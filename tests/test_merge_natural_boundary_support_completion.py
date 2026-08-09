from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.research import merge_natural_boundary_support_completion as merge
from scripts.research import run_natural_boundary_support_completion as execution


PLAN_PATH = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-06-natural-boundary-routing-history-replication/support-completion-plan-v1/plan.json")
CENSUS_PATH = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-06-natural-boundary-routing-history-replication/cpu-census-v2/admission-census.json")
PRIOR_PATH = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-05-static-dynamic-owner-interface-crossover/ledgers/s-step2444-final-support.json")


@pytest.fixture(scope="module")
def plan() -> dict:
    return json.loads(PLAN_PATH.read_text())


@pytest.fixture(scope="module")
def shards(plan: dict) -> dict[int, dict]:
    result = {}
    for shard_index in range(8):
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
        result[shard_index] = {
            "schema_version": execution.RECEIPT_SCHEMA_VERSION,
            "status": "completed",
            "unit_id": execution.UNIT_ID,
            "plan_content_sha256": plan["plan_content_sha256"],
            "shard_index": shard_index,
            "num_shards": 8,
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
    return result


def test_merge_eight_complete_shards(plan, shards):
    result = merge.merge_support_receipts(
        plan,
        shards,
        input_plan_sha256=merge.EXPECTED_PLAN_SHA256,
        prior_support=PRIOR_PATH,
        test_only=True,
    )
    assert result["ledger"]["record_count"] == 220
    assert result["ledger"]["image_count"] == 13
    assert result["receipt"]["scalar_equivalent_forward_count"] == 77428


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "wrong_shard", "count", "hash", "plan_hash"])
def test_merge_rejects_partition_or_binding_drift(plan, shards, mutation):
    candidate = {index: copy.deepcopy(receipt) for index, receipt in shards.items()}
    if mutation == "missing":
        candidate.pop(7)
    elif mutation == "duplicate":
        observations = candidate[0]["observations"]
        observations[1] = copy.deepcopy(observations[0])
    elif mutation == "wrong_shard":
        candidate[0]["shard_index"] = 7
    elif mutation == "count":
        candidate[0]["expected_scalar_forward_count"] += 1
    elif mutation == "hash":
        candidate[0]["observations"][0]["candidate_scores_sha256"] = "0" * 64
    elif mutation == "plan_hash":
        pass
    with pytest.raises(merge.MergeContractError):
        merge.merge_support_receipts(
            plan,
            candidate,
            input_plan_sha256="0" * 64 if mutation == "plan_hash" else merge.EXPECTED_PLAN_SHA256,
            prior_support=PRIOR_PATH,
            test_only=True,
        )


def test_direct_merge_rejects_resealed_nonsealed_plan(plan, shards):
    mutated = copy.deepcopy(plan)
    mutated["status"] = "resealed_for_test"
    mutated_body = dict(mutated)
    mutated_body.pop("plan_content_sha256", None)
    mutated["plan_content_sha256"] = merge.sha256_json(mutated_body)
    with pytest.raises(merge.MergeContractError):
        merge.merge_support_receipts(
            mutated,
            shards,
            input_plan_sha256=merge.EXPECTED_PLAN_SHA256,
            input_census_binding={"revision": "census-v2", "sha256": "d" * 64},
            test_only=False,
        )
