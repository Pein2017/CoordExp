from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.research import plan_natural_boundary_owner_support_completion as planner
from scripts.research import run_natural_boundary_support_completion as runner


PLAN_PATH = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-08-06-natural-boundary-routing-history-replication/"
    "support-completion-plan-v1/plan.json"
)
CENSUS_PATH = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-08-06-natural-boundary-routing-history-replication/"
    "cpu-census-v2/admission-census.json"
)


@pytest.fixture(scope="module")
def plan() -> dict[str, object]:
    value = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_plan_driven_validation_seals_s_step2444_and_full_denominator(plan: dict[str, object]) -> None:
    validated, source_info = runner.validate_execution_plan(PLAN_PATH, census_path=CENSUS_PATH)

    assert source_info["sha256"] == runner.sha256_file(PLAN_PATH)
    assert validated["checkpoint"] == "S"
    assert validated["h0_lineage"]["source"]["sha256"] == runner.EXPECTED_H0_SHA256
    assert validated["h0_lineage"]["source"]["path"].endswith("s-step2444-native-h0.json")
    assert len(validated["contexts"]) == 200
    assert sum(row["scalar_equivalent_forward_count"] for row in validated["contexts"]) == 77428
    assert validated["scope"]["native_tp_calibration_complete"] == 172
    assert validated["scope"]["native_tp_other_not_scored"] == 160
    assert all(row["native_fn"] is True and row["native_tp"] is False for row in validated["contexts"])
    census = source_info["census"]
    assert census["file_sha256"] == runner.EXPECTED_CENSUS_FILE_SHA256
    assert census["self_sha256"] == runner.EXPECTED_CENSUS_SELF_SHA256
    assert census["s_owner_ids_sha256"] == validated["scope"]["completion_owner_ids_sha256"]


def test_contract_exposes_pooled_and_shard_local_batch_estimates(plan: dict[str, object]) -> None:
    validated, _ = runner.validate_execution_plan(plan, census_path=CENSUS_PATH)
    summary = runner.contract_summary(validated, shard_index=0, census_path=CENSUS_PATH)

    assert summary["census_s_owner_ids_sha256"] == runner.EXPECTED_CENSUS_OWNER_IDS_SHA256
    assert summary["batch_estimates"] == {
        "candidate_batch_size": 16,
        "pooled_ceiling": 4840,
        "sum_shard_local_ceilings": 4843,
        "batching_admitted": False,
        "status": "estimate_only_exact_history_api_scalar_only",
    }
    assert summary["batching_admitted"] is False


def test_missing_or_tampered_census_cannot_reach_contract_ready(plan: dict[str, object], tmp_path: Path) -> None:
    with pytest.raises(runner.SupportCompletionExecutionError, match="admission census"):
        runner.validate_execution_plan(plan, census_path=tmp_path / "missing-census.json")
    tampered = json.loads(CENSUS_PATH.read_text(encoding="utf-8"))
    tampered["rows"][0]["disposition"] = "support_unassessed"
    tampered_path = tmp_path / "tampered-census.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(runner.SupportCompletionExecutionError, match="file hash"):
        runner.validate_execution_plan(plan, census_path=tampered_path)


def test_shards_use_plan_target_rows_even_if_legacy_registry_is_empty(plan: dict[str, object], monkeypatch: pytest.MonkeyPatch) -> None:
    # The old capture runner reads this registry.  The new driver must not.
    from scripts.research import materialize_static_dynamic_owner_interface_cohort as cohort

    monkeypatch.setattr(cohort, "FROZEN_CANDIDATES", ())
    validated, _ = runner.validate_execution_plan(plan)

    observed = []
    for shard_index in range(8):
        contexts = runner.shard_contexts(validated, shard_index=shard_index)
        observed.extend(contexts)
        assert len(contexts) == validated["work"]["per_shard"][shard_index]["context_count"]
        assert sum(len(row["target_rows"]) for row in contexts) == validated["work"]["per_shard"][shard_index]["scalar_equivalent_forward_count"]
    assert {row["context_id"] for row in observed} == {row["context_id"] for row in validated["contexts"]}
    assert sum(len(row["target_rows"]) for row in observed) == 77428


def test_validation_rejects_tp_context_even_when_plan_content_hash_is_resealed(plan: dict[str, object]) -> None:
    tampered = dict(plan)
    contexts = list(plan["contexts"])
    first = dict(contexts[0])
    first["native_tp"] = True
    contexts[0] = first
    tampered["contexts"] = contexts
    content = dict(tampered)
    content.pop("plan_content_sha256", None)
    tampered["plan_content_sha256"] = planner.sha256_json(content)

    with pytest.raises(runner.SupportCompletionExecutionError, match="unassessed native-FN"):
        runner.validate_execution_plan(tampered)


def test_batching_is_fail_closed_and_device_mapping_is_explicit() -> None:
    with pytest.raises(runner.SupportCompletionExecutionError, match="not admitted"):
        runner.validate_batching_request(candidate_batch_size=16)
    with pytest.raises(runner.SupportCompletionExecutionError, match="separate parity-admitted"):
        runner.validate_batching_request(candidate_batch_size=16, batching_admitted=True)

    mapping = runner.validate_device_mapping(
        shard_index=3,
        num_shards=8,
        shard_device_map={"3": "cuda:5"},
    )
    assert mapping["logical_device"] == "cuda:5"
    assert mapping["mapping_status"] == "validated"
    with pytest.raises(runner.SupportCompletionExecutionError, match="disagrees"):
        runner.validate_device_mapping(
            shard_index=3,
            num_shards=8,
            device="cuda:4",
            shard_device_map={"3": "cuda:5"},
        )


def test_runtime_device_attestation_rejects_silent_cuda_mismatch() -> None:
    mapping = runner.validate_device_mapping(shard_index=0, num_shards=8, device="cuda:0")
    attested = runner.attest_runtime_device(
        {
            "device": "cuda:0",
            "effective_device": "cuda:0",
            "normalized_device": "cuda:0",
            "torch_current_device": "cuda:0",
            "physical_device_id": "4",
        },
        mapping,
    )
    assert attested["status"] == "validated"
    assert attested["assigned_logical_device"] == "cuda:0"
    assert attested["physical_device_id"] == "4"
    with pytest.raises(runner.SupportCompletionExecutionError, match="does not match"):
        runner.attest_runtime_device({"device": "cuda:1"}, mapping)
    with pytest.raises(runner.SupportCompletionExecutionError, match="explicit logical"):
        runner.attest_runtime_device({"device": "cuda:0"}, {"logical_device": None})


def test_execute_shard_requires_caller_bound_single_device_before_open(
    plan: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    validated, _ = runner.validate_execution_plan(plan, census_path=CENSUS_PATH)
    bank = runner.CandidateBank(
        panel=None,
        h0=None,
        physical=(),
        groups={},
        by_id={},
        h0_by_owner={},
        plan_content_sha256=validated["plan_content_sha256"],
    )

    class NeverOpened:
        def __init__(self) -> None:
            self.opened = False

        def open(self) -> None:
            self.opened = True

        def score(self, _owner: object, _candidate: object) -> float:
            raise AssertionError("score must not be reached before device admission")

    scorer = NeverOpened()
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(runner.SupportCompletionExecutionError, match="single-device CUDA_VISIBLE_DEVICES"):
        runner.execute_shard(
            validated,
            shard_index=0,
            scorer=scorer,
            bank=bank,
            device="cuda:0",
            census_path=CENSUS_PATH,
        )
    assert scorer.opened is False


def test_resume_receipt_is_content_hash_bound(plan: dict[str, object], tmp_path: Path) -> None:
    validated, source_info = runner.validate_execution_plan(plan, census_path=CENSUS_PATH)
    assigned = runner.shard_contexts(validated, shard_index=0)
    census_binding = source_info["census"]
    receipt = {
        "schema_version": runner.RECEIPT_SCHEMA_VERSION,
        "plan_content_sha256": validated["plan_content_sha256"],
        "shard_index": 0,
        "num_shards": runner.NUM_SHARDS,
        "census_binding": dict(census_binding),
        "census_file_sha256": census_binding["file_sha256"],
        "census_self_sha256": census_binding["self_sha256"],
        "census_s_owner_ids_sha256": census_binding["s_owner_ids_sha256"],
        "batch_estimates": runner.batch_estimates(validated),
        "assigned_context_ids_sha256": runner.sha256_json([str(row["context_id"]) for row in assigned]),
        "observations": [
            {
                "context_id": row["context_id"],
                "status": "partial",
                "candidate_ids": list(row["candidate_ids"]),
                "candidate_scores": {},
                "candidate_scores_sha256": None,
            }
            for row in assigned
        ],
        "failure_log": [],
        "failure_log_content_sha256": runner.sha256_bytes(b""),
        "expected_scalar_forward_count": sum(int(row["scalar_equivalent_forward_count"]) for row in assigned),
        "support_calibration_sha256": validated["calibration_reuse"]["calibration_sha256"],
    }
    receipt["receipt_content_sha256"] = runner.sha256_json(receipt)
    path = tmp_path / "resume.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    loaded, observations = runner._load_resume_receipt(
        path,
        validated,
        assigned,
        0,
        census_binding=census_binding,
    )
    assert loaded["receipt_content_sha256"] == receipt["receipt_content_sha256"]
    assert len(observations) == len(assigned)

    tampered = dict(receipt)
    tampered["observations"] = list(receipt["observations"])
    tampered["observations"][0] = dict(tampered["observations"][0])
    tampered["observations"][0]["status"] = "measured"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(runner.SupportCompletionExecutionError, match="content hash"):
        runner._load_resume_receipt(
            path,
            validated,
            assigned,
            0,
            census_binding=census_binding,
        )


def test_artifacts_are_immutable_and_failure_log_is_always_materialized(tmp_path: Path) -> None:
    result = {
        "receipt": {
            "schema_version": runner.RECEIPT_SCHEMA_VERSION,
            "status": "incomplete",
            "plan_content_sha256": "a" * 64,
            "shard_index": 0,
            "num_shards": 8,
            "observations": [],
            "failure_log": [],
            "failure_log_content_sha256": runner.sha256_bytes(b""),
        },
        "failures": [],
    }
    first = runner.write_shard_artifacts(
        result,
        receipt_path=tmp_path / "shard-0.receipt.json",
        failure_log_path=tmp_path / "shard-0.failures.jsonl",
    )
    second = runner.write_shard_artifacts(
        result,
        receipt_path=tmp_path / "shard-0.receipt.json",
        failure_log_path=tmp_path / "shard-0.failures.jsonl",
    )
    assert first == second
    assert (tmp_path / "shard-0.failures.jsonl").read_bytes() == b""

    changed = copy.deepcopy(result)
    changed["receipt"]["status"] = "completed"
    with pytest.raises(runner.SupportCompletionExecutionError, match="not identical"):
        runner.write_shard_artifacts(
            changed,
            receipt_path=tmp_path / "shard-0.receipt.json",
            failure_log_path=tmp_path / "shard-0.failures.jsonl",
        )
