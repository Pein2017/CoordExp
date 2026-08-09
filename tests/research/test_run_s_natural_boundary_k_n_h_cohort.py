from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.research.run_s_natural_boundary_k_n_h_cohort import (
    ARM_ORDER,
    CHECKPOINT,
    CohortContractError,
    EXECUTOR_CODE_ROLES,
    STEP,
    SUBSTRATE,
    UNIT_ID,
    run_cohort,
    sha256_json,
    validate_manifest,
)
from scripts.research.materialize_natural_boundary_census_v3 import materialize


def _source_identity(tmp_path: Path, name: str, text: str) -> dict[str, str]:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return {"id": name, "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _manifest(tmp_path: Path, *, event_count: int = 1) -> dict[str, object]:
    thresholds = {"min_support": 1.0, "max_history_rows": 1}
    opener_contract = {
        "status": "runner_resolved",
        "resolver": "serialization_successor_runner",
        "token_name": "<|object_ref_start|>",
        "resolution_semantics": "pre_opener_natural_prefix_ends_before_object_ref_start",
        "contract_sha256": sha256_json({"token_name": "<|object_ref_start|>", "resolver": "serialization_successor_runner"}),
    }
    source_census = {"revision": "census-v3", "sha256": "a" * 64}
    events: list[dict[str, object]] = []
    for index in range(event_count):
        event_id = f"gt:{5001 + index}:15"
        event: dict[str, object] = {
            "event_index": index,
            "event_id": event_id,
            "image_id": 5001 + index,
            "owner_refs": {
                "gt_owner_id": event_id,
                "covered_owner_ids": [f"gt:{5001 + index}:14"],
                "covered_A_owner_id": f"gt:{5001 + index}:14",
                "source_panel_object_index": 15,
                "derived_panel_object_index": 15,
            },
            "checkpoint": CHECKPOINT,
            "step": STEP,
            "substrate": SUBSTRATE,
            "admission": "admitted",
            "eligibility": {
                "admitted": True,
                "rule_id": "synthetic-support-and-geometry-v3",
                "thresholds": thresholds,
                "predicates": {
                    "checkpoint": True,
                    "native_fn": True,
                    "strict_complete_row": True,
                    "natural_boundary_valid": True,
                    "verified_support": True,
                    "eligible_except_support": True,
                    "geometry_launch_eligible": True,
                    "covered_owner_ids_nonempty": True,
                },
            },
            "natural_boundary": {
                "pre_opener_natural": True,
                "opener_seeded": False,
                "opener_injected": False,
                "synthetic_opener_injections": 0,
                "opener_token_id": None,
                "opener_token_contract": opener_contract,
                "prefix_token_ids": [50, 9],
                "prefix_sha256": sha256_json([50, 9]),
                "history_token_ids": [9],
                "history_sha256": sha256_json([9]),
            },
        }
        event["event_sha256"] = sha256_json(event)
        events.append(event)
    document: dict[str, object] = {
        "schema_version": "s_natural_boundary_admitted_event_manifest.v3",
        "status": "sealed",
        "unit_id": UNIT_ID,
        "primary": {"checkpoint": CHECKPOINT, "step": STEP, "substrate": SUBSTRATE},
        "source_census": source_census,
        "panel": _source_identity(tmp_path, "panel.json", "panel\n"),
        "cohort": {
            "id": "synthetic-cohort",
            "frozen_arms": list(ARM_ORDER),
            "sha256": sha256_json({"id": "synthetic-cohort", "frozen_arms": list(ARM_ORDER)}),
        },
        "operator": {
            **_source_identity(tmp_path, "operator.json", "operator\n"),
            "object_ref_start_token_id": 1,
        },
        "backend": _source_identity(tmp_path, "backend.json", "backend\n"),
        "eligibility": {"rule_id": "synthetic-support-and-geometry-v3", "thresholds": thresholds},
        "admission_gate": {
            "minimum_event_count": 3,
            "minimum_image_count": 2,
            "status": "deferred_to_successor_runner",
        },
        "arm_order": list(ARM_ORDER),
        "events": events,
    }
    document["self_sha256"] = sha256_json(document)
    return document


def _arm_result(arm: str, index: int) -> dict[str, object]:
    statuses = {
        "K01": ("native_stop", "native_stop", {"valid_rows": 0, "duplicate_rows": 0, "unmatched_rows": 0, "ambiguous_rows": 0, "malformed_rows": 0, "invalid_rows": 0}),
        "K10": ("invalid", "invalid", {"valid_rows": 0, "duplicate_rows": 0, "unmatched_rows": 0, "ambiguous_rows": 0, "malformed_rows": 0, "invalid_rows": 1}),
        "K11": ("over_continuation", "over_continuation", {"valid_rows": 0, "duplicate_rows": 0, "unmatched_rows": 0, "ambiguous_rows": 0, "malformed_rows": 1, "invalid_rows": 1}),
        "K12": ("closure", "closure", {"valid_rows": 1, "duplicate_rows": 0, "unmatched_rows": 1, "ambiguous_rows": 0, "malformed_rows": 0, "invalid_rows": 0}),
        "K13": ("closure", "closure", {"valid_rows": 1, "duplicate_rows": 1, "unmatched_rows": 0, "ambiguous_rows": 0, "malformed_rows": 0, "invalid_rows": 0}),
    }
    status, stop_reason, parse = statuses.get(
        arm,
        ("closure", "closure", {"valid_rows": 1, "duplicate_rows": 0, "unmatched_rows": 0, "ambiguous_rows": 0, "malformed_rows": 0, "invalid_rows": 0}),
    )
    return {
        "arm_id": arm,
        "admission_mode": "pre_opener_natural",
        "opener_injected": False,
        "synthetic_opener_injections": 0,
        "opener_token_id": 1,
        "opener_generated_by_model": status == "closure",
        "first_generated_token_id": 1 if status == "closure" else None,
        "no_cache_scalar_recompute": True,
        "rows": [
            {
                "row_index": 0,
                "status": status,
                "stop_reason": stop_reason,
                "admission_mode": "pre_opener_natural",
                "opener_injected": False,
                "synthetic_opener_injections": 0,
                "opener_generated_by_model": status == "closure",
                "row_started": status == "closure",
                "token_ids": [1] if status == "closure" else [],
                "token_ids_sha256": sha256_json([1] if status == "closure" else []),
                "opener_token_id": 1,
            }
        ],
        "generated_token_ids": [1] if status == "closure" else [],
        "generated_token_ids_sha256": sha256_json([1] if status == "closure" else []),
        "scalar_forward_count": 1,
        "scalar_receipts": [{"step": 0, "finite": True, "use_cache": False}],
        "runtime_scalar_forward_count": 1,
        "runtime_scalar_receipts": [{"step": 0, "finite": True, "use_cache": False}],
        "full_logit_parity": {
            "status": "reference_captured",
            "reference_arm": "N00",
            "reference_step_count": 1,
            "candidate_step_count": 0,
            "per_forward_max_abs_delta": None,
            "tolerance": 1e-4,
            "passed": True,
        },
        "terminal_reason": stop_reason,
        "owner_bookkeeping": {
            "parse": parse,
            "stop": {"stopped": stop_reason != "closure", "stop_reason": stop_reason},
            "row_count": 1,
        },
        "synthetic_index": index,
        "executor_identity": _executor_identity(),
    }


def _full_runtime_cohort_preflight() -> dict[str, object]:
    bindings = [{"event_id": f"gt:5001:{index + 15}"} for index in range(11)]
    receipt: dict[str, object] = {
        "status": "passed",
        "event_count": 11,
        "event_identities_sha256": sha256_json([binding["event_id"] for binding in bindings]),
        "authoritative_bindings_sha256": "a" * 64,
        "processor_context_bindings": bindings,
        "processor_context_bindings_sha256": sha256_json(bindings),
        "cohort_path": "/synthetic/cohort.json",
        "cohort_sha256": "b" * 64,
        "cohort_manifest_path": "/synthetic/cohort.manifest.json",
        "cohort_manifest_sha256": "c" * 64,
    }
    receipt["receipt_sha256"] = sha256_json(receipt)
    return receipt


def _executor_identity(shard_index: int = 0) -> dict[str, object]:
    input_hashes = {
        "manifest_raw_sha256": "1" * 64,
        "manifest_self_sha256": "2" * 64,
        "census_v3_raw_sha256": "3" * 64,
        "census_v3_self_sha256": "4" * 64,
        "execution_plan_raw_sha256": "5" * 64,
        "execution_plan_sha256": "6" * 64,
        "config_sha256": "7" * 64,
        "panel_sha256": "8" * 64,
        "cohort_sha256": "9" * 64,
        "h0_identity_files_sha256": "a" * 64,
        "base_model_files_sha256": "b" * 64,
        "base_model_inventory_sha256": "f" * 64,
        "src_runtime_tree_sha256": "0" * 64,
        "event_bindings_sha256": "c" * 64,
        "authorized_event_order_sha256": "d" * 64,
        "authorized_shards_sha256": "e" * 64,
        "legacy_context_cohort_semantic_sha256": "5" * 64,
        "legacy_context_manifest_raw_sha256": "6" * 64,
        "legacy_context_manifest_semantic_sha256": "7" * 64,
        "h0_image_plan_raw_sha256": "8" * 64,
        "h0_image_plan_normalized_rows_sha256": "9" * 64,
        "legacy_context_image_plan_bindings_sha256": "a" * 64,
    }
    pre_gpu: dict[str, object] = {
        "schema_version": "s_natural_boundary_k_n_h_runtime_identity_binding.v1",
        "unit_id": UNIT_ID,
        "checkpoint": CHECKPOINT,
        "step": STEP,
        "substrate": SUBSTRATE,
        "pre_gpu_receipt_path": "/synthetic/pre-gpu-receipt.json",
        "pre_gpu_receipt_sha256": "1" * 64,
        "pre_gpu_receipt_self_sha256": "2" * 64,
        "input_hashes": input_hashes,
        "input_paths": {
            key: f"/synthetic/{key}"
            for key in (
                "manifest", "census", "execution_plan", "config", "panel",
                "cohort", "cohort_manifest", "h0_root", "h0_dir", "base_model_dir",
            )
        },
        "code_hashes": {role: sha256_json(role) for role in EXECUTOR_CODE_ROLES},
        "src_runtime_tree": {
            "root": "/synthetic/src",
            "file_count": 1,
            "inventory_sha256": input_hashes["src_runtime_tree_sha256"],
        },
        "h0_image_plan": {
            "path": "/synthetic/h0_dir/image_plan.jsonl",
            "raw_sha256": input_hashes["h0_image_plan_raw_sha256"],
            "row_count": 1,
            "normalized_rows_sha256": input_hashes["h0_image_plan_normalized_rows_sha256"],
            "row_digests": [{"row_index": 0, "row_id": "synthetic-0", "sha256": "b" * 64}],
            "row_digests_sha256": sha256_json(
                [{"row_index": 0, "row_id": "synthetic-0", "sha256": "b" * 64}]
            ),
        },
        "runtime": {
            "backend": "hf", "dtype": "fp32", "python_version": "3.11",
            "torch_version": "2.7.0", "transformers_version": "4.57.1",
        },
        "forced_math": {"enabled": True, "backend": "MATH"},
        "device_policy": {
            "device_count": 1,
            "logical_device": "cuda:0",
            "shard_physical_devices": {f"shard-{index:03d}": str(index) for index in range(8)},
        },
        "device_assignment": {
            "shard_id": f"shard-{shard_index:03d}",
            "shard_index": shard_index,
            "physical_device": str(shard_index),
            "observed_cuda_visible_devices": str(shard_index),
            "logical_device": "cuda:0",
            "device_count": 1,
            "authorization_sha256": "3" * 64,
        },
        "claim_scope": {
            "execution_scope": "checkpoint_replication_candidate",
            "event_count": 3,
            "image_count": 2,
            "minimum_checkpoint_event_count": 3,
            "minimum_checkpoint_image_count": 2,
            "checkpoint_claim_qualified": True,
            "static_direction_claim_qualified": True,
            "training_claim_qualified": False,
            "subfloor_execution_authorized": False,
        },
        "no_training": True,
    }
    pre_gpu["binding_sha256"] = sha256_json(pre_gpu)
    identity: dict[str, object] = {
        "schema_version": "s_natural_boundary_k_n_h_executor_identity.v1",
        "pre_gpu": pre_gpu,
        "full_runtime_cohort_preflight": _full_runtime_cohort_preflight(),
        "observed": {
            "model": {
                "checkpoint": CHECKPOINT,
                "step": STEP,
                "substrate": SUBSTRATE,
                "h0_dir": "/synthetic/h0_dir",
                "base_model_path": "/synthetic/base_model_dir",
                "adapter_path": "/synthetic/adapter",
                "embedding_delta_path": "/synthetic/embedding-delta",
                "resolved_config_sha256": "4" * 64,
                "h0_identity_files_sha256": input_hashes["h0_identity_files_sha256"],
                "base_model_files_sha256": input_hashes["base_model_files_sha256"],
                "base_model_inventory_sha256": input_hashes["base_model_inventory_sha256"],
            },
            "backend": {
                "type": "hf",
                "session_class": "src.inference.hf_backend.HFBackendSession",
                "model_dtype": "fp32",
                "attention_implementation": "sdpa",
                "generation": {
                    "mode": "greedy", "temperature": 0.0,
                    "top_p": 1.0,
                },
                "model_training": False,
            },
            "device": "cuda:0",
            "cuda": {
                "passed": True,
                "logical_model_device": "cuda:0",
                "cuda_visible_devices": {
                    "raw": str(shard_index),
                    "tokens": [str(shard_index)],
                    "selected_physical_device": str(shard_index),
                },
            },
            "config_sha256": input_hashes["config_sha256"],
            "runtime_versions": {
                "python_version": "3.11",
                "torch_version": "2.7.0",
                "transformers_version": "4.57.1",
            },
        },
    }
    identity["identity_sha256"] = sha256_json(identity)
    return identity


def _executor(event: dict[str, object], *, arm_order: tuple[str, ...]) -> dict[str, object]:
    return {"arms": {arm: _arm_result(arm, int(event["event_index"])) for arm in arm_order}}


def _producer_inputs(tmp_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    panel_path = tmp_path / "derived-panel.jsonl"
    panel_path.write_text("synthetic-panel\n", encoding="utf-8")
    panel_sha = hashlib.sha256(panel_path.read_bytes()).hexdigest()
    rows: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    for index in range(220):
        image_id = 7000 + (index % 13)
        owner_id = f"gt:{image_id}:{index + 1}"
        covered_a_owner_id = f"gt:{image_id}:0"
        prefix = [9]
        prefix_sha = sha256_json(prefix)
        natural_boundary = {"pre_opener_natural": True}
        rows.append(
            {
                "checkpoint": "S",
                "gt_owner_id": owner_id,
                "image_id": image_id,
                "source_panel_object_index": index,
                "derived_panel_object_index": index,
                "native_fn": True,
                "strict_complete_row": False,
                "natural_boundary": natural_boundary,
                "natural_boundary_valid": True,
                "covered_owner_ids": [covered_a_owner_id],
                "covered_A_owner_id": covered_a_owner_id,
                "covered_A_natural_boundary": 0,
                "exact_prefix_sha256": prefix_sha,
                "eligible_except_support": True,
                "geometry": {"launch_eligible": True},
            }
        )
        records.append(
            {
                "checkpoint": "S",
                "gt_owner_id": owner_id,
                "image_id": image_id,
                "source_panel_object_index": index,
                "derived_panel_object_index": index,
                "native_fn": True,
                "strict_complete_row": False,
                "natural_boundary": natural_boundary,
                "natural_boundary_valid": True,
                "covered_owner_ids": [covered_a_owner_id],
                "covered_A_owner_id": covered_a_owner_id,
                "covered_A_natural_boundary": 0,
                "exact_prefix_sha256": prefix_sha,
                "exact_prefix_token_ids": prefix,
                "support_calibration_sha256": "s" * 64,
                "support_rule": {"criterion_id": "synthetic-support-rule-v3"},
                "no_future_or_intervention_leakage": True,
                "support_features": {"assessed": True, "peak_lift": 4.0, "local_concentration": 4.0},
                "verified_support": True,
            }
        )
    rows.extend(
        {"checkpoint": "A", "gt_owner_id": f"a:{index}", "image_id": 8000 + (index % 13)}
        for index in range(564)
    )
    base: dict[str, object] = {
        "schema_version": "natural_boundary_owner_admission_census.v1",
        "status": "sealed",
        "unit_id": UNIT_ID,
        "rows": rows,
        "source_identity": {"derived_panel": {"path": str(panel_path), "sha256": panel_sha}},
    }
    base["self_sha256"] = sha256_json(base)
    ledger: dict[str, object] = {
        "schema_version": "natural_boundary_owner_support_completion_ledger.v1",
        "status": "completed",
        "checkpoint": "S",
        "records": records,
    }
    ledger["records_sha256"] = sha256_json(records)
    ledger["content_sha256"] = sha256_json({key: value for key, value in ledger.items() if key != "content_sha256"})
    return base, ledger


def test_validate_manifest_accepts_ordered_s_manifest_and_explicit_predicates(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    validated = validate_manifest(manifest)
    assert validated["arm_order"] == ARM_ORDER
    assert validated["events"][0]["natural_boundary"]["opener_seeded"] is False


def test_validate_manifest_accepts_producer_canonical_newline_hashed_census(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    census = {"schema_version": "census-v3", "status": "sealed", "rows": []}
    census["self_sha256"] = sha256_json(census)
    census_path = tmp_path / "census-v3.json"
    census_path.write_text(json.dumps(census, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    manifest["source_census"] = {
        "revision": "census-v3",
        "path": str(census_path),
        "sha256": hashlib.sha256(census_path.read_bytes()).hexdigest(),
        "hash_semantics": "canonical_json_document_with_trailing_newline",
    }
    manifest["self_sha256"] = sha256_json({key: value for key, value in manifest.items() if key != "self_sha256"})
    assert validate_manifest(manifest)["source_census_sha256"] == manifest["source_census"]["sha256"]


def test_producer_manifest_is_accepted_by_successor_runner(tmp_path: Path) -> None:
    base, ledger = _producer_inputs(tmp_path)
    result = materialize(
        base,
        ledger,
        output=tmp_path / "census-v3.json",
        records_output=tmp_path / "census-v3.records.jsonl",
        receipt_output=tmp_path / "census-v3.receipt.json",
        manifest_output=tmp_path / "admitted-events.v3.json",
        test_only=True,
    )
    manifest = result["manifest"]
    validated = validate_manifest(tmp_path / "admitted-events.v3.json")
    assert validated["arm_order"] == ARM_ORDER
    assert len(validated["events"]) == 220
    assert manifest["source_census"]["hash_semantics"] == "canonical_json_document_with_trailing_newline"


def test_contract_mode_writes_canonical_aggregate_without_model_work(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, event_count=2)
    result = run_cohort(manifest, tmp_path / "contract", executor=None, mode="contract")
    aggregate = result["aggregate"]
    assert aggregate["status"] == "contract_validated"
    assert aggregate["event_count"] == 2
    assert not list((tmp_path / "contract").glob("event-*"))
    loaded = json.loads((tmp_path / "contract" / "aggregate.json").read_text(encoding="utf-8"))
    digest = loaded.pop("aggregate_sha256")
    assert digest == sha256_json(loaded)


def test_execute_writes_one_immutable_root_per_event_and_accepts_scientific_outcomes(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, event_count=3)
    output = tmp_path / "execute"
    result = run_cohort(manifest, output, executor=_executor, mode="execute")
    assert result["aggregate"]["event_count"] == 3
    roots = sorted(output.glob("event-*"))
    assert len(roots) == 3
    assert result["aggregate"]["events"][0]["event_index"] == 0
    event = json.loads((roots[0] / "result.json").read_text(encoding="utf-8"))
    assert event["arms"]["K12"]["owner_bookkeeping"]["parse"]["unmatched_rows"] == 1
    assert event["arms"]["K13"]["owner_bookkeeping"]["parse"]["duplicate_rows"] == 1
    assert event["arms"]["K01"]["terminal_reason"] == "native_stop"
    assert event["arms"]["K11"]["terminal_reason"] == "over_continuation"


def test_execute_subthreshold_manifest_is_an_explicit_case_study(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, event_count=2)
    output = tmp_path / "case-study"
    result = run_cohort(manifest, output, executor=_executor, mode="execute")
    scope = result["aggregate"]["execution_qualification"]
    assert scope["execution_scope"] == "case_study"
    assert scope["checkpoint_claim_qualified"] is False
    assert scope["static_direction_claim_qualified"] is False
    assert scope["training_claim_qualified"] is False
    assert scope["subfloor_execution_authorized"] is True
    assert len(list(output.glob("event-*"))) == 2


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda d: d["events"][0]["eligibility"].update(rule_id="foreign-rule"), "admission predicate"),
        (lambda d: d["events"][0]["eligibility"]["predicates"].update(native_fn=False), "all-true"),
        (lambda d: d["events"][0]["owner_refs"].pop("source_panel_object_index"), "source_panel_object_index"),
        (lambda d: d["events"][0]["owner_refs"].pop("derived_panel_object_index"), "derived_panel_object_index"),
        (lambda d: d["events"][0].update(event_id="gt:9999:1"), "owner_refs"),
        (lambda d: d["events"][0].update(image_id=9999), "image_id"),
    ],
)
def test_event_identity_and_eligibility_are_bound_to_manifest(
    tmp_path: Path, mutation: object, message: str
) -> None:
    manifest = _manifest(tmp_path)
    mutation(manifest)
    event = manifest["events"][0]
    event["event_sha256"] = sha256_json({key: value for key, value in event.items() if key != "event_sha256"})
    manifest["self_sha256"] = sha256_json({key: value for key, value in manifest.items() if key != "self_sha256"})
    with pytest.raises(CohortContractError, match=message):
        validate_manifest(manifest)


@pytest.mark.parametrize("field", ["event_id", "owner_refs"])
def test_duplicate_event_or_owner_identity_fails_closed(tmp_path: Path, field: str) -> None:
    manifest = _manifest(tmp_path, event_count=2)
    manifest["events"][1]["event_id"] = manifest["events"][0]["event_id"]
    manifest["events"][1]["owner_refs"]["gt_owner_id"] = manifest["events"][0]["owner_refs"]["gt_owner_id"]
    manifest["events"][1]["image_id"] = manifest["events"][0]["image_id"]
    for event in manifest["events"]:
        body = {key: value for key, value in event.items() if key != "event_sha256"}
        event["event_sha256"] = sha256_json(body)
    manifest["self_sha256"] = sha256_json({key: value for key, value in manifest.items() if key != "self_sha256"})
    with pytest.raises(CohortContractError, match="duplicate"):
        validate_manifest(manifest)


def test_resume_is_byte_identical_and_different_manifest_cannot_reuse_roots(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, event_count=3)
    output = tmp_path / "resume"
    first = run_cohort(manifest, output, executor=_executor, mode="execute")
    first_bytes = (output / "aggregate.json").read_bytes()
    second = run_cohort(manifest, output, executor=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should resume")), mode="execute")
    assert first["aggregate"]["aggregate_sha256"] == second["aggregate"]["aggregate_sha256"]
    assert (output / "aggregate.json").read_bytes() == first_bytes

    altered = copy.deepcopy(manifest)
    altered["source_census"]["sha256"] = "b" * 64
    altered["self_sha256"] = sha256_json({key: value for key, value in altered.items() if key != "self_sha256"})
    with pytest.raises(CohortContractError, match="different manifest/event"):
        run_cohort(altered, output, executor=_executor, mode="execute")


def test_foreign_event_root_or_unexpected_cohort_file_fails_closed(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, event_count=3)
    output = tmp_path / "foreign"
    output.mkdir()
    (output / "event-999999-gt-9999-1").mkdir()
    with pytest.raises(CohortContractError, match="unexpected file or foreign event root"):
        run_cohort(manifest, output, executor=_executor, mode="execute")

    next(output.glob("event-*" )).rmdir()
    output.rmdir()
    output.mkdir()
    (output / "foreign.txt").write_text("foreign\n", encoding="utf-8")
    with pytest.raises(CohortContractError, match="unexpected file or foreign event root"):
        run_cohort(manifest, output, executor=_executor, mode="execute")


def test_resume_rejects_terminal_summary_extra_or_tampered_fields(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, event_count=3)
    output = tmp_path / "terminal"
    run_cohort(manifest, output, executor=_executor, mode="execute")
    terminal_path = next(output.glob("event-*/terminal_summary.json"))
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal["foreign"] = True
    terminal_path.write_text(json.dumps(terminal, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    with pytest.raises(CohortContractError, match="terminal receipt self hash mismatch"):
        run_cohort(manifest, output, executor=_executor, mode="execute")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda d: d["primary"].update(substrate="four-coordinate legacy"), "primary identity"),
        (lambda d: d["panel"].update(sha256="c" * 64), "panel.path hash"),
        (lambda d: d["events"][0].update(event_index=1), "event_index"),
        (lambda d: d.update(arm_order=["N00"]), "arm_order"),
        (lambda d: d["events"][0]["natural_boundary"].update(opener_token_id=50), "remain null"),
        (lambda d: d["cohort"].update(frozen_arms=["N00"]), "cohort.frozen_arms"),
    ],
)
def test_wrong_substrate_identity_duplicate_event_and_arm_set_fail_closed(
    tmp_path: Path, mutation: object, message: str
) -> None:
    manifest = _manifest(tmp_path, event_count=3)
    mutation(manifest)
    manifest["self_sha256"] = sha256_json({key: value for key, value in manifest.items() if key != "self_sha256"})
    with pytest.raises(CohortContractError, match=message):
        validate_manifest(manifest)


def test_executor_output_with_wrong_arm_set_fails_before_writes(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, event_count=3)

    def wrong_executor(event: dict[str, object], *, arm_order: tuple[str, ...]) -> dict[str, object]:
        del arm_order
        return {"arms": {"N00": _arm_result("N00", int(event["event_index"]))}}

    with pytest.raises(CohortContractError, match="arm set/order"):
        run_cohort(manifest, tmp_path / "wrong-arms", executor=wrong_executor, mode="execute")
    assert not (tmp_path / "wrong-arms" / "aggregate.json").exists()


def test_canonical_serializer_rejects_nonfinite_scientific_receipt(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, event_count=3)

    def bad_executor(event: dict[str, object], *, arm_order: tuple[str, ...]) -> dict[str, object]:
        result = {arm: _arm_result(arm, int(event["event_index"])) for arm in arm_order}
        result["N00"]["scalar_receipts"][0]["logit"] = float("nan")
        return {"arms": result}

    with pytest.raises(CohortContractError, match="canonical finite"):
        run_cohort(manifest, tmp_path / "nonfinite", executor=bad_executor, mode="execute")


@pytest.mark.parametrize("mutation", ["missing", "mismatch"])
def test_opener_provenance_is_explicit_and_tied_to_first_token(tmp_path: Path, mutation: str) -> None:
    manifest = _manifest(tmp_path, event_count=3)

    def bad_executor(event: dict[str, object], *, arm_order: tuple[str, ...]) -> dict[str, object]:
        output = _executor(event, arm_order=arm_order)
        if mutation == "missing":
            output["arms"]["N00"].pop("opener_generated_by_model")
        else:
            output["arms"]["N00"]["opener_generated_by_model"] = False
        return output

    with pytest.raises(CohortContractError, match="opener provenance"):
        run_cohort(manifest, tmp_path / f"opener-{mutation}", executor=bad_executor, mode="execute")
