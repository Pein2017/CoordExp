from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.research import seal_natural_boundary_pre_gpu_receipt as sealer


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(sealer.canonical_json_bytes(value) + b"\n")


def _self_hash(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["self_sha256"] = sealer.document_self_sha256(result)
    return result


def _fixture(tmp_path: Path) -> dict[str, Any]:
    root = tmp_path / "fixture"
    root.mkdir()
    census_root = root / sealer.CENSUS_REVISION
    unit = root / "unit.md"
    unit.write_text(
        "---\nunit_id: 2026-08-06-natural-boundary-routing-history-replication\n---\nS step-2444 primary; no training.\n",
        encoding="utf-8",
    )
    contract = _self_hash(
        {
            "schema_version": "natural_boundary_routing_history_contract.v1",
            "status": "sealed",
            "unit_id": sealer.UNIT_ID,
            "boundary_contract": {
                "primary": {"checkpoint": "S", "step": 2444},
                "training": {"authorized": False},
                "mutations": {"production_launch": False},
            },
            "contract_files": {"unit": {"sha256": sealer.sha256_file(unit)}},
        }
    )
    contract_path = root / "contract.json"
    _write_json(contract_path, contract)

    audit_md = root / "prior-evidence-semantic-audit.md"
    audit_md.write_text("# Prior audit\nS is primary; no training route.\n", encoding="utf-8")
    audit = {
        "schema_version": "prior_evidence_semantic_audit.v1",
        "audit_id": "fixture-audit",
        "audit_document_path": audit_md.name,
        "audit_document_sha256": sealer.sha256_file(audit_md),
        "status": "complete_verified_bounded_narrowing",
        "unit_id": sealer.UNIT_ID,
        "route": {
            "no_training_route_selected": True,
            "s_role": "primary_current_production_substrate",
        },
        "checks": {"s_hold": {"event_id": sealer.EVENT_ID}},
        "inputs": {"new_unit_sha256": sealer.sha256_file(unit)},
    }
    audit_json = root / "prior-evidence-semantic-audit.json"
    _write_json(audit_json, audit)

    row = {
        "checkpoint": "S",
        "gt_owner_id": sealer.EVENT_OWNER_ID,
        "image_id": sealer.EVENT_IMAGE_ID,
        "source_panel_object_index": sealer.EVENT_SOURCE_PANEL_OBJECT_INDEX,
        "disposition": "eligible_verified_pair",
        "eligible_except_support": True,
        "exact_prefix_sha256": "a" * 64,
        "geometry": {"geometry_sha256": sealer.EXPECTED_EVENT_GEOMETRY_SHA256},
    }
    support_rows = [
        {
            "checkpoint": "S",
            "gt_owner_id": f"gt:support:{index}",
            "image_id": 5000 + index,
            "native_fn": True,
            "disposition": "support_unassessed",
        }
        for index in range(200)
    ]
    census = {
        "schema_version": "natural_boundary_owner_admission_census.v1",
        "status": "sealed",
        "unit_id": sealer.UNIT_ID,
        "frozen_universe": {"row_count": 784, "physical_owner_count": 392},
        "rows": [row, *support_rows],
        "records_sha256": "placeholder",
    }
    records = census_root / "admission-census.records.jsonl"
    records.parent.mkdir(parents=True, exist_ok=True)
    records.write_bytes(b"".join(sealer.canonical_json_bytes(item) + b"\n" for item in census["rows"]))
    census["records_sha256"] = sealer.sha256_file(records)
    census = _self_hash(census)
    census_path = census_root / "admission-census.json"
    _write_json(census_path, census)
    census_receipt = _self_hash(
        {
            "schema_version": "natural_boundary_owner_admission_census.v1.receipt",
            "status": "sealed",
            "unit_id": sealer.UNIT_ID,
            "row_count": 784,
            "physical_owner_count": 392,
            "records_sha256": census["records_sha256"],
            "census_self_sha256": census["self_sha256"],
            "support_completion_candidates_count": 200,
        }
    )
    census_receipt_path = census_root / "admission-census.receipt.json"
    _write_json(census_receipt_path, census_receipt)

    plan = {
        "schema_version": "natural_boundary_owner_support_completion_plan.v1",
        "status": "sealed_cpu_plan",
        "unit_id": sealer.UNIT_ID,
        "checkpoint": "S",
        "wrapper": "object_box_closed",
        "execution_contract": {"gpu_used": False, "model_loaded": False, "training": False},
        "scope": {"support_completion_candidates": 200, "native_fn_denominator": 220},
        "work": {
            "shard_count": 8,
            "scalar_equivalent_forward_count": 77428,
            "per_shard": [
                {"shard_index": shard, "context_count": context_count, "scalar_equivalent_forward_count": forward_count}
                for shard, (context_count, forward_count) in enumerate(
                    zip(
                        [21, 22, 25, 18, 25, 24, 33, 32],
                        [6757, 7012, 11819, 8208, 9845, 8511, 13867, 11409],
                        strict=True,
                    )
                )
            ],
        },
    }
    plan["plan_content_sha256"] = sealer.sha256_json(plan)
    plan_path = root / "plan.json"
    _write_json(plan_path, plan)
    plan_receipt = {
        "schema_version": "natural_boundary_owner_support_completion_plan.v1.receipt.v1",
        "status": "sealed_cpu_plan",
        "unit_id": sealer.UNIT_ID,
        "plan_sha256": sealer.sha256_file(plan_path),
        "plan_content_sha256": plan["plan_content_sha256"],
        "context_count": 200,
        "support_completion_candidate_count": 200,
        "gpu_used": False,
        "model_loaded": False,
        "execution_receipt_sha256": None,
    }
    plan_receipt_path = root / "plan.receipt.json"
    _write_json(plan_receipt_path, plan_receipt)

    execution_root = root / "support-execution-contract-v1"
    context_counts = [21, 22, 25, 18, 25, 24, 33, 32]
    forward_counts = [6757, 7012, 11819, 8208, 9845, 8511, 13867, 11409]
    owner_ids_sha256 = sealer.sha256_json(sorted(item["gt_owner_id"] for item in support_rows))
    execution_paths: list[Path] = []
    for shard, (context_count, forward_count) in enumerate(zip(context_counts, forward_counts, strict=True)):
        execution_path = execution_root / f"shard-{shard}.json"
        _write_json(
            execution_path,
            {
                "schema_version": "natural_boundary_owner_support_completion_execution.v1",
                "unit_id": sealer.UNIT_ID,
                "status": "contract_ready",
                "plan_content_sha256": plan["plan_content_sha256"],
                "checkpoint": "S",
                "wrapper": "object_box_closed",
                "parser": "compact_object_box_closed_only",
                "shard_index": shard,
                "num_shards": 8,
                "context_count": context_count,
                "scalar_equivalent_forward_count": forward_count,
                "census_binding": {
                    "revision": sealer.CENSUS_REVISION,
                    "path": str(census_path.resolve()),
                    "file_sha256": sealer.sha256_file(census_path),
                    "self_sha256": census["self_sha256"],
                    "s_owner_ids_sha256": owner_ids_sha256,
                },
                "batching_admitted": False,
                "batch_estimates": {"batching_admitted": False},
                "gpu_used": False,
                "model_loaded": False,
                "legacy_frozen_candidate_registry_read": False,
                "native_tp_calibration_scored": False,
            },
        )
        execution_paths.append(execution_path)

    authored = root / "authored.yaml"
    authored.write_text("model:\n  checkpoint: S\n", encoding="utf-8")
    resolved = root / "resolved.json"
    resolved.write_text("{\"checkpoint\":\"S\"}\n", encoding="utf-8")
    adapter = root / "adapter.safetensors"
    adapter.write_bytes(b"adapter")
    embedding = root / "embedding.safetensors"
    embedding.write_bytes(b"embedding")
    h0 = {
        "checkpoint": "S",
        "step": 2444,
        "resolved_config": {
            "path": str(resolved),
            "sha256": sealer.sha256_file(resolved),
            "config_fingerprint": "b" * 64,
        },
        "adapter": {"path": str(adapter), "sha256": sealer.sha256_file(adapter)},
        "embedding": {"path": str(embedding), "sha256": sealer.sha256_file(embedding)},
    }

    sources: dict[str, Path] = {}
    for role in sealer.REQUIRED_SOURCE_ROLES:
        if role in {"pre_gpu_sealer", "pre_gpu_materializer"}:
            sources[role] = sealer.DEFAULT_SOURCE_FILES[role]
            continue
        path = root / f"{role}.py"
        path.write_text(f"# {role}\n", encoding="utf-8")
        sources[role] = path

    focused = root / "test_focus.py"
    focused.write_text("def test_focus(): pass\n", encoding="utf-8")
    test_receipt = root / "tests.receipt.json"
    _write_json(
        test_receipt,
        {
            "status": "passed",
            "command": "pytest -q tests/research/test_focus.py",
            "exit_code": 0,
            "fail_count": 0,
            "tests": [{"path": str(focused), "sha256": sealer.sha256_file(focused)}],
        },
    )
    mask_probe = root / "mask-probe.json"
    native_mass = [[[0.1 + 0.01 * head] for head in range(16)]]
    biased_mass = [[[0.2 + 0.02 * head] for head in range(16)]]
    delta_mass = [[[0.1 + 0.01 * head] for head in range(16)]]
    _write_json(
        mask_probe,
        {
            "status": "passed",
            "torch_version": "2.7.0",
            "transformers_version": "4.57.1",
            "qwen_model_class": "Qwen3VLForConditionalGeneration",
            "float_additive_4d_mask_passthrough": True,
            "all_layer_consumption": True,
            "block23_sdpa_mass_attestation": {
                "schema_version": "natural_boundary_attention_actuators.v1.block23_sdpa_attestor.v1",
                "status": "passed",
                "passed": True,
                "block23_call_count": 1,
                "exactly_one_block23_call": True,
                "registry_restored": True,
                "delegate_untouched": True,
                "delegate_identity": {
                    "id": 123,
                    "module": "transformers.modeling_utils",
                    "qualname": "sdpa_attention_forward",
                },
                "q_heads": 16,
                "kv_heads": 8,
                "num_key_value_groups": 2,
                "groups": 2,
                "gqa_expansion": {
                    "mode": "repeat_kv",
                    "groups": 2,
                    "head_map": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7],
                },
                "records": [
                    {
                        "q_heads": 16,
                        "kv_heads": 8,
                        "num_key_value_groups": 2,
                        "groups": 2,
                        "selected_mass": native_mass,
                    }
                ],
                "native": {
                    "delegate_identity": {
                        "id": 123,
                        "module": "transformers.modeling_utils",
                        "qualname": "sdpa_attention_forward",
                    },
                    "records": [
                        {
                            "q_heads": 16,
                            "kv_heads": 8,
                            "num_key_value_groups": 2,
                            "groups": 2,
                            "selected_mass": native_mass,
                        }
                    ]
                },
                "biased": {
                    "delegate_identity": {
                        "id": 123,
                        "module": "transformers.modeling_utils",
                        "qualname": "sdpa_attention_forward",
                    },
                    "records": [
                        {
                            "q_heads": 16,
                            "kv_heads": 8,
                            "num_key_value_groups": 2,
                            "groups": 2,
                            "selected_mass": biased_mass,
                        }
                    ]
                },
                "comparison": {
                    "native_mass": native_mass,
                    "biased_mass": biased_mass,
                    "delta_mass": delta_mass,
                    "selected_mass_shape": [1, 16, 1],
                    "q_heads": 16,
                    "kv_heads": 8,
                    "num_key_value_groups": 2,
                    "groups": 2,
                    "all_query_heads_nonzero_shift": True,
                    "all_head_nonzero_shift": True,
                    "mass_shift_observed": True,
                    "same_delegate": True,
                    "tolerance": 0.0,
                    "passed": True,
                },
                "native_delegate_output_parity": True,
                "biased_output_changed": True,
            },
        },
    )
    return {
        "output": root / sealer.RECEIPT_REVISION / sealer.RECEIPT_FILENAME,
        "gate_output_root": root / "s-gt5001-live-gate-v3",
        "contract_path": contract_path,
        "unit_path": unit,
        "audit_json_path": audit_json,
        "audit_md_path": audit_md,
        "census_path": census_path,
        "census_records_path": records,
        "census_receipt_path": census_receipt_path,
        "support_plan_path": plan_path,
        "support_plan_receipt_path": plan_receipt_path,
        "support_execution_receipts": execution_paths,
        "authored_config": authored,
        "h0_identity": h0,
        "source_files": sources,
        "focused_tests": [focused],
        "test_receipt": test_receipt,
        "mask_probe": mask_probe,
    }


def test_build_and_validate_binds_all_pre_gpu_inputs(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    document = sealer.build_receipt(**fixture)

    assert document["status"] == "sealed_pre_gpu"
    assert document["schema_version"] == sealer.SCHEMA_VERSION
    assert document["receipt_revision"] == sealer.RECEIPT_REVISION
    assert document["checkpoint"] == "S"
    assert document["step"] == 2444
    assert document["event_binding"]["event_id"] == sealer.EVENT_ID
    assert set(sealer.REQUIRED_SOURCE_ROLES) <= set(document["source_files"])
    assert document["focused_test_execution"]["status"] == "passed"
    assert document["mask_probe_identity"]["all_layer_consumption"] is True
    assert document["self_sha256"] == sealer.document_self_sha256(document)
    sealer.validate_receipt(document)


def test_seal_is_write_once_and_validate_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fixture = _fixture(tmp_path)
    first = sealer.seal(**fixture)
    second = sealer.seal(**fixture)
    assert first["byte_identical"] is False
    assert second["byte_identical"] is True
    assert first["sha256"] == second["sha256"]
    assert sealer.main(["validate", str(fixture["output"])]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "valid"

    fixture["output"].write_bytes(fixture["output"].read_bytes() + b"drift")
    with pytest.raises(FileExistsError, match="output collision"):
        sealer.seal(**fixture)


def test_rejects_non_s_h0_identity(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    bad = dict(fixture["h0_identity"])
    bad["checkpoint"] = "A"
    with pytest.raises(sealer.PreGpuReceiptError, match="checkpoint must be S"):
        sealer.build_receipt(**{**fixture, "h0_identity": bad})


def test_rejects_unclean_test_receipt_and_mask_probe(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    bad_tests = tmp_path / "bad-tests.json"
    _write_json(
        bad_tests,
        {
            "status": "failed",
            "command": "pytest -q",
            "exit_code": 1,
            "fail_count": 1,
            "tests": [],
        },
    )
    with pytest.raises(sealer.PreGpuReceiptError, match="not clean/passed"):
        sealer.build_receipt(**{**fixture, "test_receipt": bad_tests})

    bad_probe = tmp_path / "bad-probe.json"
    _write_json(bad_probe, {"status": "passed", "torch_version": "2", "transformers_version": "4"})
    with pytest.raises(sealer.PreGpuReceiptError, match="Qwen mask probe lacks"):
        sealer.build_receipt(**{**fixture, "mask_probe": bad_probe})


def test_rejects_symlink_wildcard_and_forbidden_scope(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    link = tmp_path / "unit-link.md"
    link.symlink_to(fixture["unit_path"])
    with pytest.raises(sealer.PreGpuReceiptError, match="non-symlink"):
        sealer.build_receipt(**{**fixture, "unit_path": link})
    with pytest.raises(sealer.PreGpuReceiptError, match="wildcard"):
        sealer.build_receipt(**{**fixture, "source_files": {"natural_runner": "/tmp/*.py"}})
    with pytest.raises(sealer.PreGpuReceiptError, match="forbidden scope training"):
        sealer.build_receipt(**{**fixture, "forbidden_scope": {"training": True}})


def test_rejects_superseded_cpu_census_v1_path(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    old_root = tmp_path / "cpu-census-v1"
    old_root.mkdir()
    old_paths = {
        "census_path": old_root / "admission-census.json",
        "census_records_path": old_root / "admission-census.records.jsonl",
        "census_receipt_path": old_root / "admission-census.receipt.json",
    }
    for key, destination in old_paths.items():
        destination.write_bytes(fixture[key].read_bytes())
    with pytest.raises(sealer.PreGpuReceiptError, match="cpu-census-v2"):
        sealer.build_receipt(**{**fixture, **old_paths})


def test_rejects_support_contract_receipt_bound_to_v1(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    bad = tmp_path / "bad-support-contract.json"
    document = json.loads(fixture["support_execution_receipts"][0].read_text(encoding="utf-8"))
    document["census_binding"]["revision"] = "cpu-census-v1"
    _write_json(bad, document)
    receipts = list(fixture["support_execution_receipts"])
    receipts[0] = bad
    with pytest.raises(sealer.PreGpuReceiptError, match="census binding revision"):
        sealer.build_receipt(**{**fixture, "support_execution_receipts": receipts})


def test_runtime_identity_repeats_receipt_and_code_hashes(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    result = sealer.seal(**fixture)
    receipt = json.loads(fixture["output"].read_text(encoding="utf-8"))
    runtime = {
        "unit_id": sealer.UNIT_ID,
        "checkpoint": "S",
        "event_id": sealer.EVENT_ID,
        "pre_gpu_receipt_path": str(fixture["output"].resolve()),
        "pre_gpu_receipt_self_sha256": receipt["self_sha256"],
        "pre_gpu_receipt_sha256": result["sha256"],
        "code_hashes": receipt["code_identity"]["sha256"],
    }
    checked = sealer.validate_runtime_identity(runtime, receipt_path=fixture["output"], receipt=receipt)
    assert checked["receipt_sha256"] == result["sha256"]
    runtime["code_hashes"] = dict(runtime["code_hashes"])
    runtime["code_hashes"]["natural_runner"] = "0" * 64
    with pytest.raises(sealer.PreGpuReceiptError, match="code hashes differ"):
        sealer.validate_runtime_identity(runtime, receipt_path=fixture["output"], receipt=receipt)


def test_v4_output_root_rejects_v3_without_mutating_existing_bytes(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    legacy_output = tmp_path / "pre-gpu-receipt-v3" / sealer.RECEIPT_FILENAME
    legacy_output.parent.mkdir(parents=True)
    legacy_bytes = b'{"schema_version":"natural_boundary_pre_gpu_receipt.v3"}\n'
    legacy_output.write_bytes(legacy_bytes)
    with pytest.raises(sealer.PreGpuReceiptError, match="pre-gpu-receipt-v4"):
        sealer.build_receipt(**{**fixture, "output": legacy_output})
    assert legacy_output.read_bytes() == legacy_bytes
    assert sealer.DEFAULT_OUTPUT_ROOT.name == "pre-gpu-receipt-v4"


def test_one_to_one_block23_probe_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    probe_path = fixture["mask_probe"]
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    attestation = probe["block23_sdpa_mass_attestation"]
    for container in (
        attestation,
        attestation["native"],
        attestation["biased"],
        attestation["comparison"],
        attestation["native"]["records"][0],
        attestation["biased"]["records"][0],
    ):
        container["q_heads"] = 2
        container["kv_heads"] = 2
        container["num_key_value_groups"] = 1
        container["groups"] = 1
    _write_json(probe_path, probe)
    with pytest.raises(sealer.PreGpuReceiptError, match="GQA"):
        sealer.build_receipt(**fixture)


def test_v4_payload_is_gate_consumer_compatible(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    result = sealer.seal(**fixture)
    from scripts.research import run_s_primary_natural_boundary_gate as gate

    identity = gate._load_pre_gpu_identity(fixture["output"])
    receipt = json.loads(fixture["output"].read_text(encoding="utf-8"))
    assert identity["pre_gpu_receipt_path"] == str(fixture["output"].resolve())
    assert identity["pre_gpu_receipt_sha256"] == result["sha256"]
    assert identity["pre_gpu_receipt_self_sha256"] == result["self_sha256"]
    assert identity["code_hashes"] == receipt["code_identity"]["sha256"]


def test_authority_status_source_thread_and_scope_are_bound(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    sealer.seal(**fixture)
    document = json.loads(fixture["output"].read_text(encoding="utf-8"))
    assert document["authority_binding"]["file"]["path"] == str(sealer.AUTHORITY_PATH.resolve())
    assert document["authority_binding"]["status"] == sealer.EXPECTED_AUTHORITY_STATUS
    assert document["authority_binding"]["source_thread"] == sealer.EXPECTED_AUTHORITY_SOURCE_THREAD
    assert document["authority_binding"]["scope"] == sealer.EXPECTED_AUTHORITY_SCOPE
    assert document["gate_output_binding"]["path"].endswith("/s-gt5001-live-gate-v3")
    assert not Path(document["gate_output_binding"]["path"]).exists()

    bad = json.loads(fixture["output"].read_text(encoding="utf-8"))
    bad["authority_binding"]["status"] = "inactive"
    bad["self_sha256"] = sealer.document_self_sha256(bad)
    with pytest.raises(sealer.PreGpuReceiptError, match="authority binding drifted"):
        sealer.validate_receipt(bad)

    bad_hash = json.loads(fixture["output"].read_text(encoding="utf-8"))
    bad_hash["authority_binding"]["file"]["sha256"] = "0" * 64
    bad_hash["serialization_successor_authority"] = bad_hash["authority_binding"]
    bad_hash["self_sha256"] = sealer.document_self_sha256(bad_hash)
    with pytest.raises(sealer.PreGpuReceiptError, match="SHA-256 drifted"):
        sealer.validate_receipt(bad_hash)


def test_wrong_authority_hash_status_and_gate_root_fail_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    wrong_authority = tmp_path / "wrong-authority.md"
    wrong_authority.write_text(sealer.AUTHORITY_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(sealer.PreGpuReceiptError, match="exact regular authority"):
        sealer.build_receipt(**{**fixture, "authority_path": wrong_authority})

    wrong_name = tmp_path / "wrong-gate-root"
    with pytest.raises(sealer.PreGpuReceiptError, match="end with s-gt5001-live-gate-v3"):
        sealer.build_receipt(**{**fixture, "gate_output_root": wrong_name})

    reused = tmp_path / "reused" / "s-gt5001-live-gate-v3"
    reused.mkdir(parents=True)
    with pytest.raises(sealer.PreGpuReceiptError, match="absent and unused"):
        sealer.build_receipt(**{**fixture, "gate_output_root": reused})

    symlink_parent = tmp_path / "symlink-parent"
    symlink_parent.symlink_to(tmp_path, target_is_directory=True)
    symlink_root = symlink_parent / "s-gt5001-live-gate-v3"
    with pytest.raises(sealer.PreGpuReceiptError, match="must not be a symlink"):
        sealer.build_receipt(**{**fixture, "gate_output_root": symlink_root})


def test_v4_source_identity_includes_sealer_and_materializer(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    document = sealer.build_receipt(**fixture)
    assert document["source_files"]["pre_gpu_sealer"]["path"] == str(
        (sealer.REPO_ROOT / "scripts/research/seal_natural_boundary_pre_gpu_receipt.py").resolve()
    )
    assert document["source_files"]["pre_gpu_materializer"]["path"] == str(
        (sealer.REPO_ROOT / "scripts/research/materialize_natural_boundary_pre_gpu_evidence.py").resolve()
    )


def test_validate_allows_existing_regular_gate_root_after_seal(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    sealer.seal(**fixture)
    receipt = json.loads(fixture["output"].read_text(encoding="utf-8"))
    sealer.validate_receipt(receipt, receipt_path=fixture["output"])

    gate_root = Path(receipt["gate_output_root"])
    gate_root.mkdir(parents=True)
    (gate_root / "result.json").write_text("{}\n", encoding="utf-8")
    sealer.validate_receipt(receipt, receipt_path=fixture["output"])
