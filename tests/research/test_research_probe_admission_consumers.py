from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pytest

from scripts.research import research_probe_admission_consumers as consumers
from scripts.research import resumable_natural_boundary_support_completion as support
from scripts.research import run_research_probe_admission_vertical as vertical_driver
from src.artifacts.json_values import json_sha256
from src.artifacts.research_probe_admission import (
    RegularFileBinding,
    ResearchProbeAdmission,
    ResearchProbeAdmissionError,
    ReservedOutputPath,
    StageEvidence,
    capture_binding_manifest,
)


def _write(path: Path, value: dict[str, Any]) -> Path:
    body = dict(value)
    body["content_sha256"] = json_sha256(body)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        json.dumps(
            body,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    return path


def _rewrite_receipt(source: Path, output: Path, mutate: Any) -> Path:
    document = json.loads(source.read_text(encoding="utf-8"))
    document.pop("content_sha256")
    mutate(document)
    return _write(output, document)


def _support_cpu(output_root: Path, receipt: Path) -> StageEvidence:
    return consumers.build_support_cpu_evidence(
        output_root=output_root,
        compatibility_receipt_path=receipt,
        plan_path=consumers.SUPPORT_PLAN,
        census_path=consumers.SUPPORT_CENSUS,
        consumer_path=consumers.SUPPORT_CONSUMER,
        consumer_test_path=consumers.SUPPORT_CONSUMER_TEST,
        merger_path=consumers.SUPPORT_MERGER,
        merger_test_path=consumers.SUPPORT_MERGER_TEST,
    )


def _crossover_cpu(output_root: Path, receipt: Path) -> StageEvidence:
    return consumers.build_crossover_cpu_evidence(
        output_root=output_root,
        compatibility_receipt_path=receipt,
        source_preflight_path=consumers.CROSSOVER_SOURCE_PREFLIGHT,
        accepted_evidence_path=consumers.CROSSOVER_ACCEPTED_EVIDENCE,
        runner_path=consumers.CROSSOVER_RUNNER,
        runner_test_path=consumers.CROSSOVER_RUNNER_TEST,
        sealer_path=consumers.CROSSOVER_SEALER,
        finalizer_path=consumers.CROSSOVER_FINALIZER,
        finalizer_test_path=consumers.CROSSOVER_FINALIZER_TEST,
    )


def _vertical_sources(tmp_path: Path) -> tuple[Any, ...]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    raw = {
        "schema_version": "fixture",
        "unit_id": "fixture",
        "contexts": [
            {
                "context_id": "one",
                "context_plan_position": 0,
                "scalar_equivalent_forward_count": 1,
                "shard_index": 0,
                "candidate_ids": ["a"],
            }
        ],
        "calibration_reuse": {"calibration_sha256": "a" * 64},
        "support_lineage": {"support_rule": {}},
        "scope": {"native_tp_other_not_scored": True},
        "work": {
            "candidate_batch_size": 1,
            "scalar_equivalent_forward_count": 1,
            "per_shard": [
                {"shard_index": 0, "scalar_equivalent_forward_count": 1}
            ],
        },
    }
    raw["plan_content_sha256"] = json_sha256(raw)
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(
        json.dumps(raw, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    plan = support.load_logical_plan(plan_path)
    schedule = support.plan_physical_slots(plan, slot_count=1)
    identity = {"fixture": "not-production"}
    slot = tmp_path / "slot"
    support.execute_slot(
        root=slot,
        execution_id="slot",
        execution_identity=identity,
        plan=plan,
        schedule=schedule,
        slot_index=0,
        observe_context=lambda context: {
            "context_id": context["context_id"],
            "status": "measured",
            "support_features": {},
            "candidate_scores": {"a": 0.0},
            "candidate_score_count": 1,
            "candidate_scores_sha256": json_sha256({"a": 0.0}),
        },
    )
    terminal = tmp_path / "terminal.json"
    support.materialize_bounded_mechanics_terminal(
        plan=plan,
        schedule=schedule,
        slot_roots=(slot,),
        execution_identity=identity,
        output_path=terminal,
    )
    runtime = _write(
        tmp_path / "runtime.json",
        {
            "kind": "live_worker_runtime_identity",
            "binding": identity,
            "runtime_identity": {
                "normalized_device": "cuda:0",
                "checkpoint": "S",
                "config_fingerprint": "cfg",
            },
            "claim_boundary": "live_runtime_mechanics_only_no_support_or_model_mechanism_claim",
        },
    )
    exit_receipt = _write(
        tmp_path / "exit.json",
        {
            "schema_version": support.MECHANICS_SCHEMA_VERSION,
            "kind": "worker_exit",
            "claim_boundary": "mechanics_only_no_exit_interpretation_or_automatic_retry",
            "physical_slot_index": 0,
            "logical_plan_file_sha256": plan.file_sha256,
            "schedule_sha256": schedule["content_sha256"],
            "accepted_context_count": 1,
            "missing_context_count": 0,
            "journal_terminal": True,
            "return_code": 0,
            "command": [sys.executable, "fake-worker.py"],
            "cwd": str(tmp_path.resolve()),
            "executable": {
                "path": str(Path(sys.executable).resolve()),
                "raw_sha256": support.file_sha256(sys.executable),
                "byte_count": Path(sys.executable).resolve().stat().st_size,
            },
        },
    )
    return plan_path, schedule, slot, identity, terminal, runtime, exit_receipt


def test_current_cpu_receipts_cross_shared_stage_evidence(tmp_path: Path) -> None:
    support_manifest = capture_binding_manifest(consumers.support_binding_requests())
    support_root = tmp_path / "support"
    with ResearchProbeAdmission.create(
        root=tmp_path / "support-admission",
        admission_id="support-cpu",
        bindings=support_manifest,
        reserved_output_paths=(ReservedOutputPath("support", support_root),),
        context={"consumer": "support", "cpu_only": True},
    ) as admission:
        evidence = _support_cpu(support_root, consumers.SUPPORT_COMPATIBILITY_RECEIPT)
        admission.append_stage(
            stage="cpu_preflight",
            evidence=evidence,
            attempt_id=admission.start_attempt(),
        )
    assert evidence.assertions["downstream_validator_accepted"] is True

    crossover_manifest = capture_binding_manifest(consumers.crossover_binding_requests())
    crossover_root = tmp_path / "crossover"
    with ResearchProbeAdmission.create(
        root=tmp_path / "crossover-admission",
        admission_id="crossover-cpu",
        bindings=crossover_manifest,
        reserved_output_paths=(ReservedOutputPath("crossover", crossover_root),),
        context={"consumer": "crossover", "cpu_only": True},
    ) as admission:
        evidence = _crossover_cpu(
            crossover_root, consumers.CROSSOVER_COMPATIBILITY_RECEIPT
        )
        admission.append_stage(
            stage="cpu_preflight",
            evidence=evidence,
            attempt_id=admission.start_attempt(),
        )
    assert evidence.assertions["model_free_finalizer_ran"] is True


def test_boolean_only_or_command_tampered_cpu_receipt_is_rejected(
    tmp_path: Path,
) -> None:
    missing = _rewrite_receipt(
        consumers.SUPPORT_COMPATIBILITY_RECEIPT,
        tmp_path / "missing.json",
        lambda document: document.pop("command_results"),
    )
    with pytest.raises(consumers.ConsumerAdmissionError, match="command_results"):
        _support_cpu(tmp_path / "missing-output", missing)

    def tamper(document: dict[str, Any]) -> None:
        document["command_results"]["consumer_validator"]["argv"].append("--fake")

    tampered = _rewrite_receipt(
        consumers.SUPPORT_COMPATIBILITY_RECEIPT,
        tmp_path / "tampered.json",
        tamper,
    )
    with pytest.raises(consumers.ConsumerAdmissionError, match="projection differs"):
        _support_cpu(tmp_path / "tampered-output", tampered)


def test_helper_only_paths_and_vertical_receipts_cannot_claim_production(
    tmp_path: Path,
) -> None:
    fake_consumer = tmp_path / "consumer.py"
    fake_consumer.write_text("# helper\n", encoding="utf-8")
    with pytest.raises(consumers.ConsumerAdmissionError, match="current production"):
        consumers.build_support_cpu_evidence(
            output_root=tmp_path / "cpu-output",
            compatibility_receipt_path=consumers.SUPPORT_COMPATIBILITY_RECEIPT,
            plan_path=consumers.SUPPORT_PLAN,
            census_path=consumers.SUPPORT_CENSUS,
            consumer_path=fake_consumer,
            consumer_test_path=consumers.SUPPORT_CONSUMER_TEST,
            merger_path=consumers.SUPPORT_MERGER,
            merger_test_path=consumers.SUPPORT_MERGER_TEST,
        )
    plan, schedule, slot, identity, terminal, runtime, exit_receipt = (
        _vertical_sources(tmp_path / "vertical")
    )
    manifest = capture_binding_manifest(
        (
            RegularFileBinding("support_worker", consumers.SUPPORT_WORKER),
            RegularFileBinding(
                "consumer_admission_adapter", consumers.CONSUMER_ADMISSION_ADAPTER
            ),
        )
    )
    with pytest.raises(consumers.ConsumerAdmissionError, match="production"):
        consumers.build_support_vertical_evidence(
            output_root=tmp_path / "vertical-output",
            terminal_path=terminal,
            plan_path=plan,
            schedule=schedule,
            slot_roots=(slot,),
            execution_identity=identity,
            admission_bindings=manifest,
            worker_path=consumers.SUPPORT_WORKER,
            executable_path=Path(sys.executable),
            worker_argv=[sys.executable, str(consumers.SUPPORT_WORKER)],
            worker_cwd=consumers.INFRA_ROOT,
            worker_exit_receipts=(exit_receipt,),
            runtime_receipts=(runtime,),
            expected_checkpoint="S",
            expected_config_fingerprint="cfg",
        )


def test_vertical_snapshot_is_exactly_recoverable_and_flat_bound(
    tmp_path: Path,
) -> None:
    _, _, slot, _, terminal, runtime, exit_receipt = _vertical_sources(tmp_path)
    output_root = tmp_path / "snapshot"
    outputs = consumers._publish_vertical_snapshot(
        output_root,
        {"fixture": "mechanics-only"},
        terminal,
        (runtime,),
        (exit_receipt,),
        (slot,),
    )
    expected_hashes = [support.file_sha256(path) for path in outputs]
    outputs[0].unlink()
    outputs[-1].unlink()
    recovered = consumers._publish_vertical_snapshot(
        output_root,
        {"fixture": "mechanics-only"},
        terminal,
        (runtime,),
        (exit_receipt,),
        (slot,),
    )
    assert [support.file_sha256(path) for path in recovered] == expected_hashes
    (output_root / "unexpected.json").write_text("{}", encoding="utf-8")
    with pytest.raises(consumers.ConsumerAdmissionError, match="unexpected"):
        consumers._publish_vertical_snapshot(
            output_root,
            {"fixture": "mechanics-only"},
            terminal,
            (runtime,),
            (exit_receipt,),
            (slot,),
        )


def test_post_append_snapshot_drift_blocks_final_admission(tmp_path: Path) -> None:
    _, _, slot, _, terminal, runtime, exit_receipt = _vertical_sources(
        tmp_path / "raw"
    )
    producer = tmp_path / "producer.py"
    validator = tmp_path / "validator.py"
    producer.write_text("# producer\n", encoding="utf-8")
    validator.write_text("# validator\n", encoding="utf-8")
    manifest = capture_binding_manifest(
        (
            RegularFileBinding("producer", producer),
            RegularFileBinding("validator", validator),
        )
    )
    cpu_root = tmp_path / "cpu"
    vertical_root = tmp_path / "vertical"
    with ResearchProbeAdmission.create(
        root=tmp_path / "admission",
        admission_id="drift",
        bindings=manifest,
        reserved_output_paths=(
            ReservedOutputPath("cpu", cpu_root),
            ReservedOutputPath("vertical", vertical_root),
        ),
        context={"fixture": "output-drift"},
    ) as admission:
        cpu_file = _write(cpu_root / "cpu.json", {"fixture": "cpu"})
        admission.append_stage(
            stage="cpu_preflight",
            evidence=StageEvidence(
                producer_binding="producer",
                validator_binding="validator",
                output_files=(RegularFileBinding("cpu", cpu_file),),
                assertions={
                    "production_entrypoint_resolved": True,
                    "consumer_validator_ran": True,
                    "model_free_finalizer_ran": True,
                    "downstream_validator_accepted": True,
                    "model_loaded": False,
                    "gpu_used": False,
                },
                detail={"fixture": "cpu"},
            ),
            attempt_id=admission.start_attempt(),
        )
        outputs = consumers._publish_vertical_snapshot(
            vertical_root,
            {"fixture": "mechanics-only"},
            terminal,
            (runtime,),
            (exit_receipt,),
            (slot,),
        )
        admission.append_stage(
            stage="vertical_smoke",
            evidence=StageEvidence(
                producer_binding="producer",
                validator_binding="validator",
                output_files=tuple(
                    RegularFileBinding(f"vertical_{index}", path)
                    for index, path in enumerate(outputs)
                ),
                assertions={
                    "production_entrypoint_executed": True,
                    "model_runtime_loaded": True,
                    "terminal_finalizer_completed": True,
                    "downstream_validator_accepted": True,
                    "durable_work_item_count": 1,
                },
                detail={"fixture": "vertical"},
            ),
            attempt_id=admission.start_attempt(),
        )
        outputs[-1].unlink()
        with pytest.raises(ResearchProbeAdmissionError):
            admission.finalize()


def test_exact_cpu_publication_retry_accepts_only_identical_single_file(
    tmp_path: Path,
) -> None:
    root = tmp_path / "receipt-root"
    first = consumers._publish_fresh_receipt(root, "receipt.json", {"fixture": True})
    assert consumers._publish_fresh_receipt(
        root, "receipt.json", {"fixture": True}
    ) == first
    first.write_text("{}", encoding="utf-8")
    with pytest.raises(consumers.ConsumerAdmissionError, match="differs"):
        consumers._publish_fresh_receipt(root, "receipt.json", {"fixture": True})


def test_outer_mechanics_receipt_recovers_only_exact_publication(
    tmp_path: Path,
) -> None:
    root = tmp_path / "acceptance"
    root.mkdir()
    (root / "support").mkdir()
    path = root / "mechanics-receipt.json"
    receipt = {"schema_version": "fixture.v1", "content_sha256": "fixture"}

    vertical_driver._publish_receipt_idempotent(path, receipt)
    vertical_driver._publish_receipt_idempotent(path, receipt)

    path.write_text("{}", encoding="utf-8")
    with pytest.raises(
        vertical_driver.VerticalAcceptanceError,
        match="differs from exact publication bytes",
    ):
        vertical_driver._publish_receipt_idempotent(path, receipt)


def test_crossover_not_replayed_receipt_cannot_claim_finalizer(tmp_path: Path) -> None:
    receipt = _rewrite_receipt(
        consumers.CROSSOVER_COMPATIBILITY_RECEIPT,
        tmp_path / "not-replayed.json",
        lambda document: document.__setitem__("replay_status", "not_replayed"),
    )
    with pytest.raises(consumers.ConsumerAdmissionError, match="not replayed"):
        _crossover_cpu(tmp_path / "output", receipt)


def test_binding_helpers_include_command_attestation_denominator() -> None:
    support_manifest = capture_binding_manifest(consumers.support_binding_requests())
    support_names = set(support_manifest.names)
    crossover_names = set(
        capture_binding_manifest(consumers.crossover_binding_requests()).names
    )
    assert "support_command_consumer_validator_stdout" in support_names
    assert "support_command_model_free_finalizer_receipt" in support_names
    assert "crossover_command_downstream_validator_generator" in crossover_names
    assert "consumer_admission_acceptance" in support_names & crossover_names
    verifier_names = {
        "admission_core",
        "evidence_journal_core",
        "json_values_core",
    }
    assert verifier_names <= support_names
    assert verifier_names <= crossover_names

    entries = {
        str(entry["name"]): entry
        for entry in support_manifest.to_mapping()["bindings"]
    }
    consumers._require_admission_runtime_bindings(entries)

    missing = dict(entries)
    missing.pop("admission_core")
    with pytest.raises(consumers.ConsumerAdmissionError, match="lacks admission_core"):
        consumers._require_admission_runtime_bindings(missing)

    drifted = {name: dict(entry) for name, entry in entries.items()}
    drifted["admission_core"]["sha256"] = "0" * 64
    with pytest.raises(consumers.ConsumerAdmissionError, match="identity drifted"):
        consumers._require_admission_runtime_bindings(drifted)
