from __future__ import annotations

from dataclasses import dataclass
import gc
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import weakref

import pytest

from src.artifacts.json_values import json_sha256
from scripts.research.human13_one_image_services import (
    AdmittedTask5RuntimeEvidence,
    ExistingOwnersProductionBackend,
    ProductionAcquisition,
    ProductionOneImageBackend,
    ProductionOneImageServices,
    Task5ProductionContextFailureReceipt,
    Task5ProductionContextUnavailable,
)
import scripts.research.human13_one_image_services as service_owner
from scripts.research.run_human13_all_hf_shared_surface_vertical import (
    DualGPUResourceReceipt,
    EntryConfig,
    GPUResource,
    OneImageTerminalReceipt,
    OutputRootReceipt,
    ResourceReceipt,
    SourceAssemblyReceipt,
)


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _config(root: Path) -> EntryConfig:
    return EntryConfig.from_yaml(
        Path(
            "configs/coordexp_swift/research/"
            "human13_all_hf_shared_surface_vertical/01_image1584.yaml"
        )
    )


def _resources() -> DualGPUResourceReceipt:
    return DualGPUResourceReceipt(
        cards=(
            GPUResource(0, 80 << 30, 70 << 30),
            GPUResource(1, 80 << 30, 70 << 30),
        )
    )


def _source_receipt(config: EntryConfig) -> SourceAssemblyReceipt:
    base = SourceAssemblyReceipt(
        source_plan_sha256=config.content_sha256,
        training_gpu=0,
        audit_gpu=1,
        training_surface="bf16/flash_attention_2",
        audit_surface="fp32/sdpa/batch1",
        checkpoint_path=config.source_checkpoint_path,
        base_model_path=config.base_model_path,
        adapter_path=config.adapter_path,
        special_embedding_path=config.special_embedding_path,
        adapter_sha256=config.source_adapter_sha256,
        special_embedding_sha256=config.special_embedding_sha256,
        checkpoint_sha256="d" * 64,
        tokenizer_sha256="5" * 64,
        prompt_policy_fingerprint="6" * 64,
        panel_sha256="4" * 64,
        image_sha256="3" * 64,
        manifest_sha256=config.manifest_sha256,
        source_validation_sha256="0" * 64,
        assembly_receipt_sha256="0" * 64,
    )
    from dataclasses import replace

    return replace(
        base,
        source_validation_sha256=base.source_validation_content_sha256,
        assembly_receipt_sha256=base.assembly_receipt_content_sha256,
    )


@dataclass
class _Adapter:
    events: list[str]
    rollback_calls: int = 0

    def apply_private_proposal(self) -> object:
        self.events.append("adapter_apply")
        return SimpleNamespace(
            status="private_proposal_applied", content_sha256="7" * 64
        )

    def rollback_private_proposal(self) -> object:
        self.rollback_calls += 1
        self.events.append("adapter_rollback")
        return SimpleNamespace(
            status="applied_and_rolled_back",
            rollback_decision="rejected_restored",
            source_parameter_sha256="8" * 64,
            restored_parameter_sha256="8" * 64,
            source_state_digest="9" * 64,
            restored_state_digest="9" * 64,
            content_sha256="a" * 64,
        )


class _Backend(ProductionOneImageBackend):
    def __init__(self, *, fail: str | None = None) -> None:
        self.events: list[str] = []
        self.fail = fail
        self.adapter = _Adapter(self.events)

    def preflight_source_assembly(self, config: EntryConfig, resources: Any) -> Any:
        del resources
        self.events.append("preflight_source")
        return _source_receipt(config)

    def open_training(self, config: EntryConfig, resources: Any) -> object:
        del config, resources
        self.events.append("open_training")
        return SimpleNamespace(model=object())

    def open_audit(self, config: EntryConfig, resources: Any) -> object:
        del config, resources
        self.events.append("open_audit")
        return object()

    def source_audit(
        self, session: object, repetition_penalty: float
    ) -> dict[str, Any]:
        del session
        self.events.append(f"source_audit:{repetition_penalty}")
        if self.fail == "source_audit":
            raise RuntimeError("source reconciliation failure")
        return {"rp": repetition_penalty}

    def acquire_and_replay(
        self, session: object, config: EntryConfig
    ) -> ProductionAcquisition:
        del session, config
        self.events.append("acquire_replay")
        return ProductionAcquisition(
            parity_passed=True,
            trusted_h_owner_ids=("h-1",),
            cuda_proposal_input=SimpleNamespace(
                surface_identity=object(),
                sampled_groups=(1, 2, 3, 4),
                replay_groups=(1, 2, 3, 4),
                replay_logprob_tensors={"a": object()},
                trajectory_ledger=object(),
                compiler_ledger=object(),
            ),
            task2_resource_sha256="b" * 64,
            trajectory_ledger_sha256="c" * 64,
            compiler_ledger_sha256="d" * 64,
        )

    def build_cuda_adapter(self, proposal_input: object) -> _Adapter:
        assert proposal_input is not None
        self.events.append("build_adapter")
        return self.adapter

    def write_private_checkpoint(
        self, session: object, proposal: object, output_root: Path
    ) -> object:
        del session, proposal
        self.events.append("write_private")
        if self.fail == "writer":
            raise RuntimeError("private writer failure")
        checkpoint = output_root / "private" / "checkpoints" / "step-1"
        checkpoint.mkdir(parents=True)
        (checkpoint / "payload").write_text("private", encoding="utf-8")
        return SimpleNamespace(
            checkpoint_path=str(checkpoint),
            checkpoint_payload_sha256="e" * 64,
        )

    def proposal_audit(
        self, session: object, private: object, repetition_penalty: float
    ) -> dict[str, Any]:
        del session, private
        self.events.append(f"proposal_audit:{repetition_penalty}")
        if self.fail == "audit":
            raise RuntimeError("audit failure")
        return {"rp": repetition_penalty}

    def reproduce_source(
        self, session: object, repetition_penalties: tuple[float, float]
    ) -> dict[float, dict[str, Any]]:
        del session
        self.events.append("source_reproduction")
        return {
            rp: {"rp": rp if self.fail != "reproduction" else -1.0}
            for rp in repetition_penalties
        }

    def cleanup_private_checkpoint(self, private: Any) -> None:
        self.events.append("cleanup_private")
        if self.fail == "cleanup":
            raise RuntimeError("private cleanup failure")
        path = Path(private.checkpoint_path)
        for item in sorted(path.rglob("*"), reverse=True):
            if item.is_file():
                item.unlink()
            else:
                item.rmdir()
        path.rmdir()

    def close_training(self, session: object) -> None:
        del session
        self.events.append("close_training")
        if self.fail == "close_training":
            raise RuntimeError("training close failure")

    def close_training_failed(self, session: object) -> object:
        del session
        self.events.append("close_training_failed")
        payload = {
            "sample_forward_count": 0,
            "replay_forward_count": 0,
            "source_owner_forward_count": 2,
            "total_forward_count": 2,
            "no_cache_forward_count": 2,
            "sampled_group_sha256s": [],
            "replay_group_sha256s": [],
            "model_object_id": 17,
            "retained_graph_count": 0,
            "session_held_reference_count": 0,
            "cleanup_state": "closed",
            "cleanup_reason": "failed",
            "cleanup_failures": [],
            "cleanup_call_count": 1,
        }
        return SimpleNamespace(
            to_dict=lambda: payload
            | {"content_sha256": json_sha256(payload)}
        )

    def close_audit(self, session: object) -> None:
        del session
        self.events.append("close_audit")


def _stale(path: Path) -> bytes:
    value = {
        "run_id": "20260816T032422Z-pid377949",
        "pid": 377949,
        "config_sha256": "1" * 64,
        "manifest_sha256": "2" * 64,
        "model_actions": {
            "model_loads": 0,
            "forwards": 0,
            "backwards": 0,
            "optimizer_steps": 0,
        },
    }
    payload = (json.dumps(value, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    return payload


def _services(
    tmp_path: Path,
    backend: _Backend,
    *,
    successor_name: str = "one-image-recovery-attempt-0001",
    phase_writer: Any | None = None,
) -> tuple[ProductionOneImageServices, Path, Path]:
    stale = tmp_path / "one-image" / "run-reservation.json"
    successor = tmp_path / successor_name
    if not stale.exists():
        _stale(stale)
    return (
        ProductionOneImageServices(
            backend=backend,
            reservation_mode="lost_owner_recovery",
            stale_reservation_path=stale,
            successor_root=successor,
            attempt_id="attempt-0001",
            recovery_authority="test_explicit_owner",
            pid_is_alive=lambda pid: False,
            phase_writer=phase_writer,
        ),
        stale,
        successor,
    )


def test_stale_reservation_recovery_is_parent_linked_append_only_and_dual_gpu_bound(
    tmp_path: Path,
) -> None:
    backend = _Backend()
    services, stale, successor = _services(tmp_path, backend)
    before = stale.read_bytes()

    services.preflight_source_assembly(_config(tmp_path), _resources())

    assert stale.read_bytes() == before
    recovery = json.loads((successor / "reservation-recovery.v1.json").read_text())
    reservation = json.loads((successor / "run-reservation.json").read_text())
    assert recovery["parent_reservation_sha256"] == _digest_bytes(before)
    assert recovery["parent_pid"] == 377949
    assert recovery["parent_owner_lost"] is True
    assert recovery["retry_ceiling"] == 1
    assert reservation["training_gpu"] == 0
    assert reservation["audit_gpu"] == 1
    assert reservation["recovery_successor_sha256"] == recovery["content_sha256"]
    with pytest.raises(RuntimeError, match="already reserved"):
        services.preflight_source_assembly(_config(tmp_path), _resources())
    second = ProductionOneImageServices(
        backend=_Backend(),
        reservation_mode="lost_owner_recovery",
        stale_reservation_path=stale,
        successor_root=successor,
        attempt_id="attempt-0001",
        recovery_authority="test_explicit_owner",
        pid_is_alive=lambda pid: False,
    )
    with pytest.raises(RuntimeError, match="already reserved"):
        second.preflight_source_assembly(_config(tmp_path), _resources())


def test_fixed_retry_ceiling_rejects_second_fresh_successor_for_same_parent(
    tmp_path: Path,
) -> None:
    first, stale, first_root = _services(
        tmp_path,
        _Backend(),
        successor_name="one-image-recovery-attempt-0001",
    )
    parent_before = stale.read_bytes()
    first.preflight_source_assembly(_config(tmp_path), _resources())
    assert first_root.exists()

    second, _same_stale, second_root = _services(
        tmp_path,
        _Backend(),
        successor_name="one-image-recovery-attempt-0002",
    )
    with pytest.raises(RuntimeError, match="recovery claim"):
        second.preflight_source_assembly(_config(tmp_path), _resources())

    assert stale.read_bytes() == parent_before
    assert not second_root.exists()


def test_append_only_recovery_can_chain_from_a_zero_action_successor(
    tmp_path: Path,
) -> None:
    first, _stale, first_root = _services(
        tmp_path,
        _Backend(),
        successor_name="one-image-recovery-attempt-0001",
    )
    first.preflight_source_assembly(_config(tmp_path), _resources())
    first_reservation = json.loads(
        (first_root / "run-reservation.json").read_text()
    )
    assert isinstance(first_reservation["pid"], int)

    second_root = tmp_path / "one-image-recovery-attempt-0002"
    second = ProductionOneImageServices(
        backend=_Backend(),
        reservation_mode="lost_owner_recovery",
        stale_reservation_path=first_root / "run-reservation.json",
        successor_root=second_root,
        attempt_id="attempt-0002",
        recovery_authority="test_explicit_owner",
        pid_is_alive=lambda pid: False,
    )
    second.preflight_source_assembly(_config(tmp_path), _resources())

    recovery = json.loads(
        (second_root / "reservation-recovery.v1.json").read_text()
    )
    assert recovery["parent_reservation_path"] == str(
        first_root / "run-reservation.json"
    )
    assert recovery["parent_owner_lost"] is True
    assert (first_root / ".reservation-recovery-claims").is_dir()


def test_legacy_successor_can_recover_with_explicit_lost_owner_pid_witness(
    tmp_path: Path,
) -> None:
    stale = tmp_path / "legacy-successor" / "run-reservation.json"
    _stale(stale)
    legacy_payload = json.loads(stale.read_text())
    del legacy_payload["pid"]
    stale.write_text(json.dumps(legacy_payload, sort_keys=True) + "\n")
    successor = tmp_path / "legacy-successor-next"
    services = ProductionOneImageServices(
        backend=_Backend(),
        reservation_mode="lost_owner_recovery",
        stale_reservation_path=stale,
        successor_root=successor,
        attempt_id="attempt-legacy-next",
        recovery_authority="test_explicit_owner",
        stale_owner_pid=377949,
        pid_is_alive=lambda pid: False,
    )

    services.preflight_source_assembly(_config(tmp_path), _resources())

    recovery = json.loads(
        (successor / "reservation-recovery.v1.json").read_text()
    )
    assert recovery["parent_pid"] == 377949
    assert recovery["parent_pid_source"] == "explicit_recovery_witness"


def test_fresh_primary_reservation_precedes_backend_and_binds_phase_identity(
    tmp_path: Path,
) -> None:
    root = tmp_path / "fresh-primary"

    class InspectingBackend(_Backend):
        def preflight_source_assembly(
            self, config: EntryConfig, resources: Any
        ) -> Any:
            reservation = json.loads((root / "run-reservation.json").read_text())
            assert reservation["reservation_mode"] == "fresh_primary"
            assert all(value == 0 for value in reservation["model_actions"].values())
            return super().preflight_source_assembly(config, resources)

    backend = InspectingBackend()
    services = ProductionOneImageServices(
        backend=backend,
        reservation_mode="fresh_primary",
        stale_reservation_path=None,
        successor_root=root,
        attempt_id="fresh-attempt",
        recovery_authority=None,
    )

    services.preflight_source_assembly(_config(tmp_path), _resources())

    identity = services.reservation_identity
    assert isinstance(identity, service_owner.RunReservationIdentity)
    assert identity.reservation_mode == "fresh_primary"
    assert identity.output_root == str(root.resolve())
    assert service_owner.RunReservationIdentity.from_dict(identity.to_dict()) == identity
    first_phase = json.loads(next((root / "receipts").glob("*.json")).read_text())
    assert first_phase["reservation_identity"] == identity.to_dict()
    assert backend.events == ["preflight_source"]


def test_fresh_primary_collision_has_zero_actions_and_no_admitted_identity(
    tmp_path: Path,
) -> None:
    root = tmp_path / "fresh-primary"
    root.mkdir()
    backend = _Backend()
    services = ProductionOneImageServices(
        backend=backend,
        reservation_mode="fresh_primary",
        stale_reservation_path=None,
        successor_root=root,
        attempt_id="fresh-attempt",
        recovery_authority=None,
    )

    with pytest.raises(FileExistsError):
        services.preflight_source_assembly(_config(tmp_path), _resources())

    assert services.reservation_identity is None
    assert all(value == 0 for value in services.action_counters().values())
    assert backend.events == []


@pytest.mark.parametrize("failure_point", ["reservation_write", "parent_fsync"])
def test_fresh_primary_admission_fault_does_not_strand_unique_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    root = tmp_path / "fresh-primary"
    backend = _Backend()
    services = ProductionOneImageServices(
        backend=backend,
        reservation_mode="fresh_primary",
        stale_reservation_path=None,
        successor_root=root,
        attempt_id="fresh-attempt",
        recovery_authority=None,
    )
    original_writer = service_owner._write_exclusive_json
    original_fsync = service_owner._fsync_directory

    def failing_writer(path: Path, value: Any) -> None:
        if path.name == "run-reservation.json":
            raise OSError("injected reservation write failure")
        original_writer(path, value)

    def failing_fsync(path: Path) -> None:
        if path == root.parent:
            raise OSError("injected parent fsync failure")
        original_fsync(path)

    if failure_point == "reservation_write":
        monkeypatch.setattr(service_owner, "_write_exclusive_json", failing_writer)
        expected = "reservation write failure"
    else:
        monkeypatch.setattr(service_owner, "_fsync_directory", failing_fsync)
        expected = "parent fsync failure"

    with pytest.raises(OSError, match=expected):
        services.preflight_source_assembly(_config(tmp_path), _resources())

    assert services.reservation_identity is None
    assert all(value == 0 for value in services.action_counters().values())
    assert backend.events == []
    assert not root.exists()


def test_staging_cleanup_failure_never_masks_primary_reservation_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PrimaryWriteError(OSError):
        pass

    class CleanupError(OSError):
        pass

    root = tmp_path / "fresh-primary"
    services = ProductionOneImageServices(
        backend=_Backend(),
        reservation_mode="fresh_primary",
        stale_reservation_path=None,
        successor_root=root,
        attempt_id="fresh-attempt",
        recovery_authority=None,
    )

    def fail_reservation_write(path: Path, value: Any) -> None:
        del value
        if path.name == "run-reservation.json":
            raise PrimaryWriteError("primary reservation write failed")
        raise AssertionError(path)

    def fail_staging_cleanup(path: Path) -> None:
        del path
        raise CleanupError("staging cleanup failed")

    monkeypatch.setattr(
        service_owner, "_write_exclusive_json", fail_reservation_write
    )
    monkeypatch.setattr(service_owner.shutil, "rmtree", fail_staging_cleanup)

    with pytest.raises(
        PrimaryWriteError, match="primary reservation write failed"
    ) as caught:
        services.preflight_source_assembly(_config(tmp_path), _resources())

    assert any("staging cleanup failed" in note for note in caught.value.__notes__)
    assert services.reservation_identity is None
    assert all(value == 0 for value in services.action_counters().values())
    assert not root.exists()


@pytest.mark.parametrize("reservation_mode", ["fresh_primary", "lost_owner_recovery"])
def test_both_reservation_modes_reload_hash_and_bind_phase_identity(
    tmp_path: Path, reservation_mode: Any
) -> None:
    root = tmp_path / f"root-{reservation_mode}"
    stale = tmp_path / "stale" / "run-reservation.json"
    if reservation_mode == "lost_owner_recovery":
        _stale(stale)
    services = ProductionOneImageServices(
        backend=_Backend(),
        reservation_mode=reservation_mode,
        stale_reservation_path=(
            stale if reservation_mode == "lost_owner_recovery" else None
        ),
        successor_root=root,
        attempt_id=f"attempt-{reservation_mode}",
        recovery_authority=(
            "test_explicit_owner"
            if reservation_mode == "lost_owner_recovery"
            else None
        ),
        pid_is_alive=lambda pid: False,
    )

    services.preflight_source_assembly(_config(tmp_path), _resources())

    identity = services.reservation_identity
    assert isinstance(identity, service_owner.RunReservationIdentity)
    reservation = json.loads((root / "run-reservation.json").read_text())
    reservation_payload = {
        key: value for key, value in reservation.items() if key != "content_sha256"
    }
    assert reservation["content_sha256"] == json_sha256(reservation_payload)
    assert identity.reservation_sha256 == reservation["content_sha256"]
    reloaded = service_owner.RunReservationIdentity.from_dict(identity.to_dict())
    assert reloaded == identity
    assert reloaded.content_sha256 == identity.content_sha256
    resource = ResourceReceipt(
        _resources(),
        OutputRootReceipt(identity.output_root, False),
        phase_count=len(services.phase_receipt_sha256s),
        retry_count=0,
        promoted_checkpoint=False,
        reservation_identity=identity,
        sampled_request_count=0,
        sampled_group_count=0,
        sample_forward_count=0,
        replay_forward_count=0,
        source_owner_forward_count=0,
        total_forward_count=0,
        no_cache_forward_count=0,
        backward_count=0,
    )
    assert ResourceReceipt.from_dict(resource.to_dict()) == resource
    for phase_path in sorted((root / "receipts").glob("*.json")):
        assert json.loads(phase_path.read_text())["reservation_identity"] == identity.to_dict()


def test_missing_live_owner_context_is_written_as_typed_phase_receipt(
    tmp_path: Path,
) -> None:
    receipt = Task5ProductionContextFailureReceipt(
        reason_code="live_task2_owner_publications_unavailable",
        config_sha256="1" * 64,
        manifest_sha256="2" * 64,
        source_identity_sha256="3" * 64,
        sampled_group_sha256s=("4" * 64,) * 4,
        replay_group_sha256s=("5" * 64,) * 4,
        sampled_group_object_ids=(1, 2, 3, 4),
        replay_group_object_ids=(5, 6, 7, 8),
        replay_tensor_object_ids=(("live", 11),),
        assembly_object_id=9,
        session_object_id=10,
        model_object_id=12,
        optimizer_object_id=13,
        manifest_object_id=14,
        manifest_image_object_id=15,
        process_id=16,
    )

    class MissingContextBackend(_Backend):
        def acquire_and_replay(
            self, session: object, config: EntryConfig
        ) -> ProductionAcquisition:
            del session, config
            raise Task5ProductionContextUnavailable(receipt)

    services, _stale, successor = _services(tmp_path, MissingContextBackend())
    config = _config(tmp_path)
    resources = _resources()
    services.preflight_source_assembly(config, resources)
    training = services.open_training(config, resources)

    with pytest.raises(Task5ProductionContextUnavailable):
        services.acquire_and_replay(training, config)

    phase = json.loads(sorted((successor / "receipts").glob("*.json"))[-1].read_text())
    assert phase["phase"] == "k16_acquisition_replay"
    assert phase["status"] == "runtime_context_failure"
    assert phase["evidence"]["context_failure_receipt"] == receipt.to_dict()


def test_native_owner_admission_failure_preserves_reason_and_disposition(
    tmp_path: Path,
) -> None:
    from scripts.research.human13_hf_native_one_image_owner import (
        HFNativeOneImageOwnerError,
    )

    error = HFNativeOneImageOwnerError(
        "canonical projection differs from admitted replay",
        disposition="canonical_projection_lineage_mismatch",
    )

    class NativeFailureBackend(_Backend):
        def acquire_and_replay(
            self, session: object, config: EntryConfig
        ) -> ProductionAcquisition:
            del session, config
            raise error

    services, _stale, successor = _services(tmp_path, NativeFailureBackend())
    config = _config(tmp_path)
    resources = _resources()
    services.preflight_source_assembly(config, resources)
    training = services.open_training(config, resources)

    with pytest.raises(HFNativeOneImageOwnerError) as caught:
        services.acquire_and_replay(training, config)

    assert caught.value is error
    assert caught.value.reason == "canonical projection differs from admitted replay"
    assert caught.value.disposition == "canonical_projection_lineage_mismatch"
    phase = json.loads(sorted((successor / "receipts").glob("*.json"))[-1].read_text())
    assert phase["phase"] == "k16_acquisition_replay"
    assert phase["status"] == "hf_native_owner_failure"
    assert phase["evidence"]["owner_error"] == {
        "type": "HFNativeOneImageOwnerError",
        "reason": "canonical projection differs from admitted replay",
        "disposition": "canonical_projection_lineage_mismatch",
    }


def test_source_audit_forward_is_receipted_when_owner_prepare_fails(
    tmp_path: Path,
) -> None:
    class SourceOwnerFailureBackend(_Backend):
        def source_audit(
            self, session: object, repetition_penalty: float
        ) -> dict[str, Any]:
            result = super().source_audit(session, repetition_penalty)
            if repetition_penalty == 1.1:
                error = RuntimeError("source owner preparation failed")
                setattr(error, "_source_audit_forward_observed", True)
                setattr(error, "_source_audit_result", result)
                setattr(error, "_source_audit_repetition_penalty", repetition_penalty)
                raise error
            return result

    backend = SourceOwnerFailureBackend()
    services, _stale, successor = _services(tmp_path, backend)
    config = _config(tmp_path)
    resources = _resources()
    services.preflight_source_assembly(config, resources)
    services.open_training(config, resources)
    audit = services.open_audit(config, resources)
    services.source_audit(audit, 1.0)

    with pytest.raises(RuntimeError, match="source owner preparation failed"):
        services.source_audit(audit, 1.1)

    assert services.action_counters()["forwards"] == 2
    receipts = sorted((successor / "receipts").glob("*.json"))
    failed = json.loads(
        next(path for path in receipts if "source_audit_rp_1.1" in path.name).read_text()
    )
    assert failed["status"] == "failed"
    assert failed["evidence"]["source_audit_forward_observed"] is True
    assert failed["evidence"]["repetition_penalty"] == 1.1


def test_source_only_failure_uses_aborted_training_close_without_k16_claim(
    tmp_path: Path,
) -> None:
    backend = _Backend(fail="source_audit")
    services, _stale_path, _successor = _services(tmp_path, backend)
    config = _config(tmp_path)
    resources = _resources()
    services.preflight_source_assembly(config, resources)
    training = services.open_training(config, resources)
    audit = services.open_audit(config, resources)

    with pytest.raises(RuntimeError, match="source reconciliation failure"):
        services.source_audit(audit, 1.0)

    services.close(training, audit)

    assert backend.events[-2:] == ["close_audit", "close_training_failed"]
    assert services.action_counters()["backwards"] == 0
    assert services.action_counters()["optimizer_steps"] == 0


def test_production_owner_orders_apply_checkpoint_audits_then_one_rollback_and_reproduction(
    tmp_path: Path,
) -> None:
    backend = _Backend()
    services, _stale_path, successor = _services(tmp_path, backend)
    config = _config(tmp_path)
    resources = _resources()
    services.preflight_source_assembly(config, resources)
    training = services.open_training(config, resources)
    audit = services.open_audit(config, resources)
    services.source_audit(audit, 1.0)
    services.source_audit(audit, 1.1)
    acquisition = services.acquire_and_replay(training, config)

    proposal = services.apply_private_update(training, acquisition, config)
    private = services.write_private_proposal(training, proposal, successor)
    services.proposal_audit(audit, private, 1.0)
    services.proposal_audit(audit, private, 1.1)
    assert services.rollback_and_reproduce_source(training, proposal) is True
    services.cleanup_private_proposal(private)
    services.close(training, audit)

    assert backend.events == [
        "preflight_source",
        "open_training",
        "open_audit",
        "source_audit:1.0",
        "source_audit:1.1",
        "acquire_replay",
        "build_adapter",
        "adapter_apply",
        "write_private",
        "proposal_audit:1.0",
        "proposal_audit:1.1",
        "adapter_rollback",
        "source_reproduction",
        "cleanup_private",
        "close_audit",
        "close_training",
    ]
    assert backend.adapter.rollback_calls == 1
    assert services.retry_count == 0
    assert services.fallback_used is False
    assert services.action_counters()["backwards"] == 1
    assert services.action_counters()["optimizer_steps"] == 1
    phase_files = sorted((successor / "receipts").glob("*.json"))
    assert phase_files


def test_audit_failure_still_allows_exactly_one_explicit_rollback(
    tmp_path: Path,
) -> None:
    backend = _Backend(fail="audit")
    services, _stale_path, successor = _services(tmp_path, backend)
    config = _config(tmp_path)
    resources = _resources()
    services.preflight_source_assembly(config, resources)
    training = services.open_training(config, resources)
    audit = services.open_audit(config, resources)
    services.source_audit(audit, 1.0)
    services.source_audit(audit, 1.1)
    acquisition = services.acquire_and_replay(training, config)
    proposal = services.apply_private_update(training, acquisition, config)
    private = services.write_private_proposal(training, proposal, successor)

    with pytest.raises(RuntimeError, match="audit failure"):
        services.proposal_audit(audit, private, 1.0)
    assert services.rollback_and_reproduce_source(training, proposal) is True
    assert backend.adapter.rollback_calls == 1


def test_parity_failure_cannot_build_adapter_or_run_backward(tmp_path: Path) -> None:
    backend = _Backend()
    services, _stale_path, _successor = _services(tmp_path, backend)
    acquisition = backend.acquire_and_replay(object(), _config(tmp_path))
    failed = ProductionAcquisition(
        parity_passed=False,
        trusted_h_owner_ids=acquisition.trusted_h_owner_ids,
        cuda_proposal_input=acquisition.cuda_proposal_input,
        task2_resource_sha256=acquisition.task2_resource_sha256,
        trajectory_ledger_sha256=acquisition.trajectory_ledger_sha256,
        compiler_ledger_sha256=acquisition.compiler_ledger_sha256,
    )

    with pytest.raises(RuntimeError, match="parity failure"):
        services.apply_private_update(object(), failed, _config(tmp_path))

    assert "build_adapter" not in backend.events
    assert services.action_counters()["backwards"] == 0


def test_private_writer_failure_leaves_applied_proposal_for_one_rollback(
    tmp_path: Path,
) -> None:
    backend = _Backend(fail="writer")
    services, _stale_path, successor = _services(tmp_path, backend)
    config = _config(tmp_path)
    resources = _resources()
    services.preflight_source_assembly(config, resources)
    training = services.open_training(config, resources)
    audit = services.open_audit(config, resources)
    services.source_audit(audit, 1.0)
    services.source_audit(audit, 1.1)
    acquisition = services.acquire_and_replay(training, config)
    proposal = services.apply_private_update(training, acquisition, config)

    with pytest.raises(RuntimeError, match="private writer failure"):
        services.write_private_proposal(training, proposal, successor)

    assert services.rollback_and_reproduce_source(training, proposal) is True
    assert backend.adapter.rollback_calls == 1


def test_rollback_failure_is_terminal_and_never_retried(tmp_path: Path) -> None:
    backend = _Backend()
    services, _stale_path, successor = _services(tmp_path, backend)
    config = _config(tmp_path)
    resources = _resources()
    services.preflight_source_assembly(config, resources)
    training = services.open_training(config, resources)
    audit = services.open_audit(config, resources)
    services.source_audit(audit, 1.0)
    services.source_audit(audit, 1.1)
    acquisition = services.acquire_and_replay(training, config)
    proposal = services.apply_private_update(training, acquisition, config)

    def fail_once() -> object:
        backend.adapter.rollback_calls += 1
        raise RuntimeError("rollback failure")

    backend.adapter.rollback_private_proposal = fail_once  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="rollback failure"):
        services.rollback_and_reproduce_source(training, proposal)
    with pytest.raises(RuntimeError, match="already attempted"):
        services.rollback_and_reproduce_source(training, proposal)

    assert backend.adapter.rollback_calls == 1
    assert not (successor / "terminal.json").exists()


def test_source_reproduction_requires_exact_dual_rp_output_surface(
    tmp_path: Path,
) -> None:
    backend = _Backend(fail="reproduction")
    services, _stale_path, _successor = _services(tmp_path, backend)
    config = _config(tmp_path)
    resources = _resources()
    services.preflight_source_assembly(config, resources)
    training = services.open_training(config, resources)
    audit = services.open_audit(config, resources)
    services.source_audit(audit, 1.0)
    services.source_audit(audit, 1.1)
    acquisition = services.acquire_and_replay(training, config)
    proposal = services.apply_private_update(training, acquisition, config)

    assert services.rollback_and_reproduce_source(training, proposal) is False
    assert backend.adapter.rollback_calls == 1


def test_post_apply_phase_write_failure_self_rolls_back_before_return(
    tmp_path: Path,
) -> None:
    backend = _Backend()

    def phase_writer(path: Path, value: dict[str, Any]) -> None:
        if path.name.endswith("private_update_applied.json"):
            raise OSError("injected phase journal failure")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    services, _stale_path, _successor = _services(
        tmp_path, backend, phase_writer=phase_writer
    )
    config = _config(tmp_path)
    resources = _resources()
    services.preflight_source_assembly(config, resources)
    training = services.open_training(config, resources)
    audit = services.open_audit(config, resources)
    services.source_audit(audit, 1.0)
    services.source_audit(audit, 1.1)
    acquisition = services.acquire_and_replay(training, config)

    with pytest.raises(OSError, match="phase journal failure"):
        services.apply_private_update(training, acquisition, config)

    assert backend.adapter.rollback_calls == 1
    assert backend.events[-2:] == ["adapter_rollback", "source_reproduction"]
    assert services.source_reproduced is True
    assert services.retry_count == 0


def test_production_owner_rejects_missing_task2_or_task3_lineage_before_backward(
    tmp_path: Path,
) -> None:
    backend = _Backend()
    services, _stale_path, successor = _services(tmp_path, backend)
    config = _config(tmp_path)
    services.preflight_source_assembly(config, _resources())
    invalid = ProductionAcquisition(
        parity_passed=True,
        trusted_h_owner_ids=("h-1",),
        cuda_proposal_input=SimpleNamespace(
            surface_identity=None,
            sampled_groups=(),
            replay_groups=(),
            replay_logprob_tensors={},
            trajectory_ledger=None,
            compiler_ledger=None,
        ),
        task2_resource_sha256="b" * 64,
        trajectory_ledger_sha256="c" * 64,
        compiler_ledger_sha256="d" * 64,
    )

    from scripts.research.human13_hf_native_one_image_owner import (
        HFNativeOneImageOwnerError,
    )

    with pytest.raises(HFNativeOneImageOwnerError) as caught:
        services.apply_private_update(object(), invalid, config)
    assert caught.value.reason == "complete current Task2/Task3 lineage is required"
    assert caught.value.disposition == "task2_task3_lineage_mismatch"
    assert "adapter_apply" not in backend.events
    phase = json.loads(sorted((successor / "receipts").glob("*.json"))[-1].read_text())
    assert phase["phase"] == "private_update_applied"
    assert phase["status"] == "hf_native_owner_failure"
    assert phase["evidence"]["owner_error"] == {
        "type": "HFNativeOneImageOwnerError",
        "reason": "complete current Task2/Task3 lineage is required",
        "disposition": "task2_task3_lineage_mismatch",
    }


def test_close_attempts_both_sessions_and_surfaces_typed_failure(
    tmp_path: Path,
) -> None:
    backend = _Backend(fail="close_training")
    services, _stale_path, _successor = _services(tmp_path, backend)

    with pytest.raises(RuntimeError, match="training close failure"):
        services.close(object(), object())

    assert backend.events == ["close_audit", "close_training"]


@pytest.mark.parametrize("close_fails", [False, True])
def test_close_releases_adapter_proposal_and_runtime_evidence_on_every_exit(
    tmp_path: Path,
    close_fails: bool,
) -> None:
    captured: dict[str, weakref.ReferenceType[object]] = {}

    class RuntimeEvidence:
        surface_identity = object()
        sampled_groups = (1, 2, 3, 4)
        replay_groups = (1, 2, 3, 4)
        replay_logprob_tensors = {"live": object()}
        trajectory_ledger = object()
        compiler_ledger = object()

    class Proposal:
        content_sha256 = "7" * 64

    class Adapter:
        def __init__(self, evidence: object) -> None:
            self.evidence = evidence

        def apply_private_proposal(self) -> object:
            proposal = Proposal()
            captured["proposal"] = weakref.ref(proposal)
            return proposal

        def rollback_private_proposal(self) -> object:
            return SimpleNamespace(
                content_sha256="8" * 64,
                rollback_decision="rejected_restored",
            )

    class WeakBackend(_Backend):
        def build_cuda_adapter(self, proposal_input: object) -> Any:
            adapter = Adapter(proposal_input)
            captured["adapter"] = weakref.ref(adapter)
            captured["evidence"] = weakref.ref(proposal_input)
            return adapter

    backend = WeakBackend(fail="close_training" if close_fails else None)
    services, _stale_path, _successor = _services(tmp_path, backend)
    config = _config(tmp_path)
    resources = _resources()
    services.preflight_source_assembly(config, resources)
    training = services.open_training(config, resources)
    audit = services.open_audit(config, resources)
    acquisition = ProductionAcquisition(
        parity_passed=True,
        trusted_h_owner_ids=("h-1",),
        cuda_proposal_input=RuntimeEvidence(),
        task2_resource_sha256="b" * 64,
        trajectory_ledger_sha256="c" * 64,
        compiler_ledger_sha256="d" * 64,
    )
    proposal = services.apply_private_update(training, acquisition, config)
    del acquisition, proposal

    if close_fails:
        with pytest.raises(RuntimeError, match="training close failure"):
            services.close(training, audit)
    else:
        services.close(training, audit)
    del training, audit
    gc.collect()

    assert set(captured) == {"adapter", "proposal", "evidence"}
    assert all(reference() is None for reference in captured.values())


def test_private_cleanup_failure_is_journaled_and_not_retried(tmp_path: Path) -> None:
    backend = _Backend(fail="cleanup")
    services, _stale_path, successor = _services(tmp_path, backend)
    config = _config(tmp_path)
    resources = _resources()
    services.preflight_source_assembly(config, resources)
    training = services.open_training(config, resources)
    audit = services.open_audit(config, resources)
    services.source_audit(audit, 1.0)
    services.source_audit(audit, 1.1)
    acquisition = services.acquire_and_replay(training, config)
    proposal = services.apply_private_update(training, acquisition, config)
    private = services.write_private_proposal(training, proposal, successor)
    assert services.rollback_and_reproduce_source(training, proposal) is True

    with pytest.raises(RuntimeError, match="private cleanup failure"):
        services.cleanup_private_proposal(private)

    cleanup_receipts = sorted(
        (successor / "receipts").glob("*-private_checkpoint_cleanup.json")
    )
    assert len(cleanup_receipts) == 1
    assert json.loads(cleanup_receipts[0].read_text())["status"] == "failed"


def test_terminal_receipt_is_written_immutably_with_recovery_lineage(
    tmp_path: Path,
) -> None:
    backend = _Backend()
    services, _stale_path, successor = _services(tmp_path, backend)
    config = _config(tmp_path)
    resources = _resources()
    services.preflight_source_assembly(config, resources)
    resource = ResourceReceipt(
        resources,
        OutputRootReceipt(str(successor.resolve()), False),
        phase_count=0,
        retry_count=0,
        promoted_checkpoint=False,
        reservation_identity=services.reservation_identity,
    )
    terminal = OneImageTerminalReceipt(
        terminal_status="parity_failure",
        resource_receipt=resource,
        model_actions=services.action_counters(),
        phase_receipt_sha256s=services.phase_receipt_sha256s,
        phase_ledger_sha256=services.phase_ledger_sha256,
        failure_reason="parity mismatch",
    )

    services.persist_terminal(terminal)

    payload = json.loads((successor / "terminal.json").read_text())
    assert payload["terminal_status"] == "parity_failure"
    assert payload["attempt_id"] == "attempt-0001"
    assert payload["recovery_successor_sha256"]
    assert payload["phase_receipt_count"] == len(services.phase_receipt_sha256s)
    assert payload["phase_ledger_sha256"] == services.phase_ledger_sha256
    with pytest.raises(FileExistsError):
        services.persist_terminal(terminal)


def test_fresh_primary_failure_persists_zero_action_bound_terminal(
    tmp_path: Path,
) -> None:
    class FailingBackend(_Backend):
        def preflight_source_assembly(
            self, config: EntryConfig, resources: Any
        ) -> Any:
            del config, resources
            self.events.append("preflight_source")
            raise RuntimeError("fresh primary preflight failed")

    root = tmp_path / "fresh-primary"
    backend = FailingBackend()
    services = ProductionOneImageServices(
        backend=backend,
        reservation_mode="fresh_primary",
        stale_reservation_path=None,
        successor_root=root,
        attempt_id="fresh-attempt",
        recovery_authority=None,
    )
    config = _config(tmp_path)
    resources = _resources()

    with pytest.raises(RuntimeError, match="fresh primary preflight failed"):
        services.preflight_source_assembly(config, resources)

    identity = services.reservation_identity
    assert isinstance(identity, service_owner.RunReservationIdentity)
    assert all(value == 0 for value in services.action_counters().values())
    terminal = OneImageTerminalReceipt(
        terminal_status="update_failure",
        resource_receipt=ResourceReceipt(
            resources,
            OutputRootReceipt(identity.output_root, False),
            phase_count=len(services.phase_receipt_sha256s),
            retry_count=0,
            promoted_checkpoint=False,
            reservation_identity=identity,
            sampled_request_count=0,
            sampled_group_count=0,
            sample_forward_count=0,
            replay_forward_count=0,
            source_owner_forward_count=0,
            total_forward_count=0,
            no_cache_forward_count=0,
            backward_count=0,
        ),
        model_actions=services.action_counters(),
        phase_receipt_sha256s=services.phase_receipt_sha256s,
        phase_ledger_sha256=services.phase_ledger_sha256,
        failure_reason="RuntimeError: fresh primary preflight failed",
    )

    services.persist_terminal(terminal)

    envelope = json.loads((root / "terminal.json").read_text())
    assert envelope["reservation_identity"] == identity.to_dict()
    assert envelope["terminal"]["resource_receipt"]["reservation_identity"] == (
        identity.to_dict()
    )


def test_terminal_phase_ledger_mismatch_fails_before_publication(
    tmp_path: Path,
) -> None:
    backend = _Backend()
    services, _stale_path, successor = _services(tmp_path, backend)
    config = _config(tmp_path)
    resources = _resources()
    services.preflight_source_assembly(config, resources)
    resource = ResourceReceipt(
        resources,
        OutputRootReceipt(str(successor.resolve()), False),
        phase_count=0,
        retry_count=0,
        promoted_checkpoint=False,
        reservation_identity=services.reservation_identity,
    )
    mismatched_hashes = ("f" * 64,)
    terminal = OneImageTerminalReceipt(
        terminal_status="parity_failure",
        resource_receipt=resource,
        model_actions=services.action_counters(),
        phase_receipt_sha256s=mismatched_hashes,
        phase_ledger_sha256=json_sha256(
            {
                "schema_version": "human13_all_hf_phase_ledger.v1",
                "phase_receipt_sha256s": list(mismatched_hashes),
            }
        ),
        failure_reason="parity mismatch",
    )

    with pytest.raises(ValueError, match="phase ledger"):
        services.persist_terminal(terminal)
    assert not (successor / "terminal.json").exists()


def test_concrete_existing_owner_backend_constructs_real_seams_with_injected_boundaries(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    manifest = SimpleNamespace(images=(SimpleNamespace(image_id=1584),))
    assembly = SimpleNamespace(components=object(), model=object(), optimizer=object())
    evidence = AdmittedTask5RuntimeEvidence(
        trajectory_ledger=object(),
        compiler_ledger=object(),
        compiler_compact_logits=None,
        witness_bank=object(),
        realized_margin_probe=lambda: {},
    )
    context_requests: list[Any] = []
    close_receipt = object()

    class ContextProvider:
        def provide(self, request: object) -> AdmittedTask5RuntimeEvidence:
            events.append("runtime_context")
            context_requests.append(request)
            return evidence

    class Session:
        _live_replay_tensors = {"live": object()}

        def sample_group(self, seeds: tuple[int, ...]) -> object:
            return SimpleNamespace(seeds=seeds)

        def replay_group(self, sampled: object) -> object:
            return SimpleNamespace(sampled=sampled)

        def close(self) -> object:
            events.append("surface_close")
            return close_receipt

    session = Session()

    def runtime_factory(**kwargs: Any) -> ProductionAcquisition:
        events.append("runtime_factory")
        assert kwargs["assembly"] is assembly
        assert kwargs["session"] is session
        assert kwargs["runtime_evidence"] is evidence
        return ProductionAcquisition(
            parity_passed=True,
            trusted_h_owner_ids=("h-1",),
            cuda_proposal_input=SimpleNamespace(
                surface_identity=object(),
                sampled_groups=kwargs["sampled_groups"],
                replay_groups=kwargs["replay_groups"],
                replay_logprob_tensors=kwargs["replay_logprob_tensors"],
                trajectory_ledger=object(),
                compiler_ledger=object(),
            ),
            task2_resource_sha256="b" * 64,
            trajectory_ledger_sha256="c" * 64,
            compiler_ledger_sha256="d" * 64,
        )

    backend = ExistingOwnersProductionBackend(
        manifest=manifest,
        manifest_path=tmp_path / "manifest.json",
        repo_root=Path.cwd(),
        source_config_path=tmp_path / "source.yaml",
        runtime_factory=runtime_factory,
        runtime_context_provider=ContextProvider(),
        assemble_model=lambda *args, **kwargs: (events.append("assemble"), assembly)[1],
        build_skeletons=lambda *args, **kwargs: {1584: object()},
        open_surface=lambda *args, **kwargs: (events.append("open_surface"), session)[
            1
        ],
        evaluate_checkpoint=lambda **kwargs: (
            {"image_id": 1584, "checkpoint": str(kwargs["checkpoint_path"])},
        ),
        checkpoint_writer_factory=lambda root: object(),
        checkpoint_write=lambda writer, live_assembly: SimpleNamespace(
            checkpoint_dir=root_for_checkpoint(tmp_path)
        ),
        checkpoint_readback=lambda *args, **kwargs: object(),
        checkpoint_hasher=lambda path: "e" * 64,
    )

    handle = backend.open_training(_config(tmp_path), _resources())
    acquisition = backend.acquire_and_replay(handle, _config(tmp_path))
    observed_close = backend.close_training(handle)

    assert acquisition.parity_passed is True
    assert observed_close is close_receipt
    assert events == [
        "assemble",
        "open_surface",
        "runtime_context",
        "runtime_factory",
        "surface_close",
    ]
    assert len(context_requests) == 1
    request = context_requests[0]
    assert request.assembly is assembly
    assert request.session is session
    assert request.model is assembly.model
    assert request.optimizer is assembly.optimizer
    assert request.manifest is manifest
    assert request.manifest_image is manifest.images[0]
    assert tuple(request.sampled_groups) == tuple(
        item.sampled for item in request.replay_groups
    )
    assert request.replay_logprob_tensors is session._live_replay_tensors


def test_public_backend_default_context_provider_fails_only_at_typed_live_owner_seam(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    image = SimpleNamespace(image_id=1584, image_sha256="3" * 64)
    manifest = SimpleNamespace(images=(image,))
    assembly = SimpleNamespace(
        components=object(), model=object(), optimizer=object()
    )
    tensor = object()
    sampled_objects: list[object] = []
    replayed_objects: list[object] = []

    class Session:
        # A forged hidden attribute must never satisfy the public owner seam.
        _task5_runtime_evidence = AdmittedTask5RuntimeEvidence(
            trajectory_ledger=object(),
            compiler_ledger=object(),
            compiler_compact_logits=None,
            witness_bank=object(),
            realized_margin_probe=lambda: {},
        )
        _live_replay_tensors = {"live": tensor}

        def sample_group(self, seeds: tuple[int, ...]) -> object:
            sampled = SimpleNamespace(
                seeds=seeds,
                content_sha256=json_sha256(
                    {"kind": "sampled", "seeds": list(seeds)}
                ),
            )
            sampled_objects.append(sampled)
            return sampled

        def replay_group(self, sampled: object) -> object:
            replayed = SimpleNamespace(
                sampled=sampled,
                content_sha256=json_sha256(
                    {
                        "kind": "replayed",
                        "sampled_sha256": getattr(sampled, "content_sha256"),
                    }
                ),
            )
            replayed_objects.append(replayed)
            return replayed

    session = Session()
    factory_calls: list[dict[str, Any]] = []
    backend = ExistingOwnersProductionBackend(
        manifest=manifest,
        manifest_path=tmp_path / "manifest.json",
        repo_root=Path.cwd(),
        source_config_path=tmp_path / "source.yaml",
        runtime_factory=lambda **kwargs: factory_calls.append(kwargs),  # type: ignore[arg-type,return-value]
        assemble_model=lambda *_args, **_kwargs: assembly,
        build_skeletons=lambda *_args, **_kwargs: {1584: object()},
        open_surface=lambda *_args, **_kwargs: session,
        evaluate_checkpoint=lambda **_kwargs: ({"image_id": 1584},),
        checkpoint_writer_factory=lambda _root: object(),
        checkpoint_readback=lambda *_args, **_kwargs: object(),
        checkpoint_hasher=lambda _path: "e" * 64,
    )

    handle = backend.open_training(config, _resources())
    with pytest.raises(Task5ProductionContextUnavailable) as caught:
        backend.acquire_and_replay(handle, config)

    assert factory_calls == []
    receipt = caught.value.receipt
    assert receipt.reason_code == "live_task2_owner_publications_unavailable"
    assert receipt.config_sha256 == config.content_sha256
    assert receipt.manifest_sha256 == config.manifest_sha256
    assert receipt.source_identity_sha256 == json_sha256(
        {
            "source_checkpoint_path": config.source_checkpoint_path,
            "base_model_path": config.base_model_path,
            "adapter_path": config.adapter_path,
            "special_embedding_path": config.special_embedding_path,
            "source_adapter_sha256": config.source_adapter_sha256,
            "special_embedding_sha256": config.special_embedding_sha256,
        }
    )
    assert receipt.model_object_id == id(assembly.model)
    assert receipt.optimizer_object_id == id(assembly.optimizer)
    assert receipt.assembly_object_id == id(assembly)
    assert receipt.session_object_id == id(session)
    assert receipt.sampled_group_object_ids == tuple(map(id, sampled_objects))
    assert receipt.replay_group_object_ids == tuple(map(id, replayed_objects))
    assert receipt.replay_tensor_object_ids == (("live", id(tensor)),)
    assert len(receipt.sampled_group_sha256s) == 4
    assert len(receipt.replay_group_sha256s) == 4
    assert receipt.missing_owner_publications == (
        "hf_one_image_trajectory_credit_admission",
        "pre_acquisition_source_compiler_graph",
        "pre_acquisition_frozen_witness_and_realized_probe",
    )
    assert receipt.required_owner_phase_order == (
        "source_audits",
        "source_compiler_and_witness_freeze",
        "hf_sample_and_replay",
        "one_image_trajectory_credit_admission",
    )
    assert receipt.content_sha256 in str(caught.value)


def test_default_runtime_factory_ignores_hidden_session_runtime_evidence() -> None:
    from scripts.research import human13_one_image_services as owner

    forged = owner.AdmittedTask5RuntimeEvidence(
        trajectory_ledger=object(),
        compiler_ledger=object(),
        compiler_compact_logits=None,
        witness_bank=object(),
        realized_margin_probe=lambda: {},
    )
    with pytest.raises(
        owner.Task5RuntimeEvidenceError,
        match="explicit admitted Task2/Task3 runtime evidence is required",
    ):
        owner.default_task5_runtime_factory(
            assembly=SimpleNamespace(),
            session=SimpleNamespace(_task5_runtime_evidence=forged),
            sampled_groups=(1, 2, 3, 4),
            replay_groups=(1, 2, 3, 4),
            replay_logprob_tensors={"x": object()},
            manifest=object(),
            manifest_image=object(),
            config=SimpleNamespace(),
        )


def test_default_runtime_factory_composes_existing_admitted_task2_task3_owners() -> (
    None
):
    import importlib.util
    import sys

    from scripts.research import human13_one_image_services as owner
    from scripts.research.human13_cuda_cpu_adapter import CudaProposalInput

    fixture_path = Path(__file__).with_name("test_human13_cuda_cpu_adapter.py")
    spec = importlib.util.spec_from_file_location("_task5_cuda_fixture", fixture_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    surface, fixture = module._task2_surface(module_name="_task5_vertical_fixture")
    evidence = owner.AdmittedTask5RuntimeEvidence(
        trajectory_ledger=surface.trajectory_ledger,
        compiler_ledger=surface.compiler_ledger,
        compiler_compact_logits=surface.compiler_compact_logits,
        witness_bank=surface.witness_bank,
        realized_margin_probe=surface.realized_margin_probe,
    )
    acquisition = owner.default_task5_runtime_factory(
        assembly=SimpleNamespace(
            model=surface.model,
            optimizer=surface.optimizer,
            scheduler=None,
            runtime=None,
        ),
        session=SimpleNamespace(),
        sampled_groups=surface.sampled_groups,
        replay_groups=surface.replay_groups,
        replay_logprob_tensors=surface.replay_logprob_tensors,
        manifest=object(),
        manifest_image=SimpleNamespace(h_owner_ids=("h-1",)),
        config=SimpleNamespace(),
        runtime_evidence=evidence,
    )

    assert type(acquisition.cuda_proposal_input) is CudaProposalInput
    assert acquisition.cuda_proposal_input.objective is None
    assert (
        acquisition.cuda_proposal_input.trajectory_ledger is fixture.trajectory_ledger
    )
    assert acquisition.cuda_proposal_input.compiler_ledger is fixture.compiler_ledger
    assert acquisition.trusted_h_owner_ids == ()
    assert acquisition.sample_forward_count > 0
    assert acquisition.replay_forward_count > 0


def test_default_runtime_factory_fails_closed_without_live_ledgers() -> None:
    from scripts.research import human13_one_image_services as owner

    with pytest.raises(RuntimeError, match="admitted Task2/Task3 runtime evidence"):
        owner.default_task5_runtime_factory(
            assembly=SimpleNamespace(),
            session=SimpleNamespace(),
            sampled_groups=(1, 2, 3, 4),
            replay_groups=(1, 2, 3, 4),
            replay_logprob_tensors={"x": object()},
            manifest=object(),
            manifest_image=object(),
            config=SimpleNamespace(),
        )


def test_audit_evaluator_receives_only_image1584_with_parent_binding(
    tmp_path: Path,
) -> None:
    received: list[Any] = []
    binding = SimpleNamespace(marker="parent-binding")
    manifest = SimpleNamespace(
        binding=binding,
        images=(SimpleNamespace(image_id=1584), SimpleNamespace(image_id=2299)),
    )

    def evaluate_checkpoint(**kwargs: Any) -> tuple[dict[str, Any], ...]:
        received.append(kwargs["manifest"])
        return ({"image_id": 1584},)

    backend = ExistingOwnersProductionBackend(
        manifest=manifest,
        manifest_path=tmp_path / "manifest.json",
        repo_root=Path.cwd(),
        source_config_path=tmp_path / "source.yaml",
        runtime_factory=lambda **_kwargs: (_ for _ in ()).throw(AssertionError()),
        assemble_model=lambda *_args, **_kwargs: object(),
        build_skeletons=lambda *_args, **_kwargs: {},
        open_surface=lambda *_args, **_kwargs: object(),
        evaluate_checkpoint=evaluate_checkpoint,
        checkpoint_writer_factory=lambda _root: object(),
        checkpoint_readback=lambda *_args, **_kwargs: object(),
        checkpoint_hasher=lambda _path: "e" * 64,
    )
    handle = backend.open_audit(_config(tmp_path), _resources())
    backend.source_audit(handle, 1.0)

    assert len(received) == 1
    envelope = received[0]
    assert tuple(image.image_id for image in envelope.images) == (1584,)
    assert envelope.binding is binding
    assert envelope.parent_manifest_sha256 == _config(tmp_path).manifest_sha256


def root_for_checkpoint(tmp_path: Path) -> Path:
    path = tmp_path / "private" / "checkpoints" / "step-1"
    path.mkdir(parents=True, exist_ok=True)
    return path
