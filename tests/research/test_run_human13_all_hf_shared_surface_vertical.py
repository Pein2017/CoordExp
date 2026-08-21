from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
import json
import sys

import pytest

from src.artifacts.json_values import json_sha256
from scripts.research.human13_one_image_services import (
    ProductionOneImageServices,
    Task5ProductionContextFailureReceipt,
    Task5ProductionContextUnavailable,
)

from scripts.research.run_human13_all_hf_shared_surface_vertical import (
    ALL_HF_VERTICAL_UNIT_ID,
    AuditAnalysis,
    AuditPairAnalysis,
    DualGPUResourceReceipt,
    EntryConfig,
    ExecutionAuthority,
    GPUResource,
    OneImageTerminalReceipt,
    OutputRootReceipt,
    ResourceReceipt,
    SourceAssemblyReceipt,
    analyze_audit_pair,
    confirm_absent_output_root,
    dry_run,
    evaluate_continuation_gate,
    run_full_panel,
    run_one_image,
    _seal_terminal,
    validate_dual_gpu_resources,
    _manifest_image_identity,
    _ensure_repo_root_on_sys_path,
)
from scripts.research.build_human13_k_union_manifest import default_binding
import scripts.research.run_human13_all_hf_shared_surface_vertical as entry_owner


def test_direct_entry_installs_explicit_repo_root_for_package_imports(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(entry_owner.sys, "path", ["/sentinel"])
    _ensure_repo_root_on_sys_path(root)
    assert entry_owner.sys.path[0] == str(root.resolve())


def test_direct_entry_promotes_existing_repo_root_ahead_of_competing_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(entry_owner.sys, "path", ["/competing", str(root), "/other"])
    _ensure_repo_root_on_sys_path(root)
    assert entry_owner.sys.path == [str(root.resolve()), "/competing", "/other"]


def test_module_entry_alias_preserves_terminal_receipt_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ``-m`` entry must not create a second terminal-receipt class."""

    canonical = "scripts.research.run_human13_all_hf_shared_surface_vertical"
    original_name = entry_owner.__name__
    original_main = sys.modules.get("__main__")
    monkeypatch.setattr(entry_owner, "__name__", "__main__")
    monkeypatch.setitem(sys.modules, "__main__", entry_owner)
    monkeypatch.delitem(sys.modules, canonical, raising=False)
    try:
        entry_owner._install_canonical_module_alias()
        assert sys.modules[canonical] is entry_owner
    finally:
        entry_owner.__name__ = original_name
        if original_main is None:
            sys.modules.pop("__main__", None)
        else:
            sys.modules["__main__"] = original_main
        sys.modules[canonical] = entry_owner


def _image() -> SimpleNamespace:
    owner = SimpleNamespace(
        owner_id="g-1",
        category="person",
        bbox=(0.0, 0.0, 10.0, 10.0),
        source_object_index=0,
    )
    return SimpleNamespace(
        image_id=1584,
        owners=(owner,),
        g_owner_ids=("g-1",),
        h_owner_ids=("h-1",),
        m_owner_ids=("m-1",),
        panel_sha256="4" * 64,
        panel_row_sha256="2" * 64,
        image_sha256="3" * 64,
        tokenizer_sha256="5" * 64,
        prompt_policy_fingerprint="6" * 64,
        binding=SimpleNamespace(
            matcher=SimpleNamespace(
                algorithm="cardinality_first_max_total_iou",
                same_category=True,
                duplicate_iou_threshold=0.95,
                owner_iou_threshold=0.50,
                duplicate_comparison="strictly_greater",
                target_row_rule="max_owner_iou_then_seed_then_row_index",
            )
        ),
    )


def _row(
    *,
    owner: str = "g-1",
    malformed: int = 0,
    stop: str = "im_end",
    image_id: int = 1584,
    provenance: bool = True,
    repetition_penalty: float = 1.0,
    arm_id: str = "frozen_source",
    checkpoint_payload_sha256: str = "d" * 64,
    checkpoint_path: str = "/source/checkpoint",
    trajectory_id: str = "source-trajectory-1584",
    run_id: str = "source-run",
    run_root: str = "/source/run",
) -> dict[str, Any]:
    bbox = [0.0, 0.0, 10.0, 10.0] if owner == "g-1" else [20.0, 20.0, 30.0, 30.0]
    category = "person"
    row = {
        "image_id": image_id,
        "predictions": [
            {
                "generated_order": 0,
                "description": category,
                "bbox": bbox,
            }
        ],
        "generated_token_ids": [1, 2, 3],
        "stop_reason": stop,
        "repetition_penalty": repetition_penalty,
        "malformed_row_count": malformed,
        "arm_id": arm_id,
        "milestone": 0,
        "trajectory_id": trajectory_id,
        "parser": "compact_object_box_closed_only",
        "parser_status": "accepted",
    }
    if provenance:
        row["provenance"] = {
            "manifest_sha256": "c" * 64,
            "panel_sha256": "4" * 64,
            "panel_row_sha256": "2" * 64,
            "image_id": 1584,
            "image_sha256": "3" * 64,
            "repetition_penalty": repetition_penalty,
            "tokenizer_sha256": "5" * 64,
            "prompt_policy_fingerprint": "6" * 64,
            "decode_mode": "original_prompt_clean_greedy",
            "backend": "hf",
            "physical_batch_size": 1,
            "do_sample": False,
            "arm_id": arm_id,
            "milestone": 0,
            "trajectory_id": trajectory_id,
            "source_trajectory_id": "source-trajectory-1584",
            "parser": "compact_object_box_closed_only",
            "run_id": run_id,
            "run_root": run_root,
            "checkpoint_path": checkpoint_path,
            "checkpoint_payload_sha256": checkpoint_payload_sha256,
            "source_checkpoint_identity": {
                "checkpoint_path": "/source/checkpoint",
                "base_model_path": "/source/base-model",
                "adapter_sha256": "a" * 64,
                "special_embedding_sha256": "b" * 64,
            },
        }
    return row


def _proposal_row(
    *,
    owner: str = "g-1",
    malformed: int = 0,
    stop: str = "im_end",
    image_id: int = 1584,
    provenance: bool = True,
    repetition_penalty: float = 1.0,
) -> dict[str, Any]:
    return _row(
        owner=owner,
        malformed=malformed,
        stop=stop,
        image_id=image_id,
        provenance=provenance,
        repetition_penalty=repetition_penalty,
        arm_id="private_proposal",
        checkpoint_payload_sha256="e" * 64,
        checkpoint_path="/private/proposal",
        trajectory_id="proposal-trajectory-1584",
        run_id="proposal-run",
        run_root="/private/run",
    )


def _config(tmp_path: Path) -> EntryConfig:
    return EntryConfig(
        unit_id=ALL_HF_VERTICAL_UNIT_ID,
        image_id=1584,
        seed_groups=(
            (35001, 35002, 35003, 35004),
            (35005, 35006, 35007, 35008),
            (35009, 35010, 35011, 35012),
            (35013, 35014, 35015, 35016),
        ),
        training_repetition_penalty=1.0,
        dtype="bfloat16",
        attention_backend="flash_attention_2",
        use_cache=False,
        learning_rate=3.0e-6,
        optimizer_name="adamw_torch",
        output_root=str(tmp_path / "one-image"),
        audit_repetition_penalties=(1.0, 1.10),
        source_checkpoint_path="/source/checkpoint",
        base_model_path="/source/base-model",
        adapter_path="/source/adapter",
        special_embedding_path="/source/special-embeddings",
        source_adapter_sha256="a" * 64,
        special_embedding_sha256="b" * 64,
        manifest_sha256="c" * 64,
    )


def _resources() -> tuple[GPUResource, GPUResource]:
    return (
        GPUResource(index=0, total_memory_bytes=80 << 30, free_memory_bytes=70 << 30),
        GPUResource(index=1, total_memory_bytes=80 << 30, free_memory_bytes=70 << 30),
    )


def _audit_identity_kwargs(
    *, source: str = "d" * 64, proposal: str = "e" * 64
) -> dict[str, Any]:
    return {
        "expected_source_checkpoint_sha256": source,
        "expected_proposal_checkpoint_sha256": proposal,
        "expected_source_checkpoint_path": "/source/checkpoint",
        "expected_proposal_checkpoint_path": "/private/proposal",
        "expected_source_identity": {
            "checkpoint_path": "/source/checkpoint",
            "base_model_path": "/source/base-model",
            "adapter_sha256": "a" * 64,
            "special_embedding_sha256": "b" * 64,
        },
    }


class _FakeServices:
    def __init__(self, *, fail: str | None = None) -> None:
        self.events: list[str] = []
        self.fail = fail
        self.private: object | None = None

    def preflight_source_assembly(
        self, config: EntryConfig, resources: Any
    ) -> SourceAssemblyReceipt:
        self.events.append("preflight")
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
        return replace(
            base,
            source_validation_sha256=base.source_validation_content_sha256,
            assembly_receipt_sha256=base.assembly_receipt_content_sha256,
        )

    def open_training(self, config: EntryConfig, resources: Any) -> object:
        self.events.append("open_training")
        if self.fail == "open_training":
            raise RuntimeError("open training failure")
        return object()

    def open_audit(self, config: EntryConfig, resources: Any) -> object:
        self.events.append("open_audit")
        return object()

    def source_audit(
        self, audit_session: object, repetition_penalty: float
    ) -> dict[str, Any]:
        del audit_session
        self.events.append(f"source_audit:{repetition_penalty}")
        return _row(repetition_penalty=repetition_penalty)

    def request_source_only_close(self) -> None:
        self.events.append("source_only_close_requested")

    def acquire_and_replay(
        self, training_session: object, config: EntryConfig
    ) -> object:
        del training_session, config
        self.events.append("acquire_replay")
        return SimpleNamespace(trusted_h_owner_ids=("h-1",), parity_passed=True)

    def apply_private_update(
        self, training_session: object, acquisition: object, config: EntryConfig
    ) -> object:
        del training_session, acquisition, config
        self.events.append("private_update")
        if self.fail == "private_update":
            raise RuntimeError("update failure")
        return object()

    def write_private_proposal(
        self, training_session: object, proposal: object, output_root: Path
    ) -> object:
        del training_session, proposal, output_root
        self.events.append("write_private")
        self.private = SimpleNamespace(
            checkpoint_path="/private/proposal",
            checkpoint_payload_sha256="e" * 64,
        )
        return self.private

    def proposal_audit(
        self, audit_session: object, private: object, repetition_penalty: float
    ) -> dict[str, Any]:
        del audit_session, private
        self.events.append(f"proposal_audit:{repetition_penalty}")
        return _proposal_row(owner="g-1", repetition_penalty=repetition_penalty)

    def rollback_and_reproduce_source(
        self, training_session: object, proposal: object
    ) -> bool:
        del training_session, proposal
        self.events.append("rollback_reproduce")
        return True

    def cleanup_private_proposal(self, private: object) -> None:
        assert private is self.private
        self.events.append("cleanup_private")

    def close(
        self, training_session: object | None, audit_session: object | None
    ) -> None:
        del training_session, audit_session
        self.events.append("close")


def test_typed_runtime_context_failure_terminal_binds_observed_shared_surface_counts(
    tmp_path: Path,
) -> None:
    sampled_hashes = tuple(f"{index:x}" * 64 for index in range(1, 5))
    replay_hashes = tuple(f"{index:x}" * 64 for index in range(5, 9))
    context = Task5ProductionContextFailureReceipt(
        reason_code="live_task2_owner_publications_unavailable",
        config_sha256=_config(tmp_path).content_sha256,
        manifest_sha256="c" * 64,
        source_identity_sha256="9" * 64,
        sampled_group_sha256s=sampled_hashes,
        replay_group_sha256s=replay_hashes,
        sampled_group_object_ids=(1, 2, 3, 4),
        replay_group_object_ids=(5, 6, 7, 8),
        replay_tensor_object_ids=(("live", 10),),
        assembly_object_id=11,
        session_object_id=12,
        model_object_id=13,
        optimizer_object_id=14,
        manifest_object_id=15,
        manifest_image_object_id=16,
        process_id=17,
    )
    shared_payload = {
        "sample_forward_count": 463,
        "replay_forward_count": 463,
        "source_owner_forward_count": 7,
        "total_forward_count": 933,
        "no_cache_forward_count": 933,
        "sampled_group_sha256s": list(sampled_hashes),
        "replay_group_sha256s": list(replay_hashes),
        "model_object_id": 13,
        "retained_graph_count": 0,
        "session_held_reference_count": 0,
        "cleanup_state": "closed",
        "cleanup_reason": "completed",
        "cleanup_failures": [],
        "cleanup_call_count": 1,
    }
    shared_receipt = SimpleNamespace(
        **shared_payload,
        content_sha256=json_sha256(shared_payload),
        to_dict=lambda: shared_payload
        | {"content_sha256": json_sha256(shared_payload)},
    )

    class Backend:
        def __init__(self) -> None:
            self.close_training_calls = 0
            self.close_audit_calls = 0

        def preflight_source_assembly(
            self, config: EntryConfig, resources: Any
        ) -> SourceAssemblyReceipt:
            return _FakeServices().preflight_source_assembly(config, resources)

        def open_training(self, config: EntryConfig, resources: Any) -> object:
            del config, resources
            return object()

        def open_audit(self, config: EntryConfig, resources: Any) -> object:
            del config, resources
            return object()

        def source_audit(
            self, session: object, repetition_penalty: float
        ) -> Mapping[str, Any]:
            del session
            return _row(repetition_penalty=repetition_penalty)

        def pre_acquisition_admission(
            self,
            training_session: object,
            source_outputs: Mapping[float, Mapping[str, Any]],
        ) -> object:
            del training_session, source_outputs
            return SimpleNamespace(
                content_sha256="f" * 64,
                to_artifact_dict=lambda: {"content_sha256": "f" * 64},
            )

        def acquire_and_replay(
            self, session: object, config: EntryConfig
        ) -> Any:
            del session, config
            raise Task5ProductionContextUnavailable(context)

        def build_cuda_adapter(self, proposal_input: object) -> Any:
            raise AssertionError(f"unexpected adapter build: {proposal_input!r}")

        def write_private_checkpoint(
            self, session: object, proposal: object, output_root: Path
        ) -> object:
            raise AssertionError((session, proposal, output_root))

        def proposal_audit(
            self,
            session: object,
            private: object,
            repetition_penalty: float,
        ) -> Mapping[str, Any]:
            raise AssertionError((session, private, repetition_penalty))

        def reproduce_source(
            self, session: object, repetition_penalties: tuple[float, float]
        ) -> Mapping[float, Mapping[str, Any]]:
            raise AssertionError((session, repetition_penalties))

        def cleanup_private_checkpoint(self, private: object) -> None:
            raise AssertionError(private)

        def close_training(self, session: object) -> object:
            del session
            self.close_training_calls += 1
            return shared_receipt

        def close_audit(self, session: object) -> None:
            del session
            self.close_audit_calls += 1

    config = _config(tmp_path)
    stale = tmp_path / "lost-one-image" / "run-reservation.json"
    stale.parent.mkdir()
    stale.write_text(
        json.dumps(
            {
                "run_id": "lost",
                "pid": 377949,
                "model_actions": {
                    "model_loads": 0,
                    "forwards": 0,
                    "backwards": 0,
                    "optimizer_steps": 0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    successor = tmp_path / "typed-failure-successor"
    backend = Backend()
    services = ProductionOneImageServices(
        backend=backend,
        reservation_mode="lost_owner_recovery",
        stale_reservation_path=stale,
        successor_root=successor,
        attempt_id="typed-failure-attempt",
        recovery_authority="test_explicit_owner",
        pid_is_alive=lambda _pid: False,
    )

    terminal = run_one_image(
        config,
        authority=ExecutionAuthority(user_model_gpu_authority=True),
        resources=_resources(),
        output_root=successor,
        services=services,
        manifest_image=_image(),
    )

    assert terminal.terminal_status == "update_failure"
    assert terminal.model_actions["forwards"] == 935
    assert terminal.model_actions["backwards"] == 0
    assert terminal.model_actions["optimizer_steps"] == 0
    assert terminal.resource_receipt.sample_forward_count == 463
    assert terminal.resource_receipt.replay_forward_count == 463
    assert terminal.resource_receipt.source_owner_forward_count == 7
    assert terminal.resource_receipt.total_forward_count == 933
    assert terminal.resource_receipt.no_cache_forward_count == 933
    assert backend.close_training_calls == 1
    assert backend.close_audit_calls == 1
    terminal_envelope = json.loads((successor / "terminal.json").read_text())
    assert terminal_envelope["context_failure_receipt_sha256"] == (
        context.content_sha256
    )
    assert terminal_envelope["shared_surface_resource_receipt_sha256"] == (
        shared_receipt.content_sha256
    )
    close_phase = json.loads(
        next((successor / "receipts").glob("*-training_session_closed.json")).read_text()
    )
    failure = close_phase["evidence"]["acquisition_failure_receipt"]
    assert terminal_envelope["observed_acquisition_failure_receipt"] == failure
    assert failure["sample_forward_count"] == 463
    assert failure["replay_forward_count"] == 463
    assert failure["source_owner_forward_count"] == 7
    assert failure["total_forward_count"] == 933
    assert failure["no_cache_forward_count"] == 933
    assert tuple(failure["sampled_group_sha256s"]) == sampled_hashes
    assert tuple(failure["replay_group_sha256s"]) == replay_hashes
    assert failure["config_sha256"] == config.content_sha256
    assert failure["manifest_sha256"] == "c" * 64
    assert failure["source_identity_sha256"] == "9" * 64
    assert failure["cleanup_state"] == "closed"
    assert failure["cleanup_call_count"] == 1
    assert failure["context_failure_receipt_sha256"] == context.content_sha256
    assert failure["content_sha256"] == json_sha256(
        {key: value for key, value in failure.items() if key != "content_sha256"}
    )


def test_default_dry_run_has_zero_model_gpu_network_and_output_actions(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    result = dry_run(config)

    assert result.terminal_status == "dry_run"
    assert result.model_actions == {
        "model_loads": 0,
        "forwards": 0,
        "backwards": 0,
        "optimizer_steps": 0,
        "gpu_allocations": 0,
        "network_actions": 0,
        "output_creations": 0,
    }
    assert result.resource_receipt.sample_forward_count == 2048
    assert result.resource_receipt.replay_forward_count == 2048
    assert result.resource_receipt.source_owner_forward_count == 0
    assert result.resource_receipt.total_forward_count == 4096
    assert result.resource_receipt.no_cache_forward_count == 4096
    assert not Path(config.output_root).exists()


def test_partial_failure_resource_receipt_does_not_claim_unstarted_k16_work(
    tmp_path: Path,
) -> None:
    resources = entry_owner.DualGPUResourceReceipt(
        cards=(entry_owner.GPUResource(0, 1, 1), entry_owner.GPUResource(1, 1, 1))
    )
    receipt = entry_owner._resource_receipt(
        resources,
        None,
        4,
        forward_counts={
            "sample_forward_count": 0,
            "replay_forward_count": 0,
            "source_owner_forward_count": 2,
            "total_forward_count": 2,
            "no_cache_forward_count": 2,
            "sampled_group_count": 0,
            "sampled_request_count": 0,
        },
        backward_count=0,
    )
    assert receipt.sampled_group_count == 0
    assert receipt.sampled_request_count == 0
    assert receipt.backward_count == 0
    assert receipt.total_forward_count == 2


def test_native_replay_text_is_update_failure_and_preserves_typed_reason(
    tmp_path: Path,
) -> None:
    from scripts.research.human13_hf_native_one_image_owner import (
        HFNativeOneImageOwnerError,
    )

    class NativeFailure(_FakeServices):
        def acquire_and_replay(
            self, training_session: object, config: EntryConfig
        ) -> object:
            del training_session, config
            raise HFNativeOneImageOwnerError(
                "canonical replay projection failed",
                disposition="canonical_projection_failure",
            )

    terminal = run_one_image(
        _config(tmp_path),
        authority=ExecutionAuthority(user_model_gpu_authority=True),
        resources=_resources(),
        output_root=tmp_path / "native-failure",
        services=NativeFailure(),
        manifest_image=_image(),
    )

    assert terminal.terminal_status == "update_failure"
    assert terminal.failure_reason == (
        "HFNativeOneImageOwnerError: canonical_projection_failure: "
        "canonical replay projection failed"
    )


def test_dual_gpu_admission_requires_distinct_suitable_cards() -> None:
    cards = _resources()
    receipt = validate_dual_gpu_resources(cards)
    assert receipt.training_gpu == 0
    assert receipt.audit_gpu == 1

    with pytest.raises(ValueError, match="distinct"):
        validate_dual_gpu_resources((cards[0], GPUResource(0, 80 << 30, 70 << 30)))

    with pytest.raises(ValueError, match="suitable"):
        validate_dual_gpu_resources(
            (cards[0], GPUResource(1, 80 << 30, 70 << 30, suitable=False))
        )


def test_execute_requires_explicit_model_gpu_authority_and_absent_root(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    with pytest.raises(PermissionError, match="authority"):
        run_one_image(
            config,
            authority=ExecutionAuthority(user_model_gpu_authority=False),
            resources=_resources(),
            output_root=Path(config.output_root),
            services=_FakeServices(),
        )

    root = Path(config.output_root)
    root.mkdir()
    with pytest.raises(FileExistsError, match="output root"):
        run_one_image(
            config,
            authority=ExecutionAuthority(user_model_gpu_authority=True),
            resources=_resources(),
            output_root=root,
            services=_FakeServices(),
        )


def test_phase_order_binds_gpu_roles_and_cleans_private_proposal(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    services = _FakeServices()
    result = run_one_image(
        config,
        authority=ExecutionAuthority(user_model_gpu_authority=True),
        resources=_resources(),
        output_root=Path(config.output_root),
        services=services,
        manifest_image=_image(),
    )

    assert result.terminal_status == "completed_null_or_unsafe"
    assert services.events == [
        "preflight",
        "open_training",
        "open_audit",
        "source_audit:1.0",
        "source_audit:1.1",
        "acquire_replay",
        "private_update",
        "write_private",
        "proposal_audit:1.0",
        "proposal_audit:1.1",
        "rollback_reproduce",
        "cleanup_private",
        "close",
    ]
    assert result.resource_receipt.training_gpu == 0
    assert result.resource_receipt.audit_gpu == 1
    assert result.private_proposal_cleaned is True


def test_pre_acquisition_admission_failure_happens_before_k16_call(
    tmp_path: Path,
) -> None:
    class PreAcquisitionFailure(_FakeServices):
        def pre_acquisition_admission(
            self,
            training_session: object,
            source_outputs: Mapping[float, Mapping[str, Any]],
        ) -> None:
            del training_session, source_outputs
            self.events.append("pre_acquisition_admission")
            raise RuntimeError("deterministic runtime ownership failure")

        def acquire_and_replay(
            self, training_session: object, config: EntryConfig
        ) -> object:
            self.events.append("acquire_replay")
            raise AssertionError("K16 must not start after pre-acquisition failure")

    config = _config(tmp_path)
    services = PreAcquisitionFailure()
    result = run_one_image(
        config,
        authority=ExecutionAuthority(user_model_gpu_authority=True),
        resources=_resources(),
        output_root=Path(config.output_root),
        services=services,
        manifest_image=_image(),
    )

    assert result.terminal_status == "update_failure"
    assert "deterministic runtime ownership failure" in (result.failure_reason or "")
    assert services.events[-1] == "close"
    assert "acquire_replay" not in services.events


def test_source_audit_owner_failure_preserves_observed_forward_phase(
    tmp_path: Path,
) -> None:
    class SourceOwnerFailure(_FakeServices):
        def source_audit(
            self, audit_session: object, repetition_penalty: float
        ) -> dict[str, Any]:
            result = super().source_audit(audit_session, repetition_penalty)
            if repetition_penalty == 1.1:
                error = RuntimeError("source owner preparation failed")
                setattr(error, "_source_audit_forward_observed", True)
                setattr(error, "_source_audit_phase", "source_audit_rp_1.1")
                raise error
            return result

    services = SourceOwnerFailure()
    config = _config(tmp_path)
    result = run_one_image(
        config,
        authority=ExecutionAuthority(user_model_gpu_authority=True),
        resources=_resources(),
        output_root=Path(config.output_root),
        services=services,
        manifest_image=_image(),
    )

    assert result.terminal_status == "update_failure"
    assert result.model_actions["forwards"] == 2
    assert "source_audit_rp_1.1" in result.phase_receipts
    assert "acquire_replay" not in services.events


def test_guarded_entry_hands_final_typed_terminal_to_durable_owner(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)

    class DurableServices(_FakeServices):
        terminal: OneImageTerminalReceipt | None = None

        def persist_terminal(self, terminal: OneImageTerminalReceipt) -> None:
            self.terminal = terminal

    services = DurableServices()
    result = run_one_image(
        config,
        authority=ExecutionAuthority(user_model_gpu_authority=True),
        resources=_resources(),
        output_root=Path(config.output_root),
        services=services,
        manifest_image=_image(),
    )

    assert services.terminal is result
    assert services.terminal is not None
    assert services.terminal.content_sha256 == result.content_sha256


def test_pre_reservation_error_is_not_masked_or_sent_to_terminal_writer(
    tmp_path: Path,
) -> None:
    class PreReservationError(RuntimeError):
        pass

    class FailingBeforeReservation(_FakeServices):
        reservation_identity = None

        def __init__(self) -> None:
            super().__init__()
            self.persist_calls = 0

        def preflight_source_assembly(
            self, config: EntryConfig, resources: Any
        ) -> Any:
            del config, resources
            raise PreReservationError("atomic reservation collision")

        def persist_terminal(self, terminal: OneImageTerminalReceipt) -> None:
            del terminal
            self.persist_calls += 1
            raise AssertionError("terminal writer crossed pre-reservation failure")

    services = FailingBeforeReservation()
    root = tmp_path / "fresh-primary"

    with pytest.raises(PreReservationError, match="atomic reservation collision"):
        run_one_image(
            _config(tmp_path),
            authority=ExecutionAuthority(user_model_gpu_authority=True),
            resources=_resources(),
            output_root=root,
            services=services,
            manifest_image=_image(),
        )

    assert services.persist_calls == 0
    assert not root.exists()


def test_post_reservation_preflight_failure_has_zero_observed_resource_counts(
    tmp_path: Path,
) -> None:
    root = tmp_path / "fresh-primary"
    identity = entry_owner.RunReservationIdentity(
        reservation_mode="fresh_primary",
        run_id="fresh-attempt",
        output_root=str(root.resolve()),
        reservation_sha256="a" * 64,
    )

    class FailingAfterReservation(_FakeServices):
        reservation_identity = identity
        phase_receipt_sha256s: tuple[str, ...] = ()
        phase_ledger_sha256 = None
        terminal: OneImageTerminalReceipt | None = None

        def preflight_source_assembly(
            self, config: EntryConfig, resources: Any
        ) -> Any:
            del config, resources
            raise RuntimeError("post-reservation preflight failure")

        def action_counters(self) -> dict[str, int]:
            return dict(entry_owner.ZERO_MODEL_ACTIONS)

        def persist_terminal(self, terminal: OneImageTerminalReceipt) -> None:
            self.terminal = terminal

    services = FailingAfterReservation()
    terminal = run_one_image(
        _config(tmp_path),
        authority=ExecutionAuthority(user_model_gpu_authority=True),
        resources=_resources(),
        output_root=root,
        services=services,
        manifest_image=_image(),
    )

    resource = terminal.resource_receipt
    assert services.terminal is terminal
    assert resource.reservation_identity == identity
    assert resource.sampled_request_count == 0
    assert resource.sampled_group_count == 0
    assert resource.sample_forward_count == 0
    assert resource.replay_forward_count == 0
    assert resource.source_owner_forward_count == 0
    assert resource.total_forward_count == 0
    assert resource.no_cache_forward_count == 0
    assert resource.backward_count == 0


def test_post_apply_journal_failure_is_integrated_typed_terminal_with_one_rollback(
    tmp_path: Path,
) -> None:
    from scripts.research.human13_one_image_services import (
        ProductionAcquisition,
        ProductionOneImageServices,
    )

    class Adapter:
        rollback_calls = 0

        def apply_private_proposal(self) -> object:
            return SimpleNamespace(content_sha256="7" * 64)

        def rollback_private_proposal(self) -> object:
            self.rollback_calls += 1
            return SimpleNamespace(
                content_sha256="8" * 64,
                rollback_decision="rejected_restored",
            )

    class Backend(_FakeServices):
        adapter = Adapter()

        def pre_acquisition_admission(
            self,
            training_session: object,
            source_outputs: Mapping[float, Mapping[str, Any]],
        ) -> object:
            del training_session, source_outputs
            return SimpleNamespace(
                content_sha256="f" * 64,
                to_artifact_dict=lambda: {"content_sha256": "f" * 64},
            )

        def acquire_and_replay(
            self, training_session: object, config: EntryConfig
        ) -> ProductionAcquisition:
            del training_session, config
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

        def build_cuda_adapter(self, proposal_input: object) -> Adapter:
            del proposal_input
            return self.adapter

        def reproduce_source(
            self, session: object, repetition_penalties: tuple[float, float]
        ) -> dict[float, dict[str, Any]]:
            del session
            self.events.append("source_reproduction")
            return {rp: _row(repetition_penalty=rp) for rp in repetition_penalties}

        def write_private_checkpoint(
            self, session: object, proposal: object, output_root: Path
        ) -> object:
            return self.write_private_proposal(session, proposal, output_root)

        def cleanup_private_checkpoint(self, private: object) -> None:
            self.cleanup_private_proposal(private)

        def close_training(self, session: object) -> None:
            del session
            self.events.append("close_training")

        def close_audit(self, session: object) -> None:
            del session
            self.events.append("close_audit")

    stale = tmp_path / "one-image" / "run-reservation.json"
    stale.parent.mkdir()
    stale.write_text(
        '{"run_id":"lost","pid":377949,"model_actions":{"model_loads":0}}\n',
        encoding="utf-8",
    )
    successor = tmp_path / "one-image-recovery-attempt"

    def phase_writer(path: Path, value: Mapping[str, Any]) -> None:
        if path.name.endswith("private_update_applied.json"):
            raise OSError("injected post-apply journal failure")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(__import__("json").dumps(value), encoding="utf-8")

    backend = Backend()
    services = ProductionOneImageServices(
        backend=backend,
        reservation_mode="lost_owner_recovery",
        stale_reservation_path=stale,
        successor_root=successor,
        attempt_id="attempt",
        recovery_authority="test_explicit_owner",
        pid_is_alive=lambda pid: False,
        phase_writer=phase_writer,
    )
    result = run_one_image(
        _config(tmp_path),
        authority=ExecutionAuthority(user_model_gpu_authority=True),
        resources=_resources(),
        output_root=successor,
        services=services,
        manifest_image=_image(),
    )

    assert result.terminal_status == "update_failure"
    assert result.source_reproduced is True
    assert backend.adapter.rollback_calls == 1
    assert "source_reproduction" in backend.events
    assert backend.events[-2:] == ["close_audit", "close_training"]
    terminal = __import__("json").loads((successor / "terminal.json").read_text())
    assert terminal["phase_ledger_sha256"] == result.phase_ledger_sha256


def test_public_execute_dispatches_constructed_production_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    output_root = tmp_path / "fresh-successor"
    services = _FakeServices()
    observed_resources = entry_owner.validate_dual_gpu_resources(_resources())
    execution = entry_owner.ProductionExecution(
        services=services,
        resources=observed_resources,
        output_root=output_root,
        manifest_image=_image(),
        manifest_binding=None,
    )
    monkeypatch.setattr(
        entry_owner,
        "build_production_execution",
        lambda **kwargs: execution,
    )

    result = entry_owner.main(
        [
            "--execute",
            "--user-model-gpu-authority",
            "--output-root",
            str(output_root),
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--attempt-id",
            "attempt",
        ]
    )

    assert result == 0
    terminal_payload = __import__("json").loads(capsys.readouterr().out)
    assert terminal_payload["terminal_status"] == "update_failure"
    assert (
        terminal_payload["resource_receipt"]["resources"]["content_sha256"]
        == observed_resources.content_sha256
    )
    assert terminal_payload["resource_receipt"]["output_root"]["path"] == str(
        output_root.resolve()
    )
    assert services.events


def test_production_construction_rejects_gpu1_before_manifest_or_backend(
    tmp_path: Path,
) -> None:
    args = entry_owner.build_parser().parse_args(
        [
            "--execute",
            "--user-model-gpu-authority",
            "--output-root",
            str(tmp_path / "successor"),
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--attempt-id",
            "attempt",
        ]
    )
    construction_calls: list[str] = []

    def forbidden_manifest(_path: Path, *, require_full_panel: bool) -> object:
        del require_full_panel
        construction_calls.append("manifest")
        raise AssertionError("manifest load crossed failed GPU admission")

    def forbidden_backend(**_kwargs: Any) -> object:
        construction_calls.append("backend")
        raise AssertionError("backend construction crossed failed GPU admission")

    with pytest.raises(RuntimeError, match="GPU 1.*free memory"):
        entry_owner.build_production_execution(
            config=_config(tmp_path),
            args=args,
            gpu_observer=lambda: (
                GPUResource(0, 80 << 30, 79 << 30),
                GPUResource(1, 80 << 30, 1 << 30),
            ),
            manifest_loader=forbidden_manifest,
            backend_factory=forbidden_backend,
        )
    assert construction_calls == []


def test_production_construction_uses_repository_native_owner_default(
    tmp_path: Path,
) -> None:
    from scripts.research.human13_hf_native_one_image_owner import (
        RepositoryHFNativeOneImageOwner,
    )

    args = entry_owner.build_parser().parse_args(
        [
            "--execute",
            "--user-model-gpu-authority",
            "--output-root",
            str(tmp_path / "successor"),
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--attempt-id",
            "attempt",
        ]
    )
    observed: dict[str, Any] = {}

    class Backend:
        def preflight_source_assembly(self, *_args: Any) -> Any:
            raise AssertionError

        def open_training(self, *_args: Any) -> Any:
            raise AssertionError

        def open_audit(self, *_args: Any) -> Any:
            raise AssertionError

        def source_audit(self, *_args: Any) -> Any:
            raise AssertionError

        def acquire_and_replay(self, *_args: Any) -> Any:
            raise AssertionError

        def build_cuda_adapter(self, *_args: Any) -> Any:
            raise AssertionError

        def write_private_checkpoint(self, *_args: Any) -> Any:
            raise AssertionError

        def proposal_audit(self, *_args: Any) -> Any:
            raise AssertionError

        def reproduce_source(self, *_args: Any) -> Any:
            raise AssertionError

        def cleanup_private_checkpoint(self, *_args: Any) -> Any:
            raise AssertionError

        def close_training(self, *_args: Any) -> Any:
            raise AssertionError

        def close_audit(self, *_args: Any) -> Any:
            raise AssertionError

    def backend_factory(**kwargs: Any) -> object:
        observed.update(kwargs)
        return Backend()

    entry_owner.build_production_execution(
        config=_config(tmp_path),
        args=args,
        gpu_observer=lambda: _resources(),
        manifest_loader=lambda *_args, **_kwargs: SimpleNamespace(
            images=(_image(),), binding=object()
        ),
        backend_factory=backend_factory,
    )

    assert type(observed["hf_native_owner"]) is RepositoryHFNativeOneImageOwner


def test_production_gpu_observation_uses_live_cuda_memory_for_exact_roles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    calls: list[int] = []
    observed = {
        0: (71 << 30, 80 << 30),
        1: (31 << 30, 80 << 30),
    }
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    def mem_get_info(index: int) -> tuple[int, int]:
        calls.append(index)
        return observed[index]

    monkeypatch.setattr(torch.cuda, "mem_get_info", mem_get_info)

    cards = entry_owner.observe_production_gpu_resources()
    receipt = entry_owner.admit_production_gpu_resources(cards)

    assert calls == [0, 1]
    assert tuple(card.index for card in receipt.cards) == (0, 1)
    assert tuple(card.free_memory_bytes for card in receipt.cards) == (
        71 << 30,
        31 << 30,
    )


def test_public_parser_does_not_expose_arbitrary_runtime_factory() -> None:
    with pytest.raises(SystemExit):
        entry_owner.build_parser().parse_args(
            ["--execute", "--runtime-factory", "tests.fake:factory"]
        )


def test_cli_defaults_to_fresh_primary_without_recovery_inputs() -> None:
    args = entry_owner.build_parser().parse_args([])
    assert args.reservation_mode == "fresh_primary"
    assert args.stale_reservation is None
    assert args.recovery_authority is None


@pytest.mark.parametrize(
    "extra_args, expected",
    [
        (["--stale-reservation", "/stale/run-reservation.json"], "fresh primary"),
        (["--recovery-authority", "owner"], "fresh primary"),
        (["--reservation-mode", "lost_owner_recovery"], "stale reservation"),
        (
            [
                "--reservation-mode",
                "lost_owner_recovery",
                "--stale-reservation",
                "/stale/run-reservation.json",
            ],
            "recovery authority",
        ),
    ],
)
def test_cli_rejects_mixed_or_incomplete_reservation_modes_before_gpu_observation(
    tmp_path: Path, extra_args: list[str], expected: str
) -> None:
    args = entry_owner.build_parser().parse_args(
        [
            "--execute",
            "--user-model-gpu-authority",
            "--output-root",
            str(tmp_path / "new-root"),
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--attempt-id",
            "attempt",
            *extra_args,
        ]
    )

    with pytest.raises(ValueError, match=expected):
        entry_owner.build_production_execution(
            config=_config(tmp_path),
            args=args,
            gpu_observer=lambda: (_ for _ in ()).throw(
                AssertionError("GPU observation crossed reservation CLI validation")
            ),
        )


@pytest.mark.parametrize(
    "reservation_args",
    [
        ["--stale-reservation", "/stale/run-reservation.json"],
        [
            "--reservation-mode",
            "lost_owner_recovery",
            "--stale-reservation",
            "/stale/run-reservation.json",
            "--recovery-authority",
            "owner",
        ],
    ],
)
def test_full_panel_public_main_rejects_reservation_inputs_before_gpu_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reservation_args: list[str],
) -> None:
    observed = False

    def forbidden_gpu_observer() -> tuple[GPUResource, GPUResource]:
        nonlocal observed
        observed = True
        raise AssertionError("full-panel crossed reservation validation")

    monkeypatch.setattr(
        entry_owner,
        "observe_production_gpu_resources",
        forbidden_gpu_observer,
    )

    with pytest.raises(ValueError, match="fresh primary"):
        entry_owner.main(
            [
                "--execute",
                "--full-panel",
                "--user-model-gpu-authority",
                "--expected-terminal-sha256",
                "a" * 64,
                *reservation_args,
            ]
        )

    assert observed is False


def test_terminal_from_dict_rejects_tampered_phase_receipt_count() -> None:
    terminal = OneImageTerminalReceipt(
        terminal_status="parity_failure",
        resource_receipt=ResourceReceipt(
            entry_owner.validate_dual_gpu_resources(_resources()),
            None,
            phase_count=1,
            retry_count=0,
            promoted_checkpoint=False,
        ),
        model_actions={
            "model_loads": 0,
            "forwards": 0,
            "backwards": 0,
            "optimizer_steps": 0,
            "gpu_allocations": 0,
            "network_actions": 0,
            "output_creations": 0,
        },
        phase_receipt_sha256s=("a" * 64,),
        phase_ledger_sha256=entry_owner._phase_ledger_sha256(("a" * 64,)),
        failure_reason="parity",
    )
    payload = terminal.to_dict()
    payload["phase_receipt_count"] = 2

    with pytest.raises(ValueError, match="phase receipt count"):
        OneImageTerminalReceipt.from_dict(payload)


def test_historical_v1_resource_and_terminal_without_reservation_key_reload() -> None:
    resource = ResourceReceipt(
        entry_owner.validate_dual_gpu_resources(_resources()),
        None,
        phase_count=0,
        retry_count=0,
        promoted_checkpoint=False,
    )
    terminal = OneImageTerminalReceipt(
        terminal_status="parity_failure",
        resource_receipt=resource,
        model_actions=entry_owner.ZERO_MODEL_ACTIONS,
        failure_reason="historical failure",
    )
    historical = terminal.to_dict()
    historical_resource = historical["resource_receipt"]
    assert "reservation_identity" not in historical_resource
    historical_resource["content_sha256"] = entry_owner._sha256(
        {
            key: value
            for key, value in historical_resource.items()
            if key != "content_sha256"
        }
    )
    historical["resource_receipt_sha256"] = historical_resource["content_sha256"]
    historical["content_sha256"] = entry_owner._sha256(
        {key: value for key, value in historical.items() if key != "content_sha256"}
    )

    reloaded = OneImageTerminalReceipt.from_dict(historical)

    assert "reservation_identity" not in reloaded.resource_receipt.to_dict()
    assert reloaded.to_dict() == historical


def test_reserved_resource_requires_exact_matching_output_root() -> None:
    identity = entry_owner.RunReservationIdentity(
        reservation_mode="fresh_primary",
        run_id="fresh-attempt",
        output_root="/tmp/admitted-root",
        reservation_sha256="a" * 64,
    )
    resources = entry_owner.validate_dual_gpu_resources(_resources())

    with pytest.raises(ValueError, match="reservation.*output root"):
        ResourceReceipt(
            resources,
            None,
            phase_count=0,
            retry_count=0,
            promoted_checkpoint=False,
            reservation_identity=identity,
        )
    with pytest.raises(ValueError, match="reservation.*output root"):
        ResourceReceipt(
            resources,
            OutputRootReceipt("/tmp/different-root", False),
            phase_count=0,
            retry_count=0,
            promoted_checkpoint=False,
            reservation_identity=identity,
        )


def test_private_proposal_audit_binds_distinct_proposal_checkpoint_digest(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)

    class DistinctProposal(_FakeServices):
        def write_private_proposal(
            self, training_session: object, proposal: object, output_root: Path
        ) -> object:
            del training_session, proposal, output_root
            self.events.append("write_private")
            self.private = SimpleNamespace(
                checkpoint_path="/private/proposal",
                checkpoint_payload_sha256="e" * 64,
            )
            return self.private

        def proposal_audit(
            self, audit_session: object, private: object, repetition_penalty: float
        ) -> dict[str, Any]:
            del audit_session, private
            self.events.append(f"proposal_audit:{repetition_penalty}")
            return _proposal_row(owner="g-1", repetition_penalty=repetition_penalty)

    result = run_one_image(
        config,
        authority=ExecutionAuthority(user_model_gpu_authority=True),
        resources=_resources(),
        output_root=Path(config.output_root),
        services=DistinctProposal(),
        manifest_image=_image(),
    )
    assert result.terminal_status == "completed_null_or_unsafe"


def test_private_proposal_identity_must_differ_from_source_before_audit(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)

    class SameAsSource(_FakeServices):
        def write_private_proposal(
            self, training_session: object, proposal: object, output_root: Path
        ) -> object:
            del training_session, proposal, output_root
            self.events.append("write_private")
            self.private = SimpleNamespace(
                checkpoint_path="/source/checkpoint",
                checkpoint_payload_sha256="d" * 64,
            )
            return self.private

    services = SameAsSource()
    result = run_one_image(
        config,
        authority=ExecutionAuthority(user_model_gpu_authority=True),
        resources=_resources(),
        output_root=Path(config.output_root),
        services=services,
        manifest_image=_image(),
    )
    assert result.terminal_status == "update_failure"
    assert not any(event.startswith("proposal_audit:") for event in services.events)


def test_prebuilt_resource_receipt_cannot_rebind_training_or_audit_roles(
    tmp_path: Path,
) -> None:
    del tmp_path
    services = _FakeServices()
    with pytest.raises(ValueError, match="GPU roles"):
        DualGPUResourceReceipt(cards=_resources(), training_gpu=2, audit_gpu=3)
    assert services.events == []


def test_manifest_identity_is_checked_before_any_open_or_update(tmp_path: Path) -> None:
    config = _config(tmp_path)
    services = _FakeServices()
    wrong_image = _image()
    wrong_image.image_id = 2299
    with pytest.raises(ValueError, match="image_id"):
        run_one_image(
            config,
            authority=ExecutionAuthority(user_model_gpu_authority=True),
            resources=_resources(),
            output_root=Path(config.output_root),
            services=services,
            manifest_image=wrong_image,
        )
    assert services.events == []


def test_manifest_matcher_identity_is_checked_before_any_open(tmp_path: Path) -> None:
    config = _config(tmp_path)
    services = _FakeServices()
    matcher = SimpleNamespace(
        algorithm="wrong",
        same_category=True,
        duplicate_iou_threshold=0.95,
        owner_iou_threshold=0.50,
        duplicate_comparison="strictly_greater",
        target_row_rule="max_owner_iou_then_seed_then_row_index",
    )
    with pytest.raises(ValueError, match="matcher identity"):
        run_one_image(
            config,
            authority=ExecutionAuthority(user_model_gpu_authority=True),
            resources=_resources(),
            output_root=Path(config.output_root),
            services=services,
            manifest_image=_image(),
            manifest_binding=SimpleNamespace(matcher=matcher),
        )
    assert services.events == []


def test_parent_manifest_binding_is_authoritative_over_image_local_surface(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    services = _FakeServices()
    with pytest.raises(ValueError, match="manifest binding"):
        run_one_image(
            config,
            authority=ExecutionAuthority(user_model_gpu_authority=True),
            resources=_resources(),
            output_root=Path(config.output_root),
            services=services,
            manifest_image=_image(),
            manifest_binding=default_binding(),
        )
    assert services.events == []


def test_manifest_owner_strata_must_be_disjoint_before_any_open(tmp_path: Path) -> None:
    config = _config(tmp_path)
    services = _FakeServices()
    fabricated = _image()
    fabricated.h_owner_ids = ("g-1",)
    with pytest.raises(ValueError, match="strata"):
        run_one_image(
            config,
            authority=ExecutionAuthority(user_model_gpu_authority=True),
            resources=_resources(),
            output_root=Path(config.output_root),
            services=services,
            manifest_image=fabricated,
        )
    assert services.events == []


def test_source_identity_provenance_is_required_before_open(tmp_path: Path) -> None:
    config = _config(tmp_path)

    class MissingIdentity(_FakeServices):
        def preflight_source_assembly(
            self, config: EntryConfig, resources: Any
        ) -> SourceAssemblyReceipt:
            del resources
            self.events.append("preflight")
            return SourceAssemblyReceipt(
                source_plan_sha256=config.content_sha256,
                training_gpu=0,
                audit_gpu=1,
                training_surface="bf16/flash_attention_2",
                audit_surface="fp32/sdpa/batch1",
            )

    services = MissingIdentity()
    result = run_one_image(
        config,
        authority=ExecutionAuthority(user_model_gpu_authority=True),
        resources=_resources(),
        output_root=Path(config.output_root),
        services=services,
        manifest_image=_image(),
    )
    assert result.terminal_status == "update_failure"
    assert services.events == ["preflight"]


def test_production_shaped_image_uses_parent_manifest_binding_and_preserves_owner_order(
    tmp_path: Path,
) -> None:
    image = SimpleNamespace(
        image_id=1584,
        panel_row_sha256="2" * 64,
        image_sha256="3" * 64,
        owners=(),
        g_owner_ids=("g:1584:2", "g:1584:11"),
        h_owner_ids=("h:1584:3",),
        m_owner_ids=("m:1584:4",),
    )
    identity = _manifest_image_identity(_config(tmp_path), image, default_binding())
    assert identity["panel_sha256"] == default_binding().panel.panel_sha256
    assert identity["tokenizer_sha256"] == default_binding().surface.tokenizer_sha256
    assert identity["g_owner_ids"] == ("g:1584:2", "g:1584:11")


def test_partial_audit_open_is_closed_without_leaking_training_session(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)

    class OpenAuditFail(_FakeServices):
        def open_audit(self, config: EntryConfig, resources: Any) -> object:
            self.events.append("open_audit")
            raise RuntimeError("audit open failure")

    services = OpenAuditFail()
    result = run_one_image(
        config,
        authority=ExecutionAuthority(user_model_gpu_authority=True),
        resources=_resources(),
        output_root=Path(config.output_root),
        services=services,
        manifest_image=_image(),
    )
    assert result.terminal_status == "update_failure"
    assert services.events[-1] == "close"


def test_private_bytes_failure_cleans_partial_private_proposal(tmp_path: Path) -> None:
    config = _config(tmp_path)

    class PrivateBytesFail(_FakeServices):
        def write_private_proposal(
            self, training_session: object, proposal: object, output_root: Path
        ) -> object:
            del training_session, proposal, output_root
            self.events.append("write_private")
            self.private = object()
            error = RuntimeError("private bytes failure")
            error.private_proposal = self.private  # type: ignore[attr-defined]
            raise error

    services = PrivateBytesFail()
    result = run_one_image(
        config,
        authority=ExecutionAuthority(user_model_gpu_authority=True),
        resources=_resources(),
        output_root=Path(config.output_root),
        services=services,
        manifest_image=_image(),
    )
    assert result.terminal_status == "update_failure"
    assert "cleanup_private" in services.events


def test_audit_arithmetic_reuses_canonical_matcher_and_parser() -> None:
    image = _image()
    source = {1.0: _row(), 1.1: _row(repetition_penalty=1.1)}
    proposal = {1.0: _proposal_row(), 1.1: _proposal_row(repetition_penalty=1.1)}
    result = analyze_audit_pair(
        image=image,
        source_outputs=source,
        proposal_outputs=proposal,
        acquired_h_owner_ids=("h-1",),
        **_audit_identity_kwargs(),
    )

    assert isinstance(result, AuditPairAnalysis)
    assert result.by_repetition_penalty[1.0].g_owner_ids == ("g-1",)
    assert result.by_repetition_penalty[1.0].h_gain_owner_ids == ()
    assert result.by_repetition_penalty[1.0].g_loss_owner_ids == ()
    assert result.by_repetition_penalty[1.0].net_unique_delta == 0
    assert result.by_repetition_penalty[1.0].row_count == 1
    assert result.by_repetition_penalty[1.0].token_count == 3


def test_audit_rejects_bad_image_provenance_stop_and_counts_canonical_malformed_rows() -> (
    None
):
    image = _image()
    with pytest.raises(ValueError, match="image_id"):
        analyze_audit_pair(
            image=image,
            source_outputs={
                1.0: _row(image_id=2299),
                1.1: _row(image_id=2299, repetition_penalty=1.1),
            },
            proposal_outputs={
                1.0: _proposal_row(),
                1.1: _proposal_row(repetition_penalty=1.1),
            },
            acquired_h_owner_ids=("h-1",),
            **_audit_identity_kwargs(),
        )
    with pytest.raises(ValueError, match="provenance"):
        analyze_audit_pair(
            image=image,
            source_outputs={
                1.0: _row(provenance=False),
                1.1: _row(repetition_penalty=1.1),
            },
            proposal_outputs={
                1.0: _proposal_row(),
                1.1: _proposal_row(repetition_penalty=1.1),
            },
            acquired_h_owner_ids=("h-1",),
            **_audit_identity_kwargs(),
        )
    bad_nested_identity = _row()
    bad_nested_identity["provenance"] = dict(bad_nested_identity["provenance"])
    bad_nested_identity["provenance"]["image_id"] = 2299
    with pytest.raises(ValueError, match="provenance.image_id"):
        analyze_audit_pair(
            image=image,
            source_outputs={
                1.0: bad_nested_identity,
                1.1: _row(repetition_penalty=1.1),
            },
            proposal_outputs={
                1.0: _proposal_row(),
                1.1: _proposal_row(repetition_penalty=1.1),
            },
            acquired_h_owner_ids=("h-1",),
            **_audit_identity_kwargs(),
        )
    with pytest.raises(ValueError, match="stop_reason"):
        analyze_audit_pair(
            image=image,
            source_outputs={
                1.0: _row(stop="unknown"),
                1.1: _row(repetition_penalty=1.1),
            },
            proposal_outputs={
                1.0: _proposal_row(),
                1.1: _proposal_row(repetition_penalty=1.1),
            },
            acquired_h_owner_ids=("h-1",),
            **_audit_identity_kwargs(),
        )
    bad_surface = _row()
    bad_surface["provenance"] = dict(bad_surface["provenance"])
    bad_surface["provenance"]["backend"] = "vllm"
    with pytest.raises(ValueError, match="clean HF greedy"):
        analyze_audit_pair(
            image=image,
            source_outputs={1.0: bad_surface, 1.1: _row(repetition_penalty=1.1)},
            proposal_outputs={
                1.0: _proposal_row(),
                1.1: _proposal_row(repetition_penalty=1.1),
            },
            acquired_h_owner_ids=("h-1",),
            **_audit_identity_kwargs(),
        )
    bad_identity = _proposal_row()
    bad_identity["provenance"] = dict(bad_identity["provenance"])
    bad_identity["provenance"]["checkpoint_payload_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="identity"):
        analyze_audit_pair(
            image=image,
            source_outputs={1.0: _row(), 1.1: _row(repetition_penalty=1.1)},
            proposal_outputs={
                1.0: bad_identity,
                1.1: _proposal_row(repetition_penalty=1.1),
            },
            acquired_h_owner_ids=("h-1",),
            **_audit_identity_kwargs(),
        )
    distinct_proposal = _proposal_row()
    distinct_proposal_rp = _proposal_row(repetition_penalty=1.1)
    accepted_distinct = analyze_audit_pair(
        image=image,
        source_outputs={1.0: _row(), 1.1: _row(repetition_penalty=1.1)},
        proposal_outputs={1.0: distinct_proposal, 1.1: distinct_proposal_rp},
        acquired_h_owner_ids=("h-1",),
        **_audit_identity_kwargs(proposal="e" * 64),
    )
    assert isinstance(accepted_distinct, AuditPairAnalysis)
    malformed_row = _row()
    malformed_row["predictions"].append(
        {"generated_order": 1, "description": "person", "bbox": [1.0, 1.0, 1.0, 1.0]}
    )
    projected = analyze_audit_pair(
        image=image,
        source_outputs={1.0: malformed_row, 1.1: _row(repetition_penalty=1.1)},
        proposal_outputs={
            1.0: _proposal_row(),
            1.1: _proposal_row(repetition_penalty=1.1),
        },
        acquired_h_owner_ids=("h-1",),
        **_audit_identity_kwargs(),
    )
    assert projected.by_repetition_penalty[1.0].proposal_malformed_rows == 0
    assert projected.by_repetition_penalty[1.0].source_malformed_rows == 1


def test_audit_rejects_conflicting_nested_surface_and_unbound_checkpoint_path() -> None:
    image = _image()
    conflicting_surface = _row()
    conflicting_surface["decode_mode"] = "original_prompt_clean_greedy"
    conflicting_surface["backend"] = "hf"
    conflicting_surface["provenance"] = dict(conflicting_surface["provenance"])
    conflicting_surface["provenance"]["decode_mode"] = "vllm"
    conflicting_surface["provenance"]["backend"] = "vllm"
    with pytest.raises(ValueError, match="differs between output and provenance"):
        analyze_audit_pair(
            image=image,
            source_outputs={
                1.0: conflicting_surface,
                1.1: _row(repetition_penalty=1.1),
            },
            proposal_outputs={
                1.0: _proposal_row(),
                1.1: _proposal_row(repetition_penalty=1.1),
            },
            acquired_h_owner_ids=("h-1",),
            **_audit_identity_kwargs(),
        )

    conflicting_payload = _row()
    conflicting_payload["checkpoint_payload_sha256"] = "e" * 64
    conflicting_payload["provenance"] = dict(conflicting_payload["provenance"])
    with pytest.raises(ValueError, match="checkpoint_payload_sha256"):
        analyze_audit_pair(
            image=image,
            source_outputs={
                1.0: conflicting_payload,
                1.1: _row(repetition_penalty=1.1),
            },
            proposal_outputs={
                1.0: _proposal_row(),
                1.1: _proposal_row(repetition_penalty=1.1),
            },
            acquired_h_owner_ids=("h-1",),
            **_audit_identity_kwargs(),
        )

    forged_checkpoint_path = _row()
    forged_checkpoint_path["provenance"] = dict(forged_checkpoint_path["provenance"])
    forged_checkpoint_path["provenance"]["checkpoint_path"] = "/forged/checkpoint"
    with pytest.raises(ValueError, match="checkpoint_path"):
        analyze_audit_pair(
            image=image,
            source_outputs={
                1.0: forged_checkpoint_path,
                1.1: _row(repetition_penalty=1.1),
            },
            proposal_outputs={
                1.0: _proposal_row(),
                1.1: _proposal_row(repetition_penalty=1.1),
            },
            acquired_h_owner_ids=("h-1",),
            **_audit_identity_kwargs(),
        )

    missing_checkpoint_path = _proposal_row()
    missing_checkpoint_path["provenance"] = dict(missing_checkpoint_path["provenance"])
    missing_checkpoint_path["provenance"].pop("checkpoint_path")
    with pytest.raises(ValueError, match="checkpoint_path"):
        analyze_audit_pair(
            image=image,
            source_outputs={1.0: _row(), 1.1: _row(repetition_penalty=1.1)},
            proposal_outputs={
                1.0: missing_checkpoint_path,
                1.1: _proposal_row(repetition_penalty=1.1),
            },
            acquired_h_owner_ids=("h-1",),
            **_audit_identity_kwargs(),
        )


def test_audit_requires_canonical_parser_and_lineage_fields() -> None:
    image = _image()
    missing_parser = _row()
    missing_parser.pop("parser")
    missing_parser["provenance"] = dict(missing_parser["provenance"])
    missing_parser["provenance"].pop("parser")
    with pytest.raises(ValueError, match="parser"):
        analyze_audit_pair(
            image=image,
            source_outputs={1.0: missing_parser, 1.1: _row(repetition_penalty=1.1)},
            proposal_outputs={
                1.0: _proposal_row(),
                1.1: _proposal_row(repetition_penalty=1.1),
            },
            acquired_h_owner_ids=("h-1",),
            **_audit_identity_kwargs(),
        )


def test_entry_config_rejects_mutated_frozen_leaf_fields(tmp_path: Path) -> None:
    config = _config(tmp_path).to_dict()
    mutations = (
        ("training", "temperature", 0.7, "temperature"),
        ("optimizer", "betas", [0.1, 0.2], "betas"),
        ("optimizer", "epsilon", 0.3, "epsilon"),
        ("optimizer", "weight_decay", 0.9, "weight decay"),
        ("resource", "training_gpu", 2, "resource contract"),
        ("resource", "retry_policy", "retry", "resource contract"),
        ("audit", "duplicate_iou_threshold", 0.9, "matcher_duplicate_iou"),
    )
    for section, field, value, message in mutations:
        mutated = config | {section: dict(config[section])}
        mutated[section][field] = value
        with pytest.raises(ValueError, match=message):
            EntryConfig.from_mapping(mutated)


def test_acquired_h_ids_cannot_include_fabricated_g_owner(tmp_path: Path) -> None:
    config = _config(tmp_path)

    class FabricatedH(_FakeServices):
        def acquire_and_replay(
            self, training_session: object, config: EntryConfig
        ) -> object:
            del training_session, config
            self.events.append("acquire_replay")
            return SimpleNamespace(trusted_h_owner_ids=("g-1",), parity_passed=True)

    services = FabricatedH()
    result = run_one_image(
        config,
        authority=ExecutionAuthority(user_model_gpu_authority=True),
        resources=_resources(),
        output_root=Path(config.output_root),
        services=services,
        manifest_image=_image(),
    )
    assert result.terminal_status == "update_failure"


def test_proposal_gain_of_unsampled_manifest_h_does_not_pass_continuation(
    tmp_path: Path,
) -> None:
    import importlib.util
    import sys

    fixture_path = Path(__file__).with_name("test_human13_cuda_cpu_adapter.py")
    spec = importlib.util.spec_from_file_location("_unsampled_h_fixture", fixture_path)
    assert spec is not None and spec.loader is not None
    fixture = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = fixture
    spec.loader.exec_module(fixture)
    surface, _ = fixture._task2_surface(module_name="_unsampled_h_surface")
    assert not any(
        row.outcome == "trusted_first_hit" and row.owner_stratum == "H"
        for image_ledger in surface.trajectory_ledger.images
        for trajectory in image_ledger.trajectories
        for row in trajectory.rows
    )

    base_image = _image()
    image = SimpleNamespace(
        **{
            **vars(base_image),
            "owners": (
                *base_image.owners,
                SimpleNamespace(
                    owner_id="h-sampled",
                    category="person",
                    bbox=(40.0, 40.0, 50.0, 50.0),
                    source_object_index=1,
                ),
                SimpleNamespace(
                    owner_id="h-unsampled",
                    category="person",
                    bbox=(20.0, 20.0, 30.0, 30.0),
                    source_object_index=2,
                ),
            ),
            "h_owner_ids": ("h-sampled", "h-unsampled"),
        }
    )

    class UnsampledGain(_FakeServices):
        def acquire_and_replay(
            self, training_session: object, config: EntryConfig
        ) -> object:
            del training_session, config
            self.events.append("acquire_replay")
            return SimpleNamespace(
                trusted_h_owner_ids=("h-unsampled",),
                parity_passed=True,
                cuda_proposal_input=SimpleNamespace(
                    trajectory_ledger=surface.trajectory_ledger
                ),
            )

        def proposal_audit(
            self, audit_session: object, private: object, repetition_penalty: float
        ) -> dict[str, Any]:
            del audit_session, private
            self.events.append(f"proposal_audit:{repetition_penalty}")
            result = _proposal_row(
                owner="h-unsampled", repetition_penalty=repetition_penalty
            )
            result["predictions"].insert(
                0,
                {
                    "generated_order": 0,
                    "description": "person",
                    "bbox": [0.0, 0.0, 10.0, 10.0],
                },
            )
            result["predictions"][1]["generated_order"] = 1
            return result

    services = UnsampledGain()
    result = run_one_image(
        _config(tmp_path),
        authority=ExecutionAuthority(user_model_gpu_authority=True),
        resources=_resources(),
        output_root=tmp_path / "one-image",
        services=services,
        manifest_image=image,
    )

    assert result.terminal_status == "completed_null_or_unsafe"
    assert result.continuation_gate_sha256 is not None


def test_failed_proposal_audit_attempts_rollback_exactly_once(tmp_path: Path) -> None:
    config = _config(tmp_path)

    class ProposalAuditFail(_FakeServices):
        def proposal_audit(
            self, audit_session: object, private: object, repetition_penalty: float
        ) -> dict[str, Any]:
            del audit_session, private
            self.events.append(f"proposal_audit:{repetition_penalty}")
            raise RuntimeError("proposal audit failed")

    services = ProposalAuditFail()
    result = run_one_image(
        config,
        authority=ExecutionAuthority(user_model_gpu_authority=True),
        resources=_resources(),
        output_root=Path(config.output_root),
        services=services,
        manifest_image=_image(),
    )
    assert result.terminal_status == "update_failure"
    assert services.events.count("rollback_reproduce") == 1


def test_continuation_gate_truth_table_requires_h_gain_positive_delta_and_no_burdens() -> (
    None
):
    passing = AuditPairAnalysis(
        by_repetition_penalty={
            1.0: AuditAnalysis(
                repetition_penalty=1.0,
                source_owner_ids=("g-1",),
                proposal_owner_ids=("g-1", "h-1"),
                h_gain_owner_ids=("h-1",),
                g_loss_owner_ids=(),
                m_gain_owner_ids=(),
                net_unique_delta=1,
                source_duplicate_rows=0,
                proposal_duplicate_rows=0,
                source_unmatched_rows=0,
                proposal_unmatched_rows=0,
                source_malformed_rows=0,
                proposal_malformed_rows=0,
                source_cap_stops=0,
                proposal_cap_stops=0,
                row_count=2,
                token_count=3,
            ),
            1.1: AuditAnalysis(
                repetition_penalty=1.1,
                source_owner_ids=("g-1",),
                proposal_owner_ids=("g-1", "h-1"),
                h_gain_owner_ids=(),
                g_loss_owner_ids=(),
                m_gain_owner_ids=(),
                net_unique_delta=1,
                source_duplicate_rows=0,
                proposal_duplicate_rows=0,
                source_unmatched_rows=0,
                proposal_unmatched_rows=0,
                source_malformed_rows=0,
                proposal_malformed_rows=0,
                source_cap_stops=0,
                proposal_cap_stops=0,
                row_count=2,
                token_count=3,
            ),
        }
    )
    gate = evaluate_continuation_gate(passing)
    assert gate.admitted is True

    for field in (
        "net_unique_delta",
        "proposal_duplicate_rows",
        "proposal_malformed_rows",
        "proposal_cap_stops",
    ):
        bad = dict(passing.by_repetition_penalty)
        first = bad[1.0]
        bad[1.0] = AuditAnalysis(
            **{**first.__dict__, field: 0 if field == "net_unique_delta" else 1}
        )
        assert (
            evaluate_continuation_gate(
                AuditPairAnalysis(by_repetition_penalty=bad)
            ).admitted
            is False
        )


def test_full_panel_requires_exact_passing_terminal_hash_without_model_actions(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    with pytest.raises(PermissionError, match="terminal"):
        run_full_panel(
            config,
            one_image_terminal_hash=None,
            expected_terminal_hash="a" * 64,
            authority=ExecutionAuthority(user_model_gpu_authority=True),
            resources=_resources(),
            output_root=Path(config.output_root),
            services=_FakeServices(),
        )

    with pytest.raises(PermissionError, match="sealed"):
        run_full_panel(
            config,
            one_image_terminal_hash="a" * 64,
            expected_terminal_hash="a" * 64,
            authority=ExecutionAuthority(user_model_gpu_authority=True),
            resources=_resources(),
            output_root=Path(config.output_root),
            services=_FakeServices(),
        )

    issued = run_one_image(
        config,
        authority=ExecutionAuthority(user_model_gpu_authority=True),
        resources=_resources(),
        output_root=Path(config.output_root),
        services=_FakeServices(),
        manifest_image=_image(),
    )
    fabricated = replace(
        issued,
        terminal_status="passing_one_image",
        private_proposal_cleaned=True,
        source_reproduced=True,
    )
    with pytest.raises(PermissionError, match="issued sealed"):
        run_full_panel(
            config,
            one_image_terminal_hash=fabricated.content_sha256,
            expected_terminal_hash=fabricated.content_sha256,
            one_image_terminal=fabricated,
            authority=ExecutionAuthority(user_model_gpu_authority=True),
            resources=_resources(),
            output_root=Path(config.output_root),
            services=_FakeServices(),
        )


def test_terminal_seal_requires_guarded_issuer(tmp_path: Path) -> None:
    receipt = dry_run(_config(tmp_path))
    with pytest.raises(PermissionError, match="issuer"):
        _seal_terminal(receipt)


def test_cleanup_exception_is_preserved_as_terminal_failure(tmp_path: Path) -> None:
    config = _config(tmp_path)

    class CleanupFail(_FakeServices):
        def cleanup_private_proposal(self, private: object) -> None:
            super().cleanup_private_proposal(private)
            raise RuntimeError("cleanup failed")

    result = run_one_image(
        config,
        authority=ExecutionAuthority(user_model_gpu_authority=True),
        resources=_resources(),
        output_root=Path(config.output_root),
        services=CleanupFail(),
        manifest_image=_image(),
    )
    assert result.terminal_status == "update_failure"
    assert "cleanup" in (result.failure_reason or "")


def test_confirm_absent_output_root_does_not_create_it(tmp_path: Path) -> None:
    root = tmp_path / "new-root"
    receipt = confirm_absent_output_root(root)
    assert receipt.existed_before is False
    assert not root.exists()
