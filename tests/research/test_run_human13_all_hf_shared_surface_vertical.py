from __future__ import annotations

from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.research.run_human13_all_hf_shared_surface_vertical import (
    ALL_HF_VERTICAL_UNIT_ID,
    AuditAnalysis,
    AuditPairAnalysis,
    DualGPUResourceReceipt,
    EntryConfig,
    ExecutionAuthority,
    GPUResource,
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
)
from scripts.research.build_human13_k_union_manifest import default_binding


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
        seed_groups=((35001, 35002, 35003, 35004), (35005, 35006, 35007, 35008), (35009, 35010, 35011, 35012), (35013, 35014, 35015, 35016)),
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

    def preflight_source_assembly(self, config: EntryConfig, resources: Any) -> SourceAssemblyReceipt:
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

    def source_audit(self, audit_session: object, repetition_penalty: float) -> dict[str, Any]:
        del audit_session
        self.events.append(f"source_audit:{repetition_penalty}")
        return _row(repetition_penalty=repetition_penalty)

    def acquire_and_replay(self, training_session: object, config: EntryConfig) -> object:
        del training_session, config
        self.events.append("acquire_replay")
        return SimpleNamespace(trusted_h_owner_ids=("h-1",), parity_passed=True)

    def apply_private_update(self, training_session: object, acquisition: object, config: EntryConfig) -> object:
        del training_session, acquisition, config
        self.events.append("private_update")
        if self.fail == "private_update":
            raise RuntimeError("update failure")
        return object()

    def write_private_proposal(self, training_session: object, proposal: object, output_root: Path) -> object:
        del training_session, proposal, output_root
        self.events.append("write_private")
        self.private = SimpleNamespace(
            checkpoint_path="/private/proposal",
            checkpoint_payload_sha256="e" * 64,
        )
        return self.private

    def proposal_audit(self, audit_session: object, private: object, repetition_penalty: float) -> dict[str, Any]:
        del audit_session, private
        self.events.append(f"proposal_audit:{repetition_penalty}")
        return _proposal_row(owner="g-1", repetition_penalty=repetition_penalty)

    def rollback_and_reproduce_source(self, training_session: object, proposal: object) -> bool:
        del training_session, proposal
        self.events.append("rollback_reproduce")
        return True

    def cleanup_private_proposal(self, private: object) -> None:
        assert private is self.private
        self.events.append("cleanup_private")

    def close(self, training_session: object, audit_session: object) -> None:
        del training_session, audit_session
        self.events.append("close")


def test_default_dry_run_has_zero_model_gpu_network_and_output_actions(tmp_path: Path) -> None:
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
    assert not Path(config.output_root).exists()


def test_dual_gpu_admission_requires_distinct_suitable_cards() -> None:
    cards = _resources()
    receipt = validate_dual_gpu_resources(cards)
    assert receipt.training_gpu == 0
    assert receipt.audit_gpu == 1

    with pytest.raises(ValueError, match="distinct"):
        validate_dual_gpu_resources((cards[0], GPUResource(0, 80 << 30, 70 << 30)))

    with pytest.raises(ValueError, match="suitable"):
        validate_dual_gpu_resources((cards[0], GPUResource(1, 80 << 30, 70 << 30, suitable=False)))


def test_execute_requires_explicit_model_gpu_authority_and_absent_root(tmp_path: Path) -> None:
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


def test_phase_order_binds_gpu_roles_and_cleans_private_proposal(tmp_path: Path) -> None:
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


def test_private_proposal_audit_binds_distinct_proposal_checkpoint_digest(tmp_path: Path) -> None:
    config = _config(tmp_path)

    class DistinctProposal(_FakeServices):
        def write_private_proposal(self, training_session: object, proposal: object, output_root: Path) -> object:
            del training_session, proposal, output_root
            self.events.append("write_private")
            self.private = SimpleNamespace(
                checkpoint_path="/private/proposal",
                checkpoint_payload_sha256="e" * 64,
            )
            return self.private

        def proposal_audit(self, audit_session: object, private: object, repetition_penalty: float) -> dict[str, Any]:
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


def test_private_proposal_identity_must_differ_from_source_before_audit(tmp_path: Path) -> None:
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


def test_prebuilt_resource_receipt_cannot_rebind_training_or_audit_roles(tmp_path: Path) -> None:
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


def test_parent_manifest_binding_is_authoritative_over_image_local_surface(tmp_path: Path) -> None:
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
        def preflight_source_assembly(self, config: EntryConfig, resources: Any) -> SourceAssemblyReceipt:
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


def test_partial_audit_open_is_closed_without_leaking_training_session(tmp_path: Path) -> None:
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
        def write_private_proposal(self, training_session: object, proposal: object, output_root: Path) -> object:
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


def test_audit_rejects_bad_image_provenance_stop_and_counts_canonical_malformed_rows() -> None:
    image = _image()
    with pytest.raises(ValueError, match="image_id"):
        analyze_audit_pair(
            image=image,
            source_outputs={1.0: _row(image_id=2299), 1.1: _row(image_id=2299, repetition_penalty=1.1)},
            proposal_outputs={1.0: _proposal_row(), 1.1: _proposal_row(repetition_penalty=1.1)},
            acquired_h_owner_ids=("h-1",),
            **_audit_identity_kwargs(),
        )
    with pytest.raises(ValueError, match="provenance"):
        analyze_audit_pair(
            image=image,
            source_outputs={1.0: _row(provenance=False), 1.1: _row(repetition_penalty=1.1)},
            proposal_outputs={1.0: _proposal_row(), 1.1: _proposal_row(repetition_penalty=1.1)},
            acquired_h_owner_ids=("h-1",),
            **_audit_identity_kwargs(),
        )
    bad_nested_identity = _row()
    bad_nested_identity["provenance"] = dict(bad_nested_identity["provenance"])
    bad_nested_identity["provenance"]["image_id"] = 2299
    with pytest.raises(ValueError, match="provenance.image_id"):
        analyze_audit_pair(
            image=image,
            source_outputs={1.0: bad_nested_identity, 1.1: _row(repetition_penalty=1.1)},
            proposal_outputs={1.0: _proposal_row(), 1.1: _proposal_row(repetition_penalty=1.1)},
            acquired_h_owner_ids=("h-1",),
            **_audit_identity_kwargs(),
        )
    with pytest.raises(ValueError, match="stop_reason"):
        analyze_audit_pair(
            image=image,
            source_outputs={1.0: _row(stop="unknown"), 1.1: _row(repetition_penalty=1.1)},
            proposal_outputs={1.0: _proposal_row(), 1.1: _proposal_row(repetition_penalty=1.1)},
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
            proposal_outputs={1.0: _proposal_row(), 1.1: _proposal_row(repetition_penalty=1.1)},
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
            proposal_outputs={1.0: bad_identity, 1.1: _proposal_row(repetition_penalty=1.1)},
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
        proposal_outputs={1.0: _proposal_row(), 1.1: _proposal_row(repetition_penalty=1.1)},
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
            source_outputs={1.0: conflicting_surface, 1.1: _row(repetition_penalty=1.1)},
            proposal_outputs={1.0: _proposal_row(), 1.1: _proposal_row(repetition_penalty=1.1)},
            acquired_h_owner_ids=("h-1",),
            **_audit_identity_kwargs(),
        )

    conflicting_payload = _row()
    conflicting_payload["checkpoint_payload_sha256"] = "e" * 64
    conflicting_payload["provenance"] = dict(conflicting_payload["provenance"])
    with pytest.raises(ValueError, match="checkpoint_payload_sha256"):
        analyze_audit_pair(
            image=image,
            source_outputs={1.0: conflicting_payload, 1.1: _row(repetition_penalty=1.1)},
        proposal_outputs={1.0: _proposal_row(), 1.1: _proposal_row(repetition_penalty=1.1)},
            acquired_h_owner_ids=("h-1",),
            **_audit_identity_kwargs(),
        )

    forged_checkpoint_path = _row()
    forged_checkpoint_path["provenance"] = dict(forged_checkpoint_path["provenance"])
    forged_checkpoint_path["provenance"]["checkpoint_path"] = "/forged/checkpoint"
    with pytest.raises(ValueError, match="checkpoint_path"):
        analyze_audit_pair(
            image=image,
            source_outputs={1.0: forged_checkpoint_path, 1.1: _row(repetition_penalty=1.1)},
            proposal_outputs={1.0: _proposal_row(), 1.1: _proposal_row(repetition_penalty=1.1)},
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
            proposal_outputs={1.0: _proposal_row(), 1.1: _proposal_row(repetition_penalty=1.1)},
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
        def acquire_and_replay(self, training_session: object, config: EntryConfig) -> object:
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


def test_failed_proposal_audit_attempts_rollback_exactly_once(tmp_path: Path) -> None:
    config = _config(tmp_path)

    class ProposalAuditFail(_FakeServices):
        def proposal_audit(self, audit_session: object, private: object, repetition_penalty: float) -> dict[str, Any]:
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



def test_continuation_gate_truth_table_requires_h_gain_positive_delta_and_no_burdens() -> None:
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

    for field in ("net_unique_delta", "proposal_duplicate_rows", "proposal_malformed_rows", "proposal_cap_stops"):
        bad = dict(passing.by_repetition_penalty)
        first = bad[1.0]
        bad[1.0] = AuditAnalysis(**{**first.__dict__, field: 0 if field == "net_unique_delta" else 1})
        assert evaluate_continuation_gate(AuditPairAnalysis(by_repetition_penalty=bad)).admitted is False


def test_full_panel_requires_exact_passing_terminal_hash_without_model_actions(tmp_path: Path) -> None:
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
