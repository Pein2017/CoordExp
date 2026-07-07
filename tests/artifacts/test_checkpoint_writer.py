from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest
import torch
from torch import nn

from src.adapters.dora import DoraAdapterSetupReceipt, DoraTargetDiscoveryReceipt
from src.artifacts import CheckpointWriter, MetricStreamEvent, RunArtifactManager
from src.artifacts.checkpoint_reload import (
    build_checkpoint_reload_plan,
    verify_checkpoint_reload_payloads,
)
from src.common.errors import ArtifactContractError
from src.config.models import RunDirectory
from src.optim.trainable_surface import FrozenReasonSummary, TrainableSurfaceReceipt
from src.qwen.special_token_embeddings import (
    DEFAULT_EMBED_DELTA_TENSOR_KEY,
    SPECIAL_TOKEN_EMBEDDINGS_JSON,
    SPECIAL_TOKEN_EMBEDDINGS_SAFE_TENSORS,
    SPECIAL_TOKEN_EMBEDDING_SEMANTICS,
    SpecialTokenEmbeddingInstallReceipt,
    SpecialTokenEmbeddingInstallResult,
    SpecialTokenSelection,
)


def test_checkpoint_writer_saves_payloads_metadata_and_final_alias(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path)
    writer = CheckpointWriter(manager)
    model = FakePeftModel()

    result = writer.write_checkpoint(
        planned_step_id=5,
        model=model,
        adapter_receipt=_adapter_receipt(),
        special_token_result=_special_token_result(),
        trainable_surface=_trainable_surface(),
        processor_identity={"name": "qwen3-vl-test-processor"},
        resolved_config_fingerprint="config-fingerprint",
        schedule_identity={"resolved_max_steps": 5, "fingerprint": "schedule"},
        metric_status={
            "finite_status": "finite",
            "warning_status": "warned",
            "best_selector": None,
        },
        optimizer_update_status="skipped_non_finite",
        trigger_reasons=("checkpoint.final",),
        is_final=True,
        base_model_path=Path("/models/qwen-base"),
        base_config_sha256="base-config-sha",
        tokenizer_sha256="tokenizer-sha",
    )

    checkpoint_dir = tmp_path / "run-a" / "checkpoints" / "step-5"
    metadata_path = checkpoint_dir / "checkpoint.json"
    handoff_path = checkpoint_dir / "checkpoint_handoff.json"
    final_alias_path = tmp_path / "run-a" / "checkpoints" / "checkpoint-final.json"

    assert result.metadata_path == metadata_path
    assert result.handoff_path == handoff_path
    assert metadata_path.exists()
    assert handoff_path.exists()
    assert final_alias_path.exists()
    assert not (tmp_path / "run-a" / "checkpoints" / "step-000005").exists()
    assert (checkpoint_dir / "adapter" / "adapter_config.json").exists()
    assert (checkpoint_dir / "adapter" / "adapter_model.safetensors").exists()
    assert (
        checkpoint_dir
        / "special_token_embeddings"
        / SPECIAL_TOKEN_EMBEDDINGS_SAFE_TENSORS
    ).exists()
    assert (
        checkpoint_dir / "special_token_embeddings" / SPECIAL_TOKEN_EMBEDDINGS_JSON
    ).exists()

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["checkpoint_id"] == "step-5"
    assert metadata["planned_step_id"] == 5
    assert metadata["checkpoint_path"] == "checkpoints/step-5"
    assert metadata["optimizer_update_status"] == "skipped_non_finite"
    assert metadata["resolved_config_fingerprint"] == "config-fingerprint"
    assert metadata["checkpoint_handoff"] == "checkpoints/step-5/checkpoint_handoff.json"
    assert metadata["processor_identity"] == {"name": "qwen3-vl-test-processor"}
    assert metadata["adapter"]["enabled"] is True
    assert metadata["adapter"]["payload_path"] == "checkpoints/step-5/adapter"
    adapter_identity = metadata["adapter"]["identity"]
    assert adapter_identity["payload_path"] == "checkpoints/step-5/adapter"
    assert adapter_identity["required_files"] == {
        "adapter_config.json": "checkpoints/step-5/adapter/adapter_config.json",
        "adapter_model.safetensors": "checkpoints/step-5/adapter/adapter_model.safetensors",
    }
    assert adapter_identity["adapter_config_sha256"] == _sha256(
        checkpoint_dir / "adapter" / "adapter_config.json"
    )
    assert adapter_identity["adapter_model_sha256"] == _sha256(
        checkpoint_dir / "adapter" / "adapter_model.safetensors"
    )
    assert adapter_identity["fingerprint"]
    assert metadata["special_token_embeddings"]["enabled"] is True
    assert metadata["special_token_embeddings"]["tensor_path"] == (
        "checkpoints/step-5/special_token_embeddings/"
        f"{SPECIAL_TOKEN_EMBEDDINGS_SAFE_TENSORS}"
    )
    embedding_identity = metadata["special_token_embeddings"]["identity"]
    assert embedding_identity["metadata_path"] == (
        "checkpoints/step-5/special_token_embeddings/"
        f"{SPECIAL_TOKEN_EMBEDDINGS_JSON}"
    )
    assert embedding_identity["tensor_path"] == (
        "checkpoints/step-5/special_token_embeddings/"
        f"{SPECIAL_TOKEN_EMBEDDINGS_SAFE_TENSORS}"
    )
    assert embedding_identity["metadata_sha256"] == _sha256(
        checkpoint_dir / "special_token_embeddings" / SPECIAL_TOKEN_EMBEDDINGS_JSON
    )
    assert embedding_identity["tensor_sha256"] == _sha256(
        checkpoint_dir / "special_token_embeddings" / SPECIAL_TOKEN_EMBEDDINGS_SAFE_TENSORS
    )
    assert embedding_identity["tensor_key"] == DEFAULT_EMBED_DELTA_TENSOR_KEY
    assert embedding_identity["tensor_shape"] == [2, 4]
    assert embedding_identity["tensor_dtype"] == "float32"
    assert embedding_identity["base_config_sha256"] == "base-config-sha"
    assert embedding_identity["tokenizer_sha256"] == "tokenizer-sha"
    assert embedding_identity["fingerprint"]
    assert metadata["special_token_embeddings"]["metadata"]["semantics"] == (
        SPECIAL_TOKEN_EMBEDDING_SEMANTICS
    )
    assert (
        metadata["special_token_embeddings"]["metadata"]["base_config_sha256"]
        == "base-config-sha"
    )
    assert (
        metadata["special_token_embeddings"]["metadata"]["tokenizer_sha256"]
        == "tokenizer-sha"
    )
    payload_metadata = json.loads(
        (
            checkpoint_dir
            / "special_token_embeddings"
            / SPECIAL_TOKEN_EMBEDDINGS_JSON
        ).read_text(encoding="utf-8")
    )
    assert payload_metadata["base_config_sha256"] == "base-config-sha"
    assert payload_metadata["tokenizer_sha256"] == "tokenizer-sha"
    assert metadata["trainable_surface"]["trainable_towers"] == [
        "adapter.language",
        "token_embeddings",
    ]
    assert "optimizer_state" not in metadata
    assert "scheduler_state" not in metadata
    assert "rng_state" not in metadata
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    assert handoff["base_model"] == {
        "path": "/models/qwen-base",
        "base_config_sha256": "base-config-sha",
        "tokenizer_sha256": "tokenizer-sha",
    }
    assert handoff["adapter"]["payload_path"] == "checkpoints/step-5/adapter"
    assert handoff["adapter"]["files"] == [
        "checkpoints/step-5/adapter/adapter_config.json",
        "checkpoints/step-5/adapter/adapter_model.safetensors",
    ]
    assert handoff["adapter"]["identity"] == adapter_identity
    assert handoff["special_token_embeddings"]["metadata_path"] == (
        "checkpoints/step-5/special_token_embeddings/"
        f"{SPECIAL_TOKEN_EMBEDDINGS_JSON}"
    )
    assert handoff["special_token_embeddings"]["tensor_dtype"] == "float32"
    assert handoff["special_token_embeddings"]["identity"] == embedding_identity
    assert handoff["trainable_token_set"]["token_ids"] == [2, 3]
    assert handoff["trainable_token_set"]["token_strings"] == [
        "<|object_ref_start|>",
        "<|coord_0|>",
    ]
    assert handoff["intended_inference_config_family"] == "configs/coordexp_swift/infer"
    assert handoff["accepted_eval_artifact_roots"] == []

    alias = json.loads(final_alias_path.read_text(encoding="utf-8"))
    assert alias["checkpoint_id"] == "step-5"
    assert alias["metadata_path"] == "checkpoints/step-5/checkpoint.json"
    assert alias["handoff_path"] == "checkpoints/step-5/checkpoint_handoff.json"

    manifest = manager.read_manifest()
    assert manifest["checkpoints"]["items"] == [
        "checkpoints/step-5/checkpoint.json"
    ]
    assert manifest["checkpoints"]["aliases"]["final"] == (
        "checkpoints/checkpoint-final.json"
    )


@pytest.mark.parametrize(
    ("base_config_sha256", "tokenizer_sha256", "missing_field"),
    [
        (None, "tokenizer-sha", "base_config_sha256"),
        ("", "tokenizer-sha", "base_config_sha256"),
        ("base-config-sha", None, "tokenizer_sha256"),
        ("base-config-sha", "", "tokenizer_sha256"),
    ],
)
def test_checkpoint_writer_requires_special_token_sha_evidence(
    tmp_path: Path,
    base_config_sha256: str | None,
    tokenizer_sha256: str | None,
    missing_field: str,
) -> None:
    writer = CheckpointWriter(_manager(tmp_path))

    with pytest.raises(ArtifactContractError) as exc_info:
        writer.write_checkpoint(
            planned_step_id=5,
            model=FakePeftModel(),
            adapter_receipt=_adapter_receipt(),
            special_token_result=_special_token_result(),
            trainable_surface=_trainable_surface(),
            processor_identity={"name": "qwen3-vl-test-processor"},
            resolved_config_fingerprint="config-fingerprint",
            schedule_identity={"resolved_max_steps": 5, "fingerprint": "schedule"},
            metric_status={"finite_status": "finite", "warning_status": "none"},
            optimizer_update_status="applied",
            trigger_reasons=("checkpoint.final",),
            is_final=True,
            base_model_path=Path("/models/qwen-base"),
            base_config_sha256=base_config_sha256,
            tokenizer_sha256=tokenizer_sha256,
        )

    assert exc_info.value.code == "checkpoint.special_token_identity_missing"
    assert exc_info.value.context["missing_field"] == missing_field


def test_checkpoint_reload_plan_verifies_adapter_and_special_token_payloads(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path)
    writer = CheckpointWriter(manager)
    result = writer.write_checkpoint(
        planned_step_id=5,
        model=FakePeftModel(),
        adapter_receipt=_adapter_receipt(),
        special_token_result=_special_token_result(),
        trainable_surface=_trainable_surface(),
        processor_identity={"name": "qwen3-vl-test-processor"},
        resolved_config_fingerprint="config-fingerprint",
        schedule_identity={"resolved_max_steps": 5, "fingerprint": "schedule"},
        metric_status={"finite_status": "finite", "warning_status": "none"},
        optimizer_update_status="applied",
        trigger_reasons=("checkpoint.final",),
        is_final=True,
        base_model_path=Path("model_cache/qwen-base"),
        base_config_sha256="base-config-sha",
        tokenizer_sha256="tokenizer-sha",
    )

    plan = build_checkpoint_reload_plan(result.final_alias_path)
    receipt = verify_checkpoint_reload_payloads(plan)

    assert plan.checkpoint_id == "step-5"
    assert plan.base_model_path == Path("model_cache/qwen-base")
    assert plan.adapter_dir == manager.run_dir / "checkpoints" / "step-5" / "adapter"
    assert receipt["adapter"]["enabled"] is True
    assert receipt["adapter"]["config"]["use_dora"] is True
    assert receipt["adapter"]["weight_files"] == [
        "checkpoints/step-5/adapter/adapter_model.safetensors"
    ]
    assert receipt["special_token_embeddings"]["enabled"] is True
    assert receipt["special_token_embeddings"]["tensor_key"] == DEFAULT_EMBED_DELTA_TENSOR_KEY
    assert receipt["special_token_embeddings"]["tensor_shape"] == [2, 4]
    assert receipt["reload_contract"] == "base_model_plus_dora_adapter_plus_token_embed_delta"


def test_checkpoint_writer_best_acc_top1_ignores_unsafe_candidate(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path)
    writer = CheckpointWriter(manager)

    first = writer.write_checkpoint(
        planned_step_id=1,
        model=None,
        adapter_receipt=None,
        special_token_result=None,
        trainable_surface=_trainable_surface(),
        processor_identity={"name": "processor"},
        resolved_config_fingerprint="config",
        schedule_identity={"resolved_max_steps": 2},
        metric_status={"finite_status": "finite", "warning_status": "none"},
        optimizer_update_status="applied",
        trigger_reasons=("eval.forward",),
        best_metric_event=MetricStreamEvent(
            event_type="metric",
            planned_step_id=1,
            split="eval.forward",
            name="acc_top1",
            value=0.25,
            trigger_reasons=("eval.forward",),
            optimizer_update_status="applied",
            finite_status="finite",
            warning_status="none",
        ),
    )

    writer.write_checkpoint(
        planned_step_id=2,
        model=None,
        adapter_receipt=None,
        special_token_result=None,
        trainable_surface=_trainable_surface(),
        processor_identity={"name": "processor"},
        resolved_config_fingerprint="config",
        schedule_identity={"resolved_max_steps": 2},
        metric_status={"finite_status": "non_finite", "warning_status": "warned"},
        optimizer_update_status="skipped_non_finite",
        trigger_reasons=("eval.forward",),
        best_metric_event=MetricStreamEvent(
            event_type="metric",
            planned_step_id=2,
            split="eval.forward",
            name="acc_top1",
            value=0.95,
            trigger_reasons=("eval.forward",),
            optimizer_update_status="skipped_non_finite",
            finite_status="non_finite",
            warning_status="warned",
        ),
    )

    best_alias_path = tmp_path / "run-a" / "checkpoints" / "best_acc_top1.json"
    best_alias = json.loads(best_alias_path.read_text(encoding="utf-8"))
    assert best_alias["checkpoint_id"] == "step-1"
    assert best_alias["metadata_path"] == manager.relative_artifact_path(
        first.metadata_path
    )
    assert best_alias["metric"]["value"] == 0.25

    manifest = manager.read_manifest()
    assert manifest["checkpoints"]["aliases"]["best_acc_top1"] == (
        "checkpoints/best_acc_top1.json"
    )
    assert manifest["checkpoints"]["best_acc_top1"]["checkpoint_id"] == "step-1"


def test_checkpoint_writer_rejects_best_candidate_status_mismatch(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path)
    writer = CheckpointWriter(manager)
    writer.write_checkpoint(
        planned_step_id=1,
        model=None,
        adapter_receipt=None,
        special_token_result=None,
        trainable_surface=_trainable_surface(),
        processor_identity={"name": "processor"},
        resolved_config_fingerprint="config",
        schedule_identity={"resolved_max_steps": 2},
        metric_status={"finite_status": "finite", "warning_status": "none"},
        optimizer_update_status="applied",
        trigger_reasons=("eval.forward",),
        best_metric_event=MetricStreamEvent(
            event_type="metric",
            planned_step_id=1,
            split="eval.forward",
            name="acc_top1",
            value=0.25,
            trigger_reasons=("eval.forward",),
            optimizer_update_status="applied",
            finite_status="finite",
            warning_status="none",
        ),
    )

    with pytest.raises(ArtifactContractError) as exc_info:
        writer.write_checkpoint(
            planned_step_id=2,
            model=None,
            adapter_receipt=None,
            special_token_result=None,
            trainable_surface=_trainable_surface(),
            processor_identity={"name": "processor"},
            resolved_config_fingerprint="config",
            schedule_identity={"resolved_max_steps": 2},
            metric_status={"finite_status": "finite", "warning_status": "none"},
            optimizer_update_status="skipped_non_finite",
            trigger_reasons=("eval.forward",),
            best_metric_event={
                "event_type": "metric",
                "planned_step_id": 2,
                "split": "eval.forward",
                "name": "acc_top1",
                "value": 0.95,
                "trigger_reasons": ["eval.forward"],
                "optimizer_update_status": "applied",
                "finite_status": "finite",
                "warning_status": "none",
                "reduction": "global_mean",
                "rank": None,
                "world_size": None,
                "selector_eligible": True,
                "metadata": {},
            },
        )

    assert exc_info.value.code == "checkpoint.best_selector_status_mismatch"
    best_alias_path = tmp_path / "run-a" / "checkpoints" / "best_acc_top1.json"
    best_alias = json.loads(best_alias_path.read_text(encoding="utf-8"))
    assert best_alias["checkpoint_id"] == "step-1"
    assert not (tmp_path / "run-a" / "checkpoints" / "step-2").exists()


def test_checkpoint_writer_rejects_best_candidate_without_checkpoint_finite_status(
    tmp_path: Path,
) -> None:
    writer = CheckpointWriter(_manager(tmp_path))

    with pytest.raises(ArtifactContractError) as exc_info:
        writer.write_checkpoint(
            planned_step_id=1,
            model=None,
            adapter_receipt=None,
            special_token_result=None,
            trainable_surface=_trainable_surface(),
            processor_identity={"name": "processor"},
            resolved_config_fingerprint="config",
            schedule_identity={"resolved_max_steps": 1},
            metric_status={},
            optimizer_update_status="applied",
            trigger_reasons=("eval.forward",),
            best_metric_event=MetricStreamEvent(
                event_type="metric",
                planned_step_id=1,
                split="eval.forward",
                name="acc_top1",
                value=0.25,
                trigger_reasons=("eval.forward",),
                optimizer_update_status="applied",
                finite_status="finite",
                warning_status="none",
            ),
        )

    assert exc_info.value.code == "checkpoint.best_selector_status_missing"
    assert not (tmp_path / "run-a" / "checkpoints" / "step-1").exists()


def test_checkpoint_writer_repairs_manifest_after_registration_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager(tmp_path)
    writer = CheckpointWriter(manager)
    best_event = MetricStreamEvent(
        event_type="metric",
        planned_step_id=1,
        split="eval.forward",
        name="acc_top1",
        value=0.25,
        trigger_reasons=("eval.forward",),
        optimizer_update_status="applied",
        finite_status="finite",
        warning_status="none",
    )
    _fail_next_manifest_write(monkeypatch)

    with pytest.raises(ArtifactContractError) as exc_info:
        writer.write_checkpoint(
            planned_step_id=1,
            model=None,
            adapter_receipt=None,
            special_token_result=None,
            trainable_surface=_trainable_surface(),
            processor_identity={"name": "processor"},
            resolved_config_fingerprint="config",
            schedule_identity={"resolved_max_steps": 1},
            metric_status={"finite_status": "finite", "warning_status": "none"},
            optimizer_update_status="applied",
            trigger_reasons=("checkpoint.final", "eval.forward"),
            is_final=True,
            best_metric_event=best_event,
        )

    assert exc_info.value.code == "test.manifest_write_failed"
    assert (tmp_path / "run-a" / "checkpoints" / "step-1" / "checkpoint.json").exists()
    assert manager.read_manifest()["checkpoints"]["items"] == []

    result = writer.write_checkpoint(
        planned_step_id=1,
        model=None,
        adapter_receipt=None,
        special_token_result=None,
        trainable_surface=_trainable_surface(),
        processor_identity={"name": "processor"},
        resolved_config_fingerprint="config",
        schedule_identity={"resolved_max_steps": 1},
        metric_status={"finite_status": "finite", "warning_status": "none"},
        optimizer_update_status="applied",
        trigger_reasons=("checkpoint.final", "eval.forward"),
        is_final=True,
        best_metric_event=best_event,
    )

    manifest = manager.read_manifest()
    assert manifest["checkpoints"]["items"] == [
        "checkpoints/step-1/checkpoint.json"
    ]
    assert manifest["checkpoints"]["latest"]["checkpoint_id"] == "step-1"
    assert manifest["checkpoints"]["aliases"]["final"] == (
        "checkpoints/checkpoint-final.json"
    )
    assert manifest["checkpoints"]["aliases"]["best_acc_top1"] == (
        "checkpoints/best_acc_top1.json"
    )
    assert result.metadata_path == tmp_path / "run-a" / "checkpoints" / "step-1" / "checkpoint.json"


def test_checkpoint_writer_rejects_full_model_save_pretrained_payload(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path)
    writer = CheckpointWriter(manager)

    with pytest.raises(ArtifactContractError) as exc_info:
        writer.write_checkpoint(
            planned_step_id=1,
            model=FakeFullModel(),
            adapter_receipt=_adapter_receipt(),
            special_token_result=None,
            trainable_surface=_trainable_surface(),
            processor_identity={},
            resolved_config_fingerprint="config",
            schedule_identity={},
            metric_status={"finite_status": "finite", "warning_status": "none"},
            optimizer_update_status="applied",
            trigger_reasons=("checkpoint",),
        )

    assert exc_info.value.code == "checkpoint.adapter_payload_forbidden_file"
    assert manager.read_manifest()["checkpoints"]["items"] == []
    assert not (
        tmp_path / "run-a" / "checkpoints" / "step-1" / "adapter" / "config.json"
    ).exists()
    assert not (
        tmp_path / "run-a" / "checkpoints" / "step-1" / "adapter" / "model.safetensors"
    ).exists()
    assert not (
        tmp_path / "run-a" / "checkpoints" / "checkpoint-final.json"
    ).exists()

    result = writer.write_checkpoint(
        planned_step_id=1,
        model=FakePeftModel(),
        adapter_receipt=_adapter_receipt(),
        special_token_result=None,
        trainable_surface=_trainable_surface(),
        processor_identity={},
        resolved_config_fingerprint="config",
        schedule_identity={},
        metric_status={"finite_status": "finite", "warning_status": "none"},
        optimizer_update_status="applied",
        trigger_reasons=("checkpoint",),
    )

    assert result.metadata_path.exists()
    assert manager.read_manifest()["checkpoints"]["items"] == [
        "checkpoints/step-1/checkpoint.json"
    ]


def test_checkpoint_writer_marks_lower_best_candidate_not_improved(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path)
    writer = CheckpointWriter(manager)
    writer.write_checkpoint(
        planned_step_id=1,
        model=None,
        adapter_receipt=None,
        special_token_result=None,
        trainable_surface=_trainable_surface(),
        processor_identity={"name": "processor"},
        resolved_config_fingerprint="config",
        schedule_identity={"resolved_max_steps": 2},
        metric_status={"finite_status": "finite", "warning_status": "none"},
        optimizer_update_status="applied",
        trigger_reasons=("eval.forward",),
        best_metric_event=MetricStreamEvent(
            event_type="metric",
            planned_step_id=1,
            split="eval.forward",
            name="acc_top1",
            value=0.9,
            trigger_reasons=("eval.forward",),
            optimizer_update_status="applied",
            finite_status="finite",
            warning_status="none",
        ),
    )

    second = writer.write_checkpoint(
        planned_step_id=2,
        model=None,
        adapter_receipt=None,
        special_token_result=None,
        trainable_surface=_trainable_surface(),
        processor_identity={"name": "processor"},
        resolved_config_fingerprint="config",
        schedule_identity={"resolved_max_steps": 2},
        metric_status={"finite_status": "finite", "warning_status": "none"},
        optimizer_update_status="applied",
        trigger_reasons=("eval.forward",),
        best_metric_event=MetricStreamEvent(
            event_type="metric",
            planned_step_id=2,
            split="eval.forward",
            name="acc_top1",
            value=0.1,
            trigger_reasons=("eval.forward",),
            optimizer_update_status="applied",
            finite_status="finite",
            warning_status="none",
        ),
    )

    metadata = json.loads(second.metadata_path.read_text(encoding="utf-8"))
    assert metadata["best_selection"]["selected"] is False
    assert metadata["best_selection"]["reason"] == "not_improved"
    assert metadata["best_selection"]["current_best"]["checkpoint_id"] == "step-1"
    best_alias = json.loads(
        (tmp_path / "run-a" / "checkpoints" / "best_acc_top1.json").read_text(
            encoding="utf-8"
        )
    )
    assert best_alias["checkpoint_id"] == "step-1"


def test_checkpoint_writer_allows_warning_only_best_checkpoint(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path)
    writer = CheckpointWriter(manager)
    writer.write_checkpoint(
        planned_step_id=1,
        model=None,
        adapter_receipt=None,
        special_token_result=None,
        trainable_surface=_trainable_surface(),
        processor_identity={"name": "processor"},
        resolved_config_fingerprint="config",
        schedule_identity={"resolved_max_steps": 2},
        metric_status={"finite_status": "finite", "warning_status": "none"},
        optimizer_update_status="applied",
        trigger_reasons=("eval.forward",),
        best_metric_event=MetricStreamEvent(
            event_type="metric",
            planned_step_id=1,
            split="eval.forward",
            name="acc_top1",
            value=0.25,
            trigger_reasons=("eval.forward",),
            optimizer_update_status="applied",
            finite_status="finite",
            warning_status="none",
        ),
    )

    second = writer.write_checkpoint(
        planned_step_id=2,
        model=None,
        adapter_receipt=None,
        special_token_result=None,
        trainable_surface=_trainable_surface(),
        processor_identity={"name": "processor"},
        resolved_config_fingerprint="config",
        schedule_identity={"resolved_max_steps": 2},
        metric_status={"finite_status": "finite", "warning_status": "warned"},
        optimizer_update_status="applied",
        trigger_reasons=("eval.forward",),
        best_metric_event=MetricStreamEvent(
            event_type="metric",
            planned_step_id=2,
            split="eval.forward",
            name="acc_top1",
            value=0.95,
            trigger_reasons=("eval.forward",),
            optimizer_update_status="applied",
            finite_status="finite",
            warning_status="warned",
        ),
    )

    metadata = json.loads(second.metadata_path.read_text(encoding="utf-8"))
    assert metadata["best_selection"]["selected"] is True
    assert metadata["best_selection"]["reason"] == "improved"
    best_alias = json.loads(
        (tmp_path / "run-a" / "checkpoints" / "best_acc_top1.json").read_text(
            encoding="utf-8"
        )
    )
    assert best_alias["checkpoint_id"] == "step-2"


def test_checkpoint_writer_rejects_adapter_receipt_without_savable_model(
    tmp_path: Path,
) -> None:
    writer = CheckpointWriter(_manager(tmp_path))

    with pytest.raises(ArtifactContractError) as exc_info:
        writer.write_checkpoint(
            planned_step_id=1,
            model=nn.Linear(2, 2),
            adapter_receipt=_adapter_receipt(),
            special_token_result=None,
            trainable_surface=_trainable_surface(),
            processor_identity={},
            resolved_config_fingerprint="config",
            schedule_identity={},
            metric_status={},
            optimizer_update_status="applied",
            trigger_reasons=("checkpoint",),
        )

    assert exc_info.value.code == "checkpoint.adapter_model_unsavable"


class FakePeftModel(nn.Module):
    def save_pretrained(self, output_dir: str | Path) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "adapter_config.json").write_text(
            json.dumps({"peft_type": "LORA", "use_dora": True}) + "\n",
            encoding="utf-8",
        )
        (path / "adapter_model.safetensors").write_bytes(b"adapter")


class FakeFullModel(nn.Module):
    def save_pretrained(self, output_dir: str | Path) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text("{}\n", encoding="utf-8")
        (path / "model.safetensors").write_bytes(b"full-model")


def _manager(tmp_path: Path) -> RunArtifactManager:
    return RunArtifactManager.initialize(
        run_directory=RunDirectory(
            run_name="run-a",
            artifact_root=tmp_path,
            run_dir=tmp_path / "run-a",
            collision_policy="fail",
        ),
        run_id="run-a",
        created_at="2026-06-30T00:00:00Z",
        runtime_identity={},
        backend_status={"single": ["active"], "accelerate": [], "deepspeed": []},
    )


def _fail_next_manifest_write(monkeypatch: pytest.MonkeyPatch) -> None:
    original = RunArtifactManager._write_manifest
    remaining_failures = 1

    def fail_once(self: RunArtifactManager, payload: dict[str, object]) -> None:
        nonlocal remaining_failures
        if remaining_failures:
            remaining_failures -= 1
            raise ArtifactContractError(
                "synthetic manifest write failure",
                code="test.manifest_write_failed",
            )
        original(self, payload)

    monkeypatch.setattr(RunArtifactManager, "_write_manifest", fail_once)


def _adapter_receipt() -> DoraAdapterSetupReceipt:
    return DoraAdapterSetupReceipt(
        mode="initialize_new",
        adapter_type="dora",
        adapter_name="default",
        adapter_path=None,
        base_model_path=Path("model_cache/qwen-base"),
        target_discovery=DoraTargetDiscoveryReceipt(
            target_policy="all_linear",
            target_towers=("language",),
            matched_modules=("model.language_model.q_proj",),
            counts_by_tower={"language": 1},
            lm_head_seen=True,
            lm_head_excluded=True,
        ),
        peft_config={"use_dora": True, "target_modules": ["q_proj"]},
        trainable_names=("base_model.model.language_model.q_proj.lora_A.default.weight",),
        trainable_counts={"lora_A": 1, "lora_B": 1, "lora_magnitude_vector": 1},
        package_versions={"peft": "test", "torch": "test"},
    )


def _special_token_result() -> SpecialTokenEmbeddingInstallResult:
    receipt = SpecialTokenEmbeddingInstallReceipt(
        semantics=SPECIAL_TOKEN_EMBEDDING_SEMANTICS,
        tensor_key=DEFAULT_EMBED_DELTA_TENSOR_KEY,
        tie_word_embeddings=True,
        token_selection=SpecialTokenSelection(
            token_strings=("<|object_ref_start|>", "<|coord_0|>"),
            token_ids=(2, 3),
        ),
        delta_shape=(2, 4),
        delta_dtype="float32",
        delta_parameter_names=("embed_tokens.shared_embed_delta",),
        base_embedding_parameter_name="model.embed_tokens.weight",
        base_lm_head_parameter_name="lm_head.weight",
    )
    return SpecialTokenEmbeddingInstallResult(
        model=nn.Module(),
        input_wrapper=nn.Identity(),
        output_wrapper=nn.Identity(),
        shared_embed_delta=nn.Parameter(
            torch.tensor(
                [[0.25, -0.25, 0.5, -0.5], [0.75, 0.5, -0.75, -0.5]],
                dtype=torch.float32,
            )
        ),
        receipt=receipt,
    )


def _trainable_surface() -> TrainableSurfaceReceipt:
    return TrainableSurfaceReceipt(
        phase="before_first_backward",
        frozen_towers=("language", "vision", "aligner"),
        trainable_towers=("adapter.language", "token_embeddings"),
        adapter_targets={
            "enabled": True,
            "matched_modules": ["model.language_model.q_proj"],
        },
        selected_embedding_tokens={
            "enabled": True,
            "selected_token_count": 2,
            "token_ids": [2, 3],
        },
        parameter_counts={"trainable_parameter_count": 2},
        optimizer_groups=(
            {"group_name": "adapter.language", "parameter_names": ["adapter.weight"]},
            {
                "group_name": "token_embeddings",
                "parameter_names": ["embed_tokens.shared_embed_delta"],
            },
        ),
        unmatched_trainable_names=(),
        frozen_reason_summaries=(
            FrozenReasonSummary(
                reason="base_towers_frozen_v1",
                parameter_count=1,
                scalar_count=4,
                parameter_names_preview=("model.embed_tokens.weight",),
                context={},
            ),
        ),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
