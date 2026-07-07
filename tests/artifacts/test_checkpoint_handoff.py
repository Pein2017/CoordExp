from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from src.adapters.dora import DoraAdapterSetupReceipt, DoraTargetDiscoveryReceipt
from src.artifacts import CheckpointWriter, RunArtifactManager
from src.artifacts.checkpoint_handoff import validate_checkpoint_handoff
from src.config.models import RunDirectory
from src.optim.trainable_surface import FrozenReasonSummary, TrainableSurfaceReceipt
from src.qwen.special_token_embeddings import (
    DEFAULT_EMBED_DELTA_TENSOR_KEY,
    SPECIAL_TOKEN_EMBEDDING_SEMANTICS,
    SpecialTokenEmbeddingInstallReceipt,
    SpecialTokenEmbeddingInstallResult,
    SpecialTokenSelection,
)


def test_handoff_validator_passes_complete_handoff(tmp_path: Path) -> None:
    manager, result = _write_checkpoint(tmp_path)

    verdict = validate_checkpoint_handoff(
        run_root=manager.run_dir,
        checkpoint_ref=result.handoff_path,
        gate="handoff",
    )

    assert verdict["status"] == "pass"
    assert verdict["gate"] == "handoff"
    assert verdict["checkpoint_id"] == "step-5"
    assert verdict["adapter_identity"]["fingerprint"]
    assert verdict["special_token_embedding_identity"]["fingerprint"]
    assert verdict["missing"] == []
    assert verdict["mismatches"] == []


def test_handoff_validator_holds_manifest_missing_required_identity_fields(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run-a"
    checkpoint_dir = run_dir / "checkpoints" / "step-5"
    checkpoint_dir.mkdir(parents=True)
    handoff_path = checkpoint_dir / "checkpoint_handoff.json"
    handoff_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "checkpoint_id": "step-5",
                "planned_step_id": 5,
                "adapter": {"enabled": False},
                "special_token_embeddings": {"enabled": False},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    verdict = validate_checkpoint_handoff(
        run_root=run_dir,
        checkpoint_ref=handoff_path,
        gate="handoff",
    )

    assert verdict["status"] == "hold"
    assert {
        "checkpoint_path",
        "checkpoint_metadata_path",
        "base_model.path",
        "base_model.base_config_sha256",
        "base_model.tokenizer_sha256",
        "processor_identity",
        "template_identity",
        "resolved_config_fingerprint",
        "intended_inference_config_family",
    }.issubset(set(verdict["missing"]))


def test_handoff_validator_holds_legacy_checkpoint_without_handoff(
    tmp_path: Path,
) -> None:
    manager, result = _write_checkpoint(tmp_path)
    result.handoff_path.unlink()

    verdict = validate_checkpoint_handoff(
        run_root=manager.run_dir,
        checkpoint_ref=result.metadata_path,
        gate="handoff",
    )

    assert verdict["status"] == "hold"
    assert verdict["composition_mode"] == "legacy_manual"
    assert "checkpoint_handoff" in verdict["missing"]


@pytest.mark.parametrize("ref_kind", ["handoff", "checkpoint_dir"])
def test_handoff_validator_holds_when_metadata_backlink_is_missing(
    tmp_path: Path,
    ref_kind: str,
) -> None:
    manager, result = _write_checkpoint(tmp_path)
    result.metadata_path.unlink()
    checkpoint_ref = result.handoff_path if ref_kind == "handoff" else result.checkpoint_dir

    verdict = validate_checkpoint_handoff(
        run_root=manager.run_dir,
        checkpoint_ref=checkpoint_ref,
        gate="handoff",
    )

    assert verdict["status"] == "hold"
    assert "checkpoint_metadata" in verdict["missing"]


def test_handoff_validator_holds_when_metadata_backlink_points_elsewhere(
    tmp_path: Path,
) -> None:
    manager, result = _write_checkpoint(tmp_path)
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    metadata["checkpoint_handoff"] = "checkpoints/step-5/other_handoff.json"
    result.metadata_path.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")

    verdict = validate_checkpoint_handoff(
        run_root=manager.run_dir,
        checkpoint_ref=result.handoff_path,
        gate="handoff",
    )

    assert verdict["status"] == "hold"
    assert "checkpoint_metadata.checkpoint_handoff" in verdict["mismatches"]


def test_handoff_validator_holds_alias_that_disagrees_with_handoff(
    tmp_path: Path,
) -> None:
    manager, result = _write_checkpoint(tmp_path)
    alias_payload = json.loads(result.final_alias_path.read_text(encoding="utf-8"))
    alias_payload["checkpoint_id"] = "step-6"
    result.final_alias_path.write_text(
        json.dumps(alias_payload, sort_keys=True),
        encoding="utf-8",
    )

    verdict = validate_checkpoint_handoff(
        run_root=manager.run_dir,
        checkpoint_ref=result.final_alias_path,
        gate="handoff",
    )

    assert verdict["status"] == "hold"
    assert "alias.checkpoint_id" in verdict["mismatches"]


def test_handoff_validator_holds_null_adapter_identity(tmp_path: Path) -> None:
    manager, result = _write_checkpoint(tmp_path)
    handoff = json.loads(result.handoff_path.read_text(encoding="utf-8"))
    handoff["adapter"]["identity"] = None
    result.handoff_path.write_text(json.dumps(handoff, sort_keys=True), encoding="utf-8")

    verdict = validate_checkpoint_handoff(
        run_root=manager.run_dir,
        checkpoint_ref=result.handoff_path,
        gate="handoff",
    )

    assert verdict["status"] == "hold"
    assert "adapter.identity" in verdict["missing"]


def test_handoff_validator_holds_adapter_identity_fingerprint_mismatch(
    tmp_path: Path,
) -> None:
    manager, result = _write_checkpoint(tmp_path)
    handoff = json.loads(result.handoff_path.read_text(encoding="utf-8"))
    handoff["adapter"]["identity"]["fingerprint"] = "wrong-fingerprint"
    result.handoff_path.write_text(json.dumps(handoff, sort_keys=True), encoding="utf-8")

    verdict = validate_checkpoint_handoff(
        run_root=manager.run_dir,
        checkpoint_ref=result.handoff_path,
        gate="handoff",
    )

    assert verdict["status"] == "hold"
    assert "adapter.identity.fingerprint" in verdict["mismatches"]


def test_handoff_validator_holds_adapter_tensor_hash_mismatch(
    tmp_path: Path,
) -> None:
    manager, result = _write_checkpoint(tmp_path)
    (result.checkpoint_dir / "adapter" / "adapter_model.safetensors").write_bytes(
        b"mutated-adapter"
    )

    verdict = validate_checkpoint_handoff(
        run_root=manager.run_dir,
        checkpoint_ref=result.handoff_path,
        gate="handoff",
    )

    assert verdict["status"] == "hold"
    assert "adapter.adapter_model_sha256" in verdict["mismatches"]


def test_handoff_validator_holds_special_token_tensor_hash_mismatch(
    tmp_path: Path,
) -> None:
    manager, result = _write_checkpoint(tmp_path)
    (
        result.checkpoint_dir
        / "special_token_embeddings"
        / "special_token_embeddings.safetensors"
    ).write_bytes(b"mutated-token-delta")

    verdict = validate_checkpoint_handoff(
        run_root=manager.run_dir,
        checkpoint_ref=result.handoff_path,
        gate="handoff",
    )

    assert verdict["status"] == "hold"
    assert "special_token_embeddings.tensor_sha256" in verdict["mismatches"]


def test_handoff_validator_holds_malformed_handoff_json(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run-a"
    checkpoint_dir = run_dir / "checkpoints" / "step-5"
    checkpoint_dir.mkdir(parents=True)
    handoff_path = checkpoint_dir / "checkpoint_handoff.json"
    handoff_path.write_text("{", encoding="utf-8")

    verdict = validate_checkpoint_handoff(
        run_root=run_dir,
        checkpoint_ref=handoff_path,
        gate="handoff",
    )

    assert verdict["status"] == "hold"
    assert "checkpoint_handoff.invalid_json" in verdict["mismatches"]


def test_handoff_validator_eval_gate_requires_accepted_eval_roots(
    tmp_path: Path,
) -> None:
    manager, result = _write_checkpoint(tmp_path)

    handoff_verdict = validate_checkpoint_handoff(
        run_root=manager.run_dir,
        checkpoint_ref=result.final_alias_path,
        gate="handoff",
    )
    eval_verdict = validate_checkpoint_handoff(
        run_root=manager.run_dir,
        checkpoint_ref=result.final_alias_path,
        gate="eval",
    )

    assert handoff_verdict["status"] == "pass"
    assert eval_verdict["status"] == "hold"
    assert "accepted_eval_artifact_roots" in eval_verdict["missing"]


def test_handoff_validator_production_gate_is_not_implemented(
    tmp_path: Path,
) -> None:
    manager, result = _write_checkpoint(tmp_path)
    handoff = json.loads(result.handoff_path.read_text(encoding="utf-8"))
    handoff["accepted_eval_artifact_roots"] = ["eval/accepted-val200"]
    result.handoff_path.write_text(json.dumps(handoff, sort_keys=True), encoding="utf-8")

    verdict = validate_checkpoint_handoff(
        run_root=manager.run_dir,
        checkpoint_ref=result.handoff_path,
        gate="production",
    )

    assert verdict["status"] == "hold"
    assert "production_gate_unimplemented" in verdict["missing"]


class FakePeftModel(nn.Module):
    def save_pretrained(self, output_dir: str | Path) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "adapter_config.json").write_text(
            json.dumps({"peft_type": "LORA", "use_dora": True}) + "\n",
            encoding="utf-8",
        )
        (path / "adapter_model.safetensors").write_bytes(b"adapter")


def _write_checkpoint(tmp_path: Path):
    manager = RunArtifactManager.initialize(
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
    result = CheckpointWriter(manager).write_checkpoint(
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
        template_identity={
            "object_field_order": "desc_first",
            "object_ordering": "geo_sorted",
            "assistant_format": "object_box_closed",
        },
    )
    return manager, result


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
