from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from src.artifacts.checkpoint_payload import (
    admit_inference_checkpoint_payload_identity,
    build_inference_checkpoint_payload_identity,
    load_inference_checkpoint_payload_manifest,
    write_inference_checkpoint_payload_manifest,
)
from src.artifacts.run_writer import RunWriter
from src.common.errors import ArtifactContractError
from src.qwen.special_token_embeddings import DEFAULT_EMBED_DELTA_TENSOR_KEY


def test_inference_checkpoint_payload_identity_is_complete_and_path_independent(
    tmp_path: Path,
) -> None:
    checkpoint = _write_checkpoint_payload(tmp_path / "source")
    write_inference_checkpoint_payload_manifest(
        checkpoint,
        expected_base_model_path=tmp_path / "base-model",
        expected_base_config_sha256="b" * 64,
        expected_tokenizer_sha256="t" * 64,
    )

    identity = build_inference_checkpoint_payload_identity(checkpoint)
    relocated = tmp_path / "relocated"
    shutil.copytree(checkpoint, relocated)

    assert set(identity) == {
        "schema",
        "schema_version",
        "manifest_relative_path",
        "manifest_file_sha256",
        "aggregate_digest",
    }
    assert (
        identity["schema"] == "coordexp-swift-inference-checkpoint-payload-publication"
    )
    assert identity["schema_version"] == 2
    assert identity["manifest_relative_path"] == "inference_payload_manifest.json"
    manifest = load_inference_checkpoint_payload_manifest(checkpoint)
    assert manifest["schema_version"] == 1
    assert "root" not in manifest["adapter"]["inspector_identity"]
    assert "tensor_manifest" in manifest["adapter"]["inspector_identity"]
    assert "root" not in manifest["special_token_embedding_delta"]["inspector_identity"]
    assert build_inference_checkpoint_payload_identity(relocated) == identity
    assert admit_inference_checkpoint_payload_identity(relocated, identity) == identity


@pytest.mark.parametrize(
    "mutation",
    ["adapter_byte", "embedding_metadata", "embedding_tensor", "added_file"],
)
def test_inference_checkpoint_payload_admission_rejects_committed_mutation(
    tmp_path: Path,
    mutation: str,
) -> None:
    checkpoint = _write_checkpoint_payload(tmp_path / "checkpoint")
    write_inference_checkpoint_payload_manifest(checkpoint)
    identity = build_inference_checkpoint_payload_identity(checkpoint)

    if mutation == "adapter_byte":
        path = checkpoint / "adapter" / "adapter_model.safetensors"
        payload = bytearray(path.read_bytes())
        payload[-1] ^= 1
        path.write_bytes(payload)
    elif mutation == "embedding_metadata":
        path = checkpoint / "special_token_embeddings" / "special_token_embeddings.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata["token_strings"] = ["<|coord_0001|>"]
        path.write_text(json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8")
    elif mutation == "embedding_tensor":
        save_file(
            {DEFAULT_EMBED_DELTA_TENSOR_KEY: torch.full((1, 3), 2.0)},
            str(
                checkpoint
                / "special_token_embeddings"
                / "special_token_embeddings.safetensors"
            ),
        )
    else:
        (checkpoint / "adapter" / "README.md").write_text("added later\n")

    with pytest.raises(ArtifactContractError) as exc_info:
        admit_inference_checkpoint_payload_identity(checkpoint, identity)
    assert exc_info.value.code == "checkpoint.inference_payload_identity_mismatch"


def test_absent_embedding_admission_rejects_dangling_component_symlink(
    tmp_path: Path,
) -> None:
    checkpoint = _write_checkpoint_payload(tmp_path / "checkpoint")
    shutil.rmtree(checkpoint / "special_token_embeddings")
    write_inference_checkpoint_payload_manifest(checkpoint)
    identity = build_inference_checkpoint_payload_identity(checkpoint)
    (checkpoint / "special_token_embeddings").symlink_to(
        checkpoint / "missing-embedding-payload",
        target_is_directory=True,
    )

    with pytest.raises(ArtifactContractError) as exc_info:
        admit_inference_checkpoint_payload_identity(checkpoint, identity)
    assert exc_info.value.code == "checkpoint.inference_payload_identity_mismatch"


def test_manifest_inventories_all_payload_files_and_explicit_absence(
    tmp_path: Path,
) -> None:
    checkpoint = _write_checkpoint_payload(tmp_path / "checkpoint")
    (checkpoint / "adapter" / "README.md").write_text("adapter card\n")
    shutil.rmtree(checkpoint / "special_token_embeddings")

    write_inference_checkpoint_payload_manifest(checkpoint)
    manifest = load_inference_checkpoint_payload_manifest(checkpoint)

    assert [item["relative_path"] for item in manifest["adapter"]["files"]] == [
        "README.md",
        "adapter_config.json",
        "adapter_model.safetensors",
    ]
    assert manifest["special_token_embedding_delta"] == {
        "status": "absent",
        "relative_root": "special_token_embeddings",
        "files": [],
        "inspector_identity": None,
    }


def test_completed_publication_event_atomically_binds_payload_and_progress(
    tmp_path: Path,
) -> None:
    writer = _run_writer(tmp_path)
    checkpoint = _write_checkpoint_payload(writer.checkpoints_dir / "step-3")
    write_inference_checkpoint_payload_manifest(checkpoint)
    payload_identity = build_inference_checkpoint_payload_identity(checkpoint)
    committed_progress = {
        "schema": "coordexp-swift-checkpoint-committed-progress",
        "schema_version": 1,
        "completed_steps": 3,
        "consumed_packs": 9,
        "optimizer_update_status": "applied",
        "finite_status": "finite",
    }

    writer.record_checkpoint_publication_event(
        step=3,
        status="completed",
        started_at="2026-08-11T00:00:00+00:00",
        completed_at="2026-08-11T00:00:01+00:00",
        duration_seconds=1.0,
        is_final=False,
        exact_training_state_enabled=False,
        checkpoint_identity=None,
        inference_payload_identity=payload_identity,
        committed_progress=committed_progress,
        failure_code=None,
    )

    state = writer.read_run()
    event = state["measurement"]["checkpoint_publication_events"][0]
    assert event["schema"] == "coordexp-swift-checkpoint-publication-event"
    assert event["schema_version"] == 2
    assert event["inference_payload_identity"] == payload_identity
    assert event["committed_progress"] == committed_progress
    assert state["completed_steps"] == 3
    assert state["consumed_packs"] == 9
    assert state["checkpoint_event_count"] == 1
    assert state["final_optimizer_update_status"] == "applied"
    assert state["final_finite_status"] == "finite"
    assert state["status"] == "initialized"


def test_completed_publication_rechecks_live_payload_before_event(
    tmp_path: Path,
) -> None:
    writer = _run_writer(tmp_path)
    checkpoint = _write_checkpoint_payload(writer.checkpoints_dir / "step-3")
    write_inference_checkpoint_payload_manifest(checkpoint)
    payload_identity = build_inference_checkpoint_payload_identity(checkpoint)
    adapter_path = checkpoint / "adapter" / "adapter_model.safetensors"
    payload = bytearray(adapter_path.read_bytes())
    payload[-1] ^= 1
    adapter_path.write_bytes(payload)

    with pytest.raises(ArtifactContractError) as exc_info:
        writer.record_checkpoint_publication_event(
            step=3,
            status="completed",
            started_at="2026-08-11T00:00:00+00:00",
            completed_at="2026-08-11T00:00:01+00:00",
            duration_seconds=1.0,
            is_final=False,
            exact_training_state_enabled=False,
            checkpoint_identity=None,
            inference_payload_identity=payload_identity,
            committed_progress={
                "schema": "coordexp-swift-checkpoint-committed-progress",
                "schema_version": 1,
                "completed_steps": 3,
                "consumed_packs": 9,
                "optimizer_update_status": "applied",
                "finite_status": "finite",
            },
            failure_code=None,
        )
    assert exc_info.value.code == "checkpoint.inference_payload_identity_mismatch"
    state = writer.read_run()
    assert state["measurement"]["checkpoint_publication_events"] == []
    assert state["completed_steps"] == 0


def test_failed_publication_has_null_identities_and_does_not_advance_progress(
    tmp_path: Path,
) -> None:
    writer = _run_writer(tmp_path)

    writer.record_checkpoint_publication_event(
        step=3,
        status="failed",
        started_at="2026-08-11T00:00:00+00:00",
        completed_at="2026-08-11T00:00:01+00:00",
        duration_seconds=1.0,
        is_final=False,
        exact_training_state_enabled=False,
        checkpoint_identity=None,
        inference_payload_identity=None,
        committed_progress=None,
        failure_code="checkpoint.synthetic_failure",
    )

    state = writer.read_run()
    event = state["measurement"]["checkpoint_publication_events"][0]
    assert event["inference_payload_identity"] is None
    assert event["committed_progress"] is None
    assert state["completed_steps"] == 0
    assert state["consumed_packs"] == 0
    assert state["checkpoint_event_count"] == 0
    assert state["final_optimizer_update_status"] is None
    assert state["final_finite_status"] is None


def _run_writer(tmp_path: Path) -> RunWriter:
    return RunWriter.initialize(
        run_dir=tmp_path / "run",
        run_id="run",
        run_name="run",
        artifact_root=tmp_path,
        collision_outcome="created",
        created_at="2026-08-11T00:00:00+00:00",
        config_fingerprint="f" * 64,
        resolved_config={},
        world_size=1,
        resolved_max_steps=5,
    )


def _write_checkpoint_payload(checkpoint: Path) -> Path:
    base_model = checkpoint.parent / "base-model"
    base_model.mkdir(parents=True, exist_ok=True)
    adapter = checkpoint / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": str(base_model.resolve()),
                "lora_alpha": 4,
                "peft_type": "LORA",
                "r": 2,
                "target_modules": ["q_proj"],
                "use_dora": True,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    save_file(
        {
            "base_model.model.q_proj.lora_A.default.weight": torch.ones(2, 3),
            "base_model.model.q_proj.lora_B.default.weight": torch.ones(3, 2),
            "base_model.model.q_proj.lora_magnitude_vector.default.weight": (
                torch.ones(3)
            ),
        },
        str(adapter / "adapter_model.safetensors"),
    )
    embeddings = checkpoint / "special_token_embeddings"
    embeddings.mkdir()
    (embeddings / "special_token_embeddings.json").write_text(
        json.dumps(
            {
                "base_config_sha256": "b" * 64,
                "base_model_path": str(base_model.resolve()),
                "semantics": "additive_delta",
                "tensor_dtype": "float32",
                "tensor_key": DEFAULT_EMBED_DELTA_TENSOR_KEY,
                "tensor_shape": [1, 3],
                "tie_word_embeddings": True,
                "token_ids": [7],
                "token_strings": ["<|coord_0000|>"],
                "tokenizer_sha256": "t" * 64,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    save_file(
        {DEFAULT_EMBED_DELTA_TENSOR_KEY: torch.ones(1, 3)},
        str(embeddings / "special_token_embeddings.safetensors"),
    )
    return checkpoint
