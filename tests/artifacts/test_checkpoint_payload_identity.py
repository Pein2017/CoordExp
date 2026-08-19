from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from src.artifacts import training_state
from src.artifacts.checkpoint_payload import (
    admit_inference_checkpoint_payload_identity,
    build_inference_checkpoint_payload_identity,
    load_inference_checkpoint_payload_manifest,
    write_inference_checkpoint_payload_manifest,
)
from src.artifacts.training_state import (
    RankTrainingStatePayload,
    TrainingStateExpectations,
    TrainingStatePublication,
    admit_training_state,
    publish_training_state,
)
from src.adapters.dora import inspect_dora_adapter_payload
from src.artifacts.run_writer import RunWriter
from src.common.errors import ArtifactContractError
from src.qwen.special_token_embeddings import (
    DEFAULT_EMBED_DELTA_TENSOR_KEY,
    inspect_special_token_embedding_delta_payload,
)


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


def test_payload_reading_ignores_exact_sibling_and_extra_historical_metadata(
    tmp_path: Path,
) -> None:
    """`coordexp-swift-training-artifacts` -> Scenario: Existing checkpoint is used;
    `coordexp-swift-training-resume` -> Scenario: Inference reads a checkpoint with
    exact state."""

    checkpoint = _write_checkpoint_payload(tmp_path / "checkpoint")
    write_inference_checkpoint_payload_manifest(checkpoint)
    manifest_before = load_inference_checkpoint_payload_manifest(checkpoint)
    identity = build_inference_checkpoint_payload_identity(checkpoint)

    training_state = checkpoint / "training_state"
    (training_state / "rank-00000").mkdir(parents=True)
    (training_state / "training_state_manifest.json").write_text(
        json.dumps({"schema_version": 2, "commit_status": "committed"}, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    (training_state / "rank-00000" / "model.pt").write_bytes(b"exact-state-bytes")
    (checkpoint / "checkpoint.json").write_text("{}\n", encoding="utf-8")
    (checkpoint / "checkpoint_handoff.json").write_text("{}\n", encoding="utf-8")
    (checkpoint / "setup_receipt.json").write_text("{}\n", encoding="utf-8")
    (checkpoint / "optimizer.pt").write_bytes(b"historical-resume-like-file")

    assert load_inference_checkpoint_payload_manifest(checkpoint) == manifest_before
    assert build_inference_checkpoint_payload_identity(checkpoint) == identity
    assert admit_inference_checkpoint_payload_identity(checkpoint, identity) == identity

    inventoried = {
        item["relative_path"]
        for component in ("adapter", "special_token_embedding_delta")
        for item in manifest_before[component]["files"]
    }
    assert inventoried == {
        "adapter_config.json",
        "adapter_model.safetensors",
        "special_token_embeddings.json",
        "special_token_embeddings.safetensors",
    }
    assert "training_state" not in json.dumps(manifest_before)
    assert (training_state / "rank-00000" / "model.pt").is_file()


def test_inference_reader_ignores_real_committed_training_state_without_opening_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkpoint with a REAL committed ``training_state/`` (built via the real
    ``publish_training_state``) must produce an inference-payload manifest and
    identity byte/dict-identical to a sibling-free control, and the inference
    reader must never open a ``training_state`` file while doing so. Exact
    admission of that same directory's ``training_state/`` must independently
    succeed, outside the interception window."""

    control = _write_checkpoint_payload(tmp_path / "control")
    write_inference_checkpoint_payload_manifest(control)
    control_manifest = load_inference_checkpoint_payload_manifest(control)
    control_identity = build_inference_checkpoint_payload_identity(control)

    paired = _write_checkpoint_payload(tmp_path / "paired")
    write_inference_checkpoint_payload_manifest(paired)
    publication = _real_training_state_publication()
    publish_training_state(paired, publication)
    assert (paired / "training_state" / "manifest.json").is_file()

    opened_paths: list[str] = []
    original_open = Path.open

    def _tracking_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        opened_paths.append(str(self))
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", _tracking_open)
    try:
        # (a) load/build/admit identity for the inference reader only.
        paired_manifest = load_inference_checkpoint_payload_manifest(paired)
        paired_identity = build_inference_checkpoint_payload_identity(paired)
        admitted_identity = admit_inference_checkpoint_payload_identity(
            paired, control_identity
        )
    finally:
        monkeypatch.undo()

    assert paired_manifest == control_manifest
    assert paired_identity == control_identity
    assert admitted_identity == control_identity

    # (c) file-access proof: real payload files were opened (non-vacuous), and no
    # opened path ever touched the training_state sibling.
    assert opened_paths
    assert any(path.endswith("inference_payload_manifest.json") for path in opened_paths)
    assert any(path.endswith("adapter_config.json") for path in opened_paths)
    assert any(path.endswith("adapter_model.safetensors") for path in opened_paths)
    assert any(
        path.endswith("special_token_embeddings.json") for path in opened_paths
    )
    assert any(
        path.endswith("special_token_embeddings.safetensors") for path in opened_paths
    )
    assert not any("training_state" in path for path in opened_paths)

    # (b) exact admission of the same directory's training_state/, outside the
    # interception window above.
    admitted_state = admit_training_state(
        paired,
        _real_training_state_expectations(publication),
        current_rank=0,
    )
    assert admitted_state.manifest.parent_run_id == publication.parent_run_id
    assert admitted_state.manifest.parent_segment_id == publication.parent_segment_id
    assert admitted_state.manifest.checkpoint_step == publication.checkpoint_step


def _real_training_state_identities() -> dict[str, str]:
    return {
        name: f"{index + 1:x}" * 64
        for index, name in enumerate(training_state.REQUIRED_IDENTITY_KINDS)
    }


def _real_training_state_resolved_config() -> dict[str, Any]:
    return {
        "config": {
            "adapter": {"rank": 16},
            "checkpoint": {"save_final": True, "steps": [17]},
            "data": {"train": {"path": "/data/train.jsonl"}},
            "eval": {"forward": {"steps": [17]}},
            "losses": {"coordinate": {"weight": 1.0}},
            "model": {"dtype": "float32"},
            "optimizer": {
                "lr": 0.0002,
                "scheduler": {"name": "cosine", "warmup_ratio": 0.1},
            },
            "packing": {"policy": "source_order_next_fit"},
            "resume": {"checkpoint_dir": None, "mode": "disabled"},
            "run": {
                "artifact_root": "/outputs/parent",
                "name": "parent",
                "output_dir": None,
            },
            "runtime": {"world_size": 1},
            "training": {
                "forward_input_provider_mode": "synchronous",
                "precision": "bf16",
                "seed": 17,
            },
        },
        "resolution": {
            "entry_config_path": "/configs/parent.yaml",
            "fingerprint": "parent-fingerprint",
            "loader_version": "coordexp-swift-config-v1",
            "path_origins": {},
            "schema_version": 1,
            "sources": [{"path": "/configs/parent.yaml", "sha256": "a" * 64}],
        },
    }


def _real_training_state_rank_payload(rank: int) -> RankTrainingStatePayload:
    torch.manual_seed(41)
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    values = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    loss = model(values).square().sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return training_state.serialize_rank_training_state(
        rank=rank,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        python_rng_state=random.getstate(),
        numpy_rng_state=np.random.get_state(),
        torch_cpu_rng_state=torch.get_rng_state(),
        torch_cuda_rng_states=(torch.arange(32, dtype=torch.uint8),),
        cursor={
            "data": {"epoch": 2, "ordinal": 20 + rank},
            "pack": {"ordinal": 10 + rank, "pending": []},
        },
        next_rank_local_micro_step=30 + rank,
    )


def _real_training_state_publication() -> TrainingStatePublication:
    resolved_config = _real_training_state_resolved_config()
    resume_compatibility = training_state.build_resume_compatibility_projection(
        resolved_config
    )
    identities = _real_training_state_identities()
    identities["resolved_config"] = training_state._sha256(
        training_state._canonical_json_bytes(resolved_config) + b"\n"
    )
    identities["resume_compatibility"] = training_state._sha256(
        training_state._canonical_json_bytes(resume_compatibility) + b"\n"
    )
    return TrainingStatePublication(
        parent_run_id="run-parent",
        parent_segment_id="segment-1",
        checkpoint_step=17,
        continuation_index=1,
        world_size=1,
        identities=identities,
        scheduler_applicable=True,
        scaler_applicable=False,
        rank_payloads=[_real_training_state_rank_payload(0)],
        resolved_config=resolved_config,
        resume_compatibility=resume_compatibility,
    )


def _real_training_state_expectations(
    publication: TrainingStatePublication,
) -> TrainingStateExpectations:
    decoded = training_state._decode_rank_payload(publication.rank_payloads[0])
    return TrainingStateExpectations(
        checkpoint_step=publication.checkpoint_step,
        world_size=publication.world_size,
        identities=publication.identities,
        scheduler_applicable=publication.scheduler_applicable,
        scaler_applicable=publication.scaler_applicable,
        resolved_config=publication.resolved_config,
        resume_compatibility=publication.resume_compatibility,
        runtime_state=training_state.RuntimeStateExpectations(
            structure=decoded.structure,
            signature=decoded.signature,
        ),
    )


def test_historical_payload_without_a_current_manifest_stays_inference_loadable(
    tmp_path: Path,
) -> None:
    """`coordexp-swift-training-artifacts` -> Scenario: Existing checkpoint is used;
    `coordexp-swift-training-resume` -> Scenario: Historical artifacts contain extra
    metadata."""

    checkpoint = _write_checkpoint_payload(tmp_path / "historical")
    base_model = (tmp_path / "base-model").resolve()
    (checkpoint / "checkpoint.json").write_text("{}\n", encoding="utf-8")
    (checkpoint / "checkpoint_handoff.json").write_text("{}\n", encoding="utf-8")
    (checkpoint / "training_state").mkdir()
    (checkpoint / "training_state" / "legacy_resume.pt").write_bytes(b"historical")

    adapter_identity = inspect_dora_adapter_payload(
        checkpoint / "adapter",
        expected_base_model_path=base_model,
    )
    delta_identity = inspect_special_token_embedding_delta_payload(
        checkpoint / "special_token_embeddings",
        expected_base_model_path=base_model,
        expected_base_config_sha256="b" * 64,
        expected_tokenizer_sha256="t" * 64,
    )

    assert adapter_identity["kind"] == "dora_adapter"
    assert adapter_identity["root"] == str((checkpoint / "adapter").resolve())
    assert sorted(item["relative_path"] for item in adapter_identity["files"]) == [
        "adapter_config.json",
        "adapter_model.safetensors",
    ]
    assert delta_identity["root"] == str(
        (checkpoint / "special_token_embeddings").resolve()
    )
    assert sorted(item["relative_path"] for item in delta_identity["files"]) == [
        "special_token_embeddings.json",
        "special_token_embeddings.safetensors",
    ]
    assert not (checkpoint / "inference_payload_manifest.json").exists()
    with pytest.raises(ArtifactContractError) as exc_info:
        load_inference_checkpoint_payload_manifest(checkpoint)
    assert exc_info.value.code == "checkpoint.inference_payload_manifest_invalid"
    assert (checkpoint / "training_state" / "legacy_resume.pt").is_file()


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
