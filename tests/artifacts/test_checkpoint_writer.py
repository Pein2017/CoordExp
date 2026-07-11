from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from src.artifacts.checkpoints import CheckpointWriter
from src.artifacts.run_writer import RunWriter
from src.common.errors import ArtifactContractError
from src.qwen.special_token_embeddings import (
    DEFAULT_EMBED_DELTA_TENSOR_KEY,
    SPECIAL_TOKEN_EMBEDDING_SEMANTICS,
    SpecialTokenEmbeddingInstallReceipt,
    SpecialTokenEmbeddingInstallResult,
    SpecialTokenSelection,
)


VALID_KEYS = {
    "base_model.model.q_proj.lora_A.default.weight": torch.ones(2, 3),
    "base_model.model.q_proj.lora_B.default.weight": torch.ones(3, 2),
    "base_model.model.q_proj.lora_magnitude_vector.default.weight": torch.ones(3),
}


class FakeAccelerator:
    def __init__(self, *, main: bool = True, shared: dict[str, Any] | None = None) -> None:
        self.is_main_process = main
        self.shared = shared if shared is not None else {}
        self.barriers = 0

    def wait_for_everyone(self) -> None:
        self.barriers += 1

    def unwrap_model(self, model: Any) -> Any:
        return model

    def broadcast_object_list(self, values: list[Any], *, from_process: int) -> None:
        assert from_process == 0
        if self.is_main_process:
            self.shared["status"] = values[0]
        else:
            values[0] = self.shared["status"]


class FakePeftModel(nn.Module):
    def __init__(self, tensors: dict[str, torch.Tensor] | None = None, *, fail=False) -> None:
        super().__init__()
        self.tensors = VALID_KEYS if tensors is None else tensors
        self.fail = fail
        self.calls: list[dict[str, Any]] = []

    def save_pretrained(self, output_dir: str | Path, **kwargs: Any) -> None:
        self.calls.append(kwargs)
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "adapter_config.json").write_text(
            json.dumps({"peft_type": "LORA", "use_dora": True}) + "\n"
        )
        if self.fail:
            raise RuntimeError("synthetic rank-zero save failure")
        save_file(self.tensors, str(path / "adapter_model.safetensors"))


def test_adapter_only_atomic_checkpoint_and_safe_peft_arguments(tmp_path: Path) -> None:
    model = FakePeftModel()
    accelerator = FakeAccelerator()
    result = CheckpointWriter(tmp_path).write_checkpoint(
        step=3, accelerator=accelerator, model=model, adapter_name="default"
    )
    assert result.checkpoint_dir == tmp_path / "checkpoints" / "step-3"
    assert (result.checkpoint_dir / "adapter" / "adapter_model.safetensors").is_file()
    assert model.calls == [{
        "safe_serialization": True,
        "selected_adapters": ["default"],
        "save_embedding_layers": False,
    }]
    assert accelerator.barriers == 1
    assert _staging(tmp_path) == []


def test_adapter_plus_compact_selected_token_delta(tmp_path: Path) -> None:
    result = CheckpointWriter(tmp_path).write_checkpoint(
        step=1,
        accelerator=FakeAccelerator(),
        model=FakePeftModel(),
        adapter_name="default",
        special_token_result=_special_token_result(),
        base_model_path="base",
        base_config_sha256="base-sha",
        tokenizer_sha256="tokenizer-sha",
    )
    delta = result.checkpoint_dir / "special_token_embeddings"
    assert sorted(path.name for path in delta.iterdir()) == [
        "special_token_embeddings.json", "special_token_embeddings.safetensors"
    ]


@pytest.mark.parametrize(
    ("tensors", "code"),
    [
        ({"base_model.model.q.lora_A.default.weight": torch.ones(1)},
         "checkpoint.adapter_required_tensors_missing"),
        ({**VALID_KEYS, "base_model.model.embed_tokens.weight": torch.ones(2, 2)},
         "checkpoint.adapter_forbidden_tensors"),
        ({**VALID_KEYS, "base_model.model.lm_head.weight": torch.ones(2, 2)},
         "checkpoint.adapter_forbidden_tensors"),
    ],
)
def test_missing_or_forbidden_adapter_tensors_leave_no_residue(
    tmp_path: Path, tensors: dict[str, torch.Tensor], code: str
) -> None:
    with pytest.raises(ArtifactContractError, match="checkpoint save failed") as exc_info:
        CheckpointWriter(tmp_path).write_checkpoint(
            step=1, accelerator=FakeAccelerator(), model=FakePeftModel(tensors),
            adapter_name="default"
        )
    assert exc_info.value.code == "checkpoint.save_failed"
    assert code in str(exc_info.value)
    assert not (tmp_path / "checkpoints" / "step-1").exists()
    assert _staging(tmp_path) == []


def test_failure_after_staging_begins_is_atomic(tmp_path: Path) -> None:
    with pytest.raises(ArtifactContractError) as exc_info:
        CheckpointWriter(tmp_path).write_checkpoint(
            step=7, accelerator=FakeAccelerator(), model=FakePeftModel(fail=True),
            adapter_name="default"
        )
    assert exc_info.value.code == "checkpoint.save_failed"
    assert "synthetic rank-zero save failure" in str(exc_info.value)
    assert not (tmp_path / "checkpoints" / "step-7").exists()
    assert _staging(tmp_path) == []


def test_rank_zero_failure_is_shared_with_peer_without_post_save_barrier(tmp_path: Path) -> None:
    shared: dict[str, Any] = {}
    with pytest.raises(ArtifactContractError) as main_error:
        CheckpointWriter(tmp_path).write_checkpoint(
            step=2, accelerator=FakeAccelerator(main=True, shared=shared),
            model=FakePeftModel(fail=True), adapter_name="default"
        )
    with pytest.raises(ArtifactContractError) as peer_error:
        CheckpointWriter(tmp_path).write_checkpoint(
            step=2, accelerator=FakeAccelerator(main=False, shared=shared),
            model=object(), adapter_name="default"
        )
    assert main_error.value.code == peer_error.value.code == "checkpoint.save_failed"
    assert str(main_error.value) == str(peer_error.value)


def test_preexisting_step_collision_preserves_checkpoint_and_aliases_for_all_ranks(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoints" / "step-2"
    checkpoint_dir.mkdir(parents=True)
    payload = checkpoint_dir / "existing.bin"
    payload.write_bytes(b"valid-existing-checkpoint")
    final_alias = tmp_path / "checkpoints" / "final.json"
    best_alias = tmp_path / "checkpoints" / "best.json"
    final_alias.write_bytes(b'{"step":1,"checkpoint_path":"checkpoints/step-1"}\n')
    best_alias.write_bytes(
        b'{"step":1,"checkpoint_path":"checkpoints/step-1","value":0.5}\n'
    )
    before = {
        "payload": payload.read_bytes(),
        "final": final_alias.read_bytes(),
        "best": best_alias.read_bytes(),
    }
    shared: dict[str, Any] = {}

    with pytest.raises(ArtifactContractError) as main_error:
        CheckpointWriter(tmp_path).write_checkpoint(
            step=2,
            accelerator=FakeAccelerator(main=True, shared=shared),
            model=FakePeftModel(),
            adapter_name="default",
        )
    with pytest.raises(ArtifactContractError) as peer_error:
        CheckpointWriter(tmp_path).write_checkpoint(
            step=2,
            accelerator=FakeAccelerator(main=False, shared=shared),
            model=object(),
            adapter_name="default",
        )

    assert main_error.value.code == peer_error.value.code == "checkpoint.save_failed"
    assert str(main_error.value) == str(peer_error.value)
    assert "checkpoint.step_exists" in str(main_error.value)
    assert payload.read_bytes() == before["payload"]
    assert final_alias.read_bytes() == before["final"]
    assert best_alias.read_bytes() == before["best"]
    assert sorted(path.name for path in checkpoint_dir.iterdir()) == ["existing.bin"]
    assert _staging(tmp_path) == []


def test_alias_update_and_restore_failure_still_broadcasts_safe_shared_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.artifacts import checkpoints as checkpoint_module

    writer = _run_writer(tmp_path)
    original_write_final = type(writer).write_final

    def update_final_then_fail(self: RunWriter, *, step: int) -> Path:
        original_write_final(self, step=step)
        raise RuntimeError("synthetic alias update failure")

    monkeypatch.setattr(RunWriter, "write_final", update_final_then_fail)
    monkeypatch.setattr(
        checkpoint_module,
        "_restore_aliases",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic alias restoration failure")
        ),
    )
    shared: dict[str, Any] = {}
    main_accelerator = FakeAccelerator(main=True, shared=shared)
    peer_accelerator = FakeAccelerator(main=False, shared=shared)

    with pytest.raises(ArtifactContractError) as main_error:
        CheckpointWriter(tmp_path).write_checkpoint(
            step=2, accelerator=main_accelerator, model=FakePeftModel(),
            adapter_name="default", run_writer=writer, is_final=True,
        )
    with pytest.raises(ArtifactContractError) as peer_error:
        CheckpointWriter(tmp_path).write_checkpoint(
            step=2, accelerator=peer_accelerator, model=object(),
            adapter_name="default",
        )

    assert "status" in shared
    assert main_error.value.code == peer_error.value.code == "checkpoint.save_failed"
    assert str(main_error.value) == str(peer_error.value)
    assert "synthetic alias update failure" in str(main_error.value)
    assert "synthetic alias restoration failure" in str(main_error.value)
    alias = json.loads((tmp_path / "checkpoints/final.json").read_text())
    selected = tmp_path / alias["checkpoint_path"]
    assert selected == tmp_path / "checkpoints/step-2"
    assert (selected / "adapter/adapter_model.safetensors").is_file()
    assert _staging(tmp_path) == []


def test_checkpoint_cleanup_failure_still_broadcasts_identical_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.artifacts import checkpoints as checkpoint_module

    writer = _run_writer(tmp_path)
    monkeypatch.setattr(
        RunWriter,
        "write_final",
        lambda self, **kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic alias failure")
        ),
    )
    original_rmtree = checkpoint_module.shutil.rmtree

    def fail_checkpoint_cleanup(path: str | Path, *args: Any, **kwargs: Any) -> None:
        if Path(path).name == "step-2":
            raise RuntimeError("synthetic checkpoint cleanup failure")
        original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(checkpoint_module.shutil, "rmtree", fail_checkpoint_cleanup)
    shared: dict[str, Any] = {}

    with pytest.raises(ArtifactContractError) as main_error:
        CheckpointWriter(tmp_path).write_checkpoint(
            step=2, accelerator=FakeAccelerator(main=True, shared=shared),
            model=FakePeftModel(), adapter_name="default", run_writer=writer,
            is_final=True,
        )
    with pytest.raises(ArtifactContractError) as peer_error:
        CheckpointWriter(tmp_path).write_checkpoint(
            step=2, accelerator=FakeAccelerator(main=False, shared=shared),
            model=object(), adapter_name="default",
        )

    assert "status" in shared
    assert main_error.value.code == peer_error.value.code == "checkpoint.save_failed"
    assert str(main_error.value) == str(peer_error.value)
    assert "synthetic checkpoint cleanup failure" in str(main_error.value)
    assert not (tmp_path / "checkpoints/final.json").exists()
    assert not (tmp_path / "checkpoints/best.json").exists()
    # Cleanup failed, but the residue is a complete payload and no alias selects it.
    assert (tmp_path / "checkpoints/step-2/adapter/adapter_model.safetensors").is_file()
    assert _staging(tmp_path) == []


def test_final_and_best_are_rank_zero_owned_and_safe(tmp_path: Path) -> None:
    writer = _run_writer(tmp_path)
    shared: dict[str, Any] = {}
    main = CheckpointWriter(tmp_path).write_checkpoint(
        step=1, accelerator=FakeAccelerator(main=True, shared=shared),
        model=FakePeftModel(), adapter_name="default", run_writer=writer, is_final=True,
        best_candidate={"completed": True, "selector": "eval/acc:max", "value": .5,
                        "optimizer_update_status": "applied", "finite_status": "finite"},
    )
    peer = CheckpointWriter(tmp_path).write_checkpoint(
        step=1, accelerator=FakeAccelerator(main=False, shared=shared),
        model=object(), adapter_name="default", run_writer=None, is_final=True,
    )
    assert main.final_updated and main.best_updated
    assert not peer.final_updated and not peer.best_updated
    assert json.loads((tmp_path / "checkpoints/final.json").read_text())["step"] == 1
    assert json.loads((tmp_path / "checkpoints/best.json").read_text())["step"] == 1
    assert not any(path.name.startswith("checkpoint-") for path in (tmp_path / "checkpoints").iterdir())
    assert not list(tmp_path.rglob("checkpoint.json"))
    assert not list(tmp_path.rglob("checkpoint_handoff.json"))


@pytest.mark.parametrize("candidate", [
    {"completed": False, "value": .9, "optimizer_update_status": "applied", "finite_status": "finite"},
    {"completed": True, "value": .9, "optimizer_update_status": "skipped_non_finite", "finite_status": "finite"},
    {"completed": True, "value": .9, "optimizer_update_status": "applied", "finite_status": "non_finite"},
    {"completed": True, "value": float("nan"), "optimizer_update_status": "applied", "finite_status": "finite"},
])
def test_unsafe_or_incomplete_candidate_never_advances_best(tmp_path: Path, candidate: dict[str, Any]) -> None:
    writer = _run_writer(tmp_path)
    result = CheckpointWriter(tmp_path).write_checkpoint(
        step=1, accelerator=FakeAccelerator(), model=FakePeftModel(),
        adapter_name="default", run_writer=writer, best_candidate=candidate,
    )
    assert not result.best_updated
    assert not (tmp_path / "checkpoints/best.json").exists()


def _run_writer(run_dir: Path) -> RunWriter:
    return RunWriter.initialize(
        run_dir=run_dir, run_id="run", run_name="run", artifact_root=run_dir.parent,
        collision_outcome="created", created_at="2026-07-11T00:00:00Z",
        config_fingerprint="config", resolved_config={}, world_size=2,
        resolved_max_steps=2,
    )


def _staging(run_dir: Path) -> list[Path]:
    root = run_dir / "checkpoints"
    return [] if not root.exists() else list(root.glob(".step-*.tmp"))


def _special_token_result() -> SpecialTokenEmbeddingInstallResult:
    receipt = SpecialTokenEmbeddingInstallReceipt(
        semantics=SPECIAL_TOKEN_EMBEDDING_SEMANTICS,
        tensor_key=DEFAULT_EMBED_DELTA_TENSOR_KEY,
        tie_word_embeddings=True,
        token_selection=SpecialTokenSelection(token_strings=("<x>",), token_ids=(2,)),
        delta_shape=(1, 2), delta_dtype="float32",
        delta_parameter_names=("embed.shared_embed_delta",),
        base_embedding_parameter_name="embed.weight", base_lm_head_parameter_name="lm_head.weight",
    )
    return SpecialTokenEmbeddingInstallResult(
        model=nn.Module(), input_wrapper=nn.Identity(), output_wrapper=nn.Identity(),
        shared_embed_delta=nn.Parameter(torch.ones(1, 2)), receipt=receipt,
    )
