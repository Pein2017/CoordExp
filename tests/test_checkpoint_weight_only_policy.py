import os
import json
from pathlib import Path

import pytest
import src.trainers.final_checkpoint as final_checkpoint_mod
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments

from src.callbacks.save_delay_callback import SaveDelayCallback
from src.trainers import with_final_checkpoint
from src.trainers.final_checkpoint import (
    COORDEXP_CHECKPOINT_STATE_NAME,
    load_coordexp_checkpoint_state,
    prepare_restartable_checkpoint_resume,
    validate_restartable_checkpoint,
)


class _TinyDataset(torch.utils.data.Dataset):
    def __init__(self, n: int = 8):
        super().__init__()
        self._n = int(n)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int):
        # Arbitrary small regression/classification-like batch.
        x = torch.tensor([float(idx), 1.0, 2.0, 3.0], dtype=torch.float32)
        y = torch.tensor(int(idx) % 2, dtype=torch.long)
        return {"x": x, "labels": y}


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 2)

    def forward(self, x=None, labels=None):  # type: ignore[override]
        logits = self.proj(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, 2), labels.view(-1))
        return {"loss": loss, "logits": logits}


def _collect_checkpoint_files(checkpoint_dir: Path) -> set[str]:
    out: set[str] = set()
    for root, _dirs, files in os.walk(str(checkpoint_dir)):
        for name in files:
            rel = os.path.relpath(os.path.join(root, name), str(checkpoint_dir))
            out.add(rel)
    return out


def test_hf_trainer_save_only_model_skips_optimizer_state(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=2,
        save_strategy="steps",
        save_steps=1,
        logging_steps=1,
        report_to=[],
        # The policy we care about:
        save_only_model=True,
        save_safetensors=True,
        # Avoid CUDA assumptions in unit tests (transformers>=5 prefers use_cpu).
        use_cpu=True,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=_TinyModel(),
        args=args,
        train_dataset=_TinyDataset(),
    )
    # ms-swift may patch transformers.Trainer callbacks in the runtime environment.
    # We only need to exercise the checkpoint writer, so disable callbacks and
    # call the internal save path directly (no training loop needed).
    try:
        trainer.callback_handler.callbacks = []
    except Exception:
        pass

    trainer.create_optimizer_and_scheduler(num_training_steps=1)
    trainer.state.global_step = 1
    try:
        trainer._save_checkpoint(trainer.model, trial=None)  # type: ignore[misc]
    except TypeError:
        trainer._save_checkpoint(trainer.model, trial=None, metrics=None)  # type: ignore[misc,call-arg]

    ckpt = out_dir / "checkpoint-1"
    assert ckpt.is_dir(), f"Expected checkpoint directory {ckpt} to exist"

    files = _collect_checkpoint_files(ckpt)
    # Model weights must exist in some form.
    assert any(
        name in files for name in {"model.safetensors", "pytorch_model.bin"}
    ), f"Expected model weights in checkpoint; got files={sorted(files)}"

    # Weight-only persistence: no optimizer/scheduler/rng snapshots.
    forbidden = {"optimizer.pt", "scheduler.pt", "rng_state.pth"}
    assert forbidden.isdisjoint(files), f"Found forbidden state files: {sorted(forbidden & files)}"


def test_restartable_checkpoint_writes_repo_sidecar_and_validates(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "out"
    trainer_cls = with_final_checkpoint(Trainer)
    args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=2,
        save_strategy="steps",
        save_steps=1,
        logging_steps=1,
        report_to=[],
        save_only_model=False,
        save_safetensors=True,
        use_cpu=True,
        remove_unused_columns=False,
    )
    setattr(args, "checkpoint_mode", "restartable")
    setattr(args, "max_epochs", None)

    save_delay = SaveDelayCallback(save_delay_steps=2)
    trainer = trainer_cls(
        model=_TinyModel(),
        args=args,
        train_dataset=_TinyDataset(),
        callbacks=[save_delay],
    )

    trainer.create_optimizer_and_scheduler(num_training_steps=1)
    trainer.state.global_step = 1
    save_delay._pending_reset = True
    setattr(trainer, "_coordexp_disabled_diagnostics", {"coord_diag"})

    try:
        trainer._save_checkpoint(trainer.model, trial=None)  # type: ignore[misc]
    except TypeError:
        trainer._save_checkpoint(trainer.model, trial=None, metrics=None)  # type: ignore[misc,call-arg]

    ckpt = out_dir / "checkpoint-1"
    files = _collect_checkpoint_files(ckpt)
    assert COORDEXP_CHECKPOINT_STATE_NAME in files

    payload = validate_restartable_checkpoint(ckpt)
    assert payload["checkpoint_mode"] == "restartable"
    assert payload["global_step"] == 1
    assert payload["disabled_diagnostics"] == ["coord_diag"]
    callback_payload = load_coordexp_checkpoint_state(ckpt)["callback_state"]
    assert any(key.endswith("SaveDelayCallback") for key in callback_payload)


def test_step_save_strategy_records_best_checkpoint_state(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    trainer_cls = with_final_checkpoint(Trainer)
    args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=2,
        save_strategy="steps",
        save_steps=1,
        logging_steps=1,
        report_to=[],
        save_only_model=True,
        save_safetensors=True,
        use_cpu=True,
        remove_unused_columns=False,
    )

    trainer = trainer_cls(
        model=_TinyModel(),
        args=args,
        train_dataset=_TinyDataset(),
    )
    try:
        trainer.callback_handler.callbacks = []
    except Exception:
        pass

    trainer.create_optimizer_and_scheduler(num_training_steps=1)
    trainer.state.global_step = 1
    trainer.state.best_global_step = 1

    try:
        trainer._save_checkpoint(trainer.model, trial=None)  # type: ignore[misc]
    except TypeError:
        trainer._save_checkpoint(trainer.model, trial=None, metrics=None)  # type: ignore[misc,call-arg]

    ckpt = out_dir / "checkpoint-1"
    state = json.loads((ckpt / "trainer_state.json").read_text())
    assert trainer.state.best_model_checkpoint == str(ckpt)
    assert state["best_model_checkpoint"] == str(ckpt)
    assert state["best_global_step"] == 1


def test_bounded_ddp_checkpoint_path_respects_save_total_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "out"
    trainer_cls = with_final_checkpoint(Trainer)
    args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=2,
        save_strategy="steps",
        save_steps=1,
        save_total_limit=1,
        logging_steps=1,
        report_to=[],
        save_only_model=True,
        save_safetensors=True,
        use_cpu=True,
        remove_unused_columns=False,
    )

    trainer = trainer_cls(
        model=_TinyModel(),
        args=args,
        train_dataset=_TinyDataset(),
    )
    try:
        trainer.callback_handler.callbacks = []
    except Exception:
        pass

    trainer.create_optimizer_and_scheduler(num_training_steps=2)

    monkeypatch.setattr(
        type(trainer),
        "_should_use_bounded_ddp_checkpoint_save",
        lambda self, model: True,
    )
    monkeypatch.setattr(
        final_checkpoint_mod,
        "ddp_rank0_coordinated_fail_fast",
        lambda *, fn_rank0_only, **_kwargs: fn_rank0_only(),
    )
    monkeypatch.setattr(
        final_checkpoint_mod,
        "ddp_any_rank_fail_fast",
        lambda *, fn, **_kwargs: fn(),
    )

    trainer.state.global_step = 1
    trainer.state.best_global_step = 1
    trainer._save_checkpoint(trainer.model, trial=None)  # type: ignore[misc]

    trainer.state.global_step = 2
    trainer.state.best_global_step = 2
    trainer._save_checkpoint(trainer.model, trial=None)  # type: ignore[misc]

    ckpt1 = out_dir / "checkpoint-1"
    ckpt2 = out_dir / "checkpoint-2"
    assert not ckpt1.exists()
    assert ckpt2.is_dir()
    state = json.loads((ckpt2 / "trainer_state.json").read_text())
    assert trainer.state.best_model_checkpoint == str(ckpt2)
    assert state["best_model_checkpoint"] == str(ckpt2)


def test_restartable_checkpoint_preflight_rejects_artifact_only_sidecar(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "out"
    trainer_cls = with_final_checkpoint(Trainer)
    args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=2,
        save_strategy="steps",
        save_steps=1,
        logging_steps=1,
        report_to=[],
        save_only_model=False,
        save_safetensors=True,
        use_cpu=True,
        remove_unused_columns=False,
    )
    setattr(args, "checkpoint_mode", "artifact_only")
    setattr(args, "max_epochs", None)

    trainer = trainer_cls(
        model=_TinyModel(),
        args=args,
        train_dataset=_TinyDataset(),
    )
    trainer.create_optimizer_and_scheduler(num_training_steps=1)
    trainer.state.global_step = 1

    try:
        trainer._save_checkpoint(trainer.model, trial=None)  # type: ignore[misc]
    except TypeError:
        trainer._save_checkpoint(trainer.model, trial=None, metrics=None)  # type: ignore[misc,call-arg]

    ckpt = out_dir / "checkpoint-1"
    with pytest.raises(ValueError, match="checkpoint_mode='restartable'"):
        validate_restartable_checkpoint(ckpt)


def test_prepare_restartable_checkpoint_resume_restores_callback_and_health_state(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "out"
    trainer_cls = with_final_checkpoint(Trainer)
    args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=2,
        save_strategy="steps",
        save_steps=1,
        logging_steps=1,
        report_to=[],
        save_only_model=False,
        save_safetensors=True,
        use_cpu=True,
        remove_unused_columns=False,
    )
    setattr(args, "checkpoint_mode", "restartable")
    setattr(args, "max_epochs", None)

    trainer = trainer_cls(
        model=_TinyModel(),
        args=args,
        train_dataset=_TinyDataset(),
        callbacks=[SaveDelayCallback(save_delay_steps=2)],
    )
    trainer.create_optimizer_and_scheduler(num_training_steps=1)
    trainer.state.global_step = 1
    save_delay = next(
        cb for cb in trainer.callback_handler.callbacks if isinstance(cb, SaveDelayCallback)
    )
    save_delay._pending_reset = True
    save_delay._block_logged = True
    setattr(trainer, "_coordexp_disabled_diagnostics", {"coord_diag"})

    try:
        trainer._save_checkpoint(trainer.model, trial=None)  # type: ignore[misc]
    except TypeError:
        trainer._save_checkpoint(trainer.model, trial=None, metrics=None)  # type: ignore[misc,call-arg]

    resumed_args = TrainingArguments(
        output_dir=str(out_dir / "resume"),
        per_device_train_batch_size=2,
        save_strategy="steps",
        save_steps=1,
        logging_steps=1,
        report_to=[],
        save_only_model=False,
        save_safetensors=True,
        use_cpu=True,
        remove_unused_columns=False,
    )
    setattr(resumed_args, "checkpoint_mode", "restartable")
    setattr(resumed_args, "max_epochs", None)
    resumed = trainer_cls(
        model=_TinyModel(),
        args=resumed_args,
        train_dataset=_TinyDataset(),
        callbacks=[SaveDelayCallback(save_delay_steps=2)],
    )

    payload = prepare_restartable_checkpoint_resume(resumed, out_dir / "checkpoint-1")
    restored_save_delay = next(
        cb for cb in resumed.callback_handler.callbacks if isinstance(cb, SaveDelayCallback)
    )

    assert payload["global_step"] == 1
    assert restored_save_delay._pending_reset is True
    assert restored_save_delay._block_logged is True
    assert getattr(resumed, "_coordexp_disabled_diagnostics") == {"coord_diag"}


def test_prepare_restartable_checkpoint_resume_rejects_missing_callback(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "out"
    trainer_cls = with_final_checkpoint(Trainer)
    args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=2,
        save_strategy="steps",
        save_steps=1,
        logging_steps=1,
        report_to=[],
        save_only_model=False,
        save_safetensors=True,
        use_cpu=True,
        remove_unused_columns=False,
    )
    setattr(args, "checkpoint_mode", "restartable")
    setattr(args, "max_epochs", None)

    trainer = trainer_cls(
        model=_TinyModel(),
        args=args,
        train_dataset=_TinyDataset(),
        callbacks=[SaveDelayCallback(save_delay_steps=2)],
    )
    trainer.create_optimizer_and_scheduler(num_training_steps=1)
    trainer.state.global_step = 1

    try:
        trainer._save_checkpoint(trainer.model, trial=None)  # type: ignore[misc]
    except TypeError:
        trainer._save_checkpoint(trainer.model, trial=None, metrics=None)  # type: ignore[misc,call-arg]

    resumed_args = TrainingArguments(
        output_dir=str(out_dir / "resume"),
        per_device_train_batch_size=2,
        save_strategy="steps",
        save_steps=1,
        logging_steps=1,
        report_to=[],
        save_only_model=False,
        save_safetensors=True,
        use_cpu=True,
        remove_unused_columns=False,
    )
    setattr(resumed_args, "checkpoint_mode", "restartable")
    setattr(resumed_args, "max_epochs", None)
    resumed = trainer_cls(
        model=_TinyModel(),
        args=resumed_args,
        train_dataset=_TinyDataset(),
        callbacks=[],
    )

    with pytest.raises(ValueError, match="no matching callback"):
        prepare_restartable_checkpoint_resume(resumed, out_dir / "checkpoint-1")


def test_restartable_checkpoint_preflight_fails_for_weight_only_checkpoint(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "out"
    args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=2,
        save_strategy="steps",
        save_steps=1,
        logging_steps=1,
        report_to=[],
        save_only_model=True,
        save_safetensors=True,
        use_cpu=True,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=_TinyModel(),
        args=args,
        train_dataset=_TinyDataset(),
    )
    try:
        trainer.callback_handler.callbacks = []
    except Exception:
        pass
    trainer.create_optimizer_and_scheduler(num_training_steps=1)
    trainer.state.global_step = 1
    try:
        trainer._save_checkpoint(trainer.model, trial=None)  # type: ignore[misc]
    except TypeError:
        trainer._save_checkpoint(trainer.model, trial=None, metrics=None)  # type: ignore[misc,call-arg]

    with pytest.raises(ValueError, match="Restartable checkpoint is incomplete"):
        validate_restartable_checkpoint(out_dir / "checkpoint-1")


def test_restartable_resume_preserves_global_step_schedule_continuity(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "out"
    trainer_cls = with_final_checkpoint(Trainer)

    args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=2,
        max_steps=1,
        save_strategy="steps",
        save_steps=1,
        logging_steps=1,
        report_to=[],
        save_only_model=False,
        save_safetensors=True,
        use_cpu=True,
        remove_unused_columns=False,
    )
    setattr(args, "checkpoint_mode", "restartable")
    setattr(args, "max_epochs", None)

    trainer = trainer_cls(
        model=_TinyModel(),
        args=args,
        train_dataset=_TinyDataset(),
    )
    trainer.train()

    ckpt = out_dir / "checkpoint-1"
    assert ckpt.is_dir()

    resumed_args = TrainingArguments(
        output_dir=str(out_dir / "resume"),
        per_device_train_batch_size=2,
        max_steps=2,
        save_strategy="no",
        logging_steps=1,
        report_to=[],
        save_only_model=False,
        save_safetensors=True,
        use_cpu=True,
        remove_unused_columns=False,
    )
    setattr(resumed_args, "checkpoint_mode", "restartable")
    setattr(resumed_args, "max_epochs", None)
    resumed = trainer_cls(
        model=_TinyModel(),
        args=resumed_args,
        train_dataset=_TinyDataset(),
    )
    prepare_restartable_checkpoint_resume(resumed, ckpt)
    resumed.train(resume_from_checkpoint=str(ckpt))

    assert resumed.state.global_step == 2
    assert getattr(resumed.lr_scheduler, "last_epoch", 0) >= 1


def test_audited_configs_enable_save_only_model() -> None:
    from src.config.loader import ConfigLoader

    audited = [
        "configs/stage1/profiles/4b/coord_soft_ce_gate_coco80_geometry_first.yaml",
        "configs/stage2_two_channel/prod/ab_mixed.yaml",
    ]

    for path in audited:
        training_config = ConfigLoader.load_materialized_training_config(path, None)
        training_map = getattr(training_config, "training", None)
        val = training_map.get("save_only_model") if isinstance(training_map, dict) else None
        assert bool(val) is True, (
            f"{path} must set training.save_only_model=true via YAML inheritance "
            f"(got {val!r})"
        )


def test_best_save_strategy_records_best_checkpoint_state(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    trainer_cls = with_final_checkpoint(Trainer)
    args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=2,
        save_strategy="best",
        metric_for_best_model="loss",
        greater_is_better=False,
        logging_steps=1,
        report_to=[],
        save_only_model=True,
        save_safetensors=True,
        use_cpu=True,
        remove_unused_columns=False,
    )
    setattr(args, "save_last_epoch", False)

    trainer = trainer_cls(
        model=_TinyModel(),
        args=args,
        train_dataset=_TinyDataset(),
    )
    try:
        trainer.callback_handler.callbacks = []
    except Exception:
        pass

    trainer.create_optimizer_and_scheduler(num_training_steps=1)
    trainer.state.global_step = 1
    assert trainer._determine_best_metric({"eval_loss": 0.25}, trial=None) is True

    try:
        trainer._save_checkpoint(trainer.model, trial=None)  # type: ignore[misc]
    except TypeError:
        trainer._save_checkpoint(trainer.model, trial=None, metrics=None)  # type: ignore[misc,call-arg]

    ckpt = out_dir / "checkpoint-1"
    assert trainer.state.best_model_checkpoint == str(ckpt)
    assert trainer.state.best_global_step == 1

    state = json.loads((ckpt / "trainer_state.json").read_text())
    assert state["best_model_checkpoint"] == str(ckpt)
    assert state["best_global_step"] == 1


def test_final_checkpoint_callback_installation_tracks_configured_fallback_policy(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "out"
    trainer_cls = with_final_checkpoint(Trainer)

    args_best = TrainingArguments(
        output_dir=str(out_dir / "best"),
        per_device_train_batch_size=2,
        save_strategy="best",
        metric_for_best_model="loss",
        greater_is_better=False,
        logging_steps=1,
        report_to=[],
        save_only_model=True,
        save_safetensors=True,
        use_cpu=True,
        remove_unused_columns=False,
    )
    setattr(args_best, "save_last_epoch", True)
    trainer_best = trainer_cls(
        model=_TinyModel(),
        args=args_best,
        train_dataset=_TinyDataset(),
    )
    assert hasattr(trainer_best, trainer_best._final_checkpoint_callback_attr)

    args_best_no_final = TrainingArguments(
        output_dir=str(out_dir / "best_no_final"),
        per_device_train_batch_size=2,
        save_strategy="best",
        metric_for_best_model="loss",
        greater_is_better=False,
        logging_steps=1,
        report_to=[],
        save_only_model=True,
        save_safetensors=True,
        use_cpu=True,
        remove_unused_columns=False,
    )
    setattr(args_best_no_final, "save_last_epoch", False)
    trainer_best_no_final = trainer_cls(
        model=_TinyModel(),
        args=args_best_no_final,
        train_dataset=_TinyDataset(),
    )
    assert not hasattr(
        trainer_best_no_final, trainer_best_no_final._final_checkpoint_callback_attr
    )

    args_steps = TrainingArguments(
        output_dir=str(out_dir / "steps"),
        per_device_train_batch_size=2,
        save_strategy="steps",
        save_steps=10,
        logging_steps=1,
        report_to=[],
        save_only_model=True,
        save_safetensors=True,
        use_cpu=True,
        remove_unused_columns=False,
    )
    setattr(args_steps, "save_last_epoch", True)
    trainer_steps = trainer_cls(
        model=_TinyModel(),
        args=args_steps,
        train_dataset=_TinyDataset(),
    )
    assert not hasattr(trainer_steps, trainer_steps._final_checkpoint_callback_attr)


def test_non_best_followup_save_preserves_best_checkpoint_state(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    trainer_cls = with_final_checkpoint(Trainer)
    args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=2,
        save_strategy="best",
        metric_for_best_model="loss",
        greater_is_better=False,
        logging_steps=1,
        report_to=[],
        save_only_model=True,
        save_safetensors=True,
        use_cpu=True,
        remove_unused_columns=False,
    )
    setattr(args, "save_last_epoch", False)

    trainer = trainer_cls(
        model=_TinyModel(),
        args=args,
        train_dataset=_TinyDataset(),
    )
    try:
        trainer.callback_handler.callbacks = []
    except Exception:
        pass

    trainer.create_optimizer_and_scheduler(num_training_steps=2)
    trainer.state.global_step = 1
    assert trainer._determine_best_metric({"eval_loss": 0.25}, trial=None) is True

    try:
        trainer._save_checkpoint(trainer.model, trial=None)  # type: ignore[misc]
    except TypeError:
        trainer._save_checkpoint(trainer.model, trial=None, metrics=None)  # type: ignore[misc,call-arg]

    best_ckpt = out_dir / "checkpoint-1"

    trainer.state.global_step = 2
    try:
        trainer._save_checkpoint(trainer.model, trial=None)  # type: ignore[misc]
    except TypeError:
        trainer._save_checkpoint(trainer.model, trial=None, metrics=None)  # type: ignore[misc,call-arg]

    final_ckpt = out_dir / "checkpoint-2"
    assert final_ckpt.is_dir()
    assert trainer.state.best_model_checkpoint == str(best_ckpt)
    assert trainer.state.best_global_step == 1

    state = json.loads((final_ckpt / "trainer_state.json").read_text())
    assert state["best_model_checkpoint"] == str(best_ckpt)
    assert state["best_global_step"] == 1
