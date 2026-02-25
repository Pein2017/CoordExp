import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments


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


def test_audited_configs_enable_save_only_model() -> None:
    from src.config.loader import ConfigLoader

    audited = [
        "configs/stage1/ablation/geometry_first_coco80.yaml",
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
