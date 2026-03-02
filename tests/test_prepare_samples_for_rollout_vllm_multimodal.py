from __future__ import annotations

import types

from src.trainers.stage2_rollout_aligned import RolloutMatchingSFTTrainer


def _make_trainer(system: str = "SYS") -> RolloutMatchingSFTTrainer:
    trainer = RolloutMatchingSFTTrainer.__new__(RolloutMatchingSFTTrainer)
    trainer.template = types.SimpleNamespace(system=system)
    return trainer


def test_prepare_samples_for_rollout_promotes_image_to_images_for_vllm() -> None:
    trainer = _make_trainer()

    sample = {
        "messages": [{"role": "user", "content": "hi"}],
        "image": "foo.png",
    }

    out = trainer._prepare_samples_for_rollout([sample], rollout_backend="vllm")
    assert len(out) == 1
    assert out[0] is not sample
    assert out[0]["images"] == ["foo.png"]
    assert "images" not in sample


def test_prepare_samples_for_rollout_normalizes_images_tuple_to_list_for_vllm() -> None:
    trainer = _make_trainer()

    sample = {
        "messages": [{"role": "user", "content": "hi"}],
        "images": ("a.png", "b.png"),
    }

    out = trainer._prepare_samples_for_rollout([sample], rollout_backend="vllm")
    assert len(out) == 1
    assert out[0] is not sample
    assert out[0]["images"] == ["a.png", "b.png"]
    assert isinstance(out[0]["images"], list)


def test_prepare_samples_for_rollout_leaves_images_list_unchanged_when_already_valid() -> None:
    trainer = _make_trainer()

    sample = {
        "messages": [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "hi"},
        ],
        "images": ["a.png"],
    }

    out = trainer._prepare_samples_for_rollout([sample], rollout_backend="vllm")
    assert len(out) == 1
    assert out[0] is sample
