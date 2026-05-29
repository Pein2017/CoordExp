from __future__ import annotations

import types

import pytest

from src.trainers.stage2_rollout_runtime import Stage2RolloutRuntime
from src.infer.prompt import prepare_rollout_prompt_samples_from_owner


def _make_trainer(system: str = "SYS") -> Stage2RolloutRuntime:
    trainer = Stage2RolloutRuntime.__new__(Stage2RolloutRuntime)
    trainer.template = types.SimpleNamespace(system=system)
    return trainer


def test_prepare_samples_for_rollout_promotes_image_to_images_for_vllm() -> None:
    trainer = _make_trainer()

    sample = {
        "messages": [{"role": "user", "content": "hi"}],
        "image": "foo.png",
    }

    out = prepare_rollout_prompt_samples_from_owner(
        trainer, [sample], rollout_backend="vllm"
    )
    assert len(out) == 1
    assert out[0] is not sample
    assert out[0]["images"] == ["foo.png"]
    assert "images" not in sample


def test_prepare_samples_for_rollout_normalizes_single_image_tuple_to_list_for_vllm() -> None:
    trainer = _make_trainer()

    sample = {
        "messages": [{"role": "user", "content": "hi"}],
        "images": ("a.png",),
        "width": 16,
        "height": 8,
    }

    out = prepare_rollout_prompt_samples_from_owner(
        trainer, [sample], rollout_backend="vllm"
    )
    assert len(out) == 1
    assert out[0] is not sample
    assert out[0]["images"] == ["a.png"]
    assert isinstance(out[0]["images"], list)
    assert out[0]["_coordexp_prompt_visual_metadata"] == {
        "image_count": 1,
        "image_path": "a.png",
        "image_placement": "sample.images",
        "do_resize": False,
        "original_width": 16,
        "original_height": 8,
        "post_preprocessing_width": 16,
        "post_preprocessing_height": 8,
    }


def test_prepare_samples_for_rollout_rejects_multiple_images_for_vllm() -> None:
    trainer = _make_trainer()

    sample = {
        "messages": [{"role": "user", "content": "hi"}],
        "images": ("a.png", "b.png"),
    }

    with pytest.raises(ValueError, match="exactly one image"):
        prepare_rollout_prompt_samples_from_owner(
            trainer, [sample], rollout_backend="vllm"
        )


def test_prepare_samples_for_rollout_stamps_visual_metadata_for_hf() -> None:
    trainer = _make_trainer()

    sample = {
        "messages": [{"role": "user", "content": "hi"}],
        "images": ["a.png"],
        "width": 16,
        "height": 8,
    }

    out = prepare_rollout_prompt_samples_from_owner(
        trainer, [sample], rollout_backend="hf"
    )
    assert len(out) == 1
    assert out[0] is not sample
    assert out[0]["_coordexp_prompt_visual_metadata"] == {
        "image_count": 1,
        "image_path": "a.png",
        "image_placement": "sample.images",
        "do_resize": False,
        "original_width": 16,
        "original_height": 8,
        "post_preprocessing_width": 16,
        "post_preprocessing_height": 8,
    }


def test_prepare_samples_for_rollout_leaves_images_list_unchanged_when_already_valid() -> None:
    trainer = _make_trainer()

    sample = {
        "messages": [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "hi"},
        ],
        "images": ["a.png"],
    }

    out = prepare_rollout_prompt_samples_from_owner(
        trainer, [sample], rollout_backend="vllm"
    )
    assert len(out) == 1
    assert out[0] is not sample
    assert out[0]["_coordexp_prompt_visual_metadata"]["image_count"] == 1
    assert out[0]["_coordexp_prompt_visual_metadata"]["do_resize"] is False
