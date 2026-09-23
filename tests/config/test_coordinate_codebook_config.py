from __future__ import annotations

import pytest
from pathlib import Path

from src.config.models import (
    CoordinateCodebookConfig,
    SchedulerConfig,
    SpecialTokenEmbeddingsConfig,
)
from src.config.paths import resolve_path_fields


def test_coordinate_codebook_and_untied_config_are_explicit() -> None:
    config = CoordinateCodebookConfig(initial_gain=0.05, checkpoint_path="fit.pt")
    assert config.initial_gain == pytest.approx(0.05)
    assert config.checkpoint_path == "fit.pt"
    assert config.mode == "late_masked_center"
    early = CoordinateCodebookConfig(
        initial_gain=0.05, mode="early_patch_edges", projection_seed=1729
    )
    assert early.mode == "early_patch_edges"
    assert early.projection_seed == 1729
    special = SpecialTokenEmbeddingsConfig(
        groups={
            "coordinate_tokens": "default_coord_0_999",
            "wrapper_tokens": "default_object_box_wrappers",
        },
        tie_word_embeddings=False,
    )
    assert special.tie_word_embeddings is False
    assert SchedulerConfig(name="constant_with_warmup", warmup_steps=10).name == "constant_with_warmup"


def test_coordinate_codebook_config_rejects_nonpositive_gain() -> None:
    with pytest.raises(ValueError):
        CoordinateCodebookConfig(initial_gain=0.0)


def test_codebook_and_resume_paths_resolve_from_declaring_config() -> None:
    config_path = Path("/tmp/coordexp-codebook/config.yaml")
    payload = {
        "model": {"coordinate_codebook": {"checkpoint_path": "codebook"}},
        "training": {"resume_from_checkpoint": "resume"},
    }
    resolved, origins = resolve_path_fields(
        payload,
        {
            "model.coordinate_codebook.checkpoint_path": config_path,
            "training.resume_from_checkpoint": config_path,
        },
    )
    assert resolved["model"]["coordinate_codebook"]["checkpoint_path"] == "/tmp/coordexp-codebook/codebook"
    assert resolved["training"]["resume_from_checkpoint"] == "/tmp/coordexp-codebook/resume"
    assert origins["model.coordinate_codebook.checkpoint_path"].declared_path == "codebook"
    assert origins["training.resume_from_checkpoint"].declared_path == "resume"
