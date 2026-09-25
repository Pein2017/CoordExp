from pathlib import Path

import torch

from probes.coordinate_representation.coordinate_codebook_alignment import injection_off_train, scale_train


def test_off_entry_uses_eight_rank_three_loss_hook_and_restores_injected_contract(monkeypatch):
    original = (
        scale_train.EXPECTED_WORLD_SIZE,
        scale_train.EXPECTED_GRAD_ACCUM,
        scale_train.EXPECTED_UPDATES,
        scale_train.EXPECTED_CATEGORIES,
        scale_train._patch_trainer,
    )

    def caller(config, output, packing_plan):
        assert (config, output, packing_plan) == (Path("config"), Path("receipt"), Path("packs"))
        assert (scale_train.EXPECTED_WORLD_SIZE, scale_train.EXPECTED_GRAD_ACCUM,
                scale_train.EXPECTED_UPDATES) == (8, 1, 984)
        assert scale_train.EXPECTED_CATEGORIES == (
            "language", "vision", "aligner", "input", "output"
        )
        assert scale_train._patch_trainer is not original[-1]
        return {"status": "candidate"}

    monkeypatch.setattr(scale_train, "run", caller)
    assert injection_off_train.run(Path("config"), Path("receipt"), Path("packs")) == {"status": "candidate"}
    assert (
        scale_train.EXPECTED_WORLD_SIZE,
        scale_train.EXPECTED_GRAD_ACCUM,
        scale_train.EXPECTED_UPDATES,
        scale_train.EXPECTED_CATEGORIES,
        scale_train._patch_trainer,
    ) == original


def test_gradient_categories_drop_only_an_intentionally_absent_codebook(monkeypatch):
    monkeypatch.setattr(scale_train, "_embedding_delta_ids", lambda _model: (set(), set()))
    model = torch.nn.Module()
    model.language_model = torch.nn.Module()
    model.language_model.weight = torch.nn.Parameter(torch.ones(1))
    model.language_model.weight.grad = torch.ones(1)
    off = scale_train._category_gradient_summary(model)
    assert "codebook" not in off and off["language"]["nonzero_count"] == 1

    model.coordinate_codebook = torch.nn.Module()
    model.coordinate_codebook.raw_gain = torch.nn.Parameter(torch.ones(1))
    model.coordinate_codebook.raw_gain.grad = torch.ones(1)
    injected = scale_train._category_gradient_summary(model)
    assert injected["codebook"]["nonzero_count"] == 1
