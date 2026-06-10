from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.config.loader import ConfigLoader
from src.config.schema import DetectionTrainingConfig
from src.trainers.losses.coord_soft_ce_w1 import build_coord_id_map
from src.trainers.losses.sft_gaussian_coord_soft_ce import (
    compute_sft_gaussian_coord_soft_ce_loss,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
GAUSS_CONFIG = (
    REPO_ROOT
    / "configs/stage1/recursive_detection_ce_latest/prod/"
    / "compact_full_random_sft_coord_gauss_softce_mix0p5_frac0p04_cap8_llm_lora_packed.yaml"
)
HARD_CE_CONFIG = (
    REPO_ROOT
    / "configs/stage1/recursive_detection_ce_latest/prod/"
    / "compact_full_random_sft_llm_lora_packed.yaml"
)


def test_sft_gaussian_coord_soft_ce_config_parses_as_packed_sft() -> None:
    cfg = ConfigLoader.load_materialized_training_config(str(GAUSS_CONFIG))

    assert isinstance(cfg, DetectionTrainingConfig)
    assert cfg.objective.id == "sft"
    assert cfg.objective.variant == "random_order_sft"
    assert cfg.training["packing"] is True
    assert cfg.tuner["freeze_vit"] is True
    assert cfg.tuner["freeze_aligner"] is True
    assert "model\\.visual" not in str(cfg.tuner["target_regex"])
    assert cfg.tuner["lora_rank"] == 16
    assert cfg.tuner["lora_alpha"] == 32
    assert cfg.objective.coord_soft_ce is not None
    assert cfg.objective.coord_soft_ce.target_distribution == "gaussian_around_gold"
    assert cfg.objective.coord_soft_ce.gaussian_mixture_weight == pytest.approx(0.5)
    assert cfg.objective.coord_soft_ce.gaussian_r95_axis_fraction == pytest.approx(0.04)
    assert cfg.objective.coord_soft_ce.gaussian_r95_cap_bins == 8


def test_sft_gaussian_and_hard_ce_configs_are_matched_except_objective_and_names() -> None:
    gauss = ConfigLoader.load_materialized_training_config(str(GAUSS_CONFIG))
    hard = ConfigLoader.load_materialized_training_config(str(HARD_CE_CONFIG))
    assert isinstance(gauss, DetectionTrainingConfig)
    assert isinstance(hard, DetectionTrainingConfig)

    assert gauss.model == hard.model
    assert gauss.template == hard.template
    assert gauss.data == hard.data
    assert gauss.prompt == hard.prompt
    assert gauss.detection_template == hard.detection_template
    assert gauss.token_rows == hard.token_rows
    assert gauss.packing == hard.packing
    assert gauss.tuner == hard.tuner
    assert gauss.training["packing"] == hard.training["packing"] == True
    assert gauss.training["effective_batch_size"] == hard.training["effective_batch_size"]
    assert gauss.training["learning_rate"] == hard.training["learning_rate"]
    assert gauss.training["num_train_epochs"] == hard.training["num_train_epochs"]


def test_sft_gaussian_coord_soft_ce_rejects_recursive_distribution_for_sft() -> None:
    payload = ConfigLoader.load_materialized_training_config(str(HARD_CE_CONFIG)).to_mapping()
    payload["objective"]["coord_soft_ce"] = {
        "enabled": True,
        "target_distribution": "instance_trie_gaussian",
        "gaussian_mixture_weight": 0.5,
        "gaussian_r95_axis_fraction": 0.04,
        "gaussian_r95_cap_bins": 8,
    }

    with pytest.raises(ValueError, match="gaussian_around_gold"):
        DetectionTrainingConfig.from_mapping(payload)


def test_sft_gaussian_coord_soft_ce_loss_uses_packed_coord_positions() -> None:
    coord_ids = list(range(1000, 2000))
    vocab_size = 2000
    coord_id_map = build_coord_id_map(
        vocab_size=vocab_size,
        device=torch.device("cpu"),
        coord_token_ids=coord_ids,
    )
    # Shifted labels contain two packed boxes with four adjacent coord tokens each.
    labels_next = torch.tensor(
        [[1010, 1020, 1110, 1220, 1005, 1030, 1060, 1090]],
        dtype=torch.long,
    )
    labels = torch.cat([torch.tensor([[42]], dtype=torch.long), labels_next], dim=1)
    logits = torch.zeros((1, labels.shape[1] - 1, vocab_size), dtype=torch.float32)
    for pos, token_id in enumerate(labels_next[0].tolist()):
        logits[0, pos, int(token_id)] = 4.0

    result = compute_sft_gaussian_coord_soft_ce_loss(
        logits=logits,
        labels=labels,
        coord_token_ids=coord_ids,
        coord_id_map=coord_id_map,
        cfg=SimpleNamespace(
            gaussian_mixture_weight=0.5,
            gaussian_r95_axis_fraction=0.04,
            gaussian_r95_cap_bins=8,
        ),
        average_tokens_across_devices=False,
        model_accepts_loss_kwargs=False,
        accelerator_num_processes=None,
    )

    assert result is not None
    assert result.coord_tokens == 8
    assert torch.isfinite(result.loss)
    assert float(result.loss.item()) > 0.0
    assert float(result.target_peak_prob.item()) < 1.0
    assert float(result.target_r95_radius_max.item()) <= 8.0
