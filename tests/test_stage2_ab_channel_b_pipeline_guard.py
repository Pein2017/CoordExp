from __future__ import annotations

from src.trainers.stage2_rollout_correction import Stage2RolloutCorrectionTrainer


def test_rollout_correction_pipeline_disabled_under_ddp_vllm_server() -> None:
    trainer = Stage2RolloutCorrectionTrainer.__new__(Stage2RolloutCorrectionTrainer)
    trainer._dist_info = lambda: (0, 2, object())

    enabled = trainer._stage2_rollout_correction_pipeline_enabled(
        backend="vllm",
        mode="server",
    )

    assert enabled is False
    assert bool(getattr(trainer, "_stage2_rollout_correction_pipeline_ddp_warned", False))


def test_rollout_correction_pipeline_enabled_only_for_single_rank_vllm_server() -> None:
    trainer = Stage2RolloutCorrectionTrainer.__new__(Stage2RolloutCorrectionTrainer)
    trainer._dist_info = lambda: (0, 1, None)

    assert (
        trainer._stage2_rollout_correction_pipeline_enabled(
            backend="vllm",
            mode="server",
        )
        is True
    )
    assert (
        trainer._stage2_rollout_correction_pipeline_enabled(
            backend="hf",
            mode="server",
        )
        is False
    )
    assert (
        trainer._stage2_rollout_correction_pipeline_enabled(
            backend="vllm",
            mode="colocate",
        )
        is False
    )
