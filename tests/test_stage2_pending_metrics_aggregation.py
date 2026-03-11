from __future__ import annotations

import types

import pytest

from src.trainers.stage2_two_channel import (
    Stage2ABTrainingTrainer,
    _PendingStage2Log,
    _merge_latest_stage2_metric_snapshots,
)


def test_stage2_pending_log_finalize_averages_losses_and_sums_counters() -> None:
    pending = _PendingStage2Log()

    pending.add(
        {
            "loss/B_coord/bbox_smoothl1": 1.0,
            "loss/B_coord/coord_soft_ce": 2.0,
            "stage2/channel_b": 1.0,
            "stage2/raw_rollouts": 1.0,
            "rollout/seed_base": 10.0,
            "rollout/parse_truncated": 1.0,
        }
    )
    pending.add(
        {
            "loss/B_coord/bbox_smoothl1": 3.0,
            "loss/B_coord/coord_soft_ce": 4.0,
            "stage2/channel_b": 0.0,
            "stage2/raw_rollouts": 2.0,
            "rollout/seed_base": 10.0,
            "rollout/parse_truncated": 0.0,
        }
    )

    out = pending.finalize(drop_internal=False)

    # Averaged across micro-batches (n_micro=2).
    assert out["loss/B_coord/bbox_smoothl1"] == pytest.approx(2.0)
    assert out["loss/B_coord/coord_soft_ce"] == pytest.approx(3.0)
    assert out["stage2/channel_b"] == pytest.approx(0.5)
    assert out["rollout/seed_base"] == pytest.approx(10.0)

    # Summed counters.
    assert out["stage2/raw_rollouts"] == pytest.approx(3.0)
    assert out["rollout/parse_truncated"] == pytest.approx(1.0)

    # Derived rate is always computed from numerator/denominator.
    assert out["rollout/parse_truncated_rate"] == pytest.approx(1.0 / 3.0)


def test_stage2_pending_log_finalize_uses_segment_weight_when_provided() -> None:
    pending = _PendingStage2Log()

    pending.add(
        {
            "loss/B_coord/bbox_smoothl1": 10.0,
            "stage2/_log_weight": 1.0,
            "stage2/raw_rollouts": 1.0,
        }
    )
    pending.add(
        {
            "loss/B_coord/bbox_smoothl1": 20.0,
            "stage2/_log_weight": 3.0,
            "stage2/raw_rollouts": 2.0,
        }
    )

    out = pending.finalize()

    assert out["loss/B_coord/bbox_smoothl1"] == pytest.approx((10.0 * 1.0 + 20.0 * 3.0) / 4.0)
    assert out["stage2/raw_rollouts"] == pytest.approx(3.0)
    assert "stage2/_log_weight_total" not in out


def test_stage2_pending_log_counter_suffixes_sum_not_weighted() -> None:
    pending = _PendingStage2Log()

    pending.add(
        {
            "loss/A1_text/struct_ce": 1.0,
            "rollout/fp_total": 2.0,
            "rollout/fn_total": 1.0,
            "rollout/matched_maskiou_count": 3.0,
            "stage2/_log_weight": 1.0,
        }
    )
    pending.add(
        {
            "loss/A1_text/struct_ce": 3.0,
            "rollout/fp_total": 5.0,
            "rollout/fn_total": 4.0,
            "rollout/matched_maskiou_count": 7.0,
            "stage2/_log_weight": 3.0,
        }
    )

    out = pending.finalize()

    # Mean-like loss keys are weighted by stage2/_log_weight.
    assert out["loss/A1_text/struct_ce"] == pytest.approx(
        (1.0 * 1.0 + 3.0 * 3.0) / 4.0
    )

    # Counter-like keys with suffixes are always summed.
    assert out["rollout/fp_total"] == pytest.approx(7.0)
    assert out["rollout/fn_total"] == pytest.approx(5.0)
    assert out["rollout/matched_maskiou_count"] == pytest.approx(10.0)

    # Internal helper keys are removed from final payload.
    assert "stage2/_log_weight_total" not in out
    assert "rollout/_parse_truncated_num" not in out
    assert "rollout/_parse_truncated_den" not in out


def test_stage2_pending_log_emits_canonical_loss_prefix_only() -> None:
    pending = _PendingStage2Log()
    pending.add(
        {
            "loss/A1_text/struct_ce": 0.5,
            "loss/A1_text/desc_ce": 0.25,
            "loss/A2_coord/bbox_smoothl1": 0.25,
            "loss/A2_coord/coord_soft_ce": 0.125,
        }
    )

    out = pending.finalize()

    assert "loss/A1_text/struct_ce" in out
    assert "loss/A1_text/desc_ce" in out
    assert "loss/A2_coord/bbox_smoothl1" in out
    assert "loss/A2_coord/coord_soft_ce" in out
    assert "loss/token_ce_obj" not in out
    assert "loss/bbox_geo_obj" not in out
    assert "loss/coord_reg_obj" not in out


def test_stage2_pending_log_aggregates_duplicate_metrics_with_mean_and_sum_semantics() -> None:
    pending = _PendingStage2Log()
    pending.add(
        {
            "dup/max_desc_count": 2.0,
            "dup/saturation_rate": 0.25,
            "dup/near_iou90_pairs_same_desc_count": 3.0,
            "stage2_ab/channel_b/dup/N_duplicates": 4.0,
            "stage2_ab/channel_b/dup/N_ul_boundaries": 1.0,
            "stage2/_log_weight": 1.0,
        }
    )
    pending.add(
        {
            "dup/max_desc_count": 6.0,
            "dup/saturation_rate": 0.75,
            "dup/near_iou90_pairs_same_desc_count": 5.0,
            "stage2_ab/channel_b/dup/N_duplicates": 7.0,
            "stage2_ab/channel_b/dup/N_ul_boundaries": 2.0,
            "stage2/_log_weight": 3.0,
        }
    )

    out = pending.finalize()

    assert out["dup/max_desc_count"] == pytest.approx((2.0 * 1.0 + 6.0 * 3.0) / 4.0)
    assert out["dup/saturation_rate"] == pytest.approx((0.25 * 1.0 + 0.75 * 3.0) / 4.0)
    assert out["dup/near_iou90_pairs_same_desc_count"] == pytest.approx(8.0)
    assert out["stage2_ab/channel_b/dup/N_duplicates"] == pytest.approx(11.0)
    assert out["stage2_ab/channel_b/dup/N_ul_boundaries"] == pytest.approx(3.0)

def test_latest_stage2_metric_snapshots_carry_forward_channel_specific_keys() -> None:
    latest: dict[str, float] = {}

    first = _merge_latest_stage2_metric_snapshots(
        latest,
        {
            "loss/A1_text/struct_ce": 0.5,
            "coord_diag/A1/acc_top5": 0.4,
            "stage2/channel_a": 1.0,
            "time/channel_a_teacher_encode_s": 1.2,
            "time/forward_s": 12.0,
        },
    )

    assert first["latest/loss/A1_text/struct_ce"] == pytest.approx(0.5)
    assert first["latest/coord_diag/A1/acc_top5"] == pytest.approx(0.4)
    assert first["latest/stage2/channel_a"] == pytest.approx(1.0)
    assert first["latest/time/channel_a_teacher_encode_s"] == pytest.approx(1.2)
    assert "latest/time/forward_s" not in first

    second = _merge_latest_stage2_metric_snapshots(
        latest,
        {
            "loss/B_rollout_text/struct_ce": 0.8,
            "rollout/f1": 0.3,
            "stage2/channel_b": 1.0,
            "time/rollout_generate_s": 9.0,
        },
    )

    assert second["latest/loss/A1_text/struct_ce"] == pytest.approx(0.5)
    assert second["latest/loss/B_rollout_text/struct_ce"] == pytest.approx(0.8)
    assert second["latest/rollout/f1"] == pytest.approx(0.3)
    assert second["latest/stage2/channel_b"] == pytest.approx(1.0)
    assert second["latest/time/rollout_generate_s"] == pytest.approx(9.0)


def test_stage2_log_emits_latest_snapshots_alongside_current_reduced_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)
    trainer.state = types.SimpleNamespace(global_step=1)
    trainer._stage2_pending_train_logs = {1: _PendingStage2Log()}
    trainer._stage2_pending_train_logs[1].add(
        {
            "loss/B_rollout_text/struct_ce": 0.8,
            "stage2/channel_b": 1.0,
            "rollout/f1": 0.3,
        }
    )
    trainer._stage2_latest_metric_snapshots = {
        "latest/loss/A1_text/struct_ce": 0.5,
        "latest/coord_diag/A1/acc_top5": 0.4,
    }
    trainer._ddp_assert_all_ranks_true_or_raise = (
        lambda **_kwargs: None
    )  # type: ignore[method-assign]
    trainer._reduce_stage2_pending_metrics_global = (
        lambda metrics: dict(metrics)
    )  # type: ignore[method-assign]
    trainer._stage_wallclock_metrics_local = lambda: {
        "time/sft_total_time": 12.0,
        "time/rollout_total_time": 5.0,
    }
    trainer._reduce_stage_wallclock_metrics_global = (
        lambda metrics: dict(metrics)
    )  # type: ignore[method-assign]

    captured: dict[str, float] = {}

    def _capture_super_log(self, logs):
        captured.update(dict(logs))
        return None

    monkeypatch.setattr(
        "src.trainers.stage2_rollout_aligned.RolloutMatchingSFTTrainer.log",
        _capture_super_log,
    )

    Stage2ABTrainingTrainer.log(trainer, {"loss": 1.0})

    assert captured["loss"] == pytest.approx(1.0)
    assert captured["loss/B_rollout_text/struct_ce"] == pytest.approx(0.8)
    assert captured["rollout/f1"] == pytest.approx(0.3)
    assert captured["latest/loss/A1_text/struct_ce"] == pytest.approx(0.5)
    assert captured["latest/coord_diag/A1/acc_top5"] == pytest.approx(0.4)
    assert captured["latest/loss/B_rollout_text/struct_ce"] == pytest.approx(0.8)
    assert captured["latest/rollout/f1"] == pytest.approx(0.3)
    assert captured["time/sft_total_time"] == pytest.approx(12.0)
    assert captured["time/rollout_total_time"] == pytest.approx(5.0)


def test_stage2_pending_log_preserves_sparse_gradmon_weighting() -> None:
    pending = _PendingStage2Log()
    pending.add(
        {
            "loss/B_coord/bbox_smoothl1": 1.0,
            "stage2/_log_weight": 1.0,
        }
    )
    pending.add(
        {
            "gradmon/neg_cosine_pair_frac": 0.75,
            "gradmon/num_terms": 4.0,
            "time/gradmon_s": 0.2,
            "stage2/_log_weight": 3.0,
        }
    )

    out = pending.finalize(drop_internal=False)

    assert out["loss/B_coord/bbox_smoothl1"] == pytest.approx(0.25)
    assert out["gradmon/neg_cosine_pair_frac"] == pytest.approx(0.75)
    assert out["gradmon/num_terms"] == pytest.approx(4.0)
    assert out["time/gradmon_s"] == pytest.approx(0.2)
    assert out["stage2/_log_weight_total"] == pytest.approx(4.0)
    assert out["gradmon/_log_weight_total"] == pytest.approx(3.0)
