from __future__ import annotations

import types

import pytest

from src.trainers.stage2_rollout_correction import (
    Stage2RolloutCorrectionTrainer,
    _PendingStage2Log,
    _merge_stage2_metric_snapshots,
)
from src.trainers.stage2_coordination import resolve_rollout_correction_metric_spec


def test_stage2_pending_log_finalize_averages_losses_and_sums_counters() -> None:
    pending = _PendingStage2Log()

    pending.add(
        {
            "stage2_rollout_correction/residual_set/type_loss": 1.0,
            "stage2_rollout_correction/residual_set/inner_loss": 2.0,
            "stage2/raw_rollouts": 1.0,
            "rollout/seed_base": 10.0,
            "rollout/parse_truncated": 1.0,
        }
    )
    pending.add(
        {
            "stage2_rollout_correction/residual_set/type_loss": 3.0,
            "stage2_rollout_correction/residual_set/inner_loss": 4.0,
            "stage2/raw_rollouts": 2.0,
            "rollout/seed_base": 10.0,
            "rollout/parse_truncated": 0.0,
        }
    )

    out = pending.finalize(drop_internal=False)

    # Averaged across micro-batches (n_micro=2).
    assert out["stage2_rollout_correction/residual_set/type_loss"] == pytest.approx(2.0)
    assert out["stage2_rollout_correction/residual_set/inner_loss"] == pytest.approx(3.0)
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
            "stage2_rollout_correction/residual_set/sequence_loss": 10.0,
            "stage2/_log_weight": 1.0,
            "stage2/raw_rollouts": 1.0,
        }
    )
    pending.add(
        {
            "stage2_rollout_correction/residual_set/sequence_loss": 20.0,
            "stage2/_log_weight": 3.0,
            "stage2/raw_rollouts": 2.0,
        }
    )

    out = pending.finalize()

    assert out["stage2_rollout_correction/residual_set/sequence_loss"] == pytest.approx((10.0 * 1.0 + 20.0 * 3.0) / 4.0)
    assert out["stage2/raw_rollouts"] == pytest.approx(3.0)
    assert "stage2/_log_weight_total" not in out


def test_stage2_pack_schedule_metric_specs_are_explicit_gauges() -> None:
    for key in (
        "packing/post_rollout_local_pack_count",
        "packing/post_rollout_global_slot_count",
        "packing/post_rollout_empty_slot_count",
    ):
        spec = resolve_rollout_correction_metric_spec(key)
        assert spec.local_mode == "weighted_mean"
        assert spec.ddp_mode == "max"
        assert spec.ddp_weight_key is None


def test_stage2_pending_log_counter_suffixes_sum_not_weighted() -> None:
    pending = _PendingStage2Log()

    pending.add(
        {
            "loss/text/struct_ce": 1.0,
            "rollout/fp_total": 2.0,
            "rollout/fn_total": 1.0,
            "rollout/matched_maskiou_count": 3.0,
            "stage2/_log_weight": 1.0,
        }
    )
    pending.add(
        {
            "loss/text/struct_ce": 3.0,
            "rollout/fp_total": 5.0,
            "rollout/fn_total": 4.0,
            "rollout/matched_maskiou_count": 7.0,
            "stage2/_log_weight": 3.0,
        }
    )

    out = pending.finalize()

    # Mean-like loss keys are weighted by stage2/_log_weight.
    assert out["loss/text/struct_ce"] == pytest.approx(
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


def test_stage2_pending_log_emits_rollout_correction_loss_prefix_only() -> None:
    pending = _PendingStage2Log()
    pending.add(
        {
            "stage2_rollout_correction/residual_set/sequence_loss": 0.5,
            "stage2_rollout_correction/residual_set/type_loss": 0.25,
            "stage2_rollout_correction/residual_set/inner_loss": 0.125,
        }
    )

    out = pending.finalize()

    assert "stage2_rollout_correction/residual_set/sequence_loss" in out
    assert "stage2_rollout_correction/residual_set/type_loss" in out
    assert "stage2_rollout_correction/residual_set/inner_loss" in out
    assert "loss/text/struct_ce" not in out
    assert "loss/token_ce_obj" not in out
    assert "loss/bbox_geo_obj" not in out
    assert "loss/coord_reg_obj" not in out


def test_stage2_pending_log_aggregates_duplicate_metrics_with_mean_and_sum_semantics() -> None:
    pending = _PendingStage2Log()
    pending.add(
        {
            "dup/raw/max_desc_count": 2.0,
            "dup/raw/saturation_rate": 0.25,
            "dup/raw/duplicate_like_max_cluster_size": 3.0,
            "dup/raw/desc_entropy": 0.4,
            "dup/raw/near_iou90_pairs_same_desc_count": 3.0,
            "stage2_rollout_correction/correction/dup/N_clusters_total": 4.0,
            "stage2_rollout_correction/correction/dup/N_duplicate_control_first_divergence_boundaries": 1.0,
            "stage2/_log_weight": 1.0,
        }
    )
    pending.add(
        {
            "dup/raw/max_desc_count": 6.0,
            "dup/raw/saturation_rate": 0.75,
            "dup/raw/duplicate_like_max_cluster_size": 5.0,
            "dup/raw/desc_entropy": 1.2,
            "dup/raw/near_iou90_pairs_same_desc_count": 5.0,
            "stage2_rollout_correction/correction/dup/N_clusters_total": 7.0,
            "stage2_rollout_correction/correction/dup/N_duplicate_control_first_divergence_boundaries": 2.0,
            "stage2/_log_weight": 3.0,
        }
    )

    out = pending.finalize()

    assert out["dup/raw/max_desc_count"] == pytest.approx((2.0 * 1.0 + 6.0 * 3.0) / 4.0)
    assert out["dup/raw/saturation_rate"] == pytest.approx((0.25 * 1.0 + 0.75 * 3.0) / 4.0)
    assert out["dup/raw/duplicate_like_max_cluster_size"] == pytest.approx(
        (3.0 * 1.0 + 5.0 * 3.0) / 4.0
    )
    assert out["dup/raw/desc_entropy"] == pytest.approx((0.4 * 1.0 + 1.2 * 3.0) / 4.0)
    assert out["dup/raw/near_iou90_pairs_same_desc_count"] == pytest.approx(8.0)
    assert out["stage2_rollout_correction/correction/dup/N_clusters_total"] == pytest.approx(11.0)
    assert out[
        "stage2_rollout_correction/correction/dup/N_duplicate_control_first_divergence_boundaries"
    ] == pytest.approx(3.0)

def test_stage2_metric_snapshots_carry_forward_rollout_correction_keys() -> None:
    snapshots: dict[str, float] = {}

    first = _merge_stage2_metric_snapshots(
        snapshots,
        {
            "stage2_rollout_correction/residual_set/sequence_loss": 0.5,
            "stage2_rollout_correction/residual_set/valid_set_mass": 0.4,
            "time/rollout_prepare_s": 1.2,
            "time/forward_s": 12.0,
        },
    )

    assert first["snapshot/stage2_rollout_correction/residual_set/sequence_loss"] == pytest.approx(0.5)
    assert first["snapshot/stage2_rollout_correction/residual_set/valid_set_mass"] == pytest.approx(0.4)
    assert first["snapshot/time/rollout_prepare_s"] == pytest.approx(1.2)
    assert "time/forward_s" not in first

    second = _merge_stage2_metric_snapshots(
        snapshots,
        {
            "stage2_rollout_correction/residual_set/type_loss": 0.8,
            "rollout/f1": 0.3,
            "time/rollout_generate_s": 9.0,
        },
    )

    assert second["snapshot/stage2_rollout_correction/residual_set/sequence_loss"] == pytest.approx(0.5)
    assert second["snapshot/stage2_rollout_correction/residual_set/type_loss"] == pytest.approx(0.8)
    assert second["snapshot/rollout/f1"] == pytest.approx(0.3)
    assert second["snapshot/time/rollout_generate_s"] == pytest.approx(9.0)


def test_stage2_log_emits_snapshots_alongside_current_reduced_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = Stage2RolloutCorrectionTrainer.__new__(Stage2RolloutCorrectionTrainer)
    trainer.state = types.SimpleNamespace(global_step=1)
    trainer._stage2_pending_train_logs = {1: _PendingStage2Log()}
    trainer._stage2_pending_train_logs[1].add(
        {
            "stage2_rollout_correction/residual_set/type_loss": 0.8,
            "rollout/f1": 0.3,
        }
    )
    trainer._stage2_metric_snapshots = {
        "snapshot/stage2_rollout_correction/residual_set/sequence_loss": 0.5,
        "snapshot/stage2_rollout_correction/residual_set/valid_set_mass": 0.4,
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
        "src.trainers.stage2_rollout_runtime.Stage2RolloutRuntime.log",
        _capture_super_log,
    )

    Stage2RolloutCorrectionTrainer.log(trainer, {"loss": 1.0})

    assert captured["loss"] == pytest.approx(1.0)
    assert captured["stage2_rollout_correction/residual_set/type_loss"] == pytest.approx(0.8)
    assert captured["rollout/f1"] == pytest.approx(0.3)
    assert captured["snapshot/stage2_rollout_correction/residual_set/sequence_loss"] == pytest.approx(0.5)
    assert captured["snapshot/stage2_rollout_correction/residual_set/valid_set_mass"] == pytest.approx(0.4)
    assert "snapshot/rollout/f1" not in captured
    assert captured["stage2_rollout_correction/residual_set/type_loss"] == pytest.approx(0.8)
    assert captured["rollout/f1"] == pytest.approx(0.3)
    assert captured["time/sft_total_time"] == pytest.approx(12.0)
    assert captured["time/rollout_total_time"] == pytest.approx(5.0)


def test_stage2_log_reduces_pending_metrics_once_per_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = Stage2RolloutCorrectionTrainer.__new__(Stage2RolloutCorrectionTrainer)
    trainer.state = types.SimpleNamespace(global_step=1)
    trainer._stage2_pending_train_logs = {1: _PendingStage2Log()}
    trainer._stage2_pending_train_logs[1].add(
        {
            "stage2_rollout_correction/residual_set/type_loss": 0.8,
        }
    )
    trainer._stage2_metric_snapshots = {}
    trainer._ddp_assert_all_ranks_true_or_raise = (
        lambda **_kwargs: None
    )  # type: ignore[method-assign]

    reduction_calls: list[dict[str, float]] = []

    def _reduce_pending(metrics):
        reduction_calls.append(dict(metrics))
        return dict(metrics)

    trainer._reduce_stage2_pending_metrics_global = (  # type: ignore[method-assign]
        _reduce_pending
    )
    trainer._stage_wallclock_metrics_local = lambda: {}
    trainer._reduce_stage_wallclock_metrics_global = (  # type: ignore[method-assign]
        lambda metrics: dict(metrics)
    )

    captured: dict[str, float] = {}

    def _capture_super_log(self, logs):
        captured.update(dict(logs))
        return None

    monkeypatch.setattr(
        "src.trainers.stage2_rollout_runtime.Stage2RolloutRuntime.log",
        _capture_super_log,
    )

    Stage2RolloutCorrectionTrainer.log(trainer, {"loss": 1.0})

    assert len(reduction_calls) == 1
    assert reduction_calls[0]["stage2_rollout_correction/residual_set/type_loss"] == pytest.approx(0.8)
    assert trainer._stage2_pending_train_logs == {}
    assert captured["stage2_rollout_correction/residual_set/type_loss"] == pytest.approx(0.8)


def test_stage2_pending_log_preserves_sparse_gradmon_weighting() -> None:
    pending = _PendingStage2Log()
    pending.add(
        {
            "stage2_rollout_correction/residual_set/sequence_loss": 1.0,
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

    assert out["stage2_rollout_correction/residual_set/sequence_loss"] == pytest.approx(0.25)
    assert out["gradmon/neg_cosine_pair_frac"] == pytest.approx(0.75)
    assert out["gradmon/num_terms"] == pytest.approx(4.0)
    assert out["time/gradmon_s"] == pytest.approx(0.2)
    assert out["stage2/_log_weight_total"] == pytest.approx(4.0)
    assert out["gradmon/_log_weight_total"] == pytest.approx(3.0)
