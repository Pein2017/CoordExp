
import pytest

from src.trainers.stage2_rollout_aligned import RolloutMatchingSFTTrainer, _PendingTrainRolloutLog


def test_a_only_stage2_does_not_emit_rollout_monitors_when_no_rollout_ran() -> None:
    """Regression: A-only Stage-2 should not emit rollout-only keys as constant 0.0.

    In Channel-A-only training, we still buffer PendingTrainRolloutLog for forward/packing
    telemetry, but we must not pretend a rollout executed.
    """

    trainer = object.__new__(RolloutMatchingSFTTrainer)
    trainer.rollout_matching_cfg = {}
    trainer._cfg = lambda _k, default=None: default

    pending = _PendingTrainRolloutLog()
    pending.n_micro = 1
    pending.meta = [
        {
            # This can be non-"none" for Stage-2 teacher forcing; it must NOT trigger
            # rollout monitors unless the rollout pipeline actually ran.
            "decode_mode": "exp",
            "rollout_len": 0,
            "gt_objects": 2,
            "matched_for_supervision": 1,
            "valid_pred_objects": 3,
            "excluded_from_supervision": 0,
        }
    ]

    pending.time_forward_s = 1.0
    pending.time_mask_build_s = 0.0
    pending.time_rollout_generate_s = 0.0
    pending.time_rollout_parse_match_s = 0.0
    pending.time_rollout_teacher_encode_s = 0.0

    out = trainer._build_train_rollout_log_payload(pending)

    assert out["time/forward_s"] == pytest.approx(1.0)
    assert "time/mask_build_s" not in out

    for key in ("rollout/precision", "rollout/recall", "rollout/f1"):
        assert key not in out
