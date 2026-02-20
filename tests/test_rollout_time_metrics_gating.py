from src.trainers.rollout_matching_sft import (
    RolloutMatchingSFTTrainer,
    _PendingTrainRolloutLog,
)


def test_rollout_time_metrics_only_present_when_rollout_ran() -> None:
    t = object.__new__(RolloutMatchingSFTTrainer)
    t.rollout_matching_cfg = {}

    pending = _PendingTrainRolloutLog()
    pending.meta = []
    pending.n_micro = 1

    pending.time_forward_s = 1.0
    pending.time_mask_build_s = 2.0

    # No rollout timings recorded.
    pending.time_rollout_generate_s = 0.0
    pending.time_rollout_parse_match_s = 0.0
    pending.time_rollout_teacher_encode_s = 0.0

    out = t._build_train_rollout_log_payload(pending)

    assert out["time/forward_s"] == 1.0
    assert out["time/mask_build_s"] == 2.0
    assert "time/rollout_generate_s" not in out
    assert "time/rollout_parse_match_s" not in out
    assert "time/rollout_teacher_encode_s" not in out

    # Now simulate a rollout step.
    pending.time_rollout_generate_s = 3.0
    out2 = t._build_train_rollout_log_payload(pending)

    assert out2["time/rollout_generate_s"] == 3.0
    assert out2["time/rollout_parse_match_s"] == 0.0
    assert out2["time/rollout_teacher_encode_s"] == 0.0
