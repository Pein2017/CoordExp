import types

from src.trainers.stage2_ab_training import Stage2ABTrainingTrainer


class _DummyTemplate:
    def __init__(self):
        self.tokenizer = None


def test_stage2_channel_a_does_not_emit_rollout_time_metrics() -> None:
    # Use __new__ to avoid heavy Trainer initialization; this test only exercises
    # Channel-A metric key emission (no model/encode required).
    t = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)

    t.template = _DummyTemplate()

    # Channel-A only consults these knobs before iterating over inputs.
    t._get_coord_token_ids = types.MethodType(lambda self: [], t)  # type: ignore[attr-defined]
    t._packing_enabled = types.MethodType(lambda self: False, t)  # type: ignore[attr-defined]

    segments, metrics = t._prepare_batch_inputs_a([], _segments_only=True)

    assert segments == []
    assert metrics["stage2/channel_a"] == 1.0
    assert metrics["stage2/channel_b"] == 0.0
    assert "time/channel_a_teacher_encode_s" in metrics

    # Rollout timings are Channel-B-only; emitting them on Channel-A creates
    # confusing 0-valued TB curves and hides true bottlenecks.
    for key in (
        "time/rollout_generate_s",
        "time/rollout_parse_match_s",
        "time/rollout_teacher_encode_s",
        "time/post_rollout_pack_s",
    ):
        assert key not in metrics
