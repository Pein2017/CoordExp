import types

from src.trainers.stage2_rollout_correction import Stage2RolloutCorrectionTrainer


class _DummyTemplate:
    def __init__(self):
        self.tokenizer = None


def test_stage2_rollout_correction_does_not_emit_rollout_time_metrics() -> None:
    # Use __new__ to avoid heavy Trainer initialization; this test only exercises
    # rollout-correction metric key emission (no model/encode required).
    t = Stage2RolloutCorrectionTrainer.__new__(Stage2RolloutCorrectionTrainer)

    t.template = _DummyTemplate()

    # rollout-correction only consults these knobs before iterating over inputs.
    t._get_coord_token_ids = types.MethodType(lambda self: [], t)  # type: ignore[attr-defined]
    t._packing_enabled = types.MethodType(lambda self: False, t)  # type: ignore[attr-defined]

    segments, metrics = t._prepare_rollout_correction_inputs([], _segments_only=True)

    assert segments == []
    assert metrics["stage2/rollout_correction"] == 1.0
    assert metrics["stage2/rollout_correction"] == 0.0
    assert "time/rollout_correction_teacher_encode_s" in metrics

    # Rollout timings are rollout-correction-only; emitting them on rollout-correction creates
    # confusing 0-valued TB curves and hides true bottlenecks.
    for key in (
        "time/rollout_generate_s",
        "time/rollout_parse_match_s",
        "time/rollout_teacher_encode_s",
        "time/post_rollout_pack_s",
    ):
        assert key not in metrics
