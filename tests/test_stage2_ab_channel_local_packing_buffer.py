from __future__ import annotations

from src.trainers.stage2_ab_training import Stage2ABTrainingTrainer


def test_stage2_ab_post_rollout_packing_buffers_are_channel_local() -> None:
    trainer = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)

    # Provide the minimal rollout_matching_cfg needed by packing helpers.
    trainer.rollout_matching_cfg = {
        "packing_enabled": True,
        "packing_length": 10,
        "packing_buffer": 100,
        "packing_min_fill_ratio": 0.5,
        "packing_drop_last": False,
    }

    # A gets two segments, B gets one.
    seg_a0 = ({"input_ids": [0] * 6, "length": 6}, {"id": "a0"}, 6)
    seg_a1 = ({"input_ids": [0] * 3, "length": 3}, {"id": "a1"}, 3)
    seg_b0 = ({"input_ids": [0] * 4, "length": 4}, {"id": "b0"}, 4)

    trainer._stage2_append_post_rollout_segments(channel="A", segments=[seg_a0, seg_a1])
    trainer._stage2_append_post_rollout_segments(channel="B", segments=[seg_b0])

    buf_a = trainer._stage2_post_rollout_buffer(channel="A")
    buf_b = trainer._stage2_post_rollout_buffer(channel="B")
    assert [m.get("id") for _, m, _ in buf_a] == ["a0", "a1"]
    assert [m.get("id") for _, m, _ in buf_b] == ["b0"]

    selected_a, _pm_a = trainer._stage2_pop_post_rollout_pack(channel="A")
    # Packing selection must not consume B.
    assert all(m.get("id").startswith("a") for _, m, _ in selected_a)
    assert [m.get("id") for _, m, _ in trainer._stage2_post_rollout_buffer(channel="B")] == [
        "b0"
    ]
