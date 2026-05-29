from __future__ import annotations

import pytest

from src.trainers.stage2_rollout_correction import Stage2RolloutCorrectionTrainer


def test_stage2_rollout_correction_post_rollout_packing_buffer_rejects_ab_channels() -> None:
    trainer = Stage2RolloutCorrectionTrainer.__new__(Stage2RolloutCorrectionTrainer)

    # Provide the minimal rollout_matching_cfg needed by packing helpers.
    trainer.rollout_matching_cfg = {
        "packing_enabled": True,
        "packing_length": 10,
        "packing_buffer": 100,
        "packing_min_fill_ratio": 0.5,
        "packing_drop_last": False,
    }

    seg0 = ({"input_ids": [0] * 6, "length": 6}, {"id": "rc0"}, 6)
    seg1 = ({"input_ids": [0] * 3, "length": 3}, {"id": "rc1"}, 3)

    with pytest.raises(ValueError, match="no A/B channels"):
        trainer._stage2_append_post_rollout_segments(channel="A", segments=[seg0])

    trainer._stage2_append_post_rollout_segments(
        channel="rollout_correction",
        segments=[seg0, seg1],
    )
    buf = trainer._stage2_post_rollout_buffer(channel="rollout_correction")
    assert [m.get("id") for _, m, _ in buf] == ["rc0", "rc1"]

    selected, _pm = trainer._stage2_pop_post_rollout_pack(channel="rollout_correction")
    assert [m.get("id") for _, m, _ in selected] == ["rc0", "rc1"]


def test_stage2_rollout_correction_post_rollout_pack_selector_passes_fill_target() -> None:
    trainer = Stage2RolloutCorrectionTrainer.__new__(Stage2RolloutCorrectionTrainer)
    trainer.rollout_matching_cfg = {
        "packing_enabled": True,
        "packing_length": 10,
        "packing_buffer": 100,
        "packing_min_fill_ratio": 0.5,
        "packing_drop_last": True,
    }

    captured: dict[str, object] = {}

    def _selector(
        encoded_lens,
        packing_length,
        *,
        min_fill_ratio=None,
    ):
        captured["min_fill_ratio"] = float(min_fill_ratio)
        assert list(encoded_lens) == [6, 3]
        assert int(packing_length) == 10
        return [0, 1]

    trainer._select_post_rollout_segment_indices = _selector

    seg_a0 = ({"input_ids": [0] * 6, "length": 6}, {"id": "rc0"}, 6)
    seg_a1 = ({"input_ids": [0] * 3, "length": 3}, {"id": "rc1"}, 3)
    trainer._stage2_append_post_rollout_segments(
        channel="rollout_correction",
        segments=[seg_a0, seg_a1],
    )

    selected_a, _pm_a = trainer._stage2_pop_post_rollout_pack(
        channel="rollout_correction"
    )
    assert [m.get("id") for _, m, _ in selected_a] == ["rc0", "rc1"]
    assert captured["min_fill_ratio"] == 0.5
