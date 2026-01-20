import pytest

from src.trainers.rollout_matching_sft import (
    RolloutMatchingSFTTrainer,
    schedule_post_rollout_segment_indices_window,
)


def _select(encoded_lens, packing_length):
    return RolloutMatchingSFTTrainer._select_post_rollout_segment_indices(
        encoded_lens=encoded_lens,
        packing_length=packing_length,
    )


def test_window_schedule_deterministic():
    lens = [3, 4, 5, 6, 2, 8]
    packs1 = schedule_post_rollout_segment_indices_window(
        encoded_lens=lens,
        packing_length=10,
        gas=3,
        select_indices_fn=_select,
    )
    packs2 = schedule_post_rollout_segment_indices_window(
        encoded_lens=lens,
        packing_length=10,
        gas=3,
        select_indices_fn=_select,
    )
    assert packs1 == packs2


def test_window_schedule_infeasible_fails_fast():
    lens = [9, 9, 9]
    with pytest.raises(ValueError):
        schedule_post_rollout_segment_indices_window(
            encoded_lens=lens,
            packing_length=10,
            gas=2,
            select_indices_fn=_select,
        )


def test_window_schedule_no_drop_schedules_all():
    lens = [7, 3, 7, 3]
    packs = schedule_post_rollout_segment_indices_window(
        encoded_lens=lens,
        packing_length=10,
        gas=2,
        select_indices_fn=_select,
    )
    scheduled = sorted(i for pack in packs for i in pack)
    assert scheduled == [0, 1, 2, 3]
