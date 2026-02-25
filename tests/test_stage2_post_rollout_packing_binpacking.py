from __future__ import annotations

import builtins

import pytest

from src.trainers.stage2_rollout_aligned import RolloutMatchingSFTTrainer


def _fifo_greedy_selected_total(lengths: list[int], cap: int) -> int:
    assert lengths
    used = int(lengths[0])
    for i in range(1, len(lengths)):
        sl = int(lengths[i])
        if used + sl <= cap:
            used += sl
    return int(used)


def test_post_rollout_selection_deterministic_with_stable_tiebreak():
    cap = 10
    lengths = [6, 3, 2, 2, 2]
    # Oldest=6 (pinned). Residual cap=4 can be filled by any two of the three "2"s:
    # {2,3}, {2,4}, {3,4}. Stable tie-break picks lexicographically-smallest => [2,3].
    expected = [0, 2, 3]
    out0 = RolloutMatchingSFTTrainer._select_post_rollout_segment_indices(lengths, cap)
    assert out0 == expected
    for _ in range(5):
        assert (
            RolloutMatchingSFTTrainer._select_post_rollout_segment_indices(lengths, cap)
            == expected
        )
    assert out0[0] == 0
    assert sum(lengths[i] for i in out0) <= cap


def test_post_rollout_selection_improves_fill_vs_fifo_example():
    cap = 10
    lengths = [6, 3, 2, 2]
    out = RolloutMatchingSFTTrainer._select_post_rollout_segment_indices(lengths, cap)

    fifo_total = _fifo_greedy_selected_total(lengths, cap)
    assert fifo_total == 9

    total = sum(lengths[i] for i in out)
    assert 0 in out
    assert total <= cap
    assert total >= fifo_total
    assert total == cap


def test_post_rollout_buffer_rejects_oversized_segment_on_insertion():
    trainer = RolloutMatchingSFTTrainer.__new__(RolloutMatchingSFTTrainer)
    trainer.rollout_matching_cfg = {
        "packing_enabled": True,
        "packing_length": 10,
        "packing_buffer": 100,
    }
    trainer._post_rollout_segments = []

    segments = [({"input_ids": [0] * 11, "length": 11}, {}, 11)]
    with pytest.raises(ValueError) as excinfo:
        trainer._append_post_rollout_segments(segments)
    msg = str(excinfo.value)
    assert "encoded_len=11" in msg
    assert "packing_length=10" in msg
    assert "Mitigations:" in msg


def test_post_rollout_buffer_overflow_does_not_mutate_state():
    trainer = RolloutMatchingSFTTrainer.__new__(RolloutMatchingSFTTrainer)
    trainer.rollout_matching_cfg = {
        "packing_enabled": True,
        "packing_length": 10,
        "packing_buffer": 1,
    }
    trainer._post_rollout_segments = [({"input_ids": [0] * 5, "length": 5}, {}, 5)]

    segments = [({"input_ids": [0] * 5, "length": 5}, {}, 5)]
    with pytest.raises(ValueError) as excinfo:
        trainer._append_post_rollout_segments(segments)
    assert "buffer overflow" in str(excinfo.value).lower()
    assert len(trainer._post_rollout_segments) == 1


def test_post_rollout_selection_missing_binpacking_fails_fast(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "binpacking":
            raise ImportError("no binpacking")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError) as excinfo:
        RolloutMatchingSFTTrainer._select_post_rollout_segment_indices([6, 3, 2, 2], 10)
    msg = str(excinfo.value).lower()
    assert "binpacking" in msg
    assert "install" in msg
    assert "disable" in msg
