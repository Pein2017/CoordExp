from __future__ import annotations

import math

from src.trainers.stage2_two_channel import Stage2ABTrainingTrainer


def _count_b_steps(*, b_ratio: float, steps: int) -> int:
    t = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)
    t._stage2_channel_override = None
    t.stage2_ab_cfg = {"schedule": {"b_ratio": float(b_ratio)}, "channel_b": {}}
    n_b = 0
    for s in range(int(steps)):
        n_b += 1 if t._stage2_channel_for_step(int(s)) == "B" else 0
    return int(n_b)


def test_bresenham_schedule_matches_exact_ratios_for_binary_floats() -> None:
    # 7/8 is exactly representable in binary floating point.
    steps = 1024
    b_ratio = 0.875
    n_b = _count_b_steps(b_ratio=b_ratio, steps=steps)
    assert n_b == int(math.floor(float(steps) * float(b_ratio)))


def test_bresenham_schedule_tracks_target_ratio_over_long_horizon() -> None:
    steps = 1000
    b_ratio = 0.85
    n_b = _count_b_steps(b_ratio=b_ratio, steps=steps)
    realized = float(n_b) / float(steps)

    # Allow <= 2/steps deviation to tolerate float-boundary effects while still
    # catching obvious schedule regressions.
    assert abs(realized - float(b_ratio)) <= (2.0 / float(steps))

