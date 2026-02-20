from __future__ import annotations

import types

import pytest

from src.trainers.rollout_matching_sft import RolloutMatchingSFTTrainer


def test_normalize_rollout_seed_int32_maps_zero_to_one() -> None:
    fn = RolloutMatchingSFTTrainer._normalize_rollout_seed_int32
    assert fn(0) == 1
    assert fn(1) == 1
    assert fn(-1) == 0x7FFFFFFF


def test_derive_rollout_seed_base_never_returns_zero() -> None:
    t = RolloutMatchingSFTTrainer.__new__(RolloutMatchingSFTTrainer)
    t.args = types.SimpleNamespace(seed=0)

    s0 = t._derive_rollout_seed_base(global_step=0)
    s1 = t._derive_rollout_seed_base(global_step=1)

    assert s0 != 0
    assert s1 != 0
    assert s0 == 1  # seed=0 + gs=0 canonicalizes to non-zero

    # Stability: changing global_step should change the seed deterministically.
    assert s1 != s0

