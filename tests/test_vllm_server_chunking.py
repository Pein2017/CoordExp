from __future__ import annotations

import pytest

from src.trainers.rollout_matching_sft import (
    RolloutMatchingSFTTrainer,
    _contiguous_chunk_slices,
)


def _mk_uninit_trainer(cfg):
    t = RolloutMatchingSFTTrainer.__new__(RolloutMatchingSFTTrainer)
    t.rollout_matching_cfg = cfg
    return t


def test_contiguous_chunk_slices_n0_is_all_empty():
    assert _contiguous_chunk_slices(0, 3) == [(0, 0), (0, 0), (0, 0)]


def test_contiguous_chunk_slices_matches_ceil_contract():
    # N=5, S=2 -> chunk_size=ceil(5/2)=3
    assert _contiguous_chunk_slices(5, 2) == [(0, 3), (3, 5)]

    # N=5, S=3 -> chunk_size=ceil(5/3)=2
    assert _contiguous_chunk_slices(5, 3) == [(0, 2), (2, 4), (4, 5)]


def test_contiguous_chunk_slices_preserves_order_on_reassembly():
    items = list(range(10))
    slices = _contiguous_chunk_slices(len(items), 4)

    # Simulate per-server processing that preserves per-chunk order.
    per_server_out = [items[s:e] for s, e in slices if s < e]
    reassembled = [x for chunk in per_server_out for x in chunk]

    assert reassembled == items


def test_vllm_server_specs_rejects_removed_legacy_base_url_group_port_fields():
    t = _mk_uninit_trainer(
        {
            "vllm": {
                "server": {
                    "servers": [
                        {"base_url": "http://127.0.0.1:8000", "group_port": 51216}
                    ],
                    "base_url": "http://127.0.0.1:8001",
                    "group_port": 51217,
                }
            }
        }
    )
    with pytest.raises(ValueError, match="Legacy rollout server config has been removed"):
        t._vllm_server_specs()


def test_vllm_server_timeouts_rejects_zero_timeout_s():
    t = _mk_uninit_trainer({"vllm": {"server": {"timeout_s": 0}}})
    with pytest.raises(ValueError, match="timeout_s must be > 0"):
        t._vllm_server_timeouts()
