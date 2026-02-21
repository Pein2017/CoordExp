from __future__ import annotations

import types

import pytest
import torch

from src.trainers.rollout_matching_sft import RolloutMatchingSFTTrainer
from src.trainers.stage2_ab_training import Stage2ABTrainingTrainer


def _mk_min_rm_trainer() -> RolloutMatchingSFTTrainer:
    t = RolloutMatchingSFTTrainer.__new__(RolloutMatchingSFTTrainer)
    t.model = types.SimpleNamespace(device=torch.device("cpu"))
    t.state = types.SimpleNamespace(global_step=7)
    return t


def test_ddp_guard_raises_when_not_all_ranks_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    class _ReduceOp:
        SUM = "sum"

    forced_sum = 1

    def _all_reduce(tensor: torch.Tensor, op: object) -> None:
        del op
        tensor.view(-1)[0] = int(forced_sum)

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True, raising=False)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True, raising=False)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2, raising=False)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0, raising=False)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda: "gloo", raising=False)
    monkeypatch.setattr(torch.distributed, "all_reduce", _all_reduce, raising=False)
    monkeypatch.setattr(torch.distributed, "ReduceOp", _ReduceOp, raising=False)

    t = _mk_min_rm_trainer()
    with pytest.raises(RuntimeError, match=r"readiness mismatch"):
        t._ddp_assert_all_ranks_true_or_raise(
            where="unit_test",
            local_true=True,
            global_step=7,
        )


def test_ddp_guard_allows_when_all_ranks_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    class _ReduceOp:
        SUM = "sum"

    forced_sum = 2

    def _all_reduce(tensor: torch.Tensor, op: object) -> None:
        del op
        tensor.view(-1)[0] = int(forced_sum)

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True, raising=False)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True, raising=False)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2, raising=False)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1, raising=False)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda: "gloo", raising=False)
    monkeypatch.setattr(torch.distributed, "all_reduce", _all_reduce, raising=False)
    monkeypatch.setattr(torch.distributed, "ReduceOp", _ReduceOp, raising=False)

    t = _mk_min_rm_trainer()
    t._ddp_assert_all_ranks_true_or_raise(
        where="unit_test",
        local_true=True,
        global_step=7,
    )


def test_rollout_matching_training_step_rejects_empty_raw_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True, raising=False)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True, raising=False)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2, raising=False)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0, raising=False)

    t = _mk_min_rm_trainer()

    with pytest.raises(ValueError, match=r"empty raw batch"):
        t.training_step(model=object(), inputs=[])


def test_stage2_ab_training_step_rejects_empty_raw_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True, raising=False)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True, raising=False)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2, raising=False)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1, raising=False)

    t = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)
    t.state = types.SimpleNamespace(global_step=3)

    with pytest.raises(ValueError, match=r"empty raw batch"):
        t.training_step(model=object(), inputs=[])
