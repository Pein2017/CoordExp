from __future__ import annotations

from collections import defaultdict
import types

import pytest
import torch

from src.trainers.metrics.mixins import AggregateTokenTypeMetricsMixin
from src.trainers.stage2_rollout_aligned import RolloutMatchingSFTTrainer
from src.trainers.stage2_two_channel import Stage2ABTrainingTrainer


def _mk_min_stage2_trainer() -> Stage2ABTrainingTrainer:
    t = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)
    t.model = types.SimpleNamespace(device=torch.device("cpu"))
    return t


def _mk_min_rollout_trainer() -> RolloutMatchingSFTTrainer:
    t = RolloutMatchingSFTTrainer.__new__(RolloutMatchingSFTTrainer)
    t.model = types.SimpleNamespace(device=torch.device("cpu"))
    return t


def test_stage2_pending_metric_key_sync_failure_is_not_silently_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True, raising=False)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True, raising=False)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2, raising=False)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0, raising=False)
    monkeypatch.setattr(torch.distributed, "all_gather_object", _boom, raising=False)

    t = _mk_min_stage2_trainer()
    with pytest.raises(RuntimeError, match=r"boom"):
        t._reduce_stage2_pending_metrics_global({"stage2/raw_rollouts": 1.0})


def test_stage2_pending_metric_all_reduce_failure_is_fail_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _all_gather_object(out: list[object], local: object) -> None:
        del local
        assert len(out) == 2
        out[0] = ["stage2/raw_rollouts"]
        out[1] = ["stage2/raw_rollouts"]

    def _all_reduce(_tensor: torch.Tensor, op: object) -> None:
        del op
        raise RuntimeError("boom")

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True, raising=False)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True, raising=False)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2, raising=False)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1, raising=False)
    monkeypatch.setattr(torch.distributed, "all_gather_object", _all_gather_object, raising=False)
    monkeypatch.setattr(torch.distributed, "all_reduce", _all_reduce, raising=False)

    t = _mk_min_stage2_trainer()
    with pytest.raises(RuntimeError, match=r"stage2-ab metric all-reduce failed"):
        t._reduce_stage2_pending_metrics_global({"stage2/raw_rollouts": 1.0})


def test_rollout_metric_key_sync_failure_is_not_silently_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True, raising=False)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True, raising=False)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2, raising=False)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0, raising=False)
    monkeypatch.setattr(torch.distributed, "all_gather_object", _boom, raising=False)

    t = _mk_min_rollout_trainer()
    with pytest.raises(RuntimeError, match=r"boom"):
        t._reduce_train_rollout_log_payload_global({"train/samples_total": 1.0})


def test_rollout_metric_all_reduce_failure_is_fail_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _all_gather_object(out: list[object], local: object) -> None:
        del local
        assert len(out) == 2
        out[0] = ["train/samples_total"]
        out[1] = ["train/samples_total"]

    def _all_reduce(_tensor: torch.Tensor, op: object) -> None:
        del op
        raise RuntimeError("boom")

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True, raising=False)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True, raising=False)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2, raising=False)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1, raising=False)
    monkeypatch.setattr(torch.distributed, "all_gather_object", _all_gather_object, raising=False)
    monkeypatch.setattr(torch.distributed, "all_reduce", _all_reduce, raising=False)

    t = _mk_min_rollout_trainer()
    with pytest.raises(RuntimeError, match=r"rollout metric all-reduce failed"):
        t._reduce_train_rollout_log_payload_global({"train/samples_total": 1.0})


def test_dataset_metric_key_sync_is_rank_symmetric_under_ddp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ReduceOp:
        SUM = "sum"
        MAX = "max"

    calls: dict[str, int] = {"all_reduce": 0, "all_gather_object": 0}

    def _all_reduce(tensor: torch.Tensor, op: object) -> None:
        calls["all_reduce"] += 1
        if op == _ReduceOp.SUM:
            tensor.view(-1)[0] = 2
        elif op == _ReduceOp.MAX:
            tensor.view(-1)[0] = 1
        else:
            raise AssertionError(f"unexpected op: {op!r}")

    def _all_gather_object(out: list[object], local: object) -> None:
        calls["all_gather_object"] += 1
        assert len(out) == 2
        assert isinstance(local, list)
        out[0] = list(local)
        out[1] = ["new_key"]

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True, raising=False)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True, raising=False)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2, raising=False)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0, raising=False)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda: "gloo", raising=False)
    monkeypatch.setattr(torch.distributed, "ReduceOp", _ReduceOp, raising=False)
    monkeypatch.setattr(torch.distributed, "all_reduce", _all_reduce, raising=False)
    monkeypatch.setattr(torch.distributed, "all_gather_object", _all_gather_object, raising=False)

    class _Dummy(AggregateTokenTypeMetricsMixin):
        pass

    t = _Dummy.__new__(_Dummy)
    t.model = types.SimpleNamespace(training=True)
    t.custom_metrics = {"train": defaultdict(float, {"existing": 1.0})}
    # Local cache already contains local keys; the global MAX-reduced flag forces a sync anyway.
    t._dataset_metric_key_cache = {"train": set(t.custom_metrics["train"].keys())}

    t._sync_dataset_metrics()

    metrics = t.custom_metrics["train"]
    assert "new_key" in metrics
    assert calls["all_reduce"] >= 2
    assert calls["all_gather_object"] == 1
