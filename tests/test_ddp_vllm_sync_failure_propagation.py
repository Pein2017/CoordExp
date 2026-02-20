from __future__ import annotations

import types
from typing import Any

import pytest
import torch

from src.trainers.rollout_matching_sft import RolloutMatchingSFTTrainer


class _FakeDist:
    """Single-process fake dist backend to simulate multi-rank control-flow.

    We call the target method once per "rank" sequentially and emulate the
    broadcast/barrier behavior with shared in-memory state.
    """

    def __init__(self, *, world_size: int = 2, backend: str = "gloo") -> None:
        self._world_size = int(world_size)
        self._backend = str(backend)
        self.current_rank = 0
        self.barriers = 0
        self.last_broadcast_flag: int | None = None
        self.last_broadcast_msg: str | None = None

    def is_available(self) -> bool:  # noqa: D401 - match torch.distributed API
        return True

    def is_initialized(self) -> bool:
        return True

    def get_world_size(self) -> int:
        return self._world_size

    def get_rank(self) -> int:
        return int(self.current_rank)

    def get_backend(self) -> str:
        return self._backend

    def barrier(self) -> None:
        self.barriers += 1

    def broadcast(self, tensor: torch.Tensor, src: int) -> None:
        assert tensor.numel() >= 1
        if int(self.current_rank) == int(src):
            self.last_broadcast_flag = int(tensor.view(-1)[0].item())
        else:
            assert self.last_broadcast_flag is not None
            tensor.view(-1)[0] = int(self.last_broadcast_flag)

    def broadcast_object_list(
        self,
        object_list: list[Any],
        src: int = 0,
        group: Any = None,
        device: torch.device | None = None,
        group_src: int | None = None,
    ) -> None:
        del group, device, group_src
        assert isinstance(object_list, list) and len(object_list) >= 1
        if int(self.current_rank) == int(src):
            self.last_broadcast_msg = str(object_list[0])
        else:
            assert self.last_broadcast_msg is not None
            object_list[0] = self.last_broadcast_msg


def _mk_min_trainer() -> RolloutMatchingSFTTrainer:
    t = RolloutMatchingSFTTrainer.__new__(RolloutMatchingSFTTrainer)
    t.state = types.SimpleNamespace(global_step=0)
    t.model = types.SimpleNamespace(device=torch.device("cpu"))
    t.rollout_matching_cfg = {
        "vllm": {
            "enable_lora": False,
            "sync": {"mode": "full", "fallback_to_full": True},
        }
    }
    return t


def test_ddp_vllm_weight_sync_failure_is_broadcast_and_raises_on_all_ranks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeDist(world_size=2, backend="gloo")

    # Patch the torch.distributed module functions the trainer consults.
    monkeypatch.setattr(torch.distributed, "is_available", fake.is_available, raising=False)
    monkeypatch.setattr(torch.distributed, "is_initialized", fake.is_initialized, raising=False)
    monkeypatch.setattr(torch.distributed, "get_world_size", fake.get_world_size, raising=False)
    monkeypatch.setattr(torch.distributed, "get_rank", fake.get_rank, raising=False)
    monkeypatch.setattr(torch.distributed, "get_backend", fake.get_backend, raising=False)
    monkeypatch.setattr(torch.distributed, "barrier", fake.barrier, raising=False)
    monkeypatch.setattr(torch.distributed, "broadcast", fake.broadcast, raising=False)
    monkeypatch.setattr(
        torch.distributed, "broadcast_object_list", fake.broadcast_object_list, raising=False
    )

    t0 = _mk_min_trainer()
    t1 = _mk_min_trainer()

    # Rank0 performs the sync and fails.
    t0._ensure_vllm_server_client = lambda: object()  # type: ignore[attr-defined]
    t0._ensure_vllm_server_communicator_rank0 = lambda client: None  # type: ignore[attr-defined]

    def _fail(client: Any) -> None:
        raise RuntimeError("boom")

    t0._sync_vllm_server_full_weights = _fail  # type: ignore[attr-defined]

    fake.current_rank = 0
    with pytest.raises(RuntimeError, match=r"boom"):
        t0._sync_vllm_server_rollout_model_if_needed()

    assert fake.last_broadcast_flag == 1
    assert fake.last_broadcast_msg is not None and "boom" in fake.last_broadcast_msg

    # Non-rank0 must also raise deterministically after the broadcast.
    fake.current_rank = 1
    with pytest.raises(RuntimeError, match=r"boom"):
        t1._sync_vllm_server_rollout_model_if_needed()

