from __future__ import annotations

from typing import Iterable

import torch

from src.trainers.rollout_matching_sft import RolloutMatchingSFTTrainer


def _iter_optimizer_state_tensors(optimizer: torch.optim.Optimizer) -> Iterable[torch.Tensor]:
    state = getattr(optimizer, "state", None)
    if not isinstance(state, dict):
        return []

    out: list[torch.Tensor] = []
    for st in state.values():
        if not isinstance(st, dict):
            continue
        for v in st.values():
            if isinstance(v, torch.Tensor):
                out.append(v)
    return out


def test_rollout_offload_context_smoke_cpu_keeps_state_on_cpu():
    trainer = RolloutMatchingSFTTrainer.__new__(RolloutMatchingSFTTrainer)
    trainer.rollout_matching_cfg = {
        "rollout_backend": "vllm",
        "vllm": {"mode": "colocate"},
        "offload": {"enabled": True, "offload_model": True, "offload_optimizer": True},
    }
    trainer.is_deepspeed_enabled = False
    trainer.accelerator = type(
        "_Acc",
        (),
        {"device": torch.device("cpu"), "unwrap_model": staticmethod(lambda m: m)},
    )()
    trainer.model = torch.nn.Linear(2, 2)
    trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr=1e-3)

    # Touch optimizer state so offload_optimizer code has tensors to move.
    x = torch.randn(4, 2)
    loss = trainer.model(x).sum()
    loss.backward()
    trainer.optimizer.step()
    trainer.optimizer.zero_grad(set_to_none=True)

    assert all(p.device.type == "cpu" for p in trainer.model.parameters())
    assert all(t.device.type == "cpu" for t in _iter_optimizer_state_tensors(trainer.optimizer))

    with trainer._maybe_rollout_offload_context():
        assert all(p.device.type == "cpu" for p in trainer.model.parameters())
        assert all(
            t.device.type == "cpu" for t in _iter_optimizer_state_tensors(trainer.optimizer)
        )

    assert all(p.device.type == "cpu" for p in trainer.model.parameters())
    assert all(t.device.type == "cpu" for t in _iter_optimizer_state_tensors(trainer.optimizer))


def test_rollout_offload_context_rejects_deepspeed():
    trainer = RolloutMatchingSFTTrainer.__new__(RolloutMatchingSFTTrainer)
    trainer.rollout_matching_cfg = {
        "rollout_backend": "vllm",
        "vllm": {"mode": "colocate"},
        "offload": {"enabled": True, "offload_model": True, "offload_optimizer": True},
    }
    trainer.is_deepspeed_enabled = True
    trainer.accelerator = type(
        "_Acc",
        (),
        {"device": torch.device("cpu"), "unwrap_model": staticmethod(lambda m: m)},
    )()
    trainer.model = torch.nn.Linear(2, 2)
    trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr=1e-3)

    try:
        with trainer._maybe_rollout_offload_context():
            pass
    except RuntimeError as exc:
        assert "deepspeed" in str(exc).lower()
    else:
        raise AssertionError("expected rollout offload to fail fast under DeepSpeed")
