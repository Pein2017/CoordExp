from __future__ import annotations

import torch

from src.trainers.rollout_matching_sft import (
    _AccumulationWindowRepeater,
    _RolloutWindowBuffer,
    RolloutMatchingSFTTrainer,
)


def test_accumulation_window_repeater_repeats_full_windows_only():
    # gas=3, m_steps=2 => A,B,C, A,B,C, D (partial window, no repeat)
    inner = ["A", "B", "C", "D"]
    wrapped = _AccumulationWindowRepeater(inner, gas=3, m_steps=2)
    assert list(iter(wrapped)) == ["A", "B", "C", "A", "B", "C", "D"]
    assert len(wrapped) == 7


def test_rollout_window_buffer_copy_on_reuse_avoids_mutation_footgun():
    buf = _RolloutWindowBuffer(gas=1, m_steps=2)

    # E-step micro-step at global_step=0.
    buf.on_micro_step_start(global_step=0)

    prepared_calls = {"n": 0}

    def build_prepared():
        prepared_calls["n"] += 1
        return {"input_ids": torch.tensor([1, 2, 3]), "_rollout_matching_meta": [{"k": "v"}]}

    batch0, reused0 = buf.select_batch(global_step=0, raw_fp="fp0", build_prepared=build_prepared)
    assert reused0 is False
    assert prepared_calls["n"] == 1
    assert "_rollout_matching_meta" in batch0
    # First window step should preserve batch contents (tensors are shared; dict wrapper differs).
    assert batch0["input_ids"] is buf.cached_batches[0]["input_ids"]

    # Simulate HF/Swift mutation (compute_loss pop).
    batch0.pop("_rollout_matching_meta", None)
    assert "_rollout_matching_meta" in buf.cached_batches[0]

    # Optimizer step boundary (global_step increments).
    buf.on_micro_step_start(global_step=1)

    def should_not_build():
        raise AssertionError("build_prepared must not be called on M-step reuse")

    batch1, reused1 = buf.select_batch(global_step=1, raw_fp="fp0", build_prepared=should_not_build)
    assert reused1 is True
    assert "_rollout_matching_meta" in batch1


def test_rollout_offload_context_smoke_cpu():
    # Construct a minimal trainer instance without running the heavy base-class __init__.
    trainer = RolloutMatchingSFTTrainer.__new__(RolloutMatchingSFTTrainer)
    trainer.rollout_matching_cfg = {
        "rollout_backend": "vllm",
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

    # Touch optimizer state so offload_optimizer code has something to move.
    x = torch.randn(4, 2)
    loss = trainer.model(x).sum()
    loss.backward()
    trainer.optimizer.step()
    trainer.optimizer.zero_grad(set_to_none=True)

    with trainer._maybe_rollout_offload_context():
        pass
