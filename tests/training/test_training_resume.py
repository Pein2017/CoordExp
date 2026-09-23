from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.common.errors import RuntimeContractError
from src.artifacts.checkpoints import _gather_training_states
from src.training.resume import (
    build_training_state,
    load_training_state,
    restore_runtime_state,
    save_training_state,
    validate_resume_state,
)


def _identity_bundle() -> tuple[dict[str, object], ...]:
    return (
        {"packs_per_epoch": 3, "runtime_batch": {"effective_batch_size": 1}},
        {"base": "model", "tokenizer": "tok"},
        {"cache": "data", "order": "seeded_shuffle", "seed": 1729},
        {"groups": ["adapter.language"], "scheduler": "constant"},
    )


def _run_steps(model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: object, start: int, stop: int) -> None:
    for step in range(start, stop):
        optimizer.zero_grad(set_to_none=True)
        loss = (model(torch.tensor([[float(step + 1)]])) - 1.0).square().sum()
        loss.backward()
        optimizer.step()
        scheduler.step()


def test_ddp_checkpoint_gathers_only_rank_rng_state() -> None:
    seen: list[object] = []

    class Accelerator:
        num_processes = 2

        def gather_object(self, value: object) -> list[dict[str, object]]:
            seen.append(value)
            return [{"rng": {"rank": 0}}, {"rng": {"rank": 1}}]

    gathered = _gather_training_states(
        Accelerator(),
        {"rng": {"rank": 0}, "optimizer": {"large": torch.ones(4)}, "scheduler": {"x": 1}},
    )
    assert gathered == [{"rng": {"rank": 0}}, {"rng": {"rank": 1}}]
    assert seen == [[{"rng": {"rank": 0}}]]


def test_split_run_restores_optimizer_scheduler_rng_and_matches_continuation(tmp_path) -> None:
    torch.manual_seed(1729)
    uninterrupted = torch.nn.Linear(1, 1)
    split = torch.nn.Linear(1, 1)
    split.load_state_dict(uninterrupted.state_dict())
    opt_full = torch.optim.AdamW(uninterrupted.parameters(), lr=0.01)
    opt_split = torch.optim.AdamW(split.parameters(), lr=0.01)
    sch_full = torch.optim.lr_scheduler.ConstantLR(opt_full, factor=1.0, total_iters=1)
    sch_split = torch.optim.lr_scheduler.ConstantLR(opt_split, factor=1.0, total_iters=1)
    runtime_split = SimpleNamespace(optimizer=opt_split, scheduler=sch_split)
    schedule, source, data, optimizer = _identity_bundle()

    _run_steps(uninterrupted, opt_full, sch_full, 0, 4)
    _run_steps(split, opt_split, sch_split, 0, 2)
    state = build_training_state(
        step=2,
        schedule={**schedule, "resolved_max_steps": 4},
        source_identity=source,
        data_identity=data,
        optimizer_identity=optimizer,
        runtime=runtime_split,
    )
    state["rank_rngs"] = [{"rng": state["rng"]}]
    state["model_payloads"] = {"adapter/model.safetensors": "fixture"}
    path = tmp_path / "training_state.pt"
    save_training_state(path, state)

    resumed = torch.nn.Linear(1, 1)
    resumed.load_state_dict(split.state_dict())
    opt_resumed = torch.optim.AdamW(resumed.parameters(), lr=0.01)
    sch_resumed = torch.optim.lr_scheduler.ConstantLR(opt_resumed, factor=1.0, total_iters=1)
    runtime_resumed = SimpleNamespace(
        optimizer=opt_resumed,
        scheduler=sch_resumed,
        optimizer_step_count=0,
        scheduler_step_count=0,
    )
    loaded = load_training_state(path)
    assert validate_resume_state(
        loaded,
        schedule={**schedule, "resolved_max_steps": 4},
        source_identity=source,
        data_identity=data,
        optimizer_identity=optimizer,
    ) == 2
    restore_runtime_state(runtime_resumed, loaded)
    _run_steps(resumed, opt_resumed, sch_resumed, 2, 4)

    assert runtime_resumed.optimizer_step_count == 2
    assert all(torch.equal(a, b) for a, b in zip(uninterrupted.parameters(), resumed.parameters(), strict=True))


def test_resume_fails_closed_for_missing_or_incompatible_state(tmp_path) -> None:
    schedule, source, data, optimizer = _identity_bundle()
    state = {
        "schema": 1,
        "step": 1,
        "schedule": {**schedule, "resolved_max_steps": 2},
        "source_identity": source,
        "data_identity": data,
        "optimizer_identity": optimizer,
        "optimizer": {},
        "scheduler": {},
        "rng": {},
        "rank_rngs": [{"rng": {}}],
        "model_payloads": {"adapter/model.safetensors": "fixture"},
    }
    path = tmp_path / "training_state.pt"
    save_training_state(path, state)
    loaded = load_training_state(path)
    with pytest.raises(RuntimeContractError, match="resume state is incompatible") as exc_info:
        validate_resume_state(
            loaded,
            schedule={**schedule, "resolved_max_steps": 2},
            source_identity={**source, "base": "different"},
            data_identity=data,
            optimizer_identity=optimizer,
        )
    assert exc_info.value.code == "training.resume_incompatible"

    del loaded["rng"]
    with pytest.raises(RuntimeContractError, match="resume state is incompatible"):
        validate_resume_state(
            loaded,
            schedule={**schedule, "resolved_max_steps": 2},
            source_identity=source,
            data_identity=data,
            optimizer_identity=optimizer,
        )
