from __future__ import annotations

from types import SimpleNamespace
import subprocess

from src.artifacts.git_identity import capture_source_identity

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


@pytest.fixture
def qualified_source(tmp_path):
    root = tmp_path / "source-repo"
    root.mkdir()
    for args in (("init", "-q"), ("config", "user.name", "Fixture"),
                 ("config", "user.email", "fixture@example.invalid")):
        subprocess.run(["git", "-C", str(root), *args], check=True)
    (root / "source.py").write_text("VALUE = 1\n")
    subprocess.run(["git", "-C", str(root), "add", "source.py"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "Fixture"], check=True)
    return root, capture_source_identity(["source.py"], root=root)


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


def test_split_run_restores_optimizer_scheduler_rng_and_matches_continuation(tmp_path, qualified_source) -> None:
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
    source_root, identity = qualified_source
    source["execution_source"] = identity

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
    save_training_state(path, state, source_root=source_root, required_paths=("source.py",))

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
    loaded = load_training_state(path, source_root=source_root, required_paths=("source.py",))
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


def test_resume_fails_closed_for_missing_or_incompatible_state(tmp_path, qualified_source) -> None:
    schedule, source, data, optimizer = _identity_bundle()
    source_root, identity = qualified_source
    source["execution_source"] = identity
    state = {
        "schema": 2,
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
    save_training_state(path, state, source_root=source_root, required_paths=("source.py",))
    loaded = load_training_state(path, source_root=source_root, required_paths=("source.py",))
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


def test_legacy_missing_source_gate_never_deserializes(tmp_path, monkeypatch):
    path = tmp_path / "training_state.pt"
    path.write_bytes(b"not a trusted pickle")
    calls = []
    monkeypatch.setattr(torch, "load", lambda *a, **kw: calls.append("unpickle"))
    with pytest.raises(RuntimeContractError, match="unsupported for continuation"):
        load_training_state(path)
    assert calls == []


def test_changed_source_gate_never_deserializes(tmp_path, qualified_source, monkeypatch):
    root, identity = qualified_source
    path = tmp_path / "training_state.pt"
    save_training_state(path, {"schema": 2, "source_identity": {"execution_source": identity}},
                        source_root=root, required_paths=("source.py",))
    (root / "source.py").write_text("VALUE = 2\n")
    calls = []
    monkeypatch.setattr(torch, "load", lambda *a, **kw: calls.append("unpickle"))
    with pytest.raises(RuntimeContractError, match="unsupported for continuation"):
        load_training_state(path, source_root=root, required_paths=("source.py",))
    assert calls == []
