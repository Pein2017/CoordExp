from __future__ import annotations

import copy
from dataclasses import dataclass

import pytest
import torch

from scripts.research.human13_training_transaction import (
    TrainingStateTransaction,
    UpdateCounter,
)


@dataclass
class _ToyRuntime:
    optimizer_step_count: int
    scheduler_step_count: int


def _make_stack(seed: int = 17, *, runtime: _ToyRuntime | None = None):
    torch.manual_seed(seed)
    model = torch.nn.Linear(3, 2)
    model.bias.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        (model.weight,), lr=0.03, betas=(0.8, 0.9), weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
    counter = UpdateCounter()
    transaction = TrainingStateTransaction(
        (("adapter.language.weight", model.weight),),
        optimizer=optimizer,
        scheduler=scheduler,
        update_counter=counter,
        runtime=runtime,
        capture_cuda=torch.cuda.is_available(),
    )
    return model, optimizer, scheduler, counter, transaction


def test_runtime_step_counters_participate_in_digest_and_rejected_restore() -> None:
    runtime = _ToyRuntime(optimizer_step_count=4, scheduler_step_count=3)
    _, _, _, _, transaction = _make_stack(runtime=runtime)
    snapshot = transaction.begin()

    assert snapshot.runtime_optimizer_step_count == 4
    assert snapshot.runtime_scheduler_step_count == 3

    runtime.optimizer_step_count = 5
    optimizer_counter_digest = transaction.state_digest()
    assert optimizer_counter_digest != snapshot.state_digest
    runtime.scheduler_step_count = 7
    assert transaction.state_digest() != optimizer_counter_digest

    receipt = transaction.reject(snapshot)

    assert runtime.optimizer_step_count == 4
    assert runtime.scheduler_step_count == 3
    assert receipt.after_state_digest == snapshot.state_digest
    assert transaction.state_digest() == snapshot.state_digest


def _step(model, optimizer, scheduler, counter, x, target) -> None:
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(model(x), target)
    loss.backward()
    optimizer.step()
    scheduler.step()
    counter.value += 1


def _assert_optimizer_equal(left, right) -> None:
    left_state = left.state_dict()
    right_state = right.state_dict()
    assert left_state["param_groups"] == right_state["param_groups"]
    assert left_state["state"].keys() == right_state["state"].keys()
    for key in left_state["state"]:
        assert left_state["state"][key].keys() == right_state["state"][key].keys()
        for field, left_value in left_state["state"][key].items():
            right_value = right_state["state"][key][field]
            if isinstance(left_value, torch.Tensor):
                assert torch.equal(left_value, right_value)
            else:
                assert left_value == right_value


def test_reject_restores_parameters_adamw_scheduler_counter_and_rng() -> None:
    model, optimizer, scheduler, counter, transaction = _make_stack()
    x = torch.tensor(((1.0, 2.0, 3.0),))
    target = torch.tensor(((0.25, -0.5),))
    _step(model, optimizer, scheduler, counter, x, target)
    frozen_bias = model.bias.detach().clone()
    snapshot = transaction.begin()
    expected_random = torch.rand(5)
    transaction.restore(snapshot)

    _step(model, optimizer, scheduler, counter, x * 2, target * -1)
    model.bias.data.add_(9.0)
    torch.rand(19)
    assert transaction.state_digest() != snapshot.state_digest

    receipt = transaction.reject(snapshot)

    assert receipt.decision == "rejected_restored"
    assert receipt.before_state_digest == snapshot.state_digest
    assert receipt.after_state_digest == snapshot.state_digest
    assert transaction.state_digest() == snapshot.state_digest
    assert counter.value == 1
    assert scheduler.last_epoch == 1
    assert model.bias.detach().equal(frozen_bias + 9.0)
    assert torch.equal(torch.rand(5), expected_random)


def test_accept_commits_mutated_state_and_closes_transaction() -> None:
    model, optimizer, scheduler, counter, transaction = _make_stack()
    snapshot = transaction.begin()
    _step(
        model,
        optimizer,
        scheduler,
        counter,
        torch.ones((1, 3)),
        torch.zeros((1, 2)),
    )
    mutated = transaction.state_digest()

    receipt = transaction.accept(snapshot)

    assert receipt.decision == "accepted_committed"
    assert receipt.before_state_digest == snapshot.state_digest
    assert receipt.after_state_digest == mutated
    assert transaction.state_digest() == mutated
    transaction.begin()


def test_repeated_restore_is_idempotent() -> None:
    model, optimizer, scheduler, counter, transaction = _make_stack()
    snapshot = transaction.begin()
    original = model.weight.detach().clone()
    model.weight.data.add_(4.0)
    transaction.restore(snapshot)
    model.weight.data.sub_(7.0)
    transaction.restore(snapshot)

    assert torch.equal(model.weight, original)
    assert transaction.state_digest() == snapshot.state_digest


def test_rejected_adamw_update_leaves_next_update_equal_to_no_reject_control() -> None:
    exp_model, exp_opt, exp_sched, exp_counter, transaction = _make_stack()
    ctl_model, ctl_opt, ctl_sched, ctl_counter, _ = _make_stack()
    warm_x = torch.tensor(((0.2, -0.4, 0.7),))
    warm_y = torch.tensor(((0.5, -0.2),))
    _step(exp_model, exp_opt, exp_sched, exp_counter, warm_x, warm_y)
    _step(ctl_model, ctl_opt, ctl_sched, ctl_counter, warm_x, warm_y)
    assert torch.equal(exp_model.weight, ctl_model.weight)
    _assert_optimizer_equal(exp_opt, ctl_opt)

    snapshot = transaction.begin()
    _step(exp_model, exp_opt, exp_sched, exp_counter, warm_x * 9, warm_y * -4)
    transaction.reject(snapshot)

    next_x = torch.tensor(((1.0, 0.5, -0.25),))
    next_y = torch.tensor(((-0.1, 0.8),))
    _step(exp_model, exp_opt, exp_sched, exp_counter, next_x, next_y)
    _step(ctl_model, ctl_opt, ctl_sched, ctl_counter, next_x, next_y)

    assert torch.equal(exp_model.weight, ctl_model.weight)
    _assert_optimizer_equal(exp_opt, ctl_opt)
    assert exp_sched.state_dict() == ctl_sched.state_dict()
    assert exp_counter.value == ctl_counter.value


def test_transaction_rejects_optimizer_parameter_outside_bound_surface() -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())

    with pytest.raises(ValueError, match="outside the bound trainable surface"):
        TrainingStateTransaction(
            (("adapter.language.weight", model.weight),),
            optimizer=optimizer,
            scheduler=None,
            update_counter=UpdateCounter(),
        )


def test_snapshot_payload_is_not_mutated_by_restore() -> None:
    model, optimizer, scheduler, counter, transaction = _make_stack()
    _step(
        model,
        optimizer,
        scheduler,
        counter,
        torch.ones((1, 3)),
        torch.zeros((1, 2)),
    )
    snapshot = transaction.begin()
    optimizer_payload = copy.deepcopy(snapshot.optimizer_state)
    transaction.restore(snapshot)

    for key, state in optimizer_payload["state"].items():
        for field, value in state.items():
            restored = snapshot.optimizer_state["state"][key][field]
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, restored)
            else:
                assert value == restored
