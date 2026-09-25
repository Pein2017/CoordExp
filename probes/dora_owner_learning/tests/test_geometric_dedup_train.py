from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from probes.dora_owner_learning import geometric_dedup_train as train


def test_global_eight_online_and_fifty_six_reference_normalization():
    online_ul = [torch.tensor(float(i + 1)) for i in range(8)]
    online_kl = [torch.tensor(float(i + 1) / 100) for i in range(8)]
    reference_kl = [torch.tensor(float(i + 1) / 1000) for i in range(56)]
    local = []
    for rank in range(8):
        loss = train.ddp_item_loss("online", online_ul[rank], online_kl[rank])
        loss = loss + sum(train.ddp_item_loss("reference", torch.tensor(0.0), value)
                          for value in reference_kl[rank::8])
        local.append(loss)
    ddp_average = torch.stack(local).mean()
    expected = (torch.stack(online_ul).mean() +
                10 * torch.stack(online_kl).mean() +
                100 * torch.stack(reference_kl).mean())
    assert torch.equal(ddp_average, expected)
    assert not torch.equal(torch.stack(local).sum(), expected)


class RecordingDDP:
    def __init__(self, parameter):
        self.parameter = parameter
        self.no_sync_entries = 0
        self.active_no_sync = False
        self.forward_sync_flags = []

    @contextmanager
    def no_sync(self):
        assert not self.active_no_sync
        self.no_sync_entries += 1
        self.active_no_sync = True
        try:
            yield
        finally:
            self.active_no_sync = False

    def loss(self, scale):
        self.forward_sync_flags.append(not self.active_no_sync)
        return self.parameter * scale


def test_seven_no_sync_backwards_then_one_synchronized_backward():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    ddp = RecordingDDP(parameter)
    items = [dict(kind="online")] + [dict(kind="reference") for _ in range(7)]

    def loss_for_item(item):
        loss = ddp.loss(torch.tensor(2.0 if item["kind"] == "online" else 1.0))
        return loss, dict(unlikelihood=0.0, kl=0.0, states=1)

    records = train.backward_rank_items(ddp, items, loss_for_item)
    assert ddp.no_sync_entries == 7
    assert ddp.forward_sync_flags == [False] * 7 + [True]
    assert [r["synchronized"] for r in records] == [False] * 7 + [True]
    assert parameter.grad.item() == pytest.approx(9.0)


def test_backward_rejects_unequal_local_work_before_any_gradient():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    ddp = RecordingDDP(parameter)
    with pytest.raises(ValueError, match="fixed local item count"):
        train.backward_rank_items(
            ddp, [dict(kind="online")] * 7,
            lambda _: (ddp.loss(torch.tensor(1.0)), {}),
        )
    assert parameter.grad is None and ddp.no_sync_entries == 0


def test_empty_kl_mask_still_scores_teacher_once_and_returns_detached_empty(monkeypatch):
    from src.qwen import native

    class Replay:
        def __init__(self, action_ids):
            self.inputs = {"input_ids": torch.tensor([[1]])}
            self.target_ids = torch.tensor(action_ids, dtype=torch.long)

        @staticmethod
        def aligned_logits(logits):
            return logits

    def prepare_replay(_model, _inputs, *, prompt_token_ids, continuation_token_ids):
        assert prompt_token_ids == [1, 2]
        return Replay(continuation_token_ids)

    class Teacher(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, **_inputs):
            self.calls += 1
            return SimpleNamespace(logits=torch.randn(2, 5))

    monkeypatch.setattr(native, "prepare_replay", prepare_replay)
    teacher = Teacher()
    reference = train._teacher_reference(
        teacher, {}, [1, 2], [3, 4], {"kl_positions": []},
    )
    assert teacher.calls == 1
    assert reference.shape == (0, 5)
    assert not reference.requires_grad and reference.grad_fn is None


def test_nonzero_is_only_required_by_first_two_update_gate():
    with pytest.raises(ValueError, match="two-step gate requires nonzero"):
        train.update_scalar_status(1, 0.0, 0.0, 0.0)
    assert train.update_scalar_status(3, 0.0, 0.0, 1.0) == {
        "zero_gradient": True, "zero_movement": True,
    }
    with pytest.raises(ValueError, match="finite nonnegative"):
        train.update_scalar_status(3, float("nan"), 0.0, 1.0)


def test_non_reentrant_decoder_checkpoint_closures_eval_bypass_and_parity():
    class Block(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(value))

        def forward(self, hidden_states, *, scale):
            return torch.sin(hidden_states * self.weight) + scale

    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.language_model = torch.nn.Module()
            self.model.language_model.layers = torch.nn.ModuleList(
                [Block(0.90 + index / 1000) for index in range(28)]
            )

    model = ToyModel().eval()
    state = train.install_language_decoder_checkpointing(model)
    layers = model.model.language_model.layers

    def pass_and_grad(enabled):
        state["enabled"] = enabled
        model.zero_grad(set_to_none=True)
        value = torch.tensor([0.2, -0.1], requires_grad=True)
        for index, layer in enumerate(layers):
            value = layer(value, scale=(index + 1) / 1000)
        loss = value.square().sum()
        loss.backward()
        gradient = torch.cat(
            [value_.grad.detach().reshape(-1) for value_ in model.parameters()]
        )
        return float(loss.detach()), gradient

    loss_off, gradient_off = pass_and_grad(False)
    disabled_calls = state["bypass_disabled_calls"]
    loss_on, gradient_on = pass_and_grad(True)
    metrics = train.parity_metrics(loss_off, loss_on, gradient_off, gradient_on)
    assert metrics["passed"] and disabled_calls == 28
    assert state["checkpoint_invocations"] == 28
    assert state["checkpoint_body_calls"] > state["checkpoint_invocations"]

    with torch.no_grad():
        value = torch.tensor([0.2, -0.1])
        for index, layer in enumerate(layers):
            value = layer(value, scale=(index + 1) / 1000)
    receipt = train.checkpointing_receipt(model, state)
    assert receipt["bypass_no_grad_calls"] == 28
    assert receipt["model_eval"] and receipt["all_decoder_layers_eval"]
    assert not model.training and all(not layer.training for layer in layers)


def test_only_rank_one_gets_two_parity_forward_allowance():
    assert [train.rank_model_forward_ceiling(rank) for rank in range(8)] == [
        train.MODEL_FORWARD_CEILING,
        train.MODEL_FORWARD_CEILING + 2,
        *([train.MODEL_FORWARD_CEILING] * 6),
    ]
    assert [train.rank_image_forward_count(rank) for rank in range(8)] == [
        train.IMAGE_FORWARD_COUNT,
        train.IMAGE_FORWARD_COUNT + 2,
        *([train.IMAGE_FORWARD_COUNT] * 6),
    ]


def test_checkpoint_owner_resolves_the_actual_peft_shaped_module_tree():
    # The loaded names are base_model.model.model.language_model, not
    # model.language_model as in a bare conditional-generation test model.
    model = torch.nn.Module()
    model.base_model = torch.nn.Module()
    model.base_model.model = torch.nn.Module()
    model.base_model.model.model = torch.nn.Module()
    owner = torch.nn.Module()
    owner.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(28)])
    model.base_model.model.model.language_model = owner
    model.eval()
    with pytest.raises(AttributeError):
        _ = model.model.language_model.layers
    state = train.install_language_decoder_checkpointing(model)
    assert state['decoder_module_name'] == 'base_model.model.model.language_model'
    state['enabled'] = True
    x = torch.ones(1, 2)
    for layer in owner.layers:
        x = layer(x)
    x.sum().backward()
    assert all(p.grad is not None for p in model.parameters())
    receipt = train.checkpointing_receipt(model, state)
    assert receipt['decoder_module_name'] == state['decoder_module_name']
    assert receipt['checkpoint_invocations'] == 28 and receipt['model_eval']


def test_cli_exposes_only_rank_run_finalize_verify(monkeypatch):
    monkeypatch.setattr("sys.argv", ["geometric_dedup_train", "prepare"])
    with pytest.raises(SystemExit):
        train.main()
