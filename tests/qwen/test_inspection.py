from types import SimpleNamespace

import pytest
import torch

from src.qwen.inspection import CaptureHiddenRows, CaptureInputs, resolve_text_stack


def test_capture_clones_before_inplace_injection_and_preserves_distinct_boundaries():
    layer = torch.nn.Identity()
    next_layer = torch.nn.Identity()
    hidden = torch.arange(12.0).reshape(1, 3, 4)
    with (
        CaptureHiddenRows(layer, [0, 2], boundary="output") as before,
        CaptureHiddenRows(next_layer, [0, 2], boundary="input") as after,
    ):
        returned = layer(hidden)
        returned[:, 0] += 7
        next_layer(returned)
    torch.testing.assert_close(
        before.hidden, torch.tensor([[0.0, 1.0, 2.0, 3.0], [8.0, 9.0, 10.0, 11.0]])
    )
    torch.testing.assert_close(
        after.hidden[0] - before.hidden[0], torch.full((4,), 7.0)
    )
    assert not layer._forward_hooks and not next_layer._forward_pre_hooks


def test_capture_cleanup_preserves_forward_exception_and_bounded_repeated_calls():
    layer = torch.nn.Identity()
    with pytest.raises(KeyError, match="original"):
        with CaptureHiddenRows(layer, [0]):
            raise KeyError("original")
    assert not layer._forward_pre_hooks
    with CaptureHiddenRows(layer, [0], max_calls=2) as capture:
        layer(torch.ones(1, 1, 2))
        layer(torch.zeros(1, 1, 2))
    assert capture.calls == 2
    with pytest.raises(RuntimeError, match="exactly one"):
        _ = capture.hidden
    with pytest.raises(RuntimeError, match="bound"):
        with CaptureHiddenRows(layer, [0]):
            layer(torch.ones(1, 1, 2))
            layer(torch.zeros(1, 1, 2))
    assert not layer._forward_pre_hooks


def test_actual_captured_inputs_replay_functionally_and_are_autograd_safe():
    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([2.0, 3.0]))

        def forward(self, *, inputs_embeds, position_ids):
            return inputs_embeds * self.weight + position_ids

    layer = Layer()
    with torch.inference_mode():
        inputs = dict(
            inputs_embeds=torch.tensor([5.0, 7.0]),
            position_ids=torch.tensor([0.0, 1.0]),
        )
        with CaptureInputs(layer, keys=["inputs_embeds", "position_ids"]) as capture:
            original = layer(**inputs)
    replay = torch.func.functional_call(
        layer, dict(layer.named_parameters()), capture.args, capture.kwargs
    )
    torch.testing.assert_close(replay, original)
    replay.sum().backward()
    torch.testing.assert_close(layer.weight.grad, torch.tensor([5.0, 7.0]))
    assert not layer._forward_pre_hooks


def test_resolve_aliases_once_and_reject_ambiguous_text_stacks():
    def stack():
        return SimpleNamespace(
            layers=torch.nn.ModuleList([torch.nn.Identity()]), norm=torch.nn.Identity()
        )

    owner = stack()
    head = torch.nn.Linear(2, 4)
    model = SimpleNamespace(
        language_model=owner,
        model=SimpleNamespace(language_model=owner),
        get_output_embeddings=lambda: head,
    )
    sites = resolve_text_stack(model)
    assert sites.language_model is owner and sites.head is head
    model.model.language_model = stack()
    with pytest.raises(ValueError, match="distinct"):
        resolve_text_stack(model)
