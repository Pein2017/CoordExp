"""Caller-visible characterization of the fixed Source256 binding."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from probes.dora_owner_learning import runtime
from probes.dora_owner_learning.selective_preservation import selective_loss
from src.common.errors import RuntimeContractError
from src.qwen.native import prepare_replay


def source_model():
    """Real CPU parameters with the fixed count; only one target is executed."""
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.language_model = torch.nn.Module()
    model.model.language_model.layers = torch.nn.ModuleList()
    for index in range(196):
        layer = torch.nn.Module()
        layer.q_proj = torch.nn.Module()
        for kind in ("lora_A", "lora_B", "lora_magnitude_vector"):
            adapter = torch.nn.Module()
            size = 18_006_016 - 587 if index == 195 and kind == "lora_magnitude_vector" else 1
            adapter.weight = torch.nn.Parameter(torch.zeros(size))
            setattr(layer.q_proj, kind, torch.nn.ModuleDict({"default": adapter}))
        model.model.language_model.layers.append(layer)
    model.frozen = torch.nn.Parameter(torch.tensor([0.25]))
    active = model.model.language_model.layers[0].q_proj
    with torch.no_grad():
        active.lora_A.default.weight.fill_(0.2)
        active.lora_B.default.weight.fill_(0.3)
        active.lora_magnitude_vector.default.weight.fill_(0.4)

    def forward(input_ids, logits_to_keep=0, **_):
        values = torch.cat([active.lora_A.default.weight, active.lora_B.default.weight,
                            active.lora_magnitude_vector.default.weight, model.frozen])
        logits = torch.nn.functional.pad(values, (0, 151646 - 4)).expand(1, input_ids.shape[1], -1)
        return SimpleNamespace(logits=logits[:, -logits_to_keep:] if logits_to_keep else logits)

    model.forward = forward
    model.get_rope_index = lambda input_ids, *args, **kwargs: (torch.arange(input_ids.shape[1]).expand(3, *input_ids.shape), None)
    return model


def observe_binding(bind):
    model = source_model()
    original = dict(model.named_parameters())
    named, frozen = bind(model)
    assert all(original[name] is parameter for name, parameter in (*named, *frozen))
    ids = [1, 2, 3, 151645]
    replay = prepare_replay(model, {"input_ids": torch.tensor([[7, 8]]),
                                    "image_grid_thw": torch.tensor([[1, 1, 1]])},
                            prompt_token_ids=[7, 8], continuation_token_ids=ids)
    logits = replay.aligned_logits(model(**replay.inputs).logits)
    reference = torch.log_softmax(logits[[0, 3]], -1).detach()
    loss, ce, kl = selective_loss(logits, replay.target_ids,
                                 {"state_ids": [1], "action_index": 1, "A_id": 2},
                                 ids, 3, [0, 3], reference)
    optimizer = torch.optim.AdamW([p for _, p in named], lr=1e-4, weight_decay=0, foreach=False)
    before_frozen = [p.detach().clone() for _, p in frozen]
    loss.backward()
    gradients = {n: p.grad.tolist() for n, p in named if p.grad is not None}
    optimizer.step()
    assert all(torch.equal(before, p) and p.grad is None for before, (_, p) in zip(before_frozen, frozen))
    return {
        "selected_count": len(named), "selected_scalars": sum(p.numel() for _, p in named),
        "ordered_names_sha256": hashlib.sha256(json.dumps([n for n, _ in named]).encode()).hexdigest(),
        "frozen_names": [n for n, _ in frozen],
        "trainable_names_match": [n for n, p in model.named_parameters() if p.requires_grad] == [n for n, _ in named],
        "loss": float(loss.detach()), "ce": float(ce.detach()), "kl": float(kl.detach()),
        "gradients": gradients,
        "updated": {n: p.detach().tolist() for n, p in named if p.grad is not None},
    }


def test_binding_preserves_legacy_profile_replay_and_one_step():
    expected = json.loads((Path(__file__).with_name("fixtures") / "source256_binding.json").read_text())
    actual = observe_binding(lambda model: runtime.bind_source256_language_dora(
        model, expected_tensor_count=588, expected_scalar_count=18_006_016))
    assert actual == expected


@pytest.mark.parametrize("change", ["tensor_count", "scalar_count", "adapter", "missing", "prohibited"])
def test_binding_rejects_wrong_surface_before_update(change):
    model = source_model()
    kwargs = dict(expected_tensor_count=588, expected_scalar_count=18_006_016)
    if change == "tensor_count":
        kwargs["expected_tensor_count"] = 587
    elif change == "scalar_count":
        kwargs["expected_scalar_count"] -= 1
    elif change == "adapter":
        kwargs["adapter_name"] = "absent"
    elif change == "missing":
        del model.model.language_model.layers[0].q_proj.lora_A.default.weight
    else:
        layer = model.model.language_model.layers[0]
        layer.visual = layer.q_proj
        del layer.q_proj
    with pytest.raises((ValueError, RuntimeContractError)):
        runtime.bind_source256_language_dora(model, **kwargs)
    assert not any(p.requires_grad for p in model.parameters())
