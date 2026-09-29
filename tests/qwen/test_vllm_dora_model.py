"""CPU falsification of the one-adapter vLLM DoRA arithmetic and refresh."""

import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from peft import LoraConfig
from peft.tuners.lora.layer import Linear as PeftLinear
from safetensors.torch import load_file
from torch import nn

from src.qwen.vllm_dora_model import (
    CoordExpDoRAQwen3VLForConditionalGeneration,
    _DoRALinear,
    _SelectedInput,
    _target_tensors,
    _validate_declared_targets,
)


class PackedBase(nn.Module):
    def __init__(self, weights: list[torch.Tensor], biases: list[torch.Tensor] | None):
        super().__init__()
        self.weight = nn.Parameter(torch.cat(weights), requires_grad=False)
        self.bias = None if biases is None else nn.Parameter(torch.cat(biases), requires_grad=False)
        self.skip_bias_add = False

    def forward(self, x):
        return F.linear(x, self.weight, self.bias), None


@pytest.mark.parametrize("widths", [(5,), (3, 4), (3, 2, 2)])
def test_matches_actual_peft_and_refreshes_without_replacing_buffers(widths):
    torch.manual_seed(7)
    in_features, rank, alpha = 6, 2, 4
    names = tuple(f"model.language_model.layers.0.target_{index}" for index in range(len(widths)))
    config = LoraConfig(r=rank, lora_alpha=alpha, use_dora=True, target_modules=["base"])
    references = []
    payload = {}
    weights, biases = [], []
    for name, width in zip(names, widths, strict=True):
        weight = torch.randn(width, in_features)
        bias = torch.randn(width)
        base = nn.Linear(in_features, width)
        with torch.no_grad():
            base.weight.copy_(weight)
            base.bias.copy_(bias)
        reference = PeftLinear(base, "default", config, r=rank, lora_alpha=alpha).eval()
        with torch.no_grad():
            reference.lora_A["default"].weight.copy_(torch.randn(rank, in_features) * 0.3)
            reference.lora_B["default"].weight.copy_(torch.randn(width, rank) * 0.4)
            reference.lora_magnitude_vector["default"].weight.copy_(torch.rand(width) + 0.6)
        references.append(reference)
        weights.append(weight)
        biases.append(bias)
        prefix = "base_model.model." + name
        payload[prefix + ".lora_A.weight"] = reference.lora_A["default"].weight.detach().clone()
        payload[prefix + ".lora_B.weight"] = reference.lora_B["default"].weight.detach().clone()
        payload[prefix + ".lora_magnitude_vector.weight"] = reference.lora_magnitude_vector["default"].weight.detach().clone()
    layer = _DoRALinear(PackedBase(weights, biases), names, widths, _target_tensors(payload), alpha / rank)
    x = torch.randn(4, in_features)
    expected = torch.cat([reference(x) for reference in references], dim=-1)
    torch.testing.assert_close(layer(x)[0], expected, atol=2e-6, rtol=2e-6)

    model = CoordExpDoRAQwen3VLForConditionalGeneration.__new__(CoordExpDoRAQwen3VLForConditionalGeneration)
    nn.Module.__init__(model)
    model._coordexp_linears = (layer,)
    model.coordexp_dora_identity = "initial"
    model.register_buffer("coordexp_input_embed_delta", torch.zeros(2, in_features))
    model.register_buffer("coordexp_output_embed_delta", torch.zeros(2, in_features))
    pointers = [tensor.data_ptr() for tensor in model.buffers()]
    new_payload = {key: value.clone() for key, value in payload.items()}
    for key in new_payload:
        if "magnitude" in key:
            new_payload[key] *= 1.4
    new_deltas = {"input_embed_delta": torch.ones(2, in_features), "output_embed_delta": torch.ones(2, in_features) * 2}
    assert model.refresh_coordexp_dora(new_payload, new_deltas, identity="updated") == "updated"
    assert [tensor.data_ptr() for tensor in model.buffers()] == pointers
    assert not torch.allclose(layer(x)[0], expected)
    with pytest.raises(ValueError, match="contain A, B, and magnitude"):
        model.refresh_coordexp_dora({key: value for key, value in new_payload.items() if "magnitude" not in key}, new_deltas, identity="bad")
    assert model.coordexp_dora_identity == "updated"
    with pytest.raises(ValueError, match="invalid refresh tensor"):
        malformed = dict(new_payload)
        first = next(iter(malformed))
        malformed[first] = torch.zeros(1)
        model.refresh_coordexp_dora(malformed, new_deltas, identity="bad")
    assert model.coordexp_dora_identity == "updated"


def test_selected_input_keeps_frozen_base_and_fp32_delta_separate():
    base = nn.Embedding(7, 3)
    with torch.no_grad():
        base.weight.copy_(torch.arange(21, dtype=torch.float32).view(7, 3))
    rows = torch.tensor([-1, 0, -1, -1, 1, -1, -1])
    delta = torch.tensor([[0.25, 0.5, 0.75], [-0.5, -0.25, 0.125]])
    wrapper = _SelectedInput(base, rows, delta)
    before = base.weight.detach().clone()
    ids = torch.tensor([0, 1, 4, 6])
    expected = before[ids].clone()
    expected[1] += delta[0]
    expected[2] += delta[1]
    torch.testing.assert_close(wrapper(ids), expected)
    torch.testing.assert_close(base.weight, before)


def test_peft_suffix_targets_cover_actual_module_paths():
    targets = {
        f"model.language_model.layers.{layer}.self_attn.{suffix}"
        for layer in (0, 1)
        for suffix in ("q_proj", "k_proj", "v_proj", "o_proj")
    }
    _validate_declared_targets({"q_proj", "k_proj", "v_proj", "o_proj"}, targets)
    with pytest.raises(ValueError, match="config targets"):
        _validate_declared_targets({"q_proj", "k_proj", "o_proj"}, targets)


def test_saved_and_live_magnitude_key_forms_reject_duplicates():
    prefix = "base_model.model.model.language_model.layers.0.mlp.down_proj"
    payload = {
        prefix + ".lora_A.weight": torch.zeros(2, 3),
        prefix + ".lora_B.weight": torch.zeros(4, 2),
        prefix + ".lora_magnitude_vector": torch.ones(4),
    }
    assert set(_target_tensors(payload)) == {"model.language_model.layers.0.mlp.down_proj"}
    payload[prefix + ".lora_magnitude_vector.default.weight"] = torch.ones(4)
    with pytest.raises(ValueError, match="duplicate DoRA tensor"):
        _target_tensors(payload)


_ANCHOR = Path(
    "/data/CoordExp/outputs/infra_base/start-loss-benchmark-20260928/train/"
    "instance_margin-order17/checkpoints/step-256/adapter"
)


@pytest.mark.skipif(not _ANCHOR.is_dir(), reason="local DoRA anchor unavailable")
def test_real_588_key_anchor_parses_completely():
    tensors = load_file(str(_ANCHOR / "adapter_model.safetensors"), device="cpu")
    config = json.loads((_ANCHOR / "adapter_config.json").read_text())
    assert len(tensors) == 588
    targets = _target_tensors(tensors)
    assert len(targets) == 196
    _validate_declared_targets(set(config["target_modules"]), set(targets))
    for parts in targets.values():
        a, b, m = (parts[key] for key in ("lora_A", "lora_B", "lora_magnitude_vector"))
        assert a.ndim == b.ndim == 2 and m.ndim == 1
        assert a.shape[0] == b.shape[1] == config["r"]
        assert b.shape[0] == m.shape[0]
        assert a.dtype == b.dtype == m.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_bf16_autocast_matches_peft():
    device = torch.device("cuda")
    torch.manual_seed(13)
    name = "model.language_model.layers.0.self_attn.q_proj"
    base = nn.Linear(8, 5, device=device, dtype=torch.bfloat16)
    config = LoraConfig(r=2, lora_alpha=4, use_dora=True, target_modules=["base"])
    reference = PeftLinear(base, "default", config, r=2, lora_alpha=4).eval()
    with torch.no_grad():
        reference.lora_A["default"].weight.data = torch.randn_like(reference.lora_A["default"].weight.float()) * 0.3
        reference.lora_B["default"].weight.data = torch.randn_like(reference.lora_B["default"].weight.float()) * 0.4
        reference.lora_magnitude_vector["default"].weight.data = torch.rand_like(reference.lora_magnitude_vector["default"].weight.float()) + 0.7
    payload = {
        "base_model.model." + name + "." + key + ".weight": tensor.detach().cpu().clone()
        for key, tensor in (
            ("lora_A", reference.lora_A["default"].weight),
            ("lora_B", reference.lora_B["default"].weight),
            ("lora_magnitude_vector", reference.lora_magnitude_vector["default"].weight),
        )
    }
    packed = PackedBase([base.weight.detach().clone()], [base.bias.detach().clone()]).to(device)
    candidate = _DoRALinear(packed, (name,), (5,), _target_tensors(payload), 2.0).eval()
    x = torch.randn(4, 8, device=device, dtype=torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected = reference(x)
    actual = candidate(x)[0]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
