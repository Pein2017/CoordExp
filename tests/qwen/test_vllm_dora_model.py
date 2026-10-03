"""CPU falsification of the one-adapter vLLM DoRA arithmetic and refresh."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from peft import LoraConfig
from peft.tuners.lora.layer import Linear as PeftLinear
from safetensors.torch import load_file
from torch import nn

from src.qwen.vllm_dora_model import (
    CoordExpDoRAQwen3VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
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


def _coordinate_model(*, bias: bool = False):
    model = CoordExpDoRAQwen3VLForConditionalGeneration.__new__(
        CoordExpDoRAQwen3VLForConditionalGeneration
    )
    nn.Module.__init__(model)
    coordinate_ids = tuple(range(1000))
    selected_ids = tuple(range(1000, 1004)) + coordinate_ids
    vocab_size = 1010
    head = nn.Linear(2, vocab_size, bias=bias, dtype=torch.bfloat16)
    with torch.no_grad():
        head.weight.fill_(0)
        head.weight[:, 1] = 1
    model.language_model = SimpleNamespace(lm_head=head)
    model.coordexp_dora_identity = "snapshot-1"
    model._coordexp_coordinate_token_ids = coordinate_ids
    model._coordexp_linears = ()
    model.register_buffer("coordexp_token_ids", torch.tensor(selected_ids, dtype=torch.long))
    rows = torch.full((vocab_size,), -1, dtype=torch.long)
    rows[torch.tensor(selected_ids)] = torch.arange(len(selected_ids))
    model.register_buffer("coordexp_token_rows", rows)
    model.register_buffer("coordexp_input_embed_delta", torch.zeros(len(selected_ids), 2))
    model.register_buffer("coordexp_output_embed_delta", torch.zeros(len(selected_ids), 2))
    return model, coordinate_ids


def _refresh_adapter_payload():
    target = "model.language_model.layers.0.mlp.down_proj"
    prefix = "base_model.model." + target
    return {
        prefix + ".lora_A.weight": torch.tensor([[0.2]], dtype=torch.float32),
        prefix + ".lora_B.weight": torch.tensor([[0.3]], dtype=torch.float32),
        prefix + ".lora_magnitude_vector.weight": torch.tensor([1.0], dtype=torch.float32),
    }


def _install_refresh_linear(model):
    target = "model.language_model.layers.0.mlp.down_proj"
    layer = _DoRALinear(
        PackedBase([torch.ones(1, 1)], None),
        (target,),
        (1,),
        _target_tensors(_refresh_adapter_payload()),
        1.0,
    )
    model._coordexp_linears = (layer,)
    return layer


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


def test_coordinate_output_norm_default_and_off_preserve_base_logits(monkeypatch):
    model, coordinate_ids = _coordinate_model()

    def base_compute_logits(self, hidden_states):
        return F.linear(hidden_states, self.language_model.lm_head.weight)

    monkeypatch.setattr(Qwen3VLForConditionalGeneration, "compute_logits", base_compute_logits)
    hidden = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.bfloat16)
    expected = base_compute_logits(model, hidden)
    torch.testing.assert_close(model.compute_logits(hidden), expected, atol=0, rtol=0)

    receipt = model.configure_coordinate_output_norm("off", coordinate_ids, identity="snapshot-1")
    actual = model.compute_logits(hidden)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert receipt["calls"] == 0
    receipt = model.coordinate_output_norm_receipt()
    json.dumps(receipt)
    assert receipt["calls"] == 1
    assert receipt["coordinate_positions_processed"] == 2
    assert receipt["first_call"] == {
        "native_dtype": "torch.bfloat16",
        "scaling_active": False,
        "changed_coordinates": 0,
        "max_abs_difference": 0.0,
        "non_coordinate_unchanged": True,
    }


def test_coordinate_output_norm_scales_only_coordinates_and_can_flip_greedy(monkeypatch):
    model, coordinate_ids = _coordinate_model()
    with torch.no_grad():
        model.language_model.lm_head.weight[0] = torch.tensor([0.1, 0.0])
        model.language_model.lm_head.weight[1] = torch.tensor([0.2, 1.0])

    def base_compute_logits(self, hidden_states):
        return F.linear(hidden_states, self.language_model.lm_head.weight)

    monkeypatch.setattr(Qwen3VLForConditionalGeneration, "compute_logits", base_compute_logits)
    hidden = torch.tensor([[1.0, 0.0]], dtype=torch.bfloat16)
    before = base_compute_logits(model, hidden)
    assert int(before.argmax(-1)) == 1
    model.configure_coordinate_output_norm("median", coordinate_ids, identity="snapshot-1")
    after = model.compute_logits(hidden)
    assert int(after.argmax(-1)) == 0
    assert after.dtype == before.dtype
    non_coordinates = torch.ones(before.shape[-1], dtype=torch.bool)
    non_coordinates[torch.tensor(coordinate_ids)] = False
    torch.testing.assert_close(after[..., non_coordinates], before[..., non_coordinates], atol=0, rtol=0)
    torch.testing.assert_close(after[..., 1009], before[..., 1009], atol=0, rtol=0)

    receipt = model.coordinate_output_norm_receipt()
    json.dumps(receipt)
    assert receipt["calls"] == 1
    assert receipt["coordinate_positions_processed"] == 1
    assert receipt["coordinate_tokens"] == 1000
    assert len(receipt["coordinate_ids"]) == 1000
    assert receipt["norm_min"] > 0
    assert receipt["median_norm"] == pytest.approx(1.0)
    assert receipt["first_call"]["native_dtype"] == "torch.bfloat16"
    assert receipt["first_call"]["scaling_active"] is True
    assert receipt["first_call"]["changed_coordinates"] > 0
    assert receipt["first_call"]["max_abs_difference"] > 0
    assert receipt["first_call"]["non_coordinate_unchanged"] is True


def test_coordinate_output_norm_uses_effective_base_plus_output_delta():
    model, coordinate_ids = _coordinate_model()
    with torch.no_grad():
        model.language_model.lm_head.weight[0] = torch.tensor([1.0, 0.0])
        model.coordexp_output_embed_delta[4] = torch.tensor([2.0, 0.0])
    model.configure_coordinate_output_norm("median", coordinate_ids, identity="snapshot-1")
    receipt = model.coordinate_output_norm_receipt()
    assert receipt["median_norm"] == pytest.approx(1.0)
    assert receipt["norm_max"] == pytest.approx(3.0)
    assert receipt["factor_min"] == pytest.approx(1.0 / 3.0, rel=1e-5)


def test_model_trace_capture_sees_output_delta_before_coordinate_scaling(monkeypatch):
    model, coordinate_ids = _coordinate_model()
    with torch.no_grad():
        model.language_model.lm_head.weight[0] = torch.tensor([0.25, 0.0])
        model.language_model.lm_head.weight[1009] = torch.tensor([0.75, 0.0])
        model.coordexp_output_embed_delta[4, 0] = 0.25
    model.configure_coordinate_output_norm("median", coordinate_ids, identity="snapshot-1")

    def base_compute_logits(self, hidden_states):
        return F.linear(hidden_states, self.language_model.lm_head.weight)

    class Capture:
        def capture(self, logits, identity):
            self.raw = logits.clone()
            self.identity = identity

    monkeypatch.setattr(Qwen3VLForConditionalGeneration, "compute_logits", base_compute_logits)
    state = Capture()
    model._coordexp_paired_trace_state = state
    policy = model.compute_logits(torch.tensor([[1.0, 0.0]], dtype=torch.bfloat16))
    assert state.identity == "snapshot-1"
    assert state.raw[0, 0].item() == 0.5  # base plus independent output delta
    assert policy[0, 0].item() == 1.0
    assert policy[0, 1009].item() == state.raw[0, 1009].item() == 0.75
    assert state.raw.float().log_softmax(-1)[0, 1009] != policy.float().log_softmax(-1)[0, 1009]


def test_model_refresh_rejects_active_trace_before_any_weight_change():
    model, _ = _coordinate_model()
    layer = _install_refresh_linear(model)
    before = {name: value.clone() for name, value in layer.named_buffers()}
    model._coordexp_paired_trace_state = object()
    with pytest.raises(RuntimeError, match="paired trace is active"):
        model.refresh_coordexp_dora(_refresh_adapter_payload(),
            {"input_embed_delta": torch.ones_like(model.coordexp_input_embed_delta),
             "output_embed_delta": torch.ones_like(model.coordexp_output_embed_delta)},
            identity="snapshot-2")
    assert model.coordexp_dora_identity == "snapshot-1"
    assert torch.count_nonzero(model.coordexp_input_embed_delta) == 0
    for name, value in layer.named_buffers():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)


def test_trace_off_model_keeps_no_trace_state_or_new_buffers(monkeypatch):
    model, coordinate_ids = _coordinate_model()
    model.configure_coordinate_output_norm("median", coordinate_ids, identity="snapshot-1")
    buffers = {name: value.data_ptr() for name, value in model.named_buffers()}
    monkeypatch.setattr(Qwen3VLForConditionalGeneration, "compute_logits",
        lambda self, hidden: F.linear(hidden, self.language_model.lm_head.weight))
    for _ in range(3):
        model.compute_logits(torch.tensor([[1.0, 0.0]], dtype=torch.bfloat16))
    assert not hasattr(model, "_coordexp_paired_trace_state")
    assert {name: value.data_ptr() for name, value in model.named_buffers()} == buffers


def test_coordinate_output_norm_rejects_bad_binding_and_invalid_norms():
    model, coordinate_ids = _coordinate_model()
    with pytest.raises(ValueError, match="mode"):
        model.configure_coordinate_output_norm("mean", coordinate_ids, identity="snapshot-1")
    with pytest.raises(ValueError, match="identity"):
        model.configure_coordinate_output_norm("median", coordinate_ids, identity="stale")
    with pytest.raises(ValueError, match="1,000"):
        model.configure_coordinate_output_norm("median", coordinate_ids[:-1], identity="snapshot-1")
    duplicate_ids = list(coordinate_ids)
    duplicate_ids[-1] = duplicate_ids[0]
    with pytest.raises(ValueError, match="unique"):
        model.configure_coordinate_output_norm("median", duplicate_ids, identity="snapshot-1")
    wrong_coordinate_ids = list(coordinate_ids)
    wrong_coordinate_ids[-1] = 1000
    with pytest.raises(ValueError, match="selected output coordinate rows"):
        model.configure_coordinate_output_norm("median", wrong_coordinate_ids, identity="snapshot-1")

    with torch.no_grad():
        model.language_model.lm_head.weight[0] = torch.tensor([1.0, 0.0])
        model.coordexp_output_embed_delta[4] = torch.tensor([-1.0, 0.0])
    with pytest.raises(ValueError, match="finite and positive"):
        model.configure_coordinate_output_norm("median", coordinate_ids, identity="snapshot-1")

    model.coordexp_output_embed_delta[4] = torch.tensor([float("nan"), 0.0])
    with pytest.raises(ValueError, match="finite FP32"):
        model.configure_coordinate_output_norm("median", coordinate_ids, identity="snapshot-1")


def test_coordinate_output_norm_rejects_head_bias():
    model, coordinate_ids = _coordinate_model(bias=True)
    with pytest.raises(ValueError, match="bias"):
        model.configure_coordinate_output_norm("median", coordinate_ids, identity="snapshot-1")


def test_coordinate_output_norm_refreshes_factor_in_place_and_rejects_bad_refresh():
    model, coordinate_ids = _coordinate_model()
    layer = _install_refresh_linear(model)
    with torch.no_grad():
        model.language_model.lm_head.weight[0] = torch.tensor([1.0, 0.0])
    model.configure_coordinate_output_norm("off", coordinate_ids, identity="snapshot-1")
    model.configure_coordinate_output_norm("median", coordinate_ids, identity="snapshot-1")
    factors = model.coordexp_coordinate_output_norm_factors
    factor_ptr = factors.data_ptr()

    output_delta = torch.zeros_like(model.coordexp_output_embed_delta)
    output_delta[4, 0] = 1.0
    assert model.refresh_coordexp_dora(
        _refresh_adapter_payload(),
        {"input_embed_delta": torch.zeros_like(output_delta), "output_embed_delta": output_delta},
        identity="snapshot-2",
    ) == "snapshot-2"
    assert model.coordexp_coordinate_output_norm_factors.data_ptr() == factor_ptr
    assert model.coordexp_coordinate_output_norm_factors[0].item() == pytest.approx(0.5)
    assert model.coordinate_output_norm_receipt()["identity"] == "snapshot-2"

    before_input_delta = model.coordexp_input_embed_delta.clone()
    before_delta = model.coordexp_output_embed_delta.clone()
    before_factor = factors.clone()
    before_layer_buffers = {
        name: value.clone() for name, value in layer.named_buffers()
    }
    bad_adapter = _refresh_adapter_payload()
    bad_adapter[next(key for key in bad_adapter if "lora_A" in key)] *= 2
    bad_adapter[next(key for key in bad_adapter if "magnitude" in key)] *= 2
    invalid_delta = torch.zeros_like(output_delta)
    invalid_delta[4, 0] = -1.0
    with pytest.raises(ValueError, match="finite and positive"):
        model.refresh_coordexp_dora(
            bad_adapter,
            {"input_embed_delta": torch.ones_like(output_delta), "output_embed_delta": invalid_delta},
            identity="snapshot-3",
        )
    assert model.coordexp_dora_identity == "snapshot-2"
    torch.testing.assert_close(model.coordexp_input_embed_delta, before_input_delta, atol=0, rtol=0)
    torch.testing.assert_close(model.coordexp_output_embed_delta, before_delta, atol=0, rtol=0)
    torch.testing.assert_close(factors, before_factor, atol=0, rtol=0)
    for name, value in layer.named_buffers():
        torch.testing.assert_close(value, before_layer_buffers[name], atol=0, rtol=0)


_ANCHOR = Path(
    "/data/CoordExp/outputs/shared/checkpoints/"
    "start-loss-instance-margin-order17-step256/payload/adapter"
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
