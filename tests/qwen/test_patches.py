from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest
import torch
from torch import nn

from src.common.errors import QwenForwardContractError
from src.qwen.patches import (
    PATCH_EMBED_LINEARIZATION_NAME,
    LinearizedQwen3VLPatchEmbed,
    apply_qwen3_vl_patch_embed_linearization,
)


BASE_MODEL = Path(
    "/data/CoordExp/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent"
)


def test_linearized_patch_embed_matches_conv3d_projection() -> None:
    torch.manual_seed(7)
    original = Qwen3VLVisionPatchEmbed()
    patched = LinearizedQwen3VLPatchEmbed(original)
    pixel_values = torch.randn(5, original.flat_input_features)

    conv_output = original(pixel_values)
    patched_output = patched(pixel_values)

    assert patched_output.shape == conv_output.shape
    assert torch.allclose(patched_output, conv_output, rtol=1e-5, atol=1e-5)


def test_real_qwen3_vl_patch_embed_linearization_matches_fp32_forward_and_grad() -> None:
    original, patched = _real_qwen_patch_pair(dtype=torch.float32, device="cpu")

    max_forward, max_input_grad, max_weight_grad, max_bias_grad = _compare_forward_and_grad(
        original,
        patched,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert max_forward <= 1e-4
    assert max_input_grad <= 1e-4
    assert max_weight_grad <= 1e-4
    assert max_bias_grad <= 1e-4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for bf16 parity")
def test_real_qwen3_vl_patch_embed_linearization_matches_cuda_bf16_forward_and_grad() -> None:
    original, patched = _real_qwen_patch_pair(dtype=torch.bfloat16, device="cuda")

    max_forward, max_input_grad, max_weight_grad, max_bias_grad = _compare_forward_and_grad(
        original,
        patched,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    )

    assert max_forward <= 0.05
    assert max_input_grad <= 0.05
    assert max_weight_grad <= 0.05
    assert max_bias_grad <= 0.05


def test_apply_qwen3_vl_patch_embed_linearization_replaces_module_and_receipts() -> None:
    model = FakeQwenModel()
    original_weight = model.model.visual.patch_embed.proj.weight

    receipt = apply_qwen3_vl_patch_embed_linearization(model)

    assert receipt.name == PATCH_EMBED_LINEARIZATION_NAME
    assert receipt.policy == "enabled"
    assert receipt.applied is True
    assert receipt.owner_path == "model.visual.patch_embed"
    assert receipt.original_class == "Qwen3VLVisionPatchEmbed"
    assert receipt.owner_class == "Qwen3VLVisionPatchEmbed"
    assert receipt.patched_class == "LinearizedQwen3VLPatchEmbed"
    assert receipt.weight_shape == tuple(original_weight.shape)
    assert receipt.bias is True
    assert receipt.kernel_size == (2, 4, 4)
    assert receipt.stride == (2, 4, 4)
    assert receipt.padding == (0, 0, 0)
    assert receipt.dilation == (1, 1, 1)
    assert receipt.groups == 1
    assert receipt.original_forward_sha256
    assert receipt.replacement_forward_sha256
    assert receipt.equivalence_probe is not None
    assert receipt.equivalence_probe["max_abs_diff"] <= 1e-4
    assert receipt.equivalence_probe["grad_max_abs_diff"] <= 1e-4
    assert isinstance(model.model.visual.patch_embed, LinearizedQwen3VLPatchEmbed)
    assert model.model.visual.patch_embed.proj.weight is original_weight
    assert "model.visual.patch_embed.proj.weight" in model.state_dict()


def test_apply_qwen3_vl_patch_embed_linearization_can_be_disabled_by_policy() -> None:
    model = FakeQwenModel()

    receipt = apply_qwen3_vl_patch_embed_linearization(model, policy="disabled")

    assert receipt.applied is False
    assert receipt.policy == "disabled"
    assert receipt.reason == "policy_disabled"
    assert isinstance(model.model.visual.patch_embed, Qwen3VLVisionPatchEmbed)


def test_apply_qwen3_vl_patch_embed_linearization_is_idempotent() -> None:
    model = FakeQwenModel()

    first = apply_qwen3_vl_patch_embed_linearization(model)
    second = apply_qwen3_vl_patch_embed_linearization(model)

    assert first.applied is True
    assert second.applied is False
    assert second.reason == "already_linearized"


@pytest.mark.parametrize(
    ("variant", "match"),
    (
        ("owner", "Qwen3VLVisionPatchEmbed owner"),
        ("stride", "kernel and stride"),
        ("padding", "zero Conv3d padding"),
        ("dilation", "unit Conv3d dilation"),
        ("groups", "ungrouped Conv3d"),
    ),
)
def test_apply_qwen3_vl_patch_embed_linearization_rejects_unsupported_topology(
    variant: str,
    match: str,
) -> None:
    patch_embed: nn.Module
    if variant == "owner":
        patch_embed = UnsupportedPatchEmbed()
    else:
        patch_embed = Qwen3VLVisionPatchEmbed(variant=variant)
    model = FakeQwenModel(patch_embed=patch_embed)

    with pytest.raises(QwenForwardContractError, match=match):
        apply_qwen3_vl_patch_embed_linearization(model)


class FakeQwenModel(nn.Module):
    def __init__(self, *, patch_embed: nn.Module | None = None) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.visual = nn.Module()
        self.model.visual.patch_embed = patch_embed or Qwen3VLVisionPatchEmbed()


class Qwen3VLVisionPatchEmbed(nn.Module):
    def __init__(
        self,
        *,
        variant: Literal["valid", "stride", "padding", "dilation", "groups"] = "valid",
    ) -> None:
        super().__init__()
        self.patch_size = 4
        self.temporal_patch_size = 2
        self.in_channels = 3
        self.embed_dim = 6 if variant == "groups" else 5
        self.flat_input_features = (
            self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size
        )
        stride = (1, 4, 4) if variant == "stride" else (2, 4, 4)
        padding = (0, 1, 1) if variant == "padding" else (0, 0, 0)
        dilation = (1, 2, 2) if variant == "dilation" else (1, 1, 1)
        groups = 3 if variant == "groups" else 1
        self.proj = nn.Conv3d(
            self.in_channels,
            self.embed_dim,
            kernel_size=(self.temporal_patch_size, self.patch_size, self.patch_size),
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        return self.proj(hidden_states).view(-1, self.embed_dim)


class UnsupportedPatchEmbed(Qwen3VLVisionPatchEmbed):
    pass


def _real_qwen_patch_pair(
    *,
    dtype: torch.dtype,
    device: str,
) -> tuple[nn.Module, LinearizedQwen3VLPatchEmbed]:
    from transformers import AutoConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLVisionPatchEmbed as UpstreamPatchEmbed,
    )

    vision_config = AutoConfig.from_pretrained(
        str(BASE_MODEL),
        local_files_only=True,
    ).vision_config
    original = UpstreamPatchEmbed(vision_config).to(device=device, dtype=dtype)
    linear_owner = UpstreamPatchEmbed(vision_config).to(device=device, dtype=dtype)
    linear_owner.load_state_dict(original.state_dict())
    patched = LinearizedQwen3VLPatchEmbed(linear_owner)
    return original, patched


def _compare_forward_and_grad(
    original: nn.Module,
    patched: LinearizedQwen3VLPatchEmbed,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[float, float, float, float]:
    torch.manual_seed(11)
    flat_features = patched.flat_input_features
    sample_count = 2
    source = torch.randn(
        sample_count,
        flat_features,
        dtype=dtype,
        device=device,
    )
    conv_input = source.detach().clone().requires_grad_(True)
    linear_input = source.detach().clone().requires_grad_(True)
    conv_output = original(conv_input)
    linear_output = patched(linear_input)
    max_forward = float((conv_output.float() - linear_output.float()).abs().max().item())
    conv_output.float().sum().backward()
    linear_output.float().sum().backward()
    max_input_grad = float(
        (conv_input.grad.float() - linear_input.grad.float()).abs().max().item()
    )
    max_weight_grad = float(
        (
            original.proj.weight.grad.float()
            - patched.proj.weight.grad.float()
        ).abs().max().item()
    )
    if original.proj.bias is None or patched.proj.bias is None:
        max_bias_grad = 0.0
    else:
        max_bias_grad = float(
            (
                original.proj.bias.grad.float()
                - patched.proj.bias.grad.float()
            ).abs().max().item()
        )
    return max_forward, max_input_grad, max_weight_grad, max_bias_grad
