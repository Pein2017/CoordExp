"""Owned Qwen runtime patches that preserve upstream semantics."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from src.common.errors import QwenForwardContractError


PATCH_EMBED_LINEARIZATION_NAME = "qwen3_vl_patch_embed_linearization"


@dataclass(frozen=True)
class QwenRuntimePatchReceipt:
    name: str
    policy: str
    applied: bool
    reason: str
    owner_path: str | None = None
    original_class: str | None = None
    owner_class: str | None = None
    patched_class: str | None = None
    projection_class: str | None = None
    original_forward_sha256: str | None = None
    replacement_forward_sha256: str | None = None
    in_channels: int | None = None
    temporal_patch_size: int | None = None
    patch_size: int | None = None
    embed_dim: int | None = None
    weight_shape: tuple[int, ...] | None = None
    bias: bool | None = None
    kernel_size: tuple[int, ...] | None = None
    stride: tuple[int, ...] | None = None
    padding: tuple[int, ...] | None = None
    dilation: tuple[int, ...] | None = None
    groups: int | None = None
    equivalence_probe: dict[str, Any] | None = None

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "policy": self.policy,
            "applied": self.applied,
            "reason": self.reason,
            "owner_path": self.owner_path,
            "original_class": self.original_class,
            "owner_class": self.owner_class,
            "patched_class": self.patched_class,
            "projection_class": self.projection_class,
            "original_forward_sha256": self.original_forward_sha256,
            "replacement_forward_sha256": self.replacement_forward_sha256,
            "in_channels": self.in_channels,
            "temporal_patch_size": self.temporal_patch_size,
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
            "weight_shape": None if self.weight_shape is None else list(self.weight_shape),
            "bias": self.bias,
            "kernel_size": None if self.kernel_size is None else list(self.kernel_size),
            "stride": None if self.stride is None else list(self.stride),
            "padding": None if self.padding is None else list(self.padding),
            "dilation": None if self.dilation is None else list(self.dilation),
            "groups": self.groups,
            "equivalence_probe": self.equivalence_probe,
        }


class LinearizedQwen3VLPatchEmbed(nn.Module):
    """Qwen3-VL patch embedding as the equivalent flattened projection.

    Upstream Qwen3-VL uses a Conv3d whose kernel and stride cover exactly one
    pre-extracted image patch. For Qwen's flattened `pixel_values`, that Conv3d
    is algebraically equivalent to a single linear projection with the same
    weights and bias, while avoiding a pathological Conv3d kernel choice in the
    current Torch stack.
    """

    def __init__(self, original_patch_embed: Any) -> None:
        super().__init__()
        self.patch_size = int(original_patch_embed.patch_size)
        self.temporal_patch_size = int(original_patch_embed.temporal_patch_size)
        self.in_channels = int(original_patch_embed.in_channels)
        self.embed_dim = int(original_patch_embed.embed_dim)
        self.proj = original_patch_embed.proj
        self.flat_input_features = (
            self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(-1, self.flat_input_features)
        weight = self.proj.weight.view(self.proj.out_channels, -1)
        hidden_states = F.linear(hidden_states.to(dtype=target_dtype), weight, self.proj.bias)
        return hidden_states.view(-1, self.embed_dim)


def apply_qwen3_vl_patch_embed_linearization(
    model: Any,
    *,
    policy: str = "enabled",
) -> QwenRuntimePatchReceipt:
    visual = getattr(getattr(model, "model", None), "visual", None)
    patch_embed = getattr(visual, "patch_embed", None)
    owner_path = "model.visual.patch_embed"
    if policy == "disabled":
        return QwenRuntimePatchReceipt(
            name=PATCH_EMBED_LINEARIZATION_NAME,
            policy=policy,
            applied=False,
            reason="policy_disabled",
            owner_path=owner_path,
        )
    if policy != "enabled":
        raise QwenForwardContractError(
            "unsupported Qwen runtime patch policy",
            code="qwen.runtime_patch_policy",
            context={"patch": PATCH_EMBED_LINEARIZATION_NAME, "policy": policy},
        )
    if patch_embed is None:
        return QwenRuntimePatchReceipt(
            name=PATCH_EMBED_LINEARIZATION_NAME,
            policy=policy,
            applied=False,
            reason="model_visual_patch_embed_missing",
            owner_path=owner_path,
        )
    if isinstance(patch_embed, LinearizedQwen3VLPatchEmbed):
        return _patch_receipt(
            patch_embed,
            policy=policy,
            applied=False,
            reason="already_linearized",
            owner_path=owner_path,
            original_class=type(patch_embed).__name__,
            equivalence_probe=None,
            original_forward_sha256=None,
        )
    _validate_qwen3_vl_patch_embed(patch_embed)
    original_forward_sha256 = _callable_source_sha256(type(patch_embed).forward)
    equivalence_probe = _run_patch_embed_equivalence_probe(patch_embed)
    patched = LinearizedQwen3VLPatchEmbed(patch_embed)
    visual.patch_embed = patched
    return _patch_receipt(
        patched,
        policy=policy,
        applied=True,
        reason="conv3d_kernel_stride_equivalent_linear_projection",
        owner_path=owner_path,
        original_class=type(patch_embed).__name__,
        equivalence_probe=equivalence_probe,
        original_forward_sha256=original_forward_sha256,
    )


def model_not_loaded_patch_receipts(
    *,
    patch_policy: str = "enabled",
) -> dict[str, dict[str, Any]]:
    return {
        PATCH_EMBED_LINEARIZATION_NAME: QwenRuntimePatchReceipt(
            name=PATCH_EMBED_LINEARIZATION_NAME,
            policy=patch_policy,
            applied=False,
            reason="model_not_loaded",
        ).to_artifact_dict()
    }


def _validate_qwen3_vl_patch_embed(patch_embed: Any) -> None:
    if type(patch_embed).__name__ != "Qwen3VLVisionPatchEmbed":
        raise QwenForwardContractError(
            "Qwen3-VL patch embed linearization requires the upstream Qwen3VLVisionPatchEmbed owner",
            code="qwen.patch_embed_linearization_owner",
            context={"owner_class": type(patch_embed).__name__},
        )
    proj = getattr(patch_embed, "proj", None)
    if not isinstance(proj, nn.Conv3d):
        raise QwenForwardContractError(
            "Qwen3-VL patch embed linearization requires a Conv3d projection",
            code="qwen.patch_embed_linearization_projection",
            context={"projection_class": type(proj).__name__},
        )
    kernel_size = tuple(int(item) for item in proj.kernel_size)
    stride = tuple(int(item) for item in proj.stride)
    expected_kernel = (
        int(getattr(patch_embed, "temporal_patch_size", -1)),
        int(getattr(patch_embed, "patch_size", -1)),
        int(getattr(patch_embed, "patch_size", -1)),
    )
    if kernel_size != expected_kernel or stride != expected_kernel:
        raise QwenForwardContractError(
            "Qwen3-VL patch embed linearization requires kernel and stride to match one full patch",
            code="qwen.patch_embed_linearization_shape",
            context={
                "kernel_size": list(kernel_size),
                "stride": list(stride),
                "expected_kernel": list(expected_kernel),
            },
        )
    padding = tuple(int(item) for item in proj.padding)
    dilation = tuple(int(item) for item in proj.dilation)
    if padding != (0, 0, 0):
        raise QwenForwardContractError(
            "Qwen3-VL patch embed linearization requires zero Conv3d padding",
            code="qwen.patch_embed_linearization_padding",
            context={"padding": list(padding)},
        )
    if dilation != (1, 1, 1):
        raise QwenForwardContractError(
            "Qwen3-VL patch embed linearization requires unit Conv3d dilation",
            code="qwen.patch_embed_linearization_dilation",
            context={"dilation": list(dilation)},
        )
    if int(proj.groups) != 1:
        raise QwenForwardContractError(
            "Qwen3-VL patch embed linearization requires an ungrouped Conv3d",
            code="qwen.patch_embed_linearization_groups",
            context={"groups": int(proj.groups)},
        )
    if int(proj.in_channels) != int(getattr(patch_embed, "in_channels", -1)):
        raise QwenForwardContractError(
            "Qwen3-VL patch embed in_channels mismatch",
            code="qwen.patch_embed_linearization_channels",
            context={
                "proj_in_channels": int(proj.in_channels),
                "patch_embed_in_channels": getattr(patch_embed, "in_channels", None),
            },
        )
    if int(proj.out_channels) != int(getattr(patch_embed, "embed_dim", -1)):
        raise QwenForwardContractError(
            "Qwen3-VL patch embed output channels must match embed_dim",
            code="qwen.patch_embed_linearization_embed_dim",
            context={
                "proj_out_channels": int(proj.out_channels),
                "patch_embed_embed_dim": getattr(patch_embed, "embed_dim", None),
            },
        )


def _run_patch_embed_equivalence_probe(patch_embed: Any) -> dict[str, Any]:
    proj = patch_embed.proj
    sample_count = 2
    flat_features = (
        int(patch_embed.in_channels)
        * int(patch_embed.temporal_patch_size)
        * int(patch_embed.patch_size)
        * int(patch_embed.patch_size)
    )
    x_values = torch.linspace(-0.5, 0.5, steps=sample_count * flat_features, dtype=torch.float32)
    x_conv = x_values.view(
        sample_count,
        int(patch_embed.in_channels),
        int(patch_embed.temporal_patch_size),
        int(patch_embed.patch_size),
        int(patch_embed.patch_size),
    ).detach().requires_grad_(True)
    x_linear = x_values.view(sample_count, flat_features).detach().requires_grad_(True)
    weight_conv = proj.weight.detach().to(dtype=torch.float32, device="cpu").clone().requires_grad_(True)
    weight_linear = proj.weight.detach().to(dtype=torch.float32, device="cpu").clone().requires_grad_(True)
    bias_conv = (
        None
        if proj.bias is None
        else proj.bias.detach().to(dtype=torch.float32, device="cpu").clone().requires_grad_(True)
    )
    bias_linear = (
        None
        if proj.bias is None
        else proj.bias.detach().to(dtype=torch.float32, device="cpu").clone().requires_grad_(True)
    )
    conv_out = F.conv3d(
        x_conv,
        weight_conv,
        bias_conv,
        stride=tuple(int(item) for item in proj.stride),
        padding=tuple(int(item) for item in proj.padding),
        dilation=tuple(int(item) for item in proj.dilation),
        groups=int(proj.groups),
    ).view(sample_count, int(patch_embed.embed_dim))
    linear_out = F.linear(
        x_linear,
        weight_linear.view(int(proj.out_channels), -1),
        bias_linear,
    )
    max_abs_diff = float((conv_out - linear_out).abs().max().item())
    conv_out.sum().backward()
    linear_out.sum().backward()
    grad_diffs = [
        float((x_conv.grad.view_as(x_linear) - x_linear.grad).abs().max().item()),
        float((weight_conv.grad - weight_linear.grad).abs().max().item()),
    ]
    if bias_conv is not None and bias_linear is not None:
        grad_diffs.append(float((bias_conv.grad - bias_linear.grad).abs().max().item()))
    grad_max_abs_diff = max(grad_diffs)
    return {
        "scope": "cpu_float32_forward_and_grad_small_canary",
        "sample_count": sample_count,
        "input_features": flat_features,
        "max_abs_diff": max_abs_diff,
        "grad_max_abs_diff": grad_max_abs_diff,
    }


def _patch_receipt(
    patch_embed: LinearizedQwen3VLPatchEmbed,
    *,
    policy: str,
    applied: bool,
    reason: str,
    owner_path: str,
    original_class: str,
    equivalence_probe: dict[str, Any] | None,
    original_forward_sha256: str | None,
) -> QwenRuntimePatchReceipt:
    proj = patch_embed.proj
    return QwenRuntimePatchReceipt(
        name=PATCH_EMBED_LINEARIZATION_NAME,
        policy=policy,
        applied=applied,
        reason=reason,
        owner_path=owner_path,
        original_class=original_class,
        owner_class=original_class,
        patched_class=type(patch_embed).__name__,
        projection_class=type(proj).__name__,
        original_forward_sha256=original_forward_sha256,
        replacement_forward_sha256=_callable_source_sha256(
            LinearizedQwen3VLPatchEmbed.forward
        ),
        in_channels=patch_embed.in_channels,
        temporal_patch_size=patch_embed.temporal_patch_size,
        patch_size=patch_embed.patch_size,
        embed_dim=patch_embed.embed_dim,
        weight_shape=tuple(int(item) for item in proj.weight.shape),
        bias=proj.bias is not None,
        kernel_size=tuple(int(item) for item in proj.kernel_size),
        stride=tuple(int(item) for item in proj.stride),
        padding=tuple(int(item) for item in proj.padding),
        dilation=tuple(int(item) for item in proj.dilation),
        groups=int(proj.groups),
        equivalence_probe=equivalence_probe,
    )


def _callable_source_sha256(callable_obj: Any) -> str:
    try:
        source = inspect.getsource(callable_obj)
    except (OSError, TypeError):
        source = repr(callable_obj)
    return _source_sha256(source)


def _source_sha256(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()
