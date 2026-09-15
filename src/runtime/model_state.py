"""Small deterministic model-state descriptions used by research execution paths."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

import torch


def parameter_layout(
    named: Sequence[tuple[str, torch.nn.Parameter]],
) -> list[dict[str, Any]]:
    """Describe an ordered trainable surface without assigning optimization meaning."""

    return [
        {
            "name": name,
            "shape": list(parameter.shape),
            "numel": parameter.numel(),
            "dtype": str(parameter.dtype),
        }
        for name, parameter in named
    ]


def tensor_state_sha256(named: Sequence[tuple[str, torch.Tensor]]) -> str:
    """Hash ordered tensor names, shapes, dtypes, and exact CPU bytes."""

    digest = hashlib.sha256()
    for name, tensor in named:
        value = tensor.detach().to(device="cpu").contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(memoryview(value.numpy()))
    return digest.hexdigest()
