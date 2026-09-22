"""Literal native-input receipt identity, independent of any experiment.

The hashes preserve the historical raw NumPy byte representation. They are
not a replacement for model/config identity or a shape/dtype-aware tensor ID.
"""

import hashlib
from typing import Any

import torch


def tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def input_identity(batch: Any) -> dict[str, Any]:
    tensors = {
        name: tensor_hash(value)
        for name, value in sorted(batch.inputs.items())
        if isinstance(value, torch.Tensor)
    }
    return {
        "request_ids": list(batch.request_ids),
        "prompt_token_ids": [list(row) for row in batch.prompt_token_ids],
        "media_sha256": None if batch.media_sha256 is None else list(batch.media_sha256),
        "image_grids": [None if grid is None else list(grid) for grid in batch.image_grids],
        "tensor_sha256": tensors,
    }
