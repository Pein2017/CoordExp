"""Coordinate/regression-family token masks."""

from __future__ import annotations

from typing import Iterable

import torch


def build_token_id_mask(
    token_ids: Iterable[int], *, vocab_size: int, device: torch.device | None = None
) -> torch.Tensor:
    mask = torch.zeros(int(vocab_size), dtype=torch.bool, device=device)
    for token_id in token_ids:
        token_id_i = int(token_id)
        if 0 <= token_id_i < mask.numel():
            mask[token_id_i] = True
    return mask


def build_coord_loss_mask(labels: torch.Tensor, coord_geometry_ids: Iterable[int]) -> torch.Tensor:
    coord_ids = torch.as_tensor(
        tuple(int(token_id) for token_id in coord_geometry_ids),
        dtype=labels.dtype,
        device=labels.device,
    )
    if coord_ids.numel() == 0:
        return torch.zeros_like(labels, dtype=torch.bool)
    return torch.isin(labels, coord_ids)


__all__ = ["build_token_id_mask", "build_coord_loss_mask"]
