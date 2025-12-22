from __future__ import annotations

import math
from typing import Dict, Tuple

import torch

_GRID_CACHE: Dict[Tuple[int, str, str], torch.Tensor] = {}


def _grid_cache_key(mask_size: int, device: torch.device, dtype: torch.dtype) -> Tuple[int, str, str]:
    return (int(mask_size), str(device), str(dtype))


def get_soft_raster_grid(
    mask_size: int, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    if mask_size <= 0:
        raise ValueError("mask_size must be > 0")
    key = _grid_cache_key(mask_size, device, dtype)
    grid = _GRID_CACHE.get(key)
    if grid is not None:
        return grid
    size = int(mask_size)
    ys = (torch.arange(size, device=device, dtype=dtype) + 0.5) / float(size)
    xs = (torch.arange(size, device=device, dtype=dtype) + 0.5) / float(size)
    yy = ys.unsqueeze(1).repeat(1, size)
    xx = xs.unsqueeze(0).repeat(size, 1)
    grid = torch.stack([xx, yy], dim=-1).view(-1, 2)
    _GRID_CACHE[key] = grid
    return grid


def _point_segment_distance(
    points: torch.Tensor,
    v0: torch.Tensor,
    v1: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    seg = v1 - v0
    seg_len2 = (seg * seg).sum(dim=-1).clamp_min(eps)
    diff = points[:, None, :] - v0[None, :, :]
    t = (diff * seg[None, :, :]).sum(dim=-1) / seg_len2[None, :]
    t = t.clamp(0.0, 1.0)
    proj = v0[None, :, :] + t[..., None] * seg[None, :, :]
    return (points[:, None, :] - proj).pow(2).sum(dim=-1).sqrt()


def soft_polygon_mask(
    vertices: torch.Tensor,
    *,
    mask_size: int,
    sigma_mask: float,
    tau_inside: float,
    beta_dist: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    if sigma_mask <= 0:
        raise ValueError("sigma_mask must be > 0")
    if tau_inside <= 0:
        raise ValueError("tau_inside must be > 0")
    if beta_dist <= 0:
        raise ValueError("beta_dist must be > 0")
    if vertices.numel() == 0:
        size = int(mask_size)
        return torch.zeros((size, size), device=vertices.device, dtype=vertices.dtype)
    if not torch.isfinite(vertices).all().item():
        vertices = torch.nan_to_num(
            vertices, nan=0.0, posinf=1.0, neginf=0.0
        ).clamp(0.0, 1.0)

    grid = get_soft_raster_grid(
        int(mask_size), device=vertices.device, dtype=vertices.dtype
    )
    v0 = vertices
    v1 = torch.roll(vertices, shifts=-1, dims=0)

    a = v0[None, :, :] - grid[:, None, :]
    b = v1[None, :, :] - grid[:, None, :]
    cross = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    dot = a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]
    angles = torch.atan2(cross, dot + eps)
    winding = angles.sum(dim=1) / (2.0 * math.pi)
    inside = torch.sigmoid((winding.abs() - 0.5) / float(tau_inside))

    dist = _point_segment_distance(grid, v0, v1, eps=eps)
    soft_min = -torch.logsumexp(-float(beta_dist) * dist, dim=-1) / float(beta_dist)

    signed = (2.0 * inside - 1.0) * soft_min
    mask = torch.sigmoid(signed / float(sigma_mask))
    size = int(mask_size)
    return mask.view(size, size)


__all__ = ["get_soft_raster_grid", "soft_polygon_mask"]
