"""Minimal torch.distributed mechanics with no research-level reduction policy."""

from __future__ import annotations

from typing import Any


def gather_objects(value: Any) -> list[Any]:
    """All-gather one Python object from each initialized distributed rank."""

    import torch.distributed as dist

    values: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    return values
