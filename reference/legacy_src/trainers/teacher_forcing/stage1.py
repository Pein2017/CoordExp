from __future__ import annotations

from typing import Sequence

import torch


def mask_stage1_coord_targets(
    labels: torch.Tensor,
    coord_token_ids: Sequence[int],
) -> torch.Tensor:
    if not isinstance(labels, torch.Tensor):
        raise TypeError("labels must be a torch.Tensor")

    if not coord_token_ids:
        return labels.clone()

    out = labels.clone()
    mask = torch.zeros_like(out, dtype=torch.bool)
    for tok_id in coord_token_ids:
        mask |= out.eq(int(tok_id))
    out[mask] = -100
    return out
