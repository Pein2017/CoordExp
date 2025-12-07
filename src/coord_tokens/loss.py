from __future__ import annotations

from typing import Any, List, Mapping, Sequence

import torch

from .codec import normalized_from_ints, tokens_to_ints


def coord_position_mask(
    input_ids: torch.Tensor, coord_token_mask: torch.Tensor
) -> torch.Tensor:
    """Return a boolean mask indicating positions that are coord tokens."""

    if coord_token_mask.dtype != torch.bool:
        coord_token_mask = coord_token_mask.bool()
    coord_token_mask = coord_token_mask.to(input_ids.device)
    return coord_token_mask[input_ids]


def restrict_logits_to_coords(
    logits: torch.Tensor,
    coord_token_mask: torch.Tensor,
    positions_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mask non-coord logits (optionally only at coord positions).

    Args:
        logits: [..., vocab]
        coord_token_mask: [vocab] boolean mask of coord token ids
        positions_mask: broadcastable boolean mask selecting positions to restrict
    """

    mask = coord_token_mask.to(device=logits.device, dtype=torch.bool)
    restricted = logits
    if positions_mask is not None:
        # Only apply restriction where positions_mask is True
        neg_inf = torch.tensor(-1e9, device=logits.device, dtype=logits.dtype)
        restricted = restricted.masked_fill(~mask, neg_inf)
        restricted = torch.where(positions_mask.unsqueeze(-1), restricted, logits)
    else:
        neg_inf = torch.tensor(-1e9, device=logits.device, dtype=logits.dtype)
        restricted = restricted.masked_fill(~mask, neg_inf)
    return restricted


def expectation_decode(
    logits: torch.Tensor, coord_token_mask: torch.Tensor
) -> torch.Tensor:
    """Expectation decode over coord tokens -> normalized float.

    Args:
        logits: [..., vocab]
        coord_token_mask: boolean mask of length vocab

    Returns:
        Tensor of shape logits[..., 0] (one value per position) in [0,1].
    """

    mask = coord_token_mask.to(device=logits.device, dtype=torch.bool)
    coord_values = torch.arange(mask.numel(), device=logits.device, dtype=logits.dtype)
    coord_values = coord_values[mask]
    coord_logits = logits[..., mask]
    probs = coord_logits.softmax(dim=-1)
    expected = (probs * coord_values).sum(dim=-1) / 1000.0
    return expected


def coord_targets_from_tokens(
    tokens: Sequence[Any],
    *,
    width: float | None = None,
    height: float | None = None,
) -> Mapping[str, List[float]]:
    """Convert coord tokens to numeric targets.

    Returns a mapping with keys:
      - "ints": integer bin values
      - "norm": normalized floats (k / 1000)
      - "pixel": pixel-space floats when width/height are provided
    """

    ints = tokens_to_ints(tokens, require_even=True)
    norm = normalized_from_ints(ints)
    pixel: List[float] = []
    if width is not None and height is not None:
        pixel = []
        for idx, v in enumerate(ints):
            denom = width if idx % 2 == 0 else height
            pixel.append(float(v) / 1000.0 * float(denom))
    return {"ints": ints, "norm": norm, "pixel": pixel}


__all__ = [
    "coord_position_mask",
    "restrict_logits_to_coords",
    "expectation_decode",
    "coord_targets_from_tokens",
]
