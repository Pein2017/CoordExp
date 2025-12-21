from __future__ import annotations

import math
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


def resolve_top_k(top_k: float | int, coord_vocab: int) -> int:
    """Resolve top_k fraction/count to an integer in [1, coord_vocab]."""

    try:
        top_k_val = float(top_k)
    except (TypeError, ValueError) as exc:
        raise ValueError("top_k must be numeric") from exc
    if top_k_val <= 0:
        raise ValueError("top_k must be > 0")
    if top_k_val < 1:
        k = int(math.ceil(top_k_val * coord_vocab))
    else:
        k = int(top_k_val)
    if k < 1:
        k = 1
    if k > coord_vocab:
        k = coord_vocab
    return k


def topk_expectation_decode(
    logits: torch.Tensor,
    coord_token_ids: Sequence[int] | torch.Tensor,
    *,
    top_k: float | int = 0.1,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Expectation decode over coord tokens using top-k logits.

    Args:
        logits: [..., vocab]
        coord_token_ids: ordered coord token ids for bins 0..999
        top_k: fraction (0<k<1) or integer count
        temperature: softmax temperature (>0)

    Returns:
        Tensor of shape logits[..., 0] (one value per position) in [0,1].
    """

    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    if isinstance(coord_token_ids, torch.Tensor):
        coord_ids = coord_token_ids.to(device=logits.device)
    else:
        coord_ids = torch.tensor(coord_token_ids, device=logits.device)

    if coord_ids.numel() == 0:
        return torch.zeros(logits.shape[:-1], device=logits.device, dtype=logits.dtype)

    coord_logits = logits.index_select(-1, coord_ids)
    if temperature != 1.0:
        coord_logits = coord_logits / float(temperature)

    coord_vocab = coord_logits.shape[-1]
    k = resolve_top_k(top_k, coord_vocab)

    topk_logits, topk_idx = torch.topk(coord_logits, k, dim=-1)
    probs = topk_logits.softmax(dim=-1)

    coord_values = torch.arange(
        coord_vocab, device=logits.device, dtype=logits.dtype
    )
    topk_values = coord_values[topk_idx]
    expected = (probs * topk_values).sum(dim=-1) / 1000.0
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
    "resolve_top_k",
    "topk_expectation_decode",
    "coord_targets_from_tokens",
]
