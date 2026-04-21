from __future__ import annotations

import torch


def pool_span_hidden_states(
    *,
    hidden_states: torch.Tensor,
    pooling: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    pooled: dict[str, torch.Tensor] = {}
    if "last_digit" in pooling:
        pooled["last_digit"] = hidden_states[:, -1, :]
    if "mean_digits" in pooling:
        pooled["mean_digits"] = hidden_states.mean(dim=1)
    return pooled


def representation_rsa(
    *,
    states: torch.Tensor,
    numeric_values: torch.Tensor,
) -> float:
    state_dist = torch.cdist(states.float(), states.float(), p=2).flatten()
    numeric_dist = torch.cdist(
        numeric_values.float().unsqueeze(1),
        numeric_values.float().unsqueeze(1),
        p=1,
    ).flatten()
    state_centered = state_dist - state_dist.mean()
    numeric_centered = numeric_dist - numeric_dist.mean()
    numerator = torch.sum(state_centered * numeric_centered)
    denominator = torch.sqrt(
        torch.sum(state_centered**2) * torch.sum(numeric_centered**2)
    )
    return float((numerator / denominator).item())
