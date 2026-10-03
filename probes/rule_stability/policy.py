"""Rule-stability diagnostics and compatible imports of the shared median policy."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from src.losses.token_scores import aligned_token_logprobs
from src.qwen.coordinate_policy import (
    MedianPolicy,
    _coordinate_ids,
    _MedianScoreTransform,
    scale_coordinate_logits as _scale_coordinates,
)


class TechnicalSuffixSelection:
    """Six-action qualification selection, recording scores before forcing.

    This processor follows the ordinary median processor. Its forced-selection
    likelihood is separate from the unforced normalized policy likelihood and
    never supplies a sampled training trajectory.
    """

    def __init__(self, action_ids: Sequence[int], prompt_length: int):
        self.action_ids = tuple(action_ids)
        if (len(self.action_ids) != 6 or len(set(self.action_ids)) != 6
                or any(type(token) is not int or token < 0 for token in self.action_ids)
                or type(prompt_length) is not int or prompt_length < 1):
            raise ValueError("technical suffix requires six distinct IDs and one prompt")
        self.prompt_length = prompt_length
        self.unforced_policy_logprobs: list[float] = []

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        step = len(self.unforced_policy_logprobs)
        if (input_ids.ndim != 2 or input_ids.shape != (1, self.prompt_length + step)
                or scores.ndim != 2 or scores.shape[0] != 1 or step >= 6
                or tuple(input_ids[0, self.prompt_length:].tolist()) != self.action_ids[:step]
                or scores.shape[1] <= max(self.action_ids)
                or not bool(torch.isfinite(scores).all())):
            raise ValueError("technical cached action/score alignment or support differs")
        target = self.action_ids[step]
        value = aligned_token_logprobs(scores, torch.tensor([target], device=scores.device))[0]
        self.unforced_policy_logprobs.append(float(value.detach()))
        forced = torch.full_like(scores, -torch.inf)
        forced[0, target] = 0.
        return forced


def replay_difference(
    *, request_id: str, token_ids: Sequence[int],
    behavior_raw_logprobs: Sequence[float], behavior_policy_logprobs: Sequence[float],
    replay_raw_logprobs: torch.Tensor, replay_policy_logprobs: torch.Tensor,
) -> dict[str, Any]:
    """Retain per-action numeric differences without replacing behavior scores.

    The caller supplies already causally aligned selected-action log probabilities
    including actual EOS. Both channels remain explicit; no backend tolerance or
    scientific acceptance decision is imposed by this measurement.
    """
    ids = tuple(token_ids)
    if not request_id or any(type(value) is not int or value < 0 for value in ids):
        raise ValueError("replay difference requires a request ID and literal action IDs")
    result: dict[str, Any] = dict(request_id=request_id, actions=len(ids), token_ids=list(ids))
    for name, behavior_values, replay_values in (
        ("raw", behavior_raw_logprobs, replay_raw_logprobs),
        ("policy", behavior_policy_logprobs, replay_policy_logprobs),
    ):
        behavior = torch.as_tensor(behavior_values, dtype=torch.float64)
        replay = replay_values.detach().to(device="cpu", dtype=torch.float64)
        if behavior.shape != (len(ids),) or replay.shape != (len(ids),):
            raise ValueError("behavior and replay likelihoods must align with every action")
        if not bool(torch.isfinite(behavior).all()) or not bool(torch.isfinite(replay).all()):
            raise ValueError("behavior and replay likelihoods must be finite")
        differences = replay - behavior
        result[name] = {
            "behavior_logprobs": behavior.tolist(),
            "replay_logprobs": replay.tolist(),
            "delta_replay_minus_behavior": differences.tolist(),
            "max_abs_delta": float(differences.abs().max()) if len(ids) else 0.0,
            "mean_abs_delta": float(differences.abs().mean()) if len(ids) else 0.0,
        }
    return result
