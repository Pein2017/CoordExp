"""The current-weight median policy, shared by generation and gradient replay.

The normalization arithmetic follows ``vllm_dora_model``: BF16 base rows plus
independent FP32 output deltas, an FP32 effective sum, FP64 norms/lower median,
and FP64 score multiplication followed by a cast to the input score dtype.
HF generation promotes scores to FP32 before processors; replay does the same.
No forward hook changes the raw-likelihood channel or the model's output head.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from src.losses.token_scores import aligned_token_logprobs


def _coordinate_ids(values: Sequence[int] | torch.Tensor) -> tuple[int, ...]:
    if isinstance(values, torch.Tensor):
        if values.ndim != 1 or values.dtype != torch.long:
            raise ValueError("coordinate IDs must be a one-dimensional long vector")
        values = values.detach().cpu().tolist()
    ids = tuple(values)
    if (len(ids) != 1000 or any(type(value) is not int or value < 0 for value in ids)
            or len(set(ids)) != 1000):
        raise ValueError("coordinate IDs must contain 1,000 unique nonnegative integers")
    return ids


def _scale_coordinates(
    logits: torch.Tensor, ids: torch.Tensor, factors: torch.Tensor,
) -> torch.Tensor:
    if logits.ndim < 1 or not logits.is_floating_point() or logits.shape[-1] <= int(ids.max()):
        raise ValueError("policy scores must have a floating vocabulary covering coordinate IDs")
    ids = ids.to(logits.device)
    scaled = (logits.index_select(-1, ids).to(torch.float64)
              * factors.to(device=logits.device, dtype=torch.float64)).to(logits.dtype)
    # An out-of-place operation preserves HF's captured raw logits and autograd.
    return logits.index_copy(-1, ids, scaled)


@dataclass(frozen=True)
class _MedianScoreTransform:
    coordinate_ids: torch.Tensor
    factors: torch.Tensor

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        return _scale_coordinates(scores, self.coordinate_ids, self.factors)


class MedianPolicy:
    """Bind an untied selected-output head without modifying model computation.

    ``generation_transform()`` snapshots factors from the current weights once
    per acquisition. Pass its result in ``generate_continuations``' optional
    ``logits_processor`` list. ``transform_replay`` recomputes graph-connected
    factors for each replay graph, after the same FP32 promotion used by HF.
    Causal row selection and target/action reduction remain caller-owned.
    """

    def __init__(self, model: Any, coordinate_ids: Sequence[int] | torch.Tensor) -> None:
        ids = _coordinate_ids(coordinate_ids)
        self.head = model.get_output_embeddings()
        self.input_head = model.get_input_embeddings()
        self.coordinate_ids = torch.tensor(ids, dtype=torch.long)
        selected = getattr(self.head, "selected_token_ids", None)
        if not isinstance(selected, torch.Tensor) or selected.ndim != 1 or selected.dtype != torch.long:
            raise ValueError("median policy requires the maintained selected output-delta head")
        selected_ids = selected.detach().cpu().tolist()
        if len(set(selected_ids)) != len(selected_ids):
            raise ValueError("selected output token IDs must be unique")
        rows = {token_id: row for row, token_id in enumerate(selected_ids)}
        if any(token_id not in rows for token_id in ids):
            raise ValueError("coordinate IDs are missing selected output delta rows")
        self.coordinate_rows = torch.tensor([rows[token_id] for token_id in ids], dtype=torch.long)
        self._validate_head()

    def _validate_head(self) -> tuple[torch.Tensor, torch.Tensor]:
        weight = getattr(self.head, "weight", None)
        delta = getattr(self.head, "shared_embed_delta", None)
        if (not isinstance(weight, torch.Tensor) or weight.ndim != 2
                or weight.dtype != torch.bfloat16 or weight.requires_grad):
            raise ValueError("median policy requires a frozen BF16 base output head")
        if getattr(self.head, "bias", None) is not None:
            raise ValueError("median policy does not support an output-head bias")
        if (not isinstance(delta, torch.Tensor) or delta.ndim != 2
                or delta.dtype != torch.float32 or delta.shape[1] != weight.shape[1]
                or delta.shape[0] != self.head.selected_token_ids.numel()
                or not bool(torch.isfinite(delta).all())):
            raise ValueError("median policy requires finite FP32 selected output deltas")
        input_delta = getattr(self.input_head, "shared_embed_delta", None)
        if (not isinstance(input_delta, torch.Tensor) or input_delta.dtype != torch.float32
                or input_delta.shape != delta.shape
                or input_delta.untyped_storage().data_ptr() == delta.untyped_storage().data_ptr()):
            raise ValueError("median policy requires independent input/output deltas")
        if int(self.coordinate_ids.max()) >= weight.shape[0]:
            raise ValueError("coordinate IDs lie outside the output vocabulary")
        return weight, delta

    def factors(self) -> torch.Tensor:
        """Return current FP64 factors without detaching the output-weight graph."""
        weight, delta = self._validate_head()
        ids = self.coordinate_ids.to(weight.device)
        rows = self.coordinate_rows.to(weight.device)
        effective = (weight.index_select(0, ids).to(torch.float32)
                     + delta.to(device=weight.device, dtype=torch.float32).index_select(0, rows))
        norms = torch.linalg.vector_norm(effective.to(torch.float64), ord=2, dim=1)
        if not bool(torch.isfinite(norms).all()) or bool((norms <= 0).any()):
            raise ValueError("coordinate output row norms must be finite and positive")
        factors = torch.median(norms) / norms
        if not bool(torch.isfinite(factors).all()) or bool((factors <= 0).any()):
            raise ValueError("coordinate output norm factors must be finite and positive")
        return factors

    def transform_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply exact normalization/cast semantics, preserving the input dtype."""
        return _scale_coordinates(logits, self.coordinate_ids, self.factors())

    def transform_replay(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply the HF score policy with derivatives through current factors."""
        return self.transform_logits(logits.float())

    def generation_transform(self) -> _MedianScoreTransform:
        """Capture one current-model score transform; use a fresh one after updates."""
        with torch.no_grad():
            factors = self.factors().detach().clone()
        return _MedianScoreTransform(self.coordinate_ids.clone(), factors)


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
