"""The current-weight median policy, shared by generation and gradient replay.

The normalization arithmetic uses BF16 base rows plus
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


def scale_coordinate_logits(
    logits: torch.Tensor, ids: torch.Tensor, factors: torch.Tensor,
    *, inplace: bool = False,
) -> torch.Tensor:
    """Scale coordinates in FP64, then cast to the input score dtype.

    HF uses the out-of-place default to preserve raw logits and autograd. The
    resident backend may write in place after capturing its raw-logits clone.
    That path requires IDs validated at binding; native index operations enforce
    bounds without synchronizing device scalars on each decode step.
    """
    if (logits.ndim < 1 or not logits.is_floating_point()
            or (not inplace and logits.shape[-1] <= int(ids.max()))):
        raise ValueError("policy scores must have a floating vocabulary covering coordinate IDs")
    ids = ids.to(logits.device)
    scaled = (logits.index_select(-1, ids).to(torch.float64)
              * factors.to(device=logits.device, dtype=torch.float64)).to(logits.dtype)
    if inplace:
        return logits.index_copy_(-1, ids, scaled)
    return logits.index_copy(-1, ids, scaled)


def coordinate_norm_values(
    weight: torch.Tensor, delta: torch.Tensor, ids: torch.Tensor,
    selected_rows: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute current coordinate norms and lower-median factors without detaching.

    ``ids`` addresses vocabulary rows; ``selected_rows`` maps those same IDs to
    independent output-delta rows. Model binding, bias and input/output alias
    checks remain with the caller. The effective addition is FP32, followed by
    FP64 norms, lower median and division; all returned tensors retain the graph.
    """
    if (not isinstance(weight, torch.Tensor) or weight.ndim != 2
            or weight.dtype != torch.bfloat16):
        raise ValueError("coordinate output norms require a BF16 language head")
    if (not isinstance(delta, torch.Tensor) or delta.ndim != 2
            or delta.shape[1] != weight.shape[1] or delta.dtype != torch.float32
            or not bool(torch.isfinite(delta).all())):
        raise ValueError("coordinate output norm delta must be finite FP32 rows")
    if (not isinstance(ids, torch.Tensor) or ids.ndim != 1 or ids.dtype != torch.long
            or ids.numel() != 1000 or int(ids.min()) < 0 or int(ids.max()) >= weight.shape[0]):
        raise ValueError("coordinate output norm IDs are outside the language head")
    if (not isinstance(selected_rows, torch.Tensor) or selected_rows.dtype != torch.long
            or selected_rows.shape != ids.shape or bool((selected_rows < 0).any())
            or int(selected_rows.max()) >= delta.shape[0]):
        raise ValueError("coordinate IDs are missing selected output delta rows")
    ids = ids.to(weight.device)
    rows = selected_rows.to(weight.device)
    effective = (weight.index_select(0, ids).to(torch.float32)
                 + delta.to(device=weight.device, dtype=torch.float32).index_select(0, rows))
    norms = torch.linalg.vector_norm(effective.to(torch.float64), ord=2, dim=1)
    if not bool(torch.isfinite(norms).all()) or bool((norms <= 0).any()):
        raise ValueError("coordinate output row norms must be finite and positive")
    median = torch.median(norms)
    factors = median / norms
    if not bool(torch.isfinite(factors).all()) or bool((factors <= 0).any()):
        raise ValueError("coordinate output norm factors must be finite and positive")
    return {"factors": factors, "norms": norms, "median": median}


@dataclass(frozen=True)
class _MedianScoreTransform:
    coordinate_ids: torch.Tensor
    factors: torch.Tensor

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        return scale_coordinate_logits(scores, self.coordinate_ids, self.factors)


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
        return coordinate_norm_values(
            weight, delta, self.coordinate_ids, self.coordinate_rows,
        )["factors"]

    def transform_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply exact normalization/cast semantics, preserving the input dtype."""
        return scale_coordinate_logits(logits, self.coordinate_ids, self.factors())

    def transform_replay(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply the HF score policy with derivatives through current factors."""
        return self.transform_logits(logits.float())

    def generation_transform(self) -> _MedianScoreTransform:
        """Capture one current-model score transform; use a fresh one after updates."""
        with torch.no_grad():
            factors = self.factors().detach().clone()
        return _MedianScoreTransform(self.coordinate_ids.clone(), factors)
