"""Token-pooled sidecar training over detached teacher-forced caches."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from probes.coordinate_representation.coordinate_address_readout.bridge import CoordinateAddressReadout, HIDDEN_SIZE


REQUIRED_CACHE_KEYS = frozenset(
    {"coordinate_logits", "full_lse", "hidden", "visual", "roles", "targets", "grid"}
)
CHECK_TOLERANCE = 2e-4


def _require_cache(cache: Mapping[str, Any], index: int) -> dict[str, Any]:
    if not isinstance(cache, Mapping):
        raise TypeError(f"cache {index} must be a mapping")
    missing = REQUIRED_CACHE_KEYS - set(cache)
    if missing:
        raise ValueError(f"cache {index} missing fields: {sorted(missing)}")
    coordinate_logits = cache["coordinate_logits"]
    full_lse = cache["full_lse"]
    hidden = cache["hidden"]
    visual = cache["visual"]
    roles = cache["roles"]
    targets = cache["targets"]
    grid = cache["grid"]
    tensors = {
        "coordinate_logits": coordinate_logits,
        "full_lse": full_lse,
        "hidden": hidden,
        "visual": visual,
    }
    if any(not isinstance(value, torch.Tensor) for value in tensors.values()):
        raise TypeError(f"cache {index} tensor fields must be torch tensors")
    if any(value.requires_grad for value in tensors.values()):
        raise ValueError(f"cache {index} must contain detached frozen-backbone tensors")
    if coordinate_logits.ndim != 2 or coordinate_logits.shape[1] != 1000:
        raise ValueError(f"cache {index} coordinate_logits must have shape (N, 1000)")
    count = coordinate_logits.shape[0]
    if full_lse.shape != (count,):
        raise ValueError(f"cache {index} full_lse must have shape (N,)")
    if hidden.shape != (count, HIDDEN_SIZE):
        raise ValueError(f"cache {index} hidden must have shape (N, {HIDDEN_SIZE})")
    if roles.ndim != 1 or roles.shape != (count,) or roles.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"cache {index} roles must be an integer vector aligned with N")
    if targets.ndim != 1 or targets.shape != (count,) or targets.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"cache {index} targets must be an integer vector aligned with N")
    if count <= 0:
        raise ValueError(f"cache {index} must contain at least one coordinate target")
    if bool((roles < 0).any()) or bool((roles >= 4).any()):
        raise ValueError(f"cache {index} roles must be one of x1,y1,x2,y2")
    if bool((targets < 0).any()) or bool((targets >= 1000).any()):
        raise ValueError(f"cache {index} targets must be coordinate bins 0..999")
    if not isinstance(grid, Sequence) or len(grid) != 2:
        raise ValueError(f"cache {index} grid must be (Hm, Wm)")
    hm, wm = (int(value) for value in grid)
    if hm <= 0 or wm <= 0 or visual.shape != (hm * wm, HIDDEN_SIZE):
        raise ValueError(f"cache {index} visual/grid shape mismatch")
    if not all(torch.isfinite(value).all() for value in tensors.values()):
        raise ValueError(f"cache {index} contains nonfinite values")
    return {
        "coordinate_logits": coordinate_logits,
        "full_lse": full_lse,
        "hidden": hidden,
        "visual": visual,
        "roles": roles,
        "targets": targets,
        "grid": (hm, wm),
    }


def _token_nll(sidecar: CoordinateAddressReadout, cache: Mapping[str, Any]) -> torch.Tensor:
    hm, wm = cache["grid"]
    adjusted = sidecar.adjust_coordinate_logits(
        cache["coordinate_logits"],
        cache["hidden"],
        cache["visual"],
        cache["roles"],
        hm,
        wm,
    )
    return cache["full_lse"] - adjusted.gather(1, cache["targets"][:, None]).squeeze(1)


def _gradient_norms(sidecar: CoordinateAddressReadout) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, parameter in sidecar.named_parameters():
        if parameter.grad is None:
            raise RuntimeError(f"missing gradient for sidecar parameter {name}")
        value = float(parameter.grad.detach().norm())
        if not torch.isfinite(torch.tensor(value)):
            raise RuntimeError(f"nonfinite gradient for sidecar parameter {name}")
        result[name] = value
    return result


def batch_backward(
    sidecar: CoordinateAddressReadout,
    caches: Sequence[Mapping[str, Any]],
    *,
    check: bool = False,
) -> dict[str, Any]:
    """Backprop one fixed eight-image token-pooled batch.

    The caller owns AdamW and its update.  This helper clears only sidecar
    gradients, computes ``sum(token NLL) / sum(token count)``, and calls one
    backward.  ``check=True`` builds an independent concatenated-token graph and
    compares its autograd gradients, also showing the unequal-count image-mean
    mistake explicitly.
    """

    if not isinstance(sidecar, CoordinateAddressReadout):
        raise TypeError("sidecar must be CoordinateAddressReadout")
    if len(caches) != 8:
        raise ValueError(f"expected exactly eight image caches, got {len(caches)}")
    checked = [_require_cache(cache, index) for index, cache in enumerate(caches)]
    counts = [int(cache["targets"].numel()) for cache in checked]
    denominator = sum(counts)

    # Sequential image accumulation gives one pooled numerator while retaining
    # the explicit per-image graph shape used by the production caller.
    numerator: torch.Tensor | None = None
    for cache in checked:
        token_sum = _token_nll(sidecar, cache).sum()
        numerator = token_sum if numerator is None else numerator + token_sum
    assert numerator is not None
    loss = numerator / denominator

    comparison: dict[str, Any] | None = None
    if check:
        reference_tokens = torch.cat([_token_nll(sidecar, cache) for cache in checked], dim=0)
        reference_loss = reference_tokens.sum() / denominator
        parameters = tuple(sidecar.parameters())
        reference_gradients = torch.autograd.grad(reference_loss, parameters, allow_unused=False)

        wrong_tokens = [_token_nll(sidecar, cache) for cache in checked]
        wrong_image_mean = torch.stack([tokens.mean() for tokens in wrong_tokens]).mean()
        wrong_gradients = torch.autograd.grad(wrong_image_mean, parameters, allow_unused=False)
    else:
        reference_loss = None
        reference_gradients = ()
        wrong_image_mean = None
        wrong_gradients = ()

    sidecar.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norms = _gradient_norms(sidecar)

    if check:
        actual_gradients = tuple(parameter.grad.detach() for parameter in sidecar.parameters())
        max_gradient_delta = max(
            float((actual - reference).abs().max())
            for actual, reference in zip(actual_gradients, reference_gradients, strict=True)
        )
        wrong_gradient_delta = max(
            float((wrong - reference).abs().max())
            for wrong, reference in zip(wrong_gradients, reference_gradients, strict=True)
        )
        unequal_counts = len(set(counts)) > 1
        wrong_rejected = unequal_counts and (
            abs(float(wrong_image_mean) - float(reference_loss)) > CHECK_TOLERANCE
            or wrong_gradient_delta > CHECK_TOLERANCE
        )
        if unequal_counts and not wrong_rejected:
            raise AssertionError("unweighted image-mean denominator was not rejected")
        comparison = {
            "reference_loss": float(reference_loss.detach()),
            "loss_abs_diff": abs(float(loss.detach()) - float(reference_loss.detach())),
            "max_abs_gradient_diff": max_gradient_delta,
            "wrong_image_mean_loss": float(wrong_image_mean.detach()),
            "wrong_image_mean_max_abs_gradient_diff": wrong_gradient_delta,
            "unequal_image_counts": unequal_counts,
            "wrong_image_mean_rejected": wrong_rejected,
            "tolerance": CHECK_TOLERANCE,
        }
        if comparison["loss_abs_diff"] > CHECK_TOLERANCE or max_gradient_delta > CHECK_TOLERANCE:
            raise AssertionError("pooled loss/gradient differs from independent token reference")

    return {
        "loss": float(loss.detach()),
        "denominator": denominator,
        "image_token_counts": counts,
        "gradient_norms": gradient_norms,
        "check": comparison,
    }


__all__ = ["batch_backward"]
