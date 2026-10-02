"""CPU-testable geometry regions and full-vocabulary max ranking."""
from __future__ import annotations

import math
from numbers import Real


def _real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _grid(max_bin: int) -> int:
    if type(max_bin) is not int:
        raise TypeError("max_bin must be a Python integer")
    if not 1 <= max_bin <= 999:
        raise ValueError("max_bin must be in 1..999")
    return max_bin


def _box(value, name: str, max_bin: int) -> tuple[int, int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{name} must contain four xyxy bins")
    if any(type(x) is not int for x in value):
        raise TypeError(f"{name} bins must be Python integers")
    if any(x < 0 or x > max_bin for x in value):
        raise ValueError(f"{name} bins must be in 0..{max_bin}")
    x1, y1, x2, y2 = value
    if x1 >= x2 or y1 >= y2:
        raise ValueError(f"{name} must have positive width and height")
    return x1, y1, x2, y2


def _prefix(value, max_bin: int) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or len(value) > 3:
        raise ValueError("prefix must contain zero to three coordinate bins")
    if any(type(x) is not int for x in value):
        raise TypeError("prefix bins must be Python integers")
    if any(x < 0 or x > max_bin for x in value):
        raise ValueError(f"prefix bins must be in 0..{max_bin}")
    return tuple(value)


def _actual_bins(value, max_bin: int) -> tuple[int, int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("actual_bins must contain four coordinate bins")
    if any(type(x) is not int for x in value):
        raise TypeError("actual_bins must be Python integers")
    if any(x < 0 or x > max_bin for x in value):
        raise ValueError(f"actual_bins must be in 0..{max_bin}")
    return tuple(value)


def _iou_at_least(left, right, tau: float) -> bool:
    ix = max(0, min(left[2], right[2]) - max(left[0], right[0]))
    iy = max(0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = ix * iy
    union = ((left[2] - left[0]) * (left[3] - left[1])
             + (right[2] - right[0]) * (right[3] - right[1]) - intersection)
    return union > 0 and intersection >= tau * union


def _completable(gt, prefix: tuple[int, ...], tau: float, max_bin: int) -> bool:
    """Maximize IoU for a fixed prefix by placing each free edge on the GT box."""
    size = len(prefix)
    x1, y1, x2, y2 = gt
    if size == 1:
        candidate = (prefix[0], y1, x2, y2)
    elif size == 2:
        candidate = (prefix[0], prefix[1], x2, y2)
    elif size == 3:
        candidate = (prefix[0], prefix[1], prefix[2], y2)
    else:
        candidate = prefix
    if candidate[0] >= candidate[2] or candidate[1] >= candidate[3]:
        return False
    return _iou_at_least(candidate, gt, tau)


def acceptable_bins(gt, prefix=(), tau=0.5, max_bin=999) -> list[int]:
    """Return next bins that admit some strict-positive xyxy completion at IoU >= tau."""
    max_bin = _grid(max_bin)
    gt = _box(gt, "gt", max_bin)
    prefix = _prefix(prefix, max_bin)
    tau = _real("tau", tau)
    if not 0 < tau <= 1:
        raise ValueError("tau must be in (0, 1]")
    return [bin_value for bin_value in range(max_bin + 1)
            if _completable(gt, prefix + (bin_value,), tau, max_bin)]


def region_margin(scores, acceptable_token_ids, margin=0.2):
    """Max outside-versus-acceptable logit margin over the complete vocabulary."""
    import torch

    if not isinstance(scores, torch.Tensor):
        raise TypeError("scores must be a torch tensor")
    if scores.ndim != 1 or scores.numel() == 0:
        raise ValueError("scores must be a nonempty one-dimensional vocabulary vector")
    if not scores.is_floating_point():
        raise TypeError("scores must have a floating dtype")
    margin = _real("margin", margin)
    if margin < 0:
        raise ValueError("margin must be nonnegative")
    ids = tuple(acceptable_token_ids)
    if not ids:
        raise ValueError("acceptable_token_ids must be nonempty")
    if any(type(token_id) is not int for token_id in ids):
        raise TypeError("acceptable token IDs must be Python integers")
    if any(token_id < 0 or token_id >= scores.numel() for token_id in ids):
        raise ValueError("acceptable token ID is outside the vocabulary")

    z = scores.float()
    acceptable = torch.zeros(z.shape, dtype=torch.bool, device=z.device)
    acceptable[list(ids)] = True
    outside = ~acceptable
    if len(set(ids)) == z.numel():
        return z.sum() * 0
    return torch.relu(z.new_tensor(margin) + z[outside].amax() - z[acceptable].amax())


def owner_slots(gt, actual_bins, tau=0.5, max_bin=999):
    """Return eligible slot indices, their next-bin regions, and first failure index."""
    max_bin = _grid(max_bin)
    gt = _box(gt, "gt", max_bin)
    actual_bins = _actual_bins(actual_bins, max_bin)
    tau = _real("tau", tau)
    if not 0 < tau <= 1:
        raise ValueError("tau must be in (0, 1]")

    slots, regions, prefix = [], [], ()
    first_failure = None
    for slot, actual in enumerate(actual_bins):
        acceptable = acceptable_bins(gt, prefix, tau=tau, max_bin=max_bin)
        if not acceptable:
            break
        slots.append(slot)
        regions.append(acceptable)
        if actual not in acceptable:
            first_failure = slot
            break
        prefix += (actual,)
    return slots, regions, first_failure
