from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any, Mapping


_MODE_ORDER = (
    "empty_prefix",
    "random_subset",
    "leave_one_out",
    "full_prefix",
)


@dataclass(frozen=True)
class Stage1SetContinuationSample:
    selected_mode: str
    configured_mixture: dict[str, float]
    resolved_valid_mixture: dict[str, float]
    prefix_indices: tuple[int, ...]
    remaining_indices: tuple[int, ...]
    candidate_indices: tuple[int, ...]
    candidate_scoring_mode: str
    scored_candidate_fraction: float


def _stable_seed_from_parts(seed_parts: tuple[Any, ...]) -> int:
    digest = hashlib.sha256(repr(tuple(seed_parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _configured_mixture(subset_sampling_cfg: Any) -> dict[str, float]:
    return {
        "empty_prefix": float(getattr(subset_sampling_cfg, "empty_prefix_ratio")),
        "random_subset": float(getattr(subset_sampling_cfg, "random_subset_ratio")),
        "leave_one_out": float(getattr(subset_sampling_cfg, "leave_one_out_ratio")),
        "full_prefix": float(getattr(subset_sampling_cfg, "full_prefix_ratio")),
    }


def _allowed_modes(object_count: int) -> tuple[str, ...]:
    if object_count <= 0:
        return ("empty_prefix",)
    if object_count == 1:
        return ("empty_prefix", "leave_one_out", "full_prefix")
    return _MODE_ORDER


def _resolve_valid_mixture(
    *, object_count: int, configured: Mapping[str, float]
) -> dict[str, float]:
    allowed = set(_allowed_modes(object_count))
    filtered = {
        mode: (float(configured.get(mode, 0.0)) if mode in allowed else 0.0)
        for mode in _MODE_ORDER
    }
    total = sum(filtered.values())
    if total > 0.0:
        return {mode: float(value / total) for mode, value in filtered.items()}

    fallback = next(mode for mode in _MODE_ORDER if mode in allowed)
    return {
        mode: (1.0 if mode == fallback else 0.0)
        for mode in _MODE_ORDER
    }


def _weighted_choice(rng: random.Random, mixture: Mapping[str, float]) -> str:
    total = float(sum(mixture.values()))
    if total <= 0.0:
        raise ValueError("resolved mixture must contain positive mass")
    draw = rng.random() * total
    cumulative = 0.0
    for mode in _MODE_ORDER:
        cumulative += float(mixture.get(mode, 0.0))
        if draw <= cumulative and cumulative > 0.0:
            return mode
    for mode in reversed(_MODE_ORDER):
        if float(mixture.get(mode, 0.0)) > 0.0:
            return mode
    raise AssertionError("weighted choice exhausted without a valid mode")


def _ordered_indices(
    *, object_count: int, prefix_order: str, rng: random.Random
) -> list[int]:
    indices = list(range(int(object_count)))
    if prefix_order == "random":
        rng.shuffle(indices)
    return indices


def _select_prefix_and_remaining(
    *,
    selected_mode: str,
    object_count: int,
    prefix_order: str,
    rng: random.Random,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    ordered = _ordered_indices(
        object_count=object_count,
        prefix_order=prefix_order,
        rng=rng,
    )
    if selected_mode == "empty_prefix":
        return (), tuple(ordered)
    if selected_mode == "full_prefix":
        return tuple(ordered), ()
    if selected_mode == "leave_one_out":
        drop_index = rng.randrange(len(ordered))
        drop_object = ordered[drop_index]
        prefix = tuple(idx for idx in ordered if idx != drop_object)
        return prefix, (drop_object,)
    if selected_mode == "random_subset":
        subset_size = rng.randrange(1, len(ordered))
        prefix = tuple(ordered[:subset_size])
        prefix_set = set(prefix)
        remaining = tuple(idx for idx in ordered if idx not in prefix_set)
        return prefix, remaining
    raise ValueError(f"Unsupported subset sampling mode: {selected_mode}")


def _candidate_selection(
    *,
    candidates_cfg: Any,
    remaining_indices: tuple[int, ...],
    rng: random.Random,
) -> tuple[tuple[int, ...], str, float]:
    mode = str(getattr(candidates_cfg, "mode"))
    remaining_count = len(remaining_indices)
    if mode == "exact":
        return remaining_indices, mode, 1.0
    if mode != "uniform_subsample":
        raise ValueError(f"Unsupported candidate scoring mode: {mode}")

    max_candidates = getattr(candidates_cfg, "max_candidates", None)
    if max_candidates is None or int(max_candidates) <= 0:
        raise ValueError("uniform_subsample max_candidates must be > 0")
    limit = int(max_candidates)
    if remaining_count <= limit:
        return remaining_indices, mode, 1.0

    sampled = list(remaining_indices)
    rng.shuffle(sampled)
    chosen = tuple(sampled[:limit])
    return chosen, mode, float(len(chosen) / remaining_count)


def sample_subset_and_candidates(
    *,
    object_count: int,
    subset_sampling_cfg: Any,
    candidates_cfg: Any,
    seed_parts: tuple[Any, ...],
) -> Stage1SetContinuationSample:
    object_count = int(object_count)
    if object_count < 0:
        raise ValueError("object_count must be >= 0")

    configured = _configured_mixture(subset_sampling_cfg)
    resolved = _resolve_valid_mixture(
        object_count=object_count,
        configured=configured,
    )
    rng = random.Random(_stable_seed_from_parts(tuple(seed_parts)))
    selected_mode = _weighted_choice(rng, resolved)
    prefix_indices, remaining_indices = _select_prefix_and_remaining(
        selected_mode=selected_mode,
        object_count=object_count,
        prefix_order=str(getattr(subset_sampling_cfg, "prefix_order", "random")),
        rng=rng,
    )
    candidate_indices, candidate_scoring_mode, scored_candidate_fraction = (
        _candidate_selection(
            candidates_cfg=candidates_cfg,
            remaining_indices=remaining_indices,
            rng=rng,
        )
    )
    return Stage1SetContinuationSample(
        selected_mode=selected_mode,
        configured_mixture=configured,
        resolved_valid_mixture=resolved,
        prefix_indices=prefix_indices,
        remaining_indices=remaining_indices,
        candidate_indices=candidate_indices,
        candidate_scoring_mode=candidate_scoring_mode,
        scored_candidate_fraction=float(scored_candidate_fraction),
    )


__all__ = [
    "Stage1SetContinuationSample",
    "sample_subset_and_candidates",
]
