from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch


ARTIFACT_SCHEMA_VERSION = "a3.3.v1"
ROW_SCHEMA_VERSION = "slot_posterior.v1"
R95_POLICY_ID = "axis_len_fraction_floor_cap_v1"
BOUNDARY_COORD_VALUES = {0, 999}
SLOT_TO_BBOX_INDEX = {"x1": 0, "y1": 1, "x2": 2, "y2": 3}
SLOT_TO_AXIS = {"x1": "x", "x2": "x", "y1": "y", "y2": "y"}


def strict_r95_radius(*, axis_len: int, fraction: float = 0.04, cap: int = 8) -> int:
    return int(min(int(cap), int(max(0, int(axis_len)) * float(fraction))))


def axis_len_for_slot(slot: str, bbox_xyxy: Sequence[int | float]) -> int:
    if slot not in SLOT_TO_AXIS:
        raise ValueError(f"unsupported slot: {slot}")
    if len(bbox_xyxy) != 4:
        raise ValueError("bbox_xyxy must have four values")
    x1, y1, x2, y2 = [int(value) for value in bbox_xyxy]
    return max(0, x2 - x1) if SLOT_TO_AXIS[slot] == "x" else max(0, y2 - y1)


def neighborhood_mass(probs: torch.Tensor, center: int, radius: int) -> float:
    lo = max(0, int(center) - int(radius))
    hi = min(999, int(center) + int(radius))
    return _finite_float(probs[lo : hi + 1].sum().item())


def classify_slot_posterior(
    *,
    full_vocab_logits: torch.Tensor | Sequence[float],
    coord_token_ids: Sequence[int],
    slot: str,
    target_value: int,
    target_axis_len: int,
    competitors: Sequence[Mapping[str, Any]] = (),
    other_desc_objects: Sequence[Mapping[str, Any]] = (),
    low_margin_threshold: float = 0.05,
    coord_mass_low_threshold: float = 0.01,
    strict_r95_axis_fraction: float = 0.04,
    strict_r95_cap_bins: int = 8,
    top_k: int = 8,
) -> dict[str, Any]:
    logits = _as_1d_float_tensor(full_vocab_logits)
    coord_ids = _coord_token_id_tensor(coord_token_ids, vocab_size=int(logits.numel()))
    coord_logits = logits[coord_ids]
    full_vocab_probs = torch.softmax(logits, dim=-1)
    coord_full_vocab_probs = full_vocab_probs[coord_ids]
    coord_vocab_mass = _finite_float(coord_full_vocab_probs.sum().item())
    tiny = torch.finfo(coord_full_vocab_probs.dtype).tiny
    coord_conditional_probs = coord_full_vocab_probs / max(coord_vocab_mass, tiny)

    target = _candidate_from_value(
        bucket="target",
        instance_id="target",
        value=target_value,
        axis_len=target_axis_len,
        coord_logits=coord_logits,
        coord_conditional_probs=coord_conditional_probs,
        strict_r95_axis_fraction=strict_r95_axis_fraction,
        strict_r95_cap_bins=strict_r95_cap_bins,
    )
    competitor_candidates = [
        _candidate_from_object(
            obj,
            slot=slot,
            bucket="competitor_same_desc",
            coord_logits=coord_logits,
            coord_conditional_probs=coord_conditional_probs,
            strict_r95_axis_fraction=strict_r95_axis_fraction,
            strict_r95_cap_bins=strict_r95_cap_bins,
        )
        for obj in competitors
    ]
    other_desc_candidates = [
        _candidate_from_object(
            obj,
            slot=slot,
            bucket="other_desc_object",
            coord_logits=coord_logits,
            coord_conditional_probs=coord_conditional_probs,
            strict_r95_axis_fraction=strict_r95_axis_fraction,
            strict_r95_cap_bins=strict_r95_cap_bins,
        )
        for obj in other_desc_objects
    ]
    all_identity_candidates = [target, *competitor_candidates, *other_desc_candidates]
    best_competitor = _best_candidate(competitor_candidates)
    best_other_desc = _best_candidate(other_desc_candidates)
    top_coord = _top_coord_candidate(
        coord_logits=coord_logits,
        coord_full_vocab_probs=coord_full_vocab_probs,
        coord_conditional_probs=coord_conditional_probs,
        coord_token_ids=coord_token_ids,
    )
    identity_hits = [
        candidate
        for candidate in all_identity_candidates
        if _within(candidate["value"], top_coord["coord_value"], candidate["radius"])
    ]
    coord_mass_low_flag = coord_vocab_mass < float(coord_mass_low_threshold)
    low_margin_flag = _low_margin_flag(candidates=all_identity_candidates, threshold=float(low_margin_threshold))
    boundary_extreme_flag = int(top_coord["coord_value"]) in BOUNDARY_COORD_VALUES
    winner = _winner_from_hits(
        identity_hits=identity_hits,
        low_margin_flag=low_margin_flag,
        coord_mass_low_flag=coord_mass_low_flag,
        boundary_extreme_flag=boundary_extreme_flag,
    )
    summary = {
        "slot": str(slot),
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "row_schema_version": ROW_SCHEMA_VERSION,
        "r95_policy_id": R95_POLICY_ID,
        "r95_axis_fraction": float(strict_r95_axis_fraction),
        "r95_cap_bins": int(strict_r95_cap_bins),
        "radius_used_for_classification_only": True,
        "target_value": int(target_value),
        "target_axis_len": int(target_axis_len),
        "target_radius": int(target["radius"]),
        "target_slot_mass": float(target["mass"]),
        "best_same_desc_competitor_slot_mass": None if best_competitor is None else float(best_competitor["mass"]),
        "best_other_desc_object_slot_mass": None if best_other_desc is None else float(best_other_desc["mass"]),
        "target_vs_competitor_margin": None
        if best_competitor is None
        else float(target["mass"] - best_competitor["mass"]),
        "winner_instance_id": winner["winner_instance_id"],
        "winner_bucket": winner["winner_bucket"],
        "slot_taxonomy": winner["slot_taxonomy"],
        "coord_vocab_mass": coord_vocab_mass,
        "coord_mass_low_threshold": float(coord_mass_low_threshold),
        "coord_mass_low_flag": bool(coord_mass_low_flag),
        "noncoord_top_token_id": _top_noncoord(full_vocab_probs, coord_ids)["token_id"],
        "noncoord_top_prob": _top_noncoord(full_vocab_probs, coord_ids)["prob"],
        "top_peak_value": int(top_coord["coord_value"]),
        "top_peak_token_id": int(top_coord["token_id"]),
        "top_peak_mass": float(top_coord["conditional_coord_prob"]),
        "top_peak_full_vocab_prob": float(top_coord["full_vocab_prob"]),
        "top_peak_logit": float(top_coord["logit"]),
        "top_coord_candidates": _top_coord_candidates(
            coord_logits=coord_logits,
            coord_full_vocab_probs=coord_full_vocab_probs,
            coord_conditional_probs=coord_conditional_probs,
            coord_token_ids=coord_token_ids,
            top_k=top_k,
        ),
        "target_center_logit": float(target["center_logit"]),
        "best_competitor_center_logit": None if best_competitor is None else float(best_competitor["center_logit"]),
        "target_rank": int(_coord_rank(coord_conditional_probs, int(target_value))),
        "low_margin_threshold": float(low_margin_threshold),
        "low_margin_flag": bool(low_margin_flag),
        "target_r95_hit": bool(_within(target["value"], top_coord["coord_value"], target["radius"])),
        "best_competitor_r95_hit": bool(
            best_competitor is not None
            and _within(best_competitor["value"], top_coord["coord_value"], best_competitor["radius"])
        ),
        "boundary_extreme_flag": bool(boundary_extreme_flag),
        "background_or_outlier_flag": winner["winner_bucket"] == "background",
        "candidate_masses": _candidate_rows(all_identity_candidates),
    }
    _assert_json_safe(summary)
    return summary


def _as_1d_float_tensor(values: torch.Tensor | Sequence[float]) -> torch.Tensor:
    tensor = values.detach() if isinstance(values, torch.Tensor) else torch.as_tensor(values)
    tensor = tensor.to(dtype=torch.float32)
    if tensor.ndim != 1:
        raise ValueError("full_vocab_logits must be a 1D tensor or sequence")
    if tensor.numel() == 0:
        raise ValueError("full_vocab_logits must not be empty")
    if not torch.isfinite(tensor).all():
        raise ValueError("full_vocab_logits must be finite")
    return tensor


def _coord_token_id_tensor(coord_token_ids: Sequence[int], *, vocab_size: int) -> torch.Tensor:
    if len(coord_token_ids) != 1000:
        raise ValueError("coord_token_ids must contain exactly 1000 ids for bins 0..999")
    ids = torch.as_tensor([int(token_id) for token_id in coord_token_ids], dtype=torch.long)
    if int(ids.min().item()) < 0 or int(ids.max().item()) >= int(vocab_size):
        raise ValueError("coord_token_ids contains an id outside full_vocab_logits")
    if len(set(int(token_id) for token_id in coord_token_ids)) != len(coord_token_ids):
        raise ValueError("coord_token_ids must be unique")
    return ids


def _candidate_from_object(
    obj: Mapping[str, Any],
    *,
    slot: str,
    bucket: str,
    coord_logits: torch.Tensor,
    coord_conditional_probs: torch.Tensor,
    strict_r95_axis_fraction: float,
    strict_r95_cap_bins: int,
) -> dict[str, Any]:
    return _candidate_from_value(
        bucket=bucket,
        instance_id=int(obj.get("gt_idx", obj.get("instance_id", -1))),
        value=_slot_value(obj, slot),
        axis_len=_axis_len(obj, slot),
        coord_logits=coord_logits,
        coord_conditional_probs=coord_conditional_probs,
        strict_r95_axis_fraction=strict_r95_axis_fraction,
        strict_r95_cap_bins=strict_r95_cap_bins,
    )


def _candidate_from_value(
    *,
    bucket: str,
    instance_id: str | int,
    value: int,
    axis_len: int,
    coord_logits: torch.Tensor,
    coord_conditional_probs: torch.Tensor,
    strict_r95_axis_fraction: float,
    strict_r95_cap_bins: int,
) -> dict[str, Any]:
    center = min(999, max(0, int(value)))
    radius = strict_r95_radius(
        axis_len=int(axis_len),
        fraction=float(strict_r95_axis_fraction),
        cap=int(strict_r95_cap_bins),
    )
    return {
        "bucket": bucket,
        "instance_id": instance_id,
        "value": center,
        "axis_len": int(axis_len),
        "radius": radius,
        "mass": neighborhood_mass(coord_conditional_probs, center, radius),
        "center_logit": _finite_float(coord_logits[center].item()),
        "center_conditional_prob": _finite_float(coord_conditional_probs[center].item()),
    }


def _slot_value(obj: Mapping[str, Any], slot: str) -> int:
    for key in ("value", f"{slot}_value", slot):
        if key in obj:
            return int(obj[key])
    bbox = obj.get("bbox_coord_token_xyxy", obj.get("bbox_xyxy"))
    if bbox is not None:
        return int(bbox[SLOT_TO_BBOX_INDEX[slot]])
    raise KeyError(f"object is missing a value for slot {slot}")


def _axis_len(obj: Mapping[str, Any], slot: str) -> int:
    if "axis_len" in obj:
        return int(obj["axis_len"])
    bbox = obj.get("bbox_coord_token_xyxy", obj.get("bbox_xyxy"))
    return axis_len_for_slot(slot, bbox) if bbox is not None else 0


def _best_candidate(candidates: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    return None if not candidates else max(candidates, key=lambda candidate: float(candidate["mass"]))


def _top_coord_candidate(
    *,
    coord_logits: torch.Tensor,
    coord_full_vocab_probs: torch.Tensor,
    coord_conditional_probs: torch.Tensor,
    coord_token_ids: Sequence[int],
) -> dict[str, Any]:
    top_idx = int(torch.argmax(coord_full_vocab_probs).item())
    return {
        "coord_value": top_idx,
        "token_id": int(coord_token_ids[top_idx]),
        "logit": _finite_float(coord_logits[top_idx].item()),
        "full_vocab_prob": _finite_float(coord_full_vocab_probs[top_idx].item()),
        "conditional_coord_prob": _finite_float(coord_conditional_probs[top_idx].item()),
    }


def _top_coord_candidates(
    *,
    coord_logits: torch.Tensor,
    coord_full_vocab_probs: torch.Tensor,
    coord_conditional_probs: torch.Tensor,
    coord_token_ids: Sequence[int],
    top_k: int,
) -> list[dict[str, Any]]:
    k = max(1, min(int(top_k), int(coord_full_vocab_probs.numel())))
    _, indices = torch.topk(coord_full_vocab_probs, k=k)
    rows: list[dict[str, Any]] = []
    for rank, idx_tensor in enumerate(indices, start=1):
        idx = int(idx_tensor.item())
        rows.append(
            {
                "rank": rank,
                "coord_value": idx,
                "token_id": int(coord_token_ids[idx]),
                "logit": _finite_float(coord_logits[idx].item()),
                "full_vocab_prob": _finite_float(coord_full_vocab_probs[idx].item()),
                "conditional_coord_prob": _finite_float(coord_conditional_probs[idx].item()),
            }
        )
    return rows


def _top_noncoord(full_vocab_probs: torch.Tensor, coord_ids: torch.Tensor) -> dict[str, Any]:
    mask = torch.ones_like(full_vocab_probs, dtype=torch.bool)
    mask[coord_ids] = False
    if not bool(mask.any().item()):
        return {"token_id": None, "prob": None}
    noncoord_indices = torch.nonzero(mask, as_tuple=False).flatten()
    noncoord_probs = full_vocab_probs[noncoord_indices]
    local_idx = int(torch.argmax(noncoord_probs).item())
    return {
        "token_id": int(noncoord_indices[local_idx].item()),
        "prob": _finite_float(noncoord_probs[local_idx].item()),
    }


def _winner_from_hits(
    *,
    identity_hits: Sequence[Mapping[str, Any]],
    low_margin_flag: bool,
    coord_mass_low_flag: bool,
    boundary_extreme_flag: bool,
) -> dict[str, Any]:
    if coord_mass_low_flag:
        return _winner("invalid_low_coord_mass", None)
    if low_margin_flag:
        return _winner("tied", None)
    if len(identity_hits) > 1:
        return _winner("ambiguous", None)
    if len(identity_hits) == 1:
        hit = identity_hits[0]
        return {
            "winner_bucket": str(hit["bucket"]),
            "slot_taxonomy": str(hit["bucket"]),
            "winner_instance_id": hit["instance_id"],
        }
    if boundary_extreme_flag:
        return _winner("boundary_extreme", None)
    return _winner("background", None)


def _winner(bucket: str, instance_id: str | int | None) -> dict[str, Any]:
    return {"winner_bucket": bucket, "slot_taxonomy": bucket, "winner_instance_id": instance_id}


def _low_margin_flag(*, candidates: Sequence[Mapping[str, Any]], threshold: float) -> bool:
    scored = sorted((float(candidate["mass"]) for candidate in candidates), reverse=True)
    return len(scored) >= 2 and scored[0] - scored[1] <= float(threshold)


def _coord_rank(coord_conditional_probs: torch.Tensor, coord_value: int) -> int:
    center = min(999, max(0, int(coord_value)))
    return int((coord_conditional_probs > coord_conditional_probs[center]).sum().item()) + 1


def _candidate_rows(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "bucket": str(candidate["bucket"]),
            "instance_id": candidate["instance_id"],
            "value": int(candidate["value"]),
            "axis_len": int(candidate["axis_len"]),
            "radius": int(candidate["radius"]),
            "mass": float(candidate["mass"]),
            "center_logit": float(candidate["center_logit"]),
            "center_conditional_prob": float(candidate["center_conditional_prob"]),
        }
        for candidate in candidates
    ]


def _within(left: int, right: int, radius: int) -> bool:
    return abs(int(left) - int(right)) <= int(radius)


def _finite_float(value: object) -> float:
    result = float(value)  # type: ignore[arg-type]
    if not math.isfinite(result):
        raise ValueError("posterior summary contains a non-finite float")
    return result


def _assert_json_safe(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("posterior summary contains a non-finite float")
    if isinstance(value, Mapping):
        for child in value.values():
            _assert_json_safe(child)
    elif isinstance(value, list):
        for child in value:
            _assert_json_safe(child)
