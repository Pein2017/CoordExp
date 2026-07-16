#!/usr/bin/env python3
"""Run the count-balanced finite soft cross-region one-image probe.

The experiment-local runner reuses the frozen hard-cross scorer while owning
the finite-bias operator, runtime attestation, matched controls, and
conclusion-critical classifiers for this research unit.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import scripts.research.run_fixed_encoding_cross_region_earlier_and_row_query_spatial_key_eligibility_hybrid as hard_cross  # noqa: E402

# Reuse the canonical scorer and feature/runtime helpers owned by the hard
# cross-region and uniform finite-bias experiments.  This unit adds no shared
# inference surface.
query = hard_cross.query
factorial = hard_cross.factorial


UNIT_ID = "2026-07-15-fixed-encoding-count-balanced-soft-cross-region-earlier-and-row-query-spatial-bias"
HARD_CROSS_UNIT_ID = hard_cross.UNIT_ID
HARD_CROSS_RECEIPT_SHA256 = "a50820cbc8dde6d92b964cfe9233e0e897da54e14d817dc48bd79787c8380bec"
HISTORICAL_SOFT_RECEIPT_SHA256 = "468af3e49962b4e78c69ecac0c004e774c38da93e47e4bd5e19c05814c12580a"
DEFAULT_HARD_CROSS_RECEIPT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-15-fixed-encoding-cross-region-earlier-and-row-query-spatial-key-eligibility-hybrid/"
    "image139-float32-20260715a/receipt.json"
)
DEFAULT_HISTORICAL_SOFT_RECEIPT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-15-fixed-encoding-soft-spatial-key-bias-dose-response/"
    "cohort-four-float32-20260715a/receipt.json"
)
IMAGE_ID = "139"
IMAGE_KEY_COUNT = 1014
SELECTED_KEY_COUNT = 20
LAMBDA_MATH = math.log((IMAGE_KEY_COUNT - SELECTED_KEY_COUNT) / SELECTED_KEY_COUNT)
LAMBDA_FLOAT32 = float(torch.tensor(LAMBDA_MATH, dtype=torch.float32).item())
TOLERANCE = 1e-4
OWNER_MARGIN_FLOOR = 0.10
OWNER_RELEASE_FLOOR = 0.05
MISMATCH_PENALTY_FLOOR = 0.05

ARM_REGION_MAPPING: dict[str, tuple[str, str]] = {
    "target_earlier_target_row": ("target", "target"),
    "target_earlier_competitor_row": ("target", "competitor"),
    "competitor_earlier_target_row": ("competitor", "target"),
    "competitor_earlier_competitor_row": ("competitor", "competitor"),
}

_FROZEN_HARD_CROSS_CASE: Mapping[str, Any] | None = None
_ORIGINAL_QUERY_RUN_CASE = query._run_case


def lambda_star(*, image_key_count: int = IMAGE_KEY_COUNT, selected_key_count: int = SELECTED_KEY_COUNT) -> float:
    """Return the exact cardinality-derived dose represented in float32."""

    image_count = int(image_key_count)
    selected_count = int(selected_key_count)
    if image_count <= selected_count or selected_count <= 0:
        raise ValueError("image_key_count must exceed positive selected_key_count")
    mathematical = math.log((image_count - selected_count) / selected_count)
    return float(torch.tensor(mathematical, dtype=torch.float32).item())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve(strict=True).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tensor(value: torch.Tensor) -> str:
    contiguous = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode())
    digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode())
    digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def build_soft_cross_region_additive_mask(
    *,
    sequence_length: int,
    image_key_positions: Sequence[int],
    earlier_selected_image_positions: Sequence[int],
    row_selected_image_positions: Sequence[int],
    prefix_length: int,
    row_length: int,
    bias: float = LAMBDA_FLOAT32,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Build the finite additive causal mask for one crossed arm.

    All causally visible cells start at zero.  Only selected image keys in the
    declared earlier or row-scoring query ranges receive the one float32 bias.
    Future keys remain negative infinity and are never made visible.
    """

    length = int(sequence_length)
    prefix = int(prefix_length)
    row = int(row_length)
    if length <= 0 or prefix < 2 or row < 2:
        raise ValueError("sequence_length must be positive; prefix and row lengths must be at least two")
    value = float(bias)
    if not math.isfinite(value) or value < 0:
        raise ValueError("bias must be finite and non-negative")
    image = {int(v) for v in image_key_positions}
    earlier = {int(v) for v in earlier_selected_image_positions}
    row_selected = {int(v) for v in row_selected_image_positions}
    all_positions = image | earlier | row_selected
    if any(v < 0 or v >= length for v in all_positions):
        raise ValueError("image key position is outside sequence")
    if not earlier.issubset(image) or not row_selected.issubset(image):
        raise ValueError("selected image keys must be subsets of image keys")

    mask = torch.full((length, length), float("-inf"), dtype=torch.float32, device=device)
    visible = torch.tril(torch.ones((length, length), dtype=torch.bool, device=device))
    mask[visible] = 0.0
    earlier_end = prefix - 2
    row_start = prefix - 1
    row_end = prefix + row - 2
    if row_end >= length:
        raise ValueError("row-scoring range is outside sequence")

    def add_region(query_slice: slice, selected: set[int]) -> None:
        if not selected:
            return
        keys = torch.tensor(sorted(selected), dtype=torch.long, device=device)
        visible_selected = visible[query_slice, :][:, keys]
        current = mask[query_slice, :][:, keys]
        updated = torch.where(
            visible_selected,
            torch.full_like(current, value, dtype=torch.float32),
            torch.full_like(current, float("-inf"), dtype=torch.float32),
        )
        mask[query_slice, :][:, keys] = updated

    add_region(slice(0, earlier_end + 1), earlier)
    add_region(slice(row_start, row_end + 1), row_selected)
    return mask.unsqueeze(0).unsqueeze(0).contiguous()


def build_zero_float32_causal_mask(*, sequence_length: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """Build an all-image-allowed additive no-operation mask."""

    length = int(sequence_length)
    if length <= 0:
        raise ValueError("sequence_length must be positive")
    mask = torch.full((length, length), float("-inf"), dtype=torch.float32, device=device)
    mask[torch.tril(torch.ones((length, length), dtype=torch.bool, device=device))] = 0.0
    return mask.unsqueeze(0).unsqueeze(0).contiguous()


# Readable aliases for downstream experiment code and tests.
build_count_balanced_soft_cross_region_mask = build_soft_cross_region_additive_mask
build_additive_causal_key_bias_mask = build_soft_cross_region_additive_mask


def _changed_cell_counts(mask: torch.Tensor) -> dict[str, int]:
    observed = mask[0, 0]
    length = int(observed.shape[-1])
    baseline = torch.full_like(observed, float("-inf"), dtype=torch.float32)
    baseline[torch.tril(torch.ones((length, length), dtype=torch.bool, device=observed.device))] = 0.0
    changed = observed != baseline
    finite_changed = changed & torch.isfinite(observed)
    return {
        "changed_cell_count": int(changed.sum().item()),
        "finite_changed_cell_count": int(finite_changed.sum().item()),
        "future_changed_cell_count": int((changed & ~torch.tril(torch.ones_like(changed))).sum().item()),
    }


def inspect_soft_cross_region_mask_structure(
    mask: torch.Tensor,
    *,
    sequence_length: int,
    image_key_positions: Sequence[int],
    earlier_selected_image_positions: Sequence[int],
    row_selected_image_positions: Sequence[int],
    prefix_length: int,
    row_length: int,
    bias: float = LAMBDA_FLOAT32,
) -> dict[str, Any]:
    """Return a strict structural receipt for one finite crossed mask."""

    length = int(sequence_length)
    passed_shape = tuple(mask.shape) == (1, 1, length, length)
    passed_dtype = mask.dtype is torch.float32
    receipt: dict[str, Any] = {
        "passed": False,
        "mask_shape": list(mask.shape),
        "mask_dtype": str(mask.dtype),
        "expected_bias_math": LAMBDA_MATH,
        "expected_bias_float32": float(torch.tensor(float(bias), dtype=torch.float32).item()),
        "passed_shape": passed_shape,
        "passed_dtype": passed_dtype,
    }
    if not passed_shape or not passed_dtype:
        return receipt
    expected = build_soft_cross_region_additive_mask(
        sequence_length=length,
        image_key_positions=image_key_positions,
        earlier_selected_image_positions=earlier_selected_image_positions,
        row_selected_image_positions=row_selected_image_positions,
        prefix_length=prefix_length,
        row_length=row_length,
        bias=bias,
        device=mask.device,
    )
    observed = mask[0, 0]
    expected_2d = expected[0, 0]
    image = {int(v) for v in image_key_positions}
    earlier = {int(v) for v in earlier_selected_image_positions}
    row_selected = {int(v) for v in row_selected_image_positions}
    visible = torch.tril(torch.ones((length, length), dtype=torch.bool, device=mask.device))
    changed = observed != torch.where(visible, torch.zeros_like(observed), torch.full_like(observed, float("-inf")))
    earlier_end = int(prefix_length) - 2
    row_start = int(prefix_length) - 1
    row_end = int(prefix_length) + int(row_length) - 2
    earlier_cells = torch.zeros_like(changed)
    row_cells = torch.zeros_like(changed)
    earlier_cells[: earlier_end + 1, :] = True
    row_cells[row_start : row_end + 1, :] = True
    earlier_selected = torch.zeros_like(changed)
    row_selected_cells = torch.zeros_like(changed)
    for key in earlier:
        earlier_selected[: earlier_end + 1, key] = visible[: earlier_end + 1, key]
    for key in row_selected:
        row_selected_cells[row_start : row_end + 1, key] = visible[row_start : row_end + 1, key]
    intended = earlier_selected | row_selected_cells
    baseline = torch.where(visible, torch.zeros_like(observed), torch.full_like(observed, float("-inf")))
    finite_visible_nonselected = visible & ~intended
    finite_non_image = visible.clone()
    if image:
        finite_non_image[:, torch.tensor(sorted(image), dtype=torch.long, device=mask.device)] = False
    future = ~visible
    receipt.update(
        {
            "expected_mask_digest": _sha256_tensor(expected),
            "observed_mask_digest": _sha256_tensor(mask),
            "exact_expected_mask": bool(torch.equal(observed, expected_2d)),
            "selected_key_count_earlier": len(earlier),
            "selected_key_count_row": len(row_selected),
            "image_key_count": len(image),
            "exact_1014_image_keys": len(image) == IMAGE_KEY_COUNT,
            "exact_twenty_selected_each": len(earlier) == SELECTED_KEY_COUNT and len(row_selected) == SELECTED_KEY_COUNT,
            "earlier_query_range": {"start_inclusive": 0, "end_inclusive": earlier_end},
            "row_scoring_query_range": {"start_inclusive": row_start, "end_inclusive": row_end},
            "excluded_unscored_final_query": row_end + 1,
            "earlier_changed_cell_count": int((changed & earlier_cells).sum().item()),
            "row_changed_cell_count": int((changed & row_cells).sum().item()),
            "intended_finite_changed_cell_count": int(intended.sum().item()),
            "observed_finite_changed_cell_count": int((changed & torch.isfinite(observed)).sum().item()),
            "future_changed_cell_count": int((changed & future).sum().item()),
            "changed_visible_nonselected_cell_count": int((changed & finite_visible_nonselected).sum().item()),
            "changed_visible_non_image_cell_count": int((changed & finite_non_image).sum().item()),
            "all_visible_nonselected_non_image_cells_zero": bool(torch.all(observed[finite_visible_nonselected] == 0.0).item() and torch.all(observed[finite_non_image] == 0.0).item()),
            "all_future_cells_negative_infinity": bool(torch.all(torch.isneginf(observed[future])).item()),
            "selected_cells_have_exact_bias": bool(torch.all(observed[intended] == float(torch.tensor(float(bias), dtype=torch.float32).item())).item()),
            "off_scope_cells_unchanged": bool(torch.equal(observed[~intended & ~future], baseline[~intended & ~future])),
            "earlier_and_row_changed_sets_disjoint": bool(not torch.any(earlier_selected & row_selected_cells).item()),
            "changed_cell_count": int(changed.sum().item()),
        }
    )
    receipt["passed"] = bool(
        receipt["exact_expected_mask"]
        and receipt["exact_1014_image_keys"]
        and receipt["exact_twenty_selected_each"]
        and receipt["all_visible_nonselected_non_image_cells_zero"]
        and receipt["all_future_cells_negative_infinity"]
        and receipt["selected_cells_have_exact_bias"]
        and receipt["off_scope_cells_unchanged"]
        and receipt["earlier_and_row_changed_sets_disjoint"]
        and receipt["changed_visible_nonselected_cell_count"] == 0
        and receipt["changed_visible_non_image_cell_count"] == 0
        and receipt["future_changed_cell_count"] == 0
        and receipt["observed_finite_changed_cell_count"] == receipt["intended_finite_changed_cell_count"]
    )
    return receipt


def build_soft_cross_arm_plan(
    *, frozen_case: Mapping[str, Any], target_indices: Sequence[int], competitor_indices: Sequence[int]
) -> dict[str, dict[str, Any]]:
    """Build all four arm mappings from one source of truth."""

    # The hard-cross receipt stores these frozen comparators under
    # ``frozen_comparators``; the earlier factorial receipt used the flatter
    # fields.  Accept both shapes while keeping one arm mapping source.
    frozen = frozen_case.get("frozen_comparators", frozen_case)
    earlier_refs = frozen.get("earlier_query_only_arms", frozen.get("earlier_query_only", {}))
    row_refs = frozen.get("arms", frozen.get("row_query_only", {}))
    all_refs = frozen.get("parent_hard_endpoint", frozen.get("all_query", {}))
    if "target_row_query_only_hard" not in row_refs:
        row_refs = {
            "target_row_query_only_hard": row_refs.get("target"),
            "competitor_row_query_only_hard": row_refs.get("competitor"),
        }
    if "target_eligibility" not in all_refs:
        all_refs = {"target_eligibility": all_refs.get("target"), "competitor_eligibility": all_refs.get("competitor")}
    indices = {"target": [int(v) for v in target_indices], "competitor": [int(v) for v in competitor_indices]}
    plan: dict[str, dict[str, Any]] = {}
    for name, (earlier_region, row_region) in ARM_REGION_MAPPING.items():
        plan[name] = {
            "arm_name": name,
            "earlier_region": earlier_region,
            "row_region": row_region,
            "earlier_indices": list(indices[earlier_region]),
            "row_indices": list(indices[row_region]),
            "earlier_reference": earlier_refs[earlier_region],
            "row_only_reference": row_refs[f"{row_region}_row_query_only_hard"],
            "matched_all_query_reference": all_refs[f"{row_region}_eligibility"],
        }
    return plan


def classify_owner_activation_suppression(*, gamma: float, preferred_owner_release: float, alternative_owner_release: float, margin_floor: float = OWNER_MARGIN_FLOOR, release_floor: float = OWNER_RELEASE_FLOOR) -> dict[str, Any]:
    """Apply the exclusive four-way owner movement table."""

    value = float(gamma)
    if value >= float(margin_floor):
        preferred = "vase"
    elif value <= -float(margin_floor):
        preferred = "clock"
    else:
        preferred = None
    preferred_non_destructive = preferred is not None and float(preferred_owner_release) >= -float(release_floor)
    if preferred is None:
        return {
            "preferred_owner": None,
            "owner_activated": False,
            "alternative_suppressed": False,
            "preferred_owner_non_destructive": False,
            "mechanism": "neither_activation_nor_suppression",
        }
    activated = float(preferred_owner_release) >= float(release_floor)
    suppressed = float(alternative_owner_release) <= -float(release_floor)
    labels = {
        (True, False): "activation_only",
        (False, True): "suppression_only",
        (True, True): "mixed_activation_and_suppression",
        (False, False): "neither_activation_nor_suppression",
    }
    return {
        "preferred_owner": preferred,
        "owner_activated": activated,
        "alternative_suppressed": suppressed,
        "preferred_owner_non_destructive": preferred_non_destructive,
        "mechanism": labels[(activated, suppressed)],
    }


def _phase_means(arm: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    phase_map = query._phase_score_map(arm)
    return {
        phase: {"vase": float(values["target"]["mean"]), "clock": float(values["competitor"]["mean"])}
        for phase, values in phase_map.items()
        if isinstance(values.get("target"), Mapping)
        and isinstance(values.get("competitor"), Mapping)
        and "mean" in values["target"]
        and "mean" in values["competitor"]
    }


def build_phase_signatures(
    *, unrestricted: Mapping[str, Any], hybrid: Mapping[str, Any], row_only: Mapping[str, Any], matched: Mapping[str, Any]
) -> dict[str, Any]:
    """Compute primary unrestricted deltas and named compatibility diagnostics."""

    base, current = _phase_means(unrestricted), _phase_means(hybrid)
    row, matched_means = _phase_means(row_only), _phase_means(matched)
    result: dict[str, Any] = {}
    for phase in sorted(set(base) & set(current) & set(row) & set(matched_means)):
        result[phase] = {
            "vase_mean": current[phase]["vase"],
            "clock_mean": current[phase]["clock"],
            "gamma_vase_minus_clock": current[phase]["vase"] - current[phase]["clock"],
            "gamma": current[phase]["vase"] - current[phase]["clock"],
            "Delta_vase": current[phase]["vase"] - base[phase]["vase"],
            "Delta_clock": current[phase]["clock"] - base[phase]["clock"],
            "secondary_delta_vase_to_same_row_region_row_only": current[phase]["vase"] - row[phase]["vase"],
            "secondary_delta_clock_to_same_row_region_row_only": current[phase]["clock"] - row[phase]["clock"],
            "secondary_delta_vase_to_matched_arm": current[phase]["vase"] - matched_means[phase]["vase"],
            "secondary_delta_clock_to_matched_arm": current[phase]["clock"] - matched_means[phase]["clock"],
        }
    return result


def _actual_first_token(arm: Mapping[str, Any], *, first_index: int, expected_token: int) -> dict[str, Any]:
    ids = []
    for owner in ("target", "competitor"):
        values = arm.get(owner, {}).get("top_prediction_token_ids", [])
        ids.append(int(values[first_index]) if first_index < len(values) else None)
    return {
        "expected_token_id": int(expected_token),
        "observed_token_ids": ids,
        "paths_agree": ids[0] is not None and ids[0] == ids[1],
        "matches_expected": ids[0] is not None and ids[0] == ids[1] == int(expected_token),
        "actual_token_id": ids[0] if ids[0] == ids[1] else None,
    }


def assess_matched_control_gate(*, plan: Mapping[str, Mapping[str, Any]], arms: Mapping[str, Mapping[str, Any]], phase_signatures: Mapping[str, Mapping[str, Mapping[str, Any]]], target_row: Sequence[int], competitor_row: Sequence[int], first_differing_index: int | None, owner_release_floor: float = OWNER_RELEASE_FLOOR, geometry_margin_floor: float = OWNER_MARGIN_FLOOR) -> dict[str, Any]:
    """Require both matched controls before crossed-arm interpretation."""

    checks: dict[str, Any] = {}
    if first_differing_index is None:
        return {"passed": False, "reason": "no_shared_first_differing_description_index", "checks": checks}
    expected = {"target": int(target_row[first_differing_index]), "competitor": int(competitor_row[first_differing_index])}
    for owner, arm_name in (("target", "target_earlier_target_row"), ("competitor", "competitor_earlier_competitor_row")):
        signature = phase_signatures.get(arm_name, {})
        geom = signature.get("geometry", {})
        row = arms.get(arm_name, {})
        first = _actual_first_token(row, first_index=first_differing_index, expected_token=expected[owner])
        owner_key = "vase" if owner == "target" else "clock"
        semantic_phase = signature.get("first_differing_description", signature.get("description", {}))
        owner_delta = float(semantic_phase.get(f"Delta_{owner_key}", 0.0))
        if owner == "target":
            geom_gamma_ok = float(geom.get("gamma_vase_minus_clock", 0.0)) >= float(geometry_margin_floor)
        else:
            geom_gamma_ok = float(geom.get("gamma_vase_minus_clock", 0.0)) <= -float(geometry_margin_floor)
        geom_release = float(signature.get("geometry", {}).get(f"Delta_{owner_key}", 0.0))
        checks[owner] = {
            "actual_first_token": first,
            "first_token_gate": bool(first["paths_agree"] and first["matches_expected"]),
            "description_owner_release": owner_delta,
            "description_owner_release_gate": owner_delta >= -float(owner_release_floor),
            "geometry_gamma": float(geom.get("gamma_vase_minus_clock", 0.0)),
            "geometry_gamma_gate": geom_gamma_ok,
            "geometry_owner_release": geom_release,
            "geometry_owner_release_gate": geom_release >= -float(owner_release_floor),
        }
        checks[owner]["passed"] = bool(all(checks[owner][key] for key in ("first_token_gate", "description_owner_release_gate", "geometry_gamma_gate", "geometry_owner_release_gate")))
    passed = bool(checks.get("target", {}).get("passed") and checks.get("competitor", {}).get("passed"))
    return {"passed": passed, "checks": checks, "first_differing_description_index": int(first_differing_index), "classification_if_failed": "no_adjudication_close_count_balanced_soft_cross_operator" if not passed else None}


def _penalty(signature: Mapping[str, Any], owner: str, phase: str, *, floor: float = MISMATCH_PENALTY_FLOOR) -> bool:
    owner_key = "vase" if owner == "target" else "clock"
    values = signature.get(phase, {})
    observed = values.get(f"secondary_delta_{owner_key}_to_matched_arm")
    if observed is None:
        observed = values.get(f"compatibility_difference_{owner}", 0.0)
    return float(observed) <= -float(floor)


def classify_crossed_panel(*, phase_signatures: Mapping[str, Mapping[str, Mapping[str, Any]]], arms: Mapping[str, Mapping[str, Any]], target_row: Sequence[int], competitor_row: Sequence[int], first_differing_index: int | None, matched_gate: Mapping[str, Any]) -> dict[str, Any]:
    """Classify crossed signatures with frozen strict precedence."""

    if not bool(matched_gate.get("passed")):
        return {"classification": "no_adjudication_close_count_balanced_soft_cross_operator", "matched_gate_passed": False, "interpreted": False}
    if first_differing_index is None:
        return {"classification": "mixed", "matched_gate_passed": True, "interpreted": False, "reason": "no_first_differing_description"}

    def actual(arm_name: str) -> dict[str, Any]:
        expected = target_row[first_differing_index] if arm_name.startswith("target_") else competitor_row[first_differing_index]
        return _actual_first_token(arms[arm_name], first_index=first_differing_index, expected_token=int(expected))

    cross_a = phase_signatures["competitor_earlier_target_row"]
    cross_b = phase_signatures["target_earlier_competitor_row"]
    a_top = _actual_first_token(arms["competitor_earlier_target_row"], first_index=first_differing_index, expected_token=int(competitor_row[first_differing_index]))
    b_top = _actual_first_token(arms["target_earlier_competitor_row"], first_index=first_differing_index, expected_token=int(target_row[first_differing_index]))
    a_clock = bool(a_top["paths_agree"] and a_top["actual_token_id"] == int(competitor_row[first_differing_index]))
    a_vase = bool(a_top["paths_agree"] and a_top["actual_token_id"] == int(target_row[first_differing_index]))
    b_vase = bool(b_top["paths_agree"] and b_top["actual_token_id"] == int(target_row[first_differing_index]))
    b_clock = bool(b_top["paths_agree"] and b_top["actual_token_id"] == int(competitor_row[first_differing_index]))
    a_geom_vase = float(cross_a.get("geometry", {}).get("gamma_vase_minus_clock", 0.0)) >= OWNER_MARGIN_FLOOR
    a_geom_clock = float(cross_a.get("geometry", {}).get("gamma_vase_minus_clock", 0.0)) <= -OWNER_MARGIN_FLOOR
    b_geom_vase = float(cross_b.get("geometry", {}).get("gamma_vase_minus_clock", 0.0)) >= OWNER_MARGIN_FLOOR
    b_geom_clock = float(cross_b.get("geometry", {}).get("gamma_vase_minus_clock", 0.0)) <= -OWNER_MARGIN_FLOOR
    a_chimera = a_clock and a_geom_vase
    b_chimera = b_vase and b_geom_clock
    a_penalty = _penalty(cross_a, "target", "geometry") or _penalty(cross_a, "target", "full_row")
    b_penalty = _penalty(cross_b, "competitor", "geometry") or _penalty(cross_b, "competitor", "full_row")
    def geometry_mechanism(values: Mapping[str, Any]) -> dict[str, Any]:
        geometry = values.get("geometry", {})
        gamma = float(geometry.get("gamma_vase_minus_clock", 0.0))
        preferred = "vase" if gamma >= OWNER_MARGIN_FLOOR else "clock" if gamma <= -OWNER_MARGIN_FLOOR else None
        if preferred is None:
            return {"mechanism": "neither_activation_nor_suppression", "preferred_owner_non_destructive": False}
        alternative = "clock" if preferred == "vase" else "vase"
        return classify_owner_activation_suppression(
            gamma=gamma,
            preferred_owner_release=float(geometry.get(f"Delta_{preferred}", 0.0)),
            alternative_owner_release=float(geometry.get(f"Delta_{alternative}", 0.0)),
        )
    a_geometry_mechanism = geometry_mechanism(cross_a)
    b_geometry_mechanism = geometry_mechanism(cross_b)
    a_no_penalty = not a_penalty
    b_no_penalty = not b_penalty
    if a_chimera and b_chimera:
        label = "bidirectional_soft_phase_split"
    elif a_chimera and b_geom_clock and not b_vase and (a_penalty or b_penalty):
        label = "asymmetric_hard_cross_phenotype_persists"
    elif a_vase and a_geom_vase and b_clock and b_geom_clock and a_no_penalty and b_no_penalty:
        label = "crossed_behavior_disappears_into_row_region_control"
    elif a_clock and a_geom_clock and b_vase and b_geom_vase and a_no_penalty and b_no_penalty:
        label = "crossed_behavior_follows_earlier_region"
    elif not a_chimera and not b_chimera and a_penalty and b_penalty:
        label = "hard_mismatch_penalty_persists_without_a_phase_split"
    else:
        label = "mixed"
    return {
        "classification": label,
        "matched_gate_passed": True,
        "interpreted": True,
        "crossed": {
            "competitor_earlier_target_row": {"actual_first_token": a_top, "phrase_geometry_chimera": a_chimera, "penalty_either": a_penalty, "geometry_mechanism": a_geometry_mechanism},
            "target_earlier_competitor_row": {"actual_first_token": b_top, "phrase_geometry_chimera": b_chimera, "penalty_either": b_penalty, "geometry_mechanism": b_geometry_mechanism},
        },
        "strict_precedence": [
            "bidirectional_soft_phase_split",
            "asymmetric_hard_cross_phenotype_persists",
            "crossed_behavior_disappears_into_row_region_control",
            "crossed_behavior_follows_earlier_region",
            "hard_mismatch_penalty_persists_without_a_phase_split",
            "mixed",
        ],
    }


def validate_hard_cross_receipt(path: Path) -> tuple[dict[str, Any], str]:
    """Validate the conclusion-owning hard-cross receipt before model loading."""

    resolved = path.expanduser().resolve(strict=True)
    observed = _sha256_file(resolved)
    if observed != HARD_CROSS_RECEIPT_SHA256:
        raise ValueError("hard-cross receipt SHA-256 does not match frozen contract")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("unit_id") != HARD_CROSS_UNIT_ID:
        raise ValueError("hard-cross receipt has unexpected unit_id")
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != 1:
        raise ValueError("hard-cross receipt must contain exactly one result")
    result = results[0]
    if str(result.get("image_id")) != IMAGE_ID:
        raise ValueError("hard-cross receipt is not the frozen image-139 case")
    if not bool(result.get("hybrid_execution_gate", {}).get("passed")):
        raise ValueError("hard-cross receipt did not pass its execution gate")
    if len(result.get("target_mask_indices", [])) != SELECTED_KEY_COUNT or len(result.get("competitor_mask_indices", [])) != SELECTED_KEY_COUNT:
        raise ValueError("hard-cross receipt does not contain two exact twenty-key supports")
    return payload, observed


def _runtime_attn_implementation(model: Any) -> str | None:
    config = getattr(model, "config", None)
    value = getattr(config, "_attn_implementation", None) or getattr(config, "attn_implementation", None)
    return str(value) if value is not None else None


def _model_parameter_dtypes(model: Any) -> dict[str, Any]:
    dtypes = [str(parameter.dtype) for parameter in model.parameters()]
    return {"unique": sorted(set(dtypes)), "count": len(dtypes), "all_float32": bool(dtypes) and set(dtypes) == {"torch.float32"}}


def _mask_attestation_matches_structure(observed: Mapping[str, Any], structural: Mapping[str, Any]) -> bool:
    return bool(
        observed.get("digest") == structural.get("observed_mask_digest")
        and observed.get("shape") == structural.get("mask_shape")
        and observed.get("dtype") == structural.get("mask_dtype")
        and observed.get("changed_cell_count") == structural.get("changed_cell_count")
        and observed.get("finite_changed_cell_count") == structural.get("observed_finite_changed_cell_count")
    )


def assess_runtime_forward_semantics(attestation: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one forward's raw dtype, mask consumption, and text-layer hooks."""

    hook = attestation.get("text_layer_mask_hook", {})
    passed = bool(
        attestation.get("passed_direct_forward_mask_consumption")
        and attestation.get("raw_forward_logits_dtype") == "torch.float32"
        and hook.get("passed")
    )
    return {
        "passed": passed,
        "raw_logits_float32": attestation.get("raw_forward_logits_dtype") == "torch.float32",
        "direct_mask_consumed": bool(attestation.get("passed_direct_forward_mask_consumption")),
        "text_layer_mask_hook_passed": bool(hook.get("passed")),
    }


def _register_text_layer_mask_hooks(model: Any, expected_mask: torch.Tensor) -> tuple[list[Any], dict[str, Any]]:
    """Trace effective masks at every Qwen3-VL text-attention layer."""

    expected_digest = _sha256_tensor(expected_mask)
    state: dict[str, Any] = {"layers": {}}
    handles: list[Any] = []
    for name, module in model.named_modules():
        class_name = module.__class__.__name__
        if "TextAttention" not in class_name and not ("text" in name.lower() and "attention" in name.lower()):
            continue
        state["layers"][name] = {"count": 0, "observations": [], "backend": getattr(getattr(module, "config", None), "_attn_implementation", None) or getattr(getattr(model, "config", None), "_attn_implementation", None)}

        def hook(mod: Any, args: tuple[Any, ...], kwargs: dict[str, Any], *, layer_name: str = name) -> None:
            entry = state["layers"][layer_name]
            entry["count"] += 1
            mask = kwargs.get("attention_mask")
            if mask is None:
                mask = next((value for value in args if isinstance(value, torch.Tensor) and value.ndim >= 2), None)
            if isinstance(mask, torch.Tensor):
                entry["observations"].append({"digest": _sha256_tensor(mask), "shape": list(mask.shape), "dtype": str(mask.dtype), "same_object": mask is expected_mask})
            else:
                entry["observations"].append({"digest": None, "shape": None, "dtype": None, "same_object": False})

        try:
            handles.append(module.register_forward_pre_hook(hook, with_kwargs=True))
        except TypeError:  # pragma: no cover - older torch fallback
            def legacy_hook(mod: Any, args: tuple[Any, ...], *, layer_name: str = name) -> None:
                hook(mod, args, {}, layer_name=layer_name)
            handles.append(module.register_forward_pre_hook(legacy_hook))
    return handles, {"expected_mask_digest": expected_digest, "expected_mask_shape": list(expected_mask.shape), "expected_mask_dtype": str(expected_mask.dtype), "layers": state["layers"]}


def _finalize_text_layer_mask_hook(receipt: dict[str, Any]) -> dict[str, Any]:
    layers = receipt.get("layers", {})
    checks = {
        "expected_text_layer_count_positive": len(layers) > 0,
        "every_layer_called_once": all(int(item.get("count", 0)) == 1 for item in layers.values()),
        "every_layer_backend_sdpa": all(str(item.get("backend")) in {"sdpa", "scaled_dot_product_attention"} for item in layers.values()),
        "every_layer_mask_exact": all(
            len(item.get("observations", [])) == 1
            and (item["observations"][0].get("same_object") or item["observations"][0].get("digest") == receipt.get("expected_mask_digest"))
            and item["observations"][0].get("shape") == receipt.get("expected_mask_shape")
            and item["observations"][0].get("dtype") == receipt.get("expected_mask_dtype")
            for item in layers.values()
        ),
    }
    receipt["checks"] = checks
    receipt["passed"] = bool(all(checks.values()))
    return receipt


def _module_file_sha256(module_name: str) -> str | None:
    try:
        module = importlib.import_module(module_name)
        path = Path(module.__file__).resolve(strict=True)
    except (ImportError, AttributeError, OSError):
        return None
    return _sha256_file(path)


def runtime_provenance(*, infer_config: Path) -> dict[str, Any]:
    import sys as _sys
    try:
        import transformers
        transformers_version = str(transformers.__version__)
    except ImportError:
        transformers_version = None
    return {
        **hard_cross._config_provenance(infer_config),
        "resolved_infer_config": str(infer_config.expanduser().resolve()),
        "python_version": _sys.version,
        "torch_version": str(torch.__version__),
        "transformers_version": transformers_version,
        "transformers_masking_utils_sha256": _module_file_sha256("transformers.masking_utils"),
        "transformers_sdpa_attention_sha256": _module_file_sha256("transformers.integrations.sdpa_attention"),
    }


def _decode_token(tokenizer: Any, token_id: int) -> str:
    try:
        return str(tokenizer.decode([int(token_id)], skip_special_tokens=False))
    except Exception:  # pragma: no cover - tokenizer implementations vary
        return ""


def _score_soft_row(
    *, model: Any, model_inputs: Mapping[str, Any], features: Any, grid_thw: Sequence[int], merge_size: int,
    prompt_ids: Sequence[int], row: Sequence[int], image_grid_thw: torch.Tensor, position_ids: torch.Tensor,
    custom_mask: torch.Tensor, image_token_id: int, tokenizer: Any, terminal_token_id: int | None,
) -> dict[str, Any]:
    """Score one row and attest the exact additive mask consumed by forward."""

    from types import MethodType

    device = next(model.parameters()).device
    ids = torch.tensor([list(prompt_ids) + [int(v) for v in row]], dtype=torch.long, device=device)
    observed: dict[str, Any] = {"forward_calls": 0, "mask_identity_passed": False}
    original_forward = model.forward
    hook_handles, hook_receipt = _register_text_layer_mask_hooks(model, custom_mask)

    def wrapped_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        mask = kwargs.get("attention_mask")
        observed["forward_calls"] += 1
        observed["mask_identity_passed"] = mask is custom_mask
        if isinstance(mask, torch.Tensor):
            observed["mask_digest"] = _sha256_tensor(mask)
            observed["mask_shape"] = list(mask.shape)
            observed["mask_dtype"] = str(mask.dtype)
        observed["use_cache"] = kwargs.get("use_cache")
        observed["has_logits_processor"] = "logits_processor" in kwargs
        observed["has_repetition_penalty"] = "repetition_penalty" in kwargs
        output = original_forward(*args, **kwargs)
        observed["raw_forward_logits_dtype"] = str(output.logits.dtype)
        return output

    model.forward = MethodType(wrapped_forward, model)
    try:
        logits = query._score_model_sequence(
            model=model,
            model_inputs=model_inputs,
            input_ids=ids,
            attention_mask=torch.ones_like(ids, dtype=torch.long),
            position_ids=position_ids,
            custom_mask=custom_mask,
            features=features,
            merge_size=merge_size,
            grid_thw=grid_thw,
        )
    finally:
        model.forward = original_forward
        for handle in hook_handles:
            handle.remove()
    hook_receipt = _finalize_text_layer_mask_hook(hook_receipt)

    scored = query.score_row_log_likelihoods(
        logits,
        prefix_length=len(prompt_ids),
        row_tokens=row,
        description_length=len(row) - 8,
        terminal_token_id=terminal_token_id,
    )
    row_steps = torch.stack([logits[len(prompt_ids) + index - 1] for index in range(len(row))], dim=0)
    row_token_tensor = torch.tensor([int(v) for v in row], dtype=torch.long)
    selected_logits = row_steps.gather(1, row_token_tensor.unsqueeze(1)).squeeze(1)
    top_values, top_ids = row_steps.max(dim=1)
    scored["selected_token_logits"] = [float(v) for v in selected_logits.tolist()]
    scored["top_prediction_logits"] = [float(v) for v in top_values.tolist()]
    scored["top_prediction_token_text"] = [_decode_token(tokenizer, int(v)) for v in top_ids.tolist()]
    scored["runtime_attestation"] = {
        **observed,
        "passed_direct_forward_mask_consumption": bool(
            observed.get("forward_calls") == 1 and observed.get("mask_identity_passed")
        ),
        "text_layer_mask_hook": hook_receipt,
        "attention_implementation": _runtime_attn_implementation(model),
        "model_parameter_dtype": str(next(model.parameters()).dtype),
        "scoring_logits_dtype": str(logits.dtype),
        "expected_mask_digest": _sha256_tensor(custom_mask),
        "expected_mask_shape": list(custom_mask.shape),
        "expected_mask_dtype": str(custom_mask.dtype),
        **_changed_cell_counts(custom_mask),
    }
    scored["runtime_attestation"]["runtime_forward_semantics"] = assess_runtime_forward_semantics(scored["runtime_attestation"])
    return scored


def _score_soft_arm(
    *, model: Any, model_inputs: Mapping[str, Any], features: Any, grid_thw: Sequence[int], merge_size: int,
    prompt_ids: Sequence[int], target_row: Sequence[int], competitor_row: Sequence[int], image_token_id: int,
    earlier_indices: Sequence[int], row_indices: Sequence[int], image_grid_thw: torch.Tensor,
    tokenizer: Any, terminal_token_id: int | None,
) -> dict[str, Any]:
    """Score both canonical rows under one crossed finite-bias arm."""

    device = next(model.parameters()).device
    scores: dict[str, Any] = {}
    structures: dict[str, Any] = {}
    identities: dict[str, Any] = {}
    masks: dict[str, Any] = {}
    for owner, row in (("target", target_row), ("competitor", competitor_row)):
        ids = torch.tensor([list(prompt_ids) + list(row)], dtype=torch.long, device=device)
        image_positions = [int(v) for v in torch.where(ids[0] == int(image_token_id))[0].tolist()]
        earlier_positions = [image_positions[int(v)] for v in earlier_indices]
        row_positions = [image_positions[int(v)] for v in row_indices]
        positions = factorial.derive_explicit_position_ids(
            model, input_ids=ids, attention_mask=torch.ones_like(ids), image_grid_thw=image_grid_thw.to(device=device)
        )
        mask = build_soft_cross_region_additive_mask(
            sequence_length=ids.shape[1], image_key_positions=image_positions,
            earlier_selected_image_positions=earlier_positions, row_selected_image_positions=row_positions,
            prefix_length=len(prompt_ids), row_length=len(row), device=device,
        )
        structures[owner] = inspect_soft_cross_region_mask_structure(
            mask, sequence_length=ids.shape[1], image_key_positions=image_positions,
            earlier_selected_image_positions=earlier_positions, row_selected_image_positions=row_positions,
            prefix_length=len(prompt_ids), row_length=len(row),
        )
        scores[owner] = _score_soft_row(
            model=model, model_inputs=model_inputs, features=features, grid_thw=grid_thw, merge_size=merge_size,
            prompt_ids=prompt_ids, row=row, image_grid_thw=image_grid_thw, position_ids=positions,
            custom_mask=mask, image_token_id=image_token_id, tokenizer=tokenizer, terminal_token_id=terminal_token_id,
        )
        masks[owner] = {
            "digest": _sha256_tensor(mask), "shape": list(mask.shape), "dtype": str(mask.dtype),
            **_changed_cell_counts(mask),
        }
        identities[owner] = {
            "input_ids_sha256": factorial.tensor_sha256(ids),
            "position_ids_sha256": factorial.tensor_sha256(positions),
        }
    first_diff = factorial.first_differing_description_index(target_row, competitor_row, description_length=len(target_row) - 8)
    result = {"target": scores["target"], "competitor": scores["competitor"], "structural_mask_receipt": structures, "input_and_position_identities": identities, "observed_masks": masks}
    if first_diff is not None:
        result["first_differing_description"] = {
            owner: {"sum": scores[owner]["token_log_probabilities"][first_diff], "mean": scores[owner]["token_log_probabilities"][first_diff], "count": 1}
            for owner in ("target", "competitor")
        }
    return result


def _score_soft_noop(
    *, model: Any, model_inputs: Mapping[str, Any], features: Any, grid_thw: Sequence[int], merge_size: int,
    prompt_ids: Sequence[int], target_row: Sequence[int], competitor_row: Sequence[int], image_token_id: int,
    image_grid_thw: torch.Tensor, tokenizer: Any, terminal_token_id: int | None,
) -> dict[str, Any]:
    """Score a zero float32 additive mask as the direct no-operation arm."""

    device = next(model.parameters()).device
    result: dict[str, Any] = {}
    for owner, row in (("target", target_row), ("competitor", competitor_row)):
        ids = torch.tensor([list(prompt_ids) + list(row)], dtype=torch.long, device=device)
        positions = factorial.derive_explicit_position_ids(
            model, input_ids=ids, attention_mask=torch.ones_like(ids), image_grid_thw=image_grid_thw.to(device=device)
        )
        mask = build_zero_float32_causal_mask(sequence_length=ids.shape[1], device=device)
        result[owner] = _score_soft_row(
            model=model, model_inputs=model_inputs, features=features, grid_thw=grid_thw, merge_size=merge_size,
            prompt_ids=prompt_ids, row=row, image_grid_thw=image_grid_thw, position_ids=positions,
            custom_mask=mask, image_token_id=image_token_id, tokenizer=tokenizer, terminal_token_id=terminal_token_id,
        )
    return result


def _soft_noop_parity(base: Mapping[str, Any], noop: Mapping[str, Any]) -> dict[str, Any]:
    owners: dict[str, Any] = {}
    passed = True
    for owner in ("target", "competitor"):
        left = base[owner]
        right = noop[owner]
        drift = max(abs(float(a) - float(b)) for a, b in zip(left["token_log_probabilities"], right["token_log_probabilities"], strict=True))
        ranks_equal = [int(v) for v in left["selected_token_ranks"]] == [int(v) for v in right["selected_token_ranks"]]
        owners[owner] = {"max_abs_logprob_drift": drift, "selected_token_ranks_equal": ranks_equal, "passed": bool(drift <= TOLERANCE and ranks_equal)}
        passed = passed and owners[owner]["passed"]
    return {"passed": passed, "owners": owners, "tolerance": TOLERANCE}


def _run_case_soft(**kwargs: Any) -> dict[str, Any]:
    if _FROZEN_HARD_CROSS_CASE is None:
        raise RuntimeError("hard-cross case was not initialized")
    base = _ORIGINAL_QUERY_RUN_CASE(**kwargs)
    frozen = _FROZEN_HARD_CROSS_CASE
    plan = build_soft_cross_arm_plan(
        frozen_case=frozen, target_indices=kwargs["target_indices"], competitor_indices=kwargs["competitor_indices"]
    )
    model = kwargs["qwen"].model
    common = dict(
        model=model, model_inputs=kwargs["model_inputs"], features=kwargs["features"], grid_thw=kwargs["grid_thw"],
        merge_size=kwargs["merge_size"], prompt_ids=kwargs["prompt_ids"], target_row=kwargs["target_row"],
        competitor_row=kwargs["competitor_row"], image_token_id=kwargs["image_token_id"],
        image_grid_thw=kwargs["image_grid_thw"], tokenizer=kwargs["qwen"].tokenizer,
        terminal_token_id=kwargs["qwen"].tokenizer.eos_token_id,
    )
    soft_arms = {
        name: _score_soft_arm(
            **common, earlier_indices=arm["earlier_indices"], row_indices=arm["row_indices"]
        )
        for name, arm in plan.items()
    }
    noop = _score_soft_noop(**common)
    noop_gate = _soft_noop_parity(base["arms"]["all_allowed_4d"], noop)
    hard_frozen_baseline = factorial.assess_cross_receipt_baseline(
        live=base["arms"]["all_allowed_4d"], frozen=frozen["frozen_comparators"]["unrestricted"]
    )
    frozen_identity_reference = frozen.get("hybrid_arms", {}).get("target_earlier_competitor_row", {}).get("input_and_position_identities", {})
    identity_checks = {
        name: hard_cross.compare_input_position_identities(
            observed=arm.get("input_and_position_identities", {}), expected=frozen_identity_reference
        )
        for name, arm in soft_arms.items()
    }
    signatures = {
        name: build_phase_signatures(
            unrestricted=frozen["frozen_comparators"]["unrestricted"],
            hybrid=arm,
            row_only=plan[name]["row_only_reference"],
            matched=plan[name]["matched_all_query_reference"],
        )
        for name, arm in soft_arms.items()
    }
    base_first = frozen["frozen_comparators"]["unrestricted"].get("first_differing_description", {})
    for name, arm in soft_arms.items():
        current_first = arm.get("first_differing_description", {})
        if current_first and base_first:
            target_current = float(current_first["target"]["mean"])
            competitor_current = float(current_first["competitor"]["mean"])
            target_base = float(base_first["target"]["mean"])
            competitor_base = float(base_first["competitor"]["mean"])
            signatures[name]["first_differing_description"] = {
                "vase_mean": target_current,
                "clock_mean": competitor_current,
                "gamma_vase_minus_clock": target_current - competitor_current,
                "gamma": target_current - competitor_current,
                "Delta_vase": target_current - target_base,
                "Delta_clock": competitor_current - competitor_base,
            }
        mechanisms: dict[str, Any] = {}
        for phase, values in signatures[name].items():
            if not isinstance(values, Mapping) or "gamma_vase_minus_clock" not in values:
                continue
            gamma = float(values["gamma_vase_minus_clock"])
            preferred = "vase" if gamma >= OWNER_MARGIN_FLOOR else "clock" if gamma <= -OWNER_MARGIN_FLOOR else None
            if preferred is None:
                continue
            preferred_release = float(values.get(f"Delta_{preferred}", 0.0))
            alternative = "clock" if preferred == "vase" else "vase"
            mechanisms[phase] = classify_owner_activation_suppression(
                gamma=gamma,
                preferred_owner_release=preferred_release,
                alternative_owner_release=float(values.get(f"Delta_{alternative}", 0.0)),
            )
        signatures[name]["owner_mechanisms"] = mechanisms
    first_diff = factorial.first_differing_description_index(kwargs["target_row"], kwargs["competitor_row"], description_length=len(kwargs["target_row"]) - 8)
    matched_gate = assess_matched_control_gate(
        plan=plan, arms=soft_arms, phase_signatures=signatures,
        target_row=kwargs["target_row"], competitor_row=kwargs["competitor_row"], first_differing_index=first_diff,
    )
    crossed = classify_crossed_panel(
        phase_signatures=signatures, arms=soft_arms, target_row=kwargs["target_row"],
        competitor_row=kwargs["competitor_row"], first_differing_index=first_diff, matched_gate=matched_gate,
    )
    structural_passed = all(
        bool(arm.get("structural_mask_receipt", {}).get(owner, {}).get("passed"))
        for arm in soft_arms.values() for owner in ("target", "competitor")
    )
    parent_passed = bool(base.get("parent_continuity", {}).get("passed"))
    mask_receipts_match = all(
        _mask_attestation_matches_structure(
            arm["observed_masks"][owner], arm["structural_mask_receipt"][owner]
        )
        for arm in soft_arms.values() for owner in ("target", "competitor")
    )
    noop_mask_attestations_match = all(
        noop[owner]["runtime_attestation"].get("expected_mask_digest") == noop[owner]["runtime_attestation"].get("mask_digest")
        and noop[owner]["runtime_attestation"].get("expected_mask_shape") == noop[owner]["runtime_attestation"].get("mask_shape")
        and noop[owner]["runtime_attestation"].get("expected_mask_dtype") == noop[owner]["runtime_attestation"].get("mask_dtype")
        and noop[owner]["runtime_attestation"].get("changed_cell_count") == 0
        for owner in ("target", "competitor")
    )
    model_dtypes = _model_parameter_dtypes(model)
    all_forward_records = [arm[owner] for arm in soft_arms.values() for owner in ("target", "competitor")] + [noop[owner] for owner in ("target", "competitor")]
    runtime_semantics_passed = all(bool(record["runtime_attestation"].get("runtime_forward_semantics", {}).get("passed")) for record in all_forward_records)
    execution_gate = {
        "passed": bool(structural_passed and noop_gate["passed"] and parent_passed and mask_receipts_match),
        "structural_masks_passed": structural_passed,
        "soft_noop_gate": noop_gate,
        "hard_frozen_baseline_parity": hard_frozen_baseline,
        "input_and_position_identity_checks": identity_checks,
        "parent_continuity_passed": parent_passed,
        "feature_continuity_passed": None,
        "row_contract_passed": None,
        "deferred_outer_fields": ["feature_continuity_passed", "row_contract_passed"],
        "mask_attestations_match_structural_receipts": mask_receipts_match,
        "noop_mask_attestations_match": noop_mask_attestations_match,
        "model_parameter_dtypes": model_dtypes,
        "all_runtime_forward_semantics_passed": runtime_semantics_passed,
        "runtime_requirements": {
            "all_direct_forward_masks_consumed": all(
                bool(arm[owner]["runtime_attestation"].get("passed_direct_forward_mask_consumption"))
                for arm in soft_arms.values() for owner in ("target", "competitor")
            ) and all(bool(noop[owner]["runtime_attestation"].get("passed_direct_forward_mask_consumption")) for owner in ("target", "competitor")),
            "all_float32": model_dtypes["all_float32"] and all(
                arm[owner]["runtime_attestation"].get("scoring_logits_dtype") == "torch.float32"
                for arm in soft_arms.values() for owner in ("target", "competitor")
            ) and all(noop[owner]["runtime_attestation"].get("scoring_logits_dtype") == "torch.float32" for owner in ("target", "competitor")),
            "all_use_cache_false": all(
                arm[owner]["runtime_attestation"].get("use_cache") is False
                for arm in soft_arms.values() for owner in ("target", "competitor")
            ) and all(noop[owner]["runtime_attestation"].get("use_cache") is False for owner in ("target", "competitor")),
            "no_logits_processor_or_repetition_penalty": all(
                not arm[owner]["runtime_attestation"].get("has_logits_processor") and not arm[owner]["runtime_attestation"].get("has_repetition_penalty")
                for arm in soft_arms.values() for owner in ("target", "competitor")
            ) and all(not noop[owner]["runtime_attestation"].get("has_logits_processor") and not noop[owner]["runtime_attestation"].get("has_repetition_penalty") for owner in ("target", "competitor")),
            "attention_implementation": _runtime_attn_implementation(model),
        },
    }
    execution_gate["passed"] = bool(
        execution_gate["passed"]
        and hard_frozen_baseline["passed"]
        and all(bool(value.get("passed")) for value in identity_checks.values())
        and execution_gate["mask_attestations_match_structural_receipts"]
        and execution_gate["noop_mask_attestations_match"]
        and execution_gate["all_runtime_forward_semantics_passed"]
        and execution_gate["runtime_requirements"]["all_direct_forward_masks_consumed"]
        and execution_gate["runtime_requirements"]["all_float32"]
        and execution_gate["runtime_requirements"]["all_use_cache_false"]
        and execution_gate["runtime_requirements"]["no_logits_processor_or_repetition_penalty"]
        and execution_gate["runtime_requirements"]["attention_implementation"] in {"sdpa", "scaled_dot_product_attention"}
    )
    return {
        **base,
        "soft_cross_arms": soft_arms,
        "soft_noop_arm": noop,
        "soft_phase_signatures": signatures,
        "matched_control_gate": matched_gate,
        "crossed_panel_classification": crossed,
        "soft_execution_gate": execution_gate,
        "classification": "valid_count_balanced_soft_cross_execution" if execution_gate["passed"] else "invalid_count_balanced_soft_cross_execution_gate",
    }


def finalize_soft_execution_gate(result: Mapping[str, Any]) -> dict[str, Any]:
    """Apply outer query-run fields after they are attached to the case."""

    gate = dict(result.get("soft_execution_gate", {}))
    feature_passed = bool(result.get("feature_continuity", {}).get("passed"))
    row_passed = bool(result.get("row_contract", {}).get("passed"))
    gate["feature_continuity_passed"] = feature_passed
    gate["row_contract_passed"] = row_passed
    gate["deferred_outer_fields"] = []
    gate["passed"] = bool(gate.get("passed") and feature_passed and row_passed)
    output = dict(result)
    output["soft_execution_gate"] = gate
    output["classification"] = "valid_count_balanced_soft_cross_execution" if gate["passed"] else "invalid_count_balanced_soft_cross_execution_gate"
    if not gate["passed"] and isinstance(output.get("crossed_panel_classification"), Mapping):
        crossed = dict(output["crossed_panel_classification"])
        crossed["interpreted"] = False
        crossed["invalidated_by_execution_gate"] = True
        output["crossed_panel_classification"] = crossed
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = query.build_parser()
    parser.description = __doc__
    parser.add_argument("--hard-cross-receipt", type=Path, default=DEFAULT_HARD_CROSS_RECEIPT)
    parser.add_argument("--historical-soft-receipt", type=Path, default=DEFAULT_HISTORICAL_SOFT_RECEIPT)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    global _FROZEN_HARD_CROSS_CASE
    try:
        hard_payload, hard_sha = validate_hard_cross_receipt(args.hard_cross_receipt)
        if _sha256_file(args.historical_soft_receipt) != HISTORICAL_SOFT_RECEIPT_SHA256:
            raise ValueError("historical soft receipt SHA-256 does not match frozen contract")
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    _FROZEN_HARD_CROSS_CASE = hard_payload["results"][0]
    original = query._run_case
    query._run_case = _run_case_soft
    try:
        payload = query.run(args)
    finally:
        query._run_case = original
        _FROZEN_HARD_CROSS_CASE = None
    payload["results"] = [finalize_soft_execution_gate(result) for result in payload.get("results", [])]
    payload["schema_version"] = f"{UNIT_ID}.v1"
    payload["unit_id"] = UNIT_ID
    payload["hard_cross_receipt"] = str(args.hard_cross_receipt.expanduser().resolve())
    payload["hard_cross_receipt_sha256"] = hard_sha
    payload["historical_soft_receipt"] = str(args.historical_soft_receipt.expanduser().resolve())
    payload["historical_soft_receipt_sha256"] = HISTORICAL_SOFT_RECEIPT_SHA256
    payload["runtime_provenance"] = runtime_provenance(infer_config=args.infer_config)
    payload["imported_runner_sha256"] = _sha256_file(Path(hard_cross.__file__))
    payload["imported_query_scorer_sha256"] = _sha256_file(Path(query.__file__))
    payload["runner_sha256"] = _sha256_file(Path(__file__))
    result = payload["results"][0]
    payload["panel_decision"] = {
        "classification": result.get("classification"),
        "image_id": IMAGE_ID,
        "soft_execution_gate_passed": bool(result.get("soft_execution_gate", {}).get("passed")),
        "matched_control_gate_passed": bool(result.get("matched_control_gate", {}).get("passed")),
    }
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    normalized_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    if [str(value) for value in args.image_ids] != [IMAGE_ID]:
        raise SystemExit("this unit executes exactly image 139")
    payload = run(args)
    payload["normalized_argv"] = normalized_argv
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "receipt.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
