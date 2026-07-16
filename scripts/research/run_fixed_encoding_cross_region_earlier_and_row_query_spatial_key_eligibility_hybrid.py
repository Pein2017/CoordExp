#!/usr/bin/env python3
"""Run the image-139 crossed earlier/query spatial-key hybrid probe.

This is an experiment-local teacher-forced scorer.  It reuses the accepted
factorial and query-only runners, captures one float32 feature bundle, and
scores only the two crossed regional masks that were absent from the frozen
factorial.  No generation, training, resizing, or shared-model modification is
performed.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import scripts.research.run_fixed_encoding_query_scoped_object_centered_spatial_eligibility as query  # noqa: E402
import scripts.research.run_fixed_encoding_earlier_query_only_spatial_key_eligibility_factorial as factorial  # noqa: E402


UNIT_ID = "2026-07-15-fixed-encoding-cross-region-earlier-and-row-query-spatial-key-eligibility-hybrid"
FACTORIZATION_PARENT_UNIT_ID = factorial.UNIT_ID
FACTORIZATION_PARENT_RECEIPT_SHA256 = "e819216b4b4c012bf55c332c2f6d773a2c694b1440347ade446c024c5a7d3441"
DEFAULT_FACTORIAL_RECEIPT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-15-fixed-encoding-earlier-query-only-spatial-key-eligibility-factorial/"
    "image139-float32-20260715a/receipt.json"
)
IMAGE_ID = "139"
TOLERANCE = 1e-4
OWNER_MARGIN_FLOOR = 0.10
OWNER_RELEASE_FLOOR = 0.05
OWNER_DESTRUCTION_TOLERANCE = 0.05

_FROZEN_FACTORIAL_CASE: Mapping[str, Any] | None = None
_ORIGINAL_QUERY_RUN_CASE = query._run_case
HYBRID_REGION_MAPPING = {
    "target_earlier_competitor_row": ("target", "competitor"),
    "competitor_earlier_target_row": ("competitor", "target"),
}


def hybrid_region_mapping(arm_name: str) -> tuple[str, str]:
    """Return (earlier-region, row-scoring-region) for a hybrid arm."""

    try:
        return HYBRID_REGION_MAPPING[str(arm_name)]
    except KeyError as exc:
        raise ValueError(f"unknown hybrid arm {arm_name!r}") from exc


def build_hybrid_arm_plan(
    *,
    frozen_case: Mapping[str, Any],
    target_indices: Sequence[int],
    competitor_indices: Sequence[int],
) -> dict[str, dict[str, Any]]:
    """Build one source of truth for both crossed region-assignment arms."""

    earlier_arms = frozen_case["earlier_query_only_arms"]
    row_arms = frozen_case["arms"]
    region_indices = {"target": [int(v) for v in target_indices], "competitor": [int(v) for v in competitor_indices]}
    plan: dict[str, dict[str, Any]] = {}
    for arm_name, (earlier_region, row_region) in HYBRID_REGION_MAPPING.items():
        plan[arm_name] = {
            "earlier_region": earlier_region,
            "row_region": row_region,
            "earlier_indices": region_indices[earlier_region],
            "row_indices": region_indices[row_region],
            "earlier_reference": earlier_arms[earlier_region],
            "row_only_reference": row_arms[f"{row_region}_row_query_only_hard"],
            "matched_all_query_reference": frozen_case["parent_hard_endpoint"][f"{row_region}_eligibility"],
        }
    return plan


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve(strict=True).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _config_provenance(path: Path) -> dict[str, str | None]:
    """Bind resolved config, base-model, and adapter identifiers without hardcoding them."""

    try:
        import yaml
    except ImportError:  # pragma: no cover - runtime environment owns PyYAML
        return {"infer_config": str(path.expanduser().resolve()), "base_model": None, "adapter": None}
    raw = yaml.safe_load(path.expanduser().resolve(strict=True).read_text(encoding="utf-8")) or {}
    model = raw.get("model") if isinstance(raw, Mapping) else {}
    adapter = raw.get("adapter") if isinstance(raw, Mapping) else {}
    base = model.get("base_model") if isinstance(model, Mapping) else None
    adapter_path = adapter.get("path") if isinstance(adapter, Mapping) else None
    resolved_adapter = None
    if adapter_path:
        resolved_adapter = str((path.expanduser().resolve().parent / str(adapter_path)).resolve())
    return {
        "infer_config": str(path.expanduser().resolve()),
        "base_model": str(base) if base is not None else None,
        "adapter": resolved_adapter,
    }


def validate_factorial_receipt(path: Path) -> tuple[dict[str, Any], str]:
    """Validate digest and nested receipt bindings before parsing payload data."""

    resolved = Path(path).expanduser().resolve(strict=True)
    observed = _sha256_file(resolved)
    if observed != FACTORIZATION_PARENT_RECEIPT_SHA256:
        raise ValueError("factorial receipt SHA-256 does not match frozen contract")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("unit_id") != FACTORIZATION_PARENT_UNIT_ID:
        raise ValueError("factorial receipt has unexpected unit_id")
    if payload.get("query_receipt_sha256") != factorial.QUERY_RECEIPT_SHA256:
        raise ValueError("factorial receipt does not bind the frozen query-only receipt")
    if payload.get("hard_parent_receipt_sha256") != query.PARENT_RECEIPT_SHA256:
        raise ValueError("factorial receipt does not bind the frozen hard parent receipt")
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != 1:
        raise ValueError("factorial receipt must contain exactly one result")
    if str(results[0].get("image_id")) != IMAGE_ID:
        raise ValueError("factorial receipt is not the frozen image-139 case")
    return payload, observed


def validate_exact_image_ids(values: Sequence[Any]) -> list[str]:
    requested = [str(value) for value in values]
    if requested != [IMAGE_ID]:
        raise ValueError("this hybrid executes exactly image 139")
    return requested


def hybrid_query_range(*, prefix_length: int, row_length: int) -> tuple[int, int]:
    prefix = int(prefix_length)
    length = int(row_length)
    if prefix < 2 or length < 2:
        raise ValueError("prefix_length and row_length must be at least two")
    return prefix - 1, prefix + length - 2


def build_hybrid_key_eligibility_mask(
    *,
    sequence_length: int,
    image_key_positions: Sequence[int],
    earlier_eligible_image_positions: Sequence[int],
    row_eligible_image_positions: Sequence[int],
    prefix_length: int,
    row_length: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Build the exact causal union of an earlier and current-row regional mask."""

    length = int(sequence_length)
    if length <= 0:
        raise ValueError("sequence_length must be positive")
    image = {int(v) for v in image_key_positions}
    earlier = {int(v) for v in earlier_eligible_image_positions}
    row = {int(v) for v in row_eligible_image_positions}
    if not earlier.issubset(image) or not row.issubset(image):
        raise ValueError("eligible image keys must be subsets of image keys")
    if any(v < 0 or v >= length for v in image | earlier | row):
        raise ValueError("image key position is outside the sequence")
    row_start, row_end = hybrid_query_range(prefix_length=prefix_length, row_length=row_length)
    if row_end >= length:
        raise ValueError("row-scoring query range is outside the sequence")
    mask = torch.tril(torch.ones((length, length), dtype=torch.bool, device=device))
    blocked_earlier = image - earlier
    blocked_row = image - row
    if blocked_earlier:
        keys = torch.tensor(sorted(blocked_earlier), dtype=torch.long, device=device)
        mask[:prefix_length - 1, keys] = False
    if blocked_row:
        keys = torch.tensor(sorted(blocked_row), dtype=torch.long, device=device)
        mask[row_start : row_end + 1, keys] = False
    return mask.unsqueeze(0).unsqueeze(0).contiguous()


def inspect_hybrid_mask_structure(
    hybrid_mask: torch.Tensor,
    *,
    sequence_length: int,
    image_key_positions: Sequence[int],
    earlier_eligible_image_positions: Sequence[int],
    row_eligible_image_positions: Sequence[int],
    prefix_length: int,
    row_length: int,
    earlier_reference_mask: torch.Tensor,
    row_reference_mask: torch.Tensor,
    earlier_reference_receipt: Mapping[str, Any],
    row_reference_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove exact component equality, disjointness, and union semantics."""

    length = int(sequence_length)
    if tuple(hybrid_mask.shape) != (1, 1, length, length) or hybrid_mask.dtype is not torch.bool:
        raise ValueError("hybrid mask must be boolean [1,1,S,S]")
    if tuple(earlier_reference_mask.shape) != tuple(hybrid_mask.shape) or tuple(row_reference_mask.shape) != tuple(hybrid_mask.shape):
        raise ValueError("reference masks must have the hybrid mask shape")
    device = hybrid_mask.device
    baseline = torch.tril(torch.ones((length, length), dtype=torch.bool, device=device))
    observed = hybrid_mask[0, 0]
    earlier = earlier_reference_mask[0, 0]
    row = row_reference_mask[0, 0]
    earlier_changed = earlier != baseline
    row_changed = row != baseline
    hybrid_changed = observed != baseline
    expected = build_hybrid_key_eligibility_mask(
        sequence_length=length,
        image_key_positions=image_key_positions,
        earlier_eligible_image_positions=earlier_eligible_image_positions,
        row_eligible_image_positions=row_eligible_image_positions,
        prefix_length=prefix_length,
        row_length=row_length,
        device=device,
    )[0, 0]
    expected_earlier = factorial.build_earlier_query_only_key_eligibility_mask(
        sequence_length=length,
        image_key_positions=image_key_positions,
        eligible_image_positions=earlier_eligible_image_positions,
        prefix_length=prefix_length,
        device=device,
    )[0, 0]
    expected_row = query.build_query_scoped_key_eligibility_mask(
        sequence_length=length,
        image_key_positions=image_key_positions,
        eligible_image_positions=row_eligible_image_positions,
        prefix_length=prefix_length,
        row_length=row_length,
        device=device,
    )[0, 0]
    row_start, row_end = hybrid_query_range(prefix_length=prefix_length, row_length=row_length)
    relevant_end = row_end
    scored = slice(0, relevant_end + 1)
    intersection = earlier_changed[scored] & row_changed[scored]
    union = earlier_changed[scored] | row_changed[scored]
    non_image = hybrid_changed.clone()
    image = {int(v) for v in image_key_positions}
    if image:
        key_tensor = torch.tensor(sorted(image), dtype=torch.long, device=device)
        non_image[:, key_tensor] = False
    lower_triangular = torch.tril(torch.ones_like(observed))
    future_changed = hybrid_changed & ~lower_triangular
    out_of_scope_changed = hybrid_changed.clone()
    out_of_scope_changed[: row_end + 1, :] = False
    out_of_scope_changed &= lower_triangular
    # q=P+K-1 is deliberately excluded from row scoring; every lower-triangular
    # change after that query is therefore an explicit structural failure.
    future_equal = int(future_changed.sum().item()) == 0
    frozen_metadata = {
        "earlier_query_range_matches": earlier_reference_receipt.get("earlier_query_range") == {"start_inclusive": 0, "end_inclusive": int(prefix_length) - 2},
        "row_query_range_matches": (
            row_reference_receipt.get("query_range", {}).get("start_inclusive") == row_start
            and row_reference_receipt.get("query_range", {}).get("end_inclusive") == row_end
        ),
        "earlier_blocked_key_count_matches": earlier_reference_receipt.get("blocked_image_key_count") == len(set(int(v) for v in image_key_positions) - set(int(v) for v in earlier_eligible_image_positions)),
        "row_blocked_key_count_matches": row_reference_receipt.get("blocked_image_key_count") == len(set(int(v) for v in image_key_positions) - set(int(v) for v in row_eligible_image_positions)),
        "earlier_changed_cell_count_matches": earlier_reference_receipt.get("earlier_changed_cell_count") == int(earlier_changed.sum().item()),
        "row_changed_cell_count_matches": row_reference_receipt.get("changed_cell_count") == int(row_changed.sum().item()),
    }
    frozen_metadata["passed"] = all(bool(value) for value in frozen_metadata.values())
    exact_20 = len(set(int(v) for v in earlier_eligible_image_positions)) == 20 and len(set(int(v) for v in row_eligible_image_positions)) == 20
    receipt = {
        "passed": False,
        "earlier_query_range": {"start_inclusive": 0, "end_inclusive": int(prefix_length) - 2},
        "row_scoring_query_range": {"start_inclusive": row_start, "end_inclusive": row_end},
        "excluded_unscored_final_query": row_end + 1,
        "earlier_eligible_image_key_count": len(set(int(v) for v in earlier_eligible_image_positions)),
        "row_eligible_image_key_count": len(set(int(v) for v in row_eligible_image_positions)),
        "exact_twenty_eligible_image_keys_each": exact_20,
        "exact_expected_hybrid_mask": bool(torch.equal(observed, expected)),
        "hybrid_changed_cell_count": int(hybrid_changed.sum().item()),
        "earlier_component_matches_independent_construction": bool(torch.equal(earlier, expected_earlier)),
        "row_component_matches_independent_construction": bool(torch.equal(row, expected_row)),
        "frozen_component_metadata": frozen_metadata,
        "earlier_and_row_changed_sets_disjoint": bool(not torch.any(intersection).item()),
        "component_union_equals_hybrid_on_scored_slice": bool(torch.equal(union, hybrid_changed[scored])),
        "changed_non_image_key_cell_count": int(non_image.sum().item()),
        "future_changed_cell_count": int(future_changed.sum().item()),
        "out_of_scope_lower_triangular_changed_cell_count": int(out_of_scope_changed.sum().item()),
        "causal_future_key_blocking_unchanged": future_equal,
        "mask_shape": list(hybrid_mask.shape),
        "mask_dtype": str(hybrid_mask.dtype),
    }
    receipt["passed"] = bool(
        exact_20
        and receipt["exact_expected_hybrid_mask"]
        and receipt["earlier_component_matches_independent_construction"]
        and receipt["row_component_matches_independent_construction"]
        and frozen_metadata["passed"]
        and receipt["earlier_and_row_changed_sets_disjoint"]
        and receipt["component_union_equals_hybrid_on_scored_slice"]
        and receipt["changed_non_image_key_cell_count"] == 0
        and receipt["future_changed_cell_count"] == 0
        and receipt["out_of_scope_lower_triangular_changed_cell_count"] == 0
        and future_equal
    )
    return receipt


def _phase_means(arm: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    phases = query._phase_score_map(arm)
    return {
        phase: {"vase": float(values["target"]["mean"]), "clock": float(values["competitor"]["mean"])}
        for phase, values in phases.items()
    }


def build_phase_signatures(
    *,
    unrestricted: Mapping[str, Any],
    hybrid: Mapping[str, Any],
    row_only: Mapping[str, Any],
    matched: Mapping[str, Any],
) -> dict[str, Any]:
    """Emit primary unrestricted deltas and distinctly named diagnostics."""

    base = _phase_means(unrestricted)
    current = _phase_means(hybrid)
    row = _phase_means(row_only)
    matched_means = _phase_means(matched)
    phases = sorted(set(base) & set(current) & set(row) & set(matched_means))
    output: dict[str, Any] = {}
    for phase in phases:
        gamma = current[phase]["vase"] - current[phase]["clock"]
        output[phase] = {
            "vase_mean": current[phase]["vase"],
            "clock_mean": current[phase]["clock"],
            "gamma_vase_minus_clock": gamma,
            "gamma": gamma,
            "Delta_vase": current[phase]["vase"] - base[phase]["vase"],
            "Delta_clock": current[phase]["clock"] - base[phase]["clock"],
            "secondary_delta_vase_to_same_row_region_row_only": current[phase]["vase"] - row[phase]["vase"],
            "secondary_delta_clock_to_same_row_region_row_only": current[phase]["clock"] - row[phase]["clock"],
            "secondary_delta_vase_to_matched_arm": current[phase]["vase"] - matched_means[phase]["vase"],
            "secondary_delta_clock_to_matched_arm": current[phase]["clock"] - matched_means[phase]["clock"],
        }
    return output


def classify_hybrid_effect(
    phase_signatures: Mapping[str, Mapping[str, Any]],
    *,
    margin_floor: float = OWNER_MARGIN_FLOOR,
    release_floor: float = OWNER_RELEASE_FLOOR,
    destruction_tolerance: float = OWNER_DESTRUCTION_TOLERANCE,
) -> dict[str, Any]:
    """Classify deterministic activation/suppression movement by phase.

    ``full_row`` is reported but cannot establish a phrase/geometry chimera by
    itself; chimera status requires opposite semantic and geometry phase owners.
    """

    phase_labels: dict[str, str] = {}
    for phase, values in phase_signatures.items():
        gamma = float(values.get("gamma_vase_minus_clock", 0.0))
        delta_vase = float(values.get("Delta_vase", 0.0))
        delta_clock = float(values.get("Delta_clock", 0.0))
        observed_owner = "vase" if gamma >= margin_floor else "clock" if gamma <= -margin_floor else None
        if observed_owner is None:
            phase_labels[phase] = "weak_or_ambiguous"
            continue
        owner_delta = delta_vase if observed_owner == "vase" else delta_clock
        alternative = "clock" if observed_owner == "vase" else "vase"
        alternative_delta = delta_clock if observed_owner == "vase" else delta_vase
        if owner_delta < -destruction_tolerance:
            label = "destructive_preferred_owner_lowering"
        elif owner_delta >= release_floor and alternative_delta <= -release_floor:
            label = "mixed_activation_and_suppression"
        elif owner_delta >= release_floor:
            label = f"constructive_{observed_owner}_activation"
        elif alternative_delta <= -release_floor:
            label = f"{alternative}_suppression"
        else:
            label = "weak_or_ambiguous"
        phase_labels[phase] = label
    description = phase_signatures.get("description", {})
    geometry = phase_signatures.get("geometry", {})
    def owner_of(values: Mapping[str, Any]) -> str | None:
        gamma = float(values.get("gamma_vase_minus_clock", 0.0))
        if gamma >= margin_floor:
            return "vase"
        if gamma <= -margin_floor:
            return "clock"
        return None
    description_owner = owner_of(description)
    geometry_owner = owner_of(geometry)
    chimera = description_owner is not None and geometry_owner is not None and description_owner != geometry_owner
    return {
        "phase_labels": phase_labels,
        "description_owner": description_owner,
        "geometry_owner": geometry_owner,
        "phrase_geometry_chimera": chimera,
        "full_row_is_not_sufficient_for_chimera": True,
        "classification": phase_labels.get("full_row", "weak_or_ambiguous"),
    }


def compare_input_position_identities(
    *, observed: Mapping[str, Mapping[str, Any]], expected: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Compare both canonical rows' input and explicit-position identities."""

    owners: dict[str, Any] = {}
    for owner in ("target", "competitor"):
        if owner not in observed or owner not in expected:
            owners[owner] = {"passed": False, "missing": True}
            continue
        owners[owner] = {
            "input_ids_sha256_equal": observed[owner].get("input_ids_sha256") == expected[owner].get("input_ids_sha256"),
            "position_ids_sha256_equal": observed[owner].get("position_ids_sha256") == expected[owner].get("position_ids_sha256"),
        }
        owners[owner]["passed"] = bool(
            owners[owner]["input_ids_sha256_equal"]
            and owners[owner]["position_ids_sha256_equal"]
        )
    return {"owners": owners, "passed": all(bool(value.get("passed")) for value in owners.values())}


def _score_hybrid_arm(*, model: Any, model_inputs: Mapping[str, Any], features: Any, grid_thw: Sequence[int], merge_size: int, prompt_ids: Sequence[int], target_row: Sequence[int], competitor_row: Sequence[int], image_token_id: int, earlier_indices: Sequence[int], row_indices: Sequence[int], image_grid_thw: torch.Tensor, terminal_token_id: int | None, earlier_reference: Mapping[str, Any], row_reference: Mapping[str, Any]) -> dict[str, Any]:
    device = next(model.parameters()).device
    scores: dict[str, Any] = {}
    structures: dict[str, Any] = {}
    identities: dict[str, Any] = {}
    for owner, row in (("target", target_row), ("competitor", competitor_row)):
        ids = torch.tensor([list(prompt_ids) + list(row)], dtype=torch.long, device=device)
        image_positions = [int(v) for v in torch.where(ids[0] == int(image_token_id))[0].tolist()]
        earlier_positions = [image_positions[int(v)] for v in earlier_indices]
        row_positions = [image_positions[int(v)] for v in row_indices]
        positions = factorial.derive_explicit_position_ids(model, input_ids=ids, attention_mask=torch.ones_like(ids), image_grid_thw=image_grid_thw.to(device=device))
        mask = build_hybrid_key_eligibility_mask(sequence_length=ids.shape[1], image_key_positions=image_positions, earlier_eligible_image_positions=earlier_positions, row_eligible_image_positions=row_positions, prefix_length=len(prompt_ids), row_length=len(row), device=device)
        earlier_mask = factorial.build_earlier_query_only_key_eligibility_mask(sequence_length=ids.shape[1], image_key_positions=image_positions, eligible_image_positions=earlier_positions, prefix_length=len(prompt_ids), device=device)
        row_mask = query.build_query_scoped_key_eligibility_mask(sequence_length=ids.shape[1], image_key_positions=image_positions, eligible_image_positions=row_positions, prefix_length=len(prompt_ids), row_length=len(row), device=device)
        structures[owner] = inspect_hybrid_mask_structure(
            mask,
            sequence_length=ids.shape[1],
            image_key_positions=image_positions,
            earlier_eligible_image_positions=earlier_positions,
            row_eligible_image_positions=row_positions,
            prefix_length=len(prompt_ids),
            row_length=len(row),
            earlier_reference_mask=earlier_mask,
            row_reference_mask=row_mask,
            earlier_reference_receipt=earlier_reference.get("structural_mask_receipt", {}).get(owner, {}),
            row_reference_receipt=row_reference.get("structural_mask_receipt", {}).get(owner, {}),
        )
        scores[owner] = query._score_row(model=model, model_inputs=model_inputs, features=features, grid_thw=grid_thw, merge_size=merge_size, prompt_ids=prompt_ids, row=row, image_token_id=image_token_id, image_grid_thw=image_grid_thw, position_ids=positions, custom_mask=mask, terminal_token_id=terminal_token_id)
        identities[owner] = {"input_ids_sha256": factorial.tensor_sha256(ids), "position_ids_sha256": factorial.tensor_sha256(positions)}
    first_diff = factorial.first_differing_description_index(target_row, competitor_row, description_length=len(target_row) - 8)
    result: dict[str, Any] = {"target": scores["target"], "competitor": scores["competitor"], "structural_mask_receipt": structures, "input_and_position_identities": identities}
    if first_diff is not None:
        result["first_differing_description"] = {owner: {"sum": scores[owner]["token_log_probabilities"][first_diff], "mean": scores[owner]["token_log_probabilities"][first_diff], "count": 1} for owner in ("target", "competitor")}
    return result


def _run_case_hybrid(**kwargs: Any) -> dict[str, Any]:
    if _FROZEN_FACTORIAL_CASE is None:
        raise RuntimeError("frozen factorial case was not initialized")
    base = _ORIGINAL_QUERY_RUN_CASE(**kwargs)
    frozen = _FROZEN_FACTORIAL_CASE
    arm_plan = build_hybrid_arm_plan(
        frozen_case=frozen,
        target_indices=kwargs["target_indices"],
        competitor_indices=kwargs["competitor_indices"],
    )
    common = dict(model=kwargs["qwen"].model, model_inputs=kwargs["model_inputs"], features=kwargs["features"], grid_thw=kwargs["grid_thw"], merge_size=kwargs["merge_size"], prompt_ids=kwargs["prompt_ids"], target_row=kwargs["target_row"], competitor_row=kwargs["competitor_row"], image_token_id=kwargs["image_token_id"], image_grid_thw=kwargs["image_grid_thw"], terminal_token_id=kwargs["qwen"].tokenizer.eos_token_id)
    hybrids = {
        arm_name: _score_hybrid_arm(
            **common,
            earlier_indices=plan["earlier_indices"],
            row_indices=plan["row_indices"],
            earlier_reference=plan["earlier_reference"],
            row_reference=plan["row_only_reference"],
        )
        for arm_name, plan in arm_plan.items()
    }
    position_identity_checks: dict[str, Any] = {}
    for hybrid_name, hybrid_arm in hybrids.items():
        expected = arm_plan[hybrid_name]["earlier_reference"]["input_and_position_identities"]
        position_identity_checks[hybrid_name] = compare_input_position_identities(
            observed=hybrid_arm["input_and_position_identities"], expected=expected
        )
    factorial_baseline_parity = factorial.assess_cross_receipt_baseline(
        live=base["arms"]["all_allowed_4d"], frozen=frozen["arms"]["all_allowed_4d"]
    )
    unrestricted = frozen["arms"]["all_allowed_4d"]
    base["hybrid_arms"] = hybrids
    base["frozen_comparators"] = {
        "unrestricted": unrestricted,
        "earlier_query_only": frozen["earlier_query_only_arms"],
        "row_query_only": {"target": frozen["arms"]["target_row_query_only_hard"], "competitor": frozen["arms"]["competitor_row_query_only_hard"]},
        "all_query": {"target": frozen["parent_hard_endpoint"]["target_eligibility"], "competitor": frozen["parent_hard_endpoint"]["competitor_eligibility"]},
    }
    signatures: dict[str, Any] = {}
    for name, arm in hybrids.items():
        row_only = arm_plan[name]["row_only_reference"]
        matched = arm_plan[name]["matched_all_query_reference"]
        signatures[name] = build_phase_signatures(unrestricted=unrestricted, hybrid=arm, row_only=row_only, matched=matched)
        signatures[name]["classification"] = classify_hybrid_effect(signatures[name])
    base["hybrid_phase_signatures"] = signatures
    structural_passed = all(
        bool(v.get("structural_mask_receipt", {}).get(owner, {}).get("passed"))
        for v in hybrids.values()
        for owner in ("target", "competitor")
    )
    base["hybrid_execution_gate"] = {
        "passed": bool(
            structural_passed
            and all(bool(item["passed"]) for item in position_identity_checks.values())
            and factorial_baseline_parity["passed"]
            and base.get("no_op_trust_gate", {}).get("passed")
            and base.get("parent_continuity", {}).get("passed")
            and base.get("structural_mask_gate_passed")
        ),
        "all_structural_receipts_passed": structural_passed,
        "frozen_factorial_baseline_parity": factorial_baseline_parity,
        "position_identity_checks": position_identity_checks,
    }
    base["classification"] = "valid_hybrid_execution" if base["hybrid_execution_gate"]["passed"] else "invalid_hybrid_execution_gate"
    return base


def build_parser() -> argparse.ArgumentParser:
    parser = query.build_parser()
    parser.description = __doc__
    parser.add_argument("--factorial-receipt", type=Path, default=DEFAULT_FACTORIAL_RECEIPT)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    global _FROZEN_FACTORIAL_CASE
    try:
        validate_exact_image_ids(args.image_ids)
        frozen, factorial_sha = validate_factorial_receipt(args.factorial_receipt)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    _FROZEN_FACTORIAL_CASE = frozen["results"][0]
    original = query._run_case
    query._run_case = _run_case_hybrid
    try:
        payload = query.run(args)
    finally:
        query._run_case = original
    payload["schema_version"] = "fixed_encoding_cross_region_earlier_and_row_query_spatial_key_eligibility_hybrid.v1"
    payload["unit_id"] = UNIT_ID
    payload["factorial_receipt"] = str(Path(args.factorial_receipt).expanduser().resolve())
    payload["factorial_receipt_sha256"] = factorial_sha
    payload["imported_runner_sha256"] = factorial._sha256_file(Path(factorial.__file__))
    payload["imported_query_scorer_sha256"] = query._sha256_file(Path(query.__file__))
    payload["configuration_provenance"] = _config_provenance(args.infer_config)
    result = payload["results"][0]
    result["hybrid_execution_gate"]["feature_continuity_passed"] = bool(result.get("feature_continuity", {}).get("passed"))
    result["hybrid_execution_gate"]["row_contract_passed"] = bool(result.get("row_contract", {}).get("passed"))
    result["hybrid_execution_gate"]["passed"] = bool(result["hybrid_execution_gate"]["passed"] and result["hybrid_execution_gate"]["feature_continuity_passed"] and result["hybrid_execution_gate"]["row_contract_passed"])
    result["classification"] = "valid_hybrid_execution" if result["hybrid_execution_gate"]["passed"] else "invalid_hybrid_execution_gate"
    payload["panel_decision"] = {"classification": result["classification"], "image_id": IMAGE_ID, "hybrid_execution_gate_passed": result["hybrid_execution_gate"]["passed"]}
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    normalized_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    validate_exact_image_ids(args.image_ids)
    payload = run(args)
    payload["runner_sha256"] = _sha256_file(Path(__file__))
    payload["normalized_argv"] = normalized_argv
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "receipt.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
