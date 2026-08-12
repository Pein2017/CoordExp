#!/usr/bin/env python3
"""Build the CPU-bound Human-13 payload consumed by the existing runner.

This adapter owns no model, optimizer, or checkpoint lifecycle.  It selects
manifest-derived materialized segments, maps their exact target tokens to
packed causal-logit positions, and delegates packing, micro-step, and
execution-plan construction to the existing experiment-local runner.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from scripts.research import run_human13_k_union_overfit as runner
from src.supervision import TokenSequence, build_token_sequence_from_packed_supervision


class PayloadContractError(ValueError):
    """Raised when materialized Human-13 segments cannot form a bound payload."""


@dataclass(frozen=True)
class Human13LivePayload:
    arm_id: str
    selected_segments: tuple[runner.LogicalPanelSegment, ...]
    packed_plan: runner.PackedPanelPlan
    sites_by_pack: dict[int, tuple[runner.Human13LossSite, ...]]
    token_sequences: dict[int, TokenSequence]
    micro_steps: tuple[runner.SupervisedMicroStep, ...]
    execution_plan: Any
    a6_donor_binding: Any | None = None
    a8_census_binding: Any | None = None


def _select_segments(
    materialized_segments: Any,
    *,
    arm_id: str,
    global_max_length: int,
) -> tuple[runner.LogicalPanelSegment, ...]:
    try:
        contract = runner._arm_contract(arm_id)
    except ValueError as exc:
        raise PayloadContractError(str(exc)) from exc
    if not hasattr(materialized_segments, "segments"):
        raise PayloadContractError("materialized segments must expose segments")
    if arm_id == "A4":
        preflight = getattr(materialized_segments, "preflight", None)
        if not callable(preflight):
            raise PayloadContractError("A4 requires materialized aggregate preflight")
        try:
            preflight(global_max_length, enforce_a4_aggregate=True)
        except Exception as exc:
            raise PayloadContractError(f"A4 aggregate preflight failed: {exc}") from exc
    else:
        preflight = getattr(materialized_segments, "preflight", None)
        if callable(preflight):
            preflight(global_max_length)
    source_segments = tuple(materialized_segments.segments)
    if arm_id == "A4":
        known_ids = {segment.segment_id for segment in source_segments}
        source_segments += tuple(
            segment
            for segment in getattr(materialized_segments, "a4_segments", ())
            if segment.segment_id not in known_ids
        )
    selected = tuple(
        segment for segment in source_segments if segment.role in contract.allowed_roles
    )
    if not selected:
        raise PayloadContractError(
            f"arm {arm_id} has no materialized segments for roles "
            f"{sorted(contract.allowed_roles)}"
        )
    if arm_id == "A4" and not any(segment.role == "a4_union" for segment in selected):
        raise PayloadContractError("A4 requires materialized a4_union candidate segments")
    return selected


def _required_margin(arm_id: str, binding: Any, *, objective: str) -> float | None:
    if objective != "bottleneck":
        return None
    if arm_id != "A8-prime" or binding is None:
        raise PayloadContractError("A8-prime requires a typed census binding")
    value = getattr(binding, "required_margin", None)
    if value is None and isinstance(binding, Mapping):
        value = binding.get("required_margin")
    try:
        margin = float(value)
    except (TypeError, ValueError) as exc:
        raise PayloadContractError("A8 census binding has no required margin") from exc
    if not math.isfinite(margin) or margin <= 0:
        raise PayloadContractError("A8 required margin must be finite and positive")
    return margin


def _sites_for_pack(
    packed: runner.PackedPanelMicroStep,
    *,
    arm_id: str,
    a8_census_binding: Any | None,
) -> tuple[runner.Human13LossSite, ...]:
    contract = runner._arm_contract(arm_id)
    packed_by_id = {item.example_id: item for item in packed.pack.segments}
    sites: list[runner.Human13LossSite] = []
    for logical in packed.logical_segments:
        packed_segment = packed_by_id.get(logical.segment_id)
        if packed_segment is None:
            raise PayloadContractError(
                f"packed plan lost segment {logical.segment_id}"
            )
        bindings = getattr(logical.encoded_example, "human13_row_bindings", None)
        if not isinstance(bindings, tuple) or not bindings:
            raise PayloadContractError(
                f"segment {logical.segment_id} has no typed row bindings"
            )
        for binding in bindings:
            active_families = {
                family for family, weight in contract.coefficients if weight > 0
            }
            if binding.family not in active_families:
                raise PayloadContractError(
                    f"inactive family {binding.family} appears in {logical.segment_id}"
                )
            token_indices = tuple(
                binding.token_start + offset
                for offset, included in enumerate(binding.target_token_mask)
                if included
            )
            if not token_indices or any(index <= 0 for index in token_indices):
                raise PayloadContractError(
                    f"{logical.segment_id}/{binding.manifest_row_id} has no causal target"
                )
            input_ids = tuple(logical.encoded_example.input_ids)
            if any(index >= len(input_ids) for index in token_indices):
                raise PayloadContractError(
                    f"{logical.segment_id}/{binding.manifest_row_id} target escapes segment"
                )
            objective = runner._objective_for_family(binding.family, contract)
            sites.append(
                runner.Human13LossSite(
                    family=binding.family,
                    objective=objective,
                    unit_id=binding.unit_id,
                    image_id=logical.image_id,
                    logits_positions=tuple(
                        packed_segment.start + index - 1 for index in token_indices
                    ),
                    target_token_ids=tuple(input_ids[index] for index in token_indices),
                    required_margin=_required_margin(
                        arm_id, a8_census_binding, objective=objective
                    ),
                    segment_id=logical.segment_id,
                    manifest_row_ids=(binding.manifest_row_id,),
                )
            )
    return tuple(sites)


def _enforce_a4_atomicity(
    packed_plan: runner.PackedPanelPlan,
    *,
    global_max_length: int,
) -> None:
    del global_max_length
    by_id = {
        segment.example_id: (pack.pack.pack_index, segment.image_id)
        for pack in packed_plan.packs
        for segment in pack.logical_segments
        if segment.role == "a4_union"
    }
    by_image: dict[int, set[int]] = {}
    for segment_id, (pack_index, image_id) in by_id.items():
        del segment_id
        by_image.setdefault(image_id, set()).add(pack_index)
    if any(len(pack_indices) != 1 for pack_indices in by_image.values()):
        raise PayloadContractError(
            "A4 candidate segments must remain one atomic physical pack per image"
        )


def build_live_payload(
    *,
    sealed_manifest: Any,
    materialized_segments: Any,
    arm_id: str,
    expected_vocab_size: int,
    vocab_groups: Any,
    a6_donor_binding: Any | None = None,
    a8_census_binding: Any | None = None,
    global_max_length: int = runner.GLOBAL_MAX_LENGTH,
) -> Human13LivePayload:
    """Materialize one arm's CPU payload over exact native token segments."""

    if isinstance(expected_vocab_size, bool) or expected_vocab_size <= 0:
        raise PayloadContractError("expected_vocab_size must be positive")
    selected = _select_segments(
        materialized_segments, arm_id=arm_id, global_max_length=global_max_length
    )
    packed_plan = runner.plan_panel_packs(
        selected, global_max_length=global_max_length
    )
    if arm_id == "A4":
        _enforce_a4_atomicity(packed_plan, global_max_length=global_max_length)
    sites_by_pack = {
        packed.pack.pack_index: _sites_for_pack(
            packed,
            arm_id=arm_id,
            a8_census_binding=a8_census_binding,
        )
        for packed in packed_plan.packs
    }
    token_sequences = {
        packed.pack.pack_index: build_token_sequence_from_packed_supervision(
            packed.pack, ()
        )
        for packed in packed_plan.packs
    }
    denominators = runner._manifest_denominators(
        runner._coerce_sealed_manifest(sealed_manifest).manifest,
        arm_id,
        runner._arm_contract(arm_id).coefficients,
    )
    micro_steps = runner.build_supervised_micro_steps(
        packed_plan,
        denominators=denominators,
        token_sequences=token_sequences,
        vocab_groups=vocab_groups,
        sites_by_pack=sites_by_pack,
        expected_vocab_size=expected_vocab_size,
    )
    if arm_id == "A8-prime" and a8_census_binding is None:
        raise PayloadContractError("A8-prime requires a typed census binding")
    execution_plan = runner.build_execution_plan(
        sealed_manifest,
        arm_id=arm_id,
        packed_plan=packed_plan,
        sites_by_pack=sites_by_pack,
        a6_donor_binding=a6_donor_binding,
        a8_census_binding=a8_census_binding,
    )
    return Human13LivePayload(
        arm_id=arm_id,
        selected_segments=selected,
        packed_plan=packed_plan,
        sites_by_pack=sites_by_pack,
        token_sequences=token_sequences,
        micro_steps=micro_steps,
        execution_plan=execution_plan,
        a6_donor_binding=a6_donor_binding,
        a8_census_binding=a8_census_binding,
    )


__all__ = ["Human13LivePayload", "PayloadContractError", "build_live_payload"]
