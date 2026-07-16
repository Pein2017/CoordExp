#!/usr/bin/env python3
"""Run the fixed-encoding soft spatial-key bias dose-response panel.

This is an experiment-local teacher-forced scorer.  The visual tower is run
once per image and the captured features are replayed for every arm.  The
only intervention is a finite additive bias on selected image-token *keys* in
an explicit float32 causal attention mask.  It deliberately does not train,
generate, crop, resize, or modify shared inference code.
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

from scripts.research.run_fixed_encoding_object_centered_spatial_eligibility_crossover import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_LEDGER,
    DEFAULT_SOURCE_JSONL,
    _load_ledger,
    _object_pixel_box,
    _phase_score_map,
    _row_token_ids,
    _score_model_sequence,
    assess_noop_trust_gate,
    build_translated_competitor_mask,
    derive_explicit_position_ids,
    feature_bundle_fingerprint,
    first_differing_description_index,
    score_row_log_likelihoods,
)
from src.analysis.visual_support_counterfactual import (  # noqa: E402
    build_merged_support_mask,
    capture_feature_bundle,
    validate_feature_layout,
)


DEFAULT_PARENT_RECEIPT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-15-fixed-encoding-object-centered-spatial-eligibility-crossover/"
    "cohort-six-float32-20260715b/receipt.json"
)
DEFAULT_IMAGE_IDS = ("139", "632", "12120", "12639")
DOSES = (0.5, 1.0, 2.0)
TOLERANCE = 1e-4


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


def _canonical_fingerprint(value: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Drop device labels while retaining feature content identity."""

    output: dict[str, list[dict[str, Any]]] = {}
    for stream in ("primary", "deepstack"):
        entries = value.get(stream, [])
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
            raise ValueError(f"feature fingerprint stream {stream!r} is invalid")
        output[stream] = []
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise ValueError("feature fingerprint entry is invalid")
            output[stream].append(
                {
                    "sha256": str(entry["sha256"]),
                    "shape": [int(v) for v in entry["shape"]],
                    "dtype": str(entry["dtype"]),
                }
            )
    return output


def compare_feature_fingerprints(
    expected: Mapping[str, Any], observed: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare content hash, shape, and dtype; intentionally ignore device."""

    expected_canonical = _canonical_fingerprint(expected)
    observed_canonical = _canonical_fingerprint(observed)
    return {
        "passed": expected_canonical == observed_canonical,
        "expected": expected_canonical,
        "observed": observed_canonical,
        "device_ignored": True,
    }


def build_soft_spatial_key_bias_mask(
    *,
    sequence_length: int,
    image_key_positions: Sequence[int],
    selected_image_positions: Sequence[int],
    bias: float,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Build a float32 ``[1, 1, S, S]`` additive causal key-bias mask.

    Every causally visible key starts at zero.  Future entries are negative
    infinity.  Selected image-token key columns receive ``+bias`` only where
    they are already causally visible; no query row is otherwise restricted.
    """

    length = int(sequence_length)
    if length <= 0:
        raise ValueError("sequence_length must be positive")
    value = float(bias)
    if not torch.isfinite(torch.tensor(value)) or value < 0.0:
        raise ValueError("bias must be a finite non-negative number")
    image = {int(v) for v in image_key_positions}
    selected = {int(v) for v in selected_image_positions}
    if any(v < 0 or v >= length for v in image | selected):
        raise ValueError("image key position is outside the sequence")
    if not selected.issubset(image):
        raise ValueError("selected image keys must be a subset of image keys")
    mask = torch.full((length, length), float("-inf"), dtype=torch.float32, device=device)
    visible = torch.tril(torch.ones((length, length), dtype=torch.bool, device=device))
    mask[visible] = 0.0
    if selected:
        selected_tensor = torch.tensor(sorted(selected), dtype=torch.long, device=device)
        visible_selected = visible[:, selected_tensor]
        mask[:, selected_tensor] = torch.where(
            visible_selected,
            torch.full_like(mask[:, selected_tensor], value),
            torch.full_like(mask[:, selected_tensor], float("-inf")),
        )
    return mask.unsqueeze(0).unsqueeze(0).contiguous()


# A descriptive alias keeps downstream tests and notes readable.
build_additive_causal_key_bias_mask = build_soft_spatial_key_bias_mask


def _max_vector_drift(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("token log-probability vectors have different lengths")
    return max((abs(float(a) - float(b)) for a, b in zip(left, right, strict=True)), default=0.0)


def assess_lambda05_actuation_gate(
    *,
    target_bias_target: Sequence[float],
    target_bias_competitor: Sequence[float],
    competitor_bias_target: Sequence[float],
    competitor_bias_competitor: Sequence[float],
    zero_target: Sequence[float],
    zero_competitor: Sequence[float],
    no_op_drift: float,
) -> dict[str, Any]:
    """Require a strict finite-dose actuation signal above the no-op drift."""

    drifts = {
        "target_bias_0.5_target": _max_vector_drift(target_bias_target, zero_target),
        "target_bias_0.5_competitor": _max_vector_drift(target_bias_competitor, zero_competitor),
        "competitor_bias_0.5_target": _max_vector_drift(competitor_bias_target, zero_target),
        "competitor_bias_0.5_competitor": _max_vector_drift(competitor_bias_competitor, zero_competitor),
    }
    maximum = max(drifts.values(), default=0.0)
    required = 10.0 * float(no_op_drift)
    return {
        "passed": bool(maximum > required),
        "max_token_logprob_drift_from_zero": maximum,
        "drifts": drifts,
        "required_strictly_greater_than": required,
    }


def assess_parent_all_allowed_continuity(
    *,
    parent_all_allowed: Mapping[str, Any],
    observed_all_allowed: Mapping[str, Any],
    tolerance: float = TOLERANCE,
) -> dict[str, Any]:
    """Compare complete selected-token vectors and ranks to parent arm."""

    owners: dict[str, Any] = {}
    passed = True
    for owner in ("target", "competitor"):
        expected = parent_all_allowed[owner]
        observed = observed_all_allowed[owner]
        drift = _max_vector_drift(
            expected["token_log_probabilities"], observed["token_log_probabilities"]
        )
        ranks_equal = list(map(int, expected["selected_token_ranks"])) == list(
            map(int, observed["selected_token_ranks"])
        )
        owners[owner] = {
            "max_abs_logprob_drift": drift,
            "selected_token_ranks_equal": ranks_equal,
            "passed": bool(drift <= tolerance and ranks_equal),
        }
        passed = passed and bool(owners[owner]["passed"])
    return {"passed": passed, "tolerance": float(tolerance), "owners": owners}


def signed_distance_toward_hard_endpoint(
    *,
    finite: Mapping[str, Mapping[str, float]],
    full: Mapping[str, Mapping[str, float]],
    hard: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """Report signed movement toward the hard endpoint for each phase.

    For each owner-gamma field, ``full.gamma_full`` is the common zero-bias
    owner gamma and ``hard.<owner-gamma>`` is that owner's hard endpoint.
    ``movement_from_zero`` is the exact finite-minus-zero value and
    ``signed_distance_to_hard`` is finite-minus-hard.  Sign agreement is the
    strict sign of the finite owner gamma against the corresponding hard owner
    gamma; zero agrees with neither.
    """

    result: dict[str, Any] = {}
    for phase in sorted(set(finite) & set(full) & set(hard)):
        result[phase] = {}
        zero_gamma = float(full[phase].get("gamma_full", 0.0))
        for key in ("gamma_target", "gamma_competitor"):
            hard_gamma = float(hard[phase].get(key, hard[phase].get("gamma_full", 0.0)))
            finite_value = float(finite[phase].get(key, 0.0))
            delta = finite_value - zero_gamma
            signed_distance = finite_value - hard_gamma
            result[phase][f"{key}_finite_delta_from_full"] = delta
            result[phase][f"{key}_movement_from_zero"] = delta
            result[phase][f"{key}_signed_distance_to_hard"] = signed_distance
            result[phase][f"{key}_sign_agreement"] = bool(
                finite_value != 0.0 and hard_gamma != 0.0 and finite_value * hard_gamma > 0.0
            )
        result[phase]["owner_gamma_sign_agreement"] = bool(result[phase]["gamma_target_sign_agreement"] and result[phase]["gamma_competitor_sign_agreement"])
    return result


def _sign_reversal(values: Mapping[str, float], *, floor: float) -> bool:
    return float(values.get("gamma_target", 0.0)) > floor and float(
        values.get("gamma_competitor", 0.0)
    ) < -floor


def _compatible_owner_signs(values: Mapping[str, float]) -> bool:
    """Dose-two continuity rule: owner signs may be small but must be correct."""

    return float(values.get("gamma_target", 0.0)) > 0.0 and float(
        values.get("gamma_competitor", 0.0)
    ) < 0.0


def _phase_reversal(values: Mapping[str, float], *, effect_floor: float, crossover_floor: float) -> bool:
    return float(values.get("crossover", 0.0)) >= float(crossover_floor) and _sign_reversal(values, floor=effect_floor)


def _zero_interaction_map(arm: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    """Represent an arm's unrestricted owner gamma as ``gamma_full``."""

    phase_map = _phase_score_map(arm)
    result: dict[str, dict[str, float]] = {}
    for phase, scores in phase_map.items():
        gamma = float(scores["target"]["mean"]) - float(scores["competitor"]["mean"])
        result[phase] = {
            "gamma_full": gamma,
            "gamma_target": gamma,
            "gamma_competitor": gamma,
            "crossover": 0.0,
        }
    return result


def classify_case_at_dose(
    crossover: Mapping[str, Mapping[str, float]],
    *,
    no_op_drift: float,
    different_category: bool,
    dose: float,
    dose_one_crossover: Mapping[str, Mapping[str, float]] | None = None,
) -> str:
    """Classify one case at one shared dose under the frozen unit rules."""

    effect_floor = 10.0 * float(no_op_drift)
    crossover_floor = max(0.10, effect_floor)
    release_floor = max(0.05, effect_floor)
    full = crossover.get("full_row", {})
    geometry = crossover.get("geometry", {})
    full_reversal = _phase_reversal(
        full,
        effect_floor=effect_floor,
        crossover_floor=crossover_floor,
    )
    geometry_reversal = _phase_reversal(
        geometry,
        effect_floor=effect_floor,
        crossover_floor=crossover_floor,
    )
    phase_reversal = any(
        _phase_reversal(crossover[name], effect_floor=effect_floor, crossover_floor=crossover_floor)
        for name in ("description", "first_differing_description")
        if name in crossover
    )
    if not full_reversal or not geometry_reversal:
        if phase_reversal or geometry_reversal:
            return "route_to_phase_specific_discriminator"
        return "inconclusive"
    if float(full.get("target_release", 0.0)) < release_floor or float(
        full.get("competitor_release", 0.0)
    ) < -0.05:
        return "inconclusive"
    if different_category:
        for phase_name in ("description", "first_differing_description"):
            if phase_name not in crossover:
                if phase_name == "first_differing_description":
                    continue
                return "inconclusive"
            if not _phase_reversal(crossover[phase_name], effect_floor=effect_floor, crossover_floor=crossover_floor):
                return "route_to_phase_specific_discriminator"
    if float(dose) == 2.0:
        if dose_one_crossover is None:
            return "inconclusive"
        required_phases = ["full_row", "geometry"]
        if different_category:
            required_phases.append("description")
            if "first_differing_description" in crossover:
                required_phases.append("first_differing_description")
        for phase_name in required_phases:
            if not _compatible_owner_signs(dose_one_crossover.get(phase_name, {})):
                return "inconclusive"
    return "promote_bounded_free_row_switch_replay"


def classify_shared_dose_panel(
    results_by_dose: Mapping[float | str, Sequence[Mapping[str, Any]]],
    *,
    doses: Sequence[float] = DOSES,
) -> dict[str, Any]:
    """Select the lowest shared dose with two valid cases and one semantic switch."""

    by_dose: dict[str, Any] = {}
    invalid_case_ids = sorted({
        str(r.get("image_id"))
        for rows in results_by_dose.values()
        for r in rows
        if str(r.get("classification", "")).startswith("invalid_")
    })
    for raw_dose in doses:
        dose = float(raw_dose)
        raw_results = results_by_dose.get(dose, results_by_dose.get(str(dose), []))
        eligible = [
            r for r in raw_results
            if not str(r.get("classification", "")).startswith("invalid_")
        ]
        promoting = [r for r in eligible if r.get("classification") == "promote_bounded_free_row_switch_replay"]
        semantic = [r for r in promoting if bool(r.get("different_category"))]
        by_dose[str(dose)] = {
            "eligible_case_ids": [str(r.get("image_id")) for r in eligible],
            "promoting_case_ids": [str(r.get("image_id")) for r in promoting],
            "different_category_promoting_case_ids": [str(r.get("image_id")) for r in semantic],
            "passes_shared_rule": bool(len(promoting) >= 2 and semantic),
        }
    passing = [dose for dose in doses if by_dose[str(float(dose))]["passes_shared_rule"]]
    if passing:
        chosen = float(min(passing))
        return {
            "classification": "promote_one_bounded_free_row_switch_replay",
            "selected_shared_bias": chosen,
            "invalid_case_ids": invalid_case_ids,
            "by_dose": by_dose,
        }
    phase_passes: list[tuple[float, str]] = []
    for raw_dose in doses:
        dose = float(raw_dose)
        rows = list(results_by_dose.get(dose, results_by_dose.get(str(dose), [])))
        for phase_name in ("description", "geometry"):
            count = sum(
                1
                for row in rows
                if row.get("classification") == "route_to_phase_specific_discriminator"
                and phase_name in set(row.get("phase_specific_reversals", []))
            )
            if count >= 2:
                phase_passes.append((dose, phase_name))
    if phase_passes:
        chosen, phase_name = min(phase_passes)
        return {
            "classification": "route_to_one_phase_specific_discriminator",
            "selected_shared_bias": float(chosen),
            "phase": phase_name,
            "invalid_case_ids": invalid_case_ids,
            "by_dose": by_dose,
        }
    complete_or_phase_case_ids = {
        str(r.get("image_id"))
        for rows in results_by_dose.values()
        for r in rows
        if r.get("classification") in {"promote_bounded_free_row_switch_replay", "route_to_phase_specific_discriminator"}
    }
    return {
        "classification": "inconclusive_isolated_owner_specific_case"
        if len(complete_or_phase_case_ids) == 1
        else "close_uniform_soft_spatial_key_bias",
        "invalid_case_ids": invalid_case_ids,
        "by_dose": by_dose,
    }


def apply_image139_actuation_gate(case: Mapping[str, Any]) -> dict[str, Any]:
    """Make the lambda-0.5 actuation gate invalidating only for image 139."""

    result = dict(case)
    if str(result.get("image_id")) != "139":
        return result
    if str(result.get("classification", "")).startswith("invalid_"):
        return result
    gate = result.get("lambda_0.5_actuation_gate")
    if not isinstance(gate, Mapping) or not bool(gate.get("passed")):
        result["classification"] = "invalid_lambda_0.5_actuation_gate"
    return result


def collect_panel_rows_by_dose(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Exclude invalid cases before recomputing any per-dose classification."""

    by_dose: dict[str, list[dict[str, Any]]] = {str(dose): [] for dose in DOSES}
    for result in results:
        if str(result.get("classification", "")).startswith("invalid_"):
            continue
        for dose, raw_detail in result.get("doses", {}).items():
            detail = dict(raw_detail)
            detail.update(
                {
                    "image_id": result.get("image_id"),
                    "different_category": result.get("different_category"),
                    "classification": classify_case_at_dose(
                        detail["crossover"],
                        no_op_drift=float(
                            result.get("no_op_max_abs_logprob_drift", 0.0)
                        ),
                        different_category=bool(result.get("different_category")),
                        dose=float(dose),
                        dose_one_crossover=(
                            result.get("doses", {}).get("1.0", {}).get("crossover")
                            if float(dose) == 2.0
                            else None
                        ),
                    ),
                }
            )
            by_dose[str(dose)].append(detail)
    return by_dose


def _score_row(
    *,
    model: Any,
    model_inputs: Mapping[str, Any],
    features: Any,
    grid_thw: Sequence[int],
    merge_size: int,
    prompt_ids: Sequence[int],
    row: Sequence[int],
    image_token_id: int,
    image_key_indices: Sequence[int],
    selected_key_indices: Sequence[int] | None,
    position_ids: torch.Tensor | None,
    custom_mask: torch.Tensor | None,
    image_grid_thw: torch.Tensor,
    terminal_token_id: int | None,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    ids = torch.tensor([list(prompt_ids) + [int(v) for v in row]], dtype=torch.long, device=device)
    if custom_mask is None and position_ids is None:
        mode = "implicit"
    else:
        mode = "explicit"
    logits = _score_model_sequence(
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
    del mode, image_token_id, image_key_indices, selected_key_indices, image_grid_thw
    return score_row_log_likelihoods(
        logits,
        prefix_length=len(prompt_ids),
        row_tokens=row,
        description_length=len(row) - 8,
        terminal_token_id=terminal_token_id,
    )


def _score_arm_pair(
    *,
    model: Any,
    model_inputs: Mapping[str, Any],
    features: Any,
    grid_thw: Sequence[int],
    merge_size: int,
    prompt_ids: Sequence[int],
    target_row: Sequence[int],
    competitor_row: Sequence[int],
    image_token_id: int,
    target_indices: Sequence[int],
    competitor_indices: Sequence[int],
    bias: float | None,
    selected_indices: Sequence[int] | None,
    image_grid_thw: torch.Tensor,
    terminal_token_id: int | None,
    explicit_positions: bool = True,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    def score_owner(row: Sequence[int]) -> dict[str, Any]:
        ids = torch.tensor([list(prompt_ids) + list(row)], dtype=torch.long, device=device)
        image_positions = [int(v) for v in torch.where(ids[0] == int(image_token_id))[0].tolist()]
        if not image_positions:
            raise ValueError("prompt contains no Qwen image placeholder keys")
        if len(image_positions) <= max(target_indices + competitor_indices, default=-1):
            raise ValueError("frozen key index exceeds image-token count")
        positions = derive_explicit_position_ids(
            model, input_ids=ids, attention_mask=torch.ones_like(ids), image_grid_thw=image_grid_thw.to(device=device)
        ) if explicit_positions else None
        custom = None
        if bias is not None:
            selected_positions = [image_positions[int(i)] for i in (selected_indices or [])]
            custom = build_soft_spatial_key_bias_mask(
                sequence_length=ids.shape[1], image_key_positions=image_positions,
                selected_image_positions=selected_positions, bias=float(bias), device=device,
            )
        return _score_row(
            model=model, model_inputs=model_inputs, features=features, grid_thw=grid_thw, merge_size=merge_size,
            prompt_ids=prompt_ids, row=row, image_token_id=image_token_id, image_key_indices=image_positions,
            selected_key_indices=selected_indices, position_ids=positions, custom_mask=custom, image_grid_thw=image_grid_thw,
            terminal_token_id=terminal_token_id,
        )

    target = score_owner(target_row)
    competitor = score_owner(competitor_row)
    first_diff = first_differing_description_index(target_row, competitor_row, description_length=len(target_row) - 8)
    arm: dict[str, Any] = {"target": target, "competitor": competitor}
    if first_diff is not None:
        arm["first_differing_description"] = {
            "target": {"sum": target["token_log_probabilities"][first_diff], "mean": target["token_log_probabilities"][first_diff], "count": 1},
            "competitor": {"sum": competitor["token_log_probabilities"][first_diff], "mean": competitor["token_log_probabilities"][first_diff], "count": 1},
        }
    return arm


def _interactions(arms: Mapping[str, Any], target_name: str, competitor_name: str) -> dict[str, Any]:
    return __import__(
        "scripts.research.run_fixed_encoding_object_centered_spatial_eligibility_crossover",
        fromlist=["compute_crossover_and_release"],
    ).compute_crossover_and_release(
        full_scores=_phase_score_map(arms["zero_bias_custom_4d"]),
        target_scores=_phase_score_map(arms[target_name]),
        competitor_scores=_phase_score_map(arms[competitor_name]),
    )


def _canonical_rows_hash(target_row: Sequence[int], competitor_row: Sequence[int]) -> str:
    return _sha256_json({"target": [int(v) for v in target_row], "competitor": [int(v) for v in competitor_row]})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infer-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-jsonl", type=Path, default=DEFAULT_SOURCE_JSONL)
    parser.add_argument("--audit-ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--parent-receipt", type=Path, default=DEFAULT_PARENT_RECEIPT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-ids", nargs="+", default=list(DEFAULT_IMAGE_IDS))
    return parser


def _run_case(
    *, qwen: Any, model_inputs: Mapping[str, Any], features: Any, grid_thw: Sequence[int], merge_size: int,
    prompt_ids: Sequence[int], target_row: Sequence[int], competitor_row: Sequence[int], target_indices: Sequence[int],
    competitor_indices: Sequence[int], image_grid_thw: torch.Tensor, image_token_id: int, parent_result: Mapping[str, Any],
    different_category: bool, target_id: str, competitor_id: str, target_description: str, competitor_description: str,
) -> dict[str, Any]:
    model = qwen.model
    arms: dict[str, Any] = {}
    arms["implicit_standard_2d"] = _score_arm_pair(model=model, model_inputs=model_inputs, features=features, grid_thw=grid_thw, merge_size=merge_size, prompt_ids=prompt_ids, target_row=target_row, competitor_row=competitor_row, image_token_id=image_token_id, target_indices=target_indices, competitor_indices=competitor_indices, bias=None, selected_indices=None, image_grid_thw=image_grid_thw, terminal_token_id=qwen.tokenizer.eos_token_id, explicit_positions=False)
    arms["explicit_standard_2d"] = _score_arm_pair(model=model, model_inputs=model_inputs, features=features, grid_thw=grid_thw, merge_size=merge_size, prompt_ids=prompt_ids, target_row=target_row, competitor_row=competitor_row, image_token_id=image_token_id, target_indices=target_indices, competitor_indices=competitor_indices, bias=None, selected_indices=None, image_grid_thw=image_grid_thw, terminal_token_id=qwen.tokenizer.eos_token_id, explicit_positions=True)
    arms["zero_bias_custom_4d"] = _score_arm_pair(model=model, model_inputs=model_inputs, features=features, grid_thw=grid_thw, merge_size=merge_size, prompt_ids=prompt_ids, target_row=target_row, competitor_row=competitor_row, image_token_id=image_token_id, target_indices=target_indices, competitor_indices=competitor_indices, bias=0.0, selected_indices=[], image_grid_thw=image_grid_thw, terminal_token_id=qwen.tokenizer.eos_token_id, explicit_positions=True)
    parent_arm = parent_result["arms"]["all_allowed_4d"]
    continuity = assess_parent_all_allowed_continuity(parent_all_allowed=parent_arm, observed_all_allowed=arms["zero_bias_custom_4d"])
    no_op = assess_noop_trust_gate(
        implicit_token_log_probs=arms["implicit_standard_2d"]["target"]["token_log_probabilities"],
        explicit_token_log_probs=arms["explicit_standard_2d"]["target"]["token_log_probabilities"],
        all_allowed_token_log_probs=arms["zero_bias_custom_4d"]["target"]["token_log_probabilities"],
        implicit_ranks=arms["implicit_standard_2d"]["target"]["selected_token_ranks"],
        explicit_ranks=arms["explicit_standard_2d"]["target"]["selected_token_ranks"],
        all_allowed_ranks=arms["zero_bias_custom_4d"]["target"]["selected_token_ranks"], tolerance=TOLERANCE,
    )
    competitor_no_op = assess_noop_trust_gate(
        implicit_token_log_probs=arms["implicit_standard_2d"]["competitor"]["token_log_probabilities"],
        explicit_token_log_probs=arms["explicit_standard_2d"]["competitor"]["token_log_probabilities"],
        all_allowed_token_log_probs=arms["zero_bias_custom_4d"]["competitor"]["token_log_probabilities"],
        implicit_ranks=arms["implicit_standard_2d"]["competitor"]["selected_token_ranks"],
        explicit_ranks=arms["explicit_standard_2d"]["competitor"]["selected_token_ranks"],
        all_allowed_ranks=arms["zero_bias_custom_4d"]["competitor"]["selected_token_ranks"], tolerance=TOLERANCE,
    )
    no_op_drift = max(float(no_op["explicit_max_abs_logprob_drift"]), float(no_op["all_allowed_max_abs_logprob_drift"]), float(competitor_no_op["explicit_max_abs_logprob_drift"]), float(competitor_no_op["all_allowed_max_abs_logprob_drift"]))
    result: dict[str, Any] = {
        "image_id": str(parent_result["image_id"]), "target_annotation_id": target_id, "competitor_annotation_id": competitor_id,
        "target_description": target_description, "competitor_description": competitor_description, "different_category": different_category,
        "target_row_token_ids": [int(v) for v in target_row], "competitor_row_token_ids": [int(v) for v in competitor_row],
        "canonical_rows_sha256": _canonical_rows_hash(target_row, competitor_row), "target_mask_indices": [int(v) for v in target_indices], "competitor_mask_indices": [int(v) for v in competitor_indices],
        "arms": arms, "parent_continuity": continuity, "no_op_trust_gate": {"target": no_op, "competitor": competitor_no_op}, "no_op_max_abs_logprob_drift": no_op_drift,
        "parent_hard_endpoint": {"target_eligibility": parent_result["arms"].get("target_eligibility"), "competitor_eligibility": parent_result["arms"].get("competitor_eligibility"), "crossover": parent_result.get("crossover")},
    }
    if not (no_op["passed"] and competitor_no_op["passed"] and continuity["passed"]):
        result["classification"] = "invalid_no_op_trust_gate"
        return result
    for dose in DOSES:
        target_arm_name = f"target_bias_{dose:.1f}"
        competitor_arm_name = f"competitor_bias_{dose:.1f}"
        arms[target_arm_name] = _score_arm_pair(model=model, model_inputs=model_inputs, features=features, grid_thw=grid_thw, merge_size=merge_size, prompt_ids=prompt_ids, target_row=target_row, competitor_row=competitor_row, image_token_id=image_token_id, target_indices=target_indices, competitor_indices=competitor_indices, bias=dose, selected_indices=target_indices, image_grid_thw=image_grid_thw, terminal_token_id=qwen.tokenizer.eos_token_id)
        arms[competitor_arm_name] = _score_arm_pair(model=model, model_inputs=model_inputs, features=features, grid_thw=grid_thw, merge_size=merge_size, prompt_ids=prompt_ids, target_row=target_row, competitor_row=competitor_row, image_token_id=image_token_id, target_indices=target_indices, competitor_indices=competitor_indices, bias=dose, selected_indices=competitor_indices, image_grid_thw=image_grid_thw, terminal_token_id=qwen.tokenizer.eos_token_id)
        interactions = _interactions(arms, target_arm_name, competitor_arm_name)
        hard = parent_result.get("crossover", {})
        finite = {phase: values for phase, values in interactions.items()}
        phase_effect_floor = 10.0 * no_op_drift
        phase_crossover_floor = max(0.10, phase_effect_floor)
        phase_specific_reversals = [
            phase for phase in ("description", "geometry")
            if phase in interactions and _phase_reversal(interactions[phase], effect_floor=phase_effect_floor, crossover_floor=phase_crossover_floor)
        ]
        result.setdefault("doses", {})[str(dose)] = {
            "target_arm": target_arm_name, "competitor_arm": competitor_arm_name, "crossover": interactions,
            "phase_specific_reversals": phase_specific_reversals,
            "signed_distance_toward_hard_endpoint": signed_distance_toward_hard_endpoint(finite=finite, full=_zero_interaction_map(arms["zero_bias_custom_4d"]), hard=hard),
        }
    zero_arm = arms["zero_bias_custom_4d"]
    zero_target = zero_arm["target"]["token_log_probabilities"]
    zero_competitor = zero_arm["competitor"]["token_log_probabilities"]
    result["lambda_0.5_actuation_gate"] = assess_lambda05_actuation_gate(
        target_bias_target=arms["target_bias_0.5"]["target"]["token_log_probabilities"],
        target_bias_competitor=arms["target_bias_0.5"]["competitor"]["token_log_probabilities"],
        competitor_bias_target=arms["competitor_bias_0.5"]["target"]["token_log_probabilities"],
        competitor_bias_competitor=arms["competitor_bias_0.5"]["competitor"]["token_log_probabilities"],
        zero_target=zero_target,
        zero_competitor=zero_competitor,
        no_op_drift=no_op_drift,
    )
    for dose in DOSES:
        dose_key = str(dose)
        previous = result["doses"].get("1.0", {}).get("crossover") if dose == 2.0 else None
        result["doses"][dose_key]["classification"] = classify_case_at_dose(result["doses"][dose_key]["crossover"], no_op_drift=no_op_drift, different_category=different_category, dose=dose, dose_one_crossover=previous)
    result["classification"] = "inconclusive"
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    from src.config.fingerprint import sha256_json
    from src.config.inference import load_infer_config
    from src.data import load_raw_examples
    from src.inference.image_plan import materialize_image_plan_batch, verify_processor_model_vision_parity
    from src.inference.pipeline import _processor_config, _template_config
    from src.inference.prompt import build_prompt_record
    from src.inference.runtime import assemble_runtime
    from scripts.research.run_sampled_rescue_transition import _temporary_cwd

    parent_path = Path(args.parent_receipt).expanduser().resolve(strict=True)
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if parent.get("unit_id") != "2026-07-15-fixed-encoding-object-centered-spatial-eligibility-crossover":
        raise SystemExit("parent receipt has unexpected unit_id")
    source_path = Path(args.source_jsonl).resolve(strict=True)
    config_path = Path(args.infer_config).resolve(strict=True)
    source_sha = _sha256_file(source_path)
    if source_sha != parent.get("source_jsonl_sha256"):
        raise SystemExit("source JSONL hash does not match parent receipt")
    ledger_path = Path(args.audit_ledger).resolve(strict=True)
    ledger_sha = _sha256_file(ledger_path)
    if ledger_sha != parent.get("audit_ledger_sha256"):
        raise SystemExit("audit ledger hash does not match parent receipt")
    with _temporary_cwd(config_path.parents[3]):
        resolved = load_infer_config(config_path)
    config = resolved.config.model_copy(update={"model": resolved.config.model.model_copy(update={"dtype": "fp32"})})
    config_sha = sha256_json(config.model_dump(mode="json"))
    if config_sha != parent.get("config_sha256"):
        raise SystemExit("resolved config hash does not match parent receipt")
    raw_rows = load_raw_examples(source_path)
    raw_by_image = {str(r.metadata["source"]["image_id"]): r for r in raw_rows if isinstance(r.metadata.get("source"), Mapping) and r.metadata["source"].get("image_id") is not None}
    ledger = _load_ledger(ledger_path)
    runtime = assemble_runtime(config, source_gate_root=config_path.parents[3])
    qwen = runtime.qwen
    qwen.model.eval()
    verify_processor_model_vision_parity(processor_identity=qwen.processor_identity, model_config=qwen.model.config)
    image_token_id = getattr(qwen.model.config, "image_token_id", None) or qwen.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    template = _template_config(resolved.config)
    parent_by_image = {str(r["image_id"]): r for r in parent.get("results", [])}
    results: list[dict[str, Any]] = []
    for image_id in [str(v) for v in args.image_ids]:
        if image_id not in parent_by_image:
            raise SystemExit(f"image {image_id} is absent from parent receipt")
        parent_result = parent_by_image[image_id]
        if str(parent_result.get("classification", "")).startswith("invalid_"):
            results.append({"image_id": image_id, "classification": "invalid_parent_case", "reason": "parent case invalid"})
            if image_id == "139":
                break
            continue
        raw = raw_by_image.get(image_id)
        if raw is None:
            results.append({"image_id": image_id, "classification": "inconclusive", "reason": "source image absent"})
            continue
        objects_by_id = {str(obj.object_id): obj for obj in raw.objects}
        target_id, competitor_id = str(parent_result["target_annotation_id"]), str(parent_result["competitor_annotation_id"])
        target, competitor = objects_by_id.get(target_id), objects_by_id.get(competitor_id)
        if target is None or competitor is None:
            raise SystemExit(f"frozen parent object absent for image {image_id}")
        ledger_by_id = {str(item.get("object_identifier", "")): item for item in ledger.get(image_id, [])}
        base_prompt = build_prompt_record(raw, template, processor=qwen.processor, row_index=0)
        prompt_ids = [int(v) for v in base_prompt.prompt_token_ids]
        target_row, competitor_row = _row_token_ids(qwen.tokenizer, target), _row_token_ids(qwen.tokenizer, competitor)
        image_plan = materialize_image_plan_batch([raw], components=qwen, processor_config=_processor_config(resolved.config), materialize=True, row_indices=[0])
        model_inputs = image_plan.model_inputs_by_row_id[raw.example_id]
        grid_thw = [int(v) for v in model_inputs["image_grid_thw"].reshape(-1, 3)[0].tolist()]
        if grid_thw != [int(v) for v in parent_result.get("image_grid_thw", [])]:
            raise SystemExit(f"image_grid_thw differs from parent for image {image_id}")
        if int(qwen.processor_identity.merge_size) != int(parent_result.get("merge_size")):
            raise SystemExit(f"merge_size differs from parent for image {image_id}")
        features = capture_feature_bundle(qwen.model, model_inputs)
        layout = validate_feature_layout(features, features, grid_thw=grid_thw, merge_size=int(qwen.processor_identity.merge_size))
        feature_check = compare_feature_fingerprints(parent_result["feature_fingerprint"], feature_bundle_fingerprint(features))
        target_box = _object_pixel_box(target, ledger_by_id, width=raw.image.width, height=raw.image.height)
        competitor_box = _object_pixel_box(competitor, ledger_by_id, width=raw.image.width, height=raw.image.height)
        target_support = build_merged_support_mask(bbox_xyxy=target_box, image_width=raw.image.width, image_height=raw.image.height, layout=layout, halo=1)
        target_indices = [int(v) for v in parent_result["target_mask_indices"]]
        competitor_indices = [int(v) for v in parent_result["competitor_mask_indices"]]
        if len(target_indices) != int(parent_result.get("target_mask_count", -1)) or len(competitor_indices) != int(parent_result.get("competitor_mask_count", -1)):
            raise SystemExit(f"frozen mask count differs from parent for image {image_id}")
        if len(set(target_indices)) != len(target_indices) or len(set(competitor_indices)) != len(competitor_indices):
            raise SystemExit(f"frozen mask indices are not unique for image {image_id}")
        translated_competitor = build_translated_competitor_mask(
            target_support, target_bbox_xyxy=target_box, competitor_bbox_xyxy=competitor_box,
            image_width=raw.image.width, image_height=raw.image.height, temporal=layout.temporal,
            merged_height=layout.merged_height, merged_width=layout.merged_width,
        )
        expected_target = sorted(int(v) for v in target_support.nonzero().flatten().tolist())
        if expected_target != target_indices:
            raise SystemExit(f"frozen target support differs for image {image_id}")
        expected_competitor = sorted(int(v) for v in translated_competitor.nonzero().flatten().tolist())
        if expected_competitor != competitor_indices:
            raise SystemExit(f"frozen competitor support differs from parent for image {image_id}")
        if max(target_indices + competitor_indices, default=-1) >= len(features.primary[0]):
            raise SystemExit(f"frozen mask index exceeds feature count for image {image_id}")
        if not feature_check["passed"]:
            results.append({"image_id": image_id, "classification": "invalid_feature_continuity", "feature_continuity": feature_check})
            if image_id == "139":
                break
            continue
        case = _run_case(qwen=qwen, model_inputs=model_inputs, features=features, grid_thw=grid_thw, merge_size=int(qwen.processor_identity.merge_size), prompt_ids=prompt_ids, target_row=target_row, competitor_row=competitor_row, target_indices=target_indices, competitor_indices=competitor_indices, image_grid_thw=model_inputs["image_grid_thw"], image_token_id=int(image_token_id), parent_result=parent_result, different_category=str(target.description) != str(competitor.description), target_id=target_id, competitor_id=competitor_id, target_description=str(target.description), competitor_description=str(competitor.description))
        case["feature_continuity"] = feature_check
        case["feature_fingerprint"] = feature_bundle_fingerprint(features)
        case["image_grid_thw"] = grid_thw
        case["merge_size"] = int(qwen.processor_identity.merge_size)
        case["target_mask_count"] = len(target_indices)
        case["competitor_mask_count"] = len(competitor_indices)
        case = apply_image139_actuation_gate(case)
        if image_id == "139" and str(case.get("classification", "")).startswith("invalid_"):
            results.append(case)
            break
        results.append(case)
    by_dose = collect_panel_rows_by_dose(results)
    panel = classify_shared_dose_panel(by_dose)
    panel["invalid_case_ids"] = sorted({
        str(result.get("image_id"))
        for result in results
        if str(result.get("classification", "")).startswith("invalid_")
    })
    return {"schema_version": "fixed_encoding_soft_spatial_key_bias_dose_response.v1", "unit_id": "2026-07-15-fixed-encoding-soft-spatial-key-bias-dose-response", "model_dtype": "torch.float32", "parent_receipt": str(parent_path), "parent_receipt_sha256": _sha256_file(parent_path), "source_jsonl": str(source_path), "source_jsonl_sha256": source_sha, "audit_ledger": str(Path(args.audit_ledger).resolve()), "audit_ledger_sha256": _sha256_file(Path(args.audit_ledger)), "config_sha256": config_sha, "doses": list(DOSES), "results": results, "panel_by_dose": by_dose, "panel_decision": panel}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "receipt.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
