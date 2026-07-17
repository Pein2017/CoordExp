#!/usr/bin/env python3
"""Wave One smoke for donor-state late-coordinate transport on image 7818.

This is a deliberately one-time runner.  It reuses the completed image-7818
portability probe for frozen identities and its residual-state primitives, but
does not alter that completed evidence.  The only scientific readout here is
whether a donor state changes the ``x2,y2`` suffix after an identical,
branch-supported ``x1,y1`` history.  Crossed-box ownership is reported as a
separate gate and never upgrades this smoke to a physical-owner claim.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import scripts.research.run_fixed_encoding_downstream_residual_state_portability as residual  # noqa: E402
import scripts.research.run_fixed_encoding_persistent_hard_geometry_donor_eligibility_screen as screen  # noqa: E402
import scripts.research.run_fixed_encoding_persistent_hard_geometry_state_portability_image7818 as parent  # noqa: E402
import scripts.research.run_fixed_encoding_query_scoped_object_centered_spatial_eligibility as query  # noqa: E402


UNIT_ID = "2026-07-17-object-specific-geometry-transport-and-cross-row-influence-horizon"
SCHEMA_VERSION = "donor-state-late-coordinate-transport-image7818.v1"
IMAGE_ID = "7818"
LAYERS = (23, 13)
POSITIVE_LAYER = 23
NEGATIVE_CONTROL_LAYER = 13
LOG_SEQUENCE_SUPPORT_FLOOR = math.log(10.0)
TOLERANCE = 1e-4
TRANSPORT_EFFECT_FLOOR = 0.05
MAX_LATE_GREEDY_TOKENS = 3  # x2, y2, and BOX_END at most.
EXPECTED_IMAGE_WIDTH = 1248
EXPECTED_IMAGE_HEIGHT = 832
EXPECTED_IMAGE_SHA256 = "d8c9bdea180ba1070daf5792c44187424f66fc319ada9e1ad1bbf86eda348272"
PARENT_PORTABILITY_RECEIPT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-16-fixed-encoding-persistent-hard-routing-geometry-state-portability-image7818/"
    "image7818-layer23-13-float32-20260716a/receipt.json"
)
PARENT_PORTABILITY_RECEIPT_SHA256 = "73805bc237276de1595c6d0ea071db047a6498a97b87d1f0d2cdcee9e56e0ec4"
COORDINATE_PHASES = ("x1", "y1", "x2", "y2")
LATE_PHASES = ("x2", "y2")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve(strict=True).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_parent_portability_state_receipt(
    path: Path = PARENT_PORTABILITY_RECEIPT,
) -> dict[str, Any]:
    """Load the completed float32 portability state identity contract."""

    resolved = Path(path).expanduser().resolve(strict=True)
    observed_sha = sha256_file(resolved)
    if observed_sha != PARENT_PORTABILITY_RECEIPT_SHA256:
        raise ValueError(
            "completed image-7818 portability receipt SHA-256 drifted: "
            f"expected {PARENT_PORTABILITY_RECEIPT_SHA256}, observed {observed_sha}"
        )
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("unit_id") != "2026-07-16-fixed-encoding-persistent-hard-routing-geometry-state-portability-image7818":
        raise ValueError("completed portability receipt unit_id drifted")
    expected: dict[str, Any] = {}
    for result in payload.get("results", []):
        layer = str(result.get("layer_idx"))
        if layer not in {"23", "13"}:
            continue
        position = result.get("position_attestation", {})
        for owner, donor in result.get("donor_states", {}).items():
            expected[f"{layer}:{owner}"] = {
                "state_sha256": str(donor.get("state_sha256")),
                "token_history_sha256": str(donor.get("token_history_sha256")),
                "position_ids_sha256": str(position.get("position_ids_sha256")),
                "boundary_pos": int(position.get("boundary_pos")),
            }
    required = {f"{layer}:{owner}" for layer in ("23", "13") for owner in ("target", "paired")}
    if set(expected) != required:
        raise ValueError(f"completed portability receipt lacks exact state identity set: {sorted(expected)}")
    return {"path": str(resolved), "sha256": observed_sha, "states": expected}


def validate_recaptured_state_identity(
    *,
    layer_idx: int,
    donor_receipts: Mapping[str, Mapping[str, Any]],
    expected_receipt: Mapping[str, Any],
    boundary_pos: int,
    position_ids_sha256: str,
    token_history_sha256: str,
) -> dict[str, Any]:
    """Require both donor states to reproduce the completed parent identity."""

    checks: dict[str, Any] = {}
    for owner in ("target", "paired"):
        key = f"{int(layer_idx)}:{owner}"
        expected = expected_receipt["states"][key]
        observed = donor_receipts[owner]
        checks[owner] = {
            "state_sha256_equal": observed.get("state_sha256") == expected["state_sha256"],
            "token_history_sha256_equal": token_history_sha256 == expected["token_history_sha256"],
            "position_ids_sha256_equal": position_ids_sha256 == expected["position_ids_sha256"],
            "boundary_pos_equal": int(boundary_pos) == int(expected["boundary_pos"]),
        }
        checks[owner]["passed"] = all(bool(value) for value in checks[owner].values())
    return {"passed": all(value["passed"] for value in checks.values()), "by_owner": checks}


def _coordinate_logs(score: Mapping[str, Any], row: Sequence[int]) -> list[float]:
    return parent._coordinate_log_probs(score, row)


def early_coordinate_score(score: Mapping[str, Any], row: Sequence[int]) -> float:
    """Return the selected-token log score for the forced ``x1,y1`` history."""

    values = _coordinate_logs(score, row)
    return float(values[0] + values[1])


def late_coordinate_score(score: Mapping[str, Any], row: Sequence[int]) -> float:
    """Return the selected-token log score for the forced ``x2,y2`` suffix."""

    values = _coordinate_logs(score, row)
    return float(values[2] + values[3])


def make_row(common_xy: Sequence[int], late_xy: Sequence[int]) -> list[int]:
    """Build the frozen five-token wrapper plus four coordinates and BOX_END."""

    if len(common_xy) != 2 or len(late_xy) != 2:
        raise ValueError("common and late coordinate histories must each have two tokens")
    prefix = parent.shared_row_prefix(
        parent.TARGET_REALIZED_DONOR_ROW_TOKEN_IDS,
        parent.PAIRED_REALIZED_DONOR_ROW_TOKEN_IDS,
    )
    return [*prefix, *map(int, common_xy), *map(int, late_xy), parent.BOX_END]


def candidate_common_histories() -> dict[str, list[int]]:
    """Return the predeclared target and paired common ``x1,y1`` histories."""

    return {
        "target_x1_y1": [int(v) for v in parent.TARGET_REALIZED_DONOR_ROW_TOKEN_IDS[5:7]],
        "paired_x1_y1": [int(v) for v in parent.PAIRED_REALIZED_DONOR_ROW_TOKEN_IDS[5:7]],
    }


def native_support_admission(
    *,
    candidate_scores: Mapping[str, Mapping[str, float]],
    native_scores: Mapping[str, float],
    threshold: float = LOG_SEQUENCE_SUPPORT_FLOOR,
) -> dict[str, Any]:
    """Admit only histories within ``log(10)`` of each donor's native path.

    ``candidate_scores[history][donor]`` is the selected ``x1,y1`` score of
    that history under the donor's one-shot residual state. ``native_scores``
    contains the donor's own history score under the same state treatment.
    """

    required_donors = {"target", "paired"}
    if set(native_scores) != required_donors:
        raise ValueError("native support requires exactly target and paired donors")
    decisions: dict[str, Any] = {}
    for history, by_donor in candidate_scores.items():
        if set(by_donor) != required_donors:
            raise ValueError(
                f"history {history!r} requires exactly target and paired donor scores"
            )
        donor_decisions: dict[str, Any] = {}
        for donor, score in by_donor.items():
            native = float(native_scores[donor])
            observed = float(score)
            gap = native - observed
            donor_decisions[donor] = {
                "candidate_score": observed,
                "native_score": native,
                "native_minus_candidate": gap,
                "ratio_floor_log": float(threshold),
                "passed": bool(gap <= float(threshold)),
            }
        decisions[history] = {
            "by_donor": donor_decisions,
            "passed": bool(donor_decisions) and all(item["passed"] for item in donor_decisions.values()),
        }
    return {
        "threshold_log": float(threshold),
        "histories": decisions,
        "admitted_histories": sorted(name for name, value in decisions.items() if value["passed"]),
    }


def score_margin(*, target_score: float, paired_score: float) -> float:
    """Compute target-minus-paired late-coordinate log-score margin."""

    return float(target_score) - float(paired_score)


def donor_transport_contrast(
    *, target_margin_under_target: float,
    target_margin_under_paired: float,
) -> float:
    """Difference-in-differences for a target-versus-paired donor state."""

    return float(target_margin_under_target) - float(target_margin_under_paired)


def compute_noop_epsilon(
    *,
    unrestricted_coordinate_log_probabilities: Mapping[str, Sequence[float]],
    noop_coordinate_log_probabilities: Mapping[str, Sequence[float]],
) -> float:
    """Return the max absolute raw per-coordinate no-op drift."""

    drifts: list[float] = []
    for owner in ("target", "paired"):
        left = unrestricted_coordinate_log_probabilities[owner]
        right = noop_coordinate_log_probabilities[owner]
        if len(left) != len(right):
            raise ValueError("no-op coordinate vectors differ in length")
        drifts.extend(abs(float(a) - float(b)) for a, b in zip(left, right, strict=True))
    return float(max(drifts, default=0.0))


def valid_xyxy(box: Sequence[float] | None) -> bool:
    if box is None or len(box) != 4:
        return False
    x1, y1, x2, y2 = [float(value) for value in box]
    return bool(0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0)


def classify_crossed_box_ownership(
    *,
    parsed_boxes: Mapping[str, Mapping[str, Any]],
    accepted_objects: Sequence[Mapping[str, Any]],
    iou_floor: float = 0.30,
    owner_margin: float = 0.15,
) -> dict[str, Any]:
    """Classify crossed boxes without making an owner claim on ambiguity."""

    per_box: dict[str, Any] = {}
    for name, parsed in parsed_boxes.items():
        box = parsed.get("normalized_box")
        if not parsed.get("valid") or not valid_xyxy(box):
            per_box[name] = {"status": "invalid", "owner": None, "ious": {}}
            continue
        ious = {
            str(item["annotation_id"]): float(screen._box_iou(box, item["normalized_box"]))
            for item in accepted_objects
        }
        ordered = sorted(ious.items(), key=lambda item: (-item[1], item[0]))
        best = ordered[0] if ordered else (None, 0.0)
        second = ordered[1][1] if len(ordered) > 1 else 0.0
        unique = bool(best[0] is not None and best[1] >= float(iou_floor) and best[1] - second >= float(owner_margin))
        per_box[name] = {
            "status": "unique" if unique else "ambiguous",
            "owner": best[0] if unique else None,
            "best_iou": float(best[1]),
            "second_iou": float(second),
            "owner_margin": float(best[1] - second),
            "ious": ious,
        }
    owners = [value.get("owner") for value in per_box.values()]
    passed = bool(
        len(per_box) == 2
        and all(value.get("status") == "unique" for value in per_box.values())
        and owners[0] != owners[1]
    )
    return {
        "passed": passed,
        "physical_owner_claim_permitted": passed,
        "reason": "admissible_distinct_unique_owners" if passed else "invalid_or_ambiguous_crossed_box",
        "boxes": per_box,
    }


def classify_crossed_history_owners(
    *,
    parsed_by_history: Mapping[str, Mapping[str, Mapping[str, Any]]],
    accepted_objects: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the owner gate independently to each predeclared history pair."""

    per_history = {
        str(history): classify_crossed_box_ownership(
            parsed_boxes={str(owner): parsed for owner, parsed in boxes.items()},
            accepted_objects=accepted_objects,
        )
        for history, boxes in parsed_by_history.items()
    }
    admissible = sorted(
        history for history, result in per_history.items() if result.get("passed")
    )
    return {
        "passed": bool(admissible),
        "physical_owner_claim_permitted": bool(admissible),
        "physical_owner_claim_prohibited": not bool(admissible),
        "admissible_histories": admissible,
        "reason": (
            "admissible_distinct_unique_owners"
            if admissible
            else "invalid_or_ambiguous_crossed_box"
        ),
        "per_history": per_history,
    }


def conjoin_owner_gate_with_branch_support(
    *, owner_gate: Mapping[str, Any], admitted_histories: Sequence[str],
) -> dict[str, Any]:
    """Permit owner admission only where the same history passed branch support."""

    raw_admissible = sorted(str(value) for value in owner_gate.get("admissible_histories", []))
    admitted = {str(value) for value in admitted_histories}
    supported_admissible = sorted(value for value in raw_admissible if value in admitted)
    result = dict(owner_gate)
    result.update(
        {
            "crossed_box_only_admissible_histories": raw_admissible,
            "branch_support_admitted_histories": sorted(admitted),
            "admissible_histories": supported_admissible,
            "passed": bool(supported_admissible),
            "physical_owner_claim_permitted": bool(supported_admissible),
            "physical_owner_claim_prohibited": not bool(supported_admissible),
            "owner_admissibility_conjoined_with_branch_support": True,
            "reason": (
                "branch_supported_distinct_unique_owners"
                if supported_admissible
                else "no_branch_supported_distinct_unique_owner_history"
            ),
        }
    )
    return result


def classify_smoke(
    *,
    execution_trust_passed: bool,
    admitted_histories: Sequence[str],
    contrasts: Mapping[str, Mapping[str, float]],
    owner_gate: Mapping[str, Any],
    effect_floor: float = 0.05,
    positive_layer: int = POSITIVE_LAYER,
    negative_layer: int = NEGATIVE_CONTROL_LAYER,
) -> dict[str, Any]:
    """Classify only the donor-state late-coordinate estimand."""

    positive = []
    reversed_direction = []
    weak = []
    negative = []
    for history in admitted_histories:
        cells = contrasts.get(history, {})
        block23 = float(cells.get("block23_transport", 0.0))
        block13 = float(cells.get("block13_transport", 0.0))
        if block23 >= float(effect_floor):
            positive.append(history)
        elif block23 <= -float(effect_floor):
            reversed_direction.append(history)
        else:
            weak.append(history)
        if abs(block13) >= float(effect_floor):
            negative.append(history)
    if not execution_trust_passed:
        label = "invalid_execution_trust_gate"
        interpreted = False
    elif not admitted_histories:
        label = "no_branch_supported_common_history"
        interpreted = True
    elif reversed_direction:
        label = "reversed_donor_direction"
        interpreted = True
    elif weak:
        label = "inconclusive_weak_or_mixed_donor_transport"
        interpreted = True
    elif len(positive) == len(admitted_histories) and negative:
        label = "negative_control_veto"
        interpreted = True
    elif len(positive) == len(admitted_histories):
        label = "donor_state_late_coordinate_transport_supported"
        interpreted = True
    else:
        label = "donor_state_late_coordinate_transport_not_supported"
        interpreted = True
    return {
        "conclusion_scope": "DonorStateLateCoordinateTransport",
        "classification": label,
        "interpreted": interpreted,
        "positive_layer": int(positive_layer),
        "negative_control_layer": int(negative_layer),
        "admitted_histories": list(admitted_histories),
        "positive_histories": positive,
        "reversed_direction_histories": reversed_direction,
        "weak_histories": weak,
        "negative_control_histories": negative,
        "effect_floor": float(effect_floor),
        # Image 7818 is explicitly donor-state-only in Wave One.  The
        # crossed-box gate remains informative, but it can never authorize a
        # physical-owner interpretation for this smoke.
        "physical_owner_claim_permitted": False,
        "physical_owner_claim_prohibited": True,
        "owner_gate_reason": "image_7818_donor_state_only_stop_rule",
        "crossed_box_gate_observation": owner_gate.get("reason"),
    }


def _score_row(
    *,
    model: Any,
    model_inputs: Mapping[str, Any],
    features: Any,
    grid_thw: Sequence[int],
    merge_size: int,
    prompt_ids: Sequence[int],
    row: Sequence[int],
    image_grid_thw: torch.Tensor,
    image_token_id: int,
    layer_idx: int,
    selected_indices: Sequence[int] | None = None,
    replacement: Any | None = None,
    resolved_module: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Score one complete forced row through the parent probe seam."""

    device = next(model.parameters()).device
    ids = torch.tensor([[*map(int, prompt_ids), *map(int, row)]], dtype=torch.long, device=device)
    positions = query.derive_explicit_position_ids(
        model,
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        image_grid_thw=image_grid_thw.to(device=device),
    )
    structural = None
    custom = None
    if selected_indices is not None:
        custom, structural = parent.build_dynamic_mask_for_ids(
            ids=ids,
            image_token_id=image_token_id,
            selected_indices=selected_indices,
            prefix_length=len(prompt_ids),
            device=device,
        )
    boundary = len(prompt_ids) + len(parent.shared_row_prefix(row, row)) - 1
    logits = residual._score_with_hooks(
        model=model,
        model_inputs=model_inputs,
        features=features,
        grid_thw=grid_thw,
        merge_size=merge_size,
        ids=ids,
        image_grid_thw=image_grid_thw,
        position_ids=positions,
        custom_mask=custom,
        layer_idx=int(layer_idx),
        boundary_pos=boundary,
        replacement=replacement,
        resolved_module=resolved_module,
    )
    score = query.score_row_log_likelihoods(
        logits,
        prefix_length=len(prompt_ids),
        row_tokens=row,
        description_length=len(row) - 8,
        terminal_token_id=None,
    )
    score["coordinate_log_probabilities"] = parent._coordinate_log_probs(score, row)
    score["early_coordinate_score"] = early_coordinate_score(score, row)
    score["late_coordinate_score"] = late_coordinate_score(score, row)
    score["position_ids_sha256"] = parent.sha256_tensor(positions)
    score["prefix_token_history_sha256"] = parent.sha256_tensor(ids[0, : boundary + 1])
    score["position_boundary_sha256"] = parent.sha256_tensor(positions[..., : boundary + 1])
    score["boundary_pos"] = int(boundary)
    score["replacement_count"] = None if replacement is None else int(replacement.replacement_count)
    score["non_boundary_max_abs_delta"] = None if replacement is None else float(replacement.other_position_max_abs_delta)
    return score, structural


def _capture_layer_states(
    *,
    model: Any,
    model_inputs: Mapping[str, Any],
    features: Any,
    grid_thw: Sequence[int],
    merge_size: int,
    prompt_ids: Sequence[int],
    owner_masks: Mapping[str, Sequence[int]],
    image_grid_thw: torch.Tensor,
    image_token_id: int,
    layer_idx: int,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    prefix = parent.shared_row_prefix(
        parent.TARGET_REALIZED_DONOR_ROW_TOKEN_IDS,
        parent.PAIRED_REALIZED_DONOR_ROW_TOKEN_IDS,
    )
    prefix_ids = torch.tensor([[*map(int, prompt_ids), *prefix]], dtype=torch.long, device=device)
    positions = query.derive_explicit_position_ids(
        model,
        input_ids=prefix_ids,
        attention_mask=torch.ones_like(prefix_ids),
        image_grid_thw=image_grid_thw.to(device=device),
    )
    boundary_pos = int(prefix_ids.shape[1] - 1)
    module, resolution = residual.resolve_decoder_layer(model, int(layer_idx))
    states: dict[str, torch.Tensor] = {}
    receipts: dict[str, Any] = {}
    for owner_name in ("target", "paired"):
        mask, structural = parent.build_dynamic_mask_for_ids(
            ids=prefix_ids,
            image_token_id=image_token_id,
            selected_indices=owner_masks[owner_name],
            prefix_length=len(prompt_ids),
            device=device,
        )
        capture = residual.ResidualStateCapture(module, boundary_pos=boundary_pos)
        residual._score_with_hooks(
            model=model,
            model_inputs=model_inputs,
            features=features,
            grid_thw=grid_thw,
            merge_size=merge_size,
            ids=prefix_ids,
            image_grid_thw=image_grid_thw,
            position_ids=positions,
            custom_mask=mask,
            layer_idx=int(layer_idx),
            boundary_pos=boundary_pos,
            capture=capture,
            resolved_module=module,
        )
        if capture.state is None or capture.capture_count != 1:
            raise RuntimeError("donor residual capture did not fire exactly once")
        states[owner_name] = capture.state
        receipts[owner_name] = {
            "capture_count": int(capture.capture_count),
            "state_sha256": parent.sha256_tensor(capture.state),
            "mask_indices": [int(v) for v in owner_masks[owner_name]],
            "structural_mask_gate": structural,
            "position_boundary_contract": {
                "passed": True,
                "boundary_pos": boundary_pos,
                "position_ids_sha256": parent.sha256_tensor(positions),
                "shared_token_history": True,
            },
        }
    self_capture = residual.ResidualStateCapture(module, boundary_pos=boundary_pos)
    residual._score_with_hooks(
        model=model,
        model_inputs=model_inputs,
        features=features,
        grid_thw=grid_thw,
        merge_size=merge_size,
        ids=prefix_ids,
        image_grid_thw=image_grid_thw,
        position_ids=positions,
        custom_mask=None,
        layer_idx=int(layer_idx),
        boundary_pos=boundary_pos,
        capture=self_capture,
        resolved_module=module,
    )
    if self_capture.state is None or self_capture.capture_count != 1:
        raise RuntimeError("self-state capture did not fire exactly once")
    return {
        "layer_idx": int(layer_idx),
        "module_resolution": residual.build_decoder_layer_resolution_receipt(resolution),
        "states": states,
        "self_state": self_capture.state,
        "donor_receipts": receipts,
        "self_capture_count": int(self_capture.capture_count),
        "prefix_ids": prefix_ids,
        "positions": positions,
        "boundary_pos": boundary_pos,
    }


def _late_greedy_release(
    *,
    model: Any,
    model_inputs: Mapping[str, Any],
    features: Any,
    grid_thw: Sequence[int],
    merge_size: int,
    prompt_ids: Sequence[int],
    common_xy: Sequence[int],
    image_grid_thw: torch.Tensor,
    image_token_id: int,
    layer_idx: int,
    replacement_state: torch.Tensor | None,
    max_new_tokens: int = MAX_LATE_GREEDY_TOKENS,
) -> dict[str, Any]:
    """Generate at most ``x2,y2,BOX_END`` with full-prefix recomputation."""

    device = next(model.parameters()).device
    prefix = parent.shared_row_prefix(
        parent.TARGET_REALIZED_DONOR_ROW_TOKEN_IDS,
        parent.PAIRED_REALIZED_DONOR_ROW_TOKEN_IDS,
    )
    current = torch.tensor(
        [[*map(int, prompt_ids), *prefix, *map(int, common_xy)]],
        dtype=torch.long,
        device=device,
    )
    original_boundary = len(prompt_ids) + len(prefix) - 1
    module, _ = residual.resolve_decoder_layer(model, int(layer_idx))
    generated: list[int] = []
    selected_log_probs: list[float] = []
    replacement_counts: list[int] = []
    for _ in range(int(max_new_tokens)):
        positions = query.derive_explicit_position_ids(
            model,
            input_ids=current,
            attention_mask=torch.ones_like(current),
            image_grid_thw=image_grid_thw.to(device=device),
        )
        replacement = None
        if replacement_state is not None:
            replacement = residual.ResidualStateReplacement(
                module,
                boundary_pos=original_boundary,
                replacement=replacement_state,
            )
        logits = residual._score_with_hooks(
            model=model,
            model_inputs=model_inputs,
            features=features,
            grid_thw=grid_thw,
            merge_size=merge_size,
            ids=current,
            image_grid_thw=image_grid_thw,
            position_ids=positions,
            custom_mask=None,
            layer_idx=int(layer_idx),
            boundary_pos=original_boundary,
            replacement=replacement,
            resolved_module=module,
        )
        next_logits = logits[-1].to(dtype=torch.float32)
        log_probs = torch.log_softmax(next_logits, dim=-1)
        token = int(torch.argmax(next_logits).item())
        generated.append(token)
        selected_log_probs.append(float(log_probs[token].item()))
        replacement_counts.append(0 if replacement is None else int(replacement.replacement_count))
        current = torch.cat((current, torch.tensor([[token]], dtype=torch.long, device=device)), dim=1)
        if token == int(parent.BOX_END):
            break
    full_suffix = [*map(int, common_xy), *generated]
    parsed = screen.parse_geometry_suffix(full_suffix)
    return {
        "common_xy_token_ids": [int(v) for v in common_xy],
        "generated_late_token_ids": generated,
        "selected_token_log_probabilities": selected_log_probs,
        "replacement_counts": replacement_counts,
        "generated_count": len(generated),
        "natural_box_end": bool(generated and generated[-1] == parent.BOX_END),
        "parsed_full_suffix": parsed,
        "valid_full_suffix": bool(parsed.get("valid")),
        "max_new_tokens": int(max_new_tokens),
        "full_prefix_recomputed_each_step": True,
    }


def _score_intervention(
    *,
    model: Any,
    model_inputs: Mapping[str, Any],
    features: Any,
    grid_thw: Sequence[int],
    prompt_ids: Sequence[int],
    common_xy: Sequence[int],
    late_refs: Mapping[str, Sequence[int]],
    image_grid_thw: torch.Tensor,
    image_token_id: int,
    layer_idx: int,
    intervention: str,
    states: Mapping[str, Any],
    module: Any,
) -> dict[str, Any]:
    replacements = {
        "unrestricted": None,
        "self_noop": states["self_state"],
        "block23_target": states["states"].get("target") if int(layer_idx) == POSITIVE_LAYER else None,
        "block23_paired": states["states"].get("paired") if int(layer_idx) == POSITIVE_LAYER else None,
        "block13_target": states["states"].get("target") if int(layer_idx) == NEGATIVE_CONTROL_LAYER else None,
        "block13_paired": states["states"].get("paired") if int(layer_idx) == NEGATIVE_CONTROL_LAYER else None,
    }
    if intervention not in replacements:
        raise ValueError(f"unknown intervention {intervention}")
    replacement_state = replacements[intervention]
    rows = {name: make_row(common_xy, late_refs[name]) for name in ("target", "paired")}
    scores: dict[str, Any] = {}
    for owner_name, row in rows.items():
        replacement = None
        if replacement_state is not None:
            replacement = residual.ResidualStateReplacement(
                module,
                boundary_pos=len(prompt_ids) + len(parent.shared_row_prefix(row, row)) - 1,
                replacement=replacement_state,
            )
        score, _ = _score_row(
            model=model,
            model_inputs=model_inputs,
            features=features,
            grid_thw=grid_thw,
            merge_size=2,
            prompt_ids=prompt_ids,
            row=row,
            image_grid_thw=image_grid_thw,
            image_token_id=image_token_id,
            layer_idx=int(layer_idx),
            replacement=replacement,
            resolved_module=module,
        )
        scores[owner_name] = {
            "row_token_ids": row,
            "late_coordinate_score": float(score["late_coordinate_score"]),
            "early_coordinate_score": float(score["early_coordinate_score"]),
            "coordinate_log_probabilities": score["coordinate_log_probabilities"],
            "position_ids_sha256": score["position_ids_sha256"],
            "prefix_token_history_sha256": score["prefix_token_history_sha256"],
            "position_boundary_sha256": score["position_boundary_sha256"],
            "boundary_pos": score["boundary_pos"],
            "replacement_count": score["replacement_count"],
            "non_boundary_max_abs_delta": score["non_boundary_max_abs_delta"],
        }
    margin = score_margin(
        target_score=scores["target"]["late_coordinate_score"],
        paired_score=scores["paired"]["late_coordinate_score"],
    )
    return {"intervention": intervention, "scores": scores, "target_minus_paired_late_margin": margin}


def _score_early_with_donor_state(
    *,
    model: Any,
    model_inputs: Mapping[str, Any],
    features: Any,
    grid_thw: Sequence[int],
    prompt_ids: Sequence[int],
    common_xy: Sequence[int],
    late_xy: Sequence[int],
    image_grid_thw: torch.Tensor,
    image_token_id: int,
    layer_idx: int,
    donor_state: torch.Tensor,
    module: Any,
) -> dict[str, Any]:
    """Score ``EarlyCoordinateScore(h,c)`` with one-shot state replacement."""

    row = make_row(common_xy, late_xy)
    replacement = residual.ResidualStateReplacement(
        module,
        boundary_pos=len(prompt_ids) + len(parent.shared_row_prefix(row, row)) - 1,
        replacement=donor_state,
    )
    score, _ = _score_row(
        model=model,
        model_inputs=model_inputs,
        features=features,
        grid_thw=grid_thw,
        merge_size=2,
        prompt_ids=prompt_ids,
        row=row,
        image_grid_thw=image_grid_thw,
        image_token_id=int(image_token_id),
        layer_idx=int(layer_idx),
        replacement=replacement,
        resolved_module=module,
    )
    return {
        "early_coordinate_score": float(score["early_coordinate_score"]),
        "coordinate_log_probabilities": score["coordinate_log_probabilities"],
        "replacement_count": score["replacement_count"],
        "non_boundary_max_abs_delta": score["non_boundary_max_abs_delta"],
        "prefix_token_history_sha256": score["prefix_token_history_sha256"],
        "position_boundary_sha256": score["position_boundary_sha256"],
        "boundary_pos": score["boundary_pos"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infer-config", type=Path, default=parent.DEFAULT_CONFIG)
    parser.add_argument("--source-jsonl", type=Path, default=parent.DEFAULT_SOURCE_JSONL)
    parser.add_argument("--audit-ledger", type=Path, default=parent.DEFAULT_LEDGER)
    parser.add_argument("--parent-split-receipt", type=Path, default=parent.PARENT_SPLIT_RECEIPT)
    parser.add_argument("--parent-merged-receipt", type=Path, default=parent.PARENT_MERGED_RECEIPT)
    parser.add_argument("--cohort", type=Path, default=parent.COHORT_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-id", default=IMAGE_ID)
    parser.add_argument("--layers", nargs="+", type=int, default=list(LAYERS))
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if str(args.image_id) != IMAGE_ID or [int(v) for v in args.layers] != list(LAYERS):
        raise SystemExit("Wave One smoke is frozen to image 7818 and decoder blocks 23,13")
    parent_receipt = parent.validate_parent_receipts(
        split_path=Path(args.parent_split_receipt),
        merged_path=Path(args.parent_merged_receipt),
        cohort_path=Path(args.cohort),
    )
    config_path = Path(args.infer_config).expanduser().resolve(strict=True)
    source_path = Path(args.source_jsonl).expanduser().resolve(strict=True)
    ledger_path = Path(args.audit_ledger).expanduser().resolve(strict=True)
    input_hashes = parent.validate_input_hashes(
        config_path=config_path,
        source_jsonl_path=source_path,
        audit_ledger_path=ledger_path,
    )

    from src.config.inference import load_infer_config
    from src.data import load_raw_examples
    from src.inference.image_plan import materialize_image_plan_batch
    from src.inference.pipeline import _processor_config, _template_config
    from src.inference.prompt import build_prompt_record
    from src.inference.runtime import assemble_runtime
    from src.analysis.visual_support_counterfactual import capture_feature_bundle, validate_feature_layout
    from scripts.research.run_sampled_rescue_transition import _temporary_cwd

    with _temporary_cwd(config_path.parents[3]):
        resolved = load_infer_config(config_path)
    config = resolved.config.model_copy(update={"model": resolved.config.model.model_copy(update={"dtype": "fp32"})})
    runtime = assemble_runtime(config, source_gate_root=config_path.parents[3])
    qwen = runtime.qwen
    qwen.model.eval()
    actual_dtypes = sorted({str(parameter.dtype) for parameter in qwen.model.parameters()})
    if actual_dtypes != ["torch.float32"]:
        raise RuntimeError(f"float32 execution contract failed: {actual_dtypes!r}")
    attention = str(getattr(qwen.model.config, "_attn_implementation", None) or getattr(qwen.model.config, "attn_implementation", None) or "unknown")
    if attention != "sdpa":
        raise RuntimeError(f"SDPA execution contract failed: {attention!r}")

    raw_rows = load_raw_examples(source_path)
    raw = next(row for row in raw_rows if str(row.metadata.get("source", {}).get("image_id")) == IMAGE_ID)
    image_path = Path(raw.image.path).expanduser().resolve(strict=True)
    if raw.image.width != EXPECTED_IMAGE_WIDTH or raw.image.height != EXPECTED_IMAGE_HEIGHT:
        raise ValueError("image-7818 dimensions drifted")
    image_sha = sha256_file(image_path)
    if image_sha != EXPECTED_IMAGE_SHA256:
        raise ValueError("canonical image-7818 SHA-256 drifted")
    objects = {str(obj.object_id): obj for obj in raw.objects}
    target, paired = objects[parent.TARGET_ANNOTATION_ID], objects[parent.PAIRED_ANNOTATION_ID]
    prompt_record = build_prompt_record(raw, _template_config(resolved.config), processor=qwen.processor, row_index=0)
    prompt_ids = [int(v) for v in prompt_record.prompt_token_ids]
    owner_rows = {"target": query._row_token_ids(qwen.tokenizer, target), "paired": query._row_token_ids(qwen.tokenizer, paired)}
    if owner_rows["target"] != parent.TARGET_CANONICAL_GT_ROW_TOKEN_IDS or owner_rows["paired"] != parent.PAIRED_CANONICAL_GT_ROW_TOKEN_IDS:
        raise ValueError("live image-7818 canonical rows drifted")
    plan = materialize_image_plan_batch(
        [raw],
        components=qwen,
        processor_config=_processor_config(resolved.config),
        materialize=True,
        row_indices=[0],
    )
    model_inputs = plan.model_inputs_by_row_id[raw.example_id]
    # ``pixel_values`` is patch-packed and its first dimension is not a
    # physical batch dimension.  The image-plan batch contract is therefore
    # attested by the image-grid batch axis (and any explicit input-id axis),
    # rather than by every tensor in ``model_inputs``.
    tensor_batch_sizes = {
        "image_grid_thw": int(model_inputs["image_grid_thw"].shape[0])
        if isinstance(model_inputs.get("image_grid_thw"), torch.Tensor)
        else None,
        **{
            key: int(model_inputs[key].shape[0])
            for key in ("input_ids", "attention_mask")
            if isinstance(model_inputs.get(key), torch.Tensor) and model_inputs[key].ndim > 0
        },
    }
    if any(value not in (None, 1) for value in tensor_batch_sizes.values()):
        raise RuntimeError(f"physical batch-size contract failed: {tensor_batch_sizes!r}")
    grid_thw = [int(v) for v in model_inputs["image_grid_thw"].reshape(-1, 3)[0].tolist()]
    if grid_thw != [1, 52, 78] or int(qwen.processor_identity.merge_size) != 2:
        raise ValueError("image-7818 processor grid contract drifted")
    features = capture_feature_bundle(qwen.model, model_inputs)
    layout = validate_feature_layout(features, features, grid_thw=grid_thw, merge_size=2)
    if (layout.merged_height, layout.merged_width) != (26, 39):
        raise ValueError("image-7818 merged feature layout drifted")
    image_token_id = int(getattr(qwen.model.config, "image_token_id", qwen.tokenizer.convert_tokens_to_ids("<|image_pad|>")))
    owner_masks = {"target": parent.TARGET_MASK_INDICES, "paired": parent.PAIRED_MASK_INDICES}
    ledger = query._load_ledger(ledger_path)
    accepted_ledger = [item for item in ledger.get(IMAGE_ID, []) if str(item.get("final_state")) == "accepted"]
    accepted_objects = parent.build_accepted_objects(accepted_ledger, width=raw.image.width, height=raw.image.height)
    late_refs = {
        "target": [int(v) for v in parent.TARGET_REALIZED_DONOR_ROW_TOKEN_IDS[7:9]],
        "paired": [int(v) for v in parent.PAIRED_REALIZED_DONOR_ROW_TOKEN_IDS[7:9]],
    }
    common_histories = candidate_common_histories()
    portability_state_receipt = validate_parent_portability_state_receipt()

    # Capture each layer before opening branch support.  Admission is defined
    # on the one-shot residual state h, not on a continuing visual mask.
    captured_layers: dict[int, dict[str, Any]] = {}
    for layer_idx in LAYERS:
        states = _capture_layer_states(
            model=qwen.model, model_inputs=model_inputs, features=features, grid_thw=grid_thw, merge_size=2,
            prompt_ids=prompt_ids, owner_masks=owner_masks, image_grid_thw=model_inputs["image_grid_thw"],
            image_token_id=image_token_id, layer_idx=layer_idx,
        )
        position_hash = parent.sha256_tensor(states["positions"])
        token_history_hash = parent.sha256_tensor(states["prefix_ids"][0])
        identity_gate = validate_recaptured_state_identity(
            layer_idx=layer_idx,
            donor_receipts=states["donor_receipts"],
            expected_receipt=portability_state_receipt,
            boundary_pos=states["boundary_pos"],
            position_ids_sha256=position_hash,
            token_history_sha256=token_history_hash,
        )
        states["parent_identity_gate"] = identity_gate
        states["prefix_token_history_sha256"] = token_history_hash
        states["position_ids_sha256"] = position_hash
        captured_layers[int(layer_idx)] = states

    positive_states = captured_layers[POSITIVE_LAYER]
    positive_module = residual.resolve_decoder_layer(qwen.model, POSITIVE_LAYER)[0]
    native_scores: dict[str, float] = {}
    candidate_scores: dict[str, dict[str, float]] = {}
    admission_score_receipts: dict[str, Any] = {"native": {}, "candidates": {}}
    for donor_name, donor_xy in (("target", common_histories["target_x1_y1"]), ("paired", common_histories["paired_x1_y1"])):
        native = _score_early_with_donor_state(
            model=qwen.model, model_inputs=model_inputs, features=features, grid_thw=grid_thw,
            prompt_ids=prompt_ids, common_xy=donor_xy, late_xy=late_refs[donor_name],
            image_grid_thw=model_inputs["image_grid_thw"], image_token_id=image_token_id,
            layer_idx=POSITIVE_LAYER, donor_state=positive_states["states"][donor_name], module=positive_module,
        )
        native_scores[donor_name] = float(native["early_coordinate_score"])
        admission_score_receipts["native"][donor_name] = native
    for history_name, common_xy in common_histories.items():
        candidate_scores[history_name] = {}
        admission_score_receipts["candidates"][history_name] = {}
        for donor_name in ("target", "paired"):
            candidate = _score_early_with_donor_state(
                model=qwen.model, model_inputs=model_inputs, features=features, grid_thw=grid_thw,
                prompt_ids=prompt_ids, common_xy=common_xy, late_xy=late_refs[donor_name],
                image_grid_thw=model_inputs["image_grid_thw"], image_token_id=image_token_id,
                layer_idx=POSITIVE_LAYER, donor_state=positive_states["states"][donor_name], module=positive_module,
            )
            candidate_scores[history_name][donor_name] = float(candidate["early_coordinate_score"])
            admission_score_receipts["candidates"][history_name][donor_name] = candidate
    admission = native_support_admission(
        candidate_scores=candidate_scores,
        native_scores=native_scores,
    )
    admission_execution_gate = all(
        receipt.get("replacement_count") == 1
        and receipt.get("non_boundary_max_abs_delta") == 0.0
        and receipt.get("prefix_token_history_sha256") == positive_states["prefix_token_history_sha256"]
        and receipt.get("position_boundary_sha256") == positive_states["position_ids_sha256"]
        and receipt.get("boundary_pos") == positive_states["boundary_pos"]
        for group in (admission_score_receipts["native"],)
        for receipt in group.values()
    ) and all(
        receipt.get("replacement_count") == 1
        and receipt.get("non_boundary_max_abs_delta") == 0.0
        and receipt.get("prefix_token_history_sha256") == positive_states["prefix_token_history_sha256"]
        and receipt.get("position_boundary_sha256") == positive_states["position_ids_sha256"]
        and receipt.get("boundary_pos") == positive_states["boundary_pos"]
        for histories in admission_score_receipts["candidates"].values()
        for receipt in histories.values()
    )
    admission["execution_gate"] = {"passed": bool(admission_execution_gate), "score_receipts": admission_score_receipts}

    layer_outputs: dict[str, Any] = {}
    contrasts: dict[str, dict[str, float]] = {}
    execution_trust = bool(admission_execution_gate)
    for layer_idx in LAYERS:
        states = captured_layers[int(layer_idx)]
        module = residual.resolve_decoder_layer(qwen.model, int(layer_idx))[0]
        cells_by_history: dict[str, Any] = {}
        for history_name, common_xy in common_histories.items():
            if history_name not in admission["admitted_histories"]:
                continue
            cells: dict[str, Any] = {}
            interventions = ["unrestricted", "self_noop"]
            interventions.extend(["block23_target", "block23_paired"] if layer_idx == POSITIVE_LAYER else ["block13_target", "block13_paired"])
            for intervention in interventions:
                cells[intervention] = _score_intervention(
                    model=qwen.model, model_inputs=model_inputs, features=features, grid_thw=grid_thw,
                    prompt_ids=prompt_ids, common_xy=common_xy, late_refs=late_refs,
                    image_grid_thw=model_inputs["image_grid_thw"], image_token_id=image_token_id,
                    layer_idx=layer_idx, intervention=intervention, states=states, module=module,
                )
            margin_target = float(cells["block23_target" if layer_idx == POSITIVE_LAYER else "block13_target"]["target_minus_paired_late_margin"])
            margin_paired = float(cells["block23_paired" if layer_idx == POSITIVE_LAYER else "block13_paired"]["target_minus_paired_late_margin"])
            cells["transport_contrast"] = donor_transport_contrast(
                target_margin_under_target=margin_target,
                target_margin_under_paired=margin_paired,
            )
            cells_by_history[history_name] = cells
            contrasts.setdefault(history_name, {})["block23_transport" if layer_idx == POSITIVE_LAYER else "block13_transport"] = float(cells["transport_contrast"])
        noop_gate = {"passed": True, "epsilon_noop": 0.0, "max_abs_coordinate_logprob_drift": 0.0}
        for history_cells in cells_by_history.values():
            noop = history_cells["self_noop"]
            unrestricted = history_cells["unrestricted"]
            noop_gate["max_abs_coordinate_logprob_drift"] = max(
                float(noop_gate["max_abs_coordinate_logprob_drift"]),
                compute_noop_epsilon(
                    unrestricted_coordinate_log_probabilities={
                        owner: unrestricted["scores"][owner]["coordinate_log_probabilities"]
                        for owner in ("target", "paired")
                    },
                    noop_coordinate_log_probabilities={
                        owner: noop["scores"][owner]["coordinate_log_probabilities"]
                        for owner in ("target", "paired")
                    },
                ),
            )
        noop_gate["epsilon_noop"] = float(noop_gate["max_abs_coordinate_logprob_drift"])
        noop_gate["effect_floor"] = max(0.05, 5.0 * float(noop_gate["epsilon_noop"]))
        noop_gate["passed"] = bool(noop_gate["epsilon_noop"] <= TOLERANCE)
        structural_pass = all(
            bool(item.get("structural_mask_gate", {}).get("passed"))
            for item in states["donor_receipts"].values()
        )
        replacement_pass = all(
            all(
                cells[intervention]["scores"][owner]["replacement_count"] == 1
                and cells[intervention]["scores"][owner]["non_boundary_max_abs_delta"] == 0.0
                for owner in ("target", "paired")
            )
            for cells in cells_by_history.values()
            for intervention in cells
            if intervention != "transport_contrast"
            and intervention in {"self_noop", "block23_target", "block23_paired", "block13_target", "block13_paired"}
        )
        prefix_identity_pass = all(
            cells[intervention]["scores"][owner]["boundary_pos"] == states["boundary_pos"]
            and cells[intervention]["scores"][owner]["prefix_token_history_sha256"] == states["prefix_token_history_sha256"]
            and cells[intervention]["scores"][owner]["position_boundary_sha256"] == states["position_ids_sha256"]
            for cells in cells_by_history.values()
            for intervention in cells
            if intervention != "transport_contrast"
            for owner in ("target", "paired")
        )
        execution_trust = bool(
            execution_trust
            and noop_gate["passed"]
            and structural_pass
            and replacement_pass
            and prefix_identity_pass
            and states["self_capture_count"] == 1
            and states["parent_identity_gate"]["passed"]
        )
        layer_outputs[str(layer_idx)] = {
            "layer_idx": int(layer_idx),
            "donor_states": states["donor_receipts"],
            "self_capture_count": states["self_capture_count"],
            "boundary_pos": states["boundary_pos"],
            "position_ids_sha256": parent.sha256_tensor(states["positions"]),
            "prefix_token_history_sha256": states["prefix_token_history_sha256"],
            "parent_identity_gate": states["parent_identity_gate"],
            "no_op_gate": noop_gate,
            "structural_mask_gate_passed": structural_pass,
            "replacement_gate_passed": replacement_pass,
            "prefix_identity_gate_passed": prefix_identity_pass,
            "histories": cells_by_history,
            "replacement_contract": {
                "one_shot_boundary_only": True,
                "donor_state_cache_discarded": True,
                "full_model_batch_size": 1,
            },
        }

    crossed_parsed: dict[str, Mapping[str, Any]] = {}
    for history_name, common_xy in common_histories.items():
        crossed_parsed[history_name] = {}
        for owner_name in ("target", "paired"):
            row = make_row(common_xy, late_refs[owner_name])
            crossed_parsed[history_name][owner_name] = screen.parse_geometry_suffix(row[-5:])
    owner_gate = conjoin_owner_gate_with_branch_support(
        owner_gate=classify_crossed_history_owners(
            parsed_by_history=crossed_parsed,
            accepted_objects=accepted_objects,
        ),
        admitted_histories=admission["admitted_histories"],
    )
    greedy: dict[str, Any] = {}
    # Reuse the already identity-gated capture instead of recapturing an
    # unverified state for the secondary greedy readout.
    positive_states = captured_layers[POSITIVE_LAYER]
    for history_name in admission["admitted_histories"]:
        common_xy = common_histories[history_name]
        greedy[history_name] = {
            donor: _late_greedy_release(
                model=qwen.model, model_inputs=model_inputs, features=features, grid_thw=grid_thw, merge_size=2,
                prompt_ids=prompt_ids, common_xy=common_xy, image_grid_thw=model_inputs["image_grid_thw"],
                image_token_id=image_token_id, layer_idx=POSITIVE_LAYER,
                replacement_state=positive_states["states"][donor],
            )
            for donor in ("target", "paired")
        }
    decision = classify_smoke(
        execution_trust_passed=execution_trust,
        admitted_histories=admission["admitted_histories"],
        contrasts=contrasts,
        owner_gate=owner_gate,
        effect_floor=max(
            0.05,
            5.0
            * max(
                (float(value["no_op_gate"]["epsilon_noop"]) for value in layer_outputs.values()),
                default=0.0,
            ),
        ),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "conclusion_scope": "DonorStateLateCoordinateTransport",
        "request_identity": {"image_id": IMAGE_ID, "target_annotation_id": parent.TARGET_ANNOTATION_ID, "paired_annotation_id": parent.PAIRED_ANNOTATION_ID},
        "parent_receipts": {"split_sha256": parent_receipt["split_sha256"], "merged_sha256": parent_receipt["merged_sha256"], "cohort_sha256": parent_receipt["cohort_sha256"]},
        "parent_portability_receipt": portability_state_receipt,
        "input_hashes": input_hashes,
        "image_identity": {"path": str(image_path), "width": raw.image.width, "height": raw.image.height, "sha256": image_sha},
        "runtime_contract": {"dtype": "torch.float32", "attention_implementation": attention, "physical_batch_size": 1, "model_input_tensor_batch_sizes": tensor_batch_sizes, "repetition_penalty": 1.0, "logits_processor": None},
        "frozen_contract": {"layers": list(LAYERS), "coordinate_order": "x1,y1,x2,y2", "x_axis": "horizontal", "y_axis": "vertical", "image_grid_thw": grid_thw, "merge_size": 2, "shared_prefix_token_ids": parent.shared_row_prefix(parent.TARGET_REALIZED_DONOR_ROW_TOKEN_IDS, parent.PAIRED_REALIZED_DONOR_ROW_TOKEN_IDS), "prompt_ids_sha256": sha256_json(prompt_ids), "candidate_common_histories": common_histories, "late_reference_coordinates": late_refs},
        "support_admission": admission,
        "layers": layer_outputs,
        "crossed_box_owner_gate": owner_gate,
        "bounded_greedy_late_release": greedy,
        "panel_decision": decision,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload["runner_sha256"] = sha256_file(Path(__file__))
    payload["normalized_argv"] = list(sys.argv[1:] if argv is None else argv)
    (output_dir / "receipt.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
