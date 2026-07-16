#!/usr/bin/env python3
"""Conditional residual-geometry portability probe for image 7818.

The parent eligibility screen already identified two clean, same-description
geometry donors (annotations ``664730`` and ``661523``).  This successor keeps
the intervention local: one fixed full-image encoding, one shared pre-``x1``
prefix, and two declared decoder layers (23 positive, 13 negative control).
The module deliberately exposes small pure contracts for tests; the optional
``run`` path is the only part that assembles a model and therefore is not used
by the focused CPU test suite.
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

import scripts.research.run_fixed_encoding_downstream_geometry_state_portability as geometry  # noqa: E402
import scripts.research.run_fixed_encoding_persistent_hard_geometry_donor_eligibility_screen as screen  # noqa: E402
import scripts.research.run_fixed_encoding_downstream_residual_state_portability as residual  # noqa: E402
import scripts.research.run_fixed_encoding_query_scoped_object_centered_spatial_eligibility as query  # noqa: E402


UNIT_ID = "2026-07-16-fixed-encoding-persistent-hard-routing-geometry-state-portability-image7818"
SCHEMA_VERSION = "fixed-encoding-persistent-hard-routing-geometry-state-portability-image7818.v1"
IMAGE_ID = "7818"
TARGET_ANNOTATION_ID = "664730"
PAIRED_ANNOTATION_ID = "661523"
OBJECT_NAME = "wine glass"
LAYERS = (23, 13)
POSITIVE_LAYER = 23
NEGATIVE_CONTROL_LAYER = 13
ROLE = "pre_x1"
MAX_NEW_TOKENS = 5
TOLERANCE = 1e-4
PORTABILITY_RELEASE_FLOOR = 0.10
GEOMETRY_IOU_FLOOR = 0.30

PARENT_SPLIT_RECEIPT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-15-fixed-encoding-persistent-hard-routing-geometry-donor-eligibility-screen/"
    "cohort-six-float32-20260715a/split-7818/receipt.json"
)
PARENT_MERGED_RECEIPT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-15-fixed-encoding-persistent-hard-routing-geometry-donor-eligibility-screen/"
    "cohort-six-float32-20260715a/merged/receipt.json"
)
COHORT_PATH = screen.DEFAULT_COHORT
PARENT_SPLIT_RECEIPT_SHA256 = "7c4d9ea4c6a82da103453de4189ad1041f46efde3cf0333a6b17442d322aa017"
PARENT_MERGED_RECEIPT_SHA256 = "2046d5784030f255bc039b9252b2e63770e9f9b069219cfa9478a6e94c8f59b2"
COHORT_SHA256 = "1f7d44bf8291f7b3e708372f97ed51ac85eebef3b97fa3d83456adbe7f1060bf"
CONFIG_SHA256 = "f3000588accbcf1d9ada3b2f3e0b3324d660b4810b75d8f5d050d9f184f9ca80"
SOURCE_JSONL_SHA256 = "9b524a8c20f03758e2e3939703ff35a1e35a1108fb095ac6e7af5a1539c8cfc4"
AUDIT_LEDGER_SHA256 = "52e9f21eb32f7c3793d1356125931cb4c2a1fa5a12647d0d437dec217668d8df"
DEFAULT_CONFIG = screen.DEFAULT_CONFIG
DEFAULT_SOURCE_JSONL = screen.DEFAULT_SOURCE_JSONL
DEFAULT_LEDGER = screen.DEFAULT_LEDGER

TARGET_SUPPORT_RECTANGLE = [8, 22, 7, 15]
PAIRED_SUPPORT_RECTANGLE = [9, 23, 18, 26]
TARGET_MASK_INDICES = screen.rectangle_indices(TARGET_SUPPORT_RECTANGLE, merged_height=26, merged_width=39)
PAIRED_MASK_INDICES = screen.rectangle_indices(PAIRED_SUPPORT_RECTANGLE, merged_height=26, merged_width=39)
TARGET_CANONICAL_GT_ROW_TOKEN_IDS = [151646, 71437, 8991, 151647, 151648, 151883, 152021, 152026, 152443, 151649]
PAIRED_CANONICAL_GT_ROW_TOKEN_IDS = [151646, 71437, 8991, 151647, 151648, 152189, 152042, 152316, 152496, 151649]
TARGET_REALIZED_DONOR_ROW_TOKEN_IDS = [151646, 71437, 8991, 151647, 151648, 151879, 152018, 152029, 152494, 151649]
PAIRED_REALIZED_DONOR_ROW_TOKEN_IDS = [151646, 71437, 8991, 151647, 151648, 152189, 152041, 152315, 152409, 151649]
TARGET_SUFFIX_TOKEN_IDS = TARGET_REALIZED_DONOR_ROW_TOKEN_IDS[-5:]
PAIRED_SUFFIX_TOKEN_IDS = PAIRED_REALIZED_DONOR_ROW_TOKEN_IDS[-5:]
PERSISTENT_COORDINATE_RELEASE = {
    "target": 0.4376984238624573,
    "paired": 1.0549952983856201,
}

OBJECT_REF_START = residual.OBJECT_REF_START
OBJECT_REF_END = residual.OBJECT_REF_END
BOX_START = residual.BOX_START
BOX_END = residual.BOX_END
COORDINATE_TOKEN_START = residual.COORDINATE_TOKEN_START
COORDINATE_PHASES = ("x1", "y1", "x2", "y2")


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


def sha256_tensor(value: torch.Tensor) -> str:
    return residual.sha256_tensor(value)


def build_request_identity() -> dict[str, Any]:
    return {
        "image_id": IMAGE_ID,
        "recipient": {"object_name": OBJECT_NAME, "annotation_id": TARGET_ANNOTATION_ID},
        "donors": [
            {"donor_name": "target_wine_glass", "object_name": OBJECT_NAME, "annotation_id": TARGET_ANNOTATION_ID},
            {"donor_name": "paired_wine_glass", "object_name": OBJECT_NAME, "annotation_id": PAIRED_ANNOTATION_ID},
        ],
    }


def validate_input_hashes(
    *, config_path: Path, source_jsonl_path: Path, audit_ledger_path: Path,
    expected: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Check immutable config/source/ledger identities before model assembly."""

    expected_values = {
        "config_sha256": CONFIG_SHA256,
        "source_jsonl_sha256": SOURCE_JSONL_SHA256,
        "audit_ledger_sha256": AUDIT_LEDGER_SHA256,
    }
    if expected is not None:
        expected_values.update({str(key): str(value) for key, value in expected.items()})
    paths = {
        "config_sha256": config_path,
        "source_jsonl_sha256": source_jsonl_path,
        "audit_ledger_sha256": audit_ledger_path,
    }
    observed = {key: sha256_file(path) for key, path in paths.items()}
    for key, value in observed.items():
        if value != expected_values[key]:
            raise ValueError(f"{key} drifted: expected {expected_values[key]}, observed {value}")
    return observed


def _result_for_image(payload: Mapping[str, Any], image_id: str = IMAGE_ID) -> Mapping[str, Any]:
    result = next((item for item in payload.get("results", []) if str(item.get("image_id")) == str(image_id)), None)
    if not isinstance(result, Mapping):
        raise ValueError(f"receipt lacks image {image_id}")
    return result


def _validate_parent_result(result: Mapping[str, Any]) -> None:
    if str(result.get("unit_id")) != screen.UNIT_ID:
        raise ValueError("parent result unit_id drifted")
    expected = {
        "target_annotation_id": TARGET_ANNOTATION_ID,
        "paired_annotation_id": PAIRED_ANNOTATION_ID,
        "description": OBJECT_NAME,
        "image_grid_thw": [1, 52, 78],
        "merge_size": 2,
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(f"parent image-7818 field {key!r} drifted")
    arms = result.get("arms")
    if not isinstance(arms, Mapping) or not {"target", "paired"}.issubset(arms):
        raise ValueError("parent image-7818 lacks target/paired arms")
    expected_arms = {
        "target": (TARGET_SUPPORT_RECTANGLE, TARGET_MASK_INDICES, TARGET_SUFFIX_TOKEN_IDS, TARGET_REALIZED_DONOR_ROW_TOKEN_IDS, TARGET_ANNOTATION_ID),
        "paired": (PAIRED_SUPPORT_RECTANGLE, PAIRED_MASK_INDICES, PAIRED_SUFFIX_TOKEN_IDS, PAIRED_REALIZED_DONOR_ROW_TOKEN_IDS, PAIRED_ANNOTATION_ID),
    }
    for name, (rectangle, indices, suffix, row, annotation_id) in expected_arms.items():
        arm = arms[name]
        if arm.get("support_rectangle") != rectangle or [int(v) for v in arm.get("support_indices", [])] != indices:
            raise ValueError(f"parent {name} support contract drifted")
        generated = arm.get("generated", {})
        if [int(v) for v in generated.get("generated_token_ids", [])] != suffix or not generated.get("natural_closure"):
            raise ValueError(f"parent {name} generated geometry path drifted")
        if [int(v) for v in arm.get("realized_row_token_ids", [])] != row:
            raise ValueError(f"parent {name} realized row drifted")
        observed_release = arm.get("coordinate_release")
        if observed_release is None or abs(float(observed_release) - PERSISTENT_COORDINATE_RELEASE[name]) > TOLERANCE:
            raise ValueError(f"parent {name} persistent coordinate release drifted")
        attribution = arm.get("attribution", {})
        if str(attribution.get("donor_annotation_id")) != annotation_id:
            raise ValueError(f"parent {name} donor attribution drifted")
        if not arm.get("eligibility", {}).get("passed"):
            raise ValueError(f"parent {name} is not an eligible clean donor")


def validate_parent_receipts(
    *, split_path: Path = PARENT_SPLIT_RECEIPT, merged_path: Path = PARENT_MERGED_RECEIPT,
    cohort_path: Path = COHORT_PATH,
) -> dict[str, Any]:
    """Validate both authoritative parent receipts and their shared case."""

    split_resolved = Path(split_path).expanduser().resolve(strict=True)
    merged_resolved = Path(merged_path).expanduser().resolve(strict=True)
    split_sha = sha256_file(split_resolved)
    merged_sha = sha256_file(merged_resolved)
    if split_sha != PARENT_SPLIT_RECEIPT_SHA256:
        raise ValueError("image-7818 split parent receipt SHA-256 drifted")
    if merged_sha != PARENT_MERGED_RECEIPT_SHA256:
        raise ValueError("merged parent receipt SHA-256 drifted")
    cohort_resolved = Path(cohort_path).expanduser().resolve(strict=True)
    cohort_sha = sha256_file(cohort_resolved)
    if cohort_sha != COHORT_SHA256:
        raise ValueError("frozen donor cohort SHA-256 drifted")
    split = json.loads(split_resolved.read_text(encoding="utf-8"))
    merged = json.loads(merged_resolved.read_text(encoding="utf-8"))
    for payload in (split, merged):
        if payload.get("unit_id") != screen.UNIT_ID:
            raise ValueError("parent receipt unit_id mismatch")
        for key, expected in (("config_sha256", CONFIG_SHA256), ("source_jsonl_sha256", SOURCE_JSONL_SHA256), ("audit_ledger_sha256", AUDIT_LEDGER_SHA256), ("cohort_sha256", COHORT_SHA256)):
            if str(payload.get(key)) != expected:
                raise ValueError(f"parent receipt {key} drifted")
    split_result = _result_for_image(split)
    merged_result = _result_for_image(merged)
    _validate_parent_result(split_result)
    _validate_parent_result(merged_result)
    if split_result != merged_result:
        raise ValueError("merged image-7818 case differs from split receipt")
    return {
        "split": split,
        "merged": merged,
        "split_result": split_result,
        "merged_result": merged_result,
        "split_sha256": split_sha,
        "merged_sha256": merged_sha,
        "cohort_sha256": cohort_sha,
    }


# The lower-level dynamic-mask implementation is shared with the accepted
# geometry probe.  Aliases keep the contract discoverable from this successor.
query_range_for_complete_row = geometry.query_range_for_complete_row
query_range_for_partial_row = geometry.query_range_for_partial_row
build_partial_row_query_only_key_eligibility_mask = geometry.build_partial_row_query_only_key_eligibility_mask
inspect_partial_row_mask_structure = geometry.inspect_partial_row_mask_structure


def build_dynamic_mask_for_ids(
    *, ids: torch.Tensor, image_token_id: int, selected_indices: Sequence[int],
    prefix_length: int, device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Build the exact dynamic partial-row hard mask and structural receipt."""

    return geometry.build_dynamic_mask_for_ids(
        ids=ids,
        image_token_id=image_token_id,
        selected_indices=selected_indices,
        prefix_length=prefix_length,
        device=device,
    )


def parse_geometry_suffix(generated: Sequence[int], *, tokenizer: Any | None = None) -> dict[str, Any]:
    return screen.parse_geometry_suffix(generated, tokenizer=tokenizer)


def geometry_ownership(
    parsed: Mapping[str, Any], *, donor_box_normalized: Sequence[float],
    paired_box_normalized: Sequence[float], iou_floor: float = GEOMETRY_IOU_FLOOR,
) -> dict[str, Any]:
    return geometry.geometry_ownership(
        parsed,
        donor_box_normalized=donor_box_normalized,
        paired_box_normalized=paired_box_normalized,
        iou_floor=iou_floor,
    )


def assess_owner_match(
    parsed: Mapping[str, Any], *, donor_annotation_id: str,
    attribution: Mapping[str, Any], iou_floor: float = GEOMETRY_IOU_FLOOR,
) -> dict[str, Any]:
    """Require exact intended donor attribution in addition to valid geometry."""

    observed_id = str(attribution.get("donor_annotation_id", ""))
    donor_iou = attribution.get("donor_iou")
    competing_iou = float(attribution.get("strongest_competing_iou") or 0.0)
    all_object_ious = attribution.get("all_object_ious", {})
    accepted_donor_iou = (
        all_object_ious.get(str(donor_annotation_id))
        if isinstance(all_object_ious, Mapping)
        else None
    )
    support_donor_iou = attribution.get("support_envelope_donor_iou")
    donor_l1 = attribution.get("donor_l1_distance")
    support_donor_l1 = attribution.get("support_envelope_donor_l1_distance")
    support_iou_passed = bool(
        donor_iou is not None
        and support_donor_iou is not None
        and float(donor_iou) - float(support_donor_iou)
        >= screen.SUPPORT_IOU_IMPROVEMENT_MARGIN
    )
    support_l1_passed = bool(
        donor_l1 is not None
        and support_donor_l1 is not None
        and (
            float(donor_l1) <= screen.SUPPORT_L1_IMPROVEMENT_RATIO * float(support_donor_l1)
            if float(support_donor_l1) > 0.0
            else float(donor_l1) == 0.0
        )
    )
    passed = bool(
        parsed.get("valid")
        and observed_id == str(donor_annotation_id)
        and accepted_donor_iou is not None
        and donor_iou is not None
        and math.isclose(float(accepted_donor_iou), float(donor_iou), abs_tol=1e-12)
        and bool(attribution.get("strictly_closer_to_donor"))
        and float(donor_iou) >= float(iou_floor)
        and float(donor_iou) - competing_iou >= 0.15
        and support_iou_passed
        and support_l1_passed
    )
    return {
        "passed": passed,
        "expected_donor_annotation_id": str(donor_annotation_id),
        "observed_donor_annotation_id": observed_id,
        "strictly_closer_to_donor": bool(attribution.get("strictly_closer_to_donor")),
        "donor_iou": None if donor_iou is None else float(donor_iou),
        "accepted_ledger_donor_iou": (
            None if accepted_donor_iou is None else float(accepted_donor_iou)
        ),
        "strongest_competing_iou": competing_iou,
        "competing_object_margin": None if donor_iou is None else float(donor_iou) - competing_iou,
        "donor_iou_improves_over_support": support_iou_passed,
        "donor_l1_improves_over_support": support_l1_passed,
        "support_envelope_donor_iou": (
            None if support_donor_iou is None else float(support_donor_iou)
        ),
        "support_envelope_donor_l1_distance": (
            None if support_donor_l1 is None else float(support_donor_l1)
        ),
        "iou_floor": float(iou_floor),
    }


assess_geometry_donor_eligibility = geometry.assess_geometry_donor_eligibility
assess_geometry_portability = geometry.assess_geometry_portability


def assess_self_noop_trust(
    *, max_abs_logprob_drift: float, teacher_forced_top_token_ids_equal: bool,
    teacher_forced_selected_token_ranks_equal: bool, generated_token_ids_equal: bool,
    replacement_count: int, cached_replacement_count: int, cache_hook_removed: bool,
    tolerance: float = TOLERANCE,
) -> dict[str, Any]:
    """Apply the four-coordinate no-op and one-shot cache-seam trust gate."""

    base = geometry.assess_self_noop_trust(
        max_abs_logprob_drift=max_abs_logprob_drift,
        teacher_forced_top_token_ids_equal=teacher_forced_top_token_ids_equal,
        generated_token_ids_equal=generated_token_ids_equal,
        replacement_count=replacement_count,
        tolerance=tolerance,
    )
    base.update(
        {
            "teacher_forced_selected_token_ranks_equal": bool(
                teacher_forced_selected_token_ranks_equal
            ),
            "cached_replacement_count": int(cached_replacement_count),
            "cache_hook_removed": bool(cache_hook_removed),
        }
    )
    base["passed"] = bool(
        base["passed"]
        and teacher_forced_selected_token_ranks_equal
        and int(cached_replacement_count) == 1
        and cache_hook_removed
    )
    return base


def classify_geometry_panel(
    *, trust_passed: bool, eligible_count: int,
    layer23_passed_donors: Sequence[str], layer13_passed_donors: Sequence[str],
    unrestricted_baseline_owner: str | None,
    layer23_generated_owner_by_donor: Mapping[str, str | None],
) -> dict[str, Any]:
    """Apply the layer-13 veto and owner-specificity decision exactly once."""

    positive = {str(value) for value in layer23_passed_donors}
    negative = {str(value) for value in layer13_passed_donors}
    vetoed = sorted(positive & negative)
    localized_positive = positive - negative
    intended = {
        "target": TARGET_ANNOTATION_ID,
        "paired": PAIRED_ANNOTATION_ID,
    }
    generated = {
        str(owner): None if value is None else str(value)
        for owner, value in layer23_generated_owner_by_donor.items()
    }
    passing_intended_owners = {
        intended[donor]
        for donor in localized_positive
        if donor in intended and generated.get(donor) == intended[donor]
    }
    baseline_owner = (
        None if unrestricted_baseline_owner is None else str(unrestricted_baseline_owner)
    )
    owner_switch = bool(
        passing_intended_owners
        and any(owner != baseline_owner for owner in passing_intended_owners)
    )
    both_distinct = bool(
        {"target", "paired"}.issubset(localized_positive)
        and generated.get("target") == TARGET_ANNOTATION_ID
        and generated.get("paired") == PAIRED_ANNOTATION_ID
        and generated["target"] != generated["paired"]
    )
    owner_specific = bool(owner_switch or both_distinct)
    if not trust_passed:
        classification = "invalid_execution_trust_gate"
        interpreted = False
    elif positive and not localized_positive:
        classification = "vetoed_by_negative_control_layer_13"
        interpreted = False
    elif localized_positive and owner_specific:
        classification = "bounded_owner_specific_geometry_state_portability"
        interpreted = True
    elif localized_positive:
        classification = "bounded_donor_path_confidence_portability"
        interpreted = True
    elif int(eligible_count) > 0:
        classification = "close_one_site_conditional_downstream_portability"
        interpreted = True
    else:
        classification = "no_eligible_donor_control_only"
        interpreted = True
    return {
        "classification": classification,
        "interpreted": interpreted,
        "positive_layer": POSITIVE_LAYER,
        "negative_control_layer": NEGATIVE_CONTROL_LAYER,
        "eligible_donor_count": int(eligible_count),
        "layer23_passed_donors": sorted(positive),
        "localized_layer23_passed_donors": sorted(localized_positive),
        "layer13_passed_donors": sorted(negative),
        "corresponding_donor_veto_intersection": vetoed,
        "unrestricted_baseline_owner": baseline_owner,
        "layer23_generated_owner_by_donor": generated,
        "passing_intended_annotation_ids": sorted(passing_intended_owners),
        "owner_switch_from_unrestricted_baseline": owner_switch,
        "both_donor_states_recover_distinct_intended_owners": both_distinct,
        "owner_specific_geometry_portability": owner_specific,
    }


def shared_row_prefix(target_row: Sequence[int], paired_row: Sequence[int]) -> list[int]:
    """Return the generic multi-token prefix through ``BOX_START``."""

    target_prefix = screen.row_geometry_prefix(target_row)
    paired_prefix = screen.row_geometry_prefix(paired_row)
    if target_prefix != paired_prefix:
        raise ValueError("target and paired rows differ before x1")
    return target_prefix


def shared_boundary_contract(
    *, prompt_ids: Sequence[int], target_row: Sequence[int], paired_row: Sequence[int],
) -> dict[str, Any]:
    prefix = shared_row_prefix(target_row, paired_row)
    history = torch.tensor([*map(int, prompt_ids), *prefix], dtype=torch.long)
    boundary_pos = int(history.numel() - 1)
    return {
        "shared_prefix_token_ids": prefix,
        "boundary_pos": boundary_pos,
        "boundary_token_id": int(prefix[-1]),
        "token_history_sha256": sha256_tensor(history),
        "target_paired_prefix_equal": True,
    }


def position_boundary_contract(
    *, donor_position_ids: torch.Tensor, recipient_position_ids: torch.Tensor,
    boundary_pos: int,
) -> dict[str, Any]:
    """Attest exact position equality through the patched boundary."""

    if donor_position_ids.ndim != 3 or recipient_position_ids.ndim != 3:
        raise ValueError("position_ids must have shape [3,1,S]")
    boundary = int(boundary_pos)
    if boundary < 0 or boundary >= donor_position_ids.shape[-1] or boundary >= recipient_position_ids.shape[-1]:
        raise ValueError("boundary position is outside position tensors")
    equal = bool(torch.equal(donor_position_ids[..., : boundary + 1], recipient_position_ids[..., : boundary + 1]))
    return {
        "passed": equal,
        "boundary_pos": boundary,
        "donor_position_ids_sha256": sha256_tensor(donor_position_ids),
        "recipient_position_ids_sha256": sha256_tensor(recipient_position_ids),
        "boundary_position_ids_equal": equal,
        "donor_position_shape": list(donor_position_ids.shape),
        "recipient_position_shape": list(recipient_position_ids.shape),
    }


def _coordinate_mean(score: Mapping[str, Any]) -> float:
    return float(sum(float(score[phase]["mean"]) for phase in COORDINATE_PHASES) / 4.0)


def _coordinate_log_probs(score: Mapping[str, Any], row: Sequence[int]) -> list[float]:
    """Return only x1/y1/x2/y2 raw selected-token log probabilities."""

    values = [float(value) for value in score.get("token_log_probabilities", [])]
    description_length = len(row) - 8
    start = description_length + 3  # BOX_START is followed by x1.
    if len(values) < start + 4:
        raise ValueError("score lacks all four coordinate log probabilities")
    return values[start : start + 4]


def _coordinate_values(
    score: Mapping[str, Any], row: Sequence[int], key: str,
) -> list[int]:
    values = [int(value) for value in score.get(key, [])]
    description_length = len(row) - 8
    start = description_length + 3
    if len(values) < start + 4:
        raise ValueError(f"score lacks all four coordinate {key}")
    return values[start : start + 4]


def _box_end_log_probability(score: Mapping[str, Any]) -> float:
    values = score.get("token_log_probabilities", [])
    if not values:
        raise ValueError("score lacks token_log_probabilities")
    return float(values[-1])


def assess_score_reproduction(
    *, live: Mapping[str, Any], frozen: Mapping[str, Any], row: Sequence[int],
    tolerance: float = TOLERANCE,
) -> dict[str, Any]:
    """Compare the four frozen coordinate slots to one parent score."""

    live_tokens = _coordinate_log_probs(live, row)
    frozen_tokens = _coordinate_log_probs(frozen, row)
    drift = residual.max_abs_delta(live_tokens, frozen_tokens)
    live_ranks = _coordinate_values(live, row, "selected_token_ranks")
    frozen_ranks = _coordinate_values(frozen, row, "selected_token_ranks")
    live_top = _coordinate_values(live, row, "top_prediction_token_ids")
    frozen_top = _coordinate_values(frozen, row, "top_prediction_token_ids")
    selected_ids = [int(value) for value in row[-5:-1]]
    return {
        "passed": bool(drift <= float(tolerance) and live_ranks == frozen_ranks and live_top == frozen_top),
        "coordinate_slot_names": list(COORDINATE_PHASES),
        "selected_token_ids": selected_ids,
        "selected_token_ids_equal": len(selected_ids) == 4,
        "max_abs_logprob_drift": float(drift),
        "selected_token_ranks_equal": live_ranks == frozen_ranks,
        "top_prediction_token_ids_equal": live_top == frozen_top,
        "live_coordinate_log_probabilities": live_tokens,
        "frozen_coordinate_log_probabilities": frozen_tokens,
        "tolerance": float(tolerance),
    }


def build_accepted_objects(
    accepted_ledger: Sequence[Mapping[str, Any]], *, width: int, height: int,
) -> list[dict[str, Any]]:
    """Normalize every accepted ledger box into the attribution contract."""

    if int(width) <= 0 or int(height) <= 0:
        raise ValueError("image dimensions must be positive")
    objects: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in accepted_ledger:
        identifier = str(item.get("object_identifier", ""))
        annotation_id = identifier.split(":", 1)[-1]
        box = item.get("source_canvas_box_xyxy")
        if not annotation_id or not isinstance(box, Sequence) or isinstance(box, (str, bytes)) or len(box) != 4:
            raise ValueError("accepted ledger object lacks an annotation ID or four-value box")
        if annotation_id in seen:
            raise ValueError(f"duplicate accepted annotation ID {annotation_id}")
        seen.add(annotation_id)
        objects.append(
            {
                "annotation_id": annotation_id,
                "normalized_box": [
                    float(box[0]) / int(width),
                    float(box[1]) / int(height),
                    float(box[2]) / int(width),
                    float(box[3]) / int(height),
                ],
            }
        )
    return objects


def generated_owner(
    parsed: Mapping[str, Any], accepted_objects: Sequence[Mapping[str, Any]],
) -> str | None:
    """Return a unique maximum-IoU accepted-object owner, or ``None``."""

    box = parsed.get("normalized_box")
    if not parsed.get("valid") or not isinstance(box, Sequence) or len(box) != 4:
        return None
    scored = [
        (str(item["annotation_id"]), screen._box_iou(box, item["normalized_box"]))
        for item in accepted_objects
    ]
    if not scored:
        return None
    best_iou = max(value for _, value in scored)
    winners = [annotation_id for annotation_id, value in scored if value == best_iou]
    return winners[0] if best_iou > 0.0 and len(winners) == 1 else None


def persistent_parent_reproductions_passed(persistent: Mapping[str, Any]) -> bool:
    """Require hard and unrestricted coordinate reproduction for both donors."""

    owners = persistent.get("owners", {})
    return bool(
        set(owners) == {"target", "paired"}
        and all(
            bool(owners[name].get("parent_hard_reproduction", {}).get("passed"))
            and bool(owners[name].get("parent_unrestricted_reproduction", {}).get("passed"))
            for name in ("target", "paired")
        )
    )


def _full_row_mask(
    *, ids: torch.Tensor, image_token_id: int, selected_indices: Sequence[int],
    prefix_length: int, row_length: int, device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    image_positions = [int(value) for value in torch.where(ids[0] == int(image_token_id))[0].tolist()]
    if not image_positions:
        raise ValueError("row contains no image placeholder keys")
    if max((int(value) for value in selected_indices), default=-1) >= len(image_positions):
        raise ValueError("selected image index exceeds image-token count")
    eligible = [image_positions[int(value)] for value in selected_indices]
    mask = query.build_query_scoped_key_eligibility_mask(
        sequence_length=int(ids.shape[1]), image_key_positions=image_positions,
        eligible_image_positions=eligible, prefix_length=int(prefix_length), row_length=int(row_length), device=device,
    )
    receipt = query.inspect_query_scoped_mask_structure(
        mask, sequence_length=int(ids.shape[1]), image_key_positions=image_positions,
        eligible_image_positions=eligible, prefix_length=int(prefix_length), row_length=int(row_length),
    )
    receipt["image_token_positions"] = image_positions
    receipt["eligible_image_token_positions"] = eligible
    if not receipt.get("passed"):
        raise RuntimeError("complete-row hard mask failed structural contract")
    return mask, receipt


def _score_row(
    *, model: Any, model_inputs: Mapping[str, Any], features: Any, grid_thw: Sequence[int],
    merge_size: int, prompt_ids: Sequence[int], row: Sequence[int], image_grid_thw: torch.Tensor,
    image_token_id: int, layer_idx: int, selected_indices: Sequence[int] | None = None,
    capture: Any | None = None, replacement: Any | None = None, resolved_module: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    device = next(model.parameters()).device
    ids = torch.tensor([[*map(int, prompt_ids), *map(int, row)]], dtype=torch.long, device=device)
    positions = query.derive_explicit_position_ids(
        model, input_ids=ids, attention_mask=torch.ones_like(ids), image_grid_thw=image_grid_thw.to(device=device)
    )
    custom = structural = None
    if selected_indices is not None:
        custom, structural = _full_row_mask(
            ids=ids, image_token_id=image_token_id, selected_indices=selected_indices,
            prefix_length=len(prompt_ids), row_length=len(row), device=device,
        )
    boundary = len(prompt_ids) + len(shared_row_prefix(row, row)) - 1
    logits = residual._score_with_hooks(
        model=model, model_inputs=model_inputs, features=features, grid_thw=grid_thw,
        merge_size=merge_size, ids=ids, image_grid_thw=image_grid_thw, position_ids=positions,
        custom_mask=custom, layer_idx=int(layer_idx), boundary_pos=boundary,
        capture=capture, replacement=replacement, resolved_module=resolved_module,
    )
    score = query.score_row_log_likelihoods(
        logits, prefix_length=len(prompt_ids), row_tokens=row,
        description_length=len(row) - 8, terminal_token_id=None,
    )
    score["coordinate_only_mean"] = _coordinate_mean(score)
    score["box_end_log_probability"] = _box_end_log_probability(score)
    score["position_ids_sha256"] = sha256_tensor(positions)
    return score, structural


def _greedy_geometry(
    *, model: Any, model_inputs: Mapping[str, Any], features: Any, grid_thw: Sequence[int],
    merge_size: int, prefix_ids: torch.Tensor, prompt_length: int, image_grid_thw: torch.Tensor,
    image_token_id: int, selected_indices: Sequence[int] | None, tokenizer: Any | None,
) -> dict[str, Any]:
    """Generate exactly five tokens with full-prefix recomputation and dynamic masks."""

    if prefix_ids.ndim != 2 or prefix_ids.shape[0] != 1:
        raise ValueError("prefix_ids must have shape [1,S]")
    if int(prefix_ids.shape[1]) - int(prompt_length) != len(prefix_ids[0]) - int(prompt_length):
        raise AssertionError("unreachable prefix shape guard")
    device = next(model.parameters()).device
    current = prefix_ids.detach().clone().to(device=device, dtype=torch.long)
    generated: list[int] = []
    selected_log_probs: list[float] = []
    selected_ranks: list[int] = []
    top_ids: list[int] = []
    receipts: list[dict[str, Any]] = []
    position_hashes: list[str] = []
    for _ in range(MAX_NEW_TOKENS):
        positions = query.derive_explicit_position_ids(
            model, input_ids=current, attention_mask=torch.ones_like(current), image_grid_thw=image_grid_thw.to(device=device)
        )
        position_hashes.append(sha256_tensor(positions))
        custom = None
        if selected_indices is not None:
            custom, receipt = build_dynamic_mask_for_ids(
                ids=current, image_token_id=image_token_id, selected_indices=selected_indices,
                prefix_length=int(prompt_length), device=device,
            )
            receipts.append(receipt)
        logits = residual._score_with_hooks(
            model=model, model_inputs=model_inputs, features=features, grid_thw=grid_thw,
            merge_size=merge_size, ids=current, image_grid_thw=image_grid_thw, position_ids=positions,
            custom_mask=custom, layer_idx=POSITIVE_LAYER, boundary_pos=int(prefix_ids.shape[1] - 1),
        )
        next_logits = logits[-1].to(dtype=torch.float32)
        token = int(torch.argmax(next_logits).item())
        log_probs = torch.log_softmax(next_logits, dim=-1)
        generated.append(token)
        selected_log_probs.append(float(log_probs[token].item()))
        selected_ranks.append(int(1 + (next_logits > next_logits[token]).sum().item()))
        top_ids.append(token)
        current = torch.cat((current, torch.tensor([[token]], dtype=torch.long, device=device)), dim=1)
    parsed = parse_geometry_suffix(generated, tokenizer=tokenizer)
    return {
        "generated_token_ids": generated,
        "generated_token_count": len(generated),
        "selected_token_log_probabilities": selected_log_probs,
        "selected_token_ranks": selected_ranks,
        "top_prediction_token_ids": top_ids,
        "stop_reason": "box_end" if generated[-1] == BOX_END else "max_new_tokens",
        "natural_closure": bool(parsed.get("valid") and generated[-1] == BOX_END),
        "valid": bool(parsed.get("valid")),
        "parsed": parsed,
        "cache_used": False,
        "full_prefix_recomputed_each_step": True,
        "dynamic_mask_rebuilt_each_step": selected_indices is not None,
        "structural_mask_receipts": receipts,
        "structural_mask_gate_passed": bool(selected_indices is None or len(receipts) == MAX_NEW_TOKENS and all(item.get("passed") for item in receipts)),
        "position_ids_sha256": position_hashes,
        "repetition_penalty": 1.0,
        "logits_processor": None,
    }


def _row_from_suffix(prefix: Sequence[int], suffix: Sequence[int]) -> list[int]:
    parsed = parse_geometry_suffix(suffix)
    if not parsed.get("valid"):
        raise ValueError("cannot materialize row from invalid suffix")
    return [*map(int, prefix), *map(int, suffix)]


def _run_persistent_paths(
    *, model: Any, model_inputs: Mapping[str, Any], features: Any, grid_thw: Sequence[int], merge_size: int,
    prompt_ids: Sequence[int], owner_rows: Mapping[str, Sequence[int]], owner_masks: Mapping[str, Sequence[int]],
    owner_boxes: Mapping[str, Sequence[float]], image_grid_thw: torch.Tensor, image_token_id: int,
    tokenizer: Any, frozen_arms: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    device = next(model.parameters()).device
    prefix = shared_row_prefix(owner_rows["target"], owner_rows["paired"])
    prefix_ids = torch.tensor([[*map(int, prompt_ids), *prefix]], dtype=torch.long, device=device)
    owners: dict[str, Any] = {}
    for owner_name in ("target", "paired"):
        paired_name = "paired" if owner_name == "target" else "target"
        generated = _greedy_geometry(
            model=model, model_inputs=model_inputs, features=features, grid_thw=grid_thw,
            merge_size=merge_size, prefix_ids=prefix_ids, prompt_length=len(prompt_ids),
            image_grid_thw=image_grid_thw, image_token_id=image_token_id,
            selected_indices=owner_masks[owner_name], tokenizer=tokenizer,
        )
        expected_suffix = TARGET_SUFFIX_TOKEN_IDS if owner_name == "target" else PAIRED_SUFFIX_TOKEN_IDS
        expected_row = (
            TARGET_REALIZED_DONOR_ROW_TOKEN_IDS
            if owner_name == "target"
            else PAIRED_REALIZED_DONOR_ROW_TOKEN_IDS
        )
        generated["matches_frozen_suffix"] = [int(v) for v in generated["generated_token_ids"]] == expected_suffix
        generated["matches_frozen_row"] = bool(generated["matches_frozen_suffix"] and [*prefix, *generated["generated_token_ids"]] == expected_row)
        ownership = geometry_ownership(
            generated["parsed"], donor_box_normalized=owner_boxes[owner_name], paired_box_normalized=owner_boxes[paired_name],
        )
        live_row = _row_from_suffix(prefix, generated["generated_token_ids"]) if generated.get("valid") else None
        frozen_row = [int(value) for value in frozen_arms[owner_name]["realized_row_token_ids"]]
        scores = None
        if live_row is not None and generated.get("matches_frozen_row"):
            # The parent y_d is the scientific comparison path.  Never use a
            # newly generated row to redefine the release statistic.
            hard_score, hard_structural = _score_row(
                model=model, model_inputs=model_inputs, features=features, grid_thw=grid_thw,
                merge_size=merge_size, prompt_ids=prompt_ids, row=frozen_row, image_grid_thw=image_grid_thw,
                image_token_id=image_token_id, layer_idx=POSITIVE_LAYER, selected_indices=owner_masks[owner_name],
            )
            unrestricted_score, _ = _score_row(
                model=model, model_inputs=model_inputs, features=features, grid_thw=grid_thw,
                merge_size=merge_size, prompt_ids=prompt_ids, row=frozen_row, image_grid_thw=image_grid_thw,
                image_token_id=image_token_id, layer_idx=POSITIVE_LAYER,
            )
            scores = {
                "row_token_ids": frozen_row,
                "hard": hard_score,
                "unrestricted": unrestricted_score,
                "coordinate_release": float(hard_score["coordinate_only_mean"] - unrestricted_score["coordinate_only_mean"]),
                "box_end_release": float(hard_score["box_end_log_probability"] - unrestricted_score["box_end_log_probability"]),
                "box_end_excluded_from_eligibility": True,
                "structural_mask_receipt": hard_structural,
            }
        persistent_release = None if scores is None else float(scores["coordinate_release"])
        live_eligibility = assess_geometry_donor_eligibility(
            coordinate_release=persistent_release,
            valid_path=bool(generated.get("natural_closure") and generated.get("matches_frozen_row")),
            owner_match=bool(ownership.get("passed")),
        )
        hard_reproduction = None if scores is None else assess_score_reproduction(
            live=scores["hard"], frozen=frozen_arms[owner_name]["hard_score"], row=frozen_row,
        )
        unrestricted_reproduction = None if scores is None else assess_score_reproduction(
            live=scores["unrestricted"],
            frozen=frozen_arms[owner_name]["unrestricted_score"],
            row=frozen_row,
        )
        frozen_eligibility = bool(frozen_arms[owner_name].get("eligibility", {}).get("passed"))
        eligibility = dict(live_eligibility)
        eligibility.update(
            {
                "frozen_parent_eligibility_passed": frozen_eligibility,
                "parent_hard_reproduction_passed": bool(
                    hard_reproduction and hard_reproduction.get("passed")
                ),
                "parent_unrestricted_reproduction_passed": bool(
                    unrestricted_reproduction and unrestricted_reproduction.get("passed")
                ),
            }
        )
        eligibility["passed"] = bool(
            live_eligibility.get("passed")
            and frozen_eligibility
            and eligibility["parent_hard_reproduction_passed"]
            and eligibility["parent_unrestricted_reproduction_passed"]
        )
        owners[owner_name] = {
            "support_indices": list(map(int, owner_masks[owner_name])),
            "generated": generated,
            "realized_row_token_ids": frozen_row,
            "live_realized_row_token_ids": live_row,
            "realized_row_scores": scores,
            "geometry_ownership": ownership,
            "eligibility": eligibility,
            "parent_hard_reproduction": hard_reproduction,
            "parent_unrestricted_reproduction": unrestricted_reproduction,
        }
    return {
        "owners": owners,
        "eligible_count": sum(bool(item["eligibility"].get("passed")) for item in owners.values()),
        "parent_hard_reproduction": {
            owner: value["parent_hard_reproduction"] for owner, value in owners.items()
        },
        "parent_unrestricted_reproduction": {
            owner: value["parent_unrestricted_reproduction"] for owner, value in owners.items()
        },
    }


def _run_layer(
    *, model: Any, model_inputs: Mapping[str, Any], features: Any, grid_thw: Sequence[int], merge_size: int,
    prompt_ids: Sequence[int], owner_rows: Mapping[str, Sequence[int]], owner_masks: Mapping[str, Sequence[int]],
    owner_boxes: Mapping[str, Sequence[float]], accepted_objects: Sequence[Mapping[str, Any]],
    support_boxes: Mapping[str, Sequence[float]], persistent: Mapping[str, Any], image_grid_thw: torch.Tensor,
    image_token_id: int, tokenizer: Any, layer_idx: int,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    prefix = shared_row_prefix(owner_rows["target"], owner_rows["paired"])
    prefix_ids = torch.tensor([[*map(int, prompt_ids), *prefix]], dtype=torch.long, device=device)
    boundary_pos = int(prefix_ids.shape[1] - 1)
    positions = query.derive_explicit_position_ids(
        model, input_ids=prefix_ids, attention_mask=torch.ones_like(prefix_ids), image_grid_thw=image_grid_thw.to(device=device)
    )
    module, resolution = residual.resolve_decoder_layer(model, int(layer_idx))
    donor_states: dict[str, torch.Tensor] = {}
    donor_receipts: dict[str, Any] = {}
    for owner_name in ("target", "paired"):
        hard_mask, structural = build_dynamic_mask_for_ids(
            ids=prefix_ids, image_token_id=image_token_id, selected_indices=owner_masks[owner_name],
            prefix_length=len(prompt_ids), device=device,
        )
        capture = residual.ResidualStateCapture(module, boundary_pos=boundary_pos)
        residual._score_with_hooks(
            model=model, model_inputs=model_inputs, features=features, grid_thw=grid_thw, merge_size=merge_size,
            ids=prefix_ids, image_grid_thw=image_grid_thw, position_ids=positions, custom_mask=hard_mask,
            layer_idx=int(layer_idx), boundary_pos=boundary_pos, capture=capture, resolved_module=module,
        )
        if capture.state is None or capture.capture_count != 1:
            raise RuntimeError("donor residual state capture did not complete exactly once")
        donor_states[owner_name] = capture.state
        donor_receipts[owner_name] = {
            "capture_count": capture.capture_count,
            "state_sha256": sha256_tensor(capture.state),
            "mask_indices": list(map(int, owner_masks[owner_name])),
            "structural_mask_receipt": structural,
            "token_history_sha256": sha256_tensor(prefix_ids[0]),
            "token_history_equal_to_recipient_through_boundary": True,
            "position_boundary_contract": position_boundary_contract(
                donor_position_ids=positions,
                recipient_position_ids=positions,
                boundary_pos=boundary_pos,
            ),
        }

    # Self-state capture is a separate no-cache forward.  The recipient cache
    # below is fresh and must not be reused as the no-op evidence.
    self_capture = residual.ResidualStateCapture(module, boundary_pos=boundary_pos)
    residual._score_with_hooks(
        model=model, model_inputs=model_inputs, features=features, grid_thw=grid_thw, merge_size=merge_size,
        ids=prefix_ids, image_grid_thw=image_grid_thw, position_ids=positions, custom_mask=None,
        layer_idx=int(layer_idx), boundary_pos=boundary_pos, capture=self_capture,
        resolved_module=module,
    )
    if self_capture.state is None or self_capture.capture_count != 1:
        raise RuntimeError("recipient self-state capture did not complete")
    unrestricted_prefill = residual._score_with_hooks(
        model=model, model_inputs=model_inputs, features=features, grid_thw=grid_thw, merge_size=merge_size,
        ids=prefix_ids, image_grid_thw=image_grid_thw, position_ids=positions, custom_mask=None,
        layer_idx=int(layer_idx), boundary_pos=boundary_pos, use_cache=True, return_output=True, resolved_module=module,
    )
    baseline_cached = residual.greedy_cached_one_row_continuation(
        model, prefill_output=unrestricted_prefill, input_ids=prefix_ids, prefill_position_ids=positions,
        box_end_token_id=BOX_END, eos_token_id=getattr(tokenizer, "eos_token_id", None), max_new_tokens=MAX_NEW_TOKENS,
    )
    baseline_cached["parsed"] = parse_geometry_suffix(baseline_cached.get("generated_token_ids", []), tokenizer=tokenizer)
    baseline_cached["natural_closure"] = bool(baseline_cached.get("stop_reason") == "box_end" and baseline_cached["parsed"].get("valid"))
    baseline_cached["accepted_object_owner"] = generated_owner(
        baseline_cached["parsed"], accepted_objects
    )
    baseline_cached["hook_removed_after_prefill"] = True

    noop_hook = residual.ResidualStateReplacement(module, boundary_pos=boundary_pos, replacement=self_capture.state)
    noop_prefill = residual._score_with_hooks(
        model=model, model_inputs=model_inputs, features=features, grid_thw=grid_thw, merge_size=merge_size,
        ids=prefix_ids, image_grid_thw=image_grid_thw, position_ids=positions, custom_mask=None,
        layer_idx=int(layer_idx), boundary_pos=boundary_pos, replacement=noop_hook, use_cache=True,
        return_output=True, resolved_module=module,
    )
    noop_cached = residual.greedy_cached_one_row_continuation(
        model, prefill_output=noop_prefill, input_ids=prefix_ids, prefill_position_ids=positions,
        box_end_token_id=BOX_END, eos_token_id=getattr(tokenizer, "eos_token_id", None), max_new_tokens=MAX_NEW_TOKENS,
    )
    noop_cached["parsed"] = parse_geometry_suffix(noop_cached.get("generated_token_ids", []), tokenizer=tokenizer)
    noop_cached["natural_closure"] = bool(noop_cached.get("stop_reason") == "box_end" and noop_cached["parsed"].get("valid"))
    noop_cached["hook_removed_after_prefill"] = noop_hook.hook_removed_inside_hook
    noop_cached["replacement_count"] = noop_hook.replacement_count

    donor_arms: dict[str, Any] = {}
    no_op_drifts: list[float] = []
    no_op_replacement_counts: list[int] = []
    layer_passed_donors: list[str] = []
    for owner_name in ("target", "paired"):
        paired_name = "paired" if owner_name == "target" else "target"
        row = list(persistent["owners"][owner_name]["realized_row_token_ids"])
        baseline_score, _ = _score_row(
            model=model, model_inputs=model_inputs, features=features, grid_thw=grid_thw, merge_size=merge_size,
            prompt_ids=prompt_ids, row=row, image_grid_thw=image_grid_thw, image_token_id=image_token_id, layer_idx=layer_idx,
            resolved_module=module,
        )
        noop_score_hook = residual.ResidualStateReplacement(module, boundary_pos=boundary_pos, replacement=self_capture.state)
        noop_score, _ = _score_row(
            model=model, model_inputs=model_inputs, features=features, grid_thw=grid_thw, merge_size=merge_size,
            prompt_ids=prompt_ids, row=row, image_grid_thw=image_grid_thw, image_token_id=image_token_id, layer_idx=layer_idx,
            replacement=noop_score_hook,
            resolved_module=module,
        )
        noop_drift = residual.max_abs_delta(
            _coordinate_log_probs(baseline_score, row), _coordinate_log_probs(noop_score, row)
        )
        no_op_drifts.append(float(noop_drift))
        no_op_replacement_counts.append(noop_score_hook.replacement_count)
        replacement_hook = residual.ResidualStateReplacement(module, boundary_pos=boundary_pos, replacement=donor_states[owner_name])
        replacement_score, _ = _score_row(
            model=model, model_inputs=model_inputs, features=features, grid_thw=grid_thw, merge_size=merge_size,
            prompt_ids=prompt_ids, row=row, image_grid_thw=image_grid_thw, image_token_id=image_token_id, layer_idx=layer_idx,
            replacement=replacement_hook, resolved_module=module,
        )
        replacement_release = float(replacement_score["coordinate_only_mean"] - baseline_score["coordinate_only_mean"])
        donor_hook = residual.ResidualStateReplacement(module, boundary_pos=boundary_pos, replacement=donor_states[owner_name])
        donor_prefill = residual._score_with_hooks(
            model=model, model_inputs=model_inputs, features=features, grid_thw=grid_thw, merge_size=merge_size,
            ids=prefix_ids, image_grid_thw=image_grid_thw, position_ids=positions, custom_mask=None,
            layer_idx=layer_idx, boundary_pos=boundary_pos, replacement=donor_hook, use_cache=True,
            return_output=True, resolved_module=module,
        )
        donor_cached = residual.greedy_cached_one_row_continuation(
            model, prefill_output=donor_prefill, input_ids=prefix_ids, prefill_position_ids=positions,
            box_end_token_id=BOX_END, eos_token_id=getattr(tokenizer, "eos_token_id", None), max_new_tokens=MAX_NEW_TOKENS,
        )
        donor_cached["parsed"] = parse_geometry_suffix(donor_cached.get("generated_token_ids", []), tokenizer=tokenizer)
        donor_cached["natural_closure"] = bool(donor_cached.get("stop_reason") == "box_end" and donor_cached["parsed"].get("valid"))
        donor_cached["hook_removed_after_prefill"] = donor_hook.hook_removed_inside_hook
        donor_cached["replacement_count"] = donor_hook.replacement_count
        donor_cached["accepted_object_owner"] = generated_owner(
            donor_cached["parsed"], accepted_objects
        )
        attribution = screen.geometry_attribution(
            donor_cached["parsed"], donor_box=owner_boxes[owner_name], paired_box=owner_boxes[paired_name],
            donor_annotation_id=TARGET_ANNOTATION_ID if owner_name == "target" else PAIRED_ANNOTATION_ID,
            accepted_objects=accepted_objects, support_box=support_boxes[owner_name],
        )
        owner_match = assess_owner_match(
            donor_cached["parsed"], donor_annotation_id=TARGET_ANNOTATION_ID if owner_name == "target" else PAIRED_ANNOTATION_ID,
            attribution=attribution,
        )
        persistent_gate = persistent["owners"][owner_name]["eligibility"]
        portability = assess_geometry_portability(
            persistent_release=float(persistent_gate.get("coordinate_release") or 0.0),
            replacement_release=replacement_release,
            no_op_drift=noop_drift,
            valid_path=bool(donor_cached.get("natural_closure")),
            owner_match=bool(owner_match.get("passed")),
        ) if persistent_gate.get("passed") else {"passed": False, "reason": "persistent_hard_donor_ineligible"}
        if portability.get("passed"):
            layer_passed_donors.append(owner_name)
        donor_arms[owner_name] = {
            "row_token_ids": list(map(int, row)),
            "unrestricted": baseline_score,
            "self_state_noop": noop_score,
            "donor_state_replacement": replacement_score,
            "replacement_coordinate_release": replacement_release,
            "replacement_box_end_release": float(replacement_score["box_end_log_probability"] - baseline_score["box_end_log_probability"]),
            "box_end_excluded_from_portability": True,
            "self_noop_max_abs_logprob_drift": float(noop_drift),
            "cached_donor_replacement": donor_cached,
            "owner_attribution": attribution,
            "owner_match": owner_match,
            "portability": portability,
            "replacement_count": replacement_hook.replacement_count,
            "non_boundary_max_abs_delta": replacement_hook.other_position_max_abs_delta,
        }
    no_op_drift = max(no_op_drifts, default=0.0)
    teacher_top_equal = all(
        _coordinate_values(value["unrestricted"], value["row_token_ids"], "top_prediction_token_ids")
        == _coordinate_values(value["self_state_noop"], value["row_token_ids"], "top_prediction_token_ids")
        for value in donor_arms.values()
    )
    teacher_ranks_equal = all(
        _coordinate_values(value["unrestricted"], value["row_token_ids"], "selected_token_ranks")
        == _coordinate_values(value["self_state_noop"], value["row_token_ids"], "selected_token_ranks")
        for value in donor_arms.values()
    )
    generated_equal = baseline_cached.get("generated_token_ids") == noop_cached.get("generated_token_ids")
    noop_trust = assess_self_noop_trust(
        max_abs_logprob_drift=no_op_drift, teacher_forced_top_token_ids_equal=teacher_top_equal,
        teacher_forced_selected_token_ranks_equal=teacher_ranks_equal,
        generated_token_ids_equal=generated_equal,
        replacement_count=1 if all(count == 1 for count in no_op_replacement_counts) else 0,
        cached_replacement_count=noop_hook.replacement_count,
        cache_hook_removed=noop_hook.hook_removed_inside_hook,
    )
    cached_paths_for_exactness = {
        "unrestricted": baseline_cached,
        "self_state_noop": noop_cached,
        **{
            f"{name}_donor_state": value["cached_donor_replacement"]
            for name, value in donor_arms.items()
        },
    }
    exact_cached_suffix_gate = {
        name: bool(
            path.get("natural_closure")
            and path.get("generated_token_count") == MAX_NEW_TOKENS
            and len(path.get("generated_token_ids", [])) == MAX_NEW_TOKENS
            and path.get("generated_token_ids", [])[-1:] == [BOX_END]
        )
        for name, path in cached_paths_for_exactness.items()
    }
    return {
        "layer_idx": int(layer_idx),
        "role": ROLE,
        "absolute_boundary_pos": boundary_pos,
        "decoder_layer_resolution": residual.build_decoder_layer_resolution_receipt(resolution),
        "donor_states": donor_receipts,
        "arms": donor_arms,
        "cached_paths": {"unrestricted": baseline_cached, "self_state_noop": noop_cached},
        "self_noop_max_abs_logprob_drift": no_op_drift,
        "self_noop_teacher_forced_top_token_ids_equal": teacher_top_equal,
        "self_noop_teacher_forced_selected_token_ranks_equal": teacher_ranks_equal,
        "self_noop_generated_token_ids_equal": generated_equal,
        "self_noop_trust_gate": noop_trust,
        "self_noop_passed": bool(noop_trust.get("passed")),
        "exact_cached_suffix_gate": {
            "passed": all(exact_cached_suffix_gate.values()),
            "paths": exact_cached_suffix_gate,
            "required_token_count": MAX_NEW_TOKENS,
        },
        "portability_passed_donors": sorted(layer_passed_donors),
        "portability_passed_any_donor": bool(layer_passed_donors),
        "position_attestation": {
            "position_ids_sha256": sha256_tensor(positions),
            "position_ids_shape": list(positions.shape),
            "token_history_sha256": sha256_tensor(prefix_ids[0]),
            "boundary_pos": boundary_pos,
        },
        "replacement_contract": {
            "batch_index": 0,
            "returned_full_block_output": True,
            "donor_cache_discarded": True,
            "recipient_cache_only": True,
            "hooks_removed_before_later_calls": all(value["cached_donor_replacement"].get("hook_removed_after_prefill", False) for value in donor_arms.values()),
            "cached_replacement_counts": {
                name: value["cached_donor_replacement"].get("replacement_count")
                for name, value in donor_arms.items()
            },
            "non_boundary_max_abs_delta": {name: value["non_boundary_max_abs_delta"] for name, value in donor_arms.items()},
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infer-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-jsonl", type=Path, default=DEFAULT_SOURCE_JSONL)
    parser.add_argument("--audit-ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--parent-split-receipt", type=Path, default=PARENT_SPLIT_RECEIPT)
    parser.add_argument("--parent-merged-receipt", type=Path, default=PARENT_MERGED_RECEIPT)
    parser.add_argument("--cohort", type=Path, default=COHORT_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-id", default=IMAGE_ID)
    parser.add_argument("--layers", nargs="+", type=int, default=list(LAYERS))
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if str(args.image_id) != IMAGE_ID:
        raise SystemExit("successor is frozen to image 7818")
    if [int(value) for value in args.layers] != list(LAYERS):
        raise SystemExit("successor is frozen to decoder layers 23 and 13")
    if int(args.max_new_tokens) != MAX_NEW_TOKENS:
        raise SystemExit("successor requires exactly four coordinate tokens plus BOX_END")
    parent = validate_parent_receipts(
        split_path=Path(args.parent_split_receipt), merged_path=Path(args.parent_merged_receipt), cohort_path=Path(args.cohort)
    )
    config_path = Path(args.infer_config).expanduser().resolve(strict=True)
    source_path = Path(args.source_jsonl).expanduser().resolve(strict=True)
    ledger_path = Path(args.audit_ledger).expanduser().resolve(strict=True)
    live_hashes = validate_input_hashes(config_path=config_path, source_jsonl_path=source_path, audit_ledger_path=ledger_path)

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
    objects = {str(obj.object_id): obj for obj in raw.objects}
    target, paired = objects[TARGET_ANNOTATION_ID], objects[PAIRED_ANNOTATION_ID]
    template = _template_config(resolved.config)
    prompt_record = build_prompt_record(raw, template, processor=qwen.processor, row_index=0)
    prompt_ids = [int(value) for value in prompt_record.prompt_token_ids]
    owner_rows = {"target": query._row_token_ids(qwen.tokenizer, target), "paired": query._row_token_ids(qwen.tokenizer, paired)}
    if (
        owner_rows["target"] != TARGET_CANONICAL_GT_ROW_TOKEN_IDS
        or owner_rows["paired"] != PAIRED_CANONICAL_GT_ROW_TOKEN_IDS
    ):
        raise ValueError("live image-7818 canonical ground-truth rows drifted")
    prefix = shared_row_prefix(owner_rows["target"], owner_rows["paired"])
    realized_donor_prefix = shared_row_prefix(
        TARGET_REALIZED_DONOR_ROW_TOKEN_IDS,
        PAIRED_REALIZED_DONOR_ROW_TOKEN_IDS,
    )
    if prefix != realized_donor_prefix:
        raise ValueError("live image-7818 canonical prefix drifted from frozen donor prefix")

    plan = materialize_image_plan_batch([raw], components=qwen, processor_config=_processor_config(resolved.config), materialize=True, row_indices=[0])
    model_inputs = plan.model_inputs_by_row_id[raw.example_id]
    grid_thw = [int(value) for value in model_inputs["image_grid_thw"].reshape(-1, 3)[0].tolist()]
    if grid_thw != [1, 52, 78] or int(qwen.processor_identity.merge_size) != 2:
        raise ValueError("image-7818 processor geometry drifted")
    features = capture_feature_bundle(qwen.model, model_inputs)
    layout = validate_feature_layout(features, features, grid_thw=grid_thw, merge_size=2)
    if (layout.merged_height, layout.merged_width) != (26, 39):
        raise ValueError("image-7818 merged feature layout drifted")
    image_token_id = int(getattr(qwen.model.config, "image_token_id", qwen.tokenizer.convert_tokens_to_ids("<|image_pad|>")))
    owner_masks = {"target": TARGET_MASK_INDICES, "paired": PAIRED_MASK_INDICES}
    ledger = query._load_ledger(ledger_path)
    accepted_ledger = [
        item
        for item in ledger.get(IMAGE_ID, [])
        if str(item.get("final_state")) == "accepted"
    ]
    accepted = {
        str(item.get("object_identifier")): item for item in accepted_ledger
    }
    owner_boxes = {
        "target": query._object_pixel_box(target, accepted, width=raw.image.width, height=raw.image.height),
        "paired": query._object_pixel_box(paired, accepted, width=raw.image.width, height=raw.image.height),
    }
    if owner_boxes["target"] is None or owner_boxes["paired"] is None:
        raise ValueError("image-7818 owner boxes unavailable")
    normalized_boxes = {
        name: [float(box[0]) / raw.image.width, float(box[1]) / raw.image.height, float(box[2]) / raw.image.width, float(box[3]) / raw.image.height]
        for name, box in owner_boxes.items()
    }
    accepted_objects = build_accepted_objects(
        accepted_ledger, width=raw.image.width, height=raw.image.height
    )
    accepted_annotation_ids = {
        str(item["annotation_id"]) for item in accepted_objects
    }
    if not {TARGET_ANNOTATION_ID, PAIRED_ANNOTATION_ID}.issubset(accepted_annotation_ids):
        raise ValueError("accepted-object ledger lacks one or both frozen donors")
    support_boxes = {
        "target": screen.support_envelope_box(
            TARGET_SUPPORT_RECTANGLE, merged_height=26, merged_width=39
        ),
        "paired": screen.support_envelope_box(
            PAIRED_SUPPORT_RECTANGLE, merged_height=26, merged_width=39
        ),
    }
    frozen_arms = parent["split_result"]["arms"]
    persistent = _run_persistent_paths(
        model=qwen.model, model_inputs=model_inputs, features=features, grid_thw=grid_thw, merge_size=2,
        prompt_ids=prompt_ids, owner_rows=owner_rows, owner_masks=owner_masks, owner_boxes=normalized_boxes,
        image_grid_thw=model_inputs["image_grid_thw"], image_token_id=image_token_id, tokenizer=qwen.tokenizer,
        frozen_arms=frozen_arms,
    )
    results: list[dict[str, Any]] = []
    for layer_idx in LAYERS:
        boundary = _run_layer(
            model=qwen.model, model_inputs=model_inputs, features=features, grid_thw=grid_thw, merge_size=2,
            prompt_ids=prompt_ids, owner_rows=owner_rows, owner_masks=owner_masks, owner_boxes=normalized_boxes,
            accepted_objects=accepted_objects, support_boxes=support_boxes,
            persistent=persistent, image_grid_thw=model_inputs["image_grid_thw"], image_token_id=image_token_id,
            tokenizer=qwen.tokenizer, layer_idx=layer_idx,
        )
        trust = bool(
            persistent["owners"]["target"]["generated"].get("structural_mask_gate_passed")
            and persistent["owners"]["paired"]["generated"].get("structural_mask_gate_passed")
            and all(value.get("structural_mask_receipt", {}).get("passed") for value in boundary["donor_states"].values())
            and boundary["self_noop_passed"]
            and boundary["exact_cached_suffix_gate"]["passed"]
            and all(value["generated"].get("matches_frozen_row") for value in persistent["owners"].values())
            and persistent_parent_reproductions_passed(persistent)
            and all(
                value.get("capture_count") == 1
                and value.get("position_boundary_contract", {}).get("passed")
                for value in boundary["donor_states"].values()
            )
            and all(
                value.get("replacement_count") == 1
                and value.get("non_boundary_max_abs_delta") == 0.0
                and value.get("cached_donor_replacement", {}).get("replacement_count") == 1
                and value.get("cached_donor_replacement", {}).get("hook_removed_after_prefill")
                for value in boundary["arms"].values()
            )
        )
        boundary["scientific_gate"] = {
            "passed": trust,
            "self_state_noop_passed": boundary["self_noop_passed"],
            "dynamic_donor_mask_gate_passed": all(
                value.get("structural_mask_receipt", {}).get("passed")
                for value in boundary["donor_states"].values()
            ),
            "parent_hard_and_unrestricted_reproductions_passed": (
                persistent_parent_reproductions_passed(persistent)
            ),
            "exact_cached_suffix_gate_passed": boundary[
                "exact_cached_suffix_gate"
            ]["passed"],
            "layer_is_positive_seam": int(layer_idx) == POSITIVE_LAYER,
        }
        boundary["trust_gate_passed"] = trust
        results.append(boundary)
    layer23 = next(item for item in results if item["layer_idx"] == POSITIVE_LAYER)
    layer13 = next(item for item in results if item["layer_idx"] == NEGATIVE_CONTROL_LAYER)
    baseline_paths_equal = bool(
        layer23["cached_paths"]["unrestricted"].get("generated_token_ids")
        == layer13["cached_paths"]["unrestricted"].get("generated_token_ids")
    )
    trust_passed = bool(
        layer23["trust_gate_passed"]
        and layer13["trust_gate_passed"]
        and baseline_paths_equal
    )
    decision = classify_geometry_panel(
        trust_passed=trust_passed, eligible_count=int(persistent["eligible_count"]),
        layer23_passed_donors=layer23["portability_passed_donors"], layer13_passed_donors=layer13["portability_passed_donors"],
        unrestricted_baseline_owner=layer23["cached_paths"]["unrestricted"].get(
            "accepted_object_owner"
        ),
        layer23_generated_owner_by_donor={
            donor: arm["cached_donor_replacement"].get("accepted_object_owner")
            for donor, arm in layer23["arms"].items()
        },
    )
    decision["valid_trusted_layer13_mandatory"] = True
    decision["layer13_trust_gate_passed"] = bool(layer13["trust_gate_passed"])
    decision["unrestricted_baseline_paths_equal_across_layers"] = baseline_paths_equal
    return {
        "schema_version": SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "model_dtype": "torch.float32",
        "request_identity": build_request_identity(),
        "parent_receipts": {"split_path": str(Path(args.parent_split_receipt).expanduser().resolve()), "split_sha256": parent["split_sha256"], "merged_path": str(Path(args.parent_merged_receipt).expanduser().resolve()), "merged_sha256": parent["merged_sha256"], "cohort_sha256": parent["cohort_sha256"]},
        "input_hashes": live_hashes,
        "frozen_contract": {"target_canonical_ground_truth_row_token_ids": TARGET_CANONICAL_GT_ROW_TOKEN_IDS, "paired_canonical_ground_truth_row_token_ids": PAIRED_CANONICAL_GT_ROW_TOKEN_IDS, "target_realized_donor_row_token_ids": TARGET_REALIZED_DONOR_ROW_TOKEN_IDS, "paired_realized_donor_row_token_ids": PAIRED_REALIZED_DONOR_ROW_TOKEN_IDS, "target_support_rectangle": TARGET_SUPPORT_RECTANGLE, "paired_support_rectangle": PAIRED_SUPPORT_RECTANGLE, "target_mask_indices": TARGET_MASK_INDICES, "paired_mask_indices": PAIRED_MASK_INDICES, "image_grid_thw": grid_thw, "merge_size": 2, "shared_prefix_token_ids": prefix, "boundary_role": ROLE, "boundary_pos": len(prompt_ids) + len(prefix) - 1, "normalized_boxes": normalized_boxes, "accepted_objects": accepted_objects, "support_envelopes": support_boxes},
        "persistent_hard_paths": persistent,
        "results": results,
        "runtime_contract": {"dtype": "torch.float32", "attention_implementation": attention, "recipient_mask": "unrestricted", "donor_mask": "exact_dynamic_partial_row", "donor_cache_discarded": True, "recipient_use_cache": True, "teacher_forced_use_cache": False, "max_new_tokens": MAX_NEW_TOKENS, "repetition_penalty": 1.0, "logits_processor": None, "actual_model_parameter_dtypes": actual_dtypes},
        "panel_decision": decision,
    }


def main(argv: Sequence[str] | None = None) -> int:
    normalized = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    payload = run(args)
    payload["runner_sha256"] = sha256_file(Path(__file__))
    payload["normalized_argv"] = normalized
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "receipt.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
