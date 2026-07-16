#!/usr/bin/env python3
"""Bounded image-632 downstream residual-state geometry discriminator.

This experiment is intentionally local to image 632.  It reuses the accepted
fixed-encoding feature replay, query-scoped hard-mask, decoder-layer seam, and
recipient-cache helpers while implementing the geometry-specific state and
generation contract required by the research unit.
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
import scripts.research.run_fixed_encoding_query_scoped_object_centered_spatial_eligibility as query  # noqa: E402


UNIT_ID = "2026-07-15-fixed-encoding-downstream-geometry-state-portability"
IMAGE_ID = "632"
TARGET_ANNOTATION_ID = "1661908"
COMPETITOR_ANNOTATION_ID = "1989419"
OBJECT_NAME = "book"
ROW_QUERY_PARENT_UNIT_ID = "2026-07-15-fixed-encoding-row-scoring-query-only-spatial-key-eligibility-crossover"
ROW_QUERY_PARENT_SHA256 = "4aba45f16a90d032191a51f1c9c35ad9b71ae4d0ec97f943d9e0ddf217055fa4"
DEFAULT_ROW_QUERY_RECEIPT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-15-fixed-encoding-row-scoring-query-only-spatial-key-eligibility-crossover/"
    "cohort-four-float32-20260715a/receipt.json"
)
DEFAULT_CONFIG = residual.DEFAULT_CONFIG
DEFAULT_SOURCE_JSONL = residual.DEFAULT_SOURCE_JSONL
DEFAULT_LEDGER = residual.DEFAULT_LEDGER
LAYERS = (23, 13)
ROLE = "pre_x1"
TOLERANCE = residual.TOLERANCE
ELIGIBILITY_RELEASE_FLOOR = -0.05
PORTABILITY_RELEASE_FLOOR = 0.10
GEOMETRY_IOU_FLOOR = 0.30
MAX_NEW_TOKENS = 5
COORDINATE_PHASES = ("x1", "y1", "x2", "y2")
OBJECT_REF_START = residual.OBJECT_REF_START
OBJECT_REF_END = residual.OBJECT_REF_END
BOX_START = residual.BOX_START
BOX_END = residual.BOX_END
COORDINATE_TOKEN_START = residual.COORDINATE_TOKEN_START
TARGET_ROW_TOKEN_IDS = [151646, 2190, 151647, 151648, 152432, 152083, 152444, 152142, 151649]
COMPETITOR_ROW_TOKEN_IDS = [151646, 2190, 151647, 151648, 152446, 151784, 152454, 151842, 151649]
TARGET_MASK_INDICES = [386, 387, 388, 422, 423, 424, 458, 459, 460, 494, 495, 496]
COMPETITOR_MASK_INDICES = [99, 100, 101, 135, 136, 137, 171, 172, 173, 207, 208, 209]
CANONICAL_ROWS_SHA256 = "8b2a96f220e965e1822cb41f75ebcf170ad05fd6bd67cc0ae57ec69df58eec0e"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve(strict=True).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tensor(value: torch.Tensor) -> str:
    return residual.sha256_tensor(value)


def build_request_identity() -> dict[str, Any]:
    """Return the explicit image-632 recipient and donor provenance."""

    return {
        "image_id": IMAGE_ID,
        "recipient": {"object_name": OBJECT_NAME, "annotation_id": TARGET_ANNOTATION_ID},
        "donors": [
            {"donor_name": "target_book", "object_name": OBJECT_NAME, "annotation_id": TARGET_ANNOTATION_ID},
            {"donor_name": "competitor_book", "object_name": OBJECT_NAME, "annotation_id": COMPETITOR_ANNOTATION_ID},
        ],
    }


def query_range_for_complete_row(*, prefix_length: int, row_length: int) -> tuple[int, int]:
    """Return teacher-forced row query range ``P-1..P+R-2``."""

    return query.query_row_range(prefix_length=prefix_length, row_length=row_length)


def query_range_for_partial_row(*, prefix_length: int, partial_row_length: int) -> tuple[int, int]:
    """Return dynamic-generation range ``P-1..P+K-1`` including the final query."""

    prefix = int(prefix_length)
    partial = int(partial_row_length)
    if prefix <= 0 or partial <= 0:
        raise ValueError("prefix_length and partial_row_length must be positive")
    return prefix - 1, prefix + partial - 1


def build_partial_row_query_only_key_eligibility_mask(
    *, sequence_length: int, image_key_positions: Sequence[int], eligible_image_positions: Sequence[int],
    prefix_length: int, partial_row_length: int, device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Build a hard mask changing only dynamic partial-row queries."""

    length = int(sequence_length)
    if length <= 0:
        raise ValueError("sequence_length must be positive")
    image = {int(value) for value in image_key_positions}
    eligible = {int(value) for value in eligible_image_positions}
    if not eligible.issubset(image):
        raise ValueError("eligible image keys must be a subset of image keys")
    if any(value < 0 or value >= length for value in image):
        raise ValueError("image key position is outside the sequence")
    query_start, query_end = query_range_for_partial_row(
        prefix_length=prefix_length, partial_row_length=partial_row_length
    )
    if query_end >= length:
        raise ValueError("partial-row query range is outside the sequence")
    mask = torch.tril(torch.ones((length, length), dtype=torch.bool, device=device))
    blocked = image - eligible
    if blocked:
        blocked_tensor = torch.tensor(sorted(blocked), dtype=torch.long, device=device)
        for query_index in range(query_start, query_end + 1):
            mask[query_index, blocked_tensor] = False
    return mask.unsqueeze(0).unsqueeze(0).contiguous()


def inspect_partial_row_mask_structure(
    mask: torch.Tensor, *, sequence_length: int, image_key_positions: Sequence[int],
    eligible_image_positions: Sequence[int], prefix_length: int, partial_row_length: int,
) -> dict[str, Any]:
    """Attest exact dynamic mask cells and unchanged pre-row/non-image cells."""

    length = int(sequence_length)
    if tuple(mask.shape) != (1, 1, length, length) or mask.dtype is not torch.bool:
        raise ValueError("partial-row mask must be boolean [1,1,S,S]")
    baseline = torch.tril(torch.ones((length, length), dtype=torch.bool, device=mask.device))
    observed = mask[0, 0]
    image = {int(value) for value in image_key_positions}
    eligible = {int(value) for value in eligible_image_positions}
    query_start, query_end = query_range_for_partial_row(
        prefix_length=prefix_length, partial_row_length=partial_row_length
    )
    expected = baseline.clone()
    blocked = image - eligible
    if blocked:
        blocked_tensor = torch.tensor(sorted(blocked), dtype=torch.long, device=mask.device)
        for query_index in range(query_start, query_end + 1):
            expected[query_index, blocked_tensor] = False
    changed = observed != baseline
    off_scope = changed.clone()
    off_scope[query_start : query_end + 1, :] = False
    non_image = changed.clone()
    if image:
        image_tensor = torch.tensor(sorted(image), dtype=torch.long, device=mask.device)
        non_image[:, image_tensor] = False
    blocked_cells = 0
    if blocked:
        blocked_tensor = torch.tensor(sorted(blocked), dtype=torch.long, device=mask.device)
        blocked_cells = int(changed[:, blocked_tensor].sum().item())
    eligible_unchanged = True
    if eligible:
        eligible_tensor = torch.tensor(sorted(eligible), dtype=torch.long, device=mask.device)
        eligible_unchanged = bool(torch.equal(observed[:, eligible_tensor], baseline[:, eligible_tensor]))
    return {
        "passed": bool(
            query_start >= 0
            and query_end < length
            and bool(query_end >= query_start)
            and bool(blocked)
            and blocked_cells > 0
            and int(non_image.sum().item()) == 0
            and int(off_scope.sum().item()) == 0
            and bool(torch.equal(observed, expected))
            and eligible_unchanged
        ),
        "query_range": {
            "start_inclusive": query_start,
            "end_inclusive": query_end,
            "indices": list(range(query_start, query_end + 1)),
        },
        "partial_row_length": int(partial_row_length),
        "blocked_image_key_count": len(blocked),
        "blocked_image_cell_count": blocked_cells,
        "changed_cell_count": int(changed.sum().item()),
        "changed_non_image_key_cell_count": int(non_image.sum().item()),
        "changed_off_scope_query_cell_count": int(off_scope.sum().item()),
        "exact_expected_mask_match": bool(torch.equal(observed, expected)),
        "eligible_image_cells_unchanged": bool(eligible_unchanged),
        "future_key_blocking_unchanged": bool(
            torch.equal(torch.triu(observed, diagonal=1), torch.triu(baseline, diagonal=1))
        ),
        "mask_shape": list(mask.shape),
        "mask_dtype": str(mask.dtype),
    }


def build_dynamic_mask_for_ids(
    *, ids: torch.Tensor, image_token_id: int, selected_indices: Sequence[int], prefix_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Build and attest the partial-row mask for the current materialized ids."""

    if ids.ndim != 2 or ids.shape[0] != 1:
        raise ValueError("ids must have shape [1,S]")
    image_positions = [int(v) for v in torch.where(ids[0].to(device=device) == int(image_token_id))[0].tolist()]
    if not image_positions:
        raise ValueError("current prefix contains no image placeholder keys")
    if selected_indices and max(int(v) for v in selected_indices) >= len(image_positions):
        raise ValueError("selected image index exceeds image-token count")
    partial_length = int(ids.shape[1]) - int(prefix_length)
    mask = build_partial_row_query_only_key_eligibility_mask(
        sequence_length=int(ids.shape[1]), image_key_positions=image_positions,
        eligible_image_positions=[image_positions[int(v)] for v in selected_indices],
        prefix_length=prefix_length, partial_row_length=partial_length, device=device,
    )
    receipt = inspect_partial_row_mask_structure(
        mask, sequence_length=int(ids.shape[1]), image_key_positions=image_positions,
        eligible_image_positions=[image_positions[int(v)] for v in selected_indices],
        prefix_length=prefix_length, partial_row_length=partial_length,
    )
    receipt["image_token_positions"] = image_positions
    receipt["eligible_image_token_positions"] = [image_positions[int(v)] for v in selected_indices]
    if not receipt["passed"]:
        raise RuntimeError("dynamic partial-row mask failed structural contract")
    return mask, receipt


def parse_geometry_suffix(
    generated: Sequence[int], *, tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Parse the exact ``x1 y1 x2 y2 BOX_END`` suffix without repair."""

    values = [int(value) for value in generated]
    result: dict[str, Any] = {
        "valid": False,
        "reason": None,
        "generated_token_ids": values,
        "coordinate_token_ids": [],
        "coordinate_bins": [],
        "normalized_box": None,
        "decoded_suffix": None,
    }
    if tokenizer is not None and callable(getattr(tokenizer, "decode", None)):
        try:
            result["decoded_suffix"] = str(tokenizer.decode(values, skip_special_tokens=False))
        except TypeError:
            result["decoded_suffix"] = str(tokenizer.decode(values))
    if len(values) < 5:
        result["reason"] = "truncated_geometry_suffix"
        return result
    if len(values) > 5:
        result["reason"] = "trailing_token_after_geometry_suffix"
        return result
    coordinates = values[:4]
    if any(
        token < COORDINATE_TOKEN_START or token >= COORDINATE_TOKEN_START + 1000
        for token in coordinates
    ):
        result["reason"] = "coordinate_token_out_of_range"
        return result
    if values[4] != BOX_END:
        result["reason"] = "box_end_not_final"
        return result
    bins = [token - COORDINATE_TOKEN_START for token in coordinates]
    result.update(
        {
            "valid": True,
            "coordinate_token_ids": coordinates,
            "coordinate_bins": bins,
            "normalized_box": [float(value) / 999.0 for value in bins],
        }
    )
    return result


def _box_iou(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != 4 or len(right) != 4:
        return 0.0
    lx1, ly1, lx2, ly2 = [float(value) for value in left]
    rx1, ry1, rx2, ry2 = [float(value) for value in right]
    ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    intersection = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
    left_area = max(lx2 - lx1, 0.0) * max(ly2 - ly1, 0.0)
    right_area = max(rx2 - rx1, 0.0) * max(ry2 - ry1, 0.0)
    union = left_area + right_area - intersection
    return 0.0 if union <= 0.0 else float(intersection / union)


def geometry_ownership(
    parsed: Mapping[str, Any], *, donor_box_normalized: Sequence[float],
    paired_box_normalized: Sequence[float], iou_floor: float = GEOMETRY_IOU_FLOOR,
) -> dict[str, Any]:
    """Require strict donor-relative distance and an absolute donor IoU floor."""

    box = parsed.get("normalized_box")
    if not parsed.get("valid") or not isinstance(box, Sequence) or len(box) != 4:
        return {
            "passed": False,
            "valid_natural_close": False,
            "donor_l1_distance": None,
            "paired_l1_distance": None,
            "donor_iou": None,
            "paired_iou": None,
            "strictly_closer_to_donor": False,
            "iou_floor": float(iou_floor),
        }
    observed = [float(value) for value in box]
    donor = [float(value) for value in donor_box_normalized]
    paired = [float(value) for value in paired_box_normalized]
    donor_distance = float(sum(abs(left - right) for left, right in zip(observed, donor, strict=True)))
    paired_distance = float(sum(abs(left - right) for left, right in zip(observed, paired, strict=True)))
    donor_iou = _box_iou(observed, donor)
    paired_iou = _box_iou(observed, paired)
    closer = donor_distance < paired_distance
    return {
        "passed": bool(closer and donor_iou >= float(iou_floor)),
        "valid_natural_close": True,
        "donor_l1_distance": donor_distance,
        "paired_l1_distance": paired_distance,
        "donor_iou": donor_iou,
        "paired_iou": paired_iou,
        "strictly_closer_to_donor": bool(closer),
        "iou_floor": float(iou_floor),
    }


def assess_geometry_donor_eligibility(
    *, coordinate_release: float | None, valid_path: bool, owner_match: bool,
) -> dict[str, Any]:
    threshold = ELIGIBILITY_RELEASE_FLOOR
    passed = bool(
        coordinate_release is not None
        and math.isfinite(float(coordinate_release))
        and float(coordinate_release) >= threshold
        and valid_path
        and owner_match
    )
    return {
        "passed": passed,
        "coordinate_release": None if coordinate_release is None else float(coordinate_release),
        "coordinate_release_floor": threshold,
        "valid_path": bool(valid_path),
        "owner_match": bool(owner_match),
        "box_end_excluded": True,
    }


def assess_geometry_portability(
    *, persistent_release: float, replacement_release: float,
    no_op_drift: float, valid_path: bool, owner_match: bool,
) -> dict[str, Any]:
    half_threshold = 0.5 * max(float(persistent_release), 0.0)
    effect_threshold = 10.0 * float(no_op_drift)
    passed = bool(
        float(replacement_release) >= PORTABILITY_RELEASE_FLOOR
        and float(replacement_release) >= half_threshold
        and float(replacement_release) >= effect_threshold
        and valid_path
        and owner_match
    )
    return {
        "passed": passed,
        "persistent_coordinate_release": float(persistent_release),
        "replacement_coordinate_release": float(replacement_release),
        "absolute_release_floor": PORTABILITY_RELEASE_FLOOR,
        "half_positive_persistent_release_floor": half_threshold,
        "ten_times_no_op_drift_floor": effect_threshold,
        "valid_path": bool(valid_path),
        "owner_match": bool(owner_match),
        "box_end_excluded": True,
    }


def assess_self_noop_trust(
    *, max_abs_logprob_drift: float, teacher_forced_top_token_ids_equal: bool,
    generated_token_ids_equal: bool, replacement_count: int,
    tolerance: float = TOLERANCE,
) -> dict[str, Any]:
    """Require exact discrete paths in addition to bounded float32 drift."""

    passed = bool(
        float(max_abs_logprob_drift) <= float(tolerance)
        and teacher_forced_top_token_ids_equal
        and generated_token_ids_equal
        and int(replacement_count) == 1
    )
    return {
        "passed": passed,
        "max_abs_logprob_drift": float(max_abs_logprob_drift),
        "tolerance": float(tolerance),
        "teacher_forced_top_token_ids_equal": bool(teacher_forced_top_token_ids_equal),
        "generated_token_ids_equal": bool(generated_token_ids_equal),
        "replacement_count": int(replacement_count),
    }


def classify_geometry_panel(
    *, trust_passed: bool, eligible_count: int,
    layer23_passed_donors: Sequence[str], layer13_passed_donors: Sequence[str],
) -> dict[str, Any]:
    """Apply layer-13 veto only to the corresponding layer-23 donor."""

    positive = {str(value) for value in layer23_passed_donors}
    negative = {str(value) for value in layer13_passed_donors}
    vetoed = sorted(positive & negative)
    if not trust_passed:
        classification = "invalid_execution_trust_gate"
        interpreted = False
    elif vetoed:
        classification = "vetoed_by_negative_control_layer_13"
        interpreted = False
    elif int(eligible_count) == 0:
        classification = "no_eligible_donor_control_only"
        interpreted = False
    elif positive:
        classification = "promote_bounded_one_sided_geometry_portability"
        interpreted = True
    else:
        classification = "close_one_site_conditional_downstream_portability"
        interpreted = True
    return {
        "classification": classification,
        "interpreted": interpreted,
        "positive_layer": 23,
        "negative_control_layer": 13,
        "eligible_donor_count": int(eligible_count),
        "layer23_passed_donors": sorted(positive),
        "layer13_passed_donors": sorted(negative),
        "corresponding_donor_veto_intersection": vetoed,
    }


def coordinate_mean(score: Mapping[str, Any]) -> float:
    return float(sum(float(score[phase]["mean"]) for phase in COORDINATE_PHASES) / 4.0)


def box_end_log_probability(score: Mapping[str, Any]) -> float:
    values = score.get("token_log_probabilities", [])
    if not values:
        raise ValueError("score lacks token_log_probabilities")
    return float(values[-1])


def _shared_row_prefix(row: Sequence[int]) -> list[int]:
    values = [int(value) for value in row]
    if len(values) != 9 or values[:4] != [OBJECT_REF_START, 2190, OBJECT_REF_END, BOX_START]:
        raise ValueError("image-632 row does not match the frozen shared pre-x1 history")
    return values[:4]


def _row_from_geometry_suffix(shared_prefix: Sequence[int], generated: Sequence[int]) -> list[int]:
    parsed = parse_geometry_suffix(generated)
    if not parsed["valid"]:
        raise ValueError("cannot build a realized row from an invalid geometry suffix")
    return [*map(int, shared_prefix), *map(int, generated)]


def _normalized_box_from_pixel_box(
    box: Sequence[float], *, image_width: int, image_height: int,
) -> list[float]:
    if len(box) != 4 or image_width <= 0 or image_height <= 0:
        raise ValueError("pixel box and image dimensions are invalid")
    return [
        float(box[0]) / float(image_width),
        float(box[1]) / float(image_height),
        float(box[2]) / float(image_width),
        float(box[3]) / float(image_height),
    ]


def _full_row_query_mask(
    *, ids: torch.Tensor, image_token_id: int, selected_indices: Sequence[int],
    prefix_length: int, row_length: int, device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    image_positions = [
        int(value)
        for value in torch.where(ids[0].to(device=device) == int(image_token_id))[0].tolist()
    ]
    if not image_positions:
        raise ValueError("row contains no image placeholder keys")
    if max((int(value) for value in selected_indices), default=-1) >= len(image_positions):
        raise ValueError("selected image-token index exceeds materialized image-token count")
    eligible = [image_positions[int(value)] for value in selected_indices]
    mask = query.build_query_scoped_key_eligibility_mask(
        sequence_length=int(ids.shape[1]),
        image_key_positions=image_positions,
        eligible_image_positions=eligible,
        prefix_length=int(prefix_length),
        row_length=int(row_length),
        device=device,
    )
    receipt = query.inspect_query_scoped_mask_structure(
        mask,
        sequence_length=int(ids.shape[1]),
        image_key_positions=image_positions,
        eligible_image_positions=eligible,
        prefix_length=int(prefix_length),
        row_length=int(row_length),
    )
    receipt["image_token_positions"] = image_positions
    receipt["eligible_image_token_positions"] = eligible
    if not receipt.get("passed"):
        raise RuntimeError("complete-row query-only mask failed structural contract")
    return mask, receipt


def _score_row(
    *, model: Any, model_inputs: Mapping[str, Any], features: Any,
    grid_thw: Sequence[int], merge_size: int, prompt_ids: Sequence[int], row: Sequence[int],
    image_grid_thw: torch.Tensor, image_token_id: int, layer_idx: int,
    selected_indices: Sequence[int] | None = None,
    capture: Any | None = None, replacement: Any | None = None,
    resolved_module: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    device = next(model.parameters()).device
    ids = torch.tensor([[*map(int, prompt_ids), *map(int, row)]], dtype=torch.long, device=device)
    position_ids = query.derive_explicit_position_ids(
        model,
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        image_grid_thw=image_grid_thw.to(device=device),
    )
    custom_mask = None
    structural = None
    if selected_indices is not None:
        custom_mask, structural = _full_row_query_mask(
            ids=ids,
            image_token_id=image_token_id,
            selected_indices=selected_indices,
            prefix_length=len(prompt_ids),
            row_length=len(row),
            device=device,
        )
    boundary_pos = len(prompt_ids) + len(_shared_row_prefix(row)) - 1
    logits = residual._score_with_hooks(
        model=model,
        model_inputs=model_inputs,
        features=features,
        grid_thw=grid_thw,
        merge_size=merge_size,
        ids=ids,
        image_grid_thw=image_grid_thw,
        position_ids=position_ids,
        custom_mask=custom_mask,
        layer_idx=int(layer_idx),
        boundary_pos=int(boundary_pos),
        capture=capture,
        replacement=replacement,
        resolved_module=resolved_module,
    )
    score = query.score_row_log_likelihoods(
        logits,
        prefix_length=len(prompt_ids),
        row_tokens=row,
        description_length=1,
        terminal_token_id=None,
    )
    score["coordinate_only_mean"] = coordinate_mean(score)
    score["box_end_log_probability"] = box_end_log_probability(score)
    score["position_ids_sha256"] = sha256_tensor(position_ids)
    return score, structural


def greedy_query_scoped_geometry(
    *, model: Any, model_inputs: Mapping[str, Any], features: Any,
    grid_thw: Sequence[int], merge_size: int, prefix_ids: torch.Tensor,
    prompt_length: int, image_grid_thw: torch.Tensor, image_token_id: int,
    selected_indices: Sequence[int] | None, tokenizer: Any | None,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> dict[str, Any]:
    """Generate one geometry suffix by full-prefix recomputation per token."""

    if prefix_ids.ndim != 2 or tuple(prefix_ids.shape[:1]) != (1,):
        raise ValueError("prefix_ids must have shape [1,S]")
    if int(prefix_ids.shape[1]) - int(prompt_length) != 4:
        raise ValueError("generation must begin at the frozen pre-x1 BOX_START boundary")
    device = next(model.parameters()).device
    current = prefix_ids.detach().clone().to(device=device, dtype=torch.long)
    generated: list[int] = []
    selected_log_probs: list[float] = []
    selected_ranks: list[int] = []
    top_prediction_token_ids: list[int] = []
    structural_receipts: list[dict[str, Any]] = []
    position_hashes: list[str] = []
    stop_reason = "max_new_tokens"
    for _ in range(int(max_new_tokens)):
        position_ids = query.derive_explicit_position_ids(
            model,
            input_ids=current,
            attention_mask=torch.ones_like(current),
            image_grid_thw=image_grid_thw.to(device=device),
        )
        position_hashes.append(sha256_tensor(position_ids))
        custom_mask = None
        if selected_indices is not None:
            custom_mask, structural = build_dynamic_mask_for_ids(
                ids=current,
                image_token_id=image_token_id,
                selected_indices=selected_indices,
                prefix_length=int(prompt_length),
                device=device,
            )
            structural_receipts.append(structural)
        logits = residual._score_with_hooks(
            model=model,
            model_inputs=model_inputs,
            features=features,
            grid_thw=grid_thw,
            merge_size=merge_size,
            ids=current,
            image_grid_thw=image_grid_thw,
            position_ids=position_ids,
            custom_mask=custom_mask,
            layer_idx=23,
            boundary_pos=int(prefix_ids.shape[1] - 1),
        )
        next_logits = logits[-1].to(dtype=torch.float32)
        token = int(torch.argmax(next_logits).item())
        log_probs = torch.log_softmax(next_logits, dim=-1)
        selected_log_probs.append(float(log_probs[token].item()))
        selected_ranks.append(int(1 + (next_logits > next_logits[token]).sum().item()))
        top_prediction_token_ids.append(token)
        generated.append(token)
        current = torch.cat(
            (current, torch.tensor([[token]], dtype=torch.long, device=device)), dim=1
        )
        if token == BOX_END:
            stop_reason = "box_end"
            break
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is not None and token == int(eos_token_id):
            stop_reason = "eos"
            break
    parsed = parse_geometry_suffix(generated, tokenizer=tokenizer)
    structural_passed = bool(
        selected_indices is None
        or (structural_receipts and all(item.get("passed") for item in structural_receipts))
    )
    return {
        "generated_token_ids": generated,
        "generated_token_count": len(generated),
        "selected_token_log_probabilities": selected_log_probs,
        "selected_token_ranks": selected_ranks,
        "top_prediction_token_ids": top_prediction_token_ids,
        "stop_reason": stop_reason,
        "natural_closure": bool(stop_reason == "box_end" and parsed.get("valid")),
        "valid": bool(parsed.get("valid")),
        "parsed": parsed,
        "cache_used": False,
        "full_prefix_recomputed_each_step": True,
        "dynamic_mask_rebuilt_each_step": selected_indices is not None,
        "structural_mask_receipts": structural_receipts,
        "structural_mask_gate_passed": structural_passed,
        "position_ids_sha256": position_hashes,
        "repetition_penalty": 1.0,
        "logits_processor": None,
    }


def _validate_parent_receipt(path: Path) -> tuple[dict[str, Any], str, Mapping[str, Any]]:
    receipt, digest = residual.validate_parent_receipt(
        path,
        expected_sha256=ROW_QUERY_PARENT_SHA256,
        expected_unit_id=ROW_QUERY_PARENT_UNIT_ID,
    )
    result = next(
        (item for item in receipt.get("results", []) if str(item.get("image_id")) == IMAGE_ID),
        None,
    )
    if not isinstance(result, Mapping):
        raise ValueError("row-query parent receipt lacks frozen image 632")
    exact = {
        "target_annotation_id": TARGET_ANNOTATION_ID,
        "competitor_annotation_id": COMPETITOR_ANNOTATION_ID,
        "target_row_token_ids": TARGET_ROW_TOKEN_IDS,
        "competitor_row_token_ids": COMPETITOR_ROW_TOKEN_IDS,
        "target_mask_indices": TARGET_MASK_INDICES,
        "competitor_mask_indices": COMPETITOR_MASK_INDICES,
        "canonical_rows_sha256": CANONICAL_ROWS_SHA256,
    }
    for key, expected in exact.items():
        if result.get(key) != expected:
            raise ValueError(f"frozen image-632 parent field {key!r} drifted")
    required_arms = {"target_row_query_only_hard", "competitor_row_query_only_hard", "all_allowed_4d"}
    if not required_arms.issubset(result.get("arms", {})):
        raise ValueError("row-query parent receipt lacks exact mapped image-632 arms")
    return receipt, digest, result


def _run_persistent_paths(
    *, model: Any, model_inputs: Mapping[str, Any], features: Any,
    grid_thw: Sequence[int], merge_size: int, prompt_ids: Sequence[int],
    owner_rows: Mapping[str, Sequence[int]], owner_masks: Mapping[str, Sequence[int]],
    owner_boxes: Mapping[str, Sequence[float]], image_grid_thw: torch.Tensor,
    image_token_id: int, tokenizer: Any | None, max_new_tokens: int,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    shared = _shared_row_prefix(owner_rows["target"])
    if shared != _shared_row_prefix(owner_rows["competitor"]):
        raise ValueError("image-632 donor histories differ before x1")
    prefix_ids = torch.tensor([[*map(int, prompt_ids), *shared]], dtype=torch.long, device=device)
    output: dict[str, Any] = {"owners": {}, "eligible_count": 0}
    for owner_name in ("target", "competitor"):
        paired_name = "competitor" if owner_name == "target" else "target"
        hard_generated = greedy_query_scoped_geometry(
            model=model,
            model_inputs=model_inputs,
            features=features,
            grid_thw=grid_thw,
            merge_size=merge_size,
            prefix_ids=prefix_ids,
            prompt_length=len(prompt_ids),
            image_grid_thw=image_grid_thw,
            image_token_id=image_token_id,
            selected_indices=owner_masks[owner_name],
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
        )
        ownership = geometry_ownership(
            hard_generated["parsed"],
            donor_box_normalized=owner_boxes[owner_name],
            paired_box_normalized=owner_boxes[paired_name],
        )
        hard_generated["geometry_ownership"] = ownership
        coordinate_release: float | None = None
        realized_scores: dict[str, Any] | None = None
        realized_row: list[int] | None = None
        if hard_generated["natural_closure"]:
            realized_row = _row_from_geometry_suffix(shared, hard_generated["generated_token_ids"])
            hard_score, structural = _score_row(
                model=model,
                model_inputs=model_inputs,
                features=features,
                grid_thw=grid_thw,
                merge_size=merge_size,
                prompt_ids=prompt_ids,
                row=realized_row,
                image_grid_thw=image_grid_thw,
                image_token_id=image_token_id,
                layer_idx=23,
                selected_indices=owner_masks[owner_name],
            )
            unrestricted_score, _ = _score_row(
                model=model,
                model_inputs=model_inputs,
                features=features,
                grid_thw=grid_thw,
                merge_size=merge_size,
                prompt_ids=prompt_ids,
                row=realized_row,
                image_grid_thw=image_grid_thw,
                image_token_id=image_token_id,
                layer_idx=23,
            )
            coordinate_release = float(
                hard_score["coordinate_only_mean"] - unrestricted_score["coordinate_only_mean"]
            )
            realized_scores = {
                "row_token_ids": realized_row,
                "hard": hard_score,
                "unrestricted": unrestricted_score,
                "coordinate_release": coordinate_release,
                "box_end_release": float(
                    hard_score["box_end_log_probability"]
                    - unrestricted_score["box_end_log_probability"]
                ),
                "box_end_excluded_from_eligibility": True,
                "structural_mask_receipt": structural,
            }
        eligibility = assess_geometry_donor_eligibility(
            coordinate_release=coordinate_release,
            valid_path=bool(hard_generated["natural_closure"]),
            owner_match=bool(ownership["passed"]),
        )
        output["owners"][owner_name] = {
            "hard_generated_path": hard_generated,
            "realized_row_token_ids": realized_row,
            "realized_row_scores": realized_scores,
            "geometry_ownership": ownership,
            "eligibility": eligibility,
        }
    output["eligible_count"] = sum(
        bool(value["eligibility"]["passed"]) for value in output["owners"].values()
    )
    return output


def _run_layer_boundary(
    *, model: Any, model_inputs: Mapping[str, Any], features: Any,
    grid_thw: Sequence[int], merge_size: int, prompt_ids: Sequence[int],
    owner_rows: Mapping[str, Sequence[int]], owner_masks: Mapping[str, Sequence[int]],
    owner_boxes: Mapping[str, Sequence[float]], persistent: Mapping[str, Any],
    image_grid_thw: torch.Tensor, image_token_id: int, tokenizer: Any | None,
    layer_idx: int, max_new_tokens: int,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    shared = _shared_row_prefix(owner_rows["target"])
    prefix_ids = torch.tensor([[*map(int, prompt_ids), *shared]], dtype=torch.long, device=device)
    boundary_pos = int(prefix_ids.shape[1] - 1)
    position_ids = query.derive_explicit_position_ids(
        model,
        input_ids=prefix_ids,
        attention_mask=torch.ones_like(prefix_ids),
        image_grid_thw=image_grid_thw.to(device=device),
    )
    module, resolution = residual.resolve_decoder_layer(model, int(layer_idx))
    layer_receipt = residual.build_decoder_layer_resolution_receipt(resolution)

    donor_states: dict[str, torch.Tensor] = {}
    donor_receipts: dict[str, Any] = {}
    for owner_name in ("target", "competitor"):
        hard_mask, structural = build_dynamic_mask_for_ids(
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
            position_ids=position_ids,
            custom_mask=hard_mask,
            layer_idx=int(layer_idx),
            boundary_pos=boundary_pos,
            capture=capture,
            resolved_module=module,
        )
        if capture.state is None or capture.capture_count != 1:
            raise RuntimeError("hard donor residual state capture did not complete exactly once")
        donor_states[owner_name] = capture.state
        donor_receipts[owner_name] = {
            "capture_count": capture.capture_count,
            "state_sha256": sha256_tensor(capture.state),
            "mask_indices": [int(value) for value in owner_masks[owner_name]],
            "structural_mask_receipt": structural,
            "token_history_sha256": sha256_tensor(prefix_ids[0]),
            "token_history_equal_to_recipient_through_boundary": True,
        }

    self_capture = residual.ResidualStateCapture(module, boundary_pos=boundary_pos)
    unrestricted_prefill = residual._score_with_hooks(
        model=model,
        model_inputs=model_inputs,
        features=features,
        grid_thw=grid_thw,
        merge_size=merge_size,
        ids=prefix_ids,
        image_grid_thw=image_grid_thw,
        position_ids=position_ids,
        custom_mask=None,
        layer_idx=int(layer_idx),
        boundary_pos=boundary_pos,
        capture=self_capture,
        use_cache=True,
        return_output=True,
        resolved_module=module,
    )
    if self_capture.state is None or self_capture.capture_count != 1:
        raise RuntimeError("unrestricted self-state capture did not complete exactly once")
    unrestricted_cached = residual.greedy_cached_one_row_continuation(
        model,
        prefill_output=unrestricted_prefill,
        input_ids=prefix_ids,
        prefill_position_ids=position_ids,
        box_end_token_id=BOX_END,
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
        max_new_tokens=max_new_tokens,
    )
    unrestricted_cached["parsed"] = parse_geometry_suffix(
        unrestricted_cached.get("generated_token_ids", []), tokenizer=tokenizer
    )
    unrestricted_cached["natural_closure"] = bool(
        unrestricted_cached.get("stop_reason") == "box_end"
        and unrestricted_cached["parsed"].get("valid")
    )

    self_replacement = residual.ResidualStateReplacement(
        module, boundary_pos=boundary_pos, replacement=self_capture.state
    )
    self_prefill = residual._score_with_hooks(
        model=model,
        model_inputs=model_inputs,
        features=features,
        grid_thw=grid_thw,
        merge_size=merge_size,
        ids=prefix_ids,
        image_grid_thw=image_grid_thw,
        position_ids=position_ids,
        custom_mask=None,
        layer_idx=int(layer_idx),
        boundary_pos=boundary_pos,
        replacement=self_replacement,
        use_cache=True,
        return_output=True,
        resolved_module=module,
    )
    self_cached = residual.greedy_cached_one_row_continuation(
        model,
        prefill_output=self_prefill,
        input_ids=prefix_ids,
        prefill_position_ids=position_ids,
        box_end_token_id=BOX_END,
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
        max_new_tokens=max_new_tokens,
    )
    self_cached["parsed"] = parse_geometry_suffix(
        self_cached.get("generated_token_ids", []), tokenizer=tokenizer
    )
    self_cached["natural_closure"] = bool(
        self_cached.get("stop_reason") == "box_end" and self_cached["parsed"].get("valid")
    )

    noop_drifts: list[float] = []
    no_op_rows: dict[str, Any] = {}
    donor_results: dict[str, Any] = {}
    for owner_name in ("target", "competitor"):
        paired_name = "competitor" if owner_name == "target" else "target"
        persistent_owner = persistent["owners"][owner_name]
        row = persistent_owner.get("realized_row_token_ids") or [int(value) for value in owner_rows[owner_name]]
        baseline_score, _ = _score_row(
            model=model,
            model_inputs=model_inputs,
            features=features,
            grid_thw=grid_thw,
            merge_size=merge_size,
            prompt_ids=prompt_ids,
            row=row,
            image_grid_thw=image_grid_thw,
            image_token_id=image_token_id,
            layer_idx=int(layer_idx),
            resolved_module=module,
        )
        noop_hook = residual.ResidualStateReplacement(
            module, boundary_pos=boundary_pos, replacement=self_capture.state
        )
        noop_score, _ = _score_row(
            model=model,
            model_inputs=model_inputs,
            features=features,
            grid_thw=grid_thw,
            merge_size=merge_size,
            prompt_ids=prompt_ids,
            row=row,
            image_grid_thw=image_grid_thw,
            image_token_id=image_token_id,
            layer_idx=int(layer_idx),
            replacement=noop_hook,
            resolved_module=module,
        )
        drift = residual.max_abs_delta(
            baseline_score["token_log_probabilities"], noop_score["token_log_probabilities"]
        )
        noop_drifts.append(float(drift))
        no_op_rows[owner_name] = {
            "row_token_ids": [int(value) for value in row],
            "unrestricted": baseline_score,
            "self_state_noop": noop_score,
            "max_abs_logprob_drift": float(drift),
            "replacement_count": noop_hook.replacement_count,
        }

        replacement_hook = residual.ResidualStateReplacement(
            module, boundary_pos=boundary_pos, replacement=donor_states[owner_name]
        )
        replacement_score, _ = _score_row(
            model=model,
            model_inputs=model_inputs,
            features=features,
            grid_thw=grid_thw,
            merge_size=merge_size,
            prompt_ids=prompt_ids,
            row=row,
            image_grid_thw=image_grid_thw,
            image_token_id=image_token_id,
            layer_idx=int(layer_idx),
            replacement=replacement_hook,
            resolved_module=module,
        )
        replacement_release = float(
            replacement_score["coordinate_only_mean"] - baseline_score["coordinate_only_mean"]
        )
        replacement_box_end_release = float(
            replacement_score["box_end_log_probability"] - baseline_score["box_end_log_probability"]
        )

        cached_hook = residual.ResidualStateReplacement(
            module, boundary_pos=boundary_pos, replacement=donor_states[owner_name]
        )
        replacement_prefill = residual._score_with_hooks(
            model=model,
            model_inputs=model_inputs,
            features=features,
            grid_thw=grid_thw,
            merge_size=merge_size,
            ids=prefix_ids,
            image_grid_thw=image_grid_thw,
            position_ids=position_ids,
            custom_mask=None,
            layer_idx=int(layer_idx),
            boundary_pos=boundary_pos,
            replacement=cached_hook,
            use_cache=True,
            return_output=True,
            resolved_module=module,
        )
        cached = residual.greedy_cached_one_row_continuation(
            model,
            prefill_output=replacement_prefill,
            input_ids=prefix_ids,
            prefill_position_ids=position_ids,
            box_end_token_id=BOX_END,
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
            max_new_tokens=max_new_tokens,
        )
        cached["parsed"] = parse_geometry_suffix(cached.get("generated_token_ids", []), tokenizer=tokenizer)
        cached["natural_closure"] = bool(
            cached.get("stop_reason") == "box_end" and cached["parsed"].get("valid")
        )
        cached["geometry_ownership"] = geometry_ownership(
            cached["parsed"],
            donor_box_normalized=owner_boxes[owner_name],
            paired_box_normalized=owner_boxes[paired_name],
        )
        cached["hook_removed_after_prefill"] = cached_hook.hook_removed_inside_hook
        donor_results[owner_name] = {
            "row_token_ids": [int(value) for value in row],
            "unrestricted": baseline_score,
            "replacement": replacement_score,
            "coordinate_release": replacement_release,
            "box_end_release": replacement_box_end_release,
            "box_end_excluded_from_portability": True,
            "replacement_count": replacement_hook.replacement_count,
            "cached_path": cached,
        }

    no_op_drift = max(noop_drifts, default=0.0)
    cached_equal = bool(
        unrestricted_cached.get("generated_token_ids") == self_cached.get("generated_token_ids")
    )
    teacher_forced_top_ids_equal = all(
        value["unrestricted"].get("top_prediction_token_ids")
        == value["self_state_noop"].get("top_prediction_token_ids")
        for value in no_op_rows.values()
    )
    self_noop_trust = assess_self_noop_trust(
        max_abs_logprob_drift=no_op_drift,
        teacher_forced_top_token_ids_equal=teacher_forced_top_ids_equal,
        generated_token_ids_equal=cached_equal,
        replacement_count=self_replacement.replacement_count,
    )
    portability: dict[str, Any] = {}
    for owner_name in ("target", "competitor"):
        eligibility = persistent["owners"][owner_name]["eligibility"]
        if eligibility.get("passed"):
            cached = donor_results[owner_name]["cached_path"]
            portability[owner_name] = assess_geometry_portability(
                persistent_release=float(eligibility["coordinate_release"]),
                replacement_release=float(donor_results[owner_name]["coordinate_release"]),
                no_op_drift=no_op_drift,
                valid_path=bool(cached.get("natural_closure")),
                owner_match=bool(cached.get("geometry_ownership", {}).get("passed")),
            )
        else:
            portability[owner_name] = {
                "passed": False,
                "reason": "persistent_hard_donor_ineligible",
                "donor_eligibility": eligibility,
            }
    return {
        "layer_idx": int(layer_idx),
        "role": ROLE,
        "absolute_boundary_pos": boundary_pos,
        "decoder_layer_resolution": layer_receipt,
        "position_attestation": {
            "position_ids_sha256": sha256_tensor(position_ids),
            "position_ids_shape": list(position_ids.shape),
            "token_history_sha256": sha256_tensor(prefix_ids[0]),
        },
        "donor_cache_discarded": True,
        "recipient_mask": "unrestricted",
        "teacher_forced_use_cache": False,
        "recipient_use_cache": True,
        "donor_states": donor_receipts,
        "no_op_rows": no_op_rows,
        "cached_paths": {
            "unrestricted": unrestricted_cached,
            "self_state_noop": self_cached,
        },
        "donor_results": donor_results,
        "portability_gates": portability,
        "self_noop_max_abs_logprob_drift": no_op_drift,
        "self_noop_teacher_forced_top_token_ids_equal": teacher_forced_top_ids_equal,
        "self_noop_generated_token_ids_equal": cached_equal,
        "self_noop_trust_gate": self_noop_trust,
        "self_noop_passed": bool(self_noop_trust["passed"]),
        "replacement_contract": {
            "batch_index": 0,
            "returned_full_block_output": True,
            "donor_cache_discarded": True,
            "recipient_cache_only": True,
            "hooks_removed_before_later_calls": all(
                bool(value["cached_path"].get("hook_removed_after_prefill"))
                for value in donor_results.values()
            ),
            "non_boundary_max_abs_delta": {"target": 0.0, "competitor": 0.0},
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infer-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-jsonl", type=Path, default=DEFAULT_SOURCE_JSONL)
    parser.add_argument("--audit-ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--row-query-receipt", type=Path, default=DEFAULT_ROW_QUERY_RECEIPT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-id", default=IMAGE_ID)
    parser.add_argument("--layers", nargs="+", type=int, default=list(LAYERS))
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if str(args.image_id) != IMAGE_ID:
        raise SystemExit("geometry discriminator is frozen to image 632")
    if [int(value) for value in args.layers] != list(LAYERS):
        raise SystemExit("geometry discriminator is frozen to layers 23 and 13")
    if int(args.max_new_tokens) < 5:
        raise SystemExit("max-new-tokens must allow four coordinates and BOX_END")

    _, parent_sha, parent_result = _validate_parent_receipt(
        Path(args.row_query_receipt).expanduser().resolve(strict=True)
    )

    from src.config.inference import load_infer_config
    from src.data import load_raw_examples
    from src.inference.image_plan import materialize_image_plan_batch
    from src.inference.pipeline import _processor_config, _template_config
    from src.inference.prompt import build_prompt_record
    from src.inference.runtime import assemble_runtime
    from scripts.research.run_sampled_rescue_transition import _temporary_cwd

    config_path = Path(args.infer_config).expanduser().resolve(strict=True)
    with _temporary_cwd(config_path.parents[3]):
        resolved = load_infer_config(config_path)
    config = resolved.config.model_copy(
        update={"model": resolved.config.model.model_copy(update={"dtype": "fp32"})}
    )
    runtime = assemble_runtime(config, source_gate_root=config_path.parents[3])
    qwen = runtime.qwen
    qwen.model.eval()
    actual_parameter_dtypes = sorted(
        {str(parameter.dtype) for parameter in qwen.model.parameters()}
    )
    if actual_parameter_dtypes != ["torch.float32"]:
        raise RuntimeError(f"float32 execution contract failed: {actual_parameter_dtypes!r}")
    attention_implementation = str(
        getattr(qwen.model.config, "_attn_implementation", None)
        or getattr(qwen.model.config, "attn_implementation", None)
        or "unknown"
    )
    if attention_implementation != "sdpa":
        raise RuntimeError(
            f"scaled-dot-product-attention execution contract failed: {attention_implementation!r}"
        )

    raw_rows = load_raw_examples(Path(args.source_jsonl).expanduser().resolve(strict=True))
    raw = next(
        row
        for row in raw_rows
        if str(row.metadata.get("source", {}).get("image_id")) == IMAGE_ID
    )
    objects = {str(obj.object_id): obj for obj in raw.objects}
    target = objects[TARGET_ANNOTATION_ID]
    competitor = objects[COMPETITOR_ANNOTATION_ID]
    template = _template_config(resolved.config)
    prompt_record = build_prompt_record(raw, template, processor=qwen.processor, row_index=0)
    prompt_ids = [int(value) for value in prompt_record.prompt_token_ids]
    owner_rows = {
        "target": query._row_token_ids(qwen.tokenizer, target),
        "competitor": query._row_token_ids(qwen.tokenizer, competitor),
    }
    if owner_rows["target"] != TARGET_ROW_TOKEN_IDS:
        raise ValueError("live target canonical row differs from frozen image-632 row")
    if owner_rows["competitor"] != COMPETITOR_ROW_TOKEN_IDS:
        raise ValueError("live competitor canonical row differs from frozen image-632 row")

    plan = materialize_image_plan_batch(
        [raw],
        components=qwen,
        processor_config=_processor_config(resolved.config),
        materialize=True,
        row_indices=[0],
    )
    model_inputs = plan.model_inputs_by_row_id[raw.example_id]
    grid_thw = [
        int(value)
        for value in model_inputs["image_grid_thw"].reshape(-1, 3)[0].tolist()
    ]
    if grid_thw != [1, 54, 72]:
        raise ValueError(f"image-632 grid drifted: {grid_thw!r}")
    features = query.capture_feature_bundle(qwen.model, model_inputs)
    merge_size = int(qwen.processor_identity.merge_size)
    if merge_size != 2:
        raise ValueError("image-632 merge_size drifted from 2")
    image_token_id = int(
        getattr(
            qwen.model.config,
            "image_token_id",
            qwen.tokenizer.convert_tokens_to_ids("<|image_pad|>"),
        )
    )
    owner_masks = {
        "target": list(TARGET_MASK_INDICES),
        "competitor": list(COMPETITOR_MASK_INDICES),
    }

    ledger = query._load_ledger(Path(args.audit_ledger).expanduser().resolve(strict=True))
    ledger_by_id = {
        str(item.get("object_identifier", "")): item for item in ledger.get(IMAGE_ID, [])
    }
    pixel_boxes = {
        "target": query._object_pixel_box(
            target, ledger_by_id, width=raw.image.width, height=raw.image.height
        ),
        "competitor": query._object_pixel_box(
            competitor, ledger_by_id, width=raw.image.width, height=raw.image.height
        ),
    }
    if pixel_boxes["target"] is None or pixel_boxes["competitor"] is None:
        raise ValueError("image-632 owner boxes are unavailable")
    owner_boxes = {
        name: _normalized_box_from_pixel_box(
            box, image_width=raw.image.width, image_height=raw.image.height
        )
        for name, box in pixel_boxes.items()
    }

    canonical_live_hard: dict[str, Any] = {}
    canonical_live_unrestricted: dict[str, Any] = {}
    canonical_structural: dict[str, Any] = {}
    for owner_name in ("target", "competitor"):
        canonical_live_hard[owner_name], canonical_structural[owner_name] = _score_row(
            model=qwen.model,
            model_inputs=model_inputs,
            features=features,
            grid_thw=grid_thw,
            merge_size=merge_size,
            prompt_ids=prompt_ids,
            row=owner_rows[owner_name],
            image_grid_thw=model_inputs["image_grid_thw"],
            image_token_id=image_token_id,
            layer_idx=23,
            selected_indices=owner_masks[owner_name],
        )
        canonical_live_unrestricted[owner_name], _ = _score_row(
            model=qwen.model,
            model_inputs=model_inputs,
            features=features,
            grid_thw=grid_thw,
            merge_size=merge_size,
            prompt_ids=prompt_ids,
            row=owner_rows[owner_name],
            image_grid_thw=model_inputs["image_grid_thw"],
            image_token_id=image_token_id,
            layer_idx=23,
        )
    canonical_frozen_hard = {
        "target": parent_result["arms"]["target_row_query_only_hard"]["target"],
        "competitor": parent_result["arms"]["competitor_row_query_only_hard"]["competitor"],
    }
    canonical_frozen_unrestricted = {
        "target": parent_result["arms"]["all_allowed_4d"]["target"],
        "competitor": parent_result["arms"]["all_allowed_4d"]["competitor"],
    }
    parent_hard_reproduction = residual.assess_parent_reproduction(
        live=canonical_live_hard, frozen=canonical_frozen_hard
    )
    parent_unrestricted_reproduction = residual.assess_parent_reproduction(
        live=canonical_live_unrestricted, frozen=canonical_frozen_unrestricted
    )
    canonical_structural_passed = all(
        bool(value and value.get("passed")) for value in canonical_structural.values()
    )

    persistent = _run_persistent_paths(
        model=qwen.model,
        model_inputs=model_inputs,
        features=features,
        grid_thw=grid_thw,
        merge_size=merge_size,
        prompt_ids=prompt_ids,
        owner_rows=owner_rows,
        owner_masks=owner_masks,
        owner_boxes=owner_boxes,
        image_grid_thw=model_inputs["image_grid_thw"],
        image_token_id=image_token_id,
        tokenizer=qwen.tokenizer,
        max_new_tokens=int(args.max_new_tokens),
    )

    results: list[dict[str, Any]] = []
    persistent_structural_passed = all(
        bool(value["hard_generated_path"].get("structural_mask_gate_passed"))
        for value in persistent["owners"].values()
    )
    for layer_idx in args.layers:
        boundary = _run_layer_boundary(
            model=qwen.model,
            model_inputs=model_inputs,
            features=features,
            grid_thw=grid_thw,
            merge_size=merge_size,
            prompt_ids=prompt_ids,
            owner_rows=owner_rows,
            owner_masks=owner_masks,
            owner_boxes=owner_boxes,
            persistent=persistent,
            image_grid_thw=model_inputs["image_grid_thw"],
            image_token_id=image_token_id,
            tokenizer=qwen.tokenizer,
            layer_idx=int(layer_idx),
            max_new_tokens=int(args.max_new_tokens),
        )
        donor_structural_passed = all(
            bool(value.get("structural_mask_receipt", {}).get("passed"))
            for value in boundary["donor_states"].values()
        )
        trust_passed = bool(
            parent_hard_reproduction["passed"]
            and parent_unrestricted_reproduction["passed"]
            and canonical_structural_passed
            and persistent_structural_passed
            and donor_structural_passed
            and boundary["self_noop_passed"]
        )
        boundary["scientific_gate"] = {
            "passed": trust_passed,
            "parent_hard_reproduction_passed": parent_hard_reproduction["passed"],
            "parent_unrestricted_reproduction_passed": parent_unrestricted_reproduction["passed"],
            "canonical_structural_mask_gate_passed": canonical_structural_passed,
            "persistent_dynamic_structural_mask_gate_passed": persistent_structural_passed,
            "donor_structural_mask_gate_passed": donor_structural_passed,
            "self_state_noop_passed": boundary["self_noop_passed"],
            "layer_is_positive_seam": int(layer_idx) == 23,
        }
        boundary["portability_passed_donors"] = sorted(
            owner
            for owner, value in boundary["portability_gates"].items()
            if trust_passed and value.get("passed")
        )
        boundary["portability_passed_any_donor"] = bool(
            boundary["portability_passed_donors"]
        )
        results.append(boundary)

    layer23 = next(value for value in results if int(value["layer_idx"]) == 23)
    layer13 = next(value for value in results if int(value["layer_idx"]) == 13)
    trust_passed = bool(
        layer23["scientific_gate"]["passed"] and layer13["scientific_gate"]["passed"]
    )
    panel_decision = classify_geometry_panel(
        trust_passed=trust_passed,
        eligible_count=int(persistent["eligible_count"]),
        layer23_passed_donors=layer23["portability_passed_donors"],
        layer13_passed_donors=layer13["portability_passed_donors"],
    )

    return {
        "schema_version": "fixed_encoding_downstream_geometry_state_portability.v1",
        "unit_id": UNIT_ID,
        "model_dtype": "torch.float32",
        "request_identity": build_request_identity(),
        "parent_receipt": {
            "path": str(Path(args.row_query_receipt).expanduser().resolve()),
            "sha256": parent_sha,
            "mapped_arms": {
                "target": "target_row_query_only_hard",
                "competitor": "competitor_row_query_only_hard",
                "unrestricted": "all_allowed_4d",
            },
        },
        "frozen_contract": {
            "target_row_token_ids": TARGET_ROW_TOKEN_IDS,
            "competitor_row_token_ids": COMPETITOR_ROW_TOKEN_IDS,
            "canonical_rows_sha256": CANONICAL_ROWS_SHA256,
            "target_mask_indices": TARGET_MASK_INDICES,
            "competitor_mask_indices": COMPETITOR_MASK_INDICES,
            "image_grid_thw": grid_thw,
            "merge_size": merge_size,
            "image_width": int(raw.image.width),
            "image_height": int(raw.image.height),
            "pixel_boxes": pixel_boxes,
            "normalized_boxes": owner_boxes,
        },
        "canonical_parent_reproduction": {
            "hard": parent_hard_reproduction,
            "unrestricted": parent_unrestricted_reproduction,
            "structural_mask_receipts": canonical_structural,
            "structural_mask_gate_passed": canonical_structural_passed,
        },
        "persistent_hard_paths": persistent,
        "results": results,
        "runtime_contract": {
            "recipient_mask": "unrestricted",
            "donor_mask": "exact_row_query_only_dynamic_partial_row",
            "donor_cache_discarded": True,
            "recipient_use_cache": True,
            "teacher_forced_use_cache": False,
            "logits_dtype": "torch.float32",
            "repetition_penalty": 1.0,
            "logits_processor": None,
            "max_new_tokens": int(args.max_new_tokens),
            "actual_model_parameter_dtype": str(next(qwen.model.parameters()).dtype),
            "actual_model_parameter_dtypes": actual_parameter_dtypes,
            "explicit_mrope_prefill_shape": "[3,1,S]",
            "explicit_mrope_next_token_shape": "[3,1,1]",
            "prepare_inputs_for_generation_bypassed": True,
            "rope_deltas_not_passed": True,
            "attention_implementation": attention_implementation,
        },
        "panel_decision": panel_decision,
    }


def main(argv: Sequence[str] | None = None) -> int:
    normalized = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    payload = run(args)
    payload["runner_sha256"] = sha256_file(Path(__file__))
    payload["normalized_argv"] = normalized
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "receipt.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
