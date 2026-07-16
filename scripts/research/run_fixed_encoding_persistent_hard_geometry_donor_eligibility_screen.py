#!/usr/bin/env python3
"""Run the frozen persistent hard-routing geometry-donor eligibility screen.

This is intentionally an experiment-local runner.  It encodes one image once,
restricts only row-scoring queries to one of three equal-area visual supports,
and greedily emits the exact four-coordinate plus ``BOX_END`` suffix.  The
module also exposes small pure helpers used by the focused tests and by a
deterministic receipt merger; it does not modify shared inference code.
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
import scripts.research.run_fixed_encoding_downstream_residual_state_portability as residual  # noqa: E402
import scripts.research.run_fixed_encoding_query_scoped_object_centered_spatial_eligibility as query  # noqa: E402
from src.analysis.visual_support_counterfactual import (  # noqa: E402
    capture_feature_bundle,
    validate_feature_layout,
)


UNIT_ID = "2026-07-15-fixed-encoding-persistent-hard-routing-geometry-donor-eligibility-screen"
SCHEMA_VERSION = "fixed-encoding-persistent-hard-routing-geometry-donor-eligibility-screen.v1"
DEFAULT_CONFIG = Path(
    "/data/CoordExp/.worktrees/research-probes/configs/coordexp_swift/infer/"
    "qwen3_vl_2b_desc_first_geo_sorted_gaussian_rps_dora_r16a32_step4887_val200.yaml"
)
DEFAULT_SOURCE_JSONL = Path(
    "/data/CoordExp/.worktrees/CoordExp-swift/outputs/coordexp_swift/infer/"
    "val200_inputs/coco_val200_len12000.rebased_images.coord.jsonl"
)
DEFAULT_LEDGER = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-13-spatial-scope-history-disentanglement/readiness-v2/"
    "audit-augmented-ledger.jsonl"
)
DEFAULT_COHORT = Path(__file__).resolve().parents[2] / "research/investigations/qwen3-vl-dense-enumeration/experiments/" / UNIT_ID / "cohort.json"
DEFAULT_OUTPUT_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-15-fixed-encoding-persistent-hard-routing-geometry-donor-eligibility-screen"
)

COORDINATE_TOKEN_START = residual.COORDINATE_TOKEN_START
OBJECT_REF_START = residual.OBJECT_REF_START
OBJECT_REF_END = residual.OBJECT_REF_END
BOX_START = residual.BOX_START
BOX_END = residual.BOX_END
COORDINATE_PHASES = ("x1", "y1", "x2", "y2")
EXPECTED_IMAGE_IDS = ("7818", "12576", "2157", "13923", "632", "12639")
EXPECTED_COHORT_SHA256 = "1f7d44bf8291f7b3e708372f97ed51ac85eebef3b97fa3d83456adbe7f1060bf"
LAYERS = (23, 13)
MAX_NEW_TOKENS = 5
TOLERANCE = 1e-4
ELIGIBILITY_RELEASE_FLOOR = -0.05
GEOMETRY_IOU_FLOOR = 0.30
COMPETING_IOU_MARGIN = 0.15
UNRELATED_IOU_MARGIN = 0.15
SUPPORT_IOU_IMPROVEMENT_MARGIN = 0.05
SUPPORT_L1_IMPROVEMENT_RATIO = 0.75


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


def load_cohort(path: Path = DEFAULT_COHORT) -> dict[str, Any]:
    """Load and validate the frozen six-case cohort."""

    resolved = Path(path).expanduser().resolve(strict=True)
    if sha256_file(resolved) != EXPECTED_COHORT_SHA256:
        raise ValueError("frozen cohort SHA-256 does not match the research-unit contract")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    validate_cohort(payload)
    return payload


def validate_cohort(cohort: Mapping[str, Any]) -> None:
    """Fail fast on frozen case identity, grid, and equal-area mask contracts."""

    if cohort.get("unit_id") != UNIT_ID:
        raise ValueError("cohort unit_id mismatch")
    cases = cohort.get("cases")
    if not isinstance(cases, Sequence) or isinstance(cases, (str, bytes)):
        raise ValueError("cohort cases are missing")
    ids = [str(item.get("image_id")) for item in cases if isinstance(item, Mapping)]
    if ids != list(EXPECTED_IMAGE_IDS) or len(ids) != len(set(ids)):
        raise ValueError(f"cohort image order must be {EXPECTED_IMAGE_IDS!r}")
    for case in cases:
        if not isinstance(case, Mapping):
            raise ValueError("cohort case is not an object")
        h = int(case["image_grid_thw"][1]) // int(case["merge_size"])
        w = int(case["image_grid_thw"][2]) // int(case["merge_size"])
        rects = [
            case["target_support_rectangle"],
            case["paired_support_rectangle"],
            case["unrelated_location_support_rectangle"],
        ]
        masks = [rectangle_indices(rect, merged_height=h, merged_width=w) for rect in rects]
        counts = [len(mask) for mask in masks]
        if len(set(counts)) != 1 or counts[0] <= 0:
            raise ValueError(f"case {case.get('image_id')} supports are not equal-area")
        if set(masks[0]) & set(masks[1]):
            raise ValueError(f"case {case.get('image_id')} target and paired supports overlap")
        if set(masks[2]) & (set(masks[0]) | set(masks[1])):
            raise ValueError(f"case {case.get('image_id')} unrelated support overlaps owners")


def validate_live_input_hashes(*, cohort: Mapping[str, Any], config_path: Path, source_jsonl_path: Path, audit_ledger_path: Path) -> dict[str, str]:
    """Verify the three immutable input files before constructing a model."""

    paths = {"config_sha256": config_path, "source_jsonl_sha256": source_jsonl_path, "audit_ledger_sha256": audit_ledger_path}
    expected = {key: str(cohort["config" if key == "config_sha256" else "source_jsonl" if key == "source_jsonl_sha256" else "audit_ledger"]["sha256"]) for key in paths}
    observed = {key: sha256_file(path) for key, path in paths.items()}
    for key in paths:
        if observed[key] != expected[key]:
            raise ValueError(f"{key} does not match the frozen cohort")
    return observed


def rectangle_indices(
    rectangle: Sequence[int], *, merged_height: int, merged_width: int
) -> list[int]:
    """Materialize a half-open merged-grid rectangle into row-major indices."""

    if len(rectangle) != 4:
        raise ValueError("support rectangle must contain four bounds")
    r0, r1, c0, c1 = [int(value) for value in rectangle]
    if not (0 <= r0 < r1 <= int(merged_height) and 0 <= c0 < c1 <= int(merged_width)):
        raise ValueError("support rectangle lies outside merged grid or is empty")
    return [row * int(merged_width) + col for row in range(r0, r1) for col in range(c0, c1)]


def support_envelope_box(
    rectangle: Sequence[int], *, merged_height: int, merged_width: int
) -> list[float]:
    """Return a merged-grid rectangle in normalized ``xyxy`` coordinates."""

    r0, r1, c0, c1 = [int(value) for value in rectangle]
    return [c0 / merged_width, r0 / merged_height, c1 / merged_width, r1 / merged_height]


def row_geometry_prefix(row: Sequence[int]) -> list[int]:
    """Return the generic multi-token row prefix ending at ``BOX_START``."""

    values = [int(value) for value in row]
    if len(values) < 8 or values[0] != OBJECT_REF_START:
        raise ValueError("row is too short or lacks OBJECT_REF_START")
    try:
        object_end = values.index(OBJECT_REF_END)
    except ValueError as exc:
        raise ValueError("row lacks OBJECT_REF_END") from exc
    if object_end <= 1 or object_end + 1 >= len(values) or values[object_end + 1] != BOX_START:
        raise ValueError("row wrapper is malformed")
    if len(values) - (object_end + 2) != 5:
        raise ValueError("row must end in exactly four coordinates and BOX_END")
    if values[-1] != BOX_END:
        raise ValueError("row must end with BOX_END")
    return values[: object_end + 2]


def parse_geometry_suffix(
    generated: Sequence[int], *, tokenizer: Any | None = None
) -> dict[str, Any]:
    """Parse exactly ``x1 y1 x2 y2 BOX_END`` without repair."""

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
    coords = values[:4]
    if any(token < COORDINATE_TOKEN_START or token >= COORDINATE_TOKEN_START + 1000 for token in coords):
        result["reason"] = "coordinate_token_out_of_range"
        return result
    if values[-1] != BOX_END:
        result["reason"] = "box_end_not_final"
        return result
    bins = [token - COORDINATE_TOKEN_START for token in coords]
    result.update(
        {
            "valid": True,
            "coordinate_token_ids": coords,
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
    ix1, iy1, ix2, iy2 = max(lx1, rx1), max(ly1, ry1), min(lx2, rx2), min(ly2, ry2)
    inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
    union = max(lx2 - lx1, 0.0) * max(ly2 - ly1, 0.0) + max(rx2 - rx1, 0.0) * max(ry2 - ry1, 0.0) - inter
    return float(inter / union) if union > 0 else 0.0


def _l1(left: Sequence[float], right: Sequence[float]) -> float:
    return float(sum(abs(float(a) - float(b)) for a, b in zip(left, right, strict=True)))


def geometry_attribution(
    parsed: Mapping[str, Any], *, donor_box: Sequence[float], paired_box: Sequence[float],
    accepted_objects: Sequence[Mapping[str, Any]], support_box: Sequence[float],
    donor_annotation_id: str | None = None,
) -> dict[str, Any]:
    """Attribute one generated box to donor, pair, all ledger objects, and support."""

    box = parsed.get("normalized_box")
    if not parsed.get("valid") or not isinstance(box, Sequence) or len(box) != 4:
        return {"valid": False, "donor_iou": None, "paired_iou": None, "support_iou": None, "all_object_ious": {}}
    observed = [float(value) for value in box]
    all_object_ious = {
        str(item["annotation_id"]): _box_iou(observed, item["normalized_box"])
        for item in accepted_objects
        if isinstance(item, Mapping) and "annotation_id" in item and isinstance(item.get("normalized_box"), Sequence)
    }
    donor_id = str(donor_annotation_id) if donor_annotation_id is not None else str(
        next((item["annotation_id"] for item in accepted_objects if item.get("normalized_box") == list(donor_box)), "donor")
    )
    paired_iou = _box_iou(observed, paired_box)
    donor_iou = _box_iou(observed, donor_box)
    support_donor_iou = _box_iou(support_box, donor_box)
    support_donor_l1 = _l1(support_box, donor_box)
    competing = {key: value for key, value in all_object_ious.items() if key != donor_id}
    strongest_competitor = max(competing.items(), key=lambda item: item[1], default=(None, 0.0))
    return {
        "valid": True,
        "donor_annotation_id": donor_id,
        "generated_box": observed,
        "donor_iou": donor_iou,
        "paired_iou": paired_iou,
        "donor_l1_distance": _l1(observed, donor_box),
        "paired_l1_distance": _l1(observed, paired_box),
        "strictly_closer_to_donor": _l1(observed, donor_box) < _l1(observed, paired_box),
        "support_envelope_iou": _box_iou(observed, support_box),
        "support_envelope_donor_iou": support_donor_iou,
        "support_envelope_donor_l1_distance": support_donor_l1,
        "donor_iou_improves_over_support": donor_iou >= support_donor_iou + SUPPORT_IOU_IMPROVEMENT_MARGIN,
        "donor_l1_improves_over_support": bool(
            _l1(observed, donor_box) <= SUPPORT_L1_IMPROVEMENT_RATIO * support_donor_l1
            if support_donor_l1 > 0.0
            else _l1(observed, donor_box) <= 0.0
        ),
        "all_object_ious": all_object_ious,
        "strongest_competing_annotation_id": strongest_competitor[0],
        "strongest_competing_iou": float(strongest_competitor[1]),
    }


def assess_donor_eligibility(
    *, valid_path: bool, natural_closure: bool, coordinate_release: float | None,
    attribution: Mapping[str, Any], unrelated_donor_iou: float | None = None,
) -> dict[str, Any]:
    """Apply the frozen donor gates, including competing-object specificity."""

    donor_iou = attribution.get("donor_iou")
    competing_iou = float(attribution.get("strongest_competing_iou") or 0.0)
    paired_iou = float(attribution.get("paired_iou") or 0.0)
    closer = bool(attribution.get("strictly_closer_to_donor"))
    release_ok = coordinate_release is not None and math.isfinite(float(coordinate_release)) and float(coordinate_release) >= ELIGIBILITY_RELEASE_FLOOR
    specificity_iou = unrelated_donor_iou
    checks = {
        "valid_path": bool(valid_path),
        "natural_closure": bool(natural_closure),
        "strictly_closer_to_donor": closer,
        "donor_iou_floor": donor_iou is not None and float(donor_iou) >= GEOMETRY_IOU_FLOOR,
        "improves_over_support_envelope_iou": bool(attribution.get("donor_iou_improves_over_support")),
        "improves_over_support_envelope_l1": bool(attribution.get("donor_l1_improves_over_support")),
        "competing_object_margin": donor_iou is not None and float(donor_iou) - competing_iou >= COMPETING_IOU_MARGIN,
        "coordinate_release_floor": bool(release_ok),
        "unrelated_location_margin": specificity_iou is not None and donor_iou is not None and float(donor_iou) - float(specificity_iou) >= UNRELATED_IOU_MARGIN,
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "coordinate_release": None if coordinate_release is None else float(coordinate_release),
        "coordinate_release_floor": ELIGIBILITY_RELEASE_FLOOR,
        "donor_iou": None if donor_iou is None else float(donor_iou),
        "paired_iou": paired_iou,
        "strongest_competing_iou": competing_iou,
        "unrelated_donor_iou": None if specificity_iou is None else float(specificity_iou),
        "geometry_iou_floor": GEOMETRY_IOU_FLOOR,
        "competing_object_margin": COMPETING_IOU_MARGIN,
        "unrelated_location_margin": UNRELATED_IOU_MARGIN,
        "support_iou_improvement_margin": SUPPORT_IOU_IMPROVEMENT_MARGIN,
        "support_l1_improvement_ratio": SUPPORT_L1_IMPROVEMENT_RATIO,
    }


def classify_panel(results: Sequence[Mapping[str, Any]], *, cohort: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Make the deterministic clean-vs-stress decision for a complete panel."""

    expected = [str(case["image_id"]) for case in (cohort or {}).get("cases", [])] or list(EXPECTED_IMAGE_IDS)
    ids = [str(item.get("image_id")) for item in results]
    duplicates = sorted({image_id for image_id in ids if ids.count(image_id) > 1})
    missing = sorted(set(expected) - set(ids), key=expected.index)
    unexpected = sorted(set(ids) - set(expected))
    if duplicates or missing or unexpected:
        return {"classification": "incomplete_panel", "missing_image_ids": missing, "duplicate_image_ids": duplicates, "unexpected_image_ids": unexpected, "requested_case_count": len(results)}
    by_id = {str(item["image_id"]): item for item in results}
    trust_failures = [image_id for image_id, item in by_id.items() if not isinstance(item.get("trust_gate"), Mapping) or item["trust_gate"].get("passed") is not True]
    clean_ids = [str(case["image_id"]) for case in (cohort or {}).get("cases", []) if str(case.get("stratum", "")).endswith("clean")] or list(expected[:4])
    stress_ids = [image_id for image_id in expected if image_id not in clean_ids]
    eligible_clean = [image_id for image_id in clean_ids if any(bool(arm.get("eligibility", {}).get("passed")) for arm in by_id[image_id].get("arms", {}).values())]
    eligible_stress = [image_id for image_id in stress_ids if any(bool(arm.get("eligibility", {}).get("passed")) for arm in by_id[image_id].get("arms", {}).values())]
    if trust_failures:
        classification = "invalid_execution_trust_gate"
    elif eligible_clean:
        classification = "clean_geometry_donor_found_requires_portability_successor"
    else:
        classification = "no_resolution_qualified_geometry_donor_in_curated_panel"
    return {"classification": classification, "interpreted": not bool(trust_failures), "clean_image_ids": clean_ids, "stress_image_ids": stress_ids, "eligible_clean_image_ids": eligible_clean, "eligible_stress_image_ids": eligible_stress, "trust_failed_image_ids": sorted(trust_failures), "requested_case_count": len(results)}


def merge_case_receipts(
    receipts: Sequence[Mapping[str, Any]], *, cohort: Mapping[str, Any], config_sha256: str,
    source_jsonl_sha256: str, audit_ledger_sha256: str, cohort_sha256: str,
) -> dict[str, Any]:
    """Merge exactly one receipt per frozen image and reject identity drift."""

    validate_cohort(cohort)
    expected = [str(case["image_id"]) for case in cohort["cases"]]
    expanded: list[dict[str, Any]] = []
    for receipt in receipts:
        if receipt.get("image_id") is None and isinstance(receipt.get("results"), Sequence):
            if receipt.get("unit_id") != UNIT_ID:
                raise ValueError("split wrapper unit_id mismatch")
            for item in receipt["results"]:
                if not isinstance(item, Mapping):
                    raise ValueError("split wrapper contains a non-object case")
                enriched = dict(item)
                for key in ("unit_id", "config_sha256", "source_jsonl_sha256", "audit_ledger_sha256", "cohort_sha256"):
                    if key not in enriched and key in receipt:
                        enriched[key] = receipt[key]
                expanded.append(enriched)
        else:
            expanded.append(dict(receipt))
    observed = [str(receipt.get("image_id")) for receipt in expanded]
    if len(observed) != len(set(observed)):
        raise ValueError("merged receipts contain duplicate image IDs")
    if set(observed) != set(expected):
        raise ValueError(f"merged receipts must contain exactly {expected!r}")
    for receipt in expanded:
        if receipt.get("unit_id") != UNIT_ID:
            raise ValueError("case receipt unit_id mismatch")
        if not isinstance(receipt.get("trust_gate"), Mapping) or receipt["trust_gate"].get("passed") is not True:
            raise ValueError("case receipt lacks a passed trust gate")
        for key, expected_value in (("config_sha256", config_sha256), ("source_jsonl_sha256", source_jsonl_sha256), ("audit_ledger_sha256", audit_ledger_sha256), ("cohort_sha256", cohort_sha256)):
            if str(receipt.get(key)) != str(expected_value):
                raise ValueError(f"case receipt {key} mismatch")
    ordered = sorted(expanded, key=lambda item: expected.index(str(item["image_id"])))
    merged = {
        "schema_version": SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "config_sha256": config_sha256,
        "source_jsonl_sha256": source_jsonl_sha256,
        "audit_ledger_sha256": audit_ledger_sha256,
        "cohort_sha256": cohort_sha256,
        "results": ordered,
    }
    merged["panel_decision"] = classify_panel(ordered, cohort=cohort)
    return merged


def _as_normalized_box(box: Sequence[float], *, width: int, height: int) -> list[float]:
    return [float(box[0]) / width, float(box[1]) / height, float(box[2]) / width, float(box[3]) / height]


def _row_from_suffix(prefix: Sequence[int], generated: Sequence[int]) -> list[int]:
    parsed = parse_geometry_suffix(generated)
    if not parsed["valid"]:
        raise ValueError("cannot materialize row from invalid suffix")
    return [*map(int, prefix), *map(int, generated)]


def _coordinate_mean(score: Mapping[str, Any]) -> float:
    return float(sum(float(score[phase]["mean"]) for phase in COORDINATE_PHASES) / 4.0)


def _score_row(*, model: Any, model_inputs: Mapping[str, Any], features: Any, grid_thw: Sequence[int], merge_size: int, prompt_ids: Sequence[int], row: Sequence[int], image_grid_thw: torch.Tensor, image_token_id: int, selected_indices: Sequence[int] | None, position_mode: str = "explicit") -> tuple[dict[str, Any], dict[str, Any] | None]:
    device = next(model.parameters()).device
    ids = torch.tensor([[*map(int, prompt_ids), *map(int, row)]], dtype=torch.long, device=device)
    if position_mode not in {"implicit", "explicit", "all_allowed"}:
        raise ValueError(f"unknown position_mode {position_mode!r}")
    position_ids = None if position_mode == "implicit" else query.derive_explicit_position_ids(model, input_ids=ids, attention_mask=torch.ones_like(ids), image_grid_thw=image_grid_thw.to(device=device))
    custom = None
    structural = None
    if selected_indices is not None:
        image_positions = [int(value) for value in torch.where(ids[0] == int(image_token_id))[0].tolist()]
        eligible = [image_positions[int(value)] for value in selected_indices]
        custom = query.build_query_scoped_key_eligibility_mask(sequence_length=ids.shape[1], image_key_positions=image_positions, eligible_image_positions=eligible, prefix_length=len(prompt_ids), row_length=len(row), device=device)
        structural = query.inspect_query_scoped_mask_structure(custom, sequence_length=ids.shape[1], image_key_positions=image_positions, eligible_image_positions=eligible, prefix_length=len(prompt_ids), row_length=len(row))
        if not structural["passed"]:
            raise RuntimeError("static row mask failed structural gate")
    elif position_mode == "all_allowed":
        image_positions = [int(value) for value in torch.where(ids[0] == int(image_token_id))[0].tolist()]
        custom = query.build_causal_key_eligibility_mask(sequence_length=ids.shape[1], image_key_positions=image_positions, eligible_image_positions=image_positions, device=device)
    if position_ids is None:
        logits = query._score_model_sequence(model=model, model_inputs=model_inputs, input_ids=ids, attention_mask=custom if custom is not None else torch.ones_like(ids), position_ids=None, custom_mask=custom, features=features, merge_size=merge_size, grid_thw=grid_thw)
    else:
        logits = residual._score_with_hooks(model=model, model_inputs=model_inputs, features=features, grid_thw=grid_thw, merge_size=merge_size, ids=ids, image_grid_thw=image_grid_thw, position_ids=position_ids, custom_mask=custom, layer_idx=23, boundary_pos=ids.shape[1] - 1)
    score = query.score_row_log_likelihoods(logits, prefix_length=len(prompt_ids), row_tokens=row, description_length=len(row) - 8, terminal_token_id=None)
    score["coordinate_only_mean"] = _coordinate_mean(score)
    return score, structural


def greedy_geometry_arm(*, model: Any, model_inputs: Mapping[str, Any], features: Any, grid_thw: Sequence[int], merge_size: int, prompt_ids: Sequence[int], row_prefix: Sequence[int], image_grid_thw: torch.Tensor, image_token_id: int, selected_indices: Sequence[int] | None, tokenizer: Any | None, max_new_tokens: int = MAX_NEW_TOKENS) -> dict[str, Any]:
    """Greedily generate the fixed geometry suffix with no cache or processor."""

    if int(max_new_tokens) != 5:
        raise ValueError("screen requires exactly five generated tokens")
    device = next(model.parameters()).device
    current = torch.tensor([[*map(int, prompt_ids), *map(int, row_prefix)]], dtype=torch.long, device=device)
    generated: list[int] = []
    selected_log_probs: list[float] = []
    selected_ranks: list[int] = []
    top_ids: list[int] = []
    structures: list[dict[str, Any]] = []
    for _ in range(5):
        position_ids = query.derive_explicit_position_ids(model, input_ids=current, attention_mask=torch.ones_like(current), image_grid_thw=image_grid_thw.to(device=device))
        custom = None
        if selected_indices is not None:
            custom, receipt = geometry.build_dynamic_mask_for_ids(ids=current, image_token_id=image_token_id, selected_indices=selected_indices, prefix_length=len(prompt_ids), device=device)
            structures.append(receipt)
            if not receipt["passed"]:
                raise RuntimeError("dynamic row mask failed structural gate")
        logits = residual._score_with_hooks(model=model, model_inputs=model_inputs, features=features, grid_thw=grid_thw, merge_size=merge_size, ids=current, image_grid_thw=image_grid_thw, position_ids=position_ids, custom_mask=custom, layer_idx=23, boundary_pos=current.shape[1] - 1)
        next_logits = logits[-1].to(dtype=torch.float32)
        log_probs = torch.log_softmax(next_logits, dim=-1)
        token = int(torch.argmax(next_logits).item())
        generated.append(token)
        selected_log_probs.append(float(log_probs[token].item()))
        selected_ranks.append(int(1 + (next_logits > next_logits[token]).sum().item()))
        top_ids.append(token)
        current = torch.cat((current, torch.tensor([[token]], dtype=torch.long, device=device)), dim=1)
    parsed = parse_geometry_suffix(generated, tokenizer=tokenizer)
    return {"generated_token_ids": generated, "generated_token_count": len(generated), "selected_token_log_probabilities": selected_log_probs, "selected_token_ranks": selected_ranks, "top_prediction_token_ids": top_ids, "stop_reason": "box_end" if generated[-1] == BOX_END else "max_new_tokens", "natural_closure": bool(parsed["valid"] and generated[-1] == BOX_END), "valid": bool(parsed["valid"]), "parsed": parsed, "cache_used": False, "full_prefix_recomputed_each_step": True, "repetition_penalty": 1.0, "logits_processor": None, "structural_mask_receipts": structures, "structural_mask_gate_passed": all(item["passed"] for item in structures) if selected_indices is not None else True}


def _canonical_noop_gate(*, scores: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    implicit = scores["implicit"]
    explicit = scores["explicit"]
    allowed = scores["all_allowed"]
    vectors = [implicit["token_log_probabilities"], explicit["token_log_probabilities"], allowed["token_log_probabilities"]]
    drift = max(max(abs(float(a) - float(b)) for a, b in zip(vectors[i], vectors[j], strict=True)) for i in range(3) for j in range(i + 1, 3))
    rank_equal = all(scores[name]["selected_token_ranks"] == implicit["selected_token_ranks"] for name in ("explicit", "all_allowed"))
    top_equal = all(scores[name]["top_prediction_token_ids"] == implicit["top_prediction_token_ids"] for name in ("explicit", "all_allowed"))
    return {"passed": bool(drift <= TOLERANCE and rank_equal and top_equal), "max_abs_logprob_drift": float(drift), "selected_token_ranks_equal": rank_equal, "top_prediction_token_ids_equal": top_equal, "tolerance": TOLERANCE}


def _run_case(args: argparse.Namespace, *, qwen: Any, raw: Any, case: Mapping[str, Any], ledger: Mapping[str, Any], resolved: Any) -> dict[str, Any]:
    from src.inference.image_plan import materialize_image_plan_batch
    from src.inference.pipeline import _processor_config, _template_config
    from src.inference.prompt import build_prompt_record

    image_id = str(case["image_id"])
    if int(raw.image.width) != int(case["image_width"]) or int(raw.image.height) != int(case["image_height"]):
        raise ValueError(f"image {image_id}: raw image dimensions drifted")
    objects = {str(obj.object_id): obj for obj in raw.objects}
    target = objects[str(case["target_annotation_id"])]
    paired = objects[str(case["paired_annotation_id"])]
    if str(target.description) != str(case["description"]) or str(paired.description) != str(case["description"]):
        raise ValueError(f"image {image_id}: target/paired descriptions drifted")
    if [int(value) for value in target.bbox] != [int(value) for value in case["target_coordinate_bins"]] or [int(value) for value in paired.bbox] != [int(value) for value in case["paired_coordinate_bins"]]:
        raise ValueError(f"image {image_id}: target/paired coordinate bins drifted")
    template = _template_config(resolved.config)
    prompt_record = build_prompt_record(raw, template, processor=qwen.processor, row_index=0)
    prompt_ids = [int(value) for value in prompt_record.prompt_token_ids]
    target_row = query._row_token_ids(qwen.tokenizer, target)
    paired_row = query._row_token_ids(qwen.tokenizer, paired)
    target_prefix = row_geometry_prefix(target_row)
    paired_prefix = row_geometry_prefix(paired_row)
    if target_prefix != paired_prefix:
        raise ValueError(f"image {image_id}: target and paired row prefixes differ")
    plan = materialize_image_plan_batch([raw], components=qwen, processor_config=_processor_config(resolved.config), materialize=True, row_indices=[0])
    model_inputs = plan.model_inputs_by_row_id[raw.example_id]
    grid_thw = [int(value) for value in model_inputs["image_grid_thw"].reshape(-1, 3)[0].tolist()]
    if grid_thw != [int(value) for value in case["image_grid_thw"]]:
        raise ValueError(f"image {image_id}: processor grid drift")
    merge_size = int(qwen.processor_identity.merge_size)
    if merge_size != int(case["merge_size"]):
        raise ValueError(f"image {image_id}: merge_size drift")
    features = capture_feature_bundle(qwen.model, model_inputs)
    layout = validate_feature_layout(features, features, grid_thw=grid_thw, merge_size=merge_size)
    h, w = layout.merged_height, layout.merged_width
    target_indices = rectangle_indices(case["target_support_rectangle"], merged_height=h, merged_width=w)
    paired_indices = rectangle_indices(case["paired_support_rectangle"], merged_height=h, merged_width=w)
    unrelated_indices = rectangle_indices(case["unrelated_location_support_rectangle"], merged_height=h, merged_width=w)
    image_token_id = int(getattr(qwen.model.config, "image_token_id", qwen.tokenizer.convert_tokens_to_ids("<|image_pad|>")))
    accepted_ledger = [item for item in ledger.get(image_id, []) if str(item.get("final_state")) == "accepted"]
    accepted_ledger_by_id = {str(item.get("object_identifier")): item for item in accepted_ledger}
    boxes = {"target": query._object_pixel_box(target, accepted_ledger_by_id, width=raw.image.width, height=raw.image.height), "paired": query._object_pixel_box(paired, accepted_ledger_by_id, width=raw.image.width, height=raw.image.height)}
    if boxes["target"] is None or boxes["paired"] is None:
        raise ValueError(f"image {image_id}: owner boxes unavailable")
    normalized = {name: _as_normalized_box(box, width=raw.image.width, height=raw.image.height) for name, box in boxes.items()}
    accepted_objects = []
    for item in accepted_ledger:
        identifier = str(item.get("object_identifier", ""))
        annotation_id = identifier.split(":", 1)[-1]
        box = item.get("source_canvas_box_xyxy")
        if isinstance(box, Sequence) and len(box) == 4:
            accepted_objects.append({"annotation_id": annotation_id, "normalized_box": _as_normalized_box(box, width=raw.image.width, height=raw.image.height)})
    support_boxes = {"target": support_envelope_box(case["target_support_rectangle"], merged_height=h, merged_width=w), "paired": support_envelope_box(case["paired_support_rectangle"], merged_height=h, merged_width=w), "unrelated": support_envelope_box(case["unrelated_location_support_rectangle"], merged_height=h, merged_width=w)}
    arms: dict[str, Any] = {}
    for arm_name, indices, donor_name in (("target", target_indices, "target"), ("paired", paired_indices, "paired"), ("unrelated", unrelated_indices, "target")):
        generated = greedy_geometry_arm(model=qwen.model, model_inputs=model_inputs, features=features, grid_thw=grid_thw, merge_size=merge_size, prompt_ids=prompt_ids, row_prefix=target_prefix, image_grid_thw=model_inputs["image_grid_thw"], image_token_id=image_token_id, selected_indices=indices, tokenizer=qwen.tokenizer, max_new_tokens=MAX_NEW_TOKENS)
        paired_name = "paired" if donor_name == "target" else "target"
        attribution = geometry_attribution(generated["parsed"], donor_box=normalized[donor_name], paired_box=normalized[paired_name], donor_annotation_id=str(case[f"{donor_name}_annotation_id"] if donor_name == "target" else case["paired_annotation_id"]), accepted_objects=accepted_objects, support_box=support_boxes[arm_name])
        paired_attribution = None
        if arm_name == "unrelated":
            paired_attribution = geometry_attribution(generated["parsed"], donor_box=normalized["paired"], paired_box=normalized["target"], donor_annotation_id=str(case["paired_annotation_id"]), accepted_objects=accepted_objects, support_box=support_boxes[arm_name])
        row = _row_from_suffix(target_prefix, generated["generated_token_ids"]) if generated["valid"] else None
        score_hard = score_unrestricted = None
        release = None
        if row is not None:
            score_hard, _ = _score_row(model=qwen.model, model_inputs=model_inputs, features=features, grid_thw=grid_thw, merge_size=merge_size, prompt_ids=prompt_ids, row=row, image_grid_thw=model_inputs["image_grid_thw"], image_token_id=image_token_id, selected_indices=indices, position_mode="explicit")
            score_unrestricted, _ = _score_row(model=qwen.model, model_inputs=model_inputs, features=features, grid_thw=grid_thw, merge_size=merge_size, prompt_ids=prompt_ids, row=row, image_grid_thw=model_inputs["image_grid_thw"], image_token_id=image_token_id, selected_indices=None, position_mode="implicit")
            release = float(score_hard["coordinate_only_mean"] - score_unrestricted["coordinate_only_mean"])
        arms[arm_name] = {"support_indices": indices, "support_rectangle": case[f"{'unrelated_location_' if arm_name == 'unrelated' else arm_name + '_'}support_rectangle"], "generated": generated, "attribution": attribution, "paired_attribution": paired_attribution, "realized_row_token_ids": row, "hard_score": score_hard, "unrestricted_score": score_unrestricted, "coordinate_release": release}
    # Recompute target/paired eligibility after unrelated route exists.
    unrelated_closed = bool(arms["unrelated"]["generated"].get("valid") and arms["unrelated"]["generated"].get("natural_closure"))
    unrelated_target_iou = float(arms["unrelated"]["attribution"]["donor_iou"]) if unrelated_closed and arms["unrelated"]["attribution"].get("donor_iou") is not None else None
    unrelated_paired_iou = float(arms["unrelated"]["paired_attribution"]["donor_iou"]) if unrelated_closed and arms["unrelated"]["paired_attribution"].get("donor_iou") is not None else None
    for arm_name in ("target", "paired"):
        arm = arms[arm_name]
        unrelated_iou = unrelated_target_iou if arm_name == "target" else unrelated_paired_iou
        arm["eligibility"] = assess_donor_eligibility(valid_path=bool(arm["generated"]["valid"]), natural_closure=bool(arm["generated"]["natural_closure"]), coordinate_release=arm["coordinate_release"], attribution=arm["attribution"], unrelated_donor_iou=unrelated_iou)
    arms["unrelated"]["eligibility"] = {"passed": False, "classification": "specificity_control_only"}
    canonical_scores: dict[str, Any] = {}
    for name, row in (("target", target_row), ("paired", paired_row)):
        scores = {}
        for mode in ("implicit", "explicit", "all_allowed"):
            scores[mode], _ = _score_row(model=qwen.model, model_inputs=model_inputs, features=features, grid_thw=grid_thw, merge_size=merge_size, prompt_ids=prompt_ids, row=row, image_grid_thw=model_inputs["image_grid_thw"], image_token_id=image_token_id, selected_indices=None, position_mode=mode)
        canonical_scores[name] = {"arms": scores, "trust": _canonical_noop_gate(scores=scores)}
    trust = {"passed": all(item["trust"]["passed"] for item in canonical_scores.values()) and all(bool(arm["generated"].get("structural_mask_gate_passed")) and bool(arm["generated"].get("valid")) and bool(arm["generated"].get("natural_closure")) for arm in arms.values()), "canonical": canonical_scores, "unrelated_path_closed": unrelated_closed}
    return {"image_id": image_id, "unit_id": UNIT_ID, "config_sha256": sha256_file(Path(args.infer_config)), "source_jsonl_sha256": sha256_file(Path(args.source_jsonl)), "audit_ledger_sha256": sha256_file(Path(args.audit_ledger)), "cohort_sha256": sha256_file(Path(args.cohort)), "stratum": case["stratum"], "description": case["description"], "target_annotation_id": str(case["target_annotation_id"]), "paired_annotation_id": str(case["paired_annotation_id"]), "image_grid_thw": grid_thw, "merge_size": merge_size, "feature_fingerprint": query.feature_bundle_fingerprint(features), "arms": arms, "trust_gate": trust, "runtime_contract": {"dtype": "torch.float32", "attention_implementation": str(getattr(qwen.model.config, "_attn_implementation", "unknown")), "repetition_penalty": 1.0, "logits_processor": None, "cache_used": False, "max_new_tokens": MAX_NEW_TOKENS}}


def _run(args: argparse.Namespace) -> dict[str, Any]:
    from src.config.inference import load_infer_config
    from src.data import load_raw_examples
    from src.inference.runtime import assemble_runtime
    from scripts.research.run_sampled_rescue_transition import _temporary_cwd

    cohort = load_cohort(args.cohort)
    requested = [str(value) for value in args.image_ids]
    expected = [str(case["image_id"]) for case in cohort["cases"]]
    if not requested or any(value not in expected for value in requested) or len(requested) != len(set(requested)):
        raise ValueError("image_ids must be a unique subset of the frozen cohort")
    config_path = Path(args.infer_config).expanduser().resolve(strict=True)
    source_path = Path(args.source_jsonl).expanduser().resolve(strict=True)
    ledger_path = Path(args.audit_ledger).expanduser().resolve(strict=True)
    live_hashes = validate_live_input_hashes(cohort=cohort, config_path=config_path, source_jsonl_path=source_path, audit_ledger_path=ledger_path)
    with _temporary_cwd(config_path.parents[3]):
        resolved = load_infer_config(config_path)
    config = resolved.config.model_copy(update={"model": resolved.config.model.model_copy(update={"dtype": "fp32"})})
    runtime = assemble_runtime(config, source_gate_root=config_path.parents[3])
    qwen = runtime.qwen
    qwen.model.eval()
    if sorted({str(parameter.dtype) for parameter in qwen.model.parameters()}) != ["torch.float32"]:
        raise RuntimeError("full-model float32 contract failed")
    if str(getattr(qwen.model.config, "_attn_implementation", "unknown")) != "sdpa":
        raise RuntimeError("SDPA attention contract failed")
    raw_rows = load_raw_examples(source_path)
    raw_by_id = {str(row.metadata["source"]["image_id"]): row for row in raw_rows}
    ledger = query._load_ledger(ledger_path)
    cases = {str(case["image_id"]): case for case in cohort["cases"]}
    results = [_run_case(args, qwen=qwen, raw=raw_by_id[image_id], case=cases[image_id], ledger=ledger, resolved=resolved) for image_id in requested]
    return {"schema_version": SCHEMA_VERSION, "unit_id": UNIT_ID, **live_hashes, "cohort_sha256": sha256_file(Path(args.cohort)), "results": results, "panel_decision": classify_panel(results, cohort=cohort), "requested_image_ids": requested}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infer-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-jsonl", type=Path, default=DEFAULT_SOURCE_JSONL)
    parser.add_argument("--audit-ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-ids", nargs="+", default=list(EXPECTED_IMAGE_IDS))
    parser.add_argument("--merge-receipts", nargs="*", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cohort = load_cohort(args.cohort)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.merge_receipts is not None:
        payloads = [json.loads(Path(path).expanduser().resolve(strict=True).read_text(encoding="utf-8")) for path in args.merge_receipts]
        merged = merge_case_receipts(payloads, cohort=cohort, config_sha256=sha256_file(args.infer_config), source_jsonl_sha256=sha256_file(args.source_jsonl), audit_ledger_sha256=sha256_file(args.audit_ledger), cohort_sha256=sha256_file(args.cohort))
    else:
        merged = _run(args)
    merged["runner_sha256"] = sha256_file(Path(__file__))
    (output_dir / "receipt.json").write_text(json.dumps(merged, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
