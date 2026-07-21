#!/usr/bin/env python3
"""Collect exact-prefix premature-stop rescue events for the transition pilot.

This experiment-local collector has two deliberately separate stages:

* a model-free scan of the frozen greedy rollout and token trace; and
* an optional Hugging Face (HF) replay that samples sixteen one-row
  alternatives at the exact terminal prefix.

The source terminal prefix is copied as integer token identifiers.  It is
never decoded and re-tokenized.  A sampled row is accepted only when it is a
complete row, has one same-category physical target at a clear best match, and
the target is conservatively absent from every prior prediction.  An
unmatched output is evidence only; it is never turned into a negative label.

The output is intentionally assembler-shaped but the collector does not call
the StateBank assembler.  This keeps collection usable while the active
StateBank implementation evolves, and leaves all accepted/rejected attempts
and proof receipts on disk for a later immutable assembly step.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research.run_current_seeded_sampled_rollouts import (
    _build_requests,
)
from scripts.research.run_local_branch_causal_value import (
    _annotate_owner_matches,
    _generate_row,
    _single_native_inputs,
    hash_prefix_token_ids,
)


SCHEMA_VERSION = "exact_greedy_terminal_rescue_collector.v1"
COORDINATE_TOKEN_START = 151670
COORDINATE_TOKEN_END = 152670
OBJECT_REF_START = 151646
OBJECT_REF_END = 151647
BOX_START = 151648
BOX_END = 151649
IM_END = 151645
DEFAULT_SEEDS = tuple(range(16))
DEFAULT_TEMPERATURE = 0.4
DEFAULT_TOP_P = 0.95
DEFAULT_REPETITION_PENALTY = 1.0
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_MALFORMED_LIMIT = 2
DEFAULT_MIN_IOU = 0.5
DEFAULT_MATCH_MARGIN = 0.05
DEFAULT_EXCLUSION_IOU = 0.25
DEFAULT_EXCLUSION_INTERSECTION = 0.5
DEFAULT_EXCLUSION_CENTER_DISTANCE = 0.08
DEFAULT_GEOMETRY_IOU = 0.75
DEFAULT_TOTAL_ROW_BUDGET = 16
DEFAULT_REFERENCE_STATE_BANK_MANIFEST = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-20-own-prefix-entity-transition-and-coordinate-boundary-calibration-training-screen/"
    "smoke-a-combined-v4/state-bank/manifest.json"
)
MARKERS = {OBJECT_REF_START, OBJECT_REF_END, BOX_START, BOX_END}
TOKEN_TYPES = {"schema", "desc_text", "coordinate"}


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_seeds(value: str | Iterable[int] = ",".join(map(str, DEFAULT_SEEDS))) -> tuple[int, ...]:
    """Return an ordered, unique, non-negative seed tuple."""

    pieces = value.split(",") if isinstance(value, str) else list(value)
    try:
        seeds = tuple(int(item) for item in pieces if str(item).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError("seeds must be integers") from exc
    if not seeds or any(seed < 0 for seed in seeds):
        raise ValueError("seeds must be non-empty and non-negative")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    return seeds


def released_suffix_row_count(total_row_budget: int, prefix_object_row_count: int = 0) -> int:
    """Return rows available after prefix rows and the forced candidate.

    The budget is for the complete generated trajectory, not merely the
    released suffix.  The candidate consumes one row after the immutable
    source prefix.
    """

    total = int(total_row_budget)
    prefix_rows = int(prefix_object_row_count)
    if total < 1 or prefix_rows < 0:
        raise ValueError("total row budget must be positive and prefix rows non-negative")
    return max(0, total - prefix_rows - 1)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.resolve(strict=True).open(encoding="utf-8") as handle:
        result: list[dict[str, Any]] = []
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain JSON objects")
            result.append(value)
    return result


def _box_area(box: Sequence[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def box_iou(left: Sequence[float], right: Sequence[float]) -> float:
    """Compute normalized xyxy intersection-over-union."""

    if len(left) != 4 or len(right) != 4:
        raise ValueError("boxes must contain four coordinates")
    intersection = max(0.0, min(float(left[2]), float(right[2])) - max(float(left[0]), float(right[0]))) * max(
        0.0, min(float(left[3]), float(right[3])) - max(float(left[1]), float(right[1]))
    )
    union = _box_area(left) + _box_area(right) - intersection
    return 0.0 if union <= 0.0 else intersection / union


def intersection_over_smaller_area(left: Sequence[float], right: Sequence[float]) -> float:
    intersection = max(0.0, min(float(left[2]), float(right[2])) - max(float(left[0]), float(right[0]))) * max(
        0.0, min(float(left[3]), float(right[3])) - max(float(left[1]), float(right[1]))
    )
    smaller = min(_box_area(left), _box_area(right))
    return 0.0 if smaller <= 0.0 else intersection / smaller


def _center(box: Sequence[float]) -> tuple[float, float]:
    return ((float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0)


def _center_distance(left: Sequence[float], right: Sequence[float]) -> float:
    lx, ly = _center(left)
    rx, ry = _center(right)
    return (((lx - rx) ** 2 + (ly - ry) ** 2) ** 0.5) / 1000.0


def _center_inside(point_box: Sequence[float], container: Sequence[float]) -> bool:
    x, y = _center(point_box)
    return float(container[0]) <= x <= float(container[2]) and float(container[1]) <= y <= float(container[3])


def match_target(
    prediction: Mapping[str, Any],
    targets: Sequence[Mapping[str, Any]],
    *,
    minimum_iou: float = DEFAULT_MIN_IOU,
    minimum_margin: float = DEFAULT_MATCH_MARGIN,
) -> dict[str, Any] | None:
    """Return a unique same-category target match, or ``None``."""

    description = str(prediction.get("description", "")).strip().lower()
    box = prediction.get("coord_bins")
    if not description or not isinstance(box, list) or len(box) != 4:
        return None
    scored = sorted(
        (
            (box_iou(box, target["bbox"]), target)
            for target in targets
            if str(target.get("description", "")).strip().lower() == description
        ),
        key=lambda item: (-item[0], str(item[1].get("object_id", ""))),
    )
    if not scored or scored[0][0] < float(minimum_iou):
        return None
    second = scored[1][0] if len(scored) > 1 else 0.0
    if scored[0][0] - second < float(minimum_margin):
        return None
    return {
        "owner_id": str(scored[0][1]["object_id"]),
        "category": description,
        "iou": float(scored[0][0]),
        "second_iou": float(second),
        "margin": float(scored[0][0] - second),
        "prediction_box": [int(value) for value in box],
        "target_box": [int(value) for value in scored[0][1]["bbox"]],
    }


def target_noncoverage_receipts(
    predictions: Sequence[Mapping[str, Any]],
    target: Mapping[str, Any],
    *,
    same_class_targets: Sequence[Mapping[str, Any]] = (),
    exclusion_iou: float = DEFAULT_EXCLUSION_IOU,
    exclusion_intersection: float = DEFAULT_EXCLUSION_INTERSECTION,
    exclusion_center_distance: float = DEFAULT_EXCLUSION_CENTER_DISTANCE,
) -> tuple[bool, list[dict[str, Any]]]:
    """Prove a target is absent from every prior prediction conservatively.

    This is intentionally stronger than a same-category IoU test.  A loose
    prior box, an oversized box, or a cross-category crop can still refer to
    the target, so center inclusion and intersection-over-smaller-area are
    recorded for every prior row.  Any plausible association rejects the
    target-scoped event.
    """

    target_box = target.get("bbox")
    if not isinstance(target_box, list) or len(target_box) != 4:
        raise ValueError("target must contain bbox")
    target_category = str(target.get("description", "")).strip().lower()
    receipts: list[dict[str, Any]] = []
    accepted = True
    for index, prediction in enumerate(predictions):
        box = prediction.get("coord_bins")
        category = str(prediction.get("description", "")).strip().lower()
        if not isinstance(box, list) or len(box) != 4:
            accepted = False
            receipts.append({
                "prefix_object_row_index": index,
                "target_owner_id": str(target["object_id"]),
                "excluded": False,
                "plausible": True,
                "prior_category": category,
                "target_category": target_category,
                "predicted_box": None,
                "target_box": [int(value) for value in target_box],
                "iou": 0.0,
                "intersection_over_smaller_area": 0.0,
                "prior_center_in_target": False,
                "target_center_in_prior": False,
                "center_distance_norm1000": None,
                "plausible_reasons": ["prior_prediction_has_no_valid_box"],
                "thresholds": {
                    "same_category_iou": float(exclusion_iou),
                    "intersection_over_smaller_area": float(exclusion_intersection),
                    "center_distance_norm1000": float(exclusion_center_distance),
                },
                "best_same_class_owner_id": None,
                "best_same_class_owner_margin": None,
            })
            continue
        iou = box_iou(box, target_box)
        overlap = intersection_over_smaller_area(box, target_box)
        prior_center_in_target = _center_inside(box, target_box)
        target_center_in_prior = _center_inside(target_box, box)
        distance = _center_distance(box, target_box)
        same_class_scored = sorted(
            (
                box_iou(box, same_class_target["bbox"]),
                str(same_class_target.get("object_id", "")),
            )
            for same_class_target in same_class_targets
            if str(same_class_target.get("description", "")).strip().lower() == category
        )
        best_same_class_owner_id = same_class_scored[0][1] if same_class_scored else None
        best_same_class_margin = (
            float(same_class_scored[0][0] - (same_class_scored[1][0] if len(same_class_scored) > 1 else 0.0))
            if same_class_scored
            else None
        )
        plausible_reasons: list[str] = []
        if category == target_category and iou >= float(exclusion_iou):
            plausible_reasons.append("same_category_iou")
        if overlap >= float(exclusion_intersection):
            plausible_reasons.append("intersection_over_smaller_area")
        if prior_center_in_target or target_center_in_prior:
            plausible_reasons.append("center_inclusion")
        if distance <= float(exclusion_center_distance):
            plausible_reasons.append("center_distance")
        plausible = bool(plausible_reasons)
        if plausible:
            accepted = False
        receipts.append({
            "prefix_object_row_index": index,
            "target_owner_id": str(target["object_id"]),
            "excluded": not plausible,
            "plausible": plausible,
            "prior_category": category,
            "target_category": target_category,
            "predicted_box": [int(value) for value in box],
            "target_box": [int(value) for value in target_box],
            "iou": float(iou),
            "intersection_over_smaller_area": float(overlap),
            "prior_center_in_target": bool(prior_center_in_target),
            "target_center_in_prior": bool(target_center_in_prior),
            "center_distance_norm1000": float(distance),
            "plausible_reasons": plausible_reasons,
            "thresholds": {
                "same_category_iou": float(exclusion_iou),
                "intersection_over_smaller_area": float(exclusion_intersection),
                "center_distance_norm1000": float(exclusion_center_distance),
            },
            "best_same_class_owner_id": best_same_class_owner_id,
            "best_same_class_owner_margin": best_same_class_margin,
        })
    return accepted, receipts


def trace_terminal_prefix(trace_rows: Sequence[Mapping[str, Any]], *, image_id: str) -> dict[str, Any]:
    """Extract the exact generated prefix before the first terminal token."""

    rows = [row for row in trace_rows if str(row.get("row_id")) == str(image_id) and row.get("trace_type") == "generated_token"]
    rows.sort(key=lambda row: int(row.get("generated_step_index", -1)))
    if not rows:
        raise ValueError(f"no generated token trace for {image_id}")
    terminal_positions = [index for index, row in enumerate(rows) if row.get("is_stop") or int(row.get("token_id", -1)) == IM_END]
    if len(terminal_positions) != 1:
        raise ValueError(f"expected exactly one terminal token for {image_id}, got {terminal_positions}")
    terminal_index = terminal_positions[0]
    terminal = rows[terminal_index]
    prefix = [int(row["token_id"]) for row in rows[:terminal_index]]
    generated = [int(row["token_id"]) for row in rows[: terminal_index + 1]]
    if int(terminal.get("token_id", -1)) != IM_END:
        raise ValueError(f"trace terminal token is not im_end for {image_id}")
    return {
        "image_id": str(image_id),
        "prefix_token_ids": prefix,
        "prefix_token_ids_sha256": hash_prefix_token_ids(prefix),
        "terminal_token_ids": [IM_END],
        "generated_token_ids": generated,
        "terminal_generated_step_index": int(terminal.get("generated_step_index", terminal_index)),
        "trace_token_count_before_stop": terminal_index,
    }


def _stable_shard(image_id: str, *, shard_index: int, shard_count: int) -> bool:
    if shard_count <= 0 or shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard index/count are invalid")
    return int(hashlib.sha256(str(image_id).encode()).hexdigest(), 16) % shard_count == shard_index


def scan_terminal_records(
    gt_rows: Sequence[Mapping[str, Any]],
    trace_rows: Sequence[Mapping[str, Any]],
    *,
    image_ids: set[str] | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
    minimum_iou: float = DEFAULT_MIN_IOU,
    minimum_margin: float = DEFAULT_MATCH_MARGIN,
    exclusion_iou: float = DEFAULT_EXCLUSION_IOU,
    exclusion_intersection: float = DEFAULT_EXCLUSION_INTERSECTION,
    exclusion_center_distance: float = DEFAULT_EXCLUSION_CENTER_DISTANCE,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Scan frozen source artifacts without allocating a model."""

    trace_by_id = defaultdict(list)
    for row in trace_rows:
        if row.get("trace_type") == "generated_token":
            trace_by_id[str(row.get("row_id"))].append(row)
    records: list[dict[str, Any]] = []
    exclusions: defaultdict[str, int] = defaultdict(int)
    for source in sorted(gt_rows, key=lambda row: str(row.get("example_id", row.get("row_id", "")))):
        image_id = str(source.get("example_id", source.get("row_id", "")))
        image_path_id = Path(str(source.get("image_path", ""))).stem
        source_selection_ids = {image_id, image_path_id, image_path_id.lstrip("0") or "0"}
        shard_key = image_path_id or image_id
        if not image_id or (image_ids is not None and not source_selection_ids.intersection(image_ids)) or not _stable_shard(shard_key, shard_index=shard_index, shard_count=shard_count):
            exclusions["not_selected"] += 1
            continue
        try:
            terminal = trace_terminal_prefix(trace_by_id[image_id], image_id=image_id)
        except (KeyError, ValueError) as exc:
            exclusions[f"trace:{str(exc).split(':', 1)[0]}"] += 1
            continue
        preds = source.get("pred")
        targets = source.get("gt")
        if not isinstance(preds, list) or not isinstance(targets, list) or not targets:
            exclusions["missing_predictions_or_gt"] += 1
            continue
        if str(source.get("decode_stop_reason", "")) != "im_end":
            exclusions["source_not_im_end"] += 1
            continue
        covered: list[str] = []
        full_resolved = True
        row_matches: list[dict[str, Any]] = []
        for index, prediction in enumerate(preds):
            match = match_target(prediction, targets, minimum_iou=minimum_iou, minimum_margin=minimum_margin)
            if match is None:
                full_resolved = False
                row_matches.append({"prefix_object_row_index": index, "status": "unresolved"})
                continue
            covered.append(match["owner_id"])
            row_matches.append({"prefix_object_row_index": index, "status": "resolved", **match})
        if len(set(covered)) != len(covered):
            full_resolved = False
            exclusions["prefix_duplicate_owner"] += 1
        remaining = [target for target in targets if str(target.get("object_id")) not in set(covered)]
        target_candidates: list[dict[str, Any]] = []
        for target in remaining:
            allowed, receipts = target_noncoverage_receipts(
                preds,
                target,
                same_class_targets=targets,
                exclusion_iou=exclusion_iou,
                exclusion_intersection=exclusion_intersection,
                exclusion_center_distance=exclusion_center_distance,
            )
            if allowed:
                target_candidates.append({
                    "target": dict(target),
                    "noncoverage_receipts": receipts,
                    "proof_stratum": "full_resolved" if full_resolved else "target_scoped_noncoverage",
                })
            else:
                exclusions["target_plausibly_covered"] += 1
        if not target_candidates:
            exclusions["no_target_candidate"] += 1
            continue
        records.append({
            "example_id": image_id,
            "source_physical_image_id": image_path_id,
            "source_row": source,
            "terminal": terminal,
            "prefix_object_row_count": len(preds),
            "prefix_row_matches": row_matches,
            "covered_owner_ids": sorted(set(covered)),
            "full_resolved": bool(full_resolved),
            "target_candidates": target_candidates,
        })
    return records, dict(sorted(exclusions.items()))


def token_type(token_id: int) -> str:
    if int(token_id) in {OBJECT_REF_START, OBJECT_REF_END, BOX_START, BOX_END}:
        return "schema"
    if COORDINATE_TOKEN_START <= int(token_id) < COORDINATE_TOKEN_END:
        return "coordinate"
    return "desc_text"


def selected_sites(token_ids: Sequence[int], *, end: int | None = None) -> list[dict[str, Any]]:
    limit = len(token_ids) if end is None else int(end)
    return [{"candidate_token_offset": index, "intended_token_type": token_type(int(token_id))} for index, token_id in enumerate(token_ids[:limit])]


def description_end_interval(token_ids: Sequence[int]) -> int:
    values = [int(value) for value in token_ids]
    try:
        end = values.index(OBJECT_REF_END)
    except ValueError as exc:
        raise ValueError("candidate row lacks object_ref_end") from exc
    return end + 1


def build_target_exclusion_review(
    target: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    *,
    proof_stratum: str,
    native_terminal_generated_step: int = 0,
) -> dict[str, Any]:
    prior_row_exclusions = []
    for receipt in receipts:
        prior_row_exclusions.append({
            "row_index": int(receipt["prefix_object_row_index"]),
            "prediction_description": str(receipt.get("prior_category", "")),
            "prediction_coord_bins": list(receipt.get("predicted_box", [0, 0, 0, 0])),
            "same_category": str(receipt.get("prior_category", "")).lower() == str(target.get("description", "")).lower(),
            "target_iou": float(receipt.get("iou", 0.0)),
            "target_center_inside_prediction": bool(receipt.get("target_center_in_prior", False)),
            "prediction_center_inside_target": bool(receipt.get("prior_center_in_target", False)),
            "intersection_over_smaller_area": float(receipt.get("intersection_over_smaller_area", 0.0)),
            "best_same_class_owner_id": receipt.get("best_same_class_owner_id"),
            "best_same_class_owner_margin": receipt.get("best_same_class_owner_margin"),
            "plausible_target_association": bool(receipt.get("plausible", False)),
            "evidence_reason": (
                ";".join(receipt.get("plausible_reasons", []))
                if receipt.get("plausible_reasons")
                else "no_overlap_or_center_association"
            ),
        })
    return {
        "target_owner_id": str(target["object_id"]),
        "native_terminal_generated_step": int(native_terminal_generated_step),
        "prior_row_count": len(prior_row_exclusions),
        "thresholds": {
            "same_category_iou": DEFAULT_EXCLUSION_IOU,
            "intersection_over_smaller_area": DEFAULT_EXCLUSION_INTERSECTION,
            "center_distance_norm1000": DEFAULT_EXCLUSION_CENTER_DISTANCE,
        },
        "prior_row_exclusions": prior_row_exclusions,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def _physical_image_id(example: Any) -> int:
    source = getattr(example, "metadata", {}).get("source", {})
    value = source.get("image_id") if isinstance(source, Mapping) else None
    if value is None:
        value = str(example.example_id).rsplit("_", 1)[-1]
    return int(value)


def _source_checkpoint_id(session_receipt: Mapping[str, Any]) -> str:
    candidate = session_receipt.get("model_identity_fingerprint") or session_receipt.get("model_identity")
    if isinstance(candidate, str) and len(candidate) == 64:
        return candidate
    return sha256_json(session_receipt)


def source_prompt_trace_entry(manifest: Mapping[str, Any], example_id: str) -> dict[str, Any]:
    """Return the frozen source prompt receipt for one example."""

    entries = manifest.get("prompt_trace")
    if not isinstance(entries, list):
        raise ValueError("source run manifest lacks prompt_trace")
    for entry in entries:
        if isinstance(entry, Mapping) and str(entry.get("row_id")) == str(example_id):
            return dict(entry)
    raise ValueError(f"source run manifest lacks prompt trace for {example_id}")


def validate_source_prompt_trace(
    manifest: Mapping[str, Any], example_id: str, prompt_ids: Sequence[int]
) -> dict[str, Any]:
    """Require materialized prompt ids to match the original source run."""

    entry = source_prompt_trace_entry(manifest, example_id)
    expected = str(entry.get("backend_executed_prompt_token_ids_sha256", ""))
    actual = hash_prefix_token_ids(prompt_ids)
    if not expected or actual != expected:
        raise ValueError(
            f"source prompt hash mismatch for {example_id}: expected {expected}, got {actual}"
        )
    return {
        "row_id": str(example_id),
        "source_backend_executed_prompt_token_ids_sha256": expected,
        "materialized_prompt_token_ids_sha256": actual,
        "prompt_token_parity": "verified",
    }


def _entity_ledger(targets: Sequence[Mapping[str, Any]], source_path: Path) -> list[dict[str, Any]]:
    return [
        {
            "entity_id": str(target["object_id"]),
            "category": str(target.get("description", "")),
            "entity_trusted": True,
            "geometry_trusted": True,
            "reference_bbox": [int(value) for value in target["bbox"]],
            "review_source": str(source_path),
            "reviewer": "automatic-positive-official-annotation",
            "review_confidence": "high",
            "comment": "Positive reference entity; official annotations are not treated as complete negatives.",
        }
        for target in targets
    ]


def _build_event(
    record: Mapping[str, Any],
    target_choice: Mapping[str, Any],
    sampled: Mapping[str, Any],
    *,
    prompt_ids: Sequence[int],
    image_pad_interval: Sequence[int],
    image_id: int,
    image_path: Path,
    image_width: int,
    image_height: int,
    source_gt_path: Path,
    checkpoint_id: str,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    counterfactual_admission: Mapping[str, Any],
    counterfactual_diagnostics: Mapping[str, Any] | None = None,
    split: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    target = target_choice["target"]
    positive_ids = [int(value) for value in sampled["raw_generated_token_ids"]]
    prefix_ids = [int(value) for value in record["terminal"]["prefix_token_ids"]]
    prompt_hash = hash_prefix_token_ids(prompt_ids)
    prefix_hash = hash_prefix_token_ids(prefix_ids)
    positive_hash = hash_prefix_token_ids(positive_ids)
    terminal_ids = [IM_END]
    terminal_hash = hash_prefix_token_ids(terminal_ids)
    positive_match = sampled["target_match"]
    geometry_trusted = float(positive_match["iou"]) >= DEFAULT_GEOMETRY_IOU
    # The entity transition covers the complete row even when coordinate
    # geometry is not trusted.  The StateBank geometry flag masks coordinate
    # correction while preserving schema/description continuation after the
    # coordinate span.
    interval_end = len(positive_ids)
    event_id = f"terminal-rescue-image-{image_id}-target-{target['object_id']}"
    provenance_sample = {
        "mode": "sampled",
        "seed": int(sampled["seed"]),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "repetition_penalty": float(repetition_penalty),
        "checkpoint_id": checkpoint_id,
        "prompt_token_ids_sha256": prompt_hash,
        "prefix_token_ids_sha256": prefix_hash,
    }
    provenance_greedy = {
        "mode": "greedy",
        "seed": 0,
        "temperature": 0.0,
        "top_p": 1.0,
        "repetition_penalty": float(repetition_penalty),
        "checkpoint_id": checkpoint_id,
        "prompt_token_ids_sha256": prompt_hash,
        "prefix_token_ids_sha256": prefix_hash,
    }
    positive_candidate_id = f"sampled-positive-seed-{sampled['seed']}"
    terminal_candidate_id = "greedy-premature-terminal"
    rollout = {
        "event_id": event_id,
        "image": {
            "image_id": int(image_id),
            "path": str(image_path),
            "width": int(image_width),
            "height": int(image_height),
            "content_sha256": sha256_file(image_path),
        },
        "split": split,
        "split_group_id": f"image:{image_id}",
        "executed_prompt_token_ids": [int(value) for value in prompt_ids],
        "executed_prompt_token_ids_sha256": prompt_hash,
        "image_pad_interval": [int(image_pad_interval[0]), int(image_pad_interval[1])],
        "prefix_token_ids": prefix_ids,
        "prefix_token_ids_sha256": prefix_hash,
        "candidates": [
            {
                "candidate_id": positive_candidate_id,
                "token_ids": positive_ids,
                "token_ids_sha256": positive_hash,
                "generation_provenance": provenance_sample,
                "evidence_text": str(sampled.get("raw_generated_text", "")),
            },
            {
                "candidate_id": terminal_candidate_id,
                "token_ids": terminal_ids,
                "token_ids_sha256": terminal_hash,
                "generation_provenance": provenance_greedy,
                "evidence_text": "<|im_end|>",
            },
        ],
    }
    proof_stratum = str(target_choice["proof_stratum"])
    prefix_status = "resolved" if proof_stratum == "full_resolved" else "target_scoped_noncoverage"
    prefix_proofs = [
        {
            "prefix_object_row_index": index,
            "owner_id": str(match["owner_id"]),
            "review_provenance": {
                "source": str(source_gt_path),
                "reviewer": "automatic-unique-same-category-match",
                "confidence": "high",
                "comment": f"Unique source row owner; IoU={float(match['iou']):.4f}; margin={float(match['margin']):.4f}.",
            },
        }
        for index, match in enumerate(record.get("prefix_row_matches", []))
        if match.get("status") == "resolved"
    ]
    target_noncoverage_proof = (
        build_target_exclusion_review(
            target,
            target_choice["noncoverage_receipts"],
            proof_stratum=proof_stratum,
            native_terminal_generated_step=int(record["terminal"]["terminal_generated_step_index"]),
        )
        if prefix_status == "target_scoped_noncoverage"
        else None
    )
    candidate_review = [
        {
            "candidate_id": positive_candidate_id,
            "role": "positive",
            "harmful_kind": None,
            "physical_owner_id": str(target["object_id"]),
            "coverage_status": "uncovered",
            "entity_review_status": "trusted",
            "geometry_review_status": "trusted" if geometry_trusted else "unknown",
            "entity_eligible": True,
            "geometry_eligible": False,
            "owner_resolution_interval": [0, interval_end],
            "coordinate_decision": None,
            "selected_sites": selected_sites(positive_ids, end=interval_end),
        },
        {
            "candidate_id": terminal_candidate_id,
            "role": "harmful",
            "harmful_kind": "premature_terminal",
            "physical_owner_id": None,
            "coverage_status": "unknown",
            "entity_review_status": "trusted",
            "geometry_review_status": "unknown",
            "entity_eligible": True,
            "geometry_eligible": False,
            "owner_resolution_interval": None,
            "coordinate_decision": None,
            "selected_sites": [{"candidate_token_offset": 0, "intended_token_type": "schema"}],
        },
    ]
    review = {
        "event_id": event_id,
        "admission_status": "accepted",
        "rejection_reason": None,
        "physical_entities": _entity_ledger(record["source_row"]["gt"], source_gt_path),
        "prefix_object_row_count": int(record["prefix_object_row_count"]),
        "prefix_coverage_status": prefix_status,
        "prefix_covered_owner_proofs": prefix_proofs if prefix_status == "resolved" else [],
        "target_owner_noncoverage_proof": target_noncoverage_proof,
        "entity_transition_eligible": True,
        "coordinate_boundary_eligible": False,
        "counterfactual_admission": {
            key: counterfactual_admission[key]
            for key in (
                "row_budget",
                "generated_token_budget",
                "target_owner_retained",
                "verified_owner_delta",
                "confirmed_new_duplicate_count",
                "confirmed_new_malformed_count",
                "confirmed_new_unsupported_entity_count",
                "unknown_suffix_neutral",
                "unknown_suffix_provenance",
            )
        },
        "candidates": candidate_review,
        "review_provenance": {
            "schema_version": SCHEMA_VERSION,
            "policy": "exact-terminal-prefix-sampled-uncovered-target-over-stop",
            "target_match": dict(positive_match),
            "geometry_trusted_for_continuation": bool(geometry_trusted),
            "target_noncoverage": target_noncoverage_proof,
            "counterfactual_diagnostics": dict(counterfactual_diagnostics or {}),
            "source_terminal_step": int(record["terminal"]["terminal_generated_step_index"]),
        },
    }
    return rollout, review


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-gt-vs-pred", type=Path, required=True)
    parser.add_argument("--source-token-trace", type=Path, required=True)
    parser.add_argument("--infer-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-ids", default="", help="Comma-separated example IDs or physical IDs")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--repetition-penalty", type=float, default=DEFAULT_REPETITION_PENALTY)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--total-generated-token-budget", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument(
        "--total-row-budget",
        type=int,
        default=DEFAULT_TOTAL_ROW_BUDGET,
        help="Total generated object-row budget, including immutable prefix rows and the candidate row.",
    )
    parser.add_argument(
        "--reference-state-bank-manifest",
        type=Path,
        default=DEFAULT_REFERENCE_STATE_BANK_MANIFEST,
        help="Reference StateBank manifest that defines the composite source checkpoint identity.",
    )
    parser.add_argument("--malformed-limit", type=int, default=DEFAULT_MALFORMED_LIMIT)
    parser.add_argument("--scan-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _run_forced_complete_row_suffix(
    *,
    session: Any,
    native_inputs: Mapping[str, Any],
    prefix_token_ids: Sequence[int],
    candidate_token_ids: Sequence[int],
    target_owner_id: str,
    covered_owner_ids: Sequence[str],
    entity_ledger: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    image_width: int,
    image_height: int,
    repetition_penalty: float,
    total_generated_token_budget: int,
    total_row_budget: int,
    prefix_object_row_count: int,
) -> dict[str, Any]:
    """Release a greedy suffix after a forced complete row at equal budget.

    Only uniquely matched owners count as confirmed evidence.  Unmatched or
    ambiguous suffix predictions remain neutral and are preserved in the
    receipt instead of being treated as hallucinations.
    """

    prefix = [int(value) for value in prefix_token_ids]
    candidate = [int(value) for value in candidate_token_ids]
    if (
        not candidate
        or len(prefix) + len(candidate) > int(total_generated_token_budget)
        or int(prefix_object_row_count) + 1 > int(total_row_budget)
    ):
        return {
            "target_owner_retained": False,
            "unknown_suffix_neutral": True,
            "verified_owner_delta": {"added_owner_ids": [], "removed_owner_ids": []},
            "generated_token_budget": {"native": 0, "counterfactual": len(candidate)},
            "row_budget": {"native": int(total_row_budget), "counterfactual": int(total_row_budget)},
            "suffix_rows": [],
            "confirmed_new_duplicate_count": 0,
            "confirmed_new_malformed_count": 0,
            "confirmed_new_unsupported_entity_count": 0,
            "reason": "candidate_exceeds_fixed_token_budget",
            "unknown_suffix_provenance": {"policy": "fixed-budget suffix not run"},
        }
    remaining_tokens = int(total_generated_token_budget) - len(prefix) - len(candidate)
    current_prefix = prefix + candidate
    existing_owner_ids = {str(value) for value in covered_owner_ids} | {str(target_owner_id)}
    suffix_rows: list[dict[str, Any]] = []
    duplicate_count = 0
    malformed_count = 0
    for row_index in range(released_suffix_row_count(int(total_row_budget), int(prefix_object_row_count))):
        if remaining_tokens <= 0:
            break
        row = _generate_row(
            session=session,
            native_inputs=native_inputs,
            prefix_token_ids=current_prefix,
            tokenizer=tokenizer,
            image_width=image_width,
            image_height=image_height,
            mode="greedy",
            seed=None,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=float(repetition_penalty),
            max_new_tokens=remaining_tokens,
            malformed_limit=DEFAULT_MALFORMED_LIMIT,
            row_index=row_index,
        )
        _annotate_owner_matches(
            row,
            entity_ledger=entity_ledger,
            image_width=image_width,
            image_height=image_height,
            covered_entity_ids=sorted(existing_owner_ids),
        )
        emitted = [int(value) for value in row.get("raw_generated_token_ids", [])]
        remaining_tokens -= len(emitted)
        stop_reason = str((row.get("row_stop") or {}).get("stop_reason", "unknown"))
        if stop_reason in {"malformed_limit", "contaminated_complete_row"}:
            malformed_count += 1
        owners = {str(value) for value in row.get("strict_matched_owner_ids", [])}
        duplicate_count += len(owners & existing_owner_ids)
        existing_owner_ids.update(owners)
        suffix_rows.append({
            "row_index": row_index,
            "stop_reason": stop_reason,
            "raw_generated_token_ids": emitted,
            "raw_generated_token_ids_sha256": hash_prefix_token_ids(emitted),
            "strict_matched_owner_ids": sorted(owners),
            "unmatched_or_ambiguous_prediction_indices": list(row.get("unmatched_or_ambiguous_prediction_indices", [])),
        })
        if stop_reason != "complete_row" or not emitted:
            break
        current_prefix.extend(emitted)
    return {
        "target_owner_retained": True,
        "unknown_suffix_neutral": True,
        "verified_owner_delta": {"added_owner_ids": [str(target_owner_id)], "removed_owner_ids": []},
        "generated_token_budget": {
            "native": int(total_generated_token_budget),
            "counterfactual": int(total_generated_token_budget),
        },
        "row_budget": {"native": int(total_row_budget), "counterfactual": int(total_row_budget)},
        "suffix_rows": suffix_rows,
        "confirmed_new_duplicate_count": int(duplicate_count),
        "confirmed_new_malformed_count": int(malformed_count),
        "confirmed_new_unsupported_entity_count": 0,
        "unknown_suffix_provenance": {
            "policy": "only uniquely matched owners are counted; unmatched suffix predictions are neutral",
            "suffix_row_count": len(suffix_rows),
            "prefix_object_row_count": int(prefix_object_row_count),
            "total_row_budget": int(total_row_budget),
            "remaining_token_budget_after_suffix": int(remaining_tokens),
        },
    }


def collect_runtime(args: argparse.Namespace, records: Sequence[Mapping[str, Any]], output_root: Path) -> dict[str, Any]:
    from src.config.fingerprint import sha256_json as config_sha256_json
    from src.config.inference import load_infer_config
    from src.data import load_raw_examples
    from src.inference.backend import open_backend_session
    from src.inference.runtime import assemble_frontend
    from src.rollout_calibration import load_state_bank_manifest_binding
    from scripts.research.build_inference_coordinate_boundary_state_bank import (
        checkpoint_identity_for_inference_run,
    )

    resolved = load_infer_config(args.infer_config.resolve(strict=True))
    config = resolved.config
    examples = list(load_raw_examples(config.data.input_jsonl))
    frontend = assemble_frontend(config, generation_config_fingerprint=config_sha256_json(config.generation.model_dump(mode="json")))
    requests, _ = _build_requests(config, frontend, examples)
    request_by_id = {str(request.request_id): request for request in requests}
    example_by_id = {str(example.example_id): example for example in examples}
    seeds = parse_seeds(args.seeds)
    rollout_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    checkpoint_id = ""
    source_run_manifest_path = args.source_gt_vs_pred.resolve().parent / "run_manifest.json"
    source_run_manifest = json.loads(source_run_manifest_path.read_text(encoding="utf-8"))
    reference_manifest = args.reference_state_bank_manifest.resolve(strict=True)
    reference_binding = load_state_bank_manifest_binding(reference_manifest)
    adapter_path = Path(str(source_run_manifest["adapter_identity"]["adapter_path"]))
    embedding_path = Path(str(source_run_manifest["embedding_delta_identity"]["identity"]["delta_path"]))
    checkpoint_identity = checkpoint_identity_for_inference_run(
        inference_manifest=source_run_manifest,
        reference_checkpoint=reference_binding.source_checkpoint,
        source_adapter_path=adapter_path,
        source_embedding_payload_path=embedding_path,
    )
    checkpoint_id = sha256_json(checkpoint_identity.to_artifact_dict())
    if checkpoint_id != reference_binding.source_checkpoint_id:
        raise RuntimeError(
            "source checkpoint composite identity differs from reference StateBank manifest: "
            f"expected {reference_binding.source_checkpoint_id}, got {checkpoint_id}"
        )
    image_pad_token_id = int(frontend.qwen.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>"))
    with open_backend_session(frontend.launch) as session:
        session_receipt = session.receipt.to_artifact_dict()
        runtime_model_identity = _source_checkpoint_id(session_receipt)
        for record in records:
            example_id = str(record["example_id"])
            example = example_by_id.get(example_id)
            request = request_by_id.get(example_id)
            if example is None or request is None:
                attempt_rows.append({"example_id": example_id, "status": "rejected", "reason": "missing_runtime_example"})
                continue
            native_inputs, executed_ids, observed_grids, media_sha = session._materialize_native_inputs((request,))
            prompt_ids = [int(value) for value in executed_ids[0]]
            if prompt_ids != [int(value) for value in request.expected_executed_prompt_token_ids]:
                raise RuntimeError(f"prompt token parity failed for {example_id}")
            source_prompt_receipt = validate_source_prompt_trace(source_run_manifest, example_id, prompt_ids)
            pad_positions = [index for index, token in enumerate(prompt_ids) if token == image_pad_token_id]
            if not pad_positions or pad_positions != list(range(pad_positions[0], pad_positions[-1] + 1)):
                raise RuntimeError(f"image-pad interval is not contiguous for {example_id}")
            one_native = _single_native_inputs(native_inputs)
            prefix_token_ids = [int(value) for value in record["terminal"]["prefix_token_ids"]]
            remaining_row_tokens = int(args.total_generated_token_budget) - len(prefix_token_ids)
            if remaining_row_tokens <= 0 or int(record["prefix_object_row_count"]) + 1 > int(args.total_row_budget):
                attempt_rows.append({"example_id": example_id, "status": "rejected", "reason": "candidate_exceeds_total_trajectory_budget"})
                continue
            source_terminal = _generate_row(
                session=session,
                native_inputs=one_native,
                prefix_token_ids=prefix_token_ids,
                tokenizer=session._tokenizer,
                image_width=int(example.image.width),
                image_height=int(example.image.height),
                mode="greedy",
                seed=None,
                temperature=0.0,
                top_p=1.0,
                repetition_penalty=float(args.repetition_penalty),
                max_new_tokens=min(int(args.max_new_tokens), remaining_row_tokens),
                malformed_limit=int(args.malformed_limit),
                row_index=int(record["prefix_object_row_count"]),
            )
            if source_terminal.get("row_stop", {}).get("stop_reason") != "terminal" or source_terminal.get("raw_generated_token_ids") != [IM_END]:
                attempt_rows.append({"example_id": example_id, "status": "rejected", "reason": "greedy_terminal_replay_parity_failed", "observed": source_terminal})
                continue
            _annotate_owner_matches(
                source_terminal,
                entity_ledger=_entity_ledger(record["source_row"]["gt"], args.source_gt_vs_pred),
                image_width=int(example.image.width),
                image_height=int(example.image.height),
                covered_entity_ids=record.get("covered_owner_ids", []),
            )
            accepted: list[dict[str, Any]] = []
            for seed in seeds:
                sampled = _generate_row(
                    session=session,
                    native_inputs=one_native,
                    prefix_token_ids=prefix_token_ids,
                    tokenizer=session._tokenizer,
                    image_width=int(example.image.width),
                    image_height=int(example.image.height),
                    mode="sample",
                    seed=int(seed),
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    repetition_penalty=float(args.repetition_penalty),
                    max_new_tokens=min(int(args.max_new_tokens), remaining_row_tokens),
                    malformed_limit=int(args.malformed_limit),
                    row_index=int(record["prefix_object_row_count"]),
                )
                _annotate_owner_matches(
                    sampled,
                    entity_ledger=_entity_ledger(record["source_row"]["gt"], args.source_gt_vs_pred),
                    image_width=int(example.image.width),
                    image_height=int(example.image.height),
                    covered_entity_ids=record.get("covered_owner_ids", []),
                )
                target_matches: list[dict[str, Any]] = []
                full_match = match_target(
                    sampled["parsed_predictions"][0] if len(sampled.get("parsed_predictions", [])) == 1 else {},
                    record["source_row"]["gt"],
                    minimum_iou=DEFAULT_MIN_IOU,
                    minimum_margin=DEFAULT_MATCH_MARGIN,
                ) if sampled.get("status") == "success" and sampled.get("row_stop", {}).get("stop_reason") == "complete_row" else None
                for target_choice in record["target_candidates"]:
                    match = full_match
                    if match is not None and match["owner_id"] == str(target_choice["target"]["object_id"]):
                        target_matches.append({**target_choice, "target_match": match})
                accepted_sample = target_matches[0] if len(target_matches) == 1 else None
                attempt_rows.append({
                    "example_id": example_id,
                    "seed": int(seed),
                    "status": "accepted" if accepted_sample is not None else "rejected",
                    "reason": None if accepted_sample is not None else "no_unique_complete_target_row",
                    "row_stop": sampled.get("row_stop"),
                    "parsed_predictions": sampled.get("parsed_predictions", []),
                    "target_matches": [item["target_match"] for item in target_matches],
                    "raw_generated_token_ids": sampled.get("raw_generated_token_ids", []),
                    "source_prompt_trace": source_prompt_receipt,
                })
                if accepted_sample is not None:
                    accepted.append({**dict(sampled), **accepted_sample})
            if not accepted:
                continue
            ledger = _entity_ledger(record["source_row"]["gt"], args.source_gt_vs_pred)
            accepted.sort(
                key=lambda row: (
                    -int(float(row["target_match"]["iou"]) >= DEFAULT_GEOMETRY_IOU),
                    -float(row["target_match"]["iou"]),
                    -float(row["target_match"]["margin"]),
                    str(row["target"]["object_id"]),
                    int(row["seed"]),
                )
            )
            chosen: dict[str, Any] | None = None
            forced_admission: dict[str, Any] | None = None
            for candidate in accepted:
                candidate_admission = _run_forced_complete_row_suffix(
                    session=session,
                    native_inputs=one_native,
                    prefix_token_ids=prefix_token_ids,
                    candidate_token_ids=candidate["raw_generated_token_ids"],
                    target_owner_id=str(candidate["target"]["object_id"]),
                    covered_owner_ids=record.get("covered_owner_ids", []),
                    entity_ledger=ledger,
                    tokenizer=session._tokenizer,
                    image_width=int(example.image.width),
                    image_height=int(example.image.height),
                    repetition_penalty=float(args.repetition_penalty),
                    total_generated_token_budget=int(args.total_generated_token_budget),
                    total_row_budget=int(args.total_row_budget),
                    prefix_object_row_count=int(record["prefix_object_row_count"]),
                )
                if (
                    candidate_admission.get("target_owner_retained")
                    and int(candidate_admission.get("confirmed_new_duplicate_count", 0)) == 0
                    and int(candidate_admission.get("confirmed_new_malformed_count", 0)) == 0
                    and int(candidate_admission.get("confirmed_new_unsupported_entity_count", 0)) == 0
                ):
                    chosen = candidate
                    forced_admission = candidate_admission
                    break
                attempt_rows.append({
                    "example_id": example_id,
                    "seed": int(candidate["seed"]),
                    "status": "rejected",
                    "reason": "forced_suffix_admission_harm",
                    "counterfactual_admission": candidate_admission,
                })
            if chosen is None or forced_admission is None:
                continue
            target_choice = {key: chosen[key] for key in ("target", "noncoverage_receipts", "proof_stratum")}
            rollout, review = _build_event(
                record,
                target_choice,
                chosen,
                prompt_ids=prompt_ids,
                image_pad_interval=[pad_positions[0], pad_positions[-1] + 1],
                image_id=_physical_image_id(example),
                image_path=Path(str(example.image.path)).resolve(strict=True),
                image_width=int(example.image.width),
                image_height=int(example.image.height),
                source_gt_path=args.source_gt_vs_pred.resolve(),
                checkpoint_id=checkpoint_id,
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                repetition_penalty=float(args.repetition_penalty),
                counterfactual_admission=forced_admission,
                counterfactual_diagnostics={
                    "suffix_rows": forced_admission.get("suffix_rows", []),
                    "candidate_admission_reason": forced_admission.get("reason"),
                    "source_prompt_trace": source_prompt_receipt,
                },
                split="train",
            )
            rollout_rows.append(rollout)
            review_rows.append(review)
    _write_jsonl(output_root / "rollout_rows.jsonl", rollout_rows)
    _write_jsonl(output_root / "review_rows.jsonl", review_rows)
    _write_jsonl(output_root / "sample_attempts.jsonl", attempt_rows)
    return {
        "status": "completed",
        "accepted_event_count": len(rollout_rows),
        "attempt_count": len(attempt_rows),
        "checkpoint_id": checkpoint_id,
        "runtime_model_identity": runtime_model_identity,
        "source_run_manifest": str(source_run_manifest_path),
        "source_run_manifest_sha256": sha256_file(source_run_manifest_path),
        "reference_state_bank_manifest": str(reference_manifest),
        "reference_state_bank_manifest_sha256": sha256_file(reference_manifest),
        "seeds": list(seeds),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "repetition_penalty": float(args.repetition_penalty),
    }


def main() -> int:
    args = _parse_args()
    source_gt = args.source_gt_vs_pred.resolve(strict=True)
    source_trace = args.source_token_trace.resolve(strict=True)
    output_root = args.output_dir.expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()) and not args.force:
        raise SystemExit(f"refusing to overwrite non-empty output {output_root}; use a new shard root or --force")
    requested = {item.strip() for item in args.image_ids.split(",") if item.strip()} or None
    gt_rows = load_jsonl(source_gt)
    trace_rows = load_jsonl(source_trace)
    records, exclusions = scan_terminal_records(
        gt_rows,
        trace_rows,
        image_ids=requested,
        shard_index=int(args.shard_index),
        shard_count=int(args.shard_count),
    )
    scan_receipt = {
        "schema_version": SCHEMA_VERSION,
        "mode": "scan_only" if args.scan_only else "runtime",
        "source_artifacts": {
            "gt_vs_pred": {"path": str(source_gt), "sha256": sha256_file(source_gt)},
            "pred_token_trace": {"path": str(source_trace), "sha256": sha256_file(source_trace)},
            "infer_config": {"path": str(args.infer_config.resolve()), "sha256": sha256_file(args.infer_config.resolve()) if args.infer_config.exists() else None},
            "source_run_manifest": {
                "path": str((source_gt.parent / "run_manifest.json").resolve()),
                "sha256": sha256_file(source_gt.parent / "run_manifest.json") if (source_gt.parent / "run_manifest.json").exists() else None,
            },
            "reference_state_bank_manifest": {
                "path": str(args.reference_state_bank_manifest.resolve()),
                "sha256": sha256_file(args.reference_state_bank_manifest.resolve()) if args.reference_state_bank_manifest.exists() else None,
            },
        },
        "shard": {"index": int(args.shard_index), "count": int(args.shard_count)},
        "selected_record_count": len(records),
        "exclusions": exclusions,
        "record_summaries": [
            {
                "example_id": row["example_id"],
                "prefix_object_row_count": row["prefix_object_row_count"],
                "proof_strata": sorted({str(item["proof_stratum"]) for item in row["target_candidates"]}),
                "target_candidate_count": len(row["target_candidates"]),
            }
            for row in records
        ],
    }
    _write_json(output_root / "scan.json", scan_receipt)
    if args.scan_only:
        print(json.dumps(scan_receipt, sort_keys=True))
        return 0
    runtime = collect_runtime(args, records, output_root)
    receipt = {**scan_receipt, "runtime": runtime}
    _write_json(output_root / "collection-receipt.json", receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
