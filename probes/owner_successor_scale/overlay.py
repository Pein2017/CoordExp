"""Versioned positive-only owner annotation overlay.

This module deliberately stops at a provenance-bound review/export seam.  It
does not decide whether a proposal is a physical owner, edit COCO, or admit a
new target to the training bank.  In particular, an absent review is neutral
and a worker review is not a training admission.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.eval.detection_categories import COCO_80_CATEGORY_IDS, normalize_coco_category_name


SCHEMA = "owner_successor_scale.owner_overlay.v1"
CANDIDATE_SCHEMA = "owner_successor_scale.owner_overlay_candidate.v1"
REVIEW_SCHEMA = "owner_successor_scale.owner_overlay_review_event.v1"
PATCH_SCHEMA = "owner_successor_scale.owner_positive_patch.v1"
PANEL_SCHEMA = "owner_successor_scale.owner_supplemental_validation.v1"

NATIVE_CONVENTION = "native_pixel_xyxy_half_open"
BIN_CONVENTION = "norm1000_xyxy"
STATUSES = ("candidate", "lead_reviewed_positive", "group_extent", "uncertain", "rejected")
UNCERTAINTIES = ("resolved", "uncertain", "unknown", "group_extent", "alias", "rejected")
_RESERVED_RAW_GT_KEYS = {"gt", "ground_truth", "raw_coco", "coco_annotations"}


class OverlayError(ValueError):
    """Raised when an overlay would lose provenance or violate a safety gate."""


def _fail(message: str) -> None:
    raise OverlayError(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_binding(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Return a byte identity for an existing artifact."""

    resolved = Path(path)
    _require(resolved.is_file(), f"binding path is not a file: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def verify_binding(binding: Mapping[str, Any], *, label: str = "binding") -> None:
    _require(isinstance(binding, Mapping), f"{label} must be an object")
    path = binding.get("path")
    digest = binding.get("sha256")
    size = binding.get("size_bytes")
    _require(isinstance(path, str) and path, f"{label}.path is required")
    _require(isinstance(digest, str) and len(digest) == 64, f"{label}.sha256 is required")
    _require(isinstance(size, int) and size >= 0, f"{label}.size_bytes is required")
    actual = Path(path)
    _require(actual.is_file(), f"{label} does not exist: {path}")
    _require(actual.stat().st_size == size, f"{label} size changed: {path}")
    _require(sha256_file(actual) == digest, f"{label} hash changed: {path}")


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _bbox(value: Any, *, width: int, height: int, label: str) -> list[int]:
    _require(isinstance(value, (list, tuple)) and len(value) == 4, f"{label} must be xyxy[4]")
    _require(all(_finite_number(item) for item in value), f"{label} must be finite")
    result = [int(item) for item in value]
    _require(all(float(item) == float(raw) for item, raw in zip(result, value)), f"{label} must use integer pixels")
    x1, y1, x2, y2 = result
    _require(0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height, f"{label} is outside image bounds")
    return result


def _bins(value: Any, *, label: str) -> list[int]:
    _require(isinstance(value, (list, tuple)) and len(value) == 4, f"{label} must be norm1000 xyxy[4]")
    _require(all(isinstance(item, int) and not isinstance(item, bool) for item in value), f"{label} must use ints")
    result = list(value)
    x1, y1, x2, y2 = result
    _require(0 <= x1 < x2 <= 999 and 0 <= y1 < y2 <= 999, f"{label} is outside 0..999 bounds")
    return result


def pixel_to_bins(bbox: Sequence[int], *, width: int, height: int) -> list[int]:
    """Use the same nearest integer convention as the native coordinate rows."""

    x1, y1, x2, y2 = bbox
    return [
        int(round(1000 * x1 / width)),
        int(round(1000 * y1 / height)),
        min(999, int(round(1000 * x2 / width))),
        min(999, int(round(1000 * y2 / height))),
    ]


def _image_id(value: Any) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool) and value >= 0, "image_id must be a nonnegative int")
    return value


def _iso_time(value: Any, *, label: str) -> None:
    _require(isinstance(value, str) and value.strip(), f"{label} is required")
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise OverlayError(f"{label} must be ISO-8601") from exc


def _reject_raw_gt(value: Any, *, where: str = "record") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _require(key not in _RESERVED_RAW_GT_KEYS, f"{where} must not carry raw COCO/GT field {key!r}")
            _reject_raw_gt(child, where=f"{where}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_raw_gt(child, where=f"{where}[{index}]")


def _validate_roles(roles: Mapping[str, Any], *, label: str = "roles") -> None:
    _require(isinstance(roles, Mapping), f"{label} must be an object")
    split = roles.get("source_split")
    _require(split in {"train", "val", "test", "unknown"}, f"{label}.source_split is invalid")
    for key in ("train_exposed", "eval_exposed", "confirmation"):
        _require(isinstance(roles.get(key), bool), f"{label}.{key} must be bool")
    _require(isinstance(roles.get("panel_role"), str) and roles["panel_role"], f"{label}.panel_role is required")
    if roles["confirmation"]:
        _require(roles["eval_exposed"], f"{label}.confirmation requires eval_exposed")


def validate_candidate(candidate: Mapping[str, Any], *, verify_bindings: bool = False) -> None:
    """Validate a neutral review proposal, never a training target."""

    _require(isinstance(candidate, Mapping), "candidate must be an object")
    _require(candidate.get("schema") == CANDIDATE_SCHEMA, "candidate schema mismatch")
    _require(isinstance(candidate.get("proposal_id"), str) and candidate["proposal_id"], "proposal_id is required")
    _reject_raw_gt(candidate)

    image = candidate.get("image")
    _require(isinstance(image, Mapping), "candidate.image is required")
    image_id = _image_id(image.get("image_id"))
    _require(isinstance(image.get("example_id"), str) and image["example_id"], "image.example_id is required")
    path = image.get("path")
    _require(isinstance(path, str) and path, "image.path is required")
    width, height = image.get("width"), image.get("height")
    _require(isinstance(width, int) and width > 0 and isinstance(height, int) and height > 0, "image dimensions are required")
    verify_binding(image, label="image") if verify_bindings else _validate_binding_shape(image, label="image")
    _require(isinstance(image.get("split"), str) and image["split"], "image.split is required")

    proposal = candidate.get("proposal")
    _require(isinstance(proposal, Mapping), "candidate.proposal is required")
    bbox = _bbox(proposal.get("native_bbox"), width=width, height=height, label="proposal.native_bbox")
    bins = _bins(proposal.get("coord_bins"), label="proposal.coord_bins")
    _require(proposal.get("native_coordinate_convention") == NATIVE_CONVENTION, "native coordinate convention mismatch")
    _require(proposal.get("bin_coordinate_convention") == BIN_CONVENTION, "bin coordinate convention mismatch")
    expected = pixel_to_bins(bbox, width=width, height=height)
    _require(all(abs(a - b) <= 2 for a, b in zip(expected, bins)), "pixel/binned box identity drift")
    _require(proposal.get("proposal_kind") in {"singleton", "group"}, "proposal_kind must be singleton or group")
    _require(proposal.get("owner_granularity") in {"singleton", "group", "unknown"}, "owner_granularity is required")
    _require(proposal.get("scope") in {"singleton", "group_extent", "unknown"}, "proposal scope is required")
    _require(isinstance(proposal.get("class_proposal"), str) and proposal["class_proposal"], "class_proposal is required")
    _require(isinstance(proposal.get("source_prediction_index"), int) and proposal["source_prediction_index"] >= 0, "source prediction index is required")

    provenance = candidate.get("provenance")
    _require(isinstance(provenance, Mapping), "candidate.provenance is required")
    for key in ("native_source_manifest", "native_gt_manifest", "candidate_checkpoint", "candidate_output"):
        value = provenance.get(key)
        if key == "candidate_output":
            _require(isinstance(value, Mapping), "provenance.candidate_output is required")
            _require(isinstance(value.get("row_id"), str) and value["row_id"], "candidate output row_id is required")
            _require(isinstance(value.get("row_sha256"), str) and len(value["row_sha256"]) == 64, "candidate output row hash is required")
            binding = value.get("binding")
        else:
            binding = value
        _validate_binding_shape(binding, label=f"provenance.{key}")
        if verify_bindings:
            verify_binding(binding, label=f"provenance.{key}")

    evidence = candidate.get("review_evidence")
    _require(isinstance(evidence, Mapping), "candidate.review_evidence is required")
    cards = evidence.get("cards")
    _require(isinstance(cards, list) and cards, "at least one rendered review card is required")
    for index, card in enumerate(cards):
        if verify_bindings:
            verify_binding(card, label=f"review_evidence.cards[{index}]")
        else:
            _validate_binding_shape(card, label=f"review_evidence.cards[{index}]")
    _require(evidence.get("render_route") in {"view_image", "existing_visualization_utility"}, "review render route is required")

    _validate_roles(candidate.get("roles"))
    _require(image["split"] == candidate["roles"]["source_split"], "image/role source split mismatch")
    _require(candidate.get("status") == "candidate", "candidate table status must remain candidate")
    _require(candidate.get("training_target") is False, "candidate cannot declare a training target")


def _validate_binding_shape(binding: Any, *, label: str) -> None:
    _require(isinstance(binding, Mapping), f"{label} must be a file binding")
    _require(isinstance(binding.get("path"), str) and binding["path"], f"{label}.path is required")
    _require(isinstance(binding.get("sha256"), str) and len(binding["sha256"]) == 64, f"{label}.sha256 is required")
    _require(isinstance(binding.get("size_bytes"), int) and binding["size_bytes"] >= 0, f"{label}.size_bytes is required")


def validate_review_event(event: Mapping[str, Any]) -> None:
    """Validate one append-only review event, including lead-gating."""

    _require(isinstance(event, Mapping), "review event must be an object")
    _require(event.get("schema") == REVIEW_SCHEMA, "review event schema mismatch")
    for key in ("review_id", "proposal_id", "reviewer_id", "reviewer_route"):
        _require(isinstance(event.get(key), str) and event[key], f"review event {key} is required")
    _require(event["reviewer_route"] == "view_image", "review route must be view_image")
    _require(isinstance(event.get("review_seq"), int) and event["review_seq"] >= 0, "review_seq is required")
    _iso_time(event.get("reviewed_at"), label="reviewed_at")
    status = event.get("status")
    _require(status in STATUSES, f"unknown review status: {status!r}")
    _require(event.get("reviewer_role") in {"worker", "lead"}, "reviewer_role must be worker or lead")
    _require(isinstance(event.get("lead_accepted"), bool), "lead_accepted must be bool")
    _require(
        event["lead_accepted"] is (event["reviewer_role"] == "lead" and status in {"lead_reviewed_positive", "group_extent"}),
        "lead acceptance/status mismatch",
    )
    _require(event.get("owner_granularity") in {"singleton", "group", "unknown"}, "owner_granularity is required")
    _require(event.get("scope") in {"singleton", "group_extent", "unknown"}, "review scope is required")
    for key in ("instance_entity", "geometry", "class"):
        _require(event.get(key) in UNCERTAINTIES, f"review uncertainty {key} is invalid")
    if status == "lead_reviewed_positive":
        _require(all(event[key] == "resolved" for key in ("instance_entity", "geometry", "class")), "positive review must resolve all uncertainty axes")
        _require(event["owner_granularity"] == "singleton" and event["scope"] == "singleton", "atomic positive must be a singleton")
        _require(isinstance(event.get("owner_id"), str) and event["owner_id"], "positive review requires owner_id")
    elif status == "group_extent":
        _require(event["instance_entity"] == "resolved" and event["geometry"] == "group_extent" and event["class"] in {"resolved", "uncertain"}, "group extent must keep granularity and extent separate")
        _require(event["owner_granularity"] == "group" and event["scope"] == "group_extent", "group extent review must declare group scope")
        _require(isinstance(event.get("owner_id"), str) and event["owner_id"], "accepted group extent requires owner_id")
    else:
        _require(event.get("owner_id") in (None, ""), "non-positive review cannot assign owner_id")
    parent = event.get("parent_proposal_id")
    _require(parent is None or (isinstance(parent, str) and parent), "parent_proposal_id must be null or proposal ID")
    children = event.get("child_proposal_ids", [])
    _require(isinstance(children, list) and all(isinstance(item, str) and item for item in children), "child_proposal_ids must be proposal IDs")
    alias = event.get("same_image_alias_of")
    _require(alias is None or (isinstance(alias, str) and alias), "same_image_alias_of must be null or proposal ID")
    evidence = event.get("evidence")
    _require(isinstance(evidence, list) and evidence, "review evidence is required")
    for index, item in enumerate(evidence):
        _validate_binding_shape(item, label=f"evidence[{index}]")
    _reject_raw_gt(event)


def _validate_candidate_set(candidates: Iterable[Mapping[str, Any]], *, verify_bindings: bool = False) -> dict[str, Mapping[str, Any]]:
    by_id: dict[str, Mapping[str, Any]] = {}
    for candidate in candidates:
        validate_candidate(candidate, verify_bindings=verify_bindings)
        proposal_id = str(candidate["proposal_id"])
        _require(proposal_id not in by_id, f"duplicate proposal_id: {proposal_id}")
        by_id[proposal_id] = candidate
    return by_id


def build_candidate_manifest(
    candidates: Sequence[Mapping[str, Any]],
    *,
    confirmation_selection: Mapping[str, Any] | None = None,
    verify_bindings: bool = False,
) -> dict[str, Any]:
    """Build a small review table; this function never creates target labels."""

    _require(8 <= len(candidates) <= 12, "review-ready candidate table must contain 8..12 proposals")
    by_id = _validate_candidate_set(candidates, verify_bindings=verify_bindings)
    image_ids = {int(item["image"]["image_id"]) for item in by_id.values()}
    _require(len(image_ids) <= 3, "candidate table may span at most three images")
    if confirmation_selection is not None:
        _validate_selection_shape(confirmation_selection)
    records = [by_id[key] for key in sorted(by_id)]
    return {
        "schema": SCHEMA,
        "version": 1,
        "status": "candidate_review_pending",
        "unit": "2026-09-13-owner-successor-scale-throughput",
        "records": records,
        "candidate_count": len(records),
        "image_ids": sorted(image_ids),
        "review_owner": "root",
        "admission_owner": "root",
        "review_status": "lead_review_required",
        "training_export": "blocked_until_root_lead_admission",
        "selection_conditioned": True,
        "claim_boundary": "Review aid only; proposals are not GT, not exhaustive scene annotations, and not negative labels. Unknown/unreviewed remains neutral.",
        "raw_coco_policy": "raw COCO labels remain unchanged; overlay is supplemental and positive-only",
        "confirmation_selection": confirmation_selection,
        "source_blind_cards": True,
    }


def write_new_json(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> None:
    """Write a versioned artifact without silently replacing an earlier one."""

    destination = Path(path)
    _require(not destination.exists(), f"refusing to overwrite existing artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def append_review_event(path: str | os.PathLike[str], event: Mapping[str, Any]) -> None:
    """Append exactly one review event; duplicate IDs/sequences are refused."""

    validate_review_event(event)
    destination = Path(path)
    existing_ids: set[str] = set()
    existing_keys: set[tuple[str, int]] = set()
    latest_seq: dict[str, int] = {}
    if destination.exists():
        with destination.open() as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    prior = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise OverlayError(f"invalid review JSONL at {destination}:{line_number}") from exc
                validate_review_event(prior)
                existing_ids.add(prior["review_id"])
                existing_keys.add((prior["proposal_id"], prior["review_seq"]))
                latest_seq[prior["proposal_id"]] = max(latest_seq.get(prior["proposal_id"], -1), prior["review_seq"])
    _require(event["review_id"] not in existing_ids, f"duplicate review_id: {event['review_id']}")
    _require((event["proposal_id"], event["review_seq"]) not in existing_keys, "duplicate proposal review sequence")
    _require(event["review_seq"] > latest_seq.get(event["proposal_id"], -1), "review_seq must append monotonically per proposal")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a") as handle:
        handle.write(json.dumps(dict(event), ensure_ascii=False, sort_keys=True) + "\n")


def load_review_events(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise OverlayError(f"invalid review JSONL at {path}:{line_number}") from exc
            validate_review_event(event)
            events.append(event)
    return events


def _latest_events(events: Iterable[Mapping[str, Any]], candidate_ids: set[str]) -> dict[str, Mapping[str, Any]]:
    latest: dict[str, Mapping[str, Any]] = {}
    seen_ids: set[str] = set()
    seen_seq: set[tuple[str, int]] = set()
    for event in events:
        validate_review_event(event)
        _require(event["proposal_id"] in candidate_ids, f"review references unknown proposal: {event['proposal_id']}")
        _require(event["review_id"] not in seen_ids, f"duplicate review_id: {event['review_id']}")
        key = (event["proposal_id"], event["review_seq"])
        _require(key not in seen_seq, f"duplicate review sequence: {key}")
        seen_ids.add(event["review_id"])
        seen_seq.add(key)
        prior = latest.get(event["proposal_id"])
        if prior is None or event["review_seq"] > prior["review_seq"]:
            latest[event["proposal_id"]] = event
    return latest


def _validate_selection_shape(selection: Mapping[str, Any]) -> None:
    _require(isinstance(selection, Mapping), "confirmation selection must be an object")
    _require(
        selection.get("schema") == "native_owner_successor_scale_throughput.confirmation_selection.v1",
        "frozen confirmation selection schema is required",
    )
    _require(selection.get("status") == "frozen_cpu_no_model_calls", "confirmation selection is not frozen")
    _require(isinstance(selection.get("image_ids"), list), "confirmation selection image_ids missing")
    _require(isinstance(selection.get("excluded_image_ids"), list), "confirmation selection excluded_image_ids missing")
    _require(all(isinstance(value, int) for value in selection["image_ids"]), "confirmation image IDs must be ints")
    _require(all(isinstance(value, int) for value in selection["excluded_image_ids"]), "excluded image IDs must be ints")
    expected = sha256_bytes(canonical_json(selection["image_ids"]).encode("utf-8"))
    _require(selection.get("image_ids_sha256") == expected, "confirmation image identity digest mismatch")


def _training_role_guard(
    candidates: Mapping[str, Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    *,
    confirmation_selection: Mapping[str, Any] | None,
) -> None:
    _require(confirmation_selection is not None, "frozen confirmation selection is required for training export")
    selection_ids: set[int] = set()
    excluded_ids: set[int] = set()
    _validate_selection_shape(confirmation_selection)
    selection_ids = set(confirmation_selection["image_ids"])
    excluded_ids = set(confirmation_selection["excluded_image_ids"])
    for record in records:
        proposal_id = record["proposal_id"]
        candidate = candidates[proposal_id]
        image = candidate["image"]
        roles = candidate["roles"]
        image_id = int(image["image_id"])
        _require(roles["source_split"] == "train", f"training export requires train source split: {proposal_id}")
        _require(roles["train_exposed"] is True, f"training export requires explicit exposed-train role: {proposal_id}")
        _require(roles["eval_exposed"] is False and roles["confirmation"] is False, f"evaluation/confirmation image cannot train-export: {proposal_id}")
        _require(image_id not in selection_ids, f"confirmation image leaked into training export: {image_id}")
        # The frozen exclusion chain mixes train and eval-origin IDs.  A train
        # row is allowed only because its role above is explicit; val/test rows
        # are rejected instead of blindly rejecting every excluded ID.
        if image_id in excluded_ids:
            _require(roles["source_split"] == "train" and roles["train_exposed"], f"eval-origin excluded image leaked into training export: {image_id}")


def _positive_records(
    candidates: Mapping[str, Mapping[str, Any]],
    events: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    latest = _latest_events(events, set(candidates))
    positives: list[dict[str, Any]] = []
    counts = {status: 0 for status in STATUSES}
    counts["unreviewed"] = len(set(candidates) - set(latest))
    owner_keys: set[tuple[int, str]] = set()
    for proposal_id, event in latest.items():
        status = event["status"]
        counts[status] += 1
        if status not in {"lead_reviewed_positive", "group_extent"} or event["lead_accepted"] is not True:
            continue
        candidate = candidates[proposal_id]
        image_id = int(candidate["image"]["image_id"])
        owner_key = (image_id, str(event["owner_id"]))
        _require(owner_key not in owner_keys, f"duplicate same-image owner/alias target: {owner_key}")
        owner_keys.add(owner_key)
        alias = event.get("same_image_alias_of")
        if alias is not None:
            _require(alias in candidates, f"alias references unknown proposal: {alias}")
            _require(int(candidates[alias]["image"]["image_id"]) == image_id, "same-image alias crosses image boundary")
            _fail("same-image alias cannot export as a separate atomic target; adjudicate one canonical owner first")
        parent = event.get("parent_proposal_id")
        if parent is not None:
            _require(parent in candidates, f"parent references unknown proposal: {parent}")
            _require(int(candidates[parent]["image"]["image_id"]) == image_id, "group parent crosses image boundary")
        for child in event.get("child_proposal_ids", []):
            _require(child in candidates, f"child references unknown proposal: {child}")
            _require(int(candidates[child]["image"]["image_id"]) == image_id, "group child crosses image boundary")
        proposed_class = normalize_coco_category_name(candidate["proposal"]["class_proposal"])
        canonical_class: str | None = None
        if event["class"] == "resolved":
            _require(
                proposed_class in COCO_80_CATEGORY_IDS,
                f"resolved positive class is outside canonical COCO-80: {candidate['proposal']['class_proposal']!r}",
            )
            canonical_class = proposed_class
        # Copy only the overlay fields.  This makes raw COCO GT leakage into a
        # supplemental/training target mechanically impossible.
        positives.append(
            {
                "proposal_id": proposal_id,
                "image": dict(candidate["image"]),
                "proposal": dict(candidate["proposal"]),
                "provenance": dict(candidate["provenance"]),
                "review_evidence": dict(candidate["review_evidence"]),
                "roles": dict(candidate["roles"]),
                "status": status,
                "owner_id": event["owner_id"],
                "same_image_alias_of": alias,
                "owner_granularity": event["owner_granularity"],
                "scope": event["scope"],
                "canonical_class": canonical_class,
                "parent_proposal_id": parent,
                "child_proposal_ids": list(event.get("child_proposal_ids", [])),
                "uncertainty": {key: event[key] for key in ("instance_entity", "geometry", "class")},
                "review": {
                    "review_id": event["review_id"],
                    "reviewer_id": event["reviewer_id"],
                    "reviewer_role": event["reviewer_role"],
                    "reviewer_route": event["reviewer_route"],
                    "reviewed_at": event["reviewed_at"],
                },
            }
        )
    _reject_raw_gt(positives)
    positives.sort(key=lambda item: item["proposal_id"])
    return positives, counts


def export_positive_patch(
    candidates: Sequence[Mapping[str, Any]],
    events: Iterable[Mapping[str, Any]],
    output_path: str | os.PathLike[str],
    *,
    confirmation_selection: Mapping[str, Any] | None = None,
    immutable: bool = True,
) -> dict[str, Any]:
    """Export only root-lead accepted positives; unknowns and workers stay out."""

    by_id = _validate_candidate_set(candidates, verify_bindings=False)
    positives, counts = _positive_records(by_id, events)
    payload = {
        "schema": PATCH_SCHEMA,
        "version": 1,
        "status": "lead_reviewed_owner_patch",
        "records": positives,
        "positive_count": len(positives),
        "atomic_singleton_count": sum(item["status"] == "lead_reviewed_positive" for item in positives),
        "group_extent_count": sum(item["status"] == "group_extent" for item in positives),
        "review_counts": counts,
        "unreviewed_count": counts["unreviewed"],
        "unknown_is_neutral": True,
        "negative_count": 0,
        "training_export": "not_authorized_by_this_patch",
        "confirmation_selection": confirmation_selection,
        "claim_boundary": "Supplemental positive-only owner overlay; no exhaustive annotation, negative labels, COCO mutation, or independent benchmark claim.",
    }
    if immutable:
        write_new_json(output_path, payload)
    else:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    return payload


def export_training_patch(
    candidates: Sequence[Mapping[str, Any]],
    events: Iterable[Mapping[str, Any]],
    output_path: str | os.PathLike[str],
    *,
    admission: Mapping[str, Any] | None,
    confirmation_selection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Require the root-owned admission gate before any training export."""

    _require(isinstance(admission, Mapping) and admission.get("status") == "lead_accepted", "root lead admission is required for training export")
    by_id = _validate_candidate_set(candidates, verify_bindings=False)
    positives, counts = _positive_records(by_id, events)
    _training_role_guard(by_id, positives, confirmation_selection=confirmation_selection)
    for record in positives:
        _require(record["status"] == "lead_reviewed_positive", "group extent is evidence only and cannot enter training export")
        _require(record["owner_granularity"] == "singleton" and record["scope"] == "singleton", "training export requires atomic singleton owners")
    payload = {
        "schema": PATCH_SCHEMA,
        "version": 1,
        "status": "training_target_patch_root_admitted",
        "records": positives,
        "positive_count": len(positives),
        "review_counts": counts,
        "unreviewed_count": counts["unreviewed"],
        "admission": dict(admission),
        "confirmation_selection": confirmation_selection,
        "raw_coco_unchanged": True,
        "claim_boundary": "Only root-admitted positive singleton overlay records; exact-history/bank admission remains a separate gate.",
    }
    write_new_json(output_path, payload)
    return payload


def export_supplemental_validation(
    candidates: Sequence[Mapping[str, Any]],
    events: Iterable[Mapping[str, Any]],
    output_path: str | os.PathLike[str],
    *,
    baseline_raw_metrics: Mapping[str, Any],
    confirmation_selection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a separately reported, selection-conditioned validation panel."""

    _require(isinstance(baseline_raw_metrics, Mapping) and baseline_raw_metrics, "baseline raw metrics must be preserved")
    by_id = _validate_candidate_set(candidates, verify_bindings=False)
    positives, counts = _positive_records(by_id, events)
    payload = {
        "schema": PANEL_SCHEMA,
        "version": 1,
        "status": "supplemental_known_positive_selection_conditioned",
        "records": positives,
        "positive_count": len(positives),
        "review_counts": counts,
        "unreviewed_count": counts["unreviewed"],
        "selection_conditioned": True,
        "independent_benchmark": False,
        "original_coco_metrics_untouched": True,
        "baseline_raw_metrics": dict(baseline_raw_metrics),
        "confirmation_selection": confirmation_selection,
        "claim_boundary": "Reviewed positives are a supplemental selected panel, not exhaustive recall or an independent COCO benchmark.",
    }
    write_new_json(output_path, payload)
    return payload


def training_export_is_safe(
    candidates: Sequence[Mapping[str, Any]],
    events: Iterable[Mapping[str, Any]],
    *,
    admission: Mapping[str, Any] | None,
    confirmation_selection: Mapping[str, Any] | None = None,
) -> bool:
    """Pure guard useful to callers that want a preflight before writing."""

    if not isinstance(admission, Mapping) or admission.get("status") != "lead_accepted":
        return False
    if confirmation_selection is None:
        return False
    try:
        by_id = _validate_candidate_set(candidates)
        positives, _ = _positive_records(by_id, events)
        _training_role_guard(by_id, positives, confirmation_selection=confirmation_selection)
    except OverlayError:
        return False
    return True


__all__ = [
    "BIN_CONVENTION",
    "CANDIDATE_SCHEMA",
    "NATIVE_CONVENTION",
    "OverlayError",
    "PANEL_SCHEMA",
    "PATCH_SCHEMA",
    "REVIEW_SCHEMA",
    "SCHEMA",
    "STATUSES",
    "append_review_event",
    "build_candidate_manifest",
    "canonical_json",
    "export_positive_patch",
    "export_supplemental_validation",
    "export_training_patch",
    "file_binding",
    "load_review_events",
    "pixel_to_bins",
    "sha256_file",
    "training_export_is_safe",
    "validate_candidate",
    "validate_review_event",
    "verify_binding",
    "write_new_json",
]
