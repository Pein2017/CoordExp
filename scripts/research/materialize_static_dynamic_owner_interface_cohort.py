#!/usr/bin/env python3
"""Materialize the frozen static/dynamic owner-interface cohort.

This module is deliberately a CPU-only registry seam.  It consumes an admitted
panel, its sealed ``geo_sorted_xy`` derived view, and checkpoint-specific
H0/support ledgers; it never launches inference, touches model state, or fills
missing evidence from the val200 run.  Missing or ambiguous evidence is
retained as an explicit ``indeterminate`` disposition.

Every H0 or support input is one JSON envelope with schema
``static_dynamic_native_h0_owner_ledger.v1``.  The exact machine-readable
contract is exported as :data:`LEDGER_ENVELOPE_CONTRACT`.  In brief, the
envelope binds ``unit_id``, checkpoint, resolved config fingerprint, source and
derived panel hashes, ``run_kind=native_h0``, and complete-history attestation.
Each record binds one owner by ``coco_ann_id`` or unique
``(category, pixel_bbox)``, one natural boundary, and one exact-prefix SHA-256.
Native H0 records explicitly declare support ``not_measured`` and may not carry
a support value.  Measured support records additionally must match an existing
H0 record on checkpoint, owner, pre-STOP boundary, prefix, and config; an
intervention, terminal ``im_end``, or future prefix is never accepted as
support.

The public :func:`materialize_cohort` function accepts paths or already-loaded
JSON values, making the contract easy to exercise with small CPU fixtures.  A
JSON cohort and a separate deterministic manifest are written only when paths
are explicitly supplied by the caller.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "static_dynamic_owner_interface_cohort.v1"
LEDGER_SCHEMA_VERSION = "static_dynamic_native_h0_owner_ledger.v1"
UNIT_ID = "2026-08-05-static-dynamic-owner-interface-crossover"
MIN_EVENTS = 24
MAX_EVENTS = 32
IMAGE_2299 = 2299
EXPECTED_IMAGE_IDS = (2299, 4134, 5001, 6040, 7511, 10707, 14038, 16228)
REQUIRED_GEOMETRY_STRATA = ("SS", "SO", "XC")
TP_PREFIX_SEMANTICS = "before_queried_owner_row"
FN_PREFIX_SEMANTICS = "after_strict_covered_row_pre_stop"
INVALID_PREFIX_SEMANTICS = "no_valid_post_covered_boundary"

LEDGER_ENVELOPE_CONTRACT: dict[str, Any] = {
    "schema_version": LEDGER_SCHEMA_VERSION,
    "required_envelope_fields": {
        "unit_id": UNIT_ID,
        "checkpoint": ["S", "A"],
        "config_fingerprint": "non-empty string; constant within checkpoint",
        "source_panel_sha256": "admitted source panel SHA-256",
        "derived_panel_sha256": "sealed geo_sorted_xy panel SHA-256",
        "run_kind": "native_h0",
        "history_complete": True,
        "records": "non-empty list",
    },
    "optional_envelope_fields": {"arm": ["native", "H0", "h0"]},
    "required_record_fields": {
        "image_id": "integer",
        "owner_identity": [
            "coco_ann_id",
            "unique category_name/desc plus pixel_bbox/bbox_pixel_xyxy",
        ],
        "natural_boundary": "checkpoint-native boundary identity/index",
        "exact_prefix_sha256": "64-character SHA-256",
        "exact_prefix_token_ids": "non-negative integer list hashing to exact_prefix_sha256",
        "prefix_semantics": [
            TP_PREFIX_SEMANTICS,
            FN_PREFIX_SEMANTICS,
            INVALID_PREFIX_SEMANTICS,
        ],
        "due_boundary_evidence": "mapping bound to queried/covered owners and pre-STOP steps",
        "excludes_stop": True,
        "queried_owner_not_covered": True,
    },
    "h0_record_support_contract": {
        "native_tp": "required JSON boolean",
        "native_fn": "required JSON boolean; exactly one of native_tp/native_fn is true",
        "strict_complete_row": "required JSON boolean; must equal native_tp",
        "natural_boundary_valid": "required JSON boolean",
        "tp_prefix": TP_PREFIX_SEMANTICS,
        "fn_prefix": FN_PREFIX_SEMANTICS,
        "fn_due_boundary_fields": [
            "latest_covered_owner_id",
            "non-empty covered_owner_ids containing latest owner and excluding queried owner",
            "due_boundary_index matching natural_boundary",
        ],
        "support_status": "not_measured",
        "verified_support_claim": False,
        "verified_support": "forbidden, including aliases",
    },
    "support_record_contract": {
        "support_status": "measured",
        "verified_support_claim": True,
        "verified_support": "required JSON boolean independent of native TP/FN outcome",
        "native_outcome_fields": "required complete boolean group matching H0",
    },
    "b_eligibility_fields": {
        "native_fn": True,
        "natural_boundary_valid": True,
        "strict_complete_row": False,
        "verified_support": "true from a same-prefix measured support record",
        "covered_owner_ids": "optional admitted-source gt:<image>:<source_index> list; must exclude B",
    },
    "support_rule": (
        "same checkpoint, config, source/derived panels, owner, natural boundary, "
        "and exact-prefix SHA-256 as an existing native H0 record"
    ),
}


class CohortContractError(ValueError):
    """Raised for malformed inputs or a violated frozen-cohort contract."""


def _candidate(
    image_id: int,
    owner_index: int,
    category: str,
    bbox: Sequence[int],
    labels: Sequence[str],
) -> dict[str, Any]:
    return {
        "image_id": image_id,
        "source_panel_object_index": owner_index,
        "gt_owner_id": f"gt:{image_id}:{owner_index}",
        "category": category,
        "pixel_bbox": list(bbox),
        "historical_labels": list(labels),
    }


# This is the preregistered 32-owner pool.  Keep order stable: it is part of
# the deterministic registry identity and is also the fallback tie breaker.
FROZEN_CANDIDATES: tuple[dict[str, Any], ...] = (
    _candidate(2299, 1, "person", (519, 51, 632, 263), ("TP",)),
    _candidate(2299, 11, "tie", (452, 148, 467, 172), ("A3R",)),
    _candidate(2299, 15, "person", (990, 188, 1110, 375), ("FN_DIAGNOSTIC",)),
    _candidate(2299, 2, "person", (760, 63, 865, 262), ("NK16",)),
    _candidate(4134, 28, "tie", (774, 376, 850, 710), ("TP",)),
    _candidate(4134, 22, "tie", (629, 288, 644, 324), ("SUP", "NK16", "SS")),
    _candidate(4134, 29, "wine glass", (472, 389, 488, 423), ("SUP", "A3R", "SS")),
    _candidate(4134, 13, "person", (1000, 276, 1043, 354), ("SUP", "A3R")),
    _candidate(5001, 0, "person", (764, 41, 846, 169), ("TP",)),
    _candidate(5001, 10, "person", (1085, 108, 1151, 222), ("SUP", "A3R", "SS")),
    _candidate(5001, 15, "person", (74, 172, 267, 854), ("NK16",)),
    _candidate(5001, 17, "bicycle", (901, 286, 984, 466), ("SUP", "A3R", "XC")),
    _candidate(6040, 11, "person", (621, 422, 668, 473), ("TP",)),
    _candidate(6040, 13, "person", (785, 424, 816, 469), ("SUP", "A3R", "SO")),
    _candidate(6040, 10, "person", (265, 422, 281, 441), ("SUP", "NK16")),
    _candidate(6040, 4, "person", (1130, 381, 1200, 481), ("NK16",)),
    _candidate(7511, 0, "kite", (370, 229, 399, 290), ("TP",)),
    _candidate(7511, 14, "person", (1064, 489, 1074, 520), ("A3R",)),
    _candidate(7511, 1, "kite", (999, 305, 1018, 326), ("SUP", "NK16", "XC")),
    _candidate(7511, 3, "person", (45, 466, 54, 481), ("NK16",)),
    _candidate(10707, 1, "remote", (772, 84, 806, 206), ("TP",)),
    _candidate(10707, 16, "bottle", (272, 661, 318, 751), ("SUP", "NK16", "XC")),
    _candidate(10707, 11, "bottle", (293, 627, 327, 736), ("TP", "A3R")),
    _candidate(10707, 9, "bottle", (416, 602, 456, 670), ("NK16",)),
    _candidate(14038, 0, "potted plant", (971, 75, 1085, 304), ("TP", "A3R")),
    _candidate(14038, 19, "book", (1006, 452, 1051, 467), ("SUP", "NK16", "SO")),
    _candidate(14038, 23, "book", (985, 480, 1060, 502), ("SUP", "NK16", "SO")),
    _candidate(14038, 12, "book", (1007, 372, 1073, 389), ("A3R", "FN_AMBIGUITY")),
    _candidate(16228, 0, "umbrella", (663, 245, 907, 331), ("TP",)),
    _candidate(16228, 11, "person", (1036, 315, 1079, 372), ("SUP", "A3R", "XC")),
    _candidate(16228, 38, "person", (90, 383, 118, 410), ("SUP", "NK16", "SO")),
    _candidate(16228, 47, "person", (944, 412, 1029, 539), ("SUP", "A3R", "SS")),
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _read_source(source: str | Path | Mapping[str, Any] | Sequence[Any]) -> tuple[Any, dict[str, Any]]:
    if isinstance(source, (str, Path)):
        path = Path(source).expanduser().resolve(strict=True)
        raw = path.read_bytes()
        if path.suffix.lower() == ".jsonl":
            rows = []
            for line_no, line in enumerate(raw.splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise CohortContractError(f"{path}:{line_no}: invalid JSON") from exc
                if not isinstance(row, Mapping):
                    raise CohortContractError(f"{path}:{line_no}: JSONL row is not an object")
                rows.append(dict(row))
            value: Any = rows
            row_count = len(rows)
        else:
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise CohortContractError(f"{path}: invalid JSON") from exc
            row_count = len(value) if isinstance(value, list) else None
        return value, {"path": str(path), "sha256": sha256_bytes(raw), "row_count": row_count}
    if isinstance(source, Mapping):
        value = dict(source)
    elif isinstance(source, Sequence) and not isinstance(source, (str, bytes, bytearray)):
        value = list(source)
    else:
        raise TypeError("source must be a path, mapping, or sequence")
    return value, {"inline": True, "sha256": sha256_json(value), "row_count": len(value) if isinstance(value, list) else None}


def _is_record(value: Mapping[str, Any]) -> bool:
    keys = set(value)
    return bool(
        keys & {
            "coco_ann_id",
            "owner_id",
            "gt_owner_id",
            "owner_key",
            "bbox_pixel_xyxy",
            "pixel_bbox",
            "bbox_2d",
            "strict_complete_row",
            "native_tp",
            "verified_support",
        }
    ) and ("image_id" in keys or "checkpoint" in keys or "checkpoint_id" in keys)


def _flatten_records(value: Any, context: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Flatten common image/checkpoint/owner ledger wrappers without guessing data."""

    inherited = dict(context or {})
    if isinstance(value, Mapping):
        local = dict(inherited)
        for key in (
            "image_id",
            "checkpoint",
            "checkpoint_id",
            "model",
            "row_boundary_index",
            "natural_boundary",
            "width",
            "height",
            "exact_prefix_sha256",
            "config_fingerprint",
            "history_complete",
            "run_kind",
            "unit_id",
            "source_panel_sha256",
            "panel_sha256",
            "derived_panel_sha256",
            "image_plan_identity",
            "image_grid_thw",
            "observed_image_grid_thw",
            "expected_image_grid_thw",
            "merge_size",
            "spatial_merge_size",
            "merged_visual_tokens",
            "declared_width",
            "declared_height",
            "decoded_width",
            "decoded_height",
        ):
            if key in value:
                local[key] = value[key]
        if _is_record(value) or (
            ("image_id" in local or "checkpoint" in local or "checkpoint_id" in local)
            and bool(
                set(value)
                & {
                    "coco_ann_id",
                    "owner_id",
                    "gt_owner_id",
                    "owner_key",
                    "bbox_pixel_xyxy",
                    "pixel_bbox",
                    "bbox_2d",
                    "strict_complete_row",
                    "native_tp",
                    "verified_support",
                }
            )
        ):
            return [{**local, **dict(value)}]
        records: list[dict[str, Any]] = []
        preferred = ("records", "rows", "owners", "owner_records", "events", "entries", "ledger", "primary", "h0", "support")
        consumed: set[str] = set()
        for key in preferred:
            if key in value:
                consumed.add(key)
                child_context = dict(local)
                if key not in {"records", "rows", "owners", "owner_records", "events", "entries", "ledger", "primary", "h0", "support"}:
                    child_context.setdefault("checkpoint", key)
                records.extend(_flatten_records(value[key], child_context))
        if records:
            return records
        for key, child in value.items():
            if key in consumed:
                continue
            if isinstance(child, (Mapping, list)):
                child_context = dict(local)
                if isinstance(key, str) and key.upper() in {"S", "A", "P", "PLAIN", "A3", "STEP-2444", "STEP-2445"}:
                    child_context.setdefault("checkpoint", key)
                records.extend(_flatten_records(child, child_context))
        return records
    if isinstance(value, list):
        records: list[dict[str, Any]] = []
        for child in value:
            records.extend(_flatten_records(child, inherited))
        return records
    return []


def _norm_checkpoint(value: Any) -> str | None:
    if value is None:
        return None
    token = str(value).strip().lower().replace("_", "-")
    if token in {"s", "plain", "primary", "step-2444", "2444", "native"}:
        return "S"
    if token in {"a", "a3", "step-2445", "2445", "commit"}:
        return "A"
    return str(value)


def _image_id(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result


def _category(record: Mapping[str, Any]) -> str | None:
    for key in ("category_name", "category", "desc", "description", "class_name"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return None


def _raw_bbox(record: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    value: Any = None
    for key in ("pixel_bbox", "bbox_pixel_xyxy", "bbox_xyxy", "bbox", "bbox_2d"):
        if key in record:
            value = record[key]
            break
    if isinstance(value, Mapping):
        value = value.get("xyxy", value.get("pixel_xyxy", value.get("coords")))
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)) or len(value) != 4:
        return None
    try:
        if all(isinstance(item, str) and item.startswith("<|coord_") for item in value):
            bins = [int(item[len("<|coord_") : -2]) for item in value]
            width = float(record.get("width", 1000))
            height = float(record.get("height", 1000))
            coords = tuple(
                float(bin_value * (width if axis % 2 == 0 else height) / 1000.0)
                for axis, bin_value in enumerate(bins)
            )
        else:
            coords = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in coords):
        return None
    return coords if coords[2] >= coords[0] and coords[3] >= coords[1] else None


def _bbox(record: Mapping[str, Any]) -> tuple[int, int, int, int] | None:
    """Return the launch/support identity box with historical rounding."""

    raw = _raw_bbox(record)
    if raw is None:
        return None
    return tuple(int(round(value)) for value in raw)  # type: ignore[return-value]


def _ann_id(record: Mapping[str, Any]) -> str | int | None:
    for key in ("coco_ann_id", "annotation_id", "ann_id"):
        value = record.get(key)
        if value is not None and not isinstance(value, bool):
            return value
    return None


def _as_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _flag(record: Mapping[str, Any], keys: Sequence[str]) -> bool | None:
    for key in keys:
        if key in record:
            parsed = _as_bool(record[key])
            if parsed is not None:
                return parsed
    nested = record.get("status")
    if isinstance(nested, Mapping):
        return _flag(nested, keys)
    return None


FLAG_KEYS: dict[str, tuple[str, ...]] = {
    "native_tp": ("native_tp", "native_true_positive", "is_native_tp", "greedy_tp", "tp"),
    "native_fn": ("native_fn", "native_false_negative", "is_native_fn", "greedy_fn", "fn"),
    "strict_complete_row": (
        "strict_complete_row", "strict_row", "strict_match", "strict_owner_match",
        "complete_row", "source_specific_complete_row", "owner_specific_complete_row",
    ),
    "verified_support": (
        "verified_support", "support_verified", "tested_localization_support",
        "localization_support", "positive_support", "has_tested_support", "support",
    ),
    "natural_boundary_valid": (
        "natural_boundary_valid", "boundary_valid", "has_natural_boundary", "natural_prefix", "natural",
    ),
}


def _boundary(record: Mapping[str, Any]) -> Any:
    for key in ("natural_boundary", "natural_row_boundary", "row_boundary_index", "boundary_index", "row_boundary", "boundary"):
        if key in record:
            return record[key]
    return None


def _boundary_key(value: Any) -> tuple[int, str]:
    if isinstance(value, bool):
        return (0, str(int(value)))
    if isinstance(value, (int, float)):
        return (1, f"{float(value):020.9f}")
    if isinstance(value, Mapping):
        for key in ("index", "row_index", "boundary_index", "token_index"):
            if key in value:
                return _boundary_key(value[key])
    return (2, str(value)) if value is not None else (-1, "")


def _owner_ref(record: Mapping[str, Any]) -> Any:
    for key in ("owner_id", "gt_owner_id", "owner_key", "target_owner_id"):
        if record.get(key) is not None:
            return record[key]
    return None


@dataclass(frozen=True)
class PanelOwner:
    image_id: int
    source_index: int
    category: str | None
    bbox: tuple[int, int, int, int] | None
    coco_ann_id: str | int | None
    gt_owner_id: str
    derived_index: int | None = None
    derived_mapping_method: str | None = None
    raw_bbox: tuple[float, float, float, float] | None = None


def _panel_owners(panel: Any) -> dict[int, list[PanelOwner]]:
    rows = panel if isinstance(panel, list) else [panel]
    result: dict[int, list[PanelOwner]] = {}
    for row_index, row in enumerate(rows, 1):
        if not isinstance(row, Mapping):
            raise CohortContractError(f"panel row {row_index} is not an object")
        image = _image_id(row.get("image_id"))
        if image is None:
            raise CohortContractError(f"panel row {row_index} has no integer image_id")
        objects = row.get("objects")
        if not isinstance(objects, list):
            raise CohortContractError(f"panel image {image} has no objects list")
        if image in result:
            raise CohortContractError(f"panel has duplicate image_id={image}")
        image_owners: list[PanelOwner] = []
        for index, obj in enumerate(objects):
            if not isinstance(obj, Mapping):
                continue
            payload = {
                **obj,
                "width": row.get("width", row.get("image_width", 1000)),
                "height": row.get("height", row.get("image_height", 1000)),
            }
            image_owners.append(
                PanelOwner(
                    image,
                    index,
                    _category(obj),
                    _bbox(payload),
                    _ann_id(obj),
                    f"gt:{image}:{index}",
                    raw_bbox=_raw_bbox(payload),
                )
            )
        result[image] = image_owners
        if len(result[image]) != len(objects):
            raise CohortContractError(f"panel image {image} contains a non-object owner")
    return result


def _panel_rows(panel: Any, *, label: str) -> dict[int, Mapping[str, Any]]:
    rows = panel if isinstance(panel, list) else [panel]
    result: dict[int, Mapping[str, Any]] = {}
    for row_index, row in enumerate(rows, 1):
        if not isinstance(row, Mapping):
            raise CohortContractError(f"{label} row {row_index} is not an object")
        image = _image_id(row.get("image_id"))
        if image is None or image in result:
            raise CohortContractError(f"{label} has invalid or duplicate image_id at row {row_index}")
        result[image] = row
    return result


def _validate_derived_panel(
    source_panel: Any,
    derived_panel: Any,
    receipt: Mapping[str, Any],
    *,
    source_sha256: str,
    derived_sha256: str,
) -> dict[int, list[PanelOwner]]:
    """Bind admitted source owners to the geo-sorted derived owner indices."""

    required_receipt = {
        "unit_id": UNIT_ID,
        "ordering": "geo_sorted_xy",
        "source_sha256": source_sha256,
        "derived_sha256": derived_sha256,
        "coordinate_arity_verified": True,
        "owner_multiset_preserved": True,
        "stable_sort_verified": True,
    }
    for key, expected in required_receipt.items():
        if receipt.get(key) != expected:
            raise CohortContractError(
                f"derived receipt {key} mismatch: {receipt.get(key)!r} != {expected!r}"
            )
    if receipt.get("sort_key") != ["decoded_x1", "decoded_y1", "source_index"]:
        raise CohortContractError("derived receipt has the wrong sort_key")
    mappings = receipt.get("source_to_derived")
    if not isinstance(mappings, list):
        raise CohortContractError("derived receipt has no source_to_derived mapping")
    if receipt.get("mapping_sha256") != sha256_json(mappings):
        raise CohortContractError("derived receipt mapping_sha256 mismatch")

    source_rows = _panel_rows(source_panel, label="source panel")
    derived_rows = _panel_rows(derived_panel, label="derived panel")
    if set(source_rows) != set(derived_rows):
        raise CohortContractError("source and derived panels have different image_id sets")
    source_owners = _panel_owners(source_panel)
    derived_owners = _panel_owners(derived_panel)
    mapping_by_image: dict[int, Mapping[str, Any]] = {}
    for mapping_row in mappings:
        if not isinstance(mapping_row, Mapping):
            raise CohortContractError("derived receipt contains a non-object mapping row")
        image = _image_id(mapping_row.get("image_id"))
        if image is None or image in mapping_by_image:
            raise CohortContractError("derived receipt contains an invalid or duplicate image mapping")
        mapping_by_image[image] = mapping_row
    if set(mapping_by_image) != set(source_rows):
        raise CohortContractError("derived receipt image mappings do not cover the panel")

    bound: dict[int, list[PanelOwner]] = {}
    total_owners = 0
    for image in source_rows:
        source_objects = source_rows[image].get("objects")
        derived_objects = derived_rows[image].get("objects")
        if not isinstance(source_objects, list) or not isinstance(derived_objects, list):
            raise CohortContractError(f"image {image} lacks source/derived objects")
        if len(source_objects) != len(derived_objects):
            raise CohortContractError(f"image {image} source/derived owner count mismatch")
        mapping = mapping_by_image[image].get("mapping")
        if not isinstance(mapping, list) or len(mapping) != len(source_objects):
            raise CohortContractError(f"image {image} receipt mapping count mismatch")
        by_source: dict[int, Mapping[str, Any]] = {}
        seen_derived: set[int] = set()
        for entry in mapping:
            if not isinstance(entry, Mapping):
                raise CohortContractError(f"image {image} has a non-object mapping entry")
            source_index = entry.get("source_index")
            derived_index = entry.get("derived_index")
            if (
                not isinstance(source_index, int)
                or isinstance(source_index, bool)
                or not isinstance(derived_index, int)
                or isinstance(derived_index, bool)
                or source_index in by_source
                or derived_index in seen_derived
                or not 0 <= source_index < len(source_objects)
                or not 0 <= derived_index < len(derived_objects)
            ):
                raise CohortContractError(f"image {image} has invalid source/derived mapping indices")
            by_source[source_index] = entry
            seen_derived.add(derived_index)
        if set(by_source) != set(range(len(source_objects))) or seen_derived != set(range(len(derived_objects))):
            raise CohortContractError(f"image {image} receipt mapping is not bijective")
        if any(owner.bbox is None for owner in source_owners[image]):
            raise CohortContractError(f"image {image} has an owner without a decodable four-coordinate box")
        expected_source_order = [
            owner.source_index
            for owner in sorted(
                source_owners[image],
                key=lambda owner: (owner.bbox[0], owner.bbox[1], owner.source_index),  # type: ignore[index]
            )
        ]
        receipt_source_order = [
            source_index
            for source_index, entry in sorted(
                by_source.items(), key=lambda item: item[1]["derived_index"]
            )
        ]
        if receipt_source_order != expected_source_order:
            raise CohortContractError(f"image {image} derived mapping violates geo_sorted_xy order")

        image_bound: list[PanelOwner] = []
        for owner in source_owners[image]:
            method = None
            matches: list[PanelOwner] = []
            if owner.coco_ann_id is not None:
                matches = [
                    derived_owner
                    for derived_owner in derived_owners[image]
                    if derived_owner.coco_ann_id is not None
                    and str(derived_owner.coco_ann_id) == str(owner.coco_ann_id)
                ]
                if len(matches) == 1:
                    method = "coco_ann_id"
            if len(matches) != 1:
                matches = [
                    derived_owner
                    for derived_owner in derived_owners[image]
                    if derived_owner.category == owner.category and derived_owner.bbox == owner.bbox
                ]
                method = "category_pixel_bbox" if len(matches) == 1 else None
            if len(matches) != 1 or method is None:
                raise CohortContractError(
                    f"image {image} source owner {owner.source_index} has no unique derived identity"
                )
            derived_owner = matches[0]
            receipt_entry = by_source[owner.source_index]
            if receipt_entry["derived_index"] != derived_owner.source_index:
                raise CohortContractError(
                    f"image {image} source owner {owner.source_index} receipt/identity mapping disagree"
                )
            source_object = source_objects[owner.source_index]
            derived_object = derived_objects[derived_owner.source_index]
            if canonical_json_bytes(source_object) != canonical_json_bytes(derived_object):
                raise CohortContractError(
                    f"image {image} source owner {owner.source_index} changed in derived panel"
                )
            object_sha256 = sha256_json(source_object)
            if receipt_entry.get("object_sha256") != object_sha256:
                raise CohortContractError(
                    f"image {image} source owner {owner.source_index} object_sha256 mismatch"
                )
            image_bound.append(
                replace(
                    owner,
                    derived_index=derived_owner.source_index,
                    derived_mapping_method=method,
                    raw_bbox=derived_owner.raw_bbox,
                )
            )
        bound[image] = image_bound
        total_owners += len(image_bound)
    if receipt.get("row_count") != len(source_rows):
        raise CohortContractError("derived receipt row_count mismatch")
    if receipt.get("owner_count") != total_owners or receipt.get("mapping_count") != total_owners:
        raise CohortContractError("derived receipt owner/mapping count mismatch")
    return bound


def _match_record(record: Mapping[str, Any], owners: Sequence[PanelOwner], image_id: int) -> tuple[PanelOwner | None, str | None, str | None]:
    """Match by annotation id first, then unique category/pixel bbox."""

    ann = _ann_id(record)
    if ann is not None:
        matches = [owner for owner in owners if owner.coco_ann_id is not None and str(owner.coco_ann_id) == str(ann)]
        if len(matches) == 1:
            return matches[0], "coco_ann_id", None
        if len(matches) > 1:
            return None, "coco_ann_id", "ambiguous_coco_ann_id"
    category = _category(record)
    bbox = _bbox(record)
    if category is not None and bbox is not None:
        matches = [owner for owner in owners if owner.category == category and owner.bbox == bbox]
        if len(matches) == 1:
            return matches[0], "category_pixel_bbox", None
        if len(matches) > 1:
            return None, "category_pixel_bbox", "ambiguous_category_pixel_bbox"
    return None, None, "missing_unique_owner_identity"


def _covered_refs(record: Mapping[str, Any]) -> list[Any]:
    refs: list[Any] = []
    for key in ("covered_owner_ids", "covered_owners", "strict_matched_owner_ids", "matched_owner_ids", "covered"):
        value = record.get(key)
        if isinstance(value, list):
            refs.extend(value)
    return refs


def _resolve_ref(ref: Any, owners: Sequence[PanelOwner], image_id: int) -> PanelOwner | None:
    if isinstance(ref, Mapping):
        owner, _, _ = _match_record(ref, owners, image_id)
        return owner
    if isinstance(ref, str) and ref.startswith("gt:"):
        pieces = ref.split(":")
        if len(pieces) == 3:
            try:
                index = int(pieces[2])
            except ValueError:
                return None
            return next((owner for owner in owners if owner.source_index == index), None)
    return None


def _normalise_records(
    value: Any,
    panel: dict[int, list[PanelOwner]],
    *,
    source_kind: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw in _flatten_records(value):
        image = _image_id(raw.get("image_id"))
        checkpoint = _norm_checkpoint(raw.get("checkpoint", raw.get("checkpoint_id", raw.get("model"))))
        item = dict(raw)
        item["_provided_fields"] = sorted(raw)
        item["image_id"] = image
        item["checkpoint"] = checkpoint
        item["source_kind"] = source_kind
        item["config_fingerprint"] = raw.get("config_fingerprint")
        item["exact_prefix_sha256"] = raw.get(
            "exact_prefix_sha256", raw.get("exact_prefix_hash")
        )
        item["history_complete"] = _as_bool(raw.get("history_complete"))
        item["boundary"] = _boundary(raw)
        for name, keys in FLAG_KEYS.items():
            item[name] = _flag(raw, keys)
        item["owner_ref"] = _owner_ref(raw)
        owner = None
        method = None
        reason = "missing_image_or_checkpoint"
        if image is not None and checkpoint in {"S", "A"} and image in panel:
            owner, method, reason = _match_record(raw, panel[image], image)
        item["owner"] = owner
        item["identity_method"] = method
        item["identity_error"] = reason if owner is None else None
        item["covered_owner_refs"] = _covered_refs(raw)
        item["overlap_decomposition"] = _extract_overlap(raw)
        records.append(item)
    return records


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _require_json_bool(value: Mapping[str, Any], key: str, *, context: str) -> bool:
    if key not in value or not isinstance(value[key], bool):
        raise CohortContractError(f"{context} {key} must be a present JSON boolean")
    return value[key]


def _boundary_mentions_terminal(value: Any) -> bool:
    if isinstance(value, str):
        token = value.lower()
        return "im_end" in token or "<|im_end|>" in token or token in {"stop", "terminal"}
    if isinstance(value, Mapping):
        return any(_boundary_mentions_terminal(item) for item in value.values())
    if isinstance(value, list):
        return any(_boundary_mentions_terminal(item) for item in value)
    return False


ALLOWED_DISPOSITION_STRINGS = {
    "established",
    "indeterminate",
    "invalid",
    "not_applicable",
}


def _validate_ledger_envelope(
    value: Any,
    *,
    source_info: Mapping[str, Any],
    source_panel_sha256: str,
    derived_panel_sha256: str,
    panel: dict[int, list[PanelOwner]],
    source_kind: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not isinstance(value, Mapping):
        raise CohortContractError(
            f"{source_kind} ledger {source_info.get('path', '<inline>')} must be an envelope object"
        )
    if value.get("schema_version") != LEDGER_SCHEMA_VERSION:
        raise CohortContractError(
            f"{source_kind} ledger schema_version must be {LEDGER_SCHEMA_VERSION}"
        )
    if value.get("unit_id") != UNIT_ID:
        raise CohortContractError(f"{source_kind} ledger unit_id mismatch")
    checkpoint = _norm_checkpoint(value.get("checkpoint"))
    if checkpoint not in {"S", "A"}:
        raise CohortContractError(f"{source_kind} ledger checkpoint must be S or A")
    config_fingerprint = value.get("config_fingerprint")
    if not isinstance(config_fingerprint, str) or not config_fingerprint:
        raise CohortContractError(f"{source_kind} ledger has no config_fingerprint")
    if value.get("run_kind") != "native_h0":
        raise CohortContractError(f"{source_kind} ledger run_kind must be native_h0")
    if value.get("arm") not in (None, "native", "H0", "h0"):
        raise CohortContractError(f"{source_kind} ledger is not a native H0 arm")
    declared_source = value.get("source_panel_sha256", value.get("panel_sha256"))
    if declared_source != source_panel_sha256:
        raise CohortContractError(f"{source_kind} ledger source panel hash mismatch")
    if value.get("derived_panel_sha256") != derived_panel_sha256:
        raise CohortContractError(f"{source_kind} ledger derived panel hash mismatch")
    if _require_json_bool(value, "history_complete", context=f"{source_kind} ledger") is not True:
        raise CohortContractError(f"{source_kind} ledger must attest history_complete=true")
    payload = value.get("records")
    if not isinstance(payload, list) or not payload:
        raise CohortContractError(f"{source_kind} ledger must contain a non-empty records list")
    context = {
        "unit_id": UNIT_ID,
        "checkpoint": checkpoint,
        "config_fingerprint": config_fingerprint,
        "history_complete": True,
        "run_kind": "native_h0",
    }
    flattened = _flatten_records(payload, context)
    records = _normalise_records(flattened, panel, source_kind=source_kind)
    if not records:
        raise CohortContractError(f"{source_kind} ledger contains no owner records")
    for record in records:
        for disposition_key in ("status", "disposition"):
            disposition = record.get(disposition_key)
            if isinstance(disposition, str) and disposition not in ALLOWED_DISPOSITION_STRINGS:
                raise CohortContractError(
                    f"{source_kind} record {disposition_key} has an unsupported string value"
                )
            if isinstance(disposition, Mapping):
                for keys in FLAG_KEYS.values():
                    for key in keys:
                        if key in disposition and not isinstance(disposition[key], bool):
                            raise CohortContractError(
                                f"{source_kind} record nested {key} must be a JSON boolean"
                            )
        provided_fields = set(record.get("_provided_fields", ()))
        for keys in FLAG_KEYS.values():
            for key in keys:
                if key in provided_fields and not isinstance(record.get(key), bool):
                    raise CohortContractError(
                        f"{source_kind} record {key} must be a JSON boolean when present"
                    )
        outcome_fields = {
            "native_tp",
            "native_fn",
            "strict_complete_row",
            "natural_boundary_valid",
        }
        if source_kind == "h0":
            if record.get("support_status") != "not_measured":
                raise CohortContractError(
                    "h0 record support_status must be exactly 'not_measured'"
                )
            if _require_json_bool(
                record, "verified_support_claim", context="h0 record"
            ) is not False:
                raise CohortContractError("h0 record verified_support_claim must be false")
            support_aliases = set(FLAG_KEYS["verified_support"])
            status_mapping = record.get("status")
            if provided_fields & support_aliases or (
                isinstance(status_mapping, Mapping)
                and set(status_mapping) & support_aliases
            ):
                raise CohortContractError(
                    "h0 record must not contain a verified_support claim or value"
                )
            required_outcomes = outcome_fields
        else:
            if record.get("support_status") != "measured":
                raise CohortContractError(
                    "support record support_status must be exactly 'measured'"
                )
            if _require_json_bool(
                record, "verified_support_claim", context="support record"
            ) is not True:
                raise CohortContractError(
                    "support record verified_support_claim must be true"
                )
            _require_json_bool(
                record, "verified_support", context="support record"
            )
            provided_outcomes = provided_fields & outcome_fields
            if provided_outcomes != outcome_fields:
                raise CohortContractError(
                    "support record requires one complete native outcome boolean group"
                )
            required_outcomes = outcome_fields

        canonical_values: dict[str, bool] = {}
        for field in required_outcomes:
            canonical_values[field] = _require_json_bool(
                record, field, context=f"{source_kind} record"
            )
        if required_outcomes:
            native_tp = canonical_values["native_tp"]
            native_fn = canonical_values["native_fn"]
            if native_tp == native_fn:
                raise CohortContractError(
                    f"{source_kind} record must have exactly one of native_tp/native_fn true"
                )
            if canonical_values["strict_complete_row"] != native_tp:
                raise CohortContractError(
                    f"{source_kind} record strict_complete_row contradicts native TP/FN outcome"
                )
        if source_kind == "support":
            canonical_values["verified_support"] = record["verified_support"]
        for semantic, aliases in FLAG_KEYS.items():
            if semantic not in canonical_values:
                continue
            for alias in aliases:
                if alias in provided_fields and record[alias] != canonical_values[semantic]:
                    raise CohortContractError(
                        f"{source_kind} record {alias} contradicts canonical {semantic}"
                    )
                status_mapping = record.get("status")
                if (
                    isinstance(status_mapping, Mapping)
                    and alias in status_mapping
                    and status_mapping[alias] != canonical_values[semantic]
                ):
                    raise CohortContractError(
                        f"{source_kind} record nested {alias} contradicts canonical {semantic}"
                    )
        if record.get("checkpoint") != checkpoint:
            raise CohortContractError(f"{source_kind} record checkpoint mismatches envelope")
        if record.get("unit_id") != UNIT_ID:
            raise CohortContractError(f"{source_kind} record unit_id mismatches envelope")
        if record.get("run_kind") != "native_h0":
            raise CohortContractError(f"{source_kind} record is not native_h0")
        if record.get("config_fingerprint") != config_fingerprint:
            raise CohortContractError(f"{source_kind} record config_fingerprint mismatches envelope")
        record_source_hash = record.get(
            "source_panel_sha256", record.get("panel_sha256")
        )
        if record_source_hash is not None and record_source_hash != source_panel_sha256:
            raise CohortContractError(f"{source_kind} record source panel hash mismatch")
        record_derived_hash = record.get("derived_panel_sha256")
        if record_derived_hash is not None and record_derived_hash != derived_panel_sha256:
            raise CohortContractError(f"{source_kind} record derived panel hash mismatch")
        if record.get("history_complete") is not True:
            raise CohortContractError(f"{source_kind} record lacks complete-history attestation")
        if record.get("owner") is None:
            raise CohortContractError(
                f"{source_kind} record owner identity is not unique: {record.get('identity_error')}"
            )
        if _require_json_bool(
            record, "excludes_stop", context=f"{source_kind} record"
        ) is not True:
            raise CohortContractError(f"{source_kind} record excludes_stop must be true")
        if _require_json_bool(
            record, "queried_owner_not_covered", context=f"{source_kind} record"
        ) is not True:
            raise CohortContractError(
                f"{source_kind} record queried_owner_not_covered must be true"
            )
        covered_ids = record.get("covered_owner_ids")
        if not isinstance(covered_ids, list) or any(
            not isinstance(owner_id, str) or not owner_id for owner_id in covered_ids
        ):
            raise CohortContractError(
                f"{source_kind} record covered_owner_ids must be a list of owner IDs"
            )
        if record["owner"].gt_owner_id in covered_ids:
            raise CohortContractError(
                f"{source_kind} record covered_owner_ids must exclude queried owner B"
            )
        known_owner_ids = {owner.gt_owner_id for owner in panel[record["image_id"]]}
        if any(owner_id not in known_owner_ids for owner_id in covered_ids):
            raise CohortContractError(
                f"{source_kind} record covered_owner_ids contains an owner outside the image panel"
            )
        latest_a = record.get("latest_covered_owner_id")
        if covered_ids:
            if latest_a != covered_ids[-1]:
                raise CohortContractError(
                    f"{source_kind} record latest_covered_owner_id is not the latest covered owner"
                )
        elif latest_a is not None:
            raise CohortContractError(
                f"{source_kind} record latest_covered_owner_id is set without covered owners"
            )
        evidence = record.get("due_boundary_evidence")
        if not isinstance(evidence, Mapping):
            raise CohortContractError(f"{source_kind} record due_boundary_evidence is missing")
        if evidence.get("queried_owner_id") != record["owner"].gt_owner_id:
            raise CohortContractError(
                f"{source_kind} due_boundary_evidence queried owner mismatch"
            )
        if evidence.get("covered_owner_ids") != covered_ids:
            raise CohortContractError(
                f"{source_kind} due_boundary_evidence covered owners mismatch"
            )
        if evidence.get("latest_covered_owner_id") != latest_a:
            raise CohortContractError(
                f"{source_kind} due_boundary_evidence latest covered owner mismatch"
            )
        if evidence.get("queried_owner_not_covered") is not True:
            raise CohortContractError(
                f"{source_kind} due_boundary_evidence must exclude queried owner"
            )
        boundary_disposition = record.get("boundary_disposition")
        if evidence.get("boundary_disposition") != boundary_disposition:
            raise CohortContractError(
                f"{source_kind} due_boundary_evidence boundary disposition mismatch"
            )
        if evidence.get("covered_row_count") != len(covered_ids):
            raise CohortContractError(
                f"{source_kind} due_boundary_evidence covered row count mismatch"
            )
        invalid_boundary = boundary_disposition == "no_valid_post_covered_boundary"
        if invalid_boundary:
            if source_kind != "h0" or not record["native_fn"]:
                raise CohortContractError(
                    "only an H0 native FN may use no_valid_post_covered_boundary"
                )
            if record["natural_boundary_valid"] is not False:
                raise CohortContractError(
                    "invalid-boundary H0 record must set natural_boundary_valid=false"
                )
            if (
                record.get("boundary") is not None
                or record.get("due_boundary_index") is not None
                or covered_ids
                or latest_a is not None
            ):
                raise CohortContractError(
                    "invalid-boundary H0 record must have null boundary/due index and no covered owners"
                )
            if record.get("prefix_semantics") != INVALID_PREFIX_SEMANTICS:
                raise CohortContractError(
                    "invalid-boundary H0 record prefix_semantics mismatch"
                )
            if record.get("exact_prefix_token_ids") is not None or record.get(
                "exact_prefix_sha256"
            ) is not None:
                raise CohortContractError(
                    "invalid-boundary H0 record must not claim an exact prefix"
                )
        else:
            if record["natural_boundary_valid"] is not True:
                raise CohortContractError(
                    f"{source_kind} record natural boundary is not valid"
                )
            boundary = record.get("boundary")
            if _boundary_mentions_terminal(boundary):
                raise CohortContractError(
                    f"{source_kind} record natural boundary must not be terminal im_end/STOP"
                )
            if (
                not isinstance(boundary, int)
                or isinstance(boundary, bool)
                or boundary < 0
            ):
                raise CohortContractError(
                    f"{source_kind} record natural boundary must be a non-negative integer"
                )
            due_boundary_index = record.get("due_boundary_index")
            if (
                not isinstance(due_boundary_index, int)
                or isinstance(due_boundary_index, bool)
                or due_boundary_index < 0
                or due_boundary_index != boundary
            ):
                raise CohortContractError(
                    f"{source_kind} record natural boundary and due_boundary_index disagree"
                )
            prefix_ids = record.get("exact_prefix_token_ids")
            if not isinstance(prefix_ids, list) or any(
                isinstance(token, bool) or not isinstance(token, int) or token < 0
                for token in prefix_ids
            ):
                raise CohortContractError(
                    f"{source_kind} record exact_prefix_token_ids are malformed"
                )
            if not _is_sha256(record.get("exact_prefix_sha256")) or sha256_json(
                prefix_ids
            ) != record["exact_prefix_sha256"]:
                raise CohortContractError(
                    f"{source_kind} record exact prefix token IDs/hash mismatch"
                )
            prefix_semantics = record.get("prefix_semantics")
            if not prefix_ids:
                if not record["native_tp"] or boundary != 0:
                    raise CohortContractError(
                        "only a TP root due-boundary may have an empty exact prefix"
                    )
                expected_semantics = TP_PREFIX_SEMANTICS
            else:
                expected_semantics = (
                    TP_PREFIX_SEMANTICS if record["native_tp"] else FN_PREFIX_SEMANTICS
                )
                prefix_end = evidence.get("prefix_end_step")
                stop_step = evidence.get("stop_step")
                if (
                    not isinstance(prefix_end, int)
                    or isinstance(prefix_end, bool)
                    or not isinstance(stop_step, int)
                    or isinstance(stop_step, bool)
                    or prefix_end >= stop_step
                ):
                    raise CohortContractError(
                        f"{source_kind} due boundary does not prove a pre-STOP prefix"
                    )
            if prefix_semantics != expected_semantics:
                raise CohortContractError(
                    f"{source_kind} record prefix_semantics contradicts native TP/FN boundary"
                )
            if record["native_fn"] and (
                boundary < 1 or not covered_ids or latest_a is None
            ):
                raise CohortContractError(
                    f"{source_kind} native FN requires a covered-owner pre-STOP boundary"
                )
        if record.get("intervention") not in (None, False, "none", "native"):
            raise CohortContractError(f"{source_kind} record is intervention-derived")
    envelope = {
        "checkpoint": checkpoint,
        "config_fingerprint": config_fingerprint,
        "source": dict(source_info),
    }
    return envelope, records


def _extract_overlap(record: Mapping[str, Any]) -> dict[str, Any] | None:
    for key in ("overlap_decomposition", "overlap_cells", "exclusive_shared_decomposition", "cells"):
        value = record.get(key)
        if isinstance(value, Mapping):
            normalized = {}
            for name in ("a_exclusive", "b_exclusive", "shared_core", "A_exclusive", "B_exclusive", "shared"):
                if name in value:
                    normalized[name.lower()] = value[name]
            if normalized:
                return normalized
    if any(key in record for key in ("a_exclusive", "b_exclusive", "shared_core")):
        return {key: record.get(key) for key in ("a_exclusive", "b_exclusive", "shared_core")}
    return None


def _aggregate_status(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    status: dict[str, Any] = {}
    for name in FLAG_KEYS:
        values = [record[name] for record in records if record.get(name) is not None]
        status[name] = True if True in values else (False if values and all(value is False for value in values) else None)
    valid_boundaries = [record for record in records if record.get("boundary") is not None]
    latest = max(valid_boundaries, key=lambda record: _boundary_key(record["boundary"]), default=None)
    status["natural_boundary"] = latest.get("boundary") if latest else None
    status["boundary_record"] = latest.get("row_index") if latest else None
    status["evidence_records"] = len(records)
    return status


def _pair_for(
    target: PanelOwner,
    checkpoint: str,
    owner_records: Mapping[tuple[str, int, str], list[Mapping[str, Any]]],
    image_records: Sequence[Mapping[str, Any]],
    owners: Sequence[PanelOwner],
) -> dict[str, Any]:
    target_key = (checkpoint, target.image_id, target.gt_owner_id)
    target_records = list(owner_records.get(target_key, ()))
    target_status = _aggregate_status(target_records)

    # Select the *earliest* natural boundary at which this owner is supported
    # but still uncovered.  Aggregating the target's whole history first would
    # let a later TP overwrite the earlier FN and leak future information.
    b_records: list[Mapping[str, Any]] = []
    for record in target_records:
        if record.get("native_fn") is not True:
            continue
        if record.get("verified_support") is not True:
            continue
        if record.get("strict_complete_row") is True:
            continue
        if record.get("natural_boundary_valid") is not True:
            continue
        if record.get("history_complete") is not True:
            continue
        boundary = record.get("boundary")
        if boundary is None:
            continue
        refs = record.get("covered_owner_refs", [])
        if refs:
            resolved_refs = [_resolve_ref(ref, owners, target.image_id) for ref in refs]
            if any(
                owner is not None and owner.gt_owner_id == target.gt_owner_id
                for owner in resolved_refs
            ):
                continue
        if any(
            earlier.get("strict_complete_row") is True
            and earlier.get("boundary") is not None
            and _boundary_key(earlier["boundary"]) < _boundary_key(boundary)
            for earlier in target_records
        ):
            continue
        b_records.append(record)
    earliest_b = min(b_records, key=lambda record: _boundary_key(record["boundary"]), default=None)
    target_boundary = earliest_b.get("boundary") if earliest_b is not None else None

    covered: list[tuple[PanelOwner, dict[str, Any]]] = []
    if earliest_b is not None:
        b_refs = earliest_b.get("covered_owner_refs", [])
        ref_owners = [_resolve_ref(ref, owners, target.image_id) for ref in b_refs]
        ref_owners = [owner for owner in ref_owners if owner is not None]
        # Explicit covered refs are authoritative.  Only if the B record has
        # no refs may we derive A from the owner ledgers, and only strictly
        # before B's boundary.
        search_owners = ref_owners if b_refs else list(owners)
        for owner in search_owners:
            prior_strict = [
                record
                for record in owner_records.get((checkpoint, owner.image_id, owner.gt_owner_id), ())
                if record.get("strict_complete_row") is True
                and record.get("boundary") is not None
                and _boundary_key(record["boundary"]) < _boundary_key(target_boundary)
            ]
            if prior_strict:
                prior = max(prior_strict, key=lambda record: _boundary_key(record["boundary"]))
                covered.append((owner, {
                    "strict_complete_row": True,
                    "natural_boundary": prior["boundary"],
                }))
    latest_a = max(
        covered,
        key=lambda pair: (_boundary_key(pair[1]["natural_boundary"]), -pair[0].source_index),
        default=None,
    )
    a_payload = None
    if latest_a is not None:
        a_owner, a_status = latest_a
        a_payload = {
            "gt_owner_id": a_owner.gt_owner_id,
            "source_panel_object_index": a_owner.source_index,
            "natural_boundary": a_status["natural_boundary"],
            "strict_complete_row": a_status["strict_complete_row"],
        }
    support = earliest_b.get("verified_support") if earliest_b is not None else target_status["verified_support"]
    strict = earliest_b.get("strict_complete_row") if earliest_b is not None else target_status["strict_complete_row"]
    target_is_b = earliest_b is not None and support is True and strict is not True and target_boundary is not None
    if target_status["evidence_records"] == 0:
        pair_status = "indeterminate_missing_checkpoint_h0"
    elif target_is_b and a_payload is not None:
        pair_status = "verified_pair"
    elif target_is_b:
        pair_status = "no_latest_covered_A"
    elif support is not True:
        pair_status = "no_verified_B"
    else:
        pair_status = "no_verified_B"
    return {
        "A_latest_covered": a_payload,
        "B_verified_uncovered": {
            "gt_owner_id": target.gt_owner_id,
            "verified_support": support,
            "strict_complete_row": strict,
            "natural_boundary": target_boundary,
            "exact_prefix_sha256": earliest_b.get("exact_prefix_sha256"),
        } if target_is_b and pair_status == "verified_pair" else None,
        "pair_status": pair_status,
    }


def _status_disposition(status: Mapping[str, Any]) -> str:
    if status.get("evidence_records", 0) == 0:
        return "indeterminate_missing_checkpoint_h0"
    if status.get("natural_boundary_valid") is False:
        return "indeterminate_no_valid_natural_boundary"
    if status.get("native_tp") is True or status.get("native_fn") is True:
        required = ("strict_complete_row", "natural_boundary_valid")
        if all(status.get(key) is not None for key in required):
            return "established"
    if any(status.get(key) is None for key in ("native_tp", "native_fn", "strict_complete_row", "verified_support")):
        return "indeterminate_missing_required_fields"
    return "established"


def _primary_stratum(labels: Sequence[str]) -> str:
    for label in ("SS", "SO", "XC", "TP", "SUP", "A3R", "NK16"):
        if label in labels:
            return label
    return labels[0] if labels else "UNSTRATIFIED"


NATIVE_TP_REPLACEMENT_RULE = (
    "same_image_primary_tp_by_smallest_natural_due_boundary_then_source_panel_object_index"
)


def _owner_provenance(owner: PanelOwner) -> dict[str, Any]:
    return {
        "gt_owner_id": owner.gt_owner_id,
        "coco_ann_id": owner.coco_ann_id,
        "source_panel_object_index": owner.source_index,
        "derived_panel_object_index": owner.derived_index,
        "mapping_method": owner.derived_mapping_method,
        "category": owner.category,
        "pixel_bbox": list(owner.bbox) if owner.bbox is not None else None,
        "raw_pixel_bbox": list(owner.raw_bbox) if owner.raw_bbox is not None else None,
    }


# Geometry is intentionally kept in this materializer rather than imported
# from a model/runtime module.  The cohort is a CPU-only registry seam and its
# pixel-to-merger-cell mapping must remain usable when torch/model dependencies
# are unavailable.  The semantics mirror the observational census: each
# merger cell owns a rectangular fraction of the decoded image and a box
# contributes its intersection area divided by that cell area.
def _panel_dimensions(panel: Any, *, label: str) -> dict[int, tuple[float, float] | None]:
    rows = panel if isinstance(panel, list) else [panel]
    result: dict[int, tuple[float, float] | None] = {}
    for row_index, row in enumerate(rows, 1):
        if not isinstance(row, Mapping):
            raise CohortContractError(f"{label} row {row_index} is not an object")
        image = _image_id(row.get("image_id"))
        if image is None:
            raise CohortContractError(f"{label} row {row_index} has no integer image_id")
        if image in result:
            raise CohortContractError(f"{label} has duplicate image_id={image}")
        width_value: Any = None
        height_value: Any = None
        for key in ("width", "image_width", "declared_width", "decoded_width"):
            if row.get(key) is not None:
                width_value = row[key]
                break
        for key in ("height", "image_height", "declared_height", "decoded_height"):
            if row.get(key) is not None:
                height_value = row[key]
                break
        size = row.get("image_size")
        if (
            (width_value is None or height_value is None)
            and isinstance(size, Sequence)
            and not isinstance(size, (str, bytes, bytearray))
            and len(size) == 2
        ):
            width_value = size[0] if width_value is None else width_value
            height_value = size[1] if height_value is None else height_value
        if width_value is None or height_value is None:
            result[image] = None
            continue
        try:
            width = float(width_value)
            height = float(height_value)
        except (TypeError, ValueError) as exc:
            raise CohortContractError(f"{label} image {image} has non-numeric dimensions") from exc
        if not math.isfinite(width) or not math.isfinite(height) or width <= 0 or height <= 0:
            raise CohortContractError(f"{label} image {image} dimensions must be finite and positive")
        result[image] = (width, height)
    return result


def _same_dimension(left: tuple[float, float] | None, right: tuple[float, float] | None) -> bool:
    if left is None or right is None:
        return True
    return math.isclose(left[0], right[0], rel_tol=0.0, abs_tol=1.0e-9) and math.isclose(
        left[1], right[1], rel_tol=0.0, abs_tol=1.0e-9
    )


def _plan_identity_payload(record: Mapping[str, Any]) -> Mapping[str, Any] | None:
    value = record.get("image_plan_identity")
    if isinstance(value, Mapping):
        return value
    # A few ledger producers put the selected image-plan fields directly on a
    # row.  Accept that shape while retaining one normalized identity below.
    keys = {
        "expected_image_grid_thw",
        "observed_image_grid_thw",
        "image_grid_thw",
        "merged_visual_tokens",
        "merge_size",
        "spatial_merge_size",
        "declared_width",
        "declared_height",
        "decoded_width",
        "decoded_height",
    }
    if any(key in record for key in keys):
        return record
    return None


def _integer_field(value: Any, *, name: str, context: str, positive: bool = True) -> int:
    if isinstance(value, bool):
        raise CohortContractError(f"{context} {name} must be an integer")
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise CohortContractError(f"{context} {name} must be an integer") from exc
    if integer != value or (positive and integer <= 0):
        raise CohortContractError(
            f"{context} {name} must be {'positive ' if positive else ''}integer"
        )
    return integer


def _infer_merge_size(grid: tuple[int, int, int], merged_tokens: int, *, context: str) -> int:
    temporal, rows, columns = grid
    candidates = [
        merge
        for merge in range(1, min(rows, columns) + 1)
        if rows % merge == 0
        and columns % merge == 0
        and temporal * (rows // merge) * (columns // merge) == merged_tokens
    ]
    if len(candidates) != 1:
        raise CohortContractError(
            f"{context} cannot infer a unique merge_size from grid/token count"
        )
    return candidates[0]


def _normalise_image_plan_identity(
    record: Mapping[str, Any], *, context: str
) -> dict[str, Any] | None:
    payload = _plan_identity_payload(record)
    if payload is None:
        return None
    observed = payload.get("observed_image_grid_thw", payload.get("image_grid_thw"))
    expected = payload.get("expected_image_grid_thw")
    if observed is None:
        if expected is not None:
            observed = expected
        else:
            raise CohortContractError(f"{context} image plan lacks observed_image_grid_thw")
    if not isinstance(observed, Sequence) or isinstance(observed, (str, bytes, bytearray)) or len(observed) != 3:
        raise CohortContractError(f"{context} observed_image_grid_thw must be [time,height,width]")
    grid = tuple(_integer_field(value, name="grid_thw", context=context) for value in observed)
    if expected is not None:
        if not isinstance(expected, Sequence) or isinstance(expected, (str, bytes, bytearray)) or len(expected) != 3:
            raise CohortContractError(f"{context} expected_image_grid_thw must be [time,height,width]")
        expected_grid = tuple(_integer_field(value, name="expected_grid_thw", context=context) for value in expected)
        if expected_grid != grid:
            raise CohortContractError(f"{context} expected/observed image grid mismatch")
    merged_value = payload.get("merged_visual_tokens")
    if merged_value is None:
        raise CohortContractError(f"{context} image plan lacks merged_visual_tokens")
    merged_tokens = _integer_field(merged_value, name="merged_visual_tokens", context=context)
    merge_value = payload.get("merge_size", payload.get("spatial_merge_size"))
    merge_size = (
        _integer_field(merge_value, name="merge_size", context=context)
        if merge_value is not None
        else _infer_merge_size(grid, merged_tokens, context=context)
    )
    temporal, premerge_rows, premerge_columns = grid
    if premerge_rows % merge_size or premerge_columns % merge_size:
        raise CohortContractError(f"{context} image grid is not divisible by merge_size")
    grid_rows = premerge_rows // merge_size
    grid_columns = premerge_columns // merge_size
    expected_tokens = temporal * grid_rows * grid_columns
    if expected_tokens != merged_tokens:
        raise CohortContractError(
            f"{context} merged_visual_tokens mismatch: {merged_tokens} != {expected_tokens}"
        )
    for key, expected_value in (
        ("grid_rows", grid_rows),
        ("grid_cols", grid_columns),
        ("merged_grid_rows", grid_rows),
        ("merged_grid_cols", grid_columns),
        ("cell_count", expected_tokens),
        ("raw_patch_rows", temporal * premerge_rows * premerge_columns),
    ):
        if key in payload and payload[key] is not None:
            declared = _integer_field(payload[key], name=key, context=context)
            if declared != expected_value:
                raise CohortContractError(
                    f"{context} {key} mismatch: {declared} != {expected_value}"
                )
    dimensions: tuple[float, float] | None = None
    width_value: Any = None
    height_value: Any = None
    for key in ("width", "image_width", "declared_width", "decoded_width"):
        if payload.get(key) is not None:
            width_value = payload[key]
            break
    for key in ("height", "image_height", "declared_height", "decoded_height"):
        if payload.get(key) is not None:
            height_value = payload[key]
            break
    if width_value is not None or height_value is not None:
        if width_value is None or height_value is None:
            raise CohortContractError(f"{context} image plan must bind both width and height")
        try:
            width = float(width_value)
            height = float(height_value)
        except (TypeError, ValueError) as exc:
            raise CohortContractError(f"{context} image plan dimensions are not numeric") from exc
        if not math.isfinite(width) or not math.isfinite(height) or width <= 0 or height <= 0:
            raise CohortContractError(f"{context} image plan dimensions must be finite and positive")
        dimensions = (width, height)
    normalized: dict[str, Any] = {
        "observed_image_grid_thw": list(grid),
        "grid_thw": list(grid),
        "merge_size": merge_size,
        "premerge_grid_rows": premerge_rows,
        "premerge_grid_cols": premerge_columns,
        "grid_rows": grid_rows,
        "grid_cols": grid_columns,
        "merged_visual_tokens": merged_tokens,
        "cell_count": expected_tokens,
    }
    if dimensions is not None:
        normalized["image_width"] = dimensions[0]
        normalized["image_height"] = dimensions[1]
    return normalized


def _validate_image_plan_identities(
    records: Sequence[Mapping[str, Any]],
    *,
    panel_dimensions: Mapping[int, tuple[float, float] | None],
    context: str,
) -> dict[int, dict[str, Any] | None]:
    """Validate one image-plan identity per image across H0 and support rows."""

    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        image = _image_id(record.get("image_id"))
        if image is None:
            continue
        identity = _normalise_image_plan_identity(
            record, context=f"{context} image {image} record"
        )
        if identity is not None:
            by_image[image].append(identity)
    result: dict[int, dict[str, Any] | None] = {}
    images = set(panel_dimensions) | set(by_image)
    for image in images:
        identities = by_image.get(image, [])
        if not identities:
            result[image] = None
            continue
        first = identities[0]
        comparable = (
            "observed_image_grid_thw",
            "merge_size",
            "grid_rows",
            "grid_cols",
            "merged_visual_tokens",
        )
        for item in identities[1:]:
            for key in comparable:
                if item.get(key) != first.get(key):
                    raise CohortContractError(
                        f"{context} image {image} image plan identity mismatch ({key})"
                    )
        panel_size = panel_dimensions.get(image)
        plan_size = (
            (float(first["image_width"]), float(first["image_height"]))
            if "image_width" in first and "image_height" in first
            else None
        )
        if not _same_dimension(panel_size, plan_size):
            raise CohortContractError(
                f"{context} image {image} image dimensions mismatch between panel and H0/support plan"
            )
        if plan_size is None and panel_size is not None:
            first = dict(first)
            first["image_width"], first["image_height"] = panel_size
        result[image] = first
    return result


def _fractional_bbox_cell_weights(
    owner: PanelOwner,
    *,
    identity: Mapping[str, Any],
) -> dict[int, float]:
    bbox = owner.raw_bbox or owner.bbox
    if bbox is None:
        return {}
    width = float(identity["image_width"])
    height = float(identity["image_height"])
    temporal, premerge_rows, premerge_columns = (int(value) for value in identity["observed_image_grid_thw"])
    merge_size = int(identity["merge_size"])
    rows = premerge_rows // merge_size
    columns = premerge_columns // merge_size
    x1, y1, x2, y2 = bbox
    x1, x2 = max(0.0, min(width, x1)), max(0.0, min(width, x2))
    y1, y2 = max(0.0, min(height, y1)), max(0.0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return {}
    cell_width = width / columns
    cell_height = height / rows
    cell_area = cell_width * cell_height
    result: dict[int, float] = {}
    for time_index in range(temporal):
        for row in range(rows):
            cell_y1 = row * cell_height
            cell_y2 = (row + 1) * cell_height
            overlap_y = max(0.0, min(y2, cell_y2) - max(y1, cell_y1))
            if overlap_y <= 0.0:
                continue
            for column in range(columns):
                cell_x1 = column * cell_width
                cell_x2 = (column + 1) * cell_width
                overlap_x = max(0.0, min(x2, cell_x2) - max(x1, cell_x1))
                if overlap_x <= 0.0:
                    continue
                visual_index = time_index * rows * columns + row * columns + column
                result[visual_index] = float((overlap_x * overlap_y) / cell_area)
    return dict(sorted(result.items()))


def _region_receipt(
    weights: Mapping[int, float],
    *,
    status: str = "available",
    reason: str | None = None,
) -> dict[str, Any]:
    pairs = [
        {"cell_index": int(index), "overlap_fraction": float(value)}
        for index, value in sorted(weights.items())
    ]
    cell_indices = [item["cell_index"] for item in pairs]
    return {
        "status": status,
        "available": status == "available" and bool(cell_indices),
        "not_measured_reason": reason,
        "cell_indices": cell_indices,
        "visual_indices": list(cell_indices),
        "fractional_weights": pairs,
        "weight_sum": float(sum(item["overlap_fraction"] for item in pairs)),
        "cell_count": len(cell_indices),
        "cell_indices_sha256": sha256_json(cell_indices),
        "weights_sha256": sha256_json(pairs),
    }


def _empty_region_receipt(reason: str, *, status: str = "not_measured") -> dict[str, Any]:
    return _region_receipt({}, status=status, reason=reason)


def _support_records_at_prefix(
    *,
    checkpoint: str,
    image_id: int,
    pair: Mapping[str, Any],
    owner_records: Mapping[tuple[str, int, str], Sequence[Mapping[str, Any]]],
    owners: Sequence[PanelOwner],
) -> list[dict[str, Any]]:
    b_payload = pair.get("B_verified_uncovered")
    if not isinstance(b_payload, Mapping):
        return []
    boundary = b_payload.get("natural_boundary")
    prefix_hash = b_payload.get("exact_prefix_sha256")
    if boundary is None or not isinstance(prefix_hash, str):
        return []
    support: list[dict[str, Any]] = []
    for owner in owners:
        for record in owner_records.get((checkpoint, image_id, owner.gt_owner_id), ()):
            if (
                record.get("source_kind") == "support"
                and
                record.get("verified_support") is True
                and record.get("boundary") == boundary
                and record.get("exact_prefix_sha256") == prefix_hash
            ):
                support.append(dict(record))
                break
    support.sort(key=lambda item: int(item["owner"].source_index))
    return support


def _owner_level_support_records(
    *,
    checkpoint: str,
    image_id: int,
    owner_records: Mapping[tuple[str, int, str], Sequence[Mapping[str, Any]]],
    owners: Sequence[PanelOwner],
) -> list[dict[str, Any]]:
    """Return one actual measured-support row per independently verified owner."""

    support: list[dict[str, Any]] = []
    for owner in owners:
        matching = [
            record
            for record in owner_records.get(
                (checkpoint, image_id, owner.gt_owner_id), ()
            )
            if record.get("source_kind") == "support"
            and record.get("verified_support") is True
        ]
        if matching:
            support.append(
                dict(
                    min(
                        matching,
                        key=lambda record: (
                            _boundary_key(record.get("boundary")),
                            str(record.get("exact_prefix_sha256")),
                        ),
                    )
                )
            )
    support.sort(key=lambda item: int(item["owner"].source_index))
    return support


def _materialize_event_geometry(
    *,
    candidate: Mapping[str, Any],
    checkpoint: str,
    pair: Mapping[str, Any],
    panel_owners: Mapping[int, Sequence[PanelOwner]],
    owner_records: Mapping[tuple[str, int, str], Sequence[Mapping[str, Any]]],
    image_plan: Mapping[int, dict[str, Any] | None],
    source_panel_sha256: str,
    derived_panel_sha256: str,
    h0_sources: Sequence[Mapping[str, Any]],
    support_sources: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    image = _image_id(candidate.get("image_id"))
    owners = list(panel_owners.get(image or -1, ()))
    identity = image_plan.get(image or -1)
    common = {
        "checkpoint": checkpoint,
        "status": "not_applicable",
        "launch_eligible": False,
        "mechanical_disposition": "not_applicable",
        "reason": None,
        "image_cell_regions": {
            "a_exclusive": [],
            "b_exclusive": [],
            "shared_core": [],
            "background": [],
            "same_class_competitor": [],
        },
        "image_cell_region_receipts": {},
        "verified_support_owner_ids": [],
        "exact_b_boundary_verified_support_owner_ids": [],
        "verified_support_owner_ids_uncovered": [],
        "verified_support_owner_ids_covered": [],
        "verified_support_owner_roles": {},
        "owner_region_owner_ids": [],
        "owner_region_roles": {},
        "owner_region_evidence": {},
        "covered_owner_ids_at_b_boundary": [],
        "target_owner_id": None,
        "b_support_binding": None,
        "owner_regions": {},
        "owner_sources": {},
        "source": {
            "source_panel_sha256": source_panel_sha256,
            "derived_panel_sha256": derived_panel_sha256,
            "h0_ledger_sha256": sorted(
                item.get("sha256") for item in h0_sources if item.get("sha256")
            ),
            "support_ledger_sha256": sorted(
                item.get("sha256") for item in support_sources if item.get("sha256")
            ),
        },
    }

    def finish() -> dict[str, Any]:
        payload = dict(common)
        payload.pop("geometry_sha256", None)
        common["geometry_sha256"] = sha256_json(payload)
        return common

    if identity is not None:
        common["image_plan_identity"] = dict(identity)
    else:
        common["reason"] = "missing_h0_image_plan_identity"
        common["mechanical_disposition"] = "indeterminate_missing_image_plan_identity"
        common["status"] = "indeterminate"
        return finish()
    if "image_width" not in identity or "image_height" not in identity:
        common["reason"] = "missing_authoritative_image_dimensions"
        common["mechanical_disposition"] = "indeterminate_missing_image_dimensions"
        common["status"] = "indeterminate"
        return finish()
    common["image_width"] = identity["image_width"]
    common["image_height"] = identity["image_height"]
    target_owner_id = candidate.get("gt_owner_id")
    b_payload = pair.get("B_verified_uncovered")
    a_payload = pair.get("A_latest_covered")
    if pair.get("pair_status") != "verified_pair" or not isinstance(b_payload, Mapping) or not isinstance(a_payload, Mapping):
        common["reason"] = f"pair_status:{pair.get('pair_status')}"
        common["image_cell_region_receipts"] = {
            key: _empty_region_receipt("pair_not_verified", status="not_applicable")
            for key in common["image_cell_regions"]
        }
        return finish()
    b_owner_id = str(b_payload.get("gt_owner_id", target_owner_id))
    a_owner_id = str(a_payload.get("gt_owner_id"))
    owner_by_id = {owner.gt_owner_id: owner for owner in owners}
    common["target_owner_id"] = b_owner_id
    if (
        a_owner_id == b_owner_id
        or a_owner_id not in owner_by_id
        or b_owner_id not in owner_by_id
        or b_owner_id != str(target_owner_id)
    ):
        common["status"] = "indeterminate"
        common["mechanical_disposition"] = "indeterminate_missing_pair_owner"
        common["reason"] = "A_or_B_owner_identity_invalid"
        return finish()
    a_boundary = a_payload.get("natural_boundary")
    b_boundary = b_payload.get("natural_boundary")
    if (
        a_payload.get("strict_complete_row") is not True
        or a_boundary is None
        or b_boundary is None
        or _boundary_key(a_boundary) >= _boundary_key(b_boundary)
    ):
        common["status"] = "indeterminate"
        common["mechanical_disposition"] = "indeterminate_invalid_covered_A_receipt"
        common["reason"] = "A_is_not_strictly_covered_before_B"
        return finish()
    exact_support_records = _support_records_at_prefix(
        checkpoint=checkpoint,
        image_id=int(image),
        pair=pair,
        owner_records=owner_records,
        owners=owners,
    )
    exact_support_ids = sorted(
        {str(record["owner"].gt_owner_id) for record in exact_support_records},
        key=lambda owner_id: owner_by_id[owner_id].source_index,
    )
    common["exact_b_boundary_verified_support_owner_ids"] = exact_support_ids
    exact_b_records = [
        record
        for record in exact_support_records
        if record["owner"].gt_owner_id == b_owner_id
    ]
    if not exact_b_records:
        common["status"] = "indeterminate"
        common["mechanical_disposition"] = "indeterminate_missing_exact_B_support"
        common["reason"] = "B_lacks_exact_boundary_verified_support_record"
        return finish()
    exact_b_record = exact_b_records[0]
    covered_ids: set[str] = set()
    for owner_ref in exact_b_record.get("covered_owner_refs", ()):
        resolved = _resolve_ref(owner_ref, owners, int(image))
        if resolved is not None:
            covered_ids.add(resolved.gt_owner_id)
    if a_owner_id not in covered_ids or b_owner_id in covered_ids:
        common["status"] = "indeterminate"
        common["mechanical_disposition"] = "indeterminate_invalid_B_covered_set"
        common["reason"] = "B_covered_owner_ids_do_not_bind_chosen_A"
        return finish()
    common["covered_owner_ids_at_b_boundary"] = sorted(
        covered_ids, key=lambda owner_id: owner_by_id[owner_id].source_index
    )
    common["b_support_binding"] = {
        "gt_owner_id": b_owner_id,
        "source_kind": "support",
        "verified_support": True,
        "natural_boundary": b_boundary,
        "exact_prefix_sha256": b_payload.get("exact_prefix_sha256"),
        "covered_owner_ids": common["covered_owner_ids_at_b_boundary"],
        "record_sha256": sha256_json(
            {
                key: value
                for key, value in exact_b_record.items()
                if not key.startswith("_") and key != "owner"
            }
        ),
    }
    owner_level_records = _owner_level_support_records(
        checkpoint=checkpoint,
        image_id=int(image),
        owner_records=owner_records,
        owners=owners,
    )
    support_ids = [
        str(record["owner"].gt_owner_id) for record in owner_level_records
    ]
    common["verified_support_owner_ids"] = support_ids
    common["verified_support_owner_ids_covered"] = sorted(
        (owner_id for owner_id in support_ids if owner_id in covered_ids),
        key=lambda owner_id: owner_by_id[owner_id].source_index,
    )
    common["verified_support_owner_ids_uncovered"] = sorted(
        (owner_id for owner_id in support_ids if owner_id not in covered_ids),
        key=lambda owner_id: owner_by_id[owner_id].source_index,
    )
    common["verified_support_owner_roles"] = {
        owner_id: (
            "target-B"
            if owner_id == b_owner_id
            else "covered-A"
            if owner_id in covered_ids
            else "uncovered-same-image-owner"
        )
        for owner_id in support_ids
    }
    region_owner_ids = sorted(
        set(support_ids) | {a_owner_id, b_owner_id},
        key=lambda owner_id: owner_by_id[owner_id].source_index,
    )
    common["owner_region_owner_ids"] = region_owner_ids
    common["owner_region_roles"] = {
        owner_id: (
            "target-B"
            if owner_id == b_owner_id
            else "covered-A"
            if owner_id == a_owner_id
            else "covered-owner"
            if owner_id in covered_ids
            else "uncovered-same-image-owner"
        )
        for owner_id in region_owner_ids
    }
    common["owner_region_evidence"] = {
        owner_id: {
            "strict_complete_covered_before_B": owner_id == a_owner_id,
            "exact_B_boundary_verified_support": owner_id in exact_support_ids,
            "checkpoint_owner_level_verified_support": owner_id in support_ids,
        }
        for owner_id in region_owner_ids
    }
    region_weights = {
        owner_id: _fractional_bbox_cell_weights(owner_by_id[owner_id], identity=identity)
        for owner_id in region_owner_ids
    }
    all_weights = {
        owner.gt_owner_id: _fractional_bbox_cell_weights(owner, identity=identity)
        for owner in owners
    }
    common["owner_sources"] = {
        owner_id: _owner_provenance(owner_by_id[owner_id])
        for owner_id in region_owner_ids
    }
    owner_regions: dict[str, Any] = {}
    for owner_id in region_owner_ids:
        own = region_weights[owner_id]
        colliding = {
            index: weight
            for index, weight in own.items()
            if any(
                index in other
                for other_id, other in region_weights.items()
                if other_id != owner_id
            )
        }
        exclusive = {
            index: weight for index, weight in own.items() if index not in colliding
        }
        owner_regions[owner_id] = {
            "status": "available" if exclusive else "not_measured",
            "exclusive_available": bool(exclusive),
            "shared_available": bool(colliding),
            "not_measured_reason": None if exclusive else "exclusive_support_vanished",
            "cells": sorted(exclusive),
            "exclusive": sorted(exclusive),
            "shared": sorted(colliding),
            "colliding": sorted(colliding),
            "support": sorted(own),
            "fractional_weights": _region_receipt(exclusive),
            "shared_fractional_weights": _region_receipt(colliding),
            "support_fractional_weights": _region_receipt(own),
            "role": common["owner_region_roles"][owner_id],
            "evidence": common["owner_region_evidence"][owner_id],
            "source": common["owner_sources"][owner_id],
        }
    common["owner_regions"] = owner_regions
    a_weights = region_weights[a_owner_id]
    b_weights = region_weights[b_owner_id]
    a_shared = {index: a_weights[index] for index in a_weights if index in b_weights}
    b_shared = {index: b_weights[index] for index in b_weights if index in a_weights}
    a_exclusive = {index: a_weights[index] for index in owner_regions[a_owner_id]["fractional_weights"]["cell_indices"]}
    b_exclusive = {index: b_weights[index] for index in owner_regions[b_owner_id]["fractional_weights"]["cell_indices"]}
    pair_weights: dict[str, Mapping[int, float]] = {
        "a_exclusive": a_exclusive,
        "b_exclusive": b_exclusive,
        "shared_core": {index: a_weights[index] for index in sorted(set(a_shared) & set(b_shared))},
    }
    # Background is zero-overlap against every authoritative GT owner, not
    # merely the support subset.  Equal-count selection is stable row-major.
    occupied_by_any_gt = {
        index for weights in all_weights.values() for index in weights
    }
    background_candidates = [
        index
        for index in range(int(identity["merged_visual_tokens"]))
        if index not in occupied_by_any_gt
    ]
    background_count = len(b_exclusive)
    if background_count and len(background_candidates) >= background_count:
        background = {index: 0.0 for index in background_candidates[:background_count]}
        background_status = "available"
        background_reason = None
    elif background_count:
        background = {}
        background_status = "not_measured"
        background_reason = "insufficient_zero_overlap_background"
    else:
        background = {}
        background_status = "not_measured"
        background_reason = "empty_b_exclusive"
    same_class = [
        owner_id
        for owner_id in support_ids
        if owner_id != b_owner_id
        and owner_by_id[owner_id].category == owner_by_id[b_owner_id].category
        and owner_id not in covered_ids
        and owner_regions[owner_id]["exclusive_available"]
    ]
    same_class_id = min(same_class, key=lambda owner_id: owner_by_id[owner_id].source_index) if same_class else None
    common["image_cell_regions"] = {
        "a_exclusive": sorted(a_exclusive),
        "b_exclusive": sorted(b_exclusive),
        "shared_core": sorted(pair_weights["shared_core"]),
        "background": sorted(background),
        "same_class_competitor": owner_regions[same_class_id]["cells"] if same_class_id else [],
    }
    common["image_cell_region_receipts"] = {
        "a_exclusive": _region_receipt(a_exclusive),
        "b_exclusive": _region_receipt(b_exclusive),
        "shared_core": _region_receipt(pair_weights["shared_core"]),
        "background": _region_receipt(background, status=background_status, reason=background_reason),
        "same_class_competitor": (
            owner_regions[same_class_id]["fractional_weights"]
            if same_class_id
            else _empty_region_receipt("no_verified_same_class_competitor", status="not_applicable")
        ),
    }
    common["same_class_competitor_owner_id"] = same_class_id
    common["pairwise_overlap_receipts"] = {
        name: _region_receipt(weights) for name, weights in pair_weights.items()
    }
    common["SO_exclusive_shared_decomposition"] = {
        "status": "available",
        "a_exclusive": sorted(a_exclusive),
        "b_exclusive": sorted(b_exclusive),
        "shared_core": sorted(pair_weights["shared_core"]),
        "a_exclusive_weights_sha256": common["image_cell_region_receipts"]["a_exclusive"]["weights_sha256"],
        "b_exclusive_weights_sha256": common["image_cell_region_receipts"]["b_exclusive"]["weights_sha256"],
        "shared_core_weights_sha256": common["image_cell_region_receipts"]["shared_core"]["weights_sha256"],
    }
    required_nonempty = bool(a_exclusive) and bool(b_exclusive)
    background_ok = common["image_cell_region_receipts"]["background"]["available"]
    if required_nonempty and background_ok:
        common["status"] = "available"
        common["launch_eligible"] = True
        common["mechanical_disposition"] = "eligible_verified_pair_regions"
    else:
        common["status"] = "indeterminate"
        common["mechanical_disposition"] = "indeterminate_required_regions"
        common["reason"] = (
            "empty_a_or_b_exclusive"
            if not required_nonempty
            else "background_not_available"
        )
    return finish()


def _reestablish_native_tp_candidates(
    frozen: Sequence[Mapping[str, Any]],
    h0_records: Sequence[Mapping[str, Any]],
    panel: Mapping[int, Sequence[PanelOwner]],
    checkpoint: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Replace only the primary TP slot when this checkpoint has no selected TP."""

    effective = [dict(item) for item in frozen]
    changes: list[dict[str, Any]] = []
    selected_keys = {
        (int(item["image_id"]), int(item["source_panel_object_index"]))
        for item in effective
    }
    for image_id in EXPECTED_IMAGE_IDS:
        image_positions = [
            index
            for index, item in enumerate(effective)
            if int(item["image_id"]) == image_id
        ]
        selected_indices = {
            int(effective[index]["source_panel_object_index"])
            for index in image_positions
        }
        native_tp_records = [
            record
            for record in h0_records
            if record.get("checkpoint") == checkpoint
            and record.get("image_id") == image_id
            and record.get("native_tp") is True
            and record.get("strict_complete_row") is True
            and record.get("natural_boundary_valid") is True
            and record.get("owner") is not None
        ]
        if any(
            record["owner"].source_index in selected_indices
            for record in native_tp_records
        ):
            continue
        primary_tp_positions = [
            index
            for index in image_positions
            if _primary_stratum(effective[index].get("historical_labels", ())) == "TP"
        ]
        if not primary_tp_positions:
            raise CohortContractError(
                f"candidate pool image {image_id} has no primary-stratum TP slot"
            )
        target_position = min(primary_tp_positions)
        original = dict(effective[target_position])
        original_index = int(original["source_panel_object_index"])
        original_owner = panel[image_id][original_index]

        unused_by_owner: dict[int, Mapping[str, Any]] = {}
        for record in native_tp_records:
            owner = record["owner"]
            if (image_id, owner.source_index) in selected_keys or owner.bbox is None:
                continue
            prior = unused_by_owner.get(owner.source_index)
            record_key = (int(record["boundary"]), owner.source_index)
            if prior is None or record_key < (
                int(prior["boundary"]),
                prior["owner"].source_index,
            ):
                unused_by_owner[owner.source_index] = record
        fallback: dict[str, Any] = {
            "policy": "same_image_same_stratum",
            "reason": "no_reestablished_native_tp",
            "checkpoint": checkpoint,
            "rule": NATIVE_TP_REPLACEMENT_RULE,
            "from_owner": _owner_provenance(original_owner),
            "to_owner": None,
        }
        eligible = sorted(
            unused_by_owner.values(),
            key=lambda record: (
                int(record["boundary"]),
                record["owner"].source_index,
            ),
        )
        if eligible:
            replacement_owner = eligible[0]["owner"]
            fallback["to_owner"] = _owner_provenance(replacement_owner)
            effective[target_position] = {
                "image_id": image_id,
                "source_panel_object_index": replacement_owner.source_index,
                "gt_owner_id": replacement_owner.gt_owner_id,
                "category": replacement_owner.category,
                "pixel_bbox": list(replacement_owner.bbox),
                "historical_labels": ["TP"],
                "_fallback_provenance": fallback,
            }
            selected_keys.remove((image_id, original_index))
            selected_keys.add((image_id, replacement_owner.source_index))
        else:
            fallback["reason"] = "no_same_image_native_tp"
            effective[target_position]["_fallback_provenance"] = fallback
            effective[target_position]["_forced_indeterminate_reason"] = (
                "no_same_image_native_tp"
            )
        changes.append(fallback)
    if len(
        {
            (item.get("image_id"), item.get("source_panel_object_index"))
            for item in effective
        }
    ) != len(effective):
        raise CohortContractError("checkpoint-native TP replacement duplicated an owner")
    present_strata = {
        label for item in effective for label in item.get("historical_labels", ())
    }
    missing_strata = sorted(set(REQUIRED_GEOMETRY_STRATA) - present_strata)
    if missing_strata:
        raise CohortContractError(
            "checkpoint-native TP replacement removed required geometry strata: "
            + ", ".join(missing_strata)
        )
    return effective, changes


def _overlap_for_event(candidate: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    if "SO" not in candidate["historical_labels"]:
        return None
    matches = [record["overlap_decomposition"] for record in records if record.get("overlap_decomposition") is not None]
    if matches:
        return {"status": "available", **matches[0]}
    return {"status": "indeterminate", "reason": "missing_overlap_decomposition"}


def _write_immutable(path: Path, content: bytes) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise CohortContractError(f"existing output is not identical: {path}")
        return
    path.write_bytes(content)


def materialize_cohort(
    panel: str | Path | Mapping[str, Any] | Sequence[Any],
    h0_ledgers: str | Path | Mapping[str, Any] | Sequence[Any] | Sequence[str | Path],
    support_ledgers: str | Path | Mapping[str, Any] | Sequence[Any] | Sequence[str | Path] | None = None,
    *,
    derived_panel: str | Path | Mapping[str, Any] | Sequence[Any],
    derived_receipt: str | Path | Mapping[str, Any],
    candidate_pool: Sequence[Mapping[str, Any]] | None = None,
    output: str | Path | None = None,
    manifest: str | Path | None = None,
    min_events: int = MIN_EVENTS,
    max_events: int = MAX_EVENTS,
) -> dict[str, Any]:
    """Build a deterministic cohort document and optionally write its manifest."""

    if not MIN_EVENTS <= min_events <= max_events <= MAX_EVENTS:
        raise CohortContractError(f"retention bounds must satisfy {MIN_EVENTS} <= min <= max <= {MAX_EVENTS}")
    panel_value, panel_source = _read_source(panel)
    derived_value, derived_source = _read_source(derived_panel)
    receipt_value, receipt_source = _read_source(derived_receipt)
    if not isinstance(receipt_value, Mapping):
        raise CohortContractError("derived receipt must be a JSON object")
    panel_owners = _validate_derived_panel(
        panel_value,
        derived_value,
        receipt_value,
        source_sha256=panel_source["sha256"],
        derived_sha256=derived_source["sha256"],
    )
    source_dimensions = _panel_dimensions(panel_value, label="source panel")
    derived_dimensions = _panel_dimensions(derived_value, label="derived panel")
    if set(source_dimensions) != set(derived_dimensions):
        raise CohortContractError("source and derived panels have different image dimensions")
    for image_id in sorted(source_dimensions):
        if not _same_dimension(source_dimensions[image_id], derived_dimensions[image_id]):
            raise CohortContractError(
                f"image {image_id} source/derived panel dimensions mismatch"
            )
    panel_dimensions = {
        image_id: derived_dimensions.get(image_id) or source_dimensions.get(image_id)
        for image_id in derived_dimensions
    }
    frozen = [dict(item) for item in (candidate_pool or FROZEN_CANDIDATES)]
    preregistered_frozen = [dict(item) for item in frozen]
    if len(frozen) < min_events or len(frozen) > max_events:
        raise CohortContractError(f"candidate pool count {len(frozen)} is outside {min_events}..{max_events}")
    if len({(item.get("image_id"), item.get("source_panel_object_index")) for item in frozen}) != len(frozen):
        raise CohortContractError("candidate pool contains duplicate image/owner identities")
    required_images = sorted({int(item["image_id"]) for item in frozen})
    if required_images != list(EXPECTED_IMAGE_IDS):
        raise CohortContractError(
            f"candidate pool must preserve exactly the eight frozen images: {list(EXPECTED_IMAGE_IDS)}"
        )
    for image_id in EXPECTED_IMAGE_IDS:
        image_candidates = [item for item in frozen if int(item["image_id"]) == image_id]
        if not any("TP" in item.get("historical_labels", ()) for item in image_candidates):
            raise CohortContractError(f"candidate pool image {image_id} has no TP candidate")
    present_strata = {
        label
        for item in frozen
        for label in item.get("historical_labels", ())
    }
    missing_strata = sorted(set(REQUIRED_GEOMETRY_STRATA) - present_strata)
    if missing_strata:
        raise CohortContractError(
            "candidate pool is missing required geometry strata: " + ", ".join(missing_strata)
        )

    h0_sources: list[dict[str, Any]] = []
    h0_values: list[Any] = []
    raw_h0 = h0_ledgers if isinstance(h0_ledgers, (str, Path, Mapping)) else list(h0_ledgers)
    h0_inputs = [raw_h0] if isinstance(raw_h0, (str, Path, Mapping)) else raw_h0
    for source in h0_inputs:
        value, source_info = _read_source(source)
        h0_values.append(value)
        h0_sources.append(source_info)
    support_sources: list[dict[str, Any]] = []
    support_values: list[Any] = []
    if support_ledgers is not None:
        raw_support = support_ledgers if isinstance(support_ledgers, (str, Path, Mapping)) else list(support_ledgers)
        support_inputs = [raw_support] if isinstance(raw_support, (str, Path, Mapping)) else raw_support
        for source in support_inputs:
            value, source_info = _read_source(source)
            support_values.append(value)
            support_sources.append(source_info)
    h0_envelopes: list[dict[str, Any]] = []
    h0_records: list[dict[str, Any]] = []
    for value, source_info in zip(h0_values, h0_sources, strict=True):
        envelope, records = _validate_ledger_envelope(
            value,
            source_info=source_info,
            source_panel_sha256=panel_source["sha256"],
            derived_panel_sha256=derived_source["sha256"],
            panel=panel_owners,
            source_kind="h0",
        )
        h0_envelopes.append(envelope)
        h0_records.extend(records)
    config_by_checkpoint: dict[str, str] = {}
    for envelope in h0_envelopes:
        checkpoint = envelope["checkpoint"]
        fingerprint = envelope["config_fingerprint"]
        if checkpoint in config_by_checkpoint and config_by_checkpoint[checkpoint] != fingerprint:
            raise CohortContractError(f"H0 config_fingerprint drift for checkpoint {checkpoint}")
        config_by_checkpoint[checkpoint] = fingerprint
    if len(config_by_checkpoint) > 1:
        raise CohortContractError(
            "materialization requires exactly one H0 checkpoint; S and A pools are independent"
        )
    active_checkpoint = next(iter(config_by_checkpoint), None)
    candidate_replacements: list[dict[str, Any]] = []
    if active_checkpoint is not None:
        frozen, candidate_replacements = _reestablish_native_tp_candidates(
            frozen,
            h0_records,
            panel_owners,
            active_checkpoint,
        )

    support_envelopes: list[dict[str, Any]] = []
    support_records: list[dict[str, Any]] = []
    for value, source_info in zip(support_values, support_sources, strict=True):
        envelope, records = _validate_ledger_envelope(
            value,
            source_info=source_info,
            source_panel_sha256=panel_source["sha256"],
            derived_panel_sha256=derived_source["sha256"],
            panel=panel_owners,
            source_kind="support",
        )
        checkpoint = envelope["checkpoint"]
        if config_by_checkpoint.get(checkpoint) != envelope["config_fingerprint"]:
            raise CohortContractError(
                f"support config_fingerprint does not match checkpoint {checkpoint} H0"
            )
        support_envelopes.append(envelope)
        support_records.extend(records)

    def evidence_key(record: Mapping[str, Any]) -> tuple[str, int, str, str, str]:
        owner = record["owner"]
        return (
            record["checkpoint"],
            record["image_id"],
            owner.gt_owner_id,
            sha256_json(record["boundary"]),
            record["exact_prefix_sha256"],
        )

    h0_by_boundary: dict[tuple[str, int, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in h0_records:
        key = evidence_key(record)
        existing = h0_by_boundary[key]
        if existing and any(
            prior.get(flag) != record.get(flag)
            for prior in existing
            for flag in (
                "native_tp",
                "native_fn",
                "strict_complete_row",
                "natural_boundary_valid",
            )
        ):
            raise CohortContractError("contradictory H0 records share one boundary/exact prefix")
        existing.append(record)
    for support_record in support_records:
        matches = h0_by_boundary.get(evidence_key(support_record), [])
        if not matches:
            raise CohortContractError(
                "support record has no same-checkpoint H0 boundary/exact-prefix match"
        )
        for h0_record in matches:
            for boundary_field in (
                "prefix_semantics",
                "excludes_stop",
                "queried_owner_not_covered",
                "latest_covered_owner_id",
                "covered_owner_ids",
                "boundary_disposition",
                "due_boundary_index",
                "due_boundary_evidence",
                "exact_prefix_token_ids",
            ):
                if support_record.get(boundary_field) != h0_record.get(boundary_field):
                    raise CohortContractError(
                        f"support record {boundary_field} contradicts same-prefix H0 boundary"
                    )
            support_outcomes = set(support_record.get("_provided_fields", ())) & {
                "native_tp",
                "native_fn",
                "strict_complete_row",
                "natural_boundary_valid",
            }
            for flag in support_outcomes:
                if support_record[flag] != h0_record[flag]:
                    raise CohortContractError(
                        f"support record {flag} contradicts same-prefix H0 outcome"
                    )
            if (
                h0_record.get("verified_support") is not None
                and support_record["verified_support"] != h0_record["verified_support"]
            ):
                raise CohortContractError(
                    "support record verified_support contradicts another same-prefix support claim"
                )
            h0_record["verified_support"] = support_record["verified_support"]
            if support_record.get("overlap_decomposition") is not None:
                h0_record["overlap_decomposition"] = support_record["overlap_decomposition"]
    all_records = h0_records + support_records
    image_plan_by_checkpoint: dict[str, dict[int, dict[str, Any] | None]] = {}
    for checkpoint in ("S", "A"):
        checkpoint_records = [
            record for record in all_records if record.get("checkpoint") == checkpoint
        ]
        image_plan_by_checkpoint[checkpoint] = _validate_image_plan_identities(
            checkpoint_records,
            panel_dimensions=panel_dimensions,
            context=f"checkpoint {checkpoint}",
        )
    owner_records: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    image_records: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in all_records:
        checkpoint = record.get("checkpoint")
        image = record.get("image_id")
        if checkpoint not in {"S", "A"} or image is None:
            continue
        image_records[(checkpoint, image)].append(record)
        owner = record.get("owner")
        if owner is not None:
            owner_records[(checkpoint, image, owner.gt_owner_id)].append(record)

    events: list[dict[str, Any]] = []
    for ordinal, frozen_candidate in enumerate(frozen, 1):
        candidate = dict(frozen_candidate)
        image = _image_id(candidate.get("image_id"))
        index = candidate.get("source_panel_object_index")
        expected_category = str(candidate.get("category", "")).lower()
        expected_bbox = tuple(candidate.get("pixel_bbox", ()))
        panel_match = None
        panel_reason = None
        if image in panel_owners and isinstance(index, int) and 0 <= index < len(panel_owners[image]):
            panel_match = panel_owners[image][index]
            if panel_match.category != expected_category or panel_match.bbox != expected_bbox:
                panel_reason = "frozen_candidate_panel_mismatch"
        else:
            panel_reason = "frozen_candidate_missing_from_panel"
        statuses: dict[str, Any] = {}
        pairs: dict[str, Any] = {}
        for checkpoint in ("S", "A"):
            if panel_match is None:
                status = _aggregate_status(())
            else:
                status = _aggregate_status(owner_records.get((checkpoint, image, panel_match.gt_owner_id), ()))
            status["disposition"] = panel_reason or _status_disposition(status)
            statuses[checkpoint] = status
            pairs[checkpoint] = _pair_for(
                panel_match or PanelOwner(image or -1, int(index or -1), expected_category, expected_bbox if len(expected_bbox) == 4 else None, None, str(candidate.get("gt_owner_id"))),
                checkpoint,
                owner_records,
                image_records.get((checkpoint, image), ()),
                panel_owners.get(image, ()),
            ) if image is not None else {"pair_status": "indeterminate_missing_image"}
        event_records = [record for record in all_records if record.get("image_id") == image and record.get("owner") is not None and record.get("owner").gt_owner_id == (panel_match.gt_owner_id if panel_match else None)]
        overlap = _overlap_for_event(candidate, event_records)
        geometry_by_checkpoint = {
            checkpoint: _materialize_event_geometry(
                candidate=candidate,
                checkpoint=checkpoint,
                pair=pairs[checkpoint],
                panel_owners=panel_owners,
                owner_records=owner_records,
                image_plan=image_plan_by_checkpoint[checkpoint],
                source_panel_sha256=panel_source["sha256"],
                derived_panel_sha256=derived_source["sha256"],
                h0_sources=h0_sources,
                support_sources=support_sources,
            )
            for checkpoint in ("S", "A")
        }
        active_geometry = geometry_by_checkpoint.get(active_checkpoint or "S")
        if active_geometry is None:
            active_geometry = geometry_by_checkpoint["S"]
        if active_geometry.get("SO_exclusive_shared_decomposition") is not None:
            overlap = active_geometry["SO_exclusive_shared_decomposition"]
        geometry_blocks = {
            checkpoint: dict(geometry)
            for checkpoint, geometry in geometry_by_checkpoint.items()
        }
        forced_indeterminate = candidate.get("_forced_indeterminate_reason")
        if forced_indeterminate is not None and active_checkpoint is not None:
            statuses[active_checkpoint]["disposition"] = forced_indeterminate
        event_disposition = "established" if any(status["disposition"] == "established" for status in statuses.values()) else "indeterminate"
        if panel_reason or forced_indeterminate is not None:
            event_disposition = "indeterminate"
        if active_geometry.get("launch_eligible") is not True and pairs.get(active_checkpoint or "S", {}).get("pair_status") == "verified_pair":
            event_disposition = "indeterminate"
        events.append({
            "ordinal": ordinal,
            "image_id": image,
            "gt_owner_id": candidate.get("gt_owner_id", f"gt:{image}:{index}"),
            "source_panel_object_index": index,
            "category": candidate.get("category"),
            "pixel_bbox": list(candidate.get("pixel_bbox", [])),
            "historical_labels": list(candidate.get("historical_labels", [])),
            "primary_stratum": _primary_stratum(candidate.get("historical_labels", [])),
            "panel_identity": {
                "coco_ann_id": panel_match.coco_ann_id if panel_match else None,
                "mapping_method": panel_match.derived_mapping_method if panel_match else None,
                "source_panel_object_index": panel_match.source_index if panel_match else None,
                "derived_panel_object_index": panel_match.derived_index if panel_match else None,
                "status": "matched" if panel_match and not panel_reason else "indeterminate",
                "reason": panel_reason,
            },
            "checkpoint_status": statuses,
            "A_B": pairs,
            "SO_exclusive_shared_decomposition": overlap,
            "geometry_by_checkpoint": geometry_blocks,
            "image_cell_regions": active_geometry["image_cell_regions"],
            "image_cell_region_receipts": active_geometry["image_cell_region_receipts"],
            "verified_support_owner_ids": active_geometry["verified_support_owner_ids"],
            "exact_b_boundary_verified_support_owner_ids": active_geometry["exact_b_boundary_verified_support_owner_ids"],
            "verified_support_owner_ids_uncovered": active_geometry["verified_support_owner_ids_uncovered"],
            "verified_support_owner_ids_covered": active_geometry["verified_support_owner_ids_covered"],
            "verified_support_owner_roles": active_geometry["verified_support_owner_roles"],
            "owner_region_owner_ids": active_geometry["owner_region_owner_ids"],
            "owner_region_roles": active_geometry["owner_region_roles"],
            "owner_region_evidence": active_geometry["owner_region_evidence"],
            "covered_owner_ids_at_b_boundary": active_geometry["covered_owner_ids_at_b_boundary"],
            "target_owner_id": active_geometry["target_owner_id"],
            "b_support_binding": active_geometry["b_support_binding"],
            "owner_regions": active_geometry["owner_regions"],
            "geometry_identity": active_geometry.get("image_plan_identity"),
            "geometry_status": active_geometry.get("status"),
            "geometry_mechanical_disposition": active_geometry.get("mechanical_disposition"),
            "geometry_launch_eligible": active_geometry.get("launch_eligible", False),
            "geometry_sha256": active_geometry.get("geometry_sha256"),
            "disposition": event_disposition,
            "fallback": candidate.get("_fallback_provenance"),
        })

    # A fallback is evidence-preserving, never a silent substitution: retain
    # the original frozen event and point to the first *established* owner in
    # the same image and stratum.  This allows a caller to repair a missing
    # cell without changing the preregistered pool or importing another image.
    used_fallbacks: set[str] = set()
    for event in events:
        if event["disposition"] == "established" or event["fallback"] is not None:
            continue
        alternatives = [
            candidate_event
            for candidate_event in events
            if candidate_event["image_id"] == event["image_id"]
            and candidate_event["primary_stratum"] == event["primary_stratum"]
            and candidate_event["disposition"] == "established"
            and candidate_event["gt_owner_id"] != event["gt_owner_id"]
            and candidate_event["gt_owner_id"] not in used_fallbacks
        ]
        if alternatives:
            alternative = min(alternatives, key=lambda item: item["ordinal"])
            used_fallbacks.add(alternative["gt_owner_id"])
            event["fallback"] = {
                "policy": "same_image_same_stratum",
                "source_gt_owner_id": alternative["gt_owner_id"],
                "source_ordinal": alternative["ordinal"],
                "source_disposition": alternative["disposition"],
                "note": "candidate-only fallback; original event remains indeterminate until explicitly re-established",
            }

    if len(events) < min_events:
        raise CohortContractError(f"retained event count {len(events)} is below minimum {min_events}")
    by_subset = {}
    for name, predicate in (("legacy12", lambda event: event["image_id"] != IMAGE_2299), ("image2299", lambda event: event["image_id"] == IMAGE_2299)):
        subset = [event for event in events if predicate(event)]
        by_subset[name] = {
            "event_count": len(subset),
            "established_count": sum(event["disposition"] == "established" for event in subset),
            "indeterminate_count": sum(event["disposition"] == "indeterminate" for event in subset),
            "by_primary_stratum": {stratum: sum(event["primary_stratum"] == stratum for event in subset) for stratum in sorted({event["primary_stratum"] for event in subset})},
        }
    cohort: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "execution_contract": {"cpu_only": True, "h0_execution": False, "gpu_launch": False, "val200_index_fallback": False},
        "frozen_pool": {
            "count": len(frozen),
            "sha256": sha256_json(preregistered_frozen),
            "effective_sha256": sha256_json(
                [
                    {
                        key: value
                        for key, value in item.items()
                        if not key.startswith("_")
                    }
                    for item in frozen
                ]
            ),
            "required_images": required_images,
            "active_checkpoint": active_checkpoint,
            "native_tp_replacements": candidate_replacements,
        },
        "retention": {"min_events": min_events, "max_events": max_events, "retained_events": len(events), "status": "within_bounds" if min_events <= len(events) <= max_events else "indeterminate"},
        "events": events,
        "subsets": by_subset,
        "sources": {
            "source_panel": panel_source,
            "derived_panel": derived_source,
            "derived_receipt": receipt_source,
            "h0_ledgers": h0_sources,
            "support_ledgers": support_sources,
        },
        "ledger_contract": {
            "expected_envelope": LEDGER_ENVELOPE_CONTRACT,
            "config_fingerprint_by_checkpoint": dict(sorted(config_by_checkpoint.items())),
            "support_requires_same_boundary_and_exact_prefix": True,
        },
        "indeterminate_policy": {
            "missing_or_ambiguous_identity": "retain_indeterminate",
            "missing_checkpoint_evidence": "retain_indeterminate",
            "missing_verified_B": "no_verified_B",
            "overlap_without_decomposition": "retain_indeterminate",
            "historical_image2299_labels": "comparator_only_nontransferring",
        },
    }
    cohort_bytes = canonical_json_bytes(cohort) + b"\n"
    manifest_doc = {
        "schema_version": f"{SCHEMA_VERSION}.manifest",
        "unit_id": UNIT_ID,
        "cohort_sha256": sha256_bytes(cohort_bytes),
        "cohort_content_sha256": sha256_json(cohort),
        "event_count": len(events),
        "legacy12_event_count": by_subset["legacy12"]["event_count"],
        "image2299_event_count": by_subset["image2299"]["event_count"],
        "ledger_envelope_contract_sha256": sha256_json(LEDGER_ENVELOPE_CONTRACT),
        "source_hashes": {
            "source_panel": panel_source.get("sha256"),
            "derived_panel": derived_source.get("sha256"),
            "derived_receipt": receipt_source.get("sha256"),
            "h0_ledgers": [item.get("sha256") for item in h0_sources],
            "support_ledgers": [item.get("sha256") for item in support_sources],
        },
        "execution_contract": cohort["execution_contract"],
    }
    if output is not None:
        output_path = Path(output).expanduser().resolve()
        _write_immutable(output_path, cohort_bytes)
        manifest_path = Path(manifest).expanduser().resolve() if manifest is not None else output_path.with_name(output_path.stem + ".manifest.json")
        _write_immutable(manifest_path, canonical_json_bytes(manifest_doc) + b"\n")
    elif manifest is not None:
        _write_immutable(Path(manifest), canonical_json_bytes(manifest_doc) + b"\n")
    return {"cohort": cohort, "manifest": manifest_doc}


materialize_static_dynamic_owner_interface_cohort = materialize_cohort


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True, type=Path)
    parser.add_argument("--derived-panel", required=True, type=Path)
    parser.add_argument("--derived-receipt", required=True, type=Path)
    parser.add_argument("--h0-ledger", required=True, action="append", type=Path)
    parser.add_argument("--support-ledger", action="append", type=Path, default=[])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--min-events", type=int, default=MIN_EVENTS)
    parser.add_argument("--max-events", type=int, default=MAX_EVENTS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = materialize_cohort(
        args.panel,
        args.h0_ledger,
        args.support_ledger,
        derived_panel=args.derived_panel,
        derived_receipt=args.derived_receipt,
        output=args.output,
        manifest=args.manifest,
        min_events=args.min_events,
        max_events=args.max_events,
    )
    print(json.dumps(result["manifest"], sort_keys=True))


if __name__ == "__main__":
    main()
