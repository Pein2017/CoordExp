#!/usr/bin/env python3
"""Build the sealed CPU-only 784-row natural-boundary admission census.

The preceding owner-interface materializer intentionally retained a 32-owner
pool.  This module is an additive, read-only census for the independent
2026-08-06 unit: it binds both native H0 ledgers to all 392 physical owners,
joins only exact-prefix support records, and records unknown support as
``support_unassessed``.  It never loads a model or infers a support value.

The output is immutable and self-hashed.  The public ``build_census`` function
is useful for small CPU fixtures as well as for the real sealed inputs.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:  # Allow ``python scripts/research/...py``.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research import materialize_static_dynamic_owner_interface_cohort as stable


UNIT_ID = "2026-08-06-natural-boundary-routing-history-replication"
LEGACY_UNIT_ID = stable.UNIT_ID
SCHEMA_VERSION = "natural_boundary_owner_admission_census.v1"
RECEIPT_SCHEMA_VERSION = f"{SCHEMA_VERSION}.receipt"
RECORDS_SCHEMA_VERSION = f"{SCHEMA_VERSION}.records"
EXPECTED_ROW_COUNT = 784
EXPECTED_OWNER_COUNT = 392
CHECKPOINTS = ("S", "A")
CHECKPOINT_LABELS = {"S": "S", "A": "A3"}
EXPECTED_IMAGES = (1584, 2299, 2685, 4134, 5001, 6040, 7511, 10707, 13348, 13923, 14038, 14439, 16228)
EXPECTED_SUPPORT_COUNTS = {"S": 32, "A": 32}
EXPECTED_SUPPORT_COMPLETION_COUNTS = {"S": 200, "A": 221}
EXPECTED_DISPOSITIONS = (
    "eligible_verified_pair",
    "support_unassessed",
    "support_measured_not_verified",
    "native_already_covered",
    "no_strict_covered_A_before_boundary",
    "no_valid_natural_boundary_before_stop",
    "owner_identity_or_geometry_ambiguous",
    "region_contract_unavailable",
)
SHA256_EMPTY = hashlib.sha256(b"[]").hexdigest()

DEFAULT_PRIOR_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-08-05-static-dynamic-owner-interface-crossover"
)
DEFAULT_SOURCE_PANEL = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-08-04-sorted-prospective-13-image-panel-admission/"
    "/evaluation-inputs/human-refined-13.coord.jsonl"
)
DEFAULT_DERIVED_PANEL = DEFAULT_PRIOR_ROOT / "inputs/human-refined-13.geo_sorted_xy.coord.jsonl"
DEFAULT_DERIVED_RECEIPT = DEFAULT_PRIOR_ROOT / "inputs/human-refined-13.geo_sorted_xy.coord.receipt.json"
DEFAULT_H0 = {
    "S": DEFAULT_PRIOR_ROOT / "ledgers/s-step2444-native-h0.json",
    "A": DEFAULT_PRIOR_ROOT / "ledgers/a3-step2445-native-h0.json",
}
DEFAULT_SUPPORT = {
    "S": DEFAULT_PRIOR_ROOT / "ledgers/s-step2444-final-support.json",
    "A": DEFAULT_PRIOR_ROOT / "ledgers/a3-step2445-final-support.json",
}
DEFAULT_OUTPUT_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    f"{UNIT_ID}/cpu-census-v1"
)
DEFAULT_OUTPUT = DEFAULT_OUTPUT_ROOT / "admission-census.json"
DEFAULT_RECORDS = DEFAULT_OUTPUT_ROOT / "admission-census.records.jsonl"
DEFAULT_RECEIPT = DEFAULT_OUTPUT_ROOT / "admission-census.receipt.json"


class CensusContractError(ValueError):
    """Raised when the all-owner CPU census contract is not established."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CensusContractError(f"value is not finite canonical JSON: {exc}") from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: str | Path) -> str:
    try:
        return sha256_bytes(Path(path).expanduser().resolve(strict=True).read_bytes())
    except OSError as exc:
        raise CensusContractError(f"cannot hash {path}: {exc}") from exc


def document_self_sha256(document: Mapping[str, Any]) -> str:
    payload = dict(document)
    payload.pop("self_sha256", None)
    return sha256_json(payload)


def _read_source(source: str | Path | Mapping[str, Any] | Sequence[Any], label: str) -> tuple[Any, dict[str, Any]]:
    if isinstance(source, (str, Path)):
        path = Path(source).expanduser().resolve(strict=True)
        raw = path.read_bytes()
        if path.suffix.lower() == ".jsonl":
            rows: list[Any] = []
            for line_no, line in enumerate(raw.decode("utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise CensusContractError(f"{label}:{line_no} is invalid JSON") from exc
                if not isinstance(row, Mapping):
                    raise CensusContractError(f"{label}:{line_no} must be an object")
                rows.append(dict(row))
            return rows, {"path": str(path), "sha256": sha256_bytes(raw), "row_count": len(rows)}
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CensusContractError(f"{label} is invalid JSON: {path}") from exc
        return value, {"path": str(path), "sha256": sha256_bytes(raw), "row_count": len(value) if isinstance(value, list) else None}
    if isinstance(source, Mapping):
        value: Any = dict(source)
    elif isinstance(source, Sequence) and not isinstance(source, (str, bytes, bytearray)):
        value = list(source)
    else:
        raise TypeError(f"{label} must be a path, mapping, or sequence")
    return value, {"inline": True, "sha256": sha256_json(value), "row_count": len(value) if isinstance(value, list) else None}


def _text(value: Any, label: str) -> str:
    if isinstance(value, bool) or not isinstance(value, str) or not value:
        raise CensusContractError(f"{label} must be a non-empty string")
    return value


def _bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise CensusContractError(f"{label} must be a JSON boolean")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CensusContractError(f"{label} must be a non-negative integer")
    return value


def _sha(value: Any, label: str) -> str:
    text = _text(value, label).lower()
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise CensusContractError(f"{label} must be a lowercase SHA-256")
    return text


def _checkpoint(value: Any) -> str:
    normalized = stable._norm_checkpoint(value)  # noqa: SLF001 - stable contract helper
    if normalized not in CHECKPOINTS:
        raise CensusContractError(f"checkpoint must be S or A, got {value!r}")
    return str(normalized)


def _owner_id(value: Any) -> str:
    text = _text(value, "gt_owner_id")
    parts = text.split(":")
    if len(parts) != 3 or parts[0] != "gt":
        raise CensusContractError(f"gt_owner_id is malformed: {text!r}")
    try:
        image, index = int(parts[1]), int(parts[2])
    except ValueError as exc:
        raise CensusContractError(f"gt_owner_id is malformed: {text!r}") from exc
    if image < 0 or index < 0:
        raise CensusContractError(f"gt_owner_id is malformed: {text!r}")
    return text


def _panel_inputs(
    source_panel: str | Path | Mapping[str, Any] | Sequence[Any],
    derived_panel: str | Path | Mapping[str, Any] | Sequence[Any],
    derived_receipt: str | Path | Mapping[str, Any],
) -> tuple[Any, Any, Mapping[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[int, list[Any]], dict[int, list[Any]], dict[int, Mapping[str, Any]]]:
    source, source_info = _read_source(source_panel, "source panel")
    derived, derived_info = _read_source(derived_panel, "derived panel")
    receipt, receipt_info = _read_source(derived_receipt, "derived receipt")
    if not isinstance(receipt, Mapping):
        raise CensusContractError("derived receipt must be an object")
    try:
        bound = stable._validate_derived_panel(  # noqa: SLF001
            source,
            derived,
            receipt,
            source_sha256=str(source_info["sha256"]),
            derived_sha256=str(derived_info["sha256"]),
        )
        source_owners = stable._panel_owners(source)  # noqa: SLF001
        rows = stable._panel_rows(derived, label="derived panel")  # noqa: SLF001
    except Exception as exc:
        if isinstance(exc, CensusContractError):
            raise
        raise CensusContractError(f"panel/derived identity is not established: {exc}") from exc
    images = set(source_owners)
    if images != set(EXPECTED_IMAGES):
        raise CensusContractError(f"expected 13-image panel {EXPECTED_IMAGES}, observed {sorted(images)}")
    owner_count = sum(len(items) for items in source_owners.values())
    if owner_count != EXPECTED_OWNER_COUNT:
        raise CensusContractError(f"expected {EXPECTED_OWNER_COUNT} physical owners, observed {owner_count}")
    return source, derived, receipt, source_info, derived_info, receipt_info, source_owners, bound, rows


def _load_envelope(source: str | Path | Mapping[str, Any], label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    value, info = _read_source(source, label)
    if not isinstance(value, Mapping):
        raise CensusContractError(f"{label} must be a JSON envelope object")
    return dict(value), info


def _record_owner_id(record: Mapping[str, Any]) -> str:
    value = record.get("gt_owner_id")
    if value is None:
        raise CensusContractError("ledger record has no gt_owner_id")
    return _owner_id(value)


def _validate_prefix(record: Mapping[str, Any], *, context: str, require_valid: bool) -> tuple[int | None, list[int] | None, str | None]:
    boundary = record.get("natural_boundary")
    if boundary is not None:
        boundary = _nonnegative_int(boundary, f"{context}.natural_boundary")
    valid = record.get("natural_boundary_valid")
    if valid is not None:
        valid = _bool(valid, f"{context}.natural_boundary_valid")
    if require_valid and valid is not True:
        return boundary, None, "natural_boundary_invalid"
    tokens = record.get("exact_prefix_token_ids")
    prefix_hash = record.get("exact_prefix_sha256")
    if require_valid:
        if not isinstance(tokens, list) or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in tokens):
            return boundary, None, "exact_prefix_token_ids_invalid"
        try:
            observed_hash = _sha(prefix_hash, f"{context}.exact_prefix_sha256")
        except CensusContractError:
            return boundary, None, "exact_prefix_sha256_invalid"
        if sha256_json(tokens) != observed_hash:
            return boundary, None, "exact_prefix_hash_mismatch"
        if record.get("excludes_stop") is not True:
            return boundary, None, "prefix_includes_stop"
        return boundary, [int(item) for item in tokens], observed_hash
    return boundary, None, None


def _validate_h0(
    value: Mapping[str, Any],
    *,
    info: Mapping[str, Any],
    checkpoint: str,
    source_sha256: str,
    derived_sha256: str,
    owners_by_image: Mapping[int, Sequence[Any]],
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]], dict[str, Any]]:
    context = f"{checkpoint} H0"
    if value.get("schema_version") != stable.LEDGER_SCHEMA_VERSION:
        raise CensusContractError(f"{context} schema_version is not native H0")
    if value.get("unit_id") != LEGACY_UNIT_ID:
        raise CensusContractError(f"{context} unit_id is not the immutable prior unit")
    if _checkpoint(value.get("checkpoint")) != checkpoint:
        raise CensusContractError(f"{context} checkpoint mismatch")
    if value.get("run_kind") != "native_h0":
        raise CensusContractError(f"{context} run_kind must be native_h0")
    if value.get("source_panel_sha256") != source_sha256 or value.get("derived_panel_sha256") != derived_sha256:
        raise CensusContractError(f"{context} panel identity differs from derived panel")
    if value.get("history_complete") is not True:
        raise CensusContractError(f"{context} history_complete must be true")
    records = value.get("records")
    if not isinstance(records, list) or len(records) != EXPECTED_OWNER_COUNT:
        raise CensusContractError(f"{context} must contain exactly {EXPECTED_OWNER_COUNT} records")
    by_owner: dict[str, Mapping[str, Any]] = {}
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(records):
        if not isinstance(raw, Mapping):
            raise CensusContractError(f"{context} record {index} is not an object")
        row = dict(raw)
        owner_id = _record_owner_id(row)
        if owner_id in by_owner:
            raise CensusContractError(f"{context} repeats owner {owner_id}")
        image_id, source_index = owner_id.split(":")[1:]
        image = int(image_id)
        source = int(source_index)
        if image not in owners_by_image or not any(int(owner.source_index) == source for owner in owners_by_image[image]):
            raise CensusContractError(f"{context} owner identity is not uniquely bound: {owner_id}")
        if row.get("image_id") != image or row.get("source_panel_object_index") != source:
            raise CensusContractError(f"{context} {owner_id} identity fields disagree")
        native_tp = _bool(row.get("native_tp"), f"{context} {owner_id}.native_tp")
        native_fn = _bool(row.get("native_fn"), f"{context} {owner_id}.native_fn")
        strict = _bool(row.get("strict_complete_row"), f"{context} {owner_id}.strict_complete_row")
        if native_tp == native_fn or strict != native_tp:
            raise CensusContractError(f"{context} {owner_id} native outcome group is inconsistent")
        boundary, tokens, prefix_status = _validate_prefix(
            row,
            context=f"{context} {owner_id}",
            require_valid=True,
        )
        # A malformed prefix is retained for a deterministic explicit census
        # disposition, but all current sealed ledgers are valid.
        row["_prefix_status"] = prefix_status
        row["_prefix_tokens"] = tokens
        row["_boundary"] = boundary
        row["_record_index"] = index
        by_owner[owner_id] = row
        normalized.append(row)
    expected_ids = {
        f"gt:{int(image)}:{int(owner.source_index)}"
        for image, image_owners in owners_by_image.items()
        for owner in image_owners
    }
    if set(by_owner) != expected_ids:
        raise CensusContractError(f"{context} owner universe differs from the 392-owner panel")
    metadata = {
        "path": info.get("path"),
        "sha256": info.get("sha256"),
        "record_count": len(normalized),
        "config_fingerprint": value.get("config_fingerprint"),
        "checkpoint_identity": value.get("checkpoint_identity"),
    }
    return dict(value), by_owner, metadata


def _validate_support(
    value: Mapping[str, Any],
    *,
    info: Mapping[str, Any],
    checkpoint: str,
    source_sha256: str,
    derived_sha256: str,
    h0_by_owner: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]], dict[str, Any]]:
    context = f"{checkpoint} support"
    if value.get("schema_version") != stable.LEDGER_SCHEMA_VERSION:
        raise CensusContractError(f"{context} schema_version is not native H0 support")
    if value.get("unit_id") != LEGACY_UNIT_ID or _checkpoint(value.get("checkpoint")) != checkpoint:
        raise CensusContractError(f"{context} immutable identity mismatch")
    if value.get("run_kind") != "native_h0" or value.get("history_complete") is not True:
        raise CensusContractError(f"{context} run/history identity is invalid")
    if value.get("source_panel_sha256") != source_sha256 or value.get("derived_panel_sha256") != derived_sha256:
        raise CensusContractError(f"{context} panel identity differs from H0")
    if value.get("support_rule", {}).get("teacher_forced_diagnostic_only") is not True:
        raise CensusContractError(f"{context} lacks teacher-forced diagnostic-only support rule")
    records = value.get("records")
    if not isinstance(records, list) or len(records) != EXPECTED_SUPPORT_COUNTS[checkpoint]:
        raise CensusContractError(f"{context} must contain exactly {EXPECTED_SUPPORT_COUNTS[checkpoint]} records")
    by_owner: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(records):
        if not isinstance(raw, Mapping):
            raise CensusContractError(f"{context} record {index} is not an object")
        row = dict(raw)
        owner_id = _record_owner_id(row)
        if owner_id in by_owner:
            raise CensusContractError(f"{context} repeats owner {owner_id}")
        h0 = h0_by_owner.get(owner_id)
        if h0 is None:
            raise CensusContractError(f"{context} owner {owner_id} is absent from H0")
        if row.get("natural_boundary") != h0.get("natural_boundary") or row.get("exact_prefix_sha256") != h0.get("exact_prefix_sha256"):
            raise CensusContractError(f"{context} {owner_id} is not an exact H0-boundary support row")
        for key in ("native_tp", "native_fn", "strict_complete_row", "natural_boundary_valid"):
            if row.get(key) != h0.get(key):
                raise CensusContractError(f"{context} {owner_id}.{key} differs from H0")
        if row.get("support_status") != "measured" or row.get("verified_support_claim") is not True:
            raise CensusContractError(f"{context} {owner_id} is not measured support")
        _bool(row.get("verified_support"), f"{context} {owner_id}.verified_support")
        if row.get("no_future_or_intervention_leakage") is not True:
            raise CensusContractError(f"{context} {owner_id} has future/intervention leakage")
        by_owner[owner_id] = row
    metadata = {
        "path": info.get("path"),
        "sha256": info.get("sha256"),
        "record_count": len(by_owner),
        "config_fingerprint": value.get("config_fingerprint"),
        "calibration_sha256": value.get("calibration", {}).get("calibration_sha256"),
    }
    return dict(value), by_owner, metadata


def _panel_row_dimensions(rows: Mapping[int, Mapping[str, Any]], image_id: int) -> tuple[float, float] | None:
    row = rows.get(image_id)
    if not isinstance(row, Mapping):
        return None
    width = row.get("width", row.get("image_width"))
    height = row.get("height", row.get("image_height"))
    try:
        width_value, height_value = float(width), float(height)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(width_value) or not math.isfinite(height_value) or width_value <= 0 or height_value <= 0:
        return None
    return width_value, height_value


def _geometry_for_pair(
    *,
    b_record: Mapping[str, Any],
    a_record: Mapping[str, Any],
    support_records: Mapping[str, Mapping[str, Any]],
    owners_by_image: Mapping[int, Sequence[Any]],
    panel_rows: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    image_id = int(b_record["image_id"])
    plan = b_record.get("image_plan_identity")
    base: dict[str, Any] = {
        "status": "unavailable",
        "launch_eligible": False,
        "reason": None,
        "image_id": image_id,
        "target_owner_id": b_record.get("gt_owner_id"),
        "covered_owner_id": a_record.get("gt_owner_id"),
        "image_cell_regions": {"a_exclusive": [], "b_exclusive": [], "background": []},
        "image_cell_region_receipts": {},
    }
    dimensions = _panel_row_dimensions(panel_rows, image_id)
    if not isinstance(plan, Mapping) or dimensions is None:
        base["reason"] = "missing_image_plan_or_dimensions"
        return base
    plan_payload = dict(plan)
    plan_payload.setdefault("image_width", dimensions[0])
    plan_payload.setdefault("image_height", dimensions[1])
    try:
        identity = stable._normalise_image_plan_identity(  # noqa: SLF001
            plan_payload,
            context=f"geometry image {image_id}",
        )
    except Exception as exc:
        base["reason"] = f"invalid_image_plan:{exc}"
        return base
    if identity is None:
        base["reason"] = "missing_image_plan_identity"
        return base
    base["image_plan_identity"] = identity
    owner_map = {owner.gt_owner_id: owner for owner in owners_by_image.get(image_id, ())}
    a_id = str(a_record["gt_owner_id"])
    b_id = str(b_record["gt_owner_id"])
    verified_ids = [owner_id for owner_id, row in support_records.items() if row.get("verified_support") is True and int(row.get("image_id", -1)) == image_id]
    region_ids = list(dict.fromkeys([a_id, b_id, *verified_ids]))
    if any(owner_id not in owner_map for owner_id in region_ids):
        base["reason"] = "owner_identity_or_geometry_ambiguous"
        return base
    try:
        weights = {
            owner_id: stable._fractional_bbox_cell_weights(owner_map[owner_id], identity=identity)  # noqa: SLF001
            for owner_id in region_ids
        }
        all_weights = {
            owner.gt_owner_id: stable._fractional_bbox_cell_weights(owner, identity=identity)  # noqa: SLF001
            for owner in owners_by_image.get(image_id, ())
        }
    except Exception as exc:
        base["reason"] = f"geometry_computation_failed:{exc}"
        return base
    if not weights.get(a_id) or not weights.get(b_id):
        base["reason"] = "empty_owner_support"
        return base
    def exclusive(owner_id: str) -> dict[int, float]:
        own = weights[owner_id]
        return {
            index: value
            for index, value in own.items()
            if not any(index in other for other_id, other in weights.items() if other_id != owner_id)
        }
    a_ex = exclusive(a_id)
    b_ex = exclusive(b_id)
    occupied = {index for owner_weights in all_weights.values() for index in owner_weights}
    cell_count = int(identity["merged_visual_tokens"])
    background_candidates = [index for index in range(cell_count) if index not in occupied]
    if b_ex and len(background_candidates) >= len(b_ex):
        background = {index: 0.0 for index in background_candidates[: len(b_ex)]}
        background_status, background_reason = "available", None
    else:
        background = {}
        background_status, background_reason = "unavailable", "insufficient_zero_overlap_background"
    base["image_cell_regions"] = {
        "a_exclusive": sorted(a_ex),
        "b_exclusive": sorted(b_ex),
        "background": sorted(background),
    }
    base["image_cell_region_receipts"] = {
        "a_exclusive": stable._region_receipt(a_ex),  # noqa: SLF001
        "b_exclusive": stable._region_receipt(b_ex),  # noqa: SLF001
        "background": stable._region_receipt(background, status=background_status, reason=background_reason),  # noqa: SLF001
    }
    base["region_owner_ids"] = sorted(region_ids, key=lambda owner_id: owner_map[owner_id].source_index)
    base["occupied_gt_cell_count"] = len(occupied)
    if a_ex and b_ex and background_status == "available":
        base["status"] = "available"
        base["launch_eligible"] = True
        base["reason"] = None
    else:
        base["reason"] = "required_region_empty_or_unavailable"
    # Bind the final disposition as well as the region payload.  Hashing before
    # status/launch_eligible/reason would let a later disposition drift while
    # preserving the same geometry identity.
    base["geometry_sha256"] = sha256_json(
        {key: value for key, value in base.items() if key != "geometry_sha256"}
    )
    return base


def _row_record(
    *,
    checkpoint: str,
    h0: Mapping[str, Any],
    support: Mapping[str, Mapping[str, Any]],
    h0_by_owner: Mapping[str, Mapping[str, Any]],
    owners_by_image: Mapping[int, Sequence[Any]],
    panel_rows: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    owner_id = str(h0["gt_owner_id"])
    native_tp = bool(h0["native_tp"])
    native_fn = bool(h0["native_fn"])
    support_row = support.get(owner_id)
    target_assessed = bool(native_fn and support_row is not None)
    calibration_assessed = bool(native_tp)
    pair_status = "not_applicable"
    a_record: Mapping[str, Any] | None = None
    geometry: dict[str, Any] | None = None
    eligible_except_support = False
    if native_fn:
        boundary = h0.get("_boundary")
        tokens = h0.get("_prefix_tokens")
        if h0.get("_prefix_status") is None or h0.get("natural_boundary_valid") is not True or not isinstance(tokens, list) or boundary is None:
            pair_status = "no_valid_natural_boundary_before_stop"
        else:
            covered = h0.get("covered_owner_ids")
            if not isinstance(covered, list) or not covered:
                pair_status = "no_strict_covered_A_before_boundary"
            else:
                strict_candidates: list[Mapping[str, Any]] = []
                for covered_id in covered:
                    covered_text = str(covered_id)
                    candidate = h0_by_owner.get(covered_text)
                    if candidate is None or candidate.get("native_tp") is not True or candidate.get("strict_complete_row") is not True:
                        continue
                    covered_boundary = candidate.get("natural_boundary")
                    if isinstance(covered_boundary, int) and covered_boundary < int(boundary):
                        strict_candidates.append(candidate)
                if strict_candidates:
                    a_record = max(strict_candidates, key=lambda item: (int(item.get("natural_boundary", -1)), int(item.get("source_panel_object_index", -1))))
                    pair_status = "verified_pair_candidate"
                    geometry = _geometry_for_pair(
                        b_record=h0,
                        a_record=a_record,
                        support_records=support,
                        owners_by_image=owners_by_image,
                        panel_rows=panel_rows,
                    )
                    eligible_except_support = bool(geometry.get("launch_eligible"))
                else:
                    pair_status = "no_strict_covered_A_before_boundary"
    if native_tp:
        disposition = "native_already_covered"
    elif pair_status == "no_valid_natural_boundary_before_stop":
        disposition = "no_valid_natural_boundary_before_stop"
    elif pair_status == "no_strict_covered_A_before_boundary":
        disposition = "no_strict_covered_A_before_boundary"
    elif support_row is None:
        disposition = "support_unassessed"
    elif support_row.get("verified_support") is not True:
        disposition = "support_measured_not_verified"
    elif geometry is None or not geometry.get("launch_eligible"):
        disposition = "region_contract_unavailable"
    else:
        disposition = "eligible_verified_pair"
    if disposition not in EXPECTED_DISPOSITIONS:
        raise CensusContractError(f"internal unknown disposition {disposition!r}")
    image_id = int(h0["image_id"])
    owner = next((item for item in owners_by_image[image_id] if item.gt_owner_id == owner_id), None)
    if owner is None:
        disposition = "owner_identity_or_geometry_ambiguous"
    support_status = "measured" if support_row is not None else "unassessed"
    row: dict[str, Any] = {
        "checkpoint": checkpoint,
        "checkpoint_label": CHECKPOINT_LABELS[checkpoint],
        "gt_owner_id": owner_id,
        "image_id": image_id,
        "source_panel_object_index": int(h0["source_panel_object_index"]),
        "derived_panel_object_index": (
            int(getattr(owner, "derived_index", None))
            if owner is not None and getattr(owner, "derived_index", None) is not None
            else int(h0.get("derived_panel_object_index", -1))
            if owner is not None and h0.get("derived_panel_object_index") is not None
            else None
        ),
        "category_name": h0.get("category_name"),
        "bbox_pixel_xyxy": h0.get("bbox_pixel_xyxy"),
        "native_tp": native_tp,
        "native_fn": native_fn,
        "strict_complete_row": bool(h0["strict_complete_row"]),
        "natural_boundary_valid": h0.get("natural_boundary_valid"),
        "natural_boundary": h0.get("natural_boundary"),
        "covered_owner_ids": list(h0.get("covered_owner_ids", [])),
        "latest_covered_owner_id": h0.get("latest_covered_owner_id"),
        "exact_prefix_sha256": h0.get("exact_prefix_sha256"),
        "exact_prefix_token_count": len(h0.get("_prefix_tokens", [])) if isinstance(h0.get("_prefix_tokens"), list) else None,
        "h0_record_index": int(h0["_record_index"]),
        "h0_source_sha256": h0.get("source_panel_sha256"),
        "h0_ledger_sha256": None,
        "support_status": support_status,
        "support_record_present": support_row is not None,
        "support_verified": support_row.get("verified_support") if support_row is not None else None,
        "target_B_support": (
            bool(native_fn and support_row is not None and support_row.get("verified_support") is True)
            if native_fn
            else None
        ),
        "target_B_support_assessed": target_assessed,
        "calibration_control_assessed": calibration_assessed,
        "assessment_scope": "S:expanded_in_unit" if checkpoint == "S" else "A3:frozen_at_prior_32",
        "pair_status": pair_status,
        "covered_A_owner_id": a_record.get("gt_owner_id") if a_record is not None else None,
        "covered_A_natural_boundary": a_record.get("natural_boundary") if a_record is not None else None,
        "eligible_except_support": bool(eligible_except_support),
        "eligible_except_support_semantics": (
            "support_state_dependent_upper_bound_before_verified_support"
        ),
        "disposition": disposition,
        "geometry": geometry,
    }
    if support_row is not None:
        row["support_record_sha256"] = sha256_json({key: value for key, value in support_row.items() if not str(key).startswith("_")})
        row["support_calibration_sha256"] = support_row.get("support_calibration_sha256")
    else:
        row["support_record_sha256"] = None
        row["support_calibration_sha256"] = None
    return row


def _counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_cp: dict[str, Any] = {}
    for checkpoint in CHECKPOINTS:
        selected = [row for row in rows if row.get("checkpoint") == checkpoint]
        by_image: dict[str, Any] = {}
        for image_id in EXPECTED_IMAGES:
            image_rows = [row for row in selected if int(row.get("image_id", -1)) == image_id]
            by_image[str(image_id)] = {
                "row_count": len(image_rows),
                "native_tp": sum(bool(row.get("native_tp")) for row in image_rows),
                "native_fn": sum(bool(row.get("native_fn")) for row in image_rows),
                "target_B_support_assessed": sum(bool(row.get("target_B_support_assessed")) for row in image_rows),
                "calibration_control_assessed": sum(bool(row.get("calibration_control_assessed")) for row in image_rows),
                "support_unassessed": sum(row.get("disposition") == "support_unassessed" for row in image_rows),
                "support_measured_not_verified": sum(row.get("disposition") == "support_measured_not_verified" for row in image_rows),
                "eligible_verified_pair": sum(row.get("disposition") == "eligible_verified_pair" for row in image_rows),
                "eligible_except_support": sum(bool(row.get("eligible_except_support")) for row in image_rows),
                "native_already_covered": sum(row.get("disposition") == "native_already_covered" for row in image_rows),
            }
        by_cp[checkpoint] = {
            "checkpoint_label": CHECKPOINT_LABELS[checkpoint],
            "row_count": len(selected),
            "native_tp": sum(bool(row.get("native_tp")) for row in selected),
            "native_fn": sum(bool(row.get("native_fn")) for row in selected),
            "target_B_support_assessed": sum(bool(row.get("target_B_support_assessed")) for row in selected),
            "target_B_support_verified": sum(row.get("target_B_support") is True for row in selected),
            "calibration_control_assessed": sum(bool(row.get("calibration_control_assessed")) for row in selected),
            "support_unassessed": sum(row.get("disposition") == "support_unassessed" for row in selected),
            "support_measured_not_verified": sum(row.get("disposition") == "support_measured_not_verified" for row in selected),
            "eligible_verified_pair": sum(row.get("disposition") == "eligible_verified_pair" for row in selected),
            "eligible_except_support": sum(bool(row.get("eligible_except_support")) for row in selected),
            "native_already_covered": sum(row.get("disposition") == "native_already_covered" for row in selected),
            "dispositions": dict(sorted(Counter(str(row.get("disposition")) for row in selected).items())),
            "by_image": by_image,
        }
    return by_cp


def build_census(
    source_panel: str | Path | Mapping[str, Any] | Sequence[Any] = DEFAULT_SOURCE_PANEL,
    derived_panel: str | Path | Mapping[str, Any] | Sequence[Any] = DEFAULT_DERIVED_PANEL,
    derived_receipt: str | Path | Mapping[str, Any] = DEFAULT_DERIVED_RECEIPT,
    h0_ledgers: Mapping[str, str | Path | Mapping[str, Any]] | None = None,
    support_ledgers: Mapping[str, str | Path | Mapping[str, Any]] | None = None,
    *,
    output: str | Path | None = None,
    records_output: str | Path | None = None,
    receipt_output: str | Path | None = None,
    expected_derived_panel_sha256: str | None = None,
) -> dict[str, Any]:
    """Build and optionally write the exact 784-row census."""

    h0_ledgers = dict(h0_ledgers or DEFAULT_H0)
    support_ledgers = dict(support_ledgers or DEFAULT_SUPPORT)
    if set(h0_ledgers) != set(CHECKPOINTS) or set(support_ledgers) != set(CHECKPOINTS):
        raise CensusContractError("h0_ledgers and support_ledgers must contain exactly S and A")
    (
        _source,
        _derived,
        _receipt,
        source_info,
        derived_info,
        receipt_info,
        owners_by_image,
        _bound,
        panel_rows,
    ) = _panel_inputs(source_panel, derived_panel, derived_receipt)
    source_sha256 = str(source_info["sha256"])
    derived_sha256 = str(derived_info["sha256"])
    if expected_derived_panel_sha256 is not None and derived_sha256 != expected_derived_panel_sha256:
        raise CensusContractError("derived panel SHA-256 differs from expected immutable hash")
    h0_payloads: dict[str, dict[str, Any]] = {}
    h0_by_checkpoint: dict[str, dict[str, Mapping[str, Any]]] = {}
    h0_meta: dict[str, Any] = {}
    support_payloads: dict[str, dict[str, Any]] = {}
    support_by_checkpoint: dict[str, dict[str, Mapping[str, Any]]] = {}
    support_meta: dict[str, Any] = {}
    for checkpoint in CHECKPOINTS:
        h0_value, h0_info = _load_envelope(h0_ledgers[checkpoint], f"{checkpoint} H0")
        h0_payload, h0_by_owner, meta = _validate_h0(
            h0_value,
            info=h0_info,
            checkpoint=checkpoint,
            source_sha256=source_sha256,
            derived_sha256=derived_sha256,
            owners_by_image=owners_by_image,
        )
        h0_payloads[checkpoint] = h0_payload
        h0_by_checkpoint[checkpoint] = h0_by_owner
        h0_meta[checkpoint] = meta
        support_value, support_info = _load_envelope(support_ledgers[checkpoint], f"{checkpoint} support")
        support_payload, support_by_owner, support_info_meta = _validate_support(
            support_value,
            info=support_info,
            checkpoint=checkpoint,
            source_sha256=source_sha256,
            derived_sha256=derived_sha256,
            h0_by_owner=h0_by_owner,
        )
        support_payloads[checkpoint] = support_payload
        support_by_checkpoint[checkpoint] = support_by_owner
        support_meta[checkpoint] = support_info_meta
    rows: list[dict[str, Any]] = []
    for checkpoint in CHECKPOINTS:
        for owner_id, h0 in h0_by_checkpoint[checkpoint].items():
            row = _row_record(
                checkpoint=checkpoint,
                h0=h0,
                support=support_by_checkpoint[checkpoint],
                h0_by_owner=h0_by_checkpoint[checkpoint],
                owners_by_image=owners_by_image,
                panel_rows=panel_rows,
            )
            row["h0_ledger_sha256"] = h0_meta[checkpoint]["sha256"]
            rows.append(row)
    rows.sort(key=lambda row: (CHECKPOINTS.index(str(row["checkpoint"])), int(row["image_id"]), int(row["source_panel_object_index"])))
    if len(rows) != EXPECTED_ROW_COUNT:
        raise CensusContractError(f"census row count must be exactly {EXPECTED_ROW_COUNT}")
    support_completion_candidates = [
        {
            "checkpoint": row["checkpoint"],
            "gt_owner_id": row["gt_owner_id"],
            "image_id": row["image_id"],
            "natural_boundary": row["natural_boundary"],
            "exact_prefix_sha256": row["exact_prefix_sha256"],
            "h0_record_index": row["h0_record_index"],
            "candidate_key": f"{row['checkpoint']}:{row['gt_owner_id']}:{row['exact_prefix_sha256']}",
        }
        for row in rows
        if row["checkpoint"] == "S" and row["native_fn"] and row["disposition"] == "support_unassessed"
    ]
    support_completion_candidates.sort(key=lambda row: (int(row["image_id"]), int(str(row["gt_owner_id"]).split(":")[-1])))
    if len(support_completion_candidates) != EXPECTED_SUPPORT_COMPLETION_COUNTS["S"]:
        raise CensusContractError(
            "S support_completion_candidates must contain exactly "
            f"{EXPECTED_SUPPORT_COMPLETION_COUNTS['S']} rows, observed {len(support_completion_candidates)}"
        )
    counts = _counts(rows)
    records_bytes = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    records_sha256 = sha256_bytes(records_bytes)
    source_identity = {
        "source_panel": {"path": source_info.get("path"), "sha256": source_sha256},
        "derived_panel": {"path": derived_info.get("path"), "sha256": derived_sha256},
        "derived_receipt": {"path": receipt_info.get("path"), "sha256": receipt_info.get("sha256")},
        "h0_ledgers": {cp: h0_meta[cp] for cp in CHECKPOINTS},
        "support_ledgers": {cp: support_meta[cp] for cp in CHECKPOINTS},
    }
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "sealed",
        "unit_id": UNIT_ID,
        "scope": "read-only CPU census; no support values inferred for unmeasured owners",
        "assessment_scope": {"S": "expanded_in_unit", "A3": "frozen_at_prior_32"},
        "frozen_universe": {
            "checkpoint_count": 2,
            "physical_owner_count": EXPECTED_OWNER_COUNT,
            "row_count": EXPECTED_ROW_COUNT,
            "image_count": len(EXPECTED_IMAGES),
            "image_ids": list(EXPECTED_IMAGES),
            "row_key": "checkpoint:gt_owner_id",
        },
        "source_identity": source_identity,
        "support_rule": {
            "target_B": "native_fn exact natural prefix, strict covered A before boundary, verified_support=true",
            "calibration_control": "all checkpoint-native strict native_tp H0 rows; never a target-B support value",
            "unknown_support": "support_unassessed is retained as unknown, never false",
        },
        "rows": rows,
        "summary": counts,
        "support_completion_candidates": support_completion_candidates,
        "support_completion_candidates_sha256": sha256_json(support_completion_candidates),
        "records_sha256": records_sha256,
    }
    document["self_sha256"] = document_self_sha256(document)
    receipt_document: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "sealed",
        "unit_id": UNIT_ID,
        "row_count": EXPECTED_ROW_COUNT,
        "physical_owner_count": EXPECTED_OWNER_COUNT,
        "records_sha256": records_sha256,
        "support_completion_candidates_count": len(support_completion_candidates),
        "support_completion_candidates_sha256": document["support_completion_candidates_sha256"],
        "census_self_sha256": document["self_sha256"],
        "source_identity": source_identity,
    }
    receipt_document["self_sha256"] = document_self_sha256(receipt_document)
    result = {
        "census": document,
        "receipt": receipt_document,
        "records_bytes": records_bytes,
        "records_sha256": records_sha256,
    }
    if output is not None or records_output is not None or receipt_output is not None:
        if output is None or records_output is None or receipt_output is None:
            raise CensusContractError("output, records_output, and receipt_output must be supplied together")
        _write_immutable(Path(output), canonical_json_bytes(document) + b"\n")
        _write_immutable(Path(records_output), records_bytes)
        _write_immutable(Path(receipt_output), canonical_json_bytes(receipt_document) + b"\n")
        result["paths"] = {
            "census": str(Path(output).expanduser().resolve()),
            "records": str(Path(records_output).expanduser().resolve()),
            "receipt": str(Path(receipt_output).expanduser().resolve()),
        }
    return result


def _write_immutable(path: Path, payload: bytes) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if resolved.exists():
        if resolved.read_bytes() != payload:
            raise FileExistsError(f"refusing to overwrite immutable census artifact: {resolved}")
        return
    resolved.write_bytes(payload)


def validate_census(document: Mapping[str, Any]) -> None:
    """Validate the self-hash and the exact cardinality of a census envelope."""

    if document.get("schema_version") != SCHEMA_VERSION or document.get("unit_id") != UNIT_ID:
        raise CensusContractError("census schema/unit identity mismatch")
    if document.get("status") != "sealed":
        raise CensusContractError("census status must be sealed")
    rows = document.get("rows")
    if not isinstance(rows, list) or len(rows) != EXPECTED_ROW_COUNT:
        raise CensusContractError("census rows must contain exactly 784 rows")
    keys = [(row.get("checkpoint"), row.get("gt_owner_id")) for row in rows if isinstance(row, Mapping)]
    if len(keys) != EXPECTED_ROW_COUNT or len(set(keys)) != EXPECTED_ROW_COUNT:
        raise CensusContractError("census row keys must be unique")
    if document.get("self_sha256") != document_self_sha256(document):
        raise CensusContractError("census self_sha256 mismatch")


def _parse_selector(values: Sequence[str] | None, *, label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values or ():
        left, sep, right = value.partition("=")
        if not sep or left.upper() not in CHECKPOINTS or not right:
            raise CensusContractError(f"{label} must be CHECKPOINT=PATH")
        checkpoint = left.upper()
        if checkpoint in result:
            raise CensusContractError(f"{label} repeats checkpoint {checkpoint}")
        result[checkpoint] = Path(right)
    if set(result) != set(CHECKPOINTS):
        raise CensusContractError(f"{label} must contain S=PATH and A=PATH")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-panel", type=Path, default=DEFAULT_SOURCE_PANEL)
    parser.add_argument("--derived-panel", type=Path, default=DEFAULT_DERIVED_PANEL)
    parser.add_argument("--derived-receipt", type=Path, default=DEFAULT_DERIVED_RECEIPT)
    parser.add_argument("--h0", action="append", help="CHECKPOINT=PATH (repeat for S and A)")
    parser.add_argument("--support", action="append", help="CHECKPOINT=PATH (repeat for S and A)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--records-output", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--receipt", dest="receipt_output", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--expected-derived-panel-sha256")
    args = parser.parse_args(argv)
    h0 = _parse_selector(args.h0, label="--h0") if args.h0 else dict(DEFAULT_H0)
    support = _parse_selector(args.support, label="--support") if args.support else dict(DEFAULT_SUPPORT)
    result = build_census(
        args.source_panel,
        args.derived_panel,
        args.derived_receipt,
        h0,
        support,
        output=args.output,
        records_output=args.records_output,
        receipt_output=args.receipt_output,
        expected_derived_panel_sha256=args.expected_derived_panel_sha256,
    )
    print(json.dumps(result["receipt"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
