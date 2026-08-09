#!/usr/bin/env python3
"""Materialize the S support completion into census-v3 and its event manifest.

The reducer is deliberately CPU-only: census-v2 remains an immutable input,
and only the 220 measured S support rows are replaced.  The event manifest is
the narrow hand-off consumed by the serialization successor; it is not a
training or controller authorization.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research import materialize_static_dynamic_owner_interface_cohort as stable


UNIT_ID = "2026-08-06-natural-boundary-routing-history-replication"
BASE_SCHEMA = "natural_boundary_owner_admission_census.v1"
SCHEMA_VERSION = "natural_boundary_owner_admission_census.v3"
RECEIPT_SCHEMA_VERSION = f"{SCHEMA_VERSION}.receipt"
MANIFEST_SCHEMA_VERSION = "s_natural_boundary_admitted_event_manifest.v3"
EXPECTED_SUPPORT_RECORDS = 220
EXPECTED_IMAGES = 13
EXPECTED_BASE_CENSUS_SHA256 = "dd1c61abb9acff7f4fc42380365439ee931db604525749f297bbc3e191a26a2e"
EXPECTED_PLAN_SHA256 = "1b7e97af291b58aec50849ae09aa883148d55610c281a9edb543fce46ec21d4c"
EXPECTED_SUPPORT_LEDGER_SHA256 = "c3ab9eca420c3ee2965c7ee72bd12208cf34264ec0238c226f8d7d3b399b64f6"
CHECKPOINT = "S"
PRIMARY = {"checkpoint": "S", "step": 2444, "substrate": "four-coordinate geo_sorted_xy"}
FROZEN_ARMS = ("K00", "K01", "K10", "K11", "K12", "K13", "K14T", "K14B", "N00", "N01", "N10", "N20", "H00", "H10", "H20")


class CensusV3ContractError(ValueError):
    """Raised when census-v3 inputs are not sealed and self-consistent."""


def _source_identity(value: Any, label: str) -> dict[str, Any] | None:
    """Return a source identity object, preserving only contract fields."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise CensusV3ContractError(f"{label} identity must be an object")
    result = dict(value)
    digest = result.get("sha256")
    if digest is not None and (not isinstance(digest, str) or len(digest) != 64):
        raise CensusV3ContractError(f"{label}.sha256 must be a SHA-256")
    return result


def _load_panel_source(source: Any, label: str) -> tuple[Any, dict[str, Any]]:
    """Load JSON/JSONL panel input and reject mutable/symlinked path sources."""

    try:
        value, info = stable._read_source(source)  # noqa: SLF001
    except Exception as exc:
        raise CensusV3ContractError(f"cannot load {label}: {exc}") from exc
    if isinstance(source, (str, Path)):
        path = Path(source).expanduser()
        if path.is_symlink() or not path.is_file():
            raise CensusV3ContractError(f"{label} is not a regular non-symlink file: {path.resolve()}")
    return value, dict(info)


def _assert_raw_binding(info: Mapping[str, Any], identity: Mapping[str, Any] | None, label: str) -> None:
    if identity is None or identity.get("sha256") is None:
        return
    if info.get("sha256") != identity.get("sha256"):
        raise CensusV3ContractError(f"{label} raw SHA-256 differs from bound identity")


def _panel_dimensions(panel: Any) -> dict[int, tuple[float, float] | None]:
    try:
        return stable._panel_dimensions(panel, label="geometry source panel")  # noqa: SLF001
    except Exception as exc:
        raise CensusV3ContractError(f"geometry panel dimensions are invalid: {exc}") from exc


def _normalise_geometry_source(
    base: Mapping[str, Any],
    plan: Mapping[str, Any] | None,
    *,
    geometry_source: Mapping[str, Any] | None,
    panel_source: Any,
    derived_panel_source: Any,
    derived_receipt_source: Any,
    h0_source: Any,
    allow_missing: bool = False,
) -> tuple[Any, Any, Any, Any, dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None] | None:
    """Resolve explicit geometry inputs, falling back only to sealed lineage paths."""

    # Unit tests and successor-contract fixtures may intentionally exercise
    # census-v3 admission without geometry.  Do not partially infer a source
    # from an unrelated derived-panel identity in that mode; production
    # materialization remains fail-closed below when no complete geometry
    # source can be resolved.
    if allow_missing and geometry_source is None and all(
        value is None for value in (panel_source, derived_panel_source, derived_receipt_source, h0_source)
    ):
        return None
    envelope = dict(geometry_source or {})
    panel_source = panel_source if panel_source is not None else envelope.get("panel", envelope.get("source_panel"))
    derived_panel_source = derived_panel_source if derived_panel_source is not None else envelope.get("derived_panel")
    derived_receipt_source = derived_receipt_source if derived_receipt_source is not None else envelope.get("derived_receipt")
    h0_source = h0_source if h0_source is not None else envelope.get("h0", envelope.get("h0_ledger"))
    # Geometry-source envelopes may carry identity mappings rather than the
    # payload itself; resolve those paths while retaining the declared SHA.
    source_values = {"panel": panel_source, "derived": derived_panel_source, "receipt": derived_receipt_source, "h0": h0_source}
    for key, value in tuple(source_values.items()):
        if isinstance(value, Mapping) and value.get("path") and "objects" not in value and "records" not in value:
            source_values[key] = value["path"]
    panel_source, derived_panel_source, derived_receipt_source, h0_source = (
        source_values["panel"], source_values["derived"], source_values["receipt"], source_values["h0"]
    )
    source_identity = base.get("source_identity") if isinstance(base.get("source_identity"), Mapping) else {}
    source_identity = dict(source_identity)
    source_panel_identity = _source_identity(source_identity.get("source_panel"), "source_panel")
    derived_identity = _source_identity(source_identity.get("derived_panel"), "derived_panel")
    receipt_identity = _source_identity(source_identity.get("derived_receipt"), "derived_receipt")
    if source_panel_identity is None and isinstance(envelope.get("source_panel"), Mapping):
        source_panel_identity = _source_identity(envelope.get("source_panel"), "source_panel")
    if derived_identity is None and isinstance(envelope.get("derived_panel"), Mapping):
        derived_identity = _source_identity(envelope.get("derived_panel"), "derived_panel")
    if receipt_identity is None and isinstance(envelope.get("derived_receipt"), Mapping):
        receipt_identity = _source_identity(envelope.get("derived_receipt"), "derived_receipt")
    h0_identity: dict[str, Any] | None = None
    if plan is not None:
        h0_lineage = plan.get("h0_lineage")
        if isinstance(h0_lineage, Mapping):
            h0_identity = _source_identity(h0_lineage.get("source"), "h0_ledger")
    if h0_identity is None:
        h0_binding = envelope.get("h0_ledger", envelope.get("h0"))
        if isinstance(h0_binding, Mapping):
            h0_identity = _source_identity(h0_binding, "h0_ledger")
    if panel_source is None and source_panel_identity and source_panel_identity.get("path"):
        panel_source = source_panel_identity["path"]
    if derived_panel_source is None and derived_identity and derived_identity.get("path"):
        derived_panel_source = derived_identity["path"]
    if derived_receipt_source is None and receipt_identity and receipt_identity.get("path"):
        derived_receipt_source = receipt_identity["path"]
    if h0_source is None and h0_identity and h0_identity.get("path"):
        h0_source = h0_identity["path"]
    sources = (panel_source, derived_panel_source, derived_receipt_source, h0_source)
    if all(source is None for source in sources):
        return None
    if any(source is None for source in sources):
        raise CensusV3ContractError("geometry source requires panel, derived panel, derived receipt, and exact H0 ledger")
    panel, panel_info = _load_panel_source(panel_source, "source panel")
    derived, derived_info = _load_panel_source(derived_panel_source, "derived panel")
    receipt, receipt_info = _load_panel_source(derived_receipt_source, "derived receipt")
    h0, h0_info = _load_panel_source(h0_source, "H0 ledger")
    if not isinstance(receipt, Mapping) or not isinstance(h0, Mapping):
        raise CensusV3ContractError("geometry receipt and H0 ledger must be JSON objects")
    _assert_raw_binding(panel_info, source_panel_identity, "source panel")
    _assert_raw_binding(derived_info, derived_identity, "derived panel")
    _assert_raw_binding(receipt_info, receipt_identity, "derived receipt")
    _assert_raw_binding(h0_info, h0_identity, "H0 ledger")
    return panel, derived, receipt, h0, panel_info, derived_info, receipt_info, h0_info


def _geometry_context(
    sources: tuple[Any, Any, Any, Any, dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None],
    support_records: Sequence[Mapping[str, Any]],
    support_info: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    panel, derived, receipt, h0, panel_info, derived_info, receipt_info, h0_info = sources
    source_sha = str(panel_info["sha256"] if panel_info else sha256_json(panel))
    derived_sha = str(derived_info["sha256"] if derived_info else sha256_json(derived))
    try:
        panel_owners = stable._validate_derived_panel(  # noqa: SLF001
            panel,
            derived,
            receipt,
            source_sha256=source_sha,
            derived_sha256=derived_sha,
        )
    except Exception as exc:
        raise CensusV3ContractError(f"geometry source panel binding is invalid: {exc}") from exc
    dimensions = _panel_dimensions(derived)
    try:
        envelope, h0_records = stable._validate_ledger_envelope(  # noqa: SLF001
            h0,
            source_info=h0_info or {"sha256": sha256_json(h0)},
            source_panel_sha256=source_sha,
            derived_panel_sha256=derived_sha,
            panel=panel_owners,
            source_kind="h0",
        )
        image_plan = stable._validate_image_plan_identities(  # noqa: SLF001
            h0_records,
            panel_dimensions=dimensions,
            context="census-v3 geometry H0",
        )
    except Exception as exc:
        raise CensusV3ContractError(f"exact H0 geometry source is invalid: {exc}") from exc
    h0_by_owner: dict[str, Mapping[str, Any]] = {}
    for record in h0_records:
        owner = record.get("owner")
        owner_id = getattr(owner, "gt_owner_id", None)
        if owner_id is None:
            owner_id = record.get("gt_owner_id")
        if owner_id is not None:
            h0_by_owner[str(owner_id)] = record
    support_by_owner = {str(record.get("gt_owner_id")): record for record in support_records}
    verified_ids = {
        owner_id
        for owner_id, record in support_by_owner.items()
        if record.get("verified_support") is True
    }
    return {
        "panel_owners": panel_owners,
        "image_plan": image_plan,
        "h0_by_owner": h0_by_owner,
        "h0_records": h0_records,
        "support_by_owner": support_by_owner,
        "verified_ids": verified_ids,
        "source_sha256": source_sha,
        "derived_sha256": derived_sha,
        "h0_sha256": h0_info.get("sha256") if h0_info else sha256_json(h0),
        "h0_info": dict(h0_info or {}),
        "support_sha256": support_info.get("sha256") if support_info else None,
        "support_info": dict(support_info or {}),
        "panel_info": panel_info,
        "derived_info": derived_info,
        "receipt_info": receipt_info,
    }


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CensusV3ContractError(f"non-finite/non-canonical JSON: {exc}") from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def document_hash(value: Mapping[str, Any], field: str = "self_sha256") -> str:
    body = dict(value)
    body.pop(field, None)
    return sha256_json(body)


def _read_json(source: str | Path | Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if isinstance(source, (str, Path)):
        raw_path = Path(source).expanduser()
        if raw_path.is_symlink() or not raw_path.is_file():
            raise CensusV3ContractError(f"source is not a regular non-symlink file: {raw_path}")
        path = raw_path.resolve(strict=True)
        raw = path.read_bytes()
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CensusV3ContractError(f"invalid JSON: {path}") from exc
        info = {"path": str(path), "sha256": sha256_bytes(raw)}
    elif isinstance(source, Mapping):
        value = dict(source)
        info = {"inline": True, "sha256": sha256_json(value)}
    else:
        raise CensusV3ContractError("source must be a JSON path or object")
    if not isinstance(value, Mapping):
        raise CensusV3ContractError("JSON source must be an object")
    return dict(value), info


def _write_once(path: str | Path, payload: bytes) -> str:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.read_bytes() != payload:
        raise CensusV3ContractError(f"refusing to overwrite immutable artifact: {destination}")
    if not destination.exists():
        destination.write_bytes(payload)
    return sha256_bytes(payload)


def _validate_base(base: Mapping[str, Any]) -> list[dict[str, Any]]:
    if base.get("schema_version") != BASE_SCHEMA or base.get("unit_id") != UNIT_ID or base.get("status") != "sealed":
        raise CensusV3ContractError("base census must be sealed census-v1 for the new unit")
    if base.get("self_sha256") != document_hash(base):
        raise CensusV3ContractError("base census self_sha256 mismatch")
    rows = base.get("rows")
    if not isinstance(rows, list) or len(rows) != 784 or any(not isinstance(row, Mapping) for row in rows):
        raise CensusV3ContractError("base census must contain exactly 784 rows")
    keys = [(row.get("checkpoint"), row.get("gt_owner_id")) for row in rows]
    if len(set(keys)) != len(keys):
        raise CensusV3ContractError("base census row keys are not unique")
    return [dict(row) for row in rows]


def _load_support(
    source: str | Path | Mapping[str, Any],
    *,
    calibration: Mapping[str, Any] | None = None,
    support_rule: Mapping[str, Any] | None = None,
    required_raw_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    ledger, info = _read_json(source)
    if ledger.get("schema_version") != "natural_boundary_owner_support_completion_ledger.v1" or ledger.get("status") != "completed" or ledger.get("checkpoint") != "S":
        raise CensusV3ContractError("support ledger is not the completed v1 reducer output")
    records = ledger.get("records")
    if not isinstance(records, list) or len(records) != EXPECTED_SUPPORT_RECORDS:
        raise CensusV3ContractError("support ledger must contain exactly 220 records")
    if ledger.get("records_sha256") != sha256_json(records) or ledger.get("content_sha256") != document_hash(ledger, "content_sha256"):
        raise CensusV3ContractError("support ledger hash mismatch")
    if required_raw_sha256 is not None and info.get("sha256") != required_raw_sha256:
        raise CensusV3ContractError("support ledger raw SHA-256 is not the sealed geometry-successor merge-v6")
    owners = [str(row.get("gt_owner_id")) for row in records if isinstance(row, Mapping)]
    if len(owners) != len(set(owners)) or any(not owner.startswith("gt:") for owner in owners):
        raise CensusV3ContractError("support ledger owner identities are not unique")
    image_ids = {int(row.get("image_id")) for row in records if isinstance(row, Mapping) and isinstance(row.get("image_id"), int)}
    if len(image_ids) != EXPECTED_IMAGES:
        raise CensusV3ContractError(f"support ledger must cover {EXPECTED_IMAGES} images, observed {len(image_ids)}")
    if calibration is not None:
        expected_calibration = calibration.get("calibration_sha256")
        if ledger.get("calibration", {}).get("calibration_sha256") != expected_calibration:
            raise CensusV3ContractError("support ledger calibration is detached from sealed plan")
        if support_rule is not None and ledger.get("support_rule") != support_rule:
            raise CensusV3ContractError("support ledger support rule is detached from sealed plan")
    return ledger, info


def _load_plan(source: str | Path | Mapping[str, Any] | None) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if source is None:
        return None, None
    plan, info = _read_json(source)
    if plan.get("unit_id") != UNIT_ID or plan.get("checkpoint") != CHECKPOINT or not str(plan.get("status", "")).startswith("sealed"):
        raise CensusV3ContractError("plan identity is not the sealed S support plan")
    if plan.get("plan_content_sha256") != sha256_json({key: value for key, value in plan.items() if key != "plan_content_sha256"}):
        raise CensusV3ContractError("plan content hash mismatch")
    return plan, info


def _update_row(row: dict[str, Any], record: Mapping[str, Any]) -> dict[str, Any]:
    if row.get("checkpoint") != CHECKPOINT or row.get("gt_owner_id") != record.get("gt_owner_id"):
        raise CensusV3ContractError(f"support owner {record.get('gt_owner_id')} is not an S census row")
    required = ("checkpoint", "image_id", "native_fn", "strict_complete_row", "natural_boundary", "natural_boundary_valid", "covered_owner_ids", "exact_prefix_token_ids", "exact_prefix_sha256", "support_calibration_sha256", "support_rule", "no_future_or_intervention_leakage")
    if any(key not in record for key in required):
        raise CensusV3ContractError(f"support owner {row.get('gt_owner_id')} omits required sealed fields")
    for key in required[:7]:
        if record.get(key) != row.get(key):
            raise CensusV3ContractError(f"support owner {row.get('gt_owner_id')} {key} identity drifted")
    if row.get("native_fn") is not True or row.get("strict_complete_row") is not False:
        raise CensusV3ContractError(f"support owner {row.get('gt_owner_id')} is not a native FN")
    if row.get("exact_prefix_sha256") != record.get("exact_prefix_sha256"):
        raise CensusV3ContractError(f"support owner {row.get('gt_owner_id')} exact-prefix identity drifted")
    prefix = record.get("exact_prefix_token_ids")
    if not isinstance(prefix, list) or not prefix or any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in prefix):
        raise CensusV3ContractError(f"support owner {row.get('gt_owner_id')} has invalid exact-prefix tokens")
    if record.get("exact_prefix_sha256") != sha256_json(prefix):
        raise CensusV3ContractError(f"support owner {row.get('gt_owner_id')} exact-prefix hash mismatch")
    features = record.get("support_features")
    if not isinstance(features, Mapping) or features.get("assessed") is not True or any(key not in features or not math.isfinite(float(features.get(key))) for key in ("peak_lift", "local_concentration")):
        raise CensusV3ContractError(f"support owner {row.get('gt_owner_id')} has unassessed support features")
    updated = dict(row)
    updated["support_status"] = "measured"
    updated["support_record_present"] = True
    updated["support_verified"] = record.get("verified_support")
    updated["target_B_support"] = record.get("verified_support") is True
    updated["target_B_support_assessed"] = True
    updated["support_record_sha256"] = sha256_json(dict(record))
    updated["support_calibration_sha256"] = record.get("support_calibration_sha256")
    updated["support_features"] = dict(features)
    updated["exact_prefix_token_ids"] = list(record.get("exact_prefix_token_ids", []))
    updated["natural_boundary"] = record.get("natural_boundary", updated.get("natural_boundary"))
    updated["covered_owner_ids"] = list(record.get("covered_owner_ids", updated.get("covered_owner_ids", [])))
    updated["candidate_score_count"] = record.get("candidate_score_count")
    updated["candidate_scores_sha256"] = record.get("candidate_scores_sha256")
    if record.get("verified_support") is True and updated.get("eligible_except_support") is True and updated.get("geometry", {}).get("launch_eligible") is True:
        updated["disposition"] = "eligible_verified_pair"
    else:
        updated["disposition"] = "support_measured_not_verified"
    return updated


def _geometry_from_context(
    row: Mapping[str, Any],
    *,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute frozen full geometry for one S owner from authoritative inputs."""

    owner_id = str(row.get("gt_owner_id"))
    image_id = int(row.get("image_id"))
    panel_owners = context["panel_owners"]
    owners = list(panel_owners.get(image_id, ()))
    owner_by_id = {owner.gt_owner_id: owner for owner in owners}
    h0_by_owner = context["h0_by_owner"]
    target_record = h0_by_owner.get(owner_id)
    if target_record is None:
        raise CensusV3ContractError(f"geometry H0 ledger lacks census owner {owner_id}")
    target_owner = target_record.get("owner")
    target_source = getattr(target_owner, "source_index", target_record.get("source_panel_object_index"))
    target_derived = getattr(target_owner, "derived_index", target_record.get("derived_panel_object_index"))
    if target_source != row.get("source_panel_object_index"):
        raise CensusV3ContractError(f"geometry H0 source-panel index drifted for {owner_id}")
    if row.get("derived_panel_object_index") is None:
        derived_index = target_derived
    else:
        derived_index = row.get("derived_panel_object_index")
        if target_derived != derived_index:
            raise CensusV3ContractError(f"geometry H0 derived-panel index drifted for {owner_id}")
    if owner_id not in owner_by_id:
        raise CensusV3ContractError(f"geometry panel lacks census owner {owner_id}")
    if target_record.get("exact_prefix_sha256") != row.get("exact_prefix_sha256"):
        raise CensusV3ContractError(f"geometry H0 exact-prefix identity drifted for {owner_id}")
    for key in ("native_fn", "strict_complete_row", "natural_boundary_valid", "natural_boundary"):
        if target_record.get(key) != row.get(key):
            raise CensusV3ContractError(f"geometry H0 {key} identity drifted for {owner_id}")
    covered_ids = [str(value) for value in row.get("covered_owner_ids", [])]
    covered_a = row.get("covered_A_owner_id")
    if not isinstance(covered_a, str) or covered_a not in covered_ids:
        raise CensusV3ContractError(f"geometry covered A is not bound to target B {owner_id}")
    if covered_a not in owner_by_id or covered_a not in h0_by_owner:
        raise CensusV3ContractError(f"geometry H0/panel lacks covered A {covered_a}")
    a_record = h0_by_owner[covered_a]
    if a_record.get("strict_complete_row") is not True or not isinstance(a_record.get("natural_boundary"), (int, float)):
        raise CensusV3ContractError(f"geometry covered A is not strict-complete: {covered_a}")
    if float(a_record.get("natural_boundary")) >= float(row.get("natural_boundary")):
        raise CensusV3ContractError(f"geometry covered A is not before target B: {covered_a} -> {owner_id}")
    identity = context["image_plan"].get(image_id)
    if identity is None:
        raise CensusV3ContractError(f"geometry H0 image plan is missing for image {image_id}")
    # Delegate the region semantics and geometry hash to the frozen cohort
    # implementation.  Support-completion rows are adapted into the cohort's
    # owner-record shape here; this is an identity-preserving adapter, not a
    # fallback to the legacy 32-owner event pool.
    owner_records: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for h0_record in context.get("h0_records", ()):
        owner = h0_record.get("owner")
        owner_key = getattr(owner, "gt_owner_id", None)
        if owner_key is None:
            owner_key = h0_record.get("gt_owner_id")
        if owner_key is None:
            continue
        key = (CHECKPOINT, int(h0_record.get("image_id")), str(owner_key))
        owner_records.setdefault(key, []).append(dict(h0_record))
    for support_owner_id, support_record in context.get("support_by_owner", {}).items():
        if support_record.get("verified_support") is not True:
            continue
        h0_record = context["h0_by_owner"].get(str(support_owner_id))
        if h0_record is None:
            continue
        adapted = dict(h0_record)
        adapted.update(dict(support_record))
        adapted["owner"] = h0_record.get("owner")
        adapted["source_kind"] = "support"
        adapted["boundary"] = h0_record.get("boundary", h0_record.get("natural_boundary"))
        adapted["covered_owner_refs"] = h0_record.get("covered_owner_refs", h0_record.get("covered_owner_ids", []))
        adapted["verified_support"] = True
        key = (CHECKPOINT, int(h0_record.get("image_id")), str(support_owner_id))
        owner_records.setdefault(key, []).append(adapted)
    target_h0 = context["h0_by_owner"][owner_id]
    a_h0 = context["h0_by_owner"][str(covered_a)]
    pair = {
        "pair_status": "verified_pair",
        "A_latest_covered": {
            "gt_owner_id": str(covered_a),
            "strict_complete_row": True,
            "natural_boundary": a_h0.get("natural_boundary"),
        },
        "B_verified_uncovered": {
            "gt_owner_id": owner_id,
            "verified_support": True,
            "strict_complete_row": False,
            "natural_boundary": target_h0.get("natural_boundary"),
            "exact_prefix_sha256": target_h0.get("exact_prefix_sha256"),
            "covered_owner_ids": covered_ids,
        },
    }
    try:
        frozen_geometry = stable._materialize_event_geometry(  # noqa: SLF001
            candidate={"image_id": image_id, "gt_owner_id": owner_id},
            checkpoint=CHECKPOINT,
            pair=pair,
            panel_owners=panel_owners,
            owner_records=owner_records,
            image_plan=context["image_plan"],
            source_panel_sha256=context["source_sha256"],
            derived_panel_sha256=context["derived_sha256"],
            h0_sources=[{"sha256": context["h0_sha256"]}],
            support_sources=[{"sha256": context.get("support_sha256")}],
        )
    except Exception as exc:
        raise CensusV3ContractError(f"frozen geometry computation failed for {owner_id}: {exc}") from exc
    return frozen_geometry


def _bank_provenance(
    records: Sequence[Mapping[str, Any]],
    info: Mapping[str, Any],
    *,
    population: str,
) -> dict[str, Any]:
    return {
        "path": info.get("path"),
        "raw_sha256": info.get("sha256"),
        "record_count": len(records),
        "verified_count": sum(record.get("verified_support") is True for record in records),
        "population": population,
    }


def _region_delta(old: Mapping[str, Any], new: Mapping[str, Any], key: str, owner_id: str) -> dict[str, Any]:
    old_regions = old.get("image_cell_regions")
    new_regions = new.get("image_cell_regions")
    old_receipts = old.get("image_cell_region_receipts")
    new_receipts = new.get("image_cell_region_receipts")
    if not all(isinstance(value, Mapping) for value in (old_regions, new_regions, old_receipts, new_receipts)):
        raise CensusV3ContractError(f"geometry supersession lacks {key} region/receipt for {owner_id}")
    old_cells = list(old_regions.get(key, ()))
    new_cells = list(new_regions.get(key, ()))
    if old_cells != list(old_receipts.get(key, {}).get("cell_indices", ())):
        raise CensusV3ContractError(f"old geometry {key} receipt differs from its cells for {owner_id}")
    if new_cells != list(new_receipts.get(key, {}).get("cell_indices", ())):
        raise CensusV3ContractError(f"new geometry {key} receipt differs from its cells for {owner_id}")
    old_set, new_set = set(old_cells), set(new_cells)
    if len(old_set) != len(old_cells) or len(new_set) != len(new_cells):
        raise CensusV3ContractError(f"geometry {key} repeats a cell for {owner_id}")
    if not new_set.issubset(old_set):
        raise CensusV3ContractError(f"geometry supersession added {key} cells for {owner_id}")
    return {
        "before_cells": old_cells,
        "after_cells": new_cells,
        "added_cells": sorted(new_set - old_set),
        "removed_cells": sorted(old_set - new_set),
        "before_receipt": copy.deepcopy(dict(old_receipts[key])),
        "after_receipt": copy.deepcopy(dict(new_receipts[key])),
    }


def _old_replay(
    row: Mapping[str, Any],
    *,
    target_record: Mapping[str, Any],
    target_source: Mapping[str, Any],
    target_population: str,
    injection_required: bool,
    old_bank: Mapping[str, Any],
    recomputed_old: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the fixed-B operand required by the frozen geometry API."""

    owner_id = str(row["gt_owner_id"])
    if (
        str(target_record.get("gt_owner_id")) != owner_id
        or int(target_record.get("image_id", -1)) != int(row["image_id"])
        or target_record.get("exact_prefix_sha256") != row.get("exact_prefix_sha256")
    ):
        raise CensusV3ContractError(f"old replay fixed target identity drifted for {owner_id}")
    source_sha = target_source.get("sha256")
    if not isinstance(source_sha, str) or len(source_sha) != 64:
        raise CensusV3ContractError(f"old replay fixed target source is unbound for {owner_id}")
    b_binding = recomputed_old.get("b_support_binding")
    if not isinstance(b_binding, Mapping) or (
        b_binding.get("gt_owner_id") != owner_id
        or b_binding.get("source_kind") != "support"
        or b_binding.get("verified_support") is not True
    ):
        raise CensusV3ContractError(f"old replay fixed target B binding drifted for {owner_id}")
    adapted_b_sha = b_binding.get("record_sha256")
    if not isinstance(adapted_b_sha, str) or len(adapted_b_sha) != 64 or any(ch not in "0123456789abcdef" for ch in adapted_b_sha):
        raise CensusV3ContractError(f"old replay fixed target B binding hash is invalid for {owner_id}")
    fixed_target = {
        "injection_required": injection_required,
        "target_owner_id": owner_id,
        "target_image_id": int(row["image_id"]),
        "target_exact_prefix_sha256": target_record["exact_prefix_sha256"],
        "source_record_semantic_sha256": sha256_json(dict(target_record)),
        "adapted_b_binding_record_sha256": adapted_b_sha,
        "source_ledger": {"path": target_source.get("path"), "raw_sha256": source_sha},
        "source_population_identity": target_population,
        "role": "fixed_pair_target_B",
        "included_in_old_competitor_population": False,
        "adapter_reason": (
            "old target not verified; stable geometry API support gate requires a fixed B operand"
            if injection_required
            else "old verified target record supplies the fixed B operand"
        ),
    }
    dependencies = [old_bank.get("raw_sha256")]
    if injection_required:
        dependencies.append(source_sha)
    if any(not isinstance(value, str) or len(value) != 64 for value in dependencies):
        raise CensusV3ContractError(f"old replay dependencies are unbound for {owner_id}")
    return {
        "operator": {
            "id": "stable._materialize_event_geometry",
            "semantics": "frozen_geometry_operator_old_verified_bank_plus_fixed_pair_target_B",
        },
        "fixed_target_operand": fixed_target,
        "dependency_raw_sha256s": dependencies,
        "replay_dependency_semantic_sha256": sha256_json(
            {"operator": "stable._materialize_event_geometry", "old_bank": old_bank, "fixed_target_operand": fixed_target}
        ),
    }


def _geometry_supersession(
    row: Mapping[str, Any],
    *,
    stored_old: Mapping[str, Any],
    recomputed_old: Mapping[str, Any],
    new: Mapping[str, Any],
    old_bank: Mapping[str, Any],
    new_bank: Mapping[str, Any],
    old_verified: Mapping[str, Mapping[str, Any]],
    new_verified_native_fn: Mapping[str, Mapping[str, Any]],
    old_replay: Mapping[str, Any],
) -> dict[str, Any]:
    """Explain a complete-native-FN geometry replacement, or reject it."""

    owner_id = str(row["gt_owner_id"])
    for key in ("a_exclusive", "b_exclusive", "background"):
        # The stored census-v2 operand geometry is immutable evidence.  It
        # must be reproduced exactly with the old verified bank and the same
        # frozen operator before any population change can be explained.
        old_replay_delta = _region_delta(stored_old, recomputed_old, key, owner_id)
        if (
            old_replay_delta["added_cells"]
            or old_replay_delta["removed_cells"]
            or old_replay_delta["before_receipt"] != old_replay_delta["after_receipt"]
        ):
            raise CensusV3ContractError(f"old geometry recomputation differs from census-v2 for {owner_id}")
    old_ids = list(recomputed_old.get("owner_region_owner_ids", ()))
    new_ids = list(new.get("owner_region_owner_ids", ()))
    if len(set(old_ids)) != len(old_ids) or len(set(new_ids)) != len(new_ids):
        raise CensusV3ContractError(f"geometry supersession repeats region owners for {owner_id}")
    old_set, new_set = set(old_ids), set(new_ids)
    added_ids = sorted(new_set - old_set)
    removed_ids = sorted(old_set - new_set)
    for added in added_ids:
        record = new_verified_native_fn.get(added)
        if record is None or int(record.get("image_id", -1)) != int(row["image_id"]):
            raise CensusV3ContractError(f"geometry supersession has unexplained added owner {added} for {owner_id}")
    for removed in removed_ids:
        if removed not in old_verified or removed in new_verified_native_fn:
            raise CensusV3ContractError(f"geometry supersession has unexplained removed owner {removed} for {owner_id}")
    deltas = {key: _region_delta(recomputed_old, new, key, owner_id) for key in ("a_exclusive", "b_exclusive", "background")}
    background = new.get("image_cell_region_receipts", {}).get("background", {})
    if len(deltas["background"]["after_cells"]) != len(deltas["b_exclusive"]["after_cells"]) or any(
        item.get("overlap_fraction") != 0.0 for item in background.get("fractional_weights", ())
    ):
        raise CensusV3ContractError(f"geometry supersession background contract drifted for {owner_id}")
    covered_a = row.get("covered_A_owner_id")
    if not isinstance(covered_a, str) or covered_a not in row.get("covered_owner_ids", []):
        raise CensusV3ContractError(f"geometry supersession has no frozen covered A for {owner_id}")
    for geometry in (recomputed_old, new):
        if covered_a not in geometry.get("covered_owner_ids_at_b_boundary", ()):
            raise CensusV3ContractError(f"geometry supersession changed covered A for {owner_id}")
    old_competitor = recomputed_old.get("same_class_competitor_owner_id")
    return {
        "decision": "k13_competitor_population=complete_native_fn_bank",
        "old_replay": copy.deepcopy(dict(old_replay)),
        "old_bank": dict(old_bank),
        "new_bank": dict(new_bank),
        "covered_A_owner_id_before": covered_a,
        "covered_A_owner_id_after": covered_a,
        "old_effective_region_owner_ids": old_ids,
        "new_effective_region_owner_ids": new_ids,
        "added_effective_region_owner_ids": added_ids,
        "removed_effective_region_owner_ids": removed_ids,
        "regions": deltas,
        "competitor": {
            "before_owner_id": old_competitor,
            "after_owner_id": new.get("same_class_competitor_owner_id"),
            "before_geometry_applicability": "applicable" if old_competitor is not None else "not_applicable",
            "after_geometry_applicability": "applicable" if new.get("same_class_competitor_owner_id") is not None else "not_applicable",
            "historical_reuse": False,
            "historical_execution_status": "retired_historical_diagnostic" if owner_id == "gt:5001:15" else "not_executed",
        },
        "old_geometry_sha256": stored_old.get("geometry_sha256"),
        "old_replay_geometry_semantic_sha256": recomputed_old.get("geometry_sha256"),
        "new_geometry_sha256": new.get("geometry_sha256"),
        "operand_adequacy": {
            "a_exclusive": {"status": "nonempty" if deltas["a_exclusive"]["after_cells"] else "empty", "cell_count": len(deltas["a_exclusive"]["after_cells"])},
            "launch_eligibility": "frozen_nonempty_rule",
        },
    }


def _legacy_context_event(row: Mapping[str, Any], geometry: Mapping[str, Any], ordinal: int) -> dict[str, Any]:
    """Adapt one census row into the legacy owner-interface context contract."""

    owner_id = str(row["gt_owner_id"])
    covered_a = row.get("covered_A_owner_id")
    if not isinstance(covered_a, str):
        raise CensusV3ContractError(f"legacy context {owner_id} lacks covered_A_owner_id")
    a_boundary = row.get("covered_A_natural_boundary")
    b_boundary = row.get("natural_boundary")
    if not isinstance(a_boundary, (int, float)) or not isinstance(b_boundary, (int, float)) or a_boundary >= b_boundary:
        raise CensusV3ContractError(f"legacy context {owner_id} has invalid A/B boundary")
    owner_regions = geometry.get("owner_regions")
    a_region = owner_regions.get(covered_a) if isinstance(owner_regions, Mapping) else None
    a_source = a_region.get("source") if isinstance(a_region, Mapping) else None
    if not isinstance(a_source, Mapping) or not isinstance(a_source.get("source_panel_object_index"), int):
        raise CensusV3ContractError(f"legacy context {owner_id} lacks exact covered-A source index")
    target_region = owner_regions.get(owner_id) if isinstance(owner_regions, Mapping) else None
    target_source = target_region.get("source") if isinstance(target_region, Mapping) else None
    if not isinstance(target_source, Mapping):
        raise CensusV3ContractError(f"legacy context {owner_id} lacks target-owner provenance")
    if target_source.get("source_panel_object_index") != row.get("source_panel_object_index"):
        raise CensusV3ContractError(f"legacy context {owner_id} source index differs from geometry provenance")
    if target_source.get("derived_panel_object_index") != row.get("derived_panel_object_index"):
        raise CensusV3ContractError(f"legacy context {owner_id} derived index differs from geometry provenance")
    pair = {
        "pair_status": "verified_pair",
        "A_latest_covered": {
            "gt_owner_id": covered_a,
            "source_panel_object_index": a_source["source_panel_object_index"],
            "natural_boundary": a_boundary,
            "strict_complete_row": True,
        },
        "B_verified_uncovered": {
            "gt_owner_id": owner_id,
            "verified_support": True,
            "strict_complete_row": False,
            "natural_boundary": b_boundary,
            "exact_prefix_sha256": row.get("exact_prefix_sha256"),
            "covered_owner_ids": list(row.get("covered_owner_ids", [])),
        },
    }
    event: dict[str, Any] = {
        "ordinal": ordinal,
        "image_id": int(row["image_id"]),
        "gt_owner_id": owner_id,
        "source_panel_object_index": row.get("source_panel_object_index"),
        "category": row.get("category_name", row.get("category")),
        "pixel_bbox": list(row.get("bbox_pixel_xyxy", row.get("pixel_bbox", []))),
        "panel_identity": {
            "source_panel_object_index": row.get("source_panel_object_index"),
            "derived_panel_object_index": row.get("derived_panel_object_index"),
            "coco_ann_id": target_source.get("coco_ann_id"),
            "status": "matched",
            "reason": None,
        },
        "checkpoint_status": {
            CHECKPOINT: {
                "disposition": "established",
                "native_tp": False,
                "native_fn": True,
                "strict_complete_row": False,
                "natural_boundary_valid": True,
                "natural_boundary": b_boundary,
                "boundary_record": row.get("h0_record_index"),
            }
        },
        "A_B": {CHECKPOINT: pair},
        "geometry_by_checkpoint": {CHECKPOINT: dict(geometry)},
        "SO_exclusive_shared_decomposition": geometry.get("SO_exclusive_shared_decomposition"),
        "image_cell_regions": dict(geometry.get("image_cell_regions", {})),
        "image_cell_region_receipts": dict(geometry.get("image_cell_region_receipts", {})),
        "verified_support_owner_ids": list(geometry.get("verified_support_owner_ids", [])),
        "exact_b_boundary_verified_support_owner_ids": list(geometry.get("exact_b_boundary_verified_support_owner_ids", [])),
        "verified_support_owner_ids_uncovered": list(geometry.get("verified_support_owner_ids_uncovered", [])),
        "verified_support_owner_ids_covered": list(geometry.get("verified_support_owner_ids_covered", [])),
        "verified_support_owner_roles": dict(geometry.get("verified_support_owner_roles", {})),
        "owner_region_owner_ids": list(geometry.get("owner_region_owner_ids", [])),
        "owner_region_roles": dict(geometry.get("owner_region_roles", {})),
        "owner_region_evidence": dict(geometry.get("owner_region_evidence", {})),
        "covered_owner_ids_at_b_boundary": list(geometry.get("covered_owner_ids_at_b_boundary", row.get("covered_owner_ids", []))),
        "target_owner_id": owner_id,
        "b_support_binding": geometry.get("b_support_binding"),
        "owner_regions": dict(geometry.get("owner_regions", {})),
        "geometry_identity": geometry.get("image_plan_identity"),
        "geometry_status": geometry.get("status"),
        "geometry_mechanical_disposition": geometry.get("mechanical_disposition"),
        "geometry_launch_eligible": geometry.get("launch_eligible", False),
        "geometry_sha256": geometry.get("geometry_sha256"),
        "disposition": "established" if geometry.get("launch_eligible") is True else "indeterminate",
    }
    return event


def _legacy_context_cohort(
    rows: Sequence[Mapping[str, Any]],
    *,
    geometry_context: Mapping[str, Any],
    support_info: Mapping[str, Any],
    cohort_path: str | Path | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected = [
        row for row in rows
        if row.get("checkpoint") == CHECKPOINT
        and row.get("native_fn") is True
        and row.get("disposition") in {"eligible_verified_pair", "dynamic_only"}
        and isinstance(row.get("geometry"), Mapping)
    ]
    events = [_legacy_context_event(row, row["geometry"], index + 1) for index, row in enumerate(selected)]
    cohort: dict[str, Any] = {
        "schema_version": stable.SCHEMA_VERSION,
        "unit_id": stable.UNIT_ID,
        "execution_contract": {
            "cpu_only": True,
            "h0_execution": False,
            "gpu_launch": False,
            "all_admitted_s_contexts": True,
            "selection": "none",
        },
        "frozen_pool": {
            "count": len(events),
            "sha256": sha256_json([{"gt_owner_id": event["gt_owner_id"], "image_id": event["image_id"]} for event in events]),
            "effective_sha256": sha256_json([{"gt_owner_id": event["gt_owner_id"], "image_id": event["image_id"]} for event in events]),
            "required_images": sorted({event["image_id"] for event in events}),
            "active_checkpoint": CHECKPOINT,
            "native_tp_replacements": [],
        },
        "retention": {"min_events": len(events), "max_events": len(events), "retained_events": len(events), "status": "all_admitted_s_contexts"},
        "events": events,
        "subsets": {"all_admitted_S": {"event_count": len(events), "owner_ids": [event["gt_owner_id"] for event in events]}},
        "sources": {
            "source_panel": {"path": geometry_context.get("panel_info", {}).get("path"), "sha256": geometry_context.get("panel_info", {}).get("sha256")},
            "derived_panel": {"path": geometry_context.get("derived_info", {}).get("path"), "sha256": geometry_context.get("derived_info", {}).get("sha256")},
            "derived_receipt": {"path": geometry_context.get("receipt_info", {}).get("path"), "sha256": geometry_context.get("receipt_info", {}).get("sha256")},
            "h0_ledgers": [{"path": geometry_context.get("h0_sha256") and geometry_context.get("h0_info", {}).get("path"), "sha256": geometry_context.get("h0_sha256")}],
            "support_ledgers": [{"path": support_info.get("path"), "sha256": support_info.get("sha256")}],
        },
        "indeterminate_policy": {"dynamic_only": "retained_for_explicit_nonstatic_disposition", "selection": "none"},
    }
    # Replace the H0 path placeholder with the bound path when present.
    h0_source = cohort["sources"]["h0_ledgers"][0]
    if not h0_source.get("path"):
        h0_source["path"] = geometry_context.get("h0_info", {}).get("path")
    # Outputs are written as canonical JSON followed by one newline.  Bind
    # the companion manifest to the exact bytes that a successor loader reads,
    # rather than to the in-memory semantic JSON hash.
    cohort_sha = sha256_bytes(canonical_json_bytes(cohort) + b"\n")
    manifest = {
        "schema_version": "static_dynamic_owner_interface_cohort.v1.manifest",
        "unit_id": stable.UNIT_ID,
        "status": "sealed",
        "cohort_path": str(Path(cohort_path).expanduser().resolve()) if cohort_path else None,
        "cohort_sha256": cohort_sha,
        "event_count": len(events),
        "source_hashes": {
            "source_panel": cohort["sources"]["source_panel"]["sha256"],
            "derived_panel": cohort["sources"]["derived_panel"]["sha256"],
            "derived_receipt": cohort["sources"]["derived_receipt"]["sha256"],
            "h0_ledgers": [item["sha256"] for item in cohort["sources"]["h0_ledgers"]],
            "support_ledgers": [item["sha256"] for item in cohort["sources"]["support_ledgers"]],
        },
        "owner_ids": [event["gt_owner_id"] for event in events],
        "owner_ids_sha256": sha256_json([event["gt_owner_id"] for event in events]),
    }
    return cohort, manifest


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    checkpoints = ("S", "A")
    result: dict[str, Any] = {}
    for checkpoint in checkpoints:
        selected = [row for row in rows if row.get("checkpoint") == checkpoint]
        by_image: dict[str, Any] = {}
        for image_id in sorted({int(row.get("image_id")) for row in selected}):
            image_rows = [row for row in selected if int(row.get("image_id", -1)) == image_id]
            by_image[str(image_id)] = {
                "row_count": len(image_rows),
                "native_tp": sum(bool(row.get("native_tp")) for row in image_rows),
                "native_fn": sum(bool(row.get("native_fn")) for row in image_rows),
                "target_B_support_assessed": sum(bool(row.get("target_B_support_assessed")) for row in image_rows),
                "calibration_control_assessed": sum(bool(row.get("calibration_control_assessed")) for row in image_rows),
                "support_unassessed": sum(row.get("disposition") == "support_unassessed" for row in image_rows),
                "support_measured_not_verified": sum(row.get("disposition") == "support_measured_not_verified" for row in image_rows),
                "dynamic_only": sum(row.get("disposition") == "dynamic_only" for row in image_rows),
                "eligible_verified_pair": sum(row.get("disposition") == "eligible_verified_pair" for row in image_rows),
                "eligible_except_support": sum(bool(row.get("eligible_except_support")) for row in image_rows),
                "native_already_covered": sum(row.get("disposition") == "native_already_covered" for row in image_rows),
            }
        result[checkpoint] = {
            "checkpoint_label": "S" if checkpoint == "S" else "A3",
            "row_count": len(selected),
            "native_tp": sum(bool(row.get("native_tp")) for row in selected),
            "native_fn": sum(bool(row.get("native_fn")) for row in selected),
            "target_B_support_assessed": sum(bool(row.get("target_B_support_assessed")) for row in selected),
            "target_B_support_verified": sum(row.get("target_B_support") is True for row in selected),
            "support_unassessed": sum(row.get("disposition") == "support_unassessed" for row in selected),
            "support_measured_not_verified": sum(row.get("disposition") == "support_measured_not_verified" for row in selected),
            "dynamic_only": sum(row.get("disposition") == "dynamic_only" for row in selected),
            "eligible_verified_pair": sum(row.get("disposition") == "eligible_verified_pair" for row in selected),
            "eligible_except_support": sum(bool(row.get("eligible_except_support")) for row in selected),
            "calibration_control_assessed": sum(bool(row.get("calibration_control_assessed")) for row in selected),
            "native_already_covered": sum(row.get("disposition") == "native_already_covered" for row in selected),
            "dispositions": {key: sum(row.get("disposition") == key for row in selected) for key in sorted({str(row.get("disposition")) for row in selected})},
            "by_image": by_image,
        }
    return result


def _event(row: Mapping[str, Any], event_index: int, *, rule_id: str, thresholds: Mapping[str, Any]) -> dict[str, Any]:
    prefix = row.get("exact_prefix_token_ids")
    if not isinstance(prefix, list) or not prefix or not all(isinstance(item, int) for item in prefix):
        raise CensusV3ContractError(f"admitted row {row.get('gt_owner_id')} lacks exact native prefix token IDs")
    predicates = {
        "checkpoint": row.get("checkpoint") == CHECKPOINT,
        "native_fn": row.get("native_fn") is True,
        "strict_complete_row": row.get("strict_complete_row") is False,
        "natural_boundary_valid": row.get("natural_boundary_valid") is True,
        "verified_support": row.get("support_verified") is True,
        "eligible_except_support": row.get("eligible_except_support") is True,
        "geometry_launch_eligible": row.get("geometry", {}).get("launch_eligible") is True,
        "covered_owner_ids_nonempty": bool(row.get("covered_owner_ids")),
    }
    if not all(predicates.values()):
        raise CensusV3ContractError(f"event {row.get('gt_owner_id')} failed admission predicates")
    event: dict[str, Any] = {
        "event_index": event_index,
        "event_id": f"gt:{int(row['image_id'])}:{int(str(row['gt_owner_id']).split(':')[-1])}",
        "image_id": int(row["image_id"]),
        "owner_refs": {
            "gt_owner_id": row["gt_owner_id"],
            "covered_owner_ids": list(row.get("covered_owner_ids", [])),
            "covered_A_owner_id": row.get("covered_A_owner_id"),
            "source_panel_object_index": row.get("source_panel_object_index"),
            "derived_panel_object_index": row.get("derived_panel_object_index"),
        },
        "checkpoint": CHECKPOINT,
        "step": PRIMARY["step"],
        "substrate": PRIMARY["substrate"],
        "admission": "admitted",
        "eligibility": {"admitted": True, "rule_id": rule_id, "thresholds": dict(thresholds), "predicates": predicates},
        "natural_boundary": {
            "pre_opener_natural": True,
            "opener_seeded": False,
            "opener_token_id": None,
            "opener_injected": False,
            "synthetic_opener_injections": 0,
            "opener_token_contract": {
                "status": "runner_resolved",
                "resolver": "serialization_successor_runner",
                "token_name": "<|object_ref_start|>",
                "resolution_semantics": "pre_opener_natural_prefix_ends_before_object_ref_start",
                "contract_sha256": sha256_json({"token_name": "<|object_ref_start|>", "resolver": "serialization_successor_runner"}),
            },
            "prefix_token_ids": prefix,
            "prefix_sha256": row["exact_prefix_sha256"],
            "history_token_ids": prefix,
            "history_sha256": row["exact_prefix_sha256"],
            "history_semantics": "exact_native_history_prefix",
        },
    }
    event["event_sha256"] = sha256_json(event)
    return event


def materialize(
    base_source: str | Path | Mapping[str, Any],
    support_source: str | Path | Mapping[str, Any],
    *,
    plan_source: str | Path | Mapping[str, Any] | None = None,
    geometry_source: Mapping[str, Any] | None = None,
    panel_source: str | Path | Mapping[str, Any] | Sequence[Any] | None = None,
    derived_panel_source: str | Path | Mapping[str, Any] | Sequence[Any] | None = None,
    derived_receipt_source: str | Path | Mapping[str, Any] | None = None,
    h0_source: str | Path | Mapping[str, Any] | None = None,
    output: str | Path | None = None,
    records_output: str | Path | None = None,
    receipt_output: str | Path | None = None,
    manifest_output: str | Path | None = None,
    cohort_output: str | Path | None = None,
    cohort_manifest_output: str | Path | None = None,
    test_only: bool = False,
) -> dict[str, Any]:
    base, base_info = _read_json(base_source)
    if not test_only:
        if not isinstance(base_source, (str, Path)) or base_info.get("sha256") != EXPECTED_BASE_CENSUS_SHA256:
            raise CensusV3ContractError("production materialization requires the sealed census-v2 byte hash")
        if plan_source is None:
            raise CensusV3ContractError("production materialization requires the sealed support plan")
    rows = _validate_base(base)
    plan, plan_info = _load_plan(plan_source)
    if not test_only and plan_info.get("sha256") != EXPECTED_PLAN_SHA256:
        raise CensusV3ContractError("production materialization requires the sealed plan-v1 raw byte hash")
    calibration = plan.get("calibration_reuse", {}).get("calibration") if plan else None
    support_rule = plan.get("support_lineage", {}).get("support_rule") if plan else None
    ledger, ledger_info = _load_support(
        support_source,
        calibration=calibration,
        support_rule=support_rule,
        required_raw_sha256=EXPECTED_SUPPORT_LEDGER_SHA256 if not test_only else None,
    )
    if not test_only and (not isinstance(support_source, (str, Path)) or not ledger_info.get("path")):
        raise CensusV3ContractError("production materialization requires a regular support-ledger path")
    prior_doc: dict[str, Any] | None = None
    prior_info: dict[str, Any] | None = None
    if plan is not None:
        plan_binding = ledger.get("plan_binding")
        if not isinstance(plan_binding, Mapping) or plan_binding.get("plan_content_sha256") != plan.get("plan_content_sha256"):
            raise CensusV3ContractError("support ledger is detached from the supplied sealed plan")
    by_owner = {str(row["gt_owner_id"]): row for row in rows if row.get("checkpoint") == CHECKPOINT}
    for record in ledger["records"]:
        owner = str(record.get("gt_owner_id"))
        if owner not in by_owner:
            raise CensusV3ContractError(f"support owner {owner} is absent from census S")
        by_owner[owner] = _update_row(by_owner[owner], record)
        features = record.get("support_features", {})
        if calibration is not None:
            try:
                peak = float(features.get("peak_lift"))
                concentration = float(features.get("local_concentration"))
            except (TypeError, ValueError) as exc:
                raise CensusV3ContractError(f"support owner {owner} has non-finite calibration features") from exc
            if not math.isfinite(peak) or not math.isfinite(concentration):
                raise CensusV3ContractError(f"support owner {owner} has non-finite calibration features")
            expected_verified = peak >= float(calibration["theta_peak_lift"]) + float(calibration["epsilon"]) and concentration >= float(calibration["theta_local_concentration"]) + float(calibration["epsilon"])
            if record.get("verified_support") is not expected_verified:
                raise CensusV3ContractError(f"support owner {owner} verified_support disagrees with frozen calibration")
    measured = [row for row in by_owner.values() if row.get("native_fn") is True and row.get("support_status") == "measured"]
    if len(measured) != EXPECTED_SUPPORT_RECORDS:
        raise CensusV3ContractError(f"census-v3 measured S native-FN count is {len(measured)}, expected {EXPECTED_SUPPORT_RECORDS}")
    if plan is not None:
        context_ids = {str(item.get("gt_owner_id")) for item in plan.get("contexts", [])}
        prior_binding = ledger.get("prior_support_binding")
        if not isinstance(prior_binding, Mapping) or not prior_binding.get("path") or not prior_binding.get("sha256"):
            raise CensusV3ContractError("support ledger prior lineage path/SHA is missing")
        h0_source_path = plan.get("h0_lineage", {}).get("source", {}).get("path")
        if not h0_source_path:
            raise CensusV3ContractError("sealed plan H0 source path is missing for prior-lineage derivation")
        expected_prior_path = Path(str(h0_source_path)).expanduser().resolve().with_name("s-step2444-final-support.json")
        if Path(str(prior_binding["path"])).expanduser().resolve() != expected_prior_path or prior_binding.get("sha256") != plan.get("support_lineage", {}).get("file_sha256"):
            raise CensusV3ContractError("prior support binding path/SHA differs from sealed plan lineage")
        prior_path = Path(str(prior_binding["path"])).expanduser()
        if prior_path.is_symlink() or not prior_path.is_file():
            raise CensusV3ContractError("prior support lineage is not a regular file")
        prior_raw = prior_path.read_bytes()
        if sha256_bytes(prior_raw) != prior_binding["sha256"]:
            raise CensusV3ContractError("prior support lineage raw SHA mismatch")
        prior_doc = json.loads(prior_raw)
        if not isinstance(prior_doc, Mapping) or not isinstance(prior_doc.get("records"), list):
            raise CensusV3ContractError("prior support lineage is not a record ledger")
        prior_doc = dict(prior_doc)
        prior_info = {"path": str(prior_path.resolve()), "sha256": prior_binding["sha256"]}
        prior_ids = {str(item.get("gt_owner_id")) for item in prior_doc.get("records", []) if isinstance(item, Mapping) and item.get("native_fn") is True}
        ledger_ids = {str(item.get("gt_owner_id")) for item in ledger.get("records", [])}
        if ledger_ids != context_ids | prior_ids or len(prior_ids) != 20:
            raise CensusV3ContractError("support ledger owners differ from sealed plan contexts plus prior support")
        expected_calibration_sha = plan.get("calibration_reuse", {}).get("calibration_sha256")
        expected_rule = plan.get("support_lineage", {}).get("support_rule")
        for record in ledger.get("records", []):
            if record.get("support_calibration_sha256") != expected_calibration_sha or record.get("support_rule") != expected_rule or record.get("no_future_or_intervention_leakage") is not True:
                raise CensusV3ContractError("support record calibration/rule/leakage binding drifted")
        if any(record.get("native_fn") is not True for record in ledger["records"]):
            raise CensusV3ContractError("geometry-successor ledger must be the complete native-FN bank")
    geometry_sources = _normalise_geometry_source(
        base,
        plan,
        geometry_source=geometry_source,
        panel_source=panel_source,
        derived_panel_source=derived_panel_source,
        derived_receipt_source=derived_receipt_source,
        h0_source=h0_source,
        allow_missing=test_only,
    )
    if geometry_sources is None and not test_only:
        raise CensusV3ContractError(
            "production materialization requires the authoritative panel, derived panel, derived receipt, and exact H0 ledger"
        )
    geometry_context = _geometry_context(geometry_sources, ledger["records"], support_info=ledger_info) if geometry_sources is not None else None
    old_bank: dict[str, Any] | None = None
    new_bank: dict[str, Any] | None = None
    old_records: list[dict[str, Any]] = []
    old_by_owner: dict[str, dict[str, Any]] = {}
    new_verified_native_fn: dict[str, dict[str, Any]] = {}
    ledger_by_owner = {str(record["gt_owner_id"]): dict(record) for record in ledger["records"]}
    if geometry_context is not None and prior_doc is not None and prior_info is not None:
        old_records = [dict(record) for record in prior_doc["records"] if isinstance(record, Mapping)]
        old_by_owner = {str(record.get("gt_owner_id")): record for record in old_records if record.get("verified_support") is True}
        new_verified_native_fn = {
            owner_id: record
            for owner_id, record in ledger_by_owner.items()
            if record.get("verified_support") is True and record.get("native_fn") is True
        }
        old_bank = _bank_provenance(old_records, prior_info, population="sampled_prior_verified_bank_with_native_tp_controls")
        new_bank = _bank_provenance(list(ledger_by_owner.values()), ledger_info, population="complete_native_fn_bank")
    # Keep the exact base-census order; it is the row/event ordering authority.
    updated_rows = [by_owner.get(str(row.get("gt_owner_id")), row) if row.get("checkpoint") == CHECKPOINT else row for row in rows]
    supersessions: list[dict[str, Any]] = []
    if geometry_context is not None:
        enriched_rows: list[dict[str, Any]] = []
        for row in updated_rows:
            current = dict(row)
            if (
                current.get("checkpoint") == CHECKPOINT
                and current.get("native_fn") is True
                and current.get("support_verified") is True
                and current.get("eligible_except_support") is True
            ):
                try:
                    enriched = _geometry_from_context(current, context=geometry_context)
                except CensusV3ContractError:
                    # A measured native FN with a valid natural boundary and
                    # covered A is expected to be geometry-resolvable.  Keep
                    # explicit non-pair rows untouched so admission semantics
                    # remain owned by the census-v2 predicates.
                    if current.get("eligible_except_support") is True:
                        raise
                else:
                    stored_old = current.get("geometry")
                    if isinstance(stored_old, Mapping):
                        if old_bank is None or new_bank is None:
                            raise CensusV3ContractError("geometry successor lacks sealed prior-bank provenance")
                        owner_id = str(current["gt_owner_id"])
                        # B is a fixed pair operand, never a competitor-bank
                        # member. Inject it only when the old sampled bank did
                        # not contain it, then replay the exact frozen operator.
                        injection_required = owner_id not in old_by_owner
                        fixed_target_record = ledger_by_owner[owner_id] if injection_required else old_by_owner[owner_id]
                        fixed_target_source = ledger_info if injection_required else prior_info
                        fixed_target_population = "complete_native_fn_bank" if injection_required else "sampled_prior_verified_bank_with_native_tp_controls"
                        old_operand_records = old_records if not injection_required else [*old_records, fixed_target_record]
                        old_context = _geometry_context(geometry_sources, old_operand_records, support_info=prior_info)
                        recomputed_old = _geometry_from_context(current, context=old_context)
                        old_replay = _old_replay(
                            current,
                            target_record=fixed_target_record,
                            target_source=fixed_target_source,
                            target_population=fixed_target_population,
                            injection_required=injection_required,
                            old_bank=old_bank,
                            recomputed_old=recomputed_old,
                        )
                        supersession = _geometry_supersession(
                            current,
                            stored_old=stored_old,
                            recomputed_old=recomputed_old,
                            new=enriched,
                            old_bank=old_bank,
                            new_bank=new_bank,
                            old_verified=old_by_owner,
                            new_verified_native_fn=new_verified_native_fn,
                            old_replay=old_replay,
                        )
                        current["geometry_superseded"] = copy.deepcopy(dict(stored_old))
                        current["geometry_supersession"] = supersession
                        supersessions.append({"gt_owner_id": owner_id, "image_id": int(current["image_id"]), **supersession})
                    current["geometry"] = enriched
                    current["image_cell_regions"] = dict(enriched.get("image_cell_regions", {}))
                    current["image_cell_region_receipts"] = dict(enriched.get("image_cell_region_receipts", {}))
                    current["same_class_competitor_owner_id"] = enriched.get("same_class_competitor_owner_id")
                    current["geometry_source"] = {
                        "source_panel_sha256": geometry_context["source_sha256"],
                        "derived_panel_sha256": geometry_context["derived_sha256"],
                        "derived_receipt_sha256": geometry_context["receipt_info"].get("sha256") if geometry_context.get("receipt_info") else None,
                        "h0_ledger_sha256": geometry_context["h0_sha256"],
                        "hash_semantics": "raw_source_bytes",
                    }
                    target_panel_owner = next(
                        (
                            owner
                            for owner in geometry_context["panel_owners"].get(int(current["image_id"]), ())
                            if owner.gt_owner_id == str(current["gt_owner_id"])
                        ),
                        None,
                    )
                    if target_panel_owner is None or target_panel_owner.derived_index is None:
                        raise CensusV3ContractError(f"geometry derived-panel owner index is missing for {current['gt_owner_id']}")
                    current["derived_panel_object_index"] = int(target_panel_owner.derived_index)
                    if current.get("support_verified") is True and current.get("eligible_except_support") is True:
                        current["disposition"] = "eligible_verified_pair" if enriched.get("launch_eligible") is True else "dynamic_only"
            enriched_rows.append(current)
        updated_rows = enriched_rows
    # Keep dynamic-only support explicit even when the caller did not provide
    # a geometry source (legacy/test-only materialization).  This disposition
    # is strictly downstream of verified support and never changes admission
    # predicates or exact-prefix identity.
    updated_rows = [
        {
            **row,
            "disposition": (
                "dynamic_only"
                if row.get("checkpoint") == CHECKPOINT
                and row.get("native_fn") is True
                and row.get("natural_boundary_valid") is True
                and bool(row.get("covered_owner_ids"))
                and row.get("support_verified") is True
                and row.get("eligible_except_support") is True
                and isinstance(row.get("geometry"), Mapping)
                and row.get("geometry", {}).get("launch_eligible") is not True
                else row.get("disposition")
            ),
        }
        for row in updated_rows
    ]
    records_bytes = b"".join(canonical_json_bytes(row) + b"\n" for row in updated_rows)
    base_candidates = list(base.get("support_completion_candidates", []))
    rule_id = "native_fn_verified_support_static_eligible_v3"
    calibration_for_thresholds = ledger.get("calibration", {})
    thresholds: dict[str, Any] = {
        "epsilon": float(calibration_for_thresholds.get("epsilon", 0.002)),
        "theta_peak_lift": float(calibration_for_thresholds.get("theta_peak_lift", 2.820195781841092)),
        "theta_local_concentration": float(calibration_for_thresholds.get("theta_local_concentration", 3.6645514845848086)),
        "minimum_event_count": 3,
        "minimum_image_count": 2,
    }
    admitted_rows = [row for row in updated_rows if row.get("checkpoint") == CHECKPOINT and row.get("disposition") == "eligible_verified_pair"]
    events = [_event(row, index, rule_id=rule_id, thresholds=thresholds) for index, row in enumerate(admitted_rows)]
    for event, row in zip(events, admitted_rows, strict=True):
        geometry = row.get("geometry")
        if isinstance(geometry, Mapping):
            event["geometry"] = dict(geometry)
            event["geometry_source"] = dict(row.get("geometry_source", {}))
            event["same_class_competitor_owner_id"] = geometry.get("same_class_competitor_owner_id")
            event["image_cell_regions"] = dict(geometry.get("image_cell_regions", {}))
            event["image_cell_region_receipts"] = dict(geometry.get("image_cell_region_receipts", {}))
            event["geometry_sha256"] = geometry.get("geometry_sha256")
            if isinstance(row.get("geometry_supersession"), Mapping):
                event["geometry_supersession"] = copy.deepcopy(dict(row["geometry_supersession"]))
            event["event_sha256"] = sha256_json({key: value for key, value in event.items() if key != "event_sha256"})
    image_ids = sorted({int(row.get("image_id")) for row in ledger["records"]})
    dynamic_only_rows = [
        row for row in updated_rows
        if row.get("checkpoint") == CHECKPOINT and row.get("disposition") == "dynamic_only"
    ]
    dynamic_only_ids = [str(row["gt_owner_id"]) for row in dynamic_only_rows]
    dynamic_only_images = sorted({int(row["image_id"]) for row in dynamic_only_rows})
    admitted_image_ids = sorted({int(row["image_id"]) for row in admitted_rows})
    evaluated_event_ids = [item["gt_owner_id"] for item in supersessions]
    bank_changed_event_ids = [
        item["gt_owner_id"]
        for item in supersessions
        if item["added_effective_region_owner_ids"] or item["removed_effective_region_owner_ids"]
    ]
    core_region_changed_event_ids = [
        item["gt_owner_id"]
        for item in supersessions
        if any(
            delta["added_cells"] or delta["removed_cells"]
            for delta in item["regions"].values()
        )
    ]
    competitor_changed_event_ids = [
        item["gt_owner_id"]
        for item in supersessions
        if item["competitor"]["before_owner_id"] != item["competitor"]["after_owner_id"]
    ]
    supersession_summary = {
        "status": "superseded" if supersessions else "not_applicable",
        "decision": "k13_competitor_population=complete_native_fn_bank",
        "old_population": old_bank,
        "new_population": new_bank,
        "evaluated_event_ids": evaluated_event_ids,
        "evaluated_event_count": len(evaluated_event_ids),
        "evaluated_event_image_ids": sorted({item["image_id"] for item in supersessions}),
        "effective_owner_bank_changed_event_ids": bank_changed_event_ids,
        "effective_owner_bank_changed_event_count": len(bank_changed_event_ids),
        "core_region_changed_event_ids": core_region_changed_event_ids,
        "core_region_changed_event_count": len(core_region_changed_event_ids),
        "competitor_changed_event_ids": competitor_changed_event_ids,
        "competitor_changed_event_count": len(competitor_changed_event_ids),
        "per_event_operand_adequacy": {
            item["gt_owner_id"]: item["operand_adequacy"] for item in supersessions
        },
    }
    legacy_cohort: dict[str, Any] | None = None
    legacy_cohort_manifest: dict[str, Any] | None = None
    legacy_cohort_raw_sha: str | None = None
    legacy_cohort_manifest_raw_sha: str | None = None
    if geometry_context is not None:
        legacy_cohort, legacy_cohort_manifest = _legacy_context_cohort(
            updated_rows,
            geometry_context=geometry_context,
            support_info=ledger_info,
            cohort_path=cohort_output,
        )
        legacy_cohort_raw_sha = sha256_bytes(canonical_json_bytes(legacy_cohort) + b"\n")
        legacy_cohort_manifest_raw_sha = sha256_bytes(canonical_json_bytes(legacy_cohort_manifest) + b"\n")
    panel_source = base.get("source_identity", {}).get("derived_panel", {})
    cohort_identity = {"id": "natural_boundary_s_full_13_image", "frozen_arms": list(FROZEN_ARMS)}
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "sealed",
        "unit_id": UNIT_ID,
        "primary": dict(PRIMARY),
        "source_census": {"revision": "census-v3", "path": None, "sha256": None},
        "arm_order": list(FROZEN_ARMS),
        "panel": {"id": "human-refined-13.geo_sorted_xy", "path": panel_source.get("path"), "sha256": panel_source.get("sha256")},
        "geometry_source": {
            "status": "bound" if geometry_context is not None else "not_provided",
            "source_panel_sha256": geometry_context.get("source_sha256") if geometry_context else None,
            "derived_panel_sha256": geometry_context.get("derived_sha256") if geometry_context else None,
            "derived_receipt_sha256": geometry_context.get("receipt_info", {}).get("sha256") if geometry_context else None,
            "h0_ledger_sha256": geometry_context.get("h0_sha256") if geometry_context else None,
            "hash_semantics": "raw_source_bytes" if geometry_context else None,
        },
        "legacy_context_cohort": {
            "status": "bound" if legacy_cohort is not None else "not_provided",
            "path": str(Path(cohort_output).expanduser().resolve()) if cohort_output else None,
            "sha256": legacy_cohort_raw_sha,
            "manifest_path": str(Path(cohort_manifest_output).expanduser().resolve()) if cohort_manifest_output else None,
            "manifest_sha256": legacy_cohort_manifest_raw_sha,
            "event_count": len(legacy_cohort.get("events", [])) if legacy_cohort is not None else 0,
        },
        "cohort": {**cohort_identity, "sha256": sha256_json(cohort_identity)},
        "operator": {"id": "natural_boundary_support_completion_v3", "sha256": sha256_json({"id": "natural_boundary_support_completion_v3"})},
        "backend": {"id": "hf_fp32_sdpa_greedy", "sha256": sha256_json({"id": "hf_fp32_sdpa_greedy"})},
        "eligibility": {"rule_id": rule_id, "thresholds": thresholds},
        "admission_gate": {"minimum_event_count": 3, "minimum_image_count": 2, "status": "deferred_to_successor_runner"},
        "dynamic_only": {
            "status": "explicit_not_admitted_static_geometry",
            "owner_ids": dynamic_only_ids,
            "image_ids": dynamic_only_images,
            "count": len(dynamic_only_rows),
            "semantics": "native_fn_exact_natural_boundary_covered_A_verified_support_but_static_geometry_not_launch_eligible",
        },
        "estimand_change": supersession_summary,
        "geometry_supersession": supersession_summary,
        "events": events,
        "event_count": len(events),
        # Admission image count is the unique-image denominator of emitted
        # events.  Keep the full ledger coverage separately so a 13-image
        # support ledger cannot masquerade as a multi-image admitted cohort.
        "image_count": len(admitted_image_ids),
        "ledger_image_count": len(image_ids),
    }
    manifest["self_sha256"] = document_hash(manifest)
    census: dict[str, Any] = dict(base)
    census.update({
        "schema_version": SCHEMA_VERSION,
        "status": "sealed",
        "census_revision": "census-v3",
        "rows": updated_rows,
        "summary": _summary(updated_rows),
        "support_completion": {
            "status": "completed",
            "ledger_sha256": ledger_info["sha256"],
            "ledger_content_sha256": ledger.get("content_sha256"),
            "measured_S_count": len(measured),
            "verified_S_count": sum(row.get("support_verified") is True for row in measured),
            "dynamic_only_S_count": len(dynamic_only_rows),
            "dynamic_only_S_owner_ids": dynamic_only_ids,
            "dynamic_only_S_image_ids": dynamic_only_images,
            "original_candidates_count": len(base_candidates),
            "original_candidates_sha256": sha256_json(base_candidates),
            "estimand_change": supersession_summary,
            "geometry_supersession": supersession_summary,
        },
        "admitted_event_manifest": {"path": str(Path(manifest_output).expanduser().resolve()) if manifest_output else None},
    })
    census["support_completion_candidates"] = []
    census["support_completion_candidates_sha256"] = sha256_json([])
    census.pop("self_sha256", None)
    census["records_sha256"] = sha256_bytes(records_bytes)
    census["self_sha256"] = document_hash(census)
    census_bytes = canonical_json_bytes(census) + b"\n"
    manifest["source_census"] = {
        "revision": "census-v3",
        "path": str(Path(output).expanduser().resolve()) if output else None,
        "sha256": sha256_bytes(census_bytes),
        "hash_semantics": "canonical_json_document_with_trailing_newline",
    }
    manifest["self_sha256"] = document_hash(manifest)
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "sealed",
        "unit_id": UNIT_ID,
        "census_revision": "census-v3",
        "base_census": {"path": base_info.get("path"), "sha256": base_info["sha256"]},
        "support_ledger": {"path": ledger_info.get("path"), "sha256": ledger_info["sha256"]},
        "plan": {"path": plan_info.get("path"), "sha256": plan_info.get("sha256")} if plan_info else None,
        "row_count": len(updated_rows),
        "measured_S_count": len(measured),
        "verified_S_count": sum(row.get("support_verified") is True for row in measured),
        "dynamic_only_S_count": len(dynamic_only_rows),
        "dynamic_only_S_owner_ids": dynamic_only_ids,
        "dynamic_only_S_image_ids": dynamic_only_images,
        "geometry_source": manifest["geometry_source"],
        "legacy_context_cohort": manifest["legacy_context_cohort"],
        "estimand_change": supersession_summary,
        "geometry_supersession": supersession_summary,
        "geometry_supersession_events": supersessions,
        "geometry_supersession_event_count": len(supersessions),
        "geometry_supersession_events_sha256": sha256_json(supersessions),
        "event_count": len(events),
        "image_count": len(admitted_image_ids),
        "ledger_image_count": len(image_ids),
        "records_sha256": census["records_sha256"],
        "census_self_sha256": census["self_sha256"],
        "manifest_self_sha256": manifest["self_sha256"],
    }
    receipt["self_sha256"] = document_hash(receipt)
    result = {"census": census, "receipt": receipt, "manifest": manifest, "records_bytes": records_bytes}
    if output is not None or records_output is not None or receipt_output is not None or manifest_output is not None or cohort_output is not None or cohort_manifest_output is not None:
        core_outputs = (output, records_output, receipt_output, manifest_output)
        if any(path is not None for path in core_outputs) and not all(path is not None for path in core_outputs):
            raise CensusV3ContractError("all four output paths are required together")
        if (cohort_output is None) != (cohort_manifest_output is None):
            raise CensusV3ContractError("cohort and cohort-manifest output paths are required together")
        if cohort_output is not None and cohort_manifest_output is not None:
            expected_cohort_manifest = Path(cohort_output).expanduser().resolve().with_name(
                Path(cohort_output).name.replace(".json", ".manifest.json")
            )
            if Path(cohort_manifest_output).expanduser().resolve() != expected_cohort_manifest:
                raise CensusV3ContractError("cohort manifest must be the adjacent .manifest.json path")
        if all(path is not None for path in core_outputs):
            _write_once(output, canonical_json_bytes(census) + b"\n")
            _write_once(records_output, records_bytes)
            _write_once(receipt_output, canonical_json_bytes(receipt) + b"\n")
            _write_once(manifest_output, canonical_json_bytes(manifest) + b"\n")
        if cohort_output is not None and cohort_manifest_output is not None and legacy_cohort is not None and legacy_cohort_manifest is not None:
            _write_once(cohort_output, canonical_json_bytes(legacy_cohort) + b"\n")
            _write_once(cohort_manifest_output, canonical_json_bytes(legacy_cohort_manifest) + b"\n")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-census", type=Path, required=True)
    parser.add_argument("--support-ledger", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--geometry-source", type=Path, default=None, help="sealed geometry-source envelope (optional when lineage is embedded)")
    parser.add_argument("--panel", dest="panel_source", type=Path, default=None, help="authoritative source panel (raw SHA-bound)")
    parser.add_argument("--derived-panel", dest="derived_panel_source", type=Path, default=None, help="authoritative geo_sorted_xy panel (raw SHA-bound)")
    parser.add_argument("--derived-receipt", dest="derived_receipt_source", type=Path, default=None, help="derived-panel mapping receipt (raw SHA-bound)")
    parser.add_argument("--h0-ledger", dest="h0_source", type=Path, default=None, help="exact checkpoint-native H0 ledger (raw SHA-bound)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records-output", type=Path, required=True)
    parser.add_argument("--receipt", dest="receipt_output", type=Path, required=True)
    parser.add_argument("--manifest", dest="manifest_output", type=Path, required=True)
    parser.add_argument("--cohort", dest="cohort_output", type=Path, required=True, help="write legacy-compatible full S context cohort")
    parser.add_argument("--cohort-manifest", dest="cohort_manifest_output", type=Path, required=True, help="write companion context-cohort manifest")
    args = parser.parse_args(argv)
    try:
        geometry_source = None
        if args.geometry_source is not None:
            geometry_source, _ = _read_json(args.geometry_source)
            if not isinstance(geometry_source, Mapping):
                raise CensusV3ContractError("--geometry-source must contain a JSON object")
        result = materialize(
            args.base_census,
            args.support_ledger,
            plan_source=args.plan,
            geometry_source=geometry_source,
            panel_source=args.panel_source,
            derived_panel_source=args.derived_panel_source,
            derived_receipt_source=args.derived_receipt_source,
            h0_source=args.h0_source,
            output=args.output,
            records_output=args.records_output,
            receipt_output=args.receipt_output,
            manifest_output=args.manifest_output,
            cohort_output=args.cohort_output,
            cohort_manifest_output=args.cohort_manifest_output,
        )
    except (CensusV3ContractError, OSError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result["receipt"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
