#!/usr/bin/env python3
"""Re-establish checkpoint-native physical-owner support for the crossover unit.

The probe is intentionally smaller than the P1--P4 intervention runners.  It
only binds a completed native H0 prefix, scores the frozen 17-role owner-local
candidate bank, and calibrates the prior local-peak conjunction from the same
checkpoint's native true-positive controls.  It never mutates model weights,
uses an intervention prefix, or treats a teacher-forced score as behavioural
transfer.

Three modes are exposed:

``contract``
    Validate panel/derived-panel/H0/config identity and emit a deterministic
    readiness receipt.  No model is loaded.
``dry-run``
    The same validation plus deterministic bank/event planning.  No model is
    loaded and no support ledger is emitted.
``live``
    Load the configured HF checkpoint through the shared backend, reforward
    exact native prefixes, score complete candidate rows, calibrate q10 from
    all available native TP controls, and emit a materializer-compatible
    ``static_dynamic_native_h0_owner_ledger.v1`` support envelope.

The implementation deliberately keeps coordinate localization, category
routing, and boundary-gate quantities separate.  Only the physical-owner
local-peak fields can set ``verified_support``; route/gate/readout fields are
diagnostics.  A missing/foreign/stale identity fails closed.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research import build_sorted_owner_accessibility_census_plan as legacy  # noqa: E402
from scripts.research import materialize_static_dynamic_owner_interface_cohort as cohort  # noqa: E402


UNIT_ID = "2026-08-05-static-dynamic-owner-interface-crossover"
SCHEMA_VERSION = "static_dynamic_owner_support_probe.v1"
LEDGER_SCHEMA_VERSION = cohort.LEDGER_SCHEMA_VERSION
CHECKPOINTS = ("S", "A")
EXPECTED_WRAPPERS = {"S": "object_box_closed", "A": "object_box_commit"}
EXPECTED_PARSERS = {"S": "compact_object_box_closed_only", "A": "compact_object_box_commit_only"}
EXPECTED_GENERATION = {
    "max_new_tokens": 3084,
    "repetition_penalty": 1.0,
    "temperature": 0.0,
    "top_p": 1.0,
}
SUPPORT_QUANTILE = float(legacy.SUPPORT_PRIMARY_QUANTILE)
SUPPORT_EPSILON = float(legacy.SUPPORT_EPSILON)
IMAGE_2299 = 2299
IM_END = int(getattr(legacy, "IM_END", 151645))


class SupportProbeError(ValueError):
    """Raised when a support probe identity or semantic contract is invalid."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve(strict=True).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: str | Path) -> Any:
    resolved = Path(path).expanduser().resolve(strict=True)
    try:
        return json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SupportProbeError(f"invalid JSON input: {resolved}") from exc


def _read_source(source: str | Path | Mapping[str, Any] | Sequence[Any]) -> tuple[Any, dict[str, Any]]:
    """Read JSON/JSONL using the same source hashing convention as materializer."""

    if isinstance(source, (str, Path)):
        path = Path(source).expanduser().resolve(strict=True)
        raw = path.read_bytes()
        if path.suffix.lower() == ".jsonl":
            rows: list[Any] = []
            for line_no, line in enumerate(raw.splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SupportProbeError(f"{path}:{line_no}: invalid JSONL") from exc
                if not isinstance(row, Mapping):
                    raise SupportProbeError(f"{path}:{line_no}: JSONL row is not an object")
                rows.append(dict(row))
            return rows, {"path": str(path), "sha256": sha256_bytes(raw), "row_count": len(rows)}
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SupportProbeError(f"{path}: invalid JSON") from exc
        return value, {
            "path": str(path),
            "sha256": sha256_bytes(raw),
            "row_count": len(value) if isinstance(value, list) else None,
        }
    if isinstance(source, Mapping):
        value = dict(source)
    elif isinstance(source, Sequence) and not isinstance(source, (str, bytes, bytearray)):
        value = list(source)
    else:
        raise SupportProbeError("source must be a path, mapping, or sequence")
    return value, {"inline": True, "sha256": sha256_json(value)}


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value.lower()):
        raise SupportProbeError(f"{label} must be a lowercase SHA-256")
    return value.lower()


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SupportProbeError(f"{label} is not numeric") from exc
    if not math.isfinite(result):
        raise SupportProbeError(f"{label} is not finite")
    return result


def _quantile(values: Sequence[float], level: float) -> float:
    if not values:
        raise SupportProbeError("cannot calibrate support without native TP controls")
    ordered = sorted(float(value) for value in values)
    position = float(level) * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _logsumexp(values: Sequence[float]) -> float:
    if not values:
        raise SupportProbeError("support population is empty")
    peak = max(float(value) for value in values)
    return peak + math.log(math.fsum(math.exp(float(value) - peak) for value in values))


@dataclass(frozen=True)
class PanelInputs:
    source: Any
    derived: Any
    receipt: Mapping[str, Any]
    source_info: Mapping[str, Any]
    derived_info: Mapping[str, Any]
    receipt_info: Mapping[str, Any]
    owners_by_image: Mapping[int, tuple[cohort.PanelOwner, ...]]
    rows_by_image: Mapping[int, Mapping[str, Any]]


@dataclass(frozen=True)
class H0Inputs:
    envelope: Mapping[str, Any]
    records: tuple[Mapping[str, Any], ...]
    source_info: Mapping[str, Any]


@dataclass(frozen=True)
class CheckpointContract:
    checkpoint: str
    config_path: Path
    config_sha256: str
    config_fingerprint: str
    wrapper: str
    parser: str
    source_panel_sha256: str
    derived_panel_sha256: str
    resolved_config: Mapping[str, Any] | None = None
    resolved_config_path: Path | None = None


def load_panel_inputs(
    source_panel: str | Path | Mapping[str, Any] | Sequence[Any],
    derived_panel: str | Path | Mapping[str, Any] | Sequence[Any],
    derived_receipt: str | Path | Mapping[str, Any],
) -> PanelInputs:
    source, source_info = _read_source(source_panel)
    derived, derived_info = _read_source(derived_panel)
    receipt, receipt_info = _read_source(derived_receipt)
    if not isinstance(receipt, Mapping):
        raise SupportProbeError("derived receipt must be a JSON object")
    try:
        owners = cohort._validate_derived_panel(  # noqa: SLF001
            source,
            derived,
            receipt,
            source_sha256=str(source_info["sha256"]),
            derived_sha256=str(derived_info["sha256"]),
        )
    except cohort.CohortContractError as exc:
        raise SupportProbeError(str(exc)) from exc
    rows = derived if isinstance(derived, list) else [derived]
    rows_by_image: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise SupportProbeError("derived panel row is not an object")
        try:
            image = int(row["image_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SupportProbeError("derived panel row has no integer image_id") from exc
        rows_by_image[image] = row
    return PanelInputs(
        source=source,
        derived=derived,
        receipt=receipt,
        source_info=source_info,
        derived_info=derived_info,
        receipt_info=receipt_info,
        owners_by_image={image: tuple(items) for image, items in owners.items()},
        rows_by_image=rows_by_image,
    )


def load_h0_inputs(
    source: str | Path | Mapping[str, Any],
    *,
    panel: PanelInputs,
    checkpoint: str | None = None,
) -> H0Inputs:
    value, source_info = _read_source(source)
    if not isinstance(value, Mapping):
        raise SupportProbeError("H0 ledger must be an envelope object")
    try:
        normalized_checkpoint = cohort._norm_checkpoint(value.get("checkpoint"))  # noqa: SLF001
    except Exception as exc:  # pragma: no cover - defensive around private helper
        raise SupportProbeError("H0 checkpoint is invalid") from exc
    if normalized_checkpoint not in CHECKPOINTS:
        raise SupportProbeError("H0 ledger checkpoint must be S or A")
    if checkpoint is not None and normalized_checkpoint != str(checkpoint):
        raise SupportProbeError(
            f"H0 checkpoint mismatch: {normalized_checkpoint} != {checkpoint}"
        )
    # Keep H0 semantically outcome-only.  In particular, a native TP is not a
    # measured support positive; the support field is intentionally absent.
    if value.get("schema_version") != LEDGER_SCHEMA_VERSION:
        raise SupportProbeError(f"H0 ledger schema_version must be {LEDGER_SCHEMA_VERSION}")
    if value.get("unit_id") != UNIT_ID:
        raise SupportProbeError("H0 ledger unit_id mismatch")
    if value.get("run_kind") != "native_h0":
        raise SupportProbeError("H0 ledger run_kind must be native_h0")
    if not isinstance(value.get("config_fingerprint"), str) or not value.get("config_fingerprint"):
        raise SupportProbeError("H0 ledger has no config_fingerprint")
    if value.get("arm") not in (None, "native", "H0", "h0"):
        raise SupportProbeError("H0 ledger is not a native arm")
    if value.get("native_outcome_only") is not True:
        raise SupportProbeError("H0 envelope must attest native_outcome_only=true")
    if value.get("verified_support_claim") is not False:
        raise SupportProbeError("H0 envelope must attest verified_support_claim=false")
    if value.get("source_panel_sha256") != str(panel.source_info["sha256"]):
        raise SupportProbeError("H0 ledger source panel hash mismatch")
    if value.get("derived_panel_sha256") != str(panel.derived_info["sha256"]):
        raise SupportProbeError("H0 ledger derived panel hash mismatch")
    if value.get("history_complete") is not True:
        raise SupportProbeError("H0 ledger must attest history_complete=true")
    payload = value.get("records")
    if not isinstance(payload, list) or not payload:
        raise SupportProbeError("H0 ledger must contain a non-empty records list")
    records = []
    for row in payload:
        if not isinstance(row, Mapping):
            raise SupportProbeError("H0 record is not an object")
        records.append(dict(row))
    normalized: list[Mapping[str, Any]] = []
    seen_owner_ids: set[str] = set()
    for record in records:
        item = dict(record)
        if item.get("unit_id") != UNIT_ID:
            raise SupportProbeError("H0 record unit_id mismatch")
        if item.get("config_fingerprint") != value.get("config_fingerprint"):
            raise SupportProbeError("H0 record config_fingerprint mismatch")
        if item.get("source_panel_sha256") != str(panel.source_info["sha256"]):
            raise SupportProbeError("H0 record source panel hash mismatch")
        if item.get("derived_panel_sha256") != str(panel.derived_info["sha256"]):
            raise SupportProbeError("H0 record derived panel hash mismatch")
        if item.get("run_kind") != "native_h0" or item.get("history_complete") is not True:
            raise SupportProbeError("H0 record lacks complete native H0 provenance")
        if item.get("support_status") != "not_measured":
            raise SupportProbeError("H0 record support_status must be not_measured")
        if item.get("verified_support_claim") is not False:
            raise SupportProbeError("H0 record verified_support_claim must be false")
        if "verified_support" in item:
            raise SupportProbeError("H0 record must omit verified_support")
        for alias in (
            "support_verified",
            "tested_localization_support",
            "localization_support",
            "positive_support",
            "has_tested_support",
        ):
            if alias in item:
                raise SupportProbeError(f"H0 record must omit support alias {alias}")
        try:
            image = int(item["image_id"])
            owner_id = str(item["gt_owner_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SupportProbeError("H0 record has no physical owner identity") from exc
        owner = next(
            (candidate for candidate in panel.owners_by_image.get(image, ()) if candidate.gt_owner_id == owner_id),
            None,
        )
        if owner is None:
            raise SupportProbeError("H0 record owner identity is not unique")
        if owner_id in seen_owner_ids:
            raise SupportProbeError(f"H0 contains duplicate physical owner record {owner_id}")
        seen_owner_ids.add(owner_id)
        declared_owner_id = item.get("gt_owner_id", item.get("owner_id"))
        if declared_owner_id is not None and str(declared_owner_id) != str(owner.gt_owner_id):
            raise SupportProbeError("H0 physical-owner mapping mismatch")
        declared_index = item.get("source_panel_object_index")
        if declared_index is not None and int(declared_index) != int(owner.source_index):
            raise SupportProbeError("H0 source-panel owner index mismatch")
        declared_category = item.get("category_name", item.get("category"))
        if declared_category is not None and str(declared_category).strip().lower() != str(owner.category or "").lower():
            raise SupportProbeError("H0 owner category mapping mismatch")
        declared_bbox = item.get("bbox_pixel_xyxy", item.get("pixel_bbox"))
        if declared_bbox is not None:
            try:
                if tuple(int(round(float(value))) for value in declared_bbox) != tuple(owner.bbox or ()):
                    raise SupportProbeError("H0 owner geometry mapping mismatch")
            except (TypeError, ValueError) as exc:
                raise SupportProbeError("H0 owner geometry mapping is malformed") from exc
        if item.get("checkpoint") != normalized_checkpoint:
            raise SupportProbeError("H0 record checkpoint mismatch")
        if item.get("intervention") not in (None, False, "none", "native"):
            raise SupportProbeError("H0 record is intervention-derived")
        if item.get("native_tp") not in (True, False) or item.get("native_fn") not in (True, False):
            raise SupportProbeError("H0 record must declare native_tp and native_fn")
        if item["native_tp"] == item["native_fn"]:
            raise SupportProbeError("H0 record has contradictory native TP/FN flags")
        prefix_semantics = item.get("prefix_semantics")
        root_prefix = (
            item["native_tp"]
            and int(item.get("natural_boundary", -1)) == 0
            and item.get("exact_prefix_token_ids") == []
        )
        if root_prefix and item.get("due_boundary_index") != 0:
            raise SupportProbeError("H0 TP root due_boundary_index must be zero")
        expected_prefix_semantics = (
            "before_queried_owner_row"
            if item["native_tp"]
            else "after_strict_covered_row_pre_stop"
        )
        if item["native_tp"] and root_prefix:
            if prefix_semantics != "before_queried_owner_row":
                raise SupportProbeError("H0 TP root prefix_semantics is invalid")
        elif prefix_semantics != expected_prefix_semantics:
            raise SupportProbeError(
                f"H0 {owner_id} prefix_semantics must be {expected_prefix_semantics}"
            )
        if item.get("excludes_stop") is not True:
            raise SupportProbeError("H0 exact prefix must attest excludes_stop=true")
        covered_owner_ids = item.get("covered_owner_ids")
        if not isinstance(covered_owner_ids, list) or any(
            not isinstance(owner, str) or not owner for owner in covered_owner_ids
        ):
            raise SupportProbeError("H0 covered_owner_ids must be a list of owner IDs")
        if item.get("queried_owner_not_covered") is not True:
            raise SupportProbeError("H0 queried owner must be absent from covered_owner_ids")
        if owner_id in covered_owner_ids:
            raise SupportProbeError("H0 covered_owner_ids contains queried owner")
        known_owner_ids = {candidate.gt_owner_id for candidate in panel.owners_by_image.get(image, ())}
        if any(owner not in known_owner_ids for owner in covered_owner_ids):
            raise SupportProbeError("H0 covered_owner_ids contains an owner outside the image panel")
        latest_covered = item.get("latest_covered_owner_id")
        if covered_owner_ids:
            if not isinstance(latest_covered, str) or latest_covered != covered_owner_ids[-1]:
                raise SupportProbeError("H0 latest_covered_owner_id is not the latest covered owner")
        elif latest_covered is not None:
            raise SupportProbeError("H0 latest_covered_owner_id is set without covered owners")
        evidence = item.get("due_boundary_evidence")
        if not isinstance(evidence, Mapping):
            raise SupportProbeError("H0 due_boundary_evidence is missing")
        if evidence.get("queried_owner_id") not in (None, owner_id):
            raise SupportProbeError("H0 due-boundary evidence queried owner mismatch")
        if evidence.get("covered_owner_ids") not in (None, covered_owner_ids):
            raise SupportProbeError("H0 due-boundary evidence covered-owner mismatch")
        if evidence.get("latest_covered_owner_id") not in (None, latest_covered):
            raise SupportProbeError("H0 due-boundary evidence latest owner mismatch")
        if evidence.get("covered_row_count") not in (None, len(covered_owner_ids)):
            raise SupportProbeError("H0 due-boundary evidence covered row count mismatch")
        if evidence.get("boundary_disposition") not in (None, item.get("boundary_disposition")):
            raise SupportProbeError("H0 due-boundary evidence disposition mismatch")
        if evidence.get("queried_owner_not_covered") not in (None, True):
            raise SupportProbeError("H0 due-boundary evidence queried owner is covered")
        boundary_valid = item.get("natural_boundary_valid") is True
        prefix_ids = item.get("exact_prefix_token_ids")
        if boundary_valid:
            if (
                not isinstance(prefix_ids, list)
                or any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in prefix_ids)
            ):
                raise SupportProbeError("H0 exact_prefix_token_ids are malformed")
            if sha256_json(prefix_ids) != item.get("exact_prefix_sha256"):
                raise SupportProbeError("H0 exact prefix hash does not match token IDs")
            if IM_END in prefix_ids:
                raise SupportProbeError("H0 exact prefix extends through im_end/STOP")
        elif prefix_ids is not None or item.get("exact_prefix_sha256") is not None:
            raise SupportProbeError("invalid-boundary H0 record must omit exact prefix fields")
        if boundary_valid:
            if not isinstance(item.get("natural_boundary"), (int, float)) or isinstance(item.get("natural_boundary"), bool):
                raise SupportProbeError("H0 record has no numeric natural boundary")
            if float(item["natural_boundary"]) != int(item["natural_boundary"]):
                raise SupportProbeError("H0 natural boundary must be an integer")
            if item["native_tp"] and int(item["natural_boundary"]) != len(covered_owner_ids):
                raise SupportProbeError("H0 TP natural boundary does not match covered owner count")
            if item["native_fn"] and int(item["natural_boundary"]) < len(covered_owner_ids):
                raise SupportProbeError("H0 FN natural boundary precedes covered owner count")
        else:
            if not item["native_fn"] or item.get("natural_boundary") is not None:
                raise SupportProbeError("invalid H0 boundary must be an uncovered native FN")
            if covered_owner_ids or latest_covered is not None:
                raise SupportProbeError("invalid H0 boundary must have no covered owners")
        start_step = item.get("generated_history_start_step")
        end_step = item.get("generated_history_end_step")
        if start_step not in (None, 0) or (start_step is None and prefix_ids):
            raise SupportProbeError("H0 generated_history_start_step is inconsistent")
        if prefix_ids:
            if not isinstance(end_step, int) or isinstance(end_step, bool) or end_step < 0:
                raise SupportProbeError("H0 generated_history_end_step is missing")
        else:
            if boundary_valid and (not item["native_tp"] or int(item["natural_boundary"]) != 0):
                raise SupportProbeError("only a TP root due-boundary may have an empty prefix")
            if boundary_valid and prefix_semantics != "before_queried_owner_row":
                raise SupportProbeError("H0 TP root prefix_semantics is invalid")
            if boundary_valid and start_step is not None:
                raise SupportProbeError("H0 TP root must have no generated_history_start_step")
            if end_step is not None:
                raise SupportProbeError("H0 root prefix must have no generated_history_end_step")
        if item.get("natural_boundary_valid") not in (True, False):
            raise SupportProbeError("H0 natural_boundary_valid must be a JSON boolean")
        due_index = item.get("due_boundary_index")
        if due_index is not None and boundary_valid and due_index != item.get("natural_boundary"):
            raise SupportProbeError("H0 due_boundary_index mismatches natural_boundary")
        if due_index is not None and not boundary_valid:
            raise SupportProbeError("invalid H0 boundary must have no due_boundary_index")
        if item["native_tp"] and item.get("boundary_disposition") not in (None, "native_tp_before_queried_row"):
            raise SupportProbeError("H0 TP boundary disposition is invalid")
        if item["native_fn"]:
            expected_dispositions = (
                {"native_fn_after_first_covered_row", "after_strict_covered_row_pre_stop"}
                if boundary_valid
                else {"no_valid_post_covered_boundary"}
            )
            if item.get("boundary_disposition") not in expected_dispositions:
                raise SupportProbeError("H0 FN boundary disposition is invalid")
            earliest = item.get("is_earliest_eligible_boundary")
            if earliest is not boundary_valid:
                raise SupportProbeError(
                    "H0 FN is_earliest_eligible_boundary contradicts boundary validity"
                )
        elif item.get("is_earliest_eligible_boundary") not in (None, False):
            raise SupportProbeError("H0 TP cannot attest an eligible FN boundary")
        stop_step = item.get("generated_history_stop_step")
        if stop_step is not None:
            if not isinstance(stop_step, int) or isinstance(stop_step, bool) or stop_step < 0:
                raise SupportProbeError("H0 generated_history_stop_step is malformed")
            if end_step is not None and stop_step <= end_step:
                raise SupportProbeError("H0 exact prefix reaches or exceeds STOP")
        if item.get("history_complete") is not True:
            raise SupportProbeError("H0 record lacks complete-history attestation")
        item["checkpoint"] = normalized_checkpoint
        item["image_id"] = image
        item["gt_owner_id"] = owner_id
        normalized.append(item)
    if not normalized:
        raise SupportProbeError("H0 ledger contains no owner records")
    return H0Inputs(value, tuple(normalized), source_info)


def _resolved_config_authority(
    h0: H0Inputs,
    explicit_path: str | Path | None,
) -> tuple[Mapping[str, Any] | None, Path | None, str | None]:
    """Load immutable ``configs/resolved.json`` authority when available.

    H0's resolved artifact is the fingerprint owner.  The authored leaf is
    only a semantic cross-check; its current loader fingerprint is never
    promoted into the support run identity.
    """

    candidate: Path | None = None
    if explicit_path is not None:
        candidate = Path(explicit_path).expanduser().resolve(strict=True)
    else:
        info = h0.envelope.get("infer_config")
        resolved = info.get("resolved") if isinstance(info, Mapping) else None
        declared = resolved.get("path") if isinstance(resolved, Mapping) else None
        if isinstance(declared, str) and declared:
            path = Path(declared).expanduser()
            if not path.is_file():
                raise SupportProbeError(
                    f"H0 immutable resolved config authority is missing: {path}"
                )
            candidate = path.resolve()
    if candidate is None:
        return None, None, None
    try:
        value = _read_json(candidate) if candidate.suffix.lower() == ".json" else None
    except SupportProbeError as exc:
        raise SupportProbeError(f"resolved config authority is unreadable: {candidate}") from exc
    if value is not None:
        config = value.get("config") if isinstance(value, Mapping) else None
        resolution = value.get("resolution") if isinstance(value, Mapping) else None
        fingerprint = resolution.get("fingerprint") if isinstance(resolution, Mapping) else None
        if not isinstance(config, Mapping) or not isinstance(fingerprint, str) or not fingerprint:
            raise SupportProbeError("resolved config JSON lacks config/resolution fingerprint")
        if sha256_json(config) != fingerprint:
            raise SupportProbeError("resolved config raw fingerprint does not match resolution fingerprint")
        return dict(config), candidate, fingerprint
    # A wrapped YAML artifact is accepted for provenance, but its ``config``
    # mapping is still parsed without asking the mutable leaf loader to invent
    # a new authority.
    try:
        import yaml  # type: ignore[import-not-found]

        value = yaml.safe_load(candidate.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SupportProbeError(f"resolved config authority is unreadable: {candidate}") from exc
    config = value.get("config") if isinstance(value, Mapping) else None
    if not isinstance(config, Mapping):
        raise SupportProbeError("resolved config YAML lacks config mapping")
    resolution = value.get("resolution") if isinstance(value, Mapping) else None
    fingerprint = resolution.get("fingerprint") if isinstance(resolution, Mapping) else None
    if not isinstance(fingerprint, str) or not fingerprint:
        raise SupportProbeError("resolved config YAML lacks resolution fingerprint")
    if sha256_json(config) != fingerprint:
        raise SupportProbeError("resolved config raw fingerprint does not match resolution fingerprint")
    return dict(config), candidate, fingerprint


def _config_semantics(config: Mapping[str, Any], *, checkpoint: str) -> tuple[str, str, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    model = config.get("model") if isinstance(config.get("model"), Mapping) else {}
    backend = config.get("backend") if isinstance(config.get("backend"), Mapping) else {}
    hf = backend.get("hf") if isinstance(backend.get("hf"), Mapping) else {}
    template = config.get("template") if isinstance(config.get("template"), Mapping) else {}
    generation = config.get("generation") if isinstance(config.get("generation"), Mapping) else {}
    data = config.get("data") if isinstance(config.get("data"), Mapping) else {}
    wrapper = str(template.get("assistant_format", ""))
    if wrapper != EXPECTED_WRAPPERS[checkpoint]:
        raise SupportProbeError(
            f"native wrapper mismatch: {wrapper!r} != {EXPECTED_WRAPPERS[checkpoint]!r}"
        )
    if template.get("object_ordering") != "geo_sorted_xy":
        raise SupportProbeError("config object_ordering must be geo_sorted_xy")
    if str(backend.get("type")) != "hf":
        raise SupportProbeError("support probe requires backend.type=hf")
    if str(model.get("dtype")) != "fp32":
        raise SupportProbeError("support probe requires fp32 model dtype")
    if str(hf.get("attn_implementation")) != "sdpa":
        raise SupportProbeError("support probe requires HF SDPA")
    for key, expected in EXPECTED_GENERATION.items():
        observed = generation.get(key)
        try:
            observed = int(observed) if key == "max_new_tokens" else float(observed)
        except (TypeError, ValueError) as exc:
            raise SupportProbeError(f"generation.{key} is missing") from exc
        if observed != expected:
            raise SupportProbeError(f"generation.{key} mismatch: {observed!r} != {expected!r}")
    return wrapper, str(backend.get("type")), model, template, generation, data


def validate_checkpoint_config(
    config_path: str | Path,
    *,
    checkpoint: str,
    panel: PanelInputs,
    h0: H0Inputs,
    resolved_config_path: str | Path | None = None,
) -> CheckpointContract:
    """Validate the exact native recipe without loading model weights."""

    resolved_path = Path(config_path).expanduser().resolve(strict=True)
    try:
        import yaml  # type: ignore[import-not-found]

        authored = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise SupportProbeError(f"cannot read infer config: {resolved_path}") from exc
    if not isinstance(authored, Mapping):
        raise SupportProbeError("infer config must be a mapping")
    wrapper, _backend_type, _model, _template, _generation, authored_data = _config_semantics(
        authored, checkpoint=checkpoint
    )
    configured_panel = authored_data.get("input_jsonl")
    if isinstance(configured_panel, str):
        configured_panel_path = Path(configured_panel).expanduser().resolve()
        if not configured_panel_path.is_file():
            raise SupportProbeError("config input_jsonl is missing")
        if sha256_file(configured_panel_path) != panel.derived_info["sha256"]:
            raise SupportProbeError("config input_jsonl hash does not match derived panel")
    authority, authority_path, authority_fingerprint = _resolved_config_authority(
        h0, resolved_config_path
    )
    if authority is not None:
        authority_wrapper, _authority_backend, _authority_model, authority_template, _authority_generation, authority_data = _config_semantics(
            authority, checkpoint=checkpoint
        )
        if authority_wrapper != wrapper:
            raise SupportProbeError("authored leaf and resolved config wrapper differ")
        if authority_template.get("object_ordering") != _template.get("object_ordering"):
            raise SupportProbeError("authored leaf and resolved config ordering differ")
        authority_panel = authority_data.get("input_jsonl")
        if isinstance(authority_panel, str):
            authority_panel_path = Path(authority_panel).expanduser().resolve()
            if not authority_panel_path.is_file():
                raise SupportProbeError("resolved config input_jsonl is missing")
            if sha256_file(authority_panel_path) != panel.derived_info["sha256"]:
                raise SupportProbeError("resolved config input_jsonl hash does not match derived panel")
        fingerprint = str(authority_fingerprint)
        runtime_config = authority
    else:
        # Compatibility path for synthetic/unit fixtures.  A real H0 run must
        # carry the immutable resolved artifact; live mode rejects this fallback.
        try:
            from src.config.inference import load_infer_config

            resolved = load_infer_config(resolved_path)
            fingerprint = str(resolved.fingerprint)
            runtime_config = resolved.config.model_dump(mode="json")
        except Exception as exc:
            raise SupportProbeError("current infer-config loader rejected config") from exc
    expected_fingerprint = {str(item.get("config_fingerprint")) for item in h0.records}
    declared_fingerprint = str(h0.envelope.get("config_fingerprint"))
    if len(expected_fingerprint) != 1 or declared_fingerprint not in expected_fingerprint:
        raise SupportProbeError("H0 records have inconsistent config fingerprint")
    if declared_fingerprint != fingerprint:
        raise SupportProbeError(
            "resolved config fingerprint does not match immutable H0 authority"
        )
    return CheckpointContract(
        checkpoint=checkpoint,
        config_path=resolved_path,
        config_sha256=sha256_file(resolved_path),
        config_fingerprint=fingerprint,
        wrapper=wrapper,
        parser=EXPECTED_PARSERS[checkpoint],
        source_panel_sha256=str(panel.source_info["sha256"]),
        derived_panel_sha256=str(panel.derived_info["sha256"]),
        resolved_config=runtime_config,
        resolved_config_path=authority_path,
    )


def _owners_for_bank(panel: PanelInputs) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    owners_by_image: dict[str, list[dict[str, Any]]] = {}
    panel_geometry: dict[str, dict[str, Any]] = {}
    for image_id, owners in sorted(panel.owners_by_image.items()):
        row = panel.rows_by_image[image_id]
        panel_geometry[str(image_id)] = {
            "width": int(row.get("width", row.get("image_width", 0))),
            "height": int(row.get("height", row.get("image_height", 0))),
        }
        if panel_geometry[str(image_id)]["width"] <= 0 or panel_geometry[str(image_id)]["height"] <= 0:
            raise SupportProbeError(f"image {image_id} has invalid panel dimensions")
        owners_by_image[str(image_id)] = [
            {
                "gt_owner_id": owner.gt_owner_id,
                "image_id": str(image_id),
                "normalized_description": str(owner.category or ""),
                "bbox_pixel_xyxy": list(owner.bbox or ()),
            }
            for owner in owners
        ]
    return owners_by_image, panel_geometry


def build_physical_owner_bank(panel: PanelInputs) -> tuple[tuple[dict[str, Any], ...], Mapping[str, Any]]:
    """Build the frozen 17-role local bank using the authoritative planner."""

    owners, geometry = _owners_for_bank(panel)
    try:
        physical, accounting = legacy.build_candidate_bank(owners, geometry)
    except Exception as exc:
        raise SupportProbeError(f"authoritative owner candidate bank failed: {exc}") from exc
    if not physical:
        raise SupportProbeError("authoritative owner candidate bank is empty")
    return tuple(physical), accounting


def _bank_group(physical: Sequence[Mapping[str, Any]], image_id: int, category: str) -> list[Mapping[str, Any]]:
    return [
        candidate
        for candidate in physical
        if str(candidate.get("image_id")) == str(image_id)
        and str(candidate.get("normalized_description")) == str(category).lower()
    ]


def _candidate_local_rows(
    candidates: Sequence[Mapping[str, Any]],
    *,
    owner_id: str,
) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for candidate in candidates:
        try:
            partition = legacy.classify_candidate_for_owner(candidate, owner_id)
        except Exception as exc:
            raise SupportProbeError(f"candidate partition failed for {owner_id}: {exc}") from exc
        if partition.get("counts_toward_upper_bound") is True:
            rows.append(candidate)
    return rows


def _normalize_cuda_device(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise SupportProbeError(f"{label} must be a CUDA device string")
    text = value.strip().lower()
    if text == "cuda":
        return "cuda:0"
    match = re.fullmatch(r"cuda:([0-9]+)", text)
    if match is None:
        raise SupportProbeError(f"{label} is not a supported CUDA device: {value!r}")
    return f"cuda:{int(match.group(1))}"


def _cuda_visible_mapping() -> dict[str, Any]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None or not raw.strip():
        raise SupportProbeError("CUDA_VISIBLE_DEVICES is missing; physical device mapping is ambiguous")
    tokens = [token.strip() for token in raw.split(",")]
    if len(tokens) != 1 or not tokens[0] or tokens[0] == "-1":
        raise SupportProbeError("CUDA_VISIBLE_DEVICES must expose exactly one usable device per shard")
    token = tokens[0]
    return {
        "raw": raw,
        "tokens": tokens,
        "selected_physical_device": token,
    }


def validate_live_runtime_identity(
    session: Any,
    *,
    expected_dtype: str = "fp32",
    expected_attention: str = "sdpa",
    expected_checkpoint: str | None = None,
    expected_config_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Fail closed and return the complete attested HF runtime identity."""

    model = getattr(session, "_model", None)
    if model is None:
        raise SupportProbeError("detached/wrong runtime: HF session has no model")
    try:
        first = next(model.parameters())
    except (AttributeError, StopIteration, TypeError) as exc:
        raise SupportProbeError("detached/wrong runtime: model has no parameters") from exc
    device = str(getattr(first, "device", ""))
    dtype = str(getattr(first, "dtype", ""))
    if expected_dtype == "fp32" and dtype not in {"torch.float32", "float32"}:
        raise SupportProbeError(f"detached/wrong runtime: model dtype {dtype!r}")
    if not device.startswith("cuda"):
        raise SupportProbeError(f"detached/wrong runtime: model device {device!r}")
    receipt = getattr(session, "receipt", None)
    effective = getattr(receipt, "effective_settings", {}) if receipt is not None else {}
    if not isinstance(effective, Mapping):
        raise SupportProbeError("detached/wrong runtime: effective settings are missing")
    effective_device = effective.get("device")
    normalized_device = _normalize_cuda_device(device, "model device")
    normalized_effective_device = _normalize_cuda_device(effective_device, "receipt device")
    if normalized_effective_device != normalized_device:
        raise SupportProbeError(
            f"detached/wrong runtime: model/receipt device mismatch {device!r} != {effective_device!r}"
        )
    try:
        import torch

        current_device = int(torch.cuda.current_device())
    except Exception as exc:  # pragma: no cover - live CUDA boundary
        raise SupportProbeError("detached/wrong runtime: torch current CUDA device is unavailable") from exc
    torch_current_device = f"cuda:{current_device}"
    if torch_current_device != normalized_device:
        raise SupportProbeError(
            f"detached/wrong runtime: torch current device {torch_current_device!r} != model {normalized_device!r}"
        )
    visible_devices = _cuda_visible_mapping()
    observed_attention = effective.get("observed_attn_implementation")
    if not isinstance(observed_attention, str):
        raise SupportProbeError("detached/wrong runtime: attention identity is not a string")
    if observed_attention != expected_attention:
        raise SupportProbeError(
            f"detached/wrong runtime: attention {observed_attention!r} != {expected_attention!r}"
        )
    launch = getattr(session, "_launch", None)
    if launch is None:
        raise SupportProbeError("detached/wrong runtime: session has no backend launch")
    receipt_dict_fn = getattr(receipt, "to_artifact_dict", None)
    if not callable(receipt_dict_fn):
        raise SupportProbeError("detached/wrong runtime: session receipt has no artifact identity")
    try:
        receipt_identity = receipt_dict_fn()
    except Exception as exc:  # pragma: no cover - backend-owned defensive boundary
        raise SupportProbeError("detached/wrong runtime: receipt identity is unavailable") from exc
    if not isinstance(receipt_identity, Mapping):
        raise SupportProbeError("detached/wrong runtime: receipt identity is not an object")
    required_receipt = (
        "backend",
        "backend_mode",
        "response_family",
        "backend_version",
        "model_identity",
        "tokenizer_identity",
        "processor_identity",
        "generation_config_fingerprint",
        "effective_settings",
        "likelihood_semantics",
    )
    if any(key not in receipt_identity for key in required_receipt):
        raise SupportProbeError("detached/wrong runtime: receipt identity is incomplete")
    launch_identity = {
        "backend": getattr(launch, "backend", None),
        "model_path": getattr(launch, "model_path", None),
        "model_dtype": getattr(launch, "model_dtype", None),
        "batch_size": getattr(launch, "batch_size", None),
        "generation_config_fingerprint": getattr(launch, "generation_config_fingerprint", None),
        "backend_options": getattr(launch, "backend_options", None),
        "expected_model_identity": getattr(launch, "expected_model_identity", None),
        "execution_model_identity": getattr(launch, "execution_model_identity", None),
        "adapter": getattr(launch, "adapter", None),
        "embedding_delta": getattr(launch, "embedding_delta", None),
    }
    if not isinstance(launch_identity["backend"], str) or not launch_identity["backend"]:
        raise SupportProbeError("detached/wrong runtime: launch backend identity is missing")
    if not isinstance(launch_identity["model_path"], str) or not launch_identity["model_path"]:
        raise SupportProbeError("detached/wrong runtime: launch model path is missing")
    if not isinstance(launch_identity["backend_options"], Mapping):
        raise SupportProbeError("detached/wrong runtime: launch backend options are missing")
    runtime = {
        "device": device,
        "effective_device": effective_device,
        "normalized_device": normalized_device,
        "torch_current_device": torch_current_device,
        "cuda_visible_devices": visible_devices,
        "physical_device_id": visible_devices["selected_physical_device"],
        "dtype": dtype,
        "attn_implementation": observed_attention,
        "checkpoint": expected_checkpoint,
        "config_fingerprint": expected_config_fingerprint,
        "schema_version": RUNTIME_IDENTITY_SCHEMA_VERSION,
        "status": "validated",
        "passed": True,
        "backend": receipt_identity["backend"],
        "backend_mode": receipt_identity["backend_mode"],
        "response_family": receipt_identity["response_family"],
        "backend_version": receipt_identity["backend_version"],
        "model_path": launch_identity["model_path"],
        "model_dtype": launch_identity["model_dtype"],
        "batch_size": launch_identity["batch_size"],
        "generation_config_fingerprint": receipt_identity["generation_config_fingerprint"],
        "model_identity": dict(receipt_identity["model_identity"]),
        "tokenizer_identity": dict(receipt_identity["tokenizer_identity"]),
        "processor_identity": dict(receipt_identity["processor_identity"]),
        "effective_settings": dict(receipt_identity["effective_settings"]),
        "likelihood_semantics": dict(receipt_identity["likelihood_semantics"]),
        "execution_model_identity": receipt_identity.get("execution_model_identity"),
        "adapter_identity": {
            "requested": launch_identity["adapter"],
            "observed": dict(receipt_identity["model_identity"].get("adapter") or {})
            if isinstance(receipt_identity["model_identity"], Mapping)
            else None,
        },
        "embedding_delta_identity": {
            "requested": launch_identity["embedding_delta"],
            "observed": dict(receipt_identity["model_identity"].get("embedding_delta") or {})
            if isinstance(receipt_identity["model_identity"], Mapping)
            else None,
        },
        "launch": launch_identity,
        "receipt": dict(receipt_identity),
    }
    _validate_runtime_identity_payload(runtime)
    return runtime


def _validate_runtime_identity_payload(
    runtime: Mapping[str, Any] | None,
    *,
    contract: CheckpointContract | None = None,
) -> None:
    """Validate the immutable runtime identity carried by a capture envelope."""

    if not isinstance(runtime, Mapping):
        raise SupportProbeError("capture shard runtime_identity is missing")
    required = (
        "schema_version",
        "status",
        "passed",
        "checkpoint",
        "config_fingerprint",
        "device",
        "effective_device",
        "normalized_device",
        "torch_current_device",
        "cuda_visible_devices",
        "physical_device_id",
        "dtype",
        "attn_implementation",
        "backend",
        "backend_mode",
        "response_family",
        "backend_version",
        "model_path",
        "model_dtype",
        "batch_size",
        "model_identity",
        "tokenizer_identity",
        "processor_identity",
        "effective_settings",
        "likelihood_semantics",
        "adapter_identity",
        "embedding_delta_identity",
        "launch",
        "receipt",
    )
    if any(key not in runtime for key in required):
        raise SupportProbeError("capture shard runtime_identity is incomplete")
    if runtime.get("schema_version") != RUNTIME_IDENTITY_SCHEMA_VERSION or runtime.get("status") != "validated":
        raise SupportProbeError("capture shard runtime_identity is not validated")
    if runtime.get("passed") is not True or not str(runtime.get("device", "")).startswith("cuda"):
        raise SupportProbeError("capture shard runtime_identity has no visible CUDA device")
    normalized_device = _normalize_cuda_device(runtime.get("device"), "runtime model device")
    normalized_effective = _normalize_cuda_device(runtime.get("effective_device"), "runtime receipt device")
    normalized_torch = _normalize_cuda_device(runtime.get("torch_current_device"), "runtime torch current device")
    if normalized_device != normalized_effective or normalized_device != normalized_torch:
        raise SupportProbeError("capture shard runtime_identity logical device mismatch")
    if runtime.get("normalized_device") != normalized_device:
        raise SupportProbeError("capture shard runtime_identity normalized device mismatch")
    visible = runtime.get("cuda_visible_devices")
    if not isinstance(visible, Mapping):
        raise SupportProbeError("capture shard runtime_identity CUDA_VISIBLE_DEVICES mapping is missing")
    raw_visible = visible.get("raw")
    tokens = visible.get("tokens")
    physical = visible.get("selected_physical_device")
    if not isinstance(raw_visible, str) or not raw_visible or not isinstance(tokens, list) or len(tokens) != 1:
        raise SupportProbeError("capture shard runtime_identity CUDA_VISIBLE_DEVICES mapping is ambiguous")
    if [token.strip() for token in raw_visible.split(",")] != tokens or tokens[0] == "-1":
        raise SupportProbeError("capture shard runtime_identity CUDA_VISIBLE_DEVICES mapping is inconsistent")
    if tokens[0] != physical or not isinstance(physical, str) or not physical or "," in physical:
        raise SupportProbeError("capture shard runtime_identity physical device mapping is invalid")
    if runtime.get("physical_device_id") != physical:
        raise SupportProbeError("capture shard runtime_identity physical device mismatch")
    if runtime.get("dtype") not in {"torch.float32", "float32"} or runtime.get("model_dtype") != "fp32":
        raise SupportProbeError("capture shard runtime_identity dtype is not fp32")
    if runtime.get("attn_implementation") != "sdpa":
        raise SupportProbeError("capture shard runtime_identity attention is not SDPA")
    if runtime.get("backend") != "hf" or runtime.get("backend_mode") != "generate" or runtime.get("response_family") != "hf":
        raise SupportProbeError("capture shard runtime_identity backend mismatch")
    if not isinstance(runtime.get("backend_version"), str) or not runtime["backend_version"]:
        raise SupportProbeError("capture shard runtime_identity backend version is missing")
    if not isinstance(runtime.get("model_path"), str) or not runtime["model_path"]:
        raise SupportProbeError("capture shard runtime_identity model path is missing")
    if isinstance(runtime.get("batch_size"), bool) or not isinstance(runtime.get("batch_size"), int) or runtime["batch_size"] <= 0:
        raise SupportProbeError("capture shard runtime_identity batch size is invalid")
    for key in ("model_identity", "tokenizer_identity", "processor_identity", "effective_settings", "likelihood_semantics"):
        if not isinstance(runtime.get(key), Mapping) or not runtime[key]:
            raise SupportProbeError(f"capture shard runtime_identity {key} is missing")
    model_identity = runtime["model_identity"]
    for key in ("family", "base", "qwen", "adapter", "embedding_delta"):
        if key not in model_identity:
            raise SupportProbeError(f"capture shard runtime_identity model identity lacks {key}")
    if not isinstance(model_identity["base"], Mapping) or not isinstance(model_identity["qwen"], Mapping):
        raise SupportProbeError("capture shard runtime_identity model identity base/qwen shape is invalid")
    if not isinstance(model_identity["base"].get("path"), str) or not model_identity["base"]["path"]:
        raise SupportProbeError("capture shard runtime_identity model base path is missing")
    observed_dtype = runtime["effective_settings"].get("observed_model_dtype")
    if not isinstance(observed_dtype, Mapping):
        raise SupportProbeError("capture shard runtime_identity dtype structure is missing")
    dtype_names = observed_dtype.get("parameter_dtype_names")
    dtype_counts = observed_dtype.get("parameter_dtype_counts")
    if dtype_names != ["torch.float32"] or not isinstance(dtype_counts, Mapping):
        raise SupportProbeError("capture shard runtime_identity dtype structure is invalid")
    if set(dtype_counts) != {"torch.float32"} or (
        isinstance(dtype_counts["torch.float32"], bool)
        or not isinstance(dtype_counts["torch.float32"], int)
        or dtype_counts["torch.float32"] <= 0
    ):
        raise SupportProbeError("capture shard runtime_identity dtype counts are invalid")
    observed_attention = runtime["effective_settings"].get("observed_attn_implementation")
    if not isinstance(observed_attention, str) or observed_attention != "sdpa":
        raise SupportProbeError("capture shard runtime_identity attention structure is invalid")
    effective_device = runtime["effective_settings"].get("device")
    if not isinstance(effective_device, str) or not effective_device.startswith("cuda"):
        raise SupportProbeError("capture shard runtime_identity device structure is invalid")
    launch = runtime.get("launch")
    receipt = runtime.get("receipt")
    if not isinstance(launch, Mapping) or not isinstance(receipt, Mapping):
        raise SupportProbeError("capture shard runtime_identity launch/receipt is missing")
    if launch.get("backend") != runtime.get("backend") or launch.get("model_path") != runtime.get("model_path"):
        raise SupportProbeError("capture shard runtime_identity launch mismatch")
    if launch.get("model_dtype") != runtime.get("model_dtype") or launch.get("batch_size") != runtime.get("batch_size"):
        raise SupportProbeError("capture shard runtime_identity launch settings mismatch")
    if receipt.get("backend") != runtime.get("backend") or receipt.get("backend_mode") != runtime.get("backend_mode") or receipt.get("response_family") != runtime.get("response_family"):
        raise SupportProbeError("capture shard runtime_identity receipt mismatch")
    if receipt.get("model_identity") != runtime.get("model_identity"):
        raise SupportProbeError("capture shard runtime_identity model identity mismatch")
    if receipt.get("effective_settings") != runtime.get("effective_settings"):
        raise SupportProbeError("capture shard runtime_identity effective settings mismatch")
    if contract is not None:
        if runtime.get("checkpoint") != contract.checkpoint or runtime.get("config_fingerprint") != contract.config_fingerprint:
            raise SupportProbeError("capture shard runtime_identity checkpoint/config mismatch")


def support_features(
    candidate_scores: Mapping[str, float],
    candidates: Sequence[Mapping[str, Any]],
    *,
    owner_id: str,
) -> dict[str, Any]:
    """Compute the prior owner-local peak features for one exact context.

    ``candidate_scores`` contains one score per unique physical candidate in a
    single image/category query group.  The local subset is selected by the
    authoritative generator-local/ambiguity partition; candidates strictly
    assigned to another owner are excluded from target support.  Category-route
    rank and boundary-gate values are intentionally absent from this function.
    """

    by_id = {str(key): _finite(value, f"candidate score {key}") for key, value in candidate_scores.items()}
    if not by_id:
        raise SupportProbeError("candidate score population is empty")
    expected_ids = {str(candidate.get("candidate_id")) for candidate in candidates}
    if expected_ids != set(by_id):
        raise SupportProbeError("candidate score population does not exactly match physical owner bank")
    unique_group = [score for score in by_id.values()]
    local = [str(candidate.get("candidate_id")) for candidate in _candidate_local_rows(candidates, owner_id=owner_id)]
    local_scores = [by_id[item] for item in local if item in by_id]
    if not local_scores:
        return {
            "assessed": False,
            "reason": "no_generator_local_candidates",
            "peak_lift": None,
            "local_concentration": None,
            "unique_population_size": len(unique_group),
            "local_bank_size": 0,
        }
    best = max(local_scores)
    ordered = sorted(local_scores)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0
    peak_lift = best - _logsumexp(unique_group) + math.log(len(unique_group))
    return {
        "assessed": True,
        "reason": None,
        "peak_lift": float(peak_lift),
        "local_concentration": float(best - median),
        "unique_population_size": len(unique_group),
        "local_bank_size": len(local_scores),
        "local_best_score": float(best),
        "local_bank_median": float(median),
        "local_candidate_ids": sorted(local),
        "support_semantics": legacy.SUPPORT_CRITERION_ID if hasattr(legacy, "SUPPORT_CRITERION_ID") else "local_peak_lift_and_local_concentration",
        "teacher_forced": True,
        "behavioral_transfer": False,
    }


def calibrate_support(
    observations: Sequence[Mapping[str, Any]],
    *,
    checkpoint: str,
    source_panel_sha256: str,
    derived_panel_sha256: str,
    epsilon: float = SUPPORT_EPSILON,
    quantile: float = SUPPORT_QUANTILE,
) -> dict[str, Any]:
    """Calibrate thresholds only from same-checkpoint native TP controls."""

    if checkpoint not in CHECKPOINTS:
        raise SupportProbeError("calibration checkpoint must be S or A")
    if not observations:
        raise SupportProbeError("absent TP calibrators: no native TP observations")
    accepted: list[Mapping[str, Any]] = []
    for row in observations:
        if row.get("checkpoint") != checkpoint:
            raise SupportProbeError("calibration contains a foreign checkpoint")
        if row.get("native_tp") is not True or row.get("strict_complete_row") is not True:
            raise SupportProbeError("calibration population contains a non-TP control")
        if row.get("run_kind") != "native_h0" or row.get("history_complete") is not True:
            raise SupportProbeError("calibration control is not complete native H0")
        if row.get("support_status") not in (None, "not_measured"):
            raise SupportProbeError("calibration control has a measured support status")
        if row.get("verified_support_claim") not in (None, False) or "verified_support" in row:
            raise SupportProbeError("calibration control contains a support claim")
        if row.get("source_panel_sha256") != source_panel_sha256 or row.get("derived_panel_sha256") != derived_panel_sha256:
            raise SupportProbeError("calibration control is bound to a foreign panel")
        if row.get("intervention") not in (None, False, "none", "native"):
            raise SupportProbeError("calibration control is intervention-derived")
        features = row.get("support_features")
        if not isinstance(features, Mapping) or features.get("assessed") is not True:
            continue
        if features.get("peak_lift") is None or features.get("local_concentration") is None:
            continue
        accepted.append(row)
    if not accepted:
        raise SupportProbeError("absent TP calibrators: no finite owner-local TP features")
    lifts = [_finite(row["support_features"]["peak_lift"], "TP peak_lift") for row in accepted]  # type: ignore[index]
    concentrations = [_finite(row["support_features"]["local_concentration"], "TP local_concentration") for row in accepted]  # type: ignore[index]
    payload: dict[str, Any] = {
        "schema_version": "static_dynamic_owner_support_calibration.v1",
        "unit_id": UNIT_ID,
        "checkpoint": checkpoint,
        "source_panel_sha256": source_panel_sha256,
        "derived_panel_sha256": derived_panel_sha256,
        "population": "checkpoint_native_true_positive_controls",
        "observation_count": len(accepted),
        "quantile": float(quantile),
        "quantile_is_primary_and_fixed": True,
        "theta_peak_lift": _quantile(lifts, quantile),
        "theta_local_concentration": _quantile(concentrations, quantile),
        "epsilon": float(epsilon),
        "epsilon_is_adaptive": False,
        "rule": "peak_lift >= theta_peak_lift + epsilon AND local_concentration >= theta_local_concentration + epsilon",
        "rank_is_not_a_support_input": True,
        "category_route_is_not_a_support_input": True,
        "boundary_gate_is_not_a_support_input": True,
        "teacher_forced_is_diagnostic_only": True,
        "behavioral_transfer_claim": False,
        "legacy12_control_count": sum(int(row.get("image_id")) != IMAGE_2299 for row in accepted),
        "image2299_control_count": sum(int(row.get("image_id")) == IMAGE_2299 for row in accepted),
        "control_prefixes": sorted(str(row.get("exact_prefix_sha256")) for row in accepted),
        "no_future_or_intervention_leakage": True,
    }
    payload["calibration_sha256"] = sha256_json(payload)
    return payload


def _support_decision(features: Mapping[str, Any], calibration: Mapping[str, Any]) -> bool | None:
    if features.get("assessed") is not True:
        return None
    peak = features.get("peak_lift")
    concentration = features.get("local_concentration")
    if peak is None or concentration is None:
        return None
    return bool(
        float(peak) >= float(calibration["theta_peak_lift"]) + float(calibration["epsilon"])
        and float(concentration) >= float(calibration["theta_local_concentration"]) + float(calibration["epsilon"])
    )


def _record_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    required = ("image_id", "gt_owner_id", "natural_boundary", "exact_prefix_sha256")
    for key in required:
        if key not in record:
            raise SupportProbeError(f"H0 record missing {key}")
    return {
        "image_id": int(record["image_id"]),
        "gt_owner_id": str(record["gt_owner_id"]),
        "source_panel_object_index": int(record.get("source_panel_object_index", -1)),
        "coco_ann_id": record.get("coco_ann_id"),
        "category_name": str(record.get("category_name", record.get("category", ""))).lower(),
        "bbox_pixel_xyxy": list(record.get("bbox_pixel_xyxy", record.get("pixel_bbox", []))),
        "natural_boundary": record["natural_boundary"],
        "exact_prefix_sha256": _sha256(record["exact_prefix_sha256"], "exact_prefix_sha256"),
    }


def build_support_envelope(
    *,
    contract: CheckpointContract,
    h0: H0Inputs,
    records: Sequence[Mapping[str, Any]],
    calibration: Mapping[str, Any],
    panel: PanelInputs,
) -> dict[str, Any]:
    """Build the envelope accepted by ``materialize_static_dynamic_owner_interface_cohort``."""

    if calibration.get("checkpoint") != contract.checkpoint:
        raise SupportProbeError("support calibration checkpoint mismatch")
    output_records: list[dict[str, Any]] = []
    for raw in records:
        identity = _record_identity(raw)
        if raw.get("support_status") != "measured":
            raise SupportProbeError("support record support_status must be measured")
        if raw.get("verified_support_claim") is not True:
            raise SupportProbeError("support record verified_support_claim must be true")
        if not isinstance(raw.get("verified_support"), bool):
            raise SupportProbeError("support record verified_support must be a JSON boolean")
        item = {
            "unit_id": UNIT_ID,
            "checkpoint": contract.checkpoint,
            "config_fingerprint": contract.config_fingerprint,
            "source_panel_sha256": contract.source_panel_sha256,
            "derived_panel_sha256": contract.derived_panel_sha256,
            "run_kind": "native_h0",
            "history_complete": True,
            "intervention": "none",
            "native_tp": bool(raw.get("native_tp")),
            "native_fn": bool(raw.get("native_fn")),
            "strict_complete_row": bool(raw.get("strict_complete_row")),
            "natural_boundary_valid": True,
            "due_boundary_index": raw.get("due_boundary_index", raw.get("natural_boundary")),
            "boundary_disposition": raw.get("boundary_disposition"),
            "prefix_semantics": raw.get("prefix_semantics"),
            "generated_history_start_step": raw.get("generated_history_start_step"),
            "generated_history_end_step": raw.get("generated_history_end_step"),
            "generated_history_stop_step": raw.get("generated_history_stop_step"),
            "excludes_stop": True,
            "covered_owner_ids": list(raw.get("covered_owner_ids") or []),
            "latest_covered_owner_id": raw.get("latest_covered_owner_id"),
            "queried_owner_not_covered": True,
            "is_earliest_eligible_boundary": raw.get("is_earliest_eligible_boundary"),
            "due_boundary_evidence": dict(raw.get("due_boundary_evidence") or {}),
            **identity,
            "exact_prefix_token_ids": list(raw["exact_prefix_token_ids"]) if isinstance(raw.get("exact_prefix_token_ids"), list) else None,
            "verified_support": raw["verified_support"],
            "verified_support_claim": True,
            "support_status": "measured",
            "support_features": dict(raw.get("support_features") or {}),
            "support_calibration_sha256": calibration["calibration_sha256"],
            "support_semantics": "physical_owner_specific_complete_row_geometry_local_peak",
            "category_route_axis": "diagnostic_only",
            "boundary_gate_axis": "diagnostic_only",
            "coordinate_localization_axis": "diagnostic_teacher_forced_only",
            "teacher_forced_is_behavioral_transfer": False,
            "no_future_or_intervention_leakage": True,
        }
        if item["exact_prefix_token_ids"] is None:
            # Materializer does not require the IDs, but a live support claim
            # without the attested exact prefix is not decision-bearing.
            raise SupportProbeError("support record lacks exact H0 prefix token IDs")
        if sha256_json(item["exact_prefix_token_ids"]) != item["exact_prefix_sha256"]:
            raise SupportProbeError("support exact prefix token IDs/hash mismatch")
        output_records.append(item)
    if not output_records:
        raise SupportProbeError("support envelope has no records")
    output_records.sort(key=lambda row: (int(row["image_id"]), str(row["gt_owner_id"])))
    subset = {
        "legacy12": {
            "record_count": sum(int(row["image_id"]) != IMAGE_2299 for row in output_records),
            "verified_support_count": sum(int(row["image_id"]) != IMAGE_2299 and row.get("verified_support") is True for row in output_records),
        },
        "image2299": {
            "record_count": sum(int(row["image_id"]) == IMAGE_2299 for row in output_records),
            "verified_support_count": sum(int(row["image_id"]) == IMAGE_2299 and row.get("verified_support") is True for row in output_records),
        },
    }
    envelope: dict[str, Any] = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "checkpoint": contract.checkpoint,
        "config_fingerprint": contract.config_fingerprint,
        "source_panel_sha256": contract.source_panel_sha256,
        "derived_panel_sha256": contract.derived_panel_sha256,
        "run_kind": "native_h0",
        "arm": "native",
        "history_complete": True,
        "native_outcome_only": False,
        "verified_support_claim": True,
        "support_rule": {
            "criterion_id": "local_peak_lift_and_local_concentration_under_both_ambiguity_bounds",
            "calibration_sha256": calibration["calibration_sha256"],
            "calibration_population": "checkpoint_native_true_positive_controls",
            "teacher_forced_diagnostic_only": True,
            "behavioral_transfer_claim": False,
        },
        "subset_summaries": subset,
        "records": output_records,
    }
    # Binding H0 source itself prevents a support envelope from silently being
    # detached from the native ledger used to choose its boundaries.
    envelope["h0_source_sha256"] = str(h0.source_info["sha256"])
    envelope["records_sha256"] = sha256_json(output_records)
    return envelope


def _write_immutable(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path).expanduser().resolve()
    payload = canonical_json_bytes(value) + b"\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.read_bytes() != payload:
        raise SupportProbeError(f"existing output is not identical: {destination}")
    if not destination.exists():
        destination.write_bytes(payload)


def contract_receipt(
    *,
    contract: CheckpointContract,
    h0: H0Inputs,
    panel: PanelInputs,
    physical: Sequence[Mapping[str, Any]],
    accounting: Mapping[str, Any],
    mode: str,
    event_limit: int | None,
    num_shards: int = 1,
    candidate_batch_size: int = 16,
) -> dict[str, Any]:
    candidates = list(cohort.FROZEN_CANDIDATES)
    selected = candidates if event_limit is None else candidates[:event_limit]
    h0_tp = [row for row in h0.records if row.get("native_tp") is True]
    plan = _capture_context_plan(
        contract=contract,
        h0=h0,
        panel=panel,
        physical=physical,
        accounting=accounting,
        event_limit=event_limit,
    )
    work_units = _capture_work_units(
        plan,
        num_shards=num_shards,
        candidate_batch_size=candidate_batch_size,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "status": "contract_ready" if mode == "contract" else "dry_run_ready",
        "mode": mode,
        "checkpoint": contract.checkpoint,
        "wrapper": contract.wrapper,
        "parser": contract.parser,
        "infer_config": {
            "path": str(contract.config_path),
            "sha256": contract.config_sha256,
            "fingerprint": contract.config_fingerprint,
            "resolved_authority_path": str(contract.resolved_config_path) if contract.resolved_config_path is not None else None,
            "resolved_authority_sha256": sha256_file(contract.resolved_config_path) if contract.resolved_config_path is not None else None,
            "resolved_authority_used": contract.resolved_config is not None and contract.resolved_config_path is not None,
            "dtype": "fp32",
            "attn_implementation": "sdpa",
        },
        "panel": {
            "source_sha256": contract.source_panel_sha256,
            "derived_sha256": contract.derived_panel_sha256,
            "derived_receipt_sha256": str(panel.receipt_info["sha256"]),
            "legacy12_images": sorted(image for image in panel.owners_by_image if image != IMAGE_2299),
            "image2299_present": IMAGE_2299 in panel.owners_by_image,
        },
        "h0": {
            "source_sha256": str(h0.source_info["sha256"]),
            "record_count": len(h0.records),
            "native_tp_count": len(h0_tp),
            "history_complete": True,
            "run_kind": "native_h0",
            "exact_prefix_records": sum(isinstance(row.get("exact_prefix_token_ids"), list) for row in h0.records),
        },
        "candidate_pool": {
            "frozen_count": len(candidates),
            "selected_count": len(selected),
            "sha256": sha256_json(candidates),
            "selected_gt_owner_ids": [str(row["gt_owner_id"]) for row in selected],
        },
        "owner_bank": {
            "physical_candidate_count": len(physical),
            "accounting_sha256": sha256_json(accounting),
            "semantics_source": "scripts.research.build_sorted_owner_accessibility_census_plan",
            "support_criterion": "local_peak_lift_and_local_concentration_under_both_ambiguity_bounds",
        },
        "capture_plan": {
            "schema_version": CAPTURE_SCHEMA_VERSION,
            "context_plan_sha256": plan["context_plan_sha256"],
            "context_count": len(plan["contexts"]),
            "calibration_context_count": sum(row["kind"] == "calibration" for row in plan["contexts"]),
            "candidate_context_count": sum(row["kind"] == "candidate" for row in plan["contexts"]),
            "num_shards": num_shards,
            "work_units": work_units,
        },
        "execution_contract": {
            "no_optimizer": True,
            "no_training": True,
            "native_prefix_only": True,
            "teacher_forced_diagnostic_only": True,
            "behavioral_transfer_claim": False,
            "legacy12_vs_image2299_separate": True,
            "event_limit": event_limit,
        },
    }


class HFNativeRowScorer:
    """Small adapter over the canonical HF backend exact-history API."""

    def __init__(self, config: CheckpointContract, panel: PanelInputs, h0: H0Inputs) -> None:
        self.contract = config
        self.panel = panel
        self.h0 = h0
        self._session: Any = None
        self._histories: dict[int, Any] = {}
        self._requests: dict[int, Any] = {}
        self._tokenizer: Any = None
        self._prompt_records: dict[int, Any] = {}

    def open(self) -> None:
        from src.config.fingerprint import sha256_json as config_fingerprint
        from src.data import load_raw_examples
        from src.inference.backend import DecodeRequest, GenerationPolicy, open_backend_session
        from src.inference.image_plan import plan_image_batch
        from src.inference.pipeline import _processor_config, _template_config
        from src.inference.prompt import build_prompt_record
        from src.inference.runtime import assemble_frontend

        from src.config.inference import InferConfig

        if self.contract.resolved_config is not None:
            if sha256_json(self.contract.resolved_config) != self.contract.config_fingerprint:
                raise SupportProbeError("immutable resolved config raw fingerprint drifted")
            try:
                cfg = InferConfig.model_validate(dict(self.contract.resolved_config))
            except Exception as exc:
                raise SupportProbeError("immutable resolved config cannot be loaded") from exc
        else:
            raise SupportProbeError(
                "live support requires immutable H0 resolved config authority"
            )
        # Keep the recipe exact even if a caller modified the resolved object.
        if cfg.model.dtype != "fp32" or cfg.backend.hf.attn_implementation != "sdpa":
            raise SupportProbeError("live HF config is not fp32/SDPA")
        frontend = assemble_frontend(
            cfg,
            generation_config_fingerprint=config_fingerprint(cfg.generation.model_dump(mode="json")),
        )
        raw_source = cfg.data.input_jsonl
        raw_rows = load_raw_examples(raw_source)
        raw_by_image = {
            int(item.metadata.get("source", {}).get("image_id")): item
            for item in raw_rows
        }
        selected_ids = sorted({int(row["image_id"]) for row in self.h0.records})
        selected_raw = [raw_by_image[image] for image in selected_ids if image in raw_by_image]
        if len(selected_raw) != len(selected_ids):
            raise SupportProbeError("config source is missing an H0 image")
        plans = plan_image_batch(
            selected_raw,
            components=frontend.qwen,
            processor_config=_processor_config(cfg),
            row_indices=[int(item.metadata.get("source", {}).get("row_index", index)) for index, item in enumerate(selected_raw)],
        )
        plans_by_image = {
            int(raw.metadata.get("source", {}).get("image_id")): plan
            for raw, plan in zip(selected_raw, plans.rows, strict=True)
        }
        for image in selected_ids:
            row = next(item for item in self.h0.records if int(item["image_id"]) == image)
            plan = plans_by_image[image]
            if str(row.get("image_identity", {}).get("image_path")) != str(plan.image_path):
                raise SupportProbeError(f"image path drift for H0 image {image}")
            if str(plan.image_content_sha256) != sha256_file(plan.image_path):
                raise SupportProbeError(f"image hash could not be re-established for {image}")
            prompt = build_prompt_record(
                raw_by_image[image],
                _template_config(cfg),
                processor=frontend.qwen.processor,
                row_index=int(raw_by_image[image].metadata.get("source", {}).get("row_index", 0)),
                merged_visual_tokens=plan.merged_visual_tokens,
            )
            policy = GenerationPolicy(
                max_new_tokens=3084,
                repetition_penalty=1.0,
                temperature=0.0,
                top_p=1.0,
            )
            self._requests[image] = DecodeRequest(
                request_id=f"support:{self.contract.checkpoint}:{image}",
                chat_text=prompt.chat_text,
                input_prompt_token_ids=tuple(prompt.input_prompt_token_ids),
                expected_executed_prompt_token_ids=tuple(prompt.expected_executed_prompt_token_ids),
                image_path=plan.image_path,
                declared_image_width=int(plan.declared_width),
                declared_image_height=int(plan.declared_height),
                decoded_image_width=int(plan.decoded_width),
                decoded_image_height=int(plan.decoded_height),
                image_sha256=str(plan.image_content_sha256),
                expected_image_grid_thw=tuple(plan.expected_image_grid_thw),
                generation_policy=policy,
            )
            self._prompt_records[image] = prompt
        self._tokenizer = frontend.qwen.tokenizer
        self._ctx = open_backend_session(frontend.launch)
        self._session = self._ctx.__enter__()
        validate_live_runtime_identity(self._session)
        for image, request in self._requests.items():
            history = self._session.prepare_exact_history(request)
            prompt_ids = tuple(request.expected_executed_prompt_token_ids)
            if tuple(history.conditioning_token_ids) != prompt_ids:
                raise SupportProbeError(f"HF prompt identity drift for image {image}")
            self._histories[image] = history

    def close(self) -> None:
        if getattr(self, "_ctx", None) is not None:
            self._ctx.__exit__(None, None, None)
            self._ctx = None
            self._session = None

    def _row_tokens(self, category: str, coord_token_ids: Sequence[int]) -> tuple[int, ...]:
        if self._tokenizer is None:
            raise SupportProbeError("HF scorer is not open")
        category_ids = self._category_token_ids(category)
        forbidden = {legacy.OBJECT_REF_START, legacy.OBJECT_REF_END, legacy.BOX_START, legacy.BOX_END}
        if any(value in forbidden for value in category_ids):
            raise SupportProbeError(f"category {category!r} token span collides with wrapper")
        row = [legacy.OBJECT_REF_START, *category_ids, legacy.OBJECT_REF_END, legacy.BOX_START, *map(int, coord_token_ids), legacy.BOX_END]
        if self.contract.checkpoint == "A":
            row.append(151669)
        return tuple(row)

    def _category_token_ids(self, category: str) -> list[int]:
        if self._tokenizer is None:
            raise SupportProbeError("HF scorer is not open")
        encode = getattr(self._tokenizer, "encode", None)
        if not callable(encode):
            raise SupportProbeError("native tokenizer does not expose encode")
        try:
            category_ids = [int(value) for value in encode(category, add_special_tokens=False)]
        except TypeError:
            category_ids = [int(value) for value in encode(category)]
        if not category_ids:
            raise SupportProbeError(f"category {category!r} has no tokenizer IDs")
        return category_ids

    def score(self, record: Mapping[str, Any], candidate: Mapping[str, Any]) -> float:
        image = int(record["image_id"])
        prefix = record.get("exact_prefix_token_ids")
        if not isinstance(prefix, list):
            raise SupportProbeError("exact H0 prefix is unavailable for live scoring")
        if sha256_json(prefix) != record.get("exact_prefix_sha256"):
            raise SupportProbeError("exact H0 prefix hash drifted before scoring")
        if IM_END in prefix:
            raise SupportProbeError("refusing teacher-forced scoring after H0 im_end/STOP")
        history = self._session.extend_exact_history(self._histories[image], [int(value) for value in prefix])
        row_tokens = self._row_tokens(
            str(candidate["normalized_description"]),
            candidate["coord_token_ids"],
        )
        evidence = self._session.teacher_forced_evidence(history, row_tokens)
        if len(evidence) != len(row_tokens):
            raise SupportProbeError("HF teacher-forced evidence is not row-complete")
        values = [float(item.raw_model_logprob) for item in evidence]
        if any(not math.isfinite(value) for value in values):
            raise SupportProbeError("HF teacher-forced row score is non-finite")
        # Support is a coordinate-localization statistic.  Category-route and
        # boundary-carrier likelihoods are deliberately excluded from the
        # score that feeds peak_lift/concentration; they remain separate
        # diagnostics rather than hidden support inputs.
        category_ids = self._category_token_ids(str(candidate["normalized_description"]))
        coordinate_start = 3 + len(category_ids)
        coordinate_end = coordinate_start + 4
        if coordinate_end > len(values):
            raise SupportProbeError("HF row evidence has no complete coordinate span")
        return float(sum(values[coordinate_start:coordinate_end]))


def _candidate_records(
    *,
    h0: H0Inputs,
    panel: PanelInputs,
    physical: Sequence[Mapping[str, Any]],
    accounting: Mapping[str, Any],
    contract: CheckpointContract,
    event_limit: int | None,
) -> list[dict[str, Any]]:
    by_owner: dict[str, Mapping[str, Any]] = {}
    for row in h0.records:
        owner = str(row.get("gt_owner_id"))
        if owner in by_owner:
            raise SupportProbeError(f"H0 contains duplicate physical owner record {owner}")
        by_owner[owner] = row
    selected = list(cohort.FROZEN_CANDIDATES)
    if event_limit is not None:
        selected = selected[:event_limit]
    output: list[dict[str, Any]] = []
    for candidate in selected:
        owner_id = str(candidate["gt_owner_id"])
        raw = by_owner.get(owner_id)
        if raw is None:
            raise SupportProbeError(f"frozen candidate {owner_id} is absent from checkpoint H0")
        panel_owner = next(
            (owner for owner in panel.owners_by_image[int(candidate["image_id"])] if owner.gt_owner_id == owner_id),
            None,
        )
        if panel_owner is None or panel_owner.category != str(candidate["category"]).lower() or panel_owner.bbox != tuple(candidate["pixel_bbox"]):
            raise SupportProbeError(f"frozen candidate mapping mismatch for {owner_id}")
        bank_candidates = _bank_group(physical, int(candidate["image_id"]), str(candidate["category"]).lower())
        accounting_row = accounting.get(owner_id)
        if not isinstance(accounting_row, Mapping):
            raise SupportProbeError(f"owner bank accounting missing {owner_id}")
        support_bank_status = accounting_row.get("bank_coverage_status")
        item = dict(raw)
        item["candidate"] = candidate
        item["bank_candidates"] = bank_candidates
        item["bank_accounting"] = dict(accounting_row)
        item["bank_eligible"] = support_bank_status in {"full", "adequate_reduced"}
        item["checkpoint"] = contract.checkpoint
        output.append(item)
    return output


CAPTURE_SCHEMA_VERSION = f"{SCHEMA_VERSION}.capture.v1"
CAPTURE_RECEIPT_SCHEMA_VERSION = f"{SCHEMA_VERSION}.capture-receipt.v1"
CAPTURE_CONTEXT_KINDS = ("calibration", "candidate")
RUNTIME_IDENTITY_SCHEMA_VERSION = f"{SCHEMA_VERSION}.runtime.v1"


def _stable_capture_context_key(
    *,
    kind: str,
    checkpoint: str,
    image_id: int,
    owner_id: str,
    exact_prefix_sha256: str | None,
) -> str:
    if kind not in CAPTURE_CONTEXT_KINDS:
        raise SupportProbeError(f"unknown capture context kind: {kind}")
    return "|".join(
        (
            kind,
            str(checkpoint),
            str(int(image_id)),
            str(owner_id),
            str(exact_prefix_sha256 or "<no-prefix>"),
        )
    )


def _capture_context_id(stable_key: str) -> str:
    return f"ctx:{sha256_json(stable_key)[:24]}"


def _partition_number(stable_key: str, num_shards: int) -> int:
    if isinstance(num_shards, bool) or not isinstance(num_shards, int) or num_shards <= 0:
        raise SupportProbeError("num_shards must be a positive integer")
    return int(sha256_json(stable_key)[:16], 16) % num_shards


def partition_capture_contexts(
    contexts: Sequence[Mapping[str, Any]],
    *,
    shard_index: int,
    num_shards: int,
) -> list[dict[str, Any]]:
    """Partition contexts by a content-stable key, independent of input order."""

    if isinstance(shard_index, bool) or not isinstance(shard_index, int):
        raise SupportProbeError("shard_index must be an integer")
    if isinstance(num_shards, bool) or not isinstance(num_shards, int) or num_shards <= 0:
        raise SupportProbeError("num_shards must be a positive integer")
    if not 0 <= shard_index < num_shards:
        raise SupportProbeError("shard_index must satisfy 0 <= shard_index < num_shards")
    seen: set[str] = set()
    ordered: list[dict[str, Any]] = []
    for raw in contexts:
        if not isinstance(raw, Mapping):
            raise SupportProbeError("capture context must be an object")
        context = dict(raw)
        stable_key = context.get("stable_key")
        context_id = context.get("context_id")
        if not isinstance(stable_key, str) or not stable_key:
            raise SupportProbeError("capture context has no stable_key")
        if not isinstance(context_id, str) or not context_id:
            raise SupportProbeError("capture context has no context_id")
        if stable_key in seen or context_id in seen:
            raise SupportProbeError("capture context identity is duplicated")
        seen.update((stable_key, context_id))
        context["shard_index"] = _partition_number(stable_key, num_shards)
        ordered.append(context)
    ordered.sort(key=lambda row: str(row["stable_key"]))
    return [row for row in ordered if row["shard_index"] == shard_index]


def _capture_context_plan(
    *,
    contract: CheckpointContract,
    h0: H0Inputs,
    panel: PanelInputs,
    physical: Sequence[Mapping[str, Any]],
    accounting: Mapping[str, Any],
    event_limit: int | None,
    smoke_calibrators: int = 0,
) -> dict[str, Any]:
    """Build the immutable calibration/candidate work plan for capture shards."""

    if isinstance(smoke_calibrators, bool) or not isinstance(smoke_calibrators, int) or smoke_calibrators < 0:
        raise SupportProbeError("smoke_calibrators must be a non-negative integer")
    by_owner = {str(row["gt_owner_id"]): row for row in h0.records}
    calibration_rows = [
        row
        for row in h0.records
        if row.get("native_tp") is True and row.get("strict_complete_row") is True
    ]
    calibration_rows.sort(
        key=lambda row: (
            int(row["image_id"]),
            str(row["gt_owner_id"]),
            str(row.get("exact_prefix_sha256") or "<no-prefix>"),
        )
    )
    calibration_scope = "all_native_tp"
    if smoke_calibrators:
        calibration_rows = calibration_rows[:smoke_calibrators]
        calibration_scope = "smoke_subset_non_decision"
    selected = list(cohort.FROZEN_CANDIDATES)
    if event_limit is not None:
        selected = selected[:event_limit]
    groups: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    for candidate in physical:
        groups.setdefault(
            (int(candidate["image_id"]), str(candidate["normalized_description"]).lower()),
            [],
        ).append(candidate)

    contexts: list[dict[str, Any]] = []

    def add_context(kind: str, row: Mapping[str, Any], *, candidate: Mapping[str, Any] | None = None) -> None:
        image = int(row["image_id"])
        owner_id = str(row["gt_owner_id"])
        category = str(row.get("category_name", row.get("category", ""))).lower()
        prefix_hash = row.get("exact_prefix_sha256")
        stable_key = _stable_capture_context_key(
            kind=kind,
            checkpoint=contract.checkpoint,
            image_id=image,
            owner_id=owner_id,
            exact_prefix_sha256=str(prefix_hash) if prefix_hash is not None else None,
        )
        bank = groups.get((image, category), [])
        candidate_ids = [str(item["candidate_id"]) for item in bank]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise SupportProbeError(f"physical owner bank has duplicate candidate IDs for {image}/{category}")
        bank_row = accounting.get(owner_id)
        bank_status = bank_row.get("bank_coverage_status") if isinstance(bank_row, Mapping) else None
        eligible = (
            row.get("natural_boundary_valid") is True
            and bank_status in {"full", "adequate_reduced"}
            and bool(bank)
        )
        reason = None
        if row.get("natural_boundary_valid") is not True:
            reason = "no_valid_native_due_boundary"
        elif bank_status not in {"full", "adequate_reduced"}:
            reason = "undercovered_owner_bank"
        elif not bank:
            reason = "empty_physical_owner_bank"
        context = {
            "context_id": _capture_context_id(stable_key),
            "stable_key": stable_key,
            "kind": kind,
            "checkpoint": contract.checkpoint,
            "image_id": image,
            "gt_owner_id": owner_id,
            "category_name": category,
            "native_tp": bool(row.get("native_tp")),
            "native_fn": bool(row.get("native_fn")),
            "strict_complete_row": bool(row.get("strict_complete_row")),
            "natural_boundary_valid": bool(row.get("natural_boundary_valid")),
            "natural_boundary": row.get("natural_boundary"),
            "exact_prefix_sha256": row.get("exact_prefix_sha256"),
            "exact_prefix_token_count": len(row.get("exact_prefix_token_ids") or [])
            if isinstance(row.get("exact_prefix_token_ids"), list)
            else None,
            "candidate_gt_owner_id": str(candidate["gt_owner_id"]) if candidate is not None else None,
            "candidate_labels": list(candidate.get("historical_labels", ())) if candidate is not None else [],
            "candidate_ids": sorted(candidate_ids),
            "bank_coverage_status": bank_status,
            "eligible_for_score": eligible,
            "ineligible_reason": reason,
            "support_features": None,
            "status": "planned" if eligible else "indeterminate",
        }
        contexts.append(context)

    for row in calibration_rows:
        add_context("calibration", row)
    for candidate in selected:
        owner_id = str(candidate["gt_owner_id"])
        row = by_owner.get(owner_id)
        if row is None:
            raise SupportProbeError(f"frozen candidate {owner_id} is absent from checkpoint H0")
        image = int(candidate["image_id"])
        owner = next(
            (item for item in panel.owners_by_image.get(image, ()) if item.gt_owner_id == owner_id),
            None,
        )
        if owner is None or owner.category != str(candidate["category"]).lower() or owner.bbox != tuple(candidate["pixel_bbox"]):
            raise SupportProbeError(f"frozen candidate mapping mismatch for {owner_id}")
        add_context("candidate", row, candidate=candidate)
    contexts.sort(key=lambda row: str(row["stable_key"]))
    for position, context in enumerate(contexts):
        context["context_plan_position"] = position
    context_ids = [str(row["context_id"]) for row in contexts]
    plan_core = {
        "schema_version": CAPTURE_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "checkpoint": contract.checkpoint,
        "config_fingerprint": contract.config_fingerprint,
        "source_panel_sha256": contract.source_panel_sha256,
        "derived_panel_sha256": contract.derived_panel_sha256,
        "h0_source_sha256": str(h0.source_info["sha256"]),
        "event_limit": event_limit,
        "calibration_scope": calibration_scope,
        "smoke_calibrators": smoke_calibrators,
        "contexts": contexts,
    }
    plan_core["context_plan_sha256"] = sha256_json(plan_core)
    plan_core["context_ids_sha256"] = sha256_json(context_ids)
    return plan_core


def _capture_work_units(
    plan: Mapping[str, Any], *, num_shards: int, candidate_batch_size: int
) -> dict[str, Any]:
    if isinstance(candidate_batch_size, bool) or not isinstance(candidate_batch_size, int) or candidate_batch_size <= 0:
        raise SupportProbeError("candidate_batch_size must be a positive integer")
    contexts = list(plan.get("contexts", ()))
    expected = []
    for shard in range(num_shards):
        assigned = partition_capture_contexts(contexts, shard_index=shard, num_shards=num_shards)
        scalar = sum(len(row.get("candidate_ids") or ()) for row in assigned if row.get("eligible_for_score"))
        expected.append(
            {
                "shard_index": shard,
                "context_count": len(assigned),
                "scalar_equivalent_forward_count": scalar,
                "batched_forward_estimate": math.ceil(scalar / candidate_batch_size) if scalar else 0,
                "batching_admitted": False,
            }
        )
    scalar_total = sum(row["scalar_equivalent_forward_count"] for row in expected)
    calibration_contexts = [row for row in contexts if row.get("kind") == "calibration"]
    candidate_contexts = [row for row in contexts if row.get("kind") == "candidate"]
    calibration_scalar = sum(
        len(row.get("candidate_ids") or ())
        for row in calibration_contexts
        if row.get("eligible_for_score")
    )
    candidate_scalar = sum(
        len(row.get("candidate_ids") or ())
        for row in candidate_contexts
        if row.get("eligible_for_score")
    )
    return {
        "candidate_batch_size": candidate_batch_size,
        "batching_admitted": False,
        "batching_status": "not_admitted_exact_history_api_scalar_only",
        "batching_reason": "HF exact-history API has no safe candidate-batch interface in this probe",
        "scalar_equivalent_forward_count": scalar_total,
        "calibration_scalar_equivalent_forward_count": calibration_scalar,
        "candidate_scalar_equivalent_forward_count": candidate_scalar,
        "batched_forward_estimate": math.ceil(scalar_total / candidate_batch_size) if scalar_total else 0,
        "per_shard": expected,
    }


def _quarantine_capture_error(exc: BaseException) -> tuple[str, str]:
    message = str(exc)
    lowered = message.lower()
    if "out of memory" in lowered or "cuda oom" in lowered:
        return "quarantined_oom", message
    if "parity" in lowered or "batch" in lowered:
        return "quarantined_parity", message
    return "quarantined_capture_error", message


def _capture_context_observation(
    context: Mapping[str, Any],
    *,
    h0_by_owner: Mapping[str, Mapping[str, Any]],
    groups: Mapping[tuple[int, str], Sequence[Mapping[str, Any]]],
    scorer: Any,
) -> dict[str, Any]:
    result = {
        "context_id": str(context["context_id"]),
        "stable_key": str(context["stable_key"]),
        "kind": str(context["kind"]),
        "image_id": int(context["image_id"]),
        "gt_owner_id": str(context["gt_owner_id"]),
        "candidate_gt_owner_id": context.get("candidate_gt_owner_id"),
        "category_name": str(context.get("category_name", "")),
        "candidate_ids": sorted(str(item) for item in (context.get("candidate_ids") or ())),
        "status": "indeterminate",
        "reason": context.get("ineligible_reason"),
        "support_features": None,
        "candidate_scores": None,
        "candidate_score_count": 0,
        "candidate_scores_sha256": None,
    }
    if context.get("eligible_for_score") is not True:
        return result
    raw = h0_by_owner.get(str(context["gt_owner_id"]))
    if raw is None:
        result["reason"] = "missing_h0_owner"
        return result
    group = groups.get((int(context["image_id"]), str(context["category_name"]).lower()), ())
    try:
        score_map = {str(candidate["candidate_id"]): scorer.score(raw, candidate) for candidate in group}
        score_map = {
            candidate_id: _finite(score, f"candidate score {candidate_id}")
            for candidate_id, score in score_map.items()
        }
    except Exception as exc:  # noqa: BLE001 - scorer OOM/parity must quarantine the whole shard
        status, reason = _quarantine_capture_error(exc)
        result.update({"status": status, "reason": reason})
        return result
    try:
        features = support_features(score_map, group, owner_id=str(context["gt_owner_id"]))
        result.update(
            {
                "status": "measured",
                "reason": None,
                "support_features": features,
                "candidate_scores": dict(sorted(score_map.items())),
                "candidate_score_count": len(score_map),
                "candidate_scores_sha256": sha256_json(score_map),
            }
        )
    except Exception as exc:  # owner-local bank/data issue: retain this context as indeterminate
        result.update({"status": "indeterminate", "reason": f"owner_local_support_error:{exc}"})
    return result


def _capture_observations_until_quarantine(
    contexts: Sequence[Mapping[str, Any]],
    *,
    h0_by_owner: Mapping[str, Mapping[str, Any]],
    groups: Mapping[tuple[int, str], Sequence[Mapping[str, Any]]],
    scorer: Any,
) -> list[dict[str, Any]]:
    """Capture in deterministic order and stop the shard on OOM/parity."""

    observations: list[dict[str, Any]] = []
    for context in contexts:
        observation = _capture_context_observation(
            context,
            h0_by_owner=h0_by_owner,
            groups=groups,
            scorer=scorer,
        )
        observations.append(observation)
        if str(observation.get("status", "")).startswith("quarantined_"):
            break
    return observations


def _capture_envelope(
    *,
    contract: CheckpointContract,
    h0: H0Inputs,
    plan: Mapping[str, Any],
    assigned_contexts: Sequence[Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
    shard_index: int,
    num_shards: int,
    candidate_batch_size: int,
    runtime_identity: Mapping[str, Any] | None,
) -> dict[str, Any]:
    _validate_runtime_identity_payload(runtime_identity, contract=contract)
    expected_ids = [str(row["context_id"]) for row in assigned_contexts]
    observed_ids = [str(row["context_id"]) for row in observations]
    work_units = _capture_work_units(plan, num_shards=num_shards, candidate_batch_size=candidate_batch_size)
    shard_work = work_units["per_shard"][shard_index]
    quarantined = next(
        (
            row
            for row in observations
            if str(row.get("status", "")).startswith("quarantined_")
        ),
        None,
    )
    envelope = {
        "schema_version": CAPTURE_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "checkpoint": contract.checkpoint,
        "config_fingerprint": contract.config_fingerprint,
        "source_panel_sha256": contract.source_panel_sha256,
        "derived_panel_sha256": contract.derived_panel_sha256,
        "h0_source_sha256": str(h0.source_info["sha256"]),
        "shard_index": shard_index,
        "num_shards": num_shards,
        "status": "quarantined" if quarantined is not None else "captured",
        "quarantine": (
            {
                "status": quarantined.get("status"),
                "context_id": quarantined.get("context_id"),
                "reason": quarantined.get("reason"),
                "policy": "stop_shard_no_fallback",
            }
            if quarantined is not None
            else None
        ),
        "event_limit": plan.get("event_limit"),
        "calibration_scope": plan.get("calibration_scope"),
        "smoke_calibrators": plan.get("smoke_calibrators", 0),
        "context_plan_sha256": plan["context_plan_sha256"],
        "context_ids_sha256": plan["context_ids_sha256"],
        "assigned_context_ids_sha256": sha256_json(expected_ids),
        "assigned_context_count": len(assigned_contexts),
        "observed_context_count": len(observations),
        "complete_assigned_observations": expected_ids == observed_ids and quarantined is None,
        "work_units": {
            **shard_work,
            "candidate_batch_size": candidate_batch_size,
            "batching_admitted": False,
            "batching_status": work_units["batching_status"],
            "batching_reason": work_units["batching_reason"],
        },
        "runtime_identity": dict(runtime_identity or {"status": "not_captured"}),
        "batch_admission": {
            "status": "not_admitted",
            "reason": "exact_history_api_scalar_only",
            "parity_required_before_any_batch_claim": True,
            "parity_status": "not_run",
            "oom_or_parity_policy": "quarantine_shard_no_fallback",
        },
        "contexts": [dict(row) for row in assigned_contexts],
        "observations": [dict(row) for row in observations],
        "thresholds_or_decisions_emitted": False,
        "no_future_or_intervention_leakage": True,
    }
    envelope["envelope_sha256"] = sha256_json(envelope)
    return envelope


def _capture_contexts_for_merge(plan: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    contexts = [dict(row) for row in plan.get("contexts", ())]
    calibration = [row for row in contexts if row.get("kind") == "calibration"]
    candidates = [row for row in contexts if row.get("kind") == "candidate"]
    return calibration, candidates


def _validate_capture_shard(
    shard: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    contract: CheckpointContract,
    h0: H0Inputs,
    shard_index: int,
    num_shards: int,
    candidate_batch_size: int,
) -> list[dict[str, Any]]:
    envelope_hash = shard.get("envelope_sha256")
    if not isinstance(envelope_hash, str) or envelope_hash != sha256_json(
        {key: value for key, value in shard.items() if key != "envelope_sha256"}
    ):
        raise SupportProbeError("capture shard envelope self-hash mismatch")
    _validate_runtime_identity_payload(shard.get("runtime_identity"), contract=contract)
    if shard.get("schema_version") != CAPTURE_SCHEMA_VERSION:
        raise SupportProbeError("capture shard schema_version mismatch")
    for key, expected in (
        ("unit_id", UNIT_ID),
        ("checkpoint", contract.checkpoint),
        ("config_fingerprint", contract.config_fingerprint),
        ("source_panel_sha256", contract.source_panel_sha256),
        ("derived_panel_sha256", contract.derived_panel_sha256),
        ("h0_source_sha256", str(h0.source_info["sha256"])),
        ("context_plan_sha256", plan["context_plan_sha256"]),
    ):
        if shard.get(key) != expected:
            raise SupportProbeError(f"capture shard foreign {key}")
    if shard.get("num_shards") != num_shards or shard.get("shard_index") != shard_index:
        raise SupportProbeError("capture shard index/count mismatch")
    if shard.get("event_limit") != plan.get("event_limit"):
        raise SupportProbeError("capture shard event-limit mismatch")
    if shard.get("smoke_calibrators") != plan.get("smoke_calibrators"):
        raise SupportProbeError("capture shard smoke-calibrator scope mismatch")
    if shard.get("status") == "quarantined":
        raise SupportProbeError("capture shard is quarantined and cannot be merged")
    expected = partition_capture_contexts(plan["contexts"], shard_index=shard_index, num_shards=num_shards)
    expected_ids = [str(row["context_id"]) for row in expected]
    observed_contexts = shard.get("contexts")
    if not isinstance(observed_contexts, list) or any(not isinstance(row, Mapping) for row in observed_contexts):
        raise SupportProbeError("capture shard assigned context set mismatch")
    if [str(row.get("context_id")) for row in observed_contexts] != expected_ids:
        raise SupportProbeError("capture shard assigned context set mismatch")
    observations = shard.get("observations")
    if not isinstance(observations, list):
        raise SupportProbeError("capture shard observations are missing")
    expected_work = _capture_work_units(
        plan,
        num_shards=num_shards,
        candidate_batch_size=candidate_batch_size,
    )["per_shard"][shard_index]
    work_units = shard.get("work_units")
    if not isinstance(work_units, Mapping):
        raise SupportProbeError("capture shard work-unit receipt is missing")
    for key in (
        "context_count",
        "scalar_equivalent_forward_count",
        "batched_forward_estimate",
        "batching_admitted",
        "candidate_batch_size",
    ):
        if work_units.get(key) != (candidate_batch_size if key == "candidate_batch_size" else expected_work[key]):
            raise SupportProbeError(f"capture shard work-unit mismatch for {key}")
    if work_units.get("batching_status") != "not_admitted_exact_history_api_scalar_only":
        raise SupportProbeError("capture shard batching admission is not the scalar-only contract")
    observed_ids = [str(row.get("context_id")) for row in observations]
    if len(observed_ids) != len(set(observed_ids)) or set(observed_ids) != set(expected_ids):
        raise SupportProbeError("capture shard observations are incomplete or duplicated")
    for row in observations:
        if not isinstance(row, Mapping):
            raise SupportProbeError("capture observation is not an object")
        if "verified_support" in row or "support_decision" in row:
            raise SupportProbeError("capture shard contains a threshold/decision field")
    planned_by_id = {str(row["context_id"]): row for row in expected}
    for row in observations:
        context_id = str(row["context_id"])
        planned = planned_by_id[context_id]
        for key in ("stable_key", "kind", "image_id", "gt_owner_id", "candidate_gt_owner_id", "category_name"):
            if row.get(key) != planned.get(key):
                raise SupportProbeError(f"capture observation identity mismatch for {context_id}: {key}")
        planned_candidate_ids = sorted(str(item) for item in (planned.get("candidate_ids") or ()))
        observed_candidate_ids = sorted(str(item) for item in (row.get("candidate_ids") or ()))
        if observed_candidate_ids != planned_candidate_ids:
            raise SupportProbeError(f"capture observation identity mismatch for {context_id}: candidate_ids")
        status = row.get("status")
        if status not in {"measured", "indeterminate", "quarantined_oom", "quarantined_parity", "quarantined_capture_error"}:
            raise SupportProbeError(f"capture observation has unknown status for {context_id}")
        count = row.get("candidate_score_count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise SupportProbeError(f"capture observation candidate score count is invalid for {context_id}")
        score_hash = row.get("candidate_scores_sha256")
        features = row.get("support_features")
        if status == "measured":
            if count != len(planned_candidate_ids):
                raise SupportProbeError(f"capture observation candidate score count mismatch for {context_id}")
            candidate_scores = row.get("candidate_scores")
            if not isinstance(candidate_scores, Mapping):
                raise SupportProbeError(f"capture observation candidate scores are missing for {context_id}")
            if {str(key) for key in candidate_scores} != set(planned_candidate_ids):
                raise SupportProbeError(f"capture observation candidate score IDs mismatch for {context_id}")
            normalized_scores: dict[str, float] = {}
            for key, value in candidate_scores.items():
                if not isinstance(key, str):
                    raise SupportProbeError(f"capture observation candidate score ID is not a string for {context_id}")
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise SupportProbeError(f"capture observation candidate score is not numeric for {context_id}.{key}")
                normalized_scores[key] = _finite(value, f"candidate score {context_id}.{key}")
            if not isinstance(score_hash, str):
                raise SupportProbeError(f"capture observation candidate score hash is missing for {context_id}")
            expected_score_hash = sha256_json(normalized_scores)
            if score_hash != expected_score_hash:
                raise SupportProbeError(f"capture observation candidate score hash mismatch for {context_id}")
            if not isinstance(features, Mapping):
                raise SupportProbeError(f"capture observation features are missing for {context_id}")
            for feature_name, feature_value in features.items():
                if isinstance(feature_value, bool) or not isinstance(feature_value, (int, float)):
                    continue
                _finite(feature_value, f"capture feature {context_id}.{feature_name}")
        else:
            if count != 0 or score_hash is not None or features is not None or row.get("candidate_scores") is not None:
                raise SupportProbeError(f"non-measured capture observation carries score data for {context_id}")
    return [dict(row) for row in observations]


def _recompute_observation_features(
    observation: Mapping[str, Any],
    context: Mapping[str, Any],
    *,
    groups: Mapping[tuple[int, str], Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Rebuild owner-local features from the captured raw score payload."""

    context_id = str(context["context_id"])
    candidate_scores = observation.get("candidate_scores")
    if not isinstance(candidate_scores, Mapping):
        raise SupportProbeError(f"capture observation candidate scores are missing for {context_id}")
    planned_ids = {str(item) for item in (context.get("candidate_ids") or ())}
    if {str(key) for key in candidate_scores} != planned_ids:
        raise SupportProbeError(f"capture observation candidate score IDs mismatch for {context_id}")
    scores: dict[str, float] = {}
    for key, value in candidate_scores.items():
        if not isinstance(key, str) or isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SupportProbeError(f"capture observation candidate score is invalid for {context_id}")
        scores[key] = _finite(value, f"candidate score {context_id}.{key}")
    if observation.get("candidate_score_count") != len(scores):
        raise SupportProbeError(f"capture observation candidate score count mismatch for {context_id}")
    expected_hash = sha256_json(scores)
    if observation.get("candidate_scores_sha256") != expected_hash:
        raise SupportProbeError(f"capture observation candidate score hash mismatch for {context_id}")
    group = groups.get((int(context["image_id"]), str(context.get("category_name", "")).lower()), ())
    if {str(candidate["candidate_id"]) for candidate in group} != planned_ids:
        raise SupportProbeError(f"capture observation physical bank mismatch for {context_id}")
    recomputed = support_features(scores, group, owner_id=str(context["gt_owner_id"]))
    captured = observation.get("support_features")
    if not isinstance(captured, Mapping):
        raise SupportProbeError(f"capture observation features are missing for {context_id}")
    if canonical_json_bytes(captured) != canonical_json_bytes(recomputed):
        raise SupportProbeError(f"capture observation support feature integrity mismatch for {context_id}")
    return recomputed


def merge_calibration_observations(
    observations_by_context: Mapping[str, Mapping[str, Any]],
    calibration_contexts: Sequence[Mapping[str, Any]],
    *,
    contract: CheckpointContract,
    h0: H0Inputs,
    physical: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Merge complete raw calibration features and calibrate once on CPU.

    The helper deliberately accepts only measured feature observations for
    contexts eligible for calibration.  Undercovered or invalid native TP
    contexts are recorded as plan-time exclusions; any missing, indeterminate,
    quarantined, or feature-incomplete *eligible* context prevents a threshold
    from being produced, so a partial capture cannot silently become a
    decision-bearing calibration.
    """

    groups: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    for candidate in physical:
        groups.setdefault(
            (int(candidate["image_id"]), str(candidate["normalized_description"]).lower()),
            [],
        ).append(candidate)
    h0_by_owner = {str(row["gt_owner_id"]): row for row in h0.records}
    calibration_rows: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for context in calibration_contexts:
        context_id = str(context["context_id"])
        observation = observations_by_context.get(context_id)
        if context.get("eligible_for_score") is not True:
            excluded.append(
                {
                    "context_id": context_id,
                    "image_id": int(context["image_id"]),
                    "gt_owner_id": str(context["gt_owner_id"]),
                    "reason": context.get("ineligible_reason") or "ineligible_for_calibration",
                    "capture_status": observation.get("status") if observation is not None else "missing",
                }
            )
            continue
        if observation is None:
            raise SupportProbeError(f"calibration context {context_id} is missing")
        if observation.get("status") != "measured" or not isinstance(observation.get("support_features"), Mapping):
            raise SupportProbeError(f"calibration context {context_id} is incomplete/quarantined")
        features = _recompute_observation_features(observation, context, groups=groups)
        if features.get("assessed") is not True or features.get("peak_lift") is None or features.get("local_concentration") is None:
            raise SupportProbeError(
                f"eligible calibration context {context_id} has no finite owner-local features"
            )
        owner_id = str(context["gt_owner_id"])
        raw_source = h0_by_owner.get(owner_id)
        if raw_source is None:
            raise SupportProbeError(f"calibration context {context_id} has no H0 owner")
        raw = dict(raw_source)
        raw["support_features"] = dict(features)
        raw["checkpoint"] = contract.checkpoint
        raw["source_panel_sha256"] = contract.source_panel_sha256
        raw["derived_panel_sha256"] = contract.derived_panel_sha256
        raw["intervention"] = "none"
        calibration_rows.append(raw)
    calibration = calibrate_support(
        calibration_rows,
        checkpoint=contract.checkpoint,
        source_panel_sha256=contract.source_panel_sha256,
        derived_panel_sha256=contract.derived_panel_sha256,
    )
    calibration["excluded_contexts"] = sorted(
        excluded,
        key=lambda row: (int(row["image_id"]), str(row["gt_owner_id"]), str(row["context_id"])),
    )
    calibration["excluded_context_count"] = len(excluded)
    calibration["eligible_context_count"] = len(calibration_rows)
    calibration["calibration_sha256"] = sha256_json(
        {key: value for key, value in calibration.items() if key != "calibration_sha256"}
    )
    return calibration


def merge_capture_envelopes(
    shards: Sequence[Mapping[str, Any]],
    *,
    contract: CheckpointContract,
    h0: H0Inputs,
    panel: PanelInputs,
    physical: Sequence[Mapping[str, Any]],
    accounting: Mapping[str, Any],
    event_limit: int | None,
    candidate_batch_size: int = 16,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """CPU-only completeness-checked merge; thresholds and decisions happen here."""

    plan = _capture_context_plan(
        contract=contract,
        h0=h0,
        panel=panel,
        physical=physical,
        accounting=accounting,
        event_limit=event_limit,
    )
    if not shards:
        raise SupportProbeError("merge requires at least one capture shard")
    num_shards = shards[0].get("num_shards")
    if isinstance(num_shards, bool) or not isinstance(num_shards, int) or num_shards <= 0:
        raise SupportProbeError("capture shards have no valid num_shards")
    if len(shards) != num_shards:
        raise SupportProbeError("capture merge is missing one or more shards")
    if any(shard.get("smoke_calibrators", 0) for shard in shards):
        raise SupportProbeError("smoke-calibrator captures cannot produce a final support ledger")
    by_index: dict[int, list[dict[str, Any]]] = {}
    for shard in shards:
        index = shard.get("shard_index")
        if isinstance(index, bool) or not isinstance(index, int) or index in by_index:
            raise SupportProbeError("capture merge has duplicate/invalid shard index")
        by_index[index] = _validate_capture_shard(
            shard,
            plan=plan,
            contract=contract,
            h0=h0,
            shard_index=index,
            num_shards=num_shards,
            candidate_batch_size=candidate_batch_size,
        )
    if set(by_index) != set(range(num_shards)):
        raise SupportProbeError("capture merge shard indices are incomplete")
    observations = [row for index in range(num_shards) for row in by_index[index]]
    by_context = {str(row["context_id"]): row for row in observations}
    calibration_contexts, candidate_contexts = _capture_contexts_for_merge(plan)
    calibration = merge_calibration_observations(
        by_context,
        calibration_contexts,
        contract=contract,
        h0=h0,
        physical=physical,
    )
    h0_by_owner = {str(row["gt_owner_id"]): row for row in h0.records}
    groups: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    for candidate in physical:
        groups.setdefault(
            (int(candidate["image_id"]), str(candidate["normalized_description"]).lower()),
            [],
        ).append(candidate)
    output_records: list[dict[str, Any]] = []
    indeterminate: list[dict[str, Any]] = []
    for context in candidate_contexts:
        observation = by_context[str(context["context_id"])]
        owner_id = str(context["gt_owner_id"])
        raw = h0_by_owner[owner_id]
        if observation.get("status") != "measured" or not isinstance(observation.get("support_features"), Mapping):
            indeterminate.append(
                {
                    "image_id": int(context["image_id"]),
                    "gt_owner_id": owner_id,
                    "status": "indeterminate",
                    "reason": observation.get("reason") or "capture_observation_unavailable",
                }
            )
            continue
        measured_features = _recompute_observation_features(observation, context, groups=groups)
        decision = _support_decision(measured_features, calibration)
        if decision is None:
            indeterminate.append(
                {
                    "image_id": int(context["image_id"]),
                    "gt_owner_id": owner_id,
                    "status": "indeterminate",
                    "reason": "missing_owner_local_peak_features",
                }
            )
            continue
        item = dict(raw)
        item.update(
            {
                "support_features": dict(measured_features),
                "verified_support": bool(decision),
                "verified_support_claim": True,
                "support_status": "measured",
                "source_panel_sha256": contract.source_panel_sha256,
                "derived_panel_sha256": contract.derived_panel_sha256,
                "config_fingerprint": contract.config_fingerprint,
                "checkpoint": contract.checkpoint,
                "intervention": "none",
                "no_future_or_intervention_leakage": True,
            }
        )
        output_records.append(item)
    envelope = build_support_envelope(
        contract=contract,
        h0=h0,
        records=output_records,
        calibration=calibration,
        panel=panel,
    )
    envelope["capture_merge"] = {
        "schema_version": CAPTURE_SCHEMA_VERSION,
        "num_shards": num_shards,
        "context_plan_sha256": plan["context_plan_sha256"],
        "observed_context_count": len(observations),
        "calibration_context_count": len(calibration_contexts),
        "calibration_excluded_context_count": calibration.get("excluded_context_count", 0),
        "candidate_context_count": len(candidate_contexts),
        "candidate_batch_size": candidate_batch_size,
        "batching_admitted": False,
    }
    receipt = {
        "schema_version": CAPTURE_RECEIPT_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "status": "merged",
        "mode": "merge",
        "checkpoint": contract.checkpoint,
        "config_fingerprint": contract.config_fingerprint,
        "source_panel_sha256": contract.source_panel_sha256,
        "derived_panel_sha256": contract.derived_panel_sha256,
        "context_plan_sha256": plan["context_plan_sha256"],
        "num_shards": num_shards,
        "calibration_observation_count": calibration["observation_count"],
        "calibration_excluded_context_count": calibration.get("excluded_context_count", 0),
        "calibration_excluded_contexts": list(calibration.get("excluded_contexts", ())),
        "record_count": len(output_records),
        "candidate_count": len(candidate_contexts),
        "indeterminate_count": len(indeterminate),
        "indeterminate": indeterminate,
        "batching_admitted": False,
        "teacher_forced_diagnostic_only": True,
        "behavioral_transfer_claim": False,
        "no_future_or_intervention_leakage": True,
    }
    envelope["calibration"] = calibration
    receipt["envelope_sha256"] = sha256_json(envelope)
    return envelope, receipt


def capture_live_shard(
    *,
    contract: CheckpointContract,
    h0: H0Inputs,
    panel: PanelInputs,
    physical: Sequence[Mapping[str, Any]],
    accounting: Mapping[str, Any],
    event_limit: int | None,
    shard_index: int,
    num_shards: int,
    candidate_batch_size: int = 16,
    smoke_calibrators: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Capture one deterministic shard without thresholds or support decisions."""

    plan = _capture_context_plan(
        contract=contract,
        h0=h0,
        panel=panel,
        physical=physical,
        accounting=accounting,
        event_limit=event_limit,
        smoke_calibrators=smoke_calibrators,
    )
    assigned = partition_capture_contexts(plan["contexts"], shard_index=shard_index, num_shards=num_shards)
    groups: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    for candidate in physical:
        groups.setdefault(
            (int(candidate["image_id"]), str(candidate["normalized_description"]).lower()),
            [],
        ).append(candidate)
    h0_by_owner = {str(row["gt_owner_id"]): row for row in h0.records}
    scorer = HFNativeRowScorer(contract, panel, h0)
    try:
        scorer.open()
        runtime_identity = validate_live_runtime_identity(
            scorer._session,
            expected_checkpoint=contract.checkpoint,
            expected_config_fingerprint=contract.config_fingerprint,
        )
        observations = _capture_observations_until_quarantine(
            assigned,
            h0_by_owner=h0_by_owner,
            groups=groups,
            scorer=scorer,
        )
        envelope = _capture_envelope(
            contract=contract,
            h0=h0,
            plan=plan,
            assigned_contexts=assigned,
            observations=observations,
            shard_index=shard_index,
            num_shards=num_shards,
            candidate_batch_size=candidate_batch_size,
            runtime_identity=runtime_identity,
        )
        receipt = {
            "schema_version": CAPTURE_RECEIPT_SCHEMA_VERSION,
            "unit_id": UNIT_ID,
            "status": envelope["status"],
            "mode": "capture",
            "checkpoint": contract.checkpoint,
            "shard_index": shard_index,
            "num_shards": num_shards,
            "context_plan_sha256": plan["context_plan_sha256"],
            "assigned_context_count": len(assigned),
            "observed_context_count": len(observations),
            "complete_assigned_observations": envelope["complete_assigned_observations"],
            "measured_count": sum(row.get("status") == "measured" for row in observations),
            "indeterminate_count": sum(row.get("status") == "indeterminate" for row in observations),
            "quarantined_count": sum(str(row.get("status", "")).startswith("quarantined") for row in observations),
            "quarantine": envelope.get("quarantine"),
            "thresholds_or_decisions_emitted": False,
            "runtime_identity": runtime_identity,
            "work_units": envelope["work_units"],
            "no_future_or_intervention_leakage": True,
        }
        receipt["envelope_sha256"] = sha256_json(envelope)
        return envelope, receipt
    finally:
        scorer.close()


def run_live(
    *,
    contract: CheckpointContract,
    h0: H0Inputs,
    panel: PanelInputs,
    physical: Sequence[Mapping[str, Any]],
    accounting: Mapping[str, Any],
    event_limit: int | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one checkpoint's native support re-establishment."""

    scorer = HFNativeRowScorer(contract, panel, h0)
    try:
        scorer.open()
        all_records: list[dict[str, Any]] = []
        # Calibration consumes every available checkpoint-native TP, as in the
        # prior census.  The event limit only bounds the frozen 32-owner output.
        for raw in h0.records:
            if raw.get("native_tp") is not True or raw.get("strict_complete_row") is not True:
                continue
            image = int(raw["image_id"])
            category = str(raw.get("category_name", raw.get("category", ""))).lower()
            group = _bank_group(physical, image, category)
            accounting_row = accounting.get(str(raw.get("gt_owner_id")), {})
            if not isinstance(accounting_row, Mapping) or accounting_row.get("bank_coverage_status") not in {"full", "adequate_reduced"}:
                continue
            score_map = {str(candidate["candidate_id"]): scorer.score(raw, candidate) for candidate in group}
            features = support_features(score_map, group, owner_id=str(raw["gt_owner_id"]))
            item = dict(raw)
            item["support_features"] = features
            item["source_panel_sha256"] = contract.source_panel_sha256
            item["derived_panel_sha256"] = contract.derived_panel_sha256
            item["checkpoint"] = contract.checkpoint
            item["run_kind"] = "native_h0"
            item["history_complete"] = True
            all_records.append(item)
        calibration = calibrate_support(
            all_records,
            checkpoint=contract.checkpoint,
            source_panel_sha256=contract.source_panel_sha256,
            derived_panel_sha256=contract.derived_panel_sha256,
        )
        output_records: list[dict[str, Any]] = []
        indeterminate: list[dict[str, Any]] = []
        for item in _candidate_records(
            h0=h0,
            panel=panel,
            physical=physical,
            accounting=accounting,
            contract=contract,
            event_limit=event_limit,
        ):
            raw = item
            image = int(raw["image_id"])
            category = str(raw.get("category_name", raw.get("category", ""))).lower()
            group = item["bank_candidates"]
            if raw.get("natural_boundary_valid") is not True:
                indeterminate.append(
                    {
                        "image_id": image,
                        "gt_owner_id": str(raw["gt_owner_id"]),
                        "status": "indeterminate",
                        "reason": "no_valid_native_due_boundary",
                    }
                )
                continue
            if not item["bank_eligible"]:
                indeterminate.append(
                    {
                        "image_id": image,
                        "gt_owner_id": str(raw["gt_owner_id"]),
                        "status": "indeterminate",
                        "reason": "undercovered_owner_bank",
                    }
                )
                continue
            score_map = {str(candidate["candidate_id"]): scorer.score(raw, candidate) for candidate in group}
            raw["support_features"] = support_features(score_map, group, owner_id=str(raw["gt_owner_id"]))
            decision = _support_decision(raw["support_features"], calibration)
            if decision is None:
                indeterminate.append(
                    {
                        "image_id": image,
                        "gt_owner_id": str(raw["gt_owner_id"]),
                        "status": "indeterminate",
                        "reason": "missing_owner_local_peak_features",
                    }
                )
                continue
            raw["verified_support"] = bool(decision)
            raw["verified_support_claim"] = True
            raw["support_status"] = "measured"
            raw["source_panel_sha256"] = contract.source_panel_sha256
            raw["derived_panel_sha256"] = contract.derived_panel_sha256
            raw["checkpoint"] = contract.checkpoint
            raw["config_fingerprint"] = contract.config_fingerprint
            raw["run_kind"] = "native_h0"
            raw["history_complete"] = True
            raw["intervention"] = "none"
            raw["no_future_or_intervention_leakage"] = True
            output_records.append(raw)
        envelope = build_support_envelope(
            contract=contract,
            h0=h0,
            records=output_records,
            calibration=calibration,
            panel=panel,
        )
        envelope["calibration"] = calibration
        receipt = {
            "schema_version": f"{SCHEMA_VERSION}.receipt",
            "unit_id": UNIT_ID,
            "status": "captured",
            "mode": "live",
            "checkpoint": contract.checkpoint,
            "config_fingerprint": contract.config_fingerprint,
            "source_panel_sha256": contract.source_panel_sha256,
            "derived_panel_sha256": contract.derived_panel_sha256,
            "h0_source_sha256": str(h0.source_info["sha256"]),
            "calibration_sha256": calibration["calibration_sha256"],
            "calibration_observation_count": calibration["observation_count"],
            "record_count": len(output_records),
            "candidate_count": len(output_records) + len(indeterminate),
            "indeterminate_count": len(indeterminate),
            "indeterminate": indeterminate,
            "legacy12_record_count": sum(int(row["image_id"]) != IMAGE_2299 for row in output_records),
            "image2299_record_count": sum(int(row["image_id"]) == IMAGE_2299 for row in output_records),
            "teacher_forced_diagnostic_only": True,
            "behavioral_transfer_claim": False,
            "no_future_or_intervention_leakage": True,
            "envelope_sha256": sha256_json(envelope),
        }
        return envelope, receipt
    finally:
        scorer.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", choices=CHECKPOINTS, required=True)
    parser.add_argument("--infer-config", required=True, type=Path)
    parser.add_argument(
        "--resolved-config",
        type=Path,
        help="immutable H0 configs/resolved.json authority (normally read from the H0 ledger)",
    )
    parser.add_argument("--panel", required=True, type=Path)
    parser.add_argument("--derived-panel", required=True, type=Path)
    parser.add_argument("--derived-receipt", required=True, type=Path)
    parser.add_argument("--h0-ledger", required=True, type=Path)
    parser.add_argument("--mode", choices=("contract", "dry-run", "live", "capture", "merge"), default="contract")
    parser.add_argument("--event-limit", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--candidate-batch-size", type=int, default=16)
    parser.add_argument("--smoke-calibrators", type=int, default=0)
    parser.add_argument(
        "--shard-input",
        action="append",
        type=Path,
        help="capture shard envelope; repeat for --mode merge",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--receipt", type=Path)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.event_limit is not None and (isinstance(args.event_limit, bool) or args.event_limit <= 0):
        raise SupportProbeError("event-limit must be a positive integer")
    if isinstance(args.num_shards, bool) or args.num_shards <= 0:
        raise SupportProbeError("num-shards must be a positive integer")
    if isinstance(args.shard_index, bool) or not 0 <= args.shard_index < args.num_shards:
        raise SupportProbeError("shard-index must satisfy 0 <= shard-index < num-shards")
    if isinstance(args.candidate_batch_size, bool) or args.candidate_batch_size <= 0:
        raise SupportProbeError("candidate-batch-size must be a positive integer")
    if isinstance(args.smoke_calibrators, bool) or args.smoke_calibrators < 0:
        raise SupportProbeError("smoke-calibrators must be a non-negative integer")
    if args.smoke_calibrators and args.mode != "capture":
        raise SupportProbeError("smoke-calibrators is only valid for capture mode")
    if args.mode == "merge" and args.smoke_calibrators:
        raise SupportProbeError("merge cannot use smoke-calibrators")
    panel = load_panel_inputs(args.panel, args.derived_panel, args.derived_receipt)
    h0 = load_h0_inputs(args.h0_ledger, panel=panel, checkpoint=args.checkpoint)
    contract = validate_checkpoint_config(
        args.infer_config,
        checkpoint=args.checkpoint,
        panel=panel,
        h0=h0,
        resolved_config_path=args.resolved_config,
    )
    physical, accounting = build_physical_owner_bank(panel)
    if args.mode in {"contract", "dry-run"}:
        receipt = contract_receipt(
            contract=contract,
            h0=h0,
            panel=panel,
            physical=physical,
            accounting=accounting,
            mode=args.mode,
            event_limit=args.event_limit,
            num_shards=args.num_shards,
            candidate_batch_size=args.candidate_batch_size,
        )
        if args.output is not None:
            _write_immutable(args.output, receipt)
        if args.receipt is not None:
            _write_immutable(args.receipt, receipt)
        return receipt
    if args.mode == "capture":
        if args.output is None:
            raise SupportProbeError("capture mode requires --output shard envelope path")
        envelope, receipt = capture_live_shard(
            contract=contract,
            h0=h0,
            panel=panel,
            physical=physical,
            accounting=accounting,
            event_limit=args.event_limit,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            candidate_batch_size=args.candidate_batch_size,
            smoke_calibrators=args.smoke_calibrators,
        )
        _write_immutable(args.output, envelope)
        if args.receipt is not None:
            _write_immutable(args.receipt, receipt)
        return {"envelope": envelope, "receipt": receipt}
    if args.mode == "merge":
        if not args.shard_input:
            raise SupportProbeError("merge mode requires one or more --shard-input paths")
        shards = []
        for path in args.shard_input:
            value, _info = _read_source(path)
            if not isinstance(value, Mapping):
                raise SupportProbeError(f"capture shard is not an envelope: {path}")
            shards.append(value)
        envelope, receipt = merge_capture_envelopes(
            shards,
            contract=contract,
            h0=h0,
            panel=panel,
            physical=physical,
            accounting=accounting,
            event_limit=args.event_limit,
            candidate_batch_size=args.candidate_batch_size,
        )
        if args.output is None:
            raise SupportProbeError("merge mode requires --output support ledger path")
        _write_immutable(args.output, envelope)
        if args.receipt is not None:
            _write_immutable(args.receipt, receipt)
        return {"envelope": envelope, "receipt": receipt}
    envelope, receipt = run_live(
        contract=contract,
        h0=h0,
        panel=panel,
        physical=physical,
        accounting=accounting,
        event_limit=args.event_limit,
    )
    if args.output is None:
        raise SupportProbeError("live mode requires --output support ledger path")
    _write_immutable(args.output, envelope)
    if args.receipt is not None:
        _write_immutable(args.receipt, receipt)
    return {"envelope": envelope, "receipt": receipt}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        result = run(args)
    except SupportProbeError as exc:
        print(f"support-probe: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result.get("receipt", result), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
