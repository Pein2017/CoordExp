#!/usr/bin/env python3
"""Aggregate completed static/dynamic owner-interface event shards.

The runtime runner writes one directory per checkpoint/event shard.  This
module is the CPU-only finalizer for those directories: it proves the frozen
cohort partition and shard provenance, joins the event receipts, and emits
numeric evidence without deciding which H1--H5 explanation is true.  It does
not load a model, inspect tensors, or re-run an intervention.

The runner predates a persisted shard-plan field in ``runtime_identity.json``.
Consequently the plan is an explicit CLI input and ownership is checked against
the deterministic ``cohort.events[index % count]`` partition, never inferred
from directory order.

Example::

    python scripts/research/aggregate_static_dynamic_owner_interface_shards.py \
      --cohort S=/path/cohort-S.json --cohort A=/path/cohort-A.json \
      --shard S:0/4=/runs/S-0 --shard S:1/4=/runs/S-1 \
      --shard S:2/4=/runs/S-2 --shard S:3/4=/runs/S-3 \
      --shard A:0/4=/runs/A-0 --shard A:1/4=/runs/A-1 \
      --shard A:2/4=/runs/A-2 --shard A:3/4=/runs/A-3 \
      --output /runs/summary.json
"""

from __future__ import annotations

import argparse
import copy
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any


UNIT_ID = "2026-08-05-static-dynamic-owner-interface-crossover"
SCHEMA_VERSION = "static_dynamic_owner_interface_shard_aggregation.v1"
EVIDENCE_BUNDLE_SCHEMA_VERSION = "static_dynamic_owner_interface_evidence_bundle.v1"
RAW_EVENT_EVIDENCE_SCHEMA_VERSION = "static_dynamic_owner_interface_raw_event_evidence.v1"
H0_BASELINE_EVIDENCE_SCHEMA_VERSION = "static_dynamic_owner_interface_h0_baseline_evidence.v1"
ELIGIBLE_HOLD_SCHEMA_VERSION = "static_dynamic_owner_interface_eligible_hold_leaf.v1"
RUNTIME_SCHEMA_VERSION = "static_dynamic_owner_interface_experiment.v1"
RUNTIME_ATTESTATION_SCHEMA_VERSION = f"{RUNTIME_SCHEMA_VERSION}.runtime_attestation.v1"
GRADIENT_SCHEMA_VERSION = "static_dynamic_gradient_path_audit.v2"
SHARD_COUNT = 4
CHECKPOINTS = ("S", "A")
STAGES = ("p1", "p2", "p3", "p4")
SPLITS = ("all", "legacy12", "image2299")
COHORT_SCHEMA_VERSION = "static_dynamic_owner_interface_cohort.v1"
COHORT_MANIFEST_SCHEMA_VERSION = f"{COHORT_SCHEMA_VERSION}.manifest"
LEDGER_SCHEMA_VERSION = "static_dynamic_native_h0_owner_ledger.v1"
EXPECTED_ROW_OPENER_TOKEN_ID = 151646
COHORT_MIN_EVENTS = 24
COHORT_MAX_EVENTS = 32

# These identifiers are the declared IDs emitted by
# run_static_dynamic_owner_interface_experiment.py.  Keeping the list here
# makes a missing cell an explicit finalizer error rather than a silently
# smaller denominator.
P1_PROBES = (
    "K00",
    "K01",
    "K10",
    "K11",
    "K12",
    "K13",
    *(f"{arm}_block{block}" for block in (13, 23) for arm in ("R00", "R10", "R11", "R12")),
    "R00_block27",
    "R10_block27",
)
P2_ARMS = ("D00", "D01", "D10", "D11", "D12", "D20", "D21")
P3_CELLS = ("Y00", "Y10", "Y01", "Y11")
P4_OBJECTIVES = (
    "target_b_complete_row_nll",
    "uncovered_b_vs_covered_a_margin_loss",
    "fixed_sum_coupled",
)


class AggregationError(ValueError):
    """Raised when a shard or the finalization contract is not established."""


@dataclass(frozen=True)
class ShardSpec:
    checkpoint: str
    index: int
    count: int
    path: Path

    @property
    def key(self) -> tuple[str, int, int]:
        return self.checkpoint, self.index, self.count


@dataclass(frozen=True)
class EligibleHoldSpec:
    checkpoint: str
    ordinal: int
    path: Path


@dataclass
class ShardData:
    spec: ShardSpec
    identity: dict[str, Any]
    events: list[dict[str, Any]]
    exact_prefix: dict[str, Any]
    intervention: dict[str, Any]
    gradient: dict[str, Any]
    terminal: dict[str, Any]
    input_files: list[dict[str, Any]]
    mapping_receipts: dict[str, dict[str, Any]]
    source_kind: str = "live_all"


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AggregationError(f"value is not canonical JSON: {exc}") from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise AggregationError(f"cannot hash artifact {path}: {exc}") from exc
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(_canonical(value))


def _object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AggregationError(f"{context} must be a JSON object")
    return dict(value)


def _array(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise AggregationError(f"{context} must be a JSON array")
    return list(value)


def _text(value: Any, context: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise AggregationError(f"{context} must be a non-empty string")
    result = str(value)
    if not result:
        raise AggregationError(f"{context} must be a non-empty string")
    return result


def _hash(value: Any, context: str) -> str:
    result = _text(value, context).lower()
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise AggregationError(f"{context} must be a lowercase SHA-256")
    return result


def _finite(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AggregationError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise AggregationError(f"{context} must be a finite number")
    return result


def _reject_nonfinite(value: Any, context: str) -> None:
    """Reject JSON NaN/Infinity even when the stdlib decoder accepted it."""

    if isinstance(value, float) and not math.isfinite(value):
        raise AggregationError(f"{context} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _reject_nonfinite(child, f"{context}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_nonfinite(child, f"{context}[{index}]")


def _read_json(path: Path, context: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AggregationError(f"cannot read {context}: {path}: {exc}") from exc
    result = _object(payload, context)
    _reject_nonfinite(result, context)
    return result


def _read_jsonl(path: Path, context: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise AggregationError(f"cannot read {context}: {path}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            raise AggregationError(f"blank line in {context}: {path}:{line_number}")
        try:
            row = _object(json.loads(line), f"{context}:{line_number}")
            _reject_nonfinite(row, f"{context}:{line_number}")
            rows.append(row)
        except json.JSONDecodeError as exc:
            raise AggregationError(f"invalid JSON in {context}: {path}:{line_number}") from exc
    return rows


def _resolve_path(value: Any, context: str) -> Path:
    try:
        if isinstance(value, os.PathLike):
            candidate = Path(value)
        else:
            candidate = Path(_text(value, context))
        return candidate.expanduser().resolve(strict=True)
    except OSError as exc:
        raise AggregationError(f"{context} path is unavailable: {exc}") from exc


def parse_shard_selector(value: str) -> ShardSpec:
    """Parse ``CHECKPOINT:INDEX/COUNT=PATH`` without accepting ambiguity."""

    if not isinstance(value, str):
        raise AggregationError("--shard must be CHECKPOINT:INDEX/COUNT=PATH")
    left, separator, path_text = value.partition("=")
    if not separator or not path_text:
        raise AggregationError("--shard must be CHECKPOINT:INDEX/COUNT=PATH")
    match = re.fullmatch(r"([SA]):(\d+)/(\d+)", left.strip().upper())
    if match is None:
        raise AggregationError(f"invalid shard selector: {value!r}")
    checkpoint, raw_index, raw_count = match.groups()
    index, count = int(raw_index), int(raw_count)
    if count != SHARD_COUNT or not 0 <= index < count:
        raise AggregationError(
            f"shard selector must use exactly {SHARD_COUNT} shards with 0 <= index < count"
        )
    path = _resolve_path(path_text.strip(), "shard")
    return ShardSpec(checkpoint, index, count, path)


def parse_cohort_selector(value: str) -> tuple[str, Path]:
    """Parse the checkpoint-specific ``CHECKPOINT=PATH`` cohort selector."""

    if not isinstance(value, str):
        raise AggregationError("--cohort must be CHECKPOINT=PATH")
    checkpoint, separator, path_text = value.partition("=")
    checkpoint = checkpoint.strip().upper()
    if not separator or checkpoint not in CHECKPOINTS or not path_text.strip():
        raise AggregationError("--cohort must be CHECKPOINT=PATH for S and A")
    return checkpoint, _resolve_path(path_text.strip(), "cohort")


def parse_eligible_hold_selector(value: str) -> EligibleHoldSpec:
    """Parse ``CHECKPOINT:ONE_BASED_ORDINAL=PATH`` for one sealed HOLD leaf."""

    if not isinstance(value, str):
        raise AggregationError("--eligible-hold-leaf must be CHECKPOINT:ORDINAL=PATH")
    left, separator, path_text = value.partition("=")
    match = re.fullmatch(r"([SA]):([1-9][0-9]*)", left.strip().upper())
    if not separator or match is None or not path_text.strip():
        raise AggregationError("--eligible-hold-leaf must be CHECKPOINT:ORDINAL=PATH")
    checkpoint, raw_ordinal = match.groups()
    return EligibleHoldSpec(
        checkpoint=checkpoint,
        ordinal=int(raw_ordinal),
        path=_resolve_path(path_text.strip(), "eligible HOLD leaf"),
    )


def _cohort_event_identity(event: Mapping[str, Any], index: int) -> tuple[str, str]:
    event_id = event.get("event_id", event.get("gt_owner_id"))
    if event_id is None:
        raise AggregationError(f"cohort.events[{index}] lacks event_id/gt_owner_id")
    image_id = event.get("image_id")
    if image_id is None or isinstance(image_id, bool):
        raise AggregationError(f"cohort.events[{index}] lacks image_id")
    event_id_text = _text(event_id, f"cohort.events[{index}].event_id")
    image_id_text = _text(
        image_id, f"cohort.events[{index}].image_id"
    )
    # The materializer's owner identity is deliberately boring and stable:
    # event IDs are ``gt:<image_id>:<source_panel_object_index>``.  Enforce
    # that shape at the finalizer boundary so a hand-written replacement
    # cannot silently change cohort ownership semantics.
    match = re.fullmatch(r"gt:(\d+):(\d+)", event_id_text)
    if match is None or match.group(1) != image_id_text:
        raise AggregationError(
            f"cohort.events[{index}].event_id must be gt:<image_id>:<source_index>"
        )
    return event_id_text, image_id_text


def _cohort_target_owner(event: Mapping[str, Any], checkpoint: str, fallback: str) -> str:
    """Resolve the checkpoint-specific B owner without inventing a new ID."""

    value: Any = event.get("target_row_owner_id")
    if isinstance(value, Mapping):
        value = value.get(checkpoint)
    if isinstance(value, Mapping):
        for key in ("gt_owner_id", "owner_id", "target_owner_id", "id"):
            if value.get(key) is not None:
                value = value[key]
                break
    pair = event.get("A_B")
    if value is None and isinstance(pair, Mapping):
        checkpoint_pair = pair.get(checkpoint)
        if isinstance(checkpoint_pair, Mapping):
            b_value = checkpoint_pair.get("B_verified_uncovered")
            if isinstance(b_value, Mapping):
                value = b_value.get("gt_owner_id", b_value.get("owner_id", b_value.get("target_owner_id")))
    return fallback if value is None else _text(value, "cohort target owner")


def _cohort_target_boundary(event: Mapping[str, Any], checkpoint: str) -> int | None:
    pair = event.get("A_B")
    if not isinstance(pair, Mapping) or not isinstance(pair.get(checkpoint), Mapping):
        return None
    b_value = pair[checkpoint].get("B_verified_uncovered")
    if not isinstance(b_value, Mapping):
        return None
    value = b_value.get("natural_boundary")
    if isinstance(value, Mapping):
        for key in ("index", "row_index", "boundary_index", "token_index"):
            if key in value:
                value = value[key]
                break
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, int):
        raise AggregationError("cohort B_verified_uncovered.natural_boundary must be an integer")
    if value < 0:
        raise AggregationError("cohort B_verified_uncovered.natural_boundary must be non-negative")
    return value


def _load_cohort(
    path: Path,
) -> tuple[
    dict[str, Any],
    str,
    list[tuple[str, str]],
    dict[str, Any],
    dict[tuple[str, str], dict[str, Any]],
]:
    cohort = _read_json(path, "cohort")
    if cohort.get("unit_id") != UNIT_ID:
        raise AggregationError("cohort unit_id does not match the frozen owner-interface unit")
    if cohort.get("schema_version") != COHORT_SCHEMA_VERSION:
        raise AggregationError("cohort schema_version does not match the frozen cohort contract")
    execution_contract = _object(cohort.get("execution_contract"), "cohort.execution_contract")
    for key in ("h0_execution", "gpu_launch", "val200_index_fallback"):
        if execution_contract.get(key) is not False:
            raise AggregationError(f"cohort.execution_contract.{key} must be false")
    if execution_contract.get("cpu_only") is not True:
        raise AggregationError("cohort.execution_contract.cpu_only must be true")
    retention = _object(cohort.get("retention"), "cohort.retention")
    for key, expected in (("min_events", COHORT_MIN_EVENTS), ("max_events", COHORT_MAX_EVENTS)):
        value = retention.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value != expected:
            raise AggregationError(f"cohort.retention.{key} must be {expected}")
    raw_events = _array(cohort.get("events"), "cohort.events")
    if not COHORT_MIN_EVENTS <= len(raw_events) <= COHORT_MAX_EVENTS:
        raise AggregationError(
            f"cohort.events count must be between {COHORT_MIN_EVENTS} and {COHORT_MAX_EVENTS}"
        )
    if retention.get("retained_events") != len(raw_events):
        raise AggregationError("cohort.retention.retained_events disagrees with cohort.events")
    if retention.get("status") != "within_bounds":
        raise AggregationError("cohort.retention.status must be within_bounds")
    expected: list[tuple[str, str]] = []
    seen: set[str] = set()
    for index, raw_event in enumerate(raw_events):
        event = _object(raw_event, f"cohort.events[{index}]")
        event_id, image_id = _cohort_event_identity(event, index)
        if event_id in seen:
            raise AggregationError(f"cohort.events repeats event_id {event_id}")
        seen.add(event_id)
        expected.append((event_id, image_id))
        disposition = event.get("disposition")
        if disposition not in {"established", "indeterminate"}:
            raise AggregationError(f"cohort.events[{index}].disposition is not recognized")
    if not expected:
        raise AggregationError("cohort.events must be non-empty")
    images = {image_id for _, image_id in expected}
    if "2299" not in images or len(images) < 2:
        raise AggregationError("cohort must include image2299 and at least one legacy image")
    frozen_pool = _object(cohort.get("frozen_pool"), "cohort.frozen_pool")
    required_images = frozen_pool.get("required_images")
    if required_images is not None:
        declared_images = {
            _text(value, "cohort.frozen_pool.required_images[]") for value in _array(required_images, "cohort.frozen_pool.required_images")
        }
        if declared_images != images:
            raise AggregationError("cohort frozen_pool.required_images disagrees with retained event images")
    subsets = _object(cohort.get("subsets"), "cohort.subsets")
    if set(subsets) != {"legacy12", "image2299"}:
        raise AggregationError("cohort.subsets must contain legacy12 and image2299")
    expected_subset_counts = {
        "legacy12": sum(image_id != "2299" for _, image_id in expected),
        "image2299": sum(image_id == "2299" for _, image_id in expected),
    }
    for subset_name, expected_count in expected_subset_counts.items():
        subset = _object(subsets.get(subset_name), f"cohort.subsets.{subset_name}")
        if subset.get("event_count") != expected_count:
            raise AggregationError(
                f"cohort.subsets.{subset_name}.event_count disagrees with retained events"
            )
    sources = _object(cohort.get("sources"), "cohort.sources")
    source_hashes: dict[str, Any] = {}
    for name in ("derived_panel", "source_panel"):
        source = _object(sources.get(name), f"cohort.sources.{name}")
        source_hashes[name] = _hash(source.get("sha256"), f"cohort.sources.{name}.sha256")
        source_path = _resolve_path(source.get("path"), f"cohort.sources.{name}.path")
        source_hashes[f"{name}_path"] = str(source_path)
        if sha256_file(source_path) != source_hashes[name]:
            raise AggregationError(f"cohort {name} bytes do not match declared SHA-256")

    # The first shard finalizer predates the provenance sidecars emitted by
    # the cohort materializer.  Keep them optional for legacy callers, but
    # when present bind both the descriptor bytes and the JSON identity to the
    # admitted source/derived panel.  ``build_evidence_bundle`` requires the
    # sidecars; ``aggregate_shards`` remains backward-compatible with the
    # original minimal fixture contract.
    if "derived_receipt" in sources:
        receipt_source = _object(sources.get("derived_receipt"), "cohort.sources.derived_receipt")
        receipt_path = _resolve_path(receipt_source.get("path"), "cohort.sources.derived_receipt.path")
        receipt_hash = _hash(receipt_source.get("sha256"), "cohort.sources.derived_receipt.sha256")
        if sha256_file(receipt_path) != receipt_hash:
            raise AggregationError("cohort derived_receipt bytes do not match declared SHA-256")
        receipt = _read_json(receipt_path, "cohort.sources.derived_receipt")
        required_receipt_fields = {
            "schema_version",
            "unit_id",
            "receipt_path",
            "derived_sha256",
            "source_sha256",
            "derived_path",
            "source_path",
            "row_count",
            "images_manifest",
            "images_manifest_sha256",
        }
        missing_receipt_fields = sorted(required_receipt_fields - set(receipt))
        if missing_receipt_fields:
            raise AggregationError(
                "cohort derived_receipt is missing binding field(s): "
                + ",".join(missing_receipt_fields)
            )
        if receipt.get("schema_version") not in {1, "1"} or receipt.get("unit_id") != UNIT_ID:
            raise AggregationError("cohort derived_receipt schema/unit identity is invalid")
        if _resolve_path(receipt.get("receipt_path"), "cohort derived_receipt.receipt_path") != receipt_path:
            raise AggregationError("cohort derived_receipt.receipt_path does not bind its own bytes")
        for key, expected_value in (
            ("derived_sha256", source_hashes["derived_panel"]),
            ("source_sha256", source_hashes["source_panel"]),
        ):
            declared = _hash(receipt.get(key), f"cohort derived_receipt.{key}")
            if declared != expected_value:
                raise AggregationError(f"cohort derived_receipt.{key} disagrees with admitted panel")
        for key, expected_path in (("derived_path", source_hashes.get("derived_panel_path")), ("source_path", source_hashes.get("source_panel_path"))):
            declared_path = _resolve_path(receipt.get(key), f"cohort derived_receipt.{key}")
            if expected_path is None or declared_path != Path(str(expected_path)).expanduser().resolve():
                raise AggregationError(f"cohort derived_receipt.{key} disagrees with admitted panel path")
        row_count = receipt.get("row_count")
        images_manifest = _array(receipt.get("images_manifest"), "cohort derived_receipt.images_manifest")
        if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count != 13 or len(images_manifest) != 13:
            raise AggregationError("cohort derived_receipt must bind exactly 13 admitted source-panel images")
        image_ids: list[str] = []
        for index, raw_image in enumerate(images_manifest):
            image = _object(raw_image, f"cohort derived_receipt.images_manifest[{index}]")
            image_ids.append(_text(image.get("image_id"), f"cohort derived_receipt.images_manifest[{index}].image_id"))
        if len(set(image_ids)) != 13:
            raise AggregationError("cohort derived_receipt image manifest contains duplicate image IDs")
        if _hash(receipt.get("images_manifest_sha256"), "cohort derived_receipt.images_manifest_sha256") != sha256_json(images_manifest):
            raise AggregationError("cohort derived_receipt image manifest hash mismatch")
        source_hashes["derived_receipt"] = {"path": str(receipt_path), "sha256": receipt_hash}
        source_hashes["derived_receipt_identity"] = {
            "schema_version": receipt["schema_version"],
            "unit_id": receipt["unit_id"],
            "source_panel_image_ids": sorted(image_ids, key=lambda value: int(value) if value.isdigit() else value),
            "images_manifest_sha256": receipt["images_manifest_sha256"],
        }
        source_hashes["source_panel_image_ids"] = sorted(
            image_ids,
            key=lambda value: int(value) if value.isdigit() else value,
        )
    ledger_sources = _array(sources.get("h0_ledgers"), "cohort.sources.h0_ledgers")
    if not ledger_sources:
        raise AggregationError("cohort.sources.h0_ledgers must be non-empty")
    ledger_hashes: dict[str, str] = {}
    ledger_fingerprints: dict[str, str] = {}
    prefix_bindings: dict[tuple[str, str], dict[str, Any]] = {}
    for index, raw_source in enumerate(ledger_sources):
        source = _object(raw_source, f"cohort.sources.h0_ledgers[{index}]")
        ledger_path = _resolve_path(source.get("path"), f"cohort.sources.h0_ledgers[{index}].path")
        ledger_hash = _hash(source.get("sha256"), f"cohort.sources.h0_ledgers[{index}].sha256")
        if sha256_file(ledger_path) != ledger_hash:
            raise AggregationError(f"cohort H0 ledger bytes do not match declared SHA-256: {ledger_path}")
        ledger_hashes[str(ledger_path)] = ledger_hash
        ledger = _read_json(ledger_path, f"cohort.sources.h0_ledgers[{index}]")
        if ledger.get("unit_id") != UNIT_ID:
            raise AggregationError(f"H0 ledger {ledger_path} unit_id mismatch")
        if ledger.get("schema_version") != LEDGER_SCHEMA_VERSION:
            raise AggregationError(f"H0 ledger {ledger_path} schema_version mismatch")
        ledger_checkpoint = str(ledger.get("checkpoint", "")).upper()
        if ledger_checkpoint not in CHECKPOINTS:
            raise AggregationError(f"H0 ledger {ledger_path} checkpoint is missing or invalid")
        if ledger.get("run_kind") != "native_h0" or ledger.get("history_complete") is not True:
            raise AggregationError(f"H0 ledger {ledger_path} is not a complete native H0 ledger")
        if ledger.get("source_panel_sha256") != source_hashes["source_panel"] or ledger.get("derived_panel_sha256") != source_hashes["derived_panel"]:
            raise AggregationError(f"H0 ledger {ledger_path} panel provenance differs from cohort")
        ledger_fingerprints[str(ledger_path)] = _text(ledger.get("config_fingerprint"), f"H0 ledger {ledger_path}.config_fingerprint")
        records = _array(ledger.get("records"), f"H0 ledger {ledger_path}.records")
        if not records:
            raise AggregationError(f"H0 ledger {ledger_path}.records must be non-empty")
        for record_index, raw_record in enumerate(records):
            record = _object(raw_record, f"H0 ledger {ledger_path}.records[{record_index}]")
            event_id = record.get("gt_owner_id", record.get("event_id"))
            image_id = record.get("image_id")
            if event_id is None or image_id is None:
                continue
            key = (_text(event_id, "H0 ledger event_id"), _text(image_id, "H0 ledger image_id"))
            natural_boundary = record.get("natural_boundary")
            if natural_boundary is not None and (
                isinstance(natural_boundary, bool)
                or not isinstance(natural_boundary, int)
                or natural_boundary < 0
            ):
                raise AggregationError(f"H0 ledger {ledger_path} record {event_id} has invalid natural_boundary")
            exact_prefix_hash = record.get("exact_prefix_sha256")
            if exact_prefix_hash is not None:
                _hash(exact_prefix_hash, f"H0 ledger {ledger_path} record {event_id}.exact_prefix_sha256")
            exact_prefix_token_ids = record.get("exact_prefix_token_ids")
            if exact_prefix_token_ids is not None:
                exact_prefix_token_ids = _array(
                    exact_prefix_token_ids,
                    f"H0 ledger {ledger_path} record {event_id}.exact_prefix_token_ids",
                )
                if any(
                    isinstance(token, bool) or not isinstance(token, int) or token < 0
                    for token in exact_prefix_token_ids
                ):
                    raise AggregationError(f"H0 ledger {ledger_path} record {event_id} prefix token IDs are malformed")
            if exact_prefix_hash is not None and sha256_json(exact_prefix_token_ids) != exact_prefix_hash:
                raise AggregationError(f"H0 ledger {ledger_path} record {event_id} prefix hash disagrees with token IDs")
            raw_covered = record.get("covered_owner_ids", [])
            covered_owner_ids = _array(raw_covered, f"H0 ledger {ledger_path} record {event_id}.covered_owner_ids")
            if any(not isinstance(owner, str) or not owner for owner in covered_owner_ids):
                raise AggregationError(f"H0 ledger {ledger_path} record {event_id} covered owner IDs are malformed")
            if len(set(covered_owner_ids)) != len(covered_owner_ids):
                raise AggregationError(f"H0 ledger {ledger_path} record {event_id} covered owner IDs repeat")
            metadata = {
                "checkpoint": ledger_checkpoint,
                "natural_boundary": natural_boundary,
                "exact_prefix_sha256": exact_prefix_hash,
                "exact_prefix_token_count": None if exact_prefix_token_ids is None else len(exact_prefix_token_ids),
                "exact_prefix_token_ids": exact_prefix_token_ids,
                "ledger_source_path": str(ledger_path),
                "ledger_record_index": record_index,
                "covered_owner_ids": covered_owner_ids,
            }
            prior = prefix_bindings.get(key)
            if prior is not None and prior != metadata:
                raise AggregationError(f"H0 ledgers disagree for {key[0]}/{key[1]}")
            prefix_bindings[key] = metadata
    support_sources = _array(sources.get("support_ledgers", []), "cohort.sources.support_ledgers")
    support_hashes: dict[str, str] = {}
    support_metadata: list[dict[str, Any]] = []
    for index, raw_source in enumerate(support_sources):
        source = _object(raw_source, f"cohort.sources.support_ledgers[{index}]")
        support_path = _resolve_path(source.get("path"), f"cohort.sources.support_ledgers[{index}].path")
        support_hash = _hash(source.get("sha256"), f"cohort.sources.support_ledgers[{index}].sha256")
        if sha256_file(support_path) != support_hash:
            raise AggregationError(f"cohort support ledger bytes do not match declared SHA-256: {support_path}")
        support = _read_json(support_path, f"cohort.sources.support_ledgers[{index}]")
        if support.get("unit_id") not in (None, UNIT_ID):
            raise AggregationError(f"cohort support ledger {support_path} unit_id mismatch")
        support_checkpoint = support.get("checkpoint")
        if support_checkpoint is not None and str(support_checkpoint).upper() not in CHECKPOINTS:
            raise AggregationError(f"cohort support ledger {support_path} checkpoint is invalid")
        for key, expected_value in (
            ("source_panel_sha256", source_hashes["source_panel"]),
            ("derived_panel_sha256", source_hashes["derived_panel"]),
        ):
            declared = support.get(key)
            if declared is not None and declared != expected_value:
                raise AggregationError(f"cohort support ledger {support_path} {key} disagrees with panel")
        declared_h0 = support.get("h0_source_sha256")
        if declared_h0 is not None and declared_h0 not in set(ledger_hashes.values()):
            raise AggregationError(f"cohort support ledger {support_path} h0_source_sha256 is not an admitted H0 ledger")
        support_hashes[str(support_path)] = support_hash
        support_metadata.append({"path": str(support_path), "sha256": support_hash, "checkpoint": support_checkpoint})
    for event_index, raw_event in enumerate(raw_events):
        event = _object(raw_event, f"cohort.events[{event_index}]")
        event_id, image_id = expected[event_index]
        binding = prefix_bindings.get((event_id, image_id))
        if binding is None:
            raise AggregationError(f"cohort event {event_id} has no checkpoint-native H0 ledger record")
        target_owner = _cohort_target_owner(event, binding["checkpoint"], event_id)
        binding["target_owner_id"] = target_owner
        if target_owner != event_id:
            # The target can be a checkpoint-specific replacement, but it must
            # itself be represented by the same H0 ledger source.
            target_binding = prefix_bindings.get((target_owner, image_id))
            if target_binding is None:
                raise AggregationError(f"cohort target owner {target_owner} has no checkpoint-native H0 ledger record")
            if target_binding["checkpoint"] != binding["checkpoint"]:
                raise AggregationError(f"cohort target owner {target_owner} H0 ledger checkpoint mismatch")
            binding = target_binding
            prefix_bindings[(event_id, image_id)] = binding
        cohort_boundary = _cohort_target_boundary(event, binding["checkpoint"])
        if cohort_boundary is not None and binding.get("natural_boundary") != cohort_boundary:
            raise AggregationError(f"cohort event {event_id} target boundary differs from H0 ledger")
        status_map = event.get("checkpoint_status")
        if isinstance(status_map, Mapping):
            status = status_map.get(binding.get("checkpoint"))
            if isinstance(status, Mapping) and status.get("natural_boundary") is not None and binding.get("natural_boundary") is not None:
                if status.get("natural_boundary") != binding.get("natural_boundary"):
                    raise AggregationError(f"cohort event {event_id} natural boundary differs from H0 ledger")
    source_hashes["h0_ledgers"] = ledger_hashes
    source_hashes["h0_ledger_config_fingerprints"] = ledger_fingerprints
    source_hashes["support_ledgers"] = support_hashes
    source_hashes["support_ledger_metadata"] = support_metadata
    source_hashes["derived_panel_path"] = str(_resolve_path(sources["derived_panel"].get("path"), "cohort.sources.derived_panel.path"))
    source_hashes["source_panel_path"] = str(_resolve_path(sources["source_panel"].get("path"), "cohort.sources.source_panel.path"))
    manifest_path = path.with_name(path.name.replace(".json", ".manifest.json"))
    if not manifest_path.is_file():
        raise AggregationError(f"cohort source manifest is missing: {manifest_path}")
    manifest = _read_json(manifest_path, "cohort manifest")
    if manifest.get("schema_version") != COHORT_MANIFEST_SCHEMA_VERSION or manifest.get("unit_id") != UNIT_ID:
        raise AggregationError("cohort manifest schema or unit_id mismatch")
    if manifest.get("cohort_sha256") != sha256_file(path):
        raise AggregationError("cohort manifest cohort_sha256 disagrees with cohort bytes")
    if manifest.get("cohort_content_sha256") != sha256_json(cohort):
        raise AggregationError("cohort manifest cohort_content_sha256 disagrees with cohort")
    if manifest.get("event_count") != len(expected):
        raise AggregationError("cohort manifest event_count disagrees with cohort")
    if manifest.get("legacy12_event_count") != expected_subset_counts["legacy12"] or manifest.get("image2299_event_count") != expected_subset_counts["image2299"]:
        raise AggregationError("cohort manifest subset counts disagree with cohort")
    manifest_sources = _object(manifest.get("source_hashes"), "cohort manifest.source_hashes")
    if manifest_sources.get("derived_panel") != source_hashes["derived_panel"] or manifest_sources.get("source_panel") != source_hashes["source_panel"] or manifest_sources.get("h0_ledgers") != list(ledger_hashes.values()):
        raise AggregationError("cohort manifest source hashes disagree with cohort")
    if "derived_receipt" in source_hashes and manifest_sources.get("derived_receipt") != source_hashes["derived_receipt"]["sha256"]:
        raise AggregationError("cohort manifest derived_receipt hash disagrees with cohort")
    if source_hashes["support_ledgers"] and manifest_sources.get("support_ledgers") != list(source_hashes["support_ledgers"].values()):
        raise AggregationError("cohort manifest support ledger hashes disagree with cohort")
    return cohort, sha256_file(path), expected, source_hashes, prefix_bindings


def _assert_identity_match(value: Mapping[str, Any], identity: Mapping[str, Any], context: str) -> None:
    for key in (
        "schema_version",
        "unit_id",
        "checkpoint",
        "stage",
        "config_sha256",
        "resolved_config_fingerprint",
        "panel_sha256",
        "cohort_sha256",
    ):
        if key in value and value.get(key) != identity.get(key):
            raise AggregationError(f"{context}.{key} disagrees with runtime identity")


def _normalize_attested_cuda_device(value: Any, context: str) -> str:
    if not isinstance(value, str):
        raise AggregationError(f"{context} must be a CUDA device string")
    normalized = value.strip().lower()
    if normalized == "cuda":
        return "cuda:0"
    if re.fullmatch(r"cuda:[0-9]+", normalized) is None:
        raise AggregationError(f"{context} is not a supported CUDA device: {value!r}")
    return f"cuda:{int(normalized.rsplit(':', 1)[-1])}"


def _validate_runtime_attestation(attestation: Any, *, context: str) -> None:
    if not isinstance(attestation, Mapping):
        raise AggregationError(f"{context} must be an object")
    required = {
        "schema_version",
        "status",
        "passed",
        "device",
        "effective_device",
        "normalized_device",
        "logical_selected_device",
        "model_device",
        "torch_current_device",
        "first_parameter_device",
        "cuda_visible_devices",
        "physical_device_id",
        "physical_device_index",
        "physical_device_uuid",
        "physical_device_uuid_normalized",
        "torch_device_uuid_raw",
        "pid",
        "timestamp_utc",
        "support_runtime_identity",
    }
    missing = sorted(required - set(attestation))
    if missing:
        raise AggregationError(f"{context} is missing field(s): {','.join(missing)}")
    if attestation.get("schema_version") != RUNTIME_ATTESTATION_SCHEMA_VERSION or attestation.get("status") != "validated":
        raise AggregationError(f"{context} is not validated")
    if attestation.get("passed") is not True:
        raise AggregationError(f"{context}.passed must be true")
    visible = attestation.get("cuda_visible_devices")
    if not isinstance(visible, Mapping):
        raise AggregationError(f"{context}.cuda_visible_devices must be an object")
    raw = visible.get("raw")
    tokens = visible.get("tokens")
    selected = visible.get("selected_physical_device")
    if not isinstance(raw, str) or not isinstance(tokens, list) or len(tokens) != 1 or not isinstance(tokens[0], str):
        raise AggregationError(f"{context}.cuda_visible_devices must expose exactly one token")
    token = tokens[0].strip()
    if [part.strip() for part in raw.split(",")] != tokens or re.fullmatch(r"[0-9]+", token) is None or selected != token:
        raise AggregationError(f"{context}.cuda_visible_devices mapping is inconsistent")
    if attestation.get("physical_device_id") != token:
        raise AggregationError(f"{context}.physical_device_id disagrees with CUDA_VISIBLE_DEVICES")
    if attestation.get("physical_device_index") != int(token):
        raise AggregationError(f"{context}.physical_device_index disagrees with CUDA_VISIBLE_DEVICES")
    if not isinstance(attestation.get("physical_device_uuid"), str) or re.fullmatch(
        r"GPU-[0-9A-Fa-f-]+", attestation["physical_device_uuid"]
    ) is None:
        raise AggregationError(f"{context}.physical_device_uuid is missing or malformed")
    physical_uuid = attestation["physical_device_uuid"]
    physical_uuid_normalized = str(physical_uuid).lower().removeprefix("gpu-")
    torch_uuid_raw = attestation.get("torch_device_uuid_raw")
    torch_uuid_normalized = str(torch_uuid_raw).lower().removeprefix("gpu-")
    if attestation.get("physical_device_uuid_normalized") != physical_uuid_normalized:
        raise AggregationError(f"{context}.physical_device_uuid_normalized disagrees with raw UUID")
    if re.fullmatch(r"[0-9a-f-]+", torch_uuid_normalized) is None or torch_uuid_normalized != physical_uuid_normalized:
        raise AggregationError(f"{context}.torch_device_uuid_raw disagrees with physical UUID")
    for key in ("device", "effective_device", "normalized_device", "logical_selected_device", "model_device", "torch_current_device", "first_parameter_device"):
        _normalize_attested_cuda_device(attestation.get(key), f"{context}.{key}")
    logical = _normalize_attested_cuda_device(attestation["normalized_device"], f"{context}.normalized_device")
    for key in ("device", "effective_device", "logical_selected_device", "model_device", "torch_current_device", "first_parameter_device"):
        if _normalize_attested_cuda_device(attestation[key], f"{context}.{key}") != logical:
            raise AggregationError(f"{context}.{key} disagrees with normalized device")
    if isinstance(attestation.get("pid"), bool) or not isinstance(attestation.get("pid"), int) or attestation["pid"] <= 0:
        raise AggregationError(f"{context}.pid must be a positive integer")
    timestamp = attestation.get("timestamp_utc")
    if not isinstance(timestamp, str):
        raise AggregationError(f"{context}.timestamp_utc must be an ISO-8601 UTC timestamp")
    try:
        parsed_timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AggregationError(f"{context}.timestamp_utc must be an ISO-8601 UTC timestamp") from exc
    if parsed_timestamp.tzinfo is None or parsed_timestamp.utcoffset() != timezone.utc.utcoffset(parsed_timestamp):
        raise AggregationError(f"{context}.timestamp_utc must be timezone UTC")
    support_runtime = attestation.get("support_runtime_identity")
    if not isinstance(support_runtime, Mapping) or support_runtime.get("status") != "validated" or support_runtime.get("passed") is not True:
        raise AggregationError(f"{context}.support_runtime_identity is not validated")
    for key in ("device", "effective_device", "normalized_device", "torch_current_device", "cuda_visible_devices", "physical_device_id"):
        if support_runtime.get(key) != attestation.get(key):
            raise AggregationError(f"{context}.support_runtime_identity.{key} disagrees with attestation")


def _validate_not_applicable_attestation(attestation: Any, *, context: str) -> None:
    value = _object(attestation, context)
    expected = {
        "schema_version": RUNTIME_ATTESTATION_SCHEMA_VERSION,
        "status": "not_applicable",
        "passed": False,
        "reason": "ineligible_contract_materialization_cpu_only",
    }
    if value != expected:
        raise AggregationError(
            f"{context} must be the exact ineligible_contract_materialization_cpu_only not_applicable attestation"
        )


def _runtime_source_kind(identity: Mapping[str, Any], *, path: Path) -> str:
    execution_mode = identity.get("execution_mode")
    stage = identity.get("stage")
    if execution_mode == "ineligible_contract_materialization":
        if stage != "all":
            raise AggregationError(f"{path}: CPU ineligible materialization requires stage=all")
        return "cpu_ineligible"
    if execution_mode not in {None, "live_intervention"}:
        raise AggregationError(f"{path}: unsupported execution_mode {execution_mode!r}")
    if stage == "all":
        return "live_all"
    if stage == "p4":
        return "live_p4_overlay"
    raise AggregationError(f"{path}: final aggregation requires stage=all or a P4-only repair")


def _validate_runtime_identity(
    identity: dict[str, Any],
    spec: ShardSpec,
    cohort_path: Path,
    cohort_sha256: str,
    source_hashes: Mapping[str, Any],
) -> str:
    if identity.get("schema_version") != RUNTIME_SCHEMA_VERSION:
        raise AggregationError(f"{spec.path}: unsupported runtime identity schema")
    if identity.get("unit_id") != UNIT_ID:
        raise AggregationError(f"{spec.path}: runtime identity unit_id mismatch")
    if identity.get("checkpoint") != spec.checkpoint:
        raise AggregationError(f"{spec.path}: checkpoint differs from CLI selector")
    source_kind = _runtime_source_kind(identity, path=spec.path)
    if source_kind == "cpu_ineligible":
        _validate_not_applicable_attestation(
            identity.get("runtime_attestation"),
            context=f"{spec.path}.runtime_attestation",
        )
    else:
        _validate_runtime_attestation(identity.get("runtime_attestation"), context=f"{spec.path}.runtime_attestation")
    _hash(identity.get("config_sha256"), f"{spec.path}.config_sha256")
    if not _text(identity.get("resolved_config_fingerprint"), f"{spec.path}.resolved_config_fingerprint"):
        raise AggregationError(f"{spec.path}: resolved config fingerprint is missing")
    config_path = _resolve_path(identity.get("config_path"), f"{spec.path}.config_path")
    if sha256_file(config_path) != identity.get("config_sha256"):
        raise AggregationError(f"{spec.path}: config bytes do not match config_sha256")
    panel_path = _resolve_path(identity.get("panel_path"), f"{spec.path}.panel_path")
    _hash(identity.get("panel_sha256"), f"{spec.path}.panel_sha256")
    _text(identity.get("cohort_path"), f"{spec.path}.cohort_path")
    if Path(str(identity["cohort_path"])).expanduser().resolve() != cohort_path:
        raise AggregationError(f"{spec.path}: cohort path differs from --cohort")
    if identity.get("cohort_sha256") != cohort_sha256:
        raise AggregationError(f"{spec.path}: cohort SHA-256 differs from --cohort")
    panel_identity = _object(identity.get("panel_identity"), f"{spec.path}.panel_identity")
    if panel_identity.get("derived_panel_sha256") != source_hashes["derived_panel"]:
        raise AggregationError(f"{spec.path}: derived panel provenance mismatch")
    if panel_identity.get("source_panel_sha256") != source_hashes["source_panel"]:
        raise AggregationError(f"{spec.path}: source panel provenance mismatch")
    if identity.get("panel_sha256") != source_hashes["derived_panel"]:
        raise AggregationError(f"{spec.path}: panel_sha256 is not the derived panel hash")
    derived_source = _resolve_path(
        _object(_object(_read_json(cohort_path, "cohort").get("sources"), "cohort.sources").get("derived_panel"), "cohort.sources.derived_panel").get("path"),
        "cohort.sources.derived_panel.path",
    )
    if panel_path != derived_source or sha256_file(panel_path) != identity.get("panel_sha256"):
        raise AggregationError(f"{spec.path}: panel path/bytes differ from the admitted derived panel")
    manifest_path = _resolve_path(
        panel_identity.get("cohort_manifest_path"),
        f"{spec.path}.panel_identity.cohort_manifest_path",
    )
    expected_manifest = cohort_path.with_name(cohort_path.name.replace(".json", ".manifest.json"))
    if manifest_path != expected_manifest:
        raise AggregationError(f"{spec.path}: cohort manifest path differs from --cohort sidecar")
    manifest_hash = _hash(
        panel_identity.get("cohort_manifest_sha256"),
        f"{spec.path}.panel_identity.cohort_manifest_sha256",
    )
    if sha256_file(manifest_path) != manifest_hash:
        raise AggregationError(f"{spec.path}: cohort manifest bytes do not match declared hash")
    manifest = _read_json(manifest_path, f"{spec.path}.cohort_manifest")
    if manifest.get("schema_version") != COHORT_MANIFEST_SCHEMA_VERSION:
        raise AggregationError(f"{spec.path}: unsupported cohort manifest schema")
    if manifest.get("unit_id") != UNIT_ID:
        raise AggregationError(f"{spec.path}: cohort manifest unit_id mismatch")
    if manifest.get("cohort_sha256") != cohort_sha256:
        raise AggregationError(f"{spec.path}: cohort manifest binds a different cohort hash")
    manifest_sources = _object(manifest.get("source_hashes"), f"{spec.path}.cohort_manifest.source_hashes")
    if manifest_sources.get("derived_panel") != source_hashes["derived_panel"] or manifest_sources.get("source_panel") != source_hashes["source_panel"]:
        raise AggregationError(f"{spec.path}: cohort manifest source hashes differ from cohort")
    manifest_ledgers = _array(manifest_sources.get("h0_ledgers"), f"{spec.path}.cohort_manifest.source_hashes.h0_ledgers")
    expected_ledger_hashes = list(source_hashes.get("h0_ledgers", {}).values())
    if manifest_ledgers != expected_ledger_hashes:
        raise AggregationError(f"{spec.path}: cohort manifest H0 ledger hashes differ from cohort")
    ledger_hashes = _object(panel_identity.get("h0_ledger_sha256"), f"{spec.path}.panel_identity.h0_ledger_sha256")
    if not ledger_hashes:
        raise AggregationError(f"{spec.path}: H0 ledger provenance is empty")
    normalized_identity_ledgers: dict[str, str] = {}
    for ledger_path, declared_hash in ledger_hashes.items():
        resolved_ledger = _resolve_path(ledger_path, f"{spec.path}.panel_identity.h0_ledger_sha256 path")
        normalized_hash = _hash(declared_hash, f"{spec.path}.h0 ledger hash")
        if sha256_file(resolved_ledger) != normalized_hash:
            raise AggregationError(f"{spec.path}: H0 ledger bytes do not match declared hash")
        normalized_identity_ledgers[str(resolved_ledger)] = normalized_hash
    if normalized_identity_ledgers != source_hashes.get("h0_ledgers", {}):
        raise AggregationError(f"{spec.path}: runtime H0 ledger identity is not bound to cohort.sources.h0_ledgers")
    if any(
        fingerprint != identity.get("resolved_config_fingerprint")
        for fingerprint in source_hashes.get("h0_ledger_config_fingerprints", {}).values()
    ):
        raise AggregationError(f"{spec.path}: H0 ledger config fingerprint disagrees with runtime identity")
    event_count = identity.get("event_count")
    if isinstance(event_count, bool) or not isinstance(event_count, int) or event_count <= 0:
        raise AggregationError(f"{spec.path}: event_count must be a positive integer")
    h0_map = _object(identity.get("h0"), f"{spec.path}.h0")
    h0_root = _resolve_path(h0_map.get("root"), f"{spec.path}.h0.root")
    if not h0_root.is_dir():
        raise AggregationError(f"{spec.path}: H0 root is not a directory")
    h0_files = {
        "summary.json": "summary_sha256",
        "run_manifest.json": "run_manifest_sha256",
        "configs/resolved.json": "resolved_config_sha256",
    }
    h0_payloads: dict[str, dict[str, Any]] = {}
    for relative, hash_key in h0_files.items():
        artifact = h0_root / relative
        declared = _hash(h0_map.get(hash_key), f"{spec.path}.h0.{hash_key}")
        if not artifact.is_file() or sha256_file(artifact) != declared:
            raise AggregationError(f"{spec.path}: H0 {relative} bytes do not match identity")
        h0_payloads[relative] = _read_json(artifact, f"{spec.path}.h0.{relative}")
    resolved = _object(h0_payloads["configs/resolved.json"].get("resolution"), f"{spec.path}.h0.resolved.resolution")
    resolved_fingerprint = _text(resolved.get("fingerprint"), f"{spec.path}.h0.resolved.fingerprint")
    if resolved_fingerprint != identity.get("resolved_config_fingerprint"):
        raise AggregationError(f"{spec.path}: H0 resolved config fingerprint disagrees with identity")
    run_manifest = h0_payloads["run_manifest.json"]
    summary = h0_payloads["summary.json"]
    if summary.get("terminal_status") != "completed":
        raise AggregationError(f"{spec.path}: H0 summary is not completed")
    for candidate in (summary.get("checkpoint"), summary.get("checkpoint_id")):
        if candidate is not None and str(candidate).upper() not in {spec.checkpoint, f"CHECKPOINT_{spec.checkpoint}"}:
            raise AggregationError(f"{spec.path}: H0 summary checkpoint identity disagrees with shard checkpoint")
    manifest_fingerprints = _object(run_manifest.get("resolved_config_fingerprints"), f"{spec.path}.h0.run_manifest.resolved_config_fingerprints")
    if manifest_fingerprints.get("infer_config") != identity.get("resolved_config_fingerprint"):
        raise AggregationError(f"{spec.path}: H0 run_manifest fingerprint disagrees with identity")
    if run_manifest.get("terminal_status") != "completed":
        raise AggregationError(f"{spec.path}: H0 run_manifest is not completed")
    # Keep checkpoint/model/adapter/embedding identity as one self-consistent
    # chain.  The runner records the paths in runtime_identity and repeats
    # them in run_manifest; no guessed defaults are accepted here.
    model_identity = _object(run_manifest.get("model_identity"), f"{spec.path}.h0.run_manifest.model_identity")
    base_identity = _object(model_identity.get("base"), f"{spec.path}.h0.run_manifest.model_identity.base")
    adapter_identity = _object(run_manifest.get("adapter_identity"), f"{spec.path}.h0.run_manifest.adapter_identity")
    embedding_identity = _object(run_manifest.get("embedding_delta_identity"), f"{spec.path}.h0.run_manifest.embedding_delta_identity")
    manifest_model_path = _text(base_identity.get("path"), f"{spec.path}.h0.model_identity.base.path")
    manifest_adapter_path = _text(adapter_identity.get("adapter_path"), f"{spec.path}.h0.adapter_identity.adapter_path")
    embedding_record = _object(embedding_identity.get("identity"), f"{spec.path}.h0.embedding_delta_identity.identity")
    manifest_embedding_path = _text(embedding_record.get("delta_path"), f"{spec.path}.h0.embedding_delta_path")
    for key, manifest_value in (
        ("model_base_path", manifest_model_path),
        ("adapter_path", manifest_adapter_path),
        ("embedding_delta_path", manifest_embedding_path),
    ):
        if _text(h0_map.get(key), f"{spec.path}.h0.{key}") != manifest_value:
            raise AggregationError(f"{spec.path}: H0 {key} disagrees with run_manifest")
    for candidate in (run_manifest.get("checkpoint"), run_manifest.get("checkpoint_id"), run_manifest.get("model_identity", {}).get("checkpoint")):
        if candidate is not None and str(candidate).upper() not in {spec.checkpoint, f"CHECKPOINT_{spec.checkpoint}"}:
            raise AggregationError(f"{spec.path}: H0 checkpoint identity disagrees with shard checkpoint")
    return source_kind


def _event_identity(event: Mapping[str, Any], context: str) -> tuple[str, str]:
    event_id = _text(event.get("event_id"), f"{context}.event_id")
    image_id = _text(event.get("image_id"), f"{context}.image_id")
    if event.get("checkpoint") is None:
        raise AggregationError(f"{context}.checkpoint is missing")
    return event_id, image_id


def _validate_status_value(
    record: Any,
    *,
    context: str,
    allowed: set[str],
    missing_allowed: bool = False,
) -> None:
    if not isinstance(record, Mapping):
        raise AggregationError(f"{context} must be an object")
    status = record.get("status")
    if status is None:
        if missing_allowed:
            return
        raise AggregationError(f"{context}.status is missing")
    if not isinstance(status, str) or status not in allowed:
        raise AggregationError(f"{context}.status is unknown: {status!r}")


def _validate_event_statuses(event: Mapping[str, Any], context: str) -> None:
    _validate_status_value(
        event.get("p1"),
        context=f"{context}.p1",
        allowed={"attempted", "invalid/uninterpretable", "indeterminate", "not_applicable"},
    )
    _validate_status_value(
        event.get("p2"),
        context=f"{context}.p2",
        allowed={"attempted", "invalid/uninterpretable", "indeterminate", "not_applicable"},
    )
    _validate_status_value(
        event.get("p3"),
        context=f"{context}.p3",
        allowed={"attempted", "invalid/uninterpretable", "indeterminate", "not_applicable"},
    )
    _validate_status_value(
        event.get("p4"),
        context=f"{context}.p4",
        allowed={"valid", "technical_invalid", "invalid/uninterpretable", "invalid"},
    )
    p1 = _object(event.get("p1"), f"{context}.p1")
    p1_arms = _object(p1.get("arms"), f"{context}.p1.arms")
    missing_p1 = sorted(set(P1_PROBES) - set(p1_arms))
    if missing_p1:
        raise AggregationError(f"{context}.p1.arms is missing probe(s): {','.join(missing_p1)}")
    unexpected_p1 = sorted(set(p1_arms) - set(P1_PROBES))
    if unexpected_p1:
        raise AggregationError(f"{context}.p1.arms contains unexpected probe(s): {','.join(unexpected_p1)}")
    for probe, raw in p1_arms.items():
        _validate_status_value(
            raw,
            context=f"{context}.p1.arms.{probe}",
            allowed={"valid", "complete", "not_applicable", "indeterminate", "invalid/uninterpretable", "technical_invalid", "invalid"},
            missing_allowed=True,
        )
    for stage_name, container_key, probe_allowed in (
        ("p2", "arms", {"not_applicable", "indeterminate", "invalid/uninterpretable", "technical_invalid", "invalid"}),
        ("p3", "cells", {"not_applicable", "indeterminate", "invalid/uninterpretable", "technical_invalid", "invalid"}),
    ):
        stage = _object(event.get(stage_name), f"{context}.{stage_name}")
        container = _object(stage.get(container_key), f"{context}.{stage_name}.{container_key}")
        declared = set(P2_ARMS if stage_name == "p2" else P3_CELLS)
        missing = sorted(declared - set(container))
        if missing:
            raise AggregationError(f"{context}.{stage_name}.{container_key} is missing probe(s): {','.join(missing)}")
        unexpected = sorted(set(container) - declared)
        if unexpected:
            raise AggregationError(f"{context}.{stage_name}.{container_key} contains unexpected probe(s): {','.join(unexpected)}")
        for probe, raw_probe in container.items():
            probe_map = _object(raw_probe, f"{context}.{stage_name}.{probe}")
            _validate_status_value(
                probe_map,
                context=f"{context}.{stage_name}.{probe}",
                allowed=probe_allowed,
                missing_allowed=True,
            )
            if probe_map.get("status") in {"not_applicable", "indeterminate", "invalid/uninterpretable", "technical_invalid", "invalid"}:
                continue
            for horizon in (1, 3):
                _validate_status_value(
                    probe_map.get(f"horizon_{horizon}"),
                    context=f"{context}.{stage_name}.{probe}.horizon_{horizon}",
                    allowed={"completed", "stopped_early", "not_applicable", "indeterminate", "invalid/uninterpretable", "technical_invalid", "invalid"},
                )
    p4 = _object(event.get("p4"), f"{context}.p4")
    if p4.get("status") == "valid":
        objectives = _object(p4.get("objectives"), f"{context}.p4.objectives")
        missing_objectives = sorted(set(P4_OBJECTIVES) - set(objectives))
        if missing_objectives:
            raise AggregationError(f"{context}.p4.objectives is missing objective(s): {','.join(missing_objectives)}")
        path_checks = _object(p4.get("path_checks"), f"{context}.p4.path_checks")
        required_path_checks = {"optimizer_used", "lm_head_only_path", "model_parameter_mutated", "parameter_grad_mutated", "audit_input_mutated", "visual_state_detached"}
        missing_path_checks = sorted(required_path_checks - set(path_checks))
        if missing_path_checks:
            raise AggregationError(f"{context}.p4.path_checks is missing key(s): {','.join(missing_path_checks)}")


def _validate_ineligible_event_contract(
    event: Mapping[str, Any],
    *,
    context: str,
    expected_pair_status: str | None,
) -> None:
    if expected_pair_status is None or expected_pair_status == "verified_pair":
        raise AggregationError(f"{context}: CPU materialization is not cohort-proven ineligible")
    eligibility = _object(event.get("eligibility"), f"{context}.eligibility")
    if eligibility.get("status") != "invalid/uninterpretable":
        raise AggregationError(f"{context}.eligibility.status must be invalid/uninterpretable")
    if eligibility.get("pair_status") != expected_pair_status:
        raise AggregationError(f"{context}.eligibility.pair_status differs from the cohort")
    if eligibility.get("actuators_called") is not False:
        raise AggregationError(f"{context}.eligibility.actuators_called must be false")
    if not isinstance(eligibility.get("reason"), str) or not eligibility["reason"]:
        raise AggregationError(f"{context}.eligibility.reason must be non-empty")
    for stage in STAGES:
        stage_record = _object(event.get(stage), f"{context}.{stage}")
        if stage_record.get("status") != "invalid/uninterpretable":
            raise AggregationError(f"{context}.{stage}.status must be invalid/uninterpretable")
    for stage, container_name, probes in (
        ("p1", "arms", P1_PROBES),
        ("p2", "arms", P2_ARMS),
        ("p3", "cells", P3_CELLS),
    ):
        container = _object(
            _object(event.get(stage), f"{context}.{stage}").get(container_name),
            f"{context}.{stage}.{container_name}",
        )
        if set(container) != set(probes):
            raise AggregationError(f"{context}.{stage}.{container_name} does not declare the exact frozen cells")
        for probe, raw_probe in container.items():
            record = _object(raw_probe, f"{context}.{stage}.{container_name}.{probe}")
            if record.get("status") != "invalid/uninterpretable":
                raise AggregationError(
                    f"{context}.{stage}.{container_name}.{probe}.status must be invalid/uninterpretable"
                )


def _validate_p4_overlay_event(event: Mapping[str, Any], *, context: str) -> None:
    allowed = {
        "event_id",
        "image_id",
        "checkpoint",
        "runtime_attestation",
        "eligibility",
        "prefix",
        "p4",
    }
    unexpected = sorted(set(event) - allowed)
    if unexpected:
        raise AggregationError(f"{context}: P4-only repair attempts silent replacement: {','.join(unexpected)}")
    eligibility = _object(event.get("eligibility"), f"{context}.eligibility")
    if eligibility.get("status") != "eligible" or eligibility.get("pair_status") != "verified_pair":
        raise AggregationError(f"{context}: P4-only repair is not a verified eligible event")
    if eligibility.get("actuators_called") is not True:
        raise AggregationError(f"{context}: P4-only repair must attest actuators_called=true")
    p4 = _object(event.get("p4"), f"{context}.p4")
    if p4.get("status") != "valid":
        raise AggregationError(f"{context}: P4-only repair must contain a valid P4 receipt")
    _validate_status_value(p4, context=f"{context}.p4", allowed={"valid"})
    objectives = _object(p4.get("objectives"), f"{context}.p4.objectives")
    if set(objectives) != set(P4_OBJECTIVES):
        raise AggregationError(f"{context}.p4.objectives does not declare the exact frozen objectives")


def _validate_prefix_receipt(
    prefix: Mapping[str, Any],
    *,
    spec: ShardSpec,
    event_id: str,
    image_id: str,
    expected: Mapping[str, Any] | None,
) -> dict[str, Any]:
    context = f"{spec.path}:event {event_id}.prefix"
    if expected is None:
        raise AggregationError(f"{context} has no H0 ledger binding")
    if any(key in prefix for key in ("token_ids_sha256", "token_count")):
        raise AggregationError(f"{context} contains retired top-level prefix aliases")
    model_input = _object(prefix.get("model_input"), f"{context}.model_input")
    model_hash = _hash(model_input.get("prefix_sha256"), f"{context}.model_input.prefix_sha256")
    model_count = model_input.get("prefix_token_count")
    if isinstance(model_count, bool) or not isinstance(model_count, int) or model_count <= 0:
        raise AggregationError(f"{context}.model_input.prefix_token_count must be positive")
    opener = model_input.get("row_opener_token_id")
    if isinstance(opener, bool) or not isinstance(opener, int) or opener != EXPECTED_ROW_OPENER_TOKEN_ID:
        raise AggregationError(
            f"{context}.model_input.row_opener_token_id must be {EXPECTED_ROW_OPENER_TOKEN_ID}"
        )
    wrapper = model_input.get("wrapper_mode")
    expected_wrapper = "commit" if spec.checkpoint == "A" else "closed"
    if wrapper != expected_wrapper:
        raise AggregationError(f"{context}.model_input.wrapper_mode differs from checkpoint wrapper")
    if "prefix_token_ids" in model_input:
        token_ids = _array(model_input.get("prefix_token_ids"), f"{context}.model_input.prefix_token_ids")
        if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in token_ids):
            raise AggregationError(f"{context}.model_input.prefix_token_ids are malformed")
        if sha256_json(token_ids) != model_hash or len(token_ids) != model_count:
            raise AggregationError(f"{context}.model_input prefix token IDs disagree with hash/count")
    h0 = _object(prefix.get("h0"), f"{context}.h0")
    h0_hash = _hash(h0.get("exact_generated_history_prefix_sha256"), f"{context}.h0.exact_generated_history_prefix_sha256")
    h0_count = h0.get("exact_generated_history_token_count")
    if isinstance(h0_count, bool) or not isinstance(h0_count, int) or h0_count < 0:
        raise AggregationError(f"{context}.h0.exact_generated_history_token_count must be non-negative")
    ledger_path = _resolve_path(h0.get("ledger_source_path"), f"{context}.h0.ledger_source_path")
    ledger_index = h0.get("ledger_record_index")
    if isinstance(ledger_index, bool) or not isinstance(ledger_index, int) or ledger_index < 0:
        raise AggregationError(f"{context}.h0.ledger_record_index must be non-negative")
    history_ids = h0.get("exact_prefix_token_ids")
    if history_ids is not None:
        history_ids = _array(history_ids, f"{context}.h0.exact_prefix_token_ids")
        if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in history_ids):
            raise AggregationError(f"{context}.h0.exact_prefix_token_ids are malformed")
        if sha256_json(history_ids) != h0_hash or len(history_ids) != h0_count:
            raise AggregationError(f"{context}.h0 exact history token IDs disagree with hash/count")
    if h0_hash != expected.get("exact_prefix_sha256"):
        raise AggregationError(f"{context}.h0 exact history hash differs from H0 ledger")
    if h0_count != expected.get("exact_prefix_token_count"):
        raise AggregationError(f"{context}.h0 exact history count differs from H0 ledger")
    if ledger_path != Path(str(expected.get("ledger_source_path"))).expanduser().resolve():
        raise AggregationError(f"{context}.h0 ledger path differs from cohort/H0 provenance")
    if ledger_index != expected.get("ledger_record_index"):
        raise AggregationError(f"{context}.h0 ledger record index differs from cohort/H0 provenance")
    expected_tokens = expected.get("exact_prefix_token_ids")
    if history_ids is not None and expected_tokens is not None and history_ids != expected_tokens:
        raise AggregationError(f"{context}.h0 exact history token IDs differ from H0 ledger")
    target_owner = _text(prefix.get("target_owner_id"), f"{context}.target_owner_id")
    if target_owner != expected.get("target_owner_id"):
        raise AggregationError(f"{context}.target_owner_id differs from cohort target owner")
    boundary = prefix.get("natural_boundary")
    if isinstance(boundary, bool) or not isinstance(boundary, int) or boundary < 0:
        raise AggregationError(f"{context}.natural_boundary must be non-negative")
    if boundary != expected.get("natural_boundary"):
        raise AggregationError(f"{context}.natural_boundary differs from H0 ledger/cohort")
    covered = _array(prefix.get("covered_owner_ids"), f"{context}.covered_owner_ids")
    if any(not isinstance(owner, str) or not owner for owner in covered) or len(set(covered)) != len(covered):
        raise AggregationError(f"{context}.covered_owner_ids are malformed or repeated")
    if covered != expected.get("covered_owner_ids"):
        raise AggregationError(f"{context}.covered_owner_ids differ from H0 ledger")
    if model_count <= h0_count:
        raise AggregationError(f"{context}.model_input prefix count must include prompt and row opener")
    owner_mapping = _object(prefix.get("owner_mapping"), f"{context}.owner_mapping")
    if owner_mapping.get("schema_version") != "owner_interface.source_derived_mapping.v1":
        raise AggregationError(f"{context}.owner_mapping schema_version is unsupported")
    if _text(owner_mapping.get("image_id"), f"{context}.owner_mapping.image_id") != image_id:
        raise AggregationError(f"{context}.owner_mapping.image_id differs from event image")
    source_count = owner_mapping.get("source_owner_count")
    derived_count = owner_mapping.get("derived_owner_count")
    for name, value in (("source_owner_count", source_count), ("derived_owner_count", derived_count)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise AggregationError(f"{context}.owner_mapping.{name} must be positive")
    if source_count != derived_count:
        raise AggregationError(f"{context}.owner_mapping source/derived object counts differ")
    _hash(owner_mapping.get("mapping_sha256"), f"{context}.owner_mapping.mapping_sha256")
    census = _object(owner_mapping.get("mapping_method_census"), f"{context}.owner_mapping.mapping_method_census")
    census_total = 0
    for method, value in census.items():
        if not isinstance(method, str) or not method:
            raise AggregationError(f"{context}.owner_mapping mapping method name is malformed")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise AggregationError(f"{context}.owner_mapping mapping method count is malformed")
        census_total += value
    if census_total != source_count:
        raise AggregationError(f"{context}.owner_mapping mapping method census does not cover all owners")
    source_to_derived = owner_mapping.get("source_to_derived")
    if source_to_derived is not None:
        rows = _array(source_to_derived, f"{context}.owner_mapping.source_to_derived")
        if len(rows) != source_count or sha256_json(rows) != owner_mapping.get("mapping_sha256"):
            raise AggregationError(f"{context}.owner_mapping rows disagree with mapping hash/count")
    return owner_mapping


def _validate_event_rows(
    rows: list[dict[str, Any]],
    spec: ShardSpec,
    expected_pairs: list[tuple[str, str]],
    identity: Mapping[str, Any],
    prefix_bindings: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    source_kind: str,
    pair_statuses: Mapping[tuple[str, str], str | None],
) -> dict[str, dict[str, Any]]:
    # Establish shard ownership before interpreting any nested receipts.  A
    # cross-shard event must report as a plan violation, not as an unrelated
    # malformed-prefix error.
    pre_observed: list[tuple[str, str]] = []
    pre_seen: set[str] = set()
    for index, row in enumerate(rows):
        event_id, image_id = _event_identity(row, f"{spec.path}/per_event_results:{index}")
        if event_id in pre_seen:
            raise AggregationError(f"{spec.path}: duplicate event_id {event_id}")
        pre_seen.add(event_id)
        if row.get("checkpoint") != spec.checkpoint:
            raise AggregationError(f"{spec.path}: event {event_id} checkpoint mismatch")
        pre_observed.append((event_id, image_id))
    if set(pre_observed) != set(expected_pairs) or len(pre_observed) != len(expected_pairs):
        raise AggregationError(
            f"{spec.path}: event ownership differs from the CLI shard plan; "
            f"expected {expected_pairs!r}, observed {pre_observed!r}"
        )
    observed: list[tuple[str, str]] = []
    seen: set[str] = set()
    mapping_receipts: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        event_id, image_id = _event_identity(row, f"{spec.path}/per_event_results:{index}")
        if event_id in seen:
            raise AggregationError(f"{spec.path}: duplicate event_id {event_id}")
        seen.add(event_id)
        if row.get("checkpoint") != spec.checkpoint:
            raise AggregationError(f"{spec.path}: event {event_id} checkpoint mismatch")
        if row.get("runtime_attestation") != identity.get("runtime_attestation"):
            raise AggregationError(f"{spec.path}: event {event_id} runtime attestation differs from runtime identity")
        prefix = _object(row.get("prefix"), f"{spec.path}:event {event_id}.prefix")
        mapping_receipt = _validate_prefix_receipt(
            prefix,
            spec=spec,
            event_id=event_id,
            image_id=image_id,
            expected=prefix_bindings.get((event_id, image_id)),
        )
        prior_mapping = mapping_receipts.get(image_id)
        if prior_mapping is not None and prior_mapping != mapping_receipt:
            raise AggregationError(f"{spec.path}: owner mapping receipt drifts within image {image_id}")
        mapping_receipts[image_id] = mapping_receipt
        context = f"{spec.path}:event {event_id}"
        if source_kind == "live_p4_overlay":
            _validate_p4_overlay_event(row, context=context)
        else:
            for stage in STAGES:
                if stage not in row:
                    raise AggregationError(f"{spec.path}: event {event_id} is missing stage {stage}")
                if not isinstance(row[stage], Mapping):
                    raise AggregationError(f"{spec.path}: event {event_id}.{stage} must be an object")
            _validate_event_statuses(row, context)
            if source_kind == "cpu_ineligible":
                _validate_ineligible_event_contract(
                    row,
                    context=context,
                    expected_pair_status=pair_statuses.get((event_id, image_id)),
                )
        observed.append((event_id, image_id))
    if observed != sorted(expected_pairs, key=lambda pair: expected_pairs.index(pair)):
        # Preserve the expected cohort order in the error while rejecting any
        # duplicate/missing/cross-shard identity.  Event order itself is not a
        # provenance source, so compare sets after the explicit ownership check.
        if set(observed) != set(expected_pairs):
            raise AggregationError(
                f"{spec.path}: event ownership differs from the CLI shard plan; "
                f"expected {expected_pairs!r}, observed {observed!r}"
            )
    if set(observed) != set(expected_pairs) or len(observed) != len(expected_pairs):
        raise AggregationError(
            f"{spec.path}: event ownership differs from the CLI shard plan; "
            f"expected {expected_pairs!r}, observed {observed!r}"
        )
    if identity.get("event_count") != len(expected_pairs):
        raise AggregationError(f"{spec.path}: runtime event_count does not match shard ownership")
    return mapping_receipts


def _validate_manifest_events(
    manifest: Mapping[str, Any],
    *,
    spec: ShardSpec,
    expected_pairs: Sequence[tuple[str, str]],
    name: str,
    require_image_id: bool = True,
) -> list[dict[str, Any]]:
    if manifest.get("schema_version") != RUNTIME_SCHEMA_VERSION:
        raise AggregationError(f"{spec.path}/{name}: unsupported schema")
    rows = [_object(item, f"{spec.path}/{name}.events") for item in _array(manifest.get("events"), f"{spec.path}/{name}.events")]
    seen: set[str] = set()
    observed: set[tuple[str, str]] = set()
    for index, row in enumerate(rows):
        event_id = _text(row.get("event_id"), f"{spec.path}/{name}.events[{index}].event_id")
        if event_id in seen:
            raise AggregationError(f"{spec.path}/{name}: duplicate event_id {event_id}")
        seen.add(event_id)
        if row.get("image_id") is None and not require_image_id:
            image_id = dict(expected_pairs).get(event_id)
            if image_id is None:
                raise AggregationError(f"{spec.path}/{name}: unknown event_id {event_id}")
        else:
            image_id = _text(row.get("image_id"), f"{spec.path}/{name}.events[{index}].image_id")
        if row.get("checkpoint") not in {None, spec.checkpoint}:
            raise AggregationError(f"{spec.path}/{name}: event {event_id} checkpoint mismatch")
        observed.add((event_id, image_id))
    if observed != set(expected_pairs):
        raise AggregationError(f"{spec.path}/{name}: event identity differs from per-event results")
    return rows


def _validate_shard(
    spec: ShardSpec,
    expected_pairs: list[tuple[str, str]],
    cohort_path: Path,
    cohort_sha256: str,
    source_hashes: Mapping[str, Any],
    prefix_bindings: Mapping[tuple[str, str], Mapping[str, Any]],
    pair_statuses: Mapping[tuple[str, str], str | None],
) -> ShardData:
    required = {
        "runtime_identity.json": "runtime_identity",
        "exact_prefix_manifest.json": "exact_prefix",
        "intervention_manifest.json": "intervention",
        "per_event_results.jsonl": "per_event_results",
        "gradient_receipt.json": "gradient_receipt",
        "terminal_summary.json": "terminal_summary",
    }
    paths = {name: spec.path / name for name in required}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise AggregationError(f"{spec.path}: missing required artifacts: {missing}")
    identity = _read_json(paths["runtime_identity.json"], f"{spec.path}/runtime_identity.json")
    source_kind = _validate_runtime_identity(identity, spec, cohort_path, cohort_sha256, source_hashes)
    events = _read_jsonl(paths["per_event_results.jsonl"], f"{spec.path}/per_event_results.jsonl")
    mapping_receipts = _validate_event_rows(
        events,
        spec,
        expected_pairs,
        identity,
        prefix_bindings,
        source_kind=source_kind,
        pair_statuses=pair_statuses,
    )
    exact_prefix = _read_json(paths["exact_prefix_manifest.json"], f"{spec.path}/exact_prefix_manifest.json")
    _assert_identity_match(exact_prefix.get("identity", {}), identity, f"{spec.path}/exact_prefix.identity")
    if exact_prefix.get("identity", {}).get("runtime_attestation") != identity.get("runtime_attestation"):
        raise AggregationError(f"{spec.path}: exact prefix identity runtime attestation differs from runtime identity")
    exact_rows = _validate_manifest_events(
        exact_prefix,
        spec=spec,
        expected_pairs=expected_pairs,
        name="exact_prefix_manifest.json",
    )
    event_by_id = {str(row["event_id"]): row for row in events}
    for manifest_row in exact_rows:
        event_id = str(manifest_row["event_id"])
        event_row = event_by_id[event_id]
        if manifest_row.get("prefix") != event_row.get("prefix"):
            raise AggregationError(f"{spec.path}: exact prefix manifest differs from event result for {event_id}")
        pair = (event_id, str(event_row["image_id"]))
        expected_prefix = prefix_bindings.get(pair)
        if expected_prefix is None:
            continue
        event_prefix = _object(event_row.get("prefix"), f"{spec.path}:event {event_id}.prefix")
        natural_boundary = expected_prefix.get("natural_boundary")
        if natural_boundary is not None:
            if event_prefix.get("natural_boundary") is not None and event_prefix.get("natural_boundary") != natural_boundary:
                raise AggregationError(f"{spec.path}: event {event_id} prefix natural boundary differs from H0 ledger")
        declared_h0_hash = expected_prefix.get("exact_prefix_sha256")
        if declared_h0_hash is not None:
            _hash(declared_h0_hash, f"{spec.path}:H0 exact prefix hash for {event_id}")
            for key in ("h0_exact_prefix_sha256", "exact_prefix_sha256"):
                if event_prefix.get(key) is not None and event_prefix.get(key) != declared_h0_hash:
                    raise AggregationError(f"{spec.path}: event {event_id} prefix hash differs from H0 ledger")
    intervention = _read_json(paths["intervention_manifest.json"], f"{spec.path}/intervention_manifest.json")
    _validate_manifest_events(
        intervention,
        spec=spec,
        expected_pairs=expected_pairs,
        name="intervention_manifest.json",
        require_image_id=False,
    )
    intervention_rows = _array(intervention.get("events"), f"{spec.path}/intervention.events")
    for index, raw_row in enumerate(intervention_rows):
        row = _object(raw_row, f"{spec.path}/intervention.events[{index}]")
        stages = row.get("stages")
        stage_values = _array(stages, f"{spec.path}/intervention.events[{index}].stages")
        expected_stages = {"p4"} if source_kind == "live_p4_overlay" else set(STAGES)
        if len(stage_values) != len(expected_stages) or set(stage_values) != expected_stages:
            raise AggregationError(
                f"{spec.path}: intervention manifest does not attest the exact {sorted(expected_stages)} stages"
            )
    gradient = _read_json(paths["gradient_receipt.json"], f"{spec.path}/gradient_receipt.json")
    if gradient.get("schema_version") != GRADIENT_SCHEMA_VERSION:
        raise AggregationError(f"{spec.path}: unsupported gradient receipt schema")
    if gradient.get("runtime_attestation") != identity.get("runtime_attestation"):
        raise AggregationError(f"{spec.path}: gradient receipt runtime attestation differs from runtime identity")
    gradient_rows = [_object(item, f"{spec.path}/gradient.receipts") for item in _array(gradient.get("receipts"), f"{spec.path}/gradient.receipts")]
    gradient_by_event: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(gradient_rows):
        event_id = _text(row.get("event_id"), f"{spec.path}/gradient.receipts[{index}].event_id")
        if event_id in gradient_by_event:
            raise AggregationError(f"{spec.path}: gradient receipt repeats event_id {event_id}")
        gradient_by_event[event_id] = _object(row.get("receipt"), f"{spec.path}/gradient.receipts[{index}].receipt")
    expected_ids = {event_id for event_id, _ in expected_pairs}
    if not set(gradient_by_event).issubset(expected_ids):
        raise AggregationError(f"{spec.path}: gradient receipt contains an unknown event")
    event_by_id = {str(row["event_id"]): row for row in events}
    for event_id, row in event_by_id.items():
        p4 = _object(row["p4"], f"{spec.path}:event {event_id}.p4")
        status = p4.get("status")
        if event_id in gradient_by_event:
            if gradient_by_event[event_id] != p4:
                raise AggregationError(f"{spec.path}: gradient receipt payload differs from event P4 for {event_id}")
        elif status in {"valid", "technical_invalid"}:
            raise AggregationError(f"{spec.path}: valid P4 event lacks gradient receipt: {event_id}")
    terminal = _read_json(paths["terminal_summary.json"], f"{spec.path}/terminal_summary.json")
    if terminal.get("status") != "completed":
        raise AggregationError(f"{spec.path}: terminal summary is not completed")
    _assert_identity_match(terminal, identity, f"{spec.path}/terminal_summary")
    if terminal.get("runtime_attestation") != identity.get("runtime_attestation"):
        raise AggregationError(f"{spec.path}: terminal runtime attestation differs from runtime identity")
    if terminal.get("events_attempted") != len(expected_pairs):
        raise AggregationError(f"{spec.path}: terminal events_attempted does not match shard ownership")
    input_files = [
        {"role": name, "path": str(path), "sha256": sha256_file(path)}
        for name, path in paths.items()
    ]
    return ShardData(
        spec,
        identity,
        events,
        exact_prefix,
        intervention,
        gradient,
        terminal,
        input_files,
        mapping_receipts,
        source_kind,
    )


def _status_from_record(record: Any, *, context: str, kind: str = "generic") -> tuple[str, str | None]:
    if not isinstance(record, Mapping):
        return "invalid", f"{context} is missing or not an object"
    status = record.get("status")
    if status is None:
        if kind == "p1_arm" and (
            _owner_ids(record)
            or isinstance(record.get("owner_match"), Mapping)
            or isinstance(record.get("native_parse"), Mapping)
            or isinstance(record.get("parsed"), Mapping)
        ):
            return "valid", None
        if kind == "horizon" and any(key in record for key in ("horizon_1", "horizon_3")):
            return "valid", None
        return "invalid", f"{context}.status is missing"
    allowed_by_kind = {
        "stage": {"attempted", "invalid/uninterpretable", "indeterminate", "not_applicable"},
        "p1_arm": {"valid", "complete", "not_applicable", "indeterminate", "invalid/uninterpretable", "technical_invalid", "invalid"},
        "horizon": {"completed", "stopped_early", "not_applicable", "indeterminate", "invalid/uninterpretable", "technical_invalid", "invalid"},
        "p4": {"valid", "technical_invalid", "invalid/uninterpretable", "invalid"},
    }
    if kind in allowed_by_kind and status not in allowed_by_kind[kind]:
        return "invalid", f"{context}.status is unknown: {status!r}"
    if kind == "stage":
        if status == "attempted":
            return "valid", None
        if status in {"not_applicable", "indeterminate"}:
            return "indeterminate", str(record.get("reason") or status)
        return "invalid", str(record.get("reason") or status)
    if status in {"not_applicable", "indeterminate"}:
        return "indeterminate", str(record.get("reason") or status)
    if status in {"invalid/uninterpretable", "technical_invalid", "invalid"}:
        reasons = record.get("invalid_reasons")
        if isinstance(reasons, list) and reasons:
            reason = ";".join(str(item) for item in reasons)
        else:
            reason = str(record.get("reason") or record.get("detail") or status)
        lower_reason = reason.lower()
        if any(
            marker in lower_reason
            for marker in (
                "not_applicable",
                "not applicable",
                "same-parent donor",
                "earlier equal-length completed row",
                "same-class competitor",
            )
        ):
            return "indeterminate", reason
        return "invalid", reason
    if (
        record.get("valid") is False
        or record.get("validity") is False
        or record.get("mechanically_valid") is False
    ):
        return "invalid", str(record.get("reason") or "validity=false")
    if record.get("invalid") is True or record.get("malformed") is True:
        return "invalid", str(record.get("reason") or "invalid_or_malformed_row")
    evidence = record.get("evidence_row")
    if isinstance(evidence, Mapping):
        if evidence.get("invalid") is True or evidence.get("malformed") is True or evidence.get("complete") is False:
            return "invalid", str(record.get("reason") or "invalid_evidence_row")
    return "valid", None


def _reason_counts(reasons: Iterable[str | None]) -> dict[str, int]:
    counts = Counter(str(reason) for reason in reasons if reason)
    return dict(sorted(counts.items()))


def _owner_ids(record: Mapping[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    for candidate in (record.get("owner_match"), record.get("owner_match_receipt")):
        if isinstance(candidate, Mapping):
            value = candidate.get("owner_id", candidate.get("matched_owner_id"))
            if isinstance(value, str) and value:
                values.append(value)
    evidence = record.get("evidence_row")
    if isinstance(evidence, Mapping):
        value = evidence.get("owner_id")
        if isinstance(value, str) and value:
            values.append(value)
    value = record.get("owner_id")
    if isinstance(value, str) and value:
        values.append(value)
    for row in _rows_in_result(record):
        value = row.get("owner_id")
        if isinstance(value, str) and value:
            values.append(value)
    return tuple(dict.fromkeys(values))


def _rows_in_result(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    raw_rows = record.get("rows")
    if isinstance(raw_rows, list):
        rows.extend(item for item in raw_rows if isinstance(item, Mapping))
    return rows


def _owner_delta(base: Sequence[str], treatment: Sequence[str]) -> dict[str, Any]:
    base_set, treatment_set = set(base), set(treatment)
    gained = sorted(treatment_set - base_set)
    retained = sorted(treatment_set & base_set)
    lost = sorted(base_set - treatment_set)
    return {
        "G": gained,
        "K": retained,
        "L": lost,
        "net": len(gained) - len(lost),
        "G_count": len(gained),
        "K_count": len(retained),
        "L_count": len(lost),
    }


def _p1_burden_metrics(record: Mapping[str, Any], *, context: str) -> tuple[dict[str, float], str | None]:
    """Normalize parser/owner outcomes into one mutually-exclusive burden bin."""

    parsed = record.get("native_parse")
    if not isinstance(parsed, Mapping):
        parsed = record.get("parsed")
    if not isinstance(parsed, Mapping):
        return {}, f"{context} is missing native_parse/parsed"
    parse_status = parsed.get("parse_status")
    parse_valid = parsed.get("valid")
    if parse_status not in {"accepted", "accepted_with_drops", "malformed", "parser_error", "all_spans_dropped", "empty"}:
        return {}, f"{context}.parse_status is unknown: {parse_status!r}"
    if parse_valid is not None and not isinstance(parse_valid, bool):
        return {}, f"{context}.valid must be boolean when present"
    # A native parser can report accepted_with_drops while still not producing
    # one complete row.  Treat that as malformed for the owner-release burden
    # census; the arm remains a valid mechanical observation.
    parser_ok = parse_status in {"accepted"} and parse_valid is True
    if parse_status == "accepted_with_drops":
        parser_ok = False
    if not parser_ok:
        return {"parse.malformed": 1.0}, None
    owner_match = record.get("owner_match")
    if not isinstance(owner_match, Mapping):
        return {}, f"{context}.owner_match is missing for an accepted parse"
    owner_status = owner_match.get("status")
    if owner_status in {"unique", "matched"}:
        return {"parse.valid": 1.0}, None
    if owner_status == "unmatched":
        return {"parse.unmatched": 1.0}, None
    if owner_status == "ambiguous":
        return {"parse.ambiguous": 1.0}, None
    return {}, f"{context}.owner_match.status is unknown: {owner_status!r}"


def _result_observation(record: Any, *, context: str, baseline: Sequence[str] = ()) -> dict[str, Any]:
    status, reason = _status_from_record(record, context=context, kind="p1_arm")
    observation: dict[str, Any] = {"validity": status, "reason": reason, "metrics": {}}
    if status != "valid":
        return observation
    if not isinstance(record, Mapping):
        return observation
    burden, burden_error = _p1_burden_metrics(record, context=context)
    if burden_error is not None:
        observation["validity"] = "invalid"
        observation["reason"] = burden_error
        return observation
    observation["metrics"].update(burden)
    if any(key not in record for key in ("generation_status", "complete_row", "stop_reason")):
        observation["validity"] = "invalid"
        observation["reason"] = f"{context} is missing generation_status/complete_row/stop_reason"
        observation["metrics"] = {}
        return observation
    generation_status = record.get("generation_status")
    row_state: str | None = None
    if generation_status is not None:
        if generation_status not in {"complete", "incomplete"}:
            observation["validity"] = "invalid"
            observation["reason"] = f"{context}.generation_status is unknown: {generation_status!r}"
            observation["metrics"] = {}
            return observation
        row_state = f"{generation_status}_row"
    complete_row = record.get("complete_row")
    if not isinstance(complete_row, bool):
        observation["validity"] = "invalid"
        observation["reason"] = f"{context}.complete_row must be boolean"
        observation["metrics"] = {}
        return observation
    explicit_state = "complete_row" if complete_row else "incomplete_row"
    if row_state is not None and row_state != explicit_state:
        observation["validity"] = "invalid"
        observation["reason"] = f"{context}.generation_status disagrees with complete_row"
        observation["metrics"] = {}
        return observation
    row_state = explicit_state
    if row_state is not None:
        observation["metrics"][row_state] = 1.0
    stop_reason = record.get("stop_reason")
    if not isinstance(stop_reason, str) or not stop_reason:
        observation["validity"] = "invalid"
        observation["reason"] = f"{context}.stop_reason must be a non-empty string"
        observation["metrics"] = {}
        return observation
    observation["metrics"][f"stop_reason.{stop_reason}"] = 1.0
    owners = _owner_ids(record)
    observation["owner_ids"] = list(owners)
    delta = _owner_delta(baseline, owners)
    observation["delta"] = delta
    observation["owner_utility"] = {
        "G": delta["G"],
        "K": delta["K"],
        "L": delta["L"],
        "net": delta["net"],
    }
    for key in ("G_count", "K_count", "L_count", "net"):
        observation["metrics"][key] = float(delta[key])
    return observation


def _bookkeeping_observation(record: Any, *, context: str) -> dict[str, Any]:
    status, reason = _status_from_record(record, context=context, kind="horizon")
    if status != "valid" or not isinstance(record, Mapping):
        return {"validity": status, "reason": reason, "metrics": {}}
    bookkeeping = record.get("owner_bookkeeping")
    if not isinstance(bookkeeping, Mapping):
        return {"validity": "invalid", "reason": f"{context}.owner_bookkeeping is missing", "metrics": {}}
    raw_g, raw_k, raw_l = bookkeeping.get("G"), bookkeeping.get("K"), bookkeeping.get("L")
    try:
        gained = [_text(item, f"{context}.G") for item in _array(raw_g, f"{context}.G")]
        retained = [_text(item, f"{context}.K") for item in _array(raw_k, f"{context}.K")]
        lost = [_text(item, f"{context}.L") for item in _array(raw_l, f"{context}.L")]
        net = _finite(bookkeeping.get("net"), f"{context}.net")
    except AggregationError as exc:
        return {"validity": "invalid", "reason": str(exc), "metrics": {}}
    if len(set(gained)) != len(gained) or len(set(retained)) != len(retained) or len(set(lost)) != len(lost):
        return {"validity": "invalid", "reason": f"{context}.owner_bookkeeping contains duplicate owners", "metrics": {}}
    if set(gained) & set(retained) or set(gained) & set(lost) or set(retained) & set(lost):
        return {"validity": "invalid", "reason": f"{context}.owner_bookkeeping G/K/L are not disjoint", "metrics": {}}
    expected_net = len(gained) - len(lost)
    if net != expected_net:
        return {"validity": "invalid", "reason": f"{context}.net disagrees with G/L", "metrics": {}}
    observation = {
        "validity": "valid",
        "reason": None,
        "owner_utility": {"G": gained, "K": retained, "L": lost, "net": net},
        "delta": {"G": gained, "K": retained, "L": lost, "net": net},
        "metrics": {
            "G_count": float(len(gained)),
            "K_count": float(len(retained)),
            "L_count": float(len(lost)),
            "net": net,
        },
    }
    repeat = bookkeeping.get("repeat_hazard")
    if isinstance(repeat, Mapping):
        for name, value in repeat.items():
            try:
                observation["metrics"][f"repeat_hazard.{name}"] = _finite(value, f"{context}.repeat_hazard.{name}")
            except AggregationError as exc:
                observation["validity"] = "invalid"
                observation["reason"] = str(exc)
                observation["metrics"] = {}
                return observation
    parse = bookkeeping.get("parse")
    if isinstance(parse, Mapping):
        for name in ("valid_rows", "duplicate_rows", "unmatched_rows", "ambiguous_rows", "malformed_rows", "invalid_rows"):
            if name in parse:
                try:
                    observation["metrics"][f"parse.{name}"] = _finite(parse[name], f"{context}.parse.{name}")
                except AggregationError as exc:
                    observation["validity"] = "invalid"
                    observation["reason"] = str(exc)
                    observation["metrics"] = {}
                    return observation
        try:
            if float(parse.get("malformed_rows", 0)) > 0 or float(parse.get("invalid_rows", 0)) > 0:
                return {"validity": "invalid", "reason": f"{context}.parse contains malformed/invalid rows", "metrics": {}}
        except (TypeError, ValueError):
            return {"validity": "invalid", "reason": f"{context}.parse invalid row counts", "metrics": {}}
    return observation


def _p1_observations(event: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    stage = _object(event.get("p1"), "event.p1")
    stage_status, stage_reason = _status_from_record(stage, context="event.p1", kind="stage")
    if stage_status != "valid":
        return {probe: {"validity": stage_status, "reason": stage_reason, "metrics": {}} for probe in P1_PROBES}
    arms = _object(stage.get("arms"), "event.p1.arms")
    missing = [probe for probe in P1_PROBES if probe not in arms]
    if missing:
        return {probe: {"validity": "invalid", "reason": f"missing probe(s): {','.join(missing)}", "metrics": {}} for probe in P1_PROBES}
    baseline_record = _object(arms["K00"], "event.p1.arms.K00")
    baseline_status, baseline_reason = _status_from_record(baseline_record, context="event.p1.arms.K00", kind="p1_arm")
    base = _owner_ids(baseline_record)
    observations = {
        probe: _result_observation(arms[probe], context=f"event.p1.arms.{probe}", baseline=base)
        for probe in P1_PROBES
    }
    baseline_observation = observations["K00"]
    if baseline_status != "valid" or baseline_observation.get("validity") != "valid":
        reason = f"K00 baseline unavailable: {baseline_reason or baseline_observation.get('reason') or baseline_status}"
        for observation in observations.values():
            if observation["validity"] == "valid":
                observation["validity"] = "indeterminate"
                observation["reason"] = reason
                observation.pop("delta", None)
                observation.pop("owner_utility", None)
                observation["metrics"] = {}
    return observations


def _horizon_observations(event: Mapping[str, Any], stage_name: str, probes: Sequence[str]) -> dict[str, dict[str, Any]]:
    stage = _object(event.get(stage_name), f"event.{stage_name}")
    stage_status, stage_reason = _status_from_record(stage, context=f"event.{stage_name}", kind="stage")
    if stage_status != "valid":
        return {f"{probe}.horizon_{horizon}": {"validity": stage_status, "reason": stage_reason, "metrics": {}} for probe in probes for horizon in (1, 3)}
    container_key = "arms" if stage_name == "p2" else "cells"
    container = _object(stage.get(container_key), f"event.{stage_name}.{container_key}")
    output: dict[str, dict[str, Any]] = {}
    for probe in probes:
        raw_probe = container.get(probe)
        if raw_probe is None:
            for horizon in (1, 3):
                output[f"{probe}.horizon_{horizon}"] = {"validity": "invalid", "reason": "missing probe", "metrics": {}}
            continue
        probe_map = _object(raw_probe, f"event.{stage_name}.{probe}")
        probe_status, probe_reason = _status_from_record(probe_map, context=f"event.{stage_name}.{probe}", kind="horizon")
        if probe_status in {"indeterminate", "invalid"}:
            for horizon in (1, 3):
                output[f"{probe}.horizon_{horizon}"] = {"validity": probe_status, "reason": probe_reason, "metrics": {}}
            continue
        for horizon in (1, 3):
            key = f"horizon_{horizon}"
            output[f"{probe}.{key}"] = _bookkeeping_observation(probe_map.get(key), context=f"event.{stage_name}.{probe}.{key}")
    return output


def _crossover_observation(event: Mapping[str, Any], horizon: int) -> dict[str, Any]:
    stage = _object(event.get("p3"), "event.p3")
    stage_status, stage_reason = _status_from_record(stage, context="event.p3", kind="stage")
    if stage_status != "valid":
        return {"validity": stage_status, "reason": stage_reason, "metrics": {}}
    cells = _object(stage.get("cells"), "event.p3.cells")
    values: dict[str, float] = {}
    reasons: list[str] = []
    owner_utility: dict[str, dict[str, Any]] = {}
    for cell in P3_CELLS:
        cell_map = cells.get(cell)
        if not isinstance(cell_map, Mapping):
            reasons.append(f"missing cell {cell}")
            continue
        raw = cell_map.get(f"horizon_{horizon}")
        observation = _bookkeeping_observation(raw, context=f"event.p3.cells.{cell}.horizon_{horizon}")
        if observation["validity"] != "valid":
            reasons.append(f"{cell}: {observation.get('reason') or observation['validity']}")
            continue
        values[cell] = float(observation["metrics"]["net"])
        owner_utility[cell] = dict(observation.get("owner_utility", {}))
    if reasons:
        return {"validity": "indeterminate", "reason": ";".join(reasons), "metrics": {}}
    metrics = {
        "Y00_net": values["Y00"],
        "Y10_net": values["Y10"],
        "Y01_net": values["Y01"],
        "Y11_net": values["Y11"],
        "Delta_static": values["Y10"] - values["Y00"],
        "Delta_dynamic": values["Y01"] - values["Y00"],
        "tau": (values["Y11"] - values["Y10"]) - (values["Y01"] - values["Y00"]),
    }
    return {
        "validity": "valid",
        "reason": None,
        "metrics": metrics,
        "owner_utility": owner_utility,
        "delta": {
            "Delta_static": metrics["Delta_static"],
            "Delta_dynamic": metrics["Delta_dynamic"],
            "tau": metrics["tau"],
        },
    }


def _gradient_observations(event: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    receipt = _object(event.get("p4"), "event.p4")
    status, reason = _status_from_record(receipt, context="event.p4", kind="p4")
    if status != "valid":
        return {probe: {"validity": status, "reason": reason, "metrics": {}} for probe in (*P4_OBJECTIVES, "path_checks")}
    objectives = _object(receipt.get("objectives"), "event.p4.objectives")
    result: dict[str, dict[str, Any]] = {}
    required_states = {
        "target_b_complete_row_nll": {"image_residual", "matched_background"},
        "uncovered_b_vs_covered_a_margin_loss": {"latest_terminal_carrier", "latest_row_span"},
        "fixed_sum_coupled": {"image_residual", "matched_background", "latest_terminal_carrier", "latest_row_span"},
    }
    for objective_name in P4_OBJECTIVES:
        objective = _object(objectives.get(objective_name), f"event.p4.objectives.{objective_name}")
        try:
            metrics: dict[str, float] = {"objective_value": _finite(objective.get("value"), f"event.p4.objectives.{objective_name}.value")}
            ratio = objective.get("target_control_ratio")
            if ratio is not None:
                metrics["target_control_ratio"] = _finite(ratio, f"event.p4.objectives.{objective_name}.target_control_ratio")
            gradients = _object(objective.get("gradients"), f"event.p4.objectives.{objective_name}.gradients")
            missing_required = sorted(required_states[objective_name] - set(gradients))
            if missing_required:
                raise AggregationError(
                    f"event.p4 objective {objective_name} is missing required gradient state(s): {','.join(missing_required)}"
                )
            for state_name, raw_gradient in gradients.items():
                gradient = _object(raw_gradient, f"event.p4.objectives.{objective_name}.gradients.{state_name}")
                required = state_name in required_states[objective_name]
                if required and (gradient.get("present") is not True or gradient.get("finite") is not True):
                    raise AggregationError(f"event.p4 objective {objective_name} has non-finite/missing required {state_name} gradient")
                if gradient.get("present") is True and gradient.get("finite") is not True:
                    raise AggregationError(f"event.p4 objective {objective_name} has non-finite {state_name} gradient")
                if gradient.get("present") is not True:
                    continue
                metrics[f"gradient_norm.{state_name}"] = _finite(gradient.get("norm"), f"event.p4 gradient norm {objective_name}/{state_name}")
                metrics[f"gradient_max_abs.{state_name}"] = _finite(gradient.get("max_abs"), f"event.p4 gradient max abs {objective_name}/{state_name}")
            result[objective_name] = {"validity": "valid", "reason": None, "metrics": metrics}
        except AggregationError as exc:
            result[objective_name] = {"validity": "invalid", "reason": str(exc), "metrics": {}}
    path_checks = _object(receipt.get("path_checks"), "event.p4.path_checks")
    checks = {}
    for name in ("optimizer_used", "lm_head_only_path", "model_parameter_mutated", "parameter_grad_mutated", "audit_input_mutated"):
        value = path_checks.get(name)
        if not isinstance(value, bool):
            result["path_checks"] = {"validity": "invalid", "reason": f"path_checks.{name} is not boolean", "metrics": {}}
            break
        checks[name] = 1.0 if value else 0.0
    else:
        detached = path_checks.get("visual_state_detached")
        if not isinstance(detached, list):
            result["path_checks"] = {"validity": "invalid", "reason": "visual_state_detached is not a list", "metrics": {}}
        else:
            checks["visual_state_detached_count"] = float(len(detached))
            result["path_checks"] = {"validity": "valid", "reason": None, "metrics": checks}
    return result


def _event_observations(event: Mapping[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    p1 = _p1_observations(event)
    p2 = _horizon_observations(event, "p2", P2_ARMS)
    p3 = _horizon_observations(event, "p3", P3_CELLS)
    p3["crossover.horizon_1"] = _crossover_observation(event, 1)
    p3["crossover.horizon_3"] = _crossover_observation(event, 3)
    p4 = _gradient_observations(event)
    return {"p1": p1, "p2": p2, "p3": p3, "p4": p4}


def _empty_metric() -> dict[str, Any]:
    return {"count": 0, "sum": 0.0, "mean": None, "min": None, "max": None}


def _add_metric(metric: dict[str, Any], value: float) -> None:
    metric["count"] += 1
    metric["sum"] += value
    metric["min"] = value if metric["min"] is None else min(metric["min"], value)
    metric["max"] = value if metric["max"] is None else max(metric["max"], value)
    metric["mean"] = metric["sum"] / metric["count"]


def _aggregate_observations(
    event_records: Sequence[Mapping[str, Any]],
    checkpoint: str,
) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    output: dict[str, dict[str, dict[str, dict[str, Any]]]] = {checkpoint: {}}
    for stage in STAGES:
        probe_names = sorted({probe for event in event_records for probe in _object(event["observations"], "event.observations").get(stage, {})})
        output[checkpoint][stage] = {}
        for probe in probe_names:
            output[checkpoint][stage][probe] = {}
            for split in SPLITS:
                records = [
                    _object(event["observations"], "event.observations")[stage][probe]
                    for event in event_records
                    if split == "all" or (split == "image2299") == (str(event["image_id"]) == "2299")
                ]
                counts = Counter(_text(record.get("validity"), "observation.validity") for record in records)
                reasons: list[str | None] = [record.get("reason") for record in records]
                metrics: dict[str, dict[str, Any]] = defaultdict(_empty_metric)
                utility_counts = {name: 0 for name in ("G", "K", "L")}
                utility_events = {name: 0 for name in ("G", "K", "L")}
                for record in records:
                    if record.get("validity") != "valid":
                        continue
                    for name, raw_value in _object(record.get("metrics", {}), "observation.metrics").items():
                        if isinstance(raw_value, bool):
                            continue
                        _add_metric(metrics[name], _finite(raw_value, f"observation.metrics.{name}"))
                    utility = record.get("owner_utility")
                    if isinstance(utility, Mapping):
                        for name in utility_counts:
                            values = utility.get(name)
                            if isinstance(values, list):
                                utility_counts[name] += len(values)
                                utility_events[name] += 1
                output[checkpoint][stage][probe][split] = {
                    "events_total": len(records),
                    "valid_count": counts.get("valid", 0),
                    "indeterminate_count": counts.get("indeterminate", 0),
                    "invalid_count": counts.get("invalid", 0),
                    "numeric_count": sum(metric["count"] for metric in metrics.values()),
                    "reason_counts": _reason_counts(reasons),
                    "metrics": dict(sorted(metrics.items())),
                    "owner_utility": {
                        name: {"event_count": utility_events[name], "owner_count": utility_counts[name]}
                        for name in utility_counts
                    },
                }
    return output


def _evidence_views(aggregates: Mapping[str, Any]) -> dict[str, Any]:
    """Expose H1--H5 decision inputs without assigning a disposition."""

    views: dict[str, Any] = {}
    for checkpoint, by_stage in aggregates.items():
        views[checkpoint] = {
            "static_transfer": {
                probe: by_stage.get("p1", {}).get(probe, {})
                for probe in ("K10", "K11", "R10_block13", "R10_block23", "R10_block27")
                if probe in by_stage.get("p1", {})
            },
            "dynamic_transfer": {
                probe: by_stage.get("p2", {}).get(probe, {})
                for probe in ("D10.horizon_1", "D10.horizon_3", "D20.horizon_3")
                if probe in by_stage.get("p2", {})
            },
            "crossover": {
                probe: by_stage.get("p3", {}).get(probe, {})
                for probe in ("crossover.horizon_1", "crossover.horizon_3")
                if probe in by_stage.get("p3", {})
            },
            "gradient_path": {
                probe: by_stage.get("p4", {}).get(probe, {})
                for probe in (*P4_OBJECTIVES, "path_checks")
                if probe in by_stage.get("p4", {})
            },
            "interpretation": None,
        }
    return views


def _summary_self_hash(summary: Mapping[str, Any]) -> str:
    payload = dict(summary)
    payload.pop("receipt", None)
    payload.pop("self_sha256", None)
    return sha256_json(payload)


def _atomic_write(path: Path, value: Mapping[str, Any]) -> str:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    content = _canonical(value) + b"\n"
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                Path(temporary).unlink()
            except OSError:
                pass
    return sha256_bytes(content)


def _cohort_pair_statuses(
    cohort: Mapping[str, Any], checkpoint: str
) -> dict[tuple[str, str], str | None]:
    result: dict[tuple[str, str], str | None] = {}
    for index, raw_event in enumerate(_array(cohort.get("events"), f"cohort {checkpoint}.events")):
        event = _object(raw_event, f"cohort {checkpoint}.events[{index}]")
        pair = _cohort_event_identity(event, index)
        status: str | None = None
        pairs = event.get("A_B")
        if isinstance(pairs, Mapping) and isinstance(pairs.get(checkpoint), Mapping):
            raw_status = pairs[checkpoint].get("pair_status")
            if raw_status is not None:
                status = _text(raw_status, f"cohort {checkpoint}.events[{index}].pair_status")
                if status not in _KNOWN_PAIR_STATUSES:
                    raise AggregationError(
                        f"cohort {checkpoint}.events[{index}] has unknown pair_status {status!r}"
                    )
        result[pair] = status
    return result


def _peek_shard_pairs(spec: ShardSpec) -> list[tuple[str, str]]:
    path = spec.path / "per_event_results.jsonl"
    if not path.is_file():
        raise AggregationError(f"{spec.path}: missing required artifact: {path}")
    rows = _read_jsonl(path, f"{spec.path}/per_event_results.jsonl")
    if not rows:
        raise AggregationError(f"{spec.path}: per-event results must not be empty")
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for index, row in enumerate(rows):
        pair = _event_identity(row, f"{spec.path}/per_event_results:{index}")
        if pair in seen:
            raise AggregationError(f"{spec.path}: duplicate event pair {pair!r}")
        seen.add(pair)
        pairs.append(pair)
    return pairs


def _stable_repair_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: identity.get(key)
        for key in (
            "schema_version",
            "unit_id",
            "checkpoint",
            "config_path",
            "config_sha256",
            "resolved_config_fingerprint",
            "panel_path",
            "panel_sha256",
            "panel_identity",
            "cohort_path",
            "cohort_sha256",
            "h0",
        )
    }


def _merge_p4_repair(base: ShardData, overlay: ShardData) -> dict[str, Any]:
    if len(base.events) != 1 or len(overlay.events) != 1:
        raise AggregationError("P4 repair base and overlay must each contain exactly one event")
    base_event = base.events[0]
    overlay_event = overlay.events[0]
    base_pair = _event_identity(base_event, f"{base.spec.path}/repair base")
    overlay_pair = _event_identity(overlay_event, f"{overlay.spec.path}/repair overlay")
    if base_pair != overlay_pair:
        raise AggregationError("P4 repair overlay event identity differs from its base")
    if _stable_repair_identity(base.identity) != _stable_repair_identity(overlay.identity):
        raise AggregationError("P4 repair overlay runtime identity differs from its all-stage base")
    if overlay_event.get("prefix") != base_event.get("prefix"):
        raise AggregationError("P4 repair overlay prefix differs from its all-stage base")
    base_eligibility = _object(base_event.get("eligibility"), "P4 repair base.eligibility")
    overlay_eligibility = _object(overlay_event.get("eligibility"), "P4 repair overlay.eligibility")
    if base_eligibility != overlay_eligibility:
        raise AggregationError("P4 repair overlay eligibility differs from its all-stage base")
    base_p4 = _object(base_event.get("p4"), "P4 repair base.p4")
    if base_p4.get("status") != "technical_invalid":
        raise AggregationError("P4 repair base must have status=technical_invalid")
    reasons = base_p4.get("invalid_reasons")
    if not isinstance(reasons, list) or not reasons or any(not isinstance(item, str) or not item for item in reasons):
        raise AggregationError("P4 repair base must preserve a non-empty invalid_reasons lineage")
    if not any("forward_capture_contract_invalid" in item for item in reasons):
        raise AggregationError("P4 repair base lacks the frozen forward_capture_contract_invalid reason")
    merged = copy.deepcopy(base_event)
    merged["p4"] = copy.deepcopy(overlay_event["p4"])
    merged["p4_repair_lineage"] = {
        "base": {
            "path": str(base.spec.path),
            "runtime_identity_sha256": sha256_file(base.spec.path / "runtime_identity.json"),
            "p4": base_p4,
        },
        "repair": {
            "path": str(overlay.spec.path),
            "runtime_identity_sha256": sha256_file(overlay.spec.path / "runtime_identity.json"),
            "p4": copy.deepcopy(overlay_event["p4"]),
        },
        "superseded_stages": ["p4"],
    }
    return merged


def _hold_self_hash(payload: Mapping[str, Any]) -> str:
    value = dict(payload)
    value.pop("self_sha256", None)
    return sha256_json(value)


def _validate_eligible_hold(
    spec: EligibleHoldSpec,
    *,
    cohort_path: Path,
    cohort_sha256: str,
    expected_events: Sequence[tuple[str, str]],
    prefix_bindings: Mapping[tuple[str, str], Mapping[str, Any]],
    pair_statuses: Mapping[tuple[str, str], str | None],
) -> tuple[tuple[str, str], dict[str, Any], dict[str, Any]]:
    del prefix_bindings  # The sealed leaf binds H0 directly; it is not a synthetic prefix receipt.
    payload = _read_json(spec.path, f"eligible HOLD leaf {spec.path}")
    expected_keys = {
        "schema_version",
        "unit_id",
        "status",
        "kind",
        "receipt_path",
        "frozen_contract",
        "event_binding",
        "h0_binding",
        "classification",
        "repair_policy",
        "execution_evidence",
        "attempt_lineage",
        "matrix_dispositions",
        "self_sha256",
    }
    if set(payload) != expected_keys:
        raise AggregationError("eligible HOLD leaf does not match the sealed schema")
    if (
        payload.get("schema_version") != ELIGIBLE_HOLD_SCHEMA_VERSION
        or payload.get("unit_id") != UNIT_ID
        or payload.get("status") != "sealed"
        or payload.get("kind") != "eligible_pre_actuator_technical_hold"
    ):
        raise AggregationError("eligible HOLD leaf schema/unit/status/kind mismatch")
    if _hash(payload.get("self_sha256"), "eligible HOLD self_sha256") != _hold_self_hash(payload):
        raise AggregationError("eligible HOLD leaf self hash mismatch")
    if spec.ordinal > len(expected_events):
        raise AggregationError("eligible HOLD ordinal is outside the cohort")
    expected_pair = expected_events[spec.ordinal - 1]
    event = _object(payload.get("event_binding"), "eligible HOLD.event_binding")
    if (
        event.get("checkpoint") != spec.checkpoint
        or event.get("ordinal") != spec.ordinal
        or _text(event.get("event_id"), "eligible HOLD.event_id") != expected_pair[0]
        or _text(event.get("image_id"), "eligible HOLD.image_id") != expected_pair[1]
        or event.get("cohort_sha256") != cohort_sha256
        or event.get("cohort_eligibility") != "eligible_verified_pair"
        or pair_statuses.get(expected_pair) != "verified_pair"
    ):
        raise AggregationError("eligible HOLD event is not the exact cohort-proven verified pair")
    classification = _object(payload.get("classification"), "eligible HOLD.classification")
    if classification != {
        "actuators_called": None,
        "complete_non_scored": False,
        "matrix_status": "eligible_pre_actuator_hold",
        "scored": False,
    }:
        raise AggregationError("eligible HOLD classification confuses HOLD with scored evidence")
    execution = _object(payload.get("execution_evidence"), "eligible HOLD.execution_evidence")
    expected_execution = {
        "actual_actuator_invocation_count": None,
        "actual_model_forward_count": None,
        "lead_observation_evidence_level": "unattested",
        "lead_observed_pre_actuator": True,
        "persisted_artifacts_absent": [
            "exact_prefix_manifest.json",
            "intervention_manifest.json",
            "per_event_results.jsonl",
            "gradient_receipt.json",
            "terminal_summary.json",
        ],
        "persisted_event_result_count": 0,
        "persisted_failure_log": {"path": None, "sha256": None, "status": "unavailable"},
        "persisted_gradient_receipt_count": 0,
        "persisted_intervention_receipt_count": 0,
        "persisted_terminal_count": 0,
        "receipt_bearing_actuator_cell_count": 0,
        "receipt_bearing_scientific_cell_count": 0,
    }
    if execution != expected_execution:
        raise AggregationError("eligible HOLD execution evidence is not the exact zero-receipt boundary")
    frozen = _object(payload.get("frozen_contract"), "eligible HOLD.frozen_contract")
    if set(frozen) != {"tasks", "unit"}:
        raise AggregationError("eligible HOLD frozen contract is not exact")
    for role in ("tasks", "unit"):
        ref = _object(frozen.get(role), f"eligible HOLD.frozen_contract.{role}")
        path = _resolve_path(ref.get("path"), f"eligible HOLD.frozen_contract.{role}.path")
        if sha256_file(path) != _hash(ref.get("sha256"), f"eligible HOLD.frozen_contract.{role}.sha256") or ref.get("size_bytes") != path.stat().st_size:
            raise AggregationError(f"eligible HOLD frozen {role} reference mismatch")
    repair_policy = _object(payload.get("repair_policy"), "eligible HOLD.repair_policy")
    if repair_policy != {
        "attempt_count": 2,
        "attempt_roles": ["initial", "repair1"],
        "exhausted": True,
        "repair_count": 1,
        "rule": "one_exact_repair_then_invalid_uninterpretable",
        "unit_sha256": frozen["unit"]["sha256"],
    }:
        raise AggregationError("eligible HOLD repair exhaustion contract is not exact")
    attempts = _array(payload.get("attempt_lineage"), "eligible HOLD.attempt_lineage")
    if len(attempts) != 2:
        raise AggregationError("eligible HOLD must bind exactly the base attempt and one repair attempt")
    for expected_role, raw_attempt in zip(("initial", "repair1"), attempts, strict=True):
        attempt = _object(raw_attempt, f"eligible HOLD.attempt_lineage.{expected_role}")
        if attempt.get("role") != expected_role:
            raise AggregationError("eligible HOLD attempt lineage is not initial then repair1")
        directory = _resolve_path(attempt.get("directory"), f"eligible HOLD attempt {expected_role}.directory")
        identity_path = _resolve_path(attempt.get("runtime_identity_path"), f"eligible HOLD attempt {expected_role}.runtime_identity_path")
        if identity_path != directory / "runtime_identity.json":
            raise AggregationError("eligible HOLD attempt runtime identity is outside its directory")
        digest = _hash(attempt.get("runtime_identity_sha256"), f"eligible HOLD attempt {expected_role}.runtime_identity_sha256")
        if sha256_file(identity_path) != digest or attempt.get("runtime_identity_size_bytes") != identity_path.stat().st_size:
            raise AggregationError("eligible HOLD repair attempt identity hash/size mismatch")
        census = _array(attempt.get("file_census"), f"eligible HOLD attempt {expected_role}.file_census")
        expected_census = [{"relative_path": "runtime_identity.json", "sha256": digest, "size_bytes": identity_path.stat().st_size}]
        if census != expected_census or attempt.get("file_census_sha256") != sha256_json(census):
            raise AggregationError("eligible HOLD attempt census mismatch")
        cause = _object(attempt.get("cause"), f"eligible HOLD attempt {expected_role}.cause")
        if cause.get("evidence_level") != "lead_observed_unattested" or cause.get("verbatim_stderr") is not None:
            raise AggregationError("eligible HOLD attempt cause overstates persisted evidence")
        attestation = _object(attempt.get("runtime_attestation"), f"eligible HOLD attempt {expected_role}.runtime_attestation")
        if attestation.get("checkpoint") != spec.checkpoint or attestation.get("status") != "validated" or attestation.get("passed") is not True:
            raise AggregationError("eligible HOLD attempt runtime attestation is not validated")
        if attempt.get("persisted_failure_log") != {"path": None, "sha256": None, "status": "unavailable"}:
            raise AggregationError("eligible HOLD attempt invents a persisted failure log")
    matrix = _object(payload.get("matrix_dispositions"), "eligible HOLD.matrix_dispositions")
    expected_cells = {
        "p1": P1_PROBES,
        "p2": tuple(f"{arm}.horizon_{horizon}" for arm in P2_ARMS for horizon in (1, 3)),
        "p3": tuple(f"{cell}.horizon_{horizon}" for cell in P3_CELLS for horizon in (1, 3)),
        "p4": P4_OBJECTIVES,
    }
    expected_cell = {
        "execution_status": "not_sealed",
        "metrics": None,
        "model_output": None,
        "reason_code": "pre_actuator_technical_failure_repair_exhausted",
        "scientific_observation": None,
        "status": "invalid/uninterpretable",
    }
    if set(matrix) != set(expected_cells):
        raise AggregationError("eligible HOLD matrix stages are not exact")
    for stage, cell_ids in expected_cells.items():
        stage_record = _object(matrix.get(stage), f"eligible HOLD.matrix_dispositions.{stage}")
        cells = _object(stage_record.get("cells"), f"eligible HOLD.matrix_dispositions.{stage}.cells")
        if (
            stage_record.get("status") != "invalid/uninterpretable"
            or stage_record.get("origin") != "administrative_disposition_not_model_output"
            or set(cells) != set(cell_ids)
            or any(cell != expected_cell for cell in cells.values())
        ):
            raise AggregationError(f"eligible HOLD {stage} matrix carries non-HOLD or numeric evidence")
    h0 = _object(payload.get("h0_binding"), "eligible HOLD.h0_binding")
    if h0.get("target_b_owner_id") != expected_pair[0] or h0.get("verified_support") is not True:
        raise AggregationError("eligible HOLD H0 target/support identity mismatch")
    for stem in ("native_h0_ledger", "support_ledger"):
        path = _resolve_path(h0.get(f"{stem}_path"), f"eligible HOLD.h0_binding.{stem}_path")
        if sha256_file(path) != _hash(h0.get(f"{stem}_sha256"), f"eligible HOLD.h0_binding.{stem}_sha256"):
            raise AggregationError(f"eligible HOLD {stem} hash mismatch")
    receipt_path = _resolve_path(payload.get("receipt_path"), "eligible HOLD.receipt_path")
    receipt = _read_json(receipt_path, "eligible HOLD receipt")
    receipt_keys = {
        "schema_version",
        "unit_id",
        "kind",
        "leaf_path",
        "leaf_sha256",
        "leaf_self_sha256",
        "input_set_sha256",
        "inputs",
        "self_sha256",
    }
    if set(receipt) != receipt_keys or receipt.get("schema_version") != f"{ELIGIBLE_HOLD_SCHEMA_VERSION}.receipt.v1":
        raise AggregationError("eligible HOLD receipt schema is not exact")
    if (
        receipt.get("unit_id") != UNIT_ID
        or receipt.get("kind") != payload.get("kind")
        or _resolve_path(receipt.get("leaf_path"), "eligible HOLD receipt.leaf_path") != spec.path
        or receipt.get("leaf_sha256") != sha256_file(spec.path)
        or receipt.get("leaf_self_sha256") != payload.get("self_sha256")
    ):
        raise AggregationError("eligible HOLD receipt does not bind the admitted leaf")
    receipt_inputs = _array(receipt.get("inputs"), "eligible HOLD receipt.inputs")
    if receipt.get("input_set_sha256") != sha256_json(receipt_inputs):
        raise AggregationError("eligible HOLD receipt input set hash mismatch")
    for index, raw_ref in enumerate(receipt_inputs):
        ref = _object(raw_ref, f"eligible HOLD receipt.inputs[{index}]")
        path = _resolve_path(ref.get("path"), f"eligible HOLD receipt.inputs[{index}].path")
        if sha256_file(path) != _hash(ref.get("sha256"), f"eligible HOLD receipt.inputs[{index}].sha256") or ref.get("size_bytes") != path.stat().st_size:
            raise AggregationError("eligible HOLD receipt input reference mismatch")
    if _hash(receipt.get("self_sha256"), "eligible HOLD receipt.self_sha256") != _hold_self_hash(receipt):
        raise AggregationError("eligible HOLD receipt self hash mismatch")
    reason = "pre_actuator_technical_failure_repair_exhausted"
    invalid_cell = {"status": "technical_invalid", "invalid_reasons": [reason]}
    raw_event: dict[str, Any] = {
        "event_id": expected_pair[0],
        "image_id": expected_pair[1],
        "checkpoint": spec.checkpoint,
        "eligibility": {
            "status": "eligible",
            "pair_status": "verified_pair",
            "actuators_called": None,
            "disposition": "eligible_pre_actuator_technical_hold",
        },
        "p1": {"status": "invalid/uninterpretable", "reason": reason, "arms": {probe: copy.deepcopy(invalid_cell) for probe in P1_PROBES}},
        "p2": {
            "status": "invalid/uninterpretable",
            "reason": reason,
            "arms": {
                arm: {f"horizon_{horizon}": copy.deepcopy(invalid_cell) for horizon in (1, 3)}
                for arm in P2_ARMS
            },
        },
        "p3": {
            "status": "invalid/uninterpretable",
            "reason": reason,
            "cells": {
                cell: {f"horizon_{horizon}": copy.deepcopy(invalid_cell) for horizon in (1, 3)}
                for cell in P3_CELLS
            },
        },
        "p4": {"status": "technical_invalid", "invalid_reasons": [reason]},
    }
    return expected_pair, raw_event, payload


def aggregate_shards(
    *,
    cohort_paths: Mapping[str, str | Path] | None = None,
    shard_specs: Sequence[ShardSpec],
    eligible_hold_specs: Sequence[EligibleHoldSpec] = (),
    output_path: str | Path | None = None,
    receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate and aggregate the exact sparse/hybrid final evidence topology."""

    resolved_output = None if output_path is None else Path(output_path).expanduser().resolve()
    resolved_receipt = None if receipt_path is None else Path(receipt_path).expanduser().resolve()
    if resolved_output is not None and resolved_receipt is not None and resolved_output == resolved_receipt:
        raise AggregationError("output and receipt paths must be distinct")
    for label, path in (("output", resolved_output), ("receipt", resolved_receipt)):
        if path is not None and path.exists():
            raise FileExistsError(f"{label} collision: refusing to overwrite {path}")
    if not isinstance(cohort_paths, Mapping) or set(cohort_paths) != set(CHECKPOINTS):
        raise AggregationError("cohort_paths must contain exactly one cohort for S and one for A")
    resolved_cohort_paths = {
        checkpoint: Path(cohort_paths[checkpoint]).expanduser().resolve(strict=True)
        for checkpoint in CHECKPOINTS
    }
    if len(set(resolved_cohort_paths.values())) != len(CHECKPOINTS):
        raise AggregationError("S and A must use distinct checkpoint-specific cohort artifacts")
    loaded_cohorts: dict[
        str,
        tuple[
            dict[str, Any],
            str,
            list[tuple[str, str]],
            dict[str, Any],
            dict[tuple[str, str], dict[str, Any]],
            Path,
        ],
    ] = {}
    for checkpoint in CHECKPOINTS:
        cohort_path = resolved_cohort_paths[checkpoint]
        loaded_cohorts[checkpoint] = (*_load_cohort(cohort_path), cohort_path)
    specs = list(shard_specs)
    if len(specs) < len(CHECKPOINTS) * SHARD_COUNT:
        raise AggregationError(f"at least {len(CHECKPOINTS) * SHARD_COUNT} shard selectors are required")
    by_key: dict[tuple[str, int, int], list[ShardSpec]] = defaultdict(list)
    for spec in specs:
        if spec.count != SHARD_COUNT:
            raise AggregationError("all shard selectors must use count=4")
        by_key[spec.key].append(spec)
    expected_keys = {(checkpoint, index, SHARD_COUNT) for checkpoint in CHECKPOINTS for index in range(SHARD_COUNT)}
    if set(by_key) != expected_keys:
        raise AggregationError(f"shard plan must contain S/A indices 0..3 exactly; got {sorted(by_key)}")
    if len({spec.path for spec in specs}) != len(specs):
        raise AggregationError("one artifact directory cannot serve two shard roots")
    holds = list(eligible_hold_specs)
    if len({(spec.checkpoint, spec.ordinal) for spec in holds}) != len(holds):
        raise AggregationError("duplicate eligible HOLD selector")
    if len({spec.path for spec in holds}) != len(holds):
        raise AggregationError("one eligible HOLD leaf cannot serve two events")
    if {spec.path for spec in holds} & {spec.path for spec in specs}:
        raise AggregationError("a HOLD leaf cannot also be a shard root")

    # Compute each checkpoint's immutable cohort ownership partition before
    # opening any shard.  The S and A cohorts are intentionally independent:
    # support-aware replacement can change an owner event in one checkpoint.
    expected_by_shard: dict[tuple[str, int], list[tuple[str, str]]] = {}
    for checkpoint in CHECKPOINTS:
        expected_events = loaded_cohorts[checkpoint][2]
        for index, pair in enumerate(expected_events):
            expected_by_shard.setdefault((checkpoint, index % SHARD_COUNT), []).append(pair)
    shards: list[ShardData] = []
    provenance_by_checkpoint: dict[str, dict[str, Any]] = {}
    all_inputs: list[dict[str, Any]] = []
    cohort_output: dict[str, Any] = {}
    for checkpoint in CHECKPOINTS:
        _cohort, cohort_sha256, expected_events, source_hashes, prefix_bindings, cohort_path = loaded_cohorts[checkpoint]
        cohort_output[checkpoint] = {
            "path": str(cohort_path),
            "sha256": cohort_sha256,
            "event_count": len(expected_events),
            "event_set_sha256": sha256_json(sorted(expected_events)),
            "panel_provenance": source_hashes,
        }
        all_inputs.append({"role": f"cohort:{checkpoint}", "path": str(cohort_path), "sha256": cohort_sha256, "checkpoint": checkpoint})
    pair_statuses_by_checkpoint = {
        checkpoint: _cohort_pair_statuses(loaded_cohorts[checkpoint][0], checkpoint)
        for checkpoint in CHECKPOINTS
    }
    hold_by_pair: dict[tuple[str, str, str], tuple[EligibleHoldSpec, dict[str, Any], dict[str, Any]]] = {}
    for hold in holds:
        if hold.checkpoint not in CHECKPOINTS:
            raise AggregationError(f"eligible HOLD has unknown checkpoint {hold.checkpoint!r}")
        _cohort, cohort_sha256, expected_events, _sources, prefix_bindings, cohort_path = loaded_cohorts[hold.checkpoint]
        pair, raw_event, payload = _validate_eligible_hold(
            hold,
            cohort_path=cohort_path,
            cohort_sha256=cohort_sha256,
            expected_events=expected_events,
            prefix_bindings=prefix_bindings,
            pair_statuses=pair_statuses_by_checkpoint[hold.checkpoint],
        )
        key = (hold.checkpoint, pair[0], pair[1])
        if key in hold_by_pair:
            raise AggregationError(f"duplicate eligible HOLD event {key!r}")
        hold_by_pair[key] = (hold, raw_event, payload)
        all_inputs.append(
            {
                "role": "eligible_hold_leaf",
                "path": str(hold.path),
                "sha256": sha256_file(hold.path),
                "checkpoint": hold.checkpoint,
                "shard": (hold.ordinal - 1) % SHARD_COUNT,
            }
        )
        hold_receipt_path = _resolve_path(payload.get("receipt_path"), "eligible HOLD receipt_path")
        all_inputs.append(
            {
                "role": "eligible_hold_receipt",
                "path": str(hold_receipt_path),
                "sha256": sha256_file(hold_receipt_path),
                "checkpoint": hold.checkpoint,
                "shard": (hold.ordinal - 1) % SHARD_COUNT,
            }
        )

    primary_by_pair: dict[tuple[str, str, str], tuple[ShardData, dict[str, Any]]] = {}
    overlays_by_pair: dict[tuple[str, str, str], list[ShardData]] = defaultdict(list)
    primary_pairs_by_key: dict[tuple[str, int], set[tuple[str, str]]] = defaultdict(set)
    mapping_receipts_by_checkpoint: dict[str, dict[str, dict[str, Any]]] = {checkpoint: {} for checkpoint in CHECKPOINTS}
    for checkpoint in CHECKPOINTS:
        for index in range(SHARD_COUNT):
            _cohort, cohort_sha256, expected_events, source_hashes, prefix_bindings, cohort_path = loaded_cohorts[checkpoint]
            canonical_pairs = set(expected_by_shard[(checkpoint, index)])
            for spec in by_key[(checkpoint, index, SHARD_COUNT)]:
                peek_identity = _read_json(
                    spec.path / "runtime_identity.json",
                    f"{spec.path}/runtime_identity.json",
                )
                if peek_identity.get("checkpoint") != checkpoint:
                    raise AggregationError(f"{spec.path}: checkpoint differs from CLI selector")
                declared_cohort_path = Path(
                    _text(peek_identity.get("cohort_path"), f"{spec.path}.cohort_path")
                ).expanduser().resolve()
                if declared_cohort_path != cohort_path:
                    raise AggregationError(f"{spec.path}: cohort path differs from --cohort")
                observed_pairs = _peek_shard_pairs(spec)
                if not set(observed_pairs).issubset(canonical_pairs):
                    raise AggregationError(
                        f"{spec.path}: event ownership crosses checkpoint:{index}/4 modulo partition"
                    )
                shard = _validate_shard(
                    spec,
                    observed_pairs,
                    cohort_path,
                    cohort_sha256,
                    source_hashes,
                    prefix_bindings,
                    pair_statuses_by_checkpoint[checkpoint],
                )
                shards.append(shard)
                for image_id, mapping_receipt in shard.mapping_receipts.items():
                    prior_mapping = mapping_receipts_by_checkpoint[checkpoint].get(image_id)
                    if prior_mapping is not None and prior_mapping != mapping_receipt:
                        raise AggregationError(
                            f"{checkpoint}: owner mapping receipt drifts across roots for image {image_id}"
                        )
                    mapping_receipts_by_checkpoint[checkpoint][image_id] = mapping_receipt
                all_inputs.extend(
                    {**item, "checkpoint": checkpoint, "shard": index, "source_kind": shard.source_kind}
                    for item in shard.input_files
                )
                identity = shard.identity
                checkpoint_provenance = provenance_by_checkpoint.setdefault(
                    checkpoint,
                    {
                        "config_path": identity["config_path"],
                        "config_sha256": identity["config_sha256"],
                        "resolved_config_fingerprint": identity["resolved_config_fingerprint"],
                        "panel_sha256": identity["panel_sha256"],
                        "cohort_sha256": identity["cohort_sha256"],
                        "panel_identity": identity["panel_identity"],
                    },
                )
                for provenance_key, value in checkpoint_provenance.items():
                    if identity.get(provenance_key) != value:
                        raise AggregationError(
                            f"{checkpoint}: shard {index} has inconsistent {provenance_key} provenance"
                        )
                for raw_event in shard.events:
                    event_id, image_id = _event_identity(raw_event, f"{spec.path}/event")
                    event_key = (checkpoint, event_id, image_id)
                    if shard.source_kind == "live_p4_overlay":
                        if checkpoint != "A":
                            raise AggregationError("P4-only repair overlay is admitted only for checkpoint A")
                        overlays_by_pair[event_key].append(shard)
                    else:
                        pair = (event_id, image_id)
                        if pair in primary_pairs_by_key[(checkpoint, index)] or event_key in primary_by_pair:
                            raise AggregationError(f"{checkpoint}: duplicate primary event across roots: {event_id}")
                        primary_pairs_by_key[(checkpoint, index)].add(pair)
                        primary_by_pair[event_key] = (shard, raw_event)

    hybrid = bool(holds) or len(specs) > len(CHECKPOINTS) * SHARD_COUNT or any(
        shard.source_kind != "live_all" for shard in shards
    )
    if hybrid:
        for checkpoint in CHECKPOINTS:
            if len(loaded_cohorts[checkpoint][2]) != 32:
                raise AggregationError(f"{checkpoint}: hybrid final topology requires exactly 32 cohort events")
    for checkpoint in CHECKPOINTS:
        for index in range(SHARD_COUNT):
            held_pairs = {
                (event_id, image_id)
                for hold_checkpoint, event_id, image_id in hold_by_pair
                if hold_checkpoint == checkpoint
                and (loaded_cohorts[checkpoint][2].index((event_id, image_id)) % SHARD_COUNT) == index
            }
            if primary_pairs_by_key[(checkpoint, index)] & held_pairs:
                raise AggregationError(f"{checkpoint}:{index}/4 has duplicate primary/HOLD coverage")
            coverage = primary_pairs_by_key[(checkpoint, index)] | held_pairs
            if coverage != set(expected_by_shard[(checkpoint, index)]):
                raise AggregationError(
                    f"{checkpoint}:{index}/4 has gaps or duplicate-free union drift; "
                    f"expected {expected_by_shard[(checkpoint, index)]!r}, observed {sorted(coverage)!r}"
                )
    for event_key, overlays in overlays_by_pair.items():
        if len(overlays) != 1:
            raise AggregationError(f"{event_key[0]}:{event_key[1]} has more than one P4 repair overlay")
        primary = primary_by_pair.get(event_key)
        if primary is None:
            raise AggregationError("P4 repair overlay has no all-stage base")
        if primary[0].source_kind != "live_all":
            raise AggregationError("P4 repair overlay base must be a validated GPU all-stage root")
        eligibility = _object(primary[1].get("eligibility"), "P4 repair base.eligibility")
        if eligibility.get("status") != "eligible" or eligibility.get("pair_status") != "verified_pair":
            raise AggregationError("P4 repair overlay base is not the eligible all-stage event")

    event_records_by_checkpoint: dict[str, list[dict[str, Any]]] = {checkpoint: [] for checkpoint in CHECKPOINTS}
    for checkpoint in CHECKPOINTS:
        for ordinal, pair in enumerate(loaded_cohorts[checkpoint][2], 1):
            event_key = (checkpoint, pair[0], pair[1])
            hold_entry = hold_by_pair.get(event_key)
            if hold_entry is not None:
                hold, raw_event, payload = hold_entry
                hold_reason = "pre_actuator_technical_failure_repair_exhausted"
                event_records_by_checkpoint[checkpoint].append(
                    {
                        "checkpoint": checkpoint,
                        "event_id": pair[0],
                        "image_id": pair[1],
                        "shard": {"index": (ordinal - 1) % SHARD_COUNT, "count": SHARD_COUNT, "path": str(hold.path)},
                        "source_kind": "eligible_hold",
                        "eligibility": raw_event["eligibility"],
                        "validity": {
                            stage: {
                                "status": "technical_invalid",
                                "reason": hold_reason,
                            }
                            for stage in STAGES
                        },
                        "observations": _event_observations(raw_event),
                        "hold": payload,
                    }
                )
                continue
            base_shard, base_event = primary_by_pair[event_key]
            raw_event = base_event
            repair_lineage = None
            overlay = overlays_by_pair.get(event_key)
            if overlay:
                raw_event = _merge_p4_repair(base_shard, overlay[0])
                repair_lineage = raw_event["p4_repair_lineage"]
            record = {
                "checkpoint": checkpoint,
                "event_id": pair[0],
                "image_id": pair[1],
                "shard": {
                    "index": (ordinal - 1) % SHARD_COUNT,
                    "count": SHARD_COUNT,
                    "path": str(base_shard.spec.path),
                },
                "source_kind": base_shard.source_kind,
                "validity": {
                    stage: {
                        "status": _status_from_record(raw_event.get(stage), context=f"event {pair[0]}.{stage}", kind="stage" if stage != "p4" else "p4")[0],
                        "reason": _status_from_record(raw_event.get(stage), context=f"event {pair[0]}.{stage}", kind="stage" if stage != "p4" else "p4")[1],
                    }
                    for stage in STAGES
                },
                "observations": _event_observations(raw_event),
            }
            if repair_lineage is not None:
                record["p4_repair_lineage"] = repair_lineage
                record["source_kind"] = "live_all_with_p4_repair"
            event_records_by_checkpoint[checkpoint].append(record)

    aggregates: dict[str, Any] = {}
    for checkpoint in CHECKPOINTS:
        aggregates.update(_aggregate_observations(event_records_by_checkpoint[checkpoint], checkpoint))
    event_records = [event for checkpoint in CHECKPOINTS for event in event_records_by_checkpoint[checkpoint]]
    input_hash = sha256_json(
        {
            "artifacts": sorted(
                all_inputs,
                key=lambda item: (item.get("checkpoint", ""), item.get("shard", -1), item["role"], item["path"]),
            )
        }
    )
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "status": "completed",
        "cohorts": cohort_output,
        "shard_plan": [
            {
                "checkpoint": shard.spec.checkpoint,
                "index": shard.spec.index,
                "count": shard.spec.count,
                "path": str(shard.spec.path),
                "source_kind": shard.source_kind,
            }
            for shard in sorted(shards, key=lambda item: (item.spec.checkpoint, item.spec.index, str(item.spec.path)))
        ],
        "eligible_holds": [
            {"checkpoint": spec.checkpoint, "ordinal": spec.ordinal, "path": str(spec.path)}
            for spec in sorted(holds, key=lambda item: (item.checkpoint, item.ordinal))
        ],
        "checkpoint_provenance": provenance_by_checkpoint,
        "inputs": {"sha256": input_hash, "artifacts": sorted(all_inputs, key=lambda item: (item.get("checkpoint", ""), item.get("shard", -1), item["role"], item["path"]))},
        "event_results": event_records,
        "aggregates": aggregates,
        "evidence_views": _evidence_views(aggregates),
        "hypotheses": {f"H{index}": None for index in range(1, 6)},
        "interpretation": None,
        "recommendation": None,
    }
    # ``self_sha256`` is defined over the canonical summary with that field
    # absent.  This avoids a circular digest while still making tampering
    # detectable from the published machine-readable file.
    summary["input_sha256"] = input_hash
    if receipt_path is not None:
        summary["receipt_path"] = str(Path(receipt_path).expanduser().resolve())
    summary["self_sha256"] = _summary_self_hash(summary)
    summary_sha256: str | None = None
    if resolved_output is not None:
        summary_sha256 = _atomic_write(resolved_output, summary)
    if receipt_path is not None:
        if summary_sha256 is None:
            raise AggregationError("receipt_path requires output_path so it can bind the output hash")
        receipt: dict[str, Any] = {
            "schema_version": f"{SCHEMA_VERSION}.receipt",
            "input_sha256": input_hash,
            "summary_path": str(resolved_output),
            "summary_sha256": summary_sha256,
            "summary_self_sha256": summary["self_sha256"],
            "inputs": summary["inputs"],
        }
        receipt["self_sha256"] = _summary_self_hash(receipt)
        assert resolved_receipt is not None
        _atomic_write(resolved_receipt, receipt)
    return summary


# ---------------------------------------------------------------------------
# Evidence bundle finalizer
# ---------------------------------------------------------------------------

def _read_payload(value: Any, context: str) -> tuple[Any, dict[str, Any] | None]:
    """Read a JSON object/array and return an immutable source reference.

    The shard summary already contains path/hash references, while unit tests
    and callers embedding a small synthetic receipt often pass a mapping
    directly.  Keeping this adapter local to the finalizer avoids coupling
    the historical ``aggregate_shards`` API to a new source abstraction.
    """

    if isinstance(value, (str, Path)):
        path = _resolve_path(value, context)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AggregationError(f"cannot read {context}: {path}: {exc}") from exc
        _reject_nonfinite(payload, context)
        return payload, {"path": str(path), "sha256": sha256_file(path)}
    if isinstance(value, Mapping) or isinstance(value, list):
        _reject_nonfinite(value, context)
        return value, None
    raise AggregationError(f"{context} must be a JSON path, object, or array")


def _source_descriptor(
    value: Any,
    *,
    context: str,
    expected_hash: str | None = None,
    expected_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve a ``{path, sha256}`` source descriptor and its JSON payload."""

    source = _object(value, context)
    path = _resolve_path(source.get("path"), f"{context}.path")
    declared_hash = _hash(source.get("sha256"), f"{context}.sha256")
    actual_hash = sha256_file(path)
    if actual_hash != declared_hash:
        raise AggregationError(f"{context} bytes do not match declared SHA-256")
    if expected_hash is not None and declared_hash != expected_hash:
        raise AggregationError(f"{context} hash differs from expected source")
    if expected_path is not None and path != expected_path:
        raise AggregationError(f"{context} path differs from expected source")
    payload, _ = _read_payload(path, context)
    return {"path": str(path), "sha256": declared_hash}, _object(payload, context)


def _summary_value(summary: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(summary, (str, Path)):
        return _read_json(_resolve_path(summary, "aggregate summary"), "aggregate summary")
    return _object(summary, "aggregate summary")


def _summary_artifact_refs(summary: Mapping[str, Any]) -> dict[tuple[str, int, str], dict[str, Any]]:
    refs: dict[tuple[str, int, str], dict[str, Any]] = {}
    inputs = _object(summary.get("inputs"), "aggregate summary.inputs")
    artifacts = _array(inputs.get("artifacts"), "aggregate summary.inputs.artifacts")
    for index, raw in enumerate(artifacts):
        item = _object(raw, f"aggregate summary.inputs.artifacts[{index}]")
        role = _text(item.get("role"), f"aggregate summary.inputs.artifacts[{index}].role")
        path = _resolve_path(item.get("path"), f"aggregate summary.inputs.artifacts[{index}].path")
        declared = _hash(item.get("sha256"), f"aggregate summary.inputs.artifacts[{index}].sha256")
        if sha256_file(path) != declared:
            raise AggregationError(f"aggregate summary input hash mismatch: {path}")
        checkpoint = item.get("checkpoint")
        shard = item.get("shard")
        if checkpoint not in CHECKPOINTS or isinstance(shard, bool) or not isinstance(shard, int):
            continue
        refs[(str(checkpoint), int(shard), role)] = {
            "role": role,
            "path": str(path),
            "sha256": declared,
            "checkpoint": str(checkpoint),
            "shard": int(shard),
        }
    return refs


def _bundle_shard_rows(summary: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[tuple[str, int], dict[str, Any]]]:
    """Load the owning base row for each finalized event, including HOLD/repair routing."""

    _summary_artifact_refs(summary)  # Re-hash every summary input before routing through event_results.
    plan_rows = _array(summary.get("shard_plan"), "aggregate summary.shard_plan")
    plan_keys: set[tuple[str, int]] = set()
    for index, raw_plan in enumerate(plan_rows):
        plan = _object(raw_plan, f"aggregate summary.shard_plan[{index}]")
        checkpoint = str(plan.get("checkpoint", "")).upper()
        shard_index = plan.get("index")
        if (
            checkpoint not in CHECKPOINTS
            or isinstance(shard_index, bool)
            or not isinstance(shard_index, int)
            or not 0 <= shard_index < SHARD_COUNT
            or plan.get("count") != SHARD_COUNT
        ):
            raise AggregationError("aggregate summary shard plan contains an unexpected selector")
        _resolve_path(plan.get("path"), f"aggregate summary.shard_plan[{index}].path")
        plan_keys.add((checkpoint, shard_index))
    if plan_keys != {(checkpoint, index) for checkpoint in CHECKPOINTS for index in range(SHARD_COUNT)}:
        raise AggregationError("aggregate summary shard plan is missing an S/A shard")

    def load_identity(root: Path) -> dict[str, Any]:
        event_path = root / "per_event_results.jsonl"
        exact_path = root / "exact_prefix_manifest.json"
        terminal_path = root / "terminal_summary.json"
        intervention_path = root / "intervention_manifest.json"
        gradient_path = root / "gradient_receipt.json"
        identity_path = root / "runtime_identity.json"
        for path in (event_path, exact_path, terminal_path, intervention_path, gradient_path, identity_path):
            if not path.is_file():
                raise AggregationError(f"shard is missing required provenance: {path}")
        return {
            "runtime": _read_json(identity_path, f"{root}/runtime_identity.json"),
            "exact_prefix": _read_json(exact_path, f"{root}/exact_prefix_manifest.json"),
            "terminal": _read_json(terminal_path, f"{root}/terminal_summary.json"),
            "refs": {
                "per_event_results": {"path": str(event_path), "sha256": sha256_file(event_path)},
                "runtime_identity": {"path": str(identity_path), "sha256": sha256_file(identity_path)},
                "exact_prefix_manifest": {"path": str(exact_path), "sha256": sha256_file(exact_path)},
                "terminal_summary": {"path": str(terminal_path), "sha256": sha256_file(terminal_path)},
                "intervention_manifest": {"path": str(intervention_path), "sha256": sha256_file(intervention_path)},
                "gradient_receipt": {"path": str(gradient_path), "sha256": sha256_file(gradient_path)},
            },
        }

    raw_events: list[dict[str, Any]] = []
    identities: dict[tuple[str, int], dict[str, Any]] = {}
    seen_events: set[tuple[str, str, str]] = set()
    for index, raw_result in enumerate(_array(summary.get("event_results"), "aggregate summary.event_results")):
        result = _object(raw_result, f"aggregate summary.event_results[{index}]")
        checkpoint = _text(result.get("checkpoint"), f"aggregate summary.event_results[{index}].checkpoint")
        event_id = _text(result.get("event_id"), f"aggregate summary.event_results[{index}].event_id")
        image_id = _text(result.get("image_id"), f"aggregate summary.event_results[{index}].image_id")
        shard = _object(result.get("shard"), f"aggregate summary.event_results[{index}].shard")
        shard_index = shard.get("index")
        if checkpoint not in CHECKPOINTS or isinstance(shard_index, bool) or not isinstance(shard_index, int):
            raise AggregationError("aggregate summary event contains a malformed shard selector")
        event_key = (checkpoint, event_id, image_id)
        if event_key in seen_events:
            raise AggregationError(f"aggregate summary repeats event {checkpoint}/{event_id}")
        seen_events.add(event_key)
        if result.get("source_kind") == "eligible_hold":
            leaf_path = _resolve_path(shard.get("path"), f"aggregate summary HOLD {checkpoint}/{event_id}.path")
            if not leaf_path.is_file() or _read_json(leaf_path, "eligible HOLD leaf") != result.get("hold"):
                raise AggregationError("aggregate summary HOLD payload differs from its sealed leaf")
            raw_events.append(
                {
                    "checkpoint": checkpoint,
                    "shard": {"index": shard_index, "count": SHARD_COUNT, "path": str(leaf_path)},
                    "event_id": event_id,
                    "image_id": image_id,
                    "hold": result["hold"],
                    "source_kind": "eligible_hold",
                }
            )
            continue
        root = _resolve_path(shard.get("path"), f"aggregate summary event {checkpoint}/{event_id}.path")
        if not root.is_dir():
            raise AggregationError(f"aggregate summary event root is not a directory: {root}")
        identity = load_identity(root)
        rows = _read_jsonl(root / "per_event_results.jsonl", f"{root}/per_event_results.jsonl")
        matches = [row for row in rows if _event_identity(row, f"{root}/per_event_results") == (event_id, image_id)]
        if len(matches) != 1:
            raise AggregationError(f"aggregate summary base root does not contain exactly one {checkpoint}/{event_id}")
        row = copy.deepcopy(matches[0])
        loaded: dict[str, Any] = {
            "checkpoint": checkpoint,
            "shard": {"index": shard_index, "count": SHARD_COUNT, "path": str(root)},
            "event_id": event_id,
            "image_id": image_id,
            "row": row,
            "identity": identity,
            "source_kind": result.get("source_kind"),
        }
        lineage = result.get("p4_repair_lineage")
        if lineage is not None:
            repair = _object(_object(lineage, "P4 repair lineage").get("repair"), "P4 repair lineage.repair")
            repair_root = _resolve_path(repair.get("path"), "P4 repair lineage.repair.path")
            repair_identity = load_identity(repair_root)
            if repair.get("runtime_identity_sha256") != repair_identity["refs"]["runtime_identity"]["sha256"]:
                raise AggregationError("P4 repair lineage runtime identity hash mismatch")
            repair_rows = _read_jsonl(repair_root / "per_event_results.jsonl", f"{repair_root}/per_event_results.jsonl")
            repair_matches = [candidate for candidate in repair_rows if _event_identity(candidate, "P4 repair row") == (event_id, image_id)]
            if len(repair_matches) != 1 or repair_matches[0].get("p4") != repair.get("p4"):
                raise AggregationError("P4 repair lineage row differs from the repair root")
            row["p4"] = copy.deepcopy(repair_matches[0]["p4"])
            loaded["repair_identity"] = repair_identity
            loaded["repair_shard"] = {"index": shard_index, "count": SHARD_COUNT, "path": str(repair_root)}
        identities[(checkpoint, shard_index)] = identity
        raw_events.append(loaded)
    return raw_events, identities


def _validate_bundle_partition(
    raw_events: Sequence[Mapping[str, Any]],
    *,
    cohort_payloads: Mapping[str, Any],
) -> None:
    """Re-establish exact modulo-four ownership from the admitted cohorts.

    The legacy summary is a byte-bound input receipt, but its ``event_results``
    can be recomputed after a shard mutation.  The cohort order is the sole
    partition authority for the deep evidence finalizer.
    """

    expected: dict[tuple[str, int], set[tuple[str, str]]] = {}
    for checkpoint in CHECKPOINTS:
        events = _array(
            _object(cohort_payloads[checkpoint], f"cohort payload {checkpoint}")["payload"].get("events"),
            f"cohort {checkpoint}.events",
        )
        for shard_index in range(SHARD_COUNT):
            expected[(checkpoint, shard_index)] = set()
        for ordinal, raw_event in enumerate(events):
            event = _object(raw_event, f"cohort {checkpoint}.events[{ordinal}]")
            event_id, image_id = _cohort_event_identity(event, ordinal)
            expected[(checkpoint, ordinal % SHARD_COUNT)].add((event_id, image_id))

    observed: dict[tuple[str, int], list[tuple[str, str]]] = defaultdict(list)
    for index, raw_loaded in enumerate(raw_events):
        loaded = _object(raw_loaded, f"raw bundle event[{index}]")
        checkpoint = _text(loaded.get("checkpoint"), f"raw bundle event[{index}].checkpoint")
        shard = _object(loaded.get("shard"), f"raw bundle event[{index}].shard")
        shard_index = shard.get("index")
        if checkpoint not in CHECKPOINTS or isinstance(shard_index, bool) or not isinstance(shard_index, int):
            raise AggregationError("raw evidence contains a malformed checkpoint/shard selector")
        key = (checkpoint, shard_index)
        if key not in expected:
            raise AggregationError("raw evidence contains a foreign checkpoint/shard selector")
        observed[key].append(
            (
                _text(loaded.get("event_id"), f"raw bundle event[{index}].event_id"),
                _text(loaded.get("image_id"), f"raw bundle event[{index}].image_id"),
            )
        )
    for key, expected_pairs in expected.items():
        observed_pairs = observed.get(key, [])
        if len(observed_pairs) != len(set(observed_pairs)):
            raise AggregationError(f"raw evidence shard {key[0]}:{key[1]}/4 repeats an event")
        if set(observed_pairs) != expected_pairs:
            raise AggregationError(
                f"raw evidence shard {key[0]}:{key[1]}/4 differs from exact cohort ownership"
            )


def _cohort_event_map(summary: Mapping[str, Any]) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any], dict[str, Any]]:
    cohorts = _object(summary.get("cohorts"), "aggregate summary.cohorts")
    event_map: dict[tuple[str, str], dict[str, Any]] = {}
    cohort_payloads: dict[str, Any] = {}
    source_hashes: dict[str, Any] = {}
    for checkpoint in CHECKPOINTS:
        cohort_meta = _object(cohorts.get(checkpoint), f"aggregate summary.cohorts.{checkpoint}")
        cohort_path = _resolve_path(cohort_meta.get("path"), f"aggregate summary.cohorts.{checkpoint}.path")
        cohort, cohort_hash, expected, sources, prefix_bindings = _load_cohort(cohort_path)
        if cohort_meta.get("sha256") != cohort_hash:
            raise AggregationError(f"aggregate summary cohort hash mismatch for {checkpoint}")
        for event in _array(cohort.get("events"), f"cohort {checkpoint}.events"):
            item = _object(event, f"cohort {checkpoint}.event")
            event_id, image_id = _cohort_event_identity(item, int(item.get("ordinal", 0) or 0))
            key = (checkpoint, event_id)
            if key in event_map:
                raise AggregationError(f"cohort repeats event {checkpoint}/{event_id}")
            event_map[key] = item
        cohort_payloads[checkpoint] = {
            "path": str(cohort_path),
            "sha256": cohort_hash,
            "event_count": len(expected),
            "payload": cohort,
            "prefix_bindings": prefix_bindings,
        }
        source_hashes[checkpoint] = sources
    return event_map, cohort_payloads, source_hashes


def _h0_record_identity(record: Mapping[str, Any], context: str) -> tuple[str, str]:
    event_id = record.get("gt_owner_id", record.get("event_id"))
    image_id = record.get("image_id")
    if event_id is None or image_id is None:
        raise AggregationError(f"{context} is missing gt_owner_id/image_id")
    return _text(event_id, f"{context}.gt_owner_id"), _text(image_id, f"{context}.image_id")


def _validate_h0_record(
    record: Mapping[str, Any],
    *,
    context: str,
    config_fingerprint: str,
    checkpoint: str,
    source_panel_sha256: str,
    derived_panel_sha256: str,
) -> None:
    if record.get("unit_id") != UNIT_ID or record.get("run_kind") != "native_h0" or record.get("history_complete") is not True:
        raise AggregationError(f"{context} is not a complete native H0 record")
    if record.get("checkpoint") != checkpoint:
        raise AggregationError(f"{context}.checkpoint disagrees with H0 ledger")
    if record.get("config_fingerprint") != config_fingerprint:
        raise AggregationError(f"{context}.config_fingerprint disagrees with H0 ledger")
    if record.get("source_panel_sha256") != source_panel_sha256 or record.get("derived_panel_sha256") != derived_panel_sha256:
        raise AggregationError(f"{context} panel provenance differs from the admitted source/derived panel")
    for key in ("native_tp", "native_fn", "strict_complete_row", "natural_boundary_valid"):
        if not isinstance(record.get(key), bool):
            raise AggregationError(f"{context}.{key} must be a boolean")
    if record.get("native_tp") == record.get("native_fn"):
        raise AggregationError(f"{context} must be exactly one of native TP/native FN")
    if record.get("natural_boundary_valid") is not True:
        raise AggregationError(f"{context}.natural_boundary_valid must be true")
    parse_status = record.get("parse_status")
    if parse_status not in {"accepted", "accepted_with_drops"}:
        raise AggregationError(f"{context}.parse_status is not a strict native parse: {parse_status!r}")
    stop = record.get("decode_stop_reason")
    if stop not in {"im_end", "eos", "terminal"} or record.get("excludes_stop") is not True:
        raise AggregationError(f"{context} lacks a strict native STOP/excludes_stop receipt")
    boundary = record.get("natural_boundary")
    if isinstance(boundary, bool) or not isinstance(boundary, int) or boundary < 0:
        raise AggregationError(f"{context}.natural_boundary must be a non-negative integer")
    if record.get("due_boundary_index") != boundary:
        raise AggregationError(f"{context}.due_boundary_index disagrees with natural_boundary")
    exact_hash = record.get("exact_prefix_sha256")
    token_ids = record.get("exact_prefix_token_ids")
    exact_hash = _hash(exact_hash, f"{context}.exact_prefix_sha256")
    token_ids = _array(token_ids, f"{context}.exact_prefix_token_ids")
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in token_ids):
        raise AggregationError(f"{context}.exact_prefix_token_ids are malformed")
    if sha256_json(token_ids) != exact_hash:
        raise AggregationError(f"{context} exact prefix hash disagrees with token IDs")
    declared_count = record.get("exact_prefix_token_count")
    if declared_count is not None and declared_count != len(token_ids):
        raise AggregationError(f"{context}.exact_prefix_token_count disagrees with token IDs")
    covered = _array(record.get("covered_owner_ids"), f"{context}.covered_owner_ids")
    if any(not isinstance(owner, str) or not owner for owner in covered) or len(set(covered)) != len(covered):
        raise AggregationError(f"{context}.covered_owner_ids are malformed or repeated")


def _discover_h0_attempt_lineage(ledger: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Bind nearby failed/repair attempts without admitting them as H0 data."""

    artifact_dir = ledger.get("artifact_dir")
    if not isinstance(artifact_dir, str) or not artifact_dir:
        return []
    try:
        root = Path(artifact_dir).expanduser().resolve(strict=True)
    except OSError:
        return []
    parent = root.parent
    attempts: list[dict[str, Any]] = []
    try:
        candidates = sorted(path for path in parent.iterdir() if path.is_dir())
    except OSError:
        return []
    for candidate in candidates:
        manifest_path = candidate / "run_manifest.json"
        summary_path = candidate / "summary.json"
        if not manifest_path.is_file() or not summary_path.is_file():
            continue
        try:
            manifest = _read_json(manifest_path, f"H0 attempt {candidate}/run_manifest.json")
            summary = _read_json(summary_path, f"H0 attempt {candidate}/summary.json")
        except AggregationError:
            continue
        attempts.append(
            {
                "path": str(candidate),
                "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
                "summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
                "terminal_status": summary.get("terminal_status", manifest.get("terminal_status")),
                "is_authoritative_h0": candidate == root,
            }
        )
    return attempts


def _collect_h0_baselines(
    *,
    cohort_payloads: Mapping[str, Any],
    source_hashes: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Retain one line per checkpoint/image and prove all 26 H0 baselines."""

    lines: list[dict[str, Any]] = []
    ledgers_by_checkpoint: dict[str, dict[str, Any]] = {}
    image_sets: dict[str, set[str]] = {}
    for checkpoint in CHECKPOINTS:
        sources = source_hashes[checkpoint]
        ledger_map = sources.get("h0_ledgers")
        if not isinstance(ledger_map, Mapping) or not ledger_map:
            raise AggregationError(f"{checkpoint}: H0 ledger provenance is missing")
        ledger_rows: list[dict[str, Any]] = []
        ledger_refs: list[dict[str, Any]] = []
        configs: set[str] = set()
        seen_owner_keys: set[tuple[str, str]] = set()
        for ledger_path_text, ledger_hash in ledger_map.items():
            ledger_path = _resolve_path(ledger_path_text, f"{checkpoint} H0 ledger path")
            if sha256_file(ledger_path) != ledger_hash:
                raise AggregationError(f"{checkpoint}: H0 ledger content hash mismatch")
            ledger = _read_json(ledger_path, f"{checkpoint} H0 ledger")
            if ledger.get("unit_id") != UNIT_ID or ledger.get("schema_version") != LEDGER_SCHEMA_VERSION:
                raise AggregationError(f"{checkpoint}: H0 ledger schema/unit mismatch")
            if str(ledger.get("checkpoint", "")).upper() != checkpoint or ledger.get("run_kind") != "native_h0" or ledger.get("history_complete") is not True:
                raise AggregationError(f"{checkpoint}: H0 ledger is not a complete native baseline")
            config = _text(ledger.get("config_fingerprint"), f"{checkpoint} H0 ledger.config_fingerprint")
            configs.add(config)
            records = _array(ledger.get("records"), f"{checkpoint} H0 ledger.records")
            for index, raw_record in enumerate(records):
                record = _object(raw_record, f"{checkpoint} H0 ledger.records[{index}]")
                _validate_h0_record(
                    record,
                    context=f"{checkpoint} H0 ledger.records[{index}]",
                    config_fingerprint=config,
                    checkpoint=checkpoint,
                    source_panel_sha256=str(sources.get("source_panel")),
                    derived_panel_sha256=str(sources.get("derived_panel")),
                )
                owner_key = _h0_record_identity(record, f"{checkpoint} H0 ledger.records[{index}]")
                if owner_key in seen_owner_keys:
                    raise AggregationError(f"{checkpoint}: H0 ledger repeats owner/image identity {owner_key[0]}/{owner_key[1]}")
                seen_owner_keys.add(owner_key)
                record = dict(record)
                record.setdefault("config_fingerprint", config)
                ledger_rows.append(record)
            ledger_refs.append({"path": str(ledger_path), "sha256": ledger_hash, "attempt_lineage": _discover_h0_attempt_lineage(ledger)})
        if len(configs) != 1:
            raise AggregationError(f"{checkpoint}: H0 config identity is inconsistent across ledgers")
        by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in ledger_rows:
            _event_id, image_id = _h0_record_identity(record, f"{checkpoint} H0 record")
            by_image[image_id].append(record)
        image_sets[checkpoint] = set(by_image)
        admitted_images = sources.get("source_panel_image_ids")
        if not isinstance(admitted_images, list) or len(admitted_images) != 13:
            raise AggregationError(f"{checkpoint}: admitted source-panel 13-image identity is missing")
        if image_sets[checkpoint] != set(admitted_images):
            raise AggregationError(
                f"{checkpoint}: H0 image set differs from the exact admitted source-panel image IDs"
            )
        cohort_images = {
            str(event.get("image_id"))
            for event in _array(cohort_payloads[checkpoint]["payload"].get("events"), f"cohort {checkpoint}.events")
        }
        if not cohort_images.issubset(image_sets[checkpoint]):
            raise AggregationError(f"{checkpoint}: cohort event image identity is absent from the 13-image H0 baseline set")
        ledger_identity = {
            "checkpoint": checkpoint,
            "config_fingerprint": next(iter(configs)),
            "source_panel_sha256": sources.get("source_panel"),
            "derived_panel_sha256": sources.get("derived_panel"),
            "ledger_refs": ledger_refs,
            "attempt_lineage": [
                attempt
                for ref in ledger_refs
                for attempt in ref.get("attempt_lineage", [])
            ],
        }
        ledgers_by_checkpoint[checkpoint] = ledger_identity
        for image_id in sorted(by_image, key=lambda value: (value == "2299", int(value) if value.isdigit() else value)):
            records = by_image[image_id]
            parse_statuses = sorted({record.get("parse_status") for record in records if record.get("parse_status") is not None})
            stop_reasons = sorted({record.get("decode_stop_reason") for record in records if record.get("decode_stop_reason") is not None})
            boundaries = sorted({record.get("natural_boundary") for record in records if record.get("natural_boundary") is not None})
            prefix_hashes = sorted({record.get("exact_prefix_sha256") for record in records if record.get("exact_prefix_sha256") is not None})
            token_counts = sorted({len(record["exact_prefix_token_ids"]) for record in records if isinstance(record.get("exact_prefix_token_ids"), list)})
            if not parse_statuses or not stop_reasons:
                raise AggregationError(f"{checkpoint} image {image_id}: parse/STOP identity is incomplete")
            line = {
                "schema_version": H0_BASELINE_EVIDENCE_SCHEMA_VERSION,
                "unit_id": UNIT_ID,
                "checkpoint": checkpoint,
                "image_id": image_id,
                "record_count": len(records),
                "parse": {
                    "statuses": parse_statuses,
                    "accepted_record_count": sum(record.get("parse_status") == "accepted" for record in records),
                    "malformed_record_count": sum(record.get("parse_status") in {"malformed", "parser_error", "all_spans_dropped", "empty"} for record in records),
                },
                "native_tp_count": sum(record["native_tp"] for record in records),
                "native_fn_count": sum(record["native_fn"] for record in records),
                "strict_complete_row_count": sum(record["strict_complete_row"] for record in records),
                "stop": {"reasons": stop_reasons, "record_count": len(records)},
                "boundary": {"natural_boundaries": boundaries, "valid_count": sum(record["natural_boundary_valid"] for record in records)},
                "token": {
                    "exact_prefix_token_counts": token_counts,
                    "generated_history_token_counts": sorted({record.get("generated_history_end_step") for record in records if record.get("generated_history_end_step") is not None}),
                },
                "prefix": {"exact_prefix_sha256": prefix_hashes, "record_hashes": sorted(sha256_json(record) for record in records)},
                "config_identity": ledger_identity,
                "raw_refs": ledger_refs,
            }
            line["content_sha256"] = sha256_json(line)
            lines.append(line)
    if image_sets["S"] != image_sets["A"]:
        raise AggregationError("S/A H0 image sets are not identical; the 26-baseline denominator is not established")
    if len(lines) != 26:
        raise AggregationError(f"expected 26 H0 baseline evidence rows, found {len(lines)}")
    return lines, ledgers_by_checkpoint


def _normalise_checkpoint_sources(
    values: Any,
    *,
    context: str,
) -> dict[str, list[Any]]:
    """Accept ``CHECKPOINT=PATH``-style mappings and plain source lists."""

    result: dict[str, list[Any]] = {checkpoint: [] for checkpoint in CHECKPOINTS}
    if values is None:
        return result
    if isinstance(values, Mapping):
        for checkpoint, raw in values.items():
            key = str(checkpoint).upper()
            if key not in CHECKPOINTS:
                raise AggregationError(f"{context} contains unknown checkpoint {checkpoint!r}")
            items = raw if isinstance(raw, (list, tuple)) else [raw]
            result[key].extend(items)
        return result
    if isinstance(values, (str, Path)):
        text = str(values)
        left, separator, right = text.partition("=")
        if separator and left.strip().upper() in CHECKPOINTS:
            result[left.strip().upper()].append(right.strip())
        else:
            raise AggregationError(f"{context} plain source requires CHECKPOINT=PATH")
        return result
    if isinstance(values, Sequence):
        for index, raw in enumerate(values):
            if isinstance(raw, (str, Path)):
                text = str(raw)
                left, separator, right = text.partition("=")
                if separator and left.strip().upper() in CHECKPOINTS:
                    result[left.strip().upper()].append(right.strip())
                    continue
            if isinstance(raw, Mapping):
                checkpoint = raw.get("checkpoint")
                if checkpoint is None:
                    raise AggregationError(f"{context}[{index}] lacks checkpoint")
                key = str(checkpoint).upper()
                if key not in CHECKPOINTS:
                    raise AggregationError(f"{context}[{index}] has unknown checkpoint")
                result[key].append(raw)
                continue
            raise AggregationError(f"{context}[{index}] is not a checkpoint source")
        return result
    raise AggregationError(f"{context} must be a mapping or sequence")


def _load_support_sources(
    *,
    cohort_sources: Mapping[str, Any],
    support_sources: Any,
    ledgers_by_checkpoint: Mapping[str, Mapping[str, Any]],
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[tuple[str, str, str, int, str], list[dict[str, Any]]],
    list[str],
]:
    """Validate final support-ledger provenance and index owner support."""

    rows_by_checkpoint: dict[str, list[dict[str, Any]]] = {checkpoint: [] for checkpoint in CHECKPOINTS}
    owner_support: dict[tuple[str, str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    missing: list[str] = []
    explicit = _normalise_checkpoint_sources(support_sources, context="support_ledgers")
    for checkpoint in CHECKPOINTS:
        cohort_source = cohort_sources[checkpoint]
        descriptors = cohort_source.get("support_ledgers", {})
        if isinstance(descriptors, Mapping):
            descriptor_values: list[Any] = [
                {"path": path, "sha256": digest}
                for path, digest in descriptors.items()
            ]
        elif isinstance(descriptors, Sequence) and not isinstance(descriptors, (str, bytes)):
            descriptor_values = list(descriptors)
        else:
            descriptor_values = []
        supplied = explicit[checkpoint]
        if descriptor_values and supplied:
            descriptor_paths = {
                str(_resolve_path(_object(value, "support descriptor").get("path"), "support descriptor.path"))
                for value in descriptor_values
            }
            supplied_paths = {
                str(_resolve_path(_object(value, "support source").get("path"), "support source.path"))
                if isinstance(value, Mapping)
                else str(_resolve_path(str(value).split("=", 1)[-1], "support source"))
                for value in supplied
            }
            if descriptor_paths != supplied_paths:
                raise AggregationError(f"{checkpoint}: explicit support ledgers differ from cohort.sources.support_ledgers")
        sources = descriptor_values or supplied
        for index, raw_source in enumerate(sources):
            if isinstance(raw_source, Mapping):
                descriptor = _object(raw_source, f"{checkpoint} support source[{index}]")
            else:
                descriptor = {"path": str(raw_source).split("=", 1)[-1]}
                descriptor["sha256"] = sha256_file(_resolve_path(descriptor["path"], f"{checkpoint} support source[{index}]"))
            path = _resolve_path(descriptor.get("path"), f"{checkpoint} support source[{index}].path")
            declared_hash = _hash(descriptor.get("sha256"), f"{checkpoint} support source[{index}].sha256")
            if sha256_file(path) != declared_hash:
                raise AggregationError(f"{checkpoint}: support ledger hash mismatch: {path}")
            payload, _ = _read_payload(path, f"{checkpoint} support ledger")
            envelope = _object(payload, f"{checkpoint} support ledger")
            if envelope.get("unit_id") != UNIT_ID:
                raise AggregationError(f"{checkpoint}: support ledger unit_id mismatch")
            source_checkpoint = envelope.get("checkpoint")
            if str(source_checkpoint).upper() != checkpoint:
                raise AggregationError(f"{checkpoint}: support ledger checkpoint mismatch")
            for key in ("source_panel_sha256", "derived_panel_sha256"):
                expected = cohort_source.get("source_panel" if key.startswith("source") else "derived_panel")
                if envelope.get(key) != expected:
                    raise AggregationError(f"{checkpoint}: support ledger {key} mismatch")
            h0_hash = envelope.get("h0_source_sha256")
            if h0_hash not in {ref["sha256"] for ref in ledgers_by_checkpoint[checkpoint]["ledger_refs"]}:
                raise AggregationError(f"{checkpoint}: support ledger is bound to an unadmitted H0 source")
            raw_rows = envelope.get("records", envelope.get("observations", envelope.get("contexts", [])))
            if not isinstance(raw_rows, list):
                raise AggregationError(f"{checkpoint}: support ledger rows must be an array")
            source_ref = {"path": str(path), "sha256": declared_hash}
            for row_index, raw_row in enumerate(raw_rows):
                row = _object(raw_row, f"{path}[{row_index}]")
                if row.get("unit_id") != UNIT_ID or row.get("checkpoint") != checkpoint:
                    raise AggregationError(f"{path}[{row_index}] checkpoint mismatch")
                for key, expected in (
                    ("source_panel_sha256", cohort_source.get("source_panel")),
                    ("derived_panel_sha256", cohort_source.get("derived_panel")),
                ):
                    if row.get(key) != expected:
                        raise AggregationError(f"{path}[{row_index}] {key} mismatch")
                image = _text(row.get("image_id"), f"{path}[{row_index}].image_id")
                owner = _text(row.get("gt_owner_id", row.get("event_id")), f"{path}[{row_index}].gt_owner_id")
                boundary = row.get("natural_boundary")
                if isinstance(boundary, bool) or not isinstance(boundary, int) or boundary < 0:
                    raise AggregationError(f"{path}[{row_index}].natural_boundary is malformed")
                prefix_hash = _hash(row.get("exact_prefix_sha256"), f"{path}[{row_index}].exact_prefix_sha256")
                prefix_ids = _array(row.get("exact_prefix_token_ids"), f"{path}[{row_index}].exact_prefix_token_ids")
                if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in prefix_ids):
                    raise AggregationError(f"{path}[{row_index}].exact_prefix_token_ids are malformed")
                if sha256_json(prefix_ids) != prefix_hash:
                    raise AggregationError(f"{path}[{row_index}] exact prefix hash/token IDs disagree")
                row = {**row, "_source_ref": source_ref, "_source_row_index": row_index}
                rows_by_checkpoint[checkpoint].append(row)
                owner_support[(checkpoint, image, owner, boundary, prefix_hash)].append(row)
    if not any(rows_by_checkpoint.values()):
        missing.append("support_ledgers")
    return rows_by_checkpoint, owner_support, missing


def _load_census_sources(
    census_sources: Any,
    *,
    cohort_payloads: Mapping[str, Any],
    source_hashes: Mapping[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    census_by_checkpoint: dict[str, list[dict[str, Any]]] = {checkpoint: [] for checkpoint in CHECKPOINTS}
    missing: list[str] = []
    normalized = _normalise_checkpoint_sources(census_sources, context="census")
    for checkpoint in CHECKPOINTS:
        for index, raw_source in enumerate(normalized[checkpoint]):
            if isinstance(raw_source, Mapping) and "path" in raw_source:
                descriptor = _object(raw_source, f"census.{checkpoint}[{index}]")
                path = _resolve_path(descriptor.get("path"), f"census.{checkpoint}[{index}].path")
                declared = _hash(descriptor.get("sha256"), f"census.{checkpoint}[{index}].sha256")
                if sha256_file(path) != declared:
                    raise AggregationError(f"census {checkpoint} content hash mismatch")
            else:
                path = _resolve_path(str(raw_source), f"census.{checkpoint}[{index}]")
                declared = sha256_file(path)
            expected_derived = source_hashes[checkpoint].get("derived_panel")
            cohort_images = {
                _text(event.get("image_id"), f"cohort {checkpoint}.event.image_id")
                for event in _array(cohort_payloads[checkpoint]["payload"].get("events"), f"cohort {checkpoint}.events")
            }
            receipt_ref: dict[str, Any] | None = None
            if path.suffix.lower() == ".jsonl":
                image_envelopes = _read_jsonl(path, f"census {checkpoint}")
                receipt_path = path.with_name("p1-receipt.json")
                if not receipt_path.is_file():
                    raise AggregationError(f"census {checkpoint} JSONL lacks adjacent p1-receipt.json")
                receipt = _read_json(receipt_path, f"census {checkpoint} receipt")
                if (
                    receipt.get("schema_version") != "static_dynamic_owner_observational_census.v1.receipt"
                    or receipt.get("unit_id") != UNIT_ID
                    or receipt.get("checkpoint") != checkpoint
                    or receipt.get("status") != "merged"
                    or receipt.get("technical_invalid_excluded") is not True
                    or receipt.get("shard_count") != SHARD_COUNT
                    or receipt.get("image_count") != 8
                    or receipt.get("p1_row_count") != 8
                    or receipt.get("p1_census_sha256") != declared
                    or len(image_envelopes) != 8
                ):
                    raise AggregationError(f"census {checkpoint} JSONL receipt contract mismatch")
                receipt_identity = _object(receipt.get("identity"), f"census {checkpoint} receipt.identity")
                receipt_cohort = _object(receipt_identity.get("cohort"), f"census {checkpoint} receipt.identity.cohort")
                receipt_panel = _object(receipt_identity.get("panel"), f"census {checkpoint} receipt.identity.panel")
                if receipt_cohort.get("sha256") != cohort_payloads[checkpoint]["sha256"]:
                    raise AggregationError(f"census {checkpoint} cohort identity mismatch")
                if receipt_panel.get("sha256") != expected_derived:
                    raise AggregationError(f"census {checkpoint} derived-panel identity mismatch")
                observed_images: list[str] = []
                rows: list[dict[str, Any]] = []
                for image_index, envelope in enumerate(image_envelopes):
                    if (
                        envelope.get("schema_version") != "static_dynamic_owner_observational_census.v1.p1"
                        or envelope.get("status") != "valid"
                        or envelope.get("checkpoint") != checkpoint
                    ):
                        raise AggregationError(f"census {checkpoint} JSONL row {image_index} identity/status mismatch")
                    image_id = _text(envelope.get("image_id"), f"census {checkpoint}[{image_index}].image_id")
                    if image_id not in cohort_images:
                        raise AggregationError(f"census {checkpoint} JSONL row {image_index} image is outside the cohort")
                    observed_images.append(image_id)
                    identity = _object(envelope.get("identity"), f"census {checkpoint}[{image_index}].identity")
                    for identity_key in ("cohort", "config", "h0", "panel", "runtime", "wrapper"):
                        if identity.get(identity_key) != receipt_identity.get(identity_key):
                            raise AggregationError(
                                f"census {checkpoint} JSONL row {image_index} {identity_key} identity mismatch"
                            )
                    runtime_attestation = _object(
                        envelope.get("runtime_attestation"),
                        f"census {checkpoint}[{image_index}].runtime_attestation",
                    )
                    if runtime_attestation.get("status") != "validated" or runtime_attestation.get("passed") is not True:
                        raise AggregationError(
                            f"census {checkpoint} JSONL row {image_index} runtime attestation is not validated"
                        )
                    capture = _object(
                        envelope.get("capture_receipt"),
                        f"census {checkpoint}[{image_index}].capture_receipt",
                    )
                    if (
                        capture.get("passed") is not True
                        or capture.get("cleanup_complete") is not True
                        or capture.get("forward_count") != 1
                    ):
                        raise AggregationError(
                            f"census {checkpoint} JSONL row {image_index} capture receipt is incomplete"
                        )
                    p1 = _object(envelope.get("p1_census"), f"census {checkpoint}[{image_index}].p1_census")
                    if (
                        p1.get("unit_id") != UNIT_ID
                        or p1.get("checkpoint") != checkpoint
                        or _text(p1.get("image_id"), f"census {checkpoint}[{image_index}].p1_census.image_id") != image_id
                    ):
                        raise AggregationError(f"census {checkpoint} JSONL row {image_index} inner census identity mismatch")
                    image_rows = [
                        _object(
                            item,
                            f"census {checkpoint}[{image_index}].p1_census.rows[{inner_index}]",
                        )
                        for inner_index, item in enumerate(
                            _array(p1.get("rows"), f"census {checkpoint}[{image_index}].p1_census.rows")
                        )
                    ]
                    layer_names = p1.get("layer_names")
                    if (
                        p1.get("owner_count") != 4
                        or not isinstance(layer_names, list)
                        or len(layer_names) != 31
                        or len(image_rows) != 124
                        or p1.get("no_efficacy_thresholds") is not True
                    ):
                        raise AggregationError(
                            f"census {checkpoint} JSONL row {image_index} has an incomplete 4-owner/31-layer matrix"
                        )
                    for inner_index, row in enumerate(image_rows):
                        if _text(
                            row.get("image_id"),
                            f"census {checkpoint}[{image_index}].p1_census.rows[{inner_index}].image_id",
                        ) != image_id:
                            raise AggregationError(
                                f"census {checkpoint} JSONL row {image_index} inner row image mismatch"
                            )
                    rows.extend(image_rows)
                receipt_images = [_text(value, f"census {checkpoint} receipt.image_ids") for value in _array(receipt.get("image_ids"), f"census {checkpoint} receipt.image_ids")]
                if observed_images != receipt_images or len(set(observed_images)) != 8:
                    raise AggregationError(f"census {checkpoint} JSONL image partition differs from its receipt")
                envelope = {"receipt": receipt, "image_envelopes": image_envelopes}
                receipt_ref = {"path": str(receipt_path), "sha256": sha256_file(receipt_path)}
            else:
                payload, _ = _read_payload(path, f"census {checkpoint}")
                envelope = _object(payload, f"census {checkpoint}")
                raw_rows = envelope.get("rows", envelope.get("observations", envelope.get("records", [])))
                rows = [
                    _object(item, f"census {checkpoint}.rows[{row_index}]")
                    for row_index, item in enumerate(
                        _array(raw_rows, f"census {checkpoint}.rows")
                    )
                ]
                if envelope.get("unit_id") != UNIT_ID:
                    raise AggregationError(f"census {checkpoint} unit_id mismatch")
                if envelope.get("checkpoint") != checkpoint:
                    raise AggregationError(f"census {checkpoint} checkpoint mismatch")
                identity = envelope.get("identity")
                identity = dict(identity) if isinstance(identity, Mapping) else {}
                cohort_identity = identity.get("cohort") if isinstance(identity.get("cohort"), Mapping) else {}
                panel_identity = identity.get("panel") if isinstance(identity.get("panel"), Mapping) else {}
                declared_cohort = envelope.get("cohort_sha256", cohort_identity.get("sha256"))
                declared_panel = envelope.get("derived_panel_sha256", panel_identity.get("sha256"))
                if declared_cohort != cohort_payloads[checkpoint]["sha256"]:
                    raise AggregationError(f"census {checkpoint} cohort identity mismatch")
                if declared_panel != expected_derived:
                    raise AggregationError(f"census {checkpoint} derived-panel identity mismatch")
                for row_index, row in enumerate(rows):
                    if row.get("checkpoint") != checkpoint:
                        raise AggregationError(f"census {checkpoint}.rows[{row_index}] checkpoint mismatch")
                    if _text(row.get("image_id"), f"census {checkpoint}.rows[{row_index}].image_id") not in cohort_images:
                        raise AggregationError(f"census {checkpoint}.rows[{row_index}] image is outside the cohort")
            if not rows:
                missing.append(f"observational_census.{checkpoint}:empty_rows")
            census_by_checkpoint[checkpoint].append(
                {
                    "path": str(path),
                    "sha256": declared,
                    "receipt": receipt_ref,
                    "row_count": len(rows),
                    "rows": rows,
                    "envelope": envelope,
                }
            )
    if not any(census_by_checkpoint.values()):
        missing.append("observational_census")
    return census_by_checkpoint, missing


_KNOWN_PAIR_STATUSES = {
    "verified_pair",
    "no_verified_B",
    "no_latest_covered_A",
    "indeterminate_missing_checkpoint_h0",
    "indeterminate_image2299_support_transfer",
    "indeterminate_no_valid_natural_boundary",
}


def _event_pair_contract(
    event: Mapping[str, Any],
    *,
    checkpoint: str,
    prefix: Mapping[str, Any],
    event_eligibility: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    pairs = _object(event.get("A_B"), "cohort event.A_B")
    pair = _object(pairs.get(checkpoint), f"cohort event.A_B.{checkpoint}")
    pair_status = pair.get("pair_status")
    if not isinstance(pair_status, str) or pair_status not in _KNOWN_PAIR_STATUSES:
        raise AggregationError(f"cohort event.A_B.{checkpoint}.pair_status is unknown: {pair_status!r}")
    target_owner = _cohort_target_owner(event, checkpoint, _text(event.get("gt_owner_id"), "cohort event.gt_owner_id"))
    prefix_eligibility = prefix.get("event_eligibility")
    if prefix_eligibility is None:
        prefix_eligibility = prefix.get("eligibility")
    if isinstance(prefix_eligibility, Mapping):
        declared_status = prefix_eligibility.get("pair_status")
        if declared_status is not None and declared_status != pair_status:
            raise AggregationError("prefix event_eligibility pair status differs from cohort")
        declared_target = prefix_eligibility.get("target_owner_id")
        if declared_target is not None and declared_target != target_owner:
            raise AggregationError("prefix event_eligibility target owner differs from cohort")
    if prefix.get("target_owner_id") is not None and prefix.get("target_owner_id") != target_owner:
        raise AggregationError("prefix target_owner_id differs from cohort target owner")
    b = pair.get("B_verified_uncovered")
    if pair_status == "verified_pair":
        if not isinstance(prefix_eligibility, Mapping) or prefix_eligibility.get("status") != "eligible":
            raise AggregationError("verified cohort pair lacks valid prefix eligibility")
        if not isinstance(event_eligibility, Mapping):
            raise AggregationError("verified cohort pair lacks scored event eligibility")
        if event_eligibility.get("status") != "eligible" or event_eligibility.get("pair_status") != "verified_pair":
            raise AggregationError("verified cohort pair has invalid scored event eligibility")
        if event_eligibility.get("actuators_called") is not True:
            raise AggregationError("verified cohort pair must attest actuators_called=true")
        if not isinstance(b, Mapping):
            raise AggregationError("verified cohort pair lacks B_verified_uncovered")
        if b.get("gt_owner_id") != target_owner or b.get("verified_support") is not True or b.get("strict_complete_row") is not False:
            raise AggregationError("verified cohort pair has inconsistent target support identity")
        a = pair.get("A_latest_covered")
        if not isinstance(a, Mapping):
            raise AggregationError("verified cohort pair lacks A_latest_covered")
        a_owner = a.get("gt_owner_id")
        if not isinstance(a_owner, str) or not a_owner or a_owner == target_owner:
            raise AggregationError("verified cohort pair has invalid/distinct A_latest_covered owner")
        if a.get("strict_complete_row") is not True:
            raise AggregationError("verified cohort pair A_latest_covered is not strict-complete")
        boundary = b.get("natural_boundary")
        if isinstance(boundary, Mapping):
            boundary = boundary.get("index", boundary.get("row_index", boundary.get("boundary_index")))
        if isinstance(boundary, bool) or not isinstance(boundary, int) or boundary < 0:
            raise AggregationError("verified cohort pair natural_boundary is malformed")
        a_boundary = a.get("natural_boundary")
        if isinstance(a_boundary, Mapping):
            a_boundary = a_boundary.get("index", a_boundary.get("row_index", a_boundary.get("boundary_index")))
        if isinstance(a_boundary, bool) or not isinstance(a_boundary, int) or a_boundary < 0 or a_boundary >= boundary:
            raise AggregationError("verified cohort pair A boundary must be strictly earlier than B")
        prefix_boundary = prefix.get("natural_boundary")
        if prefix_boundary != boundary:
            raise AggregationError("prefix natural_boundary differs from verified cohort pair")
        declared_prefix_hash = _hash(b.get("exact_prefix_sha256"), "verified cohort pair exact_prefix_sha256")
        h0 = _object(prefix.get("h0"), "verified cohort pair prefix.h0")
        if _hash(h0.get("exact_generated_history_prefix_sha256"), "verified cohort pair prefix.h0 exact hash") != declared_prefix_hash:
            raise AggregationError("P3/P1 prefix hash differs from verified cohort pair")
        covered = _array(prefix.get("covered_owner_ids"), "verified cohort pair prefix.covered_owner_ids")
        if a_owner not in covered or target_owner in covered:
            raise AggregationError("verified cohort pair A/B coverage state is inconsistent with exact prefix")
        owner_mapping = _object(prefix.get("owner_mapping"), "verified cohort pair prefix.owner_mapping")
        mapping_rows = _array(owner_mapping.get("source_to_derived"), "verified cohort pair prefix.owner_mapping.source_to_derived")
        panel_rows = {
            _text(row.get("owner_id"), "verified cohort pair owner mapping owner_id"): _object(row, "verified cohort pair owner mapping row")
            for row in mapping_rows
            if isinstance(row, Mapping)
        }
        if a_owner not in panel_rows or target_owner not in panel_rows:
            raise AggregationError("verified cohort pair owners are not panel-bound")
        source_index = a.get("source_panel_object_index")
        if isinstance(source_index, bool) or not isinstance(source_index, int) or source_index < 0:
            raise AggregationError("verified cohort pair A source-panel index is malformed")
        if panel_rows[a_owner].get("source_index") != source_index:
            raise AggregationError("verified cohort pair A source-panel identity disagrees with owner mapping")
    elif b is not None:
        raise AggregationError(f"cohort pair {pair_status} must not carry B_verified_uncovered")
    return {
        "pair_status": pair_status,
        "target_owner_id": target_owner,
        "natural_boundary": _cohort_target_boundary(event, checkpoint),
        "accepted_source_specific_match": pair_status == "verified_pair",
        "raw": pair,
    }


def _prefix_hashes(prefix: Mapping[str, Any]) -> set[str]:
    hashes: set[str] = set()
    for parent_key, child_key in (("h0", "exact_generated_history_prefix_sha256"), ("model_input", "prefix_sha256")):
        parent = prefix.get(parent_key)
        if isinstance(parent, Mapping) and parent.get(child_key) is not None:
            hashes.add(_hash(parent.get(child_key), f"prefix.{parent_key}.{child_key}"))
    for key in ("prefix_sha256", "prefix_token_ids_sha256", "exact_prefix_sha256"):
        if prefix.get(key) is not None:
            hashes.add(_hash(prefix.get(key), f"prefix.{key}"))
    return hashes


def _validate_probe_prefix(raw_probe: Mapping[str, Any], *, expected_hashes: set[str], context: str) -> None:
    candidates: list[Any] = []
    for key in ("prefix_token_ids_sha256", "prefix_sha256", "exact_prefix_sha256"):
        if raw_probe.get(key) is not None:
            candidates.append(raw_probe.get(key))
    for horizon_key in ("horizon_1", "horizon_3"):
        horizon = raw_probe.get(horizon_key)
        if isinstance(horizon, Mapping):
            for key in ("prefix_token_ids_sha256", "prefix_sha256", "exact_prefix_sha256"):
                if horizon.get(key) is not None:
                    candidates.append(horizon.get(key))
    for index, candidate in enumerate(candidates):
        digest = _hash(candidate, f"{context}.prefix_hash[{index}]")
        if expected_hashes and digest not in expected_hashes:
            raise AggregationError(f"{context} prefix hash differs from exact event prefix")


def _strict_nonnegative_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AggregationError(f"{context} must be a non-negative integer")
    return value


def _strict_endpoint_owner(
    raw_row: Any,
    *,
    context: str,
    checkpoint: str,
    image_id: str,
    target_owner: str,
    prefix_hashes: set[str],
) -> dict[str, Any]:
    """Validate one runner-native row/endpoint receipt and return its owner."""

    row = _object(raw_row, context)
    if row.get("complete_row") is not True:
        raise AggregationError(f"{context}.complete_row must be true")
    native = _object(row.get("native_parse"), f"{context}.native_parse")
    if native.get("valid") is not True or native.get("parse_status") != "accepted":
        raise AggregationError(f"{context}.native_parse is not a strict accepted row")
    match = _object(row.get("owner_match"), f"{context}.owner_match")
    if match.get("status") not in {"unique", "matched"}:
        raise AggregationError(f"{context}.owner_match is not unique/matched")
    owner = _text(match.get("owner_id", match.get("matched_owner_id")), f"{context}.owner_match.owner_id")
    if match.get("source_specific") is not True or match.get("physical_match") is not True:
        raise AggregationError(f"{context}.owner_match lacks source-specific physical identity")
    endpoint = _object(row.get("endpoint_evidence"), f"{context}.endpoint_evidence")
    if endpoint.get("status") != "measured" or endpoint.get("natural") is not True or endpoint.get("teacher_forced") is not False:
        raise AggregationError(f"{context}.endpoint_evidence is not a measured natural endpoint")
    strict_endpoint = _object(endpoint.get("strict_native_endpoint"), f"{context}.endpoint_evidence.strict_native_endpoint")
    if (
        strict_endpoint.get("status") != "accepted"
        or strict_endpoint.get("native_parse_status") != "accepted"
        or strict_endpoint.get("source_specific_physical_owner_match") is not True
        or strict_endpoint.get("owner_id") != owner
        or strict_endpoint.get("target_owner_id") != target_owner
    ):
        raise AggregationError(f"{context}.strict_native_endpoint identity is invalid")
    outcome = _object(endpoint.get("outcome"), f"{context}.endpoint_evidence.outcome")
    for key in ("valid_row", "duplicate", "unmatched", "ambiguous", "malformed"):
        if not isinstance(outcome.get(key), bool):
            raise AggregationError(f"{context}.endpoint_evidence.outcome.{key} must be boolean")
    if outcome.get("valid_row") is not True or any(outcome.get(key) for key in ("unmatched", "ambiguous", "malformed")):
        raise AggregationError(f"{context}.endpoint_evidence.outcome is not a strict valid row")
    generated_count = _strict_nonnegative_int(outcome.get("generated_token_count"), f"{context}.endpoint_evidence.outcome.generated_token_count")
    identity = _object(endpoint.get("identity_binding"), f"{context}.endpoint_evidence.identity_binding")
    if identity.get("checkpoint") != checkpoint or _text(identity.get("image_id"), f"{context}.identity_binding.image_id") != image_id:
        raise AggregationError(f"{context}.endpoint_evidence identity checkpoint/image mismatch")
    if identity.get("event_id") != target_owner:
        raise AggregationError(f"{context}.endpoint_evidence identity event mismatch")
    h0_hash = _hash(identity.get("h0_exact_prefix_sha256"), f"{context}.identity_binding.h0_exact_prefix_sha256")
    if prefix_hashes and h0_hash not in prefix_hashes:
        raise AggregationError(f"{context}.endpoint_evidence exact prefix hash mismatch")
    natural_hash = _hash(identity.get("natural_prefix_sha256"), f"{context}.identity_binding.natural_prefix_sha256")
    if prefix_hashes and natural_hash not in prefix_hashes:
        raise AggregationError(f"{context}.endpoint_evidence natural prefix hash mismatch")
    remaining = _object(
        endpoint.get("remaining_independently_verified_support_at_stop"),
        f"{context}.endpoint_evidence.remaining_independently_verified_support_at_stop",
    )
    if remaining.get("status") != "measured" or not isinstance(remaining.get("at_stop"), bool):
        raise AggregationError(f"{context} lacks measured later-boundary remaining-support evidence")
    remaining_ids = _array(remaining.get("owner_ids"), f"{context}.remaining_support.owner_ids")
    if any(not isinstance(value, str) or not value for value in remaining_ids) or len(set(remaining_ids)) != len(remaining_ids):
        raise AggregationError(f"{context}.remaining_support.owner_ids are malformed")
    if _strict_nonnegative_int(remaining.get("count"), f"{context}.remaining_support.count") != len(remaining_ids):
        raise AggregationError(f"{context}.remaining_support count mismatch")
    return {
        "owner_id": owner,
        "duplicate": outcome["duplicate"],
        "generated_token_count": generated_count,
        "remaining_support": remaining,
        "endpoint": endpoint,
    }


def _support_exhaustion(
    *,
    stop: Mapping[str, Any],
    endpoint_rows: Sequence[Mapping[str, Any]],
) -> tuple[bool, str | None]:
    """Use only the runner's later-boundary endpoint evidence for STOP.

    The external support ledger is an initial H0-boundary receipt.  It is
    intentionally excluded here because it cannot prove exhaustion after one
    or more newly generated rows.
    """

    if stop.get("stopped") is not True or not endpoint_rows:
        return False, None
    remaining = endpoint_rows[-1].get("remaining_support")
    if not isinstance(remaining, Mapping):
        return False, None
    if remaining.get("status") != "measured" or remaining.get("at_stop") is not True:
        return False, None
    owner_ids = remaining.get("owner_ids")
    count = remaining.get("count")
    if isinstance(owner_ids, list) and not owner_ids and count == 0:
        return True, "runner_endpoint_remaining_support_empty"
    return False, None


def _horizon_qualification(
    raw_horizon: Any,
    *,
    context: str,
    checkpoint: str,
    event_id: str,
    image_id: str,
    target_owner: str | None,
    covered_owner_ids: Sequence[str],
    expected_horizon: int,
    prefix_hashes: set[str],
    support_rows: Mapping[tuple[str, str, str, int, str], Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    status, reason = _status_from_record(raw_horizon, context=context, kind="horizon")
    if status != "valid" or not isinstance(raw_horizon, Mapping):
        return {
            "observation": {"validity": status, "reason": reason, "metrics": {}},
            "qualification": status,
            "horizon_complete": None,
            "stop": None,
            "missing_evidence": [f"{context}:{reason or status}"],
        }
    try:
        _validate_probe_prefix(raw_horizon, expected_hashes=prefix_hashes, context=context)
        rows = _array(raw_horizon.get("rows"), f"{context}.rows")
        row_count = _strict_nonnegative_int(raw_horizon.get("row_count"), f"{context}.row_count")
        bookkeeping = _object(raw_horizon.get("owner_bookkeeping"), f"{context}.owner_bookkeeping")
        requested = _strict_nonnegative_int(bookkeeping.get("horizon_requested"), f"{context}.owner_bookkeeping.horizon_requested")
        generated = _strict_nonnegative_int(bookkeeping.get("horizon_rows_generated"), f"{context}.owner_bookkeeping.horizon_rows_generated")
        if requested != expected_horizon or requested <= 0 or generated <= 0:
            raise AggregationError(f"{context} has missing/wrong/nonempty horizon requested/generated counts")
        bookkeeping_rows = _array(bookkeeping.get("rows"), f"{context}.owner_bookkeeping.rows")
        if generated != len(rows) or generated != len(bookkeeping_rows) or generated != row_count or generated > requested:
            raise AggregationError(f"{context} row counts disagree across runner/bookkeeping receipts")
        target = _text(target_owner, f"{context}.target_owner")
        endpoint_rows = [
            _strict_endpoint_owner(
                row,
                context=f"{context}.rows[{index}]",
                checkpoint=checkpoint,
                image_id=image_id,
                target_owner=target,
                prefix_hashes=prefix_hashes,
            )
            for index, row in enumerate(rows)
        ]
        for index, (endpoint_row, raw_bookkeeping_row) in enumerate(zip(endpoint_rows, bookkeeping_rows, strict=True)):
            book_row = _object(raw_bookkeeping_row, f"{context}.owner_bookkeeping.rows[{index}]")
            if (
                book_row.get("complete") is not True
                or book_row.get("parse_status") != "accepted"
                or book_row.get("owner_id") != endpoint_row["owner_id"]
                or book_row.get("duplicate") is not endpoint_row["duplicate"]
            ):
                raise AggregationError(f"{context}.owner_bookkeeping.rows[{index}] disagrees with endpoint receipt")
            for key in ("unmatched", "ambiguous", "malformed", "invalid"):
                if book_row.get(key) is not False:
                    raise AggregationError(f"{context}.owner_bookkeeping.rows[{index}].{key} must be false")
        base = {_text(owner, f"{context}.covered_owner_ids") for owner in covered_owner_ids}
        arm = {str(row["owner_id"]) for row in endpoint_rows}
        expected_g = sorted(arm - base)
        expected_k = sorted(arm & base)
        expected_l = sorted(base - arm)
        for key, expected_values in (("G", expected_g), ("K", expected_k), ("L", expected_l)):
            values = [_text(value, f"{context}.owner_bookkeeping.{key}") for value in _array(bookkeeping.get(key), f"{context}.owner_bookkeeping.{key}")]
            if values != expected_values:
                raise AggregationError(f"{context}.owner_bookkeeping.{key} disagrees with strict endpoint owners")
        expected_net = len(expected_g) - len(expected_l)
        if bookkeeping.get("net") != expected_net:
            raise AggregationError(f"{context}.owner_bookkeeping.net disagrees with strict G/L")
        parse = _object(bookkeeping.get("parse"), f"{context}.owner_bookkeeping.parse")
        expected_parse = {
            "valid_rows": len(endpoint_rows),
            "duplicate_rows": sum(bool(row["duplicate"]) for row in endpoint_rows),
            "unmatched_rows": 0,
            "ambiguous_rows": 0,
            "malformed_rows": 0,
            "invalid_rows": 0,
        }
        for key, expected_value in expected_parse.items():
            value = _strict_nonnegative_int(parse.get(key), f"{context}.owner_bookkeeping.parse.{key}")
            if value != expected_value:
                raise AggregationError(f"{context}.owner_bookkeeping.parse.{key} disagrees with endpoint rows")
        repeat = _object(bookkeeping.get("repeat_hazard"), f"{context}.owner_bookkeeping.repeat_hazard")
        repeat_metrics: dict[str, float] = {}
        for offset in range(1, 4):
            key = f"t+{offset}"
            value = _strict_nonnegative_int(repeat.get(key), f"{context}.owner_bookkeeping.repeat_hazard.{key}")
            expected_value = int(offset <= len(endpoint_rows) and endpoint_rows[offset - 1]["owner_id"] in base)
            if value != expected_value:
                raise AggregationError(f"{context}.owner_bookkeeping.repeat_hazard.{key} disagrees with endpoint rows")
            repeat_metrics[f"repeat_hazard.{key}"] = float(value)
        complete = bookkeeping.get("horizon_complete")
        if not isinstance(complete, bool) or complete != (generated >= requested):
            raise AggregationError(f"{context}.owner_bookkeeping.horizon_complete is inconsistent")
        if raw_horizon.get("status") != ("completed" if complete else "stopped_early"):
            raise AggregationError(f"{context}.status disagrees with horizon completion")
        stop = _object(bookkeeping.get("stop"), f"{context}.owner_bookkeeping.stop")
        if not isinstance(stop.get("stopped"), bool) or not isinstance(stop.get("stop_reason"), str) or not stop.get("stop_reason"):
            raise AggregationError(f"{context}.owner_bookkeeping.stop is malformed")
        exhausted, exhaustion_reason = _support_exhaustion(stop=stop, endpoint_rows=endpoint_rows)
        missing: list[str] = []
        if not complete and not exhausted:
            missing.append(f"{context}:verified_support_exhaustion")
        observation = {
            "validity": "valid",
            "reason": None,
            "owner_utility": {"G": expected_g, "K": expected_k, "L": expected_l, "net": expected_net},
            "delta": {"G": expected_g, "K": expected_k, "L": expected_l, "net": expected_net},
            "metrics": {
                "G_count": float(len(expected_g)),
                "K_count": float(len(expected_k)),
                "L_count": float(len(expected_l)),
                "net": float(expected_net),
                **repeat_metrics,
                **{f"parse.{key}": float(value) for key, value in expected_parse.items()},
            },
        }
        return {
            "observation": observation,
            "qualification": "qualified" if not missing else "indeterminate",
            "horizon_complete": complete,
            "stop": {"raw": stop, "support_exhausted": exhausted, "reason": exhaustion_reason},
            "missing_evidence": missing,
        }
    except AggregationError as exc:
        return {
            "observation": {"validity": "invalid", "reason": str(exc), "metrics": {}},
            "qualification": "technical_invalid",
            "horizon_complete": None,
            "stop": None,
            "missing_evidence": [f"{context}:{exc}"],
        }


def _p4_qualification(p4: Mapping[str, Any], *, context: str) -> tuple[dict[str, Any], list[str]]:
    """Validate CPU-visible P4 path/mass fields without loading a model."""

    status, reason = _status_from_record(p4, context=context, kind="p4")
    if status != "valid":
        return {"qualification": status, "reason": reason, "readiness": status, "missing_evidence": []}, []
    path_checks = _object(p4.get("path_checks"), f"{context}.path_checks")
    required = ("optimizer_used", "lm_head_only_path", "model_parameter_mutated", "parameter_grad_mutated", "audit_input_mutated", "visual_state_detached")
    missing: list[str] = []
    for key in required:
        if key not in path_checks:
            missing.append(f"{context}.path_checks.{key}")
    if missing:
        return {"qualification": "technical_invalid", "reason": "missing P4 path checks", "readiness": "hold", "missing_evidence": missing}, missing
    for key in required[:-1]:
        if not isinstance(path_checks.get(key), bool):
            raise AggregationError(f"{context}.path_checks.{key} must be boolean")
    if any(path_checks.get(key) is True for key in ("optimizer_used", "model_parameter_mutated", "parameter_grad_mutated", "audit_input_mutated", "lm_head_only_path")):
        raise AggregationError(f"{context} violates non-mutating/LM-head-only P4 path contract")
    if not isinstance(path_checks.get("visual_state_detached"), list) or path_checks.get("visual_state_detached"):
        raise AggregationError(f"{context}.path_checks.visual_state_detached is non-empty")
    # All numbers in a valid P4 receipt must be finite.  This is redundant
    # with JSON decoding for ordinary values, but catches a caller-provided
    # mapping before serialization as well.
    _reject_nonfinite(p4, context)
    objectives = _object(p4.get("objectives"), f"{context}.objectives")
    required_gradients = {
        "target_b_complete_row_nll": {"image_residual", "matched_background"},
        "uncovered_b_vs_covered_a_margin_loss": {"latest_terminal_carrier", "latest_row_span"},
        "fixed_sum_coupled": {"image_residual", "matched_background", "latest_terminal_carrier", "latest_row_span"},
    }
    for objective_name, required_states in required_gradients.items():
        objective = objectives.get(objective_name)
        if not isinstance(objective, Mapping):
            missing.append(f"{context}.objectives.{objective_name}")
            continue
        if objective.get("finite") is not True:
            missing.append(f"{context}.objectives.{objective_name}.finite")
        try:
            _finite(objective.get("value"), f"{context}.objectives.{objective_name}.value")
        except AggregationError:
            missing.append(f"{context}.objectives.{objective_name}.value")
        gradients = objective.get("gradients")
        if not isinstance(gradients, Mapping):
            missing.append(f"{context}.objectives.{objective_name}.gradients")
            continue
        for state_name in sorted(required_states):
            gradient = gradients.get(state_name)
            if not isinstance(gradient, Mapping):
                missing.append(f"{context}.objectives.{objective_name}.gradients.{state_name}")
                continue
            if gradient.get("present") is not True or gradient.get("finite") is not True:
                missing.append(f"{context}.objectives.{objective_name}.gradients.{state_name}:valid_finite")
                continue
            try:
                _finite(gradient.get("norm"), f"{context}.{objective_name}.{state_name}.norm")
                _finite(gradient.get("max_abs"), f"{context}.{objective_name}.{state_name}.max_abs")
            except AggregationError:
                missing.append(f"{context}.objectives.{objective_name}.gradients.{state_name}:finite_values")
    lm_head = p4.get("lm_head")
    if not isinstance(lm_head, Mapping):
        missing.append(f"{context}.lm_head")
    else:
        for objective_name in P4_OBJECTIVES:
            receipt = lm_head.get(objective_name)
            if not isinstance(receipt, Mapping):
                missing.append(f"{context}.lm_head.{objective_name}")
                continue
            if not isinstance(receipt.get("present"), bool) or receipt.get("finite") is not True:
                missing.append(f"{context}.lm_head.{objective_name}:valid_finite")
                continue
            if receipt.get("present") is True:
                try:
                    _finite(receipt.get("norm"), f"{context}.lm_head.{objective_name}.norm")
                except AggregationError:
                    missing.append(f"{context}.lm_head.{objective_name}.norm")
    raw_margin_objective = objectives.get("uncovered_b_vs_covered_a_margin_loss")
    margin_objective = dict(raw_margin_objective) if isinstance(raw_margin_objective, Mapping) else {}
    for key in ("reported_margin_higher_is_better", "reported_uncovered_b_mean_logprob", "reported_covered_a_mean_logprob"):
        if key not in margin_objective:
            missing.append(f"{context}.objectives.uncovered_b_vs_covered_a_margin_loss.{key}")
        else:
            _finite(margin_objective.get(key), f"{context}.objectives.uncovered_b_vs_covered_a_margin_loss.{key}")
    mass = p4.get("grammar_stop_invalid_mass")
    if mass is None:
        mass = p4.get("grammar-STOP-invalid-mass")
    if not isinstance(mass, Mapping):
        missing.append(f"{context}.grammar_stop_invalid_mass")
    else:
        for name in ("grammar", "stop", "invalid"):
            item = mass.get(name)
            if not isinstance(item, Mapping):
                missing.append(f"{context}.grammar_stop_invalid_mass.{name}")
                continue
            if item.get("status") not in {"reported", "measured", "valid"} or item.get("finite") is not True:
                missing.append(f"{context}.grammar_stop_invalid_mass.{name}:measured_finite")
                continue
            token_count = item.get("token_count")
            if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count <= 0:
                missing.append(f"{context}.grammar_stop_invalid_mass.{name}.token_count")
            for field in ("mean_probability", "max_probability", "min_probability"):
                try:
                    _finite(item.get(field), f"{context}.grammar_stop_invalid_mass.{name}.{field}")
                except AggregationError:
                    missing.append(f"{context}.grammar_stop_invalid_mass.{name}.{field}")
    non_target = p4.get("non_target_owner_effects")
    non_target_maxima: list[float] = []
    if not isinstance(non_target, Mapping):
        missing.append(f"{context}.non_target_owner_effects")
    else:
        for objective_name in P4_OBJECTIVES:
            effect = non_target.get(objective_name)
            if not isinstance(effect, Mapping) or effect.get("status") != "measured":
                missing.append(f"{context}.non_target_owner_effects.{objective_name}:measured")
                continue
            owners = effect.get("owners")
            if not isinstance(owners, Mapping) or not owners or effect.get("owner_count") != len(owners):
                missing.append(f"{context}.non_target_owner_effects.{objective_name}.owners")
                continue
            for owner_id, raw_gradient in owners.items():
                gradient = raw_gradient if isinstance(raw_gradient, Mapping) else {}
                if gradient.get("present") is not True or gradient.get("finite") is not True:
                    missing.append(f"{context}.non_target_owner_effects.{objective_name}.{owner_id}:valid_finite")
                    continue
                try:
                    non_target_maxima.append(abs(_finite(gradient.get("max_abs"), f"{context}.non_target.{objective_name}.{owner_id}.max_abs")))
                    _finite(gradient.get("norm"), f"{context}.non_target.{objective_name}.{owner_id}.norm")
                except AggregationError:
                    missing.append(f"{context}.non_target_owner_effects.{objective_name}.{owner_id}:finite_values")
    return {
        "qualification": "qualified" if not missing else "indeterminate",
        "reason": None if not missing else "P4 readiness evidence is incomplete",
        "readiness": "ready" if not missing else "hold",
        "missing_evidence": missing,
        "non_target_max_abs_delta": max(non_target_maxima) if non_target_maxima else None,
        "path_checks": path_checks,
        "mass": mass,
    }, missing


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical(row) + b"\n" for row in rows)


def _write_bytes_collision(path: Path, content: bytes) -> str:
    path = path.expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"evidence bundle output collision: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                Path(temporary).unlink()
            except OSError:
                pass
    return sha256_bytes(content)


def _bundle_split(image_id: str) -> str:
    return "image2299" if str(image_id) == "2299" else "legacy12"


def _qualification_for_status(status: str, *, event_eligible: bool) -> str:
    if not event_eligible:
        return "not_measured"
    if status == "valid":
        return "qualified"
    if status == "indeterminate":
        return "indeterminate"
    if status == "invalid":
        return "technical_invalid"
    return status


def _p1_bundle_rows(
    event: Mapping[str, Any],
    *,
    pair: Mapping[str, Any],
    context: str,
) -> tuple[dict[str, Any], list[str]]:
    stage = _object(event.get("p1"), f"{context}.p1")
    observations = _p1_observations(event)
    arms = _object(stage.get("arms"), f"{context}.p1.arms")
    eligible = bool(pair.get("accepted_source_specific_match"))
    target_owner = str(pair.get("target_owner_id"))
    prefix = _object(event.get("prefix"), f"{context}.prefix")
    prefix_hashes = _prefix_hashes(prefix)
    checkpoint = _text(event.get("checkpoint"), f"{context}.checkpoint")
    image_id = _text(event.get("image_id"), f"{context}.image_id")
    baseline: set[str] = set()
    if eligible:
        try:
            baseline.add(
                str(
                    _strict_endpoint_owner(
                        arms.get("K00"),
                        context=f"{context}.p1.arms.K00",
                        checkpoint=checkpoint,
                        image_id=image_id,
                        target_owner=target_owner,
                        prefix_hashes=prefix_hashes,
                    )["owner_id"]
                )
            )
        except AggregationError:
            # The individual K00 row below carries the explicit hold reason.
            pass
    missing: list[str] = []
    rows: dict[str, Any] = {}
    for probe in P1_PROBES:
        raw_arm = _object(arms.get(probe), f"{context}.p1.arms.{probe}")
        observation = observations.get(probe, {"validity": "invalid", "reason": "missing observation", "metrics": {}})
        strict_owner: str | None = None
        strict_reason: str | None = None
        if eligible:
            try:
                strict_owner = str(
                    _strict_endpoint_owner(
                        raw_arm,
                        context=f"{context}.p1.arms.{probe}",
                        checkpoint=checkpoint,
                        image_id=image_id,
                        target_owner=target_owner,
                        prefix_hashes=prefix_hashes,
                    )["owner_id"]
                )
            except AggregationError as exc:
                strict_reason = str(exc)
                missing.append(f"{context}.p1.{probe}:{exc}")
        owner_ids = [strict_owner] if strict_owner is not None else []
        valid = observation.get("validity") == "valid" and strict_owner is not None
        target_hit: bool | None
        if not eligible:
            target_hit = None
        elif valid:
            target_hit = strict_owner == target_owner and strict_owner not in baseline
        else:
            target_hit = None
        rows[probe] = {
            "raw": raw_arm,
            "observation": (
                observation
                if strict_reason is None
                else {"validity": "invalid", "reason": strict_reason, "metrics": {}}
            ),
            "qualification": "qualified" if valid and eligible else "technical_invalid" if eligible else "not_measured",
            "strict_complete_row": valid,
            "newly_covered_owner_ids": sorted(set(owner_ids) - baseline) if valid else [],
            "target_hit": target_hit,
            "scientific_status": (
                "not_measured"
                if target_hit is None
                else "measured_target_hit"
                if target_hit
                else "scientific_null"
            ),
        }
    return rows, missing


def _matrix_bundle_rows(
    event: Mapping[str, Any],
    *,
    stage_name: str,
    probes: Sequence[str],
    pair: Mapping[str, Any],
    support_rows: Mapping[tuple[str, str, str, int, str], Sequence[Mapping[str, Any]]],
    context: str,
) -> tuple[dict[str, Any], list[str]]:
    stage = _object(event.get(stage_name), f"{context}.{stage_name}")
    container = _object(stage.get("arms" if stage_name == "p2" else "cells"), f"{context}.{stage_name}")
    prefix = _object(event.get("prefix"), f"{context}.prefix")
    expected_hashes = _prefix_hashes(prefix)
    _validate_probe_prefix(stage, expected_hashes=expected_hashes, context=f"{context}.{stage_name}")
    eligible = bool(pair.get("accepted_source_specific_match"))
    target_owner = str(pair.get("target_owner_id"))
    image_id = str(event.get("image_id"))
    checkpoint = _text(event.get("checkpoint"), f"{context}.checkpoint")
    event_id = _text(event.get("event_id"), f"{context}.event_id")
    covered_owner_ids = [
        _text(owner, f"{context}.prefix.covered_owner_ids")
        for owner in _array(prefix.get("covered_owner_ids"), f"{context}.prefix.covered_owner_ids")
    ]
    output: dict[str, Any] = {}
    missing: list[str] = []
    for probe in probes:
        raw_probe = _object(container.get(probe), f"{context}.{stage_name}.{probe}")
        _validate_probe_prefix(raw_probe, expected_hashes=expected_hashes, context=f"{context}.{stage_name}.{probe}")
        output[probe] = {}
        for horizon in (1, 3):
            horizon_key = f"horizon_{horizon}"
            raw_horizon = raw_probe.get(horizon_key)
            if raw_horizon is None and raw_probe.get("status") in {"not_applicable", "indeterminate", "invalid/uninterpretable", "technical_invalid", "invalid"}:
                raw_horizon = raw_probe
            if not eligible:
                output[probe][horizon_key] = {
                    "raw": raw_horizon,
                    "observation": {"validity": "not_measured", "reason": "event pair is not actuator-eligible", "metrics": {}},
                    "qualification": "not_measured",
                    "horizon_complete": None,
                    "strict_complete_row_count": None,
                    "stop": None,
                    "newly_covered_owner_ids": [],
                    "target_hit": None,
                    "scientific_status": "not_measured",
                }
                continue
            qualification = _horizon_qualification(
                raw_horizon,
                context=f"{context}.{stage_name}.{probe}.{horizon_key}",
                checkpoint=checkpoint,
                event_id=event_id,
                image_id=image_id,
                target_owner=target_owner,
                covered_owner_ids=covered_owner_ids,
                expected_horizon=horizon,
                prefix_hashes=expected_hashes,
                support_rows=support_rows,
            )
            missing.extend(qualification.get("missing_evidence", []))
            observation = qualification["observation"]
            utility = observation.get("owner_utility", {}) if isinstance(observation, Mapping) else {}
            gained = utility.get("G", []) if isinstance(utility, Mapping) else []
            target_hit: bool | None = None
            if eligible and qualification.get("qualification") == "qualified":
                target_hit = target_owner in set(gained)
            output[probe][horizon_key] = {
                "raw": raw_horizon,
                "observation": observation,
                "qualification": qualification.get("qualification") if eligible else "not_measured",
                "horizon_complete": qualification.get("horizon_complete"),
                "strict_complete_row_count": observation.get("metrics", {}).get("parse.valid_rows") if isinstance(observation.get("metrics"), Mapping) else None,
                "stop": qualification.get("stop"),
                "newly_covered_owner_ids": list(gained) if isinstance(gained, list) else [],
                "target_hit": target_hit,
                "scientific_status": (
                    "not_measured"
                    if target_hit is None
                    else "measured_target_hit"
                    if target_hit
                    else "scientific_null"
                ),
            }
    return output, missing


def _p3_bundle_rows(
    event: Mapping[str, Any],
    *,
    pair: Mapping[str, Any],
    support_rows: Mapping[tuple[str, str, str, int, str], Sequence[Mapping[str, Any]]],
    context: str,
) -> tuple[dict[str, Any], list[str]]:
    rows, missing = _matrix_bundle_rows(
        event,
        stage_name="p3",
        probes=P3_CELLS,
        pair=pair,
        support_rows=support_rows,
        context=context,
    )
    eligible = bool(pair.get("accepted_source_specific_match"))
    for horizon in (1, 3):
        horizon_key = f"horizon_{horizon}"
        cell_rows = {
            cell: rows.get(cell, {}).get(horizon_key, {})
            for cell in P3_CELLS
        }
        if eligible and all(row.get("qualification") == "qualified" for row in cell_rows.values()):
            values = {
                cell: float(cell_rows[cell]["observation"]["metrics"]["net"])
                for cell in P3_CELLS
            }
            metrics = {
                "Y00_net": values["Y00"],
                "Y10_net": values["Y10"],
                "Y01_net": values["Y01"],
                "Y11_net": values["Y11"],
                "Delta_static": values["Y10"] - values["Y00"],
                "Delta_dynamic": values["Y01"] - values["Y00"],
                "tau": (values["Y11"] - values["Y10"]) - (values["Y01"] - values["Y00"]),
            }
            crossover = {"validity": "valid", "reason": None, "metrics": metrics, "delta": {key: metrics[key] for key in ("Delta_static", "Delta_dynamic", "tau")}}
        else:
            reason = "one or more strict P3 cell endpoints are unqualified"
            crossover = {"validity": "indeterminate" if eligible else "not_measured", "reason": reason, "metrics": {}}
            if eligible:
                missing.append(f"{context}.p3.crossover.{horizon_key}:{reason}")
        endpoint = {
            "observation": crossover,
            "qualification": "qualified" if eligible and crossover.get("validity") == "valid" else _qualification_for_status(str(crossover.get("validity")), event_eligible=eligible),
            "deltas": crossover.get("metrics", {}),
            "scientific_status": "not_measured" if not eligible or crossover.get("validity") != "valid" else "scientific_null" if all(float(crossover.get("metrics", {}).get(name, 0.0)) == 0.0 for name in ("Delta_static", "Delta_dynamic", "tau")) else "measured",
        }
        # If an implementation also emits a precomputed endpoint, require its
        # arithmetic to agree with the independently recomputed four cells.
        declared = None
        stage = _object(event.get("p3"), f"{context}.p3")
        declared_key = None
        for key in (horizon_key, f"factorial_horizon_{horizon}"):
            if isinstance(stage.get(key), Mapping):
                declared = stage[key]
                declared_key = key
                break
        if isinstance(declared, Mapping) and crossover.get("validity") == "valid":
            for name in ("Delta_static", "Delta_dynamic", "tau"):
                if declared.get(name) is not None and not math.isclose(_finite(declared.get(name), f"{context}.p3.{declared_key}.{name}"), float(crossover["metrics"][name]), rel_tol=0.0, abs_tol=1e-9):
                    raise AggregationError(f"{context}.p3 {name} arithmetic disagrees with cell endpoints")
        rows[f"crossover.horizon_{horizon}"] = endpoint
    return rows, missing


def _derive_bundle_event(
    loaded: Mapping[str, Any],
    *,
    cohort_event: Mapping[str, Any],
    support_rows: Mapping[tuple[str, str, str, int, str], Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, Any], list[str]]:
    checkpoint = str(loaded["checkpoint"])
    event_id = str(loaded["event_id"])
    image_id = str(loaded["image_id"])
    raw = _object(loaded["row"], f"raw event {checkpoint}/{event_id}")
    prefix = _object(raw.get("prefix"), f"raw event {checkpoint}/{event_id}.prefix")
    raw_eligibility = raw.get("eligibility")
    pair = _event_pair_contract(
        cohort_event,
        checkpoint=checkpoint,
        prefix=prefix,
        event_eligibility=raw_eligibility if isinstance(raw_eligibility, Mapping) else None,
    )
    identity = _object(loaded["identity"], f"raw event {checkpoint}/{event_id}.identity")
    runtime = _object(identity.get("runtime"), f"raw event {checkpoint}/{event_id}.runtime_identity")
    exact_manifest = _object(identity.get("exact_prefix"), f"raw event {checkpoint}/{event_id}.exact_prefix_manifest")
    terminal = _object(identity.get("terminal"), f"raw event {checkpoint}/{event_id}.terminal_summary")
    _assert_identity_match(exact_manifest.get("identity", {}), runtime, f"raw event {checkpoint}/{event_id}.exact_prefix.identity")
    _assert_identity_match(terminal, runtime, f"raw event {checkpoint}/{event_id}.terminal_summary")
    if terminal.get("status") != "completed":
        raise AggregationError(f"raw event {checkpoint}/{event_id}.terminal_summary is not completed")
    exact_rows = {
        str(item.get("event_id")): item
        for item in _array(exact_manifest.get("events"), f"raw event {checkpoint}/{event_id}.exact_prefix.events")
        if isinstance(item, Mapping)
    }
    exact_row = exact_rows.get(event_id)
    if exact_row is None or exact_row.get("prefix") != prefix:
        raise AggregationError(f"raw event {checkpoint}/{event_id} exact-prefix identity differs from per-event prefix")
    missing: list[str] = []
    pair_status = pair.get("pair_status")
    eligible = pair_status == "verified_pair"
    observations = _event_observations(raw)
    p1_rows, p1_missing = _p1_bundle_rows(raw, pair=pair, context=f"event {checkpoint}/{event_id}")
    missing.extend(p1_missing)
    p2_rows, p2_missing = _matrix_bundle_rows(raw, stage_name="p2", probes=P2_ARMS, pair=pair, support_rows=support_rows, context=f"event {checkpoint}/{event_id}")
    missing.extend(p2_missing)
    p3_rows, p3_missing = _p3_bundle_rows(raw, pair=pair, support_rows=support_rows, context=f"event {checkpoint}/{event_id}")
    missing.extend(p3_missing)
    p4 = _object(raw.get("p4"), f"event {checkpoint}/{event_id}.p4")
    p4_readiness, p4_missing = _p4_qualification(p4, context=f"event {checkpoint}/{event_id}.p4")
    if not eligible:
        # H0-only ineligible live-smoke leaves deliberately contain the full
        # matrix shape with zero actuator calls.  Preserve that receipt as a
        # completed non-scored matrix, never as a technical or scientific
        # negative.
        p4_readiness = {
            **p4_readiness,
            "qualification": "not_measured",
            "readiness": "not_measured",
            "missing_evidence": [],
        }
        p4_missing = []
    p4_readiness = {
        **p4_readiness,
        "scientific_status": "not_measured" if not eligible else "measured" if p4_readiness.get("readiness") == "ready" else "indeterminate",
    }
    missing.extend(p4_missing)
    stage_status = {
        stage: _status_from_record(raw.get(stage), context=f"event {checkpoint}/{event_id}.{stage}", kind="stage" if stage != "p4" else "p4")
        for stage in STAGES
    }
    event_eligibility = raw.get("eligibility", prefix.get("event_eligibility"))
    if isinstance(event_eligibility, Mapping) and pair_status != event_eligibility.get("pair_status", pair_status):
        raise AggregationError(f"event {checkpoint}/{event_id} event_eligibility pair status mismatch")
    stage_readiness: dict[str, dict[str, Any]] = {}
    if eligible:
        p1_unqualified = sorted(name for name, row in p1_rows.items() if row.get("qualification") != "qualified")
        p2_unqualified = sorted(
            f"{probe}.{horizon}"
            for probe, horizons in p2_rows.items()
            for horizon, row in horizons.items()
            if row.get("qualification") != "qualified"
        )
        p3_unqualified = sorted(
            name if name.startswith("crossover.") else f"{name}.{horizon}"
            for name, value in p3_rows.items()
            for horizon, row in (
                (("", value),)
                if name.startswith("crossover.")
                else value.items()
            )
            if row.get("qualification") != "qualified"
        )
        unqualified_by_stage = {
            "p1": p1_unqualified,
            "p2": p2_unqualified,
            "p3": p3_unqualified,
            "p4": [] if p4_readiness.get("readiness") == "ready" else ["path/objective/mass/non-target readiness"],
        }
        for stage, unqualified in unqualified_by_stage.items():
            stage_state, stage_reason = stage_status[stage]
            reasons = list(unqualified)
            if stage_state != "valid":
                reasons.append(stage_reason or stage_state)
            stage_readiness[stage] = {"status": "ready" if not reasons else "hold", "reasons": reasons}
            missing.extend(f"event {checkpoint}/{event_id}.{stage}:{reason}" for reason in reasons)
    else:
        stage_readiness = {
            stage: {"status": "not_measured", "reasons": [f"pair_status={pair_status}"]}
            for stage in STAGES
        }
    raw_refs = copy.deepcopy(_object(identity.get("refs"), f"raw event {checkpoint}/{event_id}.identity.refs"))
    repair_identity = loaded.get("repair_identity")
    if repair_identity is not None:
        repair_refs = _object(
            _object(repair_identity, f"raw event {checkpoint}/{event_id}.repair_identity").get("refs"),
            f"raw event {checkpoint}/{event_id}.repair_identity.refs",
        )
        for role, ref in repair_refs.items():
            key = f"p4_repair_overlay.{role}"
            if key in raw_refs:
                raise AggregationError(f"raw event {checkpoint}/{event_id} duplicate repair reference {key}")
            raw_refs[key] = copy.deepcopy(ref)
    result = {
        "schema_version": RAW_EVENT_EVIDENCE_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "checkpoint": checkpoint,
        "event_id": event_id,
        "image_id": image_id,
        "cohort": {
            "pair_status": pair_status,
            "target_owner_id": pair["target_owner_id"],
            "natural_boundary": pair.get("natural_boundary"),
            "accepted_source_specific_match": eligible,
            "pair": pair["raw"],
        },
        "prefix": prefix,
        "event_eligibility": event_eligibility,
        "checkpoint_pair_status": pair["raw"],
        "actuators_called": (
            event_eligibility.get("actuators_called")
            if isinstance(event_eligibility, Mapping) and event_eligibility.get("actuators_called") is not None
            else False
            if not eligible
            else None
        ),
        "stage_status": {stage: {"status": status, "reason": reason} for stage, (status, reason) in stage_status.items()},
        "stage_readiness": stage_readiness,
        "matrix_status": "complete_non_scored" if not eligible else "scored_candidate",
        "p1": {"rows": p1_rows, "raw": _object(raw.get("p1"), f"event {checkpoint}/{event_id}.p1")},
        "p2": {"rows": p2_rows, "raw": _object(raw.get("p2"), f"event {checkpoint}/{event_id}.p2")},
        "p3": {"rows": p3_rows, "raw": _object(raw.get("p3"), f"event {checkpoint}/{event_id}.p3")},
        "p4": {"raw": p4, "readiness": p4_readiness},
        "observations": observations,
        "runtime_identity": runtime,
        "terminal_identity": terminal,
        "exact_prefix_identity": {"manifest_identity": exact_manifest.get("identity"), "event": exact_row},
        "raw_refs": raw_refs,
        "shard": loaded["shard"],
        "missing_evidence": sorted(set(missing)),
    }
    result["raw_event_sha256"] = sha256_json(raw)
    result["content_sha256"] = sha256_json(result)
    return result, missing


def _derive_hold_bundle_event(
    loaded: Mapping[str, Any],
    *,
    cohort_event: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    checkpoint = _text(loaded.get("checkpoint"), "eligible HOLD checkpoint")
    event_id = _text(loaded.get("event_id"), "eligible HOLD event_id")
    image_id = _text(loaded.get("image_id"), "eligible HOLD image_id")
    leaf = _object(loaded.get("hold"), f"eligible HOLD {checkpoint}/{event_id}")
    pairs = _object(cohort_event.get("A_B"), "eligible HOLD cohort A_B")
    pair = _object(pairs.get(checkpoint), f"eligible HOLD cohort A_B.{checkpoint}")
    if pair.get("pair_status") != "verified_pair":
        raise AggregationError("eligible HOLD bundle event is not cohort-proven verified_pair")
    target_owner = _cohort_target_owner(cohort_event, checkpoint, event_id)
    if target_owner != event_id:
        raise AggregationError("eligible HOLD target owner differs from the cohort")
    reason = "pre_actuator_technical_failure_repair_exhausted"

    def invalid_row() -> dict[str, Any]:
        return {
            "observation": {"validity": "invalid", "reason": reason, "metrics": {}},
            "qualification": "technical_invalid",
            "scientific_status": "indeterminate",
            "target_hit": False,
            "newly_covered_owner_ids": [],
        }

    p1_rows = {probe: invalid_row() for probe in P1_PROBES}
    p2_rows = {
        probe: {f"horizon_{horizon}": invalid_row() for horizon in (1, 3)}
        for probe in P2_ARMS
    }
    p3_rows: dict[str, Any] = {
        probe: {f"horizon_{horizon}": invalid_row() for horizon in (1, 3)}
        for probe in P3_CELLS
    }
    p3_rows.update({f"crossover.horizon_{horizon}": invalid_row() for horizon in (1, 3)})
    observations = {
        "p1": {probe: {"validity": "invalid", "reason": reason, "metrics": {}} for probe in P1_PROBES},
        "p2": {f"{probe}.horizon_{horizon}": {"validity": "invalid", "reason": reason, "metrics": {}} for probe in P2_ARMS for horizon in (1, 3)},
        "p3": {
            **{f"{probe}.horizon_{horizon}": {"validity": "invalid", "reason": reason, "metrics": {}} for probe in P3_CELLS for horizon in (1, 3)},
            **{f"crossover.horizon_{horizon}": {"validity": "invalid", "reason": reason, "metrics": {}} for horizon in (1, 3)},
        },
        "p4": {objective: {"validity": "invalid", "reason": reason, "metrics": {}} for objective in P4_OBJECTIVES},
    }
    p4_readiness = {
        "qualification": "technical_invalid",
        "readiness": "hold",
        "scientific_status": "indeterminate",
        "missing_evidence": [reason],
    }
    leaf_path = _resolve_path(_object(loaded.get("shard"), "eligible HOLD shard").get("path"), "eligible HOLD leaf path")
    receipt_path = _resolve_path(leaf.get("receipt_path"), "eligible HOLD receipt path")
    missing = [f"event {checkpoint}/{event_id}.{stage}:{reason}" for stage in STAGES]
    result: dict[str, Any] = {
        "schema_version": RAW_EVENT_EVIDENCE_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "checkpoint": checkpoint,
        "event_id": event_id,
        "image_id": image_id,
        "cohort": {
            "pair_status": "verified_pair",
            "target_owner_id": target_owner,
            "natural_boundary": _cohort_target_boundary(cohort_event, checkpoint),
            "accepted_source_specific_match": True,
            "pair": pair,
        },
        "prefix": {"status": "sealed_h0_binding_only", "h0_binding": leaf.get("h0_binding")},
        "event_eligibility": {
            "status": "eligible",
            "pair_status": "verified_pair",
            "actuators_called": None,
            "disposition": "eligible_pre_actuator_technical_hold",
        },
        "checkpoint_pair_status": pair,
        "actuators_called": None,
        "stage_status": {stage: {"status": "technical_invalid", "reason": reason} for stage in STAGES},
        "stage_readiness": {stage: {"status": "hold", "reasons": [reason]} for stage in STAGES},
        "matrix_status": "eligible_pre_actuator_hold",
        "p1": {"rows": p1_rows, "raw": leaf["matrix_dispositions"]["p1"]},
        "p2": {"rows": p2_rows, "raw": leaf["matrix_dispositions"]["p2"]},
        "p3": {"rows": p3_rows, "raw": leaf["matrix_dispositions"]["p3"]},
        "p4": {"raw": leaf["matrix_dispositions"]["p4"], "readiness": p4_readiness},
        "observations": observations,
        "runtime_identity": None,
        "terminal_identity": None,
        "exact_prefix_identity": {"sealed_hold": True, "h0_binding": leaf.get("h0_binding")},
        "raw_refs": {
            "eligible_hold_leaf": {"path": str(leaf_path), "sha256": sha256_file(leaf_path)},
            "eligible_hold_receipt": {"path": str(receipt_path), "sha256": sha256_file(receipt_path)},
        },
        "shard": loaded["shard"],
        "missing_evidence": missing,
    }
    result["raw_event_sha256"] = leaf.get("self_sha256")
    result["content_sha256"] = sha256_json(result)
    return result, missing


def _build_evidence_readiness(
    *,
    by_checkpoint: Mapping[str, Sequence[Mapping[str, Any]]],
    census_by_checkpoint: Mapping[str, Sequence[Mapping[str, Any]]],
    missing_evidence: Sequence[str],
) -> tuple[dict[str, Any], list[str]]:
    """Derive per-stage HOLD/ready without interpreting H1--H5."""

    all_missing = list(missing_evidence)
    stage_reasons: dict[str, dict[str, list[str]]] = {
        stage: {checkpoint: [] for checkpoint in CHECKPOINTS}
        for stage in STAGES
    }
    for checkpoint in CHECKPOINTS:
        events = list(by_checkpoint.get(checkpoint, ()))
        eligible_events = [
            event
            for event in events
            if isinstance(event.get("cohort"), Mapping)
            and event["cohort"].get("accepted_source_specific_match") is True
        ]
        if not eligible_events:
            for stage in STAGES:
                stage_reasons[stage][checkpoint].append("no actuator-eligible event evidence")
        for event in eligible_events:
            for stage in STAGES:
                state = event.get("stage_readiness", {}).get(stage, {})
                if not isinstance(state, Mapping) or state.get("status") != "ready":
                    reasons = state.get("reasons") if isinstance(state, Mapping) and isinstance(state.get("reasons"), list) else ["unqualified stage"]
                    stage_reasons[stage][checkpoint].extend(
                        f"{event.get('event_id')}:{reason}" for reason in reasons
                    )
        census_row_count = sum(
            item.get("row_count", 0)
            for item in census_by_checkpoint.get(checkpoint, ())
            if isinstance(item, Mapping) and isinstance(item.get("row_count", 0), int)
        )
        if census_row_count <= 0:
            stage_reasons["p1"][checkpoint].append("checkpoint/cohort-bound observational census has no rows")
    for stage in STAGES:
        for checkpoint in CHECKPOINTS:
            all_missing.extend(
                f"readiness.{stage}.{checkpoint}:{reason}"
                for reason in stage_reasons[stage][checkpoint]
            )
    missing_unique = sorted(set(str(item) for item in all_missing if item))
    per_stage = {
        stage: {
            checkpoint: {
                "status": "ready" if not stage_reasons[stage][checkpoint] else "hold",
                "reasons": sorted(set(stage_reasons[stage][checkpoint])),
            }
            for checkpoint in CHECKPOINTS
        }
        for stage in STAGES
    }
    readiness = {
        "overall": (
            "ready"
            if not missing_unique
            and all(per_stage[stage][checkpoint]["status"] == "ready" for stage in STAGES for checkpoint in CHECKPOINTS)
            else "hold"
        ),
        "missing_evidence": missing_unique,
        **per_stage,
    }
    return readiness, missing_unique


def build_evidence_bundle(
    summary: Mapping[str, Any] | str | Path | None = None,
    *,
    aggregate_summary: Mapping[str, Any] | str | Path | None = None,
    summary_path: str | Path | None = None,
    aggregate_path: str | Path | None = None,
    cohort_paths: Mapping[str, str | Path] | None = None,
    shard_specs: Sequence[ShardSpec] | None = None,
    output_path: str | Path | None = None,
    bundle_path: str | Path | None = None,
    bundle_output_path: str | Path | None = None,
    synthesis_inputs_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    raw_event_evidence_path: str | Path | None = None,
    h0_baseline_evidence_path: str | Path | None = None,
    receipt_path: str | Path | None = None,
    census_paths: Any = None,
    observational_census: Any = None,
    support_ledger_paths: Any = None,
    support_ledgers: Any = None,
) -> dict[str, Any]:
    """Finalize additive, CPU-only evidence artifacts for the frozen unit.

    ``summary`` may be the in-memory result of :func:`aggregate_shards` or a
    previously written ``static_dynamic_owner_interface_shard_aggregation.v1``
    file.  Supplying ``cohort_paths`` and ``shard_specs`` instead invokes the
    existing validator without writing its legacy summary.  The function only
    reads JSON/JSONL and computes deterministic hashes; it never imports a
    model runtime or calls a runner.
    """

    summary_aliases = [value for value in (summary, aggregate_summary, summary_path, aggregate_path) if value is not None]
    if len(summary_aliases) > 1:
        raise AggregationError("pass one aggregate summary input")
    source_summary = summary_aliases[0] if summary_aliases else None
    if source_summary is None:
        if cohort_paths is None or shard_specs is None:
            raise AggregationError("build_evidence_bundle requires an aggregate summary or cohort/shard inputs")
        source_summary = aggregate_shards(cohort_paths=cohort_paths, shard_specs=shard_specs)
    summary_value = _summary_value(source_summary)
    if summary_value.get("schema_version") != SCHEMA_VERSION or summary_value.get("unit_id") != UNIT_ID or summary_value.get("status") != "completed":
        raise AggregationError("evidence bundle requires a completed compatible shard aggregation summary")
    declared_summary_hash = summary_value.get("self_sha256")
    if declared_summary_hash is not None:
        _hash(declared_summary_hash, "aggregate summary.self_sha256")
        if declared_summary_hash != _summary_self_hash(summary_value):
            raise AggregationError("aggregate summary self hash mismatch")

    event_map, cohort_payloads, cohort_sources = _cohort_event_map(summary_value)
    for checkpoint in CHECKPOINTS:
        if "derived_receipt" not in cohort_sources[checkpoint]:
            raise AggregationError(f"{checkpoint}: cohort.sources.derived_receipt is required for evidence finalization")
    raw_events, _identities = _bundle_shard_rows(summary_value)
    _validate_bundle_partition(raw_events, cohort_payloads=cohort_payloads)

    # Re-run the runtime identity boundary for the deep finalizer.  A summary
    # file is an input receipt, not authority to skip its shard identities.
    for loaded in raw_events:
        if loaded.get("source_kind") == "eligible_hold":
            # The sealed HOLD explicitly has no event/runtime/terminal
            # artifact.  Its leaf+receipt were revalidated by aggregation and
            # _bundle_shard_rows; never borrow an identity from another event.
            continue
        checkpoint = str(loaded["checkpoint"])
        shard = _object(loaded["shard"], f"raw event {checkpoint}/{loaded['event_id']}.shard")
        shard_path = _resolve_path(shard.get("path"), "raw event shard.path")
        shard_index = shard.get("index")
        if isinstance(shard_index, bool) or not isinstance(shard_index, int):
            raise AggregationError("raw event shard index is malformed")
        _validate_runtime_identity(
            _object(loaded["identity"].get("runtime"), "raw event runtime identity"),
            ShardSpec(checkpoint, shard_index, SHARD_COUNT, shard_path),
            Path(cohort_payloads[checkpoint]["path"]),
            str(cohort_payloads[checkpoint]["sha256"]),
            cohort_sources[checkpoint],
        )
        prefix_bindings = cohort_payloads[checkpoint].get("prefix_bindings", {})
        raw_row = _object(loaded["row"], "raw event")
        prefix = _object(raw_row.get("prefix"), "raw event.prefix")
        _validate_prefix_receipt(
            prefix,
            spec=ShardSpec(checkpoint, shard_index, SHARD_COUNT, shard_path),
            event_id=str(loaded["event_id"]),
            image_id=str(loaded["image_id"]),
            expected=prefix_bindings.get((str(loaded["event_id"]), str(loaded["image_id"]))),
        )
        repair_identity = loaded.get("repair_identity")
        if isinstance(repair_identity, Mapping):
            repair_shard = _object(loaded.get("repair_shard"), "P4 repair shard")
            repair_path = _resolve_path(repair_shard.get("path"), "P4 repair shard.path")
            _validate_runtime_identity(
                _object(repair_identity.get("runtime"), "P4 repair runtime identity"),
                ShardSpec(checkpoint, shard_index, SHARD_COUNT, repair_path),
                Path(cohort_payloads[checkpoint]["path"]),
                str(cohort_payloads[checkpoint]["sha256"]),
                cohort_sources[checkpoint],
            )

    h0_lines, h0_identity = _collect_h0_baselines(cohort_payloads=cohort_payloads, source_hashes=cohort_sources)
    support_input = support_ledger_paths if support_ledger_paths is not None else support_ledgers
    support_rows_by_checkpoint, support_owner_rows, support_missing = _load_support_sources(
        cohort_sources=cohort_sources,
        support_sources=support_input,
        ledgers_by_checkpoint=h0_identity,
    )
    census_input = census_paths if census_paths is not None else observational_census
    census_by_checkpoint, census_missing = _load_census_sources(
        census_input,
        cohort_payloads=cohort_payloads,
        source_hashes=cohort_sources,
    )

    raw_lines: list[dict[str, Any]] = []
    all_missing: list[str] = [*support_missing, *census_missing]
    by_checkpoint: dict[str, list[dict[str, Any]]] = {checkpoint: [] for checkpoint in CHECKPOINTS}
    for loaded in sorted(raw_events, key=lambda item: (str(item["checkpoint"]), str(item["event_id"]))):
        key = (str(loaded["checkpoint"]), str(loaded["event_id"]))
        cohort_event = event_map.get(key)
        if cohort_event is None:
            raise AggregationError(f"raw event {key[0]}/{key[1]} has no cohort identity")
        if loaded.get("source_kind") == "eligible_hold":
            evidence, missing = _derive_hold_bundle_event(loaded, cohort_event=cohort_event)
        else:
            evidence, missing = _derive_bundle_event(
                loaded,
                cohort_event=cohort_event,
                support_rows=support_owner_rows,
            )
        raw_lines.append(evidence)
        by_checkpoint[key[0]].append(evidence)
        all_missing.extend(missing)

    # Denominators are deliberately event-level and disjoint.  Numeric target
    # hits are only counted for an accepted source-specific ``verified_pair``;
    # ineligible H0-only rows therefore contribute to the denominator but do
    # not become negative actuator outcomes.
    denominators: dict[str, dict[str, Any]] = {}
    qualifications: dict[str, Any] = {checkpoint: {} for checkpoint in CHECKPOINTS}
    for checkpoint in CHECKPOINTS:
        events = by_checkpoint[checkpoint]
        split_values = {
            "all": events,
            "legacy12": [event for event in events if _bundle_split(str(event["image_id"])) == "legacy12"],
            "image2299": [event for event in events if _bundle_split(str(event["image_id"])) == "image2299"],
        }
        if len(split_values["all"]) != len(split_values["legacy12"]) + len(split_values["image2299"]):
            raise AggregationError(f"{checkpoint}: legacy12/image2299 splits are not exhaustive and disjoint")
        denominators[checkpoint] = {name: {"events_total": len(rows), "accepted_source_specific_pairs": sum(event["cohort"]["accepted_source_specific_match"] for event in rows), "ineligible_events": sum(not event["cohort"]["accepted_source_specific_match"] for event in rows)} for name, rows in split_values.items()}
        for stage in STAGES:
            qualifications[checkpoint][stage] = {}
            names: Sequence[str]
            if stage == "p1":
                names = P1_PROBES
            elif stage == "p2":
                names = tuple(f"{probe}.horizon_{horizon}" for probe in P2_ARMS for horizon in (1, 3))
            elif stage == "p3":
                names = tuple(f"{probe}.horizon_{horizon}" for probe in P3_CELLS for horizon in (1, 3)) + ("crossover.horizon_1", "crossover.horizon_3")
            else:
                names = (*P4_OBJECTIVES, "path_checks")
            for name in names:
                qualifications[checkpoint][stage][name] = {}
                for split, rows in split_values.items():
                    observations: list[Mapping[str, Any]] = []
                    target_hits: list[bool] = []
                    newly_covered: set[str] = set()
                    validity_counts = Counter()
                    readiness_counts = Counter()
                    scientific_counts = Counter()
                    for event in rows:
                        if stage == "p4":
                            if name == "path_checks":
                                row = event["p4"]["readiness"]
                            else:
                                row = event["observations"].get("p4", {}).get(name, {})
                            row = row if isinstance(row, Mapping) else {}
                            validity = row.get("validity", row.get("qualification", "invalid"))
                            readiness = event["p4"]["readiness"].get("readiness", "hold")
                        elif stage == "p1":
                            row = event["p1"]["rows"].get(name, {})
                            validity = row.get("qualification", "technical_invalid")
                            readiness = validity
                            if row.get("target_hit") is True:
                                target_hits.append(True)
                            newly_covered.update(row.get("newly_covered_owner_ids", []))
                        elif stage == "p2":
                            probe, horizon = name.rsplit(".", 1)
                            row = event["p2"]["rows"].get(probe, {}).get(horizon, {})
                            validity = row.get("qualification", "technical_invalid")
                            readiness = validity
                            if row.get("target_hit") is True:
                                target_hits.append(True)
                            newly_covered.update(row.get("newly_covered_owner_ids", []))
                        else:
                            if name.startswith("crossover."):
                                row = event["p3"]["rows"].get(name, {})
                            else:
                                probe, horizon = name.rsplit(".", 1)
                                row = event["p3"]["rows"].get(probe, {}).get(horizon, {})
                            validity = row.get("qualification", "technical_invalid")
                            readiness = validity
                            newly_covered.update(row.get("newly_covered_owner_ids", []))
                        validity_counts[str(validity)] += 1
                        readiness_counts[str(readiness)] += 1
                        if isinstance(row, Mapping) and row.get("scientific_status") is not None:
                            scientific_counts[str(row.get("scientific_status"))] += 1
                        if isinstance(row, Mapping) and isinstance(row.get("observation"), Mapping):
                            observations.append(row["observation"])
                    qualifications[checkpoint][stage][name][split] = {
                        "events_total": len(rows),
                        "valid_count": validity_counts.get("qualified", validity_counts.get("valid", 0)),
                        "not_measured_count": validity_counts.get("not_measured", 0),
                        "indeterminate_count": validity_counts.get("indeterminate", 0),
                        "technical_invalid_count": validity_counts.get("technical_invalid", validity_counts.get("invalid", 0)),
                        "target_hit_count": len(target_hits) if any(event["cohort"]["accepted_source_specific_match"] for event in rows) else None,
                        "newly_covered_owner_ids": sorted(newly_covered),
                        "validity_counts": dict(sorted(validity_counts.items())),
                        "readiness_counts": dict(sorted(readiness_counts.items())),
                        "scientific_status_counts": dict(sorted(scientific_counts.items())),
                        "observations": observations,
                    }

    pooled_events = [event for checkpoint in CHECKPOINTS for event in by_checkpoint[checkpoint]]
    pooled_splits = {
        "all": pooled_events,
        "legacy12": [event for event in pooled_events if _bundle_split(str(event["image_id"])) == "legacy12"],
        "image2299": [event for event in pooled_events if _bundle_split(str(event["image_id"])) == "image2299"],
    }
    denominators["pooled"] = {
        name: {
            "events_total": len(rows),
            "accepted_source_specific_pairs": sum(event["cohort"]["accepted_source_specific_match"] for event in rows),
            "ineligible_events": sum(not event["cohort"]["accepted_source_specific_match"] for event in rows),
        }
        for name, rows in pooled_splits.items()
    }

    # Readiness is evidence-facing only.  No H1--H5 label or training action
    # is inferred here; absent census/STOP/P4 pass-through evidence remains an
    # explicit hold and the interpretation slot stays user-owned.
    readiness, missing_unique = _build_evidence_readiness(
        by_checkpoint=by_checkpoint,
        census_by_checkpoint=census_by_checkpoint,
        missing_evidence=all_missing,
    )

    raw_bytes = _jsonl_bytes(raw_lines)
    h0_bytes = _jsonl_bytes(h0_lines)
    bundle_aliases = [value for value in (bundle_path, bundle_output_path, synthesis_inputs_path, output_path) if value is not None]
    if len(bundle_aliases) > 1:
        raise AggregationError("pass one evidence bundle output path")
    base_target = bundle_aliases[0] if bundle_aliases else None
    if base_target is None:
        if output_dir is not None:
            base_dir = Path(output_dir).expanduser().resolve()
        elif isinstance(source_summary, (str, Path)):
            base_dir = _resolve_path(source_summary, "aggregate summary").parent
        else:
            base_dir = Path.cwd().resolve()
        base_target = base_dir / "evidence_bundle.json"
    bundle_target = Path(base_target).expanduser().resolve()
    if output_dir is not None and bundle_path is None and output_path is None:
        bundle_target = Path(output_dir).expanduser().resolve() / "evidence_bundle.json"
    raw_target = Path(raw_event_evidence_path).expanduser().resolve() if raw_event_evidence_path is not None else bundle_target.with_name("raw_event_evidence.jsonl")
    h0_target = Path(h0_baseline_evidence_path).expanduser().resolve() if h0_baseline_evidence_path is not None else bundle_target.with_name("h0_baseline_evidence.jsonl")
    receipt_target = Path(receipt_path).expanduser().resolve() if receipt_path is not None else bundle_target.with_name("evidence_bundle.receipt.json")
    output_targets = [bundle_target, raw_target, h0_target, receipt_target]
    if len(set(output_targets)) != len(output_targets):
        raise AggregationError("evidence bundle outputs collide")
    if any(path.exists() for path in output_targets):
        collision = next(path for path in output_targets if path.exists())
        raise FileExistsError(f"evidence bundle output collision: {collision}")

    artifact_meta = {
        "raw_event_evidence": {"path": str(raw_target), "sha256": sha256_bytes(raw_bytes), "record_count": len(raw_lines)},
        "h0_baseline_evidence": {"path": str(h0_target), "sha256": sha256_bytes(h0_bytes), "record_count": len(h0_lines)},
    }
    support_refs: dict[str, list[dict[str, str]]] = {}
    for checkpoint, rows in support_rows_by_checkpoint.items():
        unique: dict[str, dict[str, str]] = {}
        for row in rows:
            ref = row.get("_source_ref")
            if isinstance(ref, Mapping) and ref.get("path") is not None and ref.get("sha256") is not None:
                unique[str(ref["path"])] = {"path": str(ref["path"]), "sha256": str(ref["sha256"])}
        support_refs[checkpoint] = [unique[path] for path in sorted(unique)]
    bundle: dict[str, Any] = {
        "schema_version": EVIDENCE_BUNDLE_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "status": "completed",
        "qualification_status": "evidence_ready" if readiness["overall"] == "ready" else "hold",
        "source_summary": {
            "schema_version": summary_value.get("schema_version"),
            "input_sha256": summary_value.get("input_sha256"),
            "self_sha256": summary_value.get("self_sha256"),
        },
        "cohorts": {
            checkpoint: {
                key: value
                for key, value in payload.items()
                if key != "prefix_bindings"
            }
            for checkpoint, payload in cohort_payloads.items()
        },
        "h0_baselines": {"count": len(h0_lines), "expected_count": 26, "by_checkpoint": h0_identity},
        "support_sources": support_refs,
        "census_sources": {checkpoint: [{key: value for key, value in item.items() if key != "rows" and key != "envelope"} for item in rows] for checkpoint, rows in census_by_checkpoint.items()},
        "artifacts": artifact_meta,
        "denominators": denominators,
        "qualifications": qualifications,
        "readiness": readiness,
        "missing_evidence": missing_unique,
        "hypotheses": {f"H{index}": None for index in range(1, 6)},
        "interpretation": None,
        "recommendation": None,
    }
    bundle["self_sha256"] = sha256_json(bundle)
    bundle_bytes = _canonical(bundle) + b"\n"
    receipt = {
        "schema_version": f"{EVIDENCE_BUNDLE_SCHEMA_VERSION}.receipt",
        "unit_id": UNIT_ID,
        "bundle_path": str(bundle_target),
        "bundle_sha256": sha256_bytes(bundle_bytes),
        "bundle_self_sha256": bundle["self_sha256"],
        "artifacts": artifact_meta,
        "input_sha256": summary_value.get("input_sha256"),
    }
    receipt["self_sha256"] = sha256_json(receipt)
    receipt_bytes = _canonical(receipt) + b"\n"
    _write_bytes_collision(raw_target, raw_bytes)
    _write_bytes_collision(h0_target, h0_bytes)
    _write_bytes_collision(bundle_target, bundle_bytes)
    _write_bytes_collision(receipt_target, receipt_bytes)
    return {
        **bundle,
        "bundle": bundle,
        "receipt": receipt,
        "raw_event_evidence": raw_lines,
        "h0_baseline_evidence": h0_lines,
        "paths": {"bundle": str(bundle_target), "receipt": str(receipt_target), "raw_event_evidence": str(raw_target), "h0_baseline_evidence": str(h0_target)},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", action="append", required=True, metavar="CHECKPOINT=PATH")
    parser.add_argument("--shard", action="append", required=True, metavar="CHECKPOINT:INDEX/4=PATH")
    parser.add_argument(
        "--eligible-hold-leaf",
        action="append",
        default=[],
        metavar="CHECKPOINT:ORDINAL=PATH",
        help="admit one independently sealed eligible technical-invalid HOLD leaf",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--evidence-bundle",
        "--bundle",
        "--synthesis-inputs",
        dest="evidence_bundle",
        type=Path,
        help="also finalize evidence_bundle.json and additive raw/H0 evidence artifacts",
    )
    parser.add_argument("--bundle-receipt", type=Path, help="receipt path for --evidence-bundle")
    parser.add_argument("--bundle-output-dir", type=Path)
    parser.add_argument("--raw-event-evidence", type=Path)
    parser.add_argument("--h0-baseline-evidence", type=Path)
    parser.add_argument("--support-ledger", "--support-ledgers", action="append", default=[], metavar="CHECKPOINT=PATH")
    parser.add_argument("--census", "--observational-census", action="append", default=[], metavar="CHECKPOINT=PATH")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cohort_paths: dict[str, Path] = {}
        for raw_cohort in args.cohort:
            checkpoint, path = parse_cohort_selector(raw_cohort)
            if checkpoint in cohort_paths:
                raise AggregationError(f"duplicate cohort selector for {checkpoint}")
            cohort_paths[checkpoint] = path
        specs = [parse_shard_selector(value) for value in args.shard]
        hold_specs = [parse_eligible_hold_selector(value) for value in args.eligible_hold_leaf]
        summary = aggregate_shards(
            cohort_paths=cohort_paths,
            shard_specs=specs,
            eligible_hold_specs=hold_specs,
            output_path=args.output,
            receipt_path=args.receipt,
        )
        bundle_result = None
        if args.evidence_bundle is not None:
            bundle_result = build_evidence_bundle(
                summary,
                bundle_path=args.evidence_bundle,
                output_dir=args.bundle_output_dir,
                receipt_path=args.bundle_receipt,
                raw_event_evidence_path=args.raw_event_evidence,
                h0_baseline_evidence_path=args.h0_baseline_evidence,
                support_ledger_paths=args.support_ledger,
                census_paths=args.census,
            )
    except (AggregationError, OSError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    result = {"status": summary["status"], "output": str(args.output), "summary_self_sha256": summary["self_sha256"]}
    if bundle_result is not None:
        result["evidence_bundle"] = bundle_result["paths"]
        result["evidence_bundle_self_sha256"] = bundle_result["bundle"]["self_sha256"]
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
