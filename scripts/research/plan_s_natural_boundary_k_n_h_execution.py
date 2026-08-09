#!/usr/bin/env python3
"""Plan deterministic eight-shard execution of the sealed S K/N/H cohort.

This planner consumes only the admitted event manifest and a sealed gate-v3
runtime receipt.  It never reads intervention payloads, executes a model, or
selects events from observed outcomes.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.run_s_natural_boundary_k_n_h_cohort import (  # noqa: E402
    ARM_ORDER,
    CohortContractError,
    UNIT_ID,
    _validate_execution_qualification,
    canonical_json_bytes,
    sha256_bytes,
    sha256_json,
    validate_manifest,
)


PLAN_SCHEMA_VERSION = "s_natural_boundary_k_n_h_execution_plan.v1"
GATE_SCHEMA_VERSION = "s_primary_natural_boundary_gate.v1"
SHARD_COUNT = 8
GATE_FORWARD_COUNT = 301
GATE_EVENT_ID = "gt:5001:15"
# Contract ceiling for one event: 15 frozen arms x three rows x 256 tokens.
# The 301-forward gate result is an empirical gt:5001:15 estimate, not this
# universal execution ceiling.
CONTRACT_MAX_ROWS = 3
CONTRACT_MAX_ROW_TOKENS = 256
CONTRACT_MAX_SCALAR_FORWARD_PER_EVENT = len(ARM_ORDER) * CONTRACT_MAX_ROWS * CONTRACT_MAX_ROW_TOKENS
CHECKPOINT = "S"
STEP = 2444
SUBSTRATE = "four-coordinate geo_sorted_xy"


class ExecutionPlanError(CohortContractError):
    """Raised when a plan, manifest, or gate receipt is incompatible."""


def _read_json(source: str | Path | Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if isinstance(source, (str, Path)):
        raw_path = Path(source).expanduser()
        if raw_path.is_symlink() or not raw_path.is_file():
            raise ExecutionPlanError(f"JSON source is not a regular file: {raw_path}")
        path = raw_path.resolve(strict=True)
        try:
            raw = path.read_bytes()
            value = json.loads(raw)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ExecutionPlanError(f"cannot read JSON source {path}: {exc}") from exc
        if not isinstance(value, Mapping):
            raise ExecutionPlanError("JSON source must be an object")
        return dict(value), {"path": str(path), "sha256": sha256_bytes(raw), "raw": raw}
    if not isinstance(source, Mapping):
        raise ExecutionPlanError("JSON source must be a path or object")
    value = dict(source)
    return value, {"inline": True, "sha256": sha256_json(value), "raw": None}


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ExecutionPlanError(f"{label} must be a lowercase SHA-256")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ExecutionPlanError(f"{label} must be a non-negative integer")
    return int(value)


def _finite_nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ExecutionPlanError(f"{label} must be a finite number")
    number = float(value)
    if number < 0:
        raise ExecutionPlanError(f"{label} must be non-negative")
    return number


def _write_once(path: str | Path, document: Mapping[str, Any]) -> bool:
    destination_source = Path(path).expanduser()
    if destination_source.is_symlink():
        raise ExecutionPlanError(f"refusing to write through symlink: {destination_source}")
    destination = destination_source.resolve()
    payload = canonical_json_bytes(document) + b"\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.is_symlink() or not destination.is_file() or destination.read_bytes() != payload:
            raise FileExistsError(f"immutable plan collision: {destination}")
        return True
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            destination.unlink()
        except OSError:
            pass
        raise
    return False


def _validate_gate_receipt(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    document, info = _read_json(source)
    try:
        canonical_json_bytes(document)
    except CohortContractError as exc:
        raise ExecutionPlanError("gate receipt is not finite canonical JSON") from exc
    # The sealed gate's result.json is an authority artifact whose raw bytes
    # are bound below; it need not be pretty/canonical serialized.  Its
    # result_sha256 still authenticates the canonical semantic body.
    if document.get("schema_version") != GATE_SCHEMA_VERSION or document.get("unit_id") != UNIT_ID:
        raise ExecutionPlanError("gate receipt schema or unit identity drifted")
    if (
        document.get("checkpoint") != CHECKPOINT
        or document.get("event_id") != GATE_EVENT_ID
        or document.get("gpu_launch_authorized") is not False
    ):
        raise ExecutionPlanError("gate receipt is not the sealed no-GPU S gate")
    if document.get("no_training") is not True:
        raise ExecutionPlanError("gate receipt permits training")
    compact_runtime_identity = document.get("runtime_identity")
    if isinstance(compact_runtime_identity, Mapping) and compact_runtime_identity.get("event_id") != GATE_EVENT_ID:
        raise ExecutionPlanError("gate compact runtime identity event differs from sealed gate event")
    arm_order = tuple(document.get("arm_order", ()))
    if arm_order != ARM_ORDER:
        raise ExecutionPlanError("gate receipt arm order differs from frozen S K/N/H order")
    arms = document.get("arms")
    # Persisted canonical JSON objects have lexicographically sorted keys; the
    # frozen arm order is carried by the explicit arm_order list above.
    if not isinstance(arms, Mapping) or set(arms) != set(ARM_ORDER):
        raise ExecutionPlanError("gate receipt arm set is incomplete")
    counts: dict[str, int] = {}
    for arm in ARM_ORDER:
        value = arms[arm]
        if not isinstance(value, Mapping):
            raise ExecutionPlanError(f"gate receipt arm {arm} is malformed")
        count = _nonnegative_int(value.get("scalar_forward_count"), f"gate arm {arm}.scalar_forward_count")
        runtime_count = _nonnegative_int(
            value.get("runtime_scalar_forward_count"), f"gate arm {arm}.runtime_scalar_forward_count"
        )
        if runtime_count != count:
            raise ExecutionPlanError(f"gate arm {arm} runtime/scalar forward counts differ")
        counts[arm] = count
    total = sum(counts.values())
    if total != GATE_FORWARD_COUNT:
        raise ExecutionPlanError(f"gate receipt scalar-forward total is {total}, expected {GATE_FORWARD_COUNT}")
    declared_result_sha = _sha(document.get("result_sha256"), "gate result_sha256")
    body = dict(document)
    body.pop("result_sha256", None)
    if declared_result_sha != sha256_json(body):
        raise ExecutionPlanError("gate result_sha256 mismatch")
    runtime_identity_sha = document.get("runtime_identity_sha256")
    if runtime_identity_sha is not None:
        runtime_identity_sha = _sha(runtime_identity_sha, "gate runtime_identity_sha256")
    runtime_identity_raw_sha: str | None = None
    runtime_identity: Mapping[str, Any] | None = None
    if info.get("path") is not None:
        runtime_path = Path(info["path"]).with_name("runtime_identity.json")
        runtime_identity, runtime_info = _read_json(runtime_path)
        runtime_identity_raw_sha = runtime_info["sha256"]
        identity_sha = _sha(runtime_identity.get("identity_sha256"), "runtime identity_sha256")
        identity_body = dict(runtime_identity)
        identity_body.pop("identity_sha256", None)
        if identity_sha != sha256_json(identity_body):
            raise ExecutionPlanError("gate runtime identity self hash mismatch")
        if runtime_identity_sha != identity_sha:
            raise ExecutionPlanError("gate result/runtime identity sibling hash mismatch")
        if (
            runtime_identity.get("schema_version") != f"{GATE_SCHEMA_VERSION}.runtime_identity.v1"
            or runtime_identity.get("unit_id") != UNIT_ID
            or runtime_identity.get("checkpoint") != CHECKPOINT
            or runtime_identity.get("event_id") != GATE_EVENT_ID
            or runtime_identity.get("gpu_launch_authorized") is not False
            or runtime_identity.get("no_training") is not True
        ):
            raise ExecutionPlanError("gate runtime identity sibling is incompatible")
        runtime_event = runtime_identity.get("event")
        if not isinstance(runtime_event, Mapping) or runtime_event.get("event_id") != GATE_EVENT_ID:
            raise ExecutionPlanError("gate runtime identity event binding differs from sealed gate event")
    wall_seconds: float | None = None
    for key in ("wall_time_seconds", "elapsed_seconds", "runtime_wall_time_seconds"):
        if key in document:
            wall_seconds = _finite_nonnegative(document[key], f"gate.{key}")
            break
    if wall_seconds is None and isinstance(document.get("runtime_identity"), Mapping):
        for key in ("wall_time_seconds", "elapsed_seconds", "runtime_wall_time_seconds"):
            if key in document["runtime_identity"]:
                wall_seconds = _finite_nonnegative(document["runtime_identity"][key], f"gate.runtime_identity.{key}")
                break
    return {
        "document": document,
        "source": info,
        "gate_sha256": info["sha256"],
        "gate_result_sha256": declared_result_sha,
        "gate_runtime_identity_sha256": runtime_identity_sha,
        "gate_runtime_identity_raw_sha256": runtime_identity_raw_sha,
        "arm_scalar_forward_counts": counts,
        "scalar_forward_count": total,
        "wall_time_seconds": wall_seconds,
    }


def _event_ref(event: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "event_index": event["event_index"],
        "event_id": event["event_id"],
        "image_id": event["image_id"],
        "event_sha256": event["event_sha256"],
    }


def _shard_id(index: int) -> str:
    return f"shard-{index:03d}"


def build_plan(
    manifest: str | Path | Mapping[str, Any],
    gate_receipt: str | Path | Mapping[str, Any],
) -> dict[str, Any]:
    manifest_info = validate_manifest(manifest)
    try:
        claim_scope = _validate_execution_qualification(manifest_info)
    except CohortContractError as exc:
        raise ExecutionPlanError(f"manifest claim scope is invalid: {exc}") from exc
    gate_info = _validate_gate_receipt(gate_receipt)
    events = manifest_info["events"]
    shards: list[dict[str, Any]] = []
    wall_per_event = (
        gate_info["wall_time_seconds"] / gate_info["scalar_forward_count"]
        if gate_info["wall_time_seconds"] is not None
        else None
    )
    for shard_index in range(SHARD_COUNT):
        assigned = [event for event in events if event["event_index"] % SHARD_COUNT == shard_index]
        refs = [_event_ref(event) for event in assigned]
        event_count = len(refs)
        shards.append(
            {
                "shard_id": _shard_id(shard_index),
                "shard_index": shard_index,
                "event_indices": [ref["event_index"] for ref in refs],
                "events": refs,
                "event_count": event_count,
                "distinct_image_count": len({ref["image_id"] for ref in refs}),
                "scalar_forward_upper_bound": event_count * CONTRACT_MAX_SCALAR_FORWARD_PER_EVENT,
                "wall_time_estimate_seconds": (
                    event_count * GATE_FORWARD_COUNT * wall_per_event if wall_per_event is not None else None
                ),
            }
        )
    body = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": "planned",
        "unit_id": UNIT_ID,
        "primary": {"checkpoint": CHECKPOINT, "step": STEP, "substrate": SUBSTRATE},
        "manifest_sha256": manifest_info["manifest_sha256"],
        "manifest_self_sha256": manifest_info["manifest_self_sha256"],
        "gate_sha256": gate_info["gate_sha256"],
        "gate_result_sha256": gate_info["gate_result_sha256"],
        "gate_runtime_identity_sha256": gate_info["gate_runtime_identity_sha256"],
        "gate_runtime_identity_raw_sha256": gate_info["gate_runtime_identity_raw_sha256"],
        "gate_scalar_forward_count": gate_info["scalar_forward_count"],
        "gate_arm_scalar_forward_counts": gate_info["arm_scalar_forward_counts"],
        "gate_wall_time_seconds": gate_info["wall_time_seconds"],
        "event_count": len(events),
        "distinct_image_count": len({event["image_id"] for event in events}),
        "events": [_event_ref(event) for event in events],
        "minimum_event_count": manifest_info["admission_gate"]["minimum_event_count"],
        "minimum_image_count": manifest_info["admission_gate"]["minimum_image_count"],
        "claim_scope": claim_scope,
        "arm_order": list(ARM_ORDER),
        "shard_count": SHARD_COUNT,
        "assignment_policy": "round_robin_event_index_mod_8",
        "no_outcome_adaptive_selection": True,
        "no_event_reorder": True,
        "no_sweep": True,
        "no_a3": True,
        "no_2x2": True,
        "no_p4": True,
        "scalar_forward_upper_bound_per_event": CONTRACT_MAX_SCALAR_FORWARD_PER_EVENT,
        "scalar_forward_upper_bound_total": len(events) * CONTRACT_MAX_SCALAR_FORWARD_PER_EVENT,
        "gate_empirical_scalar_forward_estimate_per_event": GATE_FORWARD_COUNT,
        "gate_empirical_scalar_forward_estimate_total": len(events) * GATE_FORWARD_COUNT,
        "wall_time_estimate_status": "derived_from_empirical_gate_estimate" if wall_per_event is not None else "unavailable_in_gate_receipt",
        "wall_time_estimate_seconds_total": len(events) * GATE_FORWARD_COUNT * wall_per_event if wall_per_event is not None else None,
        "shards": shards,
    }
    document = dict(body)
    document["plan_sha256"] = sha256_json(body)
    return document


def validate_plan(
    source: str | Path | Mapping[str, Any],
    *,
    manifest_info: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    document, info = _read_json(source)
    if document.get("schema_version") != PLAN_SCHEMA_VERSION or document.get("status") != "planned" or document.get("unit_id") != UNIT_ID:
        raise ExecutionPlanError("execution plan schema/status/unit identity drifted")
    if document.get("primary") != {"checkpoint": CHECKPOINT, "step": STEP, "substrate": SUBSTRATE}:
        raise ExecutionPlanError("execution plan primary identity drifted")
    if document.get("arm_order") != list(ARM_ORDER) or document.get("shard_count") != SHARD_COUNT:
        raise ExecutionPlanError("execution plan arm/shard identity drifted")
    for flag in ("no_outcome_adaptive_selection", "no_event_reorder", "no_sweep", "no_a3", "no_2x2", "no_p4"):
        if document.get(flag) is not True:
            raise ExecutionPlanError(f"execution plan flag {flag} is not sealed")
    plan_sha = _sha(document.get("plan_sha256"), "plan_sha256")
    body = dict(document)
    body.pop("plan_sha256", None)
    if plan_sha != sha256_json(body):
        raise ExecutionPlanError("execution plan plan_sha256 mismatch")
    manifest_sha = _sha(document.get("manifest_sha256"), "plan.manifest_sha256")
    manifest_self_sha = _sha(document.get("manifest_self_sha256"), "plan.manifest_self_sha256")
    _sha(document.get("gate_sha256"), "plan.gate_sha256")
    _sha(document.get("gate_result_sha256"), "plan.gate_result_sha256")
    runtime_identity_sha = document.get("gate_runtime_identity_sha256")
    if runtime_identity_sha is not None:
        _sha(runtime_identity_sha, "plan.gate_runtime_identity_sha256")
    runtime_identity_raw_sha = document.get("gate_runtime_identity_raw_sha256")
    if runtime_identity_raw_sha is not None:
        _sha(runtime_identity_raw_sha, "plan.gate_runtime_identity_raw_sha256")
    if document.get("assignment_policy") != "round_robin_event_index_mod_8":
        raise ExecutionPlanError("execution plan assignment policy drifted")
    if document.get("gate_scalar_forward_count") != GATE_FORWARD_COUNT:
        raise ExecutionPlanError("execution plan is not bound to the 301-forward gate receipt")
    if document.get("scalar_forward_upper_bound_per_event") != CONTRACT_MAX_SCALAR_FORWARD_PER_EVENT:
        raise ExecutionPlanError("execution plan contract scalar bound drifted")
    if document.get("gate_empirical_scalar_forward_estimate_per_event") != GATE_FORWARD_COUNT:
        raise ExecutionPlanError("execution plan empirical gate estimate drifted")
    if document.get("event_count") is not None:
        if document.get("scalar_forward_upper_bound_total") != document["event_count"] * CONTRACT_MAX_SCALAR_FORWARD_PER_EVENT:
            raise ExecutionPlanError("execution plan total contract scalar bound drifted")
        if document.get("gate_empirical_scalar_forward_estimate_total") != document["event_count"] * GATE_FORWARD_COUNT:
            raise ExecutionPlanError("execution plan total empirical gate estimate drifted")
    if manifest_info is not None:
        if manifest_info["manifest_sha256"] != manifest_sha or manifest_info["manifest_self_sha256"] != manifest_self_sha:
            raise ExecutionPlanError("execution plan is bound to a different manifest")
        events = manifest_info["events"]
        expected_refs = [_event_ref(event) for event in events]
        if document.get("event_count") != len(events) or document.get("distinct_image_count") != len({event["image_id"] for event in events}):
            raise ExecutionPlanError("execution plan event/image counts differ from manifest")
        if document.get("events") != [_event_ref(event) for event in events]:
            raise ExecutionPlanError("execution plan event order differs from manifest")
        if document.get("claim_scope") != _validate_execution_qualification(manifest_info):
            raise ExecutionPlanError("execution plan claim scope differs from manifest")
    else:
        expected_refs = []
    shards = document.get("shards")
    if not isinstance(shards, list) or len(shards) != SHARD_COUNT:
        raise ExecutionPlanError("execution plan must contain exactly eight shards")
    seen: list[dict[str, Any]] = []
    for shard_index, shard in enumerate(shards):
        if not isinstance(shard, Mapping) or shard.get("shard_id") != _shard_id(shard_index) or shard.get("shard_index") != shard_index:
            raise ExecutionPlanError("execution plan shard identity/order drifted")
        refs = shard.get("events")
        if not isinstance(refs, list) or shard.get("event_indices") != [ref["event_index"] for ref in refs]:
            raise ExecutionPlanError(f"execution plan {shard.get('shard_id')} event order is malformed")
        if any(not isinstance(ref, Mapping) for ref in refs):
            raise ExecutionPlanError(f"execution plan {shard.get('shard_id')} has malformed event ref")
        indices = [ref["event_index"] for ref in refs]
        if indices != sorted(indices) or len(indices) != len(set(indices)):
            raise ExecutionPlanError(f"execution plan {shard.get('shard_id')} event indices are not ordered/unique")
        if any(index % SHARD_COUNT != shard_index for index in indices):
            raise ExecutionPlanError(f"execution plan {shard.get('shard_id')} violates round-robin assignment")
        if shard.get("event_count") != len(refs) or shard.get("distinct_image_count") != len({ref["image_id"] for ref in refs}):
            raise ExecutionPlanError(f"execution plan {shard.get('shard_id')} counts are inconsistent")
        if shard.get("scalar_forward_upper_bound") != len(refs) * CONTRACT_MAX_SCALAR_FORWARD_PER_EVENT:
            raise ExecutionPlanError(f"execution plan {shard.get('shard_id')} scalar bound is inconsistent")
        for ref in refs:
            if not isinstance(ref.get("event_index"), int) or not isinstance(ref.get("event_id"), str) or not isinstance(ref.get("image_id"), int):
                raise ExecutionPlanError(f"execution plan {shard.get('shard_id')} event ref is malformed")
        seen.extend(dict(ref) for ref in refs)
    if manifest_info is not None:
        # Round-robin assignment intentionally groups indices by shard.  Event
        # order is preserved inside each shard and explicitly at top level;
        # merger reconstructs the manifest order from the event indices.
        seen_sorted = sorted(seen, key=lambda ref: ref["event_index"])
        if seen_sorted != expected_refs:
            raise ExecutionPlanError("execution plan shards do not cover the manifest event order")
        expected_indices = {event["event_index"]: event["event_index"] % SHARD_COUNT for event in manifest_info["events"]}
        observed_indices = {
            ref["event_index"]: shard_index
            for shard_index, shard in enumerate(shards)
            for ref in shard["events"]
        }
        if observed_indices != expected_indices:
            raise ExecutionPlanError("execution plan shard assignment differs from event_index modulo 8")
    if len({ref["event_index"] for ref in seen}) != len(seen):
        raise ExecutionPlanError("execution plan assigns an event more than once")
    return {
        "document": document,
        "source": info,
        "plan_sha256": plan_sha,
        "manifest_sha256": manifest_sha,
        "manifest_self_sha256": manifest_self_sha,
        "shards": [dict(shard) for shard in shards],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gate-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        plan = build_plan(args.manifest, args.gate_receipt)
        _write_once(args.output, plan)
        print(json.dumps({"status": plan["status"], "plan_sha256": plan["plan_sha256"]}, sort_keys=True))
        return 0
    except (ExecutionPlanError, FileExistsError, OSError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
