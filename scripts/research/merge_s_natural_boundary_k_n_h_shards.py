#!/usr/bin/env python3
"""Merge all eight immutable S K/N/H execution shards in manifest order."""

from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.plan_s_natural_boundary_k_n_h_execution import (  # noqa: E402
    CONTRACT_MAX_SCALAR_FORWARD_PER_EVENT,
    ExecutionPlanError,
    SHARD_COUNT,
    _event_ref,
    _read_json,
    _shard_id,
    _write_once,
    validate_plan,
)
from scripts.research.run_s_natural_boundary_k_n_h_cohort import (  # noqa: E402
    ARM_ORDER,
    CohortContractError,
    _load_completed_event,
    _validate_execution_qualification,
    CHECKPOINT,
    STEP,
    SUBSTRATE,
    UNIT_ID,
    validate_manifest,
)
from scripts.research.run_s_natural_boundary_k_n_h_shard import (  # noqa: E402
    SHARD_AGGREGATE_SCHEMA_VERSION,
    SHARD_RECEIPT_SCHEMA_VERSION,
    ShardExecutionError,
    _event_root_name,
    _event_scalars,
    _normalize_shard_id,
    _validate_event_plan_binding,
    _validate_shard_contents,
)
from scripts.research.run_s_natural_boundary_k_n_h_cohort import sha256_json  # noqa: E402


MERGED_AGGREGATE_SCHEMA_VERSION = "s_natural_boundary_k_n_h_merged_aggregate.v1"
MERGED_RECEIPT_SCHEMA_VERSION = "s_natural_boundary_k_n_h_merged_receipt.v1"


class ShardMergeError(CohortContractError):
    """Raised when a shard set is incomplete, foreign, or tampered."""


def _canonical_file(source: Path, *, label: str) -> tuple[dict[str, Any], str]:
    document, info = _read_json(source)
    raw = info.get("raw")
    if raw is None or raw != json.dumps(document, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n":
        raise ShardMergeError(f"{label} is not canonical JSON")
    return document, info["sha256"]


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ShardMergeError(f"{label} must be a lowercase SHA-256")
    return value


def _validate_hash_document(document: Mapping[str, Any], field: str, label: str) -> str:
    declared = _sha(document.get(field), f"{label}.{field}")
    body = dict(document)
    body.pop(field, None)
    if declared != sha256_json(body):
        raise ShardMergeError(f"{label} {field} mismatch")
    return declared


def _base_ref(ref: Mapping[str, Any]) -> dict[str, Any]:
    return {key: ref[key] for key in ("event_index", "event_id", "image_id", "event_sha256", "result_sha256", "root")}


def validate_shard(
    shard_root: str | Path,
    *,
    shard_id: str | int,
    manifest_info: Mapping[str, Any],
    plan_info: Mapping[str, Any],
    expected_pre_gpu_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    shard, shard_index = _normalize_shard_id(shard_id)
    root_source = Path(shard_root).expanduser()
    if root_source.is_symlink() or not root_source.is_dir():
        raise ShardMergeError(f"shard root is not a regular directory: {root_source}")
    root = root_source.resolve()
    planned = plan_info["shards"][shard_index]
    expected_events = [
        next(event for event in manifest_info["events"] if event["event_index"] == ref["event_index"])
        for ref in planned["events"]
    ]
    _validate_shard_contents(root, expected_events)
    aggregate, _ = _canonical_file(root / "aggregate.json", label=f"{shard}.aggregate")
    receipt, _ = _canonical_file(root / "aggregate.receipt.json", label=f"{shard}.receipt")
    aggregate_sha = _validate_hash_document(aggregate, "aggregate_sha256", f"{shard}.aggregate")
    receipt_sha = _validate_hash_document(receipt, "receipt_sha256", f"{shard}.receipt")
    if (
        aggregate.get("schema_version") != SHARD_AGGREGATE_SCHEMA_VERSION
        or aggregate.get("status") != "completed"
        or aggregate.get("unit_id") != UNIT_ID
        or aggregate.get("shard_id") != shard
        or aggregate.get("shard_index") != shard_index
        or aggregate.get("shard_count") != SHARD_COUNT
        or aggregate.get("plan_sha256") != plan_info["plan_sha256"]
        or aggregate.get("manifest_sha256") != manifest_info["manifest_sha256"]
        or aggregate.get("manifest_self_sha256") != manifest_info["manifest_self_sha256"]
        or aggregate.get("gate_sha256") != plan_info["document"]["gate_sha256"]
        or aggregate.get("gate_result_sha256") != plan_info["document"]["gate_result_sha256"]
        or aggregate.get("gate_runtime_identity_sha256") != plan_info["document"].get("gate_runtime_identity_sha256")
        or aggregate.get("gate_runtime_identity_raw_sha256") != plan_info["document"].get("gate_runtime_identity_raw_sha256")
        or aggregate.get("arm_order") != list(ARM_ORDER)
        or aggregate.get("primary") != {"checkpoint": CHECKPOINT, "step": STEP, "substrate": SUBSTRATE}
        or aggregate.get("claim_scope") != plan_info["document"].get("claim_scope")
        or aggregate.get("claim_scope") != receipt.get("claim_scope")
        or aggregate.get("executor_identity") != receipt.get("executor_identity")
        or aggregate.get("pre_gpu_identity") != receipt.get("pre_gpu_identity")
    ):
        raise ShardMergeError(f"{shard} aggregate identity is incompatible")
    for flag in ("no_event_reorder", "no_outcome_adaptive_selection", "no_sweep", "no_a3", "no_2x2", "no_p4"):
        if aggregate.get(flag) is not True:
            raise ShardMergeError(f"{shard} aggregate flag {flag} is not sealed")
    if (
        receipt.get("schema_version") != SHARD_RECEIPT_SCHEMA_VERSION
        or receipt.get("status") != "completed"
        or receipt.get("unit_id") != UNIT_ID
        or receipt.get("shard_id") != shard
        or receipt.get("shard_index") != shard_index
        or receipt.get("shard_count") != SHARD_COUNT
        or receipt.get("plan_sha256") != plan_info["plan_sha256"]
        or receipt.get("manifest_sha256") != manifest_info["manifest_sha256"]
        or receipt.get("aggregate_sha256") != aggregate_sha
        or receipt.get("gate_sha256") != plan_info["document"]["gate_sha256"]
        or receipt.get("gate_result_sha256") != plan_info["document"]["gate_result_sha256"]
        or receipt.get("gate_runtime_identity_sha256") != plan_info["document"].get("gate_runtime_identity_sha256")
        or receipt.get("gate_runtime_identity_raw_sha256") != plan_info["document"].get("gate_runtime_identity_raw_sha256")
    ):
        raise ShardMergeError(f"{shard} receipt identity is incompatible")
    aggregate_events = aggregate.get("events")
    receipt_events = receipt.get("event_roots")
    if not isinstance(aggregate_events, list) or not isinstance(receipt_events, list):
        raise ShardMergeError(f"{shard} event roots are missing")
    if aggregate_events != receipt_events:
        raise ShardMergeError(f"{shard} aggregate and receipt event roots differ")
    if len(aggregate_events) != len(expected_events) or aggregate.get("event_count") != len(expected_events):
        raise ShardMergeError(f"{shard} event count differs from plan")
    validated_refs: list[dict[str, Any]] = []
    for event, expected_ref, declared_ref in zip(expected_events, planned["events"], aggregate_events, strict=True):
        if not isinstance(declared_ref, Mapping):
            raise ShardMergeError(f"{shard} event reference is malformed")
        if (
            declared_ref.get("event_index") != event["event_index"]
            or declared_ref.get("event_id") != event["event_id"]
            or declared_ref.get("image_id") != event["image_id"]
            or declared_ref.get("event_sha256") != event["event_sha256"]
            or declared_ref.get("root") != _event_root_name(event)
        ):
            raise ShardMergeError(f"{shard} event reference differs from plan/manifest")
        event_root = root / _event_root_name(event)
        event_document = _load_completed_event(event_root, event=event, manifest_info=manifest_info)
        if event_document is None:
            raise ShardMergeError(f"{shard} event root is missing: {event_root}")
        _validate_event_plan_binding(
            event_document, plan_sha256=plan_info["plan_sha256"], shard=shard, shard_index=shard_index
        )
        if declared_ref.get("result_sha256") != event_document.get("result_sha256"):
            raise ShardMergeError(f"{shard} event result hash differs from aggregate")
        if declared_ref.get("executor_identity") != event_document.get("executor_identity"):
            raise ShardMergeError(f"{shard} event executor identity differs from root")
        scalar, runtime, scalar_by_arm, runtime_by_arm = _event_scalars(event_document)
        if scalar != runtime or scalar > CONTRACT_MAX_SCALAR_FORWARD_PER_EVENT:
            raise ShardMergeError(f"{shard} event scalar-forward receipt is invalid")
        if declared_ref.get("scalar_forward_count") != scalar or declared_ref.get("runtime_scalar_forward_count") != runtime:
            raise ShardMergeError(f"{shard} event scalar-forward total differs from root")
        if declared_ref.get("scalar_forward_counts") != scalar_by_arm or declared_ref.get("runtime_scalar_forward_counts") != runtime_by_arm:
            raise ShardMergeError(f"{shard} event scalar-forward arm receipts differ from root")
        if _base_ref(declared_ref) != {
            "event_index": event["event_index"],
            "event_id": event["event_id"],
            "image_id": event["image_id"],
            "event_sha256": event["event_sha256"],
            "result_sha256": event_document["result_sha256"],
            "root": _event_root_name(event),
        }:
            raise ShardMergeError(f"{shard} event reference has foreign fields")
        validated_refs.append(dict(declared_ref))
    event_identities = [ref["executor_identity"] for ref in validated_refs]
    if event_identities and any(identity != event_identities[0] for identity in event_identities[1:]):
        raise ShardMergeError(f"{shard} executor identities differ across events")
    shard_identity = event_identities[0] if event_identities else None
    if aggregate.get("executor_identity") != shard_identity:
        raise ShardMergeError(f"{shard} aggregate executor identity differs from event roots")
    shard_pre_gpu_identity = aggregate.get("pre_gpu_identity")
    if expected_pre_gpu_identity is not None:
        if shard_pre_gpu_identity != dict(expected_pre_gpu_identity):
            raise ShardMergeError(f"{shard} executor identity differs from the pre-GPU receipt")
        if validated_refs and (
            not isinstance(shard_identity, Mapping)
            or shard_identity.get("pre_gpu") != dict(expected_pre_gpu_identity)
        ):
            raise ShardMergeError(f"{shard} executor identity differs from the pre-GPU receipt")
        if not validated_refs and shard_identity is not None:
            raise ShardMergeError(f"{shard} empty shard fabricated an executor identity")
    scalar_total = sum(ref["scalar_forward_count"] for ref in validated_refs)
    runtime_total = sum(ref["runtime_scalar_forward_count"] for ref in validated_refs)
    if (
        aggregate.get("scalar_forward_count") != scalar_total
        or aggregate.get("runtime_scalar_forward_count") != runtime_total
        or aggregate.get("event_count") != len(validated_refs)
        or aggregate.get("distinct_image_count") != len({ref["image_id"] for ref in validated_refs})
        or receipt.get("scalar_forward_count") != scalar_total
        or receipt.get("runtime_scalar_forward_count") != runtime_total
        or receipt.get("event_count") != len(validated_refs)
        or receipt.get("distinct_image_count") != len({ref["image_id"] for ref in validated_refs})
        or aggregate.get("scalar_forward_upper_bound") != len(validated_refs) * CONTRACT_MAX_SCALAR_FORWARD_PER_EVENT
    ):
        raise ShardMergeError(f"{shard} scalar-forward aggregate is inconsistent")
    if aggregate.get("distinct_image_count") != len({ref["image_id"] for ref in validated_refs}):
        raise ShardMergeError(f"{shard} image count is inconsistent")
    return {
        "shard_id": shard,
        "shard_index": shard_index,
        "aggregate": dict(aggregate),
        "receipt": dict(receipt),
        "aggregate_sha256": aggregate_sha,
        "receipt_sha256": receipt_sha,
        "events": validated_refs,
        "scalar_forward_count": scalar_total,
        "runtime_scalar_forward_count": runtime_total,
        "executor_identity": shard_identity,
        "pre_gpu_identity": shard_pre_gpu_identity,
    }


def _validate_shards_root(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ShardMergeError(f"shards root is not a regular directory: {root}")
    expected = {_shard_id(index) for index in range(SHARD_COUNT)}
    children = list(root.iterdir())
    if {child.name for child in children} != expected:
        raise ShardMergeError("shards root must contain exactly shard-000 through shard-007")
    if any(child.is_symlink() or not child.is_dir() for child in children):
        raise ShardMergeError("shards root contains a non-regular shard directory")


def _validated_pre_gpu_identities(
    pre_gpu_receipt: str | Path,
    *,
    manifest_info: Mapping[str, Any],
    plan_info: Mapping[str, Any],
    shards_root: Path,
    output_root: Path,
) -> dict[str, dict[str, Any]]:
    receipt, _ = _read_json(pre_gpu_receipt)
    receipt_path = Path(pre_gpu_receipt).expanduser().resolve(strict=True)
    try:
        pre_gpu = importlib.import_module("scripts.research.seal_s_natural_boundary_k_n_h_pre_gpu_receipt")
        # runtime_identity_binding performs the merge's one final full receipt
        # validation.  The other seven identities differ only by the sealed
        # shard assignment and its resulting binding hash.
        identity = pre_gpu.runtime_identity_binding(
            receipt,
            receipt_path=receipt_path,
            shard_id=0,
            observed_cuda_visible_devices="0",
        )
    except Exception as exc:
        raise ShardMergeError(f"pre-GPU receipt validation failed: {exc}") from exc
    hashes = identity.get("input_hashes")
    if not isinstance(hashes, Mapping) or (
        hashes.get("manifest_raw_sha256") != manifest_info["manifest_sha256"]
        or hashes.get("manifest_self_sha256") != manifest_info["manifest_self_sha256"]
        or hashes.get("execution_plan_raw_sha256") != plan_info["source"]["sha256"]
        or hashes.get("execution_plan_sha256") != plan_info["plan_sha256"]
    ):
        raise ShardMergeError("pre-GPU receipt is bound to a different manifest or plan")
    identities: dict[str, dict[str, Any]] = {}
    for index, planned in enumerate(plan_info["shards"]):
        shard = _shard_id(index)
        physical_device = str(index)
        try:
            authorization = pre_gpu.validate_shard_authorization(
                receipt,
                shard_id=index,
                output_root=shards_root / shard,
                event_sha256s=[ref["event_sha256"] for ref in planned["events"]],
                observed_cuda_visible_devices=physical_device,
            )
        except Exception as exc:
            raise ShardMergeError(f"pre-GPU shard-{index:03d} authorization failed: {exc}") from exc
        shard_identity = dict(identity)
        shard_identity["device_assignment"] = {
            "shard_id": authorization["shard_id"],
            "shard_index": authorization["shard_index"],
            "physical_device": authorization["physical_device"],
            "observed_cuda_visible_devices": physical_device,
            "logical_device": authorization["logical_device"],
            "device_count": authorization["device_count"],
            "authorization_sha256": authorization["authorization_sha256"],
        }
        shard_identity.pop("binding_sha256", None)
        shard_identity["binding_sha256"] = sha256_json(shard_identity)
        identities[shard] = shard_identity
    output_binding = receipt.get("output_binding")
    final = output_binding.get("final_merge_root") if isinstance(output_binding, Mapping) else None
    if not isinstance(final, Mapping) or final.get("path") != str(output_root):
        raise ShardMergeError("merge output root differs from pre-GPU authorization")
    return identities


def _common_executor_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize only receipt-authorized physical shard differences."""

    pre_gpu = identity.get("pre_gpu")
    observed = identity.get("observed")
    if not isinstance(pre_gpu, Mapping) or not isinstance(observed, Mapping):
        raise ShardMergeError("executor identity is structurally incomplete")
    common_pre_gpu = dict(pre_gpu)
    common_pre_gpu.pop("device_assignment", None)
    common_pre_gpu.pop("binding_sha256", None)
    common_observed = dict(observed)
    cuda = common_observed.get("cuda")
    if not isinstance(cuda, Mapping):
        raise ShardMergeError("executor CUDA evidence is structurally incomplete")
    common_cuda = dict(cuda)
    common_cuda.pop("cuda_visible_devices", None)
    common_observed["cuda"] = common_cuda
    return {
        "schema_version": identity.get("schema_version"),
        "pre_gpu": common_pre_gpu,
        "observed": common_observed,
    }


def _merge_shards_core(
    manifest: str | Path | Mapping[str, Any],
    plan: str | Path | Mapping[str, Any],
    shards_root: str | Path,
    output_root: str | Path,
    *,
    expected_pre_gpu_identities: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    manifest_info = validate_manifest(manifest)
    try:
        _validate_execution_qualification(manifest_info)
    except CohortContractError as exc:
        raise ShardMergeError(f"manifest is not execution-qualified: {exc}") from exc
    plan_info = validate_plan(plan, manifest_info=manifest_info)
    root_source = Path(shards_root).expanduser()
    if root_source.is_symlink():
        raise ShardMergeError(f"shards root is a symlink: {root_source}")
    shards_dir = root_source.resolve()
    _validate_shards_root(shards_dir)
    shard_infos = [
        validate_shard(
            shards_dir / _shard_id(index),
            shard_id=index,
            manifest_info=manifest_info,
            plan_info=plan_info,
            expected_pre_gpu_identity=(
                expected_pre_gpu_identities.get(_shard_id(index))
                if expected_pre_gpu_identities is not None
                else None
            ),
        )
        for index in range(SHARD_COUNT)
    ]
    all_events = [event for shard_info in shard_infos for event in shard_info["events"]]
    expected_refs = [_event_ref(event) for event in manifest_info["events"]]
    observed_base = [
        {key: ref[key] for key in ("event_index", "event_id", "image_id", "event_sha256")}
        for ref in sorted(all_events, key=lambda ref: ref["event_index"])
    ]
    if observed_base != expected_refs or len({ref["event_index"] for ref in all_events}) != len(all_events):
        raise ShardMergeError("shards do not cover each manifest event exactly once in order")
    scalar_total = sum(info["scalar_forward_count"] for info in shard_infos)
    runtime_total = sum(info["runtime_scalar_forward_count"] for info in shard_infos)
    shard_executor_identities = {
        info["shard_id"]: info["executor_identity"]
        for info in shard_infos
        if info["executor_identity"] is not None
    }
    if shard_executor_identities and any(
        _common_executor_identity(identity)
        != _common_executor_identity(next(iter(shard_executor_identities.values())))
        for identity in list(shard_executor_identities.values())[1:]
    ):
        raise ShardMergeError("common executor identity differs across shards")
    expected_nonempty_shards = {
        info["shard_id"] for info in shard_infos if info["events"]
    }
    if expected_pre_gpu_identities is not None and set(shard_executor_identities) != expected_nonempty_shards:
        raise ShardMergeError("merged executor identities are incomplete")
    pre_gpu_identities = {
        info["shard_id"]: info["pre_gpu_identity"]
        for info in shard_infos
        if info["pre_gpu_identity"] is not None
    }
    if expected_pre_gpu_identities is not None and pre_gpu_identities != {
        key: dict(value) for key, value in expected_pre_gpu_identities.items()
    }:
        raise ShardMergeError("merged pre-GPU shard identities are incomplete")
    execution_qualification = _validate_execution_qualification(manifest_info)
    body = {
        "schema_version": MERGED_AGGREGATE_SCHEMA_VERSION,
        "status": "completed",
        "unit_id": UNIT_ID,
        "primary": dict(plan_info["document"]["primary"]),
        "claim_scope": execution_qualification,
        "plan_sha256": plan_info["plan_sha256"],
        "manifest_self_sha256": manifest_info["manifest_self_sha256"],
        "manifest_sha256": manifest_info["manifest_sha256"],
        "gate_sha256": plan_info["document"]["gate_sha256"],
        "gate_result_sha256": plan_info["document"]["gate_result_sha256"],
        "gate_runtime_identity_sha256": plan_info["document"].get("gate_runtime_identity_sha256"),
        "gate_runtime_identity_raw_sha256": plan_info["document"].get("gate_runtime_identity_raw_sha256"),
        "arm_order": list(ARM_ORDER),
        "shard_count": SHARD_COUNT,
        "shards": [
            {
                "shard_id": info["shard_id"],
                "shard_index": info["shard_index"],
                "aggregate_sha256": info["aggregate_sha256"],
                "receipt_sha256": info["receipt_sha256"],
                "event_count": len(info["events"]),
                "scalar_forward_count": info["scalar_forward_count"],
                "runtime_scalar_forward_count": info["runtime_scalar_forward_count"],
            }
            for info in shard_infos
        ],
        "event_count": len(all_events),
        "distinct_image_count": len({ref["image_id"] for ref in all_events}),
        "scalar_forward_count": scalar_total,
        "runtime_scalar_forward_count": runtime_total,
        "executor_identities": shard_executor_identities,
        "pre_gpu_identities": pre_gpu_identities,
        "scalar_forward_upper_bound": len(all_events) * CONTRACT_MAX_SCALAR_FORWARD_PER_EVENT,
        "execution_qualification": execution_qualification,
        "events": sorted(all_events, key=lambda ref: ref["event_index"]),
        "no_event_reorder": True,
        "no_outcome_adaptive_selection": True,
        "no_sweep": True,
        "no_a3": True,
        "no_2x2": True,
        "no_p4": True,
    }
    aggregate = dict(body)
    aggregate["aggregate_sha256"] = sha256_json(body)
    output_source = Path(output_root).expanduser()
    if output_source.is_symlink():
        raise ShardMergeError(f"merge output root is a symlink: {output_source}")
    output = output_source.resolve()
    if output.exists() and not output.is_dir():
        raise ShardMergeError(f"merge output root is not a directory: {output}")
    if output.exists():
        allowed = {"aggregate.json", "aggregate.receipt.json"}
        if {child.name for child in output.iterdir()} - allowed:
            raise ShardMergeError("merge output contains a foreign file")
    _write_once(output / "aggregate.json", aggregate)
    receipt = {
        "schema_version": MERGED_RECEIPT_SCHEMA_VERSION,
        "status": "completed",
        "unit_id": UNIT_ID,
        "plan_sha256": plan_info["plan_sha256"],
        "manifest_sha256": manifest_info["manifest_sha256"],
        "gate_sha256": plan_info["document"]["gate_sha256"],
        "gate_result_sha256": plan_info["document"]["gate_result_sha256"],
        "gate_runtime_identity_sha256": plan_info["document"].get("gate_runtime_identity_sha256"),
        "gate_runtime_identity_raw_sha256": plan_info["document"].get("gate_runtime_identity_raw_sha256"),
        "aggregate_sha256": aggregate["aggregate_sha256"],
        "shard_count": SHARD_COUNT,
        "event_count": len(all_events),
        "distinct_image_count": len({ref["image_id"] for ref in all_events}),
        "scalar_forward_count": scalar_total,
        "runtime_scalar_forward_count": runtime_total,
        "executor_identities": shard_executor_identities,
        "pre_gpu_identities": pre_gpu_identities,
        "execution_qualification": execution_qualification,
        "claim_scope": execution_qualification,
        "shards": aggregate["shards"],
        "event_roots": aggregate["events"],
    }
    receipt["receipt_sha256"] = sha256_json(receipt)
    _write_once(output / "aggregate.receipt.json", receipt)
    return {"aggregate": aggregate, "receipt": receipt, "manifest": manifest_info, "plan": plan_info}


def _merge_shards_test_only(
    manifest: str | Path | Mapping[str, Any],
    plan: str | Path | Mapping[str, Any],
    shards_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Private synthetic-receipt seam for CPU contract tests only."""

    return _merge_shards_core(manifest, plan, shards_root, output_root)


def merge_shards(
    manifest: str | Path,
    plan: str | Path,
    shards_root: str | Path,
    output_root: str | Path,
    *,
    pre_gpu_receipt: str | Path,
) -> dict[str, Any]:
    """Revalidate exact pre-GPU authority, then merge all shard receipts."""

    manifest_info = validate_manifest(manifest)
    plan_info = validate_plan(plan, manifest_info=manifest_info)
    shards_dir = Path(shards_root).expanduser().resolve(strict=True)
    output = Path(output_root).expanduser().resolve()
    try:
        output.relative_to(shards_dir)
    except ValueError:
        pass
    else:
        raise ShardMergeError("final merge root must be outside the exact shards root")
    identities = _validated_pre_gpu_identities(
        pre_gpu_receipt,
        manifest_info=manifest_info,
        plan_info=plan_info,
        shards_root=shards_dir,
        output_root=output,
    )
    return _merge_shards_core(
        manifest,
        plan,
        shards_dir,
        output,
        expected_pre_gpu_identities=identities,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--shards-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pre-gpu-receipt", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = merge_shards(
            args.manifest,
            args.plan,
            args.shards_root,
            args.output_root,
            pre_gpu_receipt=args.pre_gpu_receipt,
        )
        print(json.dumps({"status": result["aggregate"]["status"], "aggregate_sha256": result["aggregate"]["aggregate_sha256"]}, sort_keys=True))
        return 0
    except (ShardMergeError, ShardExecutionError, ExecutionPlanError, FileExistsError, OSError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
