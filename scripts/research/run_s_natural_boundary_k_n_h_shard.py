#!/usr/bin/env python3
"""Execute one immutable shard of the sealed S K/N/H cohort.

The shard runner is deliberately separate from the frozen cohort runner.  The
planner decides membership and the runner only executes the already assigned
event roots; it never reads interventions or performs outcome-adaptive
selection.  A shard is allowed to contain fewer events or images than the
full-manifest execution gate because qualification belongs to the merger.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.plan_s_natural_boundary_k_n_h_execution import (  # noqa: E402
    ExecutionPlanError,
    CONTRACT_MAX_SCALAR_FORWARD_PER_EVENT,
    _shard_id,
    _write_once,
    validate_plan,
)
from scripts.research.run_s_natural_boundary_k_n_h_cohort import (  # noqa: E402
    ARM_ORDER,
    CohortContractError,
    _event_result_document,
    _load_completed_event,
    _normalize_executor_output,
    RECEIPT_SCHEMA_VERSION,
    sha256_json,
    UNIT_ID,
    validate_arm_result,
    validate_manifest,
)


SHARD_AGGREGATE_SCHEMA_VERSION = "s_natural_boundary_k_n_h_shard_aggregate.v1"
SHARD_RECEIPT_SCHEMA_VERSION = "s_natural_boundary_k_n_h_shard_receipt.v1"
SHARD_COUNT = 8
ENV_MANIFEST = "S_NATURAL_BOUNDARY_MANIFEST"
ENV_CENSUS = "S_NATURAL_BOUNDARY_CENSUS"
ENV_CONFIG = "S_NATURAL_BOUNDARY_CONFIG"
ENV_PANEL = "S_NATURAL_BOUNDARY_PANEL"
ENV_COHORT = "S_NATURAL_BOUNDARY_COHORT"
ENV_COHORT_MANIFEST = "S_NATURAL_BOUNDARY_COHORT_MANIFEST"
ENV_H0_ROOT = "S_NATURAL_BOUNDARY_H0_ROOT"
ENV_H0_DIR = "S_NATURAL_BOUNDARY_H0_DIR"
ENV_PRE_GPU_RECEIPT = "S_NATURAL_BOUNDARY_PRE_GPU_RECEIPT"
ENV_SHARD_ID = "S_NATURAL_BOUNDARY_SHARD_ID"
CANONICAL_EXECUTOR_MODULE = "scripts.research.s_natural_boundary_k_n_h_live_executor"
CANONICAL_EXECUTOR_FUNCTION = "execute_event"

EventExecutor = Callable[..., Mapping[str, Any]]


class ShardExecutionError(CohortContractError):
    """Raised when a shard execution or receipt is incompatible."""


@dataclass(frozen=True)
class _PrelaunchContext:
    manifest_path: str
    manifest_sha256: str
    plan_path: str
    plan_file_sha256: str
    plan_sha256: str
    output_root: str
    shard_id: str
    shard_index: int
    receipt_path: str
    receipt_self_sha256: str
    authorization_sha256: str
    runtime_identity: dict[str, Any]
    runtime_versions: dict[str, str]
    observed_cuda_visible_devices: str


def _required_environment_path(env_name: str, *, directory: bool = False) -> Path:
    raw = os.environ.get(env_name)
    if not raw:
        raise ShardExecutionError(f"required prelaunch identity environment {env_name} is missing")
    source = Path(raw).expanduser()
    if source.is_symlink() or (not source.is_dir() if directory else not source.is_file()):
        kind = "directory" if directory else "file"
        raise ShardExecutionError(f"{env_name} is not a regular non-symlink {kind}: {source.resolve()}")
    return source.resolve(strict=True)


def _runtime_version_attestation(identity: Mapping[str, Any]) -> dict[str, str]:
    expected = identity.get("runtime")
    if not isinstance(expected, Mapping):
        raise ShardExecutionError("pre-GPU runtime version policy is missing")
    observed = {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "torch_version": importlib.metadata.version("torch"),
        "transformers_version": importlib.metadata.version("transformers"),
    }
    if any(expected.get(key) != value for key, value in observed.items()):
        raise ShardExecutionError("installed runtime versions differ from the pre-GPU binding")
    return observed


def validate_prelaunch(
    manifest: str | Path,
    plan: str | Path,
    output_root: str | Path,
    *,
    shard_id: str | int,
) -> _PrelaunchContext:
    """Validate the canonical CPU-only receipt before executor import/output.

    This function deliberately owns the one prelaunch call for a shard.  The
    live executor performs only runtime/input checks after this boundary.
    """

    shard, shard_index = _normalize_shard_id(shard_id)
    output_source = Path(output_root).expanduser()
    if output_source.is_symlink() or output_source.exists():
        raise ShardExecutionError(f"prelaunch shard output root must be absent: {output_source.resolve()}")
    output = output_source.resolve()
    manifest_source = Path(manifest).expanduser()
    plan_source = Path(plan).expanduser()
    if manifest_source.is_symlink() or not manifest_source.is_file():
        raise ShardExecutionError(f"prelaunch manifest is not a regular file: {manifest_source.resolve()}")
    if plan_source.is_symlink() or not plan_source.is_file():
        raise ShardExecutionError(f"prelaunch plan is not a regular file: {plan_source.resolve()}")
    manifest_path = manifest_source.resolve(strict=True)
    plan_path = plan_source.resolve(strict=True)
    environment_manifest = _required_environment_path(ENV_MANIFEST)
    if environment_manifest != manifest_path:
        raise ShardExecutionError("executor manifest environment differs from shard manifest")
    expected_paths = {
        "manifest": manifest_path,
        "execution_plan": plan_path,
        "census": _required_environment_path(ENV_CENSUS),
        "config": _required_environment_path(ENV_CONFIG),
        "panel": _required_environment_path(ENV_PANEL),
        "cohort": _required_environment_path(ENV_COHORT),
        "cohort_manifest": _required_environment_path(ENV_COHORT_MANIFEST),
        "h0_root": _required_environment_path(ENV_H0_ROOT, directory=True),
        "h0_dir": _required_environment_path(ENV_H0_DIR, directory=True),
    }
    receipt_path = _required_environment_path(ENV_PRE_GPU_RECEIPT)
    try:
        receipt_raw = receipt_path.read_bytes()
        receipt = json.loads(receipt_raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ShardExecutionError(f"cannot read pre-GPU receipt {receipt_path}: {exc}") from exc
    if not isinstance(receipt, Mapping):
        raise ShardExecutionError("pre-GPU receipt must be a JSON object")
    manifest_info = validate_manifest(manifest_path)
    plan_info = validate_plan(plan_path, manifest_info=manifest_info)
    event_sha256s = [ref["event_sha256"] for ref in plan_info["shards"][shard_index]["events"]]
    observed_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        pre_gpu = importlib.import_module("scripts.research.seal_s_natural_boundary_k_n_h_pre_gpu_receipt")
        # This is the shard's one full receipt validation.  Downstream event
        # execution consumes this binding and never rehashes the model tree.
        runtime_identity = pre_gpu.runtime_identity_binding(
            receipt,
            receipt_path=receipt_path,
            shard_id=shard,
            observed_cuda_visible_devices=observed_cuda_visible_devices,
        )
        bound_paths = runtime_identity.get("input_paths")
        if not isinstance(bound_paths, Mapping) or any(
            bound_paths.get(key) != str(value) for key, value in expected_paths.items()
        ):
            raise ShardExecutionError(
                "pre-GPU input paths differ from the guarded runner environment"
            )
        authorization = pre_gpu.validate_shard_authorization(
            receipt,
            shard_id=shard,
            output_root=output,
            event_sha256s=event_sha256s,
            observed_cuda_visible_devices=observed_cuda_visible_devices,
        )
    except Exception as exc:
        raise ShardExecutionError(f"pre-GPU shard authorization failed: {exc}") from exc
    receipt_self_sha256 = receipt.get("self_sha256")
    authorization_sha256 = authorization.get("authorization_sha256")
    if not isinstance(receipt_self_sha256, str) or not isinstance(authorization_sha256, str):
        raise ShardExecutionError("pre-GPU authorization hashes are missing")
    runtime_versions = _runtime_version_attestation(runtime_identity)
    return _PrelaunchContext(
        manifest_path=str(manifest_path),
        manifest_sha256=manifest_info["manifest_sha256"],
        plan_path=str(plan_path),
        plan_file_sha256=plan_info["source"]["sha256"],
        plan_sha256=plan_info["plan_sha256"],
        output_root=str(output),
        shard_id=shard,
        shard_index=shard_index,
        receipt_path=str(receipt_path),
        receipt_self_sha256=receipt_self_sha256,
        authorization_sha256=authorization_sha256,
        runtime_identity=dict(runtime_identity),
        runtime_versions=runtime_versions,
        observed_cuda_visible_devices=observed_cuda_visible_devices,
    )


def _normalize_shard_id(value: str | int) -> tuple[str, int]:
    if isinstance(value, bool):
        raise ShardExecutionError("shard id must be an integer or shard-NNN string")
    if isinstance(value, int):
        index = value
        shard = _shard_id(index) if 0 <= index < SHARD_COUNT else ""
    elif isinstance(value, str):
        if not value.startswith("shard-") or len(value) != len("shard-000"):
            raise ShardExecutionError("shard id must use shard-NNN syntax")
        try:
            index = int(value[6:])
        except ValueError as exc:
            raise ShardExecutionError("shard id must use shard-NNN syntax") from exc
        shard = value
    else:
        raise ShardExecutionError("shard id must be an integer or shard-NNN string")
    if not 0 <= index < SHARD_COUNT or shard != _shard_id(index):
        raise ShardExecutionError(f"shard id is outside the frozen 0..{SHARD_COUNT - 1} range")
    return shard, index


def _event_root_name(event: Mapping[str, Any]) -> str:
    return f"event-{event['event_index']:06d}-{str(event['event_id']).replace(':', '-') }"


def _validate_shard_contents(root: Path, events: Sequence[Mapping[str, Any]]) -> None:
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise ShardExecutionError(f"shard output root is not a regular directory: {root}")
    if not root.exists():
        return
    allowed = {"aggregate.json", "aggregate.receipt.json"} | {_event_root_name(event) for event in events}
    for child in root.iterdir():
        if child.name not in allowed:
            raise ShardExecutionError(f"shard output contains a foreign file or root: {child}")
        if child.name.startswith("event-") and (child.is_symlink() or not child.is_dir()):
            raise ShardExecutionError(f"shard event root is not a regular directory: {child}")
        if child.name in {"aggregate.json", "aggregate.receipt.json"} and child.is_symlink():
            raise ShardExecutionError(f"shard aggregate receipt is a symlink: {child}")


def _event_with_shard_binding(document: Mapping[str, Any], *, plan_sha256: str, shard: str, shard_index: int) -> dict[str, Any]:
    body = dict(document)
    body["plan_sha256"] = plan_sha256
    body["shard_id"] = shard
    body["shard_index"] = shard_index
    old_hash = body.pop("result_sha256", None)
    if not isinstance(old_hash, str):
        raise ShardExecutionError("new event result lacks result_sha256")
    body["result_sha256"] = sha256_json(body)
    return body


def _validate_event_plan_binding(document: Mapping[str, Any], *, plan_sha256: str, shard: str, shard_index: int) -> None:
    if (
        document.get("plan_sha256") != plan_sha256
        or document.get("shard_id") != shard
        or document.get("shard_index") != shard_index
    ):
        raise ShardExecutionError("event root is bound to a different plan or shard")


def _event_scalars(event_document: Mapping[str, Any]) -> tuple[int, int, dict[str, int], dict[str, int]]:
    arms = event_document.get("arms")
    if not isinstance(arms, Mapping) or set(arms) != set(ARM_ORDER):
        raise ShardExecutionError("event result arm set is incomplete")
    scalar: dict[str, int] = {}
    runtime: dict[str, int] = {}
    for arm in ARM_ORDER:
        value = arms[arm]
        if not isinstance(value, Mapping):
            raise ShardExecutionError(f"event result arm {arm} is malformed")
        count = value.get("scalar_forward_count")
        runtime_count = value.get("runtime_scalar_forward_count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ShardExecutionError(f"event result arm {arm} scalar count is invalid")
        if isinstance(runtime_count, bool) or not isinstance(runtime_count, int) or runtime_count < 0:
            raise ShardExecutionError(f"event result arm {arm} runtime scalar count is invalid")
        scalar[arm] = count
        runtime[arm] = runtime_count
    return sum(scalar.values()), sum(runtime.values()), scalar, runtime


def _validate_existing_event(
    event_document: Mapping[str, Any],
    *,
    event: Mapping[str, Any],
    plan_sha256: str,
    shard: str,
    shard_index: int,
    expected_pre_gpu_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _validate_event_plan_binding(event_document, plan_sha256=plan_sha256, shard=shard, shard_index=shard_index)
    executor_identity = event_document.get("executor_identity")
    if not isinstance(executor_identity, Mapping):
        raise ShardExecutionError(f"event {event['event_id']} lacks executor identity")
    if expected_pre_gpu_identity is not None and executor_identity.get("pre_gpu") != dict(expected_pre_gpu_identity):
        raise ShardExecutionError(f"event {event['event_id']} executor identity differs from prelaunch")
    scalar, runtime, by_arm, runtime_by_arm = _event_scalars(event_document)
    if scalar > CONTRACT_MAX_SCALAR_FORWARD_PER_EVENT:
        raise ShardExecutionError(
            f"event {event['event_id']} exceeded the contract scalar-forward bound: {scalar}"
        )
    if scalar != runtime:
        raise ShardExecutionError(f"event {event['event_id']} scalar/runtime totals differ")
    return {
        "event_index": event["event_index"],
        "event_id": event["event_id"],
        "image_id": event["image_id"],
        "event_sha256": event["event_sha256"],
        "result_sha256": event_document["result_sha256"],
        "root": _event_root_name(event),
        "scalar_forward_count": scalar,
        "runtime_scalar_forward_count": runtime,
        "scalar_forward_counts": by_arm,
        "runtime_scalar_forward_counts": runtime_by_arm,
        "executor_identity": dict(executor_identity),
    }


def _run_one_event(
    *,
    event: Mapping[str, Any],
    manifest_info: Mapping[str, Any],
    root: Path,
    plan_sha256: str,
    shard: str,
    shard_index: int,
    executor: EventExecutor,
    expected_pre_gpu_identity: Mapping[str, Any] | None,
) -> dict[str, Any]:
    event_root = root / _event_root_name(event)
    existing = _load_completed_event(event_root, event=event, manifest_info=manifest_info)
    if existing is not None:
        return _validate_existing_event(
            existing,
            event=event,
            plan_sha256=plan_sha256,
            shard=shard,
            shard_index=shard_index,
            expected_pre_gpu_identity=expected_pre_gpu_identity,
        )
    raw = executor(event, arm_order=ARM_ORDER)
    if not isinstance(raw, Mapping):
        raise ShardExecutionError(f"executor output for {event['event_id']} must be an object")
    arm_values = _normalize_executor_output(raw, event["event_id"])
    if tuple(arm_values) != ARM_ORDER:
        raise ShardExecutionError(f"executor arm set/order differs for {event['event_id']}")
    validated = {arm: validate_arm_result(arm_values[arm], arm) for arm in ARM_ORDER}
    executor_identities = [validated[arm]["executor_identity"] for arm in ARM_ORDER]
    if any(identity != executor_identities[0] for identity in executor_identities[1:]):
        raise ShardExecutionError(f"executor identities differ across arms for {event['event_id']}")
    if expected_pre_gpu_identity is not None and executor_identities[0].get("pre_gpu") != dict(expected_pre_gpu_identity):
        raise ShardExecutionError(f"executor identity differs from prelaunch for {event['event_id']}")
    resolved = {arm: validated[arm]["resolved_opener_token_id"] for arm in ARM_ORDER}
    if len(set(resolved.values())) != 1:
        raise ShardExecutionError(f"executor opener token resolution differs for {event['event_id']}")
    event_document = _event_result_document(
        manifest_info,
        event,
        {arm: validated[arm]["result"] for arm in ARM_ORDER},
        {arm: validated[arm]["outcomes"] for arm in ARM_ORDER},
        resolved,
    )
    event_document = _event_with_shard_binding(
        event_document, plan_sha256=plan_sha256, shard=shard, shard_index=shard_index
    )
    # Validate counts before creating any receipt.  This keeps a failed event
    # from leaving a seemingly complete terminal summary behind.
    scalar, runtime, _, _ = _event_scalars(event_document)
    if scalar > CONTRACT_MAX_SCALAR_FORWARD_PER_EVENT or scalar != runtime:
        raise ShardExecutionError(f"event {event['event_id']} violates scalar-forward bounds")
    _write_once(event_root / "result.json", event_document)
    terminal = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "completed",
        "unit_id": UNIT_ID,
        "event_index": event["event_index"],
        "event_id": event["event_id"],
        "event_sha256": event["event_sha256"],
        "manifest_self_sha256": manifest_info["manifest_self_sha256"],
        "manifest_sha256": manifest_info["manifest_sha256"],
        "arm_order": list(ARM_ORDER),
        "result_sha256": event_document["result_sha256"],
    }
    terminal["self_sha256"] = sha256_json(terminal)
    _write_once(event_root / "terminal_summary.json", terminal)
    return _validate_existing_event(
        event_document,
        event=event,
        plan_sha256=plan_sha256,
        shard=shard,
        shard_index=shard_index,
        expected_pre_gpu_identity=expected_pre_gpu_identity,
    )


def _run_shard_core(
    manifest: str | Path | Mapping[str, Any],
    plan: str | Path | Mapping[str, Any],
    output_root: str | Path,
    *,
    shard_id: str | int,
    executor: EventExecutor,
    expected_pre_gpu_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest_info = validate_manifest(manifest)
    plan_info = validate_plan(plan, manifest_info=manifest_info)
    shard, shard_index = _normalize_shard_id(shard_id)
    planned_shard = plan_info["shards"][shard_index]
    if planned_shard["shard_id"] != shard:
        raise ShardExecutionError("plan shard identity is inconsistent")
    events_by_index = {event["event_index"]: event for event in manifest_info["events"]}
    assigned = [events_by_index[ref["event_index"]] for ref in planned_shard["events"]]
    if [event["event_index"] for event in assigned] != planned_shard["event_indices"]:
        raise ShardExecutionError("plan shard event assignment is inconsistent with manifest")
    root_source = Path(output_root).expanduser()
    if root_source.is_symlink():
        raise ShardExecutionError(f"shard output root is a symlink: {root_source}")
    root = root_source.resolve()
    if expected_pre_gpu_identity is not None and root.exists():
        raise ShardExecutionError("shard output root appeared after canonical prelaunch validation")
    _validate_shard_contents(root, assigned)
    event_refs = [
        _run_one_event(
            event=event,
            manifest_info=manifest_info,
            root=root,
            plan_sha256=plan_info["plan_sha256"],
            shard=shard,
            shard_index=shard_index,
            executor=executor,
            expected_pre_gpu_identity=expected_pre_gpu_identity,
        )
        for event in assigned
    ]
    scalar_total = sum(ref["scalar_forward_count"] for ref in event_refs)
    runtime_total = sum(ref["runtime_scalar_forward_count"] for ref in event_refs)
    event_executor_identities = [ref["executor_identity"] for ref in event_refs]
    if event_executor_identities and any(
        identity != event_executor_identities[0] for identity in event_executor_identities[1:]
    ):
        raise ShardExecutionError("executor identities differ across shard events")
    shard_executor_identity = event_executor_identities[0] if event_executor_identities else None
    body = {
        "schema_version": SHARD_AGGREGATE_SCHEMA_VERSION,
        "status": "completed",
        "unit_id": UNIT_ID,
        "primary": dict(plan_info["document"]["primary"]),
        "claim_scope": dict(plan_info["document"]["claim_scope"]),
        "shard_id": shard,
        "shard_index": shard_index,
        "shard_count": SHARD_COUNT,
        "plan_sha256": plan_info["plan_sha256"],
        "manifest_self_sha256": manifest_info["manifest_self_sha256"],
        "manifest_sha256": manifest_info["manifest_sha256"],
        "gate_sha256": plan_info["document"]["gate_sha256"],
        "gate_result_sha256": plan_info["document"]["gate_result_sha256"],
        "gate_runtime_identity_sha256": plan_info["document"].get("gate_runtime_identity_sha256"),
        "gate_runtime_identity_raw_sha256": plan_info["document"].get("gate_runtime_identity_raw_sha256"),
        "arm_order": list(ARM_ORDER),
        "event_count": len(event_refs),
        "distinct_image_count": len({ref["image_id"] for ref in event_refs}),
        "scalar_forward_count": scalar_total,
        "runtime_scalar_forward_count": runtime_total,
        "executor_identity": shard_executor_identity,
        "pre_gpu_identity": (
            dict(expected_pre_gpu_identity) if expected_pre_gpu_identity is not None else None
        ),
        "scalar_forward_upper_bound": len(event_refs) * CONTRACT_MAX_SCALAR_FORWARD_PER_EVENT,
        "execution_qualification": {
            "status": "not_evaluated",
            "reason": "full_manifest_qualification_is_merger_owned",
            "event_count": len(event_refs),
            "distinct_image_count": len({ref["image_id"] for ref in event_refs}),
            "minimum_event_count": manifest_info["admission_gate"]["minimum_event_count"],
            "minimum_image_count": manifest_info["admission_gate"]["minimum_image_count"],
        },
        "events": event_refs,
        "no_event_reorder": True,
        "no_outcome_adaptive_selection": True,
        "no_sweep": True,
        "no_a3": True,
        "no_2x2": True,
        "no_p4": True,
    }
    aggregate = dict(body)
    aggregate["aggregate_sha256"] = sha256_json(body)
    _write_once(root / "aggregate.json", aggregate)
    receipt = {
        "schema_version": SHARD_RECEIPT_SCHEMA_VERSION,
        "status": "completed",
        "unit_id": UNIT_ID,
        "shard_id": shard,
        "shard_index": shard_index,
        "shard_count": SHARD_COUNT,
        "plan_sha256": plan_info["plan_sha256"],
        "claim_scope": aggregate["claim_scope"],
        "manifest_sha256": manifest_info["manifest_sha256"],
        "gate_sha256": plan_info["document"]["gate_sha256"],
        "gate_result_sha256": plan_info["document"]["gate_result_sha256"],
        "gate_runtime_identity_sha256": plan_info["document"].get("gate_runtime_identity_sha256"),
        "gate_runtime_identity_raw_sha256": plan_info["document"].get("gate_runtime_identity_raw_sha256"),
        "aggregate_sha256": aggregate["aggregate_sha256"],
        "event_count": len(event_refs),
        "distinct_image_count": len({ref["image_id"] for ref in event_refs}),
        "scalar_forward_count": scalar_total,
        "runtime_scalar_forward_count": runtime_total,
        "executor_identity": shard_executor_identity,
        "pre_gpu_identity": aggregate["pre_gpu_identity"],
        "event_roots": event_refs,
    }
    receipt["receipt_sha256"] = sha256_json(receipt)
    _write_once(root / "aggregate.receipt.json", receipt)
    return {"aggregate": aggregate, "receipt": receipt, "manifest": manifest_info, "plan": plan_info}


def _run_shard_test_only(
    manifest: str | Path | Mapping[str, Any],
    plan: str | Path | Mapping[str, Any],
    output_root: str | Path,
    *,
    shard_id: str | int,
    executor: EventExecutor,
) -> dict[str, Any]:
    """Private injected-executor seam for CPU contract tests only."""

    return _run_shard_core(manifest, plan, output_root, shard_id=shard_id, executor=executor)


def load_executor(
    *,
    pre_gpu_identity: Mapping[str, Any],
    runtime_versions: Mapping[str, str],
) -> EventExecutor:
    module = importlib.import_module(CANONICAL_EXECUTOR_MODULE)
    module_path = Path(str(getattr(module, "__file__", ""))).resolve(strict=True)
    expected_path = (REPO_ROOT / "scripts/research/s_natural_boundary_k_n_h_live_executor.py").resolve(strict=True)
    if module_path != expected_path or module_path.is_symlink():
        raise ShardExecutionError("canonical live executor module path drifted")
    expected_hash = pre_gpu_identity.get("code_hashes", {}).get("live_executor")
    if not isinstance(expected_hash, str):
        raise ShardExecutionError("pre-GPU live executor code hash is missing")
    digest = hashlib.sha256(module_path.read_bytes()).hexdigest()
    if digest != expected_hash:
        raise ShardExecutionError("canonical live executor code hash differs from pre-GPU binding")
    configure = getattr(module, "configure_pre_gpu_identity", None)
    if not callable(configure):
        raise ShardExecutionError("canonical live executor lacks configure_pre_gpu_identity")
    configure(pre_gpu_identity, runtime_versions=runtime_versions)
    function = getattr(module, CANONICAL_EXECUTOR_FUNCTION, None)
    if not callable(function):
        raise ShardExecutionError("canonical live executor function is not callable")
    return function


def run_shard_from_spec(
    manifest: str | Path,
    plan: str | Path,
    output_root: str | Path,
    *,
    shard_id: str | int,
) -> dict[str, Any]:
    """Cross the CPU prelaunch gate once, then import and execute the shard."""

    authorization = validate_prelaunch(manifest, plan, output_root, shard_id=shard_id)
    previous_shard_id = os.environ.get(ENV_SHARD_ID)
    if previous_shard_id is not None and previous_shard_id != authorization.shard_id:
        raise ShardExecutionError("existing shard identity environment differs from prelaunch authorization")
    os.environ[ENV_SHARD_ID] = authorization.shard_id
    try:
        executor = load_executor(
            pre_gpu_identity=authorization.runtime_identity,
            runtime_versions=authorization.runtime_versions,
        )
        if os.environ.get("CUDA_VISIBLE_DEVICES") != authorization.observed_cuda_visible_devices:
            raise ShardExecutionError("CUDA_VISIBLE_DEVICES drifted after prelaunch and before execution")
        return _run_shard_core(
            manifest,
            plan,
            output_root,
            shard_id=shard_id,
            executor=executor,
            expected_pre_gpu_identity=authorization.runtime_identity,
        )
    finally:
        if previous_shard_id is None:
            os.environ.pop(ENV_SHARD_ID, None)
        else:
            os.environ[ENV_SHARD_ID] = previous_shard_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--shard-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        try:
            shard_id: str | int = int(args.shard_id)
        except ValueError:
            shard_id = args.shard_id
        result = run_shard_from_spec(
            args.manifest,
            args.plan,
            args.output_root,
            shard_id=shard_id,
        )
        print(json.dumps({"status": result["aggregate"]["status"], "aggregate_sha256": result["aggregate"]["aggregate_sha256"]}, sort_keys=True))
        return 0
    except (ShardExecutionError, ExecutionPlanError, FileExistsError, OSError, ImportError, AttributeError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
