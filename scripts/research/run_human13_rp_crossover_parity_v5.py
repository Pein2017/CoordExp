#!/usr/bin/env python3
"""Run the reserved one-image Human-13 v5 parity-only qualification.

Dry-run is the default and performs no write or model action.  Live execution
is deliberately narrower than the production node runtime: native K16
acquisition followed by exact fp32/SDPA replay, once for RP 1.0 and then RP
1.10.  No owner, witness, dose, optimizer, update, or proposal service is in
this entry's dependency surface.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Protocol

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research.collect_human13_rp_crossover import QUALIFICATION_SEEDS
from scripts.research.human13_rp_policy import ReplayTolerance


SCHEMA_VERSION = "human13_rp_crossover_parity_v5.v1"
TERMINAL_SCHEMA_VERSION = "human13_rp_crossover_parity_v5_terminal.v1"
UNIT_ID = "2026-08-14-human13-k-trajectory-rp-crossover-screen"
IMAGE_ID = 1584
SEED_GROUP_ID = "qualification"
RPS = (1.0, 1.10)
SEEDS = QUALIFICATION_SEEDS
_REPLAY_TOLERANCE = ReplayTolerance()
PER_TOKEN_TOLERANCE_NATS = _REPLAY_TOLERANCE.per_token_nats
GROUP_MEAN_TOLERANCE_NATS = _REPLAY_TOLERANCE.group_mean_nats
V5_OUTPUT_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-08-14-human13-k-trajectory-rp-crossover-screen/"
    "vertical-dose-qualification-v5"
)


class ParityV5ContractError(RuntimeError):
    """Raised when the parity-only entry cannot preserve its sealed contract."""


class ParityBackend(Protocol):
    """The complete and intentionally tiny live dependency surface."""

    def open_sampler(self, frozen: Any) -> Any: ...

    def sample_batch(
        self, sampler: Any, batch: Any, params: tuple[Any, ...]
    ) -> Any: ...

    def close_sampler(self, sampler: Any) -> None: ...

    def open_packed_surface(self, frozen: Any) -> Any: ...

    def packed_raw_logits(self, packed: Any, execution: Any) -> Any: ...

    def parity_surface_lineage(
        self, packed: Any, execution: Any
    ) -> Mapping[str, Any]: ...

    def close_packed_surface(self, packed: Any) -> None: ...

    def resource_snapshot(self) -> Any: ...


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _bound(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["content_sha256"] = _sha256(result)
    return result


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    data = _canonical_bytes(payload)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _atomic_terminal(output_root: Path, terminal: Mapping[str, Any]) -> None:
    target = output_root / "terminal.json"
    staging = output_root / ".terminal.json.staging"
    if target.exists() or staging.exists():
        raise FileExistsError("refusing to overwrite the v5 terminal receipt")
    _write_json(staging, terminal)
    os.replace(staging, target)
    _fsync_directory(output_root)


def _publish_rp_evidence(
    output_root: Path,
    *,
    slug: str,
    payloads: Mapping[str, Mapping[str, Any]],
) -> None:
    target = output_root / slug
    staging = output_root / f".{slug}.staging"
    if target.exists() or staging.exists():
        raise FileExistsError(f"refusing to overwrite v5 RP evidence: {target}")
    staging.mkdir()
    try:
        for name, payload in payloads.items():
            _write_json(staging / name, payload)
        _fsync_directory(staging)
        os.replace(staging, target)
        _fsync_directory(output_root)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _rp_slug(repetition_penalty: float) -> str:
    return {1.0: "rp100", 1.10: "rp110"}[float(repetition_penalty)]


def _source_lineage(frozen: Any) -> dict[str, Any]:
    fields = (
        "source_checkpoint_path",
        "source_checkpoint_payload_sha256",
        "base_model_path",
        "adapter_tensor_path",
        "adapter_sha256",
        "special_embedding_tensor_path",
        "special_embedding_sha256",
        "manifest_path",
        "manifest_sha256",
        "panel_path",
        "panel_sha256",
        "tokenizer_path",
        "tokenizer_sha256",
        "prompt_config_path",
        "prompt_config_sha256",
        "prompt_policy_fingerprint",
        "alias_bank_sha256",
        "c_leaf_path",
        "c_leaf_sha256",
    )
    values: dict[str, Any] = {}
    for field in fields:
        value = getattr(frozen, field, None)
        if value is None:
            raise ParityV5ContractError(f"frozen Source lineage omitted {field}")
        values[field] = str(value) if isinstance(value, Path) else value
    return _bound(values)


def _resource_mapping(snapshot: Any) -> dict[str, Any]:
    if is_dataclass(snapshot) and not isinstance(snapshot, type):
        value = asdict(snapshot)
    elif hasattr(snapshot, "__dict__"):
        value = vars(snapshot)
    elif isinstance(snapshot, Mapping):
        value = dict(snapshot)
    else:
        raise ParityV5ContractError("resource snapshot is not serializable")
    return {str(key): item for key, item in value.items()}


def _error_payload(error: BaseException, *, phase: str) -> dict[str, Any]:
    error_field = getattr(error, "error_field", None)
    if error_field is None:
        field_payload = None
    elif is_dataclass(error_field) and not isinstance(error_field, type):
        field_payload = asdict(error_field)
    else:
        raise ParityV5ContractError("parity error field is not a typed dataclass")
    return {
        "phase": phase,
        "exception_type": type(error).__name__,
        "message": str(error),
        "error_field": field_payload,
    }


def _native_payload(execution: Any) -> dict[str, Any]:
    return _bound(
        {
            "schema_version": "human13_rp_crossover_parity_v5_native.v1",
            "plan": execution.plan.to_dict(),
            "plan_sha256": execution.plan_sha256,
            "plan_request_ids": list(execution.plan_request_ids),
            "native_receipts": execution.native_receipts_artifact.to_dict(),
            "acquisition_group": execution.group.to_dict(),
            "acquisition_group_sha256": execution.group.content_sha256,
        }
    )


def _typed_payload(value: Any, *, label: str) -> dict[str, Any]:
    to_dict = getattr(value, "to_dict", None)
    if not callable(to_dict):
        raise ParityV5ContractError(f"{label} does not expose to_dict()")
    payload = to_dict()
    if not isinstance(payload, Mapping):
        raise ParityV5ContractError(f"{label} did not serialize to a mapping")
    return _bound({"schema_version": f"v5_{label}.v1", "payload": dict(payload)})


def _run_one_rp(
    *,
    output_root: Path,
    repetition_penalty: float,
    backend_factory: Callable[[float, Path], ParityBackend],
    frozen_loader: Callable[[float], Any],
) -> dict[str, Any]:
    from scripts.research.collect_human13_rp_crossover import (
        execute_acquisition_group,
        plan_acquisition_group,
        replay_acquisition_group,
    )

    slug = _rp_slug(repetition_penalty)
    started = time.monotonic()
    phase = "frozen_source_admission"
    frozen: Any | None = None
    backend: ParityBackend | None = None
    execution: Any | None = None
    replayed: Any | None = None
    parity: Any | None = None
    source_lineage: Mapping[str, Any] | None = None
    surface_lineage: Mapping[str, Any] | None = None
    error: dict[str, Any] | None = None
    resource_snapshot: dict[str, Any] | None = None

    try:
        frozen = frozen_loader(repetition_penalty)
        source_lineage = _source_lineage(frozen)
        backend = backend_factory(repetition_penalty, output_root / slug)
        plan = plan_acquisition_group(
            image_id=IMAGE_ID,
            repetition_penalty=repetition_penalty,
            seed_group_id=SEED_GROUP_ID,
        )
        if tuple(request.seed for request in plan.requests) != SEEDS:
            raise ParityV5ContractError("qualification seed order drifted")

        phase = "native_acquisition"
        sampler = backend.open_sampler(frozen)
        try:
            execution = execute_acquisition_group(
                plan=plan,
                execute_batch=lambda batch, params: backend.sample_batch(
                    sampler, batch, params
                ),
            )
        finally:
            backend.close_sampler(sampler)

        phase = "exact_surface_replay"
        packed = backend.open_packed_surface(frozen)
        try:
            raw = backend.packed_raw_logits(packed, execution)
            surface_lineage = dict(backend.parity_surface_lineage(packed, execution))
            if (
                surface_lineage.get("mixed_precision") != "fp32"
                or surface_lineage.get("attn_implementation") != "sdpa"
                or surface_lineage.get("batch_size") != 1
            ):
                raise ParityV5ContractError(
                    "v5 replay surface is not exact fp32/SDPA batch one"
                )
            replayed, parity = replay_acquisition_group(
                sampled=execution.group, packed=raw
            )
        finally:
            backend.close_packed_surface(packed)
    except BaseException as caught:
        error = _error_payload(caught, phase=phase)
    finally:
        if backend is not None:
            try:
                resource_snapshot = _resource_mapping(backend.resource_snapshot())
            except BaseException as resource_error:
                if error is None:
                    error = _error_payload(resource_error, phase="resource_snapshot")

    elapsed = time.monotonic() - started
    token_count = (
        sum(
            len(trajectory.generated_tokens)
            for trajectory in execution.group.trajectories
        )
        if execution is not None
        else 0
    )
    runtime_counters = {
        "wall_time_seconds": elapsed,
        "native_request_count": len(execution.plan_request_ids)
        if execution is not None
        else 0,
        "native_batch_count": len(execution.native_batch_receipts)
        if execution is not None
        else 0,
        "decode_token_count": token_count,
        "replay_token_count": token_count if replayed is not None else 0,
        "surface": dict(surface_lineage or {}),
        "resources": resource_snapshot,
    }
    phase_status = {
        "native_acquisition": (
            "passed"
            if execution is not None and phase != "native_acquisition"
            else "failed"
            if error is not None and phase == "native_acquisition"
            else "not_started"
        ),
        "exact_surface_replay": (
            "passed"
            if parity is not None
            else "failed"
            if error is not None and phase == "exact_surface_replay"
            else "not_started"
        ),
    }
    evidence = _bound(
        {
            "schema_version": SCHEMA_VERSION,
            "unit_id": UNIT_ID,
            "status": "passed" if error is None else "failed",
            "repetition_penalty": repetition_penalty,
            "image_id": IMAGE_ID,
            "seeds": list(SEEDS),
            "native_request_count": runtime_counters["native_request_count"],
            "native_batch_count": runtime_counters["native_batch_count"],
            "parity_tolerance": {
                "per_token_nats": PER_TOKEN_TOLERANCE_NATS,
                "group_mean_nats": GROUP_MEAN_TOLERANCE_NATS,
            },
            "source_lineage": source_lineage,
            "surface_lineage": dict(surface_lineage or {}),
            "phase_status": phase_status,
            "runtime_counters": runtime_counters,
            "error": error,
        }
    )
    payloads: dict[str, Mapping[str, Any]] = {"parity-evidence.json": evidence}
    if execution is not None:
        payloads["native-acquisition.json"] = _native_payload(execution)
    if replayed is not None and parity is not None:
        payloads["replayed-group.json"] = _typed_payload(
            replayed, label="replayed_group"
        )
        payloads["replay-receipt.json"] = _typed_payload(parity, label="replay_receipt")
    elif error is not None:
        payloads["parity-error.json"] = _bound(
            {
                "schema_version": "human13_rp_crossover_parity_v5_error.v1",
                "error": error,
            }
        )
    _publish_rp_evidence(output_root, slug=slug, payloads=payloads)
    return evidence


def _aggregate_resources(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counters = [result["runtime_counters"] for result in results]
    resources = [
        counter["resources"]
        for counter in counters
        if isinstance(counter.get("resources"), Mapping)
    ]

    def maximum(name: str) -> int | None:
        values = [value.get(name) for value in resources if value.get(name) is not None]
        return max(int(value) for value in values) if values else None

    return {
        "wall_time_seconds": sum(float(item["wall_time_seconds"]) for item in counters),
        "native_request_count": sum(
            int(item["native_request_count"]) for item in counters
        ),
        "native_batch_count": sum(int(item["native_batch_count"]) for item in counters),
        "decode_token_count": sum(int(item["decode_token_count"]) for item in counters),
        "replay_token_count": sum(int(item["replay_token_count"]) for item in counters),
        "peak_host_rss_bytes": maximum("peak_host_rss_bytes"),
        "cuda_peak_allocated_bytes": maximum("cuda_peak_allocated_bytes"),
        "cuda_peak_reserved_bytes": maximum("cuda_peak_reserved_bytes"),
    }


def run_v5_parity_qualification(
    *,
    output_root: str | Path = V5_OUTPUT_ROOT,
    backend_factory: Callable[[float, Path], ParityBackend] | None = None,
    frozen_loader: Callable[[float], Any] | None = None,
    execution_authorized: bool = False,
) -> dict[str, Any]:
    """Execute the exact two-RP parity-only route and seal one terminal."""

    if execution_authorized is not True:
        raise PermissionError("v5 execution requires explicit execution authority")
    root = Path(output_root).expanduser().resolve()
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"refusing to reuse v5 output root: {root}")
    root.mkdir(parents=True, exist_ok=False)
    if backend_factory is None:
        backend_factory = _production_backend_factory
    if frozen_loader is None:
        frozen_loader = _production_frozen_loader

    results: list[dict[str, Any]] = []
    for repetition_penalty in RPS:
        result = _run_one_rp(
            output_root=root,
            repetition_penalty=repetition_penalty,
            backend_factory=backend_factory,
            frozen_loader=frozen_loader,
        )
        results.append(result)
        if result["status"] != "passed":
            break

    passed = len(results) == len(RPS) and all(
        result["status"] == "passed" for result in results
    )
    failed = next((result for result in results if result["status"] != "passed"), None)
    numeric_parity_failure = bool(
        failed
        and isinstance(failed.get("error"), Mapping)
        and failed["error"].get("error_field") is not None
    )
    terminal = _bound(
        {
            "schema_version": TERMINAL_SCHEMA_VERSION,
            "unit_id": UNIT_ID,
            "status": "passed" if passed else "failed",
            "output_root": str(root),
            "image_ids": [IMAGE_ID],
            "seeds": list(SEEDS),
            "rp_order": list(RPS),
            "completed_rps": [
                result["repetition_penalty"]
                for result in results
                if result["status"] == "passed"
            ],
            "failed_rp": failed["repetition_penalty"] if failed else None,
            "parity_tolerance": {
                "per_token_nats": PER_TOKEN_TOLERANCE_NATS,
                "group_mean_nats": GROUP_MEAN_TOLERANCE_NATS,
            },
            "phase_status_by_rp": {
                _rp_slug(float(result["repetition_penalty"])): result["phase_status"]
                for result in results
            },
            "prohibited_phases": [
                "owner_analysis",
                "witness_bank",
                "jacobian",
                "dose_ray",
                "optimizer_update",
                "compiler_ledger",
                "clean_greedy_proposal_audit",
            ],
            "rp_evidence": [
                {
                    "repetition_penalty": result["repetition_penalty"],
                    "status": result["status"],
                    "content_sha256": result["content_sha256"],
                    "path": _rp_slug(float(result["repetition_penalty"])),
                }
                for result in results
            ],
            "resource_counters": _aggregate_resources(results),
            "route_disposition": (
                "full_panel_successor_allowed"
                if passed
                else "retire_exact_on_policy_route"
                if numeric_parity_failure
                else "hold_infrastructure_failure"
            ),
            "error": failed["error"] if failed else None,
        }
    )
    _atomic_terminal(root, terminal)
    return terminal


class _ProductionParityBackend:
    """Narrow adapter over the accepted production backend and pack planner."""

    def __init__(self, repetition_penalty: float, rp_root: Path) -> None:
        from scripts.research.human13_rp_crossover_production_backend import (
            Human13RPCrossoverProductionBackend,
        )

        self._repetition_penalty = repetition_penalty
        self._owner = Human13RPCrossoverProductionBackend(
            {"cells": ({"output_root": str(rp_root / "parity-only")},)}
        )

    def open_sampler(self, frozen: Any) -> Any:
        return self._owner.open_sampler(frozen)

    def sample_batch(self, sampler: Any, batch: Any, params: tuple[Any, ...]) -> Any:
        return self._owner.sample_batch(sampler, batch, params)

    def close_sampler(self, sampler: Any) -> None:
        self._owner.close_sampler(sampler)

    def open_packed_surface(self, frozen: Any) -> Any:
        return self._owner.open_parity_surface(frozen, image_id=IMAGE_ID)

    def packed_raw_logits(self, packed: Any, execution: Any) -> Any:
        return self._owner.packed_raw_logits(packed, execution)

    def parity_surface_lineage(self, packed: Any, execution: Any) -> Mapping[str, Any]:
        from scripts.research.human13_rp_crossover_live_packs import (
            default_exact_history_forward,
            plan_live_packs,
        )

        assembly = getattr(packed, "assembly", None)
        skeletons = getattr(packed, "skeletons", None)
        if (
            assembly is None
            or not isinstance(skeletons, Mapping)
            or tuple(skeletons) != (IMAGE_ID,)
            or hasattr(skeletons[IMAGE_ID], "owner_row_tokens")
        ):
            raise ParityV5ContractError("production exact surface lacks assembly")
        model_plan = assembly.plan
        model_lineage = _provisional_model_plan_lineage(model_plan)
        pack_plan = plan_live_packs(
            execution=execution,
            skeleton=skeletons[IMAGE_ID],
        )
        forward_owner = (
            f"{default_exact_history_forward.__module__}:"
            f"{default_exact_history_forward.__name__}"
        )
        return {
            **model_lineage,
            "model_validation": assembly.validation.to_artifact_dict(),
            "mixed_precision": model_plan.mixed_precision,
            "attn_implementation": model_plan.attn_implementation,
            "batch_size": 1,
            "forward_owner": forward_owner,
            "live_pack_plan_sha256": pack_plan.plan_sha256,
            "acquisition_group_sha256": pack_plan.acquisition_group_sha256,
            "materialize_forward_count": len(pack_plan.pack_requests),
            "exact_history_model_forward_count": len(pack_plan.trajectory_bindings),
            "packed_token_count": sum(
                int(item.pack_length) for item in pack_plan.pack_requests
            ),
            "logical_token_count": sum(
                int(item.encoded_length)
                for item in pack_plan.packed_plan.logical_segments
            ),
            "observed_exact_surface_gate_passed": True,
        }

    def close_packed_surface(self, packed: Any) -> None:
        self._owner.close_packed_surface(packed)

    def resource_snapshot(self) -> Any:
        return self._owner.resource_snapshot()


def _provisional_model_plan_lineage(model_plan: Any) -> dict[str, Any]:
    """Admit and bind the actual pre-selection qualification plan state."""

    from scripts.research.human13_live_model import (
        Human13LiveModelPlan,
        RP_CROSSOVER_LEARNING_RATE_RAY,
    )

    if (
        type(model_plan) is not Human13LiveModelPlan
        or model_plan.unit_id != UNIT_ID
        or getattr(model_plan, "arm_id", None) != "C"
        or getattr(model_plan, "learning_rate_resolution", None)
        != "provisional_qualification"
        or getattr(model_plan, "global_learning_rate_decision_sha256", None)
        is not None
        or getattr(model_plan, "resolved_plan_sha256", None) is not None
        or getattr(model_plan, "mixed_precision", None) != "fp32"
        or getattr(model_plan, "attn_implementation", None) != "sdpa"
        or getattr(model_plan, "learning_rate", None)
        not in RP_CROSSOVER_LEARNING_RATE_RAY
    ):
        raise ParityV5ContractError(
            "production model plan is not the provisional fp32/SDPA C plan"
        )
    artifact = model_plan.to_artifact_dict()
    if not isinstance(artifact, Mapping):
        raise ParityV5ContractError("provisional model plan is not content-bindable")
    bound = dict(artifact)
    return {
        "model_plan_content_sha256": _sha256(bound),
        "model_plan": bound,
    }


def _production_backend_factory(
    repetition_penalty: float, rp_root: Path
) -> ParityBackend:
    return _ProductionParityBackend(repetition_penalty, rp_root)


def _production_frozen_loader(repetition_penalty: float) -> Any:
    from scripts.research.human13_rp_crossover_production import (
        _validate_frozen_inputs,
    )

    return _validate_frozen_inputs(repetition_penalty)


def dry_run_plan(*, output_root: str | Path = V5_OUTPUT_ROOT) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "dry_run",
        "output_root": str(Path(output_root).expanduser().resolve()),
        "image_ids": [IMAGE_ID],
        "seeds": list(SEEDS),
        "rp_order": list(RPS),
        "native_batch_size": 4,
        "trajectories_per_image": 16,
        "replay_surface": "hf_fp32_sdpa_batch_one_exact_history",
        "parity_tolerance": {
            "per_token_nats": PER_TOKEN_TOLERANCE_NATS,
            "group_mean_nats": GROUP_MEAN_TOLERANCE_NATS,
        },
        "model_actions": 0,
        "filesystem_writes": 0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--user-model-gpu-authority",
        action="store_true",
        help="Acknowledge the existing explicit user model/GPU authority.",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    _output_root: str | Path = V5_OUTPUT_ROOT,
    _runner: Callable[..., dict[str, Any]] = run_v5_parity_qualification,
) -> int:
    args = build_parser().parse_args(argv)
    if not args.execute:
        print(
            json.dumps(dry_run_plan(output_root=_output_root), sort_keys=True, indent=2)
        )
        return 0
    if not args.user_model_gpu_authority:
        raise PermissionError("--execute requires --user-model-gpu-authority")
    terminal = _runner(output_root=_output_root, execution_authorized=True)
    print(json.dumps(terminal, sort_keys=True, indent=2))
    return 0 if terminal["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "IMAGE_ID",
    "RPS",
    "SEEDS",
    "V5_OUTPUT_ROOT",
    "ParityV5ContractError",
    "dry_run_plan",
    "run_v5_parity_qualification",
    "main",
]
