"""Small, explicit state payload for deterministic training continuation."""

from __future__ import annotations

import os
import io
import random
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from src.common.errors import RuntimeContractError
from src.artifacts.git_identity import verify_source_identity, SourceIdentityError


STATE_FILE = "training_state.pt"
STATE_VERSION = 2
REQUIRED_SOURCE_PATHS = ("src/training/pipeline.py", "src/training/resume.py", "src/artifacts/git_identity.py")


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state().cpu(),
    }
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:  # pragma: no cover - numpy is a runtime dependency.
        state["numpy"] = None
    state["torch_cuda"] = (
        [item.cpu() for item in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else []
    )
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    if not isinstance(state, Mapping):
        raise RuntimeContractError(
            "resume RNG state must be a mapping",
            code="training.resume_rng_state_invalid",
        )
    try:
        random.setstate(state["python"])
        torch.set_rng_state(torch.as_tensor(state["torch_cpu"], dtype=torch.uint8).cpu())
        if state.get("numpy") is not None:
            import numpy as np

            np.random.set_state(state["numpy"])
        cuda_states = state.get("torch_cuda", ())
        if cuda_states and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(
                [torch.as_tensor(item, dtype=torch.uint8).cpu() for item in cuda_states]
            )
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise RuntimeContractError(
            "resume RNG state is malformed",
            code="training.resume_rng_state_invalid",
            cause=exc,
        ) from exc


def build_training_state(
    *,
    step: int,
    schedule: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    data_identity: Mapping[str, Any],
    optimizer_identity: Mapping[str, Any],
    runtime: Any,
) -> dict[str, Any]:
    if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
        raise RuntimeContractError(
            "resume checkpoint step must be a positive integer",
            code="training.resume_step_invalid",
        )
    optimizer = getattr(runtime, "optimizer", None)
    scheduler = getattr(runtime, "scheduler", None)
    return {
        "schema": STATE_VERSION,
        "step": step,
        "schedule": dict(schedule),
        "source_identity": dict(source_identity),
        "data_identity": dict(data_identity),
        "optimizer_identity": dict(optimizer_identity),
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "rng": capture_rng_state(),
    }


def save_training_state(
    path: Path | str, state: Mapping[str, Any], *, source_root: Path | None = None,
    required_paths: tuple[str, ...] = REQUIRED_SOURCE_PATHS,
) -> None:
    source = state.get("source_identity", {}).get("execution_source")
    if source is not None:
        verify_source_identity(source, required_paths=required_paths, root=source_root)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        torch.save(dict(state), temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        _fsync_directory(target.parent)
        # A separate JSON gate is read before torch deserialization on resume.
        # Unqualified direct callers may save state, but cannot continue it.
        source = state.get("source_identity", {}).get("execution_source")
        gate = {"schema": "training.clean_source_gate.v1", "source_identity": source,
                "payload_sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
        sidecar = target.with_suffix(".source.json")
        staged_gate = sidecar.with_name(f".{sidecar.name}.{os.getpid()}.tmp")
        try:
            staged_gate.write_text(json.dumps(gate, sort_keys=True) + "\n")
            with staged_gate.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(staged_gate, sidecar)
            _fsync_directory(target.parent)
        finally:
            staged_gate.unlink(missing_ok=True)
    finally:
        temporary.unlink(missing_ok=True)


def load_training_state(
    path: Path | str, *, source_root: Path | None = None,
    required_paths: tuple[str, ...] = REQUIRED_SOURCE_PATHS,
) -> dict[str, Any]:
    target = Path(path)
    try:
        gate = json.loads(target.with_suffix(".source.json").read_text())
        if gate.get("schema") != "training.clean_source_gate.v1":
            raise ValueError("unsupported source gate schema")
        verify_source_identity(gate.get("source_identity", {}), required_paths=required_paths, root=source_root)
        payload_bytes = target.read_bytes()
        if hashlib.sha256(payload_bytes).hexdigest() != gate.get("payload_sha256"):
            raise ValueError("training state bytes do not match the source gate")
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        raise RuntimeContractError(
            "historical/unsupported for continuation: training source identity is not verifiable",
            code="training.resume_source_identity", cause=exc,
        ) from exc
    try:
        state = torch.load(io.BytesIO(payload_bytes), map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeContractError(
            "resume checkpoint state cannot be loaded",
            code="training.resume_state_load_failed",
            context={"path": str(target)},
            cause=exc,
        ) from exc
    if not isinstance(state, dict) or state.get("schema") != STATE_VERSION:
        raise RuntimeContractError(
            "resume checkpoint state has an unsupported schema",
            code="training.resume_state_schema",
            context={"path": str(target), "schema": state.get("schema") if isinstance(state, dict) else None},
        )
    if state.get("source_identity", {}).get("execution_source") != gate["source_identity"]:
        raise RuntimeContractError("training state source gate differs", code="training.resume_source_identity")
    return state


def validate_resume_state(
    state: Mapping[str, Any],
    *,
    schedule: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    data_identity: Mapping[str, Any],
    optimizer_identity: Mapping[str, Any],
) -> int:
    if not isinstance(state, Mapping):
        raise RuntimeContractError(
            "resume state must be a mapping",
            code="training.resume_state_schema",
        )
    if state.get("schema") != STATE_VERSION:
        _incompatible("schema", state.get("schema"), STATE_VERSION)
    step = state.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
        _incompatible("step", step, "positive integer")
    for name, expected in (
        ("source_identity", source_identity),
        ("data_identity", data_identity),
        ("optimizer_identity", optimizer_identity),
    ):
        if state.get(name) != dict(expected):
            _incompatible(name, state.get(name), dict(expected))
    saved_schedule = state.get("schedule")
    if not isinstance(saved_schedule, Mapping):
        _incompatible("schedule", saved_schedule, dict(schedule))
    for key in ("resolved_max_steps", "packs_per_epoch", "runtime_batch"):
        if saved_schedule.get(key) != schedule.get(key):
            _incompatible(f"schedule.{key}", saved_schedule.get(key), schedule.get(key))
    for name in ("optimizer", "scheduler", "rng"):
        if name not in state:
            _incompatible(name, None, "present")
    rank_rngs = state.get("rank_rngs")
    if not isinstance(rank_rngs, list) or not rank_rngs:
        _incompatible("rank_rngs", rank_rngs, "non-empty list")
    if not isinstance(state.get("model_payloads"), Mapping) or not state["model_payloads"]:
        _incompatible("model_payloads", state.get("model_payloads"), "non-empty mapping")
    return step


def validate_model_payloads(checkpoint_dir: Path | str, state: Mapping[str, Any]) -> None:
    expected = state.get("model_payloads")
    if not isinstance(expected, Mapping) or not expected:
        raise RuntimeContractError(
            "resume state is missing model payload identities",
            code="training.resume_model_payload_identity_missing",
        )
    root = Path(checkpoint_dir)
    actual: dict[str, str] = {}
    for relative in expected:
        if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise RuntimeContractError(
                "resume model payload identity contains an unsafe path",
                code="training.resume_model_payload_identity_invalid",
            )
        path = root / relative
        if not path.is_file():
            raise RuntimeContractError(
                "resume model payload is missing",
                code="training.resume_model_payload_missing",
                context={"path": str(path)},
            )
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        actual[relative] = digest.hexdigest()
    if actual != dict(expected):
        raise RuntimeContractError(
            "resume model payload identity does not match checkpoint state",
            code="training.resume_model_payload_identity_mismatch",
            context={"expected": dict(expected), "actual": actual},
        )


def restore_runtime_state(runtime: Any, state: Mapping[str, Any], *, rank: int = 0) -> None:
    optimizer = getattr(runtime, "optimizer", None)
    scheduler = getattr(runtime, "scheduler", None)
    if optimizer is None or scheduler is None:
        raise RuntimeContractError(
            "resume requires optimizer and scheduler state",
            code="training.resume_runtime_state_missing",
        )
    optimizer_state = state.get("optimizer")
    scheduler_state = state.get("scheduler")
    if not isinstance(optimizer_state, Mapping) or not isinstance(scheduler_state, Mapping):
        raise RuntimeContractError(
            "resume optimizer and scheduler state must be mappings",
            code="training.resume_runtime_state_invalid",
        )
    try:
        optimizer.load_state_dict(optimizer_state)
        scheduler.load_state_dict(scheduler_state)
        restore_rng_state(state["rng"])
    except (KeyError, RuntimeError, ValueError, TypeError) as exc:
        raise RuntimeContractError(
            "resume runtime state is incompatible with the current optimizer",
            code="training.resume_runtime_state_incompatible",
            context={"rank": rank},
            cause=exc,
        ) from exc
    runtime.optimizer_step_count = int(state["step"])
    runtime.scheduler_step_count = int(state["step"])


def _incompatible(name: str, actual: Any, expected: Any) -> None:
    raise RuntimeContractError(
        f"resume state is incompatible: {name}",
        code="training.resume_incompatible",
        context={"field": name, "actual": actual, "expected": expected},
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "STATE_FILE",
    "STATE_VERSION",
    "build_training_state",
    "capture_rng_state",
    "load_training_state",
    "restore_rng_state",
    "restore_runtime_state",
    "save_training_state",
    "validate_model_payloads",
    "validate_resume_state",
]
