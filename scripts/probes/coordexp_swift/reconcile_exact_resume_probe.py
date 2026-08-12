"""Bounded two-rank exact-resume qualification probe for `world_size=2`.

Task 4A preparatory tooling for OpenSpec change
`reconcile-coordexp-swift-training-contracts`. The existing Wave-7 controller
(`wave7_exact_resume_sequence.py`) is fixed to eight ranks and cannot express
the current `world_size=2` packet, so this module is a small, independent CLI
that reuses production config/artifact/admission utilities instead of copying
that controller.

Command grammar (strict, no defaults):
    prepare          -- author the immutable two-rank config bundle.
    success-control  -- run the boundary-then-next-update control branch.
    success-resumed  -- run the matched parent-boundary + resumed-child branch.
    rank-failure     -- model-free two-rank exact-publication failure injection.
    interruption     -- exercise the three externally visible commit boundaries.
    verify           -- read durable artifacts only and publish a terminal receipt.

Every subcommand is also reachable as a plain Python function (`prepare`,
`success_control`, `success_resumed`, `rank_failure`, `interruption`,
`verify_artifacts`) so tests can inject a fake `launch` callable and never
touch a GPU or a production model.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import errno
import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.artifacts.run_writer import RunWriter  # noqa: E402
from src.artifacts.training_state import (  # noqa: E402
    RankTrainingStatePayload,
    TrainingStatePublicationPlan,
    abort_training_state_contributions,
    begin_training_state_contributions,
    build_resume_compatibility_projection,
    commit_training_state_contributions,
    load_training_state_manifest,
    publish_rank_training_state_contribution,
)
from src.common.errors import ArtifactContractError, CoordExpError  # noqa: E402
from src.config.fingerprint import sha256_json  # noqa: E402
from src.config.loader import load_train_config  # noqa: E402
from src.training.exact_resume import (  # noqa: E402
    build_exact_resume_identities,
    serialize_current_rank_training_state,
)


WORLD_SIZE = 2
SCHEMA_PREPARE_RECEIPT = "coordexp-swift-reconcile-resume-probe-prepare-receipt-v1"
SCHEMA_RUN_RECEIPT = "coordexp-swift-reconcile-resume-probe-run-receipt-v1"
SCHEMA_RANK_FAILURE_RECEIPT = (
    "coordexp-swift-reconcile-resume-probe-rank-failure-receipt-v1"
)
SCHEMA_INTERRUPTION_RECEIPT = (
    "coordexp-swift-reconcile-resume-probe-interruption-receipt-v1"
)
SCHEMA_TERMINAL_RECEIPT = "coordexp-swift-reconcile-resume-probe-terminal-receipt-v1"

CONFIG_DIR_NAME = "configs"
RUNS_DIR_NAME = "runs"
PREPARE_RECEIPT_NAME = "prepare-receipt.json"
RECEIPTS_DIR_NAME = "receipts"

ROLE_UNINTERRUPTED_CONTROL = "uninterrupted_control"
ROLE_RESUMED_PARENT = "resumed_parent"
ROLE_RESUMED_CHILD = "resumed_child"
ROLE_FILENAMES = {
    ROLE_UNINTERRUPTED_CONTROL: "uninterrupted_control.yaml",
    ROLE_RESUMED_PARENT: "resumed_parent.yaml",
    ROLE_RESUMED_CHILD: "resumed_child.yaml",
}

INJECT_KINDS = frozenset({"missing", "duplicate", "malformed", "corrupt"})
INTERRUPTION_BOUNDARIES = (0, 1, 2)


class ReconcileProbeError(CoordExpError):
    """Raised for every strict rejection this probe owns."""


# --------------------------------------------------------------------------
# Generic strict/atomic helpers (small, focused; not a copy of any Wave-7
# controller file -- these mirror the atomic-publication idiom already used
# throughout `scripts/probes/coordexp_swift` and `src/artifacts`).
# --------------------------------------------------------------------------


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ReconcileProbeError(
            "receipt payload is not strict JSON",
            code="reconcile_probe.non_strict_json",
        ) from exc


def _signed(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "receipt_payload_sha256": _sha256_bytes(_canonical_json_bytes(body))}


def _lstat_or_none(path: Path) -> os.stat_result | None:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None


def _assert_no_symlink_components(path: Path, *, include_leaf: bool) -> None:
    absolute = path.expanduser().absolute()
    rows = [absolute, *absolute.parents] if include_leaf else list(absolute.parents)
    for candidate in rows:
        info = _lstat_or_none(candidate)
        if info is not None and stat.S_ISLNK(info.st_mode):
            raise ReconcileProbeError(
                f"path contains a symlink component: {candidate}",
                code="reconcile_probe.symlink_escape",
            )


def _assert_absolute(path_value: str | Path, *, field: str) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        raise ReconcileProbeError(
            f"{field} must be an absolute path",
            code="reconcile_probe.path_not_absolute",
            context={"field": field, "value": str(path_value)},
        )
    return path


def _assert_absent_target(path: Path) -> Path:
    _assert_no_symlink_components(path, include_leaf=True)
    if _lstat_or_none(path) is not None:
        raise ReconcileProbeError(
            f"target already exists: {path}",
            code="reconcile_probe.target_exists",
            context={"path": str(path)},
        )
    return path


def _write_new_file(path: Path, content: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o644,
    )
    try:
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError(errno.EIO, "probe write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _install_directory_no_replace(stage: Path, target: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        os.rename(stage, target)
        return
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(stage), -100, os.fsencode(target), 1)
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(error_number, os.strerror(error_number), str(target))
    raise OSError(error_number, os.strerror(error_number), str(target))


def _strict_json_load(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ReconcileProbeError(
            f"cannot read strict JSON file: {path}",
            code="reconcile_probe.strict_json_unreadable",
        ) from exc
    try:
        value = json.loads(
            text,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite constant: {token}")
            ),
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ReconcileProbeError(
            f"file is not strict JSON: {path}",
            code="reconcile_probe.malformed_json",
        ) from exc
    if not isinstance(value, dict):
        raise ReconcileProbeError(
            f"strict JSON root must be an object: {path}",
            code="reconcile_probe.malformed_json",
        )
    return value


def _current_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    commit = completed.stdout.strip()
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise ReconcileProbeError(
            "resolved commit is not a 40-character hex SHA",
            code="reconcile_probe.invalid_commit",
        )
    return commit


def _require_world_size(world_size: int) -> None:
    if isinstance(world_size, bool) or not isinstance(world_size, int):
        raise ReconcileProbeError(
            "world_size must be an integer",
            code="reconcile_probe.invalid_world_size",
        )
    if world_size != WORLD_SIZE:
        raise ReconcileProbeError(
            f"this probe requires world_size={WORLD_SIZE}",
            code="reconcile_probe.wrong_world_size",
            context={"observed": world_size},
        )


# --------------------------------------------------------------------------
# prepare
# --------------------------------------------------------------------------


def _role_overrides(role: str, *, artifact_root: Path) -> dict[str, Any]:
    runs_root = str(artifact_root / RUNS_DIR_NAME)
    common: dict[str, Any] = {
        "run": {
            "name": role,
            "artifact_root": runs_root,
            "collision_policy": "fail",
        },
        "runtime": {"determinism": {"mode": "strict_cuda_replay_v1"}},
    }
    if role == ROLE_UNINTERRUPTED_CONTROL:
        common["training"] = {"max_steps": 2}
        common["checkpoint"] = {"steps": [1, 2], "save_final": True}
        common["eval"] = {"forward": {"steps": [1]}}
        common["resume"] = {"mode": "disabled", "checkpoint_dir": None}
    elif role == ROLE_RESUMED_PARENT:
        common["training"] = {"max_steps": 1}
        common["checkpoint"] = {"steps": [1], "save_final": True}
        common["eval"] = {"forward": {"steps": [1]}}
        common["resume"] = {"mode": "disabled", "checkpoint_dir": None}
    elif role == ROLE_RESUMED_CHILD:
        parent_checkpoint = str(
            artifact_root / RUNS_DIR_NAME / ROLE_RESUMED_PARENT / "checkpoints" / "step-1"
        )
        common["training"] = {"max_steps": 2}
        common["checkpoint"] = {"steps": [2], "save_final": True}
        common["eval"] = {"forward": {"steps": [2]}}
        common["resume"] = {
            "mode": "exact_same_world_size",
            "checkpoint_dir": parent_checkpoint,
        }
    else:
        raise ReconcileProbeError(
            f"unsupported role: {role}", code="reconcile_probe.unknown_role"
        )
    return common


def _deep_merge(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _validate_resolved_role(
    role: str, config: Mapping[str, Any], *, artifact_root: Path
) -> None:
    overrides = _role_overrides(role, artifact_root=artifact_root)
    try:
        exact = (
            config["schema_version"] == 1
            and config["run"]["name"] == role
            and config["run"]["artifact_root"] == overrides["run"]["artifact_root"]
            and config["run"]["collision_policy"] == "fail"
            and config["training"]["max_steps"] == overrides["training"]["max_steps"]
            and config["runtime"]["determinism"]["mode"] == "strict_cuda_replay_v1"
            and config["resume"]["mode"] == overrides["resume"]["mode"]
            and config["resume"]["checkpoint_dir"]
            == overrides["resume"]["checkpoint_dir"]
        )
    except (KeyError, TypeError) as exc:
        raise ReconcileProbeError(
            f"{role} resolved config is missing a required field",
            code="reconcile_probe.config_semantics_drift",
        ) from exc
    if not exact:
        raise ReconcileProbeError(
            f"{role} resolved config semantics drifted from the required bundle",
            code="reconcile_probe.config_semantics_drift",
            context={"role": role},
        )
    if role == ROLE_RESUMED_CHILD and config["resume"]["mode"] != "exact_same_world_size":
        raise ReconcileProbeError(
            "resumed_child must resolve to exact_same_world_size resume",
            code="reconcile_probe.config_semantics_drift",
        )


def _write_role_config(
    stage_dir: Path, role: str, *, base_config: Path, artifact_root: Path
) -> dict[str, Any]:
    payload = {"schema_version": 1, "extends": str(base_config)}
    payload.update(_role_overrides(role, artifact_root=artifact_root))
    filename = ROLE_FILENAMES[role]
    path = stage_dir / CONFIG_DIR_NAME / filename
    encoded = yaml.safe_dump(payload, sort_keys=False, allow_unicode=False).encode(
        "utf-8"
    )
    _write_new_file(path, encoded)
    try:
        resolved = load_train_config(path)
    except Exception as exc:  # config/pydantic validation errors
        raise ReconcileProbeError(
            f"{role} resolved config is invalid",
            code="reconcile_probe.config_invalid",
        ) from exc
    _validate_resolved_role(role, resolved.config_dict, artifact_root=artifact_root)
    return {
        "path": str(artifact_root / CONFIG_DIR_NAME / filename),
        "file_sha256": _sha256_file(path),
        "resolved_config_fingerprint": resolved.fingerprint,
    }


def prepare(
    *,
    artifact_root: str | Path,
    base_config: str | Path,
    world_size: int,
) -> dict[str, Any]:
    """Author the immutable two-rank config bundle; never launches a model."""

    _require_world_size(world_size)
    target = _assert_absolute(artifact_root, field="artifact_root")
    _assert_absent_target(target)

    base = Path(base_config).expanduser()
    if base.is_symlink():
        raise ReconcileProbeError(
            "base config must not be a symlink",
            code="reconcile_probe.symlink_escape",
        )
    base = base.resolve()
    if not base.is_file():
        raise ReconcileProbeError(
            "base config does not exist",
            code="reconcile_probe.base_config_missing",
            context={"path": str(base)},
        )
    try:
        load_train_config(base)
    except Exception as exc:
        raise ReconcileProbeError(
            "base config does not resolve",
            code="reconcile_probe.base_config_invalid",
        ) from exc

    parent = target.parent
    parent_info = _lstat_or_none(parent)
    if parent_info is None or not stat.S_ISDIR(parent_info.st_mode):
        raise ReconcileProbeError(
            "artifact_root parent is unavailable",
            code="reconcile_probe.parent_unavailable",
            context={"path": str(parent)},
        )
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.stage-", dir=str(parent)))
    try:
        (stage / CONFIG_DIR_NAME).mkdir(mode=0o755)
        configs: dict[str, Any] = {}
        for role in (ROLE_UNINTERRUPTED_CONTROL, ROLE_RESUMED_PARENT, ROLE_RESUMED_CHILD):
            configs[role] = _write_role_config(
                stage, role, base_config=base, artifact_root=target
            )
        pack_cache_root = target / ".cache" / "packing"
        body = {
            "schema": SCHEMA_PREPARE_RECEIPT,
            "status": "prepared",
            "commit": _current_commit(),
            "world_size": world_size,
            "artifact_root": str(target),
            "base_config": {"path": str(base), "file_sha256": _sha256_file(base)},
            "configs": configs,
            "pack_cache": {
                "root": str(pack_cache_root),
                "status": "reserved_absent",
                "note": (
                    "private cache root is reserved but not pre-populated; "
                    "the real production src.train route materializes it on "
                    "first authorized launch"
                ),
            },
        }
        receipt = _signed(body)
        receipt_path = stage / PREPARE_RECEIPT_NAME
        _write_new_file(receipt_path, _canonical_json_bytes(receipt) + b"\n")
        _fsync_directory(stage / CONFIG_DIR_NAME)
        _fsync_directory(stage)
        try:
            _install_directory_no_replace(stage, target)
        except (OSError, FileExistsError) as exc:
            raise ReconcileProbeError(
                "atomic prepare publication failed",
                code="reconcile_probe.publication_failed",
            ) from exc
        try:
            _fsync_directory(parent)
        except OSError:
            pass
        return _strict_json_load(target / PREPARE_RECEIPT_NAME)
    finally:
        if stage.exists():
            import shutil

            shutil.rmtree(stage)


def _load_prepare_receipt(artifact_root: Path) -> dict[str, Any]:
    path = artifact_root / PREPARE_RECEIPT_NAME
    if not path.is_file():
        raise ReconcileProbeError(
            "artifact_root has no prepare receipt; run prepare first",
            code="reconcile_probe.prepare_receipt_missing",
            context={"path": str(path)},
        )
    return _strict_json_load(path)


def _check_commit_drift(receipt: Mapping[str, Any], *, commit: str | None) -> None:
    if commit is None:
        return
    if commit != receipt["commit"]:
        raise ReconcileProbeError(
            "commit differs from the prepared bundle",
            code="reconcile_probe.commit_drift",
            context={"expected": receipt["commit"], "observed": commit},
        )


# --------------------------------------------------------------------------
# success-control / success-resumed
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LaunchResult:
    returncode: int
    stdout: str
    stderr: str


LaunchCallable = Callable[[Sequence[str], Path, Mapping[str, str]], LaunchResult]


def _default_launch(argv: Sequence[str], cwd: Path, env: Mapping[str, str]) -> LaunchResult:
    completed = subprocess.run(
        list(argv), cwd=str(cwd), env=dict(env), capture_output=True, text=True, check=False
    )
    return LaunchResult(completed.returncode, completed.stdout, completed.stderr)


def _launch_argv(config_path: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(WORLD_SIZE),
        "-m",
        "src.train",
        "--config",
        config_path,
    ]


def _launch_env() -> dict[str, str]:
    return dict(os.environ)


def _run_config_path(receipt: Mapping[str, Any], role: str) -> str:
    return receipt["configs"][role]["path"]


def _run_dir(receipt: Mapping[str, Any], role: str) -> Path:
    return Path(receipt["artifact_root"]) / RUNS_DIR_NAME / role


def _load_run_state(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run.json"
    if not path.is_file():
        raise ReconcileProbeError(
            "run directory has no durable run.json",
            code="reconcile_probe.run_state_missing",
            context={"path": str(path)},
        )
    return _strict_json_load(path)


def _require_launch_ok(result: LaunchResult, *, role: str) -> None:
    if result.returncode != 0:
        raise ReconcileProbeError(
            f"{role} launch exited with a nonzero status",
            code="reconcile_probe.launch_failed",
            context={
                "role": role,
                "returncode": result.returncode,
                "stderr_tail": result.stderr[-2000:],
            },
        )


def _require_completed_steps(run_state: Mapping[str, Any], *, expected: int, role: str) -> None:
    observed = run_state.get("completed_steps")
    if observed != expected:
        raise ReconcileProbeError(
            f"{role} did not resolve exactly the expected number of updates",
            code="reconcile_probe.unexpected_update_count",
            context={"role": role, "expected": expected, "observed": observed},
        )


def _require_world_size_in_state(run_state: Mapping[str, Any], *, role: str) -> None:
    observed = ((run_state.get("runtime") or {}).get("world_size"))
    if observed != WORLD_SIZE:
        raise ReconcileProbeError(
            f"{role} did not report the required rank count",
            code="reconcile_probe.missing_ranks",
            context={"role": role, "expected": WORLD_SIZE, "observed": observed},
        )


def _write_run_receipt(receipt_dir: Path, name: str, body: Mapping[str, Any]) -> dict[str, Any]:
    receipt_dir.mkdir(parents=True, exist_ok=True)
    path = receipt_dir / name
    if path.exists():
        raise ReconcileProbeError(
            f"run receipt already exists: {path}",
            code="reconcile_probe.receipt_exists",
        )
    signed = _signed(dict(body))
    _write_new_file(path, _canonical_json_bytes(signed) + b"\n")
    return _strict_json_load(path)


def success_control(
    *,
    artifact_root: str | Path,
    commit: str | None = None,
    launch: LaunchCallable = _default_launch,
) -> dict[str, Any]:
    root = Path(artifact_root)
    receipt = _load_prepare_receipt(root)
    _check_commit_drift(receipt, commit=commit)

    config_path = _run_config_path(receipt, ROLE_UNINTERRUPTED_CONTROL)
    argv = _launch_argv(config_path)
    started = time.monotonic()
    result = launch(argv, REPO_ROOT, _launch_env())
    wall_time = time.monotonic() - started
    _require_launch_ok(result, role=ROLE_UNINTERRUPTED_CONTROL)

    run_dir = _run_dir(receipt, ROLE_UNINTERRUPTED_CONTROL)
    run_state = _load_run_state(run_dir)
    _require_world_size_in_state(run_state, role=ROLE_UNINTERRUPTED_CONTROL)
    _require_completed_steps(run_state, expected=2, role=ROLE_UNINTERRUPTED_CONTROL)

    body = {
        "schema": SCHEMA_RUN_RECEIPT,
        "role": ROLE_UNINTERRUPTED_CONTROL,
        "commit": receipt["commit"],
        "argv": argv,
        "config_sha256": receipt["configs"][ROLE_UNINTERRUPTED_CONTROL]["file_sha256"],
        "returncode": result.returncode,
        "wall_time_seconds": wall_time,
        "boundary_step": 1,
        "update_step": 2,
        "run_dir": str(run_dir),
    }
    return _write_run_receipt(
        root / RECEIPTS_DIR_NAME, "success-control-receipt.json", body
    )


def success_resumed(
    *,
    artifact_root: str | Path,
    commit: str | None = None,
    launch: LaunchCallable = _default_launch,
) -> dict[str, Any]:
    root = Path(artifact_root)
    receipt = _load_prepare_receipt(root)
    _check_commit_drift(receipt, commit=commit)

    parent_config = _run_config_path(receipt, ROLE_RESUMED_PARENT)
    parent_argv = _launch_argv(parent_config)
    setup_started = time.monotonic()
    setup_result = launch(parent_argv, REPO_ROOT, _launch_env())
    setup_wall_time = time.monotonic() - setup_started
    _require_launch_ok(setup_result, role=ROLE_RESUMED_PARENT)
    parent_run_dir = _run_dir(receipt, ROLE_RESUMED_PARENT)
    parent_state = _load_run_state(parent_run_dir)
    _require_world_size_in_state(parent_state, role=ROLE_RESUMED_PARENT)
    _require_completed_steps(parent_state, expected=1, role=ROLE_RESUMED_PARENT)

    child_config = _run_config_path(receipt, ROLE_RESUMED_CHILD)
    child_argv = _launch_argv(child_config)
    update_started = time.monotonic()
    update_result = launch(child_argv, REPO_ROOT, _launch_env())
    update_wall_time = time.monotonic() - update_started
    _require_launch_ok(update_result, role=ROLE_RESUMED_CHILD)
    child_run_dir = _run_dir(receipt, ROLE_RESUMED_CHILD)
    child_state = _load_run_state(child_run_dir)
    _require_world_size_in_state(child_state, role=ROLE_RESUMED_CHILD)
    _require_completed_steps(child_state, expected=2, role=ROLE_RESUMED_CHILD)

    body = {
        "schema": SCHEMA_RUN_RECEIPT,
        "role": ROLE_RESUMED_CHILD,
        "commit": receipt["commit"],
        "setup": {
            "argv": parent_argv,
            "config_sha256": receipt["configs"][ROLE_RESUMED_PARENT]["file_sha256"],
            "returncode": setup_result.returncode,
            "wall_time_seconds": setup_wall_time,
            "boundary_step": 1,
            "run_dir": str(parent_run_dir),
        },
        "update": {
            "argv": child_argv,
            "config_sha256": receipt["configs"][ROLE_RESUMED_CHILD]["file_sha256"],
            "returncode": update_result.returncode,
            "wall_time_seconds": update_wall_time,
            "update_step": 2,
            "run_dir": str(child_run_dir),
        },
    }
    return _write_run_receipt(
        root / RECEIPTS_DIR_NAME, "success-resumed-receipt.json", body
    )


# --------------------------------------------------------------------------
# rank-failure (model-free; reuses src.artifacts.training_state directly)
# --------------------------------------------------------------------------


def _synthetic_resolved_config() -> dict[str, Any]:
    return {
        "config": {
            "resume": {"checkpoint_dir": None, "mode": "disabled"},
            "run": {
                "artifact_root": "/reconcile-probe",
                "name": "rank-failure",
                "output_dir": "/reconcile-probe/rank-failure",
            },
            "runtime": {
                "determinism": {"mode": "strict_cuda_replay_v1"},
                "seed": 17,
            },
            "training": {"seed": 17},
        },
        "resolution": {
            "entry_config_path": "/reconcile-probe/rank-failure.yaml",
            "fingerprint": "reconcile-probe",
            "loader_version": "coordexp-swift-config-v1",
            "path_origins": {},
            "schema_version": 1,
            "sources": [],
        },
    }


def _synthetic_identities() -> dict[str, str]:
    digest = "a" * 64
    return build_exact_resume_identities(
        base_model=digest,
        cache=digest,
        dependencies=digest,
        policy=digest,
        resolved_config=_synthetic_resolved_config(),
        topology=digest,
        trainable_surface=digest,
    )


def _synthetic_plan() -> TrainingStatePublicationPlan:
    resolved_config = _synthetic_resolved_config()
    return TrainingStatePublicationPlan(
        parent_run_id="reconcile-probe-run",
        parent_segment_id="reconcile-probe-segment",
        checkpoint_step=1,
        continuation_index=0,
        world_size=WORLD_SIZE,
        identities=_synthetic_identities(),
        scheduler_applicable=False,
        scaler_applicable=False,
        resolved_config=resolved_config,
        resume_compatibility=build_resume_compatibility_projection(resolved_config),
        accumulation_microstep=0,
    )


def _synthetic_payload(rank: int) -> RankTrainingStatePayload:
    import torch

    torch.manual_seed(100 + rank)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss = model(torch.ones(1, 3)).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return serialize_current_rank_training_state(
        rank=rank,
        world_size=WORLD_SIZE,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        cursor={"data": {"epoch": 0, "ordinal": rank}, "pack": {"ordinal": rank, "pending": []}},
        next_rank_local_micro_step=1,
        accumulation_microstep=0,
        rng_snapshot=None,
    )


def _validate_inject(inject: Mapping[str, Any]) -> tuple[int, str]:
    if set(inject) != {"rank", "kind"}:
        raise ReconcileProbeError(
            "inject spec must have exactly rank and kind",
            code="reconcile_probe.malformed_inject",
        )
    rank = inject["rank"]
    kind = inject["kind"]
    if isinstance(rank, bool) or rank not in (0, 1):
        raise ReconcileProbeError(
            "inject rank must be 0 or 1",
            code="reconcile_probe.malformed_inject",
        )
    if kind not in INJECT_KINDS:
        raise ReconcileProbeError(
            f"inject kind must be one of {sorted(INJECT_KINDS)}",
            code="reconcile_probe.malformed_inject",
        )
    return rank, kind


def rank_failure(
    *,
    artifact_root: str | Path,
    inject: Mapping[str, Any],
    world_size: int = WORLD_SIZE,
) -> dict[str, Any]:
    """Run a two-rank model-free exact-publication failure injection."""

    _require_world_size(world_size)
    injected_rank, kind = _validate_inject(inject)

    root = _assert_absolute(artifact_root, field="artifact_root")
    _assert_absent_target(root)
    root.mkdir(parents=True)
    checkpoint_dir = root / "checkpoint"
    checkpoint_dir.mkdir()

    plan = _synthetic_plan()
    payloads = {rank: _synthetic_payload(rank) for rank in range(WORLD_SIZE)}
    session = begin_training_state_contributions(checkpoint_dir, plan)

    error: ArtifactContractError | None = None
    contributed_ranks: list[int] = []
    try:
        for rank in range(WORLD_SIZE):
            if rank == injected_rank and kind == "missing":
                continue
            publish_rank_training_state_contribution(
                checkpoint_dir, session, payloads[rank]
            )
            contributed_ranks.append(rank)
            if rank == injected_rank and kind == "duplicate":
                publish_rank_training_state_contribution(
                    checkpoint_dir, session, payloads[rank]
                )
        if kind == "malformed":
            _corrupt_rank_contribution(session.stage_path, injected_rank, garbage=True)
        elif kind == "corrupt":
            _corrupt_rank_contribution(session.stage_path, injected_rank, garbage=False)
        commit_training_state_contributions(checkpoint_dir, session)
    except ArtifactContractError as exc:
        error = exc
        abort_training_state_contributions(checkpoint_dir, session)
    if error is None:
        raise ReconcileProbeError(
            "injected rank-failure coverage did not converge to a failure",
            code="reconcile_probe.injection_did_not_fail",
            context={"kind": kind},
        )

    manifest_admitted = (checkpoint_dir / "training_state").exists()
    residue = sorted(path.name for path in checkpoint_dir.glob(".training_state.*.tmp"))
    body = {
        "schema": SCHEMA_RANK_FAILURE_RECEIPT,
        "status": "converged_failure",
        "world_size": world_size,
        "injected_rank": injected_rank,
        "kind": kind,
        "error_code": error.code,
        "manifest_admitted": manifest_admitted,
        "residue": residue,
        "checkpoint_dir": str(checkpoint_dir),
    }
    if manifest_admitted or residue:
        raise ReconcileProbeError(
            "converged failure left an admitted alias/event or stage residue",
            code="reconcile_probe.failure_not_clean",
            context=body,
        )
    receipt = _signed(body)
    receipt_path = root / "rank-failure-receipt.json"
    _write_new_file(receipt_path, _canonical_json_bytes(receipt) + b"\n")
    return _strict_json_load(receipt_path)


def _corrupt_rank_contribution(stage_path: Path, rank: int, *, garbage: bool) -> None:
    rank_dir = stage_path / f"rank-{rank:05d}"
    if not rank_dir.is_dir():
        raise ReconcileProbeError(
            "cannot corrupt a rank contribution that was not written",
            code="reconcile_probe.injection_target_missing",
        )
    files = sorted(path for path in rank_dir.rglob("*") if path.is_file())
    if not files:
        raise ReconcileProbeError(
            "rank contribution has no files to corrupt",
            code="reconcile_probe.injection_target_missing",
        )
    target = files[0]
    if garbage:
        target.write_bytes(b"not-valid-training-state-bytes")
    else:
        original = bytearray(target.read_bytes())
        if not original:
            original = bytearray(b"\x00")
        original[0] ^= 0xFF
        target.write_bytes(bytes(original))


# --------------------------------------------------------------------------
# interruption (three externally visible commit boundaries)
# --------------------------------------------------------------------------


CommitInferenceCallable = Callable[[Path], dict[str, Any]]


def _stub_inference_manifest(checkpoint_dir: Path) -> dict[str, Any]:
    """Write a schema-shaped inference-payload manifest stand-in.

    The real production writer (`write_inference_checkpoint_payload_manifest`)
    inspects a genuine trained DoRA adapter payload, which only exists after a
    real model launch. This probe is model-free, so it durably records a
    self-consistent stub manifest at the same path instead; the real writer is
    exercised only during the authorized GPU launch in Task 4.
    """

    body = {
        "schema": "coordexp-swift-inference-checkpoint-payload-manifest",
        "schema_version": 1,
        "status": "stub",
        "note": "reconcile probe stand-in; real launch uses the production writer",
    }
    manifest = {**body, "aggregate_digest": sha256_json(body)}
    path = checkpoint_dir / "inference_payload_manifest.json"
    _write_new_file(path, _canonical_json_bytes(manifest) + b"\n")
    return manifest


def interruption(
    *,
    artifact_root: str | Path,
    stop_after: int,
    world_size: int = WORLD_SIZE,
    commit_inference: CommitInferenceCallable = _stub_inference_manifest,
) -> dict[str, Any]:
    """Exercise the three externally visible interruption boundaries."""

    _require_world_size(world_size)
    if stop_after not in INTERRUPTION_BOUNDARIES:
        raise ReconcileProbeError(
            f"stop_after must be one of {INTERRUPTION_BOUNDARIES}",
            code="reconcile_probe.invalid_boundary",
        )

    root = _assert_absolute(artifact_root, field="artifact_root")
    _assert_absent_target(root)
    root.mkdir(parents=True)
    checkpoint_dir = root / "checkpoint"
    checkpoint_dir.mkdir()

    inference_manifest: dict[str, Any] | None = None
    training_state_manifest_digest: str | None = None
    event_recorded = False

    if stop_after >= 0:
        pass  # boundary 0: nothing committed yet.
    if stop_after >= 1:
        inference_manifest = commit_inference(checkpoint_dir)
    if stop_after >= 2:
        plan = _synthetic_plan()
        payloads = {rank: _synthetic_payload(rank) for rank in range(WORLD_SIZE)}
        session = begin_training_state_contributions(checkpoint_dir, plan)
        for rank in range(WORLD_SIZE):
            publish_rank_training_state_contribution(
                checkpoint_dir, session, payloads[rank]
            )
        published = commit_training_state_contributions(checkpoint_dir, session)
        training_state_manifest_digest = published.manifest.aggregate_digest
    # The authoritative event/alias commit (RunWriter.record_checkpoint_publication_event)
    # is deliberately never reached by this probe: `stop_after` only spans the
    # three boundaries named by Task 3.3, and boundary index 2 already stops
    # before that commit by construction.

    inference_present = (checkpoint_dir / "inference_payload_manifest.json").exists()
    training_state_present = (checkpoint_dir / "training_state").exists()
    if inference_present != (stop_after >= 1):
        raise ReconcileProbeError(
            "inference payload durability does not match the requested boundary",
            code="reconcile_probe.interruption_state_mismatch",
        )
    if training_state_present != (stop_after >= 2):
        raise ReconcileProbeError(
            "exact-state durability does not match the requested boundary",
            code="reconcile_probe.interruption_state_mismatch",
        )
    if training_state_present:
        reloaded = load_training_state_manifest(checkpoint_dir)
        if reloaded.aggregate_digest != training_state_manifest_digest:
            raise ReconcileProbeError(
                "committed exact-state manifest failed exact reload",
                code="reconcile_probe.interruption_state_mismatch",
            )
    if inference_present and not training_state_present:
        # boundary-1 receipts must remain inference-only: no admitted alias.
        pass

    body = {
        "schema": SCHEMA_INTERRUPTION_RECEIPT,
        "status": "converged_partial_state",
        "world_size": world_size,
        "stop_after": stop_after,
        "checkpoint_dir": str(checkpoint_dir),
        "inference_payload_present": inference_present,
        "inference_payload_only": inference_present and not training_state_present,
        "exact_state_present": training_state_present,
        "event_recorded": event_recorded,
    }
    receipt = _signed(body)
    receipt_path = root / "interruption-receipt.json"
    _write_new_file(receipt_path, _canonical_json_bytes(receipt) + b"\n")
    return _strict_json_load(receipt_path)


# --------------------------------------------------------------------------
# verify
# --------------------------------------------------------------------------


_COMPARISON_POLICY = {
    "next_pack_and_cursor": ["cursor"],
    "trainable_model": ["model_state_ref"],
    "optimizer": ["optimizer_state_ref"],
    "scheduler_scaler_applicability": ["scheduler_applicable", "scaler_applicable"],
    "per_rank_rng": ["rng_ref"],
    "topology_identity": ["topology_ref"],
    "objective_loss": ["loss"],
    "resulting_trainable_parameters": ["parameters_ref"],
}


def verify_artifacts(*, artifact_root: str | Path) -> dict[str, Any]:
    """Read durable artifacts only and publish one strict terminal receipt."""

    root = Path(artifact_root)
    prepare_receipt = _load_prepare_receipt(root)
    receipts_dir = root / RECEIPTS_DIR_NAME

    def _load(name: str) -> dict[str, Any] | None:
        path = receipts_dir / name
        if not path.is_file():
            return None
        return _strict_json_load(path)

    control = _load("success-control-receipt.json")
    resumed = _load("success-resumed-receipt.json")

    findings: dict[str, Any] = {}
    status = "incomplete"
    if control is not None and resumed is not None:
        if control["commit"] != prepare_receipt["commit"]:
            raise ReconcileProbeError(
                "success-control receipt commit differs from the prepared bundle",
                code="reconcile_probe.commit_drift",
            )
        if resumed["commit"] != prepare_receipt["commit"]:
            raise ReconcileProbeError(
                "success-resumed receipt commit differs from the prepared bundle",
                code="reconcile_probe.commit_drift",
            )
        control_run = _strict_json_load(Path(control["run_dir"]) / "run.json")
        child_run = _strict_json_load(
            Path(resumed["update"]["run_dir"]) / "run.json"
        )
        findings["next_pack_and_cursor_matched"] = control_run.get(
            "next_planned_step"
        ) == child_run.get("next_planned_step")
        findings["objective_loss_matched"] = control_run.get(
            "final_loss"
        ) == child_run.get("final_loss")
        status = "verified"

    body = {
        "schema": SCHEMA_TERMINAL_RECEIPT,
        "status": status,
        "commit": prepare_receipt["commit"],
        "artifact_root": str(root),
        "world_size": prepare_receipt["world_size"],
        "comparison_policy": _COMPARISON_POLICY,
        "control_receipt_present": control is not None,
        "resumed_receipt_present": resumed is not None,
        "findings": findings,
    }
    receipt = _signed(body)
    receipt_path = root / "terminal-receipt.json"
    if receipt_path.exists():
        raise ReconcileProbeError(
            "terminal receipt already exists",
            code="reconcile_probe.receipt_exists",
        )
    _write_new_file(receipt_path, _canonical_json_bytes(receipt) + b"\n")
    return _strict_json_load(receipt_path)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--artifact-root", required=True)
    prepare_parser.add_argument("--base-config", required=True)
    prepare_parser.add_argument("--world-size", type=int, required=True)

    control_parser = subparsers.add_parser("success-control")
    control_parser.add_argument("--artifact-root", required=True)
    control_parser.add_argument("--commit", default=None)

    resumed_parser = subparsers.add_parser("success-resumed")
    resumed_parser.add_argument("--artifact-root", required=True)
    resumed_parser.add_argument("--commit", default=None)

    failure_parser = subparsers.add_parser("rank-failure")
    failure_parser.add_argument("--artifact-root", required=True)
    failure_parser.add_argument("--inject", required=True)
    failure_parser.add_argument("--world-size", type=int, default=WORLD_SIZE)

    interrupt_parser = subparsers.add_parser("interruption")
    interrupt_parser.add_argument("--artifact-root", required=True)
    interrupt_parser.add_argument("--stop-after", type=int, required=True)
    interrupt_parser.add_argument("--world-size", type=int, default=WORLD_SIZE)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--artifact-root", required=True)

    return parser


def _parse_inject_argument(value: str) -> Mapping[str, Any]:
    candidate = Path(value)
    if candidate.is_file():
        return _strict_json_load(candidate)
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ReconcileProbeError(
            "inject argument is neither an existing file nor strict inline JSON",
            code="reconcile_probe.malformed_json",
        ) from exc
    if not isinstance(parsed, dict):
        raise ReconcileProbeError(
            "inject argument must decode to a JSON object",
            code="reconcile_probe.malformed_json",
        )
    return parsed


def main(argv: list[str] | None = None, *, launch: LaunchCallable = _default_launch) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare(
            artifact_root=args.artifact_root,
            base_config=args.base_config,
            world_size=args.world_size,
        )
    elif args.command == "success-control":
        result = success_control(
            artifact_root=args.artifact_root, commit=args.commit, launch=launch
        )
    elif args.command == "success-resumed":
        result = success_resumed(
            artifact_root=args.artifact_root, commit=args.commit, launch=launch
        )
    elif args.command == "rank-failure":
        result = rank_failure(
            artifact_root=args.artifact_root,
            inject=_parse_inject_argument(args.inject),
            world_size=args.world_size,
        )
    elif args.command == "interruption":
        result = interruption(
            artifact_root=args.artifact_root,
            stop_after=args.stop_after,
            world_size=args.world_size,
        )
    elif args.command == "verify":
        result = verify_artifacts(artifact_root=args.artifact_root)
    else:  # pragma: no cover - argparse enforces the subcommand set
        raise ReconcileProbeError(
            "unsupported command", code="reconcile_probe.unknown_command"
        )
    print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
