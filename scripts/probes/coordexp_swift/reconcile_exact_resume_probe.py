"""Bounded two-rank exact-resume qualification probe for `world_size=2`.

Task 4A preparatory tooling for OpenSpec change
`reconcile-coordexp-swift-training-contracts`. The existing Wave-7 controller
(`wave7_exact_resume_sequence.py`) is fixed to eight ranks and cannot express
the current `world_size=2` packet, so this module is a small, independent CLI
that reuses production config/artifact/admission utilities instead of copying
that controller.

Command grammar (strict, no defaults):
    prepare          -- author the immutable two-rank config bundle and its
                        private, model-free pack cache.
    success-control  -- run the boundary-then-next-update control branch.
    success-resumed  -- run the matched parent-boundary + resumed-child branch.
    rank-failure     -- model-free two-rank exact-publication failure injection.
    interruption     -- exercise the exact-state/event-adjacent interruption
                        boundaries for real; the inference-payload boundary
                        uses a non-production stub (see `interruption`'s
                        docstring).
    verify           -- read durable artifacts only, admit and compare both
                        checkpoint boundaries and the step-2 objective, and
                        publish one fail-closed terminal receipt.

Every subcommand is also reachable as a plain Python function (`prepare`,
`success_control`, `success_resumed`, `rank_failure`, `interruption`,
`verify_artifacts`) so tests can inject a fake `launch` callable and never
touch a GPU or a production model.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import random
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.artifacts.training_state import (  # noqa: E402
    AdmittedTrainingState,
    DecodedRankTrainingState,
    RankTrainingStatePayload,
    REQUIRED_RNG_KINDS,
    TrainingStateExpectations,
    TrainingStateManifest,
    TrainingStatePublicationPlan,
    abort_training_state_contributions,
    admit_training_state,
    begin_training_state_contributions,
    build_resume_compatibility_projection,
    commit_training_state_contributions,
    load_training_state_manifest,
    publish_rank_training_state_contribution,
    serialize_rank_training_state,
)
from src.common.errors import ArtifactContractError, CoordExpError  # noqa: E402
from src.config.fingerprint import sha256_json  # noqa: E402
from src.config.loader import load_train_config  # noqa: E402
from src.runtime.seeding import _STRICT_ENVIRONMENT  # noqa: E402
from src.training.exact_resume import build_exact_resume_identities  # noqa: E402


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
ARMS_DIR_NAME = "arms"
RANK_FAILURE_ARM_NAME = "rank_failure"
INTERRUPTION_ARM_NAME = "interruption"
RANK_FAILURE_RECEIPT_NAME = "rank-failure-receipt.json"
INTERRUPTION_RECEIPT_NAME = "interruption-receipt.json"

REPRESENTATIVE_FAILURE_RANK = 1
REPRESENTATIVE_FAILURE_KIND = "missing"
REPRESENTATIVE_INTERRUPTION_STOP_AFTER = 1

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


def _load_signed_receipt(path: Path) -> dict[str, Any]:
    """Load one of this probe's own `_signed` receipts and verify its digest."""

    payload = _strict_json_load(path)
    recorded = payload.get("receipt_payload_sha256")
    if not isinstance(recorded, str):
        raise ReconcileProbeError(
            f"receipt is not signed: {path}",
            code="reconcile_probe.receipt_unsigned",
        )
    body = {key: value for key, value in payload.items() if key != "receipt_payload_sha256"}
    if _sha256_bytes(_canonical_json_bytes(body)) != recorded:
        raise ReconcileProbeError(
            f"receipt payload digest mismatch: {path}",
            code="reconcile_probe.receipt_digest_mismatch",
            context={"path": str(path)},
        )
    return payload


def _assert_absolute_existing_root(path_value: str | Path, *, field: str) -> Path:
    path = _assert_absolute(path_value, field=field)
    _assert_no_symlink_components(path, include_leaf=True)
    info = _lstat_or_none(path)
    if info is None or not stat.S_ISDIR(info.st_mode):
        raise ReconcileProbeError(
            f"{field} does not exist",
            code="reconcile_probe.target_missing",
            context={"field": field, "path": str(path)},
        )
    return path


def _assert_no_path_drift(observed: Path, *, expected: str, field: str) -> None:
    if str(observed) != expected:
        raise ReconcileProbeError(
            f"{field} differs from the prepared bundle",
            code="reconcile_probe.path_drift",
            context={"expected": expected, "observed": str(observed)},
        )


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
        # No eval forward in any role: this probe's numeric forward ceiling
        # (control=2/rank, resumed_parent=2/rank, resumed_child=1/rank) counts
        # only train-split forwards, and `steps: []` with the default unset
        # `every_fraction` is the schema-valid "never run eval forward" state.
        "eval": {"forward": {"steps": []}},
    }
    if role == ROLE_UNINTERRUPTED_CONTROL:
        common["training"] = {"max_steps": 2}
        common["checkpoint"] = {"steps": [1, 2], "save_final": True}
        # Publish-only control branch: exact mode with a null path is the
        # contract Task 2.5 established for "no restore, but still publish
        # training_state/" -- `resume.mode: disabled` would make the real
        # pipeline skip exact-state publication entirely and leave nothing
        # for `verify` to admit at the step-1/step-2 boundaries.
        common["resume"] = {"mode": "exact_same_world_size", "checkpoint_dir": None}
    elif role == ROLE_RESUMED_PARENT:
        common["training"] = {"max_steps": 2}
        common["checkpoint"] = {"steps": [1, 2], "save_final": True}
        common["resume"] = {"mode": "exact_same_world_size", "checkpoint_dir": None}
    elif role == ROLE_RESUMED_CHILD:
        parent_checkpoint = str(
            artifact_root / RUNS_DIR_NAME / ROLE_RESUMED_PARENT / "checkpoints" / "step-1"
        )
        common["training"] = {"max_steps": 2}
        common["checkpoint"] = {"steps": [1, 2], "save_final": True}
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
            and list(config["checkpoint"]["steps"])
            == list(overrides["checkpoint"]["steps"])
            and config["checkpoint"]["save_final"]
            is overrides["checkpoint"]["save_final"]
            and config["runtime"]["determinism"]["mode"] == "strict_cuda_replay_v1"
            and config["resume"]["mode"] == overrides["resume"]["mode"]
            and config["resume"]["checkpoint_dir"]
            == overrides["resume"]["checkpoint_dir"]
            and list(config["eval"]["forward"]["steps"]) == []
            and config["eval"]["forward"]["every_fraction"] is None
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
        "config_dict": resolved.config_dict,
    }


def _determinant_relevant_projection(config_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Project exactly the fields `build_packing_cache_determinants` reads.

    `_role_overrides` only ever touches `run`, `runtime.determinism.mode`,
    `training.max_steps`, `checkpoint`, `eval`, and `resume`; none of those
    are packing-cache determinants, so every role must project identically.
    """

    return {
        "data": config_dict["data"],
        "template": config_dict["template"],
        "packing": config_dict["packing"],
        "model": config_dict["model"],
        "training_precision": config_dict["training"]["precision"],
        "runtime_seed": config_dict["runtime"]["seed"],
    }


def _assert_shared_pack_cache_determinants(
    control_config_dict: Mapping[str, Any],
    other_config_dict: Mapping[str, Any],
    *,
    role: str,
) -> None:
    if _determinant_relevant_projection(
        control_config_dict
    ) != _determinant_relevant_projection(other_config_dict):
        raise ReconcileProbeError(
            f"{role} does not share the control branch's packing-cache determinants",
            code="reconcile_probe.pack_cache_determinant_drift",
            context={"role": role},
        )


PrepareCacheCallable = Callable[[Path, Path, Path], dict[str, Any]]


def _strict_determinism_env() -> dict[str, str]:
    """The exact env `src.runtime.seeding` requires before `strict_cuda_replay_v1`."""

    return dict(_STRICT_ENVIRONMENT)


def _default_prepare_pack_cache(
    config_path: Path, cache_root: Path, receipt_path: Path
) -> dict[str, Any]:
    """Run the real single-process, model-free cache preparation entrypoint."""

    env = {
        **os.environ,
        **_strict_determinism_env(),
        "COORDEXP_SWIFT_PACK_CACHE_ROOT": str(cache_root),
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.prepare_train_cache",
            "--config",
            str(config_path),
            "--receipt",
            str(receipt_path),
        ],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ReconcileProbeError(
            "pack cache preparation exited with a nonzero status",
            code="reconcile_probe.pack_cache_prepare_failed",
            context={"returncode": result.returncode, "stderr_tail": result.stderr[-2000:]},
        )
    if not receipt_path.is_file():
        raise ReconcileProbeError(
            "pack cache preparation did not publish its durable receipt",
            code="reconcile_probe.pack_cache_prepare_failed",
        )
    payload = _strict_json_load(receipt_path)
    recorded = payload.get("receipt_sha256")
    body = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    encoded = json.dumps(
        body, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")
    if not isinstance(recorded, str) or hashlib.sha256(encoded).hexdigest() != recorded:
        raise ReconcileProbeError(
            "pack cache preparation receipt failed digest verification",
            code="reconcile_probe.receipt_digest_mismatch",
        )
    if payload.get("terminal_status") != "completed":
        raise ReconcileProbeError(
            "pack cache preparation did not reach a completed terminal status",
            code="reconcile_probe.pack_cache_prepare_failed",
            context={"terminal_status": payload.get("terminal_status")},
        )
    return payload


def prepare(
    *,
    artifact_root: str | Path,
    base_config: str | Path,
    world_size: int,
    prepare_pack_cache: PrepareCacheCallable = _default_prepare_pack_cache,
) -> dict[str, Any]:
    """Author the immutable two-rank config bundle and its private pack cache.

    Never launches a model or GPU: cache preparation runs the same
    single-process, `load_model=False` route as `src.prepare_train_cache`.
    """

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

    # The private cache root and its receipt live beside `target`, not inside
    # it: cache preparation runs (and must fully succeed) before the config
    # bundle is atomically installed, so a failed prepare() leaves neither a
    # half-installed bundle nor an orphaned cache root behind.
    pack_cache_root = parent / f".{target.name}.pack-cache"
    pack_cache_receipt_path = parent / f".{target.name}.pack-cache-receipt.json"
    _assert_absent_target(pack_cache_root)
    _assert_absent_target(pack_cache_receipt_path)

    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.stage-", dir=str(parent)))
    try:
        (stage / CONFIG_DIR_NAME).mkdir(mode=0o755)
        entries: dict[str, Any] = {}
        for role in (ROLE_UNINTERRUPTED_CONTROL, ROLE_RESUMED_PARENT, ROLE_RESUMED_CHILD):
            entries[role] = _write_role_config(
                stage, role, base_config=base, artifact_root=target
            )
            if role != ROLE_UNINTERRUPTED_CONTROL:
                _assert_shared_pack_cache_determinants(
                    entries[ROLE_UNINTERRUPTED_CONTROL]["config_dict"],
                    entries[role]["config_dict"],
                    role=role,
                )

        control_config_path = (
            stage / CONFIG_DIR_NAME / ROLE_FILENAMES[ROLE_UNINTERRUPTED_CONTROL]
        )
        cache_receipt = prepare_pack_cache(
            control_config_path, pack_cache_root, pack_cache_receipt_path
        )

        configs = {
            role: {
                key: value for key, value in entry.items() if key != "config_dict"
            }
            for role, entry in entries.items()
        }
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
                "status": "prepared",
                "receipt_path": str(pack_cache_receipt_path),
                "receipt_sha256": _sha256_file(pack_cache_receipt_path),
                "prepared_for_role": ROLE_UNINTERRUPTED_CONTROL,
                "shared_with_roles": [ROLE_RESUMED_PARENT, ROLE_RESUMED_CHILD],
                "resolved_config_fingerprint": cache_receipt.get("result", {}).get(
                    "resolved_config_fingerprint"
                )
                if isinstance(cache_receipt.get("result"), Mapping)
                else None,
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
    except BaseException:
        import shutil

        if pack_cache_root.exists():
            shutil.rmtree(pack_cache_root, ignore_errors=True)
        if pack_cache_receipt_path.exists():
            pack_cache_receipt_path.unlink(missing_ok=True)
        raise
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
    return _load_signed_receipt(path)


def _require_commit_binding(
    receipt: Mapping[str, Any], *, root: Path, commit: str
) -> None:
    """Require argument == prepare receipt == current HEAD, immediately before launch."""

    current = _current_commit()
    if not (commit == receipt["commit"] == current):
        raise ReconcileProbeError(
            "commit argument, prepare receipt, and current HEAD must all match",
            code="reconcile_probe.commit_drift",
            context={
                "argument": commit,
                "prepare_receipt": receipt["commit"],
                "head": current,
            },
        )
    _assert_no_path_drift(root, expected=receipt["artifact_root"], field="artifact_root")


def _verify_role_config_not_drifted(receipt: Mapping[str, Any], role: str) -> None:
    entry = receipt["configs"][role]
    config_path = Path(entry["path"])
    if not config_path.is_file():
        raise ReconcileProbeError(
            f"{role} config is missing from the prepared bundle",
            code="reconcile_probe.config_drift",
            context={"path": str(config_path)},
        )
    if _sha256_file(config_path) != entry["file_sha256"]:
        raise ReconcileProbeError(
            f"{role} config file drifted from the prepared bundle",
            code="reconcile_probe.config_drift",
            context={"role": role, "path": str(config_path)},
        )
    try:
        resolved = load_train_config(config_path)
    except Exception as exc:
        raise ReconcileProbeError(
            f"{role} config no longer resolves",
            code="reconcile_probe.config_drift",
        ) from exc
    if resolved.fingerprint != entry["resolved_config_fingerprint"]:
        raise ReconcileProbeError(
            f"{role} resolved config fingerprint drifted from the prepared bundle",
            code="reconcile_probe.config_drift",
            context={"role": role},
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


def _launch_env(receipt: Mapping[str, Any]) -> dict[str, str]:
    return {
        **os.environ,
        **_strict_determinism_env(),
        "COORDEXP_SWIFT_PACK_CACHE_ROOT": receipt["pack_cache"]["root"],
    }


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
    return _load_signed_receipt(path)


def success_control(
    *,
    artifact_root: str | Path,
    commit: str,
    launch: LaunchCallable = _default_launch,
) -> dict[str, Any]:
    root = _assert_absolute_existing_root(artifact_root, field="artifact_root")
    receipt = _load_prepare_receipt(root)
    _require_commit_binding(receipt, root=root, commit=commit)
    _verify_role_config_not_drifted(receipt, ROLE_UNINTERRUPTED_CONTROL)

    config_path = _run_config_path(receipt, ROLE_UNINTERRUPTED_CONTROL)
    argv = _launch_argv(config_path)
    started = time.monotonic()
    result = launch(argv, REPO_ROOT, _launch_env(receipt))
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
    commit: str,
    launch: LaunchCallable = _default_launch,
) -> dict[str, Any]:
    root = _assert_absolute_existing_root(artifact_root, field="artifact_root")
    receipt = _load_prepare_receipt(root)
    _require_commit_binding(receipt, root=root, commit=commit)
    _verify_role_config_not_drifted(receipt, ROLE_RESUMED_PARENT)
    _verify_role_config_not_drifted(receipt, ROLE_RESUMED_CHILD)

    parent_config = _run_config_path(receipt, ROLE_RESUMED_PARENT)
    parent_argv = _launch_argv(parent_config)
    setup_started = time.monotonic()
    setup_result = launch(parent_argv, REPO_ROOT, _launch_env(receipt))
    setup_wall_time = time.monotonic() - setup_started
    _require_launch_ok(setup_result, role=ROLE_RESUMED_PARENT)
    parent_run_dir = _run_dir(receipt, ROLE_RESUMED_PARENT)
    parent_state = _load_run_state(parent_run_dir)
    _require_world_size_in_state(parent_state, role=ROLE_RESUMED_PARENT)
    _require_completed_steps(parent_state, expected=2, role=ROLE_RESUMED_PARENT)

    child_config = _run_config_path(receipt, ROLE_RESUMED_CHILD)
    child_argv = _launch_argv(child_config)
    update_started = time.monotonic()
    update_result = launch(child_argv, REPO_ROOT, _launch_env(receipt))
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
    """Build a payload with explicit CPU-only RNG state -- never calls CUDA.

    `serialize_current_rank_training_state(rng_snapshot=None)` falls back to
    `capture_rank_rng_snapshot`, which calls `torch.cuda.current_device()` and
    therefore requires a real GPU. This probe is required to be GPU-free, so
    it calls the lower-level `serialize_rank_training_state` directly with an
    explicit Python/NumPy/torch-CPU state plus a synthetic `torch_cuda`
    tensor, exactly as `tests/training/test_exact_resume.py` already does for
    its own CPU-only fixtures.
    """

    torch.manual_seed(100 + rank)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss = model(torch.ones(1, 3)).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    python_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()
    torch_cpu_rng_state = torch.get_rng_state().clone()
    synthetic_cuda_rng_states = (torch.arange(32, dtype=torch.uint8) + rank,)
    return serialize_rank_training_state(
        rank=rank,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        python_rng_state=python_rng_state,
        numpy_rng_state=numpy_rng_state,
        torch_cpu_rng_state=torch_cpu_rng_state,
        torch_cuda_rng_states=synthetic_cuda_rng_states,
        cursor={"data": {"epoch": 0, "ordinal": rank}, "pack": {"ordinal": rank, "pending": []}},
        next_rank_local_micro_step=1,
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
    return _load_signed_receipt(receipt_path)


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

    This is NOT the real production owner and does not close the "before
    inference commit" boundary: the real writer
    (`write_inference_checkpoint_payload_manifest`) inspects a genuine
    trained DoRA adapter payload via `inspect_dora_adapter_payload`, which
    only exists after a real model launch produces real adapter weights.
    This probe is required to stay model-free and GPU-free, so it cannot
    fabricate one; the real writer is exercised only during the authorized
    GPU launch. This stub exists solely so `interruption`'s exact-state and
    event-adjacent boundaries (which ARE exercised for real, below) have a
    durable predecessor artifact to stop after/before.
    """

    body = {
        "schema": "coordexp-swift-inference-checkpoint-payload-manifest",
        "schema_version": 1,
        "status": "stub_not_production",
        "note": (
            "reconcile probe stand-in, not write_inference_checkpoint_payload_manifest; "
            "the real GPU launch exercises the production inference-payload writer"
        ),
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
    """Exercise the exact-state/event-adjacent interruption boundaries.

    Boundary 0->1 (before/after the inference-payload commit) uses an
    injectable, non-production stub -- see `_stub_inference_manifest`. Only
    boundaries 1->2 (exact-state manifest staging/commit) use the real
    production admission primitives from `src.artifacts.training_state`.
    """

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
        "inference_commit_owner": "stub_not_production",
        "exact_state_present": training_state_present,
        "exact_state_commit_owner": "production_training_state_primitives",
        "event_recorded": event_recorded,
    }
    receipt = _signed(body)
    receipt_path = root / "interruption-receipt.json"
    _write_new_file(receipt_path, _canonical_json_bytes(receipt) + b"\n")
    return _load_signed_receipt(receipt_path)


# --------------------------------------------------------------------------
# verify
# --------------------------------------------------------------------------


_EXCLUDED_LOGGING_KEYS = frozenset(
    {"input_build_seconds", "input_wait_seconds", "per_rank_measurement"}
)
_EXCLUDED_LOGGING_KEY_MARKERS = ("duration", "resource", "wall", "timing")


_COMPARISON_POLICY = {
    "boundary_step1_control_vs_parent": [
        "identities",
        "cursor",
        "trainable_model",
        "optimizer",
        "scheduler",
        "scaler",
        "rng.python",
        "rng.numpy",
        "rng.torch_cpu",
        "rng.torch_cuda",
        "cuda_device_topology",
        "structure_signature",
    ],
    "post_update_step2_control_vs_child": [
        "identities",
        "cursor",
        "trainable_model",
        "optimizer",
        "scheduler",
        "scaler",
        "rng.python",
        "rng.numpy",
        "rng.torch_cpu",
        "rng.torch_cuda",
        "cuda_device_topology",
        "structure_signature",
    ],
    "post_update_objective_step2_control_vs_child": {
        "source": "logging.jsonl",
        "selector": {"split": "train", "step": 2},
        "excluded_key_markers": list(_EXCLUDED_LOGGING_KEY_MARKERS),
        "excluded_keys": sorted(_EXCLUDED_LOGGING_KEYS),
    },
    "rank_failure_arm": {
        "source": f"{ARMS_DIR_NAME}/{RANK_FAILURE_ARM_NAME}/{RANK_FAILURE_RECEIPT_NAME}",
        "expected": {
            "schema": SCHEMA_RANK_FAILURE_RECEIPT,
            "status": "converged_failure",
            "world_size": WORLD_SIZE,
            "injected_rank": REPRESENTATIVE_FAILURE_RANK,
            "kind": REPRESENTATIVE_FAILURE_KIND,
            "manifest_admitted": False,
            "residue": [],
        },
        "forbidden_claim_key_markers": ["selector", "event"],
    },
    "interruption_arm": {
        "source": f"{ARMS_DIR_NAME}/{INTERRUPTION_ARM_NAME}/{INTERRUPTION_RECEIPT_NAME}",
        "expected": {
            "schema": SCHEMA_INTERRUPTION_RECEIPT,
            "status": "converged_partial_state",
            "world_size": WORLD_SIZE,
            "stop_after": REPRESENTATIVE_INTERRUPTION_STOP_AFTER,
            "inference_payload_present": True,
            "inference_payload_only": True,
            "inference_commit_owner": "stub_not_production",
            "exact_state_present": False,
            "event_recorded": False,
        },
    },
}

_REQUIRED_COMPARISONS = tuple(_COMPARISON_POLICY)
_CROSS_ROLE_IDENTITY_EXCLUSIONS = frozenset({"resolved_config"})


def _exact_json_field_equal(observed: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return observed is expected
    if isinstance(expected, int):
        return type(observed) is int and observed == expected
    if isinstance(expected, str):
        return isinstance(observed, str) and observed == expected
    if isinstance(expected, list):
        return isinstance(observed, list) and observed == expected
    return observed == expected


def _bounded_observed(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= 160 else value[:157] + "..."
    if isinstance(value, (list, dict)):
        return {"type": type(value).__name__, "size": len(value)}
    return {"type": type(value).__name__}


def _compare_arm_receipt_fields(
    receipt: Mapping[str, Any],
    *,
    path_prefix: str,
    expected: Mapping[str, Any],
) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    for key, expected_value in expected.items():
        if key not in receipt:
            mismatches.append(
                {"path": f"{path_prefix}.{key}", "code": "missing_field"}
            )
            continue
        observed = receipt[key]
        if not _exact_json_field_equal(observed, expected_value):
            mismatches.append(
                {
                    "path": f"{path_prefix}.{key}",
                    "expected": expected_value,
                    "observed": _bounded_observed(observed),
                }
            )
    return mismatches


def _compare_rank_failure_arm(
    receipt: Mapping[str, Any], *, checkpoint_dir: Path
) -> list[dict[str, Any]]:
    mismatches = _compare_arm_receipt_fields(
        receipt,
        path_prefix="rank_failure_arm",
        expected={
            **_COMPARISON_POLICY["rank_failure_arm"]["expected"],
            "checkpoint_dir": str(checkpoint_dir),
        },
    )
    error_code = receipt.get("error_code")
    if not isinstance(error_code, str) or not error_code.startswith("training_state."):
        mismatches.append(
            {
                "path": "rank_failure_arm.error_code",
                "expected": "training_state.*",
                "observed": _bounded_observed(error_code),
            }
        )
    for key in receipt:
        lowered = key.lower()
        if any(marker in lowered for marker in ("selector", "event")):
            mismatches.append(
                {
                    "path": f"rank_failure_arm.{key}",
                    "code": "unsupported_admission_claim",
                }
            )
    return mismatches


def _compare_interruption_arm(
    receipt: Mapping[str, Any], *, checkpoint_dir: Path
) -> list[dict[str, Any]]:
    return _compare_arm_receipt_fields(
        receipt,
        path_prefix="interruption_arm",
        expected={
            **_COMPARISON_POLICY["interruption_arm"]["expected"],
            "checkpoint_dir": str(checkpoint_dir),
        },
    )


def _manifest_expectations(manifest: TrainingStateManifest) -> TrainingStateExpectations:
    return TrainingStateExpectations(
        checkpoint_step=manifest.checkpoint_step,
        world_size=manifest.world_size,
        identities=manifest.identities,
        scheduler_applicable=manifest.scheduler_applicable,
        scaler_applicable=manifest.scaler_applicable,
        rng_kinds=REQUIRED_RNG_KINDS,
    )


def _admit_rank_state(
    checkpoint_dir: Path, manifest: TrainingStateManifest, rank: int
) -> DecodedRankTrainingState:
    admitted = admit_training_state(
        checkpoint_dir, _manifest_expectations(manifest), current_rank=rank
    )
    if not isinstance(admitted, AdmittedTrainingState):
        raise ReconcileProbeError(
            "training-state admission returned an unexpected result",
            code="reconcile_probe.admission_failed",
        )
    return admitted.decoded_rank


def _exact_state_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape)
            and torch.equal(left, right)
        )
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (
            isinstance(left, np.ndarray)
            and isinstance(right, np.ndarray)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and bool(np.array_equal(left, right))
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_exact_state_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, Sequence) and not isinstance(left, (str, bytes, bytearray)):
        return (
            isinstance(right, Sequence)
            and not isinstance(right, (str, bytes, bytearray))
            and len(left) == len(right)
            and all(
                _exact_state_equal(a, b) for a, b in zip(left, right, strict=True)
            )
        )
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return left is right


def _compare_decoded_rank(
    left: DecodedRankTrainingState,
    right: DecodedRankTrainingState,
    *,
    path_prefix: str,
) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    if left.signature != right.signature or dict(left.structure) != dict(
        right.structure
    ):
        mismatches.append(
            {
                "path": f"{path_prefix}.structure_signature",
                "expected": left.signature,
                "observed": right.signature,
            }
        )
    if dict(left.cursor) != dict(right.cursor):
        mismatches.append({"path": f"{path_prefix}.cursor"})
    if left.cuda_device_topology != right.cuda_device_topology:
        mismatches.append({"path": f"{path_prefix}.cuda_device_topology"})
    rng_pairs = {
        "python": (left.python_rng_state, right.python_rng_state),
        "numpy": (left.numpy_rng_state, right.numpy_rng_state),
        "torch_cpu": (left.torch_cpu_rng_state, right.torch_cpu_rng_state),
        "torch_cuda": (left.torch_cuda_rng_states, right.torch_cuda_rng_states),
    }
    for name, (rng_left, rng_right) in rng_pairs.items():
        if not _exact_state_equal(rng_left, rng_right):
            mismatches.append({"path": f"{path_prefix}.rng.{name}"})
    for owner in ("trainable_model", "optimizer", "scheduler", "scaler"):
        if not _exact_state_equal(getattr(left, owner), getattr(right, owner)):
            mismatches.append({"path": f"{path_prefix}.{owner}"})
    return mismatches


def _compare_checkpoint_pair(
    left_dir: Path, right_dir: Path, *, step: int, path_prefix: str
) -> list[dict[str, Any]]:
    left_manifest = load_training_state_manifest(left_dir)
    right_manifest = load_training_state_manifest(right_dir)
    for manifest, checkpoint_dir in ((left_manifest, left_dir), (right_manifest, right_dir)):
        if (
            manifest.checkpoint_step != step
            or manifest.world_size != WORLD_SIZE
            or tuple(rank.rank for rank in manifest.ranks) != tuple(range(WORLD_SIZE))
        ):
            raise ReconcileProbeError(
                "checkpoint manifest does not match the required boundary/rank set",
                code="reconcile_probe.manifest_identity_mismatch",
                context={
                    "path": str(checkpoint_dir),
                    "checkpoint_step": manifest.checkpoint_step,
                    "world_size": manifest.world_size,
                },
            )
    mismatches: list[dict[str, Any]] = []
    for key in sorted(set(left_manifest.identities) | set(right_manifest.identities)):
        if key in _CROSS_ROLE_IDENTITY_EXCLUSIONS:
            continue
        if left_manifest.identities.get(key) != right_manifest.identities.get(key):
            mismatches.append({"path": f"{path_prefix}.identities.{key}"})
    for rank in range(WORLD_SIZE):
        left_state = _admit_rank_state(left_dir, left_manifest, rank)
        right_state = _admit_rank_state(right_dir, right_manifest, rank)
        mismatches.extend(
            _compare_decoded_rank(left_state, right_state, path_prefix=f"{path_prefix}.rank{rank}")
        )
    return mismatches


def _is_excluded_logging_key(key: str) -> bool:
    lowered = key.lower()
    return key in _EXCLUDED_LOGGING_KEYS or any(
        marker in lowered for marker in _EXCLUDED_LOGGING_KEY_MARKERS
    )


def _read_single_train_row(run_dir: Path, *, step: int) -> dict[str, Any]:
    path = run_dir / "logging.jsonl"
    if not path.is_file():
        raise ReconcileProbeError(
            "run has no durable logging.jsonl",
            code="reconcile_probe.logging_missing",
            context={"path": str(path)},
        )
    matches: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReconcileProbeError(
                "logging.jsonl row is not strict JSON",
                code="reconcile_probe.malformed_json",
                context={"path": str(path)},
            ) from exc
        if not isinstance(row, dict):
            raise ReconcileProbeError(
                "logging.jsonl row is not a JSON object",
                code="reconcile_probe.malformed_json",
                context={"path": str(path)},
            )
        if row.get("split") == "train" and row.get("step") == step:
            matches.append(row)
    if len(matches) != 1:
        raise ReconcileProbeError(
            "run does not have exactly one durable train row for the required step",
            code="reconcile_probe.logging_row_count",
            context={"run_dir": str(run_dir), "step": step, "observed": len(matches)},
        )
    return matches[0]


def _compare_loss_rows(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    for key in sorted(set(left) | set(right)):
        if _is_excluded_logging_key(key):
            continue
        if left.get(key) != right.get(key):
            mismatches.append(
                {
                    "path": f"logging.{key}",
                    "expected": left.get(key),
                    "observed": right.get(key),
                }
            )
    return mismatches


def _write_terminal_receipt(
    root: Path,
    *,
    status: str,
    commit: str,
    world_size: int,
    missing_inputs: Sequence[str],
    mismatches: Sequence[Mapping[str, Any]],
    input_file_sha256: Mapping[str, str],
) -> dict[str, Any]:
    if status not in ("verified", "failed"):
        raise ReconcileProbeError(
            "terminal receipt status must be verified or failed",
            code="reconcile_probe.invalid_status",
        )
    body = {
        "schema": SCHEMA_TERMINAL_RECEIPT,
        "status": status,
        "commit": commit,
        "artifact_root": str(root),
        "world_size": world_size,
        "required_comparisons": list(_REQUIRED_COMPARISONS),
        "comparison_policy": _COMPARISON_POLICY,
        "missing_inputs": list(missing_inputs),
        "bounded_mismatches": list(mismatches),
        "input_file_sha256": dict(sorted(input_file_sha256.items())),
    }
    receipt = _signed(body)
    receipt_path = root / "terminal-receipt.json"
    if receipt_path.exists():
        raise ReconcileProbeError(
            "terminal receipt already exists",
            code="reconcile_probe.receipt_exists",
        )
    _write_new_file(receipt_path, _canonical_json_bytes(receipt) + b"\n")
    return _load_signed_receipt(receipt_path)


def verify_artifacts(*, artifact_root: str | Path) -> dict[str, Any]:
    """Read durable artifacts only, admit+compare both boundaries, fail closed.

    Never returns `status: "verified"` unless every required durable input
    exists and every required comparison actually ran and passed; any
    missing input or mismatch yields `status: "failed"`.
    """

    root = _assert_absolute_existing_root(artifact_root, field="artifact_root")
    prepare_receipt = _load_prepare_receipt(root)
    _assert_no_path_drift(
        root, expected=prepare_receipt["artifact_root"], field="artifact_root"
    )
    current = _current_commit()
    if current != prepare_receipt["commit"]:
        raise ReconcileProbeError(
            "current commit differs from the prepared bundle",
            code="reconcile_probe.commit_drift",
            context={"prepare_receipt": prepare_receipt["commit"], "head": current},
        )

    world_size = prepare_receipt["world_size"]
    receipts_dir = root / RECEIPTS_DIR_NAME
    control_path = receipts_dir / "success-control-receipt.json"
    resumed_path = receipts_dir / "success-resumed-receipt.json"
    rank_failure_root = root / ARMS_DIR_NAME / RANK_FAILURE_ARM_NAME
    interruption_root = root / ARMS_DIR_NAME / INTERRUPTION_ARM_NAME
    rank_failure_path = rank_failure_root / RANK_FAILURE_RECEIPT_NAME
    interruption_path = interruption_root / INTERRUPTION_RECEIPT_NAME
    missing_inputs: list[str] = []
    input_hashes: dict[str, str] = {
        str(root / PREPARE_RECEIPT_NAME): _sha256_file(root / PREPARE_RECEIPT_NAME)
    }
    for path in (control_path, resumed_path, rank_failure_path, interruption_path):
        if not path.is_file():
            missing_inputs.append(str(path))
    if missing_inputs:
        return _write_terminal_receipt(
            root,
            status="failed",
            commit=current,
            world_size=world_size,
            missing_inputs=missing_inputs,
            mismatches=[],
            input_file_sha256=input_hashes,
        )

    control = _load_signed_receipt(control_path)
    resumed = _load_signed_receipt(resumed_path)
    input_hashes[str(control_path)] = _sha256_file(control_path)
    input_hashes[str(resumed_path)] = _sha256_file(resumed_path)
    for label, receipt in (("success-control", control), ("success-resumed", resumed)):
        if receipt["commit"] != prepare_receipt["commit"]:
            raise ReconcileProbeError(
                f"{label} receipt commit differs from the prepared bundle",
                code="reconcile_probe.commit_drift",
            )

    mismatches: list[dict[str, Any]] = []
    for path, path_prefix, compare in (
        (rank_failure_path, "rank_failure_arm", _compare_rank_failure_arm),
        (interruption_path, "interruption_arm", _compare_interruption_arm),
    ):
        input_hashes[str(path)] = _sha256_file(path)
        try:
            arm_receipt = _load_signed_receipt(path)
        except ReconcileProbeError as exc:
            mismatches.append({"path": f"{path_prefix}.receipt", "code": exc.code})
        else:
            mismatches.extend(
                compare(arm_receipt, checkpoint_dir=path.parent / "checkpoint")
            )

    control_run_dir = Path(control["run_dir"])
    parent_run_dir = Path(resumed["setup"]["run_dir"])
    child_run_dir = Path(resumed["update"]["run_dir"])
    checkpoints = {
        "control_step1": control_run_dir / "checkpoints" / "step-1",
        "parent_step1": parent_run_dir / "checkpoints" / "step-1",
        "control_step2": control_run_dir / "checkpoints" / "step-2",
        "child_step2": child_run_dir / "checkpoints" / "step-2",
    }
    manifest_paths = {
        name: path / "training_state" / "manifest.json"
        for name, path in checkpoints.items()
    }
    logging_paths = {
        "control_logging": control_run_dir / "logging.jsonl",
        "child_logging": child_run_dir / "logging.jsonl",
    }
    for path in (*manifest_paths.values(), *logging_paths.values()):
        if not path.is_file():
            missing_inputs.append(str(path))
    if missing_inputs:
        return _write_terminal_receipt(
            root,
            status="failed",
            commit=current,
            world_size=world_size,
            missing_inputs=missing_inputs,
            mismatches=mismatches,
            input_file_sha256=input_hashes,
        )
    for name, path in manifest_paths.items():
        input_hashes[f"{name}_manifest"] = _sha256_file(path)
    for name, path in logging_paths.items():
        input_hashes[name] = _sha256_file(path)

    mismatches.extend(
        _compare_checkpoint_pair(
            checkpoints["control_step1"],
            checkpoints["parent_step1"],
            step=1,
            path_prefix="boundary",
        )
    )
    mismatches.extend(
        _compare_checkpoint_pair(
            checkpoints["control_step2"],
            checkpoints["child_step2"],
            step=2,
            path_prefix="post_update",
        )
    )
    try:
        control_row = _read_single_train_row(control_run_dir, step=2)
        child_row = _read_single_train_row(child_run_dir, step=2)
    except ReconcileProbeError as exc:
        mismatches.append({"path": "post_update_objective", "code": exc.code})
    else:
        mismatches.extend(_compare_loss_rows(control_row, child_row))

    status = "verified" if not mismatches else "failed"
    return _write_terminal_receipt(
        root,
        status=status,
        commit=current,
        world_size=world_size,
        missing_inputs=missing_inputs,
        mismatches=mismatches,
        input_file_sha256=input_hashes,
    )


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
    control_parser.add_argument("--commit", required=True)

    resumed_parser = subparsers.add_parser("success-resumed")
    resumed_parser.add_argument("--artifact-root", required=True)
    resumed_parser.add_argument("--commit", required=True)

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
    if args.command == "verify" and result.get("status") != "verified":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
