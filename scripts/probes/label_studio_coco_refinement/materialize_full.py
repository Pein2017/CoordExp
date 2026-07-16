#!/usr/bin/env python3
"""Reproducible full-split performance and immutability probe for materialization."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.label_studio_coco_refinement.materialize import (  # noqa: E402
    OPERATOR_RECEIPT_NAME,
)
from src.label_studio_coco_refinement.models import (  # noqa: E402
    RefinementRuntimeLayout,
)

PROBE_RECEIPT_NAME = "probe.json"
PROBE_SCHEMA_VERSION = "label-studio-materialize-full-probe-v2"
OPERATOR_SCRIPT = REPO_ROOT / "scripts/materialize_label_studio_coco_refinement.py"
CODE_PATHS = (
    Path("scripts/materialize_label_studio_coco_refinement.py"),
    Path("scripts/probes/label_studio_coco_refinement/materialize_full.py"),
    Path("src/label_studio_coco_refinement/categories.py"),
    Path("src/label_studio_coco_refinement/materialize.py"),
    Path("src/label_studio_coco_refinement/models.py"),
    Path("src/label_studio_coco_refinement/project.py"),
    Path("src/label_studio_coco_refinement/store.py"),
    Path("src/data/__init__.py"),
    Path("src/data/examples.py"),
    Path("src/data/jsonl.py"),
)
_FICLONE = 0x40049409
_SENSITIVE_ENV_MARKERS = (
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "CREDENTIAL",
    "COOKIE",
    "API_KEY",
    "ACCESS_KEY",
    "PRIVATE_KEY",
)


class ProbeError(RuntimeError):
    """The probe could not establish a reproducible, immutable run."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _git_path_is_ignored(repo_root: Path, candidate: Path) -> bool:
    relative = candidate.relative_to(repo_root)
    completed = subprocess.run(
        ["git", "-C", str(repo_root), "check-ignore", "-q", "--", str(relative)],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return completed.returncode == 0


def _validate_probe_root(repo_root: Path, probe_root: Path) -> Path:
    candidate = probe_root if probe_root.is_absolute() else repo_root / probe_root
    candidate = candidate.expanduser().resolve(strict=False)
    try:
        candidate.relative_to(repo_root)
    except ValueError as exc:
        raise ProbeError("probe root must be inside the selected repository") from exc
    if os.path.lexists(candidate):
        raise ProbeError(f"probe root already exists: {candidate}")
    if not _git_path_is_ignored(repo_root, candidate):
        raise ProbeError(f"probe root is not ignored by git: {candidate}")
    return candidate


def _file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ProbeError(f"expected an ordinary file: {path}")
    return {
        "path": str(path),
        "resolved": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


@contextmanager
def _shared_snapshot_locks(split_root: Path) -> Iterator[float]:
    """Match supported-reader ordering without retaining live store objects."""

    commit_lock = split_root / ".commit.lock"
    queue_lock = split_root / ".queue.lock"
    for path in (commit_lock, queue_lock):
        if not path.is_file() or path.is_symlink():
            raise ProbeError(f"snapshot lock is missing or not ordinary: {path}")
    started = time.perf_counter()
    with commit_lock.open("rb") as commit_handle:
        fcntl.flock(commit_handle.fileno(), fcntl.LOCK_SH)
        try:
            with queue_lock.open("rb") as queue_handle:
                fcntl.flock(queue_handle.fileno(), fcntl.LOCK_SH)
                try:
                    yield started
                finally:
                    fcntl.flock(queue_handle.fileno(), fcntl.LOCK_UN)
        finally:
            fcntl.flock(commit_handle.fileno(), fcntl.LOCK_UN)


def _clone_or_copy_file(
    source: str,
    destination: str,
    *,
    statistics: dict[str, Any] | None = None,
    working_source: Path | None = None,
) -> str:
    """Use Linux CoW when available, falling back to a byte copy."""

    source_path = Path(source)
    destination_path = Path(destination)
    source_fd = os.open(source_path, os.O_RDONLY)
    destination_fd: int | None = None
    try:
        destination_fd = os.open(
            destination_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        try:
            fcntl.ioctl(destination_fd, _FICLONE, source_fd)
        except OSError:
            os.close(destination_fd)
            destination_fd = None
            destination_path.unlink(missing_ok=True)
            shutil.copy2(source_path, destination_path)
            if statistics is not None:
                statistics["copy_file_count"] += 1
                statistics["copy_bytes"] += source_path.stat().st_size
                if working_source is not None and source_path == working_source:
                    statistics["working_mode"] = "copy"
            return str(destination_path)
    finally:
        os.close(source_fd)
        if destination_fd is not None:
            os.close(destination_fd)
    shutil.copystat(source_path, destination_path)
    if statistics is not None:
        statistics["cow_clone_file_count"] += 1
        statistics["cow_clone_bytes"] += source_path.stat().st_size
        if working_source is not None and source_path == working_source:
            statistics["working_mode"] = "cow_clone"
    return str(destination_path)


def _rebind_copied_manifest(
    layout: RefinementRuntimeLayout,
    split: str,
    *,
    final_managed_image_link: Path | None = None,
) -> dict[str, Any]:
    manifest_path = layout.project_manifest(split)  # type: ignore[arg-type]
    try:
        before = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProbeError(f"cannot read copied manifest: {manifest_path}") from exc
    if not isinstance(before, dict):
        raise ProbeError("copied project manifest must be a JSON object")
    after = copy.deepcopy(before)
    original = after.get("managed_image_link")
    rebound_link = (
        final_managed_image_link
        if final_managed_image_link is not None
        else layout.images_link(split)  # type: ignore[arg-type]
    )
    after["managed_image_link"] = str(rebound_link)
    changed = sorted(key for key in set(before) | set(after) if before.get(key) != after.get(key))
    if changed != ["managed_image_link"]:
        raise ProbeError(f"manifest rebind changed unexpected fields: {changed}")
    _atomic_write_json(manifest_path, after)
    return {
        "changed_fields": changed,
        "original_managed_image_link": original,
        "rebound_managed_image_link": after["managed_image_link"],
        "sha256": sha256_file(manifest_path),
    }


def _snapshot_live_split(
    *, repo_root: Path, probe_root: Path, split: str
) -> tuple[RefinementRuntimeLayout, dict[str, Any]]:
    live_layout = RefinementRuntimeLayout.under_repository(repo_root)
    live_split = live_layout.split_root(split)  # type: ignore[arg-type]
    if not live_split.is_dir() or live_split.is_symlink():
        raise ProbeError(f"live split root is missing or not ordinary: {live_split}")

    staging = probe_root / f".repository.snapshot-{uuid.uuid4().hex}"
    snapshot_layout = RefinementRuntimeLayout.under_repository(staging)
    snapshot_split = snapshot_layout.split_root(split)  # type: ignore[arg-type]
    snapshot_split.parent.mkdir(parents=True, exist_ok=True)
    snapshot_layout.repository_root.mkdir(parents=True, exist_ok=True)
    snapshot_layout.repository_root.joinpath("public_data").symlink_to(
        repo_root / "public_data", target_is_directory=True
    )
    working_source = live_layout.working_norm(split)  # type: ignore[arg-type]
    copy_statistics: dict[str, Any] = {
        "cow_clone_file_count": 0,
        "cow_clone_bytes": 0,
        "copy_file_count": 0,
        "copy_bytes": 0,
        "working_mode": None,
    }

    def clone_or_copy(source: str, destination: str) -> str:
        return _clone_or_copy_file(
            source,
            destination,
            statistics=copy_statistics,
            working_source=working_source,
        )

    started = time.perf_counter()
    with _shared_snapshot_locks(live_split) as lock_started:
        shutil.copytree(
            live_split,
            snapshot_split,
            symlinks=True,
            copy_function=clone_or_copy,
        )
    lock_seconds = time.perf_counter() - lock_started
    snapshot_seconds = time.perf_counter() - started

    final_repository = probe_root / "repository"
    final_layout = RefinementRuntimeLayout.under_repository(final_repository)
    manifest_rebind = _rebind_copied_manifest(
        snapshot_layout,
        split,
        final_managed_image_link=final_layout.images_link(split),  # type: ignore[arg-type]
    )
    os.replace(staging, final_repository)
    _fsync_directory(probe_root)
    live_files = [path for path in live_split.rglob("*") if path.is_file()]
    snapshot_files = [
        path for path in final_layout.split_root(split).rglob("*") if path.is_file()  # type: ignore[arg-type]
    ]
    return final_layout, {
        "repository": str(final_repository),
        "split_root": str(final_layout.split_root(split)),  # type: ignore[arg-type]
        "public_data": {
            "path": str(final_repository / "public_data"),
            "resolved": str((final_repository / "public_data").resolve(strict=True)),
            "copied": False,
        },
        "locks": {
            "mode": "shared",
            "order": [".commit.lock", ".queue.lock"],
            "held_only_during_snapshot": True,
            "hold_seconds": lock_seconds,
        },
        "snapshot_seconds": snapshot_seconds,
        "live_regular_file_count": len(live_files),
        "snapshot_regular_file_count": len(snapshot_files),
        "copy_strategy": copy_statistics,
        "manifest_rebind": manifest_rebind,
    }


def _cleanup_snapshot_artifacts(probe_root: Path) -> None:
    """Remove only this probe's copied repository and unpublished staging trees."""

    candidates = [probe_root / "repository", *probe_root.glob(".repository.snapshot-*")]
    removed = False
    for candidate in candidates:
        if not os.path.lexists(candidate):
            continue
        if candidate.is_symlink() or not candidate.is_dir():
            candidate.unlink()
        else:
            shutil.rmtree(candidate)
        removed = True
    if removed:
        _fsync_directory(probe_root)


def _sanitized_child_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not any(marker in key.upper() for marker in _SENSITIVE_ENV_MARKERS)
    }
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONHASHSEED"] = "0"
    return environment


def _peak_rss_bytes(max_rss: int) -> int:
    return max_rss if sys.platform == "darwin" else max_rss * 1024


def _run_materializer_child(
    *, fake_repo_root: Path, split: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    command = [
        sys.executable,
        str(OPERATOR_SCRIPT),
        "--repo-root",
        str(fake_repo_root),
        "--split",
        split,
    ]
    started = time.perf_counter()
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=_sanitized_child_environment(),
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
        )
        try:
            _, status, usage = os.wait4(process.pid, 0)
        except BaseException:
            process.kill()
            os.waitpid(process.pid, 0)
            raise
        wall_seconds = time.perf_counter() - started
        returncode = os.waitstatus_to_exitcode(status)
        stdout.seek(0)
        stdout_bytes = stdout.read()
        stderr.seek(0)
        stderr_bytes = stderr.read()
    if returncode != 0:
        raise ProbeError(
            "materializer child failed "
            f"(returncode={returncode}, stderr_bytes={len(stderr_bytes)}, "
            f"stderr_sha256={hashlib.sha256(stderr_bytes).hexdigest()})"
        )
    try:
        receipt = json.loads(stdout_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProbeError(
            "materializer child did not emit one JSON receipt "
            f"(stdout_bytes={len(stdout_bytes)}, "
            f"stdout_sha256={hashlib.sha256(stdout_bytes).hexdigest()})"
        ) from exc
    if not isinstance(receipt, dict):
        raise ProbeError("materializer child receipt must be a JSON object")
    return receipt, {
        "wall_seconds": wall_seconds,
        "peak_rss_bytes": _peak_rss_bytes(usage.ru_maxrss),
        "returncode": returncode,
    }


def _code_identity() -> dict[str, Any]:
    sources: dict[str, Any] = {}
    for relative in CODE_PATHS:
        path = REPO_ROOT / relative
        sources[str(relative)] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }
    return {
        "python": {
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "platform": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "source_files": sources,
    }


def _validate_operator_receipt(
    receipt: Mapping[str, Any], destination: Path
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    materialization = receipt.get("materialization")
    loader = receipt.get("loader_attestation")
    if not isinstance(materialization, Mapping) or not isinstance(loader, Mapping):
        raise ProbeError("operator receipt lacks materialization/loader attestation")
    destination_hash = sha256_file(destination)
    if materialization.get("destination_sha256") != destination_hash:
        raise ProbeError("destination hash differs from operator receipt")
    if loader.get("status") != "passed" or loader.get("row_count") != materialization.get(
        "row_count"
    ):
        raise ProbeError("operator loader attestation does not match row count")
    return materialization, loader


def run_probe(
    *,
    repo_root: Path,
    probe_root: Path,
    split: str,
    retain_snapshot: bool = False,
) -> dict[str, Any]:
    repo_root = repo_root.expanduser().resolve(strict=True)
    if split not in {"train", "val"}:
        raise ProbeError(f"unsupported split: {split}")
    if not OPERATOR_SCRIPT.is_file():
        raise ProbeError(f"operator script is missing: {OPERATOR_SCRIPT}")
    probe_root = _validate_probe_root(repo_root, probe_root)
    live_layout = RefinementRuntimeLayout.under_repository(repo_root)
    source = live_layout.selected_source(split)  # type: ignore[arg-type]
    live_working = live_layout.working_norm(split)  # type: ignore[arg-type]
    before = {
        "selected_source": _file_identity(source),
        "live_working": _file_identity(live_working),
    }
    code_before = _code_identity()
    probe_root.mkdir(parents=True)
    completed = False
    try:
        snapshot_layout, snapshot = _snapshot_live_split(
            repo_root=repo_root, probe_root=probe_root, split=split
        )
        cloned_working = snapshot_layout.working_norm(split)  # type: ignore[arg-type]
        cloned_hash_before = sha256_file(cloned_working)
        child_error: BaseException | None = None
        child_receipt: dict[str, Any] | None = None
        performance: dict[str, Any] | None = None
        try:
            child_receipt, performance = _run_materializer_child(
                fake_repo_root=snapshot_layout.repository_root, split=split
            )
        except BaseException as exc:
            child_error = exc

        after = {
            "selected_source": _file_identity(source),
            "live_working": _file_identity(live_working),
        }
        code_after = _code_identity()
        cloned_hash_after = sha256_file(cloned_working)
        unchanged = {
            name: before[name]["sha256"] == after[name]["sha256"]
            and before[name]["size_bytes"] == after[name]["size_bytes"]
            for name in before
        }
        unchanged["cloned_working"] = cloned_hash_before == cloned_hash_after
        unchanged["source_code"] = (
            code_before["source_files"] == code_after["source_files"]
        )
        if not all(unchanged.values()):
            raise ProbeError(f"probe changed immutable/live bytes: {unchanged}")
        if child_error is not None:
            raise child_error
        assert child_receipt is not None and performance is not None

        destination = snapshot_layout.working_coord(split)  # type: ignore[arg-type]
        durable_receipt = (
            snapshot_layout.split_root(split) / OPERATOR_RECEIPT_NAME  # type: ignore[arg-type]
        )
        if not destination.is_file() or not durable_receipt.is_file():
            raise ProbeError("materializer child did not publish its fixed atomic outputs")
        persisted_child = json.loads(durable_receipt.read_text(encoding="utf-8"))
        if persisted_child != child_receipt:
            raise ProbeError("printed and durable operator receipts differ")
        materialization, loader = _validate_operator_receipt(
            child_receipt, destination
        )

        snapshot["retained"] = retain_snapshot
        artifact = {
            "schema_version": PROBE_SCHEMA_VERSION,
            "code": "label_studio.materialize_full_probe_passed",
            "split": split,
            "probe_root": str(probe_root),
            "snapshot": snapshot,
            "performance": performance,
            "code_identity": code_before,
            "immutability": {
                "selected_source": {
                    "before": before["selected_source"],
                    "after": after["selected_source"],
                    "unchanged": unchanged["selected_source"],
                },
                "live_working": {
                    "before": before["live_working"],
                    "after": after["live_working"],
                    "unchanged": unchanged["live_working"],
                },
                "cloned_working": {
                    "path": str(cloned_working),
                    "sha256": cloned_hash_before,
                    "unchanged": unchanged["cloned_working"],
                    "retained": retain_snapshot,
                },
                "source_code_unchanged": unchanged["source_code"],
                "all_unchanged": all(unchanged.values()),
            },
            "result": {
                "row_count": materialization["row_count"],
                "object_count": materialization["object_count"],
                "destination": {
                    "path": str(destination),
                    "size_bytes": destination.stat().st_size,
                    "sha256": materialization["destination_sha256"],
                    "retained": retain_snapshot,
                },
                "loader_attestation": dict(loader),
                "operator_receipt": child_receipt,
                "operator_receipt_path": str(durable_receipt),
                "operator_receipt_path_retained": retain_snapshot,
            },
            "credential_material_recorded": False,
        }
        if not retain_snapshot:
            _cleanup_snapshot_artifacts(probe_root)
            if os.path.lexists(snapshot_layout.repository_root):  # pragma: no cover
                raise ProbeError("default snapshot cleanup did not remove repository")
        receipt_path = probe_root / PROBE_RECEIPT_NAME
        _atomic_write_json(receipt_path, artifact)
        completed = True
        print(str(receipt_path))
        return artifact
    finally:
        if not completed:
            _cleanup_snapshot_artifacts(probe_root)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Snapshot one live refinement split and run the real current-coord "
            "materializer with full performance/immutability attestation."
        )
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument(
        "--probe-root",
        type=Path,
        required=True,
        help="Caller-selected fresh Git-ignored path inside the repository.",
    )
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument(
        "--retain-snapshot",
        action="store_true",
        help=(
            "Retain the successful copied repository and materialized outputs. "
            "Failed runs are always cleaned."
        ),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    run_probe(
        repo_root=args.repo_root,
        probe_root=args.probe_root,
        split=args.split,
        retain_snapshot=args.retain_snapshot,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
