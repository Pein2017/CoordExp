from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.coco_refinement.preflight import (
    LaunchPreflightError,
    RuntimeRootBusyError,
    RuntimeRootLock,
    SUPPORTED_DEPENDENCIES,
    UnsafeRuntimeRootLockError,
    resolve_dependency_versions,
    run_launch_preflight,
    validate_process_shape,
)


def _supported_version(distribution: str) -> str:
    return SUPPORTED_DEPENDENCIES[distribution]


def test_supported_preflight_holds_lock_and_writes_json_receipt(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"

    with run_launch_preflight(
        runtime_root,
        environment={},
        version_resolver=_supported_version,
    ) as accepted:
        assert accepted.writer_lock.acquired is True
        receipt = json.loads(accepted.receipt_path.read_text(encoding="utf-8"))
        assert receipt == accepted.receipt.as_dict()
        assert json.loads(accepted.receipt.to_json()) == receipt
        assert receipt["dependencies"] == dict(SUPPORTED_DEPENDENCIES)
        assert receipt["process_shape"] == {
            "reload": False,
            "workers": 1,
            "web_concurrency": 1,
            "pid": accepted.receipt.pid,
        }
        assert receipt["runtime_root"] == str(runtime_root.resolve())
        assert receipt["status"] == "accepted"

    assert accepted.writer_lock.acquired is False


def test_missing_dependency_is_rejected_before_runtime_root_creation(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"

    def missing(distribution: str) -> str:
        if distribution == "uvicorn":
            raise KeyError(distribution)
        return SUPPORTED_DEPENDENCIES[distribution]

    with pytest.raises(LaunchPreflightError, match="missing: uvicorn==0.37.0"):
        run_launch_preflight(
            runtime_root,
            environment={},
            version_resolver=missing,
        )

    assert not runtime_root.exists()


@pytest.mark.parametrize(
    ("distribution", "actual"),
    [
        ("fastapi", "0.136.2"),
        ("uvicorn", "0.38.0"),
        ("starlette", "0.52.0"),
    ],
)
def test_version_mismatch_is_rejected(distribution: str, actual: str) -> None:
    def resolve(name: str) -> str:
        return actual if name == distribution else SUPPORTED_DEPENDENCIES[name]

    with pytest.raises(
        LaunchPreflightError,
        match=rf"unsupported {distribution} version",
    ):
        resolve_dependency_versions(resolve)


@pytest.mark.parametrize(
    ("reload", "workers", "environment", "message"),
    [
        (True, 1, {}, "reload must be disabled"),
        (False, 0, {}, "workers must equal 1"),
        (False, 2, {}, "workers must equal 1"),
        (False, True, {}, "workers must equal 1"),
        (False, 1, {"WEB_CONCURRENCY": "2"}, "WEB_CONCURRENCY must equal 1"),
        (False, 1, {"WEB_CONCURRENCY": "invalid"}, "WEB_CONCURRENCY must equal 1"),
        (False, 1, {"WEB_CONCURRENCY": "01"}, "WEB_CONCURRENCY must equal 1"),
    ],
)
def test_unsupported_process_shape_is_rejected_before_lock(
    tmp_path: Path,
    reload: bool,
    workers: int,
    environment: dict[str, str],
    message: str,
) -> None:
    runtime_root = tmp_path / "runtime"

    with pytest.raises(LaunchPreflightError, match=message):
        run_launch_preflight(
            runtime_root,
            reload=reload,
            workers=workers,
            environment=environment,
            version_resolver=_supported_version,
        )

    assert not runtime_root.exists()


def test_unset_and_exact_web_concurrency_are_supported() -> None:
    assert validate_process_shape(reload=False, workers=1, environment={}) == 1
    assert (
        validate_process_shape(
            reload=False,
            workers=1,
            environment={"WEB_CONCURRENCY": "1"},
        )
        == 1
    )


def test_second_runtime_root_writer_is_rejected(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    first = RuntimeRootLock(runtime_root).acquire()
    try:
        with pytest.raises(RuntimeRootBusyError, match="already has a writer"):
            RuntimeRootLock(runtime_root).acquire()
    finally:
        first.release()


def test_writer_lock_rejects_symlink_without_mutating_target(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    sentinel = tmp_path / "outside-sentinel"
    original = b"must-not-change\n"
    sentinel.write_bytes(original)
    (runtime_root / ".writer.lock").symlink_to(sentinel)

    with pytest.raises(UnsafeRuntimeRootLockError, match="unsafe writer lock"):
        RuntimeRootLock(runtime_root).acquire()

    assert sentinel.read_bytes() == original


def test_writer_lock_rejects_preexisting_hardlink_without_mutation(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    sentinel = tmp_path / "outside-sentinel"
    original = b"must-not-change\n"
    sentinel.write_bytes(original)
    os.link(sentinel, runtime_root / ".writer.lock")

    with pytest.raises(UnsafeRuntimeRootLockError, match="unsafe writer lock"):
        RuntimeRootLock(runtime_root).acquire()

    assert sentinel.read_bytes() == original


def test_release_is_idempotent_and_allows_reacquire(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    first = RuntimeRootLock(runtime_root).acquire()
    lock_path = first.path

    first.release()
    first.release()
    assert lock_path.exists()

    with RuntimeRootLock(runtime_root) as second:
        assert second.acquired is True
        owner = json.loads(lock_path.read_text(encoding="utf-8"))
        assert owner["kind"] == "coco_refinement_runtime_writer"

    assert second.acquired is False


def test_released_preflight_cannot_be_reentered(tmp_path: Path) -> None:
    accepted = run_launch_preflight(
        tmp_path / "runtime",
        environment={},
        version_resolver=_supported_version,
    )
    accepted.release()

    with pytest.raises(LaunchPreflightError, match="already released"):
        accepted.__enter__()
