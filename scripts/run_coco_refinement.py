#!/usr/bin/env python3
"""Run the standalone loopback-only COCO refinement workspace."""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import uvicorn  # noqa: E402

from src.coco_refinement.bootstrap import DEFAULT_RUNTIME_RELATIVE  # noqa: E402
from src.coco_refinement.http_security import (  # noqa: E402
    HttpSecurityError,
    validate_loopback_authority,
)
from src.coco_refinement.runtime import create_standalone_runtime  # noqa: E402
from src.coco_refinement.service import create_runtime_service_app  # noqa: E402
from src.label_studio_coco_refinement.roi_runtime import (  # noqa: E402
    InferenceReceiptStore,
)


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 19172
DEFAULT_SHUTDOWN_TIMEOUT = 5
RESERVED_LEGACY_PORT = 8080
RECEIPT_STORE_NAME = "roi-receipts.jsonl"


def _existing_directory(value: str) -> Path:
    try:
        path = Path(value).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise argparse.ArgumentTypeError(
            f"path does not resolve to an existing directory: {value}"
        ) from exc
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"path is not a directory: {value}")
    return path


def _numeric_loopback(value: str) -> str:
    try:
        return validate_loopback_authority(value, DEFAULT_PORT).host
    except HttpSecurityError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _tcp_port(value: str) -> int:
    try:
        port = int(value, 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "port must be an integer from 1 through 65535"
        ) from exc
    try:
        validated = validate_loopback_authority(DEFAULT_HOST, port).port
    except HttpSecurityError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if validated == RESERVED_LEGACY_PORT:
        raise argparse.ArgumentTypeError("port 8080 is reserved for the legacy workspace")
    return validated


def _positive_timeout(value: str) -> int:
    try:
        timeout = int(value, 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "shutdown timeout must be a positive integer number of seconds"
        ) from exc
    if timeout <= 0 or not math.isfinite(timeout):
        raise argparse.ArgumentTypeError(
            "shutdown timeout must be a positive integer number of seconds"
        )
    return timeout


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the build-free COCO refinement editor on one numeric loopback "
            "address with reload disabled and exactly one Uvicorn worker."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=_existing_directory,
        default=REPO_ROOT,
        help=f"CoordExp repository root (default: {REPO_ROOT}).",
    )
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=None,
        help=(
            "Mutable standalone runtime root. Relative paths are resolved below "
            f"--repo-root (default: {DEFAULT_RUNTIME_RELATIVE})."
        ),
    )
    parser.add_argument(
        "--host",
        type=_numeric_loopback,
        default=DEFAULT_HOST,
        help=f"Numeric loopback bind address (default: {DEFAULT_HOST}).",
    )
    parser.add_argument(
        "--port",
        type=_tcp_port,
        default=DEFAULT_PORT,
        help=f"TCP port from 1 through 65535 (default: {DEFAULT_PORT}).",
    )
    parser.add_argument(
        "--shutdown-timeout",
        type=_positive_timeout,
        default=DEFAULT_SHUTDOWN_TIMEOUT,
        help=(
            "Positive seconds allowed for Uvicorn and worker shutdown "
            f"(default: {DEFAULT_SHUTDOWN_TIMEOUT})."
        ),
    )
    return parser


def _resolve_runtime_root(repo_root: Path, runtime_root: Path | None) -> Path:
    selected = DEFAULT_RUNTIME_RELATIVE if runtime_root is None else runtime_root
    if not selected.is_absolute():
        selected = repo_root / selected
    resolved = selected.expanduser().resolve()
    approved_parent = (repo_root / "outputs" / "coco_refinement").resolve()
    if resolved == approved_parent or approved_parent not in resolved.parents:
        raise ValueError(
            "runtime root must be a child of <repo-root>/outputs/coco_refinement"
        )
    return resolved


def run_server(
    *,
    repo_root: Path,
    runtime_root: Path,
    host: str,
    port: int,
    shutdown_timeout: int,
    receipt_store_factory: Callable[[Path], object] = InferenceReceiptStore,
    runtime_factory: Callable[..., Any] = create_standalone_runtime,
    app_factory: Callable[..., Any] = create_runtime_service_app,
    config_factory: Callable[..., Any] = uvicorn.Config,
    server_factory: Callable[[Any], Any] = uvicorn.Server,
) -> None:
    """Assemble, serve, and always release one standalone runtime."""

    # Keep argument validation before any runtime-root mutation. The runtime
    # factory then performs dependency/process preflight and takes the sole
    # writer lock before Uvicorn receives a bind-capable server configuration.
    authority = validate_loopback_authority(host, port)
    if authority.port == RESERVED_LEGACY_PORT:
        raise ValueError("port 8080 is reserved for the legacy workspace")
    if shutdown_timeout <= 0:
        raise ValueError("shutdown_timeout must be positive")

    selected_runtime = _resolve_runtime_root(repo_root.resolve(strict=True), runtime_root)
    runtime = runtime_factory(
        repo_root,
        runtime_root=selected_runtime,
        inference_receipt_store_factory=lambda: receipt_store_factory(
            selected_runtime / RECEIPT_STORE_NAME
        ),
        reload=False,
        workers=1,
    )
    try:
        runtime.start()
        app = app_factory(
            runtime,
            bind_host=authority.host,
            port=authority.port,
        )
        config = config_factory(
            app,
            host=authority.host,
            port=authority.port,
            reload=False,
            workers=1,
            access_log=False,
            timeout_graceful_shutdown=shutdown_timeout,
        )
        server_factory(config).run()
    finally:
        runtime.shutdown(timeout=shutdown_timeout)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve(strict=True)
    try:
        runtime_root = _resolve_runtime_root(repo_root, args.runtime_root)
    except ValueError as exc:
        parser.error(str(exc))
    run_server(
        repo_root=repo_root,
        runtime_root=runtime_root,
        host=args.host,
        port=args.port,
        shutdown_timeout=args.shutdown_timeout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
