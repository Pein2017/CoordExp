"""Minimal Wave 2 inference entry delegation."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from src.config.inference import load_infer_config, resolve_infer_run_directory
from src.config.writer import write_resolved_config_artifacts
from src.inference.runtime import assemble_runtime


def run(*, config_path: str | Path) -> int:
    resolved = load_infer_config(config_path)
    run_dir = resolve_infer_run_directory(
        resolved.config,
        timestamp=_timestamp_suffix(),
    ).run_dir
    run_dir.mkdir(parents=True, exist_ok=False)
    write_resolved_config_artifacts(resolved, run_dir)
    if not resolved.config.debug.dry_run:
        assemble_runtime(resolved.config)
    return 0


def _timestamp_suffix() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
