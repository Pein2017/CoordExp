"""Minimal Wave 2 inference entry delegation."""

from __future__ import annotations

from pathlib import Path

from src.config.inference import load_infer_config
from src.config.writer import write_resolved_config_artifacts
from src.inference.runtime import assemble_runtime


def run(*, config_path: str | Path) -> int:
    resolved = load_infer_config(config_path)
    run_dir = _run_dir(resolved.config)
    write_resolved_config_artifacts(resolved, run_dir, overwrite=True)
    if not resolved.config.debug.dry_run:
        assemble_runtime(resolved.config)
    return 0


def _run_dir(config) -> Path:
    root = Path(config.run.artifact_root)
    run_name = config.run.output_dir or config.run.name
    run_dir = root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
