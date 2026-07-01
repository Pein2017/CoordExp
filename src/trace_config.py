"""Dry config tracing entrypoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config.loader import load_train_config
from src.config.writer import write_resolved_config_artifacts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resolve a CoordExp-swift config.")
    parser.add_argument("--config", required=True, help="Path to runnable YAML config.")
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Optional directory where configs/resolved.{yaml,json} are written.",
    )
    args = parser.parse_args(argv)

    resolved = load_train_config(args.config)
    artifacts = None
    if args.run_dir is not None:
        artifacts = write_resolved_config_artifacts(resolved, Path(args.run_dir))

    summary = {
        "entry_config_path": str(resolved.entry_config_path),
        "fingerprint": resolved.fingerprint,
        "schema_version": resolved.schema_version,
        "source_count": len(resolved.sources),
        "path_origin_count": len(resolved.path_origins),
        "artifacts": None
        if artifacts is None
        else {
            "yaml_path": str(artifacts.yaml_path),
            "json_path": str(artifacts.json_path),
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
