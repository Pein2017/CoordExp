"""Config-first training entrypoint for CoordExp-swift."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from src.training.pipeline import run_training_pipeline


TrainingRunner = Callable[[Path], Mapping[str, Any]]


def main(
    argv: list[str] | None = None,
    *,
    runner: TrainingRunner = run_training_pipeline,
) -> int:
    parser = argparse.ArgumentParser(description="Run CoordExp-swift supervised training.")
    parser.add_argument("--config", required=True, help="Path to runnable YAML config.")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    result = dict(runner(config_path))
    summary = {"entry_config_path": str(config_path), **result}
    print(json.dumps(summary, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
