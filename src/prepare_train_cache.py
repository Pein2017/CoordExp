"""Single-process packing-cache preparation before distributed training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.training.pipeline import prepare_training_pack_caches


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare CoordExp-Swift packing caches without loading the model."
    )
    parser.add_argument("--config", required=True, help="Path to runnable YAML config.")
    args = parser.parse_args(argv)
    result = prepare_training_pack_caches(Path(args.config))
    print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
