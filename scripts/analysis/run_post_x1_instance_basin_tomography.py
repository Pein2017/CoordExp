#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.post_x1_instance_basin_tomography.runner import run_stages


def main() -> None:
    parser = argparse.ArgumentParser(description="Run A3.3 post-x1 instance-basin tomography stages.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stages", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mock-runtime", action="store_true")
    parser.add_argument("--real-runtime", action="store_true")
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--shard-id", type=int, default=None)
    args = parser.parse_args()
    result = run_stages(
        Path(args.config),
        stages=args.stages,
        dry_run=args.dry_run,
        mock_runtime=args.mock_runtime,
        real_runtime=args.real_runtime,
        allow_overwrite=args.allow_overwrite,
        shard_id=args.shard_id,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
