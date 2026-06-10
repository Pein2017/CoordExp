#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.prefix_state_transition_tomography.finalize import finalize_if_ready


def main() -> int:
    parser = argparse.ArgumentParser(description="Finalize prefix-state transition tomography when shard outputs are ready.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--log-root", type=Path, default=None)
    parser.add_argument("--no-allow-overwrite", action="store_true")
    args = parser.parse_args()
    result = finalize_if_ready(
        config_path=args.config,
        artifact_root=args.artifact_root,
        log_root=args.log_root,
        allow_overwrite=not args.no_allow_overwrite,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
