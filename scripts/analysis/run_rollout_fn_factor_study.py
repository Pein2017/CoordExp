#!/usr/bin/env python
"""Run the fixed-checkpoint rollout FN-factor analysis study."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.rollout_fn_factor_study import run_rollout_fn_factor_study  # noqa: E402


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to the study YAML config.")
    args = parser.parse_args()
    result = run_rollout_fn_factor_study(args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
