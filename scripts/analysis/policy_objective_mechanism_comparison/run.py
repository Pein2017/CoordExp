from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.analysis.policy_objective_mechanism_comparison.runner import run_from_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate FullObj policy/objective mechanism artifacts.")
    parser.add_argument("--config", required=True, help="Path to comparison YAML config.")
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()
    result = run_from_config(Path(args.config), allow_overwrite=bool(args.allow_overwrite))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

