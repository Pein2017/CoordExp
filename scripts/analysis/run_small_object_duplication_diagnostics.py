#!/usr/bin/env python3
"""Run the fixed-checkpoint small-object duplication diagnostics study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.analysis.small_object_duplication_diagnostics import run_study


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to the study YAML config.")
    args = parser.parse_args()
    result = run_study(args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
