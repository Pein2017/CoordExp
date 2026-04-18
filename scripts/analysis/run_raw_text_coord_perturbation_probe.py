#!/usr/bin/env python3
"""Run the raw-text prefix perturbation probe."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    import argparse
    from src.analysis.raw_text_coord_perturbation_probe import run_perturbation_probe

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to the perturbation YAML config.")
    args = parser.parse_args()
    print(json.dumps(run_perturbation_probe(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
