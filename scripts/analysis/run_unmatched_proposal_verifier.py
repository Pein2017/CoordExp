#!/usr/bin/env python
"""Run the unmatched proposal verifier offline ablation study."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.unmatched_proposal_verifier import (  # noqa: E402
    run_unmatched_proposal_verifier_study,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the study YAML config.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Apply only the 8-sample / 1-checkpoint cardinality override on top of the provided config.",
    )
    parser.add_argument(
        "--limit-checkpoints",
        type=int,
        default=None,
        help="Optional checkpoint cap for debugging or smoke workflows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_unmatched_proposal_verifier_study(
        config_path=args.config,
        smoke=bool(args.smoke),
        limit_checkpoints=args.limit_checkpoints,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
