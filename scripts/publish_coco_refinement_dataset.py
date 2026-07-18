#!/usr/bin/env python3
"""Publish one standalone COCO refinement split into the mutable training view."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.coco_refinement.dataset_publisher import (  # noqa: E402
    DEFAULT_TRAINING_CONFIG,
    CommittedGenerationPublisher,
    CoordExpSwiftTokenBudgetValidator,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate one terminal standalone working generation and atomically "
            "replace only its train/val norm and coord training files."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument(
        "--training-config",
        type=Path,
        default=DEFAULT_TRAINING_CONFIG,
        help="Current CoordExp-Swift config used for exact 12000-token validation.",
    )
    parser.add_argument("--max-total-tokens", type=int, default=12000)
    return parser


def main() -> int:
    args = _parser().parse_args()
    repo_root = args.repo_root.expanduser().resolve(strict=True)
    config_path = args.training_config
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    validator = CoordExpSwiftTokenBudgetValidator(
        config_path,
        max_total_tokens=args.max_total_tokens,
    )
    receipt = CommittedGenerationPublisher(
        repository_root=repo_root,
        runtime_root=args.runtime_root,
        split=args.split,
        token_budget_validator=validator,
    ).publish()
    print(
        json.dumps(
            receipt.to_artifact_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
