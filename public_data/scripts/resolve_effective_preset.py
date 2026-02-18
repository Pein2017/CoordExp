#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from public_data.pipeline.naming import resolve_effective_preset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve effective preset naming with legacy max-suffix reuse")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--base-preset", type=str, required=True)
    parser.add_argument("--max-objects", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(resolve_effective_preset(args.dataset_dir, args.base_preset, args.max_objects))


if __name__ == "__main__":
    main()
