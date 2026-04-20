from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.coord_family_text_subset import materialize_text_pixel_subset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize a text-mode pixel subset from a norm1000 coordinate JSONL."
    )
    parser.add_argument("--src-jsonl", type=Path, required=True)
    parser.add_argument("--dst-jsonl", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = materialize_text_pixel_subset(args.src_jsonl, args.dst_jsonl)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
