#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    from src.analysis.raw_text_coord_manual_review import build_manual_review_bundle

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_manual_review_bundle(args.config), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
