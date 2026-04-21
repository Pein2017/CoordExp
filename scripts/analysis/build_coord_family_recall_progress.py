"""Build the coordinate-family recall progress snapshot from artifact summaries."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.coord_family_recall_progress import build_recall_progress  # noqa: E402


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to the recall-progress YAML config.")
    args = parser.parse_args()
    print(json.dumps(build_recall_progress(args.config, repo_root=ROOT), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
