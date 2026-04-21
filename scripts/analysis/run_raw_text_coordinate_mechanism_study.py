from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    from src.analysis.raw_text_coordinate_mechanism_study import run_study

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage")
    parser.add_argument("--model-alias")
    parser.add_argument("--branch")
    args = parser.parse_args()
    result = run_study(
        Path(args.config),
        stage_override=args.stage,
        model_alias=args.model_alias,
        branch_name=args.branch,
    )
    print(result["run_dir"])


if __name__ == "__main__":
    main()
