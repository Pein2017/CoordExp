from __future__ import annotations

import argparse
from pathlib import Path

from src.analysis.raw_text_coordinate_mechanism_study import load_study_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_study_config(Path(args.config))
    print(cfg.run.name)


if __name__ == "__main__":
    main()
