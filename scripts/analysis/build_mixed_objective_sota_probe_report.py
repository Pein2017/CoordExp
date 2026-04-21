"""Build a minimal mixed-objective SOTA probe report bundle from a YAML config."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.mixed_objective_sota_probe_report import (  # noqa: E402
    build_mixed_objective_sota_probe_report,
)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to the report YAML config.")
    args = parser.parse_args()
    print(
        json.dumps(
            build_mixed_objective_sota_probe_report(args.config, repo_root=ROOT),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
