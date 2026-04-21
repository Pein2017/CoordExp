"""Run the coordinate-family inventory and contract audit from a YAML manifest."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.coord_family_contract_audit import run_contract_audit  # noqa: E402


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to the audit YAML config.")
    args = parser.parse_args()
    print(json.dumps(run_contract_audit(args.config, repo_root=ROOT), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
