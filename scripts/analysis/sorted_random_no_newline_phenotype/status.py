#!/usr/bin/env python
"""Print semantic A3.2 artifact status as JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "configs/analysis/sorted_random_no_newline_phenotype/"
    "fullobj_purece_random_vs_sorted_ckpt3668_phase_a3_2.yaml"
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.sorted_random_no_newline_phenotype.config import load_config
from src.analysis.sorted_random_no_newline_phenotype.status import evaluate_status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print A3.2 sorted-vs-random no-newline artifact status.",
    )
    parser.add_argument(
        "--artifact-root",
        default=None,
        help="Artifact root to inspect. Defaults to the root in the A3.2 config.",
    )
    parser.add_argument(
        "--log-root",
        default=None,
        help="Optional log root to echo in the JSON status payload.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Config used only when --artifact-root is omitted.",
    )
    args = parser.parse_args(argv)

    try:
        artifact_root = (
            Path(args.artifact_root)
            if args.artifact_root is not None
            else load_config(args.config).artifact_root
        )
        result = evaluate_status(artifact_root)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    result = {
        "artifact_root": str(artifact_root),
        "log_root": None if args.log_root is None else str(Path(args.log_root)),
        **result,
    }
    print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
