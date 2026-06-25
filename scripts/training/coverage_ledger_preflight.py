#!/usr/bin/env python
"""Run the coverage-ledger preflight artifact writer without training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.coverage_ledger.preflight import run_coverage_ledger_preflight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize coverage-ledger smoke preflight artifacts."
    )
    parser.add_argument("--config", required=True, help="Ledger smoke config path.")
    parser.add_argument(
        "--baseline-config",
        required=True,
        help="Paired baseline smoke config path.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Output root; artifacts are written under <output-root>/ledger/.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_coverage_ledger_preflight(
        config_path=args.config,
        baseline_config_path=args.baseline_config,
        output_root=args.output_root,
    )
    print(f"ledger_root={result.artifact_result.ledger_root}")
    print(f"selected_samples={result.artifact_result.selected_samples_path}")
    print(f"alignment_debug={result.artifact_result.alignment_debug_path}")
    print(f"overlay_index={result.artifact_result.overlay_index_path}")
    print(f"overlay_count={len(result.artifact_result.overlay_paths)}")
    if result.artifact_result.packed_materialization_path is not None:
        print(
            "packed_materialization="
            f"{result.artifact_result.packed_materialization_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
