#!/usr/bin/env python3
"""Assemble immutable post-run merge and metric evidence for the five-arm panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.analysis.spatial_scope_history.postrun_assembler import (  # noqa: E402
    assemble_supported_postrun_metrics,
    write_supported_postrun_assembly,
)
from src.analysis.spatial_scope_history.postrun_loader import (  # noqa: E402
    load_postrun_evidence,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--schedule",
        type=Path,
        required=True,
        help="Absolute path to the canonical PrimaryScheduleArtifact JSON wrapper.",
    )
    parser.add_argument(
        "--cohort",
        type=Path,
        required=True,
        help="Absolute path to the exact CohortLedger JSON Lines artifact.",
    )
    parser.add_argument(
        "--attempt-ledger",
        type=Path,
        required=True,
        help="Absolute path to the complete terminal AttemptLedger JSON Lines artifact.",
    )
    parser.add_argument(
        "--readiness-root",
        type=Path,
        required=True,
        help="Absolute path to the sealed readiness-v2 reference-ledger root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="New absolute output root; existing paths are rejected.",
    )
    parser.add_argument(
        "--source-commit",
        required=True,
        help="Lowercase Git commit identifier for the executed source tree.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    evidence = load_postrun_evidence(
        schedule_path=arguments.schedule,
        cohort_path=arguments.cohort,
        attempt_ledger_path=arguments.attempt_ledger,
        readiness_root=arguments.readiness_root,
    )
    assembly = assemble_supported_postrun_metrics(evidence)
    receipt = write_supported_postrun_assembly(
        assembly,
        output_root=arguments.output_root,
        source_commit=arguments.source_commit,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
