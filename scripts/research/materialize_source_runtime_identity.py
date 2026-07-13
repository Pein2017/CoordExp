#!/usr/bin/env python3
"""Materialize the verifier-derived source/runtime identity receipt once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.analysis.spatial_scope_history.calibration import (  # noqa: E402
    FROZEN_SPATIAL_GRID_SPEC_SHA256,
    SourceRuntimeIdentityReceipt,
    write_immutable_json,
)
from src.analysis.spatial_scope_history.cohort_ledger import (  # noqa: E402
    sha256_file,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-attestation", type=Path, required=True)
    parser.add_argument(
        "--sampled-runtime-attestation-aggregate", type=Path, required=True
    )
    parser.add_argument("--resolved-inference-config", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--readiness-ledger-seal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def run(arguments: argparse.Namespace) -> SourceRuntimeIdentityReceipt:
    source_attestation_path = arguments.source_attestation.expanduser().resolve()
    aggregate_path = (
        arguments.sampled_runtime_attestation_aggregate.expanduser().resolve()
    )
    config_path = arguments.resolved_inference_config.expanduser().resolve()
    checkpoint_manifest_path = arguments.checkpoint_manifest.expanduser().resolve()
    ledger_seal_path = arguments.readiness_ledger_seal.expanduser().resolve()
    receipt = SourceRuntimeIdentityReceipt.from_verified_artifacts(
        source_attestation_path=source_attestation_path,
        sampled_runtime_attestation_aggregate_path=aggregate_path,
        resolved_inference_config_path=config_path,
        checkpoint_manifest_path=checkpoint_manifest_path,
        ledger_seal_sha256=sha256_file(ledger_seal_path),
        processor_contract_sha256=FROZEN_SPATIAL_GRID_SPEC_SHA256,
    )
    write_immutable_json(arguments.output, receipt.to_artifact_dict())
    return receipt


def main() -> int:
    arguments = _parser().parse_args()
    receipt = run(arguments)
    print(
        json.dumps(
            {
                "output": str(arguments.output.expanduser().resolve()),
                "receipt_sha256": receipt.receipt_sha256,
                "schema_version": receipt.schema_version,
                "status": "verified_and_written",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
