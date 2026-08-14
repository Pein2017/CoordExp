#!/usr/bin/env python3
"""Fail-closed one-cell entry point for the Human13 RP crossover screen."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import json
from pathlib import Path
import sys
from typing import TextIO

from scripts.research.human13_rp_crossover_matrix_contracts import (
    CellReceipt,
    CellSpec,
    DRY_RUN_COUNTER_KEYS,
)
from scripts.research.human13_rp_crossover_runtime import (
    CellRuntimeServices,
    run_cell,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-spec", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorize-model-gpu-use", action="store_true")
    return parser


def _load_spec(path: Path) -> CellSpec:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return CellSpec.from_dict(payload)


def build_dry_run_plan(spec: CellSpec) -> dict[str, object]:
    """Describe the requested cell without loading a model or creating outputs."""

    return {
        "schema_version": "human13_rp_crossover_cell_dry_run.v1",
        "mode": "dry_run",
        "cell_spec_sha256": spec.content_sha256,
        "cell_key": spec.cell_key.to_dict(),
        "execution_ready": False,
        "requires_explicit_model_gpu_authority": True,
        "actions": {key: 0 for key in DRY_RUN_COUNTER_KEYS},
    }


def run_cli(
    argv: Sequence[str] | None = None,
    *,
    services_factory: Callable[[CellSpec], CellRuntimeServices] | None = None,
    receipt_writer: Callable[[CellReceipt], None] | None = None,
    stdout: TextIO | None = None,
) -> int:
    """Run the dry plan or one explicitly authorized injected cell."""

    args = _parser().parse_args(argv)
    output = sys.stdout if stdout is None else stdout
    spec = _load_spec(args.cell_spec)
    if not args.execute:
        json.dump(build_dry_run_plan(spec), output, sort_keys=True)
        output.write("\n")
        return 0

    if args.authorize_model_gpu_use is not True:
        raise PermissionError("execute requires explicit model/GPU authority")
    if services_factory is None or receipt_writer is None:
        raise RuntimeError(
            "execute requires an injected production runtime adapter and receipt writer"
        )
    services = services_factory(spec)
    receipt = run_cell(spec, services=services, receipt_writer=receipt_writer)
    json.dump(receipt.to_dict(), output, sort_keys=True)
    output.write("\n")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run_cli(argv)


if __name__ == "__main__":
    raise SystemExit(main())
