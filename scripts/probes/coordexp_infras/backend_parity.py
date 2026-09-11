#!/usr/bin/env python3
"""Write a read-only HF/vLLM artifact parity and likelihood receipt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference.backend_parity import (  # noqa: E402
    BackendParityInputError,
    write_backend_parity_receipt,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-run-dir", type=Path, required=True)
    parser.add_argument("--vllm-run-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--require-raw", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        receipt = write_backend_parity_receipt(
            hf_run_dir=args.hf_run_dir,
            vllm_run_dir=args.vllm_run_dir,
            receipt_path=args.receipt,
            require_raw=args.require_raw,
        )
    except BackendParityInputError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "failed_gates": receipt["failed_gates"],
                "receipt": str(args.receipt.expanduser().resolve()),
            },
            sort_keys=True,
        )
    )
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
