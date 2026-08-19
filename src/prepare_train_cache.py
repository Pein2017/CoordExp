"""Single-process packing-cache preparation before distributed training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from src.artifacts.identity import (
    assert_absent_artifact_target,
    write_strict_json_atomic,
)
from src.training.cache_workflow import prepare_training_pack_caches


_RECEIPT_SCHEMA = "coordexp-swift-pack-cache-preparation-receipt-v1"
_VERIFICATION_RECEIPT_SCHEMA = "coordexp-swift-pack-cache-verification-receipt-v1"


def _receipt_payload(
    *,
    config_path: Path,
    terminal_status: str,
    result: dict[str, Any] | None,
    failure: BaseException | None,
    require_all_hit: bool = False,
) -> dict[str, Any]:
    body = {
        "schema": (
            _VERIFICATION_RECEIPT_SCHEMA if require_all_hit else _RECEIPT_SCHEMA
        ),
        "terminal_status": terminal_status,
        "config_path": str(config_path.resolve()),
        "result": result,
        "failure": (
            None
            if failure is None
            else {
                "error_type": type(failure).__name__,
                "error_code": str(getattr(failure, "code", "python_exception"))[:128],
            }
        ),
    }
    encoded = json.dumps(
        body,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return {**body, "receipt_sha256": hashlib.sha256(encoded).hexdigest()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare CoordExp-Swift packing caches without loading the model."
    )
    parser.add_argument("--config", required=True, help="Path to runnable YAML config.")
    parser.add_argument(
        "--receipt",
        help="Optional absent strict-JSON target for the terminal preparation receipt.",
    )
    parser.add_argument(
        "--require-all-hit",
        action="store_true",
        help=(
            "Verify that both published split targets already exist and "
            "validate. This mode has no cache-materialization authority: it "
            "fails before any render, tokenize, pack, build, or publication "
            "path when either target is missing or invalid."
        ),
    )
    args = parser.parse_args(argv)
    config_path = Path(args.config)
    require_all_hit = bool(args.require_all_hit)
    receipt_target = (
        None
        if args.receipt is None
        else assert_absent_artifact_target(Path(args.receipt))
    )
    try:
        result = prepare_training_pack_caches(
            config_path,
            require_all_hit=require_all_hit,
        )
    except BaseException as exc:
        if receipt_target is not None:
            write_strict_json_atomic(
                receipt_target,
                _receipt_payload(
                    config_path=config_path,
                    terminal_status="failed",
                    result=None,
                    failure=exc,
                    require_all_hit=require_all_hit,
                ),
            )
        raise
    if receipt_target is not None:
        write_strict_json_atomic(
            receipt_target,
            _receipt_payload(
                config_path=config_path,
                terminal_status="completed",
                result=result,
                failure=None,
                require_all_hit=require_all_hit,
            ),
        )
    print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
