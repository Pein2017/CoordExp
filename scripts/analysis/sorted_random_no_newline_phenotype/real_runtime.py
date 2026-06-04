#!/usr/bin/env python
"""Dedicated real GPU runtime entrypoint for A3.2 mechanism probes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.sorted_random_no_newline_phenotype.config import load_config
from src.analysis.sorted_random_no_newline_phenotype.fn_hint_runtime import (
    run_real_fn_hint_probe,
)
from src.analysis.sorted_random_no_newline_phenotype.native_rollout import (
    run_real_native_rollout,
)
from src.analysis.sorted_random_no_newline_phenotype.paired_probe import (
    run_real_paired_checkpoint_probe,
)


REAL_RUNTIME_STAGES = {
    "paired_checkpoint_probe",
    "native_rollout",
    "fn_hint_probe",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run real A3.2 GPU runtime stages with positive provenance.",
    )
    parser.add_argument("--config", required=True, help="Path to the A3.2 YAML config.")
    parser.add_argument(
        "--stages",
        required=True,
        help="Single real runtime stage name.",
    )
    parser.add_argument(
        "--shard-id",
        type=int,
        default=None,
        help="Shard id for sharded real runtime stages.",
    )
    parser.add_argument(
        "--launch-context",
        action="store_true",
        help="Required guard proving invocation came from the controlled launcher.",
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Allow overwriting this stage's real runtime artifacts.",
    )
    args = parser.parse_args(argv)

    try:
        result = _run(args)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))
    return 0


def _run(args: argparse.Namespace) -> dict[str, object]:
    stage = str(args.stages).strip()
    if "," in stage:
        raise ValueError("real_runtime.py accepts exactly one stage per invocation")
    if stage not in REAL_RUNTIME_STAGES:
        raise ValueError(f"unsupported real runtime stage: {stage}")
    if not args.launch_context:
        raise ValueError("real runtime stages require --launch-context")
    config = load_config(args.config)
    if stage == "paired_checkpoint_probe":
        if args.shard_id is None:
            raise ValueError("paired_checkpoint_probe requires --shard-id")
        return run_real_paired_checkpoint_probe(
            config,
            shard_id=int(args.shard_id),
            allow_overwrite=bool(args.allow_overwrite),
        )
    if stage == "native_rollout":
        return run_real_native_rollout(
            config,
            allow_overwrite=bool(args.allow_overwrite),
        )
    if stage == "fn_hint_probe":
        if args.shard_id is None:
            raise ValueError("fn_hint_probe requires --shard-id")
        return run_real_fn_hint_probe(
            config,
            shard_id=int(args.shard_id),
            allow_overwrite=bool(args.allow_overwrite),
        )
    raise NotImplementedError(
        f"{stage} real runtime is not implemented yet; keep GPU launch gated"
    )


if __name__ == "__main__":
    raise SystemExit(main())
