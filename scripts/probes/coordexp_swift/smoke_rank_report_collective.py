#!/usr/bin/env python3
"""Exercise the bounded rank-report control collective under torchrun/NCCL."""

from __future__ import annotations

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch
import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.pipeline import _build_rank_report_gatherer  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=256)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(seconds=int(args.timeout_seconds)),
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    original_all_gather_object = dist.all_gather_object
    succeeded = False
    gather: Any | None = None

    def forbid_object_collective(*unused_args: Any, **unused_kwargs: Any) -> None:
        raise AssertionError("unbounded all_gather_object was invoked")

    dist.all_gather_object = forbid_object_collective  # type: ignore[assignment]
    try:
        gather = _build_rank_report_gatherer(world_size)
        if gather is None:
            raise RuntimeError("multi-rank smoke received no report gatherer")
        torch.cuda.reset_peak_memory_stats()
        for step in range(1, int(args.iterations) + 1):
            reports = gather(
                {
                    "kind": "metrics",
                    "planned_step_id": step,
                    "split": "collective.smoke",
                    "rank": rank,
                    "world_size": world_size,
                    "metrics": {
                        "loss/total": float(rank),
                        "acc_top1": float(rank) / max(world_size - 1, 1),
                    },
                }
            )
            observed_ranks = [int(report["rank"]) for report in reports]
            if observed_ranks != list(range(world_size)):
                raise RuntimeError(f"rank order mismatch: {observed_ranks}")
        print(
            json.dumps(
                {
                    "rank": rank,
                    "world_size": world_size,
                    "iterations": int(args.iterations),
                    "default_backend": str(dist.get_backend()),
                    "max_cuda_memory_allocated": int(
                        torch.cuda.max_memory_allocated(local_rank)
                    ),
                    "status": "passed",
                },
                sort_keys=True,
            ),
            flush=True,
        )
        succeeded = True
        return 0
    finally:
        dist.all_gather_object = original_all_gather_object  # type: ignore[assignment]
        if succeeded and dist.is_initialized():
            dist.barrier()
        close = getattr(gather, "close", None)
        if callable(close):
            close()
        if succeeded and dist.is_initialized():
            dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
