#!/usr/bin/env python
"""Wave-2 task 2.5: real two-process gloo probe for typed metric reduction.

Runs two OS processes over a real `gloo` CPU process group and drives the
production collective (`src.training.control_plane._build_rank_report_gatherer`
through `TrainRuntime.gather_metrics`) with deliberately asymmetric rank-local
values. Nothing here re-implements a reducer: every assertion is against the
value the production path returns.

Arms
----
1. **asymmetric reduction** -- unequal rank-local pre-clip gradient norms,
   timings, counts, and ratio numerator/denominator statistics prove `MAX`,
   `SUM`, `IDENTICAL`, `BOOL_ALL`, and sum-before-divide ratio pooling.
2. **schema disagreement** -- one rank declares a different reducer for the
   same metric name; the collective must fail closed on every rank rather
   than reduce mismatched semantics.
3. **replicated-eval divergence measurement** -- both ranks run the SAME
   computation over the SAME data (the shape a replicated evaluator has) and
   the probe reports the maximum absolute cross-rank divergence. That number
   decides whether the replicated-eval `IDENTICAL` reducer can be exact.

Emits one strict-JSON receipt on stdout and exits non-zero on any failure.
CPU only; the rendezvous port is chosen by binding port 0; child processes are
always joined and terminated.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import socket
import sys
import traceback
from datetime import timedelta
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from src.common.errors import RuntimeContractError  # noqa: E402
from src.config.models import RuntimeBatchResolution, RuntimeConfig  # noqa: E402
from src.runtime.metrics import (  # noqa: E402
    IDENTICAL_ABS_TOLERANCE,
    REDUCER_BOOL_ALL,
    REDUCER_IDENTICAL,
    REDUCER_MAX,
    REDUCER_SUM,
    MetricBatch,
    RatioSample,
    ScalarSample,
)
from src.runtime.train_runtime import TrainRuntime  # noqa: E402

WORLD_SIZE = 2
PROCESS_GROUP_TIMEOUT_SECONDS = 60
JOIN_TIMEOUT_SECONDS = 240

# Deliberately asymmetric rank-local observations. Every value differs across
# ranks except the two that are declared IDENTICAL.
RANK_LOCAL = {
    0: {
        "grad_norm/pre_clip_rank_max": 0.25,
        "step_duration_seconds": 0.4,
        "input_wait_seconds": 0.01,
        "count/packs": 1.0,
        "count/examples": 5.0,
        "count/eligible_segments": 4.0,
        "lr/group_0": 1e-5,
        "finite/total_loss": 1.0,
        # ratio: rank-local per-atom mean 0.5 over 2 selected atoms
        "diag_value": 0.5,
        "diag_weight": 2.0,
    },
    1: {
        "grad_norm/pre_clip_rank_max": 1.75,
        "step_duration_seconds": 0.9,
        "input_wait_seconds": 0.30,
        "count/packs": 3.0,
        "count/examples": 9.0,
        "count/eligible_segments": 4.0,
        "lr/group_0": 1e-5,
        "finite/total_loss": 0.0,
        # ratio: rank-local per-atom mean 1.5 over 6 selected atoms
        "diag_value": 1.5,
        "diag_weight": 6.0,
    },
}

EXPECTED = {
    "grad_norm/pre_clip_rank_max": 1.75,
    "step_duration_seconds": 0.9,
    "input_wait_seconds": 0.30,
    "count/packs": 4.0,
    "count/examples": 14.0,
    "count/eligible_segments": 4.0,
    "lr/group_0": 1e-5,
    "finite/total_loss": 0.0,
    "loss/base_ce/token_weighted_diag": (0.5 * 2.0 + 1.5 * 6.0) / 8.0,
}
# What an implicit mean would have produced for the same inputs -- the probe
# proves the reduced values are NOT these.
SUPERSEDED_MEAN = {
    "count/packs": 2.0,
    "count/examples": 7.0,
    "finite/total_loss": 0.5,
    "loss/base_ce/token_weighted_diag": (0.5 + 1.5) / 2.0,
}


class _Accelerator:
    def __init__(self, *, rank: int) -> None:
        self.process_index = rank
        self.num_processes = WORLD_SIZE
        self.device = torch.device("cpu")
        self.is_main_process = rank == 0
        self.distributed_type = type("_T", (), {"name": "NO"})()
        self.gradient_accumulation_steps = 1
        self.mixed_precision = "no"
        self.scaler = None

    def prepare(self, *objects: Any) -> tuple[Any, ...]:
        return objects


def _runtime(*, rank: int, gatherer: Any) -> TrainRuntime:
    return TrainRuntime(
        runtime_config=RuntimeConfig.model_validate(
            {"seed": 17, "determinism": {"mode": "legacy"}}
        ),
        runtime_batch=RuntimeBatchResolution(
            world_size=WORLD_SIZE,
            effective_batch_size=WORLD_SIZE,
            resolved_grad_accum_steps=1,
        ),
        model=torch.nn.Linear(1, 1),
        optimizer=None,
        scheduler=None,
        expected_mixed_precision="no",
        accelerator=_Accelerator(rank=rank),
        rank_report_gatherer=gatherer,
    )


def _asymmetric_batch(rank: int) -> MetricBatch:
    local = RANK_LOCAL[rank]
    return MetricBatch(
        planned_step_id=7,
        split="train",
        samples=(
            ScalarSample(
                "grad_norm/pre_clip_rank_max",
                REDUCER_MAX,
                local["grad_norm/pre_clip_rank_max"],
            ),
            ScalarSample(
                "step_duration_seconds", REDUCER_MAX, local["step_duration_seconds"]
            ),
            ScalarSample(
                "input_wait_seconds", REDUCER_MAX, local["input_wait_seconds"]
            ),
            ScalarSample(
                "count/packs", REDUCER_SUM, local["count/packs"], integral=True
            ),
            ScalarSample(
                "count/examples", REDUCER_SUM, local["count/examples"], integral=True
            ),
            ScalarSample(
                "count/eligible_segments",
                REDUCER_IDENTICAL,
                local["count/eligible_segments"],
                integral=True,
            ),
            ScalarSample("lr/group_0", REDUCER_IDENTICAL, local["lr/group_0"]),
            ScalarSample(
                "finite/total_loss", REDUCER_BOOL_ALL, local["finite/total_loss"]
            ),
            RatioSample(
                "loss/base_ce/token_weighted_diag",
                local["diag_value"] * local["diag_weight"],
                local["diag_weight"],
                empty_value=0.0,
            ),
        ),
    )


def _disagreeing_batch(rank: int) -> MetricBatch:
    """Rank 1 declares MAX where rank 0 declares SUM for the same name."""

    return MetricBatch(
        planned_step_id=8,
        split="train",
        samples=(
            ScalarSample(
                "count/packs",
                REDUCER_SUM if rank == 0 else REDUCER_MAX,
                float(rank + 1),
                integral=True,
            ),
        ),
    )


def _replicated_reference_value() -> float:
    """A deterministic replicated-eval-shaped computation over identical data.

    Both ranks execute the identical sequence of float operations on identical
    inputs -- exactly what a replicated evaluator does when every rank
    evaluates the whole eval set -- so any nonzero cross-rank divergence would
    be a real reason to reject an exact IDENTICAL reducer.
    """

    torch.manual_seed(20260820)
    logits = torch.linspace(-3.0, 3.0, steps=257, dtype=torch.float32).reshape(1, 257)
    targets = torch.arange(257, dtype=torch.long).reshape(1, 257) % 7
    weights = torch.linspace(0.25, 4.0, steps=257, dtype=torch.float32)
    per_atom = torch.nn.functional.cross_entropy(
        logits.repeat(257, 1), targets.reshape(-1), reduction="none"
    )
    weighted = (per_atom * weights).sum() / weights.sum()
    return float(weighted.item())


def _measure_replicated_divergence(rank: int, gatherer: Any) -> dict[str, Any]:
    local_value = _replicated_reference_value()
    reports = gatherer(
        {
            "kind": "replicated_divergence_probe",
            "planned_step_id": 0,
            "split": "eval",
            "rank": rank,
            "world_size": WORLD_SIZE,
            "value": local_value,
            "value_bits": float(local_value).hex(),
        }
    )
    values = [float(report["value"]) for report in reports]
    bits = [str(report["value_bits"]) for report in reports]
    max_abs_divergence = max(abs(value - values[0]) for value in values)
    return {
        "per_rank_values": values,
        "per_rank_value_hex": bits,
        "max_abs_divergence": max_abs_divergence,
        "bitwise_identical": len(set(bits)) == 1,
    }


def _worker(rank: int, port: int, output: Any) -> None:
    result: dict[str, Any] = {"rank": rank}
    gatherer = None
    try:
        os.environ.setdefault("PYTHONHASHSEED", "0")
        dist.init_process_group(
            backend="gloo",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=WORLD_SIZE,
            timeout=timedelta(seconds=PROCESS_GROUP_TIMEOUT_SECONDS),
        )
        from src.training.control_plane import _build_rank_report_gatherer

        gatherer = _build_rank_report_gatherer(WORLD_SIZE)
        runtime = _runtime(rank=rank, gatherer=gatherer)

        # Arm 1: asymmetric reduction through the production collective.
        reduced = runtime.gather_metrics(_asymmetric_batch(rank))
        result["reduced_metrics"] = dict(reduced["metrics"])
        result["reduction"] = reduced["reduction"]
        checks: dict[str, bool] = {}
        for name, expected in EXPECTED.items():
            observed = reduced["metrics"][name]
            checks[f"reduced:{name}"] = math.isclose(
                observed, expected, rel_tol=0.0, abs_tol=1e-12
            )
        for name, superseded in SUPERSEDED_MEAN.items():
            checks[f"not_mean:{name}"] = not math.isclose(
                reduced["metrics"][name], superseded, rel_tol=0.0, abs_tol=1e-12
            )
        # Rank-local scalars stay available to lifecycle accounting only.
        per_rank = reduced["per_rank_metrics"]
        checks["per_rank_projection_has_both_ranks"] = set(per_rank) == {"0", "1"}

        # Arm 2: schema/reducer disagreement must fail closed on every rank.
        try:
            runtime.gather_metrics(_disagreeing_batch(rank))
        except RuntimeContractError as exc:
            checks["schema_disagreement_fails_closed"] = (
                exc.code == "runtime.metric_schema_mismatch"
            )
            result["schema_disagreement_code"] = exc.code
        else:
            checks["schema_disagreement_fails_closed"] = False
            result["schema_disagreement_code"] = None

        # Arm 3: empirical replicated-eval divergence measurement.
        divergence = _measure_replicated_divergence(rank, gatherer)
        result["replicated_divergence"] = divergence
        checks["replicated_divergence_within_declared_tolerance"] = (
            divergence["max_abs_divergence"] <= IDENTICAL_ABS_TOLERANCE
        )

        # The collective still works after the failure arm, and one more
        # reduction converges on both ranks (clean termination, no desync).
        final = runtime.gather_metrics(
            MetricBatch(
                planned_step_id=9,
                split="train",
                samples=(ScalarSample("count/packs", REDUCER_SUM, 1.0, integral=True),),
            )
        )
        checks["collective_usable_after_failure"] = (
            final["metrics"]["count/packs"] == 2.0
        )
        result["checks"] = checks
        result["ok"] = all(checks.values())
    except BaseException:
        result["ok"] = False
        result["traceback"] = traceback.format_exc()
    finally:
        close = getattr(gatherer, "close", None)
        if callable(close):
            try:
                close()
            except BaseException:
                result.setdefault("close_error", traceback.format_exc())
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        result["clean_shutdown"] = not (
            dist.is_available() and dist.is_initialized()
        )
        output.put((rank, result))


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    context = mp.get_context("spawn")
    output = context.Queue()
    port = _free_port()
    processes = [
        context.Process(target=_worker, args=(rank, port, output))
        for rank in range(WORLD_SIZE)
    ]
    results: dict[int, dict[str, Any]] = {}
    try:
        for process in processes:
            process.start()
        for _ in processes:
            rank, payload = output.get(timeout=JOIN_TIMEOUT_SECONDS)
            results[rank] = payload
        for process in processes:
            process.join(timeout=JOIN_TIMEOUT_SECONDS)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=30)

    exit_codes = [process.exitcode for process in processes]
    ok = (
        set(results) == set(range(WORLD_SIZE))
        and all(results[rank].get("ok") for rank in results)
        and all(results[rank].get("clean_shutdown") for rank in results)
        and all(code == 0 for code in exit_codes)
    )
    ranks_agree = (
        len(results) == WORLD_SIZE
        and results[0].get("reduced_metrics") == results[1].get("reduced_metrics")
    )
    divergence = results.get(0, {}).get("replicated_divergence", {})
    receipt = {
        "schema": "coordexp-swift-obs-wave2-gloo-probe-v1",
        "world_size": WORLD_SIZE,
        "backend": "gloo",
        "device": "cpu",
        "rendezvous_port": port,
        "process_exit_codes": exit_codes,
        "every_rank_derives_the_same_global_row": ranks_agree,
        "declared_identical_abs_tolerance": IDENTICAL_ABS_TOLERANCE,
        "replicated_eval_divergence": divergence,
        "identical_reducer_decision": (
            "exact"
            if divergence.get("bitwise_identical") and divergence.get(
                "max_abs_divergence"
            ) == 0.0
            else "bounded_tolerance_required"
        ),
        "per_rank": {str(rank): results[rank] for rank in sorted(results)},
        "ok": bool(ok and ranks_agree),
    }
    print(json.dumps(receipt, sort_keys=True, indent=2))
    return 0 if receipt["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
