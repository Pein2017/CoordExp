#!/usr/bin/env python
"""Wave-3 task 3.8: bounded two-rank INJECTED-OUTCOME optimizer-boundary probe.

Scope and evidence class
------------------------
This probe drives the production optimizer boundary
(``TrainRuntime.post_backward`` -> ``TrainRuntime.execute_optimizer_boundary``
-> ``src.training.reporting.publish_terminal_boundary_row``) over a REAL
two-process ``gloo`` group using the production rank-report gatherer
(``src.training.control_plane._build_rank_report_gatherer``) and a real
``RunWriter`` on rank zero.

CPU/gloo is the correct backend here, and that is a deliberate reading of task
3.8: the task attaches "genuine CUDA ... through real Accelerate" only to the
fp16 finite/overflow sentence, and calls THIS probe "bounded". Every outcome
covered below is INJECTED precisely because a real CUDA GradScaler cannot be
made to produce it on demand (a rank-divergent candidacy, an action-
contradictory post-wrapper skip flag). Nothing in these arms depends on CUDA
arithmetic; what is under test is consensus, receipt truthfulness, terminal-row
publication, and non-progression. Genuine CUDA fp16 evidence is the separate
``obs_wave3_fp16_cuda_probe.py`` receipt, and no fp16 claim may rest on this
one alone.

Arms (task 3.8: "pre-wrapper mixed/unsupported, post-wrapper mixed,
``apply+all_skipped``, and ``scaler_skip+none_skipped``")
--------------------------------------------------------------------------
1. ``pre_wrapper_mixed_scaler_overflow``   - rank 0 overflowed, rank 1 finite.
2. ``pre_wrapper_unrelated_unsafe``        - unsupported fp16 state (a rank
   with an unusable gradient norm and no overflow candidacy).
3. ``pre_wrapper_scaler_candidacy_divergent`` - unsupported fp16 state (only
   one rank has an active scaler).
4. ``post_wrapper_mixed``                  - only rank 0's wrapper reports a skip.
5. ``post_wrapper_apply_all_skipped``      - unanimous skip after an ``apply``.
6. ``post_wrapper_scaler_skip_none_skipped`` - unanimous update after a
   ``scaler_skip`` (the known ``applied_unsafe`` branch).

Per arm the probe asserts, on BOTH ranks:

* truthful terminal receipt fields, including pre-wrapper
  ``mutation_state == "divergent_or_unknown"`` once any rank unscaled;
* EXACTLY ONE terminal row at the CURRENT planned-step id in rank zero's
  ``logging.jsonl`` (and no run tree at all on rank one);
* common failed finalization: byte-identical terminal receipt projections and a
  common publication outcome broadcast on every rank;
* NO rank-local raise before consensus: the rank whose own local state was
  clean terminates identically, and post-wrapper arms show the wrapper
  completed on every rank before any rank raised;
* ZERO scheduler progression (counter, ``last_epoch``, and param-group LR) and
  zero scheduled-handler dispatch.

Emits one strict-JSON receipt to ``--output`` (which MUST NOT already exist),
also echoed on stdout, and exits non-zero on any failed assertion. CPU only.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import socket
import sys
import tempfile
import traceback
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from src.config.models import RuntimeBatchResolution, RuntimeConfig  # noqa: E402
from src.runtime.optimizer_boundary import (  # noqa: E402
    OptimizerBoundaryTerminal,
)
from src.runtime.train_runtime import TrainRuntime  # noqa: E402

WORLD_SIZE = 2
PROCESS_GROUP_TIMEOUT_SECONDS = 120
JOIN_TIMEOUT_SECONDS = 600
LEARNING_RATE = 0.125

PROBE_SCHEMA = "coordexp-swift-obs-wave3-injected-outcome-probe-v1"

ARMS: tuple[str, ...] = (
    "pre_wrapper_mixed_scaler_overflow",
    "pre_wrapper_unrelated_unsafe",
    "pre_wrapper_scaler_candidacy_divergent",
    "post_wrapper_mixed",
    "post_wrapper_apply_all_skipped",
    "post_wrapper_scaler_skip_none_skipped",
)

# One distinct planned-step id per arm so "exactly one terminal row at the
# CURRENT planned-step id" is a real check rather than a tautology.
PLANNED_STEP_IDS: dict[str, int] = {
    arm: 101 + index for index, arm in enumerate(ARMS)
}

# Injected rank-local state. `grad_value=None` means the rank's parameters
# carry NO gradient at all (an unusable norm, not an overflow).
RANK_INJECTION: dict[str, dict[int, dict[str, Any]]] = {
    "pre_wrapper_mixed_scaler_overflow": {
        0: {"scaler": True, "found_inf": True, "grad_value": float("inf"),
            "suppress": False, "skip_flag": False},
        1: {"scaler": True, "found_inf": False, "grad_value": 0.5,
            "suppress": False, "skip_flag": False},
    },
    "pre_wrapper_unrelated_unsafe": {
        0: {"scaler": True, "found_inf": False, "grad_value": 0.5,
            "suppress": False, "skip_flag": False},
        1: {"scaler": True, "found_inf": False, "grad_value": None,
            "suppress": False, "skip_flag": False},
    },
    "pre_wrapper_scaler_candidacy_divergent": {
        0: {"scaler": True, "found_inf": False, "grad_value": 0.5,
            "suppress": False, "skip_flag": False},
        1: {"scaler": False, "found_inf": False, "grad_value": 0.5,
            "suppress": False, "skip_flag": False},
    },
    "post_wrapper_mixed": {
        0: {"scaler": True, "found_inf": False, "grad_value": 0.5,
            "suppress": True, "skip_flag": True},
        1: {"scaler": True, "found_inf": False, "grad_value": 0.5,
            "suppress": False, "skip_flag": False},
    },
    "post_wrapper_apply_all_skipped": {
        0: {"scaler": True, "found_inf": False, "grad_value": 0.5,
            "suppress": True, "skip_flag": True},
        1: {"scaler": True, "found_inf": False, "grad_value": 0.5,
            "suppress": True, "skip_flag": True},
    },
    "post_wrapper_scaler_skip_none_skipped": {
        0: {"scaler": True, "found_inf": True, "grad_value": float("inf"),
            "suppress": False, "skip_flag": False},
        1: {"scaler": True, "found_inf": True, "grad_value": float("inf"),
            "suppress": False, "skip_flag": False},
    },
}

# The truth each terminal receipt MUST report. Nothing here is derived from the
# production code at runtime: these are the spec-authored expectations.
EXPECTED: dict[str, dict[str, Any]] = {
    "pre_wrapper_mixed_scaler_overflow": {
        "terminal_reason": "pre_wrapper_mixed_scaler_overflow",
        "clip_events": 0,
        "action": None,
        "attempted": False,
        "applied": False,
        "step_was_skipped": False,
        "mutation_state": "divergent_or_unknown",
        "post_wrapper_outcome": None,
        "optimizer_step_count": 0,
        "wrapper_called": False,
        "lrs_all_null": True,
        "unscaled_on_every_rank": True,
    },
    "pre_wrapper_unrelated_unsafe": {
        "terminal_reason": "pre_wrapper_unrelated_unsafe",
        "clip_events": 0,
        "action": None,
        "attempted": False,
        "applied": False,
        "step_was_skipped": False,
        "mutation_state": "divergent_or_unknown",
        "post_wrapper_outcome": None,
        "optimizer_step_count": 0,
        "wrapper_called": False,
        "lrs_all_null": True,
        "unscaled_on_every_rank": True,
    },
    "pre_wrapper_scaler_candidacy_divergent": {
        "terminal_reason": "pre_wrapper_scaler_candidacy_divergent",
        "clip_events": 0,
        "action": None,
        "attempted": False,
        "applied": False,
        "step_was_skipped": False,
        "mutation_state": "divergent_or_unknown",
        "post_wrapper_outcome": None,
        "optimizer_step_count": 0,
        "wrapper_called": False,
        "lrs_all_null": True,
        # Only rank zero has an active scaler here; the COMPOSITE state is
        # still divergent_or_unknown because SOME rank unscaled.
        "unscaled_on_every_rank": False,
    },
    "post_wrapper_mixed": {
        "terminal_reason": "post_wrapper_mixed",
        "clip_events": 1,
        "action": "apply",
        "attempted": True,
        "applied": None,
        "step_was_skipped": None,
        "mutation_state": "divergent_or_unknown",
        "post_wrapper_outcome": "mixed",
        "optimizer_step_count": 1,
        "wrapper_called": True,
        "lrs_all_null": True,
        "unscaled_on_every_rank": True,
    },
    "post_wrapper_apply_all_skipped": {
        "terminal_reason": "post_wrapper_apply_all_skipped",
        "clip_events": 1,
        "action": "apply",
        "attempted": True,
        "applied": False,
        "step_was_skipped": True,
        "mutation_state": "scaler_suppressed",
        "post_wrapper_outcome": "all_skipped",
        "optimizer_step_count": 1,
        "wrapper_called": True,
        "lrs_all_null": True,
        "unscaled_on_every_rank": True,
    },
    "post_wrapper_scaler_skip_none_skipped": {
        "terminal_reason": "post_wrapper_scaler_skip_none_skipped",
        "clip_events": 0,
        "action": "scaler_skip",
        "attempted": True,
        "applied": True,
        "step_was_skipped": False,
        "mutation_state": "applied_unsafe",
        "post_wrapper_outcome": "none_skipped",
        "optimizer_step_count": 1,
        "wrapper_called": True,
        # The one branch that retains the identical pre-call LRs that were in
        # fact applied on every rank.
        "lrs_all_null": False,
        "unscaled_on_every_rank": True,
    },
}


# ---------------------------------------------------------------------------
# Injected fp16 surface (the bounded set of attributes TrainRuntime reads)
# ---------------------------------------------------------------------------


class _InjectedScaler:
    def __init__(self, *, found_inf: bool) -> None:
        self.found_inf = bool(found_inf)

    def _found_inf_per_device(self, optimizer: Any = None) -> dict[str, torch.Tensor]:
        del optimizer
        return {"cpu": torch.tensor(1.0 if self.found_inf else 0.0)}


class _InjectedAccelerator:
    """A bounded fp16 accelerator surface bound to a REAL gloo rank.

    `broadcast_object_list` is implemented over the real process group on
    purpose: `src.training.reporting._append_logging_row_shared` prefers the
    accelerator's own method, and this probe must not depend on Accelerate's
    ambient PartialState detection to decide whether the publication outcome is
    actually made common.
    """

    def __init__(
        self,
        *,
        rank: int,
        scaler_active: bool,
        found_inf: bool,
        skip_flag: bool,
    ) -> None:
        self.process_index = rank
        self.num_processes = WORLD_SIZE
        self.device = torch.device("cpu")
        self.is_main_process = rank == 0
        self.distributed_type = SimpleNamespace(name="MULTI_GPU")
        self.gradient_accumulation_steps = 1
        self.mixed_precision = "fp16"
        self.scaler = _InjectedScaler(found_inf=found_inf) if scaler_active else None
        self.unscale_calls: list[Any] = []
        self.accelerator_clip_calls: list[Any] = []
        self.optimizer_step_was_skipped = bool(skip_flag)
        # The production observation seam in `TrainRuntime._record_boundary_event`.
        self.events: list[str] = []

    def prepare(self, *objects: Any) -> tuple[Any, ...]:
        return tuple(objects)

    def backward(self, loss: torch.Tensor) -> None:  # pragma: no cover - unused
        loss.backward()

    def unscale_gradients(self, optimizer: Any = None) -> None:
        self.unscale_calls.append("unscale_gradients")

    def clip_grad_norm_(self, parameters: Any, max_norm: float) -> None:
        # Prohibited on every branch. Recorded AND raised so it can never pass
        # silently.
        self.accelerator_clip_calls.append(float(max_norm))
        raise AssertionError("accelerator.clip_grad_norm_ is prohibited")

    def broadcast_object_list(self, values: list[Any], from_process: int = 0) -> None:
        dist.broadcast_object_list(values, src=int(from_process))


class _CountingOptimizer(torch.optim.SGD):
    """Counts wrapper invocations and can suppress the underlying update."""

    def __init__(self, parameters: Any, *, suppress: bool) -> None:
        super().__init__(parameters, lr=LEARNING_RATE, momentum=0.9)
        self.suppress = bool(suppress)
        self.step_calls = 0
        self.applied_calls = 0

    def step(self, closure: Any | None = None) -> Any:
        self.step_calls += 1
        if self.suppress:
            return None
        self.applied_calls += 1
        return super().step(closure)


def _build_runtime(
    *,
    rank: int,
    arm: str,
    gatherer: Any,
) -> tuple[TrainRuntime, _InjectedAccelerator, _CountingOptimizer, Any]:
    injection = RANK_INJECTION[arm][rank]
    torch.manual_seed(20260820 + rank)
    model = torch.nn.Linear(3, 2, bias=False)
    accelerator = _InjectedAccelerator(
        rank=rank,
        scaler_active=bool(injection["scaler"]),
        found_inf=bool(injection["found_inf"]),
        skip_flag=bool(injection["skip_flag"]),
    )
    optimizer = _CountingOptimizer(
        model.parameters(), suppress=bool(injection["suppress"])
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: 0.5**step
    )
    runtime = TrainRuntime(
        runtime_config=RuntimeConfig.model_validate(
            {"seed": 17, "determinism": {"mode": "legacy"}}
        ),
        runtime_batch=RuntimeBatchResolution(
            world_size=WORLD_SIZE,
            effective_batch_size=WORLD_SIZE,
            resolved_grad_accum_steps=1,
        ),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        expected_mixed_precision="fp16",
        max_grad_norm=1.0,
        accelerator=accelerator,
        rank_report_gatherer=gatherer,
    )
    grad_value = injection["grad_value"]
    for parameter in runtime.model.parameters():
        parameter.grad = (
            None if grad_value is None else torch.full_like(parameter, float(grad_value))
        )
    return runtime, accelerator, optimizer, scheduler


def _make_writer(run_root: Path) -> Any:
    from src.artifacts.run_writer import RunWriter

    return RunWriter.initialize(
        run_dir=run_root / "run",
        run_id="obs-wave3-injected-outcome",
        run_name="obs-wave3-injected-outcome",
        artifact_root=run_root,
        collision_outcome="created",
        created_at="1970-01-01T00:00:00Z",
        config_fingerprint="probe",
        resolved_config={},
        world_size=WORLD_SIZE,
        resolved_max_steps=200,
    )


def _run_arm(*, arm: str, rank: int, gatherer: Any, workspace: Path) -> dict[str, Any]:
    from src.training import reporting

    planned_step_id = PLANNED_STEP_IDS[arm]
    expected = EXPECTED[arm]
    checks: dict[str, bool] = {}
    observed: dict[str, Any] = {"arm": arm, "planned_step_id": planned_step_id}

    runtime, accelerator, optimizer, scheduler = _build_runtime(
        rank=rank, arm=arm, gatherer=gatherer
    )
    initial_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    initial_last_epoch = int(scheduler.last_epoch)

    decision = runtime.post_backward(planned_step_id=planned_step_id)
    observed["decision"] = {
        "optimizer_boundary_action": decision.optimizer_boundary_action,
        "terminal_reason": decision.terminal_reason,
        "optimizer_update_status": decision.optimizer_update_status,
        "scaler_active": bool(decision.scaler_active),
        "unscale_completed": bool(decision.unscale_completed),
        "pre_clip_grad_norm_rank_max": (
            None
            if decision.pre_clip_grad_norm_rank_max is None
            else float(decision.pre_clip_grad_norm_rank_max)
        ),
        "all_ranks_safe": bool(decision.all_ranks_safe),
    }
    checks["converged_decision_matches_expected_terminal_or_action"] = (
        decision.terminal_reason == expected["terminal_reason"]
        if expected["optimizer_step_count"] == 0
        else decision.optimizer_boundary_action == expected["action"]
    )

    # A rank-local branch here would be the bug under test: EVERY rank enters
    # the same call with the same converged decision.
    run_root = workspace / arm
    writer = None
    if rank == 0:
        run_root.mkdir(parents=True, exist_ok=False)
        writer = _make_writer(run_root)

    lifecycle_before: dict[str, Any] = {"completed_steps": 40, "consumed_packs": 80}
    lifecycle = dict(lifecycle_before)

    terminal_error: Exception | None = None
    receipt_dict: dict[str, Any] | None = None
    try:
        runtime.execute_optimizer_boundary(decision, planned_step_id=planned_step_id)
    except OptimizerBoundaryTerminal as terminal:
        terminal_error = terminal
        receipt_dict = terminal.receipt.to_artifact_dict()
        reporting.publish_terminal_boundary_row(
            writer=writer,
            runtime=runtime,
            lifecycle=lifecycle,
            terminal=terminal,
        )
    checks["terminal_raised_on_this_rank"] = terminal_error is not None
    checks["terminal_error_code_is_the_boundary_code"] = (
        getattr(terminal_error, "code", None) == "runtime.optimizer_boundary_terminal"
    )
    observed["receipt"] = receipt_dict

    # -- truthful terminal receipt fields ---------------------------------
    if receipt_dict is None:
        for name in (
            "terminal_receipt_fields_truthful",
            "pre_wrapper_mutation_state_divergent_or_unknown_after_unscale",
            "applied_lr_truth",
        ):
            checks[name] = False
    else:
        checks["terminal_receipt_fields_truthful"] = (
            receipt_dict["planned_step_id"] == planned_step_id
            and receipt_dict["terminal"] is True
            and receipt_dict["terminal_reason"] == expected["terminal_reason"]
            and receipt_dict["optimizer_boundary_action"] == expected["action"]
            and receipt_dict["attempted"] is expected["attempted"]
            and receipt_dict["applied"] is expected["applied"]
            and receipt_dict["step_was_skipped"] is expected["step_was_skipped"]
            and receipt_dict["mutation_state"] == expected["mutation_state"]
            and receipt_dict["post_wrapper_outcome"]
            == expected["post_wrapper_outcome"]
            and receipt_dict["optimizer_update_status"]
            == f"terminal_{expected['terminal_reason']}"
        )
        lrs = list(receipt_dict["group_learning_rates"])
        if expected["lrs_all_null"]:
            checks["applied_lr_truth"] = all(value is None for value in lrs)
        else:
            checks["applied_lr_truth"] = lrs == initial_lrs
        if arm.startswith("pre_wrapper_"):
            checks[
                "pre_wrapper_mutation_state_divergent_or_unknown_after_unscale"
            ] = receipt_dict["mutation_state"] == "divergent_or_unknown"
        else:
            checks[
                "pre_wrapper_mutation_state_divergent_or_unknown_after_unscale"
            ] = True

    # -- exactly one terminal row at the current planned-step id -----------
    row_lines: list[str] = []
    row: dict[str, Any] | None = None
    if rank == 0 and writer is not None:
        row_lines = writer.logging_path.read_text(encoding="utf-8").splitlines()
        if len(row_lines) == 1:
            row = json.loads(row_lines[0])
        checks["exactly_one_terminal_row"] = len(row_lines) == 1
        checks["terminal_row_at_current_planned_step_id"] = bool(
            row is not None and row.get("step") == planned_step_id
        )
        checks["terminal_row_carries_boundary_truth"] = bool(
            row is not None
            and row.get("optimizer_boundary_terminal") is True
            and row.get("optimizer_terminal_reason") == expected["terminal_reason"]
            and row.get("finite_status") == "unavailable"
            and row.get("optimizer_update_status")
            == f"terminal_{expected['terminal_reason']}"
            and row.get("optimizer_step_count") == expected["optimizer_step_count"]
            and row.get("scheduler_step_count") == 0
        )
        observed["terminal_row"] = row
    else:
        # Rank one owns no run tree at all: no directory, no logging file.
        checks["exactly_one_terminal_row"] = not run_root.exists()
        checks["terminal_row_at_current_planned_step_id"] = not run_root.exists()
        checks["terminal_row_carries_boundary_truth"] = not run_root.exists()
        observed["terminal_row"] = None
    observed["row_line_count"] = len(row_lines)

    # -- non-progression ---------------------------------------------------
    checks["zero_scheduler_progression"] = (
        runtime.scheduler_step_count == 0
        and int(scheduler.last_epoch) == initial_last_epoch
        and [float(group["lr"]) for group in optimizer.param_groups] == initial_lrs
    )
    # The probe dispatches no scheduled handler at all; the seam that would
    # dispatch one (session.py's `except OptimizerBoundaryTerminal`) re-raises
    # before any handler, which is asserted in
    # tests/training/test_reporting.py::test_session_publishes_the_terminal_row_before_failed_finalization.
    checks["zero_scheduled_handler_progression"] = lifecycle == lifecycle_before
    checks["completed_wrapper_counter_truth"] = (
        runtime.optimizer_step_count == expected["optimizer_step_count"]
    )
    checks["wrapper_invocation_truth"] = optimizer.step_calls == (
        1 if expected["wrapper_called"] else 0
    )
    checks["accelerator_clip_grad_norm_never_called"] = (
        accelerator.accelerator_clip_calls == []
    )
    # `apply` legitimately clips (once, with the non-unscaling primitive)
    # BEFORE the wrapper; `scaler_skip` and every pre-wrapper terminal must not
    # clip at all. `accelerator.events` is the production observation seam in
    # `TrainRuntime._record_boundary_event`.
    checks["clip_event_count_matches_action"] = accelerator.events == (
        ["clip"] * int(expected["clip_events"])
    )

    observed["counters"] = {
        "optimizer_step_count": runtime.optimizer_step_count,
        "scheduler_step_count": runtime.scheduler_step_count,
        "zero_grad_count": runtime.zero_grad_count,
        "wrapper_step_calls": optimizer.step_calls,
        "underlying_applied_calls": optimizer.applied_calls,
        "unscale_calls": len(accelerator.unscale_calls),
        "scheduler_last_epoch": int(scheduler.last_epoch),
        "param_group_learning_rates": [
            float(group["lr"]) for group in optimizer.param_groups
        ],
    }
    checks["unscale_called_at_most_once_on_this_rank"] = (
        len(accelerator.unscale_calls) <= 1
    )
    checks["unscale_called_exactly_once_when_scaler_active"] = (
        len(accelerator.unscale_calls) == 1
    ) is bool(RANK_INJECTION[arm][rank]["scaler"])

    observed["checks"] = checks
    observed["ok"] = all(checks.values())
    return observed


def _worker(rank: int, port: int, output: Any) -> None:
    payload: dict[str, Any] = {"rank": rank, "arms": {}}
    gatherer = None
    workspace_dir: tempfile.TemporaryDirectory[str] | None = None
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=WORLD_SIZE,
            timeout=timedelta(seconds=PROCESS_GROUP_TIMEOUT_SECONDS),
        )
        from src.training.control_plane import _build_rank_report_gatherer

        gatherer = _build_rank_report_gatherer(WORLD_SIZE)
        if gatherer is None:
            raise RuntimeError("production rank report gatherer was not constructed")
        workspace_dir = tempfile.TemporaryDirectory(prefix="obs-wave3-injected-")
        workspace = Path(workspace_dir.name) / f"rank{rank}"
        workspace.mkdir(parents=True, exist_ok=True)

        for arm in ARMS:
            payload["arms"][arm] = _run_arm(
                arm=arm, rank=rank, gatherer=gatherer, workspace=workspace
            )

        # Cross-rank agreement over the SAME collective the production boundary
        # uses. Kept small: the gatherer frame is bounded at 64 KiB.
        compact = {
            "kind": "obs_wave3_injected_outcome_summary",
            "planned_step_id": 0,
            "rank": rank,
            "world_size": WORLD_SIZE,
            "arms": {
                arm: {
                    "receipt": payload["arms"][arm]["receipt"],
                    "checks_ok": bool(payload["arms"][arm]["ok"]),
                    "counters": payload["arms"][arm]["counters"],
                }
                for arm in ARMS
            },
        }
        gathered = gatherer(compact)
        agreement: dict[str, bool] = {}
        for arm in ARMS:
            receipts = [report["arms"][arm]["receipt"] for report in gathered]
            agreement[arm] = all(item == receipts[0] for item in receipts)
        payload["cross_rank_receipt_identical"] = agreement
        payload["ok"] = all(
            payload["arms"][arm]["ok"] for arm in ARMS
        ) and all(agreement.values())
    except BaseException:
        payload["ok"] = False
        payload["traceback"] = traceback.format_exc()
    finally:
        close = getattr(gatherer, "close", None)
        if callable(close):
            try:
                close()
            except BaseException:
                payload.setdefault("close_error", traceback.format_exc())
        if dist.is_available() and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except BaseException:
                payload.setdefault("shutdown_error", traceback.format_exc())
        payload["clean_shutdown"] = not (
            dist.is_available() and dist.is_initialized()
        )
        if workspace_dir is not None:
            try:
                workspace_dir.cleanup()
            except BaseException:
                pass
        output.put((rank, payload))


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _drive() -> tuple[dict[int, dict[str, Any]], list[int | None], int]:
    context = mp.get_context("spawn")
    output = context.Queue()
    port = _free_port()
    processes = [
        context.Process(target=_worker, args=(rank, port, output), daemon=False)
        for rank in range(WORLD_SIZE)
    ]
    results: dict[int, dict[str, Any]] = {}
    try:
        for process in processes:
            process.start()
        for _ in processes:
            rank, item = output.get(timeout=JOIN_TIMEOUT_SECONDS)
            results[int(rank)] = item
        for process in processes:
            process.join(timeout=JOIN_TIMEOUT_SECONDS)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=30)
    return results, [process.exitcode for process in processes], port


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="strict-JSON receipt path; MUST NOT already exist",
    )
    args = parser.parse_args()
    output_path: Path = args.output
    if output_path.exists():
        print(f"[result] FAILED: --output already exists: {output_path}")
        return 1

    results, exit_codes, port = _drive()

    findings: list[str] = []
    if sorted(results) != list(range(WORLD_SIZE)):
        findings.append(f"missing worker payloads: {sorted(results)}")
    for rank in sorted(results):
        if results[rank].get("traceback"):
            findings.append(f"rank {rank} raised:\n{results[rank]['traceback']}")
        if not results[rank].get("clean_shutdown"):
            findings.append(f"rank {rank} did not shut its process group down")
    if exit_codes != [0] * WORLD_SIZE:
        findings.append(f"worker exit codes {exit_codes}")

    arms_receipt: dict[str, Any] = {}
    for arm in ARMS:
        per_rank = {}
        arm_ok = True
        for rank in sorted(results):
            view = results[rank].get("arms", {}).get(arm)
            if view is None:
                findings.append(f"{arm}: rank {rank} produced no observation")
                arm_ok = False
                continue
            failed = sorted(name for name, ok in view["checks"].items() if not ok)
            if failed:
                findings.append(f"{arm}: rank {rank} failed checks {failed}")
                arm_ok = False
            per_rank[str(rank)] = view
        identical = all(
            results[rank].get("cross_rank_receipt_identical", {}).get(arm)
            for rank in sorted(results)
        )
        if not identical:
            findings.append(f"{arm}: terminal receipts diverged across ranks")
            arm_ok = False
        # Common failed finalization: both ranks raised the SAME boundary
        # terminal, and the rank whose own local state was clean did not
        # short-circuit before the consensus.
        raised = [
            bool(per_rank.get(str(rank), {}).get("checks", {}).get(
                "terminal_raised_on_this_rank"
            ))
            for rank in sorted(results)
        ]
        if not all(raised):
            findings.append(f"{arm}: not every rank converged the failed finalization")
            arm_ok = False
        arms_receipt[arm] = {
            "planned_step_id": PLANNED_STEP_IDS[arm],
            "expected": EXPECTED[arm],
            "injection": {
                str(rank): RANK_INJECTION[arm][rank] for rank in range(WORLD_SIZE)
            },
            "per_rank": per_rank,
            "cross_rank_receipt_identical": bool(identical),
            "ok": bool(arm_ok),
        }

    receipt: dict[str, Any] = {
        "schema": PROBE_SCHEMA,
        "change": "add-coordexp-swift-training-observability",
        "task": "3.8",
        "evidence_class": "injected_outcome_two_rank_cpu",
        "claims_fp16_cuda_evidence": False,
        "world_size": WORLD_SIZE,
        "backend": "gloo",
        "device": "cpu",
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_initialized": bool(torch.cuda.is_initialized()),
        "torch_version": torch.__version__,
        "rendezvous_port": port,
        "process_exit_codes": exit_codes,
        "production_seams_driven": [
            "src.runtime.train_runtime.TrainRuntime.post_backward",
            "src.runtime.train_runtime.TrainRuntime.execute_optimizer_boundary",
            "src.training.reporting.publish_terminal_boundary_row",
            "src.training.control_plane._build_rank_report_gatherer",
            "src.artifacts.run_writer.RunWriter.append_logging_row",
        ],
        "instrumentation": {
            "monkeypatched_modules": [],
            "injection_surface": (
                "a bounded accelerator double implementing only the attributes "
                "TrainRuntime reads (mixed_precision, scaler, unscale_gradients, "
                "clip_grad_norm_, optimizer_step_was_skipped, events, "
                "broadcast_object_list); every decision, receipt, row, and "
                "counter comes from unmodified production code"
            ),
        },
        "arms": arms_receipt,
        "findings": findings,
        "ok": not findings,
    }
    text = json.dumps(receipt, sort_keys=True, indent=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"[result] {'PASSED' if receipt['ok'] else 'FAILED'}: {output_path}")
    return 0 if receipt["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
