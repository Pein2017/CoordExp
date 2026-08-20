#!/usr/bin/env python
"""Wave-3 task 3.8: genuine CUDA fp16 finite + overflow arms through Accelerate.

What this proves
----------------
Two arms, one planned optimizer step each, driven through the PRODUCTION
boundary (`TrainRuntime.backward` -> `TrainRuntime.post_backward` ->
`TrainRuntime.execute_optimizer_boundary` -> `TrainRuntime.scheduler_step`) with
a REAL `accelerate.Accelerator(mixed_precision="fp16")`, a real
`torch.amp.GradScaler`, a real CUDA device, and a tiny real `nn.Module`:

* **FINITE arm** - exactly one `accelerator.unscale_gradients(optimizer)`, then
  finite/norm authority over the ALREADY-UNSCALED gradients, ZERO
  `accelerator.clip_grad_norm_` calls, exactly one NON-unscaling clip
  (`torch.nn.utils.clip_grad_norm_`), an APPLIED update (parameters actually
  changed, underlying optimizer state actually mutated), applied LR equal to the
  PRE-call param-group value (and different from the post-scheduler value), and
  a post-wrapper consensus of `none_skipped`.
* **OVERFLOW arm** - a poisoned loss makes the SCALED backward overflow fp32, so
  the real GradScaler records `found_inf`. The probe proves an all-rank-confirmed
  `scaler_skip`, ZERO clip calls of either kind, exactly one wrapper call solely
  for scaler finalization, post-call skipped truth (the real GradScaler skipped:
  parameters byte-identical and the underlying optimizer never stepped), null
  applied LRs in the receipt, exactly one completed-wrapper counter increment,
  planned scheduler progression, and no false applied status.

Modes
-----
* default              - single process, `world_size == 1`, real CUDA.
* `--two-rank`         - launched under `torch.distributed.run` with
  `--nproc_per_node=2`; both ranks run both arms through the production
  rank-report gatherer and rank zero writes the merged receipt.
* `--dry-structure`    - NO CUDA, NO Accelerate. Runs the identical arm driver
  and the identical assertion functions against a duck-typed CPU environment so
  the assertion plumbing itself is validated. Its receipt is stamped
  `evidence_class="plumbing_only"`, `cuda_used=false`,
  `claims_fp16_evidence=false` and MUST NEVER be cited as fp16 evidence.

Instrumentation (disclosed in the receipt)
------------------------------------------
* `accelerator.events` - the PRODUCTION observation seam already implemented in
  `TrainRuntime._record_boundary_event`; the probe only supplies the list.
* instance-attribute spies (no module patching) that DELEGATE to the real bound
  methods: `accelerator.unscale_gradients`, `accelerator.clip_grad_norm_`, and
  the prepared `AcceleratedOptimizer.step`.
* ONE module-level delegating spy: `torch.nn.utils.clip_grad_norm_`, installed
  for the duration of the run and restored in a `finally`. It exists to prove
  the IDENTITY and CALL COUNT of the non-unscaling clip primitive, which cannot
  be observed from outside the process any other way. It is paired with a
  physical proof that does not depend on it: the post-boundary gradient norm is
  bounded by `--max-grad-norm` while the pre-clip norm was larger.

Emits one strict-JSON receipt to `--output` (which MUST NOT already exist) and
exits non-zero on any failed assertion or missing CUDA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from src.config.models import RuntimeBatchResolution, RuntimeConfig  # noqa: E402
from src.runtime.train_runtime import TrainRuntime  # noqa: E402

PROBE_SCHEMA = "coordexp-swift-obs-wave3-fp16-cuda-probe-v1"
ARMS: tuple[str, ...] = ("finite", "overflow")
PLANNED_STEP_IDS: dict[str, int] = {"finite": 1, "overflow": 1}

# Ordered boundary-event log shared by the delegating spies. Reset per arm.
_ORDER_LOG: list[str] = []
_CLIP_PRIMITIVE_CALLS: list[float] = []

NORM_ABS_TOLERANCE = 1e-3
NORM_REL_TOLERANCE = 1e-3


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def _parameter_digest(parameters: Any) -> str:
    digest = hashlib.sha256()
    for parameter in parameters:
        digest.update(
            parameter.detach().to(torch.float32).cpu().contiguous().numpy().tobytes()
        )
    return digest.hexdigest()


def _gradient_norm(parameters: Any) -> float | None:
    total = 0.0
    seen = False
    for parameter in parameters:
        grad = parameter.grad
        if grad is None:
            continue
        seen = True
        value = float(torch.linalg.vector_norm(grad.detach().float()).cpu())
        if not math.isfinite(value) or not math.isfinite(total):
            return float("inf")
        total += value**2
    if not seen:
        return None
    return math.sqrt(total)


def _close(observed: float, expected: float) -> bool:
    return abs(observed - expected) <= (
        NORM_ABS_TOLERANCE + NORM_REL_TOLERANCE * abs(expected)
    )


class _CountingOptimizer(torch.optim.SGD):
    """Counts wrapper entries and ACTUALLY-APPLIED underlying updates."""

    def __init__(self, parameters: Any, *, lr: float, suppress: bool = False) -> None:
        super().__init__(parameters, lr=lr, momentum=0.9)
        self.suppress = bool(suppress)
        self.step_calls = 0
        self.applied_calls = 0

    def step(self, closure: Any | None = None) -> Any:
        self.step_calls += 1
        if self.suppress:
            return None
        self.applied_calls += 1
        return super().step(closure)


def _spy(name: str, target: Callable[..., Any], sink: list[Any]) -> Callable[..., Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        _ORDER_LOG.append(name)
        sink.append(name)
        return target(*args, **kwargs)

    return wrapper


def _build_model(*, width: int, seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(width, 2 * width, bias=True),
        torch.nn.Tanh(),
        torch.nn.Linear(2 * width, width, bias=True),
    )


# ---------------------------------------------------------------------------
# arm environments
# ---------------------------------------------------------------------------


class _CudaAccelerateEnvironment:
    """A fresh real `Accelerator` + `TrainRuntime` for ONE arm.

    A fresh Accelerator per arm is mandatory, not stylistic: `prepare` registers
    the optimizer on `Accelerator._optimizers` and `AcceleratedOptimizer`
    LATCHES `_is_overflow`, so `accelerator.optimizer_step_was_skipped` would
    otherwise leak the previous arm's skip flag into this arm's post-wrapper
    consensus. A fresh instance also restores the scaler's initial scale.
    """

    kind = "cuda_accelerate"

    def __init__(self, *, args: argparse.Namespace, rank: int, world_size: int,
                 gatherer: Any) -> None:
        from accelerate import Accelerator

        self.accelerator = Accelerator(mixed_precision="fp16")
        self.accelerator.events = []
        self.unscale_calls: list[Any] = []
        self.accelerator_clip_calls: list[Any] = []
        self.wrapper_calls: list[Any] = []
        self.accelerator.unscale_gradients = _spy(
            "unscale_gradients", self.accelerator.unscale_gradients, self.unscale_calls
        )
        self.accelerator.clip_grad_norm_ = _spy(
            "accelerator_clip_grad_norm_",
            self.accelerator.clip_grad_norm_,
            self.accelerator_clip_calls,
        )
        model = _build_model(width=args.width, seed=args.seed)
        self.inner_optimizer = _CountingOptimizer(model.parameters(), lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.inner_optimizer, lr_lambda=lambda step: 0.5**step
        )
        self.runtime = TrainRuntime(
            runtime_config=RuntimeConfig.model_validate(
                {"seed": args.seed, "determinism": {"mode": "legacy"}}
            ),
            runtime_batch=RuntimeBatchResolution(
                world_size=world_size,
                effective_batch_size=world_size * args.batch_size,
                resolved_grad_accum_steps=1,
            ),
            model=model,
            optimizer=self.inner_optimizer,
            scheduler=self.scheduler,
            expected_mixed_precision="fp16",
            max_grad_norm=args.max_grad_norm,
            accelerator=self.accelerator,
            rank_report_gatherer=gatherer,
        )
        # The prepared wrapper is the object the boundary actually calls.
        self.prepared_optimizer = self.runtime.optimizer
        self.prepared_optimizer.step = _spy(
            "wrapper_step", self.prepared_optimizer.step, self.wrapper_calls
        )
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.device = self.runtime.device

    def describe(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "device": str(self.device),
            "distributed_type": str(
                getattr(self.accelerator.distributed_type, "name", "")
            ),
            "mixed_precision": str(self.accelerator.mixed_precision),
            "native_amp": bool(getattr(self.accelerator, "native_amp", False)),
            "scaler_class": type(self.accelerator.scaler).__name__,
            "initial_scale": float(self.accelerator.scaler.get_scale()),
            "model_class": type(self.runtime.model).__name__,
            "parameter_count": int(
                sum(p.numel() for p in self.runtime.model.parameters())
            ),
        }

    def produce_gradients(self, *, poison: bool, planned_step_id: int) -> dict[str, Any]:
        args = self.args
        generator = torch.Generator(device="cpu").manual_seed(args.seed + self.rank)
        inputs = torch.randn(
            args.batch_size, args.width, generator=generator
        ).to(self.device)
        targets = torch.randn(
            args.batch_size, args.width, generator=generator
        ).to(self.device)
        outputs = self.runtime.model(inputs)
        loss = torch.nn.functional.mse_loss(outputs, targets)
        if loss.dtype is not torch.float32:
            raise RuntimeError(
                "fp16 probe requires an fp32 loss before poisoning; observed "
                f"{loss.dtype}. A non-fp32 loss cannot carry the overflow "
                "multiplier and would silently degenerate the overflow arm."
            )
        base_loss = float(loss.detach().cpu())
        if poison:
            loss = loss * args.overflow_loss_multiplier
        poisoned_loss = float(loss.detach().cpu())
        self.runtime.backward(loss, planned_step_id=planned_step_id)
        return {
            "base_loss": base_loss,
            "poisoned_loss": poisoned_loss,
            "loss_is_finite": math.isfinite(poisoned_loss),
            "grad_scale_before_wrapper": float(self.accelerator.scaler.get_scale()),
        }

    def prime_post_wrapper_flag(self, *, expect_skip: bool) -> None:
        # Real Accelerate owns this flag; the probe must not write it.
        del expect_skip

    def scaler_ground_truth_found_inf(self) -> bool | None:
        """`found_inf` keyed by the optimizer the scaler was ACTUALLY given.

        `Accelerator.unscale_gradients` unwraps `AcceleratedOptimizer` before
        calling `scaler.unscale_(opt)`, so the scaler's per-optimizer state is
        keyed on the INNER optimizer. Must be read before the wrapper call:
        `scaler.update()` clears `_per_optimizer_states`.
        """

        scaler = self.accelerator.scaler
        states = getattr(scaler, "_per_optimizer_states", None)
        if states is None:
            return None
        key = id(self.inner_optimizer)
        if key not in states:
            return None
        found = states[key].get("found_inf_per_device", {})
        return bool(any(bool(torch.as_tensor(v).item()) for v in found.values()))

    def wrapper_step_calls(self) -> int:
        return len(self.wrapper_calls)

    def underlying_applied_calls(self) -> int:
        return int(self.inner_optimizer.applied_calls)

    def optimizer_step_was_skipped(self) -> bool | None:
        return bool(self.accelerator.optimizer_step_was_skipped)


class _DryStructureEnvironment:
    """CPU, duck-typed fp16 surface. Validates the assertion plumbing only."""

    kind = "dry_structure"

    def __init__(self, *, args: argparse.Namespace, rank: int, world_size: int,
                 gatherer: Any, poison: bool) -> None:
        self.unscale_calls: list[Any] = []
        self.accelerator_clip_calls: list[Any] = []
        self.wrapper_calls: list[Any] = []
        order_log = _ORDER_LOG
        unscale_sink = self.unscale_calls
        clip_sink = self.accelerator_clip_calls

        class _Scaler:
            def __init__(self) -> None:
                self.found_inf = False

            def _found_inf_per_device(self, optimizer: Any = None) -> dict[str, Any]:
                del optimizer
                return {"cpu": torch.tensor(1.0 if self.found_inf else 0.0)}

            def get_scale(self) -> float:
                return 65536.0

        class _Accelerator:
            def __init__(self) -> None:
                self.process_index = rank
                self.num_processes = world_size
                self.device = torch.device("cpu")
                self.is_main_process = rank == 0
                self.distributed_type = SimpleNamespace(name="NO")
                self.gradient_accumulation_steps = 1
                self.mixed_precision = "fp16"
                self.native_amp = True
                self.scaler = _Scaler()
                self.optimizer_step_was_skipped = False
                self.events: list[str] = []

            def prepare(self, *objects: Any) -> tuple[Any, ...]:
                return tuple(objects)

            def backward(self, loss: torch.Tensor) -> None:
                loss.backward()

            def unscale_gradients(self, optimizer: Any = None) -> None:
                order_log.append("unscale_gradients")
                unscale_sink.append("unscale_gradients")

            def clip_grad_norm_(self, parameters: Any, max_norm: float) -> None:
                order_log.append("accelerator_clip_grad_norm_")
                clip_sink.append(float(max_norm))
                raise AssertionError("accelerator.clip_grad_norm_ is prohibited")

        self.accelerator = _Accelerator()
        model = _build_model(width=args.width, seed=args.seed)
        self.inner_optimizer = _CountingOptimizer(
            model.parameters(), lr=args.lr, suppress=poison
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.inner_optimizer, lr_lambda=lambda step: 0.5**step
        )
        self.runtime = TrainRuntime(
            runtime_config=RuntimeConfig.model_validate(
                {"seed": args.seed, "determinism": {"mode": "legacy"}}
            ),
            runtime_batch=RuntimeBatchResolution(
                world_size=world_size,
                effective_batch_size=world_size * args.batch_size,
                resolved_grad_accum_steps=1,
            ),
            model=model,
            optimizer=self.inner_optimizer,
            scheduler=self.scheduler,
            expected_mixed_precision="fp16",
            max_grad_norm=args.max_grad_norm,
            accelerator=self.accelerator,
            rank_report_gatherer=gatherer,
        )
        self.prepared_optimizer = self.runtime.optimizer
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.device = self.runtime.device
        self._poison = poison

    def describe(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "device": str(self.device),
            "distributed_type": "NO",
            "mixed_precision": "fp16(duck-typed)",
            "native_amp": False,
            "scaler_class": "probe-local duck type",
            "initial_scale": 65536.0,
            "model_class": type(self.runtime.model).__name__,
            "parameter_count": int(
                sum(p.numel() for p in self.runtime.model.parameters())
            ),
        }

    def produce_gradients(self, *, poison: bool, planned_step_id: int) -> dict[str, Any]:
        args = self.args
        generator = torch.Generator(device="cpu").manual_seed(args.seed + self.rank)
        inputs = torch.randn(args.batch_size, args.width, generator=generator)
        targets = torch.randn(args.batch_size, args.width, generator=generator)
        outputs = self.runtime.model(inputs)
        loss = torch.nn.functional.mse_loss(outputs, targets)
        base_loss = float(loss.detach())
        self.runtime.backward(loss, planned_step_id=planned_step_id)
        if poison:
            # EMULATED overflow: the duck-typed scaler has no real scale, so
            # the non-finite gradient state is written directly.
            for parameter in self.runtime.model.parameters():
                parameter.grad = torch.full_like(parameter, float("inf"))
            self.accelerator.scaler.found_inf = True
        return {
            "base_loss": base_loss,
            "poisoned_loss": base_loss,
            "loss_is_finite": math.isfinite(base_loss),
            "grad_scale_before_wrapper": 65536.0,
        }

    def prime_post_wrapper_flag(self, *, expect_skip: bool) -> None:
        # EMULATED: no real GradScaler sets this flag on CPU.
        self.accelerator.optimizer_step_was_skipped = bool(expect_skip)

    def scaler_ground_truth_found_inf(self) -> bool | None:
        return bool(self.accelerator.scaler.found_inf)

    def wrapper_step_calls(self) -> int:
        return int(self.inner_optimizer.step_calls)

    def underlying_applied_calls(self) -> int:
        return int(self.inner_optimizer.applied_calls)

    def optimizer_step_was_skipped(self) -> bool | None:
        return bool(self.accelerator.optimizer_step_was_skipped)


# ---------------------------------------------------------------------------
# the shared arm driver (identical for every environment)
# ---------------------------------------------------------------------------


def run_arm(*, arm: str, env: Any, args: argparse.Namespace) -> dict[str, Any]:
    planned_step_id = PLANNED_STEP_IDS[arm]
    poison = arm == "overflow"
    runtime = env.runtime
    _ORDER_LOG.clear()
    clip_primitive_start = len(_CLIP_PRIMITIVE_CALLS)

    parameters = list(runtime.model.parameters())
    pre_digest = _parameter_digest(parameters)
    pre_lrs = [float(group["lr"]) for group in env.inner_optimizer.param_groups]
    pre_scheduler_last_epoch = int(env.scheduler.last_epoch)
    pre_optimizer_step_count = int(runtime.optimizer_step_count)
    pre_scheduler_step_count = int(runtime.scheduler_step_count)
    pre_optimizer_state_entries = len(env.inner_optimizer.state)

    backward_info = env.produce_gradients(poison=poison, planned_step_id=planned_step_id)

    decision = runtime.post_backward(planned_step_id=planned_step_id)
    measured_unscaled_norm = _gradient_norm(parameters)
    scaler_ground_truth = env.scaler_ground_truth_found_inf()

    env.prime_post_wrapper_flag(expect_skip=poison)

    receipt = runtime.execute_optimizer_boundary(
        decision, planned_step_id=planned_step_id
    )
    post_boundary_norm = _gradient_norm(parameters)
    post_digest = _parameter_digest(parameters)
    post_lrs_before_scheduler = [
        float(group["lr"]) for group in env.inner_optimizer.param_groups
    ]
    skipped_flag = env.optimizer_step_was_skipped()

    scheduler_payload = runtime.scheduler_step(planned_step_id=planned_step_id)
    post_scheduler_lrs = [
        float(group["lr"]) for group in env.inner_optimizer.param_groups
    ]

    clip_primitive_calls = _CLIP_PRIMITIVE_CALLS[clip_primitive_start:]
    order_log = list(_ORDER_LOG)

    observed: dict[str, Any] = {
        "arm": arm,
        "planned_step_id": planned_step_id,
        "environment": env.describe(),
        "backward": backward_info,
        "decision": {
            "optimizer_boundary_action": decision.optimizer_boundary_action,
            "terminal_reason": decision.terminal_reason,
            "optimizer_update_status": decision.optimizer_update_status,
            "finite_status": decision.finite_status,
            "all_ranks_safe": bool(decision.all_ranks_safe),
            "scaler_active": bool(decision.scaler_active),
            "unscale_completed": bool(decision.unscale_completed),
            "pre_clip_grad_norm_rank_max": (
                None
                if decision.pre_clip_grad_norm_rank_max is None
                else float(decision.pre_clip_grad_norm_rank_max)
            ),
            "world_size": int(decision.world_size),
            "ranks": list(decision.ranks),
            "rank_diagnostics": [
                {
                    "rank": item["rank"],
                    "gradients_finite": item["gradients_finite"],
                    "backend_overflow": item["backend_overflow"],
                    "grad_norm_finite": item["grad_norm_finite"],
                    "scaler_active": item["scaler_active"],
                    "unscale_completed": item["unscale_completed"],
                    "scaler_found_inf": item["scaler_found_inf"],
                    "report_error_code": item["report_error_code"],
                }
                for item in decision.rank_diagnostics
            ],
        },
        "receipt": receipt.to_artifact_dict(),
        "order_log": order_log,
        "accelerator_events": list(env.accelerator.events),
        "clip_primitive_call_count": len(clip_primitive_calls),
        "clip_primitive_max_norms": [float(value) for value in clip_primitive_calls],
        "accelerator_clip_call_count": len(env.accelerator_clip_calls),
        "unscale_call_count": len(env.unscale_calls),
        "wrapper_step_calls": env.wrapper_step_calls(),
        "underlying_applied_calls": env.underlying_applied_calls(),
        "optimizer_step_was_skipped_after_wrapper": skipped_flag,
        "scaler_found_inf_ground_truth_inner_optimizer": scaler_ground_truth,
        "scaler_found_inf_visible_to_runtime": bool(
            any(item["scaler_found_inf"] for item in decision.rank_diagnostics)
        ),
        "measured_unscaled_pre_clip_norm": measured_unscaled_norm,
        "post_boundary_grad_norm": post_boundary_norm,
        "parameter_digest_before": pre_digest,
        "parameter_digest_after": post_digest,
        "learning_rates": {
            "pre_call": pre_lrs,
            "post_boundary_before_scheduler": post_lrs_before_scheduler,
            "post_scheduler": post_scheduler_lrs,
            "receipt": list(receipt.group_learning_rates),
        },
        "counters": {
            "optimizer_step_count_before": pre_optimizer_step_count,
            "optimizer_step_count_after": int(runtime.optimizer_step_count),
            "scheduler_step_count_before": pre_scheduler_step_count,
            "scheduler_step_count_after": int(runtime.scheduler_step_count),
            "zero_grad_count": int(runtime.zero_grad_count),
            "optimizer_state_entries_before": pre_optimizer_state_entries,
            "optimizer_state_entries_after": len(env.inner_optimizer.state),
            "scheduler_last_epoch_before": pre_scheduler_last_epoch,
            "scheduler_last_epoch_after": int(env.scheduler.last_epoch),
        },
        "scheduler_payload": {
            "scheduler_present": scheduler_payload["scheduler_present"],
            "scheduler_step_count": scheduler_payload["scheduler_step_count"],
            "scheduler_last_epoch": scheduler_payload["scheduler_last_epoch"],
        },
    }
    observed["checks"] = (
        _check_finite_arm(observed, args=args)
        if arm == "finite"
        else _check_overflow_arm(observed, args=args)
    )
    observed["failed_checks"] = sorted(
        name for name, ok in observed["checks"].items() if not ok
    )
    observed["ok"] = not observed["failed_checks"]
    return observed


def _shared_checks(observed: dict[str, Any]) -> dict[str, bool]:
    order = observed["order_log"]
    decision = observed["decision"]
    checks: dict[str, bool] = {}
    # 3.8: "exactly one accelerator.unscale_gradients(optimizer)"
    checks["unscale_gradients_called_exactly_once"] = (
        observed["unscale_call_count"] == 1
        and order.count("unscale_gradients") == 1
    )
    # 3.8: "... followed by finite/norm authority"
    checks["unscale_precedes_every_other_boundary_event"] = bool(
        order and order[0] == "unscale_gradients"
    )
    checks["runtime_observed_unscale_completed_on_every_rank"] = all(
        item["unscale_completed"] for item in decision["rank_diagnostics"]
    )
    checks["runtime_observed_active_scaler_on_every_rank"] = all(
        item["scaler_active"] for item in decision["rank_diagnostics"]
    )
    # 3.8: "with no accelerator.clip_grad_norm_"
    checks["zero_accelerator_clip_grad_norm_calls"] = (
        observed["accelerator_clip_call_count"] == 0
        and "accelerator_clip_grad_norm_" not in order
    )
    checks["no_terminal_outcome"] = (
        decision["terminal_reason"] is None
        and observed["receipt"]["terminal"] is False
    )
    return checks


def _check_finite_arm(observed: dict[str, Any], *, args: argparse.Namespace) -> dict[str, bool]:
    checks = _shared_checks(observed)
    decision = observed["decision"]
    receipt = observed["receipt"]
    order = observed["order_log"]
    lrs = observed["learning_rates"]
    counters = observed["counters"]
    measured = observed["measured_unscaled_pre_clip_norm"]
    declared = decision["pre_clip_grad_norm_rank_max"]

    checks["converged_action_is_apply"] = (
        decision["optimizer_boundary_action"] == "apply"
        and decision["optimizer_update_status"] == "ready_to_step"
        and decision["all_ranks_safe"] is True
    )
    # Norm authority is over the ALREADY-UNSCALED gradients, not the scaled ones.
    checks["norm_authority_is_over_unscaled_gradients"] = bool(
        measured is not None
        and declared is not None
        and declared >= measured - NORM_ABS_TOLERANCE
        and (
            _close(declared, measured)
            or declared > measured  # rank-max over an asymmetric peer
        )
        and not _close(
            declared, measured * observed["backward"]["grad_scale_before_wrapper"]
        )
    )
    # 3.8 finite arm: "one non-unscaling clip"
    checks["exactly_one_non_unscaling_clip_primitive"] = (
        observed["clip_primitive_call_count"] == 1
        and order.count("clip_primitive") == 1
        and observed["accelerator_events"] == ["clip"]
    )
    checks["clip_ordered_after_unscale_and_before_the_wrapper"] = (
        "clip_primitive" in order
        and "wrapper_step" in order
        and order.index("unscale_gradients")
        < order.index("clip_primitive")
        < order.index("wrapper_step")
        if observed["environment"]["kind"] == "cuda_accelerate"
        else (
            "clip_primitive" in order
            and order.index("unscale_gradients") < order.index("clip_primitive")
        )
    )
    checks["clip_actually_bounded_the_gradient_norm"] = bool(
        measured is not None
        and measured > args.max_grad_norm
        and observed["post_boundary_grad_norm"] is not None
        and observed["post_boundary_grad_norm"]
        <= args.max_grad_norm * (1.0 + 1e-3) + 1e-9
    )
    checks["clip_primitive_used_the_configured_max_norm"] = observed[
        "clip_primitive_max_norms"
    ] == [float(args.max_grad_norm)]
    # 3.8 finite arm: "an applied update"
    checks["update_actually_applied_parameters_changed"] = (
        observed["parameter_digest_before"] != observed["parameter_digest_after"]
    )
    checks["underlying_optimizer_actually_stepped"] = (
        observed["underlying_applied_calls"] == 1
        and counters["optimizer_state_entries_after"]
        > counters["optimizer_state_entries_before"]
    )
    checks["exactly_one_wrapper_invocation"] = (
        observed["wrapper_step_calls"] == 1
        and counters["optimizer_step_count_after"]
        == counters["optimizer_step_count_before"] + 1
    )
    checks["gradscaler_did_not_skip"] = (
        observed["optimizer_step_was_skipped_after_wrapper"] is False
    )
    # 3.8 finite arm: "and pre-call LR"
    checks["applied_lr_is_the_pre_call_group_value"] = (
        list(lrs["receipt"]) == list(lrs["pre_call"])
        and lrs["pre_call"] == lrs["post_boundary_before_scheduler"]
    )
    checks["applied_lr_is_not_the_post_scheduler_value"] = (
        list(lrs["receipt"]) != list(lrs["post_scheduler"])
    )
    # post-wrapper consensus
    checks["post_wrapper_consensus_none_skipped"] = (
        receipt["post_wrapper_outcome"] == "none_skipped"
    )
    checks["receipt_reports_a_truthful_applied_update"] = (
        receipt["optimizer_boundary_action"] == "apply"
        and receipt["attempted"] is True
        and receipt["applied"] is True
        and receipt["step_was_skipped"] is False
        and receipt["mutation_state"] == "applied"
        and receipt["optimizer_update_status"] == "applied"
        and receipt["planned_step_id"] == observed["planned_step_id"]
    )
    checks["scheduler_advanced_once"] = (
        counters["scheduler_step_count_after"]
        == counters["scheduler_step_count_before"] + 1
        and counters["scheduler_last_epoch_after"]
        == counters["scheduler_last_epoch_before"] + 1
    )
    return checks


def _check_overflow_arm(observed: dict[str, Any], *, args: argparse.Namespace) -> dict[str, bool]:
    checks = _shared_checks(observed)
    decision = observed["decision"]
    receipt = observed["receipt"]
    order = observed["order_log"]
    lrs = observed["learning_rates"]
    counters = observed["counters"]

    # 3.8 overflow arm: "an all-rank-confirmed scaler_skip"
    checks["all_rank_confirmed_scaler_skip"] = (
        decision["optimizer_boundary_action"] == "scaler_skip"
        and decision["optimizer_update_status"] == "ready_to_scaler_skip"
        and decision["terminal_reason"] is None
        and list(decision["ranks"]) == list(range(decision["world_size"]))
        and all(item["scaler_active"] for item in decision["rank_diagnostics"])
        and all(
            (not item["gradients_finite"])
            or item["scaler_found_inf"]
            or item["backend_overflow"]
            or (not item["grad_norm_finite"])
            for item in decision["rank_diagnostics"]
        )
    )
    checks["overflow_was_genuinely_produced"] = all(
        (not item["gradients_finite"]) or item["scaler_found_inf"]
        for item in decision["rank_diagnostics"]
    )
    # 3.8 overflow arm: "zero clip calls"
    checks["zero_clip_calls_of_any_kind"] = (
        observed["clip_primitive_call_count"] == 0
        and observed["accelerator_clip_call_count"] == 0
        and observed["accelerator_events"] == []
        and "clip_primitive" not in order
    )
    # 3.8 overflow arm: "one wrapper call solely for scaler finalization"
    checks["exactly_one_wrapper_call_for_scaler_finalization"] = (
        observed["wrapper_step_calls"] == 1
    )
    # 3.8 overflow arm: "post-call skipped truth"
    checks["gradscaler_actually_skipped"] = (
        observed["optimizer_step_was_skipped_after_wrapper"] is True
    )
    checks["parameters_byte_unchanged"] = (
        observed["parameter_digest_before"] == observed["parameter_digest_after"]
    )
    # 3.8 overflow arm: "no underlying optimizer mutation"
    checks["no_underlying_optimizer_mutation"] = (
        observed["underlying_applied_calls"] == 0
        and counters["optimizer_state_entries_after"]
        == counters["optimizer_state_entries_before"]
        == 0
        and lrs["post_boundary_before_scheduler"] == lrs["pre_call"]
    )
    # 3.8 overflow arm: "null applied LR"
    checks["null_applied_learning_rates_in_the_receipt"] = (
        len(lrs["receipt"]) == len(lrs["pre_call"])
        and all(value is None for value in lrs["receipt"])
    )
    # 3.8 overflow arm: "one completed-wrapper counter increment"
    checks["one_completed_wrapper_counter_increment"] = (
        counters["optimizer_step_count_after"]
        == counters["optimizer_step_count_before"] + 1
    )
    # 3.8 overflow arm: "planned scheduler progression"
    checks["planned_scheduler_progression"] = (
        counters["scheduler_step_count_after"]
        == counters["scheduler_step_count_before"] + 1
        and counters["scheduler_last_epoch_after"]
        == counters["scheduler_last_epoch_before"] + 1
        and lrs["post_scheduler"] != lrs["pre_call"]
    )
    # 3.8 overflow arm: "no false applied status"
    checks["no_false_applied_status"] = (
        receipt["optimizer_boundary_action"] == "scaler_skip"
        and receipt["attempted"] is True
        and receipt["applied"] is False
        and receipt["step_was_skipped"] is True
        and receipt["mutation_state"] == "scaler_suppressed"
        and receipt["optimizer_update_status"] == "skipped_scaler_overflow"
        and receipt["post_wrapper_outcome"] == "all_skipped"
        and receipt["planned_step_id"] == observed["planned_step_id"]
    )
    del args
    return checks


# ---------------------------------------------------------------------------
# entry
# ---------------------------------------------------------------------------


def _install_clip_primitive_spy() -> Callable[[], None]:
    original = torch.nn.utils.clip_grad_norm_

    def spy(parameters: Any, max_norm: Any, *rest: Any, **kwargs: Any) -> Any:
        _ORDER_LOG.append("clip_primitive")
        _CLIP_PRIMITIVE_CALLS.append(float(max_norm))
        return original(parameters, max_norm, *rest, **kwargs)

    torch.nn.utils.clip_grad_norm_ = spy  # type: ignore[assignment]

    def restore() -> None:
        torch.nn.utils.clip_grad_norm_ = original  # type: ignore[assignment]

    return restore


def _distributed_identity(args: argparse.Namespace) -> tuple[int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if args.two_rank and world_size != 2:
        raise RuntimeError(
            "--two-rank requires a torch.distributed.run launch with "
            f"--nproc_per_node=2; observed WORLD_SIZE={world_size}"
        )
    if not args.two_rank and world_size != 1:
        raise RuntimeError(
            "the single-rank form must run outside a distributed launcher; "
            f"observed WORLD_SIZE={world_size}"
        )
    return rank, world_size


def _build_environment(
    *, arm: str, args: argparse.Namespace, rank: int, world_size: int, gatherer: Any
) -> Any:
    if args.dry_structure:
        return _DryStructureEnvironment(
            args=args,
            rank=rank,
            world_size=world_size,
            gatherer=gatherer,
            poison=arm == "overflow",
        )
    return _CudaAccelerateEnvironment(
        args=args, rank=rank, world_size=world_size, gatherer=gatherer
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="strict-JSON receipt path; MUST NOT already exist",
    )
    parser.add_argument(
        "--two-rank",
        action="store_true",
        help="assert a two-process torch.distributed.run launch (WORLD_SIZE=2)",
    )
    parser.add_argument(
        "--dry-structure",
        action="store_true",
        help=(
            "CPU plumbing validation of the assertion structure. NOT fp16 "
            "evidence; the receipt is stamped evidence_class=plumbing_only."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.125)
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1e-3,
        help="deliberately small so the clip visibly bounds the gradient norm",
    )
    parser.add_argument(
        "--overflow-loss-multiplier",
        type=float,
        default=1e35,
        help=(
            "poisons the loss so the SCALED backward overflows fp32 and the "
            "real GradScaler records found_inf"
        ),
    )
    args = parser.parse_args()

    output_path: Path = args.output
    if output_path.exists():
        print(f"[result] FAILED: --output already exists: {output_path}")
        return 1
    if args.two_rank and args.dry_structure:
        print("[result] FAILED: --two-rank and --dry-structure are exclusive")
        return 1

    started = time.monotonic()
    findings: list[str] = []
    arms_receipt: dict[str, Any] = {}
    rank = 0
    world_size = 1
    gatherer = None
    restore_spy = _install_clip_primitive_spy()
    try:
        rank, world_size = _distributed_identity(args)
        if not args.dry_structure:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "the fp16 arms require a real CUDA device; refusing to "
                    "emit a receipt that could be mistaken for fp16 evidence"
                )
            if world_size > 1:
                # The first Accelerator initializes the process group; build
                # the production gatherer from it.
                from accelerate import Accelerator

                Accelerator(mixed_precision="fp16")
                from src.training.control_plane import _build_rank_report_gatherer

                gatherer = _build_rank_report_gatherer(world_size)
                if gatherer is None:
                    raise RuntimeError("rank report gatherer was not constructed")

        for arm in ARMS:
            env = _build_environment(
                arm=arm,
                args=args,
                rank=rank,
                world_size=world_size,
                gatherer=gatherer,
            )
            observed = run_arm(arm=arm, env=env, args=args)
            arms_receipt[arm] = observed
            if not observed["ok"]:
                findings.append(f"{arm}: failed checks {observed['failed_checks']}")
            if not args.dry_structure:
                inner = observed["scaler_found_inf_ground_truth_inner_optimizer"]
                visible = observed["scaler_found_inf_visible_to_runtime"]
                if inner is True and visible is False:
                    findings.append(
                        "FINDING(non-fatal, recorded for the lead): the runtime's "
                        "`scaler_found_inf` reported False while the real "
                        "GradScaler's found_inf record for the INNER optimizer "
                        "was True. `Accelerator.unscale_gradients` unwraps "
                        "`AcceleratedOptimizer`, so `_scaler_found_inf(scaler, "
                        "self.optimizer)` in src/runtime/train_runtime.py looks "
                        "up a defaultdict miss. The converged action was still "
                        "correct because the gradient scan is independent "
                        f"evidence (arm={arm})."
                    )
            del env
            if torch.cuda.is_available() and not args.dry_structure:
                torch.cuda.synchronize()

        if world_size > 1 and gatherer is not None:
            compact = {
                "kind": "obs_wave3_fp16_arm_summary",
                "planned_step_id": 0,
                "rank": rank,
                "world_size": world_size,
                "arms": {
                    arm: {
                        "receipt": arms_receipt[arm]["receipt"],
                        "checks": arms_receipt[arm]["checks"],
                        "counters": arms_receipt[arm]["counters"],
                        "parameter_digest_before": arms_receipt[arm][
                            "parameter_digest_before"
                        ],
                        "parameter_digest_after": arms_receipt[arm][
                            "parameter_digest_after"
                        ],
                        "measured_unscaled_pre_clip_norm": arms_receipt[arm][
                            "measured_unscaled_pre_clip_norm"
                        ],
                    }
                    for arm in ARMS
                },
            }
            gathered = gatherer(compact)
            for arm in ARMS:
                receipts = [report["arms"][arm]["receipt"] for report in gathered]
                if any(item != receipts[0] for item in receipts):
                    findings.append(f"{arm}: update receipts diverged across ranks")
                digests = [
                    report["arms"][arm]["parameter_digest_after"]
                    for report in gathered
                ]
                if any(item != digests[0] for item in digests):
                    findings.append(
                        f"{arm}: post-boundary parameter digests diverged across ranks"
                    )
                for report in gathered:
                    failed = sorted(
                        name
                        for name, ok in report["arms"][arm]["checks"].items()
                        if not ok
                    )
                    if failed:
                        findings.append(
                            f"{arm}: rank {report['rank']} failed checks {failed}"
                        )
            arms_receipt = {
                arm: {
                    **arms_receipt[arm],
                    "per_rank_summary": {
                        str(report["rank"]): report["arms"][arm]
                        for report in gathered
                    },
                }
                for arm in ARMS
            }
    except BaseException:
        findings.append("probe raised:\n" + traceback.format_exc())
    finally:
        restore_spy()
        close = getattr(gatherer, "close", None)
        if callable(close):
            try:
                close()
            except BaseException:
                findings.append("gatherer close raised:\n" + traceback.format_exc())

    elapsed = time.monotonic() - started
    cuda_used = bool(not args.dry_structure and torch.cuda.is_available())
    bounds: dict[str, Any] = {
        "declared_world_size": world_size,
        "declared_device_count": world_size if cuda_used else 0,
        "arms": len(ARMS),
        "planned_steps_total": len(ARMS),
        "model_forwards_total": len(ARMS),
        "cache_or_materialization_passes": 0,
        "wall_clock_seconds": round(elapsed, 3),
        "peak_cuda_allocated_bytes": (
            int(torch.cuda.max_memory_allocated()) if cuda_used else None
        ),
        "peak_cuda_reserved_bytes": (
            int(torch.cuda.max_memory_reserved()) if cuda_used else None
        ),
        "artifact_paths": [str(output_path)],
        # The exact on-disk size is printed after the write as
        # `[bounds] receipt_bytes=<n>`; it cannot be embedded without making
        # the field describe a document other than the one written.
        "artifact_bytes_measurement": "printed_after_write",
    }

    receipt: dict[str, Any] = {
        "schema": PROBE_SCHEMA,
        "change": "add-coordexp-swift-training-observability",
        "task": "3.8",
        "mode": (
            "dry_structure"
            if args.dry_structure
            else ("two_rank_cuda_fp16" if args.two_rank else "single_rank_cuda_fp16")
        ),
        "evidence_class": (
            "plumbing_only" if args.dry_structure else "genuine_cuda_fp16_accelerate"
        ),
        "cuda_used": cuda_used,
        "claims_fp16_evidence": bool(not args.dry_structure),
        "rank": rank,
        "world_size": world_size,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_names": (
            [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ]
            if torch.cuda.is_available()
            else []
        ),
        "arguments": {
            "seed": args.seed,
            "width": args.width,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "max_grad_norm": args.max_grad_norm,
            "overflow_loss_multiplier": args.overflow_loss_multiplier,
        },
        "production_seams_driven": [
            "src.runtime.train_runtime.TrainRuntime.backward",
            "src.runtime.train_runtime.TrainRuntime.post_backward",
            "src.runtime.train_runtime.TrainRuntime.execute_optimizer_boundary",
            "src.runtime.train_runtime.TrainRuntime.scheduler_step",
            "src.training.control_plane._build_rank_report_gatherer",
        ],
        "instrumentation": {
            "production_seam_used": (
                "accelerator.events (TrainRuntime._record_boundary_event)"
            ),
            "instance_attribute_spies": [
                "accelerator.unscale_gradients",
                "accelerator.clip_grad_norm_",
                "AcceleratedOptimizer.step (prepared wrapper)",
            ],
            "module_level_spies": ["torch.nn.utils.clip_grad_norm_"],
            "module_level_spy_rationale": (
                "the only way to observe the identity and call count of the "
                "non-unscaling clip primitive from outside src/; delegating, "
                "restored in a finally, and corroborated by the physical "
                "post-boundary norm bound"
            ),
            "monkeypatched_production_modules": [],
        },
        "observed_bounds": bounds,
        "arms": arms_receipt,
        "findings": findings,
        "ok": not [item for item in findings if not item.startswith("FINDING(")],
    }
    if args.dry_structure:
        receipt["dry_structure_emulation_notes"] = [
            "the fp16 surface is duck-typed on CPU; no real GradScaler runs",
            "the overflow arm writes non-finite gradients directly instead of "
            "overflowing a scaled backward",
            "the post-wrapper skip flag is set by the probe, not by Accelerate",
            "the underlying update is suppressed by a probe optimizer, not by "
            "GradScaler",
        ]

    text = json.dumps(receipt, sort_keys=True, indent=2, default=str)
    if rank == 0:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
        print(text)
        print(f"[bounds] receipt_bytes={output_path.stat().st_size}")
        print(f"[result] {'PASSED' if receipt['ok'] else 'FAILED'}: {output_path}")
    else:
        print(
            f"[rank {rank}] {'PASSED' if receipt['ok'] else 'FAILED'} "
            f"({len(findings)} findings)"
        )
        if findings:
            print(json.dumps(findings, indent=2, default=str))
    return 0 if receipt["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
