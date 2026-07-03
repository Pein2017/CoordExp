"""Concrete training runtime boundary for V1 supervised training."""

from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

import torch

from src.common.errors import RuntimeContractError
from src.config.models import RuntimeBatchResolution, RuntimeConfig
from src.runtime.finite_gates import (
    GateDecision,
    RankGradientFiniteReport,
    RankScalarFiniteReport,
    build_gradient_finite_report,
    reduce_gradient_overflow_reports,
    reduce_scalar_finite_reports,
)
from src.runtime.seeding import seed_training_runtime

if TYPE_CHECKING:
    from src.training.supervised_trainer import SupervisedMicroStep


@dataclass(frozen=True)
class TrainRuntimeSetupReceipt:
    backend: str
    rank: int
    world_size: int
    device: str
    seed: int
    runtime_batch: RuntimeBatchResolution
    backend_status: dict[str, tuple[str, ...]]
    max_grad_norm: float | None

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "rank": self.rank,
            "world_size": self.world_size,
            "device": self.device,
            "seed": self.seed,
            "runtime_batch": self.runtime_batch.to_artifact_dict(),
            "backend_status": {
                backend: list(labels)
                for backend, labels in sorted(self.backend_status.items())
            },
            "max_grad_norm": self.max_grad_norm,
        }


class TrainRuntime:
    def __init__(
        self,
        *,
        runtime_config: RuntimeConfig,
        runtime_batch: RuntimeBatchResolution,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None,
        scheduler: Any | None,
        device: torch.device | str,
        rank: int,
        world_size: int,
        max_grad_norm: float | None = None,
        accelerator: Any | None = None,
        rank_report_gatherer: Callable[
            [RankScalarFiniteReport | RankGradientFiniteReport],
            Sequence[RankScalarFiniteReport | RankGradientFiniteReport],
        ]
        | None = None,
    ) -> None:
        self.runtime_config = runtime_config
        self.runtime_batch = runtime_batch
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device(device)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.max_grad_norm = max_grad_norm
        self.accelerator = accelerator
        self.rank_report_gatherer = rank_report_gatherer
        self._accelerate_prepared = False
        self.optimizer_step_count = 0
        self.scheduler_step_count = 0
        self.zero_grad_count = 0
        self._validate_runtime_contract()
        seed_training_runtime(
            runtime_config.seed,
            deterministic=False,
            phase="runtime_setup_reapplied",
        )
        self.model.to(self.device)
        self._prepare_backend()
        self.setup_receipt = TrainRuntimeSetupReceipt(
            backend=runtime_config.backend,
            rank=self.rank,
            world_size=self.world_size,
            device=str(self.device),
            seed=runtime_config.seed,
            runtime_batch=runtime_batch,
            backend_status=_backend_status(
                runtime_config,
                accelerate_prepared=self._accelerate_prepared,
            ),
            max_grad_norm=max_grad_norm,
        )

    @property
    def is_main_process(self) -> bool:
        if self.accelerator is not None and hasattr(self.accelerator, "is_main_process"):
            return bool(self.accelerator.is_main_process)
        return self.rank == 0

    def move_micro_step(
        self,
        micro_step: SupervisedMicroStep,
        *,
        planned_step_id: int,
        local_micro_step_index: int,
    ) -> SupervisedMicroStep:
        del planned_step_id, local_micro_step_index
        self._ensure_training_backend_can_execute()
        return replace(micro_step, forward_device=self.device)

    def pre_backward(
        self,
        bundle: Any,
        *,
        planned_step_id: int,
    ) -> GateDecision:
        report = RankScalarFiniteReport.from_loss_bundle(
            bundle,
            planned_step_id=planned_step_id,
            rank=self.rank,
            world_size=self.world_size,
        )
        return reduce_scalar_finite_reports(self._gather_rank_reports(report))

    def accumulation_context(self, *, sync_gradients: bool) -> Any:
        self._ensure_training_backend_can_execute()
        if sync_gradients or self.accelerator is None:
            return nullcontext()
        if self.runtime_config.backend == "deepspeed":
            return nullcontext()
        no_sync = getattr(self.accelerator, "no_sync", None)
        if callable(no_sync):
            return no_sync(self.model)
        return nullcontext()

    def backward(
        self,
        loss: torch.Tensor,
        *,
        planned_step_id: int,
        sync_gradients: bool = True,
    ) -> None:
        del planned_step_id
        self._ensure_training_backend_can_execute()
        if self.accelerator is not None and hasattr(self.accelerator, "backward"):
            if self.runtime_config.backend == "deepspeed":
                old_sync = getattr(self.accelerator, "sync_gradients", None)
                has_sync_state = hasattr(self.accelerator, "sync_gradients")
                if has_sync_state:
                    setattr(self.accelerator, "sync_gradients", bool(sync_gradients))
                try:
                    self.accelerator.backward(loss, scale_wrt_gas=False)
                finally:
                    if has_sync_state:
                        setattr(self.accelerator, "sync_gradients", old_sync)
                return
            self.accelerator.backward(loss)
        else:
            loss.backward()

    def post_backward(self, *, planned_step_id: int) -> GateDecision:
        self._ensure_training_backend_can_execute()
        if self.runtime_config.backend == "deepspeed":
            report = RankGradientFiniteReport(
                planned_step_id=planned_step_id,
                rank=self.rank,
                world_size=self.world_size,
                gradients_finite=True,
                backend_overflow=_backend_overflow(self.accelerator),
                grad_norm=_deepspeed_global_grad_norm(self.accelerator),
            )
        else:
            report = build_gradient_finite_report(
                self.model.parameters(),
                planned_step_id=planned_step_id,
                rank=self.rank,
                world_size=self.world_size,
                backend_overflow=_backend_overflow(self.accelerator),
            )
        return reduce_gradient_overflow_reports(self._gather_rank_reports(report))

    def clip_gradients(self, *, planned_step_id: int) -> None:
        del planned_step_id
        self._ensure_training_backend_can_execute()
        if self.max_grad_norm is None:
            return
        if self.accelerator is not None and hasattr(self.accelerator, "clip_grad_norm_"):
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

    def optimizer_step(self, *, planned_step_id: int) -> None:
        del planned_step_id
        self._ensure_training_backend_can_execute()
        if self.optimizer is None:
            raise RuntimeContractError(
                "optimizer_step requires an optimizer",
                code="runtime.optimizer_missing",
            )
        self.optimizer.step()
        self.optimizer_step_count += 1

    def scheduler_step(self, *, planned_step_id: int) -> None:
        del planned_step_id
        self._ensure_training_backend_can_execute()
        if self.scheduler is None:
            return
        self.scheduler.step()
        self.scheduler_step_count += 1

    def zero_gradients(self, *, planned_step_id: int) -> None:
        del planned_step_id
        self._ensure_training_backend_can_execute()
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)
        else:
            for parameter in self.model.parameters():
                parameter.grad = None
        self.zero_grad_count += 1

    def gather_metrics(
        self,
        metrics: Mapping[str, float],
        *,
        planned_step_id: int,
        split: str,
    ) -> dict[str, Any]:
        values = {str(key): float(metrics[key]) for key in sorted(metrics)}
        return {
            "planned_step_id": int(planned_step_id),
            "split": split,
            "rank": self.rank,
            "world_size": self.world_size,
            "metrics": values,
            "reduction": "single_rank" if self.world_size == 1 else "rank_local",
        }

    def safe_save_json(self, payload: Mapping[str, Any], path: Path) -> Path | None:
        if not self.is_main_process:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        return path

    def _validate_runtime_contract(self) -> None:
        if self.world_size <= 0:
            raise RuntimeContractError(
                "runtime world_size must be positive",
                code="runtime.world_size",
                context={"world_size": self.world_size},
            )
        if self.rank < 0 or self.rank >= self.world_size:
            raise RuntimeContractError(
                "runtime rank must be inside world size",
                code="runtime.rank",
                context={"rank": self.rank, "world_size": self.world_size},
            )
        if self.runtime_batch.world_size != self.world_size:
            raise RuntimeContractError(
                "runtime batch world size must match runtime world size",
                code="runtime.batch_world_size",
                context={
                    "runtime_batch_world_size": self.runtime_batch.world_size,
                    "world_size": self.world_size,
                },
            )
        if self.world_size > 1 and self.rank_report_gatherer is None:
            raise RuntimeContractError(
                "multi-rank finite gates require a rank report gatherer",
                code="runtime.report_gather_unavailable",
                context={"rank": self.rank, "world_size": self.world_size},
            )
        accelerate = self.runtime_config.accelerate
        if (
            self.runtime_config.backend == "accelerate"
            and accelerate is not None
            and accelerate.gradient_accumulation_steps not in (
                None,
                self.runtime_batch.resolved_grad_accum_steps,
            )
        ):
            raise RuntimeContractError(
                "accelerate accumulation conflicts with runtime-derived value",
                code="runtime.accumulation_conflict",
                context={
                    "backend": "accelerate",
                    "authored": accelerate.gradient_accumulation_steps,
                    "resolved": self.runtime_batch.resolved_grad_accum_steps,
                },
            )
        if self.runtime_config.backend == "accelerate" and self.accelerator is None:
            raise RuntimeContractError(
                "accelerate backend requires an accelerator instance",
                code="runtime.accelerator_required",
            )
        if self.runtime_config.backend == "deepspeed" and self.accelerator is None:
            raise RuntimeContractError(
                "deepspeed backend requires an accelerator instance",
                code="runtime.deepspeed_accelerator_required",
            )
        deepspeed = self.runtime_config.deepspeed
        if (
            self.runtime_config.backend == "deepspeed"
            and deepspeed is not None
            and deepspeed.gradient_accumulation_steps not in (
                None,
                self.runtime_batch.resolved_grad_accum_steps,
            )
        ):
            raise RuntimeContractError(
                "deepspeed accumulation conflicts with runtime-derived value",
                code="runtime.accumulation_conflict",
                context={
                    "backend": "deepspeed",
                    "authored": deepspeed.gradient_accumulation_steps,
                    "resolved": self.runtime_batch.resolved_grad_accum_steps,
                },
            )
        if (
            self.runtime_config.backend == "deepspeed"
            and deepspeed is not None
            and deepspeed.train_batch_size not in (
                None,
                self.runtime_batch.effective_batch_size,
            )
        ):
            raise RuntimeContractError(
                "deepspeed train batch size conflicts with effective batch size",
                code="runtime.deepspeed_batch_conflict",
                context={
                    "authored": deepspeed.train_batch_size,
                    "effective_batch_size": self.runtime_batch.effective_batch_size,
                },
            )

    def _prepare_backend(self) -> None:
        if self.runtime_config.backend not in ("accelerate", "deepspeed"):
            return
        prepare = getattr(self.accelerator, "prepare", None)
        if not callable(prepare):
            raise RuntimeContractError(
                f"{self.runtime_config.backend} backend requires accelerator.prepare",
                code="runtime.accelerator_prepare_missing",
            )
        names: list[str] = ["model"]
        objects: list[Any] = [self.model]
        if self.optimizer is not None:
            names.append("optimizer")
            objects.append(self.optimizer)
        if self.scheduler is not None:
            names.append("scheduler")
            objects.append(self.scheduler)
        prepared = prepare(*objects)
        if isinstance(prepared, tuple):
            prepared_objects = prepared
        elif isinstance(prepared, list):
            prepared_objects = tuple(prepared)
        else:
            prepared_objects = (prepared,)
        if len(prepared_objects) != len(objects):
            raise RuntimeContractError(
                "accelerator.prepare returned an unexpected number of objects",
                code="runtime.accelerator_prepare_arity",
                context={"expected": len(objects), "observed": len(prepared_objects)},
            )
        prepared_by_name = dict(zip(names, prepared_objects, strict=True))
        self.model = prepared_by_name["model"]
        if "optimizer" in prepared_by_name:
            self.optimizer = prepared_by_name["optimizer"]
        if "scheduler" in prepared_by_name:
            self.scheduler = prepared_by_name["scheduler"]
        self._accelerate_prepared = True

    def _gather_rank_reports(
        self,
        local_report: RankScalarFiniteReport | RankGradientFiniteReport,
    ) -> tuple[RankScalarFiniteReport | RankGradientFiniteReport, ...]:
        if self.world_size == 1:
            return (local_report,)
        if self.rank_report_gatherer is None:
            raise RuntimeContractError(
                "multi-rank finite gates require a rank report gatherer",
                code="runtime.report_gather_unavailable",
                context={"rank": self.rank, "world_size": self.world_size},
            )
        try:
            reports = tuple(self.rank_report_gatherer(local_report))
        except TypeError as exc:
            raise RuntimeContractError(
                "rank report gatherer must return an iterable of rank reports",
                code="runtime.report_gather_invalid",
                context={"rank": self.rank, "world_size": self.world_size},
            ) from exc
        return reports

    def _ensure_training_backend_can_execute(self) -> None:
        if self.runtime_config.backend in ("accelerate", "deepspeed") and not self._accelerate_prepared:
            raise RuntimeContractError(
                f"{self.runtime_config.backend} execution requires prepared runtime objects",
                code="runtime.accelerator_unprepared",
            )


def _backend_status(
    runtime_config: RuntimeConfig,
    *,
    accelerate_prepared: bool = False,
) -> dict[str, tuple[str, ...]]:
    if runtime_config.backend == "single":
        return {
            "single": ("active",),
            "accelerate": (),
            "deepspeed": (),
        }
    if runtime_config.backend == "accelerate":
        return {
            "single": (),
            "accelerate": (
                "schema_accepted",
                "prepared",
                "active",
            )
            if accelerate_prepared
            else ("schema_accepted",),
            "deepspeed": (),
        }
    if runtime_config.backend == "deepspeed":
        return {
            "single": (),
            "accelerate": (),
            "deepspeed": (
                "schema_accepted",
                "conflict_validation_implemented",
                "prepared",
                "active",
            )
            if accelerate_prepared
            else (
                "schema_accepted",
                "conflict_validation_implemented",
            ),
        }
    raise RuntimeContractError(
        "unsupported runtime backend",
        code="runtime.backend_unsupported",
        context={"backend": runtime_config.backend},
    )


def _backend_overflow(accelerator: Any | None) -> bool:
    if accelerator is None:
        return False
    scaler = getattr(accelerator, "scaler", None)
    if scaler is None:
        return False
    found_inf = getattr(scaler, "_found_inf_per_device", None)
    if callable(found_inf):
        try:
            values = found_inf({})
        except TypeError:
            values = found_inf()
        if isinstance(values, Mapping):
            return any(bool(torch.as_tensor(value).item()) for value in values.values())
    return False


def _deepspeed_global_grad_norm(accelerator: Any | None) -> float | None:
    if accelerator is None:
        return None
    engine_wrapper = getattr(accelerator, "deepspeed_engine_wrapped", None)
    get_global_grad_norm = getattr(engine_wrapper, "get_global_grad_norm", None)
    if not callable(get_global_grad_norm):
        return None
    grad_norm = get_global_grad_norm()
    if grad_norm is None:
        return None
    try:
        return float(torch.as_tensor(grad_norm).detach().cpu().item())
    except (TypeError, ValueError, RuntimeError):
        return None


__all__ = ["TrainRuntime", "TrainRuntimeSetupReceipt"]
