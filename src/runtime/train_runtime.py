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


def validate_accelerator_runtime(
    accelerator: Any,
    *,
    expected_mixed_precision: str,
) -> None:
    """Reject unsupported launcher state before training-side mutation."""

    distributed_type = getattr(accelerator, "distributed_type", None)
    distributed_name = getattr(distributed_type, "name", str(distributed_type))
    if distributed_name not in {"NO", "MULTI_GPU"}:
        raise RuntimeContractError(
            f"unsupported Accelerate distributed type: {distributed_name}",
            code="runtime.distributed_type_unsupported",
            context={"distributed_type": distributed_name},
        )
    expected_precision = _normalize_mixed_precision(expected_mixed_precision)
    observed_precision = _normalize_mixed_precision(
        getattr(accelerator, "mixed_precision", None)
    )
    if observed_precision != expected_precision:
        raise RuntimeContractError(
            "constructed Accelerator mixed precision does not match resolved training precision",
            code="runtime.mixed_precision_mismatch",
            context={
                "expected_mixed_precision": expected_precision,
                "observed_mixed_precision": observed_precision,
            },
        )
    world_size = int(accelerator.num_processes)
    rank = int(accelerator.process_index)
    if world_size <= 0:
        raise RuntimeContractError(
            "runtime world_size must be positive",
            code="runtime.world_size",
            context={"world_size": world_size},
        )
    if rank < 0 or rank >= world_size:
        raise RuntimeContractError(
            "runtime rank must be inside world size",
            code="runtime.rank",
            context={"rank": rank, "world_size": world_size},
        )
    accelerator_accumulation = int(
        getattr(accelerator, "gradient_accumulation_steps", 1)
    )
    if accelerator_accumulation != 1:
        raise RuntimeContractError(
            "Accelerate accumulation must remain neutral; CoordExp owns accumulation",
            code="runtime.accelerator_accumulation_non_neutral",
            context={
                "accelerator_gradient_accumulation_steps": accelerator_accumulation,
            },
        )


@dataclass(frozen=True)
class TrainRuntimeSetupReceipt:
    rank: int
    world_size: int
    device: str
    seed: int
    runtime_batch: RuntimeBatchResolution
    max_grad_norm: float | None
    scheduler_semantics: Mapping[str, Any]

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "device": self.device,
            "seed": self.seed,
            "runtime_batch": self.runtime_batch.to_artifact_dict(),
            "max_grad_norm": self.max_grad_norm,
            "scheduler_semantics": dict(self.scheduler_semantics),
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
        expected_mixed_precision: str,
        max_grad_norm: float | None = None,
        accelerator: Any,
        rank_report_gatherer: Callable[
            [Any],
            Sequence[Any],
        ]
        | None = None,
    ) -> None:
        self.runtime_config = runtime_config
        self.runtime_batch = runtime_batch
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.expected_mixed_precision = expected_mixed_precision
        self.accelerator = accelerator
        self.device = torch.device(accelerator.device)
        self.rank = int(accelerator.process_index)
        self.world_size = int(accelerator.num_processes)
        self.max_grad_norm = max_grad_norm
        self.rank_report_gatherer = rank_report_gatherer
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
        self._prepare_training_objects()
        self.setup_receipt = TrainRuntimeSetupReceipt(
            rank=self.rank,
            world_size=self.world_size,
            device=str(self.device),
            seed=runtime_config.seed,
            runtime_batch=runtime_batch,
            max_grad_norm=max_grad_norm,
            scheduler_semantics=_scheduler_semantics(),
        )

    @property
    def is_main_process(self) -> bool:
        return bool(self.accelerator.is_main_process)

    def move_micro_step(
        self,
        micro_step: SupervisedMicroStep,
        *,
        planned_step_id: int,
        local_micro_step_index: int,
    ) -> SupervisedMicroStep:
        del planned_step_id, local_micro_step_index
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
        if sync_gradients:
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
        self.accelerator.backward(loss)

    def post_backward(self, *, planned_step_id: int) -> GateDecision:
        report = build_gradient_finite_report(
            self.model.parameters(),
            planned_step_id=planned_step_id,
            rank=self.rank,
            world_size=self.world_size,
            backend_overflow=_accelerator_overflow(self.accelerator),
        )
        return reduce_gradient_overflow_reports(self._gather_rank_reports(report))

    def clip_gradients(self, *, planned_step_id: int) -> None:
        del planned_step_id
        if self.max_grad_norm is None:
            return
        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

    def optimizer_step(self, *, planned_step_id: int) -> None:
        del planned_step_id
        if self.optimizer is None:
            raise RuntimeContractError(
                "optimizer_step requires an optimizer",
                code="runtime.optimizer_missing",
            )
        self.optimizer.step()
        self.optimizer_step_count += 1

    def scheduler_step(self, *, planned_step_id: int) -> dict[str, Any]:
        if self.scheduler is None:
            return {
                "planned_step_id": int(planned_step_id),
                "scheduler_present": False,
                "scheduler_step_count": self.scheduler_step_count,
                "scheduler_last_epoch": None,
                "learning_rates": [],
                "semantics": _scheduler_semantics(),
            }
        self.scheduler.step()
        self.scheduler_step_count += 1
        return {
            "planned_step_id": int(planned_step_id),
            "scheduler_present": True,
            "scheduler_step_count": self.scheduler_step_count,
            "scheduler_last_epoch": _scheduler_last_epoch(self.scheduler),
            "learning_rates": _scheduler_learning_rates(
                self.scheduler,
                optimizer=self.optimizer,
            ),
            "semantics": _scheduler_semantics(),
        }

    def zero_gradients(self, *, planned_step_id: int) -> None:
        del planned_step_id
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

    def gather_loss_denominators(
        self,
        denominators: Mapping[str, Mapping[str, Any]],
        *,
        planned_step_id: int,
    ) -> tuple[Mapping[str, Mapping[str, Any]], ...]:
        payload = {
            "kind": "loss_denominators",
            "planned_step_id": int(planned_step_id),
            "rank": self.rank,
            "world_size": self.world_size,
            "denominators": {
                str(name): dict(value) for name, value in denominators.items()
            },
        }
        if self.world_size == 1:
            return (payload["denominators"],)
        reports = self._gather_rank_reports(payload)
        gathered: list[Mapping[str, Mapping[str, Any]]] = []
        for rank_index, report in enumerate(reports):
            if not isinstance(report, Mapping):
                raise RuntimeContractError(
                    "loss denominator gatherer must return mapping payloads",
                    code="runtime.loss_denominator_gather_invalid",
                    context={
                        "rank": self.rank,
                        "world_size": self.world_size,
                        "rank_index": rank_index,
                        "value_type": type(report).__name__,
                    },
                )
            denominators_payload = report.get("denominators", report)
            if not isinstance(denominators_payload, Mapping):
                raise RuntimeContractError(
                    "loss denominator gatherer payload must include denominators",
                    code="runtime.loss_denominator_gather_invalid",
                    context={
                        "rank": self.rank,
                        "world_size": self.world_size,
                        "rank_index": rank_index,
                        "value_type": type(denominators_payload).__name__,
                    },
                )
            gathered.append(
                {
                    str(name): dict(value)
                    for name, value in denominators_payload.items()
                    if isinstance(value, Mapping)
                }
            )
        return tuple(gathered)

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
        validate_accelerator_runtime(
            self.accelerator,
            expected_mixed_precision=self.expected_mixed_precision,
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

    def _prepare_training_objects(self) -> None:
        prepare = getattr(self.accelerator, "prepare", None)
        if not callable(prepare):
            raise RuntimeContractError(
                "TrainRuntime requires accelerator.prepare",
                code="runtime.accelerator_prepare_missing",
            )
        names: list[str] = ["model"]
        objects: list[Any] = [self.model]
        if self.optimizer is not None:
            names.append("optimizer")
            objects.append(self.optimizer)
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

    def _gather_rank_reports(
        self,
        local_report: Any,
    ) -> tuple[Any, ...]:
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

def _accelerator_overflow(accelerator: Any) -> bool:
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


def _normalize_mixed_precision(value: Any) -> str:
    if value is None:
        return "no"
    normalized = str(getattr(value, "value", value)).strip().lower()
    if normalized in {"", "none", "null", "false", "no"}:
        return "no"
    return normalized


def _scheduler_semantics() -> dict[str, Any]:
    return {
        "scheduler_owner": "coordexp_runtime",
        "scheduler_prepared_by_accelerate": False,
        "step_policy": "once_per_planned_step",
    }


def _scheduler_last_epoch(scheduler: Any) -> int | None:
    last_epoch = getattr(scheduler, "last_epoch", None)
    if last_epoch is None:
        return None
    try:
        return int(last_epoch)
    except (TypeError, ValueError):
        return None


def _scheduler_learning_rates(
    scheduler: Any,
    *,
    optimizer: torch.optim.Optimizer | None,
) -> list[dict[str, float | int]]:
    get_last_lr = getattr(scheduler, "get_last_lr", None)
    if callable(get_last_lr):
        values: Sequence[Any] = get_last_lr()
    elif optimizer is not None:
        values = [group.get("lr") for group in optimizer.param_groups]
    else:
        values = ()
    return [
        {
            "group_index": index,
            "lr": _float_scalar(value),
        }
        for index, value in enumerate(values)
    ]


def _float_scalar(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(torch.as_tensor(value).detach().cpu().item())


__all__ = [
    "TrainRuntime",
    "TrainRuntimeSetupReceipt",
    "validate_accelerator_runtime",
]
