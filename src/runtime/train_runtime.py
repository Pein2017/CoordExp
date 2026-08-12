"""Concrete training runtime boundary for V1 supervised training."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
import math
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

import torch

from src.common.errors import RuntimeContractError
from src.config.models import RuntimeBatchResolution, RuntimeConfig
from src.runtime.finite_gates import (
    GateDecision,
    RankScalarFiniteReport,
    build_gradient_finite_report,
    reduce_gradient_overflow_reports,
    reduce_scalar_finite_reports,
)
from src.runtime.seeding import seed_training_runtime

if TYPE_CHECKING:
    from src.training.supervised_trainer import SupervisedMicroStep

# Reducer semantics per key are a fixed declared mapping, not a framework:
# plain mean is the default; these keys override it.
_ACCURACY_METRIC_CORRECT_FIELDS: dict[str, str] = {
    "acc_top1": "top1_correct",
    "acc_top5": "top5_correct",
}
_ACCURACY_ATOM_COUNT_FIELD = "atom_count"
_MAX_REDUCED_METRIC_KEYS = frozenset(
    {
        "input_build_seconds",
        "input_wait_seconds",
        "eval_duration_seconds",
        "resource/cpu_io_read_bytes",
        "resource/cpu_io_write_bytes",
        "resource/cpu_max_rss_bytes",
        "resource/gpu_max_memory_allocated_bytes",
        "resource/gpu_max_memory_reserved_bytes",
        "step_duration_seconds",
    }
)

# Sharded eval.forward reduction (design Seam C): a compact, mode-explicit
# extension of the same metric collective above, never inferred from payload
# shape. Only active when the caller's `reduction_mode` matches this literal
# string (the replicated eval path, and every train call, never set it and
# therefore reach the unchanged reducers above unmodified).
EVAL_DISJOINT_SHARD_REDUCTION_MODE = "disjoint_shard"
_EVAL_SUM_METRIC_KEY_NAMES = frozenset(
    {"example_count", "pack_count", "count/packs", "count/examples"}
)
_EVAL_IDENTICAL_METRIC_KEY_NAMES = frozenset(
    {"count/supervised_atoms", "count/eligible_segments", "count/skipped_segments"}
)
_TOKEN_WEIGHTED_DIAG_SUFFIX = "/token_weighted_diag"
_TOKEN_WEIGHTED_DIAG_WEIGHT_SUFFIX = "/token_weighted_diag/__weight__"
_FINITE_METRIC_KEY_PREFIX = "finite/"


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
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.accelerator = accelerator
        self.device = torch.device(accelerator.device)
        self.rank = int(accelerator.process_index)
        self.world_size = int(accelerator.num_processes)
        self.max_grad_norm = max_grad_norm
        self.rank_report_gatherer = rank_report_gatherer
        self.optimizer_step_count = 0
        self.scheduler_step_count = 0
        self.zero_grad_count = 0
        self._validate_runtime_contract(
            runtime_batch=runtime_batch,
            expected_mixed_precision=expected_mixed_precision,
        )
        self.seed_receipt = seed_training_runtime(
            runtime_config.seed,
            determinism_mode=runtime_config.determinism.mode,
            phase="runtime_setup_reapplied",
        )
        self.model.to(self.device)
        self._prepare_training_objects()

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
        accuracy_stats: Mapping[str, int] | None = None,
        reduction_mode: str | None = None,
    ) -> dict[str, Any]:
        values = {str(key): float(metrics[key]) for key in sorted(metrics)}
        checked_accuracy_stats = _checked_accuracy_stats(accuracy_stats)
        accuracy_keys = tuple(
            key for key in values if key in _ACCURACY_METRIC_CORRECT_FIELDS
        )
        if accuracy_keys and checked_accuracy_stats is None:
            raise RuntimeContractError(
                "accuracy metrics require exact integer accuracy_stats",
                code="runtime.accuracy_stats_missing",
                context={"metrics": list(accuracy_keys)},
            )
        if self.world_size > 1:
            values, per_rank_metrics, global_accuracy_stats = (
                self._reduce_metric_reports(
                    {
                        "kind": "metrics",
                        "planned_step_id": int(planned_step_id),
                        "split": str(split),
                        "rank": self.rank,
                        "world_size": self.world_size,
                        "metrics": values,
                        "accuracy_stats": checked_accuracy_stats,
                        "reduction_mode": reduction_mode,
                    }
                )
            )
        else:
            global_accuracy_stats = checked_accuracy_stats
            if accuracy_keys and global_accuracy_stats is not None:
                _validate_accuracy_metric_ratios(
                    values,
                    global_accuracy_stats,
                    metric_keys=accuracy_keys,
                    rank=self.rank,
                )
                values = _derive_accuracy_metrics(
                    values,
                    global_accuracy_stats,
                    metric_keys=accuracy_keys,
                )
            per_rank_metrics = {
                str(self.rank): {key: values[key] for key in sorted(values)}
            }
        result = {
            "planned_step_id": int(planned_step_id),
            "split": split,
            "rank": self.rank,
            "world_size": self.world_size,
            "metrics": values,
            "per_rank_metrics": per_rank_metrics,
            "reduction": "single_rank" if self.world_size == 1 else "all_rank_mixed",
        }
        if global_accuracy_stats is not None:
            result["accuracy_stats"] = global_accuracy_stats
        return result

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

    def validate_eval_reduction_consensus(
        self,
        *,
        reduction_mode: str,
        pack_count: int | None,
    ) -> None:
        """One-time pipeline-assembly consensus check (Opus HOLD P2-B).

        Every world rank MUST call this before any mode-dependent branching
        (sharding the eval micro-step sequence, or eval.forward's own
        conditional `gather_loss_denominators` collective). Unlike the
        per-invocation `reduction_mode` identity check inside
        `_reduce_metric_reports`, this runs unconditionally on every rank
        regardless of what each rank's own locally-resolved mode is, so a
        divergence is caught here -- before it could otherwise leave one
        rank waiting forever on a collective a diverged peer never joins.
        """

        if self.world_size <= 1:
            return
        local_report = {
            "kind": "eval_reduction_consensus",
            "planned_step_id": 0,
            "split": "consensus",
            "rank": self.rank,
            "world_size": self.world_size,
            "reduction_mode": str(reduction_mode),
            "pack_count": None if pack_count is None else int(pack_count),
        }
        reports = self._gather_rank_reports(local_report)
        if len(reports) != self.world_size:
            raise RuntimeContractError(
                "eval reduction consensus gather must return exactly one "
                "report per world rank",
                code="runtime.eval_reduction_consensus_count",
                context={
                    "expected_report_count": self.world_size,
                    "observed_report_count": len(reports),
                },
            )
        ranks_seen: set[int] = set()
        modes: set[str] = set()
        pack_counts: set[int | None] = set()
        for report_index, report in enumerate(reports):
            if not isinstance(report, Mapping):
                raise RuntimeContractError(
                    "eval reduction consensus gatherer must return mapping reports",
                    code="runtime.eval_reduction_consensus_invalid",
                    context={
                        "report_index": report_index,
                        "value_type": type(report).__name__,
                    },
                )
            try:
                report_rank = int(report["rank"])
                report_mode = str(report["reduction_mode"])
                report_pack_count = report["pack_count"]
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeContractError(
                    "eval reduction consensus report is missing required identity fields",
                    code="runtime.eval_reduction_consensus_invalid",
                    context={"report_index": report_index},
                ) from exc
            if (
                report_rank < 0
                or report_rank >= self.world_size
                or report_rank in ranks_seen
            ):
                raise RuntimeContractError(
                    "eval reduction consensus must contain one unique report per world rank",
                    code="runtime.eval_reduction_consensus_ranks",
                    context={"report_index": report_index, "rank": report_rank},
                )
            ranks_seen.add(report_rank)
            modes.add(report_mode)
            pack_counts.add(
                None if report_pack_count is None else int(report_pack_count)
            )
        if ranks_seen != set(range(self.world_size)):
            raise RuntimeContractError(
                "eval reduction consensus must contain every world rank",
                code="runtime.eval_reduction_consensus_ranks",
                context={"observed_ranks": sorted(ranks_seen)},
            )
        if len(modes) != 1:
            raise RuntimeContractError(
                "eval reduction mode diverged across ranks before sharding; the "
                "active mode must be identical on every rank before any "
                "mode-dependent collective is attempted",
                code="runtime.eval_reduction_consensus_mode_mismatch",
                context={"observed_modes": sorted(modes)},
            )
        if len(pack_counts) != 1:
            raise RuntimeContractError(
                "eval pack_count diverged across ranks before sharding",
                code="runtime.eval_reduction_consensus_pack_count_mismatch",
                context={
                    "observed_pack_counts": sorted(
                        (-1 if value is None else value) for value in pack_counts
                    )
                },
            )

    def _validate_runtime_contract(
        self,
        *,
        runtime_batch: RuntimeBatchResolution,
        expected_mixed_precision: str,
    ) -> None:
        validate_accelerator_runtime(
            self.accelerator,
            expected_mixed_precision=expected_mixed_precision,
        )
        if runtime_batch.world_size != self.world_size:
            raise RuntimeContractError(
                "runtime batch world size must match runtime world size",
                code="runtime.batch_world_size",
                context={
                    "runtime_batch_world_size": runtime_batch.world_size,
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

    def _reduce_metric_reports(
        self, local_report: Mapping[str, Any]
    ) -> tuple[
        dict[str, float],
        dict[str, dict[str, float]],
        dict[str, int] | None,
    ]:
        reports = self._gather_rank_reports(local_report)
        if len(reports) != self.world_size:
            raise RuntimeContractError(
                "metric gather must return exactly one report per world rank",
                code="runtime.metric_gather_count",
                context={
                    "expected_report_count": self.world_size,
                    "observed_report_count": len(reports),
                },
            )
        expected_step = int(local_report["planned_step_id"])
        expected_split = str(local_report["split"])
        expected_keys = tuple(local_report["metrics"])
        expected_reduction_mode = local_report.get("reduction_mode")
        local_accuracy_stats = local_report.get("accuracy_stats")
        reports_by_rank: dict[int, Mapping[str, Any]] = {}
        accuracy_stats_by_rank: dict[int, Mapping[str, Any] | None] = {}
        for report_index, report in enumerate(reports):
            if not isinstance(report, Mapping):
                raise RuntimeContractError(
                    "metric gatherer must return mapping reports",
                    code="runtime.metric_gather_invalid",
                    context={
                        "report_index": report_index,
                        "value_type": type(report).__name__,
                    },
                )
            try:
                report_rank = int(report["rank"])
                report_step = int(report["planned_step_id"])
                report_split = str(report["split"])
                report_metrics = report["metrics"]
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeContractError(
                    "metric gather report is missing required identity fields",
                    code="runtime.metric_gather_invalid",
                    context={"report_index": report_index},
                ) from exc
            if (
                report_rank < 0
                or report_rank >= self.world_size
                or report_rank in reports_by_rank
            ):
                raise RuntimeContractError(
                    "metric gather must contain one unique report per world rank",
                    code="runtime.metric_gather_ranks",
                    context={"report_index": report_index, "rank": report_rank},
                )
            if report_step != expected_step:
                raise RuntimeContractError(
                    "metric gather reports disagree on planned step",
                    code="runtime.metric_gather_step",
                    context={
                        "rank": report_rank,
                        "expected_planned_step_id": expected_step,
                        "observed_planned_step_id": report_step,
                    },
                )
            if report_split != expected_split:
                raise RuntimeContractError(
                    "metric gather reports disagree on split",
                    code="runtime.metric_gather_split",
                    context={
                        "rank": report_rank,
                        "expected_split": expected_split,
                        "observed_split": report_split,
                    },
                )
            report_reduction_mode = report.get("reduction_mode")
            if report_reduction_mode != expected_reduction_mode:
                raise RuntimeContractError(
                    "metric gather reports disagree on the explicit reduction mode; the "
                    "active mode must never be inferred ambiguously from payload shape",
                    code="runtime.metric_gather_reduction_mode",
                    context={
                        "rank": report_rank,
                        "expected_reduction_mode": expected_reduction_mode,
                        "observed_reduction_mode": report_reduction_mode,
                    },
                )
            if not isinstance(report_metrics, Mapping):
                raise RuntimeContractError(
                    "metric gather report metrics must be a mapping",
                    code="runtime.metric_gather_invalid",
                    context={"rank": report_rank},
                )
            observed_keys = tuple(sorted(str(key) for key in report_metrics))
            if observed_keys != expected_keys:
                raise RuntimeContractError(
                    "metric gather reports disagree on metric keys",
                    code="runtime.metric_gather_keys",
                    context={
                        "rank": report_rank,
                        "expected_metric_keys": list(expected_keys),
                        "observed_metric_keys": list(observed_keys),
                    },
                )
            reports_by_rank[report_rank] = report_metrics
            accuracy_stats_by_rank[report_rank] = report.get("accuracy_stats")
        if tuple(sorted(reports_by_rank)) != tuple(range(self.world_size)):
            raise RuntimeContractError(
                "metric gather must contain every world rank",
                code="runtime.metric_gather_ranks",
                context={"observed_ranks": sorted(reports_by_rank)},
            )
        accuracy_keys_present = tuple(
            key for key in expected_keys if key in _ACCURACY_METRIC_CORRECT_FIELDS
        )
        if accuracy_keys_present and local_accuracy_stats is None:
            raise RuntimeContractError(
                "accuracy metric reduction requires local accuracy_stats whenever "
                "acc_top1/acc_top5 are present in the gathered metric payload; a "
                "plain rank mean must never silently substitute for the exact "
                "summed-integer ratio",
                code="runtime.accuracy_stats_missing",
                context={"metrics": list(accuracy_keys_present)},
            )
        global_accuracy_stats = (
            self._reduce_accuracy_stats(
                reports_by_rank=reports_by_rank,
                accuracy_stats_by_rank=accuracy_stats_by_rank,
                metric_keys=accuracy_keys_present,
                replicated=(
                    expected_split == "eval" and expected_reduction_mode is None
                ),
            )
            if accuracy_keys_present
            else None
        )
        eval_sharded = expected_reduction_mode == EVAL_DISJOINT_SHARD_REDUCTION_MODE
        reduced: dict[str, float] = {}
        for key in expected_keys:
            correct_field = _ACCURACY_METRIC_CORRECT_FIELDS.get(key)
            if correct_field is not None:
                assert global_accuracy_stats is not None
                reduced[key] = _accuracy_ratio(
                    global_accuracy_stats[correct_field],
                    global_accuracy_stats[_ACCURACY_ATOM_COUNT_FIELD],
                    metric=key,
                )
            elif key in _MAX_REDUCED_METRIC_KEYS:
                reduced[key] = max(
                    float(reports_by_rank[rank][key]) for rank in range(self.world_size)
                )
            elif eval_sharded and _is_eval_sum_metric_key(key):
                reduced[key] = self._reduce_eval_sum_metric(key, reports_by_rank)
            elif eval_sharded and _is_eval_identical_metric_key(key):
                reduced[key] = self._reduce_eval_identical_metric(key, reports_by_rank)
            elif eval_sharded and key.startswith(_FINITE_METRIC_KEY_PREFIX):
                reduced[key] = self._reduce_eval_finite_metric(key, reports_by_rank)
            else:
                reduced[key] = (
                    sum(
                        float(reports_by_rank[rank][key])
                        for rank in range(self.world_size)
                    )
                    / self.world_size
                )
        per_rank_metrics = {
            str(rank): {key: float(reports_by_rank[rank][key]) for key in expected_keys}
            for rank in range(self.world_size)
        }
        return reduced, per_rank_metrics, global_accuracy_stats

    def _reduce_eval_sum_metric(
        self, key: str, reports_by_rank: Mapping[int, Mapping[str, Any]]
    ) -> float:
        # Disjoint per-rank shards make a plain sum exact: an example or pack
        # belongs to exactly one rank's shard, and the token-weighted-diag
        # keys already carry pre-weighted (value * local selected_count)
        # products / their companion weights, summed here and divided back
        # into an exact global weighted average by the eval caller.
        count_like = not key.endswith(_TOKEN_WEIGHTED_DIAG_SUFFIX)
        total = 0.0
        for rank in range(self.world_size):
            value = float(reports_by_rank[rank][key])
            if count_like:
                _checked_eval_count_value(value, key=key, rank=rank)
            total += value
        return total

    def _reduce_eval_identical_metric(
        self, key: str, reports_by_rank: Mapping[int, Mapping[str, Any]]
    ) -> float:
        # These fields are already global (derived from the merged
        # cross-rank denominator gathered once in prepare_planned_step), so
        # every rank must report the identical value; summing them again
        # would multiply the true value by world_size.
        observed: dict[float, int] = {}
        for rank in range(self.world_size):
            value = float(reports_by_rank[rank][key])
            _checked_eval_count_value(value, key=key, rank=rank)
            observed.setdefault(value, rank)
        if len(observed) != 1:
            raise RuntimeContractError(
                "eval sharded reduction requires an already-global field to be "
                "identical across ranks; it must be emitted once, not rank-summed",
                code="runtime.eval_identical_metric_mismatch",
                context={"key": key, "observed_values": sorted(observed)},
            )
        return next(iter(observed))

    def _reduce_eval_finite_metric(
        self, key: str, reports_by_rank: Mapping[int, Mapping[str, Any]]
    ) -> float:
        # `finite/*` flags are per-rank binary indicators (1.0 finite, 0.0
        # not) computed from each rank's own LOCAL contribution. The
        # replicated evaluator's reference value is 0.0 whenever ANY
        # contribution to the underlying scalar is non-finite (NaN/Inf
        # propagates through the sum that derives it) -- so the correct
        # cross-rank reducer is a logical AND, implemented as `min` over the
        # 0.0/1.0 values. A plain mean would silently report a fractional,
        # meaningless value (e.g. 0.5 for two ranks split finite/non-finite)
        # instead of the correct binary 0.0.
        result = 1.0
        for rank in range(self.world_size):
            value = float(reports_by_rank[rank][key])
            _checked_eval_finite_flag(value, key=key, rank=rank)
            result = min(result, value)
        return result

    def _reduce_accuracy_stats(
        self,
        *,
        reports_by_rank: Mapping[int, Mapping[str, Any]],
        accuracy_stats_by_rank: Mapping[int, Mapping[str, Any] | None],
        metric_keys: Sequence[str],
        replicated: bool,
    ) -> dict[str, int]:
        totals = {
            "top1_correct": 0,
            "top5_correct": 0,
            _ACCURACY_ATOM_COUNT_FIELD: 0,
        }
        checked_by_rank: dict[int, dict[str, int]] = {}
        for rank in range(self.world_size):
            stats = accuracy_stats_by_rank.get(rank)
            if not isinstance(stats, Mapping):
                raise RuntimeContractError(
                    "accuracy metric reduction requires accuracy_stats from every rank",
                    code="runtime.accuracy_stats_missing",
                    context={"metrics": list(metric_keys), "rank": rank},
                )
            checked = _checked_accuracy_stats(stats, rank=rank)
            assert checked is not None
            checked_by_rank[rank] = checked

        for rank, checked in checked_by_rank.items():
            _validate_accuracy_metric_ratios(
                reports_by_rank[rank],
                checked,
                metric_keys=metric_keys,
                rank=rank,
            )

        if replicated:
            reference = checked_by_rank[0]
            mismatched_ranks = [
                rank
                for rank in range(1, self.world_size)
                if checked_by_rank[rank] != reference
            ]
            if mismatched_ranks:
                raise RuntimeContractError(
                    "replicated eval accuracy_stats must be identical on every rank",
                    code="runtime.accuracy_stats_replicated_mismatch",
                    context={"mismatched_ranks": mismatched_ranks},
                )
            return dict(reference)

        for checked in checked_by_rank.values():
            for field_name in totals:
                totals[field_name] += checked[field_name]
        if totals[_ACCURACY_ATOM_COUNT_FIELD] <= 0:
            raise RuntimeContractError(
                "accuracy metric reduction requires a positive summed atom count",
                code="runtime.accuracy_stats_zero_atoms",
                context={"metrics": list(metric_keys)},
            )
        return totals


def _is_eval_sum_metric_key(key: str) -> bool:
    return (
        key in _EVAL_SUM_METRIC_KEY_NAMES
        or key.endswith(_TOKEN_WEIGHTED_DIAG_SUFFIX)
        or key.endswith(_TOKEN_WEIGHTED_DIAG_WEIGHT_SUFFIX)
    )


def _is_eval_identical_metric_key(key: str) -> bool:
    return key in _EVAL_IDENTICAL_METRIC_KEY_NAMES or key.endswith("/segment_count")


def _checked_eval_count_value(value: float, *, key: str, rank: int) -> None:
    if key.endswith(_TOKEN_WEIGHTED_DIAG_SUFFIX):
        return
    if not math.isfinite(value) or value < 0.0 or value != math.floor(value):
        raise RuntimeContractError(
            "eval sharded reduction count field must be a finite non-negative integer",
            code="runtime.eval_count_metric_invalid",
            context={"key": key, "rank": rank, "value": value},
        )


def _checked_eval_finite_flag(value: float, *, key: str, rank: int) -> None:
    if value not in (0.0, 1.0):
        raise RuntimeContractError(
            "eval sharded reduction finite/* field must be exactly 0.0 or 1.0",
            code="runtime.eval_finite_metric_invalid",
            context={"key": key, "rank": rank, "value": value},
        )


def _checked_accuracy_stats(
    accuracy_stats: Mapping[str, int] | None,
    *,
    rank: int = -1,
) -> dict[str, int] | None:
    if accuracy_stats is None:
        return None
    required_fields = (
        *_ACCURACY_METRIC_CORRECT_FIELDS.values(),
        _ACCURACY_ATOM_COUNT_FIELD,
    )
    if set(accuracy_stats) != set(required_fields):
        raise RuntimeContractError(
            "accuracy_stats must contain exactly the declared sufficient-statistic fields",
            code="runtime.accuracy_stats_fields",
            context={
                "rank": rank,
                "expected_fields": sorted(required_fields),
                "observed_fields": sorted(str(field) for field in accuracy_stats),
            },
        )
    checked: dict[str, int] = {}
    for field_name in required_fields:
        if field_name not in accuracy_stats:
            raise RuntimeContractError(
                "accuracy_stats payload is missing a required sufficient-statistic field",
                code="runtime.accuracy_stats_field_missing",
                context={
                    "field": field_name,
                    "available_fields": sorted(accuracy_stats),
                },
            )
        checked[field_name] = _checked_stat_int(
            accuracy_stats, field_name, metric="accuracy_stats", rank=rank
        )
    atom_count = checked[_ACCURACY_ATOM_COUNT_FIELD]
    for correct_field in _ACCURACY_METRIC_CORRECT_FIELDS.values():
        _check_correct_within_atom_count(
            checked[correct_field],
            atom_count,
            field=correct_field,
            metric="accuracy_stats",
            rank=rank,
        )
    return checked


def _validate_accuracy_metric_ratios(
    metrics: Mapping[str, Any],
    accuracy_stats: Mapping[str, int],
    *,
    metric_keys: Sequence[str],
    rank: int,
) -> None:
    atom_count = accuracy_stats[_ACCURACY_ATOM_COUNT_FIELD]
    for metric_key in metric_keys:
        correct_field = _ACCURACY_METRIC_CORRECT_FIELDS[metric_key]
        expected = _accuracy_ratio(
            accuracy_stats[correct_field], atom_count, metric=metric_key
        )
        observed = float(metrics[metric_key])
        if observed != expected:
            raise RuntimeContractError(
                "accuracy metric must match the ratio derived from exact integer stats",
                code="runtime.accuracy_metric_stats_mismatch",
                context={
                    "metric": metric_key,
                    "rank": rank,
                    "observed": observed,
                    "expected": expected,
                },
            )


def _derive_accuracy_metrics(
    metrics: Mapping[str, float],
    accuracy_stats: Mapping[str, int],
    *,
    metric_keys: Sequence[str],
) -> dict[str, float]:
    derived = dict(metrics)
    atom_count = accuracy_stats[_ACCURACY_ATOM_COUNT_FIELD]
    for metric_key in metric_keys:
        derived[metric_key] = _accuracy_ratio(
            accuracy_stats[_ACCURACY_METRIC_CORRECT_FIELDS[metric_key]],
            atom_count,
            metric=metric_key,
        )
    return derived


def _accuracy_ratio(correct: int, atom_count: int, *, metric: str) -> float:
    if atom_count <= 0:
        raise RuntimeContractError(
            "accuracy metric requires a positive atom count",
            code="runtime.accuracy_stats_zero_atoms",
            context={"metric": metric},
        )
    return float(correct) / float(atom_count)


def _checked_stat_int(
    stats: Mapping[str, Any], field_name: str, *, metric: str, rank: int
) -> int:
    value = stats.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeContractError(
            "accuracy metric reduction stat field must be a non-negative integer",
            code="runtime.accuracy_stats_field_type",
            context={
                "metric": metric,
                "rank": rank,
                "field": field_name,
                "value": value,
            },
        )
    return value


def _check_correct_within_atom_count(
    correct: int, atom_count: int, *, field: str, metric: str, rank: int
) -> None:
    if correct > atom_count:
        raise RuntimeContractError(
            "accuracy metric reduction stat field cannot exceed the atom count",
            code="runtime.accuracy_stats_correct_exceeds_atoms",
            context={
                "metric": metric,
                "rank": rank,
                "field": field,
                "correct": correct,
                "atom_count": atom_count,
            },
        )


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
    "EVAL_DISJOINT_SHARD_REDUCTION_MODE",
    "TrainRuntime",
    "validate_accelerator_runtime",
]
