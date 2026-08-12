"""Artifact-independent packed forward evaluation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import math
import os
from types import MappingProxyType
from typing import Any, Protocol

import torch

from src.common.errors import RuntimeContractError
from src.runtime.train_runtime import EVAL_DISJOINT_SHARD_REDUCTION_MODE
from src.training.supervised_trainer import (
    LossContextFactory,
    LossRunnerBoundary,
    QwenForwardFn,
    SupervisedMicroStep,
    _default_loss_context,
    _default_qwen_forward,
    _runtime_loss_denominator_gatherer,
)


EVAL_FORWARD_SPLIT = "eval"

# Rank-sharded eval.forward (design Seam C). `EVAL_REDUCTION_REPLICATED` is
# the pre-Wave-4 behavior, byte-identical: every rank evaluates the full
# eval set and cross-rank reduction is the existing identical-value mean.
# `EVAL_REDUCTION_DISJOINT_SHARD` partitions eval packs disjointly by
# `sequence_ordinal % world_size == rank` (the canonical micro-step
# sequence position, not the `pack_index` identity label -- see
# `partition_eval_micro_steps_for_rank`) and reduces exact sufficient
# statistics through the same bounded gatherer used by training.
EVAL_REDUCTION_REPLICATED = "replicated"
EVAL_REDUCTION_DISJOINT_SHARD = EVAL_DISJOINT_SHARD_REDUCTION_MODE
_EVAL_REDUCTION_MODES = frozenset(
    {EVAL_REDUCTION_REPLICATED, EVAL_REDUCTION_DISJOINT_SHARD}
)

_EVAL_REDUCTION_CONTROL_ENV = "COORDEXP_SWIFT_EVAL_REDUCTION_MODE"
_EVAL_REDUCTION_CONTROL_AUTO = "auto"
_EVAL_REDUCTION_CONTROLS = frozenset(
    {_EVAL_REDUCTION_CONTROL_AUTO, EVAL_REDUCTION_REPLICATED}
)

_TOKEN_WEIGHTED_DIAG_SUFFIX = "/token_weighted_diag"
_TOKEN_WEIGHTED_DIAG_WEIGHT_SUFFIX = "/token_weighted_diag/__weight__"


def resolve_eval_reduction_control() -> str:
    """Internal reduction-mode override (env var, not YAML).

    Disjoint-shard eval reduction is implemented and its full-row exactness
    against the replicated evaluator is proven by test (tasks.md 4.1-4.3)
    and the exact 8-rank M4 gate recorded in `implementation-notes.md`.
    `auto` is therefore the shipped default and selects disjoint-shard
    reduction whenever the eval pack count is at least the world size. The
    automatic `pack_count < world_size` replicated fallback stays active
    regardless of this control and is never overridden by it. Explicit
    `replicated` remains available for internal regression measurements; this
    is not a public YAML/CLI compatibility surface.
    """

    raw = os.environ.get(_EVAL_REDUCTION_CONTROL_ENV)
    if raw is None:
        return _EVAL_REDUCTION_CONTROL_AUTO
    if raw not in _EVAL_REDUCTION_CONTROLS:
        raise RuntimeContractError(
            "eval reduction mode override must be 'auto' or 'replicated'",
            code="eval_forward.reduction_control_invalid",
            context={"value": raw, "env_var": _EVAL_REDUCTION_CONTROL_ENV},
        )
    return raw


def resolve_active_eval_reduction_mode(*, pack_count: int, world_size: int) -> str:
    """Decide the active mode once, explicitly -- never inferred later.

    The automatic `pack_count < world_size` replicated fallback is
    unconditional (design Seam C); the `auto`-vs-`replicated` control below
    only decides whether disjoint-shard reduction is even attempted once
    that structural fallback does not apply.
    """

    if world_size <= 1:
        return EVAL_REDUCTION_REPLICATED
    if pack_count < world_size:
        return EVAL_REDUCTION_REPLICATED
    if resolve_eval_reduction_control() == _EVAL_REDUCTION_CONTROL_AUTO:
        return EVAL_REDUCTION_DISJOINT_SHARD
    return EVAL_REDUCTION_REPLICATED


def partition_eval_micro_steps_for_rank(
    micro_steps: Sequence[SupervisedMicroStep], *, rank: int, world_size: int
) -> tuple[SupervisedMicroStep, ...]:
    """Deterministic disjoint covering assignment: `sequence_ordinal % world_size == rank`.

    The sharding key is each micro-step's POSITION in the given `micro_steps`
    sequence (`enumerate(micro_steps)`'s index) -- the canonical sequence
    ordinal -- never `pack.pack_index`. `pack_index` is an identity/label
    assigned at pack-planning time; it is not guaranteed to be a contiguous
    0..N-1 restatement of a micro-step's position in whatever sequence this
    function is given (a filtered or re-materialized cache could carry a
    non-contiguous or offset `pack_index`), so using its VALUE for the
    modulo would silently produce a different, non-canonical partition.
    Every micro-step's pack identity is still validated fail-closed (a
    missing/malformed `pack.pack_index`) even though that value plays no
    role in the partition itself.

    `micro_steps` MUST already be in canonical pack order (as produced by
    `load_all_micro_steps_from_cache`); this only filters, it never
    reorders, so forward order within the rank's shard is unchanged.
    """

    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if rank < 0 or rank >= world_size:
        raise ValueError("rank must be inside world_size")
    selected: list[SupervisedMicroStep] = []
    for sequence_ordinal, micro_step in enumerate(micro_steps):
        _validate_pack_identity(micro_step)
        if sequence_ordinal % world_size == rank:
            selected.append(micro_step)
    return tuple(selected)


def _validate_pack_identity(micro_step: SupervisedMicroStep) -> int:
    """Fail-closed pack identity check; the returned value is NOT the
    sharding ordinal (see `partition_eval_micro_steps_for_rank`)."""

    pack = getattr(micro_step, "pack", None)
    pack_index = getattr(pack, "pack_index", None)
    if not isinstance(pack_index, int) or isinstance(pack_index, bool):
        raise RuntimeContractError(
            "disjoint eval sharding requires a micro-step pack with an integer pack_index",
            code="eval_forward.pack_identity_missing",
            context={"pack_type": type(pack).__name__},
        )
    return pack_index


class EvalRuntimeBoundary(Protocol):
    def move_micro_step(
        self,
        micro_step: SupervisedMicroStep,
        *,
        planned_step_id: int,
        local_micro_step_index: int,
    ) -> SupervisedMicroStep: ...

    def gather_metrics(
        self,
        metrics: Mapping[str, float],
        *,
        planned_step_id: int,
        split: str,
        accuracy_stats: Mapping[str, int] | None = None,
        reduction_mode: str | None = None,
    ) -> Mapping[str, Any]: ...

    def gather_loss_denominators(
        self,
        denominators: Mapping[str, Mapping[str, Any]],
        *,
        planned_step_id: int,
    ) -> tuple[Mapping[str, Mapping[str, Any]], ...]: ...


@dataclass(frozen=True)
class ForwardEvalObservation:
    """One completed eval invocation, ready for a canonical wide logging row."""

    planned_step_id: int
    split: str
    trigger_reasons: tuple[str, ...]
    example_count: int
    pack_count: int
    scalars: Mapping[str, float | None]
    accuracy_stats: Mapping[str, int] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "scalars", MappingProxyType(dict(self.scalars)))
        if self.accuracy_stats is not None:
            object.__setattr__(
                self,
                "accuracy_stats",
                MappingProxyType(dict(self.accuracy_stats)),
            )

    def to_logging_row(self) -> dict[str, Any]:
        """Return writer input without normalizing non-finite scalar values."""

        row = {
            "step": self.planned_step_id,
            "split": self.split,
            "trigger_reasons": list(self.trigger_reasons),
            "example_count": self.example_count,
            "pack_count": self.pack_count,
            **self.scalars,
        }
        if self.accuracy_stats is not None:
            row["accuracy_stats"] = dict(self.accuracy_stats)
        return row


class ForwardEvalRunner:
    def __init__(
        self,
        *,
        model: Any,
        micro_step_stream: Iterable[SupervisedMicroStep],
        loss_runner: LossRunnerBoundary,
        eval_source: Mapping[str, Any] | None,
        qwen_forward: QwenForwardFn | None = None,
        loss_context_factory: LossContextFactory | None = None,
        runtime: EvalRuntimeBoundary | None = None,
        reduction_mode: str = EVAL_REDUCTION_REPLICATED,
        world_size: int = 1,
        rank: int = 0,
    ) -> None:
        if not all(
            callable(getattr(loss_runner, name, None))
            for name in (
                "prepare_planned_step",
                "compute_micro_step",
                "finalize_planned_step",
            )
        ):
            raise RuntimeContractError(
                "eval.forward requires a streaming-capable loss runner "
                "(prepare_planned_step/compute_micro_step/finalize_planned_step); "
                "the non-streaming batch eval path has been removed",
                code="eval_forward.loss_runner_requires_streaming_protocol",
                context={"loss_runner": type(loss_runner).__name__},
            )
        if reduction_mode not in _EVAL_REDUCTION_MODES:
            raise RuntimeContractError(
                "eval.forward reduction_mode must be 'replicated' or 'disjoint_shard'",
                code="eval_forward.reduction_mode_invalid",
                context={"reduction_mode": reduction_mode},
            )
        if world_size <= 0:
            raise RuntimeContractError(
                "eval.forward world_size must be positive",
                code="eval_forward.world_size_invalid",
                context={"world_size": world_size},
            )
        if rank < 0 or rank >= world_size:
            raise RuntimeContractError(
                "eval.forward rank must be inside world_size",
                code="eval_forward.rank_invalid",
                context={"rank": rank, "world_size": world_size},
            )
        if reduction_mode == EVAL_REDUCTION_DISJOINT_SHARD and world_size <= 1:
            raise RuntimeContractError(
                "eval.forward disjoint_shard reduction requires world_size "
                "greater than one; use the automatic replicated fallback instead",
                code="eval_forward.disjoint_shard_requires_multi_rank",
                context={"world_size": world_size},
            )
        self.model = model
        self.micro_step_stream = micro_step_stream
        self.loss_runner = loss_runner
        self.eval_source = None if eval_source is None else dict(eval_source)
        self.qwen_forward = qwen_forward or _default_qwen_forward
        self.loss_context_factory = loss_context_factory or _default_loss_context
        self.runtime = runtime
        self.reduction_mode = reduction_mode
        self.world_size = int(world_size)
        self.rank = int(rank)

    def run(
        self,
        *,
        planned_step_id: int,
        trigger_reasons: Sequence[str],
    ) -> ForwardEvalObservation:
        if planned_step_id <= 0:
            raise RuntimeContractError(
                "eval.forward planned_step_id must be positive",
                code="eval_forward.planned_step_id",
                context={"planned_step_id": planned_step_id},
            )
        if not self.eval_source:
            raise RuntimeContractError(
                "eval.forward requires an explicit eval source",
                code="eval_forward.source_required",
            )

        was_training = getattr(self.model, "training", None)
        eval_method = getattr(self.model, "eval", None)
        train_method = getattr(self.model, "train", None)
        try:
            if callable(eval_method):
                eval_method()
            with torch.no_grad():
                example_count, pack_count, scalars, accuracy_stats = (
                    self._run_forward_only(planned_step_id=planned_step_id)
                )
        finally:
            if was_training is not None and callable(train_method):
                train_method(bool(was_training))

        return ForwardEvalObservation(
            planned_step_id=planned_step_id,
            split=EVAL_FORWARD_SPLIT,
            trigger_reasons=tuple(str(reason) for reason in trigger_reasons),
            example_count=example_count,
            pack_count=pack_count,
            scalars=scalars,
            accuracy_stats=accuracy_stats,
        )

    def _run_forward_only(
        self, *, planned_step_id: int
    ) -> tuple[int, int, dict[str, float | None], dict[str, int] | None]:
        micro_steps = tuple(self.micro_step_stream)
        if not micro_steps:
            raise RuntimeContractError(
                "eval.forward requires at least one eval micro-step",
                code="eval_forward.empty_stream",
                context={"planned_step_id": planned_step_id},
            )
        return self._run_streaming_forward_only(
            micro_steps, planned_step_id=planned_step_id
        )

    def _run_streaming_forward_only(
        self,
        micro_steps: Sequence[SupervisedMicroStep],
        *,
        planned_step_id: int,
    ) -> tuple[int, int, dict[str, float | None]]:
        sharded = self.reduction_mode == EVAL_REDUCTION_DISJOINT_SHARD
        if sharded:
            denominator_gatherer = _runtime_loss_denominator_gatherer(
                self.runtime,
                planned_step_id=planned_step_id,
                world_size=self.world_size,
            )
            plan = self.loss_runner.prepare_planned_step(
                tuple(micro_steps),
                denominator_gatherer=denominator_gatherer,
                world_size=self.world_size,
                rank=self.rank,
            )
        else:
            plan = self.loss_runner.prepare_planned_step(tuple(micro_steps))
        micro_loss_artifacts: list[Mapping[str, Any]] = []
        example_count = 0
        for local_index, micro_step in enumerate(micro_steps):
            example_count += len(tuple(micro_step.encoded_examples))
            moved = self._move_micro_step(
                micro_step,
                planned_step_id=planned_step_id,
                local_micro_step_index=local_index,
            )
            forward_result = self.qwen_forward(
                _runtime_model(self.runtime, self.model), moved
            )
            context = self.loss_context_factory(moved, forward_result)
            loss_bundle = self.loss_runner.compute_micro_step(
                context, plan, local_micro_step_index=local_index
            )
            micro_loss_artifacts.append(_artifact(loss_bundle))
            del loss_bundle, context, forward_result, moved

        loss_artifact = self.loss_runner.finalize_planned_step(
            tuple(dict(item) for item in micro_loss_artifacts), plan
        )
        scalars = _metric_scalars(None, loss_artifact)
        accuracy_stats = _accuracy_stats_from(None, loss_artifact)
        pack_count = len(micro_steps)
        if sharded:
            scalars = _prepare_disjoint_shard_scalars(
                scalars,
                loss_artifact,
                example_count=example_count,
                pack_count=pack_count,
            )
        gathered, global_accuracy_stats = self._gather_scalars(
            scalars, planned_step_id=planned_step_id, accuracy_stats=accuracy_stats
        )
        if sharded:
            gathered, example_count, pack_count = _finalize_disjoint_shard_scalars(
                gathered
            )
        return example_count, pack_count, gathered, global_accuracy_stats

    def _move_micro_step(
        self,
        micro_step: SupervisedMicroStep,
        *,
        planned_step_id: int,
        local_micro_step_index: int,
    ) -> SupervisedMicroStep:
        if self.runtime is None:
            return micro_step
        return self.runtime.move_micro_step(
            micro_step,
            planned_step_id=planned_step_id,
            local_micro_step_index=local_micro_step_index,
        )

    def _gather_scalars(
        self,
        scalars: Mapping[str, float | None],
        *,
        planned_step_id: int,
        accuracy_stats: Mapping[str, int] | None = None,
    ) -> tuple[dict[str, float | None], dict[str, int] | None]:
        gather = getattr(self.runtime, "gather_metrics", None)
        if not callable(gather):
            return dict(scalars), _strict_accuracy_stats_or_none(accuracy_stats)
        finite_or_nonfinite = {
            name: float(value) for name, value in scalars.items() if value is not None
        }
        gather_kwargs: dict[str, Any] = {
            "planned_step_id": planned_step_id,
            "split": EVAL_FORWARD_SPLIT,
            "accuracy_stats": accuracy_stats,
        }
        if self.reduction_mode == EVAL_REDUCTION_DISJOINT_SHARD:
            gather_kwargs["reduction_mode"] = EVAL_REDUCTION_DISJOINT_SHARD
        gathered = gather(finite_or_nonfinite, **gather_kwargs)
        reduced = gathered.get("metrics") if isinstance(gathered, Mapping) else None
        if not isinstance(reduced, Mapping):
            raise RuntimeContractError(
                "eval.forward runtime metric reduction returned no scalar mapping",
                code="eval_forward.metric_reduction",
                context={"planned_step_id": planned_step_id},
            )
        gathered_accuracy_stats = (
            gathered.get("accuracy_stats") if isinstance(gathered, Mapping) else None
        )
        if accuracy_stats is not None and not isinstance(
            gathered_accuracy_stats, Mapping
        ):
            raise RuntimeContractError(
                "eval.forward runtime metric reduction returned no global accuracy_stats",
                code="eval_forward.accuracy_stats_reduction",
                context={"planned_step_id": planned_step_id},
            )
        checked_accuracy_stats = _strict_accuracy_stats_or_none(gathered_accuracy_stats)
        result = {str(name): _optional_float(value) for name, value in reduced.items()}
        for name, value in scalars.items():
            if value is None:
                result.setdefault(name, None)
        return result, checked_accuracy_stats


def _strict_accuracy_stats_or_none(
    accuracy_stats: Mapping[str, Any] | None,
) -> dict[str, int] | None:
    if accuracy_stats is None:
        return None
    expected_fields = {"top1_correct", "top5_correct", "atom_count"}
    if set(accuracy_stats) != expected_fields:
        raise RuntimeContractError(
            "eval.forward accuracy_stats must contain exactly the declared fields",
            code="eval_forward.accuracy_stats_fields",
            context={
                "expected_fields": sorted(expected_fields),
                "observed_fields": sorted(str(field) for field in accuracy_stats),
            },
        )
    checked: dict[str, int] = {}
    for field_name in sorted(expected_fields):
        value = accuracy_stats[field_name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeContractError(
                "eval.forward accuracy_stats fields must be non-negative integers",
                code="eval_forward.accuracy_stats_field_type",
                context={"field": field_name, "value": value},
            )
        checked[field_name] = value
    atom_count = checked["atom_count"]
    if checked["top1_correct"] > atom_count or checked["top5_correct"] > atom_count:
        raise RuntimeContractError(
            "eval.forward accuracy correct counts cannot exceed atom_count",
            code="eval_forward.accuracy_stats_correct_exceeds_atoms",
            context=checked,
        )
    return checked


def _runtime_model(runtime: EvalRuntimeBoundary | None, fallback_model: Any) -> Any:
    return getattr(runtime, "model", fallback_model)


def _prepare_disjoint_shard_scalars(
    scalars: Mapping[str, float | None],
    loss_artifact: Mapping[str, Any],
    *,
    example_count: int,
    pack_count: int,
) -> dict[str, float | None]:
    """Inject rank-local sufficient statistics before the metric gather.

    `example_count`/`pack_count` become ordinary sum-reduced metric keys
    (never durable row fields themselves -- extracted back out after
    reduction). Each term's `token_weighted_diag` value is replaced by its
    pre-weighted product (`value * local selected_count`) plus a companion
    `.../__weight__` key, so the exact cross-rank weighted average
    `sum_r(value_r * count_r) / sum_r(count_r)` can be recovered from two
    plain sums.
    """

    prepared = dict(scalars)
    prepared["example_count"] = float(example_count)
    prepared["pack_count"] = float(pack_count)
    for term in loss_artifact.get("terms", ()):
        name = str(term.get("name"))
        weighted_key = f"loss/{name}{_TOKEN_WEIGHTED_DIAG_SUFFIX}"
        if weighted_key not in prepared:
            continue
        local_value = prepared[weighted_key]
        local_value = 0.0 if local_value is None else float(local_value)
        selected_count = int(term.get("selected_count", 0))
        prepared[weighted_key] = local_value * float(selected_count)
        prepared[f"loss/{name}{_TOKEN_WEIGHTED_DIAG_WEIGHT_SUFFIX}"] = float(
            selected_count
        )
    return prepared


def _finalize_disjoint_shard_scalars(
    scalars: Mapping[str, float | None],
) -> tuple[dict[str, float | None], int, int]:
    result = dict(scalars)
    example_count = _required_nonnegative_int(
        result.pop("example_count", None), field="example_count"
    )
    pack_count = _required_nonnegative_int(
        result.pop("pack_count", None), field="pack_count"
    )
    weight_keys = [
        key for key in result if key.endswith(_TOKEN_WEIGHTED_DIAG_WEIGHT_SUFFIX)
    ]
    for weight_key in weight_keys:
        base_key = weight_key[: -len("/__weight__")]
        weight_value = result.pop(weight_key)
        if base_key not in result:
            continue
        weight = 0.0 if weight_value is None else float(weight_value)
        if weight <= 0.0:
            # Mirrors `_weighted_average`'s own zero-total-weight convention
            # (`src/losses/runner.py`) exactly: a term with globally zero
            # selected tokens across the sharded eval set (e.g. an optional
            # coordinate term on a text-only eval split) is not corrupted
            # data -- `TrainRuntime._reduce_eval_sum_metric` already
            # fail-closes on a negative per-rank weight before this runs, so
            # reaching here with `weight <= 0.0` only ever means an honest
            # global zero, which the replicated reference this reduction
            # must reproduce also resolves to 0.0, never a raise.
            result[base_key] = 0.0
            continue
        numerator = result[base_key]
        numerator = 0.0 if numerator is None else float(numerator)
        result[base_key] = numerator / weight
    return result, example_count, pack_count


def _required_nonnegative_int(value: float | None, *, field: str) -> int:
    if (
        value is None
        or not math.isfinite(value)
        or value < 0
        or value != math.floor(value)
    ):
        raise RuntimeContractError(
            "eval sharded reduction count field must resolve to a finite "
            "non-negative integer",
            code="eval_forward.count_field_invalid",
            context={"field": field, "value": value},
        )
    return int(value)


def _artifact(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_artifact_dict"):
        return value.to_artifact_dict()
    if isinstance(value, Mapping):
        return dict(value)
    return {"type": type(value).__name__}


def _metric_scalars(
    loss_bundle: Any, loss_artifact: Mapping[str, Any]
) -> dict[str, float | None]:
    metrics = getattr(loss_bundle, "metrics", None)
    if not isinstance(metrics, Mapping):
        metrics = loss_artifact.get("metrics")
    if isinstance(metrics, Mapping):
        return {str(name): _optional_float(value) for name, value in metrics.items()}
    return {"loss/total": _optional_float(loss_artifact.get("total_loss"))}


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _accuracy_stats_from(
    loss_bundle: Any, loss_artifact: Mapping[str, Any] | None
) -> Mapping[str, int] | None:
    stats = getattr(loss_bundle, "accuracy_stats", None)
    if not isinstance(stats, Mapping) and isinstance(loss_artifact, Mapping):
        stats = loss_artifact.get("accuracy_stats")
    return stats if isinstance(stats, Mapping) else None


__all__ = [
    "EVAL_FORWARD_SPLIT",
    "EVAL_REDUCTION_DISJOINT_SHARD",
    "EVAL_REDUCTION_REPLICATED",
    "ForwardEvalObservation",
    "ForwardEvalRunner",
    "partition_eval_micro_steps_for_rank",
    "resolve_active_eval_reduction_mode",
    "resolve_eval_reduction_control",
]
