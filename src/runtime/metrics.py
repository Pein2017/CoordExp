"""Typed distributed metric samples and their explicit reducers.

This module is the single owner of what a durable scalar *means* when it
crosses ranks. Every sample carries its reducer as data declared by the code
that produced the value; nothing here inspects any part of a metric's name to
guess semantics, and there is no fallback reducer. A metric that arrives
without a declared reducer fails closed before any durable publication.

Two sample forms exist:

* :class:`ScalarSample` -- one finite-or-non-finite value (or an explicit
  backend unavailability) reduced by exactly one of ``SUM``, ``MAX``,
  ``IDENTICAL``, or ``BOOL_ALL``;
* :class:`RatioSample` -- an exact numerator/denominator pair reduced by
  summing both sides before dividing, so a global ratio is never the mean of
  already-normalized rank-local ratios.

:class:`AccuracySufficientStats` is the one structured side channel: exact
integer top-1/top-5 correct counts over an atom count, reduced by declared
``SUM`` (disjoint work) or ``IDENTICAL`` (replicated work), from which the
producer-named accuracy ratios are derived.

The reduction entry point is a pure function over already-gathered per-rank
payloads: the caller owns the collective, this module owns the meaning. The
world-size-one path runs the identical validation and reduction code, so a
single-rank run cannot accept a batch that a distributed run would reject.

There is deliberately no mutable registry, no event name, no subscription,
and no reducer named ``ALL`` (``BOOL_ALL`` is the only boolean conjunction).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.common.errors import RuntimeContractError

REDUCER_SUM = "SUM"
REDUCER_MAX = "MAX"
REDUCER_IDENTICAL = "IDENTICAL"
REDUCER_BOOL_ALL = "BOOL_ALL"

#: The complete closed reducer set for scalar samples. `ALL` is intentionally
#: absent: boolean conjunction is only ever spelled `BOOL_ALL`.
SCALAR_REDUCERS = frozenset(
    {REDUCER_SUM, REDUCER_MAX, REDUCER_IDENTICAL, REDUCER_BOOL_ALL}
)
#: Accuracy sufficient statistics are either disjoint (summed) or replicated
#: (identical on every rank); no other reduction of exact integer statistics
#: is meaningful.
ACCURACY_REDUCERS = frozenset({REDUCER_SUM, REDUCER_IDENTICAL})

SCALAR_FORM = "scalar"
RATIO_FORM = "ratio"

METRIC_PAYLOAD_KIND = "metrics"

#: Absolute tolerance for the `IDENTICAL` reducer. Exact equality is the
#: declared contract: the two-process gloo probe
#: `scripts/probes/coordexp_swift/obs_wave2_gloo_probe.py` measured a maximum
#: cross-rank divergence of exactly 0.0 for a replicated-eval-shaped
#: computation over identical data under the enforced deterministic
#: environment, so no bounded-tolerance variant is warranted. A future
#: measured divergence would raise this constant and be recorded with its
#: receipt rather than silently softening the comparison at a call site.
IDENTICAL_ABS_TOLERANCE = 0.0


def _checked_name(name: Any, *, context_field: str) -> str:
    if not isinstance(name, str) or not name:
        raise RuntimeContractError(
            "metric sample name must be a non-empty string",
            code="runtime.metric_sample_name_invalid",
            context={"field": context_field, "value_type": type(name).__name__},
        )
    return name


def _checked_float(value: Any, *, name: str, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeContractError(
            "metric sample value must be a real number",
            code="runtime.metric_sample_value_invalid",
            context={"metric": name, "field": field, "value_type": type(value).__name__},
        )
    return float(value)


def _checked_integral(value: float, *, name: str, rank: int = -1) -> None:
    if not math.isfinite(value) or value < 0.0 or value != math.floor(value):
        raise RuntimeContractError(
            "integral metric sample must be a finite non-negative integer value",
            code="runtime.metric_count_value_invalid",
            context={"metric": name, "rank": rank, "value": value},
        )


def _checked_bool_flag(value: float, *, name: str, rank: int = -1) -> None:
    if value not in (0.0, 1.0):
        raise RuntimeContractError(
            "BOOL_ALL metric sample must be exactly 0.0 or 1.0",
            code="runtime.metric_bool_all_value_invalid",
            context={"metric": name, "rank": rank, "value": value},
        )


def _values_identical(left: float, right: float) -> bool:
    if math.isnan(left) and math.isnan(right):
        # Two ranks that both observed the same non-finite result agree; a
        # NaN never equals itself under `==` and must not be read as drift.
        return True
    if left == right:
        return True
    if IDENTICAL_ABS_TOLERANCE <= 0.0:
        return False
    if not (math.isfinite(left) and math.isfinite(right)):
        return False
    return abs(left - right) <= IDENTICAL_ABS_TOLERANCE


@dataclass(frozen=True, slots=True)
class ScalarSample:
    """One rank-local scalar with its producer-declared reducer.

    ``value is None`` means the metric is genuinely unavailable on this
    backend; it is only accepted for a sample the producer marked
    ``required=False``, and every rank must agree about that availability.
    ``integral=True`` additionally binds the value to an exact non-negative
    integer (exact summed counts).
    """

    name: str
    reducer: str
    value: float | None
    required: bool = True
    integral: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _checked_name(self.name, context_field="name"))
        if self.reducer not in SCALAR_REDUCERS:
            raise RuntimeContractError(
                "metric sample requires one exact declared reducer; the reduction "
                "boundary never infers mean, sum, maximum, or ratio semantics from "
                "a metric name",
                code="runtime.metric_reducer_undeclared",
                context={
                    "metric": self.name,
                    "declared_reducer": self.reducer,
                    "supported_reducers": sorted(SCALAR_REDUCERS),
                },
            )
        if not isinstance(self.required, bool) or not isinstance(self.integral, bool):
            raise RuntimeContractError(
                "metric sample required/integral flags must be booleans",
                code="runtime.metric_sample_flag_invalid",
                context={"metric": self.name},
            )
        if self.value is None:
            if self.required:
                raise RuntimeContractError(
                    "a required metric sample cannot be unavailable; declare the "
                    "field conditional to report it as backend-unavailable",
                    code="runtime.metric_sample_required_unavailable",
                    context={"metric": self.name},
                )
            return
        value = _checked_float(self.value, name=self.name, field="value")
        object.__setattr__(self, "value", value)
        if self.integral:
            _checked_integral(value, name=self.name)
        if self.reducer == REDUCER_BOOL_ALL:
            _checked_bool_flag(value, name=self.name)

    @property
    def form(self) -> str:
        return SCALAR_FORM

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "form": SCALAR_FORM,
            "reducer": self.reducer,
            "required": self.required,
            "integral": self.integral,
            "value": self.value,
        }


@dataclass(frozen=True, slots=True)
class RatioSample:
    """A sum-before-divide ratio: never the mean of rank-local ratios.

    ``empty_value`` is the declared result when the *global* denominator sums
    to zero. Leaving it ``None`` makes a globally empty denominator a hard
    failure; producers whose upstream calculation resolves an empty weighted
    average to a value (see ``_weighted_average`` in ``src/losses/runner.py``)
    declare that same value here.
    """

    name: str
    numerator: float
    denominator: float
    empty_value: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _checked_name(self.name, context_field="name"))
        numerator = _checked_float(self.numerator, name=self.name, field="numerator")
        denominator = _checked_float(
            self.denominator, name=self.name, field="denominator"
        )
        if not math.isfinite(denominator) or denominator < 0.0:
            raise RuntimeContractError(
                "ratio metric sample requires a finite non-negative denominator",
                code="runtime.metric_ratio_denominator_invalid",
                context={"metric": self.name, "denominator": denominator},
            )
        object.__setattr__(self, "numerator", numerator)
        object.__setattr__(self, "denominator", denominator)
        if self.empty_value is not None:
            object.__setattr__(
                self,
                "empty_value",
                _checked_float(self.empty_value, name=self.name, field="empty_value"),
            )

    @property
    def form(self) -> str:
        return RATIO_FORM

    @property
    def required(self) -> bool:
        return True

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "form": RATIO_FORM,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "empty_value": self.empty_value,
        }


@dataclass(frozen=True, slots=True)
class AccuracySufficientStats:
    """Exact integer accuracy statistics plus the producer-named ratios."""

    reducer: str
    top1_correct: int
    top5_correct: int
    atom_count: int
    top1_metric_name: str = "acc_top1"
    top5_metric_name: str = "acc_top5"

    def __post_init__(self) -> None:
        if self.reducer not in ACCURACY_REDUCERS:
            raise RuntimeContractError(
                "accuracy sufficient statistics require a declared SUM or "
                "IDENTICAL reducer",
                code="runtime.metric_reducer_undeclared",
                context={
                    "metric": "accuracy_stats",
                    "declared_reducer": self.reducer,
                    "supported_reducers": sorted(ACCURACY_REDUCERS),
                },
            )
        _checked_name(self.top1_metric_name, context_field="top1_metric_name")
        _checked_name(self.top5_metric_name, context_field="top5_metric_name")
        for field_name in ("top1_correct", "top5_correct", "atom_count"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RuntimeContractError(
                    "accuracy metric reduction stat field must be a non-negative integer",
                    code="runtime.accuracy_stats_field_type",
                    context={
                        "metric": "accuracy_stats",
                        "field": field_name,
                        "value": value,
                    },
                )
        for field_name in ("top1_correct", "top5_correct"):
            if getattr(self, field_name) > self.atom_count:
                raise RuntimeContractError(
                    "accuracy metric reduction stat field cannot exceed the atom count",
                    code="runtime.accuracy_stats_correct_exceeds_atoms",
                    context={
                        "metric": "accuracy_stats",
                        "field": field_name,
                        "correct": getattr(self, field_name),
                        "atom_count": self.atom_count,
                    },
                )

    def to_payload(self) -> dict[str, Any]:
        return {
            "reducer": self.reducer,
            "top1_correct": self.top1_correct,
            "top5_correct": self.top5_correct,
            "atom_count": self.atom_count,
            "top1_metric_name": self.top1_metric_name,
            "top5_metric_name": self.top5_metric_name,
        }


@dataclass(frozen=True, slots=True)
class MetricBatch:
    """The closed set of typed samples one rank contributes for one boundary."""

    planned_step_id: int
    split: str
    samples: tuple[ScalarSample | RatioSample, ...] = ()
    accuracy: AccuracySufficientStats | None = None
    reduction_mode: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "planned_step_id", int(self.planned_step_id))
        object.__setattr__(self, "split", str(self.split))
        object.__setattr__(self, "samples", tuple(self.samples))
        seen: set[str] = set()
        for sample in self.samples:
            if not isinstance(sample, (ScalarSample, RatioSample)):
                raise RuntimeContractError(
                    "metric batch accepts only typed scalar or ratio samples",
                    code="runtime.metric_sample_form_invalid",
                    context={"value_type": type(sample).__name__},
                )
            if sample.name in seen:
                raise RuntimeContractError(
                    "metric batch names must be unique",
                    code="runtime.metric_sample_duplicate",
                    context={"metric": sample.name},
                )
            seen.add(sample.name)
        if self.accuracy is not None:
            for derived in (
                self.accuracy.top1_metric_name,
                self.accuracy.top5_metric_name,
            ):
                if derived in seen:
                    raise RuntimeContractError(
                        "accuracy ratio names must not collide with a declared sample",
                        code="runtime.metric_sample_duplicate",
                        context={"metric": derived},
                    )

    def to_rank_payload(self, *, rank: int, world_size: int) -> dict[str, Any]:
        return {
            "kind": METRIC_PAYLOAD_KIND,
            "planned_step_id": self.planned_step_id,
            "split": self.split,
            "rank": int(rank),
            "world_size": int(world_size),
            "reduction_mode": self.reduction_mode,
            "samples": [sample.to_payload() for sample in self.samples],
            "accuracy_stats": (
                None if self.accuracy is None else self.accuracy.to_payload()
            ),
        }


@dataclass(frozen=True, slots=True)
class ReducedMetricBatch:
    """Reduced global values plus the ephemeral per-rank scalar projection."""

    metrics: dict[str, float | None]
    per_rank_metrics: dict[str, dict[str, float]]
    accuracy_stats: dict[str, int] | None


def _payload_identity(
    payload: Mapping[str, Any], *, index: int
) -> tuple[int, int, str, Any]:
    try:
        rank = int(payload["rank"])
        planned_step_id = int(payload["planned_step_id"])
        split = str(payload["split"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeContractError(
            "metric gather report is missing required identity fields",
            code="runtime.metric_gather_invalid",
            context={"report_index": index},
        ) from exc
    return rank, planned_step_id, split, payload.get("reduction_mode")


def _sample_schema(payload: Mapping[str, Any]) -> tuple[Any, ...]:
    form = payload.get("form")
    if form == SCALAR_FORM:
        return (
            str(payload.get("name")),
            SCALAR_FORM,
            payload.get("reducer"),
            bool(payload.get("required")),
            bool(payload.get("integral")),
            payload.get("value") is None,
        )
    if form == RATIO_FORM:
        return (
            str(payload.get("name")),
            RATIO_FORM,
            None,
            True,
            False,
            False,
        )
    raise RuntimeContractError(
        "metric sample payload must declare a known sample form",
        code="runtime.metric_sample_form_invalid",
        context={"metric": payload.get("name"), "form": form},
    )


def _checked_sample_payloads(
    payload: Mapping[str, Any], *, rank: int
) -> tuple[Mapping[str, Any], ...]:
    samples = payload.get("samples")
    if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
        raise RuntimeContractError(
            "metric gather report samples must be a sequence",
            code="runtime.metric_gather_invalid",
            context={"rank": rank},
        )
    checked: list[Mapping[str, Any]] = []
    for item in samples:
        if not isinstance(item, Mapping):
            raise RuntimeContractError(
                "metric gather report sample must be a mapping",
                code="runtime.metric_gather_invalid",
                context={"rank": rank},
            )
        checked.append(item)
    return tuple(checked)


def _reduce_scalar(
    name: str,
    reducer: str,
    values: Sequence[tuple[int, float]],
    *,
    integral: bool,
) -> float:
    for rank, value in values:
        if integral:
            _checked_integral(value, name=name, rank=rank)
        if reducer == REDUCER_BOOL_ALL:
            _checked_bool_flag(value, name=name, rank=rank)
    if reducer == REDUCER_SUM:
        # Ordered rank-major accumulation: the reduced float must not depend on
        # a summation strategy, and this reproduces the pre-migration value
        # bit-for-bit for every already-summed family.
        total = 0.0
        for _, value in values:
            total += value
        return total
    if reducer == REDUCER_MAX:
        maximum = values[0][1]
        for _, value in values[1:]:
            if math.isnan(value) or math.isnan(maximum):
                # Deterministic non-finite propagation: an unordered value must
                # not depend on rank iteration order.
                return float("nan")
            if value > maximum:
                maximum = value
        return maximum
    if reducer == REDUCER_BOOL_ALL:
        return 1.0 if all(value == 1.0 for _, value in values) else 0.0
    if reducer == REDUCER_IDENTICAL:
        reference_rank, reference = values[0]
        for rank, value in values[1:]:
            if not _values_identical(reference, value):
                raise RuntimeContractError(
                    "a metric declared IDENTICAL must carry the same value on "
                    "every rank; it is a shared global value, not a rank-local "
                    "contribution to be averaged",
                    code="runtime.metric_identical_mismatch",
                    context={
                        "metric": name,
                        "reference_rank": reference_rank,
                        "reference_value": reference,
                        "rank": rank,
                        "value": value,
                    },
                )
        return reference
    raise RuntimeContractError(
        "metric sample requires one exact declared reducer",
        code="runtime.metric_reducer_undeclared",
        context={"metric": name, "declared_reducer": reducer},
    )


def _reduce_ratio(name: str, payloads: Sequence[tuple[int, Mapping[str, Any]]]) -> float:
    numerator = 0.0
    denominator = 0.0
    empty_value = payloads[0][1].get("empty_value")
    for rank, payload in payloads:
        rank_denominator = _checked_float(
            payload.get("denominator"), name=name, field="denominator"
        )
        if not math.isfinite(rank_denominator) or rank_denominator < 0.0:
            raise RuntimeContractError(
                "ratio metric sample requires a finite non-negative denominator",
                code="runtime.metric_ratio_denominator_invalid",
                context={"metric": name, "rank": rank, "denominator": rank_denominator},
            )
        if payload.get("empty_value") != empty_value:
            raise RuntimeContractError(
                "ratio metric samples disagree on the declared empty-denominator value",
                code="runtime.metric_schema_mismatch",
                context={"metric": name, "rank": rank},
            )
        numerator += _checked_float(
            payload.get("numerator"), name=name, field="numerator"
        )
        denominator += rank_denominator
    if denominator == 0.0:
        if empty_value is None:
            raise RuntimeContractError(
                "ratio metric sample reduced to a zero global denominator without a "
                "declared empty-denominator value",
                code="runtime.metric_ratio_zero_denominator",
                context={"metric": name},
            )
        return float(empty_value)
    return numerator / denominator


def _reduce_accuracy(
    payloads: Sequence[tuple[int, Any]],
) -> tuple[dict[str, int], str, str]:
    present = [(rank, item) for rank, item in payloads if isinstance(item, Mapping)]
    if len(present) != len(payloads):
        raise RuntimeContractError(
            "accuracy metric reduction requires accuracy_stats from every rank",
            code="runtime.accuracy_stats_missing",
            context={
                "missing_ranks": [
                    rank for rank, item in payloads if not isinstance(item, Mapping)
                ]
            },
        )
    reference_rank, reference = present[0]
    reducer = reference.get("reducer")
    top1_name = str(reference.get("top1_metric_name"))
    top5_name = str(reference.get("top5_metric_name"))
    checked: list[tuple[int, AccuracySufficientStats]] = []
    for rank, item in present:
        if (
            item.get("reducer") != reducer
            or str(item.get("top1_metric_name")) != top1_name
            or str(item.get("top5_metric_name")) != top5_name
        ):
            raise RuntimeContractError(
                "accuracy sufficient statistics disagree on their declared reducer "
                "or derived metric names",
                code="runtime.metric_schema_mismatch",
                context={"metric": "accuracy_stats", "rank": rank},
            )
        checked.append(
            (
                rank,
                AccuracySufficientStats(
                    reducer=str(reducer),
                    top1_correct=item.get("top1_correct"),
                    top5_correct=item.get("top5_correct"),
                    atom_count=item.get("atom_count"),
                    top1_metric_name=top1_name,
                    top5_metric_name=top5_name,
                ),
            )
        )
    if reducer == REDUCER_IDENTICAL:
        mismatched = [
            rank
            for rank, stats in checked[1:]
            if (stats.top1_correct, stats.top5_correct, stats.atom_count)
            != (
                checked[0][1].top1_correct,
                checked[0][1].top5_correct,
                checked[0][1].atom_count,
            )
        ]
        if mismatched:
            raise RuntimeContractError(
                "replicated accuracy_stats must be identical on every rank",
                code="runtime.accuracy_stats_replicated_mismatch",
                context={
                    "reference_rank": reference_rank,
                    "mismatched_ranks": mismatched,
                },
            )
        totals = {
            "top1_correct": checked[0][1].top1_correct,
            "top5_correct": checked[0][1].top5_correct,
            "atom_count": checked[0][1].atom_count,
        }
    else:
        totals = {"top1_correct": 0, "top5_correct": 0, "atom_count": 0}
        for _, stats in checked:
            totals["top1_correct"] += stats.top1_correct
            totals["top5_correct"] += stats.top5_correct
            totals["atom_count"] += stats.atom_count
    if totals["atom_count"] <= 0:
        raise RuntimeContractError(
            "accuracy metric reduction requires a positive summed atom count",
            code="runtime.accuracy_stats_zero_atoms",
            context={"metrics": [top1_name, top5_name]},
        )
    return totals, top1_name, top5_name


def reduce_rank_payloads(
    payloads: Sequence[Mapping[str, Any]], *, world_size: int
) -> ReducedMetricBatch:
    """Reduce one already-gathered payload per rank into global metric values.

    The caller owns the collective; this function performs no communication.
    It is used unchanged for world size one so a single-rank run validates the
    same schema a distributed run would.
    """

    if len(payloads) != world_size:
        raise RuntimeContractError(
            "metric gather must return exactly one report per world rank",
            code="runtime.metric_gather_count",
            context={
                "expected_report_count": world_size,
                "observed_report_count": len(payloads),
            },
        )
    by_rank: dict[int, Mapping[str, Any]] = {}
    expected_step: int | None = None
    expected_split: str | None = None
    expected_mode: Any = None
    for index, payload in enumerate(payloads):
        if not isinstance(payload, Mapping):
            raise RuntimeContractError(
                "metric gatherer must return mapping reports",
                code="runtime.metric_gather_invalid",
                context={"report_index": index, "value_type": type(payload).__name__},
            )
        rank, planned_step_id, split, reduction_mode = _payload_identity(
            payload, index=index
        )
        if rank < 0 or rank >= world_size or rank in by_rank:
            raise RuntimeContractError(
                "metric gather must contain one unique report per world rank",
                code="runtime.metric_gather_ranks",
                context={"report_index": index, "rank": rank},
            )
        if index == 0:
            expected_step, expected_split, expected_mode = (
                planned_step_id,
                split,
                reduction_mode,
            )
        if planned_step_id != expected_step:
            raise RuntimeContractError(
                "metric gather reports disagree on planned step",
                code="runtime.metric_gather_step",
                context={
                    "rank": rank,
                    "expected_planned_step_id": expected_step,
                    "observed_planned_step_id": planned_step_id,
                },
            )
        if split != expected_split:
            raise RuntimeContractError(
                "metric gather reports disagree on split",
                code="runtime.metric_gather_split",
                context={
                    "rank": rank,
                    "expected_split": expected_split,
                    "observed_split": split,
                },
            )
        if reduction_mode != expected_mode:
            raise RuntimeContractError(
                "metric gather reports disagree on the explicit reduction mode; the "
                "active mode must never be inferred ambiguously from payload shape",
                code="runtime.metric_gather_reduction_mode",
                context={
                    "rank": rank,
                    "expected_reduction_mode": expected_mode,
                    "observed_reduction_mode": reduction_mode,
                },
            )
        by_rank[rank] = payload
    if tuple(sorted(by_rank)) != tuple(range(world_size)):
        raise RuntimeContractError(
            "metric gather must contain every world rank",
            code="runtime.metric_gather_ranks",
            context={"observed_ranks": sorted(by_rank)},
        )

    samples_by_rank: dict[int, tuple[Mapping[str, Any], ...]] = {
        rank: _checked_sample_payloads(by_rank[rank], rank=rank)
        for rank in range(world_size)
    }
    reference_schema = tuple(
        _sample_schema(sample) for sample in samples_by_rank[0]
    )
    reference_names = tuple(entry[0] for entry in reference_schema)
    if len(set(reference_names)) != len(reference_names):
        raise RuntimeContractError(
            "metric batch names must be unique",
            code="runtime.metric_sample_duplicate",
            context={"metric_names": list(reference_names)},
        )
    for rank in range(1, world_size):
        schema = tuple(_sample_schema(sample) for sample in samples_by_rank[rank])
        names = tuple(entry[0] for entry in schema)
        if names != reference_names:
            raise RuntimeContractError(
                "metric gather reports disagree on metric keys",
                code="runtime.metric_gather_keys",
                context={
                    "rank": rank,
                    "expected_metric_keys": list(reference_names),
                    "observed_metric_keys": list(names),
                },
            )
        if schema != reference_schema:
            mismatched = [
                {
                    "metric": reference_names[index],
                    "expected": list(reference_schema[index]),
                    "observed": list(entry),
                }
                for index, entry in enumerate(schema)
                if entry != reference_schema[index]
            ]
            raise RuntimeContractError(
                "metric gather reports disagree on sample form, declared reducer, "
                "required status, or backend availability",
                code="runtime.metric_schema_mismatch",
                context={"rank": rank, "mismatched_metrics": mismatched},
            )

    metrics: dict[str, float | None] = {}
    per_rank_metrics: dict[str, dict[str, float]] = {
        str(rank): {} for rank in range(world_size)
    }
    for position, reference_sample in enumerate(samples_by_rank[0]):
        name = reference_names[position]
        form = reference_schema[position][1]
        if form == RATIO_FORM:
            metrics[name] = _reduce_ratio(
                name,
                [(rank, samples_by_rank[rank][position]) for rank in range(world_size)],
            )
            continue
        if reference_sample.get("value") is None:
            metrics[name] = None
            continue
        values: list[tuple[int, float]] = []
        for rank in range(world_size):
            value = _checked_float(
                samples_by_rank[rank][position].get("value"), name=name, field="value"
            )
            values.append((rank, value))
            per_rank_metrics[str(rank)][name] = value
        metrics[name] = _reduce_scalar(
            name,
            str(reference_schema[position][2]),
            values,
            integral=bool(reference_schema[position][4]),
        )

    accuracy_payloads = [
        (rank, by_rank[rank].get("accuracy_stats")) for rank in range(world_size)
    ]
    accuracy_stats: dict[str, int] | None = None
    if any(item is not None for _, item in accuracy_payloads):
        accuracy_stats, top1_name, top5_name = _reduce_accuracy(accuracy_payloads)
        atom_count = float(accuracy_stats["atom_count"])
        metrics[top1_name] = float(accuracy_stats["top1_correct"]) / atom_count
        metrics[top5_name] = float(accuracy_stats["top5_correct"]) / atom_count
    return ReducedMetricBatch(
        metrics={name: metrics[name] for name in sorted(metrics)},
        per_rank_metrics={
            rank: {key: values[key] for key in sorted(values)}
            for rank, values in per_rank_metrics.items()
        },
        accuracy_stats=accuracy_stats,
    )


ACCURACY_STAT_FIELDS = ("top1_correct", "top5_correct", "atom_count")


def checked_accuracy_stats(
    accuracy_stats: Mapping[str, Any] | None,
    *,
    reducer: str,
    reported_metrics: Mapping[str, Any] | None = None,
    top1_metric_name: str = "acc_top1",
    top5_metric_name: str = "acc_top5",
) -> AccuracySufficientStats | None:
    """Validate a producer's exact integer accuracy statistics.

    When the producer also reports already-divided accuracy ratios, they are
    checked against the integer statistics here, on the producing rank, before
    anything crosses the collective: the durable ratio is then always derived
    from the reduced integers, never from a rank-local float.
    """

    reported = reported_metrics or {}
    derived_names = (top1_metric_name, top5_metric_name)
    if accuracy_stats is None:
        if any(name in reported for name in derived_names):
            raise RuntimeContractError(
                "accuracy metrics require exact integer accuracy_stats; a plain "
                "rank mean must never silently substitute for the exact "
                "summed-integer ratio",
                code="runtime.accuracy_stats_missing",
                context={
                    "metrics": [name for name in derived_names if name in reported]
                },
            )
        return None
    if set(accuracy_stats) != set(ACCURACY_STAT_FIELDS):
        raise RuntimeContractError(
            "accuracy_stats must contain exactly the declared sufficient-statistic fields",
            code="runtime.accuracy_stats_fields",
            context={
                "expected_fields": sorted(ACCURACY_STAT_FIELDS),
                "observed_fields": sorted(str(field) for field in accuracy_stats),
            },
        )
    stats = AccuracySufficientStats(
        reducer=reducer,
        top1_correct=accuracy_stats["top1_correct"],
        top5_correct=accuracy_stats["top5_correct"],
        atom_count=accuracy_stats["atom_count"],
        top1_metric_name=top1_metric_name,
        top5_metric_name=top5_metric_name,
    )
    for metric_name, correct in (
        (top1_metric_name, stats.top1_correct),
        (top5_metric_name, stats.top5_correct),
    ):
        if metric_name not in reported:
            continue
        if stats.atom_count <= 0:
            raise RuntimeContractError(
                "accuracy metric requires a positive atom count",
                code="runtime.accuracy_stats_zero_atoms",
                context={"metric": metric_name},
            )
        expected = float(correct) / float(stats.atom_count)
        observed = _checked_float(
            reported[metric_name], name=metric_name, field="value"
        )
        if observed != expected:
            raise RuntimeContractError(
                "accuracy metric must match the ratio derived from exact integer stats",
                code="runtime.accuracy_metric_stats_mismatch",
                context={
                    "metric": metric_name,
                    "observed": observed,
                    "expected": expected,
                },
            )
    return stats


def _loss_term_declarations(
    loss_artifact: Mapping[str, Any],
    *,
    objective_reducer: str,
) -> dict[str, str]:
    """Names this planned-step loss telemetry declares, with their reducers.

    Names are CONSTRUCTED from the finalized loss artifact's own structure --
    its term list and its count mapping -- never parsed back out of a metric
    key. `src/losses/runner.py` is the calculation authority and this is the
    single declaration of what its published telemetry means across ranks, so
    train and eval can never classify the same family two different ways.

    * objective contributions (`loss/total`, `loss/<term>/raw`,
      `loss/<term>/weighted`, `loss/<term>/selected_count`) are this rank's own
      partial semantic contribution over the globally merged denominator, so
      they SUM wherever the planned-step window is partitioned and are
      IDENTICAL where every rank already holds the whole window;
    * `loss/<term>/segment_count` and the three `count/` denominator fields
      come from the merged cross-rank denominator resolved in
      `prepare_planned_step`, so they are already global on every rank;
    * `count/packs` and `count/examples` are rank-local disjoint work counts;
    * `finite/*` flags are per-rank binary indicators of a globally shared
      decision, so their conjunction is the only meaningful reduction.

    `loss/<term>/token_weighted_diag` is not listed here: it is an
    already-normalized per-atom average and is declared as a ratio sample by
    the caller below.
    """

    declarations: dict[str, str] = {"loss/total": objective_reducer}
    for term in loss_artifact.get("terms", ()):
        name = str(term.get("name"))
        declarations[f"loss/{name}/raw"] = objective_reducer
        declarations[f"loss/{name}/weighted"] = objective_reducer
        declarations[f"loss/{name}/selected_count"] = objective_reducer
        declarations[f"loss/{name}/segment_count"] = REDUCER_IDENTICAL
        declarations[f"finite/{name}"] = REDUCER_BOOL_ALL
    declarations["finite/total_loss"] = REDUCER_BOOL_ALL
    for count_name in (
        "count/supervised_atoms",
        "count/eligible_segments",
        "count/skipped_segments",
    ):
        declarations[count_name] = REDUCER_IDENTICAL
    for count_name in ("count/packs", "count/examples"):
        declarations[count_name] = objective_reducer
    return declarations


def loss_telemetry_batch(
    *,
    planned_step_id: int,
    split: str,
    loss_metrics: Mapping[str, Any],
    loss_artifact: Mapping[str, Any],
    partial_rank_contributions: bool,
    accuracy_stats: Mapping[str, int] | None = None,
    extra_samples: Sequence[ScalarSample | RatioSample] = (),
    reduction_mode: str | None = None,
) -> MetricBatch:
    """Declare one planned-step (or eval) loss-telemetry batch.

    `loss_metrics` supplies the authoritative published values; the finalized
    `loss_artifact` structure supplies the meaning. Every published value must
    be covered by a declaration or by an explicitly declared `extra_samples`
    entry; anything else fails closed rather than acquiring a default reducer.
    """

    objective_reducer = (
        REDUCER_SUM if partial_rank_contributions else REDUCER_IDENTICAL
    )
    declarations = _loss_term_declarations(
        loss_artifact, objective_reducer=objective_reducer
    )
    integral_names = {
        "count/supervised_atoms",
        "count/eligible_segments",
        "count/skipped_segments",
        "count/packs",
        "count/examples",
    }
    ratio_weights: dict[str, float] = {}
    for term in loss_artifact.get("terms", ()):
        name = str(term.get("name"))
        integral_names.add(f"loss/{name}/selected_count")
        integral_names.add(f"loss/{name}/segment_count")
        ratio_weights[f"loss/{name}/token_weighted_diag"] = float(
            term.get("selected_count", 0)
        )
    declared_extra = {sample.name for sample in extra_samples}
    accuracy = checked_accuracy_stats(
        accuracy_stats,
        reducer=objective_reducer,
        reported_metrics=loss_metrics,
    )
    derived_accuracy_names = (
        set()
        if accuracy is None
        else {accuracy.top1_metric_name, accuracy.top5_metric_name}
    )
    samples: list[ScalarSample | RatioSample] = list(extra_samples)
    for metric_name in sorted(str(name) for name in loss_metrics):
        if metric_name in derived_accuracy_names or metric_name in declared_extra:
            continue
        value = loss_metrics[metric_name]
        if metric_name in ratio_weights:
            if partial_rank_contributions:
                weight = ratio_weights[metric_name]
                local = 0.0 if value is None else float(value)
                samples.append(
                    RatioSample(
                        name=metric_name,
                        numerator=local * weight,
                        denominator=weight,
                        # Matches `_weighted_average` in `src/losses/runner.py`:
                        # a globally zero selected-token weight resolves to 0.0.
                        empty_value=0.0,
                    )
                )
                continue
            samples.append(
                ScalarSample(
                    name=metric_name,
                    reducer=REDUCER_IDENTICAL,
                    value=None if value is None else float(value),
                    required=value is not None,
                )
            )
            continue
        reducer = declarations.get(metric_name)
        if reducer is None:
            raise RuntimeContractError(
                "planned-step telemetry published a metric with no declared "
                "reducer; every durable scalar must declare exact reduction "
                "semantics at its producer before the metric collective runs",
                code="runtime.metric_reducer_undeclared",
                context={
                    "metric": metric_name,
                    "split": split,
                    "declared_metrics": sorted(declarations),
                },
            )
        samples.append(
            ScalarSample(
                name=metric_name,
                reducer=reducer,
                value=None if value is None else float(value),
                required=value is not None,
                integral=metric_name in integral_names and value is not None,
            )
        )
    return MetricBatch(
        planned_step_id=planned_step_id,
        split=split,
        samples=tuple(samples),
        accuracy=accuracy,
        reduction_mode=reduction_mode,
    )


__all__ = [
    "ACCURACY_REDUCERS",
    "ACCURACY_STAT_FIELDS",
    "checked_accuracy_stats",
    "loss_telemetry_batch",
    "AccuracySufficientStats",
    "IDENTICAL_ABS_TOLERANCE",
    "METRIC_PAYLOAD_KIND",
    "MetricBatch",
    "RATIO_FORM",
    "REDUCER_BOOL_ALL",
    "REDUCER_IDENTICAL",
    "REDUCER_MAX",
    "REDUCER_SUM",
    "ReducedMetricBatch",
    "RatioSample",
    "SCALAR_FORM",
    "SCALAR_REDUCERS",
    "ScalarSample",
    "reduce_rank_payloads",
]
