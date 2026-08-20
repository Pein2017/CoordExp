"""Task 2.1/2.2: typed metric samples, batches, and explicit reducers.

RED-first module for `src/runtime/metrics.py`: every test here was observed
failing before the module existed (import error), and each value/rejection
assertion below was additionally observed failing against the superseded
name-table reducer in `src/runtime/train_runtime.py` (see the wave-2 RED
receipt: implicit mean produced 2.0 for a summed count pair, 2.5 for a
count-weighted ratio, and 0.5 for a split finite flag, and never rejected an
undeclared metric).
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from src.common.errors import RuntimeContractError
from src.runtime.metrics import (
    AccuracySufficientStats,
    MetricBatch,
    RatioSample,
    ScalarSample,
    reduce_rank_payloads,
)

SUM = "SUM"
MAX = "MAX"
IDENTICAL = "IDENTICAL"
BOOL_ALL = "BOOL_ALL"


def _payloads(
    *rank_batches: MetricBatch,
) -> list[dict[str, Any]]:
    world_size = len(rank_batches)
    return [
        batch.to_rank_payload(rank=rank, world_size=world_size)
        for rank, batch in enumerate(rank_batches)
    ]


def _batch(
    *samples: ScalarSample | RatioSample,
    accuracy: AccuracySufficientStats | None = None,
    split: str = "train",
    planned_step_id: int = 7,
    reduction_mode: str | None = None,
) -> MetricBatch:
    return MetricBatch(
        planned_step_id=planned_step_id,
        split=split,
        samples=samples,
        accuracy=accuracy,
        reduction_mode=reduction_mode,
    )


def _reduce(*rank_batches: MetricBatch) -> Any:
    return reduce_rank_payloads(
        _payloads(*rank_batches), world_size=len(rank_batches)
    )


# ---------------------------------------------------------------------------
# Scalar samples and their four declared reducers
# ---------------------------------------------------------------------------


def test_sum_reducer_adds_rank_local_contributions() -> None:
    reduced = _reduce(
        _batch(ScalarSample("count/packs", SUM, 1.0, integral=True)),
        _batch(ScalarSample("count/packs", SUM, 3.0, integral=True)),
    )
    assert reduced.metrics["count/packs"] == 4.0


def test_max_reducer_takes_the_slowest_rank_not_the_mean() -> None:
    reduced = _reduce(
        _batch(ScalarSample("step_duration_seconds", MAX, 0.4)),
        _batch(ScalarSample("step_duration_seconds", MAX, 0.9)),
    )
    assert reduced.metrics["step_duration_seconds"] == 0.9


def test_identical_reducer_returns_the_shared_global_value() -> None:
    reduced = _reduce(
        _batch(ScalarSample("lr/group_0", IDENTICAL, 1e-5)),
        _batch(ScalarSample("lr/group_0", IDENTICAL, 1e-5)),
    )
    assert reduced.metrics["lr/group_0"] == 1e-5


def test_identical_reducer_rejects_a_cross_rank_value_mismatch() -> None:
    with pytest.raises(RuntimeContractError) as exc_info:
        _reduce(
            _batch(ScalarSample("lr/group_0", IDENTICAL, 1e-5)),
            _batch(ScalarSample("lr/group_0", IDENTICAL, 2e-5)),
        )
    assert exc_info.value.code == "runtime.metric_identical_mismatch"
    assert exc_info.value.context["metric"] == "lr/group_0"


def test_identical_reducer_accepts_a_shared_non_finite_value() -> None:
    reduced = _reduce(
        _batch(ScalarSample("loss/total", IDENTICAL, float("nan"))),
        _batch(ScalarSample("loss/total", IDENTICAL, float("nan"))),
    )
    assert math.isnan(reduced.metrics["loss/total"])


def test_bool_all_reducer_is_a_conjunction_never_a_fraction() -> None:
    reduced = _reduce(
        _batch(ScalarSample("finite/total_loss", BOOL_ALL, 1.0)),
        _batch(ScalarSample("finite/total_loss", BOOL_ALL, 0.0)),
    )
    assert reduced.metrics["finite/total_loss"] == 0.0

    all_finite = _reduce(
        _batch(ScalarSample("finite/total_loss", BOOL_ALL, 1.0)),
        _batch(ScalarSample("finite/total_loss", BOOL_ALL, 1.0)),
    )
    assert all_finite.metrics["finite/total_loss"] == 1.0


def test_bool_all_reducer_rejects_a_non_boolean_value() -> None:
    with pytest.raises(RuntimeContractError) as exc_info:
        ScalarSample("finite/total_loss", BOOL_ALL, 0.5)
    assert exc_info.value.code == "runtime.metric_bool_all_value_invalid"


def test_integral_sample_rejects_a_fractional_count() -> None:
    with pytest.raises(RuntimeContractError) as exc_info:
        ScalarSample("count/packs", SUM, 1.5, integral=True)
    assert exc_info.value.code == "runtime.metric_count_value_invalid"


def test_max_reducer_propagates_a_non_finite_value_independent_of_rank_order() -> None:
    nan_first = _reduce(
        _batch(ScalarSample("step_duration_seconds", MAX, float("nan"))),
        _batch(ScalarSample("step_duration_seconds", MAX, 0.4)),
    )
    nan_second = _reduce(
        _batch(ScalarSample("step_duration_seconds", MAX, 0.4)),
        _batch(ScalarSample("step_duration_seconds", MAX, float("nan"))),
    )
    assert math.isnan(nan_first.metrics["step_duration_seconds"])
    assert math.isnan(nan_second.metrics["step_duration_seconds"])


# ---------------------------------------------------------------------------
# No implicit reducer, ever
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("declared", ["MEAN", "ALL", "mean", "", None, "avg"])
def test_a_metric_without_an_exact_declared_reducer_is_rejected(
    declared: Any,
) -> None:
    with pytest.raises(RuntimeContractError) as exc_info:
        ScalarSample("unclassified/scientific_metric", declared, 1.0)
    assert exc_info.value.code == "runtime.metric_reducer_undeclared"
    assert "MEAN" not in exc_info.value.context["supported_reducers"]
    assert "ALL" not in exc_info.value.context["supported_reducers"]


def test_reduction_rejects_a_gathered_sample_with_an_undeclared_reducer() -> None:
    # A peer rank that fabricates a payload the typed constructor would have
    # rejected must still fail closed, never fall back to a mean.
    payloads = _payloads(
        _batch(ScalarSample("loss/total", SUM, 1.0)),
        _batch(ScalarSample("loss/total", SUM, 3.0)),
    )
    payloads[1]["samples"][0]["reducer"] = "MEAN"
    with pytest.raises(RuntimeContractError) as exc_info:
        reduce_rank_payloads(payloads, world_size=2)
    assert exc_info.value.code == "runtime.metric_schema_mismatch"


def test_reduction_rejects_an_unknown_sample_form() -> None:
    payloads = _payloads(_batch(ScalarSample("loss/total", SUM, 1.0)))
    payloads[0]["samples"][0]["form"] = "histogram"
    with pytest.raises(RuntimeContractError) as exc_info:
        reduce_rank_payloads(payloads, world_size=1)
    assert exc_info.value.code == "runtime.metric_sample_form_invalid"


# ---------------------------------------------------------------------------
# Ratio samples: sum-before-divide
# ---------------------------------------------------------------------------


def test_ratio_sample_sums_numerators_and_denominators_before_dividing() -> None:
    reduced = _reduce(
        _batch(RatioSample("loss/base_ce/token_weighted_diag", 1.0 * 1, 1)),
        _batch(RatioSample("loss/base_ce/token_weighted_diag", 4.0 * 3, 3)),
    )
    assert reduced.metrics["loss/base_ce/token_weighted_diag"] == (1.0 + 12.0) / 4.0
    # The rejected alternative -- the mean of the two rank-local ratios -- is a
    # different number whenever the rank denominators differ.
    assert reduced.metrics["loss/base_ce/token_weighted_diag"] != (1.0 + 4.0) / 2.0


def test_ratio_sample_with_a_globally_zero_denominator_uses_its_declared_value() -> (
    None
):
    reduced = _reduce(
        _batch(RatioSample("loss/coord/token_weighted_diag", 0.0, 0, empty_value=0.0)),
        _batch(RatioSample("loss/coord/token_weighted_diag", 0.0, 0, empty_value=0.0)),
    )
    assert reduced.metrics["loss/coord/token_weighted_diag"] == 0.0


def test_ratio_sample_without_a_declared_empty_value_fails_closed() -> None:
    with pytest.raises(RuntimeContractError) as exc_info:
        _reduce(
            _batch(RatioSample("acc/pooled", 0.0, 0)),
            _batch(RatioSample("acc/pooled", 0.0, 0)),
        )
    assert exc_info.value.code == "runtime.metric_ratio_zero_denominator"


def test_ratio_sample_rejects_a_negative_denominator() -> None:
    with pytest.raises(RuntimeContractError) as exc_info:
        RatioSample("acc/pooled", 1.0, -1.0)
    assert exc_info.value.code == "runtime.metric_ratio_denominator_invalid"


# ---------------------------------------------------------------------------
# Required versus backend-unavailable fields
# ---------------------------------------------------------------------------


def test_required_sample_cannot_be_unavailable() -> None:
    with pytest.raises(RuntimeContractError) as exc_info:
        ScalarSample("loss/total", SUM, None)
    assert exc_info.value.code == "runtime.metric_sample_required_unavailable"


def test_conditional_sample_reduces_to_an_explicit_unavailable_value() -> None:
    reduced = _reduce(
        _batch(ScalarSample("input_h2d_seconds", MAX, None, required=False)),
        _batch(ScalarSample("input_h2d_seconds", MAX, None, required=False)),
    )
    assert reduced.metrics["input_h2d_seconds"] is None


def test_ranks_disagreeing_on_availability_fail_closed() -> None:
    with pytest.raises(RuntimeContractError) as exc_info:
        _reduce(
            _batch(ScalarSample("input_h2d_seconds", MAX, 0.01, required=False)),
            _batch(ScalarSample("input_h2d_seconds", MAX, None, required=False)),
        )
    assert exc_info.value.code == "runtime.metric_schema_mismatch"


def test_ranks_disagreeing_on_required_status_fail_closed() -> None:
    with pytest.raises(RuntimeContractError) as exc_info:
        _reduce(
            _batch(ScalarSample("resource/gpu_bytes", MAX, 4.0, required=True)),
            _batch(ScalarSample("resource/gpu_bytes", MAX, 4.0, required=False)),
        )
    assert exc_info.value.code == "runtime.metric_schema_mismatch"


# ---------------------------------------------------------------------------
# Cross-rank schema agreement
# ---------------------------------------------------------------------------


def test_ranks_disagreeing_on_metric_names_fail_closed() -> None:
    with pytest.raises(RuntimeContractError) as exc_info:
        _reduce(
            _batch(ScalarSample("loss/total", SUM, 1.0)),
            _batch(ScalarSample("loss/other", SUM, 1.0)),
        )
    assert exc_info.value.code == "runtime.metric_gather_keys"


def test_ranks_disagreeing_on_the_declared_reducer_fail_closed() -> None:
    with pytest.raises(RuntimeContractError) as exc_info:
        _reduce(
            _batch(ScalarSample("count/packs", SUM, 1.0)),
            _batch(ScalarSample("count/packs", MAX, 1.0)),
        )
    assert exc_info.value.code == "runtime.metric_schema_mismatch"
    assert exc_info.value.context["mismatched_metrics"][0]["metric"] == "count/packs"


def test_ranks_disagreeing_on_the_sample_form_fail_closed() -> None:
    with pytest.raises(RuntimeContractError) as exc_info:
        _reduce(
            _batch(ScalarSample("loss/a/token_weighted_diag", SUM, 1.0)),
            _batch(RatioSample("loss/a/token_weighted_diag", 1.0, 1.0)),
        )
    assert exc_info.value.code == "runtime.metric_schema_mismatch"


def test_ranks_disagreeing_on_planned_step_split_or_mode_fail_closed() -> None:
    with pytest.raises(RuntimeContractError) as step_exc:
        _reduce(
            _batch(ScalarSample("loss/total", SUM, 1.0), planned_step_id=7),
            _batch(ScalarSample("loss/total", SUM, 1.0), planned_step_id=8),
        )
    assert step_exc.value.code == "runtime.metric_gather_step"

    with pytest.raises(RuntimeContractError) as split_exc:
        _reduce(
            _batch(ScalarSample("loss/total", SUM, 1.0), split="train"),
            _batch(ScalarSample("loss/total", SUM, 1.0), split="eval"),
        )
    assert split_exc.value.code == "runtime.metric_gather_split"

    with pytest.raises(RuntimeContractError) as mode_exc:
        _reduce(
            _batch(ScalarSample("loss/total", SUM, 1.0), reduction_mode=None),
            _batch(
                ScalarSample("loss/total", SUM, 1.0),
                reduction_mode="disjoint_shard",
            ),
        )
    assert mode_exc.value.code == "runtime.metric_gather_reduction_mode"


def test_missing_or_duplicated_rank_reports_fail_closed() -> None:
    payloads = _payloads(
        _batch(ScalarSample("loss/total", SUM, 1.0)),
        _batch(ScalarSample("loss/total", SUM, 1.0)),
    )
    with pytest.raises(RuntimeContractError) as count_exc:
        reduce_rank_payloads(payloads[:1], world_size=2)
    assert count_exc.value.code == "runtime.metric_gather_count"

    payloads[1]["rank"] = 0
    with pytest.raises(RuntimeContractError) as rank_exc:
        reduce_rank_payloads(payloads, world_size=2)
    assert rank_exc.value.code == "runtime.metric_gather_ranks"


def test_duplicate_metric_names_are_rejected_in_the_batch() -> None:
    with pytest.raises(RuntimeContractError) as exc_info:
        _batch(
            ScalarSample("loss/total", SUM, 1.0),
            ScalarSample("loss/total", MAX, 1.0),
        )
    assert exc_info.value.code == "runtime.metric_sample_duplicate"


# ---------------------------------------------------------------------------
# Accuracy sufficient statistics
# ---------------------------------------------------------------------------


def test_accuracy_stats_reduce_as_a_pooled_ratio_of_summed_integers() -> None:
    reduced = _reduce(
        _batch(
            accuracy=AccuracySufficientStats(
                reducer=SUM, top1_correct=1, top5_correct=2, atom_count=3
            )
        ),
        _batch(
            accuracy=AccuracySufficientStats(
                reducer=SUM, top1_correct=1, top5_correct=2, atom_count=6
            )
        ),
    )
    assert reduced.accuracy_stats == {
        "top1_correct": 2,
        "top5_correct": 4,
        "atom_count": 9,
    }
    assert reduced.metrics["acc_top1"] == pytest.approx(2 / 9)
    assert reduced.metrics["acc_top5"] == pytest.approx(4 / 9)
    # The rejected alternative: the mean of the two rank-local ratios.
    assert reduced.metrics["acc_top1"] != pytest.approx(((1 / 3) + (1 / 6)) / 2)


def test_replicated_accuracy_stats_are_identical_not_summed() -> None:
    reduced = _reduce(
        _batch(
            accuracy=AccuracySufficientStats(
                reducer=IDENTICAL, top1_correct=1, top5_correct=2, atom_count=4
            ),
            split="eval",
        ),
        _batch(
            accuracy=AccuracySufficientStats(
                reducer=IDENTICAL, top1_correct=1, top5_correct=2, atom_count=4
            ),
            split="eval",
        ),
    )
    assert reduced.accuracy_stats == {
        "top1_correct": 1,
        "top5_correct": 2,
        "atom_count": 4,
    }
    assert reduced.metrics["acc_top1"] == 0.25


def test_replicated_accuracy_stats_reject_a_cross_rank_mismatch() -> None:
    with pytest.raises(RuntimeContractError) as exc_info:
        _reduce(
            _batch(
                accuracy=AccuracySufficientStats(
                    reducer=IDENTICAL, top1_correct=1, top5_correct=2, atom_count=4
                ),
                split="eval",
            ),
            _batch(
                accuracy=AccuracySufficientStats(
                    reducer=IDENTICAL, top1_correct=2, top5_correct=2, atom_count=4
                ),
                split="eval",
            ),
        )
    assert exc_info.value.code == "runtime.accuracy_stats_replicated_mismatch"


def test_accuracy_stats_missing_on_one_rank_fail_closed() -> None:
    payloads = _payloads(
        _batch(
            accuracy=AccuracySufficientStats(
                reducer=SUM, top1_correct=1, top5_correct=1, atom_count=2
            )
        ),
        _batch(
            accuracy=AccuracySufficientStats(
                reducer=SUM, top1_correct=1, top5_correct=1, atom_count=2
            )
        ),
    )
    payloads[1]["accuracy_stats"] = None
    with pytest.raises(RuntimeContractError) as exc_info:
        reduce_rank_payloads(payloads, world_size=2)
    assert exc_info.value.code == "runtime.accuracy_stats_missing"


@pytest.mark.parametrize("malformed", [1.0, True, -1])
def test_accuracy_stats_reject_non_integer_fields(malformed: Any) -> None:
    with pytest.raises(RuntimeContractError) as exc_info:
        AccuracySufficientStats(
            reducer=SUM, top1_correct=malformed, top5_correct=1, atom_count=2
        )
    assert exc_info.value.code == "runtime.accuracy_stats_field_type"


def test_accuracy_stats_reject_correct_exceeding_the_atom_count() -> None:
    with pytest.raises(RuntimeContractError) as exc_info:
        AccuracySufficientStats(
            reducer=SUM, top1_correct=3, top5_correct=1, atom_count=2
        )
    assert exc_info.value.code == "runtime.accuracy_stats_correct_exceeds_atoms"


def test_accuracy_stats_reject_a_zero_global_atom_count() -> None:
    with pytest.raises(RuntimeContractError) as exc_info:
        _reduce(
            _batch(
                accuracy=AccuracySufficientStats(
                    reducer=SUM, top1_correct=0, top5_correct=0, atom_count=0
                )
            )
        )
    assert exc_info.value.code == "runtime.accuracy_stats_zero_atoms"


def test_accuracy_stats_require_a_declared_reducer() -> None:
    with pytest.raises(RuntimeContractError) as exc_info:
        AccuracySufficientStats(
            reducer=BOOL_ALL, top1_correct=1, top5_correct=1, atom_count=2
        )
    assert exc_info.value.code == "runtime.metric_reducer_undeclared"


# ---------------------------------------------------------------------------
# World size one runs the identical validation and reduction path
# ---------------------------------------------------------------------------


def test_world_size_one_reduction_returns_rank_local_values() -> None:
    reduced = _reduce(
        _batch(
            ScalarSample("count/packs", SUM, 2.0, integral=True),
            ScalarSample("step_duration_seconds", MAX, 0.5),
            ScalarSample("finite/total_loss", BOOL_ALL, 1.0),
            RatioSample("loss/base_ce/token_weighted_diag", 6.0, 3.0),
        )
    )
    assert reduced.metrics == {
        "count/packs": 2.0,
        "finite/total_loss": 1.0,
        "loss/base_ce/token_weighted_diag": 2.0,
        "step_duration_seconds": 0.5,
    }


def test_world_size_one_still_rejects_an_undeclared_reducer() -> None:
    payloads = _payloads(_batch(ScalarSample("loss/total", SUM, 1.0)))
    payloads[0]["samples"][0]["reducer"] = "MEAN"
    with pytest.raises(RuntimeContractError) as exc_info:
        reduce_rank_payloads(payloads, world_size=1)
    assert exc_info.value.code == "runtime.metric_reducer_undeclared"


def test_per_rank_scalar_projection_stays_available_for_lifecycle_accounting() -> None:
    reduced = _reduce(
        _batch(ScalarSample("resource/cpu_max_rss_bytes", MAX, 10.0)),
        _batch(ScalarSample("resource/cpu_max_rss_bytes", MAX, 20.0)),
    )
    assert reduced.per_rank_metrics == {
        "0": {"resource/cpu_max_rss_bytes": 10.0},
        "1": {"resource/cpu_max_rss_bytes": 20.0},
    }
    assert reduced.metrics["resource/cpu_max_rss_bytes"] == 20.0


def test_samples_and_batches_are_immutable() -> None:
    sample = ScalarSample("loss/total", SUM, 1.0)
    batch = _batch(sample)
    with pytest.raises(Exception):
        sample.value = 2.0  # type: ignore[misc]
    with pytest.raises(Exception):
        batch.split = "eval"  # type: ignore[misc]
