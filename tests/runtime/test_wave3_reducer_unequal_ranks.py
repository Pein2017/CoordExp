"""Wave-3 task 3.5: train and forward-eval reducers with unequal ranks.

Once per-term telemetry stops inheriting the backend mean-gradient
compensation (task 3.2), each rank reports its own *partial* semantic
contribution to the global planned-step value. The cross-rank reducer must
therefore SUM those contributions wherever the denominators were globally
merged (train at any world size, and sharded forward eval), and must keep
plain-mean semantics for replicated eval where every rank already holds the
identical global value.

`acc_top1`/`acc_top5` must stay pooled ratios of summed integer sufficient
statistics, never a mean of per-rank ratios, and must equal the rank-local
value at world size one.

No collective is added, removed, or reordered by any of this: every
assertion below rides the single existing metric gather.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from src.common.errors import RuntimeContractError
from src.config.models import RuntimeBatchResolution, RuntimeConfig
from src.runtime.train_runtime import TrainRuntime


class _PeerGatherer:
    """One report per rank, in rank order - the production gather shape."""

    def __init__(self, *peers: dict[str, Any]) -> None:
        self.peers = peers
        self.calls = 0

    def __call__(self, local_report: dict[str, Any]) -> tuple[Any, ...]:
        self.calls += 1
        reports = [local_report]
        for index, peer in enumerate(self.peers, start=1):
            reports.append({**local_report, "rank": index, **peer})
        return tuple(reports)


class _Accelerator:
    def __init__(self, *, process_index: int = 0, num_processes: int = 1) -> None:
        self.process_index = process_index
        self.num_processes = num_processes
        self.device = torch.device("cpu")
        self.is_main_process = process_index == 0
        self.distributed_type = SimpleNamespace(name="NO")
        self.gradient_accumulation_steps = 1
        self.mixed_precision = "no"
        self.scaler = None

    def prepare(self, *objects: Any) -> tuple[Any, ...]:
        return objects


def _runtime(*, world_size: int, gatherer: Any | None = None) -> TrainRuntime:
    return TrainRuntime(
        runtime_config=RuntimeConfig.model_validate(
            {"seed": 17, "determinism": {"mode": "legacy"}}
        ),
        runtime_batch=RuntimeBatchResolution(
            world_size=world_size,
            effective_batch_size=world_size,
            resolved_grad_accum_steps=1,
        ),
        model=torch.nn.Linear(1, 1),
        optimizer=None,
        scheduler=None,
        expected_mixed_precision="no",
        accelerator=_Accelerator(num_processes=world_size),
        rank_report_gatherer=gatherer,
    )


def test_train_reducer_sums_unequal_rank_local_semantic_contributions() -> None:
    """Two ranks, unequal rank-local denominators, one global objective.

    Rank 0 owns 1 of 4 globally eligible segments and rank 1 owns 3, so the
    rank-local semantic contributions are unequal partial values whose SUM is
    the global planned-step value. A plain mean would report exactly half of
    the true objective at world size two.
    """

    gatherer = _PeerGatherer(
        {
            "metrics": {
                "loss/base_ce/raw": 2.4,
                "loss/base_ce/weighted": 2.4,
                "loss/base_ce/selected_count": 4.0,
                "loss/token_type_gate/raw": 3.0,
                "loss/token_type_gate/weighted": 0.3,
                "loss/total": 2.7,
                "loss/base_ce/segment_count": 4.0,
                "loss/base_ce/token_weighted_diag": 1.5,
                "count/eligible_segments": 4.0,
            },
            "accuracy_stats": {
                "top1_correct": 3,
                "top5_correct": 5,
                "atom_count": 6,
            },
        }
    )
    runtime = _runtime(world_size=2, gatherer=gatherer)

    reduced = runtime.gather_metrics(
        {
            "loss/base_ce/raw": 0.8,
            "loss/base_ce/weighted": 0.8,
            "loss/base_ce/selected_count": 2.0,
            "loss/token_type_gate/raw": 1.0,
            "loss/token_type_gate/weighted": 0.1,
            "loss/total": 0.9,
            "loss/base_ce/segment_count": 4.0,
            "loss/base_ce/token_weighted_diag": 0.5,
            "count/eligible_segments": 4.0,
        },
        planned_step_id=7,
        split="train",
    )["metrics"]

    # raw/weighted aggregation: exact sum of the rank-local contributions.
    # The Wave-4 rename must keep BOTH explicit families classified as partial
    # objective contributions -- a `/raw` key that fell through to the mean
    # branch would silently report half the global raw value at world size two.
    assert reduced["loss/base_ce/raw"] == pytest.approx(3.2)
    assert reduced["loss/token_type_gate/raw"] == pytest.approx(4.0)
    assert reduced["loss/base_ce/weighted"] == pytest.approx(3.2)
    assert reduced["loss/token_type_gate/weighted"] == pytest.approx(0.4)
    # Rank-local selected counts are partial too: summed, never averaged.
    assert reduced["loss/base_ce/selected_count"] == pytest.approx(6.0)
    assert reduced["loss/total"] == pytest.approx(3.6)
    # ... and never the plain mean it used to be.
    assert reduced["loss/total"] != pytest.approx(1.8)
    # Already-global fields stay identical, never summed (no double scaling).
    assert reduced["loss/base_ce/segment_count"] == pytest.approx(4.0)
    assert reduced["count/eligible_segments"] == pytest.approx(4.0)
    # Token-weighted diagnostics are per-rank means, not partial sums.
    assert reduced["loss/base_ce/token_weighted_diag"] == pytest.approx(1.0)
    assert gatherer.calls == 1


def test_train_reducer_total_equals_sum_of_reduced_weighted_terms() -> None:
    """`loss/total` must stay the sum of the reduced weighted objective terms."""

    gatherer = _PeerGatherer(
        {
            "metrics": {
                "loss/base_ce/weighted": 2.4,
                "loss/coord_gaussian_rps/weighted": 0.75,
                "loss/total": 3.15,
            }
        }
    )
    runtime = _runtime(world_size=2, gatherer=gatherer)
    reduced = runtime.gather_metrics(
        {
            "loss/base_ce/weighted": 0.8,
            "loss/coord_gaussian_rps/weighted": 0.25,
            "loss/total": 1.05,
        },
        planned_step_id=7,
        split="train",
    )["metrics"]

    assert reduced["loss/total"] == pytest.approx(
        reduced["loss/base_ce/weighted"] + reduced["loss/coord_gaussian_rps/weighted"]
    )
    assert reduced["loss/total"] == pytest.approx(4.2)


def test_world_size_one_train_reduction_equals_rank_local_values() -> None:
    runtime = _runtime(world_size=1)
    reduced = runtime.gather_metrics(
        {"loss/base_ce/weighted": 0.8, "loss/total": 0.9, "acc_top1": 0.5, "acc_top5": 1.0},
        planned_step_id=7,
        split="train",
        accuracy_stats={"top1_correct": 1, "top5_correct": 2, "atom_count": 2},
    )
    assert reduced["metrics"]["loss/base_ce/weighted"] == pytest.approx(0.8)
    assert reduced["metrics"]["loss/total"] == pytest.approx(0.9)
    assert reduced["metrics"]["acc_top1"] == pytest.approx(0.5)
    assert reduced["metrics"]["acc_top5"] == pytest.approx(1.0)
    assert reduced["accuracy_stats"] == {
        "top1_correct": 1,
        "top5_correct": 2,
        "atom_count": 2,
    }


def test_replicated_eval_reduction_keeps_identical_global_values_unsummed() -> None:
    """Replicated eval never partitions the window, so summing would double."""

    gatherer = _PeerGatherer(
        {
            "metrics": {"loss/base_ce/weighted": 1.25, "loss/total": 1.25},
            "accuracy_stats": {
                "top1_correct": 2,
                "top5_correct": 3,
                "atom_count": 4,
            },
        }
    )
    runtime = _runtime(world_size=2, gatherer=gatherer)
    reduced = runtime.gather_metrics(
        {"loss/base_ce/weighted": 1.25, "loss/total": 1.25},
        planned_step_id=7,
        split="eval",
    )["metrics"]

    assert reduced["loss/base_ce/weighted"] == pytest.approx(1.25)
    assert reduced["loss/total"] == pytest.approx(1.25)
    assert reduced["loss/total"] != pytest.approx(2.5)


def test_sharded_eval_reduction_sums_partial_semantic_contributions() -> None:
    gatherer = _PeerGatherer(
        {
            "metrics": {
                "loss/base_ce/weighted": 2.4,
                "loss/total": 2.4,
                "loss/base_ce/segment_count": 4.0,
                "count/supervised_atoms": 6.0,
                "example_count": 3.0,
                "pack_count": 2.0,
            },
            "accuracy_stats": {
                "top1_correct": 3,
                "top5_correct": 5,
                "atom_count": 6,
            },
        }
    )
    runtime = _runtime(world_size=2, gatherer=gatherer)
    reduced = runtime.gather_metrics(
        {
            "loss/base_ce/weighted": 0.8,
            "loss/total": 0.8,
            "loss/base_ce/segment_count": 4.0,
            "count/supervised_atoms": 6.0,
            "example_count": 2.0,
            "pack_count": 1.0,
        },
        planned_step_id=7,
        split="eval",
        reduction_mode="disjoint_shard",
    )["metrics"]

    assert reduced["loss/base_ce/weighted"] == pytest.approx(3.2)
    assert reduced["loss/total"] == pytest.approx(3.2)
    assert reduced["loss/base_ce/segment_count"] == pytest.approx(4.0)
    assert reduced["count/supervised_atoms"] == pytest.approx(6.0)
    assert reduced["example_count"] == pytest.approx(5.0)
    assert reduced["pack_count"] == pytest.approx(3.0)


def test_accuracy_reduces_as_pooled_ratio_of_summed_integer_statistics() -> None:
    """Unequal rank-local atom counts: pooled ratio, never mean-of-ratios."""

    gatherer = _PeerGatherer(
        {
            "metrics": {"acc_top1": 1.0, "acc_top5": 1.0, "loss/total": 1.0},
            "accuracy_stats": {
                "top1_correct": 9,
                "top5_correct": 9,
                "atom_count": 9,
            },
        }
    )
    runtime = _runtime(world_size=2, gatherer=gatherer)
    gathered = runtime.gather_metrics(
        {"acc_top1": 0.0, "acc_top5": 0.0, "loss/total": 1.0},
        planned_step_id=7,
        split="train",
        accuracy_stats={"top1_correct": 0, "top5_correct": 0, "atom_count": 1},
    )
    reduced = gathered["metrics"]

    # Pooled: (0 + 9) / (1 + 9) = 0.9. Mean of ratios would be 0.5.
    assert reduced["acc_top1"] == pytest.approx(0.9)
    assert reduced["acc_top5"] == pytest.approx(0.9)
    assert reduced["acc_top1"] != pytest.approx(0.5)
    assert gathered["accuracy_stats"] == {
        "top1_correct": 9,
        "top5_correct": 9,
        "atom_count": 10,
    }
    # Accuracy is a pooled ratio, NOT one of the summed objective keys.
    assert reduced["acc_top1"] <= 1.0


def test_accuracy_reduction_rejects_rank_ratio_inconsistent_with_its_statistics() -> (
    None
):
    gatherer = _PeerGatherer(
        {
            "metrics": {"acc_top1": 0.25, "acc_top5": 1.0, "loss/total": 1.0},
            "accuracy_stats": {
                "top1_correct": 9,
                "top5_correct": 9,
                "atom_count": 9,
            },
        }
    )
    runtime = _runtime(world_size=2, gatherer=gatherer)
    with pytest.raises(RuntimeContractError) as exc_info:
        runtime.gather_metrics(
            {"acc_top1": 0.0, "acc_top5": 0.0, "loss/total": 1.0},
            planned_step_id=7,
            split="train",
            accuracy_stats={"top1_correct": 0, "top5_correct": 0, "atom_count": 1},
        )
    assert exc_info.value.code.startswith("runtime.accuracy")
