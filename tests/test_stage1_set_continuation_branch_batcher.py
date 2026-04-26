from __future__ import annotations

import builtins

from src.trainers.stage1_set_continuation.branch_batcher import (
    BranchBatchWorkItem,
    plan_smart_branch_batches,
)


def test_smart_branch_batcher_respects_row_and_token_caps() -> None:
    plan = plan_smart_branch_batches(
        [
            BranchBatchWorkItem(index=0, sequence_length=80, suffix_keep=20),
            BranchBatchWorkItem(index=1, sequence_length=75, suffix_keep=20),
            BranchBatchWorkItem(index=2, sequence_length=30, suffix_keep=12),
            BranchBatchWorkItem(index=3, sequence_length=25, suffix_keep=12),
        ],
        max_branch_rows=2,
        max_branch_tokens=160,
    )

    flattened = [item.index for batch in plan.batches for item in batch.items]
    assert sorted(flattened) == [0, 1, 2, 3]
    assert plan.batch_count == len(plan.batches)
    assert plan.scheduler in {"constant_volume", "deterministic_fallback"}
    for batch in plan.batches:
        assert len(batch.items) <= 2
        assert batch.padded_token_volume <= 160


def test_smart_branch_batcher_reports_padding_fraction() -> None:
    plan = plan_smart_branch_batches(
        [
            BranchBatchWorkItem(index=0, sequence_length=50, suffix_keep=8),
            BranchBatchWorkItem(index=1, sequence_length=50, suffix_keep=8),
            BranchBatchWorkItem(index=2, sequence_length=25, suffix_keep=8),
        ],
        max_branch_rows=3,
        max_branch_tokens=200,
    )

    assert plan.total_real_tokens == 125
    assert plan.total_padded_tokens >= plan.total_real_tokens
    assert 0.0 <= plan.padding_fraction <= 1.0


def test_smart_branch_batcher_falls_back_when_binpacking_is_unavailable(
    monkeypatch,
) -> None:
    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name == "binpacking":
            raise ImportError("blocked by test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)

    plan = plan_smart_branch_batches(
        [
            BranchBatchWorkItem(index=0, sequence_length=70, suffix_keep=20),
            BranchBatchWorkItem(index=1, sequence_length=60, suffix_keep=20),
            BranchBatchWorkItem(index=2, sequence_length=25, suffix_keep=12),
        ],
        max_branch_rows=2,
        max_branch_tokens=140,
    )

    assert plan.scheduler == "deterministic_fallback"
    assert sorted(item.index for batch in plan.batches for item in batch.items) == [
        0,
        1,
        2,
    ]
    assert all(batch.padded_token_volume <= 140 for batch in plan.batches)
