from __future__ import annotations

import pytest

from src.sft import _validate_stage2_step_budget_windows


def test_stage2_step_budget_guard_noop_for_non_stage2_variant() -> None:
    _validate_stage2_step_budget_windows(
        trainer_variant="sft",
        packing_enabled=True,
        per_rank_batches_est=1,
        gradient_accumulation_steps=8,
        dataloader_drop_last=False,
    )


def test_stage2_step_budget_guard_requires_full_window() -> None:
    with pytest.raises(ValueError, match=r"requires per-rank batches >= gradient_accumulation_steps"):
        _validate_stage2_step_budget_windows(
            trainer_variant="stage2_two_channel",
            packing_enabled=True,
            per_rank_batches_est=1,
            gradient_accumulation_steps=2,
            dataloader_drop_last=False,
        )


def test_stage2_step_budget_guard_rejects_partial_tail_when_not_drop_last() -> None:
    with pytest.raises(ValueError, match=r"does not support a partial gradient-accumulation window"):
        _validate_stage2_step_budget_windows(
            trainer_variant="stage2_two_channel",
            packing_enabled=True,
            per_rank_batches_est=5,
            gradient_accumulation_steps=2,
            dataloader_drop_last=False,
        )


def test_stage2_step_budget_guard_allows_drop_last_partial_tail() -> None:
    _validate_stage2_step_budget_windows(
        trainer_variant="stage2_two_channel",
        packing_enabled=True,
        per_rank_batches_est=5,
        gradient_accumulation_steps=2,
        dataloader_drop_last=True,
    )
