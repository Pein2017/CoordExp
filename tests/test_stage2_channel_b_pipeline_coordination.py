import pytest
import torch

from src.trainers.rollout_correction.coordination import (
    finalize_rollout_correction_pipeline_step,
)


class _DoneThread:
    def join(self, timeout=None):
        return None

    def is_alive(self):
        return False


class _Owner:
    def __init__(self):
        self.flushed = []

    def _stage2_flush_train_monitor_dump(self, *, global_step):
        self.flushed.append(int(global_step))


def test_pipeline_finalize_allows_multiple_segments_per_raw_rollout() -> None:
    owner = _Owner()

    loss = finalize_rollout_correction_pipeline_step(
        thread_obj=_DoneThread(),
        owner=owner,
        target_log_step=7,
        producer_exc=[],
        total_segments_target=16,
        seen_raw=16,
        seen_segments=64,
        loss_total=torch.tensor(1.0),
    )

    assert loss.item() == pytest.approx(1.0)
    assert owner.flushed == [7]


def test_pipeline_finalize_still_rejects_wrong_raw_rollout_count() -> None:
    owner = _Owner()

    with pytest.raises(ValueError, match="unexpected raw-rollout count"):
        finalize_rollout_correction_pipeline_step(
            thread_obj=_DoneThread(),
            owner=owner,
            target_log_step=7,
            producer_exc=[],
            total_segments_target=16,
            seen_raw=15,
            seen_segments=64,
            loss_total=torch.tensor(1.0),
        )
