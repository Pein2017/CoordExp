from __future__ import annotations

import inspect
from typing import get_type_hints

from src.trainers.rollout_correction import coordination


def test_rollout_correction_coordination_owner_surface_is_protocol_typed() -> None:
    expected = coordination.RolloutCorrectionCoordinationOwner

    for fn_name in (
        "consume_rollout_correction_queue_item",
        "prepare_rollout_correction_pipeline_pack_step",
        "run_rollout_correction_train_one_pack",
        "gather_rollout_correction_local_pack_counts",
        "build_rollout_correction_pack_schedule",
        "run_rollout_correction_nonpipeline_learning_loop",
        "run_rollout_correction_pipeline_learning_loop",
        "run_rollout_correction_pipeline_producer",
    ):
        hints = get_type_hints(getattr(coordination, fn_name))
        assert hints["owner"] is expected


def test_rollout_correction_pack_execution_does_not_construct_targets() -> None:
    source = inspect.getsource(coordination.run_rollout_correction_train_one_pack)

    for forbidden in (
        "construct_rollout_correction_target_context",
        "_build_rollout_correction_triage",
        "_build_rollout_correction_supervision_targets",
        "_prepare_rollout_correction_inputs",
        "_rollout_many",
    ):
        assert forbidden not in source


def test_rollout_correction_pack_schedule_surface_is_bounded() -> None:
    schedule_source = inspect.getsource(coordination.build_rollout_correction_pack_schedule)
    gather_source = inspect.getsource(coordination.gather_rollout_correction_local_pack_counts)

    assert "Stage2PackSchedule.from_rank_pack_counts" in schedule_source
    assert "_stage2_post_rollout_buffer" not in schedule_source
    assert "_stage2_pop_post_rollout_pack" not in schedule_source
    assert "all_gather_object" in gather_source or "all_gather" in gather_source
