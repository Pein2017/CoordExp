"""Compatibility test module for canonical Stage-2 two-channel naming.

Keeps legacy command paths working while the canonical test body lives in
`tests/test_stage2_ab_training.py`.
"""

from test_stage2_ab_training import *  # noqa: F401,F403


def test_legacy_stage2_ab_import_paths_resolve() -> None:
    import src.trainers.stage2_ab as legacy_stage2_ab
    import src.trainers.stage2_ab_training as legacy_stage2_ab_training
    from src.trainers.stage2_two_channel import (
        Stage2ABChannelExecutorsMixin,
        Stage2ABSchedulerMixin,
        Stage2ABTrainingTrainer,
    )

    assert legacy_stage2_ab_training.Stage2ABTrainingTrainer is Stage2ABTrainingTrainer
    assert (
        legacy_stage2_ab.Stage2ABChannelExecutorsMixin
        is Stage2ABChannelExecutorsMixin
    )
    assert legacy_stage2_ab.Stage2ABSchedulerMixin is Stage2ABSchedulerMixin
