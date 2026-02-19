from __future__ import annotations

import inspect

from src.trainers.stage2_ab_training import Stage2ABTrainingTrainer


def test_stage2_ab_enforces_prompt_tokenization_alignment() -> None:
    src = inspect.getsource(Stage2ABTrainingTrainer._prepare_batch_inputs_b)
    assert (
        "prompt tokenization mismatch between generation and teacher-forced encoding" in src
    )
