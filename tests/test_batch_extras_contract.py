from __future__ import annotations

from collections import defaultdict

import torch

from src.metrics.dataset_metrics import GradAccumLossScaleMixin
from src.trainers.batch_extras import (
    DATASET_LABELS_KEY,
    DATASET_SEGMENTS_KEY,
    INSTABILITY_META_JSON_KEY,
    PACK_NUM_SAMPLES_KEY,
    TOKEN_TYPES_KEY,
    get_stashed_batch_extras,
)


def test_batch_extras_are_stripped_before_model_forward_and_stashed() -> None:
    class DummyBase:
        def compute_loss(
            self, model, inputs, return_outputs: bool = False, num_items_in_batch=None
        ):
            # The contract: batch extras MUST NOT be forwarded into model(**inputs).
            assert DATASET_LABELS_KEY not in inputs
            assert DATASET_SEGMENTS_KEY not in inputs
            assert PACK_NUM_SAMPLES_KEY not in inputs
            assert TOKEN_TYPES_KEY not in inputs
            assert INSTABILITY_META_JSON_KEY not in inputs

            loss = torch.tensor(1.0)
            outputs = object()
            return (loss, outputs) if return_outputs else loss

    class DummyMetric:
        def __init__(self) -> None:
            self.last = None

        def update(self, value: float) -> None:
            self.last = float(value)

    class DummyModel:
        def __init__(self, training: bool) -> None:
            self.training = training

    class DummyTrainer(GradAccumLossScaleMixin, DummyBase):
        def __init__(self) -> None:
            self.model = DummyModel(training=True)
            self.custom_metrics = {
                "train": defaultdict(DummyMetric),
                "eval": defaultdict(DummyMetric),
            }
            self._get_learning_rate = lambda: 1e-5  # noqa: E731

    trainer = DummyTrainer()

    inputs = {
        "labels": torch.zeros((2, 4), dtype=torch.long),
        DATASET_LABELS_KEY: ["a", "b"],
        DATASET_SEGMENTS_KEY: [4, 4],
        PACK_NUM_SAMPLES_KEY: torch.tensor([2, 1], dtype=torch.long),
        TOKEN_TYPES_KEY: torch.zeros((2, 4), dtype=torch.long),
        INSTABILITY_META_JSON_KEY: "[]",
    }

    loss = trainer.compute_loss(model=None, inputs=inputs, return_outputs=False, num_items_in_batch=None)
    assert torch.is_tensor(loss)

    extras = get_stashed_batch_extras(trainer)
    assert extras.dataset_labels == ["a", "b"]
    assert extras.dataset_segments == [4, 4]
    assert isinstance(extras.pack_num_samples, torch.Tensor)
    assert isinstance(extras.token_types, torch.Tensor)
    assert extras.instability_meta_json == "[]"
