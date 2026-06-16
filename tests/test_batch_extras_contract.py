from __future__ import annotations

from collections import defaultdict

import pytest
import torch

from src.metrics.dataset_metrics import GradAccumLossScaleMixin
from src.data_collators.dataset_metrics import build_dataset_metrics_collator
from src.detection.prefix_denoising.types import (
    HybridPrefixDenoisingSample,
    PrefixDenoisingSegment,
)
from src.trainers.batch_extras import (
    DATASET_LABELS_KEY,
    DATASET_SEGMENTS_KEY,
    INSTABILITY_META_JSON_KEY,
    PACK_NUM_SAMPLES_KEY,
    PREFIX_DENOISING_HYBRID_KEY,
    RECURSIVE_DETECTION_TARGETS_KEY,
    TEACHER_FORCING_TARGET_IR_KEY,
    TOKEN_TYPES_KEY,
    get_stashed_batch_extras,
)
from src.trainers.metrics.teacher_forcing import _resolve_sample_ids


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
            assert RECURSIVE_DETECTION_TARGETS_KEY not in inputs
            assert TEACHER_FORCING_TARGET_IR_KEY not in inputs

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
        RECURSIVE_DETECTION_TARGETS_KEY: ("target-a", "target-b"),
        TEACHER_FORCING_TARGET_IR_KEY: ("ir-a", "ir-b"),
    }

    loss = trainer.compute_loss(model=None, inputs=inputs, return_outputs=False, num_items_in_batch=None)
    assert torch.is_tensor(loss)

    extras = get_stashed_batch_extras(trainer)
    assert extras.dataset_labels == ["a", "b"]
    assert extras.dataset_segments == [4, 4]
    assert isinstance(extras.pack_num_samples, torch.Tensor)
    assert isinstance(extras.token_types, torch.Tensor)
    assert extras.instability_meta_json == "[]"
    assert extras.recursive_detection_targets == ("target-a", "target-b")
    assert extras.teacher_forcing_target_ir == ("ir-a", "ir-b")


class _DummyTemplate:
    tokenizer = None
    template_meta = None


def _base_collator(batch):
    bsz = len(batch)
    labels = torch.ones((bsz, 4), dtype=torch.long)
    return {
        "input_ids": labels.clone(),
        "labels": labels,
        "attention_mask": torch.ones_like(labels),
    }


def test_recursive_detection_targets_are_collated_from_unpacked_samples() -> None:
    collator = build_dataset_metrics_collator(_DummyTemplate(), _base_collator)

    out = collator(
        [
            {"dataset": "coco", RECURSIVE_DETECTION_TARGETS_KEY: "target-a"},
            {"dataset": "coco", RECURSIVE_DETECTION_TARGETS_KEY: "target-b"},
        ]
    )

    assert out[RECURSIVE_DETECTION_TARGETS_KEY] == ("target-a", "target-b")


def test_recursive_detection_targets_require_all_unpacked_samples_to_have_sidecar() -> None:
    collator = build_dataset_metrics_collator(_DummyTemplate(), _base_collator)

    with pytest.raises(ValueError, match="recursive_detection_targets"):
        collator(
            [
                {"dataset": "coco", RECURSIVE_DETECTION_TARGETS_KEY: "target-a"},
                {"dataset": "coco"},
            ]
        )


def test_recursive_detection_targets_reject_packed_batches() -> None:
    collator = build_dataset_metrics_collator(_DummyTemplate(), _base_collator)

    with pytest.raises(ValueError, match="recursive_detection_targets.*packing"):
        collator(
            [
                [{"dataset": "coco", RECURSIVE_DETECTION_TARGETS_KEY: "target-a"}],
                [{"dataset": "coco", RECURSIVE_DETECTION_TARGETS_KEY: "target-b"}],
            ]
        )


def test_teacher_forcing_target_ir_is_collated_from_unpacked_samples() -> None:
    collator = build_dataset_metrics_collator(_DummyTemplate(), _base_collator)

    out = collator(
        [
            {"dataset": "coco", TEACHER_FORCING_TARGET_IR_KEY: "ir-a"},
            {"dataset": "coco", TEACHER_FORCING_TARGET_IR_KEY: "ir-b"},
        ]
    )

    assert out[TEACHER_FORCING_TARGET_IR_KEY] == ("ir-a", "ir-b")


def test_prefix_denoising_sidecar_survives_mutating_base_collator() -> None:
    def mutating_base_collator(batch):
        for row in batch:
            row.pop(PREFIX_DENOISING_HYBRID_KEY, None)
        return _base_collator(batch)

    clean = PrefixDenoisingSegment(
        segment_id="sample.clean",
        branch_id="clean_full",
        input_ids=(1, 2),
        labels=(-100, 2),
        attention_mask=(1, 1),
        supervised_positions=(1,),
        ce_denominator=1,
    )
    noisy = PrefixDenoisingSegment(
        segment_id="sample.noisy",
        branch_id="noisy_full",
        input_ids=(1, 3),
        labels=(-100, 2),
        attention_mask=(1, 1),
        supervised_positions=(1,),
        ce_denominator=1,
    )
    hybrid = HybridPrefixDenoisingSample(
        ok=True,
        hybrid_sample_id="sample",
        base_sample_id="base",
        clean_full=clean,
        noisy_full=noisy,
    )
    row = {"dataset": "coco", PREFIX_DENOISING_HYBRID_KEY: hybrid}
    collator = build_dataset_metrics_collator(
        _DummyTemplate(),
        mutating_base_collator,
    )

    out = collator([row])

    assert PREFIX_DENOISING_HYBRID_KEY not in row
    assert out[PREFIX_DENOISING_HYBRID_KEY] == (hybrid,)


def test_teacher_forcing_target_ir_collates_sample_id_sidecar_from_raw_batch() -> None:
    collator = build_dataset_metrics_collator(_DummyTemplate(), _base_collator)

    out = collator(
        [
            {"dataset": "coco", "sample_id": 101, TEACHER_FORCING_TARGET_IR_KEY: "ir-a"},
            {"dataset": "coco", "sample_id": 202, TEACHER_FORCING_TARGET_IR_KEY: "ir-b"},
        ]
    )

    assert out[TEACHER_FORCING_TARGET_IR_KEY] == ("ir-a", "ir-b")
    assert out["sample_id"] == (101, 202)


def test_teacher_forcing_sample_ids_resolve_tensor_array_list_and_scalar() -> None:
    class ArrayLike:
        def __init__(self, value):
            self.value = value

        def tolist(self):
            return self.value

    assert _resolve_sample_ids(torch.tensor([101, 202]), count=2) == ("101", "202")
    assert _resolve_sample_ids(ArrayLike([303, 404]), count=2) == ("303", "404")
    assert _resolve_sample_ids([505, 606], count=2) == ("505", "606")
    assert _resolve_sample_ids(torch.tensor(707), count=1) == ("707",)

    with pytest.raises(ValueError, match="sample_id sidecar length"):
        _resolve_sample_ids(torch.tensor(808), count=2)


def test_teacher_forcing_target_ir_requires_all_unpacked_samples_to_have_sidecar() -> None:
    collator = build_dataset_metrics_collator(_DummyTemplate(), _base_collator)

    with pytest.raises(ValueError, match="teacher_forcing_target_ir"):
        collator(
            [
                {"dataset": "coco", TEACHER_FORCING_TARGET_IR_KEY: "ir-a"},
                {"dataset": "coco"},
            ]
        )


def test_teacher_forcing_target_ir_rejects_packed_batches() -> None:
    collator = build_dataset_metrics_collator(_DummyTemplate(), _base_collator)

    with pytest.raises(ValueError, match="teacher_forcing_target_ir.*packing"):
        collator(
            [
                [{"dataset": "coco", TEACHER_FORCING_TARGET_IR_KEY: "ir-a"}],
                [{"dataset": "coco", TEACHER_FORCING_TARGET_IR_KEY: "ir-b"}],
            ]
        )
