from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from src.data_collators.dataset_metrics import build_dataset_metrics_collator
from src.trainers.batch_extras import pop_batch_extras
from src.training.bridge import TrainerLossBridge
from src.training.objectives.types import ObjectiveSpec
from src.training.sidecars import SupervisionSidecars, TrainingSidecars
from src.training.supervision.batch import SupervisionBatch
from src.training.teacher_forcing.constants import TEACHER_FORCING_TARGET_IR_KEY


class _DummyTemplate:
    tokenizer = None
    template_meta = None


def _base_collator(batch: list[dict[str, Any]]) -> dict[str, Any]:
    bsz = len(batch)
    labels = torch.ones((bsz, 4), dtype=torch.long)
    return {
        "input_ids": labels.clone(),
        "labels": labels,
        "attention_mask": torch.ones_like(labels),
    }


@dataclass
class _FakeOutputs:
    logits: torch.Tensor
    loss: torch.Tensor


class _FakeModel:
    def __init__(self, logits: torch.Tensor) -> None:
        self.config = SimpleNamespace(model_type="qwen3_vl")
        self.logits = logits
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> _FakeOutputs:
        self.calls.append(dict(kwargs))
        return _FakeOutputs(
            logits=self.logits,
            loss=torch.tensor(999.0, dtype=torch.float32),
        )


def test_teacher_forcing_target_ir_survives_collator_to_batch_extras() -> None:
    collator = build_dataset_metrics_collator(_DummyTemplate(), _base_collator)
    target_ir = {"schema_version": 1, "atoms": []}

    collated = collator(
        [
            {"dataset": "coco", TEACHER_FORCING_TARGET_IR_KEY: target_ir},
        ]
    )
    extras = pop_batch_extras(collated)

    assert extras.teacher_forcing_target_ir == (target_ir,)
    assert TEACHER_FORCING_TARGET_IR_KEY not in collated


def test_teacher_forcing_target_ir_has_semantic_supervision_sidecar_home() -> None:
    sidecars = TrainingSidecars(
        supervision=SupervisionSidecars(
            teacher_forcing_target_ir=("ir-a", "ir-b"),
        )
    )

    assert sidecars.supervision.teacher_forcing_target_ir == ("ir-a", "ir-b")


def test_trainer_loss_bridge_strips_teacher_forcing_target_ir_and_carries_sidecar() -> None:
    model = _FakeModel(torch.zeros((1, 2, 5), dtype=torch.float32))
    target_ir = {"schema_version": 1, "atoms": []}
    raw_batch = {
        "input_ids": torch.ones((1, 2), dtype=torch.long),
        "attention_mask": torch.ones((1, 2), dtype=torch.long),
        "labels": torch.ones((1, 2), dtype=torch.long),
        "cu_seq_lens_q": torch.tensor([0, 2], dtype=torch.int32),
        "cu_seq_lens_k": torch.tensor([0, 2], dtype=torch.int32),
        "max_length_q": 2,
        "max_length_k": 2,
        TEACHER_FORCING_TARGET_IR_KEY: (target_ir,),
    }

    result = TrainerLossBridge().compute_loss(
        model=model,
        raw_batch=raw_batch,
        supervision=SupervisionBatch(),
        objectives=(ObjectiveSpec("token_ce"),),
    )

    assert len(model.calls) == 1
    assert TEACHER_FORCING_TARGET_IR_KEY not in model.calls[0]
    assert TEACHER_FORCING_TARGET_IR_KEY not in result.model_inputs.payload
    assert result.training_sidecars.supervision.teacher_forcing_target_ir == (target_ir,)

    assert torch.equal(model.calls[0]["cu_seq_lens_q"], raw_batch["cu_seq_lens_q"])
    assert torch.equal(model.calls[0]["cu_seq_lens_k"], raw_batch["cu_seq_lens_k"])
    assert model.calls[0]["max_length_q"] == 2
    assert model.calls[0]["max_length_k"] == 2


def test_trainer_loss_bridge_carries_teacher_forcing_ir_after_batch_extras_pop() -> None:
    collator = build_dataset_metrics_collator(_DummyTemplate(), _base_collator)
    target_ir = {"schema_version": 1, "atoms": []}
    raw_batch = collator(
        [
            {"dataset": "coco", TEACHER_FORCING_TARGET_IR_KEY: target_ir},
        ]
    )
    batch_extras = pop_batch_extras(raw_batch)
    model = _FakeModel(torch.zeros((1, 4, 5), dtype=torch.float32))

    assert TEACHER_FORCING_TARGET_IR_KEY not in raw_batch

    result = TrainerLossBridge().compute_loss(
        model=model,
        raw_batch=raw_batch,
        batch_extras=batch_extras,
        supervision=SupervisionBatch(),
        objectives=(ObjectiveSpec("token_ce"),),
    )

    assert TEACHER_FORCING_TARGET_IR_KEY not in model.calls[0]
    assert result.training_sidecars.supervision.teacher_forcing_target_ir == (target_ir,)


def test_trainer_loss_bridge_rejects_logits_to_keep_for_teacher_forcing_sidecar() -> None:
    model = _FakeModel(torch.zeros((1, 2, 5), dtype=torch.float32))

    with pytest.raises(ValueError, match="logits_to_keep.*projection"):
        TrainerLossBridge().compute_loss(
            model=model,
            raw_batch={
                "input_ids": torch.ones((1, 2), dtype=torch.long),
                "logits_to_keep": 1,
                TEACHER_FORCING_TARGET_IR_KEY: ("ir-a",),
            },
            supervision=SupervisionBatch(),
            objectives=(ObjectiveSpec("token_ce"),),
        )

    assert model.calls == []
