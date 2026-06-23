from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from src.data_collators.dataset_metrics import build_dataset_metrics_collator
from src.trainers.batch_extras import BatchExtras, pop_batch_extras
from src.training.bridge import TrainerLossBridge, TrainerLossBridgeSettings
from src.training.coverage_ledger import (
    CoverageLedgerObjectEntry,
    CoverageLedgerSidecar,
)
from src.training.objectives.types import ObjectiveSpec
from src.training.sidecars import (
    DatasetSidecars,
    Stage2OwnershipSidecars,
    SupervisionSidecars,
    TrainingSidecars,
)
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


def _coverage_ledger_sidecar(sample_id: str = "coco:0") -> CoverageLedgerSidecar:
    return CoverageLedgerSidecar(
        sample_id=sample_id,
        prompt_end_position=9,
        object_entries=(
            CoverageLedgerObjectEntry(
                object_instance_id=f"{sample_id}:object-0",
                source_object_index=7,
                emitted_order_index=0,
                image_index=0,
                bbox_norm1000_xyxy=(10, 20, 300, 400),
                box_start_position=11,
                coord_label_positions=(12, 13, 14, 15),
                object_ref_end_position=10,
                box_end_position=16,
            ),
        ),
        image_grid_thw=(1, 16, 16),
        processed_width=640,
        processed_height=480,
        image_identity=f"{sample_id}.jpg",
    )


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


def _minimal_raw_batch(
    *,
    teacher_forcing_target_ir: object | None = None,
) -> dict[str, Any]:
    raw_batch: dict[str, Any] = {
        "input_ids": torch.ones((1, 2), dtype=torch.long),
        "attention_mask": torch.ones((1, 2), dtype=torch.long),
    }
    if teacher_forcing_target_ir is not None:
        raw_batch[TEACHER_FORCING_TARGET_IR_KEY] = teacher_forcing_target_ir
    return raw_batch


def _run_bridge(
    *,
    raw_ir: object | None = None,
    batch_ir: object | None = None,
    semantic_ir: object | None = None,
):
    training_sidecars = None
    if semantic_ir is not None:
        training_sidecars = TrainingSidecars(
            supervision=SupervisionSidecars(
                teacher_forcing_target_ir=semantic_ir,
            )
        )

    batch_extras = None
    if batch_ir is not None:
        batch_extras = BatchExtras(teacher_forcing_target_ir=batch_ir)

    return TrainerLossBridge().compute_loss(
        model=_FakeModel(torch.zeros((1, 2, 5), dtype=torch.float32)),
        raw_batch=_minimal_raw_batch(teacher_forcing_target_ir=raw_ir),
        batch_extras=batch_extras,
        training_sidecars=training_sidecars,
        supervision=SupervisionBatch(),
        objectives=(ObjectiveSpec("token_ce"),),
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


def test_coverage_ledger_sidecar_survives_collator_to_training_sidecars() -> None:
    collator = build_dataset_metrics_collator(
        _DummyTemplate(),
        _base_collator,
        coverage_ledger_cfg={"enabled": True},
    )
    ledger_sidecar = _coverage_ledger_sidecar()

    collated = collator(
        [
            {
                "dataset": "coco",
                "training_sidecars": TrainingSidecars(
                    supervision=SupervisionSidecars(payloads=(ledger_sidecar,))
                ),
            },
        ]
    )
    model = _FakeModel(torch.zeros((1, 4, 5), dtype=torch.float32))

    result = TrainerLossBridge().compute_loss(
        model=model,
        raw_batch=collated,
        batch_extras=pop_batch_extras(collated),
        supervision=SupervisionBatch(),
        objectives=(ObjectiveSpec("token_ce"),),
    )

    assert "training_sidecars" not in model.calls[0]
    assert "training_sidecars" not in result.model_inputs.payload
    assert result.training_sidecars.supervision.payloads == (ledger_sidecar,)


def test_coverage_ledger_collator_preserves_existing_supervision_payloads() -> None:
    collator = build_dataset_metrics_collator(
        _DummyTemplate(),
        _base_collator,
        coverage_ledger_cfg={"enabled": True},
    )
    first_ledger_sidecar = _coverage_ledger_sidecar("coco:0")
    second_ledger_sidecar = _coverage_ledger_sidecar("coco:1")

    collated = collator(
        [
            {
                "dataset": "coco",
                "training_sidecars": TrainingSidecars(
                    supervision=SupervisionSidecars(
                        payloads=("non-ledger-a", first_ledger_sidecar)
                    )
                ),
            },
            {
                "dataset": "coco",
                "training_sidecars": TrainingSidecars(
                    supervision=SupervisionSidecars(
                        payloads=("non-ledger-b", second_ledger_sidecar)
                    )
                ),
            },
        ]
    )

    assert collated["training_sidecars"].supervision.payloads == (
        "non-ledger-a",
        first_ledger_sidecar,
        "non-ledger-b",
        second_ledger_sidecar,
    )


def test_coverage_ledger_collator_rejects_unaggregated_row_sidecar_fields() -> None:
    collator = build_dataset_metrics_collator(
        _DummyTemplate(),
        _base_collator,
        coverage_ledger_cfg={"enabled": True},
    )
    ledger_sidecar = _coverage_ledger_sidecar()

    with pytest.raises(ValueError, match="coverage ledger.*cannot aggregate"):
        collator(
            [
                {
                    "dataset": "coco",
                    "training_sidecars": TrainingSidecars(
                        supervision=SupervisionSidecars(
                            payloads=(ledger_sidecar,),
                            teacher_forcing_target_ir=("ir-a",),
                        )
                    ),
                }
            ]
        )


def test_coverage_ledger_enabled_requires_one_payload_per_unpacked_sample() -> None:
    collator = build_dataset_metrics_collator(
        _DummyTemplate(),
        _base_collator,
        coverage_ledger_cfg={"enabled": True},
    )

    with pytest.raises(ValueError, match="CoverageLedgerSidecar.*every sample"):
        collator([{"dataset": "coco"}])


def test_coverage_ledger_rejects_duplicate_payloads_per_sample() -> None:
    collator = build_dataset_metrics_collator(
        _DummyTemplate(),
        _base_collator,
        coverage_ledger_cfg={"enabled": True},
    )
    ledger_sidecar = _coverage_ledger_sidecar()

    with pytest.raises(ValueError, match="duplicate CoverageLedgerSidecar"):
        collator(
            [
                {
                    "dataset": "coco",
                    "training_sidecars": TrainingSidecars(
                        supervision=SupervisionSidecars(
                            payloads=(ledger_sidecar, ledger_sidecar)
                        )
                    ),
                }
            ]
        )


def test_coverage_ledger_rejects_packed_sidecar_offset_rewriting() -> None:
    collator = build_dataset_metrics_collator(
        _DummyTemplate(),
        lambda batch: _base_collator([{"dataset": "pack"} for _pack in batch]),
        coverage_ledger_cfg={"enabled": True},
    )
    ledger_sidecar = _coverage_ledger_sidecar()

    with pytest.raises(ValueError, match="CoverageLedgerSidecar.*packing"):
        collator(
            [
                [
                    {
                        "dataset": "coco",
                        "training_sidecars": TrainingSidecars(
                            supervision=SupervisionSidecars(payloads=(ledger_sidecar,))
                        ),
                    }
                ]
            ]
        )


def test_coverage_ledger_full_training_runtime_uses_bridge_consumer_guards() -> None:
    import src.sft as sft_module

    assert not hasattr(sft_module, "_reject_coverage_ledger_without_loss_consumer")

    enabled_settings = TrainerLossBridgeSettings(
        coverage_ledger={"enabled": True}
    )
    with pytest.raises(ValueError, match="exactly one CoverageLedgerSidecar"):
        TrainerLossBridge(settings=enabled_settings).compute_loss(
            model=_FakeModel(torch.zeros((1, 2, 5), dtype=torch.float32)),
            raw_batch=_minimal_raw_batch(),
            supervision=SupervisionBatch(),
            objectives=(ObjectiveSpec("token_ce"),),
        )

    with pytest.raises(ValueError, match="exactly one coverage_ledger_head"):
        TrainerLossBridge(settings=enabled_settings).compute_loss(
            model=_FakeModel(torch.zeros((1, 2, 5), dtype=torch.float32)),
            raw_batch={
                **_minimal_raw_batch(),
                "training_sidecars": TrainingSidecars(
                    supervision=SupervisionSidecars(
                        payloads=(_coverage_ledger_sidecar(),)
                    )
                ),
            },
            supervision=SupervisionBatch(),
            objectives=(ObjectiveSpec("token_ce"),),
        )


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


def test_semantic_teacher_forcing_sidecar_is_preserved_when_compat_sources_match() -> None:
    semantic_ir = ({"payload": "same-ir"},)
    raw_ir = ({"payload": "same-ir"},)
    batch_ir = ({"payload": "same-ir"},)

    result = _run_bridge(
        raw_ir=raw_ir,
        batch_ir=batch_ir,
        semantic_ir=semantic_ir,
    )

    assert result.training_sidecars.supervision.teacher_forcing_target_ir is semantic_ir


def test_batch_extras_teacher_forcing_ir_fills_absent_semantic_sidecar() -> None:
    batch_ir = ("batch-ir",)

    result = _run_bridge(batch_ir=batch_ir)

    assert result.training_sidecars.supervision.teacher_forcing_target_ir is batch_ir


def test_raw_and_batch_teacher_forcing_ir_conflict_fails() -> None:
    with pytest.raises(ValueError, match="raw_batch.*batch_extras"):
        _run_bridge(raw_ir=("raw-ir",), batch_ir=("batch-ir",))


def test_semantic_and_batch_teacher_forcing_ir_conflict_fails() -> None:
    with pytest.raises(ValueError, match="training_sidecars.*batch_extras"):
        _run_bridge(semantic_ir=("semantic-ir",), batch_ir=("batch-ir",))


def test_semantic_and_raw_teacher_forcing_ir_conflict_fails() -> None:
    with pytest.raises(ValueError, match="training_sidecars.*raw_batch"):
        _run_bridge(semantic_ir=("semantic-ir",), raw_ir=("raw-ir",))


def test_explicit_and_raw_training_sidecars_same_ir_preserves_explicit() -> None:
    explicit_ir = ({"payload": "same-ir"},)
    raw_ir = ({"payload": "same-ir"},)
    explicit_sidecars = TrainingSidecars(
        supervision=SupervisionSidecars(
            teacher_forcing_target_ir=explicit_ir,
        )
    )
    raw_batch = _minimal_raw_batch()
    raw_batch["training_sidecars"] = TrainingSidecars(
        supervision=SupervisionSidecars(
            teacher_forcing_target_ir=raw_ir,
        )
    )

    result = TrainerLossBridge().compute_loss(
        model=_FakeModel(torch.zeros((1, 2, 5), dtype=torch.float32)),
        raw_batch=raw_batch,
        training_sidecars=explicit_sidecars,
        supervision=SupervisionBatch(),
        objectives=(ObjectiveSpec("token_ce"),),
    )

    assert result.training_sidecars is not raw_batch["training_sidecars"]
    assert result.training_sidecars.supervision.teacher_forcing_target_ir is explicit_ir


def test_equal_explicit_and_raw_training_sidecars_preserves_explicit() -> None:
    explicit_sidecars = TrainingSidecars(
        supervision=SupervisionSidecars(
            teacher_forcing_target_ir=("same-ir",),
        ),
        dataset=DatasetSidecars(sample_id="sample-1"),
        stage2=Stage2OwnershipSidecars(assignment_result={"matched": 1}),
    )
    raw_batch = _minimal_raw_batch()
    raw_batch["training_sidecars"] = TrainingSidecars(
        supervision=SupervisionSidecars(
            teacher_forcing_target_ir=("same-ir",),
        ),
        dataset=DatasetSidecars(sample_id="sample-1"),
        stage2=Stage2OwnershipSidecars(assignment_result={"matched": 1}),
    )

    result = TrainerLossBridge().compute_loss(
        model=_FakeModel(torch.zeros((1, 2, 5), dtype=torch.float32)),
        raw_batch=raw_batch,
        training_sidecars=explicit_sidecars,
        supervision=SupervisionBatch(),
        objectives=(ObjectiveSpec("token_ce"),),
    )

    assert result.training_sidecars is explicit_sidecars


def test_explicit_and_raw_training_sidecars_full_payload_conflict_fails() -> None:
    explicit_sidecars = TrainingSidecars(
        dataset=DatasetSidecars(sample_id="explicit-sample"),
    )
    raw_batch = _minimal_raw_batch()
    raw_batch["training_sidecars"] = TrainingSidecars(
        dataset=DatasetSidecars(sample_id="raw-sample"),
        stage2=Stage2OwnershipSidecars(assignment_result={"matched": 1}),
    )

    with pytest.raises(
        ValueError,
        match="training_sidecars.*raw_batch\\.training_sidecars",
    ):
        TrainerLossBridge().compute_loss(
            model=_FakeModel(torch.zeros((1, 2, 5), dtype=torch.float32)),
            raw_batch=raw_batch,
            training_sidecars=explicit_sidecars,
            supervision=SupervisionBatch(),
            objectives=(ObjectiveSpec("token_ce"),),
        )


def test_explicit_and_raw_training_sidecars_coverage_payload_conflict_fails() -> None:
    explicit_sidecars = TrainingSidecars(
        supervision=SupervisionSidecars(
            payloads=(_coverage_ledger_sidecar("explicit"),)
        )
    )
    raw_batch = _minimal_raw_batch()
    raw_batch["training_sidecars"] = TrainingSidecars(
        supervision=SupervisionSidecars(
            payloads=(_coverage_ledger_sidecar("raw"),)
        )
    )

    with pytest.raises(
        ValueError,
        match="training_sidecars.*raw_batch\\.training_sidecars",
    ):
        TrainerLossBridge().compute_loss(
            model=_FakeModel(torch.zeros((1, 2, 5), dtype=torch.float32)),
            raw_batch=raw_batch,
            training_sidecars=explicit_sidecars,
            supervision=SupervisionBatch(),
            objectives=(ObjectiveSpec("token_ce"),),
        )


def test_explicit_and_raw_training_sidecars_different_ir_conflict_fails() -> None:
    explicit_sidecars = TrainingSidecars(
        supervision=SupervisionSidecars(
            teacher_forcing_target_ir=("explicit-ir",),
        )
    )
    raw_batch = _minimal_raw_batch()
    raw_batch["training_sidecars"] = TrainingSidecars(
        supervision=SupervisionSidecars(
            teacher_forcing_target_ir=("raw-sidecar-ir",),
        )
    )

    with pytest.raises(
        ValueError,
        match="training_sidecars.*raw_batch\\.training_sidecars",
    ):
        TrainerLossBridge().compute_loss(
            model=_FakeModel(torch.zeros((1, 2, 5), dtype=torch.float32)),
            raw_batch=raw_batch,
            training_sidecars=explicit_sidecars,
            supervision=SupervisionBatch(),
            objectives=(ObjectiveSpec("token_ce"),),
        )


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
