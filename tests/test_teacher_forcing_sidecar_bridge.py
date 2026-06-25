from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest
import torch

from src.data_collators.dataset_metrics import build_dataset_metrics_collator
from src.datasets.wrappers.packed_caption import build_static_packed_dataset
from src.trainers.batch_extras import BatchExtras, pop_batch_extras
from src.trainers.teacher_forcing.forwards import prepare_forward_inputs
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
from test_detection_training_dataset import (
    ImageExpandingSwiftTemplate,
    _ensure_image,
    _raw_row,
    _write_jsonl,
)
from src.detection.dataset import DetectionTrainingDataset


class _DummyTemplate:
    tokenizer = None
    template_meta = None


class _ImageGridExpandingSwiftTemplate(ImageExpandingSwiftTemplate):
    def encode(
        self,
        payload: dict[str, Any],
        *,
        return_length: bool,
    ) -> dict[str, Any]:
        encoded = super().encode(payload, return_length=return_length)
        encoded["image_grid_thw"] = (1, 8, 8)
        return encoded


class _QwenStaticPackingSwiftTemplate(_ImageGridExpandingSwiftTemplate):
    def __init__(self) -> None:
        super().__init__()
        self.max_length = 4096
        self.packing = False
        self.padding_free = False

    def data_collator(self, batch: list[Any]) -> dict[str, Any]:
        assert self.packing is True
        assert self.padding_free is True

        input_rows: list[list[int]] = []
        label_rows: list[list[int]] = []
        attention_rows: list[list[int]] = []
        packed_offsets_rows: list[list[int]] = []

        for pack in batch:
            pack_seq = pack if isinstance(pack, (list, tuple)) else [pack]
            input_ids: list[int] = []
            labels: list[int] = []
            attention_mask: list[int] = []
            offsets = [0]
            for sample in pack_seq:
                sample_ids = [int(x) for x in sample["input_ids"]]
                sample_labels = [int(x) for x in sample["labels"]]
                sample_attention = [
                    int(x)
                    for x in sample.get("attention_mask", [1 for _ in sample_ids])
                ]
                assert len(sample_ids) == len(sample_labels) == len(sample_attention)
                input_ids.extend(sample_ids)
                labels.extend(sample_labels)
                attention_mask.extend(sample_attention)
                offsets.append(len(input_ids))
            input_rows.append(input_ids)
            label_rows.append(labels)
            attention_rows.append(attention_mask)
            packed_offsets_rows.append(offsets)

        assert len(input_rows) == 1
        offsets = packed_offsets_rows[0]
        seq_len = len(input_rows[0])
        text_position_ids = torch.empty((1, seq_len), dtype=torch.long)
        for start, end in zip(offsets, offsets[1:]):
            text_position_ids[0, start:end] = torch.arange(end - start, dtype=torch.long)

        return {
            "input_ids": torch.tensor(input_rows, dtype=torch.long),
            "labels": torch.tensor(label_rows, dtype=torch.long),
            "attention_mask": torch.tensor(attention_rows, dtype=torch.long),
            "position_ids": text_position_ids.unsqueeze(0).repeat(3, 1, 1),
            "text_position_ids": text_position_ids,
            "cu_seq_lens_q": torch.tensor(offsets, dtype=torch.int32),
            "cu_seq_lens_k": torch.tensor(offsets, dtype=torch.int32),
            "max_length_q": max(b - a for a, b in zip(offsets, offsets[1:])),
            "max_length_k": max(b - a for a, b in zip(offsets, offsets[1:])),
            "packed_segment_offsets": torch.tensor(offsets, dtype=torch.int32),
        }


class _ModelInputOnlyDataset:
    object_ordering = "sorted"

    def __init__(
        self,
        dataset: DetectionTrainingDataset,
        *,
        order: Sequence[int],
    ) -> None:
        self._dataset = dataset
        self._order = tuple(int(x) for x in order)

    def __len__(self) -> int:
        return len(self._order)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self._dataset[self._order[int(index)]]
        return {
            "dataset": "unit",
            "input_ids": list(sample["input_ids"]),
            "labels": list(sample["labels"]),
            "attention_mask": list(sample["attention_mask"]),
            "length": int(sample["length"]),
        }

    def _static_packing_precompute_info(self) -> dict[str, object]:
        return {"thread_safe": True}


@dataclass(frozen=True)
class _PackedDirection:
    name: str
    target_label: str
    source_label: str
    target_range: tuple[int, int]
    source_range: tuple[int, int]


@dataclass
class _QwenProbeOutputs:
    logits: torch.Tensor
    final_hidden_state: torch.Tensor


class _QwenCompatibleAttentionProbe(torch.nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int = 16,
        use_packed_boundaries: bool,
    ) -> None:
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen3_vl")
        self.use_packed_boundaries = bool(use_packed_boundaries)
        self.embed = torch.nn.Embedding(int(vocab_size), int(hidden_size))
        self.lm_head = torch.nn.Linear(int(hidden_size), int(vocab_size), bias=False)
        with torch.no_grad():
            token = torch.arange(int(vocab_size), dtype=torch.float32).unsqueeze(1)
            dim = torch.arange(1, int(hidden_size) + 1, dtype=torch.float32).unsqueeze(0)
            self.embed.weight.copy_(((token + 1.0) * dim).remainder(97.0) / 97.0)
            self.lm_head.weight.copy_(
                torch.sin((token + 1.0) * dim / float(hidden_size + 3))
            )

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seq_lens_q: torch.Tensor,
        cu_seq_lens_k: torch.Tensor,
        max_length_q: int,
        max_length_k: int,
        **_: Any,
    ) -> _QwenProbeOutputs:
        assert tuple(cu_seq_lens_q.tolist()) == tuple(cu_seq_lens_k.tolist())
        assert int(max_length_q) == int(max_length_k)
        assert position_ids.ndim == 3
        assert int(position_ids.shape[0]) == 4

        x = self.embed(input_ids)
        scores = torch.matmul(x, x.transpose(-1, -2)) / (x.shape[-1] ** 0.5)
        seq_len = int(input_ids.shape[1])
        visible = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool))
        if self.use_packed_boundaries:
            segment_mask = torch.zeros_like(visible)
            offsets = [int(x) for x in cu_seq_lens_q.detach().cpu().tolist()]
            for start, end in zip(offsets, offsets[1:]):
                segment_mask[start:end, start:end] = True
            visible &= segment_mask
        scores = scores.masked_fill(~visible.unsqueeze(0), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        final_hidden = torch.matmul(weights, x)
        return _QwenProbeOutputs(
            logits=self.lm_head(final_hidden),
            final_hidden_state=final_hidden,
        )


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


def _raw_row_variant(desc_prefix: str, image_id: int) -> dict[str, Any]:
    row = copy.deepcopy(_raw_row())
    row["image_id"] = int(image_id)
    for index, obj in enumerate(row["objects"]):
        obj["desc"] = f"{desc_prefix}-{index}"
        obj["category_name"] = f"{desc_prefix}-{index}"
        obj["coco_ann_id"] = int(image_id * 100 + index)
    return row


def _build_static_packed_detection_batch(
    tmp_path: Path,
    *,
    order: Sequence[int],
) -> tuple[dict[str, Any], tuple[int, int, int]]:
    jsonl_path = tmp_path / f"train-{'-'.join(str(x) for x in order)}.coord.jsonl"
    _write_jsonl(
        jsonl_path,
        [
            _raw_row_variant("alpha", 101),
            _raw_row_variant("beta", 202),
        ],
    )
    _ensure_image(tmp_path)
    swift_template = _QwenStaticPackingSwiftTemplate()
    detection_dataset = DetectionTrainingDataset.from_jsonl(
        jsonl_path,
        swift_template=swift_template,
        image_root=tmp_path / "image-root",
        detection_template_id="compact_object_box_closed",
        mode="random_order_sft",
        object_ordering="sorted",
        user_prompt="Detect every object.",
        system_prompt="You are a detector.",
        seed=20260625,
        state_weighting="uniform_permutation",
        normalization="semantic_image_bucket_balanced",
        object_field_order="desc_first",
        teacher_forcing_profile="hard_sft",
        coverage_ledger_enabled=False,
        dataset_name="unit",
    )
    model_input_dataset = _ModelInputOnlyDataset(detection_dataset, order=order)
    packing_length = sum(int(model_input_dataset[i]["length"]) for i in range(2))
    packed_dataset = build_static_packed_dataset(
        model_input_dataset,
        template=swift_template,
        packing_length=packing_length,
        min_fill_ratio=0.01,
        packing_drop_last=False,
        dataloader_drop_last=False,
        allow_single_long=True,
        cache_dir=tmp_path / f"static-pack-cache-{'-'.join(str(x) for x in order)}",
        fingerprint={"test": "static_packed_forward_attention_isolation"},
        length_precompute_workers=1,
    )
    assert len(packed_dataset) == 1
    pack = packed_dataset[0]
    assert len(pack) == 2

    collator = build_dataset_metrics_collator(
        swift_template,
        swift_template.data_collator,
    )
    batch = collator([pack])
    offsets = tuple(int(x) for x in batch["packed_segment_offsets"].tolist())
    return batch, offsets


def _assert_packed_metadata_matches_offsets(
    batch: Mapping[str, Any],
    offsets: Sequence[int],
) -> None:
    expected = [int(x) for x in offsets]
    assert batch["cu_seq_lens_q"].tolist() == expected
    assert batch["cu_seq_lens_k"].tolist() == expected
    max_segment_len = max(b - a for a, b in zip(expected, expected[1:]))
    assert int(batch["max_length_q"]) == int(max_segment_len)
    assert int(batch["max_length_k"]) == int(max_segment_len)
    assert batch["packed_segment_offsets"].tolist() == expected

    text_position_ids = batch["text_position_ids"]
    reset_points = torch.nonzero(text_position_ids[0] == 0, as_tuple=False).flatten()
    assert reset_points.tolist() == expected[:-1]
    position_ids = batch["position_ids"]
    assert tuple(position_ids.shape) == (3, 1, expected[-1])
    for start, end in zip(expected, expected[1:]):
        local_positions = torch.arange(end - start, dtype=torch.long)
        assert torch.equal(text_position_ids[0, start:end], local_positions)
        for component in range(3):
            assert torch.equal(position_ids[component, 0, start:end], local_positions)


def _supervised_positions_in_range(
    labels: torch.Tensor,
    segment_range: tuple[int, int],
) -> torch.Tensor:
    start, end = segment_range
    segment_labels = labels[0, start:end]
    positions = torch.nonzero(segment_labels != -100, as_tuple=False).flatten() + start
    assert int(positions.numel()) > 0
    return positions


def _perturb_source_text_tokens(
    batch: Mapping[str, Any],
    source_range: tuple[int, int],
) -> dict[str, Any]:
    perturbed = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    labels = perturbed["labels"]
    input_ids = perturbed["input_ids"]
    start, end = source_range
    candidates = torch.nonzero(labels[0, start:end] == -100, as_tuple=False).flatten()
    assert int(candidates.numel()) >= 2
    positions = candidates[: min(8, int(candidates.numel()))] + start
    input_ids[0, positions] = input_ids[0, positions] + 211
    return perturbed


def _run_probe(
    probe: _QwenCompatibleAttentionProbe,
    batch: Mapping[str, Any],
) -> _QwenProbeOutputs:
    _, inputs_for_model, _ = prepare_forward_inputs(
        model=probe,
        inputs=batch,
        ignored_keys=(
            "labels",
            "attention_mask",
            "dataset_labels",
            "dataset_segments",
            "pack_num_samples",
            "packed_segment_offsets",
        ),
        packing_enabled=True,
        where="static packed attention isolation test",
    )
    return probe(**inputs_for_model)


def _max_delta_at_positions(
    before: torch.Tensor,
    after: torch.Tensor,
    positions: torch.Tensor,
) -> float:
    return float((before[0, positions] - after[0, positions]).abs().max().item())


def _assert_direction_isolated(
    *,
    batch: Mapping[str, Any],
    direction: _PackedDirection,
    vocab_size: int,
) -> None:
    supervised_positions = _supervised_positions_in_range(
        batch["labels"],
        direction.target_range,
    )
    perturbed = _perturb_source_text_tokens(batch, direction.source_range)

    isolated_probe = _QwenCompatibleAttentionProbe(
        vocab_size=vocab_size,
        use_packed_boundaries=True,
    )
    baseline = _run_probe(isolated_probe, batch)
    changed = _run_probe(isolated_probe, perturbed)

    hidden_delta = _max_delta_at_positions(
        baseline.final_hidden_state,
        changed.final_hidden_state,
        supervised_positions,
    )
    logits_delta = _max_delta_at_positions(
        baseline.logits,
        changed.logits,
        supervised_positions,
    )
    assert hidden_delta <= 1e-7, (
        f"{direction.name}: segment {direction.target_label} hidden states changed "
        f"after perturbing segment {direction.source_label}: delta={hidden_delta}"
    )
    assert logits_delta <= 1e-6, (
        f"{direction.name}: segment {direction.target_label} logits changed "
        f"after perturbing segment {direction.source_label}: delta={logits_delta}"
    )

    leaky_probe = _QwenCompatibleAttentionProbe(
        vocab_size=vocab_size,
        use_packed_boundaries=False,
    )
    leaky_baseline = _run_probe(leaky_probe, batch)
    leaky_changed = _run_probe(leaky_probe, perturbed)
    leaky_hidden_delta = _max_delta_at_positions(
        leaky_baseline.final_hidden_state,
        leaky_changed.final_hidden_state,
        supervised_positions,
    )
    assert leaky_hidden_delta > 1e-5, (
        f"{direction.name}: sensitivity check failed; an unsegmented causal "
        f"attention pass did not expose leakage at supervised positions"
    )


def test_static_packed_forward_has_no_cross_segment_attention_leakage(tmp_path) -> None:
    batch_ab, offsets_ab = _build_static_packed_detection_batch(tmp_path, order=(0, 1))
    _assert_packed_metadata_matches_offsets(batch_ab, offsets_ab)
    assert len(offsets_ab) == 3
    assert int(batch_ab["pack_num_samples"].reshape(-1)[0].item()) == 2

    vocab_size = int(batch_ab["input_ids"].max().item()) + 512
    _assert_direction_isolated(
        batch=batch_ab,
        direction=_PackedDirection(
            name="segment_0_to_segment_1",
            target_label="segment_1",
            source_label="segment_0",
            target_range=(offsets_ab[1], offsets_ab[2]),
            source_range=(offsets_ab[0], offsets_ab[1]),
        ),
        vocab_size=vocab_size,
    )

    # Qwen decoder attention is causal, so future segment text cannot affect an
    # earlier segment even without packed boundaries. Reversing the static pack
    # order puts the other sample in the previous-segment leakage-risk position.
    batch_ba, offsets_ba = _build_static_packed_detection_batch(tmp_path, order=(1, 0))
    _assert_packed_metadata_matches_offsets(batch_ba, offsets_ba)
    assert len(offsets_ba) == 3
    assert int(batch_ba["pack_num_samples"].reshape(-1)[0].item()) == 2

    vocab_size = max(vocab_size, int(batch_ba["input_ids"].max().item()) + 512)
    _assert_direction_isolated(
        batch=batch_ba,
        direction=_PackedDirection(
            name="segment_1_to_segment_0_when_ordered_first",
            target_label="segment_0",
            source_label="segment_1",
            target_range=(offsets_ba[1], offsets_ba[2]),
            source_range=(offsets_ba[0], offsets_ba[1]),
        ),
        vocab_size=vocab_size,
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


def test_coverage_ledger_dataset_sidecar_positions_follow_swift_encoded_labels(
    tmp_path,
) -> None:
    jsonl_path = tmp_path / "train.coord.jsonl"
    _write_jsonl(jsonl_path, [_raw_row()])
    _ensure_image(tmp_path)
    dataset = DetectionTrainingDataset.from_jsonl(
        jsonl_path,
        swift_template=_ImageGridExpandingSwiftTemplate(),
        image_root=tmp_path / "image-root",
        detection_template_id="compact_object_box_closed",
        mode="random_order_sft",
        object_ordering="sorted",
        user_prompt="Detect every object.",
        system_prompt="You are a detector.",
        seed=20260623,
        state_weighting="uniform_permutation",
        normalization="semantic_image_bucket_balanced",
        object_field_order="desc_first",
        teacher_forcing_profile="hard_sft",
        teacher_forcing_rollin_base_seed=17,
        coverage_ledger_enabled=True,
        dataset_name="unit",
    )

    sample = dataset[0]
    sidecar = sample["training_sidecars"].supervision.payloads[0]
    assert isinstance(sidecar, CoverageLedgerSidecar)
    ledger_entry = sidecar.object_entries[0]

    raw_coord_positions = tuple(
        sample["detection_supervision_view_metadata"]["coord_token_positions"]
    )
    assert ledger_entry.coord_label_positions[0] != raw_coord_positions[0]

    tokenizer = dataset.tokenizer
    input_ids = tuple(int(token_id) for token_id in sample["input_ids"])
    labels = tuple(int(token_id) for token_id in sample["labels"])
    expected_control_ids = {
        ledger_entry.object_ref_end_position: tokenizer.convert_tokens_to_ids(
            "<|object_ref_end|>"
        ),
        ledger_entry.box_start_position: tokenizer.convert_tokens_to_ids(
            "<|box_start|>"
        ),
        ledger_entry.box_end_position: tokenizer.convert_tokens_to_ids("<|box_end|>"),
    }
    for position, token_id in expected_control_ids.items():
        assert input_ids[position] == token_id
        assert labels[position] == token_id

    assert ledger_entry.coord_label_positions[0] == ledger_entry.box_start_position + 1
    for position, coord_value in zip(
        ledger_entry.coord_label_positions,
        ledger_entry.bbox_norm1000_xyxy,
        strict=True,
    ):
        coord_token_id = tokenizer.convert_tokens_to_ids(f"<|coord_{coord_value}|>")
        assert input_ids[position] == coord_token_id
        assert labels[position] == coord_token_id


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
