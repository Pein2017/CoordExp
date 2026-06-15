from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import torch

from src.data_collators.batch_extras_collator import build_batch_extras_collator
from src.detection.dataset import strip_non_model_detection_sidecars
from src.detection.prefix_denoising.dataset import materialize_hybrid_model_ready_item
from src.detection.prefix_denoising.types import (
    HybridPrefixDenoisingSample,
    PrefixDenoisingKLSite,
    PrefixDenoisingSegment,
)
from src.trainers.batch_extras import (
    maybe_pop_and_stash_batch_extras,
    pop_batch_extras,
)
from src.trainers.metrics.mixins import PrefixDenoisingObjectiveMixin


class _DummyTemplate:
    tokenizer = None
    template_meta = None

    def data_collator(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        return _base_collator(batch)


def _base_collator(batch: list[dict[str, Any]]) -> dict[str, Any]:
    def _mask_1d(value: Any) -> torch.Tensor:
        tensor = value if isinstance(value, torch.Tensor) else torch.tensor(value)
        if tensor.ndim == 2 and tensor.shape[0] == 1:
            tensor = tensor[0]
        return tensor.long()

    return {
        "input_ids": torch.tensor([row["input_ids"] for row in batch], dtype=torch.long),
        "labels": torch.tensor([row["labels"] for row in batch], dtype=torch.long),
        "attention_mask": torch.stack(
            [_mask_1d(row["attention_mask"]) for row in batch]
        ),
        "pixel_values": torch.cat(
            [row["pixel_values"] for row in batch],
            dim=0,
        ),
        "image_grid_thw": torch.cat(
            [row["image_grid_thw"] for row in batch],
            dim=0,
        ),
    }


class _FuturePrefixDenoisingObjective:
    def compute_loss(
        self,
        *,
        labels: torch.Tensor,
        prefix_denoising_segment_meta: tuple[Any, ...],
        prefix_denoising_hybrid: tuple[HybridPrefixDenoisingSample, ...],
    ) -> dict[str, int]:
        return {
            "batch": int(labels.shape[0]),
            "segment_meta": len(prefix_denoising_segment_meta[0]),
            "hybrids": len(prefix_denoising_hybrid),
            "kl_sites": len(prefix_denoising_hybrid[0].kl_sites),
        }


class _DummyTrainer:
    pass


class _Metric:
    def __init__(self) -> None:
        self.values: list[float] = []

    def update(self, value: float) -> None:
        self.values.append(float(value))


class _CoordTokenizer:
    def convert_tokens_to_ids(self, tokens):
        return [
            1000 + int(str(token).split("_")[1].split("|")[0])
            for token in tokens
        ]


def _hybrid_sample() -> HybridPrefixDenoisingSample:
    clean = PrefixDenoisingSegment(
        segment_id="unit-0:clean",
        branch_id="clean_full",
        input_ids=(10, 101, 12, 13),
        labels=(-100, 101, -100, -100),
        attention_mask=(1, 1, 1, 1),
        supervised_positions=(1,),
        ce_denominator=1,
        metadata={
            "encoded_extras": {
                "pixel_values": torch.ones((1, 2, 3), dtype=torch.float32),
                "image_grid_thw": torch.tensor([[1, 1, 1]], dtype=torch.long),
            }
        },
    )
    noisy = PrefixDenoisingSegment(
        segment_id="unit-0:noisy",
        branch_id="noisy_full",
        input_ids=(20, 21, 22, 202),
        labels=(-100, -100, -100, 101),
        attention_mask=(1, 1, 1, 1),
        supervised_positions=(3,),
        ce_denominator=1,
        metadata={
            "encoded_extras": {
                "pixel_values": torch.zeros((1, 2, 3), dtype=torch.float32),
                "image_grid_thw": torch.tensor([[1, 1, 1]], dtype=torch.long),
            }
        },
    )
    return HybridPrefixDenoisingSample(
        ok=True,
        hybrid_sample_id="unit-0",
        base_sample_id="unit-base-0",
        clean_full=clean,
        noisy_full=noisy,
        kl_sites=(
            PrefixDenoisingKLSite(
                clean_segment_id=clean.segment_id,
                noisy_segment_id=noisy.segment_id,
                object_index=0,
                history_object_count=0,
                coord_slot="x1",
                clean_label_position=1,
                noisy_label_position=3,
                clean_gt_bin=101,
                support_bins=(99, 100, 101, 102),
            ),
        ),
    )


def _coord_hybrid_sample(sample_id: str) -> HybridPrefixDenoisingSample:
    clean = PrefixDenoisingSegment(
        segment_id=f"{sample_id}:clean",
        branch_id="clean_full",
        input_ids=(0, 9, 1010),
        labels=(-100, -100, 1010),
        attention_mask=(1, 1, 1),
        supervised_positions=(2,),
        ce_denominator=1,
    )
    noisy = PrefixDenoisingSegment(
        segment_id=f"{sample_id}:noisy",
        branch_id="noisy_full",
        input_ids=(0, 9, 1011),
        labels=(-100, -100, 1010),
        attention_mask=(1, 1, 1),
        supervised_positions=(2,),
        ce_denominator=1,
    )
    return HybridPrefixDenoisingSample(
        ok=True,
        hybrid_sample_id=sample_id,
        base_sample_id=f"base-{sample_id}",
        clean_full=clean,
        noisy_full=noisy,
        kl_sites=(
            PrefixDenoisingKLSite(
                clean_segment_id=clean.segment_id,
                noisy_segment_id=noisy.segment_id,
                object_index=0,
                history_object_count=0,
                coord_slot="x1",
                clean_label_position=2,
                noisy_label_position=2,
                clean_gt_bin=10,
                support_bins=(9, 10, 11),
            ),
        ),
    )


def _packed_prefix_base_collator(batch: list[Any]) -> dict[str, torch.Tensor]:
    rows: list[dict[str, list[int]]] = []
    for pack in batch:
        pack_seq = pack if isinstance(pack, (list, tuple)) else [pack]
        row = {"input_ids": [], "labels": [], "attention_mask": []}
        for sample in pack_seq:
            row["input_ids"].extend(sample["input_ids"])
            row["labels"].extend(sample["labels"])
            mask = sample["attention_mask"]
            if isinstance(mask, torch.Tensor):
                mask = mask.reshape(-1).tolist()
            row["attention_mask"].extend(mask)
        rows.append(row)
    return {
        "input_ids": torch.tensor([row["input_ids"] for row in rows], dtype=torch.long),
        "labels": torch.tensor([row["labels"] for row in rows], dtype=torch.long),
        "attention_mask": torch.tensor(
            [row["attention_mask"] for row in rows],
            dtype=torch.long,
        ),
    }


class _PackedPrefixTrainer(PrefixDenoisingObjectiveMixin):
    prefix_denoising_packing_enabled = True
    prefix_denoising_kl_weight = 0.25
    model = None
    args = SimpleNamespace(gradient_accumulation_steps=1)

    def __init__(self) -> None:
        self.custom_metrics = {"train": defaultdict(_Metric)}
        self.tokenizer = _CoordTokenizer()


class _UnpackedPrefixTrainer(PrefixDenoisingObjectiveMixin):
    prefix_denoising_packing_enabled = False
    prefix_denoising_kl_weight = 0.0
    model = None
    args = SimpleNamespace(gradient_accumulation_steps=1)

    def __init__(self) -> None:
        self.custom_metrics = {"train": defaultdict(_Metric)}
        self.tokenizer = _CoordTokenizer()


class _CleanPrefixLeakProbeModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.training = True
        self.config = SimpleNamespace(model_type="unit")

    def __call__(self, **kwargs):
        self.calls.append(dict(kwargs))
        input_ids = kwargs["input_ids"]
        logits = torch.full(
            (int(input_ids.shape[0]), int(input_ids.shape[1]), 2100),
            -8.0,
            dtype=torch.float32,
        )
        for row_index, row in enumerate(input_ids.tolist()):
            # Clean branch loss is intentionally invariant to the perturbation.
            logits[row_index, 0, 101] = 8.0
            if len(row) >= 8:
                # Simulate a causal leak: noisy branch logits depend on clean tokens
                # that precede the noisy segment in the same model row.
                if 999 in row[:4]:
                    logits[row_index, 6, 202] = 8.0
                else:
                    logits[row_index, 6, 101] = 8.0
            else:
                # Isolated noisy segment forward: clean tokens are absent, so the
                # noisy prediction cannot depend on them.
                logits[row_index, 2, 101] = 8.0
        return SimpleNamespace(logits=logits)


class _PackedPrefixModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.training = True
        self.config = SimpleNamespace(model_type="unit")

    def __call__(self, **kwargs):
        self.calls.append(dict(kwargs))
        input_ids = kwargs["input_ids"]
        logits = torch.full(
            (int(input_ids.shape[0]), int(input_ids.shape[1]), 2100),
            -8.0,
            dtype=torch.float32,
        )
        for row_index, row in enumerate(input_ids.tolist()):
            if int(row[2]) == 1010:
                logits[row_index, 1, 1010] = 7.0
            else:
                logits[row_index, 1, 1010] = 5.0
                logits[row_index, 1, 1011] = 9.0
        return SimpleNamespace(logits=logits)


def _replace_clean_token(
    sample: HybridPrefixDenoisingSample,
    *,
    token_index: int,
    token_id: int,
) -> HybridPrefixDenoisingSample:
    assert sample.clean_full is not None
    clean_ids = list(sample.clean_full.input_ids)
    clean_ids[int(token_index)] = int(token_id)
    return replace(
        sample,
        clean_full=replace(sample.clean_full, input_ids=tuple(clean_ids)),
    )


def test_prefix_denoising_model_ready_sidecars_survive_until_model_boundary() -> None:
    sample = _hybrid_sample()
    item = materialize_hybrid_model_ready_item(
        sample=sample,
        dataset_name="unit",
        base_idx=0,
    )
    assert isinstance(item["attention_mask"], torch.Tensor)
    assert item["attention_mask"].dtype == torch.long
    assert item["attention_mask"].shape == torch.Size([1, 8])
    collator = build_batch_extras_collator(
        _DummyTemplate(),
        base_collator=_base_collator,
    )

    collated = collator([item])

    assert set(["input_ids", "labels", "attention_mask"]).issubset(collated)
    assert collated["input_ids"].shape == torch.Size([1, 8])
    assert collated["labels"].shape == torch.Size([1, 8])
    assert collated["attention_mask"].shape == torch.Size([1, 8])
    assert collated["pixel_values"].shape == torch.Size([2, 2, 3])
    assert collated["image_grid_thw"].tolist() == [[1, 1, 1], [1, 1, 1]]

    segment_meta = collated["prefix_denoising_segment_meta"]
    assert segment_meta[0][0]["branch_id"] == "clean_full"
    assert segment_meta[0][1]["branch_id"] == "noisy_full"
    assert collated["prefix_denoising_hybrid"] == (sample,)

    clean_meta, noisy_meta = segment_meta[0]
    kl_site = sample.kl_sites[0]
    clean_local = int(kl_site.clean_label_position)
    noisy_local = int(kl_site.noisy_label_position)
    assert clean_local in clean_meta["local_supervised_positions"]
    assert noisy_local in noisy_meta["local_supervised_positions"]
    clean_physical = int(clean_meta["local_token_start"]) + clean_local
    noisy_physical = int(noisy_meta["local_token_start"]) + noisy_local
    assert collated["labels"][0, clean_physical].item() == kl_site.clean_gt_bin
    assert collated["labels"][0, noisy_physical].item() == kl_site.clean_gt_bin

    future_loss_inputs = _FuturePrefixDenoisingObjective().compute_loss(
        labels=collated["labels"],
        prefix_denoising_segment_meta=collated["prefix_denoising_segment_meta"],
        prefix_denoising_hybrid=collated["prefix_denoising_hybrid"],
    )
    assert future_loss_inputs == {
        "batch": 1,
        "segment_meta": 2,
        "hybrids": 1,
        "kl_sites": 1,
    }

    model_inputs = dict(collated)
    extras = pop_batch_extras(model_inputs)
    assert extras.prefix_denoising_hybrid == (sample,)
    assert extras.prefix_denoising_segment_meta == segment_meta
    assert extras.prefix_denoising_resolved_kl_sites is None
    assert extras.packed_hybrid_boundary_map is None
    assert "prefix_denoising_hybrid" not in model_inputs
    assert "prefix_denoising_segment_meta" not in model_inputs

    stripped = strip_non_model_detection_sidecars(model_inputs)
    assert "prefix_denoising_hybrid" not in stripped
    assert "prefix_denoising_segment_meta" not in stripped
    assert "packed_hybrid_boundary_map" not in stripped
    assert "prefix_denoising_resolved_kl_sites" not in stripped
    assert set(stripped) == {
        "input_ids",
        "labels",
        "attention_mask",
        "pixel_values",
        "image_grid_thw",
    }


def test_prefix_denoising_template_data_collator_path_stashes_batch_extras() -> None:
    sample = _hybrid_sample()
    item = materialize_hybrid_model_ready_item(
        sample=sample,
        dataset_name="unit",
        base_idx=0,
    )
    collator = build_batch_extras_collator(_DummyTemplate())
    collated = collator([item])
    trainer = _DummyTrainer()

    extras = maybe_pop_and_stash_batch_extras(trainer, collated)
    stripped = strip_non_model_detection_sidecars(collated)

    assert extras.prefix_denoising_hybrid == (sample,)
    assert extras.prefix_denoising_segment_meta[0][0]["branch_id"] == "clean_full"
    assert extras.prefix_denoising_segment_meta[0][1]["branch_id"] == "noisy_full"
    assert "prefix_denoising_hybrid" not in stripped
    assert "prefix_denoising_segment_meta" not in stripped
    assert set(stripped) == {
        "input_ids",
        "labels",
        "attention_mask",
        "pixel_values",
        "image_grid_thw",
    }


def test_noisy_branch_logits_are_isolated_from_clean_branch_tokens() -> None:
    base = _hybrid_sample()
    perturbed = _replace_clean_token(base, token_index=1, token_id=999)
    base_item = materialize_hybrid_model_ready_item(
        sample=base,
        dataset_name="unit",
        base_idx=0,
    )
    perturbed_item = materialize_hybrid_model_ready_item(
        sample=perturbed,
        dataset_name="unit",
        base_idx=0,
    )
    collator = build_batch_extras_collator(
        _DummyTemplate(),
        base_collator=_base_collator,
    )

    trainer = _UnpackedPrefixTrainer()
    base_model = _CleanPrefixLeakProbeModel()
    base_loss = trainer.compute_loss(base_model, collator([base_item]))
    perturbed_model = _CleanPrefixLeakProbeModel()
    perturbed_loss = trainer.compute_loss(perturbed_model, collator([perturbed_item]))

    torch.testing.assert_close(base_loss, perturbed_loss)
    assert all(
        call["input_ids"].shape[-1] == len(base.clean_full.input_ids)  # type: ignore[index, union-attr]
        for call in perturbed_model.calls
    )


def test_prefix_denoising_packed_collator_enriches_and_objective_runs_ce_kl() -> None:
    first = materialize_hybrid_model_ready_item(
        sample=_coord_hybrid_sample("pack-0"),
        dataset_name="unit",
        base_idx=0,
    )
    second = materialize_hybrid_model_ready_item(
        sample=_coord_hybrid_sample("pack-1"),
        dataset_name="unit",
        base_idx=1,
    )
    collator = build_batch_extras_collator(
        _DummyTemplate(),
        base_collator=_packed_prefix_base_collator,
    )
    collated = collator([[first, second]])

    assert collated["packed_hybrid_boundary_map"] == (
        (
            {"sample_index": 0, "start": 0, "end": 6},
            {"sample_index": 1, "start": 6, "end": 12},
        ),
    )
    second_clean = collated["prefix_denoising_segment_meta"][0][1][0]
    assert second_clean["local_token_start"] == 0
    assert second_clean["token_start"] == 6
    second_site = collated["prefix_denoising_resolved_kl_sites"][0][1][0]
    assert second_site.clean_label_position == 8
    assert second_site.noisy_label_position == 11

    trainer = _PackedPrefixTrainer()
    model = _PackedPrefixModel()
    loss = trainer.compute_loss(model, collated)

    assert torch.isfinite(loss)
    assert trainer.custom_metrics["train"][
        "prefix_denoising/kl/local_window/raw"
    ].values[-1] > 0.0
    assert len(model.calls) == 2
    assert all(call["input_ids"].shape == torch.Size([2, 3]) for call in model.calls)
    assert "prefix_denoising_segment_meta" not in model.calls[0]
    assert "packed_hybrid_boundary_map" not in model.calls[0]
    assert "prefix_denoising_resolved_kl_sites" not in model.calls[0]
