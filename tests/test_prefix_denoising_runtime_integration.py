from __future__ import annotations

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


class _DummyTemplate:
    tokenizer = None
    template_meta = None

    def data_collator(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        return _base_collator(batch)


def _base_collator(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "input_ids": torch.tensor([row["input_ids"] for row in batch], dtype=torch.long),
        "labels": torch.tensor([row["labels"] for row in batch], dtype=torch.long),
        "attention_mask": torch.tensor(
            [row["attention_mask"] for row in batch],
            dtype=torch.long,
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


def test_prefix_denoising_model_ready_sidecars_survive_until_model_boundary() -> None:
    sample = _hybrid_sample()
    item = materialize_hybrid_model_ready_item(
        sample=sample,
        dataset_name="unit",
        base_idx=0,
    )
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
