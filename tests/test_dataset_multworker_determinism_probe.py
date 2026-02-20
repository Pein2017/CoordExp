from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from src.config.schema import CoordTokensConfig
from src.datasets.dense_caption import BaseCaptionDataset


class _FakeTemplate:
    # Minimal template shim used by BaseCaptionDataset for CPU-only determinism probes.
    max_pixels = 786432
    system = None

    def encode(self, merged: Dict[str, Any], return_length: bool = True) -> Dict[str, Any]:
        # The dataset attaches messages/assistant_payload downstream; we only need a mutable mapping.
        return {"input_ids": [0], "labels": [0], "length": 1}


def _first(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    assert len(batch) == 1
    return batch[0]


def _make_record(*, image: str, objects: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "images": [image],
        "width": 32,
        "height": 32,
        "objects": objects,
    }


def _collect_samples(*, num_workers: int) -> List[Tuple[int, List[str]]]:
    records = [
        _make_record(
            image="img_0.png",
            objects=[
                {"desc": "a", "bbox_2d": ["<|coord_0|>", "<|coord_0|>", "<|coord_1|>", "<|coord_1|>"]},
                {"desc": "b", "bbox_2d": ["<|coord_2|>", "<|coord_2|>", "<|coord_3|>", "<|coord_3|>"]},
                {"desc": "c", "bbox_2d": ["<|coord_4|>", "<|coord_4|>", "<|coord_5|>", "<|coord_5|>"]},
                {"desc": "d", "bbox_2d": ["<|coord_6|>", "<|coord_6|>", "<|coord_7|>", "<|coord_7|>"]},
            ],
        ),
        _make_record(
            image="img_1.png",
            objects=[
                {"desc": "e", "bbox_2d": ["<|coord_10|>", "<|coord_10|>", "<|coord_11|>", "<|coord_11|>"]},
                {"desc": "f", "bbox_2d": ["<|coord_12|>", "<|coord_12|>", "<|coord_13|>", "<|coord_13|>"]},
            ],
        ),
        _make_record(
            image="img_2.png",
            objects=[
                {"desc": "g", "bbox_2d": ["<|coord_20|>", "<|coord_20|>", "<|coord_21|>", "<|coord_21|>"]},
            ],
        ),
    ]

    ds = BaseCaptionDataset(
        base_records=records,
        template=_FakeTemplate(),
        user_prompt="prompt",
        emit_norm="none",
        json_format="standard",
        seed=123,
        coord_tokens=CoordTokensConfig(enabled=True, skip_bbox_norm=True),
        object_ordering="random",
        object_field_order="desc_first",
    )

    dl = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=_first,
        persistent_workers=False,
    )

    out: List[Tuple[int, List[str]]] = []
    for sample in dl:
        sample_id = int(sample["sample_id"])
        payload = sample.get("assistant_payload") or {}
        objects = payload.get("objects") or []
        descs = [str(obj.get("desc", "")) for obj in objects]
        out.append((sample_id, descs))
    return out


def test_dataset_multiworker_determinism_probe() -> None:
    # Determinism contract: changing worker count must not change per-sample randomness.
    # This catches order-sensitive RNG usage in __getitem__ under multi-worker prefetching.
    torch.manual_seed(0)

    single = _collect_samples(num_workers=0)
    multi = _collect_samples(num_workers=2)

    assert single == multi

