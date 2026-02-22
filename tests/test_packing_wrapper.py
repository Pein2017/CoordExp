import json
from pathlib import Path

import torch
import pytest

from src.datasets.wrappers.packed_caption import (
    build_packed_dataset,
    build_static_packed_dataset,
)


class _FakeTemplate:
    def __init__(self, max_length=80):
        self.max_length = max_length
        self.packing = False
        self.padding_free = False


class _FakeDataset:
    def __init__(self, lengths):
        self.lengths = list(lengths)
        self.epoch_set = None

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, idx):
        length = self.lengths[idx]
        return {
            "idx": idx,
            "input_ids": [0] * length,
            "labels": [0] * length,
            "length": length,
            "pixel_values": torch.zeros(1),
            "image_grid_thw": torch.tensor([1, 1, 1]),
        }

    def set_epoch(self, epoch):
        self.epoch_set = epoch


class _OrderSensitiveDataset:
    def __init__(self, size: int = 8):
        self.size = int(size)
        self.calls = 0

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        self.calls += 1
        jitter = self.calls % 2
        length = 16 + ((int(idx) + jitter) % 3)
        return {
            "idx": int(idx),
            "input_ids": [0] * int(length),
            "labels": [0] * int(length),
            "length": int(length),
            "pixel_values": torch.zeros(1),
            "image_grid_thw": torch.tensor([1, 1, 1]),
        }


class FusionCaptionDataset:
    def __init__(self, lengths):
        self.lengths = list(lengths)

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, idx):
        length = self.lengths[idx]
        return {
            "idx": idx,
            "input_ids": [0] * length,
            "labels": [0] * length,
            "length": length,
            "pixel_values": torch.zeros(1),
            "image_grid_thw": torch.tensor([1, 1, 1]),
        }


def _collect_pack_lengths(packs):
    return [sum(item["length"] for item in pack) for pack in packs]


def _static_fingerprint(tag: str) -> dict:
    return {
        "test_case": tag,
        "seed": 13,
    }


def test_packing_respects_bounds_and_fill_ratio():
    dataset = _FakeDataset([30, 30, 20, 60, 50, 10])
    template = _FakeTemplate(max_length=80)
    wrapped = build_packed_dataset(
        dataset,
        template=template,
        packing_length=80,
        buffer_size=6,
        min_fill_ratio=0.6,
        drop_last=True,
        allow_single_long=True,
    )
    packs = list(wrapped)
    lengths = _collect_pack_lengths(packs)
    assert all(l <= 80 for l in lengths)
    assert all(l >= 48 for l in lengths)


def test_drop_last_underfilled_group():
    dataset = _FakeDataset([20, 15])
    template = _FakeTemplate(max_length=80)
    wrapped = build_packed_dataset(
        dataset,
        template=template,
        packing_length=80,
        buffer_size=2,
        min_fill_ratio=0.6,
        drop_last=True,
        allow_single_long=True,
    )
    packs = list(wrapped)
    assert packs == []


def test_allow_single_long():
    dataset = _FakeDataset([90])
    template = _FakeTemplate(max_length=80)
    wrapped = build_packed_dataset(
        dataset,
        template=template,
        packing_length=80,
        buffer_size=1,
        min_fill_ratio=0.6,
        drop_last=True,
        allow_single_long=True,
    )
    packs = list(wrapped)
    assert len(packs) == 1
    assert _collect_pack_lengths(packs)[0] == 90


def test_skip_single_long_when_disallowed(caplog):
    dataset = _FakeDataset([90])
    template = _FakeTemplate(max_length=80)
    wrapped = build_packed_dataset(
        dataset,
        template=template,
        packing_length=80,
        buffer_size=1,
        min_fill_ratio=0.6,
        drop_last=True,
        allow_single_long=False,
    )
    with caplog.at_level(
        "WARNING", logger="swift.custom.src.datasets.wrappers.packed_caption"
    ):
        packs = list(wrapped)
    assert packs == []


def test_set_epoch_forwarding():
    dataset = _FakeDataset([10, 10])
    template = _FakeTemplate(max_length=32)
    wrapped = build_packed_dataset(
        dataset,
        template=template,
        packing_length=32,
        buffer_size=2,
    )
    wrapped.set_epoch(3)
    assert dataset.epoch_set == 3


def test_template_flags_set_on_init():
    dataset = _FakeDataset([10])
    template = _FakeTemplate(max_length=32)
    _ = build_packed_dataset(
        dataset,
        template=template,
        packing_length=32,
        buffer_size=1,
    )
    assert template.packing is True
    assert template.padding_free is True


def test_packing_preserves_stable_intra_pack_order():
    dataset = _FakeDataset([50, 30, 50, 30])
    template = _FakeTemplate(max_length=80)
    wrapped = build_packed_dataset(
        dataset,
        template=template,
        packing_length=80,
        buffer_size=4,
        min_fill_ratio=0.6,
        drop_last=False,
        allow_single_long=True,
    )
    packs = list(wrapped)
    assert packs

    for pack in packs:
        idxs = [int(item["idx"]) for item in pack]
        assert idxs == sorted(idxs)


def test_static_packing_deterministic_plan(tmp_path: Path):
    lengths = [30, 30, 20, 60, 50, 10, 40, 30]

    run1 = build_static_packed_dataset(
        _FakeDataset(lengths),
        template=_FakeTemplate(max_length=80),
        packing_length=80,
        min_fill_ratio=0.5,
        packing_drop_last=False,
        dataloader_drop_last=False,
        allow_single_long=True,
        cache_dir=tmp_path / "run1",
        fingerprint=_static_fingerprint("deterministic"),
        world_size=2,
        train_dataloader_shuffle=True,
    )
    run2 = build_static_packed_dataset(
        _FakeDataset(lengths),
        template=_FakeTemplate(max_length=80),
        packing_length=80,
        min_fill_ratio=0.5,
        packing_drop_last=False,
        dataloader_drop_last=False,
        allow_single_long=True,
        cache_dir=tmp_path / "run2",
        fingerprint=_static_fingerprint("deterministic"),
        world_size=2,
        train_dataloader_shuffle=True,
    )

    assert run1.raw_plan == run2.raw_plan
    assert run1.pack_plan == run2.pack_plan
    assert run1.raw_plan_checksum == run2.raw_plan_checksum
    assert run1.aligned_plan_checksum == run2.aligned_plan_checksum


def test_static_packing_ddp_drop_last_truncates_tail(tmp_path: Path):
    dataset = _FakeDataset([40] * 10)
    static_ds = build_static_packed_dataset(
        dataset,
        template=_FakeTemplate(max_length=80),
        packing_length=80,
        min_fill_ratio=0.5,
        packing_drop_last=False,
        dataloader_drop_last=True,
        allow_single_long=True,
        cache_dir=tmp_path / "drop_true",
        fingerprint=_static_fingerprint("drop_true"),
        world_size=4,
        train_dataloader_shuffle=False,
    )

    assert len(static_ds.raw_plan) == 5
    assert len(static_ds) == 4
    assert static_ds.pack_plan == static_ds.raw_plan[:4]
    assert static_ds.pad_needed == 0
    assert static_ds.repeated_pack_indices == []


def test_static_packing_ddp_pad_repeats_prefix(tmp_path: Path):
    dataset = _FakeDataset([40] * 10)
    static_ds = build_static_packed_dataset(
        dataset,
        template=_FakeTemplate(max_length=80),
        packing_length=80,
        min_fill_ratio=0.5,
        packing_drop_last=False,
        dataloader_drop_last=False,
        allow_single_long=True,
        cache_dir=tmp_path / "drop_false",
        fingerprint=_static_fingerprint("drop_false"),
        world_size=4,
        train_dataloader_shuffle=False,
    )

    assert len(static_ds.raw_plan) == 5
    assert len(static_ds) == 8
    assert static_ds.pack_plan[:5] == static_ds.raw_plan
    assert static_ds.pack_plan[5:] == static_ds.raw_plan[:3]
    assert static_ds.pad_needed == 3
    assert static_ds.repeated_pack_indices == [0, 1, 2]

    raw_set = {tuple(pack) for pack in static_ds.raw_plan}
    aligned_set = {tuple(pack) for pack in static_ds.pack_plan}
    assert raw_set.issubset(aligned_set)


def test_static_packing_computes_missing_length_cache_entries(tmp_path: Path):
    cache_dir = tmp_path / "resume_cache"
    lengths = [20, 24, 28, 32]

    _ = build_static_packed_dataset(
        _FakeDataset(lengths),
        template=_FakeTemplate(max_length=64),
        packing_length=64,
        min_fill_ratio=0.5,
        packing_drop_last=False,
        dataloader_drop_last=False,
        allow_single_long=True,
        cache_dir=cache_dir,
        fingerprint=_static_fingerprint("resume"),
        world_size=1,
        train_dataloader_shuffle=False,
    )

    length_cache_path = cache_dir / "lengths.json"
    payload = json.loads(length_cache_path.read_text(encoding="utf-8"))
    payload["lengths"][2] = None
    length_cache_path.write_text(json.dumps(payload), encoding="utf-8")

    _ = build_static_packed_dataset(
        _FakeDataset(lengths),
        template=_FakeTemplate(max_length=64),
        packing_length=64,
        min_fill_ratio=0.5,
        packing_drop_last=False,
        dataloader_drop_last=False,
        allow_single_long=True,
        cache_dir=cache_dir,
        fingerprint=_static_fingerprint("resume"),
        world_size=1,
        train_dataloader_shuffle=False,
    )

    payload_after = json.loads(length_cache_path.read_text(encoding="utf-8"))
    assert all(v is not None for v in payload_after["lengths"])


def test_static_packing_rejects_order_sensitive_dataset(tmp_path: Path):
    with pytest.raises(ValueError, match="order-invariant"):
        build_static_packed_dataset(
            _OrderSensitiveDataset(size=12),
            template=_FakeTemplate(max_length=64),
            packing_length=64,
            min_fill_ratio=0.5,
            packing_drop_last=False,
            dataloader_drop_last=False,
            allow_single_long=True,
            cache_dir=tmp_path / "order_sensitive",
            fingerprint=_static_fingerprint("order_sensitive"),
            world_size=1,
            train_dataloader_shuffle=False,
        )


def test_static_packing_rejects_fusion_dataset(tmp_path: Path):
    with pytest.raises(ValueError, match="fusion/mixing"):
        build_static_packed_dataset(
            FusionCaptionDataset([10, 12, 14, 16]),
            template=_FakeTemplate(max_length=32),
            packing_length=32,
            min_fill_ratio=0.5,
            packing_drop_last=False,
            dataloader_drop_last=False,
            allow_single_long=True,
            cache_dir=tmp_path / "fusion",
            fingerprint=_static_fingerprint("fusion"),
            world_size=1,
            train_dataloader_shuffle=False,
        )
