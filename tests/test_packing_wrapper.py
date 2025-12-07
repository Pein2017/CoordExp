import torch
import pytest

from src.datasets.wrappers.packed_caption import build_packed_dataset


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
            "input_ids": [0] * length,
            "labels": [0] * length,
            "length": length,
            "pixel_values": torch.zeros(1),
            "image_grid_thw": torch.tensor([1, 1, 1]),
        }

    def set_epoch(self, epoch):
        self.epoch_set = epoch


def _collect_pack_lengths(packs):
    return [sum(item["length"] for item in pack) for pack in packs]


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
    assert all(l >= 48 for l in lengths)  # 0.6 * 80


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
    assert packs == []  # underfilled and dropped


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
