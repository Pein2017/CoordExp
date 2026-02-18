from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Sequence

from .base import DatasetAdapter
from .coco import CocoAdapter
from .lvis import LvisAdapter
from .vg import VgAdapter


class AdapterRegistry:
    def __init__(self) -> None:
        self._adapters: Dict[str, DatasetAdapter] = {}

    def register(self, adapter: DatasetAdapter) -> None:
        existing = self._adapters.get(adapter.dataset_id)
        if existing is not None:
            raise ValueError(
                f"Dataset id '{adapter.dataset_id}' is already registered "
                f"with adapter type '{type(existing).__name__}'."
            )
        self._adapters[adapter.dataset_id] = adapter

    def get(self, dataset_id: str) -> DatasetAdapter:
        if dataset_id not in self._adapters:
            available = ", ".join(sorted(self._adapters.keys()))
            raise KeyError(f"Unknown dataset id '{dataset_id}'. Available datasets: {available}")
        return self._adapters[dataset_id]

    def ids(self) -> Iterable[str]:
        return tuple(sorted(self._adapters.keys()))


class AliasAdapter(DatasetAdapter):
    def __init__(self, *, dataset_id: str, delegate: DatasetAdapter) -> None:
        self.dataset_id = dataset_id
        self._delegate = delegate

    def download_raw_images(self, dataset_dir: Path, *, passthrough_args: Sequence[str] = ()) -> None:
        self._delegate.download_raw_images(dataset_dir, passthrough_args=passthrough_args)

    def download_and_parse_annotations(
        self,
        dataset_dir: Path,
        *,
        passthrough_args: Sequence[str] = (),
    ) -> None:
        self._delegate.download_and_parse_annotations(dataset_dir, passthrough_args=passthrough_args)

    def source_normalize_record(self, record: dict, split: str) -> dict:
        return self._delegate.source_normalize_record(record, split)


def build_default_registry() -> AdapterRegistry:
    reg = AdapterRegistry()
    coco = CocoAdapter()
    lvis = LvisAdapter()
    vg = VgAdapter()

    reg.register(coco)
    reg.register(lvis)
    reg.register(vg)
    reg.register(AliasAdapter(dataset_id="vg_ref", delegate=vg))

    # Smoke aliases keep synthetic-runner tests isolated from real dataset folders.
    reg.register(AliasAdapter(dataset_id="smoke_coco", delegate=coco))
    reg.register(AliasAdapter(dataset_id="smoke_lvis", delegate=lvis))
    reg.register(AliasAdapter(dataset_id="smoke_vg", delegate=vg))
    reg.register(AliasAdapter(dataset_id="smoke_vg_ref", delegate=vg))
    return reg
