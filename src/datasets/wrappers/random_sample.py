"""Random-sample dataset wrapper for eval-time subsampling."""

from __future__ import annotations

import random
from typing import Any

from torch.utils.data import Dataset, get_worker_info



class RandomSampleDataset(Dataset):
    """Sample with replacement from a base dataset."""

    def __init__(self, dataset: Any, *, sample_size: int, seed: int = 0) -> None:
        super().__init__()
        if sample_size is None:
            raise ValueError("sample_size must be provided for RandomSampleDataset")
        sample_size = int(sample_size)
        if sample_size <= 0:
            raise ValueError("sample_size must be > 0 for RandomSampleDataset")
        self.dataset = dataset
        self.sample_size = sample_size
        self.seed = int(seed)
        self._epoch = 0
        self._rng = random.Random(self._seed_for_epoch(self._epoch))
        self.base_length = len(dataset)

    def _seed_for_epoch(self, epoch: int) -> int:
        base_seed = self.seed & 0xFFFFFFFF
        mixed = (base_seed ^ ((int(epoch) + 1) * 0x9E3779B1)) & 0xFFFFFFFF
        return mixed

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
        self._rng = random.Random(self._seed_for_epoch(self._epoch))
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

    def __len__(self) -> int:
        return self.sample_size

    def __getitem__(self, index: int) -> Any:
        try:
            base_len = len(self.dataset)
        except TypeError as exc:
            raise TypeError("RandomSampleDataset requires a sized base dataset") from exc
        if base_len <= 0:
            raise IndexError("RandomSampleDataset base dataset is empty")

        worker = get_worker_info()
        seed_local = self._rng.randrange(0, 2**32 - 1)
        if worker is not None:
            seed_local ^= ((worker.id + 1) * 0xC2B2AE35) & 0xFFFFFFFF
        rng_local = random.Random(seed_local & 0xFFFFFFFF)

        base_idx = rng_local.randrange(base_len)
        return self.dataset[base_idx]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.dataset, name)
