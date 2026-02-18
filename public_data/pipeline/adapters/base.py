from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterable, Iterator

from public_data.pipeline.types import SPLITS


class DatasetAdapter(ABC):
    """Source adapter boundary for dataset-specific ingestion hooks."""

    dataset_id: str

    @abstractmethod
    def download_raw_images(self, dataset_dir: Path) -> None:
        """Dataset-specific raw image download hook."""

    @abstractmethod
    def download_and_parse_annotations(self, dataset_dir: Path) -> None:
        """Dataset-specific annotation acquisition/parsing hook."""

    def source_normalize_record(self, record: dict, split: str) -> dict:
        """Source-specific normalization into canonical intermediate record."""
        return record

    def split_input_paths(self, raw_dir: Path) -> Dict[str, Path]:
        result: Dict[str, Path] = {}
        for split in SPLITS:
            p = raw_dir / f"{split}.jsonl"
            if p.exists():
                result[split] = p
        if "train" not in result:
            raise FileNotFoundError(f"Missing required train split JSONL: {raw_dir / 'train.jsonl'}")
        return result

    def iter_canonical_records(self, raw_dir: Path, split: str) -> Iterator[dict]:
        path = raw_dir / f"{split}.jsonl"
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                yield self.source_normalize_record(rec, split)

    def available_splits(self, raw_dir: Path) -> Iterable[str]:
        return tuple(self.split_input_paths(raw_dir).keys())
