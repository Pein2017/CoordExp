from __future__ import annotations

import json
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterable, Iterator, Sequence

from public_data.pipeline.types import SPLITS


class DatasetAdapter(ABC):
    """Source adapter boundary for dataset-specific ingestion hooks."""

    dataset_id: str

    @abstractmethod
    def download_raw_images(self, dataset_dir: Path, *, passthrough_args: Sequence[str] = ()) -> None:
        """Dataset-specific raw image download hook."""

    @abstractmethod
    def download_and_parse_annotations(
        self,
        dataset_dir: Path,
        *,
        passthrough_args: Sequence[str] = (),
    ) -> None:
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

    def _run_plugin_ingestion(
        self,
        *,
        dataset_dir: Path,
        subcommand: str,
        passthrough_args: Sequence[str] = (),
    ) -> None:
        if subcommand not in {"download", "convert"}:
            raise ValueError(f"Unsupported ingestion subcommand: {subcommand}")

        repo_root = self._infer_repo_root(dataset_dir)
        plugin_path = repo_root / "public_data" / "datasets" / f"{self.dataset_id}.sh"
        if not plugin_path.exists():
            raise FileNotFoundError(
                f"Missing plugin for dataset '{self.dataset_id}': {plugin_path}. "
                "Expected public_data/datasets/<dataset>.sh to exist."
            )

        dataset_dir = dataset_dir.resolve()
        raw_dir = dataset_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "bash",
            str(plugin_path),
            subcommand,
            "--repo-root",
            str(repo_root),
            "--dataset",
            self.dataset_id,
            "--dataset-dir",
            str(dataset_dir),
            "--raw-dir",
            str(raw_dir),
            "--raw-image-dir",
            str(raw_dir / "images"),
            "--raw-train-jsonl",
            str(raw_dir / "train.jsonl"),
            "--raw-val-jsonl",
            str(raw_dir / "val.jsonl"),
        ]

        if passthrough_args:
            cmd.extend(["--", *passthrough_args])

        subprocess.run(cmd, cwd=repo_root, check=True)

    @staticmethod
    def _infer_repo_root(dataset_dir: Path) -> Path:
        dataset_dir = dataset_dir.resolve()
        if dataset_dir.parent.name == "public_data":
            repo_root = dataset_dir.parent.parent
        else:
            repo_root = Path.cwd()

        datasets_dir = repo_root / "public_data" / "datasets"
        if not datasets_dir.exists():
            raise FileNotFoundError(
                f"Could not infer repo root for dataset dir '{dataset_dir}'. "
                f"Expected datasets directory at '{datasets_dir}'."
            )
        return repo_root
