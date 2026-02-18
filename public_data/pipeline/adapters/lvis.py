from __future__ import annotations

from pathlib import Path

from .base import DatasetAdapter


class LvisAdapter(DatasetAdapter):
    dataset_id = "lvis"

    def download_raw_images(self, dataset_dir: Path) -> None:
        raise RuntimeError(
            "LVIS downloads are owned by runner/plugin compatibility wrappers. "
            "Use public_data/run.sh lvis download."
        )

    def download_and_parse_annotations(self, dataset_dir: Path) -> None:
        raise RuntimeError(
            "LVIS annotation parsing is owned by runner/plugin compatibility wrappers. "
            "Use public_data/run.sh lvis convert."
        )
