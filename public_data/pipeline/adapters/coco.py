from __future__ import annotations

from pathlib import Path

from .base import DatasetAdapter


class CocoAdapter(DatasetAdapter):
    dataset_id = "coco"

    def download_raw_images(self, dataset_dir: Path) -> None:
        raise RuntimeError(
            "COCO downloads are owned by runner/plugin compatibility wrappers. "
            "Use public_data/run.sh coco download."
        )

    def download_and_parse_annotations(self, dataset_dir: Path) -> None:
        raise RuntimeError(
            "COCO annotation parsing is owned by runner/plugin compatibility wrappers. "
            "Use public_data/run.sh coco convert."
        )
