from __future__ import annotations

from pathlib import Path

from .base import DatasetAdapter


class VgAdapter(DatasetAdapter):
    dataset_id = "vg"

    def download_raw_images(self, dataset_dir: Path) -> None:
        raise RuntimeError(
            "VG downloads are owned by runner/plugin compatibility wrappers. "
            "Use public_data/run.sh vg download."
        )

    def download_and_parse_annotations(self, dataset_dir: Path) -> None:
        raise RuntimeError(
            "VG annotation parsing is owned by runner/plugin compatibility wrappers. "
            "Use public_data/run.sh vg convert."
        )
