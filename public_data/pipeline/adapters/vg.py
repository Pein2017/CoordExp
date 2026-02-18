from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .base import DatasetAdapter


class VgAdapter(DatasetAdapter):
    dataset_id = "vg"

    def download_raw_images(self, dataset_dir: Path, *, passthrough_args: Sequence[str] = ()) -> None:
        self._run_plugin_ingestion(
            dataset_dir=dataset_dir,
            subcommand="download",
            passthrough_args=passthrough_args,
        )

    def download_and_parse_annotations(
        self,
        dataset_dir: Path,
        *,
        passthrough_args: Sequence[str] = (),
    ) -> None:
        self._run_plugin_ingestion(
            dataset_dir=dataset_dir,
            subcommand="convert",
            passthrough_args=passthrough_args,
        )
