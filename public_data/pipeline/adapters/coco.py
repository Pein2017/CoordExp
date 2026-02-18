from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from .base import DatasetAdapter


class CocoAdapter(DatasetAdapter):
    dataset_id = "coco"

    def download_raw_images(self, dataset_dir: Path, *, passthrough_args: Sequence[str] = ()) -> None:
        if not passthrough_args and self._download_raw_images_aria2c(dataset_dir):
            return

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

    @staticmethod
    def _download_raw_images_aria2c(dataset_dir: Path) -> bool:
        required_cmds = ("aria2c", "unzip", "sha256sum")
        if any(shutil.which(cmd) is None for cmd in required_cmds):
            return False

        dataset_dir = dataset_dir.resolve()
        raw_dir = dataset_dir / "raw"
        raw_image_dir = raw_dir / "images"
        downloads_dir = raw_dir / "downloads"
        downloads_dir.mkdir(parents=True, exist_ok=True)
        raw_image_dir.mkdir(parents=True, exist_ok=True)

        zip_targets = {
            "train2017.zip": "http://images.cocodataset.org/zips/train2017.zip",
            "val2017.zip": "http://images.cocodataset.org/zips/val2017.zip",
            "annotations_trainval2017.zip": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        }

        for filename, url in zip_targets.items():
            target = downloads_dir / filename
            if target.exists() and not (downloads_dir / f"{filename}.aria2").exists():
                print(f"[coco] skip download: already present {target}", file=sys.stderr)
                continue
            subprocess.run(
                ["aria2c", "-c", "-x", "16", "-s", "16", "-k", "1M", "-d", str(downloads_dir), url],
                check=True,
            )

        with (downloads_dir / "SHA256SUMS.txt").open("w", encoding="utf-8") as checksum_file:
            subprocess.run(
                ["sha256sum", "train2017.zip", "val2017.zip", "annotations_trainval2017.zip"],
                cwd=downloads_dir,
                stdout=checksum_file,
                check=True,
            )

        subprocess.run(["unzip", "-q", "-n", str(downloads_dir / "train2017.zip"), "-d", str(raw_image_dir)], check=True)
        subprocess.run(["unzip", "-q", "-n", str(downloads_dir / "val2017.zip"), "-d", str(raw_image_dir)], check=True)
        subprocess.run(
            ["unzip", "-q", "-n", str(downloads_dir / "annotations_trainval2017.zip"), "-d", str(raw_dir)],
            check=True,
        )
        return True
