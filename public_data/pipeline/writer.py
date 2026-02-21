from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from .types import SplitArtifactPaths


def build_split_artifact_paths(preset_dir: Path, split: str) -> SplitArtifactPaths:
    return SplitArtifactPaths(
        split=split,
        raw=preset_dir / f"{split}.jsonl",
        norm=preset_dir / f"{split}.norm.jsonl",
        coord=preset_dir / f"{split}.coord.jsonl",
        filter_stats=preset_dir / f"{split}.filter_stats.json",
    )


def count_jsonl_records(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def write_pipeline_manifest(
    *,
    preset_dir: Path,
    dataset_id: str,
    preset: str,
    max_objects: int | None,
    split_paths: Dict[str, SplitArtifactPaths],
    stage_stats: dict,
) -> Path:
    manifest = {
        "dataset_id": dataset_id,
        "preset": preset,
        "max_objects": max_objects,
        "splits": {},
        "stage_stats": stage_stats,
    }
    for split, paths in split_paths.items():
        manifest["splits"][split] = {
            "raw": str(paths.raw),
            "norm": str(paths.norm),
            "coord": str(paths.coord),
            "counts": {
                "raw": count_jsonl_records(paths.raw),
                "norm": count_jsonl_records(paths.norm),
                "coord": count_jsonl_records(paths.coord),
            },
        }

    out = preset_dir / "pipeline_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def collect_existing_artifacts(split_paths: Dict[str, SplitArtifactPaths]) -> list[Path]:
    out: list[Path] = []
    for split in sorted(split_paths.keys()):
        paths = split_paths[split]
        for candidate in (paths.raw, paths.norm, paths.coord):
            if candidate.exists():
                out.append(candidate)
    return out
