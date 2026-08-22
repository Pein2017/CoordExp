#!/usr/bin/env python3
"""Build current COCO80 full or deterministic length-budget annotation views."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Sequence

from public_data.scripts.convert_to_coord_tokens import convert_record_to_ints
from public_data.view_contracts import (
    ASSISTANT_COORDINATE_RENDERING_QWEN_COORD_TOKENS,
    COORDINATE_CHART_XYXY,
    COORDINATE_RANGE_NORM1000,
    COORDINATE_SPACE_NORM1000,
    COORDINATE_STORAGE_INTEGER,
    IMAGE_PATH_SEMANTICS_IMAGE_STORE_RELATIVE,
    SCHEMA_VERSION_V1,
    write_image_store_metadata,
    write_view_metadata,
)


def estimate_total_tokens(row: dict[str, Any]) -> int:
    """Deterministic preparation-only upper-bound used for view membership."""
    width, height = int(row["width"]), int(row["height"])
    visual = ((width + 31) // 32) * ((height + 31) // 32)
    text = 96
    for obj in row.get("objects") or []:
        if isinstance(obj, dict):
            text += 12 + (len(str(obj.get("desc", ""))) + 3) // 4
    return visual + text


def build_views(
    *, source_preset: Path, image_store_root: Path, views_root: Path,
    splits: Sequence[str], views: Sequence[str], max_total_tokens: int,
    image_store_mode: str,
) -> dict[str, int]:
    if image_store_mode not in {"copy", "hardlink", "reuse-existing"}:
        raise ValueError("image-store-mode must be copy, hardlink, or reuse-existing")
    _materialize_image_store(source_preset / "images", image_store_root / "images", image_store_mode)
    write_image_store_metadata(
        image_store_root / "meta.json", dataset="coco", image_factor=32,
        image_path_semantics=IMAGE_PATH_SEMANTICS_IMAGE_STORE_RELATIVE,
    )
    totals: dict[str, int] = {}
    for view in views:
        if view not in {"coco80/full", "coco80/len-12000"}:
            raise ValueError(f"unsupported current view: {view}")
        view_root = views_root / Path(view)
        if view_root.exists() and any(view_root.iterdir()):
            raise RuntimeError(f"view target is not fresh: {view_root}")
        view_root.mkdir(parents=True, exist_ok=True)
        primary: dict[str, str] = {}
        view_records = 0
        view_objects = 0
        comparisons: dict[str, Any] = {}
        for split in splits:
            source = _split_source(source_preset, split)
            output = view_root / f"{split}.jsonl"
            accepted = 0
            rejected = 0
            object_count = 0
            with source.open("r", encoding="utf-8") as src, output.open("w", encoding="utf-8") as dst:
                for line_number, line in enumerate(src, start=1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if source.name.endswith(".jsonl") and not source.name.endswith(".norm.jsonl"):
                        row = convert_record_to_ints(
                            row, ("bbox_2d", "poly", "line"), assume_normalized=False
                        )
                    estimate = estimate_total_tokens(row)
                    if view.endswith("len-12000") and estimate > max_total_tokens:
                        rejected += 1
                        continue
                    row["images"] = [_image_store_relative(str(row["images"][0]))]
                    dst.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    accepted += 1
                    object_count += len(row.get("objects") or [])
            if accepted == 0:
                raise RuntimeError(f"view would be empty for {split}: {view}")
            primary[split] = output.name
            comparisons[split] = {
                "source": str(source), "source_sha256": _sha256(source),
                "accepted": accepted, "rejected": rejected,
            }
            view_records += accepted
            view_objects += object_count
        (view_root / "source_comparison.json").write_text(
            json.dumps(comparisons, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        write_view_metadata(
            view_root / "meta.json", schema_version=SCHEMA_VERSION_V1,
            kind="annotation_view", dataset="coco", view=view,
            image_store=str(image_store_root), path_anchor="repository_root",
            image_path_semantics=IMAGE_PATH_SEMANTICS_IMAGE_STORE_RELATIVE,
            coordinate_space=COORDINATE_SPACE_NORM1000,
            coordinate_storage=COORDINATE_STORAGE_INTEGER,
            coordinate_range=COORDINATE_RANGE_NORM1000,
            coordinate_chart=COORDINATE_CHART_XYXY,
            assistant_coordinate_rendering=ASSISTANT_COORDINATE_RENDERING_QWEN_COORD_TOKENS,
            primary_jsonl=primary,
            summary={"records": view_records, "rendered_object_count": view_objects},
            sample_policy="length_budget" if view.endswith("len-12000") else "full",
            max_total_tokens=max_total_tokens if view.endswith("len-12000") else None,
        )
        totals[view] = view_records
    return totals


def _split_source(root: Path, split: str) -> Path:
    for name in (f"{split}.norm.jsonl", f"{split}.jsonl"):
        candidate = root / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"missing source JSONL for split {split!r} under {root}")


def _image_store_relative(value: str) -> str:
    path = Path(value)
    parts = path.parts
    if "images" in parts:
        return Path(*parts[parts.index("images") :]).as_posix()
    return f"images/{path.name}"


def _materialize_image_store(source: Path, target: Path, mode: str) -> None:
    if mode == "reuse-existing":
        if not target.is_dir() or not any(target.rglob("*")):
            raise FileNotFoundError(f"reusable image store is absent or empty: {target}")
        return
    if target.exists() and any(target.rglob("*")):
        raise RuntimeError(f"image-store target is not fresh: {target}")
    for image in source.rglob("*"):
        if not image.is_file():
            continue
        destination = target / image.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if mode == "copy":
            shutil.copy2(image, destination)
        else:
            os.link(image, destination)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-preset", type=Path, required=True)
    parser.add_argument("--image-store-root", type=Path, required=True)
    parser.add_argument("--views-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--views", nargs="+", default=["coco80/full", "coco80/len-12000"])
    parser.add_argument("--max-total-tokens", type=int, default=12000)
    parser.add_argument("--image-store-mode", choices=["copy", "hardlink", "reuse-existing"], default="hardlink")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    totals = build_views(**vars(args))
    for name, records in totals.items():
        print(f"[view] {name}: {records} records")


if __name__ == "__main__":
    main()
