from __future__ import annotations

from pathlib import Path

import pytest

from public_data.view_contracts import (
    ImageStoreMetadata,
    ViewMetadata,
    resolve_image_path,
    resolve_view_image_root,
)


def test_resolve_image_path_is_image_store_relative(tmp_path: Path) -> None:
    repo_root = tmp_path
    image_store_ref = "public_data/coco/images/res-1024"
    image_store = repo_root / image_store_ref
    image_path = image_store / "images" / "train2017" / "000000000001.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"fake")

    view_root = tmp_path / "public_data" / "coco" / "views" / "coco80" / "len-12000"
    view_root.mkdir(parents=True)
    meta = ViewMetadata(
        schema_version=1,
        kind="annotation_view",
        dataset="coco",
        view="coco80/len-12000",
        image_store=image_store_ref,
        path_anchor="repo_root",
        image_path_semantics="image_store_relative",
        coordinate_space="norm1000",
        coordinate_storage="integer",
        coordinate_range=(0, 999),
        coordinate_chart="xyxy",
        assistant_coordinate_rendering="qwen_coord_tokens",
        primary_jsonl={"train": "train.jsonl", "val": "val.jsonl"},
        sample_policy={"type": "length_budget", "max_total_tokens": 12000},
        length_budget_scope={
            "rendered_families": ["objects"],
            "excluded_sidecars": ["metadata.supervision.support_objects"],
        },
        summary={"records": 1, "rendered_object_count": 1, "support_sidecar_count": 0},
    )

    resolved_root = resolve_view_image_root(meta, view_root=view_root, repo_root=repo_root)
    assert resolved_root == image_store.resolve()
    assert (
        resolve_image_path(
            "images/train2017/000000000001.jpg",
            image_root=resolved_root,
        )
        == image_path.resolve()
    )


def test_rejects_escaped_image_path(tmp_path: Path) -> None:
    image_store = tmp_path / "store"
    image_store.mkdir()

    with pytest.raises(ValueError, match="outside image_root"):
        resolve_image_path("../raw/images/leak.jpg", image_root=image_store)


def test_rejects_absolute_image_store_with_absolute_anchor(tmp_path: Path) -> None:
    meta = ViewMetadata(
        schema_version=1,
        kind="annotation_view",
        dataset="coco",
        view="coco80/len-12000",
        image_store=str(tmp_path / "public_data" / "coco" / "images" / "res-1024"),
        path_anchor="absolute",
        image_path_semantics="image_store_relative",
        coordinate_space="norm1000",
        coordinate_storage="integer",
        coordinate_range=(0, 999),
        coordinate_chart="xyxy",
        assistant_coordinate_rendering="qwen_coord_tokens",
        primary_jsonl={"train": "train.jsonl", "val": "val.jsonl"},
    )

    with pytest.raises(ValueError, match="absolute image_store requires test/debug"):
        resolve_view_image_root(meta, view_root=tmp_path)


def test_rejects_relative_image_store_without_repo_root_anchor(tmp_path: Path) -> None:
    meta = ViewMetadata(
        schema_version=1,
        kind="annotation_view",
        dataset="coco",
        view="coco80/len-12000",
        image_store="public_data/coco/images/res-1024",
        path_anchor="debug_absolute",
        image_path_semantics="image_store_relative",
        coordinate_space="norm1000",
        coordinate_storage="integer",
        coordinate_range=(0, 999),
        coordinate_chart="xyxy",
        assistant_coordinate_rendering="qwen_coord_tokens",
        primary_jsonl={"train": "train.jsonl", "val": "val.jsonl"},
    )

    with pytest.raises(ValueError, match="relative image_store requires repo_root"):
        resolve_view_image_root(meta, view_root=tmp_path)
