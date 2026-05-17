from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from public_data.view_contracts import (
    ImageStoreMetadata,
    ViewMetadata,
    load_view_metadata,
    resolve_image_path,
    resolve_view_image_root,
    write_view_metadata,
)


def _valid_view_metadata(**overrides: object) -> dict[str, object]:
    metadata: dict[str, object] = {
        "schema_version": 1,
        "kind": "annotation_view",
        "dataset": "coco",
        "view": "coco80/len-12000",
        "image_store": "public_data/coco/images/res-1024",
        "path_anchor": "repo_root",
        "image_path_semantics": "image_store_relative",
        "coordinate_space": "norm1000",
        "coordinate_storage": "integer",
        "coordinate_range": [0, 999],
        "coordinate_chart": "xyxy",
        "assistant_coordinate_rendering": "qwen_coord_tokens",
        "primary_jsonl": {"train": "train.jsonl", "val": "val.jsonl"},
    }
    metadata.update(overrides)
    return metadata


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


def test_infers_repo_root_from_innermost_public_data_component(tmp_path: Path) -> None:
    outer_public_data = tmp_path / "public_data"
    repo_root = outer_public_data / "checkout"
    view_root = repo_root / "public_data" / "coco" / "views" / "coco80" / "len-12000"
    meta = ViewMetadata(**_valid_view_metadata())

    resolved_root = resolve_view_image_root(meta, view_root=view_root, repo_root=None)

    assert resolved_root == (
        repo_root / "public_data" / "coco" / "images" / "res-1024"
    ).resolve()
    assert resolved_root != (
        outer_public_data / "coco" / "images" / "res-1024"
    ).resolve()


def test_write_and_load_view_metadata_round_trips_valid_metadata(tmp_path: Path) -> None:
    metadata_path = tmp_path / "nested" / "view" / "metadata.json"

    write_view_metadata(metadata_path, _valid_view_metadata())

    assert metadata_path.exists()
    loaded = load_view_metadata(metadata_path)
    assert loaded.dataset == "coco"
    assert loaded.view == "coco80/len-12000"
    assert loaded.image_store == "public_data/coco/images/res-1024"
    assert loaded.path_anchor == "repo_root"
    assert loaded.coordinate_range == (0, 999)


def test_image_store_metadata_is_frozen_public_contract() -> None:
    metadata = ImageStoreMetadata(
        schema_version=1,
        kind="image_store",
        dataset="coco",
        image_store="res-1024",
        image_path_semantics="image_store_relative",
        max_pixels=1048576,
        visual_token_budget=1024,
        image_factor=28,
        image_root="public_data/coco/images/res-1024",
        splits=("train", "val"),
    )

    assert metadata.splits == ("train", "val")
    with pytest.raises(FrozenInstanceError):
        metadata.dataset = "lvis"


def test_rejects_escaped_image_path(tmp_path: Path) -> None:
    image_store = tmp_path / "store"
    image_store.mkdir()

    with pytest.raises(ValueError, match="outside image_root"):
        resolve_image_path("../raw/images/leak.jpg", image_root=image_store)


def test_rejects_bare_images_image_path(tmp_path: Path) -> None:
    image_store = tmp_path / "store"
    image_store.mkdir()

    with pytest.raises(ValueError, match="image_ref must start with images/"):
        resolve_image_path("images", image_root=image_store)


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


def test_load_view_metadata_rejects_absolute_image_store_with_absolute_anchor(
    tmp_path: Path,
) -> None:
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        """{
  "schema_version": 1,
  "kind": "annotation_view",
  "dataset": "coco",
  "view": "coco80/len-12000",
  "image_store": "/tmp/public_data/coco/images/res-1024",
  "path_anchor": "absolute",
  "image_path_semantics": "image_store_relative",
  "coordinate_space": "norm1000",
  "coordinate_storage": "integer",
  "coordinate_range": [0, 999],
  "coordinate_chart": "xyxy",
  "assistant_coordinate_rendering": "qwen_coord_tokens",
  "primary_jsonl": {"train": "train.jsonl", "val": "val.jsonl"}
}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="absolute image_store requires test/debug"):
        load_view_metadata(metadata_path)


def test_load_view_metadata_rejects_relative_image_store_without_repo_root_anchor(
    tmp_path: Path,
) -> None:
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        """{
  "schema_version": 1,
  "kind": "annotation_view",
  "dataset": "coco",
  "view": "coco80/len-12000",
  "image_store": "public_data/coco/images/res-1024",
  "path_anchor": "debug_absolute",
  "image_path_semantics": "image_store_relative",
  "coordinate_space": "norm1000",
  "coordinate_storage": "integer",
  "coordinate_range": [0, 999],
  "coordinate_chart": "xyxy",
  "assistant_coordinate_rendering": "qwen_coord_tokens",
  "primary_jsonl": {"train": "train.jsonl", "val": "val.jsonl"}
}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="relative image_store requires repo_root"):
        load_view_metadata(metadata_path)


def test_write_view_metadata_rejects_absolute_image_store_with_absolute_anchor(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="absolute image_store requires test/debug"):
        write_view_metadata(
            tmp_path / "metadata.json",
            {
                "schema_version": 1,
                "kind": "annotation_view",
                "dataset": "coco",
                "view": "coco80/len-12000",
                "image_store": "/tmp/public_data/coco/images/res-1024",
                "path_anchor": "absolute",
                "image_path_semantics": "image_store_relative",
                "coordinate_space": "norm1000",
                "coordinate_storage": "integer",
                "coordinate_range": [0, 999],
                "coordinate_chart": "xyxy",
                "assistant_coordinate_rendering": "qwen_coord_tokens",
                "primary_jsonl": {"train": "train.jsonl", "val": "val.jsonl"},
            },
        )


def test_write_view_metadata_rejects_relative_image_store_without_repo_root_anchor(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="relative image_store requires repo_root"):
        write_view_metadata(
            tmp_path / "metadata.json",
            {
                "schema_version": 1,
                "kind": "annotation_view",
                "dataset": "coco",
                "view": "coco80/len-12000",
                "image_store": "public_data/coco/images/res-1024",
                "path_anchor": "debug_absolute",
                "image_path_semantics": "image_store_relative",
                "coordinate_space": "norm1000",
                "coordinate_storage": "integer",
                "coordinate_range": [0, 999],
                "coordinate_chart": "xyxy",
                "assistant_coordinate_rendering": "qwen_coord_tokens",
                "primary_jsonl": {"train": "train.jsonl", "val": "val.jsonl"},
            },
        )
