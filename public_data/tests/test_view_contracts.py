from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from public_data.view_contracts import (
    ImageStoreMetadata,
    ViewMetadata,
    load_image_store_metadata,
    load_view_metadata,
    resolve_image_path,
    resolve_view_image_root,
    write_image_store_metadata,
    write_view_metadata,
)


def _valid_image_store_metadata(**overrides: object) -> dict[str, object]:
    metadata: dict[str, object] = {
        "schema_version": 1,
        "kind": "image_store",
        "dataset": "coco",
        "image_store": "res-1024",
        "image_path_semantics": "image_store_relative",
        "max_pixels": 1048576,
        "visual_token_budget": 1024,
        "image_factor": 32,
        "image_root": "public_data/coco/images/res-1024",
        "splits": ["train", "val"],
    }
    metadata.update(overrides)
    return metadata


def _valid_view_metadata(**overrides: object) -> dict[str, object]:
    metadata: dict[str, object] = {
        "schema_version": 1,
        "kind": "annotation_view",
        "dataset": "coco",
        "view": "coco80/full",
        "image_store": "public_data/coco/images/res-1024",
        "path_anchor": "repo_root",
        "image_path_semantics": "image_store_relative",
        "coordinate_space": "norm1000",
        "coordinate_storage": "integer",
        "coordinate_range": [0, 999],
        "coordinate_chart": "xyxy",
        "assistant_coordinate_rendering": "qwen_coord_tokens",
        "primary_jsonl": {"train": "train.jsonl", "val": "val.jsonl"},
        "summary": {"records": 2, "rendered_object_count": 3},
    }
    metadata.update(overrides)
    return metadata


def _valid_len_view_metadata(**overrides: object) -> dict[str, object]:
    metadata = _valid_view_metadata(
        view="coco80/len-12000",
        sample_policy={"type": "length_budget", "max_total_tokens": 12000},
        length_budget_scope={"rendered_families": ["objects"]},
        length_budget_template_id="compact-detection-v1",
        length_stats={
            "train": {
                "filename": "train.length_stats.json",
                "sha256": "a" * 64,
            },
            "val": {
                "filename": "val.length_stats.json",
                "sha256": "b" * 64,
            },
        },
    )
    metadata.update(overrides)
    return metadata


def _valid_proxy_view_metadata(**overrides: object) -> dict[str, object]:
    metadata = _valid_len_view_metadata(
        view="coco80-lvis-proxy/len-12000",
        annotation_policy="all_proxy",
        parent_view="coco80/len-12000",
        proxy_policy={
            "source_artifacts": [
                {
                    "kind": "lvis_proxy_jsonl",
                    "path": "public_data/coco/rescale_32_1024_bbox_max60_lvis_proxy/train.coord.jsonl",
                }
            ]
        },
        summary={
            "records": 2,
            "rendered_object_count": 3,
            "object_supervision_count": 3,
        },
    )
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
        length_budget_template_id="compact-detection-v1",
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


def test_rejects_repo_root_image_store_with_parent_component(tmp_path: Path) -> None:
    repo_root = tmp_path / "checkout"
    view_root = repo_root / "public_data" / "coco" / "views" / "coco80" / "len-12000"
    meta = ViewMetadata(
        **_valid_view_metadata(
            image_store="../outside/images",
            path_anchor="repo_root",
        )
    )

    with pytest.raises(ValueError, match=r"image_store must not contain \.\."):
        resolve_view_image_root(meta, view_root=view_root, repo_root=repo_root)


def test_write_and_load_view_metadata_round_trips_valid_metadata(tmp_path: Path) -> None:
    metadata_path = tmp_path / "nested" / "view" / "metadata.json"

    write_view_metadata(metadata_path, _valid_view_metadata())

    assert metadata_path.exists()
    loaded = load_view_metadata(metadata_path)
    assert loaded.dataset == "coco"
    assert loaded.view == "coco80/full"
    assert loaded.image_store == "public_data/coco/images/res-1024"
    assert loaded.path_anchor == "repo_root"
    assert loaded.coordinate_range == (0, 999)


def test_write_and_load_image_store_metadata_round_trips(tmp_path: Path) -> None:
    metadata_path = tmp_path / "images" / "res-1024" / "meta.json"

    write_image_store_metadata(metadata_path, _valid_image_store_metadata())

    loaded = load_image_store_metadata(metadata_path)
    assert loaded.kind == "image_store"
    assert loaded.image_store == "res-1024"
    assert loaded.splits == ("train", "val")


@pytest.mark.parametrize(
    "required_field",
    [
        "schema_version",
        "kind",
        "dataset",
        "image_store",
        "image_path_semantics",
        "max_pixels",
        "visual_token_budget",
        "image_factor",
        "image_root",
        "splits",
    ],
)
def test_load_image_store_metadata_rejects_missing_required_fields(
    tmp_path: Path,
    required_field: str,
) -> None:
    metadata_path = tmp_path / "meta.json"
    metadata = _valid_image_store_metadata()
    metadata.pop(required_field)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match=rf"image store metadata .*{required_field}"):
        load_image_store_metadata(metadata_path)


def test_load_image_store_metadata_rejects_wrong_splits_shape(
    tmp_path: Path,
) -> None:
    metadata_path = tmp_path / "meta.json"
    metadata = _valid_image_store_metadata(splits={"train": "train2017"})
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match=r"splits"):
        load_image_store_metadata(metadata_path)


@pytest.mark.parametrize("schema_version", ["1", True, 2])
def test_load_image_store_metadata_rejects_wrong_schema_version(
    tmp_path: Path,
    schema_version: object,
) -> None:
    metadata_path = tmp_path / "meta.json"
    metadata = _valid_image_store_metadata(schema_version=schema_version)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match=r"schema_version.*integer 1"):
        load_image_store_metadata(metadata_path)


def test_load_view_metadata_rejects_unknown_field(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.json"
    metadata = _valid_view_metadata(unexpected_contract="nope")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match=r"view metadata .*unexpected_contract"):
        load_view_metadata(metadata_path)


@pytest.mark.parametrize("schema_version", ["1", False, 2])
def test_load_view_metadata_rejects_wrong_schema_version(
    tmp_path: Path,
    schema_version: object,
) -> None:
    metadata_path = tmp_path / "metadata.json"
    metadata = _valid_view_metadata(schema_version=schema_version)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match=r"schema_version.*integer 1"):
        load_view_metadata(metadata_path)


@pytest.mark.parametrize("coordinate_chart", ["", "cxcywh"])
def test_load_view_metadata_rejects_unsupported_coordinate_chart(
    tmp_path: Path,
    coordinate_chart: object,
) -> None:
    metadata_path = tmp_path / "metadata.json"
    metadata = _valid_view_metadata(coordinate_chart=coordinate_chart)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match=r"coordinate_chart.*xyxy"):
        load_view_metadata(metadata_path)


@pytest.mark.parametrize("assistant_coordinate_rendering", ["", "plain_numbers"])
def test_load_view_metadata_rejects_unsupported_assistant_coordinate_rendering(
    tmp_path: Path,
    assistant_coordinate_rendering: object,
) -> None:
    metadata_path = tmp_path / "metadata.json"
    metadata = _valid_view_metadata(
        assistant_coordinate_rendering=assistant_coordinate_rendering
    )
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match=r"assistant_coordinate_rendering.*qwen"):
        load_view_metadata(metadata_path)


def test_load_view_metadata_rejects_bad_coordinate_range_shape(
    tmp_path: Path,
) -> None:
    metadata_path = tmp_path / "metadata.json"
    metadata = _valid_view_metadata(coordinate_range=[0, 500, 999])
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match=r"coordinate_range.*two integers"):
        load_view_metadata(metadata_path)


def test_base_annotation_view_requires_summary(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.json"
    metadata = _valid_view_metadata()
    metadata.pop("summary")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match=r"summary"):
        load_view_metadata(metadata_path)


def test_length_budget_view_requires_len_contract_fields(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.json"
    metadata = _valid_len_view_metadata(length_budget_template_id=None)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match=r"length_budget_template_id"):
        load_view_metadata(metadata_path)


def test_length_budget_view_validates_optional_length_stats(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.json"
    metadata = _valid_len_view_metadata(
        length_stats={"train": {"filename": "train.length_stats.json"}}
    )
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match=r"length_stats.train.sha256"):
        load_view_metadata(metadata_path)


def test_max_objects_view_accepts_legacy_cap_key(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.json"
    metadata = _valid_view_metadata(
        view="coco80/max-60",
        sample_policy={"type": "max_objects_legacy", "max_objects_per_image": 60},
    )
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    loaded = load_view_metadata(metadata_path)

    assert loaded.sample_policy == {
        "type": "max_objects_legacy",
        "max_objects_per_image": 60,
    }


def test_max_objects_view_requires_legacy_cap(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.json"
    metadata = _valid_view_metadata(
        view="coco80/max-60",
        sample_policy={"type": "max_objects_legacy"},
    )
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match=r"max_objects"):
        load_view_metadata(metadata_path)


def test_proxy_view_requires_source_artifacts(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.json"
    metadata = _valid_proxy_view_metadata(proxy_policy={"source_artifacts": []})
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match=r"proxy_policy.source_artifacts"):
        load_view_metadata(metadata_path)


def test_proxy_view_requires_all_proxy_annotation_policy(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.json"
    metadata = _valid_proxy_view_metadata(annotation_policy="mixed_proxy")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match=r"annotation_policy.*all_proxy"):
        load_view_metadata(metadata_path)


@pytest.mark.parametrize(
    ("source_artifact", "expected_error"),
    [
        (None, r"proxy_policy.source_artifacts\[0\] must be a mapping"),
        ({}, r"proxy_policy.source_artifacts\[0\].kind"),
        (
            {"kind": "lvis_proxy_jsonl"},
            r"proxy_policy.source_artifacts\[0\].path",
        ),
        (
            {
                "kind": "",
                "path": "public_data/coco/rescale_32_1024_bbox_max60_lvis_proxy/train.coord.jsonl",
            },
            r"proxy_policy.source_artifacts\[0\].kind",
        ),
        (
            {
                "kind": "lvis_proxy_jsonl",
                "path": "",
            },
            r"proxy_policy.source_artifacts\[0\].path",
        ),
        (
            {
                "kind": "lvis_proxy_jsonl",
                "path": "/tmp/proxy/train.coord.jsonl",
            },
            r"proxy_policy.source_artifacts\[0\].path.*relative",
        ),
        (
            {
                "kind": "lvis_proxy_jsonl",
                "path": "public_data/../proxy/train.coord.jsonl",
            },
            r"proxy_policy.source_artifacts\[0\].path.*\.\.",
        ),
        (
            {
                "kind": "lvis_proxy_jsonl",
                "path": r"public_data\proxy\train.coord.jsonl",
            },
            r"proxy_policy.source_artifacts\[0\].path.*POSIX",
        ),
    ],
)
def test_proxy_view_rejects_malformed_source_artifacts(
    tmp_path: Path,
    source_artifact: object,
    expected_error: str,
) -> None:
    metadata_path = tmp_path / "metadata.json"
    metadata = _valid_proxy_view_metadata(
        proxy_policy={"source_artifacts": [source_artifact]}
    )
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match=expected_error):
        load_view_metadata(metadata_path)


def test_proxy_view_requires_supervision_summary(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.json"
    metadata = _valid_proxy_view_metadata(
        summary={"records": 2, "rendered_object_count": 3}
    )
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match=r"summary.*object_supervision_count"):
        load_view_metadata(metadata_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("object_supervision_count", None),
        ("object_supervision_count", True),
        ("object_supervision_count", -1),
        ("rendered_proxy_candidate_count", None),
        ("rendered_proxy_candidate_count", False),
        ("rendered_proxy_candidate_count", -1),
        ("support_sidecar_count", None),
        ("support_sidecar_count", True),
        ("support_sidecar_count", -1),
    ],
)
def test_proxy_view_requires_non_negative_integer_supervision_summary_values(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    metadata_path = tmp_path / "metadata.json"
    metadata = _valid_proxy_view_metadata(
        summary={"records": 2, "rendered_object_count": 3, field: value}
    )
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match=rf"summary\.{field}"):
        load_view_metadata(metadata_path)


def test_image_store_metadata_is_frozen_public_contract() -> None:
    metadata = ImageStoreMetadata(
        schema_version=1,
        kind="image_store",
        dataset="coco",
        image_store="res-1024",
        image_path_semantics="image_store_relative",
        max_pixels=1048576,
        visual_token_budget=1024,
        image_factor=32,
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


def test_rejects_symlink_escape_image_path(tmp_path: Path) -> None:
    image_store = tmp_path / "store"
    image_store.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "leak.jpg").write_bytes(b"fake")
    images = image_store / "images"
    images.mkdir()
    (images / "outside").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="outside image_root"):
        resolve_image_path("images/outside/leak.jpg", image_root=image_store)


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
