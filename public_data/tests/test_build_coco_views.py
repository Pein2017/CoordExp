from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import pytest

import public_data.scripts.build_coco_views as build_coco_views
from public_data.scripts.build_coco_length_budget_artifacts import (
    TokenBudgetBreakdown,
    _model_facing_objects,
)
from public_data.scripts.build_coco_views import (
    AllProxyResearchViewBuilder,
    CocoViewFactoryConfig,
    ImageStoreAdopter,
    LegacyMaxObjectsViewBuilder,
    LengthBudgetViewBuilder,
    Norm1000ViewWriter,
    SourceComparisonWriter,
    ViewManifestPayloadBuilder,
    ViewStatsWriter,
    main,
    parse_args,
)
from public_data.view_contracts import load_image_store_metadata, load_view_metadata


class FakeEstimator:
    """Estimator returning deterministic budgets from the row metadata."""

    def measure(self, record: Mapping[str, Any]) -> TokenBudgetBreakdown:
        total_tokens = int(record.get("metadata", {}).get("fake_total_tokens", 0))
        return TokenBudgetBreakdown(
            total_tokens=total_tokens,
            text_tokens_without_image_placeholders=total_tokens,
            image_patch_tokens=0,
            image_placeholders=1,
            assistant_tokens=len(record.get("objects") or []),
            object_count=len(record.get("objects") or []),
        )


def test_coco80_full_writes_norm1000_integer_boxes_and_image_store_refs(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "public_data" / "coco" / "rescale_32_1024_bbox"
    _write_image(source_root / "images" / "train2017" / "000000000001.jpg")
    _write_jsonl(
        source_root / "train.jsonl",
        [
            {
                "image_id": 1,
                "file_name": "train2017/000000000001.jpg",
                "width": 100,
                "height": 50,
                "images": ["images/train2017/000000000001.jpg"],
                "metadata": {"source": "coco", "split": "train"},
                "objects": [
                    {
                        "desc": "cat",
                        "bbox_2d": [10, 5, 20, 25],
                    }
                ],
            }
        ],
    )
    config = _config(tmp_path, source_preset=source_root, splits=("train",))
    ImageStoreAdopter(config).prepare()

    writer = Norm1000ViewWriter(config=config)
    summary = writer.write_view(
        source_root=source_root,
        view_name="coco80/full",
        view_root=config.view_root("coco80/full"),
        sample_policy=None,
    )

    row = _read_jsonl(config.view_root("coco80/full") / "train.jsonl")[0]
    meta = load_view_metadata(config.view_root("coco80/full") / "meta.json")
    image_meta = load_image_store_metadata(config.image_store_root / "meta.json")
    assert summary["records"] == 1
    assert row["images"] == ["images/train2017/000000000001.jpg"]
    assert row["objects"][0]["bbox_2d"] == [100, 101, 202, 510]
    assert all(isinstance(coord, int) for coord in row["objects"][0]["bbox_2d"])
    assert row["objects"][0]["object_id"] == "1:0"
    assert meta.view == "coco80/full"
    assert meta.primary_jsonl == {"train": "train.jsonl"}
    assert image_meta.image_root == "public_data/coco/images/res-1024"
    assert image_meta.image_factor == 32


def test_coco80_len12000_drops_over_budget_row_with_fake_estimator(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, splits=("train",))
    full_root = config.view_root("coco80/full")
    _write_jsonl(
        full_root / "train.jsonl",
        [
            _norm_row(image_id=1, fake_total_tokens=11999),
            _norm_row(image_id=2, fake_total_tokens=12001),
        ],
    )

    builder = LengthBudgetViewBuilder(
        config=config,
        estimator=FakeEstimator(),
        stats_writer=ViewStatsWriter(),
        manifest_builder=ViewManifestPayloadBuilder(config),
    )
    summary = builder.build(
        source_view_root=full_root,
        view_name="coco80/len-12000",
        max_total_tokens=12000,
    )

    rows = _read_jsonl(config.view_root("coco80/len-12000") / "train.jsonl")
    stats = json.loads(
        (config.view_root("coco80/len-12000") / "train.length_stats.json").read_text()
    )
    meta = load_view_metadata(config.view_root("coco80/len-12000") / "meta.json")
    assert [row["image_id"] for row in rows] == [1]
    assert summary["records"] == 1
    assert str(tmp_path) not in json.dumps(stats)
    assert stats["source_jsonl"] == (
        "public_data/coco/views/coco80/full/train.jsonl"
    )
    assert stats["output_jsonl"] == (
        "public_data/coco/views/coco80/len-12000/train.jsonl"
    )
    assert stats["records_seen"] == 2
    assert stats["records_dropped"] == 1
    assert stats["length_over_budget_examples"][0]["image_id"] == 2
    assert meta.sample_policy == {
        "type": "length_budget",
        "max_total_tokens": 12000,
        "budget_includes": [
            "image_patch_tokens",
            "system_prompt_tokens",
            "user_prompt_tokens",
            "assistant_response_tokens",
        ],
    }


def test_coco80_max60_preserves_historical_membership_from_norm1000_source(
    tmp_path: Path,
) -> None:
    legacy_root = tmp_path / "public_data" / "coco" / "rescale_32_1024_bbox_max60"
    _write_jsonl(
        legacy_root / "train.norm.jsonl",
        [
            _norm_row(image_id=10, object_count=60),
            _norm_row(image_id=11, object_count=1),
        ],
    )
    config = _config(tmp_path, legacy_max_objects_source=legacy_root, splits=("train",))

    builder = LegacyMaxObjectsViewBuilder(
        config=config,
        writer=Norm1000ViewWriter(config=config),
    )
    summary = builder.build(view_name="coco80/max-60", max_objects=60)

    rows = _read_jsonl(config.view_root("coco80/max-60") / "train.jsonl")
    meta = load_view_metadata(config.view_root("coco80/max-60") / "meta.json")
    assert [row["image_id"] for row in rows] == [10, 11]
    assert [len(row["objects"]) for row in rows] == [60, 1]
    assert summary["records"] == 2
    assert meta.sample_policy == {
        "type": "max_objects_legacy",
        "max_objects": 60,
        "membership_source": "public_data/coco/rescale_32_1024_bbox_max60",
    }


def test_coco80_lvis_proxy_len12000_records_object_supervision_by_object_id(
    tmp_path: Path,
) -> None:
    proxy_root = tmp_path / "public_data" / "coco" / "rescale_32_1024_bbox_lvis_proxy"
    _write_jsonl(
        proxy_root / "train.norm.jsonl",
        [
            _norm_row(
                image_id=20,
                fake_total_tokens=12000,
                objects=[
                    {
                        "object_id": "coco:20:0",
                        "desc": "dog",
                        "bbox_2d": [10, 10, 100, 100],
                        "source_role": "coco_ground_truth",
                    },
                    {
                        "object_id": "lvis:20:1",
                        "desc": "leash",
                        "bbox_2d": [20, 20, 110, 110],
                        "source_role": "lvis_proxy_candidate",
                        "coordinate_weight": 0.0,
                        "regression_weight": 0.0,
                    },
                ],
            )
        ],
    )
    config = _config(tmp_path, proxy_source=proxy_root, splits=("train",))

    builder = AllProxyResearchViewBuilder(
        config=config,
        estimator=FakeEstimator(),
        stats_writer=ViewStatsWriter(),
        manifest_builder=ViewManifestPayloadBuilder(config),
    )
    summary = builder.build(view_name="coco80-lvis-proxy/len-12000", max_total_tokens=12000)

    row = _read_jsonl(config.view_root("coco80-lvis-proxy/len-12000") / "train.jsonl")[0]
    stats = json.loads(
        (
            config.view_root("coco80-lvis-proxy/len-12000")
            / "train.length_stats.json"
        ).read_text()
    )
    object_supervision = row["metadata"]["supervision"]["object_supervision"]
    meta = load_view_metadata(config.view_root("coco80-lvis-proxy/len-12000") / "meta.json")
    assert str(tmp_path) not in json.dumps(stats)
    assert stats["source_jsonl"] == (
        "public_data/coco/rescale_32_1024_bbox_lvis_proxy/train.norm.jsonl"
    )
    assert stats["output_jsonl"] == (
        "public_data/coco/views/coco80-lvis-proxy/len-12000/train.jsonl"
    )
    assert set(object_supervision) == {"coco:20:0", "lvis:20:1"}
    assert object_supervision["lvis:20:1"]["source_role"] == "lvis_proxy_candidate"
    assert object_supervision["lvis:20:1"]["coordinate_weight"] == 0.0
    assert summary["object_supervision_count"] == 2
    assert meta.annotation_policy == "all_proxy"
    assert meta.parent_view == "coco80/full"


def test_coco80_lvis_proxy_len12000_infers_proxy_supervision_from_lvis_evidence(
    tmp_path: Path,
) -> None:
    proxy_root = tmp_path / "public_data" / "coco" / "rescale_32_1024_bbox_lvis_proxy"
    _write_jsonl(
        proxy_root / "train.norm.jsonl",
        [
            _norm_row(
                image_id=21,
                fake_total_tokens=12000,
                objects=[
                    {
                        "object_id": "coco:21:0",
                        "desc": "dog",
                        "bbox_2d": [10, 10, 100, 100],
                    },
                    {
                        "object_id": "lvis:21:1",
                        "desc": "muzzle",
                        "bbox_2d": [20, 20, 110, 110],
                        "source": "lvis",
                        "proxy_source": "lvis",
                        "lvis_ann_id": 987,
                        "lvis_category_id": 4321,
                        "lvis_category_name": "muzzle",
                        "coco_category_id": 18,
                    },
                ],
            )
        ],
    )
    config = _config(tmp_path, proxy_source=proxy_root, splits=("train",))

    builder = AllProxyResearchViewBuilder(
        config=config,
        estimator=FakeEstimator(),
        stats_writer=ViewStatsWriter(),
        manifest_builder=ViewManifestPayloadBuilder(config),
    )
    summary = builder.build(view_name="coco80-lvis-proxy/len-12000", max_total_tokens=12000)

    row = _read_jsonl(config.view_root("coco80-lvis-proxy/len-12000") / "train.jsonl")[0]
    lvis_object = row["objects"][1]
    object_supervision = row["metadata"]["supervision"]["object_supervision"]
    lvis_supervision = object_supervision["lvis:21:1"]
    assert lvis_supervision["source_role"] == "lvis_proxy_candidate"
    assert lvis_supervision["coordinate_weight"] == 0.0
    assert lvis_supervision["regression_weight"] == 0.0
    assert lvis_supervision["hard_bbox_supervision"] is False
    assert lvis_supervision["source"] == "lvis"
    assert lvis_supervision["proxy_source"] == "lvis"
    assert lvis_supervision["lvis_ann_id"] == 987
    assert lvis_supervision["lvis_category_id"] == 4321
    assert lvis_supervision["lvis_category_name"] == "muzzle"
    assert lvis_supervision["coco_category_id"] == 18
    assert "coordinate_weight" not in lvis_object
    assert "regression_weight" not in lvis_object
    assert "hard_bbox_supervision" not in lvis_object
    assert summary["object_supervision_count"] == 2
    assert summary["rendered_proxy_candidate_count"] == 1


def test_source_comparison_records_expected_length_budget_subset(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "public_data" / "coco" / "rescale_32_1024_bbox"
    config = _config(
        tmp_path,
        source_preset=source_root,
        legacy_length_budget_source=tmp_path / "missing_len12000",
        splits=("train",),
    )
    view_root = config.view_root("coco80/len-12000")
    _write_jsonl(
        source_root / "train.jsonl",
        [_norm_row(image_id=1), _norm_row(image_id=2)],
    )
    _write_jsonl(view_root / "train.jsonl", [_norm_row(image_id=1)])
    _write_json(
        view_root / "train.length_stats.json",
        {
            "records_seen": 2,
            "records_written": 1,
            "records_dropped": 1,
            "kept_image_ids": [1],
            "dropped_image_ids": [2],
        },
    )
    _write_json(
        view_root / "meta.json",
        ViewManifestPayloadBuilder(config).build_view_metadata(
            view_name="coco80/len-12000",
            primary_jsonl={"train": "train.jsonl"},
            summary={"records": 1},
            sample_policy={"type": "length_budget", "max_total_tokens": 12000},
            length_budget_scope={"rendered_families": ["objects"]},
            length_budget_template_id="compact-detection-v1",
            length_stats={
                "train": {
                    "filename": "train.length_stats.json",
                    "sha256": "0" * 64,
                }
            },
        ),
    )

    ref = SourceComparisonWriter(config).write_for_view("coco80/len-12000")

    comparison = json.loads((view_root / "source_comparison.json").read_text())
    meta = load_view_metadata(view_root / "meta.json")
    assert ref["filename"] == "source_comparison.json"
    assert comparison["source_mode"] == "length_budget_subset"
    assert comparison["unexpected_deltas"] == []
    assert comparison["splits"]["train"]["missing_from_generated_count"] == 1
    assert meta.summary["source_comparison"] == ref


def test_source_comparison_fails_on_unverifiable_length_budget_subset(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "public_data" / "coco" / "rescale_32_1024_bbox"
    config = _config(
        tmp_path,
        source_preset=source_root,
        legacy_length_budget_source=tmp_path / "missing_len12000",
        splits=("train",),
    )
    view_root = config.view_root("coco80/len-12000")
    _write_jsonl(
        source_root / "train.jsonl",
        [_norm_row(image_id=1), _norm_row(image_id=2), _norm_row(image_id=3)],
    )
    _write_jsonl(
        view_root / "train.jsonl",
        [_norm_row(image_id=1), _norm_row(image_id=2)],
    )
    _write_json(
        view_root / "train.length_stats.json",
        {
            "records_seen": 3,
            "records_written": 2,
            "records_dropped": 1,
            "kept_image_ids": [1, 3],
            "dropped_image_ids": [2],
        },
    )
    _write_json(
        view_root / "meta.json",
        ViewManifestPayloadBuilder(config).build_view_metadata(
            view_name="coco80/len-12000",
            primary_jsonl={"train": "train.jsonl"},
            summary={"records": 2},
            sample_policy={"type": "length_budget", "max_total_tokens": 12000},
            length_budget_scope={"rendered_families": ["objects"]},
            length_budget_template_id="compact-detection-v1",
            length_stats={
                "train": {
                    "filename": "train.length_stats.json",
                    "sha256": "0" * 64,
                }
            },
        ),
    )

    with pytest.raises(ValueError, match="length_budget_kept_image_ids_mismatch"):
        SourceComparisonWriter(config).write_for_view("coco80/len-12000")

    comparison = json.loads((view_root / "source_comparison.json").read_text())
    assert comparison["unexpected_deltas"][0]["type"] == (
        "length_budget_kept_image_ids_mismatch"
    )


def test_length_budget_build_writes_membership_ids_for_fallback_comparison(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        legacy_length_budget_source=tmp_path / "missing_len12000",
        splits=("train",),
    )
    full_root = config.view_root("coco80/full")
    source_rows = [
        _norm_row(image_id=1, fake_total_tokens=11999),
        _norm_row(image_id=2, fake_total_tokens=12001),
    ]
    _write_jsonl(config.source_preset / "train.jsonl", source_rows)
    _write_jsonl(
        full_root / "train.jsonl",
        source_rows,
    )

    builder = LengthBudgetViewBuilder(
        config=config,
        estimator=FakeEstimator(),
        stats_writer=ViewStatsWriter(),
        manifest_builder=ViewManifestPayloadBuilder(config),
    )
    builder.build(
        source_view_root=full_root,
        view_name="coco80/len-12000",
        max_total_tokens=12000,
    )

    view_root = config.view_root("coco80/len-12000")
    stats = json.loads((view_root / "train.length_stats.json").read_text())
    ref = SourceComparisonWriter(config).write_for_view("coco80/len-12000")

    comparison = json.loads((view_root / "source_comparison.json").read_text())
    assert stats["kept_image_ids"] == [1]
    assert stats["dropped_image_ids"] == [2]
    assert ref["filename"] == "source_comparison.json"
    assert comparison["unexpected_deltas"] == []


def test_source_comparison_fails_on_object_count_drift(tmp_path: Path) -> None:
    source_root = tmp_path / "public_data" / "coco" / "rescale_32_1024_bbox"
    config = _config(tmp_path, source_preset=source_root, splits=("train",))
    view_root = config.view_root("coco80/full")
    _write_jsonl(source_root / "train.jsonl", [_norm_row(image_id=1, object_count=2)])
    _write_jsonl(view_root / "train.jsonl", [_norm_row(image_id=1, object_count=1)])
    _write_json(
        view_root / "meta.json",
        ViewManifestPayloadBuilder(config).build_view_metadata(
            view_name="coco80/full",
            primary_jsonl={"train": "train.jsonl"},
            summary={"records": 1},
        ),
    )

    with pytest.raises(ValueError, match="object_count_mismatch"):
        SourceComparisonWriter(config).write_for_view("coco80/full")

    comparison = json.loads((view_root / "source_comparison.json").read_text())
    meta = load_view_metadata(view_root / "meta.json")
    assert comparison["unexpected_deltas"][0]["type"] == "object_count_mismatch"
    assert "source_comparison" not in meta.summary


def test_image_store_rejects_non_empty_target_without_reuse(tmp_path: Path) -> None:
    source_root = tmp_path / "public_data" / "coco" / "rescale_32_1024_bbox"
    _write_image(source_root / "images" / "train2017" / "000000000001.jpg")
    config = _config(tmp_path, source_preset=source_root, image_store_mode="copy")
    _write_image(config.image_store_root / "images" / "train2017" / "existing.jpg")

    with pytest.raises(FileExistsError, match="reuse-existing"):
        ImageStoreAdopter(config).prepare()


def test_hardlink_image_store_rejects_non_empty_target_without_reuse(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "public_data" / "coco" / "rescale_32_1024_bbox"
    _write_image(source_root / "images" / "train2017" / "000000000001.jpg")
    config = _config(tmp_path, source_preset=source_root, image_store_mode="hardlink")
    _write_image(config.image_store_root / "images" / "train2017" / "existing.jpg")

    with pytest.raises(FileExistsError, match="reuse-existing"):
        ImageStoreAdopter(config).prepare()


def test_dry_run_report_records_counts_and_does_not_write_artifacts(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "public_data" / "coco" / "rescale_32_1024_bbox"
    _write_image(source_root / "images" / "train2017" / "000000000001.jpg")
    report_path = tmp_path / "report.json"
    config = _config(
        tmp_path,
        source_preset=source_root,
        image_store_mode="copy",
        dry_run=True,
        dry_run_report=report_path,
    )

    ImageStoreAdopter(config).prepare()

    report = json.loads(report_path.read_text())
    assert not config.image_store_root.exists()
    assert report["source_image_count"] == 1
    assert report["target_image_count"] == 0
    assert report["sample_resolution_checks"][0]["new_ref"].startswith("images/")
    assert "No files were copied" in report["rollback_notes"]


def test_cli_dry_run_default_views_plans_without_materialized_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_root = tmp_path / "public_data" / "coco" / "rescale_32_1024_bbox"
    legacy_root = tmp_path / "public_data" / "coco" / "rescale_32_1024_bbox_max60"
    proxy_root = (
        tmp_path / "public_data" / "coco" / "rescale_32_1024_bbox_lvis_proxy_len12000"
    )
    image_store_root = tmp_path / "public_data" / "coco" / "images" / "res-1024"
    views_root = tmp_path / "public_data" / "coco" / "views"
    report_path = tmp_path / "dry-run-report.json"
    source_image = source_root / "images" / "train2017" / "000000000001.jpg"
    _write_image(source_image)
    _write_jsonl(source_root / "train.jsonl", [_raw_row(image_id=1)])
    _write_jsonl(legacy_root / "train.norm.jsonl", [_norm_row(image_id=1)])
    _write_jsonl(proxy_root / "train.norm.jsonl", [_norm_row(image_id=1)])

    def fail_if_estimator_loads(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("dry-run should not load the real length estimator")

    monkeypatch.setattr(
        build_coco_views.CompactFullTokenBudgetEstimator,
        "from_config",
        fail_if_estimator_loads,
    )

    main(
        [
            "--source-preset",
            str(source_root),
            "--image-store-root",
            str(image_store_root),
            "--views-root",
            str(views_root),
            "--legacy-max-objects-source",
            str(legacy_root),
            "--proxy-source",
            str(proxy_root),
            "--splits",
            "train",
            "--image-store-mode",
            "copy",
            "--dry-run",
            "--dry-run-report",
            str(report_path),
        ]
    )

    report = json.loads(report_path.read_text())
    summary = json.loads(capsys.readouterr().out)
    assert not image_store_root.exists()
    assert not views_root.exists()
    assert source_image.is_file()
    assert set(summary) == {
        "image_store",
        "coco80/full",
        "coco80/len-12000",
        "coco80/max-60",
        "coco80-lvis-proxy/len-12000",
    }
    assert summary["coco80/len-12000"]["dry_run"] is True
    assert report["source_image_count"] == 1
    assert report["target_image_count"] == 0
    assert report["estimated_copy_bytes"] == 4
    assert report["sample_resolution_checks"][0]["old_exists"] is True
    assert report["sample_resolution_checks"][0]["new_ref"] == (
        "images/train2017/000000000001.jpg"
    )
    assert str(views_root / "coco80" / "full") in report["planned_outputs"]["views"]
    assert "No files were copied" in report["rollback_notes"]


def test_parse_args_requires_image_store_mode() -> None:
    with pytest.raises(SystemExit):
        parse_args([])


def test_comparison_only_does_not_require_image_store_mode(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_root = tmp_path / "public_data" / "coco" / "rescale_32_1024_bbox"
    views_root = tmp_path / "public_data" / "coco" / "views"
    image_store_root = tmp_path / "public_data" / "coco" / "images" / "res-1024"
    config = _config(
        tmp_path,
        source_preset=source_root,
        splits=("train",),
        image_store_mode="reuse-existing",
    )
    view_root = views_root / "coco80" / "full"
    _write_jsonl(source_root / "train.jsonl", [_norm_row(image_id=1)])
    _write_jsonl(view_root / "train.jsonl", [_norm_row(image_id=1)])
    _write_json(
        view_root / "meta.json",
        ViewManifestPayloadBuilder(config).build_view_metadata(
            view_name="coco80/full",
            primary_jsonl={"train": "train.jsonl"},
            summary={"records": 1},
        ),
    )

    main(
        [
            "--comparison-only",
            "--source-preset",
            str(source_root),
            "--views-root",
            str(views_root),
            "--image-store-root",
            str(image_store_root),
            "--splits",
            "train",
            "--views",
            "coco80/full",
        ]
    )

    summary = json.loads(capsys.readouterr().out)
    meta = load_view_metadata(view_root / "meta.json")
    assert summary["coco80/full"]["source_comparison"]["filename"] == (
        "source_comparison.json"
    )
    assert meta.summary["source_comparison"]["filename"] == "source_comparison.json"


def test_reuse_existing_accepts_non_empty_canonical_store_without_copying(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "public_data" / "coco" / "rescale_32_1024_bbox"
    source_image = source_root / "images" / "train2017" / "source-only.jpg"
    _write_image(source_image)
    config = _config(
        tmp_path,
        source_preset=source_root,
        image_store_mode="reuse-existing",
    )
    existing_target = config.target_image_dir / "train2017" / "existing.jpg"
    _write_image(existing_target)

    summary = ImageStoreAdopter(config).prepare()

    assert summary["target_image_count"] == 1
    assert existing_target.is_file()
    assert not (config.target_image_dir / "train2017" / "source-only.jpg").exists()
    assert source_image.is_file()


def test_copy_adoption_leaves_legacy_source_image_tree_intact(tmp_path: Path) -> None:
    source_root = tmp_path / "public_data" / "coco" / "rescale_32_1024_bbox"
    source_image = source_root / "images" / "train2017" / "000000000001.jpg"
    _write_image(source_image)
    config = _config(tmp_path, source_preset=source_root, image_store_mode="copy")

    ImageStoreAdopter(config).prepare()

    assert source_image.is_file()
    assert (config.target_image_dir / "train2017" / "000000000001.jpg").is_file()


def test_hardlink_adoption_links_target_and_leaves_source_intact(tmp_path: Path) -> None:
    source_root = tmp_path / "public_data" / "coco" / "rescale_32_1024_bbox"
    source_image = source_root / "images" / "train2017" / "000000000001.jpg"
    _write_image(source_image)
    _skip_if_hardlinks_are_unsupported(tmp_path)
    config = _config(tmp_path, source_preset=source_root, image_store_mode="hardlink")

    ImageStoreAdopter(config).prepare()

    target_image = config.target_image_dir / "train2017" / "000000000001.jpg"
    source_stat = source_image.stat()
    target_stat = target_image.stat()
    assert source_image.is_file()
    assert target_image.is_file()
    assert (source_stat.st_dev, source_stat.st_ino) == (
        target_stat.st_dev,
        target_stat.st_ino,
    )
    assert source_stat.st_nlink >= 2


def test_reflink_adoption_uses_reflink_and_leaves_source_intact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "public_data" / "coco" / "rescale_32_1024_bbox"
    source_image = source_root / "images" / "train2017" / "000000000001.jpg"
    _write_image(source_image)
    config = _config(tmp_path, source_preset=source_root, image_store_mode="reflink")
    reflink_calls: list[list[str]] = []

    def fake_run(command: list[str], check: bool) -> None:
        reflink_calls.append(command)
        assert check is True
        Path(command[3]).write_bytes(Path(command[2]).read_bytes())

    monkeypatch.setattr(build_coco_views.subprocess, "run", fake_run)

    ImageStoreAdopter(config).prepare()

    assert reflink_calls == [
        [
            "cp",
            "--reflink=always",
            str(source_image),
            str(config.target_image_dir / "train2017" / "000000000001.jpg"),
        ]
    ]
    assert source_image.is_file()
    assert (config.target_image_dir / "train2017" / "000000000001.jpg").is_file()


def test_cli_rejects_phase1_move_image_store_mode() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--image-store-mode", "move"])


def test_cli_accepts_phase1_hardlink_image_store_mode() -> None:
    args = parse_args(["--image-store-mode", "hardlink"])

    assert args.image_store_mode == "hardlink"


def test_compact_full_estimator_model_view_renders_norm1000_ints_as_coord_tokens() -> None:
    model_objects = _model_facing_objects(
        {
            "objects": [
                {
                    "object_id": "1:0",
                    "desc": "cat",
                    "bbox_2d": [1, 2, 3, 4],
                }
            ]
        }
    )

    assert model_objects == [
        {
            "desc": "cat",
            "bbox_2d": [
                "<|coord_1|>",
                "<|coord_2|>",
                "<|coord_3|>",
                "<|coord_4|>",
            ],
        }
    ]


def _config(
    tmp_path: Path,
    *,
    source_preset: Path | None = None,
    legacy_length_budget_source: Path | None = None,
    legacy_max_objects_source: Path | None = None,
    proxy_source: Path | None = None,
    splits: tuple[str, ...] = ("train",),
    image_store_mode: str = "copy",
    dry_run: bool = False,
    dry_run_report: Path | None = None,
) -> CocoViewFactoryConfig:
    repo_root = tmp_path
    return CocoViewFactoryConfig(
        repo_root=repo_root,
        source_preset=source_preset
        or repo_root / "public_data" / "coco" / "rescale_32_1024_bbox",
        image_store_root=repo_root / "public_data" / "coco" / "images" / "res-1024",
        views_root=repo_root / "public_data" / "coco" / "views",
        legacy_length_budget_source=legacy_length_budget_source
        or repo_root / "public_data" / "coco" / "rescale_32_1024_bbox_len12000",
        legacy_max_objects_source=legacy_max_objects_source,
        proxy_source=proxy_source,
        splits=splits,
        views=("coco80/full",),
        max_total_tokens=12000,
        image_store_mode=image_store_mode,
        reuse_existing_image_store=image_store_mode == "reuse-existing",
        dry_run=dry_run,
        dry_run_report=dry_run_report,
    )


def _norm_row(
    *,
    image_id: int,
    fake_total_tokens: int = 0,
    object_count: int = 1,
    objects: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "image_id": image_id,
        "file_name": f"train2017/{image_id:012d}.jpg",
        "width": 1024,
        "height": 1024,
        "images": [f"images/train2017/{image_id:012d}.jpg"],
        "metadata": {
            "source": "coco",
            "split": "train",
            "fake_total_tokens": fake_total_tokens,
        },
        "objects": objects
        if objects is not None
        else [
            {
                "object_id": f"{image_id}:{index}",
                "desc": f"object {index}",
                "bbox_2d": [index, index + 1, index + 2, index + 3],
            }
            for index in range(object_count)
        ],
    }


def _raw_row(*, image_id: int, object_count: int = 1) -> dict[str, Any]:
    row = _norm_row(image_id=image_id, object_count=object_count)
    row["width"] = 100
    row["height"] = 50
    row["objects"] = [
        {
            **obj,
            "bbox_2d": [10, 5, 20, 25],
        }
        for obj in row["objects"]
    ]
    return row


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake")


def _skip_if_hardlinks_are_unsupported(tmp_path: Path) -> None:
    source = tmp_path / "hardlink-probe-source"
    target = tmp_path / "hardlink-probe-target"
    source.write_bytes(b"probe")
    try:
        os.link(source, target)
    except OSError as exc:
        pytest.skip(f"hardlinks are unsupported in tmp_path: {exc}")
