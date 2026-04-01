from __future__ import annotations

import csv
import json
from pathlib import Path

from src.analysis.coco_lvis_missing_objects import (
    ProxyAugmentConfig,
    export_augmented_coco_with_lvis_proxies,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_export_augmented_coco_with_lvis_proxies_merges_and_aligns_metadata(
    tmp_path: Path,
) -> None:
    base_jsonl = tmp_path / "base.coord.jsonl"
    projection_dir = tmp_path / "projection"
    mapping_csv = tmp_path / "determined_proxy_mappings.csv"
    raw_ann = tmp_path / "instances_val2017.json"
    output_jsonl = tmp_path / "augmented.coord.jsonl"

    _write_jsonl(
        base_jsonl,
        [
            {
                "images": ["images/val2017/000000000001.jpg"],
                "objects": [
                    {
                        "bbox_2d": [
                            "<|coord_500|>",
                            "<|coord_500|>",
                            "<|coord_700|>",
                            "<|coord_700|>",
                        ],
                        "desc": "clock",
                        "category_id": 85,
                        "category_name": "clock",
                        "coco_ann_id": 1,
                    }
                ],
                "width": 640,
                "height": 480,
                "image_id": 1,
                "file_name": "images/val2017/000000000001.jpg",
                "metadata": {"source": "coco2017", "split": "val"},
            }
        ],
    )

    _write_json(
        raw_ann,
        {
            "images": [
                {
                    "id": 1,
                    "width": 100,
                    "height": 100,
                    "file_name": "000000000001.jpg",
                }
            ],
            "annotations": [],
            "categories": [],
        },
    )

    _write_jsonl(
        projection_dir / "recovered_coco80_instances.jsonl",
        [
            {
                "image_id": 1,
                "lvis_ann_id": 101,
                "lvis_category_id": 201,
                "lvis_category_name": "mug",
                "mapped_coco_category_id": 47,
                "mapped_coco_category_name": "cup",
                "mapping_kind": "semantic_evidence",
                "mapping_tier": "strict",
                "bbox_xyxy": [10.0, 20.0, 30.0, 40.0],
                "recovery_view": "strict_plus_usable",
                "why_recovered": "semantic strict",
            },
            {
                "image_id": 1,
                "lvis_ann_id": 102,
                "lvis_category_id": 202,
                "lvis_category_name": "tablecloth",
                "mapped_coco_category_id": 67,
                "mapped_coco_category_name": "dining table",
                "mapping_kind": "semantic_evidence",
                "mapping_tier": "usable",
                "bbox_xyxy": [5.0, 10.0, 15.0, 20.0],
                "recovery_view": "strict_plus_usable",
                "why_recovered": "semantic plausible",
            },
            {
                "image_id": 1,
                "lvis_ann_id": 103,
                "lvis_category_id": 301,
                "lvis_category_name": "water_bottle",
                "mapped_coco_category_id": 44,
                "mapped_coco_category_name": "bottle",
                "mapping_kind": "exact_canonical",
                "mapping_tier": "strict",
                "bbox_xyxy": [40.0, 10.0, 50.0, 20.0],
                "recovery_view": "strict_only",
                "why_recovered": "exact strict duplicate older view",
            },
            {
                "image_id": 1,
                "lvis_ann_id": 103,
                "lvis_category_id": 301,
                "lvis_category_name": "water_bottle",
                "mapped_coco_category_id": 44,
                "mapped_coco_category_name": "bottle",
                "mapping_kind": "exact_canonical",
                "mapping_tier": "strict",
                "bbox_xyxy": [40.0, 10.0, 50.0, 20.0],
                "recovery_view": "strict_plus_usable",
                "why_recovered": "exact strict preferred view",
            },
            {
                "image_id": 1,
                "lvis_ann_id": 104,
                "lvis_category_id": 203,
                "lvis_category_name": "car_(automobile)",
                "mapped_coco_category_id": 3,
                "mapped_coco_category_name": "car",
                "mapping_kind": "semantic_evidence",
                "mapping_tier": "usable",
                "bbox_xyxy": [60.0, 10.0, 80.0, 30.0],
                "recovery_view": "strict_plus_usable",
                "why_recovered": "semantic reject by mapping csv",
            },
        ],
    )

    _write_csv(
        mapping_csv,
        [
            {
                "lvis_category_id": 201,
                "lvis_category_name": "mug",
                "mapped_coco_category_id": 47,
                "mapped_coco_category_name": "cup",
                "mapping_kind": "semantic_evidence",
                "prior_kind": "",
                "confidence_tier": "strict",
                "determination_tier": "strict",
                "mapping_class": "same_extent_proxy",
                "candidate_source": "empirical_match",
                "candidate_kind": "semantic_evidence",
                "decision_rule_version": "unit_test",
                "decision_rule": "strict_cut",
                "n_match": 25,
                "n_images": 10,
                "precision_like": 0.97,
                "coverage_like": 0.8,
                "mean_iou": 0.89,
                "median_iou": 0.91,
                "iou_ge_05_rate": 0.99,
                "iou_ge_075_rate": 0.92,
                "support_score": 0.8,
                "geometry_score": 0.9,
                "proxy_score": 0.91,
                "determination_reason": "strict",
                "tier_rank": 1,
            },
            {
                "lvis_category_id": 202,
                "lvis_category_name": "tablecloth",
                "mapped_coco_category_id": 67,
                "mapped_coco_category_name": "dining table",
                "mapping_kind": "semantic_evidence",
                "prior_kind": "",
                "confidence_tier": "usable",
                "determination_tier": "plausible",
                "mapping_class": "cue_only_proxy",
                "candidate_source": "empirical_match",
                "candidate_kind": "semantic_evidence",
                "decision_rule_version": "unit_test",
                "decision_rule": "plausible_cut",
                "n_match": 30,
                "n_images": 12,
                "precision_like": 0.93,
                "coverage_like": 0.58,
                "mean_iou": 0.81,
                "median_iou": 0.88,
                "iou_ge_05_rate": 0.89,
                "iou_ge_075_rate": 0.67,
                "support_score": 0.7,
                "geometry_score": 0.79,
                "proxy_score": 0.81,
                "determination_reason": "plausible",
                "tier_rank": 1,
            },
            {
                "lvis_category_id": 203,
                "lvis_category_name": "car_(automobile)",
                "mapped_coco_category_id": 3,
                "mapped_coco_category_name": "car",
                "mapping_kind": "semantic_evidence",
                "prior_kind": "",
                "confidence_tier": "usable",
                "determination_tier": "reject",
                "mapping_class": "reject",
                "candidate_source": "empirical_match",
                "candidate_kind": "semantic_evidence",
                "decision_rule_version": "unit_test",
                "decision_rule": "reject_cut",
                "n_match": 200,
                "n_images": 70,
                "precision_like": 0.89,
                "coverage_like": 0.73,
                "mean_iou": 0.86,
                "median_iou": 0.90,
                "iou_ge_05_rate": 0.99,
                "iou_ge_075_rate": 0.85,
                "support_score": 0.99,
                "geometry_score": 0.87,
                "proxy_score": 0.87,
                "determination_reason": "reject",
                "tier_rank": 1,
            },
        ],
        fieldnames=[
            "lvis_category_id",
            "lvis_category_name",
            "mapped_coco_category_id",
            "mapped_coco_category_name",
            "mapping_kind",
            "prior_kind",
            "confidence_tier",
            "determination_tier",
            "mapping_class",
            "candidate_source",
            "candidate_kind",
            "decision_rule_version",
            "decision_rule",
            "n_match",
            "n_images",
            "precision_like",
            "coverage_like",
            "mean_iou",
            "median_iou",
            "iou_ge_05_rate",
            "iou_ge_075_rate",
            "support_score",
            "geometry_score",
            "proxy_score",
            "determination_reason",
            "tier_rank",
        ],
    )

    result = export_augmented_coco_with_lvis_proxies(
        base_jsonl_path=base_jsonl,
        projection_dir=projection_dir,
        determined_mapping_csv_path=mapping_csv,
        output_jsonl_path=output_jsonl,
        raw_coco_annotation_paths=(raw_ann,),
        config=ProxyAugmentConfig(include_plausible=True),
    )

    rows = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert [obj["desc"] for obj in row["objects"]] == [
        "dining table",
        "bottle",
        "cup",
        "clock",
    ]
    assert row["images"] == ["images/val2017/000000000001.jpg"]
    ns = row["metadata"]["coordexp_proxy_supervision"]
    assert len(ns["object_supervision"]) == len(row["objects"]) == 4
    assert [entry["proxy_tier"] for entry in ns["object_supervision"]] == [
        "plausible",
        "strict",
        "strict",
        "real",
    ]
    assert ns["object_supervision"][0]["coord_weight"] == 0.0
    assert ns["object_supervision"][1]["coord_weight"] == 1.0
    assert ns["object_supervision"][0]["desc_ce_weight"] == 0.25
    assert ns["summary"] == {
        "real_count": 1,
        "strict_count": 2,
        "plausible_count": 1,
        "include_plausible": True,
    }
    assert result.summary["accepted_proxy_counts"] == {"plausible": 1, "strict": 2}
    assert result.summary["record_count_with_added_proxies"] == 1


def test_export_augmented_coco_with_lvis_proxies_can_exclude_plausible(
    tmp_path: Path,
) -> None:
    base_jsonl = tmp_path / "base.coord.jsonl"
    projection_dir = tmp_path / "projection"
    mapping_csv = tmp_path / "determined_proxy_mappings.csv"
    raw_ann = tmp_path / "instances_val2017.json"
    output_jsonl = tmp_path / "augmented.coord.jsonl"

    _write_jsonl(
        base_jsonl,
        [
            {
                "images": ["images/val2017/000000000001.jpg"],
                "objects": [],
                "width": 640,
                "height": 480,
                "image_id": 1,
                "file_name": "images/val2017/000000000001.jpg",
                "metadata": {"source": "coco2017", "split": "val"},
            }
        ],
    )
    _write_json(
        raw_ann,
        {"images": [{"id": 1, "width": 100, "height": 100}], "annotations": [], "categories": []},
    )
    _write_jsonl(
        projection_dir / "recovered_coco80_instances.jsonl",
        [
            {
                "image_id": 1,
                "lvis_ann_id": 201,
                "lvis_category_id": 401,
                "lvis_category_name": "tablecloth",
                "mapped_coco_category_id": 67,
                "mapped_coco_category_name": "dining table",
                "mapping_kind": "semantic_evidence",
                "mapping_tier": "usable",
                "bbox_xyxy": [10.0, 10.0, 20.0, 20.0],
                "recovery_view": "strict_plus_usable",
                "why_recovered": "semantic plausible",
            }
        ],
    )
    _write_csv(
        mapping_csv,
        [
            {
                "lvis_category_id": 401,
                "lvis_category_name": "tablecloth",
                "mapped_coco_category_id": 67,
                "mapped_coco_category_name": "dining table",
                "mapping_kind": "semantic_evidence",
                "prior_kind": "",
                "confidence_tier": "usable",
                "determination_tier": "plausible",
                "mapping_class": "cue_only_proxy",
                "candidate_source": "empirical_match",
                "candidate_kind": "semantic_evidence",
                "decision_rule_version": "unit_test",
                "decision_rule": "plausible_cut",
                "n_match": 30,
                "n_images": 12,
                "precision_like": 0.93,
                "coverage_like": 0.58,
                "mean_iou": 0.81,
                "median_iou": 0.88,
                "iou_ge_05_rate": 0.89,
                "iou_ge_075_rate": 0.67,
                "support_score": 0.7,
                "geometry_score": 0.79,
                "proxy_score": 0.81,
                "determination_reason": "plausible",
                "tier_rank": 1,
            }
        ],
        fieldnames=[
            "lvis_category_id",
            "lvis_category_name",
            "mapped_coco_category_id",
            "mapped_coco_category_name",
            "mapping_kind",
            "prior_kind",
            "confidence_tier",
            "determination_tier",
            "mapping_class",
            "candidate_source",
            "candidate_kind",
            "decision_rule_version",
            "decision_rule",
            "n_match",
            "n_images",
            "precision_like",
            "coverage_like",
            "mean_iou",
            "median_iou",
            "iou_ge_05_rate",
            "iou_ge_075_rate",
            "support_score",
            "geometry_score",
            "proxy_score",
            "determination_reason",
            "tier_rank",
        ],
    )

    export_augmented_coco_with_lvis_proxies(
        base_jsonl_path=base_jsonl,
        projection_dir=projection_dir,
        determined_mapping_csv_path=mapping_csv,
        output_jsonl_path=output_jsonl,
        raw_coco_annotation_paths=(raw_ann,),
        config=ProxyAugmentConfig(include_plausible=False),
    )
    rows = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines()]
    row = rows[0]
    assert row["objects"] == []
    ns = row["metadata"]["coordexp_proxy_supervision"]
    assert ns["summary"]["plausible_count"] == 0
