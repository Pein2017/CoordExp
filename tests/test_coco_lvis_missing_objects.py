from __future__ import annotations

from src.analysis.coco_lvis_missing_objects import (
    AnalysisConfig,
    MappingDecisionThresholds,
    analyze_coco_lvis_overlap,
    analyze_lvis_to_coco80_projection,
    build_category_mapping,
)


def _payload(
    *,
    categories,
    images,
    annotations,
):
    return {
        "categories": categories,
        "images": images,
        "annotations": annotations,
    }


def _projection_test_config(
    *,
    strict_exact_min_match_count: int = 2,
    usable_exact_min_match_count: int = 1,
    strict_semantic_min_match_count: int = 2,
    usable_semantic_min_match_count: int = 1,
) -> AnalysisConfig:
    strict_exact = MappingDecisionThresholds(
        min_match_count=strict_exact_min_match_count,
        min_precision_like=0.90,
        min_coverage_like=0.20,
        min_mean_iou=0.90,
        min_median_iou=0.90,
        min_iou_ge_05_rate=1.00,
        min_iou_ge_075_rate=1.00,
        min_image_count=1,
        max_runner_up_ratio=0.25,
    )
    usable_exact = MappingDecisionThresholds(
        min_match_count=usable_exact_min_match_count,
        min_precision_like=0.70,
        min_coverage_like=0.10,
        min_mean_iou=0.70,
        min_median_iou=0.70,
        min_iou_ge_05_rate=1.00,
        min_iou_ge_075_rate=1.00,
        min_image_count=1,
        max_runner_up_ratio=0.60,
    )
    strict_semantic = MappingDecisionThresholds(
        min_match_count=strict_semantic_min_match_count,
        min_precision_like=0.90,
        min_coverage_like=0.20,
        min_mean_iou=0.90,
        min_median_iou=0.90,
        min_iou_ge_05_rate=1.00,
        min_iou_ge_075_rate=1.00,
        min_image_count=1,
        max_runner_up_ratio=0.25,
    )
    usable_semantic = MappingDecisionThresholds(
        min_match_count=usable_semantic_min_match_count,
        min_precision_like=0.70,
        min_coverage_like=0.10,
        min_mean_iou=0.70,
        min_median_iou=0.70,
        min_iou_ge_05_rate=1.00,
        min_iou_ge_075_rate=1.00,
        min_image_count=1,
        max_runner_up_ratio=0.60,
    )
    return AnalysisConfig(
        evidence_pair_iou_threshold=0.5,
        recovery_iou_threshold=0.5,
        recovery_max_conflicting_coco_iou=0.8,
        strict_exact_thresholds=strict_exact,
        usable_exact_thresholds=usable_exact,
        strict_semantic_thresholds=strict_semantic,
        usable_semantic_thresholds=usable_semantic,
    )


def test_build_category_mapping_strict_and_expanded_modes() -> None:
    coco_categories = {
        1: {"id": 1, "name": "couch"},
        2: {"id": 2, "name": "laptop"},
        3: {"id": 3, "name": "car"},
        4: {"id": 4, "name": "sports ball"},
        5: {"id": 5, "name": "hot dog"},
    }
    lvis_categories = {
        11: {"id": 11, "name": "sofa", "synonyms": ["sofa", "couch"]},
        12: {"id": 12, "name": "laptop_computer", "synonyms": ["laptop_computer"]},
        13: {"id": 13, "name": "car_(automobile)", "synonyms": ["car_(automobile)"]},
        14: {"id": 14, "name": "race_car", "synonyms": ["race_car"]},
        15: {"id": 15, "name": "soccer_ball", "synonyms": ["soccer_ball"]},
        16: {"id": 16, "name": "basketball", "synonyms": ["basketball"]},
    }

    strict_mapping = build_category_mapping(
        coco_categories,
        lvis_categories,
        mapping_mode="strict",
    )
    assert strict_mapping.coco_to_lvis[1].strategy == "synonym"
    assert strict_mapping.coco_to_lvis[2].strategy == "manual_alias"
    assert 3 not in strict_mapping.coco_to_lvis
    assert 4 not in strict_mapping.coco_to_lvis
    assert "hot dog" in strict_mapping.summary["unmapped_coco_categories"]

    expanded_mapping = build_category_mapping(
        coco_categories,
        lvis_categories,
        mapping_mode="expanded",
    )
    assert expanded_mapping.coco_to_lvis[3].strategy == "manual_broad"
    assert expanded_mapping.coco_to_lvis[3].lvis_category_names == (
        "car_(automobile)",
        "race_car",
    )
    assert expanded_mapping.coco_to_lvis[4].lvis_category_names == (
        "basketball",
        "soccer_ball",
    )


def test_analysis_skips_not_exhaustive_lvis_categories() -> None:
    coco_payload = _payload(
        categories=[{"id": 1, "name": "book"}],
        images=[
            {
                "id": 10,
                "width": 100,
                "height": 100,
                "file_name": "000000000010.jpg",
            }
        ],
        annotations=[],
    )
    lvis_payload = _payload(
        categories=[
            {
                "id": 101,
                "name": "book",
                "synonyms": ["book"],
                "frequency": "f",
            }
        ],
        images=[
            {
                "id": 10,
                "width": 100,
                "height": 100,
                "coco_url": "http://images.cocodataset.org/val2017/000000000010.jpg",
                "neg_category_ids": [],
                "not_exhaustive_category_ids": [101],
            }
        ],
        annotations=[
            {
                "id": 1001,
                "image_id": 10,
                "category_id": 101,
                "bbox": [10, 10, 20, 20],
                "area": 400,
                "iscrowd": 0,
            }
        ],
    )

    result = analyze_coco_lvis_overlap(
        [coco_payload],
        [lvis_payload],
        coco_source_names=["instances_val2017"],
        lvis_source_names=["lvis_v1_val"],
        config=AnalysisConfig(),
    )

    assert result.summary["analysis"]["mappable_lvis_instances"] == 1
    assert result.summary["analysis"]["skipped_not_exhaustive"] == 1
    assert result.summary["analysis"].get("unmatched_recoverable_instances", 0) == 0
    assert result.unmatched_instances == []
    assert result.per_image_rows[0]["skipped_not_exhaustive"] == 1


def test_analysis_reports_unmatched_and_respects_one_to_one_matching() -> None:
    coco_payload = _payload(
        categories=[{"id": 1, "name": "person"}],
        images=[
            {
                "id": 20,
                "width": 200,
                "height": 200,
                "file_name": "000000000020.jpg",
            }
        ],
        annotations=[
            {
                "id": 2001,
                "image_id": 20,
                "category_id": 1,
                "bbox": [10, 10, 40, 40],
                "area": 1600,
                "iscrowd": 0,
            }
        ],
    )
    lvis_payload = _payload(
        categories=[
            {
                "id": 201,
                "name": "person",
                "synonyms": ["person"],
                "frequency": "f",
            }
        ],
        images=[
            {
                "id": 20,
                "width": 200,
                "height": 200,
                "coco_url": "http://images.cocodataset.org/train2017/000000000020.jpg",
                "neg_category_ids": [],
                "not_exhaustive_category_ids": [],
            }
        ],
        annotations=[
            {
                "id": 3001,
                "image_id": 20,
                "category_id": 201,
                "bbox": [10, 10, 40, 40],
                "area": 1600,
                "iscrowd": 0,
            },
            {
                "id": 3002,
                "image_id": 20,
                "category_id": 201,
                "bbox": [120, 120, 40, 40],
                "area": 1600,
                "iscrowd": 0,
            },
        ],
    )

    result = analyze_coco_lvis_overlap(
        [coco_payload],
        [lvis_payload],
        coco_source_names=["instances_train2017"],
        lvis_source_names=["lvis_v1_train"],
        config=AnalysisConfig(iou_threshold=0.5),
    )

    assert result.summary["shared_images"]["by_coco_image_split"] == {"train2017": 1}
    assert result.summary["analysis"]["candidate_lvis_instances"] == 2
    assert result.summary["analysis"]["matched_lvis_instances"] == 1
    assert result.summary["analysis"]["unmatched_recoverable_instances"] == 1
    assert len(result.unmatched_instances) == 1
    unmatched = result.unmatched_instances[0]
    assert unmatched["mapped_coco_category_name"] == "person"
    assert unmatched["best_same_category_iou"] == 0.0
    assert result.per_image_rows[0]["unmatched_recoverable_instances"] == 1


def test_projection_learns_exact_semantic_and_reject_mappings() -> None:
    coco_payload = _payload(
        categories=[
            {"id": 1, "name": "person"},
            {"id": 2, "name": "dog"},
            {"id": 3, "name": "cat"},
        ],
        images=[
            {"id": 1, "width": 100, "height": 100, "file_name": "000000000001.jpg"},
            {"id": 2, "width": 100, "height": 100, "file_name": "000000000002.jpg"},
            {"id": 3, "width": 100, "height": 100, "file_name": "000000000003.jpg"},
            {"id": 4, "width": 100, "height": 100, "file_name": "000000000004.jpg"},
        ],
        annotations=[
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [5, 5, 30, 60], "area": 1800, "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [55, 20, 25, 25], "area": 625, "iscrowd": 0},
            {"id": 3, "image_id": 2, "category_id": 1, "bbox": [8, 6, 30, 60], "area": 1800, "iscrowd": 0},
            {"id": 4, "image_id": 2, "category_id": 2, "bbox": [55, 20, 25, 25], "area": 625, "iscrowd": 0},
            {"id": 5, "image_id": 3, "category_id": 2, "bbox": [20, 20, 30, 30], "area": 900, "iscrowd": 0},
            {"id": 6, "image_id": 4, "category_id": 3, "bbox": [60, 55, 20, 20], "area": 400, "iscrowd": 0},
        ],
    )
    lvis_payload = _payload(
        categories=[
            {"id": 101, "name": "person", "synonyms": ["person"], "frequency": "f"},
            {"id": 102, "name": "golden_retriever", "synonyms": ["golden retriever"], "frequency": "f"},
            {"id": 103, "name": "ambiguous_pet", "synonyms": ["ambiguous pet"], "frequency": "f"},
        ],
        images=[
            {
                "id": 1,
                "width": 100,
                "height": 100,
                "coco_url": "http://images.cocodataset.org/val2017/000000000001.jpg",
                "neg_category_ids": [],
                "not_exhaustive_category_ids": [],
            },
            {
                "id": 2,
                "width": 100,
                "height": 100,
                "coco_url": "http://images.cocodataset.org/val2017/000000000002.jpg",
                "neg_category_ids": [],
                "not_exhaustive_category_ids": [],
            },
            {
                "id": 3,
                "width": 100,
                "height": 100,
                "coco_url": "http://images.cocodataset.org/val2017/000000000003.jpg",
                "neg_category_ids": [],
                "not_exhaustive_category_ids": [],
            },
            {
                "id": 4,
                "width": 100,
                "height": 100,
                "coco_url": "http://images.cocodataset.org/val2017/000000000004.jpg",
                "neg_category_ids": [],
                "not_exhaustive_category_ids": [],
            },
        ],
        annotations=[
            {"id": 1001, "image_id": 1, "category_id": 101, "bbox": [5, 5, 30, 60], "area": 1800, "iscrowd": 0},
            {"id": 1002, "image_id": 1, "category_id": 102, "bbox": [55, 20, 25, 25], "area": 625, "iscrowd": 0},
            {"id": 1004, "image_id": 2, "category_id": 101, "bbox": [8, 6, 30, 60], "area": 1800, "iscrowd": 0},
            {"id": 1005, "image_id": 2, "category_id": 102, "bbox": [55, 20, 25, 25], "area": 625, "iscrowd": 0},
            {"id": 1006, "image_id": 3, "category_id": 103, "bbox": [20, 20, 30, 30], "area": 900, "iscrowd": 0},
            {"id": 1007, "image_id": 4, "category_id": 103, "bbox": [60, 55, 20, 20], "area": 400, "iscrowd": 0},
        ],
    )

    result = analyze_lvis_to_coco80_projection(
        [coco_payload],
        [lvis_payload],
        coco_source_names=["instances_val2017"],
        lvis_source_names=["lvis_v1_val"],
        config=_projection_test_config(),
    )

    learned_by_name = {
        row["lvis_category_name"]: row
        for row in result.learned_mapping_rows
    }
    assert learned_by_name["person"]["confidence_tier"] == "strict"
    assert learned_by_name["person"]["mapping_kind"] == "exact_canonical"
    assert learned_by_name["person"]["mapped_coco_category_name"] == "person"

    assert learned_by_name["golden_retriever"]["confidence_tier"] == "strict"
    assert learned_by_name["golden_retriever"]["mapping_kind"] == "semantic_evidence"
    assert learned_by_name["golden_retriever"]["mapped_coco_category_name"] == "dog"

    assert learned_by_name["ambiguous_pet"]["confidence_tier"] == "reject"
    assert learned_by_name["ambiguous_pet"]["mapped_coco_category_name"] is None
    assert "runner_up_ratio" in learned_by_name["ambiguous_pet"]["rejection_reason"]
    assert result.summary["learned_mapping"]["strict_mapping_count"] == 2
    assert result.summary["learned_mapping"]["reject_mapping_count"] == 1


def test_projection_recovery_blocks_mapped_coco_category_when_sibling_is_not_exhaustive() -> None:
    coco_payload = _payload(
        categories=[{"id": 1, "name": "dog"}],
        images=[
            {"id": 1, "width": 100, "height": 100, "file_name": "000000000001.jpg"},
            {"id": 2, "width": 100, "height": 100, "file_name": "000000000002.jpg"},
            {"id": 3, "width": 100, "height": 100, "file_name": "000000000003.jpg"},
            {"id": 4, "width": 100, "height": 100, "file_name": "000000000004.jpg"},
            {"id": 5, "width": 100, "height": 100, "file_name": "000000000005.jpg"},
        ],
        annotations=[
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 30, 30], "area": 900, "iscrowd": 0},
            {"id": 2, "image_id": 2, "category_id": 1, "bbox": [12, 12, 30, 30], "area": 900, "iscrowd": 0},
            {"id": 3, "image_id": 3, "category_id": 1, "bbox": [70, 70, 10, 10], "area": 100, "iscrowd": 0},
            {"id": 4, "image_id": 5, "category_id": 1, "bbox": [14, 14, 30, 30], "area": 900, "iscrowd": 0},
        ],
    )
    lvis_payload = _payload(
        categories=[
            {"id": 201, "name": "golden_retriever", "synonyms": ["golden retriever"], "frequency": "f"},
            {"id": 202, "name": "poodle", "synonyms": ["poodle"], "frequency": "f"},
        ],
        images=[
            {
                "id": 1,
                "width": 100,
                "height": 100,
                "coco_url": "http://images.cocodataset.org/val2017/000000000001.jpg",
                "neg_category_ids": [],
                "not_exhaustive_category_ids": [],
            },
            {
                "id": 2,
                "width": 100,
                "height": 100,
                "coco_url": "http://images.cocodataset.org/val2017/000000000002.jpg",
                "neg_category_ids": [],
                "not_exhaustive_category_ids": [],
            },
            {
                "id": 3,
                "width": 100,
                "height": 100,
                "coco_url": "http://images.cocodataset.org/val2017/000000000003.jpg",
                "neg_category_ids": [],
                "not_exhaustive_category_ids": [202],
            },
            {
                "id": 4,
                "width": 100,
                "height": 100,
                "coco_url": "http://images.cocodataset.org/val2017/000000000004.jpg",
                "neg_category_ids": [],
                "not_exhaustive_category_ids": [],
            },
            {
                "id": 5,
                "width": 100,
                "height": 100,
                "coco_url": "http://images.cocodataset.org/val2017/000000000005.jpg",
                "neg_category_ids": [],
                "not_exhaustive_category_ids": [],
            },
        ],
        annotations=[
            {"id": 2001, "image_id": 1, "category_id": 201, "bbox": [10, 10, 30, 30], "area": 900, "iscrowd": 0},
            {"id": 2002, "image_id": 2, "category_id": 201, "bbox": [12, 12, 30, 30], "area": 900, "iscrowd": 0},
            {"id": 2003, "image_id": 3, "category_id": 201, "bbox": [10, 10, 30, 30], "area": 900, "iscrowd": 0},
            {"id": 2004, "image_id": 4, "category_id": 201, "bbox": [20, 20, 30, 30], "area": 900, "iscrowd": 0},
            {"id": 2005, "image_id": 5, "category_id": 202, "bbox": [14, 14, 30, 30], "area": 900, "iscrowd": 0},
        ],
    )

    result = analyze_lvis_to_coco80_projection(
        [coco_payload],
        [lvis_payload],
        coco_source_names=["instances_val2017"],
        lvis_source_names=["lvis_v1_val"],
        config=_projection_test_config(
            strict_semantic_min_match_count=2,
            usable_semantic_min_match_count=1,
        ),
    )

    learned_by_name = {
        row["lvis_category_name"]: row
        for row in result.learned_mapping_rows
    }
    assert learned_by_name["golden_retriever"]["confidence_tier"] == "strict"
    assert learned_by_name["poodle"]["confidence_tier"] == "usable"

    recovered_by_image = {
        row["image_id"]: row
        for row in result.recovered_instances
    }
    assert recovered_by_image[3]["included_in_strict_only"] is True
    assert recovered_by_image[3]["included_in_strict_plus_usable"] is False
    assert recovered_by_image[4]["included_in_strict_only"] is True
    assert recovered_by_image[4]["included_in_strict_plus_usable"] is True
    assert result.summary["recovery"]["strict_only_count"] == 2
    assert result.summary["recovery"]["strict_plus_usable_count"] == 1
