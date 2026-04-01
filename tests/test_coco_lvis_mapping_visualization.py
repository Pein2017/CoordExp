from __future__ import annotations

from src.analysis.coco_lvis_mapping_visualization import (
    SemanticVisConfig,
    build_semantic_example_record,
    select_representative_usable_semantic_examples,
)
from src.analysis.coco_lvis_missing_objects import LoadedDataset


def _loaded_dataset(
    *,
    dataset_kind: str,
    categories_by_id,
    images_by_id,
    annotations_by_image,
) -> LoadedDataset:
    return LoadedDataset(
        dataset_kind=dataset_kind,
        source_name=f"{dataset_kind}_test",
        categories_by_id=categories_by_id,
        images_by_id=images_by_id,
        annotations_by_image=annotations_by_image,
        invalid_annotation_count=0,
    )


def test_select_representative_examples_prefers_cleaner_image() -> None:
    coco_dataset = _loaded_dataset(
        dataset_kind="coco",
        categories_by_id={
            53: {"id": 53, "name": "apple"},
        },
        images_by_id={
            1: {"id": 1, "width": 100, "height": 100, "file_name": "000000000001.jpg", "coco_image_split": "val2017"},
            2: {"id": 2, "width": 100, "height": 100, "file_name": "000000000002.jpg", "coco_image_split": "val2017"},
        },
        annotations_by_image={
            1: [
                {
                    "annotation_id": 11,
                    "image_id": 1,
                    "category_id": 53,
                    "category_name": "apple",
                    "bbox_xyxy": [10.0, 10.0, 40.0, 40.0],
                }
            ],
            2: [
                {
                    "annotation_id": 21,
                    "image_id": 2,
                    "category_id": 53,
                    "category_name": "apple",
                    "bbox_xyxy": [12.0, 12.0, 42.0, 42.0],
                }
            ],
        },
    )
    lvis_dataset = _loaded_dataset(
        dataset_kind="lvis",
        categories_by_id={
            301: {"id": 301, "name": "pear", "frequency": "f"},
            302: {"id": 302, "name": "apple", "frequency": "f"},
        },
        images_by_id={
            1: {
                "id": 1,
                "width": 100,
                "height": 100,
                "file_name": "000000000001.jpg",
                "coco_image_split": "val2017",
                "source_name": "lvis_v1_val",
            },
            2: {
                "id": 2,
                "width": 100,
                "height": 100,
                "file_name": "000000000002.jpg",
                "coco_image_split": "val2017",
                "source_name": "lvis_v1_val",
            },
        },
        annotations_by_image={
            1: [
                {
                    "annotation_id": 1001,
                    "image_id": 1,
                    "category_id": 301,
                    "category_name": "pear",
                    "bbox_xyxy": [10.0, 10.0, 40.0, 40.0],
                },
                {
                    "annotation_id": 1002,
                    "image_id": 1,
                    "category_id": 302,
                    "category_name": "apple",
                    "bbox_xyxy": [50.0, 10.0, 80.0, 40.0],
                },
            ],
            2: [
                {
                    "annotation_id": 2001,
                    "image_id": 2,
                    "category_id": 301,
                    "category_name": "pear",
                    "bbox_xyxy": [12.0, 12.0, 42.0, 42.0],
                }
            ],
        },
    )
    learned_mapping_rows = [
        {
            "lvis_category_id": 301,
            "lvis_category_name": "pear",
            "confidence_tier": "usable",
            "mapping_kind": "semantic_evidence",
            "mapped_coco_category_id": 53,
            "mapped_coco_category_name": "apple",
            "evidence_summary": {"n_match": 10, "precision_like": 0.91},
        },
        {
            "lvis_category_id": 302,
            "lvis_category_name": "apple",
            "confidence_tier": "strict",
            "mapping_kind": "exact_canonical",
            "mapped_coco_category_id": 53,
            "mapped_coco_category_name": "apple",
            "evidence_summary": {"n_match": 50, "precision_like": 0.99},
        },
    ]
    recovered_rows = [
        {
            "image_id": 1,
            "lvis_category_id": 301,
            "lvis_category_name": "pear",
            "mapped_coco_category_name": "apple",
            "mapping_tier": "usable",
            "mapping_kind": "semantic_evidence",
            "included_in_strict_plus_usable": True,
        },
        {
            "image_id": 2,
            "lvis_category_id": 301,
            "lvis_category_name": "pear",
            "mapped_coco_category_name": "apple",
            "mapping_tier": "usable",
            "mapping_kind": "semantic_evidence",
            "included_in_strict_plus_usable": True,
        },
    ]

    examples = select_representative_usable_semantic_examples(
        coco_dataset,
        lvis_dataset,
        learned_mapping_rows=learned_mapping_rows,
        recovered_rows=recovered_rows,
        explicit_lvis_category_names=("pear",),
        config=SemanticVisConfig(
            examples_per_mapping=1,
            max_total_gt_objects=4,
            max_total_pred_objects=4,
            max_sibling_lvis_instances=0,
            auto_top_mappings=1,
        ),
    )

    assert len(examples) == 1
    assert examples[0].image_id == 2
    assert examples[0].sibling_lvis_count == 0


def test_build_semantic_example_record_optionally_includes_siblings() -> None:
    coco_dataset = _loaded_dataset(
        dataset_kind="coco",
        categories_by_id={1: {"id": 1, "name": "remote"}},
        images_by_id={
            5: {
                "id": 5,
                "width": 100,
                "height": 80,
                "file_name": "000000000005.jpg",
                "coco_image_split": "val2017",
                "source_name": "instances_val2017",
            }
        },
        annotations_by_image={},
    )
    lvis_dataset = _loaded_dataset(
        dataset_kind="lvis",
        categories_by_id={
            11: {"id": 11, "name": "control", "frequency": "f"},
            12: {"id": 12, "name": "remote_control", "frequency": "f"},
        },
        images_by_id={
            5: {
                "id": 5,
                "width": 100,
                "height": 80,
                "file_name": "000000000005.jpg",
                "coco_image_split": "val2017",
                "source_name": "lvis_v1_val",
            }
        },
        annotations_by_image={},
    )
    learned_mapping_rows = [
        {
            "lvis_category_id": 11,
            "lvis_category_name": "control",
            "confidence_tier": "usable",
            "mapping_kind": "semantic_evidence",
            "mapped_coco_category_id": 1,
            "mapped_coco_category_name": "remote",
            "evidence_summary": {"n_match": 78, "precision_like": 0.987},
        },
        {
            "lvis_category_id": 12,
            "lvis_category_name": "remote_control",
            "confidence_tier": "strict",
            "mapping_kind": "exact_canonical",
            "mapped_coco_category_id": 1,
            "mapped_coco_category_name": "remote",
            "evidence_summary": {"n_match": 100, "precision_like": 0.99},
        },
    ]
    recovered_rows = [
        {
            "image_id": 5,
            "lvis_category_id": 11,
            "lvis_category_name": "control",
            "mapped_coco_category_name": "remote",
            "mapping_tier": "usable",
            "mapping_kind": "semantic_evidence",
            "included_in_strict_plus_usable": True,
        }
    ]
    target_annotation = {
        "annotation_id": 501,
        "image_id": 5,
        "category_id": 11,
        "category_name": "control",
        "bbox_xyxy": [5.0, 5.0, 20.0, 20.0],
    }
    sibling_annotation = {
        "annotation_id": 502,
        "image_id": 5,
        "category_id": 12,
        "category_name": "remote_control",
        "bbox_xyxy": [30.0, 10.0, 50.0, 30.0],
    }
    pred_annotation = {
        "annotation_id": 601,
        "image_id": 5,
        "category_id": 1,
        "category_name": "remote",
        "bbox_xyxy": [6.0, 6.0, 21.0, 21.0],
    }
    lvis_dataset = _loaded_dataset(
        dataset_kind="lvis",
        categories_by_id=lvis_dataset.categories_by_id,
        images_by_id=lvis_dataset.images_by_id,
        annotations_by_image={5: [target_annotation, sibling_annotation]},
    )
    coco_dataset = _loaded_dataset(
        dataset_kind="coco",
        categories_by_id=coco_dataset.categories_by_id,
        images_by_id=coco_dataset.images_by_id,
        annotations_by_image={5: [pred_annotation]},
    )

    example = select_representative_usable_semantic_examples(
        coco_dataset,
        lvis_dataset,
        learned_mapping_rows=learned_mapping_rows,
        recovered_rows=recovered_rows,
        explicit_lvis_category_names=("control",),
        config=SemanticVisConfig(
            examples_per_mapping=1,
            max_total_gt_objects=4,
            max_total_pred_objects=4,
            max_sibling_lvis_instances=1,
            auto_top_mappings=1,
        ),
    )[0]

    record_without_siblings = build_semantic_example_record(
        example,
        include_sibling_lvis_in_gt=False,
    )
    assert [obj["desc"] for obj in record_without_siblings["gt"]] == ["LVIS:control"]
    assert [obj["desc"] for obj in record_without_siblings["pred"]] == ["COCO:remote"]

    record_with_siblings = build_semantic_example_record(
        example,
        include_sibling_lvis_in_gt=True,
    )
    assert [obj["desc"] for obj in record_with_siblings["gt"]] == [
        "LVIS:control",
        "LVIS-sibling:remote_control",
    ]
