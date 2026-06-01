from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

import src.detection as detection
from src.detection.data import (
    CoordinateTokenBox,
    DetectionMetadata,
    NormalizedDetectionObject,
    NormalizedDetectionSample,
    ObjectOrderingPlan,
)
from src.detection.ir import DetectionDocument, detection_document_from_normalized_sample


def test_detection_document_adapts_normalized_sample_without_behavior_change() -> None:
    sample = _sample(
        object_ordering=ObjectOrderingPlan.sorted(
            seed_source="source_order",
        ).with_realized((0, 1)),
    )

    doc = DetectionDocument.from_normalized_sample(sample)

    assert doc.images == ("images/train2017/example.jpg",)
    assert doc.width == 640
    assert doc.height == 480
    assert doc.image_id == 123
    assert doc.file_name == "example.jpg"
    assert doc.object_ordering is sample.object_ordering
    assert doc.realized_source_object_indices == (0, 1)
    assert doc.coordinate_surface == "coord_token"
    assert doc.bbox_format == "xyxy"
    assert doc.metadata == {"source": "coco", "split": "train2017"}

    assert [obj.object_instance_id for obj in doc.objects] == [
        "coco:train2017:image_id=123:source_object_index=0:coco_ann_id=101",
        "coco:train2017:image_id=123:source_object_index=1:coco_ann_id=102",
    ]
    assert [obj.object_index for obj in doc.objects] == [0, 1]
    assert [obj.source_object_index for obj in doc.objects] == [0, 1]
    assert [obj.desc for obj in doc.objects] == ["red car", "traffic light"]
    assert [obj.category_id for obj in doc.objects] == [3, 10]
    assert [obj.category_name for obj in doc.objects] == ["car", "traffic light"]
    assert [obj.coco_ann_id for obj in doc.objects] == [101, 102]

    first_slots = doc.objects[0].geometry.slots
    assert [slot.slot_name for slot in first_slots] == ["x1", "y1", "x2", "y2"]
    assert [slot.coord_token for slot in first_slots] == [
        "<|coord_10|>",
        "<|coord_20|>",
        "<|coord_30|>",
        "<|coord_40|>",
    ]
    assert [slot.render_text for slot in first_slots] == [
        "<|coord_10|>",
        "<|coord_20|>",
        "<|coord_30|>",
        "<|coord_40|>",
    ]
    assert [slot.norm1000_value for slot in first_slots] == [10, 20, 30, 40]
    assert [slot.coord_token_id for slot in first_slots] == [None, None, None, None]
    assert [slot.object_instance_id for slot in first_slots] == [
        "coco:train2017:image_id=123:source_object_index=0:coco_ann_id=101",
        "coco:train2017:image_id=123:source_object_index=0:coco_ann_id=101",
        "coco:train2017:image_id=123:source_object_index=0:coco_ann_id=101",
        "coco:train2017:image_id=123:source_object_index=0:coco_ann_id=101",
    ]
    assert [slot.object_index for slot in first_slots] == [0, 0, 0, 0]


def test_detection_document_preserves_random_permutation_ordering() -> None:
    sample = _sample(
        object_ordering=ObjectOrderingPlan.random_permutation(
            seed=17,
            seed_source="unit-test",
        ).with_realized((1, 0)),
        source_indices=(1, 0),
    )

    doc = DetectionDocument.from_normalized_sample(sample)

    assert doc.object_ordering.strategy == "random_permutation"
    assert doc.object_ordering.seed == 17
    assert doc.object_ordering.seed_source == "unit-test"
    assert doc.realized_source_object_indices == (1, 0)
    assert [obj.source_object_index for obj in doc.objects] == [1, 0]


def test_detection_document_field_order_policy_is_coordinate_surface_independent() -> None:
    sample = _sample(object_ordering=ObjectOrderingPlan.sorted())

    doc = DetectionDocument.from_normalized_sample(
        sample,
        coordinate_surface="numeric_text",
    )

    assert doc.coordinate_surface == "numeric_text"
    assert doc.objects[0].field_order_policy == "desc_then_bbox_2d"


def test_detection_document_falls_back_to_object_source_indices() -> None:
    sample = _sample(object_ordering=ObjectOrderingPlan.sorted())

    doc = DetectionDocument.from_normalized_sample(sample)

    assert sample.realized_source_object_indices == ()
    assert doc.realized_source_object_indices == (0, 1)


def test_detection_ir_dataclasses_are_frozen() -> None:
    doc = DetectionDocument.from_normalized_sample(
        _sample(object_ordering=ObjectOrderingPlan.sorted())
    )

    with pytest.raises(FrozenInstanceError):
        doc.coordinate_surface = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        doc.objects[0].desc = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        doc.metadata["source"] = "changed"  # type: ignore[index]


def test_detection_document_is_semantic_only() -> None:
    doc = DetectionDocument.from_normalized_sample(
        _sample(object_ordering=ObjectOrderingPlan.sorted())
    )

    assert not hasattr(doc, "rendered_text")
    assert not hasattr(doc, "input_ids")
    assert not hasattr(doc, "labels")
    assert not hasattr(doc, "recursive_detection_targets")


def test_detection_ir_bridge_is_private_not_root_public() -> None:
    sample = _sample(object_ordering=ObjectOrderingPlan.sorted())
    doc = detection_document_from_normalized_sample(sample)

    assert not hasattr(detection, "DetectionDocument")
    assert not hasattr(detection, "detection_document_from_normalized_sample")
    assert isinstance(doc, DetectionDocument)
    assert doc.objects[0].object_instance_id == sample.objects[0].object_instance_id


def _sample(
    *,
    object_ordering: ObjectOrderingPlan,
    source_indices: tuple[int, int] = (0, 1),
) -> NormalizedDetectionSample:
    source_index_a, source_index_b = source_indices
    return NormalizedDetectionSample(
        images=("images/train2017/example.jpg",),
        objects=(
            NormalizedDetectionObject(
                normalized_object_index=0,
                source_object_index=source_index_a,
                object_instance_id=(
                    "coco:train2017:image_id=123:"
                    f"source_object_index={source_index_a}:coco_ann_id=101"
                ),
                desc="red car",
                bbox_2d=CoordinateTokenBox(
                    "<|coord_10|>",
                    "<|coord_20|>",
                    "<|coord_30|>",
                    "<|coord_40|>",
                ),
                category_id=3,
                category_name="car",
                coco_ann_id=101,
            ),
            NormalizedDetectionObject(
                normalized_object_index=1,
                source_object_index=source_index_b,
                object_instance_id=(
                    "coco:train2017:image_id=123:"
                    f"source_object_index={source_index_b}:coco_ann_id=102"
                ),
                desc="traffic light",
                bbox_2d=CoordinateTokenBox(
                    "<|coord_50|>",
                    "<|coord_60|>",
                    "<|coord_70|>",
                    "<|coord_80|>",
                ),
                category_id=10,
                category_name="traffic light",
                coco_ann_id=102,
            ),
        ),
        width=640,
        height=480,
        image_id=123,
        file_name="example.jpg",
        metadata=DetectionMetadata(source="coco", split="train2017"),
        object_ordering=object_ordering,
    )
