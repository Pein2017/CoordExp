from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

import src.detection as detection
from src.detection.data import ObjectOrderingPlan, parse_raw_detection_row
from src.detection.dataset import DetectionDatasetRuntimeConfig, DetectionTrainingDataset
from src.detection.ir import DetectionGeometry as DetectionDocumentIRGeometry
from src.detection.scene import DetectionGeometry as DetectionSceneModuleGeometry
import src.detection.scene as scene_module
from src.detection.scene import (
    detection_scene_from_raw_row,
    normalized_detection_sample_from_scene,
)


class _FakeSwiftTemplate:
    tokenizer = object()


def _raw_row() -> dict:
    return {
        "images": ["images/train2017/example.jpg"],
        "objects": [
            {
                "bbox_2d": ["<|coord_10|>", "<|coord_20|>", "<|coord_30|>", "<|coord_40|>"],
                "desc": "striped cat",
                "category_id": 17,
                "category_name": "cat",
                "coco_ann_id": 101,
                "object_id": "obj-cat",
            },
            {
                "bbox_2d": ["<|coord_50|>", "<|coord_60|>", "<|coord_70|>", "<|coord_80|>"],
                "desc": "brown dog",
                "category_id": 18,
                "category_name": "dog",
                "coco_ann_id": 102,
                "object_id": "obj-dog",
            },
        ],
        "width": 640,
        "height": 480,
        "image_id": 9,
        "file_name": "images/train2017/example.jpg",
        "metadata": {
            "source": "unit",
            "split": "train",
            "supervision": {
                "object_supervision": {
                    "obj-cat": {"source_role": "query", "relation": "left"},
                    "obj-dog": {"source_role": "support", "relation": "right"},
                }
            },
        },
    }


def test_detection_scene_from_raw_row_preserves_explicit_semantics() -> None:
    raw = parse_raw_detection_row(_raw_row())

    scene = detection_scene_from_raw_row(
        raw,
        object_ordering=ObjectOrderingPlan.sorted(),
        image_reference="/resolved/images/train2017/example.jpg",
    )

    assert scene.image_id == 9
    assert scene.image_reference == "/resolved/images/train2017/example.jpg"
    assert scene.source_image_reference == "images/train2017/example.jpg"
    assert scene.width == 640
    assert scene.height == 480
    assert scene.coordinate_frame == "image"
    assert scene.coordinate_space == "norm1000"
    assert scene.bbox_chart == "xyxy"
    assert scene.object_ordering.strategy == "sorted"
    assert scene.realized_source_object_indices == (0, 1)
    assert scene.metadata.source == "unit"
    assert scene.metadata.split == "train"

    cat = scene.objects[0]
    assert cat.scene_object_index == 0
    assert cat.source_object_index == 0
    assert cat.label == "cat"
    assert cat.desc == "striped cat"
    assert cat.object_id == "obj-cat"
    assert cat.source_role == "query"
    assert cat.relation_snapshot == {"source_role": "query", "relation": "left"}
    assert cat.geometry.kind == "bbox_2d"
    assert cat.geometry.coordinate_frame == "image"
    assert cat.geometry.coordinate_space == "norm1000"
    assert cat.geometry.bbox_chart == "xyxy"
    assert cat.geometry.bbox_2d is not None
    assert cat.geometry.bbox_2d.tokens == (
        "<|coord_10|>",
        "<|coord_20|>",
        "<|coord_30|>",
        "<|coord_40|>",
    )
    assert cat.geometry.bbox_2d.values == (10, 20, 30, 40)

    assert not hasattr(scene, "rendered_assistant_text")
    assert not hasattr(scene, "token_ids")
    assert not hasattr(scene, "rollout_assignments")
    assert not hasattr(scene, "eval_metrics")


def test_detection_scene_from_raw_row_requires_resolved_image_reference() -> None:
    raw = parse_raw_detection_row(_raw_row())

    with pytest.raises(TypeError, match="image_reference"):
        detection_scene_from_raw_row(raw, object_ordering=ObjectOrderingPlan.sorted())
    with pytest.raises(ValueError, match="resolved image_reference"):
        detection_scene_from_raw_row(
            raw,
            object_ordering=ObjectOrderingPlan.sorted(),
            image_reference=None,  # type: ignore[arg-type]
        )


def test_detection_scene_from_raw_row_rejects_relative_local_image_reference() -> None:
    raw = parse_raw_detection_row(_raw_row())

    with pytest.raises(ValueError, match="absolute"):
        detection_scene_from_raw_row(
            raw,
            object_ordering=ObjectOrderingPlan.sorted(),
            image_reference="images/train2017/example.jpg",
        )


def test_detection_scene_from_raw_row_expands_tilde_image_reference() -> None:
    raw = parse_raw_detection_row(_raw_row())

    scene = detection_scene_from_raw_row(
        raw,
        object_ordering=ObjectOrderingPlan.sorted(),
        image_reference="~/coordexp-scene-test/example.jpg",
    )

    expected = str(Path("~/coordexp-scene-test/example.jpg").expanduser().resolve(strict=False))
    assert scene.image_reference == expected
    assert scene.images == (expected,)
    assert not scene.image_reference.startswith("~")


def test_detection_root_exports_keep_geometry_names_disambiguated() -> None:
    assert detection.DetectionGeometry is DetectionSceneModuleGeometry
    assert detection.DetectionSceneGeometry is DetectionSceneModuleGeometry
    assert detection.DetectionGeometry is not DetectionDocumentIRGeometry
    for name in [
        "DetectionScene",
        "DetectionObject",
        "DetectionGeometry",
        "RenderedDetectionSequence",
        "DetectionSequenceTemplate",
        "DetectionSupervisionView",
    ]:
        assert hasattr(detection, name), name
    assert not hasattr(detection, "DetectionDocumentGeometry")
    assert not hasattr(detection, "DetectionDocument")
    assert not hasattr(detection, "NormalizedDetectionSample")
    assert not hasattr(detection, "NormalizedDetectionObject")
    assert not hasattr(detection, "RenderedAssistantSequence")
    assert not hasattr(detection, "TokenizedDetectionExample")
    assert not hasattr(detection, "compute_recursive_detection_ce_batch_loss")
    assert not hasattr(detection, "normalize_recursive_detection_token_losses")
    assert not hasattr(detection, "RecursiveDetectionTargets")
    assert not hasattr(detection, "RecursiveDetectionLossResult")
    assert not hasattr(detection, "RecursiveDetectionLossWeights")


def test_detection_scene_projects_back_to_normalized_sample_without_raw_authority() -> None:
    raw = parse_raw_detection_row(_raw_row())
    ordering = ObjectOrderingPlan.random_permutation(seed=7, seed_source="unit")

    scene = detection_scene_from_raw_row(
        raw,
        object_ordering=ordering,
        image_reference="/resolved/images/train2017/example.jpg",
    )
    sample = normalized_detection_sample_from_scene(scene)

    assert sample.images == ("images/train2017/example.jpg",)
    assert sample.width == raw.width
    assert sample.height == raw.height
    assert sample.image_id == raw.image_id
    assert sample.file_name == raw.file_name
    assert sample.object_ordering.strategy == "random_permutation"
    assert sample.object_ordering.seed == 7
    assert sample.realized_source_object_indices == scene.realized_source_object_indices
    assert tuple(obj.source_object_index for obj in sample.objects) == scene.realized_source_object_indices
    assert tuple(obj.object_instance_id for obj in sample.objects) == tuple(
        obj.object_instance_id for obj in scene.objects
    )
    assert tuple(obj.bbox_2d.values for obj in sample.objects) == tuple(
        obj.geometry.require_bbox_2d().values for obj in scene.objects
    )


def test_detection_scene_from_normalized_sample_bridge_requires_valid_ordering() -> None:
    raw = parse_raw_detection_row(_raw_row())
    scene = detection_scene_from_raw_row(
        raw,
        object_ordering=ObjectOrderingPlan.sorted(),
        image_reference="/resolved/images/train2017/example.jpg",
    )
    sample = normalized_detection_sample_from_scene(scene)

    assert not hasattr(scene_module, "detection_scene_from_normalized_sample")
    bridge = scene_module.detection_scene_from_normalized_sample_bridge(
        sample,
        image_reference="/resolved/images/train2017/example.jpg",
    )

    assert bridge.realized_source_object_indices == (0, 1)
    assert tuple(obj.scene_object_index for obj in bridge.objects) == (0, 1)

    bad_object_index = replace(sample.objects[0], normalized_object_index=4)
    bad_sample = replace(
        sample,
        objects=(bad_object_index, *sample.objects[1:]),
    )
    with pytest.raises(ValueError, match="normalized_object_index"):
        scene_module.detection_scene_from_normalized_sample_bridge(
            bad_sample,
            image_reference="/resolved/images/train2017/example.jpg",
        )

    bad_ordering = sample.object_ordering.with_realized((1, 0))
    bad_sample = replace(sample, object_ordering=bad_ordering)
    with pytest.raises(ValueError, match="realized_source_object_indices"):
        scene_module.detection_scene_from_normalized_sample_bridge(
            bad_sample,
            image_reference="/resolved/images/train2017/example.jpg",
        )


def test_detection_scene_rejects_multiple_image_references() -> None:
    row = _raw_row()
    row["images"] = ["images/a.jpg", "images/b.jpg"]
    raw = parse_raw_detection_row(row)

    with pytest.raises(ValueError, match="exactly one image reference"):
        detection_scene_from_raw_row(
            raw,
            object_ordering=ObjectOrderingPlan.sorted(),
            image_reference="/resolved/images/a.jpg",
        )


def test_detection_training_dataset_exposes_resolved_scene_for_raw_jsonl_rows(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "image-root/images/train2017/example.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"placeholder")
    dataset = DetectionTrainingDataset(
        [_raw_row()],
        swift_template=_FakeSwiftTemplate(),
        config=DetectionDatasetRuntimeConfig(
            image_root=str(tmp_path / "image-root"),
            detection_template_id="compact",
            mode="teacher_forcing",
            object_ordering="sorted",
            user_prompt="Detect every object.",
            system_prompt=None,
            seed=123,
            state_weighting="uniform_permutation",
            normalization="semantic_image_bucket_balanced",
        ),
        dataset_name="unit",
    )

    scene = dataset.scene_for_row(0)

    assert scene.image_reference == str(image_path.resolve())
    assert scene.source_image_reference == "images/train2017/example.jpg"
    assert scene.object_ordering.strategy == "sorted"
    assert scene.realized_source_object_indices == (0, 1)
