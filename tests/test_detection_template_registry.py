import pytest

from src.detection.data import (
    CoordinateTokenBox,
    DetectionMetadata,
    NormalizedDetectionObject,
    NormalizedDetectionSample,
    ObjectOrderingPlan,
)
from src.detection.template import (
    CompactFullTemplate,
    DetectionSequenceTemplate,
    Stage1JsonPrettyTemplate,
    get_detection_template,
)


def _sample() -> NormalizedDetectionSample:
    return NormalizedDetectionSample(
        images=("image.jpg",),
        objects=(
            NormalizedDetectionObject(
                normalized_object_index=0,
                source_object_index=7,
                object_instance_id="img-1:ann-11:src-7",
                desc="traffic light",
                bbox_2d=CoordinateTokenBox(
                    "<|coord_10|>",
                    "<|coord_20|>",
                    "<|coord_30|>",
                    "<|coord_40|>",
                ),
                category_id=10,
                category_name="traffic light",
                coco_ann_id=11,
            ),
        ),
        width=640,
        height=480,
        image_id=1,
        file_name="image.jpg",
        metadata=DetectionMetadata(source="unit", split="test"),
        object_ordering=ObjectOrderingPlan.sorted().with_realized((7,)),
    )


def test_template_registry_exposes_only_greenfield_template_ids() -> None:
    stage1 = get_detection_template("stage1_json_pretty")
    compact = get_detection_template("compact")

    assert isinstance(stage1, Stage1JsonPrettyTemplate)
    assert isinstance(compact, CompactFullTemplate)
    assert isinstance(stage1, DetectionSequenceTemplate)
    assert isinstance(compact, DetectionSequenceTemplate)

    assert stage1.template_id == "stage1_json_pretty"
    assert compact.template_id == "compact"
    assert stage1.capabilities.version == 1
    assert compact.capabilities.version == 1
    assert stage1.capabilities.coordinate_surface == "coord_token"
    assert compact.capabilities.coordinate_surface == "coord_token"
    assert stage1.capabilities.bbox_format == "xyxy"
    assert compact.capabilities.bbox_format == "xyxy"
    assert stage1.capabilities.object_field_order == "desc_first"
    assert compact.capabilities.object_field_order == "compact_row"
    assert stage1.capabilities.supports_recursive_detection_ce is True
    assert compact.capabilities.supports_recursive_detection_ce is True
    assert stage1.capabilities.supports_et_rmp_ce is True
    assert compact.capabilities.supports_et_rmp_ce is True
    assert stage1.capabilities.supports_static_packing is True
    assert compact.capabilities.supports_static_packing is True
    assert stage1.capabilities.geometry_kinds == ("bbox_2d",)
    assert compact.capabilities.geometry_kinds == ("bbox_2d",)


def test_template_registry_rejects_unknown_or_legacy_template_ids() -> None:
    with pytest.raises(ValueError, match="Unsupported detection template"):
        get_detection_template("coordjson_legacy")

    with pytest.raises(ValueError, match="Unsupported detection template"):
        get_detection_template("compact_no_desc")

    with pytest.raises(ValueError, match="Unsupported detection template"):
        get_detection_template("compact_full")


def test_templates_reject_unsupported_surfaces_before_rendering() -> None:
    sample = _sample()
    template = get_detection_template("stage1_json_pretty")

    with pytest.raises(ValueError, match="coordinate_surface=coord_token"):
        template.validate_sample(sample, coordinate_surface="norm1000")

    with pytest.raises(ValueError, match="bbox_format=xyxy"):
        template.validate_sample(sample, bbox_format="cxcywh")

    with pytest.raises(ValueError, match="prompt template mismatch"):
        template.validate_sample(sample, prompt_template_id="compact")
