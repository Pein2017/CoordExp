from __future__ import annotations

from xml.etree import ElementTree

import pytest

from src.label_studio_coco_refinement.categories import COCO80_REGISTRY
from src.label_studio_coco_refinement.label_config import (
    build_label_config,
    label_config_fingerprint,
    validate_label_config,
)
from src.label_studio_coco_refinement.project import ProjectContractError


def test_fixed_config_contains_only_bbox_controls_and_canonical_coco80() -> None:
    config = build_label_config()
    root = ElementTree.fromstring(config)

    assert {element.tag for element in root.iter()} == {
        "View",
        "Image",
        "RectangleLabels",
        "Label",
    }
    assert root.find("Image").attrib["rotateControl"] == "false"
    rectangle = root.find("RectangleLabels")
    assert rectangle is not None
    assert rectangle.attrib == {
        "name": "bbox",
        "toName": "image",
        "canRotate": "false",
    }
    assert tuple(label.attrib["value"] for label in rectangle.findall("Label")) == (
        COCO80_REGISTRY.names
    )
    lowered = config.lower()
    for forbidden in ("<polygon", "<brush", "<mask", "crowd", "<textarea", "alias="):
        assert forbidden not in lowered


def test_label_config_and_fingerprint_are_deterministic() -> None:
    first = build_label_config()
    second = build_label_config()
    assert first == second
    assert label_config_fingerprint(first) == label_config_fingerprint(second)


@pytest.mark.parametrize(
    "mutated",
    [
        lambda config: config.replace('canRotate="false"', 'canRotate="true"'),
        lambda config: config.replace('rotateControl="false"', 'rotateControl="true"'),
        lambda config: config.replace("</View>", '<PolygonLabels name="p" toName="image"/></View>'),
        lambda config: config.replace('value="person"', 'value="human"'),
    ],
)
def test_label_config_rejects_rotation_unsupported_controls_and_noncanonical_names(mutated) -> None:
    with pytest.raises(ProjectContractError):
        validate_label_config(mutated(build_label_config()))
