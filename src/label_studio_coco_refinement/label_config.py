"""Deterministic bbox-only Label Studio configuration for canonical COCO-80."""

from __future__ import annotations

from html import escape
from typing import Sequence
from xml.etree import ElementTree

from .project import Coco80RegistryProtocol, ProjectContractError, default_registry, fingerprint_json


_LABEL_COLORS: tuple[str, ...] = (
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#B279A2",
    "#FF9DA6",
    "#9D755D",
    "#BAB0AC",
    "#5F9ED1",
)


def build_label_config(registry: Coco80RegistryProtocol | None = None) -> str:
    """Return fixed static RectangleLabels XML with all rotation controls disabled."""

    registry = registry or default_registry()
    names = tuple(registry.names)
    _validate_names(names)
    labels = "\n".join(
        f'    <Label value="{escape(name, quote=True)}" '
        f'background="{_LABEL_COLORS[index % len(_LABEL_COLORS)]}"/>'
        for index, name in enumerate(names)
    )
    config = (
        '<View>\n'
        '  <Image name="image" value="$image" zoom="true" zoomControl="true" '
        'rotateControl="false"/>\n'
        '  <RectangleLabels name="bbox" toName="image" canRotate="false">\n'
        f"{labels}\n"
        '  </RectangleLabels>\n'
        '</View>\n'
    )
    validate_label_config(config, expected_names=names)
    return config


def label_config_fingerprint(config: str) -> str:
    """Fingerprint the exact XML bytes embedded in the project manifest."""

    validate_label_config(config)
    return fingerprint_json({"label_config_xml": config})


def validate_label_config(config: str, *, expected_names: Sequence[str] | None = None) -> None:
    """Fail closed on any non-bbox control, dynamic class, or rotation surface."""

    try:
        root = ElementTree.fromstring(config)
    except ElementTree.ParseError as exc:
        raise ProjectContractError("label config is not valid XML") from exc
    if root.tag != "View":
        raise ProjectContractError("label config root must be View")
    images = root.findall(".//Image")
    rectangles = root.findall(".//RectangleLabels")
    if len(images) != 1 or len(rectangles) != 1:
        raise ProjectContractError("label config must contain one Image and one RectangleLabels")
    image = images[0]
    rectangle = rectangles[0]
    if image.attrib.get("name") != "image" or image.attrib.get("value") != "$image":
        raise ProjectContractError("Image must use the fixed image/$image binding")
    if image.attrib.get("rotateControl") != "false":
        raise ProjectContractError("Image rotateControl must be false")
    if rectangle.attrib != {"name": "bbox", "toName": "image", "canRotate": "false"}:
        raise ProjectContractError("RectangleLabels must use fixed bbox/image binding and canRotate=false")
    allowed_tags = {"View", "Image", "RectangleLabels", "Label"}
    unsupported = sorted({element.tag for element in root.iter()} - allowed_tags)
    if unsupported:
        raise ProjectContractError("unsupported label-config controls: " + ", ".join(unsupported))
    labels = rectangle.findall("Label")
    if any(set(label.attrib) != {"value", "background"} for label in labels):
        raise ProjectContractError("labels may contain only canonical value and fixed background")
    names = tuple(label.attrib["value"] for label in labels)
    _validate_names(names)
    canonical_names = tuple(expected_names) if expected_names is not None else tuple(default_registry().names)
    if names != canonical_names:
        raise ProjectContractError("label config order/content differs from the canonical registry")


def _validate_names(names: Sequence[str]) -> None:
    if len(names) != 80 or len(set(names)) != 80:
        raise ProjectContractError("label config requires exactly 80 unique canonical names")
    if any(not isinstance(name, str) or not name or name != name.strip() for name in names):
        raise ProjectContractError("canonical names must be non-empty normalized strings")
