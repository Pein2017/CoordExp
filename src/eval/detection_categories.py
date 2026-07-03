"""Canonical COCO-80 category registry for CoordExp-swift detection eval."""

from __future__ import annotations

import re


COCO_80_CLASS_NAMES: tuple[str, ...] = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)

if len(COCO_80_CLASS_NAMES) != 80:
    raise ValueError(
        f"COCO_80_CLASS_NAMES must contain 80 classes, got {len(COCO_80_CLASS_NAMES)}"
    )
if len(set(COCO_80_CLASS_NAMES)) != len(COCO_80_CLASS_NAMES):
    raise ValueError("COCO_80_CLASS_NAMES must not contain duplicate names")


def normalize_coco_category_name(value: object) -> str:
    """Normalize category text for closed COCO-80 matching."""

    return re.sub(r"\s+", " ", str(value or "").strip().lower())


COCO_80_CATEGORY_IDS: dict[str, int] = {
    normalize_coco_category_name(name): index + 1
    for index, name in enumerate(COCO_80_CLASS_NAMES)
}
