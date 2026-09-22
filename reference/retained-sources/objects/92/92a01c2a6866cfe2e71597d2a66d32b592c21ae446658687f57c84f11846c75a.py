"""Common Objects in Context 80-class category namespaces for detection eval.

The Swift evaluator historically writes a contiguous one-based category ID in
its private ``coco_gt.json`` and ``coco_predictions.json`` sidecars.  Those
values are evaluator-local identifiers, not the gapped category identifiers in
the official Common Objects in Context annotation files.  Both namespaces are
declared here so research artifacts cannot silently join them as though they
were interchangeable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
from types import MappingProxyType
from typing import Any, Mapping


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


COCO_80_EVALUATOR_LOCAL_CATEGORY_ID_BY_NAME: dict[str, int] = {
    normalize_coco_category_name(name): index + 1
    for index, name in enumerate(COCO_80_CLASS_NAMES)
}

# Official Common Objects in Context category identifiers are intentionally
# gapped.  The tuple is aligned one-to-one with ``COCO_80_CLASS_NAMES``.
COCO_80_OFFICIAL_CATEGORY_IDS: tuple[int, ...] = (
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    27,
    28,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    67,
    70,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
)

if len(COCO_80_OFFICIAL_CATEGORY_IDS) != len(COCO_80_CLASS_NAMES):
    raise ValueError(
        "COCO_80_OFFICIAL_CATEGORY_IDS must align one-to-one with "
        f"COCO_80_CLASS_NAMES, got {len(COCO_80_OFFICIAL_CATEGORY_IDS)} IDs"
    )
if len(set(COCO_80_OFFICIAL_CATEGORY_IDS)) != len(COCO_80_OFFICIAL_CATEGORY_IDS):
    raise ValueError("COCO_80_OFFICIAL_CATEGORY_IDS must not contain duplicates")

COCO_80_OFFICIAL_CATEGORY_ID_BY_NAME: Mapping[str, int] = MappingProxyType(
    {
        normalize_coco_category_name(name): category_id
        for name, category_id in zip(
            COCO_80_CLASS_NAMES,
            COCO_80_OFFICIAL_CATEGORY_IDS,
            strict=True,
        )
    }
)
COCO_80_OFFICIAL_CATEGORY_NAME_BY_ID: Mapping[int, str] = MappingProxyType(
    {
        category_id: normalize_coco_category_name(name)
        for name, category_id in zip(
            COCO_80_CLASS_NAMES,
            COCO_80_OFFICIAL_CATEGORY_IDS,
            strict=True,
        )
    }
)
COCO_80_EVALUATOR_LOCAL_CATEGORY_NAME_BY_ID: Mapping[int, str] = MappingProxyType(
    {
        category_id: name
        for name, category_id in COCO_80_EVALUATOR_LOCAL_CATEGORY_ID_BY_NAME.items()
    }
)


@dataclass(frozen=True)
class Coco80CategoryNamespaceEntry:
    """One category joined across evaluator-local and official namespaces."""

    normalized_category_name: str
    evaluator_category_id: int
    official_coco_category_id: int


COCO_80_CATEGORY_NAMESPACE_ENTRIES: tuple[Coco80CategoryNamespaceEntry, ...] = tuple(
    Coco80CategoryNamespaceEntry(
        normalized_category_name=normalize_coco_category_name(name),
        evaluator_category_id=index + 1,
        official_coco_category_id=official_category_id,
    )
    for index, (name, official_category_id) in enumerate(
        zip(COCO_80_CLASS_NAMES, COCO_80_OFFICIAL_CATEGORY_IDS, strict=True)
    )
)


def coco_80_category_namespace_payload() -> list[dict[str, Any]]:
    """Return the canonical JSON-ready three-column category registry."""

    return [asdict(entry) for entry in COCO_80_CATEGORY_NAMESPACE_ENTRIES]


def coco_80_category_namespace_sha256() -> str:
    """Return the stable SHA-256 fingerprint of the canonical registry."""

    payload = json.dumps(
        coco_80_category_namespace_payload(),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


COCO_80_CATEGORY_NAMESPACE_SHA256 = coco_80_category_namespace_sha256()

# Compatibility alias: the direct Swift evaluator's existing COCO sidecars use
# contiguous one-based evaluator-local IDs.  New code must prefer the explicit
# namespace name above.
COCO_80_CATEGORY_IDS = COCO_80_EVALUATOR_LOCAL_CATEGORY_ID_BY_NAME

if set(COCO_80_EVALUATOR_LOCAL_CATEGORY_ID_BY_NAME) != set(
    COCO_80_OFFICIAL_CATEGORY_ID_BY_NAME
):
    raise ValueError(
        "Evaluator-local and official category namespaces must share names"
    )
if set(COCO_80_EVALUATOR_LOCAL_CATEGORY_NAME_BY_ID.values()) != set(
    COCO_80_EVALUATOR_LOCAL_CATEGORY_ID_BY_NAME
):
    raise ValueError("Evaluator-local category mappings must be bijective")
if set(COCO_80_OFFICIAL_CATEGORY_NAME_BY_ID.values()) != set(
    COCO_80_OFFICIAL_CATEGORY_ID_BY_NAME
):
    raise ValueError("Official category mappings must be bijective")
