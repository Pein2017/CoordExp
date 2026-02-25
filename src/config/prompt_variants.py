"""Central registry for dense prompt variants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

DEFAULT_PROMPT_VARIANT = "default"

COCO_80_CLASS_NAMES = (
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
        "COCO_80_CLASS_NAMES must contain exactly 80 classes "
        f"(got {len(COCO_80_CLASS_NAMES)})."
    )
if len(set(COCO_80_CLASS_NAMES)) != len(COCO_80_CLASS_NAMES):
    raise ValueError("COCO_80_CLASS_NAMES must not include duplicated class names.")

COCO_80_CLASS_LIST_COMPACT = ", ".join(COCO_80_CLASS_NAMES)


@dataclass(frozen=True)
class PromptVariant:
    key: str
    dense_system_suffix: str = ""
    dense_user_suffix: str = ""


PROMPT_VARIANT_REGISTRY: Mapping[str, PromptVariant] = {
    DEFAULT_PROMPT_VARIANT: PromptVariant(key=DEFAULT_PROMPT_VARIANT),
    "coco_80": PromptVariant(
        key="coco_80",
        dense_system_suffix=(
            "- COCO-80 closed-class policy: `desc` must be exactly one canonical class name from this list "
            "(case-sensitive): "
            f"{COCO_80_CLASS_LIST_COMPACT}.\n"
            "- Do not emit any class outside this list; if uncertain, choose the closest canonical class and keep details concise.\n"
            "- Coverage: locate each clearly visible object instance; when multiple instances of the same class exist, output multiple records (one per instance).\n"
            "- Atomic instances: each record must refer to exactly one object instance; each bbox_2d must tightly cover a single instance.\n"
            "- Do not output group boxes that cover multiple instances (e.g., a long thin strip over a crowd or shelf row).\n"
            "- Avoid duplicates: do not output multiple near-identical boxes for the same instance.\n"
            "- If you cannot localize a single instance, omit it.\n"
        ),
        dense_user_suffix=(
            " Restrict `desc` to this COCO-80 class list: "
            f"{COCO_80_CLASS_LIST_COMPACT}."
            " Locate each clearly visible object instance; output one record per instance."
            " Each record must correspond to exactly one object instance with an atomic bbox; do not use one box to cover multiple objects and do not repeat near-identical boxes."
        ),
    ),
}


def available_prompt_variant_keys() -> tuple[str, ...]:
    """Return registered prompt-variant keys in deterministic order."""
    return tuple(PROMPT_VARIANT_REGISTRY.keys())


def resolve_prompt_variant_key(prompt_variant: Optional[str] = None) -> str:
    """Resolve prompt variant key with strict validation."""
    if prompt_variant is None:
        return DEFAULT_PROMPT_VARIANT
    if not isinstance(prompt_variant, str):
        raise TypeError("prompt_variant must be a string when provided")

    key = prompt_variant.strip().lower()
    if not key:
        return DEFAULT_PROMPT_VARIANT

    if key not in PROMPT_VARIANT_REGISTRY:
        available = ", ".join(available_prompt_variant_keys())
        raise ValueError(
            f"Unknown prompt variant '{prompt_variant}'. Available variants: [{available}]"
        )
    return key


def resolve_prompt_variant(prompt_variant: Optional[str] = None) -> PromptVariant:
    """Resolve and return prompt-variant payload."""
    return PROMPT_VARIANT_REGISTRY[resolve_prompt_variant_key(prompt_variant)]


__all__ = [
    "COCO_80_CLASS_LIST_COMPACT",
    "COCO_80_CLASS_NAMES",
    "DEFAULT_PROMPT_VARIANT",
    "PROMPT_VARIANT_REGISTRY",
    "PromptVariant",
    "available_prompt_variant_keys",
    "resolve_prompt_variant",
    "resolve_prompt_variant_key",
]
