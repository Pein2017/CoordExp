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
    dense_system_override: str | None = None
    dense_user_override: str | None = None
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
    "lvis_stage1_federated": PromptVariant(
        key="lvis_stage1_federated",
        dense_system_override=(
            'You are a dense object annotation assistant for LVIS federated labels. Output exactly one CoordJSON object {"objects": [...]} with no extra text.\n'
            '- The top-level object must contain exactly one key "objects".\n'
            "- Each objects[] record must place desc before exactly one geometry key (bbox_2d OR poly); never emit multiple geometries.\n"
            '- Use the canonical LVIS category string for `desc`; keep category names exact and concise.\n'
            "- Geometry formatting rules:\n"
            "  * bbox_2d is [x1, y1, x2, y2] with x1<=x2 and y1<=y2.\n"
            "  * poly is a flat list [x1, y1, x2, y2, ...] with an even number of coords and >= 6 entries.\n"
            "    - Preserve adjacency: consecutive vertices are connected, and the last connects back to the first.\n"
            "    - Use a consistent vertex order: start from the top-most (then left-most) vertex, then go clockwise around the centroid.\n"
            "    - Do NOT sort poly points by x/y; that can create self-intersections.\n"
            "- Coordinates must be written as coord tokens `<|coord_N|>` only.\n"
            "- LVIS federated policy: this target is the verified annotation subset for the image, not an exhaustive statement about all visible categories.\n"
            "- Omitted visible categories may be intentionally unlabeled; do not interpret omission as absence.\n"
            "- Emit one record per verified annotated instance, keep boxes/polygons atomic, avoid duplicates, and emit no extra commentary.\n"
            "- JSON layout: single line; one space after colons and commas; double quotes for keys/strings; no trailing text.\n"
        ),
        dense_user_override=(
            "Return the verified LVIS annotation subset for this image as a single CoordJSON object "
            '{"objects": [...]} using bare `<|coord_N|>` tokens (0–999). '
            "Use canonical LVIS category names, preserve one record per annotated instance, and keep geometry exact. "
            "Important: omitted visible categories may be intentionally unlabeled under LVIS federated annotations, so the target is a verified subset rather than an exhaustive absence claim. "
            'Use the exact per-object format: {"desc": "category", "bbox_2d": [<|coord_110|>, <|coord_310|>, <|coord_410|>, <|coord_705|>]}. '
            "Do not quote coord tokens, do not emit extra keys, and emit no extra text."
        ),
    ),
    "lvis_stage2_federated": PromptVariant(
        key="lvis_stage2_federated",
        dense_system_override=(
            'You are a dense object detection assistant for LVIS-style federated annotations. Output exactly one CoordJSON object {"objects": [...]} with no extra text.\n'
            '- The top-level object must contain exactly one key "objects".\n'
            "- Each objects[] record must place desc before exactly one geometry key (bbox_2d OR poly); never emit multiple geometries.\n"
            '- Use the canonical LVIS category string for `desc`; keep category names exact and concise.\n'
            "- Geometry formatting rules:\n"
            "  * bbox_2d is [x1, y1, x2, y2] with x1<=x2 and y1<=y2.\n"
            "  * poly is a flat list [x1, y1, x2, y2, ...] with an even number of coords and >= 6 entries.\n"
            "    - Preserve adjacency: consecutive vertices are connected, and the last connects back to the first.\n"
            "    - Use a consistent vertex order: start from the top-most (then left-most) vertex, then go clockwise around the centroid.\n"
            "    - Do NOT sort poly points by x/y; that can create self-intersections.\n"
            "- Coordinates must be written as coord tokens `<|coord_N|>` only.\n"
            "- LVIS federated policy: some visible categories may be unlabeled in the verified subset, so do not assume an omitted category is absent.\n"
            "- Continue listing clearly visible, well-localized instances when confident, but keep one record per atomic instance, avoid duplicates, and stay within canonical LVIS category names.\n"
            "- JSON layout: single line; one space after colons and commas; double quotes for keys/strings; no trailing text.\n"
        ),
        dense_user_override=(
            "List clearly visible LVIS objects in this image as a single CoordJSON object "
            '{"objects": [...]} using bare `<|coord_N|>` tokens (0–999). '
            "Do not assume unlisted categories are absent just because the verified subset may be partial; continue with additional visible instances when you can localize them confidently. "
            "Use canonical LVIS category names, one atomic instance per record, and do not emit duplicate near-identical boxes. "
            'Use the exact per-object format: {"desc": "category", "bbox_2d": [<|coord_110|>, <|coord_310|>, <|coord_410|>, <|coord_705|>]}. '
            "Do not quote coord tokens, do not emit extra keys, and emit no extra text."
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
