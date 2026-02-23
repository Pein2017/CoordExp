from __future__ import annotations

from typing import Optional

from src.common.object_field_order import normalize_object_field_order

from .prompt_variants import (
    DEFAULT_PROMPT_VARIANT,
    available_prompt_variant_keys,
    resolve_prompt_variant,
    resolve_prompt_variant_key,
)

# Shared prior rules (kept flat for easy embedding in system prompt)
PRIOR_RULES = '- Open-domain object detection/grounding on public datasets; cover all visible targets. If none, return {"objects": []}.\n'

_USER_EXAMPLE_DESC_FIRST = (
    '{"desc": "black cat", "bbox_2d": [<|coord_110|>, <|coord_310|>, <|coord_410|>, <|coord_705|]}'
)
_USER_EXAMPLE_GEOMETRY_FIRST = (
    '{"bbox_2d": [<|coord_110|>, <|coord_310|>, <|coord_410|>, <|coord_705|>], "desc": "black cat near sofa"}'
)

# Ordering instructions (shared across coord modes)
_ORDER_RULE_SORTED = (
    "- Order objects by anchor (y, x): top-to-bottom then left-to-right.\n"
    "  For bbox_2d, anchor is (y1, x1) from [x1, y1, x2, y2].\n"
    "  For poly, anchor is (minY, minX) over polygon vertices.\n"
)
_ORDER_RULE_RANDOM = (
    "- Object order is unrestricted; any ordering is acceptable.\n"
)

# Coord-token mode (model asked to emit <|coord_N|> tokens)
_SYSTEM_PREFIX_TOKENS = (
    'You are a general-purpose object detection and grounding assistant. Output exactly one CoordJSON object {"objects": [...]} with no extra text.\n'
    '- The top-level object must contain exactly one key "objects".\n'
    "- Each objects[] record must place desc before exactly one geometry key (bbox_2d OR poly); never emit multiple geometries.\n"
    '- If uncertain, set desc="unknown" and give the reason succinctly.\n'
    "- Geometry formatting rules:\n"
    "  * bbox_2d is [x1, y1, x2, y2] with x1<=x2 and y1<=y2.\n"
    "  * poly is a flat list [x1, y1, x2, y2, ...] with an even number of coords and >= 6 entries.\n"
    "    - Preserve adjacency: consecutive vertices are connected, and the last connects back to the first.\n"
    "    - Use a consistent vertex order: start from the top-most (then left-most) vertex, then go clockwise around the centroid.\n"
    "    - Do NOT sort poly points by x/y; that can create self-intersections.\n"
    "- Coordinates must be written as coord tokens `<|coord_N|>` only. Examples that tokenize correctly:\n"
    "  * bbox_2d: [<|coord_12|>, <|coord_34|>, <|coord_256|>, <|coord_480|>]\n"
    "  * poly: [<|coord_10|>, <|coord_20|>, <|coord_200|>, <|coord_20|>, <|coord_200|>, <|coord_220|>, <|coord_10|>, <|coord_220|>]\n"
    "- Coordinates are integers in [0,999]; always keep bare `<|coord_...|>` literals (no quotes) in geometry arrays.\n"
    "- JSON layout: single line; one space after colons and commas; double quotes for keys/strings; no trailing text.\n"
    "Prior rules:\n" + PRIOR_RULES
)
SYSTEM_PROMPT_SORTED_TOKENS = _SYSTEM_PREFIX_TOKENS.replace(
    "Prior rules:\n", _ORDER_RULE_SORTED + "Prior rules:\n"
)
SYSTEM_PROMPT_RANDOM_TOKENS = _SYSTEM_PREFIX_TOKENS.replace(
    "Prior rules:\n", _ORDER_RULE_RANDOM + "Prior rules:\n"
)
USER_PROMPT_SORTED_TOKENS = (
    "Detect and list every object in the image, ordered by (minY, minX) "
    "(top-to-bottom then left-to-right). For bbox_2d anchors, use (y1, x1) from [x1, y1, x2, y2]. "
    "For poly anchors, use (minY, minX) over all vertices. "
    "Return a single CoordJSON object {\"objects\": [...]} where each record has desc before one geometry (bbox_2d or poly) using bare `<|coord_N|>` tokens (0–999). "
    "Use the exact per-object format: "
    f"{_USER_EXAMPLE_DESC_FIRST}. "
    "Do not quote coord tokens, do not emit extra keys, and emit no extra text."
)
USER_PROMPT_RANDOM_TOKENS = (
    "Detect and list every object in the image (any ordering is acceptable). "
    "Return a single CoordJSON object {\"objects\": [...]} where each record has desc before one geometry (bbox_2d or poly) using bare `<|coord_N|>` tokens (0–999). "
    "Use the exact per-object format: "
    f"{_USER_EXAMPLE_DESC_FIRST}. "
    "Do not quote coord tokens, do not emit extra keys, and emit no extra text."
)

# Coord-token-only contract: numeric dense prompt variants are intentionally unsupported.

# Defaults (coord-token, sorted)
SYSTEM_PROMPT = SYSTEM_PROMPT_SORTED_TOKENS
USER_PROMPT = USER_PROMPT_SORTED_TOKENS

# Summary prompts remain unchanged
SYSTEM_PROMPT_SUMMARY = "You are an assistant that writes a concise English one-sentence summary of the image contents."
USER_PROMPT_SUMMARY = "Summarize the image in one short English sentence."


def _apply_geometry_first_system_wording(base_prompt: str) -> str:
    return base_prompt.replace(
        "Each objects[] record must place desc before exactly one geometry key (bbox_2d OR poly); never emit multiple geometries.",
        "Each objects[] record must place exactly one geometry key (bbox_2d OR poly) before desc; never emit multiple geometries.",
    )


def _apply_geometry_first_user_wording(base_prompt: str) -> str:
    return base_prompt.replace(
        "has desc before one geometry (bbox_2d or poly)",
        "has one geometry (bbox_2d or poly) before desc",
    ).replace(
        f"Use the exact per-object format: {_USER_EXAMPLE_DESC_FIRST}.",
        f"Use the exact per-object format: {_USER_EXAMPLE_GEOMETRY_FIRST}.",
    )


def build_dense_system_prompt(
    ordering: str = "sorted",
    coord_mode: str = "coord_tokens",
    prompt_variant: Optional[str] = None,
    object_field_order: str = "desc_first",
) -> str:
    """Return system prompt for dense mode with coord-tokens-only contract."""
    coord_mode_key = str(coord_mode).lower()
    if coord_mode_key != "coord_tokens":
        raise ValueError(
            "Coord-token-only contract: coord_mode must be 'coord_tokens'."
        )

    ordering_key = "random" if str(ordering).lower() == "random" else "sorted"
    base_prompt = (
        SYSTEM_PROMPT_RANDOM_TOKENS
        if ordering_key == "random"
        else SYSTEM_PROMPT_SORTED_TOKENS
    )

    field_order = normalize_object_field_order(
        object_field_order, path="custom.object_field_order"
    )
    if field_order == "geometry_first":
        base_prompt = _apply_geometry_first_system_wording(base_prompt)

    variant = resolve_prompt_variant(prompt_variant)
    return f"{base_prompt}{variant.dense_system_suffix}"


def build_dense_user_prompt(
    ordering: str = "sorted",
    coord_mode: str = "coord_tokens",
    prompt_variant: Optional[str] = None,
    object_field_order: str = "desc_first",
) -> str:
    """Return user prompt for dense mode with coord-tokens-only contract."""
    coord_mode_key = str(coord_mode).lower()
    if coord_mode_key != "coord_tokens":
        raise ValueError(
            "Coord-token-only contract: coord_mode must be 'coord_tokens'."
        )

    ordering_key = "random" if str(ordering).lower() == "random" else "sorted"
    base_prompt = (
        USER_PROMPT_RANDOM_TOKENS
        if ordering_key == "random"
        else USER_PROMPT_SORTED_TOKENS
    )

    field_order = normalize_object_field_order(
        object_field_order, path="custom.object_field_order"
    )
    if field_order == "geometry_first":
        base_prompt = _apply_geometry_first_user_wording(base_prompt)

    variant = resolve_prompt_variant(prompt_variant)
    return f"{base_prompt}{variant.dense_user_suffix}"


def get_template_prompts(
    ordering: str = "sorted",
    coord_mode: str = "coord_tokens",
    prompt_variant: Optional[str] = None,
    object_field_order: str = "desc_first",
) -> tuple[str, str]:
    """Return (system, user) prompts for dense mode with variant support."""
    return (
        build_dense_system_prompt(
            ordering=ordering,
            coord_mode=coord_mode,
            prompt_variant=prompt_variant,
            object_field_order=object_field_order,
        ),
        build_dense_user_prompt(
            ordering=ordering,
            coord_mode=coord_mode,
            prompt_variant=prompt_variant,
            object_field_order=object_field_order,
        ),
    )


def resolve_dense_prompt_variant_key(prompt_variant: Optional[str] = None) -> str:
    """Resolve dense prompt variant with default fallback and strict validation."""
    return resolve_prompt_variant_key(prompt_variant)


__all__ = [
    # primary helpers
    "build_dense_system_prompt",
    "build_dense_user_prompt",
    "get_template_prompts",
    "resolve_dense_prompt_variant_key",
    "available_prompt_variant_keys",
    # prompt variant defaults
    "DEFAULT_PROMPT_VARIANT",
    # coord-token variants
    "SYSTEM_PROMPT",
    "USER_PROMPT",
    "SYSTEM_PROMPT_SORTED_TOKENS",
    "SYSTEM_PROMPT_RANDOM_TOKENS",
    "USER_PROMPT_SORTED_TOKENS",
    "USER_PROMPT_RANDOM_TOKENS",
    # summary
    "SYSTEM_PROMPT_SUMMARY",
    "USER_PROMPT_SUMMARY",
]
