"""Unified English prompts for CoordExp general detection/grounding."""

from __future__ import annotations

from typing import Optional

from .prompt_variants import (
    DEFAULT_PROMPT_VARIANT,
    available_prompt_variant_keys,
    resolve_prompt_variant,
    resolve_prompt_variant_key,
)

# Shared prior rules (kept flat for easy embedding in system prompt)
PRIOR_RULES = "- Open-domain object detection/grounding on public datasets; cover all visible targets. If none, return an empty JSON {}.\n"

# Ordering instructions (shared across coord modes)
_ORDER_RULE_SORTED = (
    "- Order objects by (minY, minX): top-to-bottom then left-to-right. Index from 1.\n"
)
_ORDER_RULE_RANDOM = "- Object order is unrestricted; any ordering is acceptable. Index objects consecutively starting from 1.\n"

# Coord-token mode (model asked to emit <|coord_N|> tokens)
_SYSTEM_PREFIX_TOKENS = (
    'You are a general-purpose object detection and grounding assistant. Output exactly one JSON object like {"object_1":{...}} with no extra text.\n'
    "- Each object must have a plain English desc and exactly one geometry key (bbox_2d OR poly), never multiple geometries.\n"
    '- If uncertain, set desc="unknown" and give the reason succinctly.\n'
    "- Geometry formatting rules:\n"
    "  * bbox_2d is [x1, y1, x2, y2] with x1<=x2 and y1<=y2.\n"
    "  * poly is a single closed polygon as an ordered list of [x, y] vertices.\n"
    "    - Preserve adjacency: consecutive vertices are connected, and the last connects back to the first.\n"
    "    - Use a consistent vertex order: start from the top-most (then left-most) vertex, then go clockwise around the centroid.\n"
    "    - Do NOT sort poly points by x/y; that can create self-intersections.\n"
    "- Coordinates must be written as coord tokens `<|coord_N|>` only. Examples that tokenize correctly:\n"
    "  * bbox_2d: ['<|coord_12|>', '<|coord_34|>', '<|coord_256|>', '<|coord_480|>']\n"
    "  * poly: [['<|coord_10|>', '<|coord_20|>'], ['<|coord_200|>', '<|coord_20|>'], ['<|coord_200|>', '<|coord_220|>'], ['<|coord_10|>', '<|coord_220|>']]\n"
    "- Coordinates are integers in [0,999]; always keep the `<|coord_...|>` form so the tokenizer parses them as special tokens.\n"
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
    "(top-to-bottom then left-to-right). "
    "Return a single JSON object where each entry has desc plus one geometry (bbox_2d or poly) using `<|coord_N|>` tokens (0–999)."
)
USER_PROMPT_RANDOM_TOKENS = (
    "Detect and list every object in the image (any ordering is acceptable). "
    "Return a single JSON object where each entry has desc plus one geometry (bbox_2d or poly) using `<|coord_N|>` tokens (0–999)."
)

# Coord-token-only contract: numeric dense prompt variants are intentionally unsupported.

# Defaults (coord-token, sorted)
SYSTEM_PROMPT = SYSTEM_PROMPT_SORTED_TOKENS
USER_PROMPT = USER_PROMPT_SORTED_TOKENS

# Summary prompts remain unchanged
SYSTEM_PROMPT_SUMMARY = "You are an assistant that writes a concise English one-sentence summary of the image contents."
USER_PROMPT_SUMMARY = "Summarize the image in one short English sentence."


def build_dense_system_prompt(
    ordering: str = "sorted",
    coord_mode: str = "coord_tokens",
    prompt_variant: Optional[str] = None,
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

    variant = resolve_prompt_variant(prompt_variant)
    return f"{base_prompt}{variant.dense_system_suffix}"


def build_dense_user_prompt(
    ordering: str = "sorted",
    coord_mode: str = "coord_tokens",
    prompt_variant: Optional[str] = None,
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

    variant = resolve_prompt_variant(prompt_variant)
    return f"{base_prompt}{variant.dense_user_suffix}"


def get_template_prompts(
    ordering: str = "sorted",
    coord_mode: str = "coord_tokens",
    prompt_variant: Optional[str] = None,
) -> tuple[str, str]:
    """Return (system, user) prompts for dense mode with variant support."""
    return (
        build_dense_system_prompt(
            ordering=ordering,
            coord_mode=coord_mode,
            prompt_variant=prompt_variant,
        ),
        build_dense_user_prompt(
            ordering=ordering,
            coord_mode=coord_mode,
            prompt_variant=prompt_variant,
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
