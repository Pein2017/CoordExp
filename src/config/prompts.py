from __future__ import annotations

import hashlib
import json
from typing import Optional

from src.common.geometry.bbox_parameterization import normalize_bbox_format
from src.common.object_field_order import normalize_object_field_order

from .prompt_variants import (
    DEFAULT_PROMPT_VARIANT,
    available_prompt_variant_keys,
    resolve_prompt_variant,
    resolve_prompt_variant_key,
)

# Shared prior rules (kept flat for easy embedding in system prompt)
PRIOR_RULES = '- Open-domain object detection/grounding on public datasets; cover all visible targets. If none, return {"objects": []}.\n'

_USER_EXAMPLE_DESC_FIRST_XYXY = (
    '{"desc": "black cat", "bbox_2d": [<|coord_110|>, <|coord_310|>, <|coord_410|>, <|coord_705|>]}'
)
_USER_EXAMPLE_GEOMETRY_FIRST_XYXY = (
    '{"bbox_2d": [<|coord_110|>, <|coord_310|>, <|coord_410|>, <|coord_705|>], "desc": "black cat near sofa"}'
)
_USER_EXAMPLE_DESC_FIRST_CENTER_LOG_SIZE = (
    '{"desc": "black cat", "bbox_2d": [<|coord_260|>, <|coord_507|>, <|coord_408|>, <|coord_612|>]}'
)
_USER_EXAMPLE_GEOMETRY_FIRST_CENTER_LOG_SIZE = (
    '{"bbox_2d": [<|coord_260|>, <|coord_507|>, <|coord_408|>, <|coord_612|>], "desc": "black cat near sofa"}'
)

_BBOX_SYSTEM_RULE_PLACEHOLDER = "__BBOX_SYSTEM_RULE__"
_BBOX_USER_RULE_PLACEHOLDER = "__BBOX_USER_RULE__"
_OBJECT_FIELD_ORDER_SYSTEM_RULE_PLACEHOLDER = "__OBJECT_FIELD_ORDER_SYSTEM_RULE__"
_OBJECT_FIELD_ORDER_USER_RULE_PLACEHOLDER = "__OBJECT_FIELD_ORDER_USER_RULE__"
_USER_EXAMPLE_PLACEHOLDER = "__USER_EXAMPLE__"

# Ordering instructions (shared across coord modes)
_ORDER_RULE_SORTED = (
    "- Order objects by image-space top-left anchor (y1, x1): top-to-bottom then left-to-right.\n"
    "  For poly, anchor is (minY, minX) over polygon vertices.\n"
)
_ORDER_RULE_RANDOM = (
    "- Object order is unrestricted; any ordering is acceptable.\n"
)

# Coord-token mode (model asked to emit <|coord_N|> tokens)
_SYSTEM_PREFIX_TOKENS = (
    'You are a general-purpose object detection and grounding assistant. Output exactly one CoordJSON object {"objects": [...]} with no extra text.\n'
    '- The top-level object must contain exactly one key "objects".\n'
    f"- {_OBJECT_FIELD_ORDER_SYSTEM_RULE_PLACEHOLDER}\n"
    '- If uncertain, set desc="unknown" and give the reason succinctly.\n'
    "- Geometry formatting rules:\n"
    f"  * {_BBOX_SYSTEM_RULE_PLACEHOLDER}\n"
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
    "Detect and list every object in the image, ordered by image-space top-left "
    "(y1, x1) top-to-bottom then left-to-right. For poly anchors, use (minY, minX) over all vertices. "
    "Return a single CoordJSON object {\"objects\": [...]} where each record "
    f"{_OBJECT_FIELD_ORDER_USER_RULE_PLACEHOLDER} using bare `<|coord_N|>` tokens (0–999). "
    f"{_BBOX_USER_RULE_PLACEHOLDER} "
    "Use the exact per-object format: "
    f"{_USER_EXAMPLE_PLACEHOLDER}. "
    "Do not quote coord tokens, do not emit extra keys, and emit no extra text."
)
USER_PROMPT_RANDOM_TOKENS = (
    "Detect and list every object in the image (any ordering is acceptable). "
    "Return a single CoordJSON object {\"objects\": [...]} where each record "
    f"{_OBJECT_FIELD_ORDER_USER_RULE_PLACEHOLDER} using bare `<|coord_N|>` tokens (0–999). "
    f"{_BBOX_USER_RULE_PLACEHOLDER} "
    "Use the exact per-object format: "
    f"{_USER_EXAMPLE_PLACEHOLDER}. "
    "Do not quote coord tokens, do not emit extra keys, and emit no extra text."
)

# Coord-token-only contract: numeric dense prompt variants are intentionally unsupported.

# Defaults (coord-token, sorted)
SYSTEM_PROMPT = SYSTEM_PROMPT_SORTED_TOKENS
USER_PROMPT = USER_PROMPT_SORTED_TOKENS

# Summary prompts remain unchanged
SYSTEM_PROMPT_SUMMARY = "You are an assistant that writes a concise English one-sentence summary of the image contents."
USER_PROMPT_SUMMARY = "Summarize the image in one short English sentence."


def _prompt_fragments(*, bbox_format: str, object_field_order: str) -> dict[str, str]:
    bbox_format_key = normalize_bbox_format(bbox_format, path="bbox_format")
    field_order = normalize_object_field_order(
        object_field_order,
        path="custom.object_field_order",
    )
    if field_order == "geometry_first":
        field_order_system_rule = (
            "Each objects[] record must place exactly one geometry key (bbox_2d OR poly) before desc; never emit multiple geometries."
        )
        field_order_user_rule = "has one geometry (bbox_2d or poly) before desc"
        user_example = _USER_EXAMPLE_GEOMETRY_FIRST_CENTER_LOG_SIZE
    else:
        field_order_system_rule = (
            "Each objects[] record must place desc before exactly one geometry key (bbox_2d OR poly); never emit multiple geometries."
        )
        field_order_user_rule = "has desc before one geometry (bbox_2d or poly)"
        user_example = _USER_EXAMPLE_DESC_FIRST_CENTER_LOG_SIZE
    if bbox_format_key == "center_log_size":
        return {
            _BBOX_SYSTEM_RULE_PLACEHOLDER: (
                "bbox_2d is [cx, cy, u(w), u(h)] where u(s) = "
                "(log(max(s, 1/1024)) - log(1/1024)) / -log(1/1024); "
                "cx/cy are normalized box centers, and u(w)/u(h) are normalized log-size slots."
            ),
            _BBOX_USER_RULE_PLACEHOLDER: (
                "Use bbox_2d as [cx, cy, u(w), u(h)] where "
                "u(s) = (log(max(s, 1/1024)) - log(1/1024)) / -log(1/1024)."
            ),
            _OBJECT_FIELD_ORDER_SYSTEM_RULE_PLACEHOLDER: field_order_system_rule,
            _OBJECT_FIELD_ORDER_USER_RULE_PLACEHOLDER: field_order_user_rule,
            _USER_EXAMPLE_PLACEHOLDER: user_example,
        }
    if field_order == "geometry_first":
        user_example = _USER_EXAMPLE_GEOMETRY_FIRST_XYXY
    else:
        user_example = _USER_EXAMPLE_DESC_FIRST_XYXY
    return {
        _BBOX_SYSTEM_RULE_PLACEHOLDER: "bbox_2d is [x1, y1, x2, y2] with x1<=x2 and y1<=y2.",
        _BBOX_USER_RULE_PLACEHOLDER: "Use bbox_2d as [x1, y1, x2, y2].",
        _OBJECT_FIELD_ORDER_SYSTEM_RULE_PLACEHOLDER: field_order_system_rule,
        _OBJECT_FIELD_ORDER_USER_RULE_PLACEHOLDER: field_order_user_rule,
        _USER_EXAMPLE_PLACEHOLDER: user_example,
    }


def _render_prompt_text(
    template: str,
    *,
    bbox_format: str,
    object_field_order: str,
    variant_key: str,
    field_name: str,
    required_placeholders: tuple[str, ...] = (),
) -> str:
    rendered = str(template)
    missing = [p for p in required_placeholders if p not in rendered]
    if missing:
        raise ValueError(
            f"Prompt variant '{variant_key}' {field_name} must contain placeholders {missing}."
        )
    for placeholder, replacement in _prompt_fragments(
        bbox_format=bbox_format,
        object_field_order=object_field_order,
    ).items():
        rendered = rendered.replace(placeholder, replacement)
    return rendered


def build_dense_system_prompt(
    ordering: str = "sorted",
    coord_mode: str = "coord_tokens",
    prompt_variant: Optional[str] = None,
    object_field_order: str = "desc_first",
    bbox_format: str = "xyxy",
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
    variant = resolve_prompt_variant(prompt_variant)
    if variant.dense_system_override is not None:
        return _render_prompt_text(
            variant.dense_system_override,
            bbox_format=bbox_format,
            object_field_order=field_order,
            variant_key=variant.key,
            field_name="dense_system_override",
            required_placeholders=(
                _BBOX_SYSTEM_RULE_PLACEHOLDER,
                _OBJECT_FIELD_ORDER_SYSTEM_RULE_PLACEHOLDER,
            ),
        )
    return _render_prompt_text(
        f"{base_prompt}{variant.dense_system_suffix}",
        bbox_format=bbox_format,
        object_field_order=field_order,
        variant_key=variant.key,
        field_name="dense_system_prompt",
    )


def build_dense_user_prompt(
    ordering: str = "sorted",
    coord_mode: str = "coord_tokens",
    prompt_variant: Optional[str] = None,
    object_field_order: str = "desc_first",
    bbox_format: str = "xyxy",
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
    variant = resolve_prompt_variant(prompt_variant)
    if variant.dense_user_override is not None:
        return _render_prompt_text(
            str(variant.dense_user_override),
            bbox_format=bbox_format,
            object_field_order=field_order,
            variant_key=variant.key,
            field_name="dense_user_override",
            required_placeholders=(
                _BBOX_USER_RULE_PLACEHOLDER,
                _OBJECT_FIELD_ORDER_USER_RULE_PLACEHOLDER,
                _USER_EXAMPLE_PLACEHOLDER,
            ),
        )
    return _render_prompt_text(
        f"{base_prompt}{variant.dense_user_suffix}",
        bbox_format=bbox_format,
        object_field_order=field_order,
        variant_key=variant.key,
        field_name="dense_user_prompt",
    )


def get_template_prompts(
    ordering: str = "sorted",
    coord_mode: str = "coord_tokens",
    prompt_variant: Optional[str] = None,
    object_field_order: str = "desc_first",
    bbox_format: str = "xyxy",
) -> tuple[str, str]:
    """Return (system, user) prompts for dense mode with variant support."""
    return (
        build_dense_system_prompt(
            ordering=ordering,
            coord_mode=coord_mode,
            prompt_variant=prompt_variant,
            object_field_order=object_field_order,
            bbox_format=bbox_format,
        ),
        build_dense_user_prompt(
            ordering=ordering,
            coord_mode=coord_mode,
            prompt_variant=prompt_variant,
            object_field_order=object_field_order,
            bbox_format=bbox_format,
        ),
    )


def get_template_prompt_hash(
    *,
    ordering: str = "sorted",
    coord_mode: str = "coord_tokens",
    prompt_variant: Optional[str] = None,
    object_field_order: str = "desc_first",
    bbox_format: str = "xyxy",
) -> str:
    system_prompt, user_prompt = get_template_prompts(
        ordering=ordering,
        coord_mode=coord_mode,
        prompt_variant=prompt_variant,
        object_field_order=object_field_order,
        bbox_format=bbox_format,
    )
    payload = {
        "ordering": ordering,
        "coord_mode": coord_mode,
        "prompt_variant": resolve_prompt_variant_key(prompt_variant),
        "object_field_order": normalize_object_field_order(
            object_field_order, path="custom.object_field_order"
        ),
        "bbox_format": normalize_bbox_format(bbox_format, path="bbox_format"),
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "do_resize": False,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def resolve_dense_prompt_variant_key(prompt_variant: Optional[str] = None) -> str:
    """Resolve dense prompt variant with default fallback and strict validation."""
    return resolve_prompt_variant_key(prompt_variant)


__all__ = [
    # primary helpers
    "build_dense_system_prompt",
    "build_dense_user_prompt",
    "get_template_prompts",
    "get_template_prompt_hash",
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
