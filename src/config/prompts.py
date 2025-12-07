"""Unified English prompts for CoordExp general detection/grounding."""

# Shared prior rules (kept flat for easy embedding in system prompt)
PRIOR_RULES = (
    "- Open-domain object detection/grounding on public datasets; cover all visible targets. If none, return an empty JSON {}.\n"
    '- For occluded/blurred/cropped targets, avoid guessing; use desc="unknown" with a short reason.\n'
    "- Use only visible evidence; never rely on external knowledge.\n"
    "- Nearby instances of the same class must be annotated separately; long/thin targets can use line geometry.\n"
)

# Ordering instructions (shared across coord modes)
_ORDER_RULE_SORTED = (
    "- Order objects top-to-bottom then left-to-right (lines: use the leftmost endpoint). Index from 1.\n"
)
_ORDER_RULE_RANDOM = (
    "- Object order is unrestricted; any ordering is acceptable. Index objects consecutively starting from 1.\n"
)

# Coord-token mode (model asked to emit <|coord_N|> tokens)
_SYSTEM_PREFIX_TOKENS = (
    'You are a general-purpose object detection and grounding assistant. Output exactly one JSON object like {"object_1":{...}} with no extra text.\n'
    "- Each object must have a plain English desc and exactly one geometry key (bbox_2d OR poly OR line), never multiple geometries.\n"
    "- If uncertain, set desc=\"unknown\" and give the reason succinctly.\n"
    "- Coordinates must be written as coord tokens `<|coord_N|>` only. Examples that tokenize correctly:\n"
    "  * bbox_2d: ['<|coord_12|>', '<|coord_34|>', '<|coord_256|>', '<|coord_480|>']\n"
    "  * poly: [['<|coord_10|>', '<|coord_20|>'], ['<|coord_200|>', '<|coord_20|>'], ['<|coord_200|>', '<|coord_220|>'], ['<|coord_10|>', '<|coord_220|>']]\n"
    "  * line: [['<|coord_100|>', '<|coord_300|>'], ['<|coord_300|>', '<|coord_320|>']]\n"
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
    "Detect and list every object in the image, ordered top-to-bottom then left-to-right. "
    "Return a single JSON object where each entry has desc plus one geometry (bbox_2d, poly, or line) using `<|coord_N|>` tokens (0–999)."
)
USER_PROMPT_RANDOM_TOKENS = (
    "Detect and list every object in the image (any ordering is acceptable). "
    "Return a single JSON object where each entry has desc plus one geometry (bbox_2d, poly, or line) using `<|coord_N|>` tokens (0–999)."
)

# Numeric mode (model asked to emit plain integers already normalized to 0–999)
_SYSTEM_PREFIX_NUMERIC = (
    'You are a general-purpose object detection and grounding assistant. Output exactly one JSON object like {"object_1":{...}} with no extra text.\n'
    "- Each object must have a plain English desc and exactly one geometry key (bbox_2d OR poly OR line), never multiple geometries.\n"
    "- If uncertain, set desc=\"unknown\" and give the reason succinctly.\n"
    "- Coordinates must be plain integers in [0,999] (already normalized from pixels); do not wrap them in coord tokens.\n"
    "- JSON layout: single line; one space after colons and commas; double quotes for keys/strings; no trailing text.\n"
    "Prior rules:\n" + PRIOR_RULES
)
SYSTEM_PROMPT_SORTED_NUMERIC = _SYSTEM_PREFIX_NUMERIC.replace(
    "Prior rules:\n", _ORDER_RULE_SORTED + "Prior rules:\n"
)
SYSTEM_PROMPT_RANDOM_NUMERIC = _SYSTEM_PREFIX_NUMERIC.replace(
    "Prior rules:\n", _ORDER_RULE_RANDOM + "Prior rules:\n"
)
USER_PROMPT_SORTED_NUMERIC = (
    "Detect and list every object in the image, ordered top-to-bottom then left-to-right. "
    "Return a single JSON object where each entry has desc plus one geometry (bbox_2d, poly, or line) using plain integer coordinates 0–999 (no coord tokens)."
)
USER_PROMPT_RANDOM_NUMERIC = (
    "Detect and list every object in the image (any ordering is acceptable). "
    "Return a single JSON object where each entry has desc plus one geometry (bbox_2d, poly, or line) using plain integer coordinates 0–999 (no coord tokens)."
)

# Defaults (coord-token, sorted) kept for backward compatibility
SYSTEM_PROMPT = SYSTEM_PROMPT_SORTED_TOKENS
USER_PROMPT = USER_PROMPT_SORTED_TOKENS
USER_PROMPT_JSON = USER_PROMPT  # legacy alias

# Summary prompts remain unchanged
SYSTEM_PROMPT_SUMMARY = (
    "You are an assistant that writes a concise English one-sentence summary of the image contents."
)
USER_PROMPT_SUMMARY = "Summarize the image in one short English sentence."


def build_dense_system_prompt(ordering: str = "sorted", coord_mode: str = "coord_tokens") -> str:
    """Return system prompt for dense mode given ordering and coordinate representation."""
    ordering_key = "random" if str(ordering).lower() == "random" else "sorted"
    if str(coord_mode).lower() == "numeric":
        return SYSTEM_PROMPT_RANDOM_NUMERIC if ordering_key == "random" else SYSTEM_PROMPT_SORTED_NUMERIC
    return SYSTEM_PROMPT_RANDOM_TOKENS if ordering_key == "random" else SYSTEM_PROMPT_SORTED_TOKENS


def build_dense_user_prompt(ordering: str = "sorted", coord_mode: str = "coord_tokens") -> str:
    """Return user prompt for dense mode given ordering and coordinate representation."""
    ordering_key = "random" if str(ordering).lower() == "random" else "sorted"
    if str(coord_mode).lower() == "numeric":
        return USER_PROMPT_RANDOM_NUMERIC if ordering_key == "random" else USER_PROMPT_SORTED_NUMERIC
    return USER_PROMPT_RANDOM_TOKENS if ordering_key == "random" else USER_PROMPT_SORTED_TOKENS


def get_template_prompts(ordering: str = "sorted", coord_mode: str = "coord_tokens") -> tuple[str, str]:
    """Return (system, user) prompts for dense mode; coord_mode in {'coord_tokens','numeric'}."""
    return (
        build_dense_system_prompt(ordering=ordering, coord_mode=coord_mode),
        build_dense_user_prompt(ordering=ordering, coord_mode=coord_mode),
    )


__all__ = [
    # primary helpers
    "build_dense_system_prompt",
    "build_dense_user_prompt",
    "get_template_prompts",
    # coord-token variants
    "SYSTEM_PROMPT",
    "USER_PROMPT",
    "SYSTEM_PROMPT_SORTED_TOKENS",
    "SYSTEM_PROMPT_RANDOM_TOKENS",
    "USER_PROMPT_SORTED_TOKENS",
    "USER_PROMPT_RANDOM_TOKENS",
    # numeric variants
    "SYSTEM_PROMPT_SORTED_NUMERIC",
    "SYSTEM_PROMPT_RANDOM_NUMERIC",
    "USER_PROMPT_SORTED_NUMERIC",
    "USER_PROMPT_RANDOM_NUMERIC",
    # summary / legacy
    "SYSTEM_PROMPT_SUMMARY",
    "USER_PROMPT_SUMMARY",
    "USER_PROMPT_JSON",
]
