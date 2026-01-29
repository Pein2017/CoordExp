from __future__ import annotations

import re
from typing import Any, Iterable, List, Sequence

COORD_TOKEN_PATTERN = re.compile(r"^<\|coord_(\d{1,4})\|>$")


def is_coord_token(value: Any) -> bool:
    return isinstance(value, str) and bool(COORD_TOKEN_PATTERN.match(value))


def sequence_has_coord_tokens(values: Sequence[Any]) -> bool:
    return any(is_coord_token(v) for v in values)


def value_in_coord_range(value: int) -> bool:
    return 0 <= value <= 999


def token_to_int(token: str) -> int:
    match = COORD_TOKEN_PATTERN.match(token)
    if not match:
        raise ValueError(f"Malformed coord token: {token!r}")
    value = int(match.group(1))
    if not value_in_coord_range(value):
        raise ValueError(f"Coord token value {value} out of range 0..999")
    return value


def int_to_token(value: int) -> str:
    if not value_in_coord_range(value):
        raise ValueError(f"Coord value {value} out of range 0..999")
    return f"<|coord_{int(value)}|>"


def _coerce_to_int(value: Any) -> int:
    if is_coord_token(value):
        return token_to_int(value)
    try:
        v_float = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Coordinate {value!r} is not numeric or coord token") from exc
    v_int = int(round(v_float))
    if abs(v_float - v_int) > 1e-6:
        raise ValueError(f"Coordinate {value!r} is not an integer value")
    if not value_in_coord_range(v_int):
        raise ValueError(f"Coordinate {v_int} out of range 0..999")
    return v_int


def tokens_to_ints(
    values: Sequence[Any], *, require_even: bool = True
) -> List[int]:
    ints: List[int] = [_coerce_to_int(v) for v in values]
    if require_even and len(ints) % 2 != 0:
        raise ValueError("Coordinate list length must be even (x,y pairs)")
    return ints


def ints_to_tokens(values: Iterable[int]) -> List[str]:
    return [int_to_token(int(v)) for v in values]


def normalized_from_ints(values: Sequence[int], scale: float = 999.0) -> List[float]:
    return [float(v) / float(scale) for v in values]


def get_coord_token_ids(tokenizer: Any, *, validate: bool = False) -> List[int]:
    """Return tokenizer ids for `<|coord_0|>` ... `<|coord_999|>`.

    If `validate=True`, fail fast when the tokenizer does not contain the full coord
    sub-vocabulary (e.g. missing tokens mapping to `unk_token_id` or a shared id).
    """
    tokens = ints_to_tokens(range(0, 1000))
    ids_raw = tokenizer.convert_tokens_to_ids(tokens)
    # HF tokenizers return a list for list input, but keep this defensive for custom tokenizers.
    if isinstance(ids_raw, int):
        ids = [int(ids_raw)]
    else:
        ids = [int(x) if x is not None else -1 for x in list(ids_raw)]

    if validate:
        if len(ids) != 1000:
            raise ValueError(
                f"coord token id lookup must return 1000 ids; got {len(ids)}. "
                "Check tokenizer.convert_tokens_to_ids implementation."
            )
        if any(int(i) < 0 for i in ids):
            bad = [j for j, i in enumerate(ids) if int(i) < 0][:5]
            raise ValueError(
                "Tokenizer is missing required coord tokens <|coord_0|>.. <|coord_999|> "
                f"(example missing indices: {bad}). Ensure vocab expansion/special tokens include coord tokens."
            )
        if len(set(ids)) != 1000:
            # Most commonly happens when coord tokens map to unk_token_id.
            raise ValueError(
                "Tokenizer does not map coord tokens to 1000 distinct ids. "
                "This will silently break coord-bin semantics; ensure coord tokens are added to the tokenizer vocab."
            )

    return ids


def build_coord_token_id_mask(tokenizer: Any, *, device=None):
    import torch

    ids = get_coord_token_ids(tokenizer, validate=False)
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is None:
        vocab_size = max(ids) + 1 if ids else 0
    mask = torch.zeros(int(vocab_size), dtype=torch.bool, device=device)
    for idx in ids:
        if int(idx) < 0:
            continue
        if int(idx) < mask.numel():
            mask[int(idx)] = True
    return mask
