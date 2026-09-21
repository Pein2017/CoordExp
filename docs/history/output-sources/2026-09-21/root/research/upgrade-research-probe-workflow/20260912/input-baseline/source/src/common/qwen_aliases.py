"""Shared Qwen alias safety checks."""

from __future__ import annotations


INVALID_QWEN_WRAPPER_ALIASES = {
    "<|object_start|>": "<|object_ref_start|>",
    "<|object_end|>": "<|object_ref_end|>",
}


def find_invalid_qwen_wrapper_alias(text: str) -> tuple[str, str] | None:
    for alias, canonical in INVALID_QWEN_WRAPPER_ALIASES.items():
        if alias in text:
            return alias, canonical
    return None
