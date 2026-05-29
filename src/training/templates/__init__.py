"""Training template codec facades.

Exports are resolved lazily to keep import-only contract checks independent of
the detection stack and any optional tensor dependencies it may transitively use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.training.templates.codec import (
        DetectionTemplateCodec,
        DetectionTemplateRenderOptions,
    )
    from src.training.templates.compact_full import CompactFullEncodingCodec
    from src.training.templates.json_chat import Stage1JsonChatEncodingCodec

_LAZY_EXPORTS = {
    "CompactFullEncodingCodec": "src.training.templates.compact_full",
    "DetectionTemplateCodec": "src.training.templates.codec",
    "DetectionTemplateRenderOptions": "src.training.templates.codec",
    "Stage1JsonChatEncodingCodec": "src.training.templates.json_chat",
    "create_compact_full_codec": "src.training.templates.compact_full",
    "create_stage1_json_chat_codec": "src.training.templates.json_chat",
}


def __getattr__(name: str) -> Any:
    """Resolve facade exports on first access."""

    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value

__all__ = [
    "CompactFullEncodingCodec",
    "DetectionTemplateCodec",
    "DetectionTemplateRenderOptions",
    "Stage1JsonChatEncodingCodec",
    "create_compact_full_codec",
    "create_stage1_json_chat_codec",
]
