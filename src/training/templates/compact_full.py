"""Compact-full detection encoding codec."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.training.templates.codec import (
    DetectionTemplateCodec,
    DetectionTemplateRenderOptions,
)

if TYPE_CHECKING:
    from src.detection.template import DetectionSequenceTemplate
else:
    DetectionSequenceTemplate = Any


def _compact_full_template() -> "DetectionSequenceTemplate":
    """Return the compact-full detection template owner."""

    from src.detection.template import get_detection_template

    return get_detection_template("compact_full")


@dataclass(frozen=True, slots=True)
class CompactFullEncodingCodec(DetectionTemplateCodec):
    """Codec wrapping the existing ``compact_full`` detection template."""

    template: "DetectionSequenceTemplate" = field(default_factory=_compact_full_template)
    options: DetectionTemplateRenderOptions = field(
        default_factory=DetectionTemplateRenderOptions
    )


def create_compact_full_codec(
    *,
    options: DetectionTemplateRenderOptions | None = None,
) -> CompactFullEncodingCodec:
    """Return a compact-full codec with optional render/tokenization options."""

    from src.detection.template import get_detection_template

    return CompactFullEncodingCodec(
        template=get_detection_template("compact_full"),
        options=options or DetectionTemplateRenderOptions(),
    )


__all__ = [
    "CompactFullEncodingCodec",
    "create_compact_full_codec",
]
