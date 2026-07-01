"""Stage-1 JSON-chat detection encoding codec."""

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


def _stage1_json_chat_template() -> "DetectionSequenceTemplate":
    """Return the Stage-1 JSON-chat detection template owner."""

    from src.detection.template import get_detection_template

    return get_detection_template("stage1_json_pretty")


@dataclass(frozen=True, slots=True)
class Stage1JsonChatEncodingCodec(DetectionTemplateCodec):
    """Codec wrapping the existing ``stage1_json_pretty`` detection template."""

    template: "DetectionSequenceTemplate" = field(
        default_factory=_stage1_json_chat_template
    )
    options: DetectionTemplateRenderOptions = field(
        default_factory=DetectionTemplateRenderOptions
    )


def create_stage1_json_chat_codec(
    *,
    options: DetectionTemplateRenderOptions | None = None,
) -> Stage1JsonChatEncodingCodec:
    """Return a Stage-1 JSON-chat codec with optional encoding options."""

    from src.detection.template import get_detection_template

    return Stage1JsonChatEncodingCodec(
        template=get_detection_template("stage1_json_pretty"),
        options=options or DetectionTemplateRenderOptions(),
    )


__all__ = [
    "Stage1JsonChatEncodingCodec",
    "create_stage1_json_chat_codec",
]
