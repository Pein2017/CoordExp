"""Detection template codecs for training encoding contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.detection.data import NormalizedDetectionSample
    from src.detection.template import (
        DetectionSequenceTemplate,
        RenderedAssistantSequence,
        TemplateId,
    )
    from src.detection.tokenization import (
        TokenizedDetectionExample,
        TokenizerWithOffsets,
    )
    from src.training.encoding.view import EncodedDetectionView
else:
    DetectionSequenceTemplate = Any
    EncodedDetectionView = Any
    NormalizedDetectionSample = Any
    RenderedAssistantSequence = Any
    TemplateId = str
    TokenizedDetectionExample = Any
    TokenizerWithOffsets = Any


@dataclass(frozen=True, slots=True)
class DetectionTemplateRenderOptions:
    """Rendering and chat-template options for detection encoding.

    :param coordinate_surface: Coordinate surface understood by the detection
        template owner.
    :param bbox_format: Bounding-box format understood by the detection owner.
    :param prompt_template_id: Optional prompt-template identity for validation.
    :param system_prompt: Optional system prompt for chat-template projection.
    :param user_content: User turn content for chat-template projection.
    :param messages: Optional fully-owned chat message sequence.
    :param assistant_stop_markers: Optional assistant stop markers.
    :param include_rendered_assistant_text: Whether to store rendered assistant
        text on the diagnostic view.
    """

    coordinate_surface: str = "coord_token"
    bbox_format: str = "xyxy"
    prompt_template_id: str | None = None
    system_prompt: str | None = None
    user_content: str = "<image>"
    messages: Sequence[Mapping[str, Any]] | None = None
    assistant_stop_markers: Sequence[str] | None = None
    include_rendered_assistant_text: bool = True

    def __post_init__(self) -> None:
        """Validate and freeze caller-owned rendering options."""

        if type(self.coordinate_surface) is not str:
            raise TypeError("coordinate_surface must be a plain string")
        if type(self.bbox_format) is not str:
            raise TypeError("bbox_format must be a plain string")
        if (
            self.prompt_template_id is not None
            and type(self.prompt_template_id) is not str
        ):
            raise TypeError("prompt_template_id must be a plain string or None")
        if self.system_prompt is not None and type(self.system_prompt) is not str:
            raise TypeError("system_prompt must be a plain string or None")
        if type(self.user_content) is not str:
            raise TypeError("user_content must be a plain string")
        if type(self.include_rendered_assistant_text) is not bool:
            raise TypeError("include_rendered_assistant_text must be a plain bool")

        object.__setattr__(
            self,
            "assistant_stop_markers",
            self._freeze_assistant_stop_markers(self.assistant_stop_markers),
        )
        object.__setattr__(
            self,
            "messages",
            self._freeze_messages(self.messages),
        )

    @staticmethod
    def _freeze_assistant_stop_markers(
        markers: Sequence[str] | None,
    ) -> tuple[str, ...] | None:
        """Return immutable assistant stop markers from caller-owned input."""

        if markers is None:
            return None
        if isinstance(markers, (str, bytes, Mapping)) or not isinstance(
            markers,
            Sequence,
        ):
            raise TypeError("assistant_stop_markers must be a sequence of strings")

        normalized: list[str] = []
        for marker in markers:
            if type(marker) is not str:
                raise TypeError("assistant_stop_markers must contain plain strings")
            if not marker:
                raise ValueError("assistant_stop_markers must contain non-empty strings")
            normalized.append(marker)

        return tuple(normalized)

    @staticmethod
    def _freeze_messages(
        messages: Sequence[Mapping[str, Any]] | None,
    ) -> tuple[Mapping[str, Any], ...] | None:
        """Return immutable message mappings from caller-owned input."""

        if messages is None:
            return None
        if isinstance(messages, (str, bytes, Mapping)) or not isinstance(
            messages,
            Sequence,
        ):
            raise TypeError("messages must be a sequence of mappings")

        normalized: list[Mapping[str, Any]] = []
        for message in messages:
            if not isinstance(message, Mapping):
                raise TypeError("messages must contain mappings")

            copied_message = dict(message)
            if any(type(key) is not str for key in copied_message):
                raise TypeError("message keys must be plain strings")
            normalized.append(MappingProxyType(copied_message))

        return tuple(normalized)


@dataclass(frozen=True, slots=True)
class DetectionTemplateCodec:
    """Thin wrapper around the existing detection template/tokenization owners.

    :param template: Existing detection sequence template implementation.
    :param options: Rendering and chat-template options.
    """

    template: "DetectionSequenceTemplate"
    options: DetectionTemplateRenderOptions = field(
        default_factory=DetectionTemplateRenderOptions
    )

    @classmethod
    def from_template_id(
        cls,
        template_id: "TemplateId | str",
        *,
        options: DetectionTemplateRenderOptions | None = None,
    ) -> "DetectionTemplateCodec":
        """Build a codec around the registered detection template owner."""

        from src.detection.template import get_detection_template

        return cls(
            template=get_detection_template(template_id),
            options=options or DetectionTemplateRenderOptions(),
        )

    def render_assistant(
        self,
        sample: "NormalizedDetectionSample",
    ) -> "RenderedAssistantSequence":
        """Render a sample through the existing detection template owner."""

        return self.template.render_assistant(
            sample,
            coordinate_surface=self.options.coordinate_surface,
            bbox_format=self.options.bbox_format,
            prompt_template_id=self.options.prompt_template_id,
        )

    def tokenize(
        self,
        rendered: "RenderedAssistantSequence",
        *,
        tokenizer: "TokenizerWithOffsets",
    ) -> "TokenizedDetectionExample":
        """Tokenize a rendered assistant through the existing tokenizer owner."""

        from src.detection.tokenization import tokenize_rendered_detection_conversation

        return tokenize_rendered_detection_conversation(
            rendered,
            tokenizer=tokenizer,
            system_prompt=self.options.system_prompt,
            user_content=self.options.user_content,
            messages=self.options.messages,
            assistant_stop_markers=self.options.assistant_stop_markers,
        )

    def encode_sample(
        self,
        sample: "NormalizedDetectionSample",
        *,
        tokenizer: "TokenizerWithOffsets",
    ) -> "EncodedDetectionView":
        """Render and tokenize a sample into the authoritative training view."""

        from src.training.encoding.view import EncodedDetectionView

        rendered = self.render_assistant(sample)
        tokenized = self.tokenize(rendered, tokenizer=tokenizer)

        return EncodedDetectionView.from_tokenized(
            tokenized,
            include_rendered_assistant_text=(
                self.options.include_rendered_assistant_text
            ),
        )


__all__ = [
    "DetectionTemplateCodec",
    "DetectionTemplateRenderOptions",
]
