"""Shared detection chat-message construction.

The Qwen3-VL chat template preserves multimodal content order. Detection
training and inference must therefore share one builder instead of assembling
image/text parts independently.
"""

from __future__ import annotations

from typing import Any, Sequence


def build_detection_chat_messages(
    *,
    system_prompt: str | None,
    user_prompt: str,
    images: Sequence[Any],
    assistant_text: str | None = None,
    image_content_type: str = "image",
) -> list[dict[str, Any]]:
    """Build Qwen-compatible detection messages with image-before-text order."""

    if image_content_type not in {"image", "image_url"}:
        raise ValueError(
            "image_content_type must be either 'image' or 'image_url', "
            f"got {image_content_type!r}"
        )

    messages: list[dict[str, Any]] = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": str(system_prompt)})

    user_content: list[dict[str, Any]] = []
    for image in images:
        if image_content_type == "image_url":
            user_content.append({"type": "image_url", "image_url": image})
        else:
            user_content.append({"type": "image", "image": image})
    user_content.append({"type": "text", "text": str(user_prompt)})
    messages.append({"role": "user", "content": user_content})

    if assistant_text is not None:
        messages.append({"role": "assistant", "content": str(assistant_text)})

    return messages


__all__ = ["build_detection_chat_messages"]
