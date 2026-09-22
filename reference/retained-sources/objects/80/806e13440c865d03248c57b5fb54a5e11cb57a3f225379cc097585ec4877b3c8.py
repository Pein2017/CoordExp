"""Stable inspection snapshots for rendered examples."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from src.templates.renderer import RenderedExample


SNAPSHOT_SCHEMA_VERSION = 1


def rendered_example_snapshot(
    rendered: RenderedExample,
    *,
    image_root: Path | None = None,
) -> dict[str, Any]:
    payload = rendered.to_artifact_dict()
    if image_root is not None:
        payload["messages"] = _normalize_message_images(payload["messages"], image_root.resolve())
    return payload


def rendered_examples_snapshot(
    rendered_examples: Iterable[RenderedExample],
    *,
    image_root: Path | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "examples": [
            rendered_example_snapshot(rendered, image_root=image_root)
            for rendered in rendered_examples
        ],
    }


def _normalize_message_images(messages: list[dict[str, Any]], image_root: Path) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for message in messages:
        copied = dict(message)
        content = copied.get("content")
        if isinstance(content, list):
            copied["content"] = [
                _normalize_content_item(item, image_root)
                if isinstance(item, dict)
                else item
                for item in content
            ]
        normalized.append(copied)
    return normalized


def _normalize_content_item(item: dict[str, Any], image_root: Path) -> dict[str, Any]:
    copied = dict(item)
    image = copied.get("image")
    if isinstance(image, str):
        image_path = Path(image)
        if image_path.is_absolute():
            try:
                copied["image"] = str(image_path.resolve().relative_to(image_root))
            except ValueError:
                copied["image"] = str(image_path)
    return copied


__all__ = [
    "SNAPSHOT_SCHEMA_VERSION",
    "rendered_example_snapshot",
    "rendered_examples_snapshot",
]
