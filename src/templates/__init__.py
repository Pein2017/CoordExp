"""Template rendering for validated raw examples."""

from src.templates.renderer import (
    BOX_END_TOKEN,
    BOX_START_TOKEN,
    IMAGE_PLACEHOLDER,
    IM_END_SUFFIX,
    IM_END_TOKEN,
    OBJECT_REF_END_TOKEN,
    OBJECT_REF_START_TOKEN,
    RenderedExample,
    RenderedObjectOrder,
    render_example,
)
from src.templates.snapshot import rendered_example_snapshot, rendered_examples_snapshot
from src.templates.spans import RenderedSpan, validate_rendered_spans

__all__ = [
    "BOX_END_TOKEN",
    "BOX_START_TOKEN",
    "IMAGE_PLACEHOLDER",
    "IM_END_SUFFIX",
    "IM_END_TOKEN",
    "OBJECT_REF_END_TOKEN",
    "OBJECT_REF_START_TOKEN",
    "RenderedExample",
    "RenderedObjectOrder",
    "RenderedSpan",
    "rendered_example_snapshot",
    "rendered_examples_snapshot",
    "render_example",
    "validate_rendered_spans",
]
