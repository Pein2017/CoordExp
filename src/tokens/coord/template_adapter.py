from __future__ import annotations

from types import MethodType
from typing import Any


def apply_coord_template_adapter(template: Any, coord_cfg: Any) -> Any:
    """Patch template to skip bbox normalization in coord-token mode.

    The adapter is a no-op when coord_cfg.enabled is False. When enabled and
    skip_bbox_norm is True, template.normalize_bbox is replaced with a no-op
    to avoid double-scaling pre-quantized coord tokens.
    """

    if not getattr(coord_cfg, "enabled", False):
        return template

    if getattr(coord_cfg, "skip_bbox_norm", False):
        # Only patch once per template instance
        if not getattr(template, "_coord_tokens_skip_norm", False):
            original = getattr(template, "normalize_bbox", None)

            def _skip_norm(self, inputs):
                return None

            if original is not None:
                setattr(template, "_coord_tokens_original_normalize_bbox", original)
            template.normalize_bbox = MethodType(_skip_norm, template)
            template._coord_tokens_skip_norm = True  # type: ignore[attr-defined]
    return template


__all__ = ["apply_coord_template_adapter"]
