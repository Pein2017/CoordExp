"""Visualization stage for the unified inference pipeline.

This module keeps the historical public entry point, but the actual review
semantics now live in the shared canonical GT-vs-Pred visualization layer.
Raw inference artifacts are first materialized into a canonical sidecar under
``vis_resources/gt_vs_pred.jsonl``, then rendered with the shared 1x2 reviewer.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from src.utils import get_logger
from src.vis import ensure_gt_vs_pred_vis_resource, render_gt_vs_pred_review

logger = get_logger(__name__)


def render_vis_from_jsonl(
    jsonl_path: Path,
    *,
    out_dir: Path,
    limit: int = 20,
    root_image_dir: Optional[Path] = None,
    root_source: str = "none",
) -> None:
    if root_image_dir is None:
        root_env = str(os.environ.get("ROOT_IMAGE_DIR") or "").strip()
        if root_env:
            root_image_dir = Path(root_env).resolve()
            root_source = "env"

    if root_image_dir is not None:
        logger.info(
            "vis: resolved ROOT_IMAGE_DIR (source=%s): %s", root_source, root_image_dir
        )
    else:
        logger.info(
            "vis: ROOT_IMAGE_DIR not set; falling back to JSONL directory for image resolution"
        )

    vis_resource_path = ensure_gt_vs_pred_vis_resource(
        jsonl_path,
        source_kind="offline_single_run",
        materialize_matching=True,
    )
    logger.info("vis: rendering canonical GT-vs-Pred sidecar from %s", vis_resource_path)
    render_gt_vs_pred_review(
        vis_resource_path,
        out_dir=out_dir,
        limit=limit,
        root_image_dir=root_image_dir,
        root_source=root_source,
        record_order="input",
    )
