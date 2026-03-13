"""Shared visualization helpers."""

from .gt_vs_pred import (
    canonicalize_gt_vs_pred_record,
    default_vis_resource_path,
    ensure_gt_vs_pred_vis_resource,
    materialize_eval_gt_vs_pred_vis_resource,
    materialize_gt_vs_pred_vis_resource,
    render_gt_vs_pred_review,
)
from .comparison import compose_comparison_scenes_from_jsonls

__all__ = [
    "canonicalize_gt_vs_pred_record",
    "compose_comparison_scenes_from_jsonls",
    "default_vis_resource_path",
    "ensure_gt_vs_pred_vis_resource",
    "materialize_eval_gt_vs_pred_vis_resource",
    "materialize_gt_vs_pred_vis_resource",
    "render_gt_vs_pred_review",
]
