from __future__ import annotations


def build_prefix_intervention_matrix(
    *,
    source_object: dict[str, object],
    gt_next: dict[str, object],
) -> list[dict[str, object]]:
    return [
        {"label": "baseline", "mode": "identity"},
        {"label": "drop_previous_object", "mode": "drop_previous"},
        {"label": "geometry_only_swap", "mode": "replace_bbox_keep_text"},
        {"label": "text_only_swap", "mode": "replace_text_keep_bbox"},
        {"label": "x1y1_only", "mode": "replace_x1y1"},
        {"label": "x2y2_only", "mode": "replace_x2y2"},
        {"label": "full_gt_next_geometry", "mode": "replace_bbox_with_gt_next"},
        {
            "label": "nonlocal_same_desc_geometry",
            "mode": "replace_bbox_with_nonlocal_same_desc",
        },
    ]


def label_fn_bucket(
    *,
    recovered_by_sampling: bool,
    recovered_by_clean_prefix: bool,
    recovered_by_stop_probe: bool,
    has_teacher_forced_support: bool,
    ambiguity_flag: bool,
) -> str:
    if ambiguity_flag:
        return "unlabeled_positive_or_eval_ambiguity"
    if recovered_by_sampling:
        return "decode_selection_fn"
    if recovered_by_clean_prefix:
        return "continuation_blocked_fn"
    if recovered_by_stop_probe and has_teacher_forced_support:
        return "stop_pressure_fn"
    if not has_teacher_forced_support:
        return "never_supported_fn"
    return "never_supported_fn"
