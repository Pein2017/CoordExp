from __future__ import annotations

import json
import re


def select_duplicate_burst_cases(
    *,
    rows: list[dict[str, object]],
    max_cases_per_model: int,
    serializer_surface: str = "model_native",
) -> list[dict[str, object]]:
    filtered_rows = [
        dict(row)
        for row in rows
        if str(row.get("review_bucket") or "") == "FP"
        and str(row.get("serializer_surface") or "") == serializer_surface
    ]
    selected: list[dict[str, object]] = []
    counts_by_model: dict[str, int] = {}
    for row in sorted(
        filtered_rows,
        key=lambda item: (
            int(item.get("selection_rank") or 10**9),
            str(item.get("model_alias") or ""),
            str(item.get("case_uid") or ""),
        ),
    ):
        model_alias = str(row.get("model_alias") or "")
        if counts_by_model.get(model_alias, 0) >= max(0, int(max_cases_per_model)):
            continue
        selected.append(row)
        counts_by_model[model_alias] = counts_by_model.get(model_alias, 0) + 1
    return selected


def _coerce_generated_text(native_assistant_text: object) -> str:
    if isinstance(native_assistant_text, str):
        return native_assistant_text
    if isinstance(native_assistant_text, list):
        return "".join(str(piece) for piece in native_assistant_text)
    raise TypeError("native_assistant_text must be a string or token-text list")


def render_model_native_prefix_assistant_text(
    *,
    objects: list[dict[str, object]],
    native_assistant_text: object,
) -> str:
    raw_text = _coerce_generated_text(native_assistant_text).strip()
    if raw_text.endswith("<|im_end|>"):
        raw_text = raw_text[: -len("<|im_end|>")].rstrip()
    payload = {"objects": [dict(obj) for obj in objects]}
    if raw_text.startswith("```"):
        rendered_json = json.dumps(payload, ensure_ascii=False, indent=2)
        rendered_json = re.sub(
            r"\[\n(?P<body>(?:\s*-?\d+,\n)+\s*-?\d+)\n(?P<indent>\s*)\]",
            lambda match: (
                "["
                + ", ".join(
                    part.strip().rstrip(",")
                    for part in match.group("body").splitlines()
                )
                + "]"
            ),
            rendered_json,
        )
        return f"```json\n{rendered_json}\n```"
    return json.dumps(payload, ensure_ascii=False)


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
