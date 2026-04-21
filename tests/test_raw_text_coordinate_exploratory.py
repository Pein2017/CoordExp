from src.analysis.raw_text_coordinate_exploratory import (
    build_prefix_intervention_matrix,
    label_fn_bucket,
    render_model_native_prefix_assistant_text,
    select_duplicate_burst_cases,
)


def test_build_prefix_intervention_matrix_exposes_all_required_variants() -> None:
    variants = build_prefix_intervention_matrix(
        source_object={"bbox": [1, 2, 3, 4]},
        gt_next={"bbox": [4, 5, 6, 7]},
    )

    assert [variant["label"] for variant in variants] == [
        "baseline",
        "drop_previous_object",
        "geometry_only_swap",
        "text_only_swap",
        "x1y1_only",
        "x2y2_only",
        "full_gt_next_geometry",
        "nonlocal_same_desc_geometry",
    ]


def test_label_fn_bucket_prefers_stop_pressure_only_when_other_recovery_is_absent() -> None:
    bucket = label_fn_bucket(
        recovered_by_sampling=False,
        recovered_by_clean_prefix=False,
        recovered_by_stop_probe=True,
        has_teacher_forced_support=True,
        ambiguity_flag=False,
    )

    assert bucket == "stop_pressure_fn"


def test_select_duplicate_burst_cases_balances_models_after_surface_filter() -> None:
    rows = [
        {
            "case_uid": "base_only:1",
            "review_bucket": "FP",
            "model_alias": "base_only",
            "selection_rank": 1,
            "serializer_surface": "model_native",
        },
        {
            "case_uid": "base_only:1:pretty",
            "review_bucket": "FP",
            "model_alias": "base_only",
            "selection_rank": 1,
            "serializer_surface": "pretty_inline",
        },
        {
            "case_uid": "base_plus_adapter:1",
            "review_bucket": "FP",
            "model_alias": "base_plus_adapter",
            "selection_rank": 1,
            "serializer_surface": "model_native",
        },
        {
            "case_uid": "base_plus_adapter:2",
            "review_bucket": "FP",
            "model_alias": "base_plus_adapter",
            "selection_rank": 2,
            "serializer_surface": "model_native",
        },
        {
            "case_uid": "base_only:fn",
            "review_bucket": "FN",
            "model_alias": "base_only",
            "selection_rank": 1,
            "serializer_surface": "model_native",
        },
    ]

    selected = select_duplicate_burst_cases(
        rows=rows,
        max_cases_per_model=1,
    )

    assert [row["case_uid"] for row in selected] == [
        "base_only:1",
        "base_plus_adapter:1",
    ]


def test_render_model_native_prefix_assistant_text_preserves_fenced_style() -> None:
    rendered = render_model_native_prefix_assistant_text(
        objects=[{"desc": "book", "bbox_2d": [1, 2, 3, 4]}],
        native_assistant_text=(
            '```json\n{"objects": [{"desc": "book", "bbox_2d": [9, 9, 9, 9]}]}\n```'
        ),
    )

    assert rendered.startswith("```json\n")
    assert rendered.endswith("\n```")
    assert '"bbox_2d": [\n' not in rendered
    assert '"bbox_2d": [1, 2, 3, 4]' in rendered


def test_render_model_native_prefix_assistant_text_strips_im_end_for_inline_json() -> None:
    rendered = render_model_native_prefix_assistant_text(
        objects=[{"desc": "book", "bbox_2d": [1, 2, 3, 4]}],
        native_assistant_text=(
            '{"objects": [{"desc": "book", "bbox_2d": [9, 9, 9, 9]}]}<|im_end|>'
        ),
    )

    assert rendered == '{"objects": [{"desc": "book", "bbox_2d": [1, 2, 3, 4]}]}'
