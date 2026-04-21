from src.analysis.raw_text_coordinate_exploratory import (
    build_prefix_intervention_matrix,
    label_fn_bucket,
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
