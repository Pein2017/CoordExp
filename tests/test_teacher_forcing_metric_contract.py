from __future__ import annotations

import pytest

from src.metrics.events import flatten_metric_events
from src.training.teacher_forcing.metrics import (
    REQUIRED_TEACHER_FORCING_METRIC_KEYS,
    builder_rejection_events,
    compact_full_parse_error_events,
    decode_quality_events,
    summarize_metric_events,
    teacher_forcing_diagnostic_events,
    teacher_forcing_loss_events,
)


def test_required_teacher_forcing_metric_keys_are_canonical() -> None:
    required = set(REQUIRED_TEACHER_FORCING_METRIC_KEYS)

    assert {
        "teacher_forcing/loss/total",
        "teacher_forcing/loss/token_type_mass",
        "teacher_forcing/loss/conditional_valid_set_likelihood",
        "teacher_forcing/loss/within_valid_coverage",
        "teacher_forcing/valid_set/mass",
        "teacher_forcing/coverage/kl",
        "teacher_forcing/continuation/eos_margin",
        "teacher_forcing/ambiguity/coordinate_onset_count",
        "teacher_forcing/ambiguity/mixed_role_count",
        "teacher_forcing/builder/rejected_samples",
        "teacher_forcing/builder/rejection_reason/<code>",
        "teacher_forcing/branch_coherence/rate",
        "teacher_forcing/residual_set/remaining_count",
        "teacher_forcing/permutation_probe/nll_std",
        "teacher_forcing/decode/object_coherence_rate",
        "teacher_forcing/decode/duplicate_rate",
        "teacher_forcing/decode/missed_object_rate",
        "teacher_forcing/decode/malformed_sequence_rate",
        "infer/parse/compact_full/error/<code>",
    }.issubset(required)
    assert not any(key.startswith("recursive_detection/") for key in required)


def test_teacher_forcing_events_flatten_to_required_metric_names() -> None:
    events = [
        *teacher_forcing_loss_events(
            loss=2.5,
            denominator=5,
            span_count=2,
            atom_count=8,
        ),
        *teacher_forcing_diagnostic_events(
            token_type_mass=0.75,
            conditional_valid_set_likelihood=0.6,
            within_valid_coverage=0.8,
            valid_set_mass=0.7,
            coverage_kl=0.12,
            eos_margin=-0.3,
            coordinate_onset_count=3,
            mixed_role_count=2,
            branch_coherent=7,
            branch_total=10,
            residual_remaining_count=4,
            permutation_nll_std=0.09,
        ),
        *builder_rejection_events(
            {
                "missing_detection_list": 1,
                "empty_detection_list": 2,
                "overlength": 3,
                "description_tokenization_failed": 4,
            }
        ),
        *decode_quality_events(
            artifact_present=True,
            object_coherent=8,
            duplicate=2,
            missed_object=1,
            malformed_sequence=3,
            total=10,
        ),
        *compact_full_parse_error_events({"bad_marker": 2}),
    ]

    flat = flatten_metric_events(events)

    assert flat["teacher_forcing/loss/total"] == pytest.approx(2.5)
    assert flat["teacher_forcing/loss/token_type_mass"] == pytest.approx(0.75)
    assert flat["teacher_forcing/loss/conditional_valid_set_likelihood"] == pytest.approx(0.6)
    assert flat["teacher_forcing/loss/within_valid_coverage"] == pytest.approx(0.8)
    assert flat["teacher_forcing/valid_set/mass"] == pytest.approx(0.7)
    assert flat["teacher_forcing/coverage/kl"] == pytest.approx(0.12)
    assert flat["teacher_forcing/continuation/eos_margin"] == pytest.approx(-0.3)
    assert flat["teacher_forcing/ambiguity/coordinate_onset_count"] == pytest.approx(3)
    assert flat["teacher_forcing/ambiguity/mixed_role_count"] == pytest.approx(2)
    assert flat["teacher_forcing/builder/rejected_samples"] == pytest.approx(10)
    assert flat["teacher_forcing/builder/rejection_reason/missing_detection_list"] == pytest.approx(1)
    assert flat["teacher_forcing/builder/rejection_reason/empty_detection_list"] == pytest.approx(2)
    assert flat["teacher_forcing/builder/rejection_reason/overlength"] == pytest.approx(3)
    assert flat[
        "teacher_forcing/builder/rejection_reason/description_tokenization_failed"
    ] == pytest.approx(4)
    assert flat["teacher_forcing/branch_coherence/rate"] == pytest.approx(0.7)
    assert flat["teacher_forcing/residual_set/remaining_count"] == pytest.approx(4)
    assert flat["teacher_forcing/permutation_probe/nll_std"] == pytest.approx(0.09)
    assert flat["teacher_forcing/decode/object_coherence_rate"] == pytest.approx(0.8)
    assert flat["teacher_forcing/decode/duplicate_rate"] == pytest.approx(0.2)
    assert flat["teacher_forcing/decode/missed_object_rate"] == pytest.approx(0.1)
    assert flat["teacher_forcing/decode/malformed_sequence_rate"] == pytest.approx(0.3)
    assert flat["infer/parse/compact_full/error/bad_marker"] == pytest.approx(2)
    assert not any(key.startswith("training/objectives/teacher_forcing") for key in flat)
    assert not any(key.startswith("recursive_detection/") for key in flat)


def test_decode_quality_events_record_absent_artifact_without_zero_rates() -> None:
    summary = summarize_metric_events(
        decode_quality_events(
            artifact_present=False,
            object_coherent=0,
            duplicate=0,
            missed_object=0,
            malformed_sequence=0,
            total=0,
        )
    )

    assert summary["artifact_status"]["teacher_forcing/decode"] == "absent"
    assert "teacher_forcing/decode/object_coherence_rate" not in summary["metrics"]
    assert "teacher_forcing/decode/duplicate_rate" not in summary["metrics"]
    assert "teacher_forcing/decode/missed_object_rate" not in summary["metrics"]
    assert "teacher_forcing/decode/malformed_sequence_rate" not in summary["metrics"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"branch_coherent": 1, "branch_total": None},
        {"branch_coherent": None, "branch_total": 1},
    ],
)
def test_branch_coherence_requires_complete_ratio_inputs(kwargs: dict[str, int | None]) -> None:
    with pytest.raises(ValueError, match="branch_coherent and branch_total"):
        teacher_forcing_diagnostic_events(**kwargs)


def test_builder_rejection_reason_code_normalization_rejects_collisions() -> None:
    with pytest.raises(ValueError, match="code collision"):
        builder_rejection_events({"a/b": 1, "a_b": 2})


def test_compact_full_parse_error_code_normalization_rejects_collisions() -> None:
    with pytest.raises(ValueError, match="code collision"):
        compact_full_parse_error_events({"a/b": 1, "a_b": 2})
