from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.config.schema import (
    Stage1SetContinuationCandidatesConfig,
    Stage1SetContinuationSubsetSamplingConfig,
)
from src.trainers.stage1_set_continuation.sampling import sample_subset_and_candidates


def test_random_subset_sampler_reports_mixtures_and_is_deterministic() -> None:
    subset_cfg = Stage1SetContinuationSubsetSamplingConfig(
        empty_prefix_ratio=0.0,
        random_subset_ratio=1.0,
        leave_one_out_ratio=0.0,
        full_prefix_ratio=0.0,
        prefix_order="random",
    )
    candidate_cfg = Stage1SetContinuationCandidatesConfig(mode="exact")

    first = sample_subset_and_candidates(
        object_count=5,
        subset_sampling_cfg=subset_cfg,
        candidates_cfg=candidate_cfg,
        seed_parts=(17, 3, 41, 0, "micro0"),
    )
    second = sample_subset_and_candidates(
        object_count=5,
        subset_sampling_cfg=subset_cfg,
        candidates_cfg=candidate_cfg,
        seed_parts=(17, 3, 41, 0, "micro0"),
    )

    assert first == second
    assert first.selected_mode == "random_subset"
    assert first.configured_mixture == {
        "empty_prefix": pytest.approx(0.0),
        "random_subset": pytest.approx(1.0),
        "leave_one_out": pytest.approx(0.0),
        "full_prefix": pytest.approx(0.0),
    }
    assert first.resolved_valid_mixture == first.configured_mixture
    assert sorted(first.prefix_indices + first.remaining_indices) == [0, 1, 2, 3, 4]
    assert set(first.prefix_indices).isdisjoint(first.remaining_indices)
    assert 0 < len(first.prefix_indices) < 5
    assert first.candidate_indices == first.remaining_indices
    assert first.candidate_scoring_mode == "exact"
    assert first.scored_candidate_fraction == pytest.approx(1.0)


def test_epoch_seed_part_can_change_selection() -> None:
    subset_cfg = Stage1SetContinuationSubsetSamplingConfig(
        empty_prefix_ratio=0.0,
        random_subset_ratio=1.0,
        leave_one_out_ratio=0.0,
        full_prefix_ratio=0.0,
        prefix_order="random",
    )
    candidate_cfg = Stage1SetContinuationCandidatesConfig(mode="exact")

    selections = {
        sample_subset_and_candidates(
            object_count=6,
            subset_sampling_cfg=subset_cfg,
            candidates_cfg=candidate_cfg,
            seed_parts=(17, epoch, 41, 0, "micro0"),
        ).prefix_indices
        for epoch in range(8)
    }

    assert len(selections) >= 2


def test_zero_and_one_object_renormalize_invalid_mixtures_deterministically() -> None:
    subset_cfg = Stage1SetContinuationSubsetSamplingConfig(
        empty_prefix_ratio=0.30,
        random_subset_ratio=0.45,
        leave_one_out_ratio=0.20,
        full_prefix_ratio=0.05,
        prefix_order="random",
    )
    candidate_cfg = Stage1SetContinuationCandidatesConfig(mode="exact")

    zero = sample_subset_and_candidates(
        object_count=0,
        subset_sampling_cfg=subset_cfg,
        candidates_cfg=candidate_cfg,
        seed_parts=(17, 0, 0, 0, "micro0"),
    )
    one = sample_subset_and_candidates(
        object_count=1,
        subset_sampling_cfg=subset_cfg,
        candidates_cfg=candidate_cfg,
        seed_parts=(17, 0, 1, 0, "micro0"),
    )

    assert zero.selected_mode == "empty_prefix"
    assert zero.resolved_valid_mixture == {
        "empty_prefix": pytest.approx(1.0),
        "random_subset": pytest.approx(0.0),
        "leave_one_out": pytest.approx(0.0),
        "full_prefix": pytest.approx(0.0),
    }
    assert zero.prefix_indices == ()
    assert zero.remaining_indices == ()
    assert zero.candidate_indices == ()
    assert zero.scored_candidate_fraction == pytest.approx(1.0)

    assert one.selected_mode in {"empty_prefix", "leave_one_out", "full_prefix"}
    assert one.resolved_valid_mixture == {
        "empty_prefix": pytest.approx(0.30 / 0.55),
        "random_subset": pytest.approx(0.0),
        "leave_one_out": pytest.approx(0.20 / 0.55),
        "full_prefix": pytest.approx(0.05 / 0.55),
    }
    assert sorted(one.prefix_indices + one.remaining_indices) == [0]
    assert set(one.prefix_indices).isdisjoint(one.remaining_indices)


def test_leave_one_out_excludes_exactly_one_object_and_scores_remaining() -> None:
    subset_cfg = Stage1SetContinuationSubsetSamplingConfig(
        empty_prefix_ratio=0.0,
        random_subset_ratio=0.0,
        leave_one_out_ratio=1.0,
        full_prefix_ratio=0.0,
        prefix_order="dataset",
    )
    candidate_cfg = Stage1SetContinuationCandidatesConfig(mode="exact")

    selection = sample_subset_and_candidates(
        object_count=4,
        subset_sampling_cfg=subset_cfg,
        candidates_cfg=candidate_cfg,
        seed_parts=(17, 0, 9, 0, "micro0"),
    )

    assert selection.selected_mode == "leave_one_out"
    assert len(selection.prefix_indices) == 3
    assert len(selection.remaining_indices) == 1
    assert selection.candidate_indices == selection.remaining_indices
    assert selection.scored_candidate_fraction == pytest.approx(1.0)


def test_full_prefix_uses_all_objects_and_scores_no_candidates() -> None:
    subset_cfg = Stage1SetContinuationSubsetSamplingConfig(
        empty_prefix_ratio=0.0,
        random_subset_ratio=0.0,
        leave_one_out_ratio=0.0,
        full_prefix_ratio=1.0,
        prefix_order="dataset",
    )
    candidate_cfg = Stage1SetContinuationCandidatesConfig(mode="exact")

    selection = sample_subset_and_candidates(
        object_count=4,
        subset_sampling_cfg=subset_cfg,
        candidates_cfg=candidate_cfg,
        seed_parts=(17, 0, 9, 0, "micro0"),
    )

    assert selection.selected_mode == "full_prefix"
    assert selection.prefix_indices == (0, 1, 2, 3)
    assert selection.remaining_indices == ()
    assert selection.candidate_indices == ()
    assert selection.candidate_scoring_mode == "exact"
    assert selection.scored_candidate_fraction == pytest.approx(1.0)


def test_uniform_subsample_caps_at_remaining_count_and_reports_fraction() -> None:
    subset_cfg = Stage1SetContinuationSubsetSamplingConfig(
        empty_prefix_ratio=1.0,
        random_subset_ratio=0.0,
        leave_one_out_ratio=0.0,
        full_prefix_ratio=0.0,
        prefix_order="dataset",
    )

    limited = sample_subset_and_candidates(
        object_count=5,
        subset_sampling_cfg=subset_cfg,
        candidates_cfg=Stage1SetContinuationCandidatesConfig(
            mode="uniform_subsample",
            max_candidates=2,
        ),
        seed_parts=(17, 0, 9, 0, "micro0"),
    )
    capped = sample_subset_and_candidates(
        object_count=5,
        subset_sampling_cfg=subset_cfg,
        candidates_cfg=Stage1SetContinuationCandidatesConfig(
            mode="uniform_subsample",
            max_candidates=99,
        ),
        seed_parts=(17, 0, 9, 0, "micro0"),
    )

    assert limited.candidate_scoring_mode == "uniform_subsample"
    assert len(limited.remaining_indices) == 5
    assert len(limited.candidate_indices) == 2
    assert set(limited.candidate_indices).issubset(limited.remaining_indices)
    assert limited.scored_candidate_fraction == pytest.approx(2.0 / 5.0)

    assert capped.candidate_indices == capped.remaining_indices
    assert capped.scored_candidate_fraction == pytest.approx(1.0)


def test_uniform_subsample_keeps_tail_positive_candidates() -> None:
    subset_cfg = Stage1SetContinuationSubsetSamplingConfig(
        empty_prefix_ratio=1.0,
        random_subset_ratio=0.0,
        leave_one_out_ratio=0.0,
        full_prefix_ratio=0.0,
        prefix_order="dataset",
    )

    selection = sample_subset_and_candidates(
        object_count=5,
        subset_sampling_cfg=subset_cfg,
        candidates_cfg=Stage1SetContinuationCandidatesConfig(
            mode="uniform_subsample",
            max_candidates=2,
            tail_positive_count=1,
        ),
        seed_parts=(17, 0, 9, 0, "micro0"),
    )

    assert len(selection.candidate_indices) == 2
    assert 4 in selection.candidate_indices
    assert set(selection.candidate_indices).issubset(selection.remaining_indices)


def test_uniform_subsample_requires_positive_k() -> None:
    subset_cfg = Stage1SetContinuationSubsetSamplingConfig(
        empty_prefix_ratio=1.0,
        random_subset_ratio=0.0,
        leave_one_out_ratio=0.0,
        full_prefix_ratio=0.0,
        prefix_order="dataset",
    )
    invalid_candidate_cfg = SimpleNamespace(mode="uniform_subsample", max_candidates=0)

    with pytest.raises(ValueError, match="max_candidates must be > 0"):
        sample_subset_and_candidates(
            object_count=3,
            subset_sampling_cfg=subset_cfg,
            candidates_cfg=invalid_candidate_cfg,
            seed_parts=(17, 0, 9, 0, "micro0"),
        )
