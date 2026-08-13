from __future__ import annotations

import pytest

from scripts.research.human13_frontier_selection import (
    CandidatePath,
    ContinuationOutcome,
    PackedCandidateScore,
    SurfaceEvidence,
    packed_prefilter,
    score_packed_candidate,
    protected_owner_coverable,
    score_candidate,
    select_continuation,
    shortlist_candidates,
)


def _surface(
    name: str, rows: list[list[float]], targets: tuple[int, ...]
) -> SurfaceEvidence:
    return SurfaceEvidence(
        surface=name, logits=tuple(tuple(row) for row in rows), target_token_ids=targets
    )


def test_hf_owns_barrier_first_strict_bottleneck_and_tie() -> None:
    path = CandidatePath(1, "h1", "alias", (1, 2, 3))
    packed = _surface(
        "packed_bf16_fa2", [[0, 4, 1, 0], [0, 0, 5, 1], [0, 0, 1, 6]], path.token_ids
    )
    hf = _surface(
        "hf_fp32_sdpa", [[0, 4, 1, 0], [0, 6, 5, 1], [0, 0, 6, 6]], path.token_ids
    )

    score = score_candidate(path, packed=packed, hf=hf)

    assert score.hf_barrier == pytest.approx(1.0)
    assert score.first_bottleneck_index == 1
    assert score.hf_sites[2].tie_count == 2
    assert score.hf_sites[2].strict_margin == pytest.approx(0.0)
    assert score.max_surface_margin_drift == pytest.approx(5.0)
    assert score.packed_first_bottleneck_index is None
    assert score.first_bottleneck_disagreement
    assert len(score.aligned_sites) == 3


def test_missing_hf_evidence_fails_even_with_good_packed_path() -> None:
    path = CandidatePath(1, "h1", "alias", (1,))
    packed = _surface("packed_bf16_fa2", [[0, 2]], path.token_ids)
    with pytest.raises(ValueError, match="HF|hf"):
        score_candidate(path, packed=packed, hf=None)


def test_shortlist_reduces_aliases_by_hf_barrier_and_keeps_two_to_four_owners() -> None:
    scores = []
    for owner, alias, deficit in (
        ("h1", "a", 3.0),
        ("h1", "b", 1.0),
        ("h2", "a", 2.0),
        ("h3", "a", 4.0),
        ("h4", "a", 5.0),
        ("h5", "a", 6.0),
    ):
        path = CandidatePath(1, owner, alias, (1,))
        scores.append(
            score_candidate(
                path, packed=None, hf=_surface("hf_fp32_sdpa", [[deficit, 0]], (1,))
            )
        )

    shortlist = shortlist_candidates(scores, limit=4)

    assert [(item.path.owner_id, item.path.alias_id) for item in shortlist] == [
        ("h1", "b"),
        ("h2", "a"),
        ("h3", "a"),
        ("h4", "a"),
    ]
    with pytest.raises(ValueError, match="two|2"):
        shortlist_candidates(scores, limit=1)


def test_packed_only_prefilter_bounds_hf_work_to_two_aliases_per_owner() -> None:
    packed_scores: list[PackedCandidateScore] = []
    for alias, deficit in (("a", 3.0), ("b", 1.0), ("c", 2.0)):
        path = CandidatePath(1, "h1", alias, (1,))
        packed_scores.append(
            score_packed_candidate(
                path,
                _surface("packed_bf16_fa2", [[deficit, 0]], path.token_ids),
            )
        )
    survivors = packed_prefilter(packed_scores, aliases_per_owner=2)
    assert [item.path.alias_id for item in survivors] == ["b", "c"]
    assert all(not hasattr(item, "hf_sites") for item in survivors)
    with pytest.raises(ValueError, match="at most two|<=2|one or two"):
        packed_prefilter(packed_scores, aliases_per_owner=3)


def test_cross_surface_tie_bottleneck_disagrees_even_with_same_argmax_id() -> None:
    path = CandidatePath(1, "h1", "a", (0,))
    score = score_candidate(
        path,
        packed=_surface("packed_bf16_fa2", [[2, 1]], path.token_ids),
        hf=_surface("hf_fp32_sdpa", [[2, 2]], path.token_ids),
    )
    assert score.hf_sites[0].actual_argmax_token_id == 0
    assert score.packed_sites[0].actual_argmax_token_id == 0
    assert score.first_bottleneck_disagreement
    assert score.surface_rank_disagreement
    with pytest.raises(ValueError, match="vocabulary"):
        score_candidate(
            path,
            packed=_surface("packed_bf16_fa2", [[2, 1, 0]], path.token_ids),
            hf=_surface("hf_fp32_sdpa", [[2, 2]], path.token_ids),
        )


def test_select_continuation_is_lexicographic_not_barrier_only() -> None:
    bad_easy = ContinuationOutcome(
        "h1", 0.1, False, 2, "natural_im_end", False, 0, 0, 10, 30
    )
    safe_gain = ContinuationOutcome(
        "h2", 4.0, True, 1, "natural_im_end", False, 1, 0, 12, 40
    )
    more_burden = ContinuationOutcome(
        "h3", 2.0, True, 1, "natural_im_end", False, 2, 0, 20, 80
    )

    assert select_continuation((bad_easy, more_burden, safe_gain)) == safe_gain


def test_cap_hit_is_harm_and_zero_gain_is_ineligible() -> None:
    capped = ContinuationOutcome("h1", 0.1, True, 2, "length_cap", True, 0, 0, 10, 30)
    zero_gain = ContinuationOutcome(
        "h2", 0.2, True, 0, "natural_im_end", False, 0, 0, 10, 30
    )
    with pytest.raises(ValueError, match="eligible"):
        select_continuation((capped, zero_gain))


@pytest.mark.parametrize(
    "termination", ("backend_error", "cancelled", "premature_stop")
)
def test_only_natural_continuation_termination_is_eligible(termination: str) -> None:
    bad = ContinuationOutcome("h1", 0.1, True, 1, termination, False, 0, 0, 10, 30)
    with pytest.raises(ValueError, match="eligible"):
        select_continuation((bad,))


def test_surface_shapes_and_targets_are_fail_closed() -> None:
    path = CandidatePath(1, "h1", "a", (1, 2))
    with pytest.raises(ValueError, match="length|shape"):
        score_candidate(path, packed=None, hf=_surface("hf_fp32_sdpa", [[0, 2]], (1,)))
    with pytest.raises(ValueError, match="target"):
        score_candidate(
            path, packed=None, hf=_surface("hf_fp32_sdpa", [[0, 2], [0, 2]], (1, 1))
        )


def test_hf_alias_rank_can_reverse_packed_rank_and_remains_authoritative() -> None:
    left = CandidatePath(1, "h1", "a", (1,))
    right = CandidatePath(1, "h1", "b", (1,))
    packed = (
        score_packed_candidate(left, _surface("packed_bf16_fa2", [[1, 2]], (1,))),
        score_packed_candidate(right, _surface("packed_bf16_fa2", [[3, 2]], (1,))),
    )
    survivors = packed_prefilter(packed, aliases_per_owner=2)
    scored = (
        score_candidate(
            survivors[0].path,
            packed=None,
            hf=_surface("hf_fp32_sdpa", [[4, 2]], (1,)),
        ),
        score_candidate(
            survivors[1].path,
            packed=None,
            hf=_surface("hf_fp32_sdpa", [[1, 2]], (1,)),
        ),
    )
    assert shortlist_candidates(scored, limit=2)[0].path.alias_id == "b"


def test_protected_coverability_is_a_matching_question_not_canonical_identity() -> None:
    owners = (
        ("g1", "book", (0.0, 0.0, 10.0, 10.0)),
        ("g2", "book", (1.0, 0.0, 11.0, 10.0)),
    )
    predictions = (
        ("book", (0.0, 0.0, 10.0, 10.0)),
        ("book", (1.0, 0.0, 11.0, 10.0)),
    )
    assert protected_owner_coverable(owners, predictions, protected_owner_ids=("g2",))
    assert not protected_owner_coverable(
        owners, predictions[:1], protected_owner_ids=("g1", "g2")
    )
