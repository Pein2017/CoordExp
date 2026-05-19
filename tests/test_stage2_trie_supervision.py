from __future__ import annotations

import json

import pytest

from src.trainers.stage2_two_channel.trie_supervision import (
    Stage2TrieCandidate,
    Stage2TrieObjectSpan,
    Stage2TrieSpanScoreRecord,
    Stage2TrieTokenTarget,
    build_fp_object_span,
    compile_stage2_trie_targets,
    compile_stage2_trie_targets_for_rollout_group,
    stage2_trie_span_score_record_to_json,
)


def test_compile_stage2_trie_merges_multiple_positive_next_tokens() -> None:
    candidates = [
        Stage2TrieCandidate(
            sample_id="img-1",
            rollout_index=0,
            source="valid_rollout",
            token_ids=[101, 11, 12, 102],
            loss_weight=1.0,
            object_spans=[],
        ),
        Stage2TrieCandidate(
            sample_id="img-1",
            rollout_index=1,
            source="valid_rollout",
            token_ids=[101, 11, 13, 102],
            loss_weight=1.0,
            object_spans=[],
        ),
    ]

    targets = compile_stage2_trie_targets(candidates, label_position_start=5)

    targets_by_position = {target.position: target for target in targets.token_targets}
    assert sorted(targets_by_position) == [5, 6, 7, 8]
    assert targets_by_position[7].positive_token_ids == (12, 13)
    assert targets_by_position[7].source_weights == (1.0, 1.0)
    assert targets.summary.candidate_count == 2
    assert targets.summary.branch_points == 1
    assert targets.summary.max_branching_factor == 2


def test_compile_stage2_trie_downweights_fallback_candidate() -> None:
    candidates = [
        Stage2TrieCandidate(
            sample_id="img-1",
            rollout_index=-1,
            source="fallback_gt_fn_append_only",
            token_ids=[101, 21, 102],
            loss_weight=0.25,
            object_spans=[
                Stage2TrieObjectSpan(
                    role="fallback_fn",
                    token_start=1,
                    token_end=2,
                    object_iou=None,
                    support_count=0,
                    loss_weight=0.25,
                )
            ],
        )
    ]

    targets = compile_stage2_trie_targets(candidates, label_position_start=1)

    assert targets.summary.fallback_candidate_count == 1
    assert targets.summary.fallback_loss_weight_sum == pytest.approx(0.25)
    assert targets.span_score_records[0].span_role == "fallback_fn"
    assert targets.span_score_records[0].loss_weight == pytest.approx(0.25)


def test_compile_stage2_trie_assigns_semantic_roles_from_position_map() -> None:
    candidates = [
        Stage2TrieCandidate(
            sample_id="img-roles",
            rollout_index=0,
            source="valid_rollout",
            token_ids=[101, 21, 102],
            loss_weight=1.0,
            object_spans=[],
        )
    ]

    targets = compile_stage2_trie_targets(
        candidates,
        label_position_start=5,
        semantic_role_by_position={5: "struct", 6: "desc"},
    )

    targets_by_position = {target.position: target for target in targets.token_targets}
    assert targets_by_position[5].semantic_role == "struct"
    assert targets_by_position[6].semantic_role == "desc"
    assert targets_by_position[7].semantic_role == "text"


def test_compile_stage2_trie_merges_extra_terminal_target_deterministically() -> None:
    candidates = [
        Stage2TrieCandidate(
            sample_id="img-eos",
            rollout_index=0,
            source="valid_rollout",
            token_ids=[101, 21],
            loss_weight=1.0,
            object_spans=[],
        )
    ]
    terminal = Stage2TrieTokenTarget(
        position=7,
        positive_token_ids=(999,),
        source_weights=(1.0,),
        semantic_role="eos",
    )

    targets = compile_stage2_trie_targets(
        candidates,
        label_position_start=5,
        semantic_role_by_position={5: "struct", 6: "coord"},
        extra_token_targets=[terminal],
    )

    assert [
        (target.position, target.positive_token_ids, target.semantic_role)
        for target in targets.token_targets
    ] == [
        (5, (101,), "struct"),
        (6, (21,), "coord"),
        (7, (999,), "eos"),
    ]
    assert targets.summary.target_positions == 3


def test_compile_stage2_trie_extra_target_role_precedence_and_weight_merge() -> None:
    candidates = [
        Stage2TrieCandidate(
            sample_id="img-merge",
            rollout_index=-1,
            source="fallback_gt_fn_append_only",
            token_ids=[101, 21],
            loss_weight=0.25,
            object_spans=[],
        )
    ]
    terminal = Stage2TrieTokenTarget(
        position=6,
        positive_token_ids=(21, 999),
        source_weights=(0.5, 0.25),
        semantic_role="eos",
    )

    targets = compile_stage2_trie_targets(
        candidates,
        label_position_start=5,
        semantic_role_by_position={6: "desc"},
        extra_token_targets=[terminal],
    )

    target = {target.position: target for target in targets.token_targets}[6]
    assert target.positive_token_ids == (21, 999)
    assert target.source_weights == pytest.approx((0.5, 0.25))
    assert target.semantic_role == "eos"


def test_compile_stage2_trie_rollout_group_merges_candidates_and_fallback() -> None:
    candidates = [
        Stage2TrieCandidate(
            sample_id="img-group",
            rollout_index=0,
            source="valid_rollout",
            token_ids=[101, 11, 12, 102],
            loss_weight=1.0,
            object_spans=[],
        ),
        Stage2TrieCandidate(
            sample_id="img-group",
            rollout_index=1,
            source="valid_rollout",
            token_ids=[101, 11, 13, 102],
            loss_weight=1.0,
            object_spans=[],
        ),
        Stage2TrieCandidate(
            sample_id="img-group",
            rollout_index=-1,
            source="fallback_gt_fn_append_only",
            token_ids=[101, 11, 14, 102],
            loss_weight=0.25,
            object_spans=[
                Stage2TrieObjectSpan(
                    role="fallback_fn",
                    token_start=2,
                    token_end=3,
                    object_iou=None,
                    support_count=0,
                    loss_weight=0.25,
                )
            ],
        ),
    ]

    targets = compile_stage2_trie_targets_for_rollout_group(
        candidates,
        label_position_start=7,
    )

    targets_by_position = {target.position: target for target in targets.token_targets}
    assert targets_by_position[9].positive_token_ids == (12, 13, 14)
    assert targets_by_position[9].source_weights == (1.0, 1.0, 0.25)
    assert targets.summary.candidate_count == 3
    assert targets.summary.fallback_candidate_count == 1
    assert targets.summary.fallback_loss_weight_sum == pytest.approx(0.25)
    assert targets.summary.branch_points == 1
    assert targets.summary.max_branching_factor == 3
    assert targets.span_score_records[0].sample_id == "img-group"
    assert targets.span_score_records[0].span_role == "fallback_fn"


def test_compile_stage2_trie_rollout_group_rejects_empty_candidates() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        compile_stage2_trie_targets_for_rollout_group([], label_position_start=1)


def test_compile_stage2_trie_rollout_group_rejects_mixed_sample_ids() -> None:
    candidates = [
        Stage2TrieCandidate(
            sample_id="img-1",
            rollout_index=0,
            source="valid_rollout",
            token_ids=[101],
            loss_weight=1.0,
            object_spans=[],
        ),
        Stage2TrieCandidate(
            sample_id="img-2",
            rollout_index=1,
            source="valid_rollout",
            token_ids=[102],
            loss_weight=1.0,
            object_spans=[],
        ),
    ]

    with pytest.raises(ValueError, match="share exactly one sample_id"):
        compile_stage2_trie_targets_for_rollout_group(
            candidates,
            label_position_start=1,
        )


def test_compile_stage2_trie_stops_candidate_after_teacher_prefix_divergence() -> None:
    candidates = [
        Stage2TrieCandidate(
            sample_id="img-1",
            rollout_index=0,
            source="valid_rollout",
            token_ids=[101, 11, 12, 102],
            loss_weight=1.0,
            object_spans=[],
        ),
        Stage2TrieCandidate(
            sample_id="img-1",
            rollout_index=1,
            source="valid_rollout",
            token_ids=[101, 99, 13, 103],
            loss_weight=1.0,
            object_spans=[],
        ),
    ]

    targets = compile_stage2_trie_targets(candidates, label_position_start=5)

    targets_by_position = {target.position: target for target in targets.token_targets}
    assert targets_by_position[5].positive_token_ids == (101,)
    assert targets_by_position[6].positive_token_ids == (11, 99)
    assert targets_by_position[7].positive_token_ids == (12,)
    assert targets_by_position[8].positive_token_ids == (102,)
    assert targets.summary.branch_points == 1
    assert targets.summary.max_branching_factor == 2


def test_compile_stage2_trie_keeps_identical_prefix_alternatives_branching() -> None:
    candidates = [
        Stage2TrieCandidate(
            sample_id="img-1",
            rollout_index=0,
            source="valid_rollout",
            token_ids=[101, 11, 12, 102],
            loss_weight=1.0,
            object_spans=[],
        ),
        Stage2TrieCandidate(
            sample_id="img-1",
            rollout_index=1,
            source="valid_rollout",
            token_ids=[101, 11, 13, 103],
            loss_weight=0.5,
            object_spans=[],
        ),
    ]

    targets = compile_stage2_trie_targets(candidates, label_position_start=5)

    targets_by_position = {target.position: target for target in targets.token_targets}
    assert targets_by_position[6].positive_token_ids == (11,)
    assert targets_by_position[7].positive_token_ids == (12, 13)
    assert targets_by_position[7].source_weights == pytest.approx((1.0, 0.5))
    assert targets_by_position[8].positive_token_ids == (102,)
    assert targets.summary.branch_points == 1
    assert targets.summary.max_branching_factor == 2


def test_compile_stage2_trie_records_weak_fp_span_weight() -> None:
    candidates = [
        Stage2TrieCandidate(
            sample_id="img-1",
            rollout_index=2,
            source="valid_rollout",
            token_ids=[101, 31, 32, 102],
            loss_weight=1.0,
            object_spans=[
                Stage2TrieObjectSpan(
                    role="weak_positive_fp",
                    token_start=1,
                    token_end=3,
                    object_iou=None,
                    support_count=2,
                    loss_weight=0.05,
                )
            ],
        )
    ]

    targets = compile_stage2_trie_targets(candidates, label_position_start=10)

    assert targets.summary.weak_positive_fp_count == 1
    assert targets.span_score_records[0].token_start == 11
    assert targets.span_score_records[0].token_end == 13
    assert targets.span_score_records[0].loss_weight == pytest.approx(0.05)


def test_compile_stage2_trie_neutral_fp_context_removes_covered_positive_tokens() -> None:
    candidates = [
        Stage2TrieCandidate(
            sample_id="img-1",
            rollout_index=0,
            source="valid_rollout",
            token_ids=[101, 31, 32, 102],
            loss_weight=1.0,
            object_spans=[
                Stage2TrieObjectSpan(
                    role="neutral_fp",
                    token_start=1,
                    token_end=3,
                    object_iou=None,
                    support_count=1,
                    loss_weight=0.0,
                )
            ],
        )
    ]

    targets = compile_stage2_trie_targets(candidates, label_position_start=10)

    targets_by_position = {target.position: target for target in targets.token_targets}
    assert sorted(targets_by_position) == [10, 13]
    assert targets_by_position[10].positive_token_ids == (101,)
    assert targets_by_position[13].positive_token_ids == (102,)
    assert targets.summary.weak_positive_fp_count == 0


def test_compile_stage2_trie_weak_positive_fp_context_downweights_covered_tokens() -> None:
    candidates = [
        Stage2TrieCandidate(
            sample_id="img-1",
            rollout_index=0,
            source="valid_rollout",
            token_ids=[101, 31, 32, 102],
            loss_weight=1.0,
            object_spans=[
                Stage2TrieObjectSpan(
                    role="weak_positive_fp",
                    token_start=1,
                    token_end=3,
                    object_iou=None,
                    support_count=2,
                    loss_weight=0.05,
                )
            ],
        )
    ]

    targets = compile_stage2_trie_targets(candidates, label_position_start=10)

    targets_by_position = {target.position: target for target in targets.token_targets}
    assert sorted(targets_by_position) == [10, 11, 12, 13]
    assert targets_by_position[11].positive_token_ids == (31,)
    assert targets_by_position[11].source_weights == pytest.approx((0.05,))
    assert targets_by_position[12].positive_token_ids == (32,)
    assert targets_by_position[12].source_weights == pytest.approx((0.05,))
    assert targets.summary.weak_positive_fp_count == 1


def test_compile_stage2_trie_nested_neutral_fp_allows_marker_only_weak_positive() -> None:
    candidates = [
        Stage2TrieCandidate(
            sample_id="img-1",
            rollout_index=0,
            source="valid_rollout",
            token_ids=[101, 31, 32, 33, 102],
            loss_weight=1.0,
            object_spans=[
                Stage2TrieObjectSpan(
                    role="neutral_fp",
                    token_start=1,
                    token_end=4,
                    object_iou=None,
                    support_count=2,
                    loss_weight=0.0,
                ),
                Stage2TrieObjectSpan(
                    role="weak_positive_fp",
                    token_start=1,
                    token_end=2,
                    object_iou=None,
                    support_count=2,
                    loss_weight=0.05,
                ),
            ],
        )
    ]

    targets = compile_stage2_trie_targets(candidates, label_position_start=10)

    targets_by_position = {target.position: target for target in targets.token_targets}
    assert sorted(targets_by_position) == [10, 11, 14]
    assert targets_by_position[11].positive_token_ids == (31,)
    assert targets_by_position[11].source_weights == pytest.approx((0.05,))
    assert targets.summary.weak_positive_fp_count == 1


def test_build_fp_object_span_zero_loss_context_returns_neutral_span() -> None:
    span = build_fp_object_span(
        token_start=4,
        token_end=7,
        policy_mode="zero_loss_context",
        support_count=3,
        weak_positive_weight=0.05,
    )

    assert span.role == "neutral_fp"
    assert span.loss_weight == pytest.approx(0.0)
    assert span.support_count == 3


def test_build_fp_object_span_weak_positive_requires_support_count() -> None:
    supported = build_fp_object_span(
        token_start=1,
        token_end=2,
        policy_mode="weak_positive_context",
        support_count=1,
        weak_positive_weight=0.05,
    )
    unsupported = build_fp_object_span(
        token_start=1,
        token_end=2,
        policy_mode="weak_positive_context",
        support_count=0,
        weak_positive_weight=0.05,
    )

    assert supported.role == "weak_positive_fp"
    assert supported.loss_weight == pytest.approx(0.05)
    assert unsupported.role == "neutral_fp"
    assert unsupported.loss_weight == pytest.approx(0.0)


def test_stage2_trie_span_score_record_to_json_is_serializable() -> None:
    candidates = [
        Stage2TrieCandidate(
            sample_id="img-1",
            rollout_index=0,
            source="valid_rollout",
            token_ids=[101, 41, 102],
            loss_weight=1.0,
            object_spans=[
                Stage2TrieObjectSpan(
                    role="matched_clean",
                    token_start=1,
                    token_end=2,
                    object_iou=0.75,
                    support_count=4,
                    loss_weight=1.0,
                )
            ],
        )
    ]
    targets = compile_stage2_trie_targets(candidates, label_position_start=5)

    payload = stage2_trie_span_score_record_to_json(targets.span_score_records[0])

    assert payload["sample_id"] == "img-1"
    assert payload["candidate_source"] == "valid_rollout"
    assert payload["span_role"] == "matched_clean"
    assert payload["object_role"] == "matched_clean"
    assert payload["token_start"] == 6
    assert payload["token_end"] == 7
    assert payload["mean_token_logprob"] is None
    assert payload["min_token_logprob"] is None
    assert payload["object_iou"] == pytest.approx(0.75)
    json.dumps(payload, allow_nan=False)


def test_compile_stage2_trie_allows_empty_input_at_position_zero() -> None:
    targets = compile_stage2_trie_targets([], label_position_start=0)

    assert targets.token_targets == ()
    assert targets.span_score_records == ()
    assert targets.summary.candidate_count == 0
    assert targets.summary.target_positions == 0
    assert targets.summary.max_branching_factor == 0


def test_compile_stage2_trie_rejects_position_zero_for_non_empty_candidates() -> None:
    candidates = [
        Stage2TrieCandidate(
            sample_id="img-1",
            rollout_index=0,
            source="valid_rollout",
            token_ids=[101],
            loss_weight=1.0,
            object_spans=[],
        )
    ]

    with pytest.raises(ValueError, match="label_position_start.*> 0"):
        compile_stage2_trie_targets(candidates, label_position_start=0)


def test_stage2_trie_candidate_stores_sequence_fields_as_tuples() -> None:
    span = Stage2TrieObjectSpan(
        role="matched_clean",
        token_start=0,
        token_end=1,
        object_iou=0.5,
        support_count=1,
        loss_weight=1.0,
    )

    candidate = Stage2TrieCandidate(
        sample_id="img-1",
        rollout_index=0,
        source="valid_rollout",
        token_ids=[101, 102],
        loss_weight=1.0,
        object_spans=[span],
    )

    assert candidate.token_ids == (101, 102)
    assert candidate.object_spans == (span,)


@pytest.mark.parametrize(
    ("sample_id", "rollout_index", "error_type", "match"),
    [
        (1, 0, TypeError, "sample_id"),
        ("img-1", True, TypeError, "rollout_index"),
    ],
)
def test_stage2_trie_candidate_rejects_bad_identity_fields(
    sample_id: object,
    rollout_index: object,
    error_type: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error_type, match=match):
        Stage2TrieCandidate(
            sample_id=sample_id,  # type: ignore[arg-type]
            rollout_index=rollout_index,  # type: ignore[arg-type]
            source="valid_rollout",
            token_ids=[101],
            loss_weight=1.0,
            object_spans=[],
        )


def test_validation_rejects_invalid_source_and_role() -> None:
    with pytest.raises(ValueError, match="source"):
        Stage2TrieCandidate(
            sample_id="img-1",
            rollout_index=0,
            source="invalid_source",
            token_ids=[101],
            loss_weight=1.0,
            object_spans=[],
        )

    with pytest.raises(ValueError, match="role"):
        Stage2TrieObjectSpan(
            role="bad_role",
            token_start=0,
            token_end=1,
            object_iou=None,
            support_count=0,
            loss_weight=1.0,
        )


@pytest.mark.parametrize("token_ids", ([True], [-1], [1.5], [], ["101"]))
def test_validation_rejects_bad_candidate_token_ids(token_ids: list[object]) -> None:
    error_type = ValueError if token_ids in ([-1], []) else TypeError

    with pytest.raises(error_type):
        Stage2TrieCandidate(
            sample_id="img-1",
            rollout_index=0,
            source="valid_rollout",
            token_ids=token_ids,
            loss_weight=1.0,
            object_spans=[],
        )


@pytest.mark.parametrize("bad_weight", (float("nan"), float("inf"), -0.1))
def test_validation_rejects_non_finite_or_negative_weights(bad_weight: float) -> None:
    with pytest.raises(ValueError, match="finite|non-negative"):
        Stage2TrieCandidate(
            sample_id="img-1",
            rollout_index=0,
            source="valid_rollout",
            token_ids=[101],
            loss_weight=bad_weight,
            object_spans=[],
        )

    with pytest.raises(ValueError, match="finite|non-negative"):
        Stage2TrieObjectSpan(
            role="matched_clean",
            token_start=0,
            token_end=1,
            object_iou=None,
            support_count=0,
            loss_weight=bad_weight,
        )


def test_validation_rejects_malformed_token_targets() -> None:
    with pytest.raises(ValueError, match="position.*> 0"):
        Stage2TrieTokenTarget(
            position=0,
            positive_token_ids=(101,),
            source_weights=(1.0,),
            semantic_role="text",
        )

    with pytest.raises(ValueError, match="non-empty"):
        Stage2TrieTokenTarget(
            position=1,
            positive_token_ids=(),
            source_weights=(),
            semantic_role="text",
        )

    with pytest.raises(ValueError, match="equal lengths"):
        Stage2TrieTokenTarget(
            position=1,
            positive_token_ids=(101,),
            source_weights=(1.0, 0.5),
            semantic_role="text",
        )

    with pytest.raises(ValueError, match="unique"):
        Stage2TrieTokenTarget(
            position=1,
            positive_token_ids=(101, 101),
            source_weights=(1.0, 0.5),
            semantic_role="text",
        )

    with pytest.raises(TypeError, match="positive_token_ids"):
        Stage2TrieTokenTarget(
            position=1,
            positive_token_ids=(True,),
            source_weights=(1.0,),
            semantic_role="text",
        )

    with pytest.raises(ValueError, match="finite"):
        Stage2TrieTokenTarget(
            position=1,
            positive_token_ids=(101,),
            source_weights=(float("nan"),),
            semantic_role="text",
        )


@pytest.mark.parametrize("bad_value", (float("nan"), float("inf")))
def test_span_score_record_rejects_non_finite_json_fields(bad_value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        Stage2TrieSpanScoreRecord(
            sample_id="img-1",
            rollout_index=0,
            candidate_source="valid_rollout",
            span_role="matched_clean",
            object_role="matched_clean",
            token_start=1,
            token_end=2,
            mean_token_logprob=bad_value,
            min_token_logprob=None,
            object_iou=None,
            support_count=0,
            loss_weight=1.0,
        )

    with pytest.raises(ValueError, match="finite"):
        Stage2TrieSpanScoreRecord(
            sample_id="img-1",
            rollout_index=0,
            candidate_source="valid_rollout",
            span_role="matched_clean",
            object_role="matched_clean",
            token_start=1,
            token_end=2,
            mean_token_logprob=None,
            min_token_logprob=None,
            object_iou=bad_value,
            support_count=0,
            loss_weight=1.0,
        )


def test_span_score_record_rejects_invalid_object_role() -> None:
    with pytest.raises(ValueError, match="object_role"):
        Stage2TrieSpanScoreRecord(
            sample_id="img-1",
            rollout_index=0,
            candidate_source="valid_rollout",
            span_role="matched_clean",
            object_role="bad_role",
            token_start=1,
            token_end=2,
            mean_token_logprob=None,
            min_token_logprob=None,
            object_iou=None,
            support_count=0,
            loss_weight=1.0,
        )


def test_span_score_record_rejects_non_string_sample_id() -> None:
    with pytest.raises(TypeError, match="sample_id"):
        Stage2TrieSpanScoreRecord(
            sample_id=123,
            rollout_index=0,
            candidate_source="valid_rollout",
            span_role="matched_clean",
            object_role="matched_clean",
            token_start=1,
            token_end=2,
            mean_token_logprob=None,
            min_token_logprob=None,
            object_iou=None,
            support_count=0,
            loss_weight=1.0,
        )


@pytest.mark.parametrize("bad_rollout_index", (True, 1.5, "0"))
def test_span_score_record_rejects_bool_or_non_int_rollout_index(
    bad_rollout_index: object,
) -> None:
    with pytest.raises(TypeError, match="rollout_index"):
        Stage2TrieSpanScoreRecord(
            sample_id="img-1",
            rollout_index=bad_rollout_index,
            candidate_source="valid_rollout",
            span_role="matched_clean",
            object_role="matched_clean",
            token_start=1,
            token_end=2,
            mean_token_logprob=None,
            min_token_logprob=None,
            object_iou=None,
            support_count=0,
            loss_weight=1.0,
        )


def test_span_score_record_allows_negative_fallback_rollout_index() -> None:
    record = Stage2TrieSpanScoreRecord(
        sample_id="img-1",
        rollout_index=-1,
        candidate_source="fallback_gt_fn_append_only",
        span_role="fallback_fn",
        object_role="fallback_fn",
        token_start=1,
        token_end=2,
        mean_token_logprob=None,
        min_token_logprob=None,
        object_iou=None,
        support_count=0,
        loss_weight=0.25,
    )

    payload = stage2_trie_span_score_record_to_json(record)

    assert payload["rollout_index"] == -1
    json.dumps(payload, allow_nan=False)


def test_validation_rejects_unknown_fp_policy_and_invalid_spans() -> None:
    with pytest.raises(ValueError, match="zero_loss_context|weak_positive_context"):
        build_fp_object_span(
            token_start=1,
            token_end=2,
            policy_mode="score_threshold_context",
            support_count=0,
            weak_positive_weight=0.05,
        )

    with pytest.raises(ValueError, match="token_start <= token_end"):
        Stage2TrieObjectSpan(
            role="matched_clean",
            token_start=3,
            token_end=2,
            object_iou=None,
            support_count=0,
            loss_weight=1.0,
        )

    with pytest.raises(ValueError, match="non-negative"):
        compile_stage2_trie_targets([], label_position_start=-1)
