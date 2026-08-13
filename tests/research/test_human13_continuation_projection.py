from __future__ import annotations

from dataclasses import replace
import math

import pytest

from scripts.research.build_human13_k_union_manifest import (
    ImageRecord,
    OwnerRecord,
    PredictionRowInput,
    PrefixRecord,
    RequestIdentity,
    SelectedRowRecord,
    TrajectoryRecord,
)
from scripts.research.build_human13_on_policy_frontier import (
    FrontierCandidateAlias,
    FrontierDuplicateEvent,
    FrontierImage,
    FrontierRow,
    natural_pre_stop_prefix,
)
from scripts.research.human13_continuation_projection import (
    project_continuation,
    select_projected_continuation,
)
from scripts.research.human13_forced_continuation import (
    ForcedContinuationResult,
    source_continuation_cap,
)
from scripts.research.human13_frontier_selection import (
    CandidatePath,
    CandidateScore,
)
from scripts.research.run_local_branch_causal_value import hash_prefix_token_ids


def _owner(
    owner_id: str,
    category: str,
    bbox: tuple[float, float, float, float],
    source_index: int,
    stratum: str,
) -> OwnerRecord:
    return OwnerRecord(
        owner_id=owner_id,
        category=category,
        bbox=bbox,
        source_object_index=source_index,
        stratum=stratum,  # type: ignore[arg-type]
        source_row_ids=(),
        sampled_row_ids=(),
    )


def _image() -> ImageRecord:
    owners = (
        _owner("g1", "person", (0.0, 0.0, 10.0, 10.0), 0, "G"),
        _owner("h1", "dog", (20.0, 0.0, 30.0, 10.0), 1, "H"),
        _owner("h2", "cat", (40.0, 0.0, 50.0, 10.0), 2, "H"),
    )
    selected = tuple(
        SelectedRowRecord(
            owner_id=owner_id,
            row_id=f"row-{owner_id}",
            trajectory_id=f"trajectory-{owner_id}",
            seed=index,
            row_index=0,
            owner_iou=1.0,
            token_ids=tokens,
            target_token_mask=tuple(True for _ in tokens),
        )
        for index, (owner_id, tokens) in enumerate(
            (("h1", (101, 102)), ("h2", (201, 202))), start=1
        )
    )
    source = TrajectoryRecord(
        trajectory_id="source",
        request=RequestIdentity("hf", "test", "source_greedy", 1, None, 0, 0.0, 1.0, 1.0, 64),
        raw_token_ids=(1, 2, 3, 4, 5, 99),
        terminal_token_index=5,
        stop_reason="im_end",
        parser_status="accepted",
        rows=(
            PredictionRowInput("source:row:0", 0, "person", (0.0, 0.0, 10.0, 10.0), 0, 2, 1),
            PredictionRowInput("source:row:1", 1, "person", (60.0, 0.0, 70.0, 10.0), 2, 5, 4),
        ),
        prefix=PrefixRecord((1, 2, 3, 4, 5), (1, 2, 3, 4, 5), ()),
        retained_row_ids=("source:row:0", "source:row:1"),
        duplicate_row_ids=(),
        matched_row_ids=("source:row:0",),
        replay_token_mask=(True, True, True, True, True, False),
        duplicate_target_mask=(False, False, False, False, False, False),
    )
    return ImageRecord(
        image_id=7,
        panel_row_sha256=None,
        image_sha256=None,
        owners=owners,
        trajectories=(source,),
        duplicate_events=(),
        selected_rows=selected,
        g_owner_ids=("g1",),
        h_owner_ids=("h1", "h2"),
        m_owner_ids=(),
        replay_row_ids=(),
        target_row_ids=tuple(row.row_id for row in selected),
        candidate_row_ids=tuple(row.row_id for row in selected),
    )


def _frontier() -> FrontierImage:
    rows = (
        FrontierRow(0, "person", (0.0, 0.0, 10.0, 10.0), 0, 2, (11, 12)),
        FrontierRow(1, "different label", (0.1, 0.0, 10.1, 10.0), 2, 4, (13, 14)),
    )
    aliases = (
        FrontierCandidateAlias("h1", "row-h1", "trajectory-h1", 1, 1.0, (101, 102)),
        FrontierCandidateAlias("h2", "row-h2", "trajectory-h2", 2, 1.0, (201, 202)),
    )
    return FrontierImage(
        image_id=7,
        trajectory_id="natural",
        generated_token_ids=(11, 12, 13, 14, 99),
        parser="compact_object_box_closed_v1",
        parser_status="complete",
        stop_reason="natural_im_end",
        rows=rows,
        canonical_owner_ids=("g1",),
        constrained_protected_owner_ids=("g1",),
        covered_h_owner_ids=(),
        uncovered_h_owner_ids=("h1", "h2"),
        candidate_aliases=aliases,
        duplicate_events=(FrontierDuplicateEvent(1, 0, 0.98),),
        terminal_token_index=4,
        malformed_row_count=0,
    )


def _score(owner_id: str = "h1", barrier: float = 2.0) -> CandidateScore:
    tokens = (101, 102) if owner_id == "h1" else (201, 202)
    return CandidateScore(
        path=CandidatePath(7, owner_id, f"row-{owner_id}", tokens),
        hf_sites=(),
        packed_sites=(),
        aligned_sites=(),
        hf_barrier=barrier,
        first_bottleneck_index=0,
        packed_first_bottleneck_index=None,
        first_bottleneck_disagreement=None,
        packed_barrier=None,
        max_surface_margin_drift=None,
        surface_rank_disagreement=None,
    )


def _prediction(
    category: str,
    bbox: tuple[float, float, float, float],
    *,
    generated_order: int = 0,
) -> dict:
    return {
        "description": category,
        "bbox": list(bbox),
        "generated_order": generated_order,
    }


def _parse(*predictions: dict, dropped: tuple[dict, ...] = ()) -> dict:
    if predictions and dropped:
        status = "accepted_with_drops"
    elif predictions:
        status = "accepted"
    elif dropped:
        status = "all_spans_dropped"
    else:
        status = "empty"
    return {
        "parse_status": status,
        "predictions": list(predictions),
        "dropped_predictions": list(dropped),
    }


def _result(
    *,
    owner_id: str = "h1",
    released_predictions: tuple[dict, ...] = (),
    dropped: tuple[dict, ...] = (),
    released_tokens: tuple[int, ...] = (301, 302, 99),
    termination_status: str = "natural_im_end",
    cap_hit: bool = False,
) -> ForcedContinuationResult:
    forced = (
        _prediction("dog", (20.0, 0.0, 30.0, 10.0))
        if owner_id == "h1"
        else _prediction("cat", (40.0, 0.0, 50.0, 10.0))
    )
    forced_tokens = (101, 102) if owner_id == "h1" else (201, 202)
    natural_prefix = natural_pre_stop_prefix(_frontier())
    minimum_cap = source_continuation_cap(source_row_count=2, source_token_count=6)
    return ForcedContinuationResult(
        forced_row_token_ids=forced_tokens,
        released_token_ids=released_tokens,
        termination_status=termination_status,
        cap_hit=cap_hit,
        generated_text="ignored by pure projection",
        forced_row_parse_evidence=_parse(forced),
        parse_evidence=_parse(*released_predictions, dropped=dropped),
        natural_prefix_token_ids_sha256=hash_prefix_token_ids(natural_prefix),
        forced_row_token_ids_sha256=hash_prefix_token_ids(forced_tokens),
        forced_context_sha256=hash_prefix_token_ids((*natural_prefix, *forced_tokens)),
        released_token_ids_sha256=hash_prefix_token_ids(released_tokens),
        requested_continuation_cap=minimum_cap,
        minimum_continuation_cap=minimum_cap,
        repetition_penalty=1.0,
        current_checkpoint_payload_sha256="c" * 64,
    )


def _project(
    score: CandidateScore,
    result: ForcedContinuationResult,
    *,
    expected_continuation_cap: int = 518,
    expected_repetition_penalty: float = 1.0,
    current_checkpoint_payload_sha256: str = "c" * 64,
):
    return project_continuation(
        _image(),
        _frontier(),
        score,
        result,
        expected_continuation_cap=expected_continuation_cap,
        expected_repetition_penalty=expected_repetition_penalty,
        current_checkpoint_payload_sha256=current_checkpoint_payload_sha256,
    )


def test_projection_composes_branch_without_awarding_duplicate_owner_credit() -> None:
    released = (
        _prediction("cat", (40.0, 0.0, 50.0, 10.0)),
        _prediction(
            "anything",
            (20.05, 0.0, 30.05, 10.0),
            generated_order=1,
        ),
    )
    projection = _project(
        _score(),
        _result(released_predictions=released, dropped=({"reason": "junk"},)),
    )

    assert projection.matched_owner_ids == ("g1", "h1", "h2")
    assert [row.origin for row in projection.full_branch_predictions] == [
        "current",
        "current",
        "forced",
        "released",
        "released",
    ]
    assert projection.retained_prediction_indices == (0, 2, 3)
    assert projection.duplicate_prediction_indices == (1, 4)
    assert projection.duplicate_count == 2
    assert projection.duplicate_increase == 1
    assert projection.malformed_count == 1
    assert projection.row_count == 5
    assert projection.generated_tokens == 5
    assert projection.outcome.protected_coverable is True
    assert projection.outcome.unique_owner_delta == 2
    assert projection.outcome.duplicate_increase == 1
    assert projection.outcome.malformed_increase == 1


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda result: replace(
                result,
                forced_row_parse_evidence=_parse(
                    _prediction("dog", (20.0, 0.0, 30.0, 10.0)),
                    _prediction(
                        "dog",
                        (20.0, 0.0, 30.0, 10.0),
                        generated_order=1,
                    ),
                ),
            ),
            "exactly one",
        ),
        (
            lambda result: replace(
                result,
                forced_row_parse_evidence=_parse(
                    _prediction("dog", (20.0, 0.0, math.nan, 10.0))
                ),
            ),
            "finite|rectangle",
        ),
        (
            lambda result: replace(result, forced_row_token_ids=(999,)),
            "forced.*candidate",
        ),
    ),
)
def test_projection_fails_closed_on_invalid_forced_row(mutation, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _project(_score(), mutation(_result()))


@pytest.mark.parametrize(
    ("mutation", "kwargs", "message"),
    (
        (lambda result: replace(result, natural_prefix_token_ids_sha256="d" * 64), {}, "natural prefix"),
        (lambda result: replace(result, forced_row_token_ids_sha256="d" * 64), {}, "forced row"),
        (lambda result: replace(result, forced_context_sha256="d" * 64), {}, "forced context"),
        (lambda result: replace(result, released_token_ids_sha256="d" * 64), {}, "released"),
        (lambda result: replace(result, requested_continuation_cap=517), {}, "cap"),
        (lambda result: result, {"expected_continuation_cap": 519}, "cap"),
        (lambda result: replace(result, minimum_continuation_cap=519), {}, "minimum"),
        (
            lambda result: replace(
                result,
                termination_status="cap_hit",
                cap_hit=True,
            ),
            {},
            "cap status",
        ),
        (lambda result: replace(result, repetition_penalty=1.1), {}, "repetition"),
        (lambda result: result, {"expected_repetition_penalty": 1.1}, "repetition"),
        (lambda result: replace(result, current_checkpoint_payload_sha256="d" * 64), {}, "checkpoint"),
        (lambda result: result, {"current_checkpoint_payload_sha256": "d" * 64}, "checkpoint"),
    ),
)
def test_projection_fails_closed_on_continuation_binding_mismatch(
    mutation, kwargs: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _project(_score(), mutation(_result()), **kwargs)


def test_projection_rejects_forged_frontier_alias_not_bound_to_manifest() -> None:
    frontier = _frontier()
    forged_alias = replace(frontier.candidate_aliases[0], token_ids=(777,))
    frontier = replace(
        frontier,
        candidate_aliases=(forged_alias, frontier.candidate_aliases[1]),
    )
    score = replace(
        _score(),
        path=replace(_score().path, token_ids=(777,)),
    )
    result = replace(_result(), forced_row_token_ids=(777,))

    with pytest.raises(ValueError, match="manifest"):
        project_continuation(
            _image(),
            frontier,
            score,
            result,
            expected_continuation_cap=518,
            expected_repetition_penalty=1.0,
            current_checkpoint_payload_sha256="c" * 64,
        )


def test_projection_rejects_nonchronological_released_parse_rows() -> None:
    released = (
        _prediction("cat", (40.0, 0.0, 50.0, 10.0)),
        _prediction("dog", (60.0, 0.0, 70.0, 10.0)),
    )
    with pytest.raises(ValueError, match="generated order"):
        _project(_score(), _result(released_predictions=released))


def test_select_projected_continuation_returns_the_score_and_result_pair() -> None:
    h1 = _project(
        _score("h1", barrier=4.0),
        _result(
            owner_id="h1",
            released_predictions=(
                _prediction("cat", (40.0, 0.0, 50.0, 10.0)),
            ),
        ),
    )
    h2 = _project(_score("h2", barrier=0.1), _result(owner_id="h2"))

    selected = select_projected_continuation((h2, h1))

    assert selected is h1
    assert selected.score is h1.score
    assert selected.result is h1.result
    assert len(selected.artifact_sha256) == 64


def test_selector_rejects_a_projection_whose_artifact_hash_was_forged() -> None:
    projection = _project(_score(), _result())
    with pytest.raises(ValueError, match="artifact"):
        select_projected_continuation(
            (replace(projection, artifact_sha256="d" * 64),)
        )
