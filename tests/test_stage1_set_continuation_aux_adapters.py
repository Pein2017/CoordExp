from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from src.trainers.stage1_set_continuation.aux_adapters import (
    AggregatedAuxResult,
    BBoxGeoAuxAdapter,
    BBoxGeoDecodedState,
    BBoxSizeAuxAdapter,
    BranchCandidateAuxState,
    CandidateAuxResult,
    CoordAuxAdapter,
    aggregate_candidate_aux_losses,
)


def _candidate_state(
    *, name: str = "cand-0", scored_valid: bool = True
) -> BranchCandidateAuxState:
    return BranchCandidateAuxState(
        candidate_name=name,
        scored_valid=scored_valid,
        coord_logits=torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        coord_logits_full=torch.tensor(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            dtype=torch.float32,
        ),
        coord_target_bins=torch.tensor([1, 0], dtype=torch.long),
        coord_slot_mask=torch.tensor([True, False], dtype=torch.bool),
    )


def test_coord_aux_enabled_without_helper_fails_fast() -> None:
    adapter = CoordAuxAdapter(enabled=True, helper=None)

    with pytest.raises(ValueError, match="coord aux adapter helper"):
        adapter.compute_candidate(_candidate_state())


def test_coord_aux_adapter_calls_helper_with_branch_local_masks() -> None:
    calls: list[dict[str, torch.Tensor]] = []

    def _helper(**kwargs):
        calls.append(kwargs)
        return torch.tensor(0.75, dtype=torch.float32)

    adapter = CoordAuxAdapter(enabled=True, helper=_helper)
    result = adapter.compute_candidate(_candidate_state())

    assert float(result.loss.detach().cpu().item()) == pytest.approx(0.75)
    assert len(calls) == 1
    assert torch.equal(calls[0]["coord_logits"], _candidate_state().coord_logits)
    assert torch.equal(
        calls[0]["coord_logits_full"],
        _candidate_state().coord_logits_full,
    )
    assert torch.equal(
        calls[0]["coord_target_bins"],
        _candidate_state().coord_target_bins,
    )
    assert torch.equal(calls[0]["coord_slot_mask"], _candidate_state().coord_slot_mask)


def test_bbox_geo_adapter_fails_fast_without_decoded_bbox_state() -> None:
    adapter = BBoxGeoAuxAdapter(enabled=True, helper=lambda **_: torch.tensor(1.0))

    with pytest.raises(ValueError, match="decoded bbox state"):
        adapter.compute_candidate(_candidate_state())


def test_bbox_size_aux_depends_on_bbox_geo_decoded_state() -> None:
    adapter = BBoxSizeAuxAdapter(enabled=True, helper=lambda **_: torch.tensor(1.0))

    with pytest.raises(ValueError, match="bbox_geo decoded state"):
        adapter.compute_candidate(_candidate_state())


def test_aux_aggregation_is_uniform_over_scored_valid_candidates_and_counts_skips() -> None:
    candidates = [
        _candidate_state(name="cand-a"),
        _candidate_state(name="cand-b"),
        _candidate_state(name="cand-c"),
        _candidate_state(name="cand-d", scored_valid=False),
    ]

    def _runner(candidate: BranchCandidateAuxState) -> CandidateAuxResult:
        if candidate.candidate_name == "cand-b":
            return CandidateAuxResult.skipped(
                torch.tensor(0.0, dtype=torch.float32),
                reason="missing geometry",
            )
        if candidate.candidate_name == "cand-c":
            return CandidateAuxResult(loss=torch.tensor(4.0, dtype=torch.float32))
        return CandidateAuxResult(loss=torch.tensor(2.0, dtype=torch.float32))

    result = aggregate_candidate_aux_losses(
        candidates=candidates,
        compute_candidate=_runner,
        metric_prefix="coord_aux",
    )

    assert isinstance(result, AggregatedAuxResult)
    assert float(result.loss.detach().cpu().item()) == pytest.approx(3.0)
    assert result.scored_valid_candidates == 3
    assert result.contributing_candidates == 2
    assert result.skipped_candidates == 1
    assert result.metrics["coord_aux/skipped_candidates"] == pytest.approx(1.0)


def test_responsibility_weighted_aux_is_rejected_in_v1() -> None:
    with pytest.raises(ValueError, match="responsibility-weighted"):
        CoordAuxAdapter(
            enabled=True,
            helper=lambda **_: torch.tensor(1.0),
            weighting_mode="responsibility_weighted",
        )


def test_bbox_size_aux_reads_bbox_geo_decoded_state() -> None:
    calls: list[dict[str, torch.Tensor | None]] = []

    def _helper(**kwargs):
        calls.append(kwargs)
        return torch.tensor(0.25, dtype=torch.float32)

    decoded = BBoxGeoDecodedState(
        pred_boxes_xyxy=torch.tensor([[0.1, 0.2, 0.4, 0.5]], dtype=torch.float32),
        target_boxes_xyxy=torch.tensor([[0.1, 0.2, 0.4, 0.5]], dtype=torch.float32),
        box_weights=torch.tensor([1.0], dtype=torch.float32),
    )
    adapter = BBoxSizeAuxAdapter(enabled=True, helper=_helper)
    result = adapter.compute_candidate(
        replace(_candidate_state(), bbox_geo_decoded_state=decoded)
    )

    assert float(result.loss.detach().cpu().item()) == pytest.approx(0.25)
    assert len(calls) == 1
    assert torch.equal(calls[0]["pred_boxes_xyxy"], decoded.pred_boxes_xyxy)
    assert torch.equal(calls[0]["target_boxes_xyxy"], decoded.target_boxes_xyxy)
    assert torch.equal(calls[0]["box_weights"], decoded.box_weights)
