from __future__ import annotations

from pathlib import Path

from scripts.research import run_person25_commit_closeout as probe


def _sample(outcome: str, *, owner: int | None = None, iou: float = 0.0, mode: str = "sample"):
    matched = owner is not None
    return {
        "decode_mode": mode,
        "outcome": outcome,
        "person25_iou": iou,
        "parsed": {"status": "row"},
        "match": {
            "status": "matched" if matched else "unmatched",
            "matched_person_only_rank": owner,
        },
        "nearest_person_rank": owner,
    }


def test_common_seed_zero_donor_is_token_identical_across_histories():
    receipt = probe.common_donor_attestation(probe.DEFAULT_SOURCE_ROOT, seed=0)
    assert receipt["common_across_history_owners"] is True
    assert receipt["history_owners"] == [2, 3, 4, 14]
    assert len(receipt["raw_token_ids"]) == 7


def test_summarize_samples_keeps_strict_match_and_overlap_separate():
    samples = [
        _sample("person25_repeat", owner=25, iou=0.9),
        _sample("unresolved_geometry", owner=None, iou=0.49),
        _sample("other_uncovered_person", owner=6, iou=0.0),
        _sample("person25_repeat", owner=25, iou=0.95, mode="greedy"),
    ]
    summary = probe.summarize_samples(samples)
    assert summary["sample_count"] == 3
    assert summary["valid_row_count"] == 3
    assert summary["strict_person25_recurrence_count"] == 1
    assert summary["outcome_counts"]["unresolved_geometry"] == 1
    assert summary["matched_person_owner_counts"] == {"6": 1, "25": 1}
    assert summary["greedy"]["outcome"] == "person25_repeat"


def test_immediate_gate_promotes_only_consistent_suppression():
    control = {
        "arm_name": "control-person3",
        "summary": {"strict_person25_recurrence_count": 24, "valid_row_count": 24},
    }
    treatments = [
        {
            "arm_name": f"donor-seed-{seed}",
            "summary": {"strict_person25_recurrence_count": count, "valid_row_count": 24},
        }
        for seed, count in zip(probe.DEFAULT_DONOR_SEEDS, (0, 2, 4))
    ]
    result = probe.evaluate_immediate_gate([control, *treatments])
    assert result["status"] == "immediate_suppression_count_gate_passed_pending_overlap_review"
    assert result["conditional_common_final_history_panel_authorized"] is True


def test_immediate_gate_stops_when_person25_remains_dominant():
    arms = [{
        "arm_name": "control-person3",
        "summary": {"strict_person25_recurrence_count": 24, "valid_row_count": 24},
    }]
    arms.extend({
        "arm_name": f"donor-seed-{seed}",
        "summary": {"strict_person25_recurrence_count": 22, "valid_row_count": 24},
    } for seed in probe.DEFAULT_DONOR_SEEDS)
    result = probe.evaluate_immediate_gate(arms)
    assert result["status"] == "no_reliable_dominant_owner_commit"
    assert result["conditional_common_final_history_panel_authorized"] is False


def test_source_variants_remain_raw_seven_token_rows():
    for seed in probe.DEFAULT_DONOR_SEEDS:
        sample = probe.source_sample(probe.DEFAULT_SOURCE_ROOT, owner_rank=2, seed=seed)
        assert len(sample["generated_token_ids"]) == 7
        assert sample["match"]["matched_person_only_rank"] == 25
        assert Path(sample["source_path"]).is_file()


def test_history_builder_accepts_observed_successor_owner_outside_original_panel():
    packet = probe.historical.select_image2299_rows()
    candidates, rows = probe.candidates_and_history_rows(packet, earlier_owner=18)
    assert len(rows) == 3
    assert [item["person_only_rank"] for item in candidates if item["emitted_person"]] == [0, 1, 18]
