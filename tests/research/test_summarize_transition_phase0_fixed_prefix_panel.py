from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.research.summarize_transition_phase0_fixed_prefix_panel import summarize


def _score_receipt(manifest_sha: str, *, margin: float, uncovered: float, covered: float):
    def candidate(candidate_id: str, role: str, value: float, is_covered: bool):
        return {
            "candidate_id": candidate_id,
            "role": role,
            "covered": is_covered,
            "owner": candidate_id,
            "category": "object",
            "full_row": {"count": 3, "sum": value - 0.5},
            "row_entry": {"count": 1, "sum": -0.5},
            "description": {"mean": -0.2},
            "geometry": {"mean": -0.3},
        }

    return {
        "schema_version": "complete_candidate_row_scoring.receipt.v1",
        "manifest": {"sha256": manifest_sha},
        "images": [
            {
                "image_id": "1",
                "boundaries": [
                    {
                        "boundary_id": "case-1",
                        "terminal_boundary": {"row_entry_minus_terminal": margin},
                        "candidate_scores": [
                            candidate("u", "gained_uncovered_owner", uncovered, False),
                            candidate("c", "covered_owner_control", covered, True),
                        ],
                    }
                ],
            }
        ],
    }


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_summarizes_boundary_routing_and_release(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    treatment = tmp_path / "treatment.json"
    releases = tmp_path / "releases"
    _write(source, _score_receipt("a" * 64, margin=-1.0, uncovered=-4.0, covered=-3.0))
    _write(treatment, _score_receipt("a" * 64, margin=0.5, uncovered=-2.0, covered=-3.5))
    release = {
        "schema_version": "transition_phase0_fixed_prefix_release.receipt.v1",
        "manifest": {"sha256": "a" * 64},
        "checkpoint": {"checkpoint_role": "transition-step36"},
        "release_plan": {
            "case_id": "case-1",
            "force_mode": "complete-description",
            "expected_owner_id": "u",
        },
        "released_row": {
            "status": "success",
            "raw_generated_text": "row",
            "strict_matched_owner_ids": ["u"],
            "uncovered_ledger_owner_ids": ["u"],
            "covered_prefix_owner_ids": [],
            "intended_owner_realized": True,
        },
    }
    _write(releases / "one.json", release)

    result = summarize(source, treatment, releases)

    assert result["aggregate"]["boundary_margin_increased_count"] == 1
    assert result["aggregate"]["source_stop_to_treatment_continue_count"] == 1
    assert result["aggregate"]["best_uncovered_vs_covered_gap_improved_count"] == 1
    assert result["aggregate"]["release"]["transition-step36"][
        "intended_owner_realized_count"
    ] == 1


def test_rejects_misaligned_score_manifests(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    treatment = tmp_path / "treatment.json"
    releases = tmp_path / "releases"
    releases.mkdir()
    _write(source, _score_receipt("a" * 64, margin=0.0, uncovered=-1.0, covered=-2.0))
    _write(treatment, _score_receipt("b" * 64, margin=0.0, uncovered=-1.0, covered=-2.0))

    with pytest.raises(ValueError, match="different manifests"):
        summarize(source, treatment, releases)
