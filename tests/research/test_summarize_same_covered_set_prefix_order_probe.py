from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.research.run_same_covered_set_prefix_order_probe import ARMS, SCHEMA_VERSION
from scripts.research.summarize_same_covered_set_prefix_order_probe import summarize


def _run(*, mode: str, seed: int | None, owner: str | None, token: int, status: str = "success") -> dict:
    matches = [] if owner is None else [{"status": "matched", "matched_entity_id": owner}]
    return {
        "mode": mode,
        "seed": seed,
        "status": status,
        "raw_generated_token_ids": [token],
        "raw_generated_token_ids_sha256": str(token),
        "raw_generated_text": str(token),
        "row_stop": {"stop_reason": "complete_row" if status == "success" else "failed"},
        "parse_evidence": {"parse_status": "accepted" if status == "success" else "not_run", "predictions": [], "dropped_predictions": []},
        "entity_matches": matches,
    }


def _payload(*, case_id: str = "case-1", seed_tokens: tuple[int, int] = (11, 21)) -> dict:
    ledger = [
        {"entity_id": "A", "bbox_norm1000": [0, 0, 100, 100]},
        {"entity_id": "B", "bbox_norm1000": [200, 0, 300, 100]},
        {"entity_id": "C", "bbox_norm1000": [400, 0, 500, 100]},
    ]
    arm_rows = {
        "a_then_b_then_c": [
            _run(mode="greedy", seed=None, owner="C", token=1),
            _run(mode="sample", seed=seed_tokens[0], owner="A", token=2),
            _run(mode="sample", seed=seed_tokens[1], owner=None, token=3),
        ],
        "b_then_a_then_c": [
            _run(mode="greedy", seed=None, owner="B", token=4),
            _run(mode="sample", seed=seed_tokens[0], owner="A", token=2),
            _run(mode="sample", seed=seed_tokens[1], owner="B", token=30),
        ],
        "b_then_c_coverage_control": [
            _run(mode="greedy", seed=None, owner="A", token=5),
            _run(mode="sample", seed=seed_tokens[0], owner="A", token=6),
            _run(mode="sample", seed=seed_tokens[1], owner="C", token=7),
        ],
    }
    arms = {
        arm_name: {
            "entity_ids": (["A", "B", "C"] if arm_name == "a_then_b_then_c" else ["B", "A", "C"] if arm_name == "b_then_a_then_c" else ["B", "C"]),
            "prefix_token_ids": [1],
            "prefix_token_ids_sha256": "x",
            "row_count": 3 if arm_name != "b_then_c_coverage_control" else 2,
            "runs": rows,
        }
        for arm_name, rows in arm_rows.items()
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "config": {},
        "cases": [{"case_id": case_id, "image_id": "image-1", "a_entity_id": "A", "b_entity_id": "B", "c_entity_id": "C", "entity_ledger": ledger, "invariants": {}, "arms": arms}],
    }


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_summary_counts_owners_coverage_activation_and_pair_agreement(tmp_path: Path) -> None:
    payload = _payload()
    # Real runner artifacts identify A/B/C through arm entity_ids; they do not
    # need to duplicate those fields at the case level.
    payload["cases"][0].pop("a_entity_id")
    payload["cases"][0].pop("b_entity_id")
    payload["cases"][0].pop("c_entity_id")
    result = summarize([_write(tmp_path / "one.json", payload)])
    case = result["cases"][0]
    abc = case["arms"]["a_then_b_then_c"]
    assert abc["physical_owner_counts"] == {"A": 1, "C": 1}
    assert abc["covered_owner_counts"] == {"A": 1, "C": 1}
    assert abc["uncovered_owner_counts"] == {}
    assert case["a_recurrence"]["a_then_b_then_c"]["a_owner_run_count"] == 1
    assert case["b_then_c_activation_of_a"]["a_owner_run_count"] == 2
    assert case["b_then_c_activation_of_a"]["run_count"] == 3
    assert case["greedy_owner_comparison"] == {"a_then_b_then_c": "C", "b_then_a_then_c": "B", "equal": False}
    paired = case["paired_sample_comparison"]
    assert paired["shared_seed_count"] == 2
    assert paired["owner_agreement_count"] == 1
    assert paired["owner_agreement_rate"] == 0.5
    assert paired["both_success_count"] == 2
    assert paired["raw_generated_row_agreement_count"] == 1
    assert paired["raw_generated_row_agreement_rate"] == 0.5


def test_merge_accepts_disjoint_cases_and_rejects_duplicate_runs(tmp_path: Path) -> None:
    first = _payload(case_id="case-1")
    second = _payload(case_id="case-2", seed_tokens=(31, 41))
    result = summarize([_write(tmp_path / "a.json", first), _write(tmp_path / "b.json", second)])
    assert result["source"]["case_count"] == 2
    assert result["source"]["run_count"] == 18
    with pytest.raises(ValueError, match="duplicate run key"):
        summarize([tmp_path / "a.json", tmp_path / "a.json"])


def test_summary_distinguishes_failed_and_unmatched_runs(tmp_path: Path) -> None:
    payload = _payload()
    payload["cases"][0]["arms"][ARMS[0]]["runs"].append(_run(mode="sample", seed=99, owner=None, token=0, status="failed"))
    result = summarize([_write(tmp_path / "failed.json", payload)])
    arm = result["cases"][0]["arms"][ARMS[0]]
    assert arm["primary_owner_counts"]["failed"] == 1
    assert arm["status_counts"]["failed"] == 1
    assert arm["primary_owner_counts"]["none"] == 1
