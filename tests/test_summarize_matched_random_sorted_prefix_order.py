from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.research.summarize_matched_random_sorted_prefix_order import summarize


SEEDS = [101, 102, 103, 104]
COMPARISON_ID = "sorted_vs_permuted"


def _run(mode: str, seed: int | None, owner: str | None = "owner_a", *, outcome: str = "accepted") -> dict:
    accepted = outcome == "accepted"
    row = {
        "accepted_complete_row": accepted,
        "strict_matched_owner_ids": [owner] if owner else [],
        "owner_recurrence": [False] if owner else [],
        "covered_prefix_owner_ids": [],
        "raw_generated_token_ids_sha256": f"row-{mode}-{seed}-{owner}-{outcome}",
        "parse_evidence": {"parse_status": "accepted" if accepted else ("malformed" if outcome == "malformed" else "accepted")},
        "row_stop": {"stop_reason": "complete_row" if accepted else ("eos" if outcome == "terminal" else "missing")},
    }
    return {
        "comparison_id": COMPARISON_ID,
        "mode": mode,
        "seed": seed,
        "rows": [row] if outcome != "generic_corruption" else [],
        "status": "success" if accepted else "failed",
        "initial_prefix_token_ids": [1, 2],
        "initial_prefix_token_ids_sha256": _json_hash([1, 2]),
    }


def _json_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _payload(*, owner: str = "owner_a") -> dict:
    runs = [_run("greedy", None, owner)] + [_run("sample", seed, owner) for seed in SEEDS]
    other_runs = copy.deepcopy(runs)
    return {
        "schema_version": "same_covered_set_prefix_order.v2",
        "experiment": "same_covered_set_prefix_order",
        "config": {"seeds": SEEDS, "include_greedy": True},
        "image": {"image_id": "42", "image_sha256": "image-hash"},
        "case_spec": {"schema_version": "same_covered_set_prefix_order.case.v2", "sha256": "case-hash"},
        "base_prompt": {"image_sha256": "image-hash", "observed_prompt_token_ids_sha256": "prompt-hash"},
        "cases": [{
            "case_id": "case_42",
            "image_id": "42",
            "comparisons": [{"comparison_id": COMPARISON_ID, "arm_names": ["sorted", "permuted"], "shared_suffix_length": 2}],
            "invariants_by_comparison": {COMPARISON_ID: {"same_physical_covered_set": True, "byte_identical_shared_suffix": True}},
            "arms": {
                "sorted": {"prefix_token_ids": [1, 2], "prefix_token_ids_sha256": _json_hash([1, 2]), "runs": runs},
                "permuted": {"prefix_token_ids": [1, 2], "prefix_token_ids_sha256": _json_hash([1, 2]), "runs": other_runs},
            },
        }],
    }


def _write_pair(tmp_path: Path, sorted_payload: dict, random_payload: dict) -> tuple[Path, Path]:
    sorted_dir, random_dir = tmp_path / "sorted", tmp_path / "random"
    sorted_dir.mkdir()
    random_dir.mkdir()
    (sorted_dir / "case.json").write_text(json.dumps(sorted_payload), encoding="utf-8")
    (random_dir / "case.json").write_text(json.dumps(random_payload), encoding="utf-8")
    return sorted_dir, random_dir


def test_promotes_checkpoint_owner_switch_and_selects_for_both(tmp_path: Path) -> None:
    sorted_payload = _payload()
    random_payload = _payload()
    random_payload["cases"][0]["arms"]["permuted"]["runs"][0]["rows"][0]["strict_matched_owner_ids"] = ["owner_b"]
    sorted_dir, random_dir = _write_pair(tmp_path, sorted_payload, random_payload)

    receipt = summarize(sorted_dir, random_dir)

    assert receipt["selected_pair_count"] == 1
    pair = receipt["pairs"][0]
    assert pair["promotion_checkpoints"] == ["random"]
    assert pair["promoted_for_identical_scoring_under_both_checkpoints"] is True
    assert pair["checkpoint_results"]["random"]["promotion_reasons"] == ["greedy_strict_owner_switch"]
    assert pair["checkpoint_results"]["random"]["greedy"]["strict_owner_switch"] is True


def test_rejects_generic_corruption_only_and_reports_exact_row_difference(tmp_path: Path) -> None:
    sorted_payload = _payload()
    random_payload = _payload()
    random_payload["cases"][0]["arms"]["permuted"]["runs"][0] = _run("greedy", None, outcome="generic_corruption")
    sorted_dir, random_dir = _write_pair(tmp_path, sorted_payload, random_payload)

    receipt = summarize(sorted_dir, random_dir)

    pair = receipt["pairs"][0]
    assert receipt["selected_pair_count"] == 0
    assert pair["checkpoint_results"]["random"]["generic_corruption_only"] is True
    assert pair["checkpoint_results"]["random"]["promotion_reasons"] == []
    assert pair["checkpoint_results"]["random"]["greedy"]["exact_generated_row_difference"] is True


def test_promotes_frozen_prefix_covered_owner_recurrence_not_horizon_recurrence(tmp_path: Path) -> None:
    sorted_payload = _payload()
    random_payload = _payload()
    # The runner's horizon-local owner_recurrence remains false; this non-empty
    # ledger is the only evidence that the owner recurs from the frozen prefix.
    random_payload["cases"][0]["arms"]["permuted"]["runs"][0]["rows"][0]["covered_prefix_owner_ids"] = ["owner_a"]
    sorted_dir, random_dir = _write_pair(tmp_path, sorted_payload, random_payload)

    receipt = summarize(sorted_dir, random_dir)

    result = receipt["pairs"][0]["checkpoint_results"]["random"]
    assert result["promotion_reasons"] == ["greedy_covered_recurrence_difference"]
    assert result["greedy"]["right"]["covered_recurrence"] is True


def test_rejects_cross_checkpoint_base_prompt_identity_mismatch(tmp_path: Path) -> None:
    sorted_payload = _payload()
    random_payload = _payload()
    random_payload["base_prompt"]["observed_prompt_token_ids_sha256"] = "different"
    sorted_dir, random_dir = _write_pair(tmp_path, sorted_payload, random_payload)

    with pytest.raises(ValueError, match="artifact identity"):
        summarize(sorted_dir, random_dir)
