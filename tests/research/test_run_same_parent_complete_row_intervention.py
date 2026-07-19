from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.research.run_same_parent_complete_row_intervention as stage3
from scripts.research.validate_same_parent_complete_row_intervention_union import validate_union


def _row(index: int, token_ids: list[int], owner: str, *, coords: list[int] | None = None) -> dict[str, object]:
    row: dict[str, object] = {
        "row_index": index,
        "status": "success",
        "accepted_complete_row": True,
        "raw_generated_token_ids": token_ids,
        "row_stop": {"stop_reason": "complete_row"},
        "strict_matched_owner_ids": [owner],
        "entity_matches": [{"status": "matched", "matched_entity_id": owner}],
        "parse_evidence": {"parse_status": "accepted"},
    }
    if coords is not None:
        row["parsed_predictions"] = [{"coord_bins": coords}]
    return row


def test_exact_suffix_comparison_checks_tokens_and_status() -> None:
    left = _row(1, [10, 11], "a")
    right = dict(left)
    assert stage3.compare_exact_rows(left, right)["passed"] is True
    changed = dict(right)
    changed["raw_generated_token_ids"] = [10, 12]
    assert stage3.compare_exact_rows(left, changed)["passed"] is False
    assert stage3.compare_suffix_rows([left], [right])["passed"] is True
    assert stage3.compare_suffix_rows([left], [changed])["passed"] is False


def test_continuation_parity_rejects_metadata_or_sampled_replay_drift() -> None:
    row = _row(1, [10], "owner")
    row.update({"generated_token_count_after_row": 1, "total_budget_remaining_before_row": 512})
    frozen = {
        "rows": [row],
        "final_prefix_token_ids_sha256": stage3.hash_prefix_token_ids([1, 10]),
        "horizon_rows_requested": 1,
        "horizon_rows_complete": True,
        "generated_token_count": 1,
        "total_token_budget": 512,
        "budget_exhausted": False,
    }
    replay = dict(frozen)
    replay["rows"] = [dict(row)]
    assert stage3.compare_continuation_parity(
        frozen, replay, frozen_includes_branch=False, branch_token_count=0
    )["passed"] is True
    replay["final_prefix_token_ids_sha256"] = "drift"
    assert stage3.compare_continuation_parity(
        frozen, replay, frozen_includes_branch=False, branch_token_count=0
    )["passed"] is False


def test_branch_structure_rejects_non_coordinate_semantic_drift() -> None:
    native = [151646, 151670, 151671, 151672, 151673, 151647]
    changed = [151645, 151670, 151671, 151672, 151673, 151647]
    assert stage3.compare_branch_structure(native, changed)["passed"] is False


def test_prepare_frozen_candidates_and_coordinate_control() -> None:
    admission = Path(
        "research/investigations/qwen3-vl-dense-enumeration/experiments/"
        "2026-07-19-sampled-history-target-reachability-and-complete-row-value/stage3-admission.json"
    )
    sources = stage3.load_stage_three_sources(admission)
    for image_id in ("7816", "12576", "18380"):
        source = stage3.prepare_candidate_source(sources, image_id)
        assert len(source["native_branch_token_ids"]) == 9
        assert len(source["sampled_branch_token_ids"]) == 9
        assert source["branch_box_iou"] > 0.8
        assert len(source["branch_coordinate_delta_sampled_minus_native"]) == 4


def test_monkeypatched_runtime_replay_uses_exact_parent_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    direct_suffix = _row(2, [12], "other")
    direct_suffix.update({"generated_token_count_after_row": 2, "total_budget_remaining_before_row": 511})
    native_branch = _row(1, [10], "branch", coords=[100, 100, 200, 200])
    native_branch.update({"generated_token_count_after_row": 1, "total_budget_remaining_before_row": 512})
    sampled_branch = _row(1, [11], "branch", coords=[101, 100, 201, 200])
    sampled_branch.update({"generated_token_count_after_row": 1, "total_budget_remaining_before_row": 512})
    frozen_native = {
        "rows": [native_branch, direct_suffix],
        "final_prefix_token_ids_sha256": stage3.hash_prefix_token_ids([7, 10, 12]),
        "horizon_rows_requested": 7,
        "horizon_rows_complete": True,
        "generated_token_count": 2,
        "total_token_budget": 512,
        "budget_exhausted": False,
    }
    frozen_sampled = {
        "rows": [{**_row(2, [13], "target"), "generated_token_count_after_row": 1, "total_budget_remaining_before_row": 512}],
        "final_prefix_token_ids_sha256": stage3.hash_prefix_token_ids([7, 11, 13]),
        "horizon_rows_requested": 6,
        "horizon_rows_complete": True,
        "generated_token_count": 1,
        "total_token_budget": 512,
        "budget_exhausted": False,
    }
    source = {
        "candidate": {"parent_prefix_row_count": 1, "branch_owner_id": "branch", "target_owner_id": "target", "stage_two_transition": ["clean_miss", "hit"]},
        "parent_token_ids": [7],
        "prefix_owner_ids": ["old"],
        "native_branch_token_ids": [10],
        "sampled_branch_token_ids": [11],
        "native_branch": native_branch,
        "sampled_branch": sampled_branch,
        "direct_continuation_rows": [native_branch, direct_suffix],
        "frozen_native_continuation": frozen_native,
        "frozen_sampled_continuation": frozen_sampled,
        "structural_gate": {"passed": True, "checks": {}},
    }
    calls: list[dict[str, object]] = []

    def fake_generate(**kwargs):
        calls.append({"prefix": list(kwargs["prefix_token_ids"]), "budget": kwargs["total_token_budget"]})
        owner = "target" if kwargs["prefix_token_ids"][-1] == 11 else "other"
        token = 13 if owner == "target" else 12
        row = _row(2, [token], owner)
        row.update({"generated_token_count_after_row": 1, "total_budget_remaining_before_row": kwargs["total_token_budget"]})
        return {
            "horizon_rows_complete": True,
            "horizon_rows_requested": 6,
            "rows": [row],
            "final_prefix_token_ids_sha256": stage3.hash_prefix_token_ids(kwargs["prefix_token_ids"] + [token]),
            "generated_token_count": 1,
            "total_token_budget": kwargs["total_token_budget"],
            "budget_exhausted": False,
        }

    monkeypatch.setattr(stage3, "_generate_prefix_continuation", fake_generate)
    result = stage3.run_candidate_replays(
        source=source,
        session=object(),
        native_inputs={},
        tokenizer=object(),
        image_width=100,
        image_height=100,
        entity_ledger=[],
        malformed_limit=2,
    )
    assert calls == [{"prefix": [7, 10], "budget": 511}, {"prefix": [7, 11], "budget": 512}]
    assert result["comparison"]["native_no_op_parity"]["passed"] is True
    assert result["comparison"]["sampled_source_replay_parity"]["passed"] is True
    assert result["comparison"]["target_retrieved_native"] is False
    assert result["comparison"]["target_retrieved_sampled"] is True


def test_union_requires_exact_three_candidates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    candidates = {str(i): {"image_id": str(i)} for i in ("1", "2", "3")}
    shard_digests = {image_id: {"path": str(tmp_path / f"source-{image_id}.json"), "sha256": "source"} for image_id in candidates}
    sources = {
        "admission_path": str(tmp_path / "admission.json"),
        "admission_sha256": "admission",
        "stage_two_union_path": str(tmp_path / "stage2-union.json"),
        "stage_two_union_sha256": "stage2-union",
        "stage_two_admission_path": str(tmp_path / "stage2-admission.json"),
        "stage_two_admission_sha256": "stage2-admission",
        "stage_two_shards": shard_digests,
        "candidates": candidates,
    }
    monkeypatch.setattr("scripts.research.validate_same_parent_complete_row_intervention_union.load_stage_three_sources", lambda _path: sources)
    shard_paths = []
    digest_map = {key: value["sha256"] for key, value in shard_digests.items()}
    for image_id in candidates:
        path = tmp_path / f"shard-{image_id}.json"
        path.write_text(
            json.dumps({
                "schema_version": stage3.SCHEMA_VERSION,
                "phase": stage3.PHASE,
                "frozen_inputs": {
                    "stage_three_admission": sources["admission_path"],
                    "stage_three_admission_sha256": sources["admission_sha256"],
                    "stage_two_union": sources["stage_two_union_path"],
                    "stage_two_union_sha256": sources["stage_two_union_sha256"],
                    "stage_two_admission": sources["stage_two_admission_path"],
                    "stage_two_admission_sha256": sources["stage_two_admission_sha256"],
                    "stage_two_shard_sha256": digest_map,
                },
                "model_identity": {"model": "same"},
                "config": {"model_dtype": "fp32", "device": "cuda:0"},
                "source_identity": {"stage_three_runner_sha256": "runner", "stage_two_runner_sha256": "stage2"},
                "images": [{
                    "image_id": image_id,
                    "stage3_admission_entry": candidates[image_id],
                    "comparison": {"no_op_parity": {"passed": True}, "primary_causal_claim_allowed": True},
                }],
            }),
            encoding="utf-8",
        )
        shard_paths.append(path)
    with pytest.raises((ValueError, KeyError)):
        validate_union(shard_paths=shard_paths, admission_path=tmp_path / "admission.json")
    with pytest.raises(ValueError, match="exactly three"):
        validate_union(shard_paths=shard_paths[:2], admission_path=tmp_path / "admission.json")
