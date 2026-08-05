"""CPU contract tests for the native-greedy boundary census planner."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil

import pytest

from scripts.research import (
    build_sorted_all_person_greedy_boundary_census as census,
)


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _build(tmp_path: Path, name: str = "plan") -> Path:
    output = tmp_path / name
    result = census.build_sorted_all_person_greedy_boundary_census(output)
    assert result["status"] == "created"
    return output


# ---------------------------------------------------------------------------
# Real-data end-to-end smoke: 11 boundaries / 41 owners / 369 candidates
# ---------------------------------------------------------------------------


def test_exact_boundary_owner_candidate_and_request_counts(tmp_path: Path) -> None:
    output = _build(tmp_path)
    owners = _jsonl(output / "owner-ledger.jsonl")
    candidates = _jsonl(output / "primary-candidates.jsonl")
    contexts = _jsonl(output / "contexts.jsonl")
    owner_map = _jsonl(output / "owner-boundary-map.jsonl")
    sidecars = _jsonl(output / "sidecars.jsonl")
    requests = _jsonl(output / "scoring-requests.jsonl")
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))

    assert [row["gt_owner_id"] for row in owners] == [
        f"gt:7511:{index}" for index in range(2, 43)
    ]
    assert len(candidates) == 369
    assert len(contexts) == 11
    assert [row["context_id"] for row in contexts] == [
        f"boundary-{index:02d}" for index in range(11)
    ]
    assert [row["boundary_index"] for row in contexts] == list(range(11))
    assert contexts[0]["self_prefix_row_indices"] == []
    assert contexts[-1]["self_prefix_row_indices"] == list(range(10))
    assert contexts[-1]["terminal_status"]["is_terminal"] is True
    assert all(not row["terminal_status"]["is_terminal"] for row in contexts[:-1])

    assert len(owner_map) == 41
    assert {row["gt_owner_id"] for row in owner_map} == {row["gt_owner_id"] for row in owners}

    # nine token-distinct native greedy person rows (kite at row 0 excluded)
    assert len(sidecars) == 9
    assert all(row["description"] == "person" for row in sidecars)
    assert all(row["excluded_from_primary_ranks"] is True for row in sidecars)
    assert all(row["requires_new_score_row"] is True for row in sidecars)

    assert len(requests) == 11 * 369 + 11 * 9
    assert receipt["counts"]["scoring_requests_by_kind"] == {
        "primary": 11 * 369,
        "sidecar": 11 * 9,
    }
    for context in contexts:
        primary_ids = {
            row["candidate_id"]
            for row in requests
            if row["request_kind"] == "primary" and row["context_id"] == context["context_id"]
        }
        assert primary_ids == {row["candidate_id"] for row in candidates}
        sidecar_ids = {
            row["sidecar_id"]
            for row in requests
            if row["request_kind"] == "sidecar" and row["context_id"] == context["context_id"]
        }
        assert sidecar_ids == {row["sidecar_id"] for row in sidecars}


def test_emitted_vs_missed_owner_role_split_on_real_data(tmp_path: Path) -> None:
    output = _build(tmp_path)
    owner_map = _jsonl(output / "owner-boundary-map.jsonl")
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))

    emitted = [row for row in owner_map if row["mapping_kind"] == "strict_matched_emit"]
    overtaken = [row for row in owner_map if row["mapping_kind"] == "spatial_overtake"]
    assert len(emitted) == 5
    assert len(overtaken) == 36
    assert {row["gt_owner_id"] for row in emitted} == {
        "gt:7511:2",
        "gt:7511:16",
        "gt:7511:22",
        "gt:7511:32",
        "gt:7511:42",
    }
    for row in emitted:
        assert row["post_boundary_index"] == row["pre_boundary_index"] + 1
        assert row["no_observed_overtake_before_stop"] is False
    assert receipt["owner_role_summary"] == {
        "strict_matched_emit_count": 5,
        "spatial_overtake_count": 36,
        "no_observed_overtake_before_stop_count": 0,
    }
    # real rollout's row-key sequence happens to be monotone
    assert receipt["row_key_monotonicity"]["is_monotonic_nondecreasing"] is True
    assert receipt["row_key_monotonicity"]["backtracking_pairs"] == []


def test_sealed_two_layer_prefix_ends_at_box_start_root_and_nonroot(tmp_path: Path) -> None:
    output = _build(tmp_path)
    contexts = _jsonl(output / "contexts.jsonl")
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))

    query_suffix = receipt["greedy_rollout_binding"]["query_suffix_token_ids"]
    assert len(query_suffix) == 4  # object_ref_start, person token(s), object_ref_end, box_start
    box_start = query_suffix[-1]

    root = contexts[0]
    assert root["boundary_index"] == 0
    assert root["observed_self_prefix_token_ids"] == root["prompt_token_ids"]
    assert root["query_suffix_token_ids"] == query_suffix
    assert root["full_prefix_token_ids"] == root["observed_self_prefix_token_ids"] + query_suffix
    assert root["full_prefix_token_ids"][-1] == box_start
    assert root["full_prefix_token_ids_sha256"] == census.sha256_json(root["full_prefix_token_ids"])
    assert root["observed_self_prefix_token_ids_sha256"] == census.sha256_json(
        root["observed_self_prefix_token_ids"]
    )
    assert root["query_suffix_token_ids_sha256"] == census.sha256_json(query_suffix)

    nonroot = contexts[3]
    assert nonroot["boundary_index"] == 3
    assert nonroot["query_suffix_token_ids"] == query_suffix
    assert (
        nonroot["full_prefix_token_ids"]
        == nonroot["observed_self_prefix_token_ids"] + query_suffix
    )
    assert nonroot["full_prefix_token_ids"][-1] == box_start
    # the self-prefix itself must never contain a forced/injected token: it is
    # exactly prompt + native generated rows
    assert (
        nonroot["observed_self_prefix_token_ids"]
        == nonroot["prompt_token_ids"] + nonroot["generated_prefix_token_ids"]
    )
    # and every boundary's own observed self-prefix must literally end at the
    # previous row's box_end (or the bare prompt for root) -- never at box_start
    assert nonroot["observed_self_prefix_token_ids"][-1] != box_start

    for context in contexts:
        assert context["full_prefix_token_ids"][-1] == box_start


def test_full_prefix_fails_fast_if_it_would_not_end_at_box_start() -> None:
    census.assert_full_prefix_ends_at_box_start([1, 2, 3, 12], box_start=12, boundary_index=0)
    with pytest.raises(census.BoundaryCensusContractError, match="does not end at box_start"):
        census.assert_full_prefix_ends_at_box_start([1, 2, 3, 99], box_start=12, boundary_index=0)
    with pytest.raises(census.BoundaryCensusContractError, match="does not end at box_start"):
        census.assert_full_prefix_ends_at_box_start([], box_start=12, boundary_index=0)


def test_reused_candidate_bank_is_byte_identical_to_plan_v2(tmp_path: Path) -> None:
    output = _build(tmp_path)
    plan_v2 = census.PLAN_V2_DIR
    assert (output / "primary-candidates.jsonl").read_bytes() == (
        plan_v2 / "primary-candidates.jsonl"
    ).read_bytes()
    assert (output / "owner-ledger.jsonl").read_bytes() == (plan_v2 / "owner-ledger.jsonl").read_bytes()

    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    plan_v2_receipt = json.loads((plan_v2 / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["lineage"]["plan_v2_receipt_content_sha256"] == plan_v2_receipt["receipt_content_sha256"]
    assert receipt["unit_id"] == census.PLAN_V2_UNIT_ID
    assert receipt["schema_version"] == census.PLAN_V2_SCHEMA_VERSION
    assert receipt["plan_strategy"] == "native_greedy_boundary_census"


def test_compatibility_seam_is_reported_not_silently_bridged(tmp_path: Path) -> None:
    output = _build(tmp_path)
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    compat = receipt["compatibility"]
    assert compat["target_scorer_load_plan_status"] == "incompatible_missing_required_files"
    assert compat["missing_required_files"] == ["sampling-seeds.jsonl"]
    assert not (output / "sampling-seeds.jsonl").exists()
    assert (output / "sidecars.jsonl").exists()


# ---------------------------------------------------------------------------
# Pure owner-role logic: terminal no-overtake, backtracking, dedup -- driven
# by small synthetic inputs so real-data monotonicity does not hide them.
# ---------------------------------------------------------------------------


def _owner(owner_id: str, xyxy: list[int]) -> dict[str, object]:
    return {"gt_owner_id": owner_id, "bbox_pixel_xyxy": xyxy}


def test_terminal_no_overtake_is_explicitly_flagged() -> None:
    owners = [_owner("gt:7511:A", [0, 900, 10, 910])]  # far below all row keys
    row_keys = [(100, 100), (200, 200), (300, 300)]
    roles = census.compute_owner_boundary_roles(
        owners=owners,
        matched_owner_to_row={},
        row_keys=row_keys,
        total_complete_row_count=len(row_keys),
    )
    assert len(roles) == 1
    role = roles[0]
    assert role["mapping_kind"] == "spatial_overtake"
    assert role["no_observed_overtake_before_stop"] is True
    assert role["pre_boundary_index"] == role["post_boundary_index"] == len(row_keys)
    assert role["matched_row_index"] is None


def test_first_overtake_is_used_even_with_later_backtracking() -> None:
    # owner key (150, 150): row 0 key is below it, row 1 overtakes it, row 2
    # falls back below it again -- the "first" rule must still fire at row 1,
    # and the reversion must be recorded rather than silently dropped.
    owners = [_owner("gt:7511:A", [140, 150, 160, 170])]
    row_keys = [(100, 100), (200, 200), (120, 120)]
    roles = census.compute_owner_boundary_roles(
        owners=owners,
        matched_owner_to_row={},
        row_keys=row_keys,
        total_complete_row_count=len(row_keys),
    )
    role = roles[0]
    assert role["mapping_kind"] == "spatial_overtake"
    assert role["matched_row_index"] == 1
    assert role["pre_boundary_index"] == 1
    assert role["post_boundary_index"] == 2
    assert role["no_observed_overtake_before_stop"] is False
    assert role["post_overtake_reversion_observed"] is True


def test_row_key_monotonicity_records_backtracking_without_gating() -> None:
    row_keys = [(100, 100), (300, 300), (150, 150), (400, 400)]
    result = census.row_key_monotonicity(row_keys)
    assert result["is_monotonic_nondecreasing"] is False
    assert result["backtracking_pairs"] == [
        {
            "from_index": 1,
            "to_index": 2,
            "from_key_y1_x1": [300, 300],
            "to_key_y1_x1": [150, 150],
        }
    ]


def test_strict_matched_emit_takes_priority_over_overtake() -> None:
    owners = [_owner("gt:7511:A", [140, 150, 160, 170])]
    row_keys = [(100, 100), (200, 200)]
    roles = census.compute_owner_boundary_roles(
        owners=owners,
        matched_owner_to_row={"gt:7511:A": 0},
        row_keys=row_keys,
        total_complete_row_count=len(row_keys),
    )
    role = roles[0]
    assert role["mapping_kind"] == "strict_matched_emit"
    assert role["pre_boundary_index"] == 0
    assert role["post_boundary_index"] == 1
    assert role["no_observed_overtake_before_stop"] is False


def _ledger_row(index: int, owner_id: str | None) -> census.CanonicalLedgerRow:
    return census.CanonicalLedgerRow(
        original_row_index=index,
        strict_match_status="matched" if owner_id else "unmatched",
        strict_match_gt_owner_id=owner_id,
        raw_span_sha256=f"span-{index}",
        execution_receipt_content_sha256="receipt",
    )


def test_duplicate_owner_claim_across_rows_fails_closed() -> None:
    ledger_rows = {
        0: _ledger_row(0, "gt:7511:A"),
        1: _ledger_row(1, "gt:7511:A"),
    }
    with pytest.raises(census.BoundaryCensusContractError, match="claimed by more than one"):
        census.build_matched_owner_to_row(ledger_rows, ["gt:7511:A"])


def test_matched_owner_outside_scope_is_ignored_not_an_error() -> None:
    ledger_rows = {0: _ledger_row(0, "gt:7511:0")}  # e.g. the non-person kite row
    assert census.build_matched_owner_to_row(ledger_rows, ["gt:7511:2"]) == {}


# ---------------------------------------------------------------------------
# Owner-boundary structural dedup: many owners onto one boundary; the
# boundary is still scored exactly once.
# ---------------------------------------------------------------------------


def test_owner_boundary_dedup_is_structural() -> None:
    owners = [_owner(f"gt:7511:{i}", [140, 150, 160, 170]) for i in range(5)]
    row_keys = [(200, 200)]
    roles = census.compute_owner_boundary_roles(
        owners=owners,
        matched_owner_to_row={},
        row_keys=row_keys,
        total_complete_row_count=len(row_keys),
    )
    # every owner overtaken by the same single row -> same boundary pair
    assert {role["pre_boundary_index"] for role in roles} == {0}
    assert {role["post_boundary_index"] for role in roles} == {1}
    owner_map = census.materialize_owner_boundary_map(roles)
    # five owners map onto exactly two boundary rows total (pre + post)
    assert {row["pre_boundary_context_id"] for row in owner_map} == {"boundary-00"}
    assert {row["post_boundary_context_id"] for row in owner_map} == {"boundary-01"}


# ---------------------------------------------------------------------------
# Tamper detection and create-or-identical
# ---------------------------------------------------------------------------


def test_create_or_identical_and_receipt_self_digest(tmp_path: Path) -> None:
    first = _build(tmp_path, "first")
    second = _build(tmp_path, "second")
    first_files = sorted(path.name for path in first.iterdir())
    assert first_files == sorted(path.name for path in second.iterdir())
    for name in first_files:
        assert (first / name).read_bytes() == (second / name).read_bytes()

    result = census.build_sorted_all_person_greedy_boundary_census(first)
    assert result["status"] == "identical_existing_output"

    receipt = json.loads((first / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["receipt_content_sha256"] == census._receipt_digest(receipt)

    original = (first / "receipt.json").read_bytes()
    (first / "receipt.json").write_bytes(b"{}\n")
    with pytest.raises(census.BoundaryCensusContractError, match="non-identical content"):
        census.build_sorted_all_person_greedy_boundary_census(first)
    assert (first / "receipt.json").read_bytes() == b"{}\n"
    (first / "receipt.json").write_bytes(original)


def test_tampered_plan_v2_owner_ledger_fails_closed(tmp_path: Path) -> None:
    plan_v2 = census.PLAN_V2_DIR
    bad_plan_v2 = tmp_path / "plan-v2-tampered"
    bad_plan_v2.mkdir()
    for name in ["receipt.json", "primary-candidates.jsonl", "contexts.jsonl"]:
        source = plan_v2 / name
        if source.exists():
            shutil.copy(source, bad_plan_v2 / name)
    (bad_plan_v2 / "owner-ledger.jsonl").write_bytes(
        (plan_v2 / "owner-ledger.jsonl").read_bytes() + b"\n"
    )
    sources = replace(census.DEFAULT_SOURCES, plan_v2_dir=bad_plan_v2)
    with pytest.raises(census.BoundaryCensusContractError, match="owner-ledger.jsonl"):
        census.build_sorted_all_person_greedy_boundary_census(tmp_path / "must-not-exist", sources=sources)
    assert not (tmp_path / "must-not-exist").exists()


def test_foreign_image_greedy_rollout_fails_closed(tmp_path: Path) -> None:
    doc = json.loads(census.GREEDY_PATH.read_text(encoding="utf-8"))
    for rollout in doc["rollouts"]:
        if str(rollout.get("image_id")) == census.IMAGE_ID:
            rollout["image_id"] = "9999"
    bad_greedy = tmp_path / "greedy-foreign-image.json"
    bad_greedy.write_text(json.dumps(doc), encoding="utf-8")
    sources = replace(census.DEFAULT_SOURCES, greedy=bad_greedy)
    with pytest.raises(census.BoundaryCensusContractError, match="exactly one image-7511 rollout"):
        census.build_sorted_all_person_greedy_boundary_census(tmp_path / "must-not-exist", sources=sources)
    assert not (tmp_path / "must-not-exist").exists()


def test_dropped_predictions_are_rejected_as_incomplete(tmp_path: Path) -> None:
    doc = json.loads(census.GREEDY_PATH.read_text(encoding="utf-8"))
    for rollout in doc["rollouts"]:
        if str(rollout.get("image_id")) == census.IMAGE_ID:
            rollout["predictions"]["dropped_prediction_count"] = 1
    bad_greedy = tmp_path / "greedy-dropped.json"
    bad_greedy.write_text(json.dumps(doc), encoding="utf-8")
    sources = replace(census.DEFAULT_SOURCES, greedy=bad_greedy)
    with pytest.raises(census.BoundaryCensusContractError, match="dropped/incomplete predictions"):
        census.build_sorted_all_person_greedy_boundary_census(tmp_path / "must-not-exist", sources=sources)
    assert not (tmp_path / "must-not-exist").exists()
