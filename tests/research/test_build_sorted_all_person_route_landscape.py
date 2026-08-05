"""CPU contract tests for the all-person owner-relative route planner."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from scripts.research import build_sorted_all_person_route_landscape as planner


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _build(tmp_path: Path, name: str = "plan") -> Path:
    output = tmp_path / name
    result = planner.build_sorted_all_person_route_landscape(output)
    assert result["status"] == "created"
    return output


def test_exact_shared_bank_context_sidecar_and_request_counts(tmp_path: Path) -> None:
    output = _build(tmp_path)
    owners = _jsonl(output / "owner-ledger.jsonl")
    candidates = _jsonl(output / "primary-candidates.jsonl")
    contexts = _jsonl(output / "contexts.jsonl")
    sidecars = _jsonl(output / "sidecars.jsonl")
    seeds = _jsonl(output / "sampling-seeds.jsonl")
    requests = _jsonl(output / "scoring-requests.jsonl")
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))

    assert [row["gt_owner_id"] for row in owners] == [
        f"gt:7511:{index}" for index in range(2, 43)
    ]
    assert len(candidates) == 369
    assert candidates[0]["candidate_id"] == "primary:gt:7511:2:00:exact_gt_anchor"
    assert all("context_id" not in row for row in candidates)
    assert all(len(row["coord_token_ids"]) == 4 for row in candidates)
    assert all(
        row["coord_token_ids_sha256"] == planner.sha256_json(row["coord_token_ids"])
        for row in candidates
    )
    for owner_id in [f"gt:7511:{index}" for index in range(2, 43)]:
        owner_rows = [
            row for row in candidates if row["generator_gt_owner_id"] == owner_id
        ]
        assert len(owner_rows) == 9
        assert len({tuple(row["coord_token_ids"]) for row in owner_rows}) == 9

    context_ids = [row["context_id"] for row in contexts]
    assert context_ids == [
        "root",
        "self-due-gt2",
        "self-due-gt17",
        "self-due-gt22",
        "self-due-gt32",
        "skip-post-gt17",
    ]
    assert "skip-post-gt22" not in context_ids
    assert all(
        row["teacher_forced_chosen_token_parity"]["claimed_pass"] is False
        for row in contexts
    )
    assert all(
        row["candidate_universe_join"] == "primary-candidates.jsonl:candidate_id"
        for row in contexts
    )

    assert len(sidecars) == 5
    assert {row["source_kind"] for row in sidecars} == {
        "seed_21010_sampled_row",
        "corrected_greedy_v2",
    }
    assert all(row["excluded_from_primary_ranks"] is True for row in sidecars)
    assert all(
        row["coord_token_ids_sha256"] == planner.sha256_json(row["coord_token_ids"])
        for row in sidecars
    )
    assert all(
        row["sidecar_id"] not in {item["candidate_id"] for item in candidates}
        for row in sidecars
    )

    assert len(seeds) == 6 * 32
    assert len(requests) == 2214 + 30 + 8
    assert sum(row["request_kind"] == "primary" for row in requests) == 6 * 369
    assert sum(row["request_kind"] == "sidecar" for row in requests) == 6 * 5
    repeats = [row for row in requests if row["request_kind"] == "numerical_repeat"]
    assert len(repeats) == 8
    assert {row["context_id"] for row in repeats} == {"self-due-gt17"}
    assert {row["candidate_id"] for row in repeats} == {
        "primary:gt:7511:2:00:exact_gt_anchor"
    }
    assert sorted(row["repeat_index"] for row in repeats) == list(range(8))
    assert receipt["counts"]["scoring_requests_by_kind"] == {
        "numerical_repeat": 8,
        "primary": 2214,
        "sidecar": 30,
    }


def test_context_swap_assignments_and_primary_joins_are_semantically_complete(
    tmp_path: Path,
) -> None:
    output = _build(tmp_path)
    candidates = _jsonl(output / "primary-candidates.jsonl")
    contexts = _jsonl(output / "contexts.jsonl")
    requests = _jsonl(output / "scoring-requests.jsonl")
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))

    candidate_ids = {row["candidate_id"] for row in candidates}
    for context in contexts:
        primary = {
            row["candidate_id"]
            for row in requests
            if row["request_kind"] == "primary"
            and row["context_id"] == context["context_id"]
        }
        assert primary == candidate_ids

    swap = receipt["context_admission"]["same_length_owner_swap"]
    assert swap["equal_donor_row_count"] is True
    assert swap["equal_row_grammar_token_shapes"] is True
    assert swap["equal_total_generated_token_count"] is True
    assert swap["total_generated_token_count"] == sum(
        swap["row_generated_token_counts"]
    )
    assert swap["prefixes_are_content_distinct"] is True

    for candidate in candidates:
        status = candidate["strict_assignment_status"]
        lower = candidate["lower_bound_owner_ids"]
        upper = candidate["upper_bound_owner_ids"]
        if status == "matched":
            assert lower == upper == [candidate["strict_assignment_gt_owner_id"]]
        elif status == "ambiguous_neutral":
            assert lower == []
            assert upper == candidate["ambiguity_owner_ids"]
        else:
            assert status == "unmatched"
            assert lower == upper == []


def test_sidecar_bank_dedup_branch_uses_existing_primary_candidate(
    tmp_path: Path,
) -> None:
    output = _build(tmp_path)
    owners = _jsonl(output / "owner-ledger.jsonl")
    candidate = _jsonl(output / "primary-candidates.jsonl")[0]
    duplicate = planner._sidecar_assignment(
        sidecar_id="test:duplicate",
        source_kind="test",
        coord_token_ids=candidate["coord_token_ids"],
        raw_token_ids=[
            planner.BOX_START,
            *candidate["coord_token_ids"],
            planner.BOX_END,
        ],
        source_bbox=tuple(candidate["decoded_bbox_pixel_xyxy"]),
        owners=owners,
        candidates=[candidate],
        source_ref={},
    )
    assert duplicate["bank_member_candidate_ids"] == [candidate["candidate_id"]]
    assert duplicate["requires_new_score_row"] is False
    assert duplicate["excluded_from_primary_ranks"] is True


def test_sampling_seeds_match_independent_utf8_sha256_derivation(
    tmp_path: Path,
) -> None:
    output = _build(tmp_path)
    rows = _jsonl(output / "sampling-seeds.jsonl")
    by_role: dict[str, set[int]] = {}
    for row in rows:
        payload = f"{planner.UNIT_ID}|{row['role_id']}|{row['sample_index']}".encode(
            "utf-8"
        )
        digest = hashlib.sha256(payload).digest()
        assert row["utf8_sha256"] == digest.hex()
        assert row["seed"] == int.from_bytes(digest[:4], "big") & 0x7FFFFFFF
        by_role.setdefault(str(row["role_id"]), set()).add(int(row["seed"]))
    assert set(by_role) == {
        "root",
        "self-due-gt2",
        "self-due-gt17",
        "self-due-gt22",
        "self-due-gt32",
        "skip-post-gt17",
    }
    assert all(len(seeds) == 32 for seeds in by_role.values())


def test_seed_collision_and_retention_shortfall_fail_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(planner.PlanContractError, match="only 0/9"):
        planner.enforce_retention(
            [
                {
                    "generator_gt_owner_id": "gt:7511:2",
                    "strict_assignment_status": "unmatched",
                    "strict_assignment_gt_owner_id": None,
                }
                for _ in range(9)
            ]
        )

    monkeypatch.setattr(planner, "sample_seed", lambda _role, _index: (7, "0" * 64))
    with pytest.raises(planner.PlanContractError, match="sampling seed collision"):
        planner.build_sampling_seeds([{"context_id": "root"}])

    bad_output = tmp_path / "must-not-exist"
    bad_owner = tmp_path / "tampered-owner-ledger.jsonl"
    bad_owner.write_bytes(planner.OWNER_LEDGER_PATH.read_bytes() + b"\n")
    sources = replace(planner.DEFAULT_SOURCES, owner_ledger=bad_owner)
    with pytest.raises(planner.PlanContractError, match="owner_ledger SHA-256"):
        planner.build_sorted_all_person_route_landscape(bad_output, sources=sources)
    assert not bad_output.exists()


def test_stable_bytes_receipt_self_digest_and_create_or_identical(
    tmp_path: Path,
) -> None:
    first = _build(tmp_path, "first")
    second = _build(tmp_path, "second")
    first_files = sorted(path.name for path in first.iterdir())
    assert first_files == sorted(path.name for path in second.iterdir())
    for name in first_files:
        assert (first / name).read_bytes() == (second / name).read_bytes()

    result = planner.build_sorted_all_person_route_landscape(first)
    assert result["status"] == "identical_existing_output"
    receipt = json.loads((first / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["receipt_content_sha256"] == planner._receipt_digest(receipt)
    assert receipt["source_digests"]["unit"] == planner.UNIT_SHA256
    assert receipt["source_digests"]["panel"] == planner.PANEL_SHA256
    assert receipt["source_digests"]["owner_ledger"] == planner.OWNER_LEDGER_SHA256
    assert receipt["source_digests"]["donor"] == planner.DONOR_SHA256
    assert receipt["source_digests"]["behavior"] == planner.BEHAVIOR_SHA256

    original = (first / "receipt.json").read_bytes()
    (first / "receipt.json").write_bytes(b"{}\n")
    with pytest.raises(planner.PlanContractError, match="non-identical content"):
        planner.build_sorted_all_person_route_landscape(first)
    assert (first / "receipt.json").read_bytes() == b"{}\n"
    (first / "receipt.json").write_bytes(original)
