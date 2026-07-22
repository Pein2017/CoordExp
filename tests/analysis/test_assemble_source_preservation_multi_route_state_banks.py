"""Model-free contracts for the matched Source-route assembler."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts/research/assemble_source_preservation_multi_route_state_banks.py"
)
SPEC = importlib.util.spec_from_file_location("source_preservation_multi_route", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _row(index: int, owner: str | None, *, iou: float = 0.9) -> dict[str, object]:
    return {
        "generated_row_index": index,
        "owner_id": owner,
        "entity_status": "verified_owner" if owner is not None else "unresolved_pending_crop_review",
        "category": "person",
        "intersection_over_union": iou,
        "raw": {"coord_bins": [10, 20, 30, 40], "raw_span_text": "exact"},
    }


def _assignment(rows: list[dict[str, object]], *, unresolved: int = 0) -> dict[str, object]:
    return {
        "row_assignment_receipts": rows,
        "row_counts": {"duplicate": 0, "malformed": 0, "unresolved": unresolved},
    }


def _route_image() -> dict[str, object]:
    return {
        "image_id": "42",
        "greedy_trajectory_id": "greedy",
        "sampled_trajectory_ids": ["seed-7", "seed-8", "seed-9"],
        "trajectory_evidence": {
            "greedy": {"seed": 0, "stop_reason": "im_end", "parser": {"parse_status": "accepted"}},
            "seed-7": {"seed": 7, "stop_reason": "im_end", "parser": {"parse_status": "accepted"}},
            "seed-8": {"seed": 8, "stop_reason": "im_end", "parser": {"parse_status": "accepted"}},
            "seed-9": {"seed": 9, "stop_reason": "im_end", "parser": {"parse_status": "accepted"}},
        },
        "budgets": [
            {
                "budget": 16,
                "owner_sets": {
                    "greedy": ["42:a"],
                    "seed-7": ["42:a", "42:b", "42:c"],
                    "seed-8": ["42:a", "42:c", "42:d"],
                    "seed-9": ["42:a", "42:b"],
                },
                "trajectory_assignments": {
                    "greedy": _assignment([_row(0, "42:a")]),
                    "seed-7": _assignment([_row(0, "42:a"), _row(1, "42:b"), _row(2, "42:c")]),
                    "seed-8": _assignment([_row(0, "42:a"), _row(1, "42:c"), _row(2, "42:d")]),
                    "seed-9": _assignment([_row(0, "42:a"), _row(1, "42:b")]),
                },
            }
        ],
    }


def test_multi_route_selection_is_deterministic_and_marginal() -> None:
    first = MODULE.select_complementary_routes(_route_image())
    second = MODULE.select_multi_route_routes(_route_image())
    assert first == second
    assert first["selected_route_ids"] == ["seed-7", "seed-8"]
    assert first["selected_added_owner_union"] == ["42:b", "42:c", "42:d"]
    assert first["selected_routes"][0]["last_marginal_owner_row_index"] == 2
    assert first["selected_routes"][1]["marginal_added_owner_ids"] == ["42:d"]


def test_exact_integer_slices_are_preserved_without_retokenization() -> None:
    row_a = [151646, 987654, 151647, 151648, 151670, 151671, 151672, 151673, 151649]
    row_b = [151646, 123, 151647, 151648, 151700, 151800, 151900, 152000, 151649]
    prefix, row = MODULE.exact_row_slices(row_a + row_b, 1)
    assert prefix == row_a
    assert row == row_b
    assert hashlib.sha256(bytes(str(row), "utf-8")).hexdigest() == hashlib.sha256(bytes(str(row_b), "utf-8")).hexdigest()


def test_source_first_owner_rows_are_trusted_and_deduplicated() -> None:
    owners = [
        {"owner_id": "42:a", "category": "person"},
        {"owner_id": "42:b", "category": "person"},
    ]
    rows = MODULE.select_source_anchor_rows(
        _assignment([_row(0, "42:a"), _row(1, "42:a"), _row(2, "42:b"), _row(3, None)]),
        owners,
    )
    assert [item["owner_id"] for item in rows] == ["42:a", "42:b"]
    assert [item["generated_row_index"] for item in rows] == [0, 2]


def test_identical_source_anchor_candidates_have_identical_hashes_across_arms() -> None:
    source = [
        {
            "image_id": "1",
            "prefix_token_ids_sha256": "p1",
            "candidate_token_ids_sha256": "r1",
            "owner_id": "1:a",
            "event_id": "source-a",
        },
        {
            "image_id": "2",
            "prefix_token_ids_sha256": "p2",
            "candidate_token_ids_sha256": "r2",
            "owner_id": "2:a",
            "event_id": "source-b",
        },
    ]
    arm_a = MODULE.deduplicate_route_events(source)
    arm_b = MODULE.deduplicate_route_events(source)
    hashes_a = {(item["image_id"], item["prefix_token_ids_sha256"], item["candidate_token_ids_sha256"], item["owner_id"]) for item in arm_a}
    hashes_b = {(item["image_id"], item["prefix_token_ids_sha256"], item["candidate_token_ids_sha256"], item["owner_id"]) for item in arm_b}
    assert hashes_a == hashes_b


def test_image_diverse_selector_reaches_512_and_keeps_every_image() -> None:
    image_ids = [str(index) for index in range(1, 119)]
    pools = {
        image: [
            {
                "image_id": image,
                "owner_id": f"{image}:owner-{offset}",
                "prefix_token_ids_sha256": f"{image}-prefix-{offset}",
                "candidate_token_ids_sha256": f"{image}-row-{offset}",
                "generated_row_index": offset,
                "event_id": f"{image}-{offset}",
                "marginal_owner_count": 1,
            }
            for offset in range(5)
        ]
        for image in image_ids
    }
    selected, receipt = MODULE.select_image_diverse_events(pools, image_ids=image_ids, event_count=512)
    assert len(selected) == 512
    assert receipt["image_count"] == 118
    assert {item["image_id"] for item in selected} == set(image_ids)


def test_family_weights_are_equal_per_image_and_mean_one() -> None:
    image_ids = [str(index) for index in range(1, 119)]
    counts = {image: {"source_preservation": 4, "treatment": 4} for image in image_ids}
    counts["1"] = {"source_preservation": 20, "treatment": 20}
    counts["2"] = {"source_preservation": 20, "treatment": 20}
    weights = MODULE.image_family_event_weights(counts, event_count=1024)
    assert weights[("1", "source_preservation")] == weights[("1", "treatment")]
    assert weights[("1", "source_preservation")] != weights[("3", "source_preservation")]
    total = sum(weights[(image, family)] * counts[image][family] for image in image_ids for family in ("source_preservation", "treatment"))
    assert total / 1024 == pytest.approx(1.0)
    image_totals = [sum(weights[(image, family)] * counts[image][family] for family in ("source_preservation", "treatment")) for image in image_ids]
    assert len(set(round(value, 12) for value in image_totals)) == 1


def test_blind_image_rejection_is_fail_fast() -> None:
    with pytest.raises(ValueError, match="blind"):
        MODULE.reject_blind_images(["1584", "42"])


def test_output_refusal_is_immutable_and_rerun_safe(tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"
    MODULE._write_json(output, {"status": "first"})
    with pytest.raises(ValueError, match="already exists"):
        MODULE._write_json(output, {"status": "second"})
