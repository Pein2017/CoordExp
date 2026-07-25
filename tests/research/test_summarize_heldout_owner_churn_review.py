from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from scripts.research.summarize_heldout_owner_churn_review import (
    ENTITY_CATEGORY_DISPOSITIONS,
    GEOMETRY_DISPOSITIONS,
    ReviewSummaryContractError,
    summarize_review,
)


@dataclass(frozen=True)
class ReviewFixture:
    manifest: Path
    unblinding: Path
    ledger: Path
    shards: tuple[Path, ...]
    output: Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _target(*, matched: bool) -> dict[str, Any]:
    if not matched:
        return {"status": "unmatched_unresolved"}
    return {
        "status": "matched",
        "prediction_index": 0,
        "intersection_over_union": 0.75,
    }


def _fixture(tmp_path: Path) -> ReviewFixture:
    packet = tmp_path / "packet"
    reviewer = packet / "reviewer"
    private = packet / "private"
    case_ids = [f"case_{index:04d}" for index in range(1, 6)]
    review_values = [
        ("real_owner_change", "acceptable", "clean treatment-only owner"),
        ("real_owner_change", "localization_error", "same owner, shifted box"),
        ("duplicate", "acceptable", None),
        ("category_alias_or_disagreement", "localization_error", "category alias"),
        ("uncertain", "acceptable", "cannot resolve owner"),
    ]

    manifest_cases: list[dict[str, Any]] = []
    for case_id in case_ids:
        full_rel = f"images/{case_id}.full.png"
        crop_rel = f"images/{case_id}.crop.png"
        case_rel = f"cases/{case_id}.json"
        full_path = reviewer / full_rel
        crop_path = reviewer / crop_rel
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_bytes(f"full:{case_id}".encode())
        crop_path.write_bytes(f"crop:{case_id}".encode())
        case_payload = {
            "schema_version": "heldout_owner_change_review_packet.v1.blind_case",
            "blinded": True,
            "case_id": case_id,
            "image": {"path": f"images/{case_id}.jpg"},
            "reference_objects": [],
            "target_reference": {},
            "views": {"view_a": {}, "view_b": {}},
            "artifacts": {
                "full_image": full_rel,
                "full_image_sha256": _sha256(full_path),
                "enlarged_crop": crop_rel,
                "enlarged_crop_sha256": _sha256(crop_path),
                "crop_pixel_xyxy": [0, 0, 10, 10],
            },
        }
        case_path = reviewer / case_rel
        _write_json(case_path, case_payload)
        manifest_cases.append(
            {
                "case_id": case_id,
                "case_file": case_rel,
                "case_file_sha256": _sha256(case_path),
                "full_image": full_rel,
                "full_image_sha256": _sha256(full_path),
                "enlarged_crop": crop_rel,
                "enlarged_crop_sha256": _sha256(crop_path),
            }
        )

    readme = reviewer / "README.md"
    readme.write_text("Arm-blind review fixture.\n", encoding="utf-8")
    template = reviewer / "dispositions.jsonl"
    _write_jsonl(
        template,
        [
            {
                "case_id": case_id,
                "entity_category": "pending",
                "geometry": "pending",
                "notes": None,
            }
            for case_id in case_ids
        ],
    )
    manifest = reviewer / "manifest.json"
    _write_json(
        manifest,
        {
            "schema_version": "heldout_owner_change_review_packet.v1.blind_manifest",
            "blinded": True,
            "case_count": 5,
            "hidden_fields": [
                "checkpoint_identity",
                "physical_owner_change_direction",
            ],
            "official_matching": {
                "method": "cardinality_first_maximum_total_intersection_over_union",
                "threshold": 0.5,
                "unmatched_prediction_semantics": "unresolved_pending_human_review",
            },
            "review_schema": {
                "entity_category": {
                    "required": True,
                    "allowed_values": list(ENTITY_CATEGORY_DISPOSITIONS),
                },
                "geometry": {
                    "required": True,
                    "allowed_values": list(GEOMETRY_DISPOSITIONS),
                },
                "notes": {"required": False, "type": "string_or_null"},
            },
            "instructions": "README.md",
            "instructions_sha256": _sha256(readme),
            "disposition_template": "dispositions.jsonl",
            "disposition_template_sha256": _sha256(template),
            "cases": manifest_cases,
        },
    )

    source_refs = [
        {"row_id": "row-2", "owner_index": 1},
        {"row_id": "row-4", "owner_index": 3},
    ]
    treatment_refs = [
        {"row_id": "row-1", "owner_index": 0},
        {"row_id": "row-3", "owner_index": 2},
        {"row_id": "row-5", "owner_index": 4},
    ]
    arm_inputs = {
        "arm_a": {"path": "/synthetic/source.jsonl", "sha256": "a" * 64},
        "arm_b": {"path": "/synthetic/treatment.jsonl", "sha256": "b" * 64},
    }
    ledger = private / "original_comparison_ledger.json"
    _write_json(
        ledger,
        {
            "inputs": {
                **arm_inputs,
                "cohort": {"selection": "all_artifact_rows", "included_row_ids": None},
            },
            "policy": {
                "delta_convention": "arm_b_minus_arm_a",
                "match_iou_threshold": 0.5,
                "unmatched_predictions_are_not_hallucinations": True,
            },
            "common_owner_geometry": {
                "owner_count": 10,
                "arm_a_only_owner_count": 2,
                "arm_a_only_owner_refs": source_refs,
                "arm_b_only_owner_count": 3,
                "arm_b_only_owner_refs": treatment_refs,
            },
        },
    )
    unblinding_cases: list[dict[str, Any]] = []
    for index, case_id in enumerate(case_ids, start=1):
        gain = index in {1, 3, 5}
        matched_arm = "arm_b" if gain else "arm_a"
        unblinding_cases.append(
            {
                "case_id": case_id,
                "row_id": f"row-{index}",
                "owner_index": index - 1,
                "ledger_reference_side": "arm_b_only" if gain else "arm_a_only",
                "physical_owner_change_direction": "gain" if gain else "loss",
                "views": (
                    {"view_a": "arm_a", "view_b": "arm_b"}
                    if index % 2
                    else {"view_a": "arm_b", "view_b": "arm_a"}
                ),
                "authoritative_attribution": {
                    arm: {"matches": [], "target": _target(matched=arm == matched_arm)}
                    for arm in ("arm_a", "arm_b")
                },
            }
        )
    unblinding = private / "unblinding.json"
    _write_json(
        unblinding,
        {
            "schema_version": (
                "heldout_owner_change_review_packet.v1.private_unblinding"
            ),
            "comparison_ledger": {
                "path": "/synthetic/original-comparison.json",
                "sha256": _sha256(ledger),
                "private_copy": "original_comparison_ledger.json",
                "private_copy_sha256": _sha256(ledger),
            },
            "checkpoint_roles": {
                "arm_a": "Source",
                "arm_b": "transition step 36",
            },
            "inputs": arm_inputs,
            "attribution": {"threshold": 0.5},
            "reviewer_manifest": {
                "path": "../reviewer/manifest.json",
                "sha256": _sha256(manifest),
            },
            "cases": unblinding_cases,
        },
    )

    shards: list[Path] = []
    for index, (case_id, values) in enumerate(
        zip(case_ids, review_values, strict=True), start=1
    ):
        entity, geometry, notes = values
        shard = reviewer / "review-shards" / f"shard-{index}.jsonl"
        _write_jsonl(
            shard,
            [
                {
                    "case_id": case_id,
                    "entity_category": entity,
                    "geometry": geometry,
                    "notes": notes,
                }
            ],
        )
        shards.append(shard)
    return ReviewFixture(
        manifest=manifest,
        unblinding=unblinding,
        ledger=ledger,
        shards=tuple(shards),
        output=tmp_path / "audit-summary-v1",
    )


def _run(fixture: ReviewFixture, *, output: Path | None = None) -> dict[str, Any]:
    result = summarize_review(
        fixture.manifest,
        fixture.unblinding,
        fixture.ledger,
        fixture.shards,
        output or fixture.output,
        expected_case_count=5,
        expected_treatment_only_count=3,
        expected_source_only_count=2,
    )
    return dict(result)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _rewrite_unblinding_hashes(fixture: ReviewFixture) -> None:
    unblinding = _load_json(fixture.unblinding)
    digest = _sha256(fixture.ledger)
    unblinding["comparison_ledger"]["sha256"] = digest
    unblinding["comparison_ledger"]["private_copy_sha256"] = digest
    _write_json(fixture.unblinding, unblinding)


def test_summarizes_synthetic_gain_and_loss_deterministically(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    summary = _run(fixture)
    assert summary["official_geometry_ledger"]["direction_counts"] == {
        "gain_count": 3,
        "loss_count": 2,
        "net_gain_minus_loss": 1,
        "total_count": 5,
    }
    refined = summary["human_refined_review"][
        "interpretation_counts_by_official_direction"
    ]
    assert refined["genuine_real_owner_change"] == {
        "gain_count": 1,
        "loss_count": 0,
        "net_gain_minus_loss": 1,
        "total_count": 1,
    }
    assert summary["official_geometry_ledger"]["common_owner_geometry"] == _load_json(
        fixture.ledger
    )["common_owner_geometry"]
    assert len((fixture.output / "per-case.jsonl").read_text().splitlines()) == 5
    assert len((fixture.output / "per-case.tsv").read_text().splitlines()) == 6

    second_output = tmp_path / "audit-summary-v1-repeat"
    _run(fixture, output=second_output)
    for name in ("summary.json", "per-case.jsonl", "per-case.tsv"):
        assert (fixture.output / name).read_bytes() == (second_output / name).read_bytes()


def test_geometry_only_review_is_not_genuine_owner_change(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    summary = _run(fixture)
    rows = [
        json.loads(line)
        for line in (fixture.output / "per-case.jsonl").read_text().splitlines()
    ]
    geometry_loss = next(row for row in rows if row["case_id"] == "case_0002")
    assert geometry_loss["official_direction"] == "loss"
    assert geometry_loss["blind_entity_category_disposition"] == "real_owner_change"
    assert geometry_loss["blind_geometry_disposition"] == "localization_error"
    assert geometry_loss["refined_interpretation"] == "geometry"
    refined = summary["human_refined_review"][
        "interpretation_counts_by_official_direction"
    ]
    assert refined["geometry"]["loss_count"] == 1
    assert refined["genuine_real_owner_change"]["loss_count"] == 0


def test_fails_closed_on_manifest_contract_change(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    manifest = _load_json(fixture.manifest)
    manifest["blinded"] = False
    _write_json(fixture.manifest, manifest)
    with pytest.raises(ReviewSummaryContractError, match="blinded=true"):
        _run(fixture)
    assert not fixture.output.exists()


def test_fails_closed_on_manifest_listed_hash_mismatch(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    manifest = _load_json(fixture.manifest)
    image = fixture.manifest.parent / manifest["cases"][0]["full_image"]
    image.write_bytes(image.read_bytes() + b"tamper")
    with pytest.raises(ReviewSummaryContractError, match="hash mismatch"):
        _run(fixture)


def test_fails_closed_on_unblinding_ledger_bijection_break(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    unblinding = _load_json(fixture.unblinding)
    unblinding["cases"][0]["row_id"] = "not-in-ledger"
    _write_json(fixture.unblinding, unblinding)
    with pytest.raises(ReviewSummaryContractError, match="not bijective"):
        _run(fixture)


def test_fails_closed_on_missing_review_case(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture.shards[-1].write_text("", encoding="utf-8")
    with pytest.raises(ReviewSummaryContractError, match="not the exact manifest set once"):
        _run(fixture)


def test_fails_closed_on_duplicate_review_case(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    first = fixture.shards[0].read_text(encoding="utf-8")
    with fixture.shards[-1].open("a", encoding="utf-8") as handle:
        handle.write(first)
    with pytest.raises(ReviewSummaryContractError, match="occurs more than once"):
        _run(fixture)


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("entity_category", "hallucination", "invalid entity/category"),
        ("geometry", "close_enough", "invalid geometry"),
    ],
)
def test_fails_closed_on_review_enum_value(
    tmp_path: Path,
    field: str,
    bad_value: str,
    message: str,
) -> None:
    fixture = _fixture(tmp_path)
    row = json.loads(fixture.shards[0].read_text(encoding="utf-8"))
    row[field] = bad_value
    _write_jsonl(fixture.shards[0], [row])
    with pytest.raises(ReviewSummaryContractError, match=message):
        _run(fixture)


def test_fails_closed_on_official_direction_counts(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    ledger = _load_json(fixture.ledger)
    geometry = ledger["common_owner_geometry"]
    geometry["arm_b_only_owner_count"] = 2
    geometry["arm_b_only_owner_refs"] = geometry["arm_b_only_owner_refs"][:2]
    _write_json(fixture.ledger, ledger)
    _rewrite_unblinding_hashes(fixture)
    with pytest.raises(ReviewSummaryContractError, match="expected 3, got 2"):
        _run(fixture)


def test_refuses_to_overwrite_immutable_output(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture.output.mkdir()
    with pytest.raises(ReviewSummaryContractError, match="already exists"):
        _run(fixture)
