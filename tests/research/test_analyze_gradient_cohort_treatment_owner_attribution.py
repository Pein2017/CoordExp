from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from scripts.research.analyze_gradient_cohort_treatment_owner_attribution import (
    OwnerAttributionError,
    analyze_gradient_cohort_owner_attribution,
)


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _event(
    arm: str, image_id: str, owner_id: str, index: int, event_family: str
) -> dict[str, object]:
    return {
        "event_id": f"{arm}-{image_id}-{index}",
        "image_id": image_id,
        "owner_id": owner_id,
        "event_family": event_family,
    }


def _record(event: dict[str, object], owners: list[str]) -> dict[str, object]:
    return {
        "event_id": event["event_id"],
        "image": {"image_id": event["image_id"]},
        "physical_entities": [{"entity_id": owner} for owner in owners],
        "candidates": [{"physical_owner_id": event["owner_id"]}],
    }


def _per_image(
    image_id: str, source: list[str], treatment: list[str], *, eligible: bool = True
) -> dict[str, object]:
    row = {"source_owner_ids": source, "treatment_owner_ids": treatment}
    return {
        "image_id": image_id,
        "annotated_object_count": 3,
        "object_count_band": "one_to_three_annotated_objects",
        "owner_comparison": {
            "eligible": eligible,
            "by_intersection_over_union": None if not eligible else {"0.30": row, "0.50": row},
        },
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    # Every arm has sampled-route treatment and source-preservation selections.
    # This also makes a2 selected under broad but not under concentrated.
    broad_events = [
        _event("broad", "1", "1:a1", 0, "treatment"),
        _event("broad", "1", "1:a2", 1, "source_preservation"),
    ]
    concentrated_events = [
        _event("concentrated", "1", "1:a3", 0, "treatment"),
        _event("concentrated", "1", "1:a1", 1, "source_preservation"),
    ]
    receipt = _write_json(
        tmp_path / "state-banks/selection-receipt.json",
        {
            "arms": {
                "broad": {"events": broad_events, "image_ids": ["1"]},
                "concentrated": {"events": concentrated_events, "image_ids": ["1"]},
            }
        },
    )
    owners = ["1:a1", "1:a2", "1:a3"]
    for arm, events in (("broad", broad_events), ("concentrated", concentrated_events)):
        records = "\n".join(json.dumps(_record(event, owners)) for event in events) + "\n"
        path = tmp_path / "state-banks" / f"{arm}-plus-source-preservation/state-bank/records.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(records, encoding="utf-8")
    cohort_path = tmp_path / "gradient-cohorts" / "cohort-a.jsonl"
    cohort_path.parent.mkdir(parents=True, exist_ok=True)
    cohort_path.write_text(json.dumps({"image_id": "1"}) + "\n", encoding="utf-8")
    cohort_sha256 = hashlib.sha256(cohort_path.read_bytes()).hexdigest()
    gradient_cohort_receipt = _write_json(
        tmp_path / "gradient-cohorts" / "receipt.json",
        {
            "schema_version": "constant_dose_breadth_gradient_cohorts.v1",
            "cohorts": {
                "cohort-a": {
                    "path": str(cohort_path),
                    "sha256": cohort_sha256,
                    "image_count": 1,
                }
            },
        },
    )
    candidate_jsonl = {
        "path": str(cohort_path),
        "sha256": cohort_sha256,
        "image_count": 1,
    }
    ledger = {
        "inputs": {"candidate_jsonl": candidate_jsonl},
        "treatments": {
            "broad-seed19-step30": {"per_image": [_per_image("1", ["1:a1"], ["1:a2", "1:a3"])]},
            "concentrated-seed19-step30": {"per_image": [_per_image("1", ["1:a1"], ["1:a2", "1:a3"])]},
        }
    }
    ledger_root = tmp_path / "ledgers"
    _write_json(ledger_root / "cohort-a" / "seed19.json", ledger)
    return receipt, tmp_path / "state-banks", ledger_root, gradient_cohort_receipt


def test_attributes_only_the_current_arms_selected_owners(tmp_path: Path) -> None:
    receipt, state_banks, ledgers, gradient_cohorts = _fixture(tmp_path)
    result = analyze_gradient_cohort_owner_attribution(
        selection_receipt=receipt,
        state_banks_root=state_banks,
        ledger_root=ledgers,
        gradient_cohort_root_or_receipt=gradient_cohorts,
    )
    broad = result["cohorts"]["cohort-a"]["treatments"]["broad-seed19-step30"]
    concentrated = result["cohorts"]["cohort-a"]["treatments"]["concentrated-seed19-step30"]
    broad_summary = broad["by_intersection_over_union"]["0.30"]
    broad_union = broad_summary["selected_owners_all_event_families"]
    broad_treatment = broad_summary["selected_owners_by_event_family"][
        "sampled_route_treatment"
    ]
    broad_source_preservation = broad_summary["selected_owners_by_event_family"][
        "source_preservation"
    ]
    assert broad["selected_owner_set_sizes_by_event_family"] == {
        "sampled_route_treatment": 1,
        "source_preservation": 1,
    }
    assert broad_union["selection_scope"] == "union of all StateBank event families"
    assert broad_union["selected_owners"]["count"] == 2
    assert broad_treatment["selected_owners"]["loss_among_source_found"] == {
        "numerator": 1,
        "denominator": 1,
        "rate": 1.0,
    }
    assert broad_source_preservation["selected_owners"]["exact_recovery_among_source_missed"] == {
        "numerator": 1, "denominator": 1, "rate": 1.0
    }
    concentrated_summary = concentrated["by_intersection_over_union"]["0.50"]
    concentrated_treatment = concentrated_summary["selected_owners_by_event_family"][
        "sampled_route_treatment"
    ]
    assert concentrated_treatment["selected_owners"]["count"] == 1
    assert concentrated_treatment["selected_owners"]["source_missed"]["numerator"] == 1
    # a2 is selected in broad but deliberately remains non-selected here.
    assert (
        concentrated_treatment["non_selected_annotated_owners"]
        ["exact_recovery_among_source_missed"]["numerator"]
        == 1
    )
    assert (
        concentrated["object_count_band_breakdown"]["one_to_three_annotated_objects"]
        ["0.30"]["selected_owners_all_event_families"]
        ["complete_case_denominator"]["image_count"]
        == 1
    )


def test_rejects_ledger_owner_outside_the_annotated_image(tmp_path: Path) -> None:
    receipt, state_banks, ledgers, gradient_cohorts = _fixture(tmp_path)
    ledger_path = ledgers / "cohort-a/seed19.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["treatments"]["broad-seed19-step30"]["per_image"][0]["owner_comparison"]["by_intersection_over_union"]["0.30"]["treatment_owner_ids"] = ["1:not-annotated"]
    _write_json(ledger_path, ledger)
    with pytest.raises(OwnerAttributionError, match="absent from annotated"):
        analyze_gradient_cohort_owner_attribution(
            selection_receipt=receipt,
            state_banks_root=state_banks,
            ledger_root=ledgers,
            gradient_cohort_root_or_receipt=gradient_cohorts,
        )


def test_rejects_receipt_statebank_owner_ambiguity(tmp_path: Path) -> None:
    receipt, state_banks, ledgers, gradient_cohorts = _fixture(tmp_path)
    records_path = state_banks / "broad-plus-source-preservation/state-bank/records.jsonl"
    rows = records_path.read_text(encoding="utf-8").splitlines()
    corrupted = json.loads(rows[0])
    corrupted["candidates"].append({"physical_owner_id": "1:a2"})
    rows[0] = json.dumps(corrupted)
    records_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(OwnerAttributionError, match="selected-owner mismatch"):
        analyze_gradient_cohort_owner_attribution(
            selection_receipt=receipt,
            state_banks_root=state_banks,
            ledger_root=ledgers,
            gradient_cohort_root_or_receipt=gradient_cohorts,
        )


@pytest.mark.parametrize(
    ("replacement_image_id", "error"),
    [(None, "missing="), ("2", "unexpected=")],
)
def test_rejects_gradient_cohort_per_image_drop_or_substitution(
    tmp_path: Path, replacement_image_id: str | None, error: str
) -> None:
    receipt, state_banks, ledgers, gradient_cohorts = _fixture(tmp_path)
    ledger_path = ledgers / "cohort-a/seed19.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    per_image = ledger["treatments"]["broad-seed19-step30"]["per_image"]
    if replacement_image_id is None:
        per_image.clear()
    else:
        per_image[0]["image_id"] = replacement_image_id
    _write_json(ledger_path, ledger)
    with pytest.raises(OwnerAttributionError, match=error):
        analyze_gradient_cohort_owner_attribution(
            selection_receipt=receipt,
            state_banks_root=state_banks,
            ledger_root=ledgers,
            gradient_cohort_root_or_receipt=gradient_cohorts,
        )
