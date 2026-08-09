from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.research.materialize_natural_boundary_census_v3 import (
    CensusV3ContractError,
    canonical_json_bytes,
    document_hash,
    materialize,
    sha256_json,
    _geometry_context,
    _geometry_from_context,
)
from scripts.research import materialize_static_dynamic_owner_interface_cohort as stable
from scripts.research.run_s_natural_boundary_k_n_h_cohort import validate_manifest


IMAGES = (1584, 2299, 2685, 4134, 5001, 6040, 7511, 10707, 13348, 13923, 14038, 14439, 16228)


def _row(checkpoint: str, image: int, index: int, *, measured: bool = False) -> dict:
    owner = f"gt:{image}:{index}"
    prefix = [151646, 151647, 151648, 151649]
    return {
        "checkpoint": checkpoint,
        "gt_owner_id": owner,
        "image_id": image,
        "source_panel_object_index": index,
        "derived_panel_object_index": index,
        "native_tp": False,
        "native_fn": True,
        "strict_complete_row": False,
        "natural_boundary_valid": True,
        "natural_boundary": 1,
        "covered_owner_ids": [f"gt:{image}:0"],
        "exact_prefix_sha256": sha256_json(prefix),
        "exact_prefix_token_count": len(prefix),
        "exact_prefix_token_ids": prefix,
        "eligible_except_support": True,
        "support_status": "measured" if measured else "unassessed",
        "support_record_present": measured,
        "support_verified": True if measured else None,
        "target_B_support": True if measured else None,
        "target_B_support_assessed": measured,
        "geometry": {"launch_eligible": True, "image_cell_regions": {"b_exclusive": [1]}},
        "disposition": "eligible_verified_pair" if measured else "support_unassessed",
        "covered_A_owner_id": f"gt:{image}:0",
    }


def _fixture() -> tuple[dict, dict]:
    rows = []
    for checkpoint in ("S", "A"):
        for index in range(392):
            image = IMAGES[index % len(IMAGES)]
            rows.append(_row(checkpoint, image, index, measured=False))
    base = {
        "schema_version": "natural_boundary_owner_admission_census.v1",
        "status": "sealed",
        "unit_id": "2026-08-06-natural-boundary-routing-history-replication",
        "frozen_universe": {"row_count": 784, "physical_owner_count": 392},
        "source_identity": {"derived_panel": {"path": None, "sha256": "a" * 64}},
        "support_completion_candidates": [{"gt_owner_id": row["gt_owner_id"]} for row in rows[:200]],
        "rows": rows,
    }
    base["self_sha256"] = document_hash(base)
    records = []
    for index in range(220):
        image = IMAGES[index % len(IMAGES)]
        owner = f"gt:{image}:{index}"
        records.append(
            {
                "gt_owner_id": owner,
                "image_id": image,
                "checkpoint": "S",
                "native_fn": True,
                "strict_complete_row": False,
                "natural_boundary_valid": True,
                "natural_boundary": 1,
                "covered_owner_ids": [f"gt:{image}:0"],
                "exact_prefix_sha256": sha256_json([151646, 151647, 151648, 151649]),
                "exact_prefix_token_ids": [151646, 151647, 151648, 151649],
                "verified_support": True,
                "support_status": "measured",
                "support_verified": True,
                "support_features": {"assessed": True, "peak_lift": 4.0, "local_concentration": 5.0},
                "support_calibration_sha256": "b" * 64,
                "support_rule": {"criterion_id": "fixture"},
                "no_future_or_intervention_leakage": True,
                "candidate_score_count": 1,
                "candidate_scores_sha256": sha256_json({"0": 0.0}),
            }
        )
    ledger = {
        "schema_version": "natural_boundary_owner_support_completion_ledger.v1",
        "status": "completed",
        "unit_id": "2026-08-06-natural-boundary-routing-history-replication",
        "checkpoint": "S",
        "records": records,
        "records_sha256": sha256_json(records),
        "support_rule": {"criterion_id": "fixture"},
    }
    ledger["content_sha256"] = document_hash(ledger)
    return base, ledger


def test_materialize_census_v3_and_manifest(tmp_path):
    base, ledger = _fixture()
    result = materialize(base, ledger, test_only=True)
    census, manifest = result["census"], result["manifest"]
    assert census["schema_version"] == "natural_boundary_owner_admission_census.v3"
    assert census["support_completion"]["measured_S_count"] == 220
    assert len(census["rows"]) == 784
    assert manifest["schema_version"] == "s_natural_boundary_admitted_event_manifest.v3"
    assert manifest["event_count"] == 220
    assert manifest["image_count"] == 13
    assert manifest["self_sha256"] == document_hash(manifest)
    assert all(event["natural_boundary"]["pre_opener_natural"] for event in manifest["events"])
    assert manifest["events"][0]["natural_boundary"]["opener_token_id"] is None
    assert manifest["events"][0]["natural_boundary"]["opener_token_contract"]["status"] == "runner_resolved"


def test_materialize_allows_deferred_subthreshold_admission():
    base, ledger = _fixture()
    # The support ledger still covers all 13 images, but only two rows are
    # geometry-admissible and both are from one image. The successor runner,
    # not this materializer, applies the >=3-event/>=2-image gate.
    for row in base["rows"]:
        if row["checkpoint"] == "S":
            row["geometry"]["launch_eligible"] = row["gt_owner_id"] in {"gt:1584:0", "gt:1584:13"}
    base["self_sha256"] = document_hash(base)
    result = materialize(base, ledger, test_only=True)
    assert result["manifest"]["event_count"] == 2
    assert result["manifest"]["image_count"] == 1
    assert result["manifest"]["ledger_image_count"] == 13
    assert result["manifest"]["admission_gate"]["status"] == "deferred_to_successor_runner"


def test_manifest_matches_serialization_successor_contract():
    base, ledger = _fixture()
    manifest = materialize(base, ledger, test_only=True)["manifest"]
    validated = validate_manifest(manifest)
    assert validated["arm_order"] == tuple(manifest["arm_order"])
    assert len(validated["events"]) == 220


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "wrong_prefix", "missing_field"])
def test_materialize_rejects_support_identity_drift(mutation):
    base, ledger = _fixture()
    if mutation == "missing":
        ledger["records"] = ledger["records"][:-1]
    elif mutation == "duplicate":
        ledger["records"][-1] = copy.deepcopy(ledger["records"][0])
    else:
        ledger["records"][0]["exact_prefix_sha256"] = "c" * 64
    if mutation == "missing_field":
        ledger["records"][0].pop("covered_owner_ids")
    ledger["records_sha256"] = sha256_json(ledger["records"])
    ledger["content_sha256"] = document_hash(ledger)
    with pytest.raises(CensusV3ContractError):
        materialize(base, ledger, test_only=True)


@pytest.fixture(scope="module")
def actual_geometry_context():
    root = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration")
    panel = root / "2026-08-04-sorted-prospective-13-image-panel-admission/evaluation-inputs/human-refined-13.coord.jsonl"
    derived = root / "2026-08-05-static-dynamic-owner-interface-crossover/inputs/human-refined-13.geo_sorted_xy.coord.jsonl"
    receipt = root / "2026-08-05-static-dynamic-owner-interface-crossover/inputs/human-refined-13.geo_sorted_xy.coord.receipt.json"
    h0 = root / "2026-08-05-static-dynamic-owner-interface-crossover/ledgers/s-step2444-native-h0.json"
    support = root / "2026-08-05-static-dynamic-owner-interface-crossover/ledgers/s-step2444-final-support.json"
    if not all(path.is_file() for path in (panel, derived, receipt, h0, support)):
        pytest.skip("sealed actual geometry sources are unavailable")
    panel_value, panel_info = stable._read_source(panel)  # noqa: SLF001
    derived_value, derived_info = stable._read_source(derived)  # noqa: SLF001
    receipt_value, receipt_info = stable._read_source(receipt)  # noqa: SLF001
    h0_value, h0_info = stable._read_source(h0)  # noqa: SLF001
    support_value, support_info = stable._read_source(support)  # noqa: SLF001
    return _geometry_context(
        (panel_value, derived_value, receipt_value, h0_value, panel_info, derived_info, receipt_info, h0_info),
        support_value["records"],
        support_info=support_info,
    )


def test_actual_gt5001_geometry_matches_frozen_full_cohort(actual_geometry_context):
    root = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration")
    base_path = root / "2026-08-06-natural-boundary-routing-history-replication/cpu-census-v2/admission-census.json"
    cohort_path = root / "2026-08-05-static-dynamic-owner-interface-crossover/cohort/s-step2444-final-support.json"
    if not base_path.is_file() or not cohort_path.is_file():
        pytest.skip("sealed actual census/cohort artifacts are unavailable")
    base = json.loads(base_path.read_text())
    row = next(row for row in base["rows"] if row.get("checkpoint") == "S" and row.get("gt_owner_id") == "gt:5001:15")
    enriched = _geometry_from_context(row, context=actual_geometry_context)
    cohort = json.loads(cohort_path.read_text())
    expected = next(event for event in cohort["events"] if event["gt_owner_id"] == "gt:5001:15")["geometry_by_checkpoint"]["S"]
    for key in ("a_exclusive", "b_exclusive", "shared_core", "background", "same_class_competitor"):
        assert enriched["image_cell_regions"][key] == expected["image_cell_regions"][key]
    assert enriched["same_class_competitor_owner_id"] == expected["same_class_competitor_owner_id"]


def test_geometry_enriches_new_owner_outside_legacy_32(actual_geometry_context):
    root = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration")
    base_path = root / "2026-08-06-natural-boundary-routing-history-replication/cpu-census-v2/admission-census.json"
    cohort_path = root / "2026-08-05-static-dynamic-owner-interface-crossover/cohort/s-step2444-final-support.json"
    if not base_path.is_file() or not cohort_path.is_file():
        pytest.skip("sealed actual census/cohort artifacts are unavailable")
    base = json.loads(base_path.read_text())
    cohort = json.loads(cohort_path.read_text())
    legacy_ids = {event["gt_owner_id"] for event in cohort["events"]}
    owner_id = "gt:5001:21"
    assert owner_id not in legacy_ids
    row = dict(next(row for row in base["rows"] if row.get("checkpoint") == "S" and row.get("gt_owner_id") == owner_id))
    target_h0 = actual_geometry_context["h0_by_owner"][owner_id]
    row["support_verified"] = True
    row["eligible_except_support"] = True
    row["exact_prefix_sha256"] = target_h0["exact_prefix_sha256"]
    row["natural_boundary"] = target_h0["natural_boundary"]
    row["covered_owner_ids"] = list(target_h0["covered_owner_ids"])
    row["covered_A_owner_id"] = str(target_h0["covered_owner_ids"][0])
    row["geometry"] = {"launch_eligible": True, "image_cell_regions": {"a_exclusive": [], "b_exclusive": [], "background": []}}
    context = copy.deepcopy(actual_geometry_context)
    support_record = next(item for item in context["support_by_owner"].values() if item.get("verified_support") is True)
    context["support_by_owner"][owner_id] = {**dict(support_record), "gt_owner_id": owner_id, "verified_support": True}
    context["verified_ids"].add(owner_id)
    enriched = _geometry_from_context(row, context=context)
    assert enriched["target_owner_id"] == owner_id
    assert set(enriched["image_cell_regions"]) == {"a_exclusive", "b_exclusive", "shared_core", "background", "same_class_competitor"}


def _actual_production_materialization(tmp_path: Path) -> dict:
    """Materialize two real S contexts, including one outside the old 32 pool."""

    root = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration")
    base_path = root / "2026-08-06-natural-boundary-routing-history-replication/cpu-census-v2/admission-census.json"
    plan_path = root / "2026-08-06-natural-boundary-routing-history-replication/support-completion-plan-v1/plan.json"
    panel_path = root / "2026-08-04-sorted-prospective-13-image-panel-admission/evaluation-inputs/human-refined-13.coord.jsonl"
    derived_path = root / "2026-08-05-static-dynamic-owner-interface-crossover/inputs/human-refined-13.geo_sorted_xy.coord.jsonl"
    receipt_path = root / "2026-08-05-static-dynamic-owner-interface-crossover/inputs/human-refined-13.geo_sorted_xy.coord.receipt.json"
    h0_path = root / "2026-08-05-static-dynamic-owner-interface-crossover/ledgers/s-step2444-native-h0.json"
    prior_path = root / "2026-08-05-static-dynamic-owner-interface-crossover/ledgers/s-step2444-final-support.json"
    paths = (base_path, plan_path, panel_path, derived_path, receipt_path, h0_path, prior_path)
    if not all(path.is_file() for path in paths):
        pytest.skip("sealed production geometry inputs are unavailable")
    base = json.loads(base_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    records = []
    verified_ids = {"gt:5001:15", "gt:5001:21"}
    for record in [*copy.deepcopy(plan["contexts"]), *[copy.deepcopy(item) for item in prior["records"] if item.get("native_fn") is True]]:
        owner_id = str(record["gt_owner_id"])
        verified = owner_id in verified_ids
        features = dict(record.get("support_features") or {})
        features.update({"assessed": True, "peak_lift": 5.0 if verified else 0.0, "local_concentration": 5.0 if verified else 0.0})
        record.update(
            {
                "support_features": features,
                "verified_support": verified,
                "support_status": "measured",
                "support_rule": copy.deepcopy(plan["support_lineage"]["support_rule"]),
                "support_calibration_sha256": plan["calibration_reuse"]["calibration_sha256"],
                "no_future_or_intervention_leakage": True,
                "candidate_score_count": record.get("candidate_score_count", 0),
                "candidate_scores_sha256": record.get("candidate_scores_sha256", sha256_json([])),
            }
        )
        records.append(record)
    ledger = {
        "schema_version": "natural_boundary_owner_support_completion_ledger.v1",
        "status": "completed",
        "unit_id": base["unit_id"],
        "checkpoint": "S",
        "records": records,
        "calibration": copy.deepcopy(plan["calibration_reuse"]["calibration"]),
        "support_rule": copy.deepcopy(plan["support_lineage"]["support_rule"]),
        "plan_binding": {"plan_content_sha256": plan["plan_content_sha256"], "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(), "census_binding": {}},
        "prior_support_binding": {"path": str(prior_path.resolve()), "sha256": hashlib.sha256(prior_path.read_bytes()).hexdigest(), "record_count": 20},
    }
    ledger["records_sha256"] = sha256_json(records)
    ledger["content_sha256"] = document_hash(ledger, "content_sha256")
    support_path = tmp_path / "support-completion-ledger.json"
    support_path.write_bytes(canonical_json_bytes(ledger) + b"\n")
    return materialize(
        base_path,
        support_path,
        plan_source=plan_path,
        panel_source=panel_path,
        derived_panel_source=derived_path,
        derived_receipt_source=receipt_path,
        h0_source=h0_path,
        output=tmp_path / "census-v3.json",
        records_output=tmp_path / "census-v3.records.jsonl",
        receipt_output=tmp_path / "census-v3.receipt.json",
        manifest_output=tmp_path / "admitted-events.v3.json",
        cohort_output=tmp_path / "cohort.json",
        cohort_manifest_output=tmp_path / "cohort.manifest.json",
        test_only=True,
    )


def test_production_context_cohort_raw_bindings_and_admitted_image_count(tmp_path: Path):
    result = _actual_production_materialization(tmp_path)
    manifest_path = tmp_path / "admitted-events.v3.json"
    cohort_path = tmp_path / "cohort.json"
    cohort_manifest_path = tmp_path / "cohort.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    cohort_manifest = json.loads(cohort_manifest_path.read_text(encoding="utf-8"))
    raw_cohort_sha = hashlib.sha256(cohort_path.read_bytes()).hexdigest()
    raw_cohort_manifest_sha = hashlib.sha256(cohort_manifest_path.read_bytes()).hexdigest()
    binding = manifest["legacy_context_cohort"]
    assert binding["sha256"] == raw_cohort_sha
    assert binding["manifest_sha256"] == raw_cohort_manifest_sha
    assert cohort_manifest["cohort_sha256"] == raw_cohort_sha
    assert binding["path"] == str(cohort_path.resolve())
    assert binding["manifest_path"] == str(cohort_manifest_path.resolve())
    assert manifest["event_count"] == 2
    assert manifest["image_count"] == 1
    assert manifest["ledger_image_count"] == 13
    assert [event["gt_owner_id"] for event in cohort["events"]] == ["gt:5001:15", "gt:5001:21"]
    assert all(event["panel_identity"].get("coco_ann_id") is not None for event in cohort["events"])
    assert validate_manifest(manifest_path)["events"]
    assert result["manifest"]["legacy_context_cohort"] == binding
