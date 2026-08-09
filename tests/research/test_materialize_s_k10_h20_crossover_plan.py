from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.research import materialize_s_k10_h20_crossover_plan as planner


BASE = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-06-natural-boundary-routing-history-replication"
)
EVIDENCE = BASE / "s-k-n-h-evidence-native-fn-supersession-v3/evidence.json"
RECEIPT = BASE / "s-k-n-h-evidence-native-fn-supersession-v3/evidence.receipt.json"
MANIFEST = BASE / "cpu-census-v3-native-fn-supersession-v1/admitted-event-manifest.json"
CENSUS = BASE / "cpu-census-v3-native-fn-supersession-v1/admission-census.json"
ORIGINAL_PLAN = BASE / "execution-plan-native-fn-supersession-v1/plan.json"
GATE = BASE / "s-gt5001-live-gate-v3/result.json"


def _build(output: Path | None = None) -> dict[str, Any]:
    return planner.build_plan(EVIDENCE, RECEIPT, MANIFEST, CENSUS, ORIGINAL_PLAN, GATE, output)


def _copy_json(path: Path, destination: Path, mutate: Any) -> Path:
    document = json.loads(path.read_bytes())
    mutate(document)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(planner.canonical_json_bytes(document) + b"\n")
    return destination


def test_materializes_exact_v3_intersection_and_three_device_shards(tmp_path: Path) -> None:
    document = _build(tmp_path / "plan.json")
    assert document["schema_version"] == "s_k10_h20_crossover_plan.v1"
    assert document["self_sha256"] == planner.document_self_sha256(document)
    assert [event["event_id"] for event in document["events"]] == list(planner.EVENT_IDS)
    assert [event["event_index"] for event in document["events"]] == [2, 5, 8]
    assert [event["prefix_sha256"] for event in document["events"]] == [
        "371e68a445522e7f789686e385aac432facb76afc2e14c88f044e894f737020a",
        "93afaba4bd7c0f2a4e315160a3af936a0099e031325e0de638202ad15a23f6ed",
        "f19993eb5c17e0d4fc84274c6c26cc4fb2939f742604c2575e08020dcd1a4dd6",
    ]
    assert document["cell_order"] == ["C00", "C10", "C01", "C11"]
    assert document["device_plan"] == planner.DEVICE_PLAN
    assert "no_2x2" not in document
    assert document["no_legacy_no_2x2_reuse"] is True
    assert planner.validate_plan(tmp_path / "plan.json")["self_sha256"] == document["self_sha256"]


def test_source_raw_hash_tamper_is_rejected(tmp_path: Path) -> None:
    tampered = _copy_json(EVIDENCE, tmp_path / "evidence.json", lambda value: value.update({"status": "tampered"}))
    with pytest.raises(planner.PlanError, match="raw SHA-256"):
        planner.build_plan(tampered, RECEIPT, MANIFEST, CENSUS, ORIGINAL_PLAN, GATE)


def test_selection_tamper_is_rejected_even_when_fixture_hashes_are_rebound(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original = json.loads(EVIDENCE.read_bytes())
    original["events"] = [event for event in original["events"] if event["event_id"] != "gt:16228:15"]
    # Keep the synthetic fixture self-consistent, then rebind the frozen raw
    # constants only for this test: the selection rule must still fail closed.
    original["self_sha256"] = planner.document_self_sha256(original)
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_bytes(planner.canonical_json_bytes(original) + b"\n")
    monkeypatch.setattr(planner, "EXPECTED_SOURCE_EVIDENCE_RAW_SHA256", planner.sha256_file(evidence_path))
    monkeypatch.setattr(planner, "EXPECTED_SOURCE_EVIDENCE_SELF_SHA256", original["self_sha256"])
    with pytest.raises(planner.PlanError, match="selection changed"):
        planner.build_plan(evidence_path, RECEIPT, MANIFEST, CENSUS, ORIGINAL_PLAN, GATE)


def test_plan_write_once_rejects_self_or_event_tamper(tmp_path: Path) -> None:
    output = tmp_path / "plan.json"
    document = _build(output)
    assert planner.validate_plan(output)["plan"]["self_sha256"] == document["self_sha256"]
    tampered = dict(document)
    tampered["events"] = list(document["events"])
    tampered["events"][0] = dict(tampered["events"][0], geometry_sha256="0" * 64)
    output.write_bytes(planner.canonical_json_bytes(tampered) + b"\n")
    with pytest.raises(planner.PlanError, match="self_sha256"):
        planner.validate_plan(output)
