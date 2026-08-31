from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from src.common.errors import ArtifactContractError


def test_complete_minimal_infra_base_surface_is_public() -> None:
    from src.artifacts import (
        AbsoluteExecutableBinding,
        AdmissionInspection,
        BindingManifest,
        DirectoryTreeBinding,
        ExecutionEvidenceJournal,
        RegularFileBinding,
        ResearchProbeAdmission,
        ResearchProbeAdmissionError,
        ReservedOutputPath,
        ResolvedDataFileBinding,
        StageEvidence,
        StrictValueBinding,
        TargetTreeBinding,
        TargetTreeIdentity,
        canonical_json_bytes,
        capture_binding_manifest,
        capture_target_tree_binding,
        json_sha256,
        load_canonical_json,
        publish_json_exclusive,
        revalidate_binding_manifest,
        revalidate_target_tree_binding,
        validate_json_value,
    )

    assert all(
        value is not None
        for value in (
            AbsoluteExecutableBinding,
            AdmissionInspection,
            BindingManifest,
            DirectoryTreeBinding,
            ExecutionEvidenceJournal,
            RegularFileBinding,
            ResearchProbeAdmission,
            ResearchProbeAdmissionError,
            ReservedOutputPath,
            ResolvedDataFileBinding,
            StageEvidence,
            StrictValueBinding,
            TargetTreeBinding,
            TargetTreeIdentity,
            canonical_json_bytes,
            capture_binding_manifest,
            capture_target_tree_binding,
            json_sha256,
            load_canonical_json,
            publish_json_exclusive,
            revalidate_binding_manifest,
            revalidate_target_tree_binding,
            validate_json_value,
        )
    )

    from src.artifacts import json_values, research_probe_admission

    assert publish_json_exclusive is json_values.publish_json_exclusive
    assert BindingManifest is research_probe_admission.BindingManifest
    assert TargetTreeBinding is research_probe_admission.TargetTreeBinding
    assert TargetTreeIdentity is research_probe_admission.TargetTreeIdentity
    assert AdmissionInspection is research_probe_admission.AdmissionInspection
    assert (
        capture_target_tree_binding
        is research_probe_admission.capture_target_tree_binding
    )
    assert revalidate_binding_manifest is (
        research_probe_admission.revalidate_binding_manifest
    )
    assert revalidate_target_tree_binding is (
        research_probe_admission.revalidate_target_tree_binding
    )


def test_artifact_facade_is_lazy_and_cpu_only() -> None:
    script = """
import sys
import src.artifacts as artifacts

deep_modules = {
    "src.artifacts.evidence_journal",
    "src.artifacts.json_values",
    "src.artifacts.research_probe_admission",
}
assert deep_modules.isdisjoint(sys.modules)
assert "torch" not in sys.modules
for name in (
    "ExecutionEvidenceJournal",
    "ResearchProbeAdmission",
    "publish_json_exclusive",
    "BindingManifest",
    "TargetTreeBinding",
    "TargetTreeIdentity",
    "AdmissionInspection",
    "capture_target_tree_binding",
    "revalidate_binding_manifest",
    "revalidate_target_tree_binding",
):
    getattr(artifacts, name)
assert "torch" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_one_shot_publication_refuses_occupied_path_without_mutation(
    tmp_path: Path,
) -> None:
    from src.artifacts import publish_json_exclusive

    result_path = tmp_path / "result.json"
    publish_json_exclusive(result_path, {"status": "caller-owned", "value": 1})
    original_bytes = result_path.read_bytes()
    original_hash = hashlib.sha256(original_bytes).hexdigest()

    with pytest.raises(ArtifactContractError) as occupied:
        publish_json_exclusive(result_path, {"status": "replacement", "value": 2})

    assert occupied.value.code == "artifact.path_already_exists"
    assert result_path.read_bytes() == original_bytes
    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == original_hash
    assert tuple(tmp_path.iterdir()) == (result_path,)


def test_independent_producer_journals_preserve_caller_topology(
    tmp_path: Path,
) -> None:
    from src.artifacts import (
        ExecutionEvidenceJournal,
        json_sha256,
        load_canonical_json,
        publish_json_exclusive,
    )

    common_input = tmp_path / "common-input.json"
    common_value = {"items": ["alpha", "beta"], "owner": "caller"}
    publish_json_exclusive(common_input, common_value)
    common_bytes = common_input.read_bytes()
    identity = {"common_input_sha256": json_sha256(common_value)}
    expected = ("alpha", "beta")
    roots = (tmp_path / "producer-a", tmp_path / "producer-b")
    producer_a = ExecutionEvidenceJournal.create(
        root=roots[0],
        execution_id="producer-a",
        execution_identity=identity,
        expected_work_item_ids=expected,
        context={"stop_rule": "caller-a", "claim": None},
    )
    producer_b = ExecutionEvidenceJournal.create(
        root=roots[1],
        execution_id="producer-b",
        execution_identity=identity,
        expected_work_item_ids=expected,
        context={"threshold": 0.25, "outcome": "caller-b"},
    )

    attempt_a = producer_a.start_attempt()
    attempt_b = producer_b.start_attempt()
    producer_a.append_record(
        work_item_id="alpha",
        payload={"scientific_outcome": "opaque-a-alpha"},
        attempt_id=attempt_a,
    )
    producer_b.append_record(
        work_item_id="beta",
        payload={"scientific_outcome": "opaque-b-beta"},
        attempt_id=attempt_b,
    )
    producer_a.append_record(
        work_item_id="beta",
        payload={"scientific_outcome": "opaque-a-beta"},
        attempt_id=attempt_a,
    )
    producer_a.finalize()
    producer_a.close()

    assert ExecutionEvidenceJournal.inspect(roots[0]).terminal is not None
    assert ExecutionEvidenceJournal.inspect(roots[1]).terminal is None
    producer_a_bytes = {
        path.relative_to(roots[0]): path.read_bytes()
        for path in roots[0].rglob("*.json")
    }

    producer_b.append_record(
        work_item_id="alpha",
        payload={"scientific_outcome": "opaque-b-alpha"},
        attempt_id=attempt_b,
    )
    producer_b.finalize()
    producer_b.close()

    diagnostics_a = ExecutionEvidenceJournal.inspect_diagnostics(roots[0])
    diagnostics_b = ExecutionEvidenceJournal.inspect_diagnostics(roots[1])
    assert [(record.sequence, record.work_item_id) for record in diagnostics_a.records] == [
        (0, "alpha"),
        (1, "beta"),
    ]
    assert [(record.sequence, record.work_item_id) for record in diagnostics_b.records] == [
        (0, "beta"),
        (1, "alpha"),
    ]
    assert diagnostics_b.records[0].payload["scientific_outcome"] == (
        "opaque-b-beta"
    )
    assert common_input.read_bytes() == common_bytes
    assert producer_a_bytes == {
        path.relative_to(roots[0]): path.read_bytes()
        for path in roots[0].rglob("*.json")
    }
    for root in roots:
        terminal = load_canonical_json(root / "terminal.json")
        assert terminal["status"] == "completed"
        assert "scientific_outcome" not in terminal
