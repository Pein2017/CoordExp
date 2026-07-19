from pathlib import Path
import json

import pytest

from scripts.research import run_row_four_coordinate_factorial as stage6
from scripts.research.run_complete_candidate_row_scoring import validate_manifest


ADMISSION = Path(
    "research/investigations/qwen3-vl-dense-enumeration/experiments/"
    "2026-07-19-sampled-history-target-reachability-and-complete-row-value/"
    "stage6-row-four-coordinate-factorial-admission.json"
)


def test_frozen_source_has_ten_arms_and_exact_budget() -> None:
    frozen = stage6.load_stage_six_source(ADMISSION)
    assert len(frozen["arms"]) == 10
    assert [len(arm["prefix_token_ids"]) for arm in frozen["arms"]] == [45] * 10
    contract = frozen["admission"]["execution_contract"]
    assert contract["post_prefix_generated_token_budget"] + contract["forced_prefix_token_count"] == 512
    assert {arm["prefix_token_ids_sha256"] for arm in frozen["arms"]}.__len__() == 10


def test_scoring_manifest_is_literal_and_valid() -> None:
    frozen = stage6.load_stage_six_source(ADMISSION)
    manifest = stage6.build_scoring_manifest(frozen)
    normalized = validate_manifest(manifest, manifest_path=ADMISSION)
    assert normalized["schema_version"] == "complete_candidate_row_scoring.manifest.v1"
    assert len(normalized["images"][0]["boundaries"]) == 10
    assert all(len(boundary["candidates"]) == 2 for boundary in normalized["images"][0]["boundaries"])


def test_source_hash_failure_is_fail_closed(tmp_path: Path) -> None:
    document = json.loads(ADMISSION.read_text())
    document["source_stage_five_admission_sha256"] = "0" * 64
    tampered = tmp_path / "admission.json"
    tampered.write_text(json.dumps(document))
    with pytest.raises(stage6.StageSixValidationError):
        stage6.load_stage_six_source(tampered)


def test_invalid_coordinate_fails_closed(tmp_path: Path) -> None:
    document = json.loads(ADMISSION.read_text())
    document["arms"][1]["coordinates"] = [900, 200, 100, 300]
    tampered = tmp_path / "admission.json"
    tampered.write_text(json.dumps(document))
    with pytest.raises(stage6.StageSixValidationError):
        stage6.load_stage_six_source(tampered)


def test_candidate_row_hash_failure_is_fail_closed(tmp_path: Path) -> None:
    document = json.loads(ADMISSION.read_text())
    document["candidate_rows_for_teacher_forced_scoring"][0]["row_token_ids_sha256"] = "0" * 64
    tampered = tmp_path / "admission.json"
    tampered.write_text(json.dumps(document))
    with pytest.raises(stage6.StageSixValidationError):
        stage6.load_stage_six_source(tampered)


def test_primary_and_scoring_outputs_must_be_distinct(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    with pytest.raises(stage6.StageSixValidationError):
        stage6._resolved_output_paths(path, path)
