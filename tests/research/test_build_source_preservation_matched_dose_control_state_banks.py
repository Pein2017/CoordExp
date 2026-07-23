"""Focused contracts for matched-dose Source-preservation repeat controls."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.research.build_source_preservation_matched_dose_control_state_banks import (
    MatchedDoseControlError,
    build_matched_dose_source_preservation_controls,
)
from src.config.fingerprint import sha256_file
from src.rollout_calibration import CheckpointIdentity, assemble_state_bank, load_state_bank


_FIXTURE_SPEC = importlib.util.spec_from_file_location(
    "rollout_calibration_test_conftest",
    Path(__file__).parents[1] / "rollout_calibration" / "conftest.py",
)
assert _FIXTURE_SPEC and _FIXTURE_SPEC.loader
_FIXTURE_MODULE = importlib.util.module_from_spec(_FIXTURE_SPEC)
_FIXTURE_SPEC.loader.exec_module(_FIXTURE_MODULE)
SYNTHETIC_SOURCE_CHECKPOINT = _FIXTURE_MODULE.SYNTHETIC_SOURCE_CHECKPOINT
source_artifacts = _FIXTURE_MODULE.source_artifacts
synthetic_inputs = _FIXTURE_MODULE.synthetic_inputs


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _parent_source_arm(tmp_path: Path) -> Path:
    """Create a two-event canonical Source arm with real core validation."""

    rollouts: list[dict] = []
    reviews: list[dict] = []
    for index in range(2):
        fixture_root = tmp_path / f"fixture-{index}"
        fixture_root.mkdir()
        source_rollouts, source_reviews, image_path = synthetic_inputs(
            fixture_root,
            event_id=f"source-event-{index}",
            image_id=100 + index,
        )
        rollout = source_rollouts[0]
        review = source_reviews[0]
        Image.new("RGB", (64, 64), color=(12 + index, 34, 56)).save(image_path)
        rollout["image"]["content_sha256"] = sha256_file(image_path)
        rollout["candidates"] = [rollout["candidates"][0]]
        rollout["candidates"][0]["generation_provenance"].update(
            {"mode": "greedy", "seed": 0, "temperature": 0.0, "top_p": 1.0}
        )
        review["candidates"] = [review["candidates"][0]]
        review["candidates"][0].update(
            {
                "owner_resolution_interval": [0, 3],
                "selected_sites": [
                    {"candidate_token_offset": 0, "intended_token_type": "desc_text"},
                    {"candidate_token_offset": 1, "intended_token_type": "coordinate"},
                    {"candidate_token_offset": 2, "intended_token_type": "schema"},
                ],
            }
        )
        review.update(
            {
                "entity_transition_eligible": False,
                "coordinate_boundary_eligible": False,
                "positive_path_imitation_eligible": False,
                "source_route_imitation_eligible": True,
                "image_balanced_event_weight": 1.0,
            }
        )
        rollouts.append(rollout)
        reviews.append(review)
    arm_root = tmp_path / "source_preservation_only"
    manifest = assemble_state_bank(
        output_dir=arm_root / "state-bank",
        rollout_rows=rollouts,
        review_rows=reviews,
        source_checkpoint=CheckpointIdentity(**SYNTHETIC_SOURCE_CHECKPOINT),
        prompt_identity_sha256="8" * 64,
        source_artifacts=source_artifacts(),
    )
    _write_jsonl(arm_root / "pre-state-bank" / "rollout_rows.jsonl", rollouts)
    _write_jsonl(arm_root / "pre-state-bank" / "review_rows.jsonl", reviews)
    assert manifest.record_count == 2
    return arm_root / "state-bank" / "manifest.json"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _without_event_id(value: dict) -> dict:
    result = copy.deepcopy(value)
    result.pop("event_id")
    return result


def test_builds_validated_2x_and_3x_controls_without_changing_row_semantics(
    tmp_path: Path,
) -> None:
    parent_manifest = _parent_source_arm(tmp_path)
    parent_root = parent_manifest.parent.parent
    source_rollouts = _read_jsonl(parent_root / "pre-state-bank" / "rollout_rows.jsonl")
    source_reviews = _read_jsonl(parent_root / "pre-state-bank" / "review_rows.jsonl")
    parent_rollout_bytes = (parent_root / "pre-state-bank" / "rollout_rows.jsonl").read_bytes()
    parent_review_bytes = (parent_root / "pre-state-bank" / "review_rows.jsonl").read_bytes()

    output_dir = tmp_path / "matched-dose-controls"
    receipt = build_matched_dose_source_preservation_controls(
        reference_manifest=parent_manifest,
        output_dir=output_dir,
    )

    policy = json.loads((output_dir / "repetition-policy.json").read_text())
    assert policy["source_state_bank"]["manifest_sha256"] == sha256_file(parent_manifest)
    assert policy["source_state_bank"]["records_sha256"] == json.loads(
        parent_manifest.read_text()
    )["records_sha256"]
    assert policy["repetition_policy"]["repeat_factors"] == [2, 3]
    assert policy["repetition_policy"]["mutated_fields"] == ["event_id"]
    assert (parent_root / "pre-state-bank" / "rollout_rows.jsonl").read_bytes() == parent_rollout_bytes
    assert (parent_root / "pre-state-bank" / "review_rows.jsonl").read_bytes() == parent_review_bytes

    parent_binding = json.loads(parent_manifest.read_text())
    parent_records = load_state_bank(
        parent_manifest,
        expected_source_checkpoint=parent_binding["source_checkpoint"],
        expected_prompt_identity_sha256=parent_binding["prompt_identity_sha256"],
    )
    parent_record_by_id = {
        record.event_id: record.to_artifact_dict() for record in parent_records.records
    }
    source_rollout_by_id = {row["event_id"]: row for row in source_rollouts}
    source_review_by_id = {row["event_id"]: row for row in source_reviews}

    for factor, expected_count in ((2, 4), (3, 6)):
        bank_name = f"source_preservation_repeat_factor_{factor}"
        bank_root = output_dir / "state-banks" / bank_name
        rollouts = _read_jsonl(bank_root / "pre-state-bank" / "rollout_rows.jsonl")
        reviews = _read_jsonl(bank_root / "pre-state-bank" / "review_rows.jsonl")
        mapping = receipt["banks"][bank_name]["event_id_mapping"]
        assert len(rollouts) == len(reviews) == len(mapping) == expected_count
        assert [row["event_id"] for row in rollouts] == [row["event_id"] for row in reviews]
        assert len({row["event_id"] for row in rollouts}) == expected_count
        for copied_rollout, copied_review, item in zip(rollouts, reviews, mapping):
            source_event_id = item["source_event_id"]
            assert item["repeat_factor"] == factor
            assert copied_rollout["event_id"] == copied_review["event_id"] == item["event_id"]
            assert _without_event_id(copied_rollout) == _without_event_id(
                source_rollout_by_id[source_event_id]
            )
            assert _without_event_id(copied_review) == _without_event_id(
                source_review_by_id[source_event_id]
            )

        manifest = json.loads((bank_root / "state-bank" / "manifest.json").read_text())
        artifacts = {item["artifact_id"]: item["sha256"] for item in manifest["source_artifacts"]}
        assert artifacts["source-preservation-control-parent-manifest"] == sha256_file(
            parent_manifest
        )
        assert artifacts["source-preservation-control-parent-records"] == parent_binding[
            "records_sha256"
        ]
        assert artifacts["source-preservation-control-repetition-policy"] == sha256_file(
            output_dir / "repetition-policy.json"
        )
        loaded = load_state_bank(
            bank_root / "state-bank" / "manifest.json",
            expected_source_checkpoint=parent_binding["source_checkpoint"],
            expected_prompt_identity_sha256=parent_binding["prompt_identity_sha256"],
        )
        assert loaded.manifest.record_count == expected_count
        assert dict(loaded.manifest.event_family_counts) == {
            "source_route_imitation": expected_count
        }
        for item in mapping:
            record = next(record for record in loaded.records if record.event_id == item["event_id"])
            assert _without_event_id(record.to_artifact_dict()) == _without_event_id(
                parent_record_by_id[item["source_event_id"]]
            )


def test_rejects_an_existing_output_root(tmp_path: Path) -> None:
    parent_manifest = _parent_source_arm(tmp_path)
    output_dir = tmp_path / "matched-dose-controls"
    build_matched_dose_source_preservation_controls(
        reference_manifest=parent_manifest,
        output_dir=output_dir,
    )

    with pytest.raises(MatchedDoseControlError, match="immutable output already exists"):
        build_matched_dose_source_preservation_controls(
            reference_manifest=parent_manifest,
            output_dir=output_dir,
        )


def test_rejects_parent_rows_that_do_not_match_the_canonical_bank(tmp_path: Path) -> None:
    parent_manifest = _parent_source_arm(tmp_path)
    review_path = parent_manifest.parent.parent / "pre-state-bank" / "review_rows.jsonl"
    rows = _read_jsonl(review_path)
    rows[0]["event_id"] = "wrong-event-id"
    _write_jsonl(review_path, rows)

    with pytest.raises(MatchedDoseControlError, match="must exactly equal validated"):
        build_matched_dose_source_preservation_controls(
            reference_manifest=parent_manifest,
            output_dir=tmp_path / "should-not-exist",
        )
