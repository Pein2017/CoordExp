from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

from src.detection.coord_soft_targets import CoordSoftTargetRuntimeConfig

from scripts.diagnostics import audit_instance_trie_gaussian_targets as audit


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    REPO_ROOT
    / "configs/stage1/recursive_detection_ce/smoke/"
    "compact_full_support2_instance_trie_focused_cap8_frac0p04_mix0p1_tiny.yaml"
)
REQUIRED_METRIC_KEYS = {
    "candidate_count",
    "effective_candidate_count",
    "posterior_entropy",
    "posterior_top1",
    "target_entropy",
    "target_peak_prob",
    "target_r95_radius",
    "target_std",
    "union_coordinate_probability_ratio",
    "component_mass_by_candidate_id",
}


def _fixture_by_id(artifact: dict, fixture_id: str) -> dict:
    fixtures = {
        str(fixture["fixture_id"]): fixture
        for fixture in artifact["synthetic_fixtures"]
    }
    return fixtures[fixture_id]


def test_build_audit_artifact_schema_and_required_synthetic_fixtures() -> None:
    artifact = audit.build_audit_artifact(
        config_path=CONFIG_PATH,
        include_record_idx=None,
    )

    assert artifact["target_distribution"] == "instance_trie_gaussian"
    assert artifact["gaussian_mixture_weight"] == pytest.approx(0.1)
    assert artifact["gaussian_r95_axis_fraction"] == pytest.approx(0.04)
    assert artifact["gaussian_r95_cap_bins"] == 8
    assert artifact["config_path"] == str(CONFIG_PATH)
    assert artifact["summary"] == {
        "teacher_candidate_missing_count": 0,
        "candidate_leak_count": 0,
        "nonfinite_target_count": 0,
    }
    assert set(artifact["slots"]) == {"x1", "y1", "x2", "y2"}
    assert artifact["real_probe_records"] == []

    fixture_ids = {
        fixture["fixture_id"] for fixture in artifact["synthetic_fixtures"]
    }
    assert {
        "far_apart_same_desc",
        "near_shared_top_left",
        "same_previous_coord_tiny_large",
        "already_emitted_same_desc_excluded",
        "structural_boundary_tiny_boxes",
        "prefix_disambiguates_far_candidate",
    } <= fixture_ids

    for fixture in artifact["synthetic_fixtures"]:
        assert REQUIRED_METRIC_KEYS <= fixture.keys()
        for value in fixture["slots"].values():
            assert REQUIRED_METRIC_KEYS <= value.keys()

    for slot_metrics in artifact["slots"].values():
        assert REQUIRED_METRIC_KEYS <= slot_metrics.keys()


def test_synthetic_fixture_gates_are_recorded() -> None:
    artifact = audit.build_audit_artifact(
        config_path=CONFIG_PATH,
        include_record_idx=None,
    )

    far_apart = _fixture_by_id(artifact, "far_apart_same_desc")
    assert far_apart["slots"]["x1"]["effective_candidate_count"] > 1.5

    near_shared = _fixture_by_id(artifact, "near_shared_top_left")
    assert near_shared["slots"]["x2"]["effective_candidate_count"] > 1.1
    assert near_shared["slots"]["x2"]["posterior_top1"] < 0.95

    prefix_disambiguated = _fixture_by_id(
        artifact,
        "prefix_disambiguates_far_candidate",
    )
    assert (
        prefix_disambiguated["slots"]["x2"]["component_mass_by_candidate_id"][
            "far"
        ]
        < 0.01
    )
    assert (
        prefix_disambiguated["slots"]["x2"][
            "union_coordinate_probability_ratio"
        ]
        < 0.05
    )

    boundary = _fixture_by_id(artifact, "structural_boundary_tiny_boxes")
    assert boundary["slots"]["x1"]["candidate_count"] == 2
    assert boundary["slots"]["y2"]["target_peak_prob"] > 0.0

    excluded = _fixture_by_id(
        artifact,
        "already_emitted_same_desc_excluded",
    )
    assert excluded["candidate_count"] == 1
    assert excluded["component_mass_by_candidate_id"] == {"remaining": 1.0}


def test_missing_sidecar_fixture_reports_hard_error() -> None:
    with pytest.raises(ValueError, match="coord soft target candidates must be non-empty"):
        audit.audit_slot(
            candidates=(),
            runtime_cfg=audit.resolve_runtime_config(CONFIG_PATH),
            slot_name="x1",
        )


def test_real_probe_parses_canonical_bbox_2d_coord_tokens() -> None:
    candidates, desc = audit._real_probe_candidates(
        {
            "objects": [
                {
                    "desc": "orange",
                    "bbox_2d": [
                        "<|coord_100|>",
                        "<|coord_110|>",
                        "<|coord_200|>",
                        "<|coord_240|>",
                    ],
                },
                {
                    "desc": "orange",
                    "bbox_2d": [
                        "<|coord_700|>",
                        "<|coord_120|>",
                        "<|coord_820|>",
                        "<|coord_260|>",
                    ],
                },
                {
                    "desc": "bottle",
                    "bbox_2d": [
                        "<|coord_10|>",
                        "<|coord_20|>",
                        "<|coord_30|>",
                        "<|coord_40|>",
                    ],
                },
            ]
        }
    )

    assert desc == "orange"
    assert candidates == (
        ("record0", (100, 110, 200, 240)),
        ("record1", (700, 120, 820, 260)),
    )


def test_real_probe_replaces_unusable_requested_record_with_fixed_prefix_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jsonl_path = tmp_path / "train.coord.jsonl"
    jsonl_path.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "image_id": "replacement-image",
                        "objects": [
                            {
                                "desc": "orange",
                                "bbox_2d": [
                                    "<|coord_100|>",
                                    "<|coord_110|>",
                                    "<|coord_200|>",
                                    "<|coord_240|>",
                                ],
                            },
                            {
                                "desc": "orange",
                                "bbox_2d": [
                                    "<|coord_700|>",
                                    "<|coord_120|>",
                                    "<|coord_820|>",
                                    "<|coord_260|>",
                                ],
                            },
                        ],
                    }
                ),
                json.dumps(
                    {
                        "image_id": "requested-image",
                        "objects": [
                            {
                                "desc": "bottle",
                                "bbox_2d": [
                                    "<|coord_10|>",
                                    "<|coord_20|>",
                                    "<|coord_30|>",
                                    "<|coord_40|>",
                                ],
                            }
                        ],
                    }
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        audit.ConfigLoader,
        "load_materialized_training_config",
        lambda _path: SimpleNamespace(
            data=SimpleNamespace(train_jsonl=str(jsonl_path))
        ),
    )

    records = audit._build_real_probe_records(
        config_path=tmp_path / "fake.yaml",
        runtime_cfg=CoordSoftTargetRuntimeConfig(
            target_distribution="instance_trie_gaussian",
            coord_token_start=10,
            coord_token_end=1009,
        ),
        include_record_idx=1,
    )

    assert len(records) == 1
    record = records[0]
    assert record["status"] == "ok"
    assert record["requested_record_idx"] == 1
    assert record["record_idx"] == 0
    assert record["selection_reason"] == "replacement_for_unusable_requested_record"
    assert record["image_id"] == "replacement-image"
    assert record["repeated_desc"] == "orange"
    assert record["candidate_count"] == 2


def test_cli_writes_artifact_with_best_effort_real_probe(tmp_path: Path) -> None:
    output_path = tmp_path / "target_shape_audit.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/diagnostics/audit_instance_trie_gaussian_targets.py",
            "--config",
            str(CONFIG_PATH),
            "--output",
            str(output_path),
            "--include-record-idx",
            "27",
        ],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0, result.stderr
    assert output_path.exists()

    artifact = json.loads(output_path.read_text(encoding="utf-8"))
    assert artifact["target_distribution"] == "instance_trie_gaussian"
    assert artifact["summary"]["teacher_candidate_missing_count"] == 0
    assert len(artifact["real_probe_records"]) == 1
    real_probe = artifact["real_probe_records"][0]
    assert real_probe.get("requested_record_idx", real_probe["record_idx"]) == 27
    assert real_probe["status"] in {"ok", "skipped"}
    if real_probe["status"] == "ok":
        assert real_probe["candidate_count"] >= 2
        assert REQUIRED_METRIC_KEYS <= real_probe["slots"]["x1"].keys()
