from __future__ import annotations

import inspect
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from src.analysis.sorted_random_no_newline_phenotype import (
    PHASE_ID,
    PROJECT_ID,
    RUN_ID,
    SCHEMA_VERSION,
)
from src.analysis.sorted_random_no_newline_phenotype.config import (
    A32Config,
    CheckpointConfig,
    FNProbeConfig,
    PeakConfig,
    RolloutConfig,
    SamplingConfig,
    TemplateContractConfig,
)
from src.analysis.sorted_random_no_newline_phenotype import paired_probe
from src.analysis.sorted_random_no_newline_phenotype.paired_probe import (
    REAL_PREFIX_RUNTIME_KIND,
    build_real_paired_readout_rows,
    build_real_shard_summary_row,
    build_mocked_paired_readout_rows,
    build_shard_manifest_row,
    checkpoint_roles_from_config,
)


def test_checkpoint_roles_are_read_dynamically_from_config(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        checkpoint_roles=(
            ("alpha_random_ckpt9001", "random_permutation"),
            ("beta_sorted_ckpt9001", "sorted"),
            ("gamma_curriculum_ckpt9001", "curriculum_yx"),
        ),
    )

    assert checkpoint_roles_from_config(config) == [
        "alpha_random_ckpt9001",
        "beta_sorted_ckpt9001",
        "gamma_curriculum_ckpt9001",
    ]


def test_shard_manifest_preserves_a3_2_prefix_and_training_provenance(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        checkpoint_roles=(
            ("alpha_random_ckpt9001", "random_permutation"),
            ("beta_sorted_ckpt9001", "sorted"),
        ),
    )
    manifest = build_shard_manifest_row(
        config,
        shard_id=3,
        prefix_state_rows=[
            {"prefix_state_id": "prefix-a", "shard_id": 3},
            {"prefix_state_id": "prefix-b", "shard_id": 3},
        ],
    )

    assert manifest["project_id"] == PROJECT_ID
    assert manifest["phase_id"] == PHASE_ID
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["run_id"] == RUN_ID
    assert manifest["shard_id"] == 3
    assert manifest["checkpoint_roles"] == [
        "alpha_random_ckpt9001",
        "beta_sorted_ckpt9001",
    ]
    assert manifest["checkpoint_training_ordering"] == {
        "alpha_random_ckpt9001": "random_permutation",
        "beta_sorted_ckpt9001": "sorted",
    }
    assert manifest["checkpoint_readout_prompt_ordering"] == {
        "alpha_random_ckpt9001": "sorted",
        "beta_sorted_ckpt9001": "sorted",
    }
    assert manifest["prefix_source_policy"] == "canonical_sorted_teacher_prefix_readout"
    assert manifest["readout_prompt_ordering"] == "sorted"
    assert manifest["teacher_prefix_ordering"] == "canonical_sorted_yx_teacher_v1"
    assert manifest["readout_behavior"] == "canonical sorted teacher-prefix readout"
    assert manifest["prefix_state_ids"] == ["prefix-a", "prefix-b"]
    assert manifest["prefix_state_count"] == 2
    serialized = json.dumps(manifest, sort_keys=True)
    json.dumps(manifest, allow_nan=False, sort_keys=True)
    assert "phase_a3_1" not in serialized
    assert "ckpt3664" not in serialized
    assert "et_rmp_ce" not in serialized
    assert "native rollout behavior" not in serialized


def test_mocked_paired_readout_combines_prefix_rows_and_per_role_summaries(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        checkpoint_roles=(
            ("alpha_random_ckpt9001", "random_permutation"),
            ("beta_sorted_ckpt9001", "sorted"),
        ),
    )
    prefix_rows = [
        {
            "prefix_state_id": "prefix-a",
            "shard_id": 5,
            "image_id": 17,
            "prefix_source_policy": "canonical_sorted_teacher_prefix_readout",
        }
    ]
    boundary_summaries = {
        "alpha_random_ckpt9001": {
            "prefix-a": {
                "winner_desc": "chair",
                "winner_roles": ["residual_same_desc"],
                "boundary_winner_class": "residual_same_desc_favored",
                "candidate_descs_with_roles": [
                    {"desc": "chair", "roles": ["residual_same_desc"], "score": 1.2}
                ],
            }
        },
        "beta_sorted_ckpt9001": {
            "prefix-a": {
                "winner_desc": "lamp",
                "winner_roles": ["hard_competitor"],
                "boundary_winner_class": "hard_competitor_favored",
                "candidate_descs_with_roles": [
                    {"desc": "lamp", "roles": ["hard_competitor"], "score": 1.1}
                ],
            }
        },
    }

    rows = build_mocked_paired_readout_rows(
        config,
        prefix_state_rows=prefix_rows,
        boundary_summaries_by_role=boundary_summaries,
    )

    assert [row["checkpoint_role"] for row in rows] == [
        "alpha_random_ckpt9001",
        "beta_sorted_ckpt9001",
    ]
    assert all(
        row["prefix_source_policy"] == "canonical_sorted_teacher_prefix_readout"
        for row in rows
    )
    assert all(row["readout_prompt_ordering"] == "sorted" for row in rows)
    assert all(
        row["teacher_prefix_ordering"] == "canonical_sorted_yx_teacher_v1"
        for row in rows
    )
    assert rows[0]["checkpoint_training_ordering"] == "random_permutation"
    assert rows[1]["checkpoint_training_ordering"] == "sorted"
    assert rows[0]["boundary_summary"]["winner_roles"] == ["residual_same_desc"]
    assert rows[1]["boundary_summary"]["winner_roles"] == ["hard_competitor"]
    assert rows[0]["readout_behavior"] == "canonical sorted teacher-prefix readout"
    serialized = json.dumps(rows, sort_keys=True)
    json.dumps(rows, allow_nan=False, sort_keys=True)
    assert "phase_a3_1" not in serialized
    assert "ckpt3664" not in serialized
    assert "et_rmp_ce" not in serialized
    assert "native rollout behavior" not in serialized


def test_real_paired_readout_rows_have_runtime_markers_and_metrics(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        checkpoint_roles=(
            ("alpha_random_ckpt9001", "random_permutation"),
            ("beta_sorted_ckpt9001", "sorted"),
        ),
    )
    prefix_rows = [
        {
            "prefix_state_id": "prefix-a",
            "shard_id": 2,
            "image_id": 17,
            "candidate_descs_with_roles": [
                {
                    "desc": "person",
                    "roles": ["residual_same_desc"],
                }
            ],
        }
    ]
    boundary_summaries = {
        "alpha_random_ckpt9001": {
            "prefix-a": {
                "winner_desc": "person",
                "winner_roles": ["residual_same_desc"],
                "boundary_winner_class": "residual_same_desc_favored",
                "residual_vs_eos_margin": 0.25,
                "residual_vs_winner_margin": 0.0,
                "strict_r95_x1_hit_rate": 0.5,
                "boundary_residual_favored_rate": 1.0,
                "candidate_descs_with_roles": [
                    {"desc": "person", "roles": ["residual_same_desc"], "score": 1.0}
                ],
            }
        },
        "beta_sorted_ckpt9001": {
            "prefix-a": {
                "winner_desc": "chair",
                "winner_roles": ["hard_competitor"],
                "boundary_winner_class": "hard_competitor_favored",
                "residual_vs_eos_margin": -0.1,
                "residual_vs_winner_margin": -0.2,
                "strict_r95_x1_hit_rate": 0.0,
                "boundary_residual_favored_rate": 0.0,
                "candidate_descs_with_roles": [
                    {"desc": "chair", "roles": ["hard_competitor"], "score": 1.1}
                ],
            }
        },
    }

    rows = build_real_paired_readout_rows(
        config,
        prefix_state_rows=prefix_rows,
        boundary_summaries_by_role=boundary_summaries,
        shard_id=2,
        gpu_id="7",
        checkpoint_fingerprints={
            "alpha_random_ckpt9001": "checkpoint:a",
            "beta_sorted_ckpt9001": "checkpoint:b",
        },
    )

    assert len(rows) == 2
    assert {row["runtime_kind"] for row in rows} == {REAL_PREFIX_RUNTIME_KIND}
    assert {row["gpu_id"] for row in rows} == {"7"}
    assert rows[0]["shard_id"] == 2
    assert rows[0]["constraint_policy"] == "none"
    assert rows[0]["checkpoint_fingerprint"] == "checkpoint:a"
    assert rows[0]["residual_vs_eos_margin"] == 0.25
    assert rows[0]["strict_r95_x1_hit_rate"] == 0.5
    json.dumps(rows, allow_nan=False, sort_keys=True)


def test_real_shard_summary_row_is_status_compatible(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        checkpoint_roles=(
            ("alpha_random_ckpt9001", "random_permutation"),
            ("beta_sorted_ckpt9001", "sorted"),
        ),
    )

    row = build_real_shard_summary_row(
        config,
        shard_id=6,
        gpu_id="0",
        prefix_state_count=3,
        readout_row_count=6,
    )

    assert row["runtime_kind"] == REAL_PREFIX_RUNTIME_KIND
    assert row["gpu_id"] == "0"
    assert row["shard_id"] == 6
    assert row["prefix_state_count"] == 3
    assert row["readout_row_count"] == 6
    assert row["checkpoint_roles"] == [
        "alpha_random_ckpt9001",
        "beta_sorted_ckpt9001",
    ]
    json.dumps(row, allow_nan=False, sort_keys=True)


def test_paired_probe_source_does_not_define_or_import_a3_1_role_tuple() -> None:
    source = inspect.getsource(paired_probe)

    assert "CHECKPOINT_ROLES" not in source
    assert "et_rmp_ce" not in source
    assert "ckpt3664" not in source
    assert "phase_a3_1" not in source


def test_prefix_token_alignment_allows_cpu_prefix_and_cuda_full_ids() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for cross-device alignment regression")

    prefix_ids = torch.tensor([[11, 22, 33]], device="cpu")
    full_ids = torch.tensor([[11, 22, 33, 44]], device="cuda")

    paired_probe._assert_prefix_token_alignment(
        prefix_ids,
        full_ids,
        suffix_start=3,
    )


def test_mocked_paired_readout_rejects_nested_non_finite_prefix_state_row(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        checkpoint_roles=(("alpha_random_ckpt9001", "random_permutation"),),
    )

    with pytest.raises(ValueError, match="prefix_state_row.*finite"):
        build_mocked_paired_readout_rows(
            config,
            prefix_state_rows=[
                {
                    "prefix_state_id": "prefix-a",
                    "nested": {"bad_score": math.inf},
                }
            ],
            boundary_summaries_by_role={
                "alpha_random_ckpt9001": {
                    "prefix-a": {
                        "winner_desc": "chair",
                        "winner_roles": ["residual_same_desc"],
                        "boundary_winner_class": "residual_same_desc_favored",
                    }
                }
            },
        )


def test_mocked_paired_readout_rejects_nested_non_finite_boundary_summary(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        checkpoint_roles=(("alpha_random_ckpt9001", "random_permutation"),),
    )

    with pytest.raises(ValueError, match="boundary_summary.*finite"):
        build_mocked_paired_readout_rows(
            config,
            prefix_state_rows=[{"prefix_state_id": "prefix-a"}],
            boundary_summaries_by_role={
                "alpha_random_ckpt9001": {
                    "prefix-a": {
                        "winner_desc": "chair",
                        "winner_roles": ["residual_same_desc"],
                        "boundary_winner_class": "residual_same_desc_favored",
                        "candidate_descs_with_roles": [
                            {"desc": "chair", "score": math.nan}
                        ],
                    }
                }
            },
        )


def test_paired_probe_import_does_not_pull_heavy_or_yaml_modules() -> None:
    script = """
import json
import sys
import src.analysis.sorted_random_no_newline_phenotype.paired_probe
blocked = ["yaml", "torch", "transformers", "PIL", "numpy"]
print(json.dumps({name: name in sys.modules for name in blocked}, sort_keys=True))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        text=True,
        capture_output=True,
    )

    assert json.loads(result.stdout) == {
        "PIL": False,
        "numpy": False,
        "torch": False,
        "transformers": False,
        "yaml": False,
    }


def _config(
    tmp_path: Path,
    *,
    checkpoint_roles: tuple[tuple[str, str], ...],
) -> A32Config:
    checkpoints: dict[str, CheckpointConfig] = {}
    for role, training_ordering in checkpoint_roles:
        checkpoint_path = tmp_path / role / "checkpoint"
        checkpoint_path.mkdir(parents=True)
        checkpoints[role] = CheckpointConfig(
            checkpoint_path=checkpoint_path,
            training_ordering=training_ordering,
            readout_prompt_ordering="sorted",
        )

    return A32Config(
        project_id=PROJECT_ID,
        phase_id=PHASE_ID,
        schema_version=SCHEMA_VERSION,
        run_id=RUN_ID,
        artifact_root=tmp_path / "artifacts",
        train_jsonl=tmp_path / "train.coord.jsonl",
        val_jsonl=tmp_path / "val.coord.jsonl",
        image_root=tmp_path / "rescale_32_1024_bbox",
        checkpoints=checkpoints,
        template_contract=TemplateContractConfig(
            detection_sequence_format="compact_full",
            coordinate_surface="coord_token",
            bbox_format="xyxy",
            row_separator="none",
        ),
        sampling=SamplingConfig(max_prefix_states=8, num_shards=8, seed=3668),
        rollout=RolloutConfig(),
        fn_probe=FNProbeConfig(),
        peak=PeakConfig(),
    )
