from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.analysis.sorted_random_no_newline_phenotype.runner import (
    STAGE_NAMES,
    run_stages,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_SCRIPT = REPO_ROOT / "scripts/analysis/sorted_random_no_newline_phenotype/run.py"
STATUS_SCRIPT = REPO_ROOT / "scripts/analysis/sorted_random_no_newline_phenotype/status.py"
LAUNCH_SCRIPT = (
    REPO_ROOT
    / "scripts/analysis/sorted_random_no_newline_phenotype/launch_a3_2_tmux.sh"
)
REAL_RUNTIME_SCRIPT = (
    REPO_ROOT / "scripts/analysis/sorted_random_no_newline_phenotype/real_runtime.py"
)

EXPECTED_STAGES = (
    "data_root_audit",
    "prefix_state_index",
    "validate",
    "paired_checkpoint_probe",
    "prefix_merge",
    "prefix_report",
    "prefix_gallery",
    "native_rollout",
    "rollout_phenotype",
    "fn_case_index",
    "fn_hint_probe",
    "fn_merge",
    "fn_report",
    "fn_gallery",
    "finalize",
)
EXPECTED_ROLES = (
    "fullobj_random_pure_ce_ckpt3668",
    "fullobj_sorted_pure_ce_ckpt3668",
)
LEGACY_LABELS = ("phase_a3_1", "ckpt3664", "et_rmp_ce", "pure_minus_et")


def test_runner_dry_run_lists_a3_2_stages_roles_and_shards(
    tmp_path: Path,
) -> None:
    config_path = _write_tiny_config(tmp_path)

    result = run_stages(config_path, dry_run=True)
    payload = json.dumps(result, sort_keys=True)

    assert STAGE_NAMES == EXPECTED_STAGES
    assert tuple(result["stages"]) == EXPECTED_STAGES
    assert result["project_id"] == "sorted_random_no_newline_phenotype"
    assert result["phase_id"] == "phase_a3_2"
    assert result["run_id"] == "fullobj_purece_random_vs_sorted_ckpt3668_phase_a3_2"
    assert tuple(result["checkpoint_roles"]) == EXPECTED_ROLES
    assert result["sampling"]["num_shards"] == 8
    assert result["shard"]["num_shards"] == 8
    assert all(label not in payload for label in LEGACY_LABELS)


def test_runner_rejects_unknown_stage_and_invalid_shards(tmp_path: Path) -> None:
    config_path = _write_tiny_config(tmp_path)

    with pytest.raises(ValueError, match="unknown stage"):
        run_stages(config_path, stages=["not_a_stage"], dry_run=True)

    with pytest.raises(ValueError, match="shard_id"):
        run_stages(
            config_path,
            stages=["paired_checkpoint_probe"],
            dry_run=True,
            shard_id=8,
        )

    with pytest.raises(ValueError, match="requires --shard-id"):
        run_stages(
            config_path,
            stages=["paired_checkpoint_probe"],
            dry_run=False,
        )

    with pytest.raises(ValueError, match="launch context"):
        run_stages(
            config_path,
            stages=["native_rollout"],
            dry_run=False,
        )

    with pytest.raises(ValueError, match="real GPU runtime is not wired"):
        run_stages(
            config_path,
            stages=["paired_checkpoint_probe"],
            dry_run=False,
            shard_id=0,
        )

    with pytest.raises(ValueError, match="real GPU runtime is not wired"):
        run_stages(
            config_path,
            stages=["native_rollout"],
            dry_run=False,
            launch_context=True,
        )


def test_cpu_index_validate_materializes_required_artifacts(
    tmp_path: Path,
) -> None:
    config_path = _write_tiny_config(tmp_path)
    artifact_root = tmp_path / "artifacts"

    result = run_stages(
        config_path,
        stages=["data_root_audit", "prefix_state_index", "validate"],
        dry_run=False,
    )

    assert result["stage_status"] == "index_ready_pending_gpu"
    assert result["launch_eligible"] is True
    assert result["failed_launch_gates"] == []
    for rel_path in (
        "data_root_audit.json",
        "resolved_config.yaml",
        "prefix_state_index.jsonl",
        "prefix_state_sampled_rows.jsonl",
        "prefix_state_index_summary.json",
        "sample_manifest.json",
    ):
        assert (artifact_root / rel_path).is_file()
    status = result["stage_results"]["validate"]["status"]
    assert status["status"] == "index_ready_pending_gpu"
    assert status["index_ready_pending_gpu"] is True


def test_paired_checkpoint_probe_requires_shard_id_when_non_dry(
    tmp_path: Path,
) -> None:
    config_path = _write_tiny_config(tmp_path)
    run_stages(
        config_path,
        stages=["data_root_audit", "prefix_state_index", "validate"],
        dry_run=False,
    )

    with pytest.raises(ValueError, match="requires --shard-id"):
        run_stages(
            config_path,
            stages=["paired_checkpoint_probe"],
            dry_run=False,
        )


def test_cli_does_not_expose_mock_runtime_flag(tmp_path: Path) -> None:
    config_path = _write_tiny_config(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            str(RUN_SCRIPT),
            "--config",
            str(config_path),
            "--stages",
            "native_rollout",
            "--launch-context",
            "--mock-runtime-for-tests",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert completed.returncode != 0
    assert "unrecognized arguments" in completed.stderr


def test_real_runtime_cli_requires_launch_context_and_stage_guards(
    tmp_path: Path,
) -> None:
    config_path = _write_tiny_config(tmp_path)

    missing_launch_context = subprocess.run(
        [
            sys.executable,
            str(REAL_RUNTIME_SCRIPT),
            "--config",
            str(config_path),
            "--stages",
            "paired_checkpoint_probe",
            "--shard-id",
            "0",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    assert missing_launch_context.returncode != 0
    assert "require --launch-context" in missing_launch_context.stderr

    missing_shard = subprocess.run(
        [
            sys.executable,
            str(REAL_RUNTIME_SCRIPT),
            "--config",
            str(config_path),
            "--stages",
            "paired_checkpoint_probe",
            "--launch-context",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    assert missing_shard.returncode != 0
    assert "requires --shard-id" in missing_shard.stderr

    missing_fn_shard = subprocess.run(
        [
            sys.executable,
            str(REAL_RUNTIME_SCRIPT),
            "--config",
            str(config_path),
            "--stages",
            "fn_hint_probe",
            "--launch-context",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    assert missing_fn_shard.returncode != 0
    assert "fn_hint_probe requires --shard-id" in missing_fn_shard.stderr


def test_internal_mock_runtime_respects_overwrite_guard(tmp_path: Path) -> None:
    config_path = _write_tiny_config(tmp_path)
    existing = (
        tmp_path
        / "artifacts"
        / "rollout"
        / "fullobj_random_pure_ce_ckpt3668"
        / "gt_vs_pred.jsonl"
    )
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_text('{"real": true}\n', encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_stages(
            config_path,
            stages=["native_rollout"],
            dry_run=False,
            launch_context=True,
            mock_runtime=True,
        )

    assert existing.read_text(encoding="utf-8") == '{"real": true}\n'


def test_loop_stages_preflight_all_outputs_before_mutation(tmp_path: Path) -> None:
    config_path = _write_tiny_config(tmp_path)
    artifact_root = tmp_path / "artifacts"
    run_stages(
        config_path,
        stages=["data_root_audit", "prefix_state_index", "validate"],
        dry_run=False,
    )

    sorted_existing = (
        artifact_root
        / "rollout"
        / "fullobj_sorted_pure_ce_ckpt3668"
        / "gt_vs_pred.jsonl"
    )
    sorted_existing.parent.mkdir(parents=True, exist_ok=True)
    sorted_existing.write_text("sorted sentinel\n", encoding="utf-8")
    random_rollout_dir = (
        artifact_root / "rollout" / "fullobj_random_pure_ce_ckpt3668"
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_stages(
            config_path,
            stages=["native_rollout"],
            dry_run=False,
            launch_context=True,
            mock_runtime=True,
        )

    assert not (random_rollout_dir / "gt_vs_pred.jsonl").exists()
    assert sorted_existing.read_text(encoding="utf-8") == "sorted sentinel\n"

    summary_path = artifact_root / "prefix_state_shard_summaries.jsonl"
    summary_path.write_text("summary sentinel\n", encoding="utf-8")
    shard_1_path = artifact_root / "prefix_readout_shards" / "shard_1.jsonl"

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_stages(
            config_path,
            stages=["paired_checkpoint_probe"],
            dry_run=False,
            shard_id=1,
            mock_runtime=True,
        )

    assert not shard_1_path.exists()
    assert summary_path.read_text(encoding="utf-8") == "summary sentinel\n"

    fn_cases = artifact_root / "fn_probe" / "fn_cases.jsonl"
    _write_jsonl(
        fn_cases,
        [
            {"fn_case_id": "case-0", "fn_desc": "person", "fn_bbox": [1, 2, 3, 4]},
            {"fn_case_id": "case-1", "fn_desc": "person", "fn_bbox": [5, 6, 7, 8]},
        ],
    )
    later_shard = (
        artifact_root
        / "fn_probe"
        / "fn_hint_shards"
        / "shard_1_probe_rows.jsonl"
    )
    later_shard.parent.mkdir(parents=True, exist_ok=True)
    later_shard.write_text("fn sentinel\n", encoding="utf-8")
    earlier_shard = later_shard.parent / "shard_0_probe_rows.jsonl"

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_stages(
            config_path,
            stages=["fn_hint_probe"],
            dry_run=False,
            launch_context=True,
            mock_runtime=True,
        )

    assert not earlier_shard.exists()
    assert later_shard.read_text(encoding="utf-8") == "fn sentinel\n"


def test_materializer_stages_respect_overwrite_guard(tmp_path: Path) -> None:
    config_path = _write_tiny_config(tmp_path)
    artifact_root = tmp_path / "artifacts"
    run_stages(
        config_path,
        stages=["data_root_audit", "prefix_state_index", "validate"],
        dry_run=False,
    )

    cases = (
        ("prefix_gallery", artifact_root / "gallery" / "index.md", {}),
        (
            "rollout_phenotype",
            artifact_root / "rollout" / "rollout_phenotype_rows.jsonl",
            {},
        ),
        (
            "fn_case_index",
            artifact_root / "fn_probe" / "fn_case_universe.jsonl",
            {},
        ),
        (
            "fn_merge",
            artifact_root / "fn_probe" / "fn_probe_rows.jsonl",
            {},
        ),
        (
            "fn_gallery",
            artifact_root / "fn_probe" / "gallery" / "index.md",
            {},
        ),
    )
    for stage, sentinel, kwargs in cases:
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.write_text("sentinel\n", encoding="utf-8")

        with pytest.raises(FileExistsError, match="refusing to overwrite"):
            run_stages(
                config_path,
                stages=[stage],
                dry_run=False,
                **kwargs,
            )

        assert sentinel.read_text(encoding="utf-8") == "sentinel\n"

    run_stages(
        config_path,
        stages=["paired_checkpoint_probe"],
        dry_run=False,
        shard_id=0,
        mock_runtime=True,
    )
    summary_path = artifact_root / "prefix_state_shard_summaries.jsonl"
    original_summary = summary_path.read_text(encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_stages(
            config_path,
            stages=["paired_checkpoint_probe"],
            dry_run=False,
            shard_id=1,
            mock_runtime=True,
        )

    assert summary_path.read_text(encoding="utf-8") == original_summary


def test_fn_case_index_uses_real_native_rollout_rows(tmp_path: Path) -> None:
    config_path = _write_tiny_config(tmp_path)
    artifact_root = tmp_path / "artifacts"
    roles = (
        "fullobj_random_pure_ce_ckpt3668",
        "fullobj_sorted_pure_ce_ckpt3668",
    )
    gt = [
        {"gt_idx": 0, "desc": "person", "bbox_xyxy": [0, 0, 10, 10]},
        {"gt_idx": 1, "desc": "chair", "bbox_xyxy": [20, 0, 30, 10]},
    ]
    rollout_common = {
        "runtime_kind": "real_gpu_native_rollout_v1",
        "decode_policy": "free_text_unconstrained_greedy_temp0",
        "constraint_policy": "none",
        "gpu_id": "0",
        "source_line_idx": 0,
        "image_id": 3,
        "image": "images/val/000000000003.jpg",
        "width": 1024,
        "height": 768,
        "gt": gt,
        "template_contract": {
            "detection_sequence_format": "compact_full",
            "coordinate_surface": "coord_token",
            "bbox_format": "xyxy",
            "row_separator": "none",
        },
        "errors": [],
    }
    _write_jsonl(
        artifact_root / "rollout" / roles[0] / "gt_vs_pred.jsonl",
        [
            {
                **rollout_common,
                "checkpoint_role": roles[0],
                "checkpoint_fingerprint": "model:random",
                "pred": [
                    {"pred_idx": 0, "desc": "person", "bbox_xyxy": [0, 0, 10, 10]}
                ],
            }
        ],
    )
    _write_jsonl(
        artifact_root / "rollout" / roles[1] / "gt_vs_pred.jsonl",
        [
            {
                **rollout_common,
                "checkpoint_role": roles[1],
                "checkpoint_fingerprint": "model:sorted",
                "pred": [
                    {"pred_idx": 0, "desc": "person", "bbox_xyxy": [0, 0, 10, 10]},
                    {"pred_idx": 1, "desc": "chair", "bbox_xyxy": [20, 0, 30, 10]},
                ],
            }
        ],
    )

    result = run_stages(config_path, stages=["fn_case_index"], dry_run=False)

    assert result["stage_results"]["fn_case_index"]["fn_cases"] == 1
    universe_rows = [
        json.loads(line)
        for line in (artifact_root / "fn_probe" / "fn_case_universe.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    case_rows = [
        json.loads(line)
        for line in (artifact_root / "fn_probe" / "fn_cases.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert {row["fn_membership"] for row in universe_rows} == {
        "not_fn",
        "random_only_fn",
    }
    assert case_rows[0]["checkpoint_role"] == roles[0]
    assert case_rows[0]["fn_desc"] == "chair"
    assert case_rows[0]["checkpoint_fingerprint"] == "model:random"
    serialized = json.dumps({"universe": universe_rows, "cases": case_rows})
    assert "mocked_cpu_status_case" not in serialized
    assert "mocked_runtime" not in serialized


def test_mocked_runtime_artifacts_do_not_satisfy_final_status(
    tmp_path: Path,
) -> None:
    config_path = _write_tiny_config(tmp_path)

    result = run_stages(
        config_path,
        stages=[
            "data_root_audit",
            "prefix_state_index",
            "paired_checkpoint_probe",
            "prefix_merge",
            "prefix_report",
            "prefix_gallery",
            "native_rollout",
            "rollout_phenotype",
            "fn_case_index",
            "fn_hint_probe",
            "fn_merge",
            "fn_report",
            "fn_gallery",
            "finalize",
        ],
        dry_run=False,
        launch_context=True,
        allow_overwrite=True,
        mock_runtime=True,
    )

    status = result["stage_results"]["finalize"]["status"]
    assert status["status"] == "incomplete"
    assert status["final_artifacts_present"] is False
    assert "real_runtime_artifacts_present" in status["failed_gates"]


def test_run_py_dry_run_prints_json(tmp_path: Path) -> None:
    config_path = _write_tiny_config(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            str(RUN_SCRIPT),
            "--config",
            str(config_path),
            "--stages",
            "validate",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    result = json.loads(completed.stdout)
    assert result["dry_run"] is True
    assert result["stages"] == ["validate"]
    assert result["checkpoint_roles"] == list(EXPECTED_ROLES)


def test_status_py_prints_json(tmp_path: Path) -> None:
    config_path = _write_tiny_config(tmp_path)
    artifact_root = tmp_path / "artifacts"
    run_stages(
        config_path,
        stages=["data_root_audit", "prefix_state_index", "validate"],
        dry_run=False,
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(STATUS_SCRIPT),
            "--artifact-root",
            str(artifact_root),
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    result = json.loads(completed.stdout)
    assert result["status"] == "index_ready_pending_gpu"
    assert result["index_ready_pending_gpu"] is True


def test_launch_a3_2_tmux_dry_run_plan_and_refusals(tmp_path: Path) -> None:
    config_path = _write_tiny_config(tmp_path)
    env = {
        **os.environ,
        "DRY_RUN": "1",
        "CONFIG": str(config_path),
        "SESSION": "sorted_random_no_newline_a3_2_ckpt3668",
    }

    completed = subprocess.run(
        ["bash", str(LAUNCH_SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    output = completed.stdout
    assert "DRY_RUN=1" in output
    assert "sorted_random_no_newline_phenotype" in output
    assert "phase_a3_2" in output
    assert "num_shards=8" in output
    assert "fullobj_random_pure_ce_ckpt3668" in output
    assert "fullobj_sorted_pure_ce_ckpt3668" in output
    assert "tmux new-session" not in output
    assert all(label not in output for label in LEGACY_LABELS)

    refused_root = subprocess.run(
        ["bash", str(LAUNCH_SCRIPT)],
        cwd=REPO_ROOT,
        env={**env, "ARTIFACT_ROOT": str(tmp_path / "elsewhere")},
        text=True,
        capture_output=True,
    )
    assert refused_root.returncode != 0
    assert "ARTIFACT_ROOT override" in refused_root.stderr

    refused_gpu_count = subprocess.run(
        ["bash", str(LAUNCH_SCRIPT)],
        cwd=REPO_ROOT,
        env={**env, "GPU_IDS": "0 1"},
        text=True,
        capture_output=True,
    )
    assert refused_gpu_count.returncode != 0
    assert "GPU_IDS count" in refused_gpu_count.stderr

    run_stages(
        config_path,
        stages=["data_root_audit", "prefix_state_index", "validate"],
        dry_run=False,
    )
    real_launch_refused = subprocess.run(
        ["bash", str(LAUNCH_SCRIPT)],
        cwd=REPO_ROOT,
        env={**env, "DRY_RUN": "0"},
        text=True,
        capture_output=True,
    )
    assert real_launch_refused.returncode != 0
    assert "real GPU runtime is not wired" in real_launch_refused.stderr
    assert "tmux new-session" not in real_launch_refused.stdout

    artifact_root = tmp_path / "artifacts"
    (artifact_root / "shard_pids.tsv").write_text("stale\n", encoding="utf-8")
    stale_refused = subprocess.run(
        ["bash", str(LAUNCH_SCRIPT)],
        cwd=REPO_ROOT,
        env={
            **env,
            "DRY_RUN": "0",
            "ENABLE_A3_2_REAL_GPU_RUNTIME": "1",
            "REAL_GPU_RUNTIME_CMD": "/bin/true",
            "SKIP_GPU_PREFLIGHT": "1",
        },
        text=True,
        capture_output=True,
    )
    assert stale_refused.returncode != 0
    assert "refusing to overwrite" in stale_refused.stderr

    external_log_root = tmp_path / "external_logs"
    external_log_root.mkdir()
    marker = external_log_root / "marker.txt"
    marker.write_text("keep me\n", encoding="utf-8")
    disabled_runtime_refused = subprocess.run(
        ["bash", str(LAUNCH_SCRIPT)],
        cwd=REPO_ROOT,
        env={
            **env,
            "DRY_RUN": "0",
            "ALLOW_OVERWRITE": "1",
            "LOG_ROOT": str(external_log_root),
        },
        text=True,
        capture_output=True,
    )
    assert disabled_runtime_refused.returncode != 0
    assert "real GPU runtime is not wired" in disabled_runtime_refused.stderr
    assert marker.read_text(encoding="utf-8") == "keep me\n"

    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_tmux = fake_bin / "tmux"
    fake_tmux.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$1\" == \"has-session\" ]]; then exit 0; fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake_tmux.chmod(0o755)
    (artifact_root / "shard_status.log").write_text("stale-status\n", encoding="utf-8")
    metadata_before = (artifact_root / "shard_pids.tsv").read_text(encoding="utf-8")
    status_before = (artifact_root / "shard_status.log").read_text(encoding="utf-8")
    later_refusal = subprocess.run(
        ["bash", str(LAUNCH_SCRIPT)],
        cwd=REPO_ROOT,
        env={
            **env,
            "DRY_RUN": "0",
            "ALLOW_OVERWRITE": "1",
            "ENABLE_A3_2_REAL_GPU_RUNTIME": "1",
            "REAL_GPU_RUNTIME_CMD": "/bin/true",
            "SKIP_GPU_PREFLIGHT": "1",
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        },
        text=True,
        capture_output=True,
    )
    assert later_refusal.returncode != 0
    assert "tmux session already exists" in later_refusal.stderr
    assert (artifact_root / "shard_pids.tsv").read_text(encoding="utf-8") == metadata_before
    assert (artifact_root / "shard_status.log").read_text(encoding="utf-8") == status_before

    fresh_root = tmp_path / "fresh-artifacts"
    fresh_config = _write_tiny_config(
        tmp_path / "fresh",
        artifact_root=fresh_root,
    )
    run_stages(
        fresh_config,
        stages=["data_root_audit", "prefix_state_index", "validate"],
        dry_run=False,
    )
    fresh_log_root = tmp_path / "fresh-logs"
    fake_tmux_clean = fake_bin / "tmux-clean"
    fake_tmux_clean.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_tmux_clean.chmod(0o755)
    clean_fake_bin = tmp_path / "fake-bin-clean"
    clean_fake_bin.mkdir()
    (clean_fake_bin / "tmux").write_text(fake_tmux_clean.read_text(encoding="utf-8"), encoding="utf-8")
    (clean_fake_bin / "tmux").chmod(0o755)
    unwired_refused = subprocess.run(
        ["bash", str(LAUNCH_SCRIPT)],
        cwd=REPO_ROOT,
        env={
            **env,
            "CONFIG": str(fresh_config),
            "DRY_RUN": "0",
            "ENABLE_A3_2_REAL_GPU_RUNTIME": "1",
            "SKIP_GPU_PREFLIGHT": "1",
            "LOG_ROOT": str(fresh_log_root),
            "PATH": f"{clean_fake_bin}{os.pathsep}{os.environ['PATH']}",
        },
        text=True,
        capture_output=True,
    )
    assert unwired_refused.returncode != 0
    assert "real GPU runtime command is not configured" in unwired_refused.stderr
    assert not (fresh_root / "shard_pids.tsv").exists()
    assert not (fresh_root / "shard_status.log").exists()
    assert not fresh_log_root.exists()


def _write_tiny_config(tmp_path: Path, *, artifact_root: Path | None = None) -> Path:
    data_root = tmp_path / "rescale_32_1024_bbox_len12000"
    image_root = tmp_path / "rescale_32_1024_bbox"
    train_jsonl = data_root / "train.coord.jsonl"
    val_jsonl = data_root / "val.coord.jsonl"
    _write_jsonl(train_jsonl, [_record(1, "train"), _record(2, "train")])
    _write_jsonl(val_jsonl, [_record(3, "val")])
    for split, image_id in (("train", 1), ("train", 2), ("val", 3)):
        image_path = image_root / "images" / split / f"{image_id:012d}.jpg"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"tiny")

    random_checkpoint = tmp_path / "random" / "checkpoint-3668"
    sorted_checkpoint = tmp_path / "sorted" / "checkpoint-3668"
    random_checkpoint.mkdir(parents=True)
    sorted_checkpoint.mkdir(parents=True)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "project_id": "sorted_random_no_newline_phenotype",
                "phase_id": "phase_a3_2",
                "schema_version": "a3.2.v1",
                "run_id": "fullobj_purece_random_vs_sorted_ckpt3668_phase_a3_2",
                "artifact_root": str(artifact_root or tmp_path / "artifacts"),
                "train_jsonl": str(train_jsonl),
                "val_jsonl": str(val_jsonl),
                "image_root": str(image_root),
                "template_contract": {
                    "detection_sequence_format": "compact_full",
                    "coordinate_surface": "coord_token",
                    "bbox_format": "xyxy",
                    "row_separator": "none",
                },
                "sampling": {
                    "max_prefix_states": 8,
                    "num_shards": 8,
                    "seed": 3668,
                    "easy_sanity_max_fraction": 0.20,
                },
                "rollout": {
                    "limit_images": 4,
                    "decode_policy": "free_text_unconstrained_greedy_temp0",
                    "native_prompt_ordering": True,
                    "constraint_policy": "none",
                },
                "fn_probe": {
                    "max_fn_objects_per_checkpoint": 4,
                    "hint_policy_id": "desc_x1_r95_ladder_v1",
                    "strict_r95_axis_fraction": 0.04,
                    "strict_r95_cap_bins": 8,
                    "broad_x1_radius": 24,
                },
                "peak": {
                    "absolute_mass_floor": 0.002,
                    "relative_floor": 0.10,
                    "primary_merge_radius": 24,
                    "gt_x1_neighborhood_radius": 24,
                    "raw_topk_k": 32,
                },
                "checkpoints": {
                    "fullobj_random_pure_ce_ckpt3668": {
                        "training_ordering": "random_permutation",
                        "readout_prompt_ordering": "sorted",
                        "checkpoint_path": str(random_checkpoint),
                    },
                    "fullobj_sorted_pure_ce_ckpt3668": {
                        "training_ordering": "sorted",
                        "readout_prompt_ordering": "sorted",
                        "checkpoint_path": str(sorted_checkpoint),
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config_path


def _record(image_id: int, split: str) -> dict[str, Any]:
    return {
        "image_id": image_id,
        "file_name": f"images/{split}/{image_id:012d}.jpg",
        "width": 1024,
        "height": 768,
        "objects": [
            {"label": "person", "bbox_2d": [50, 10, 100, 90]},
            {"label": "chair", "bbox_2d": [140, 40, 200, 120]},
            {"label": "person", "bbox_2d": [20, 200, 80, 280]},
        ],
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
