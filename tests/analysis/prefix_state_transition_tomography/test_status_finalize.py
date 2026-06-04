from __future__ import annotations

import json
from pathlib import Path

import yaml

from src.analysis.prefix_state_transition_tomography.finalize import finalize_if_ready
from src.analysis.prefix_state_transition_tomography.jsonl import write_jsonl
from src.analysis.prefix_state_transition_tomography.status import build_status_report


def test_status_infers_expected_shards_and_shard_completion(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    write_jsonl(
        root / "prefix_state_sampled_rows.jsonl",
        [
            {"prefix_state_id": "a", "shard_id": 0},
            {"prefix_state_id": "b", "shard_id": 1},
        ],
    )
    for shard_id in (0, 1):
        shard_root = root / "shards" / f"shard_{shard_id:02d}"
        write_jsonl(shard_root / "boundary_score_rows.jsonl", [{"row": shard_id}])
        write_jsonl(shard_root / "forced_x1_rows.jsonl", [{"row": shard_id}])
        (shard_root / "shard_manifest.json").write_text("{}", encoding="utf-8")

    status = build_status_report(artifact_root=root)

    assert status["expected_shards"] == 2
    assert status["planned_prefix_state_rows"] == 2
    assert status["stage_status"] == "shards_complete_pending_merge"
    assert status["shards_complete"] is True


def test_finalize_not_ready_until_shards_complete(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"project_id": "prefix_state_transition_tomography"}), encoding="utf-8")

    result = finalize_if_ready(config_path=config, artifact_root=root)

    assert result["action"] == "not_ready"
    assert result["status"]["stage_status"] == "missing_artifact_root"


def test_status_reports_final_artifacts_present(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    write_jsonl(root / "prefix_state_sampled_rows.jsonl", [{"prefix_state_id": "a", "shard_id": 0}])
    write_jsonl(root / "boundary_score_rows.jsonl", [{"row": 1}])
    write_jsonl(root / "forced_x1_rows.jsonl", [{"row": 1}])
    write_jsonl(root / "paired_state_rows.jsonl", [{"row": 1}])
    write_jsonl(root / "quadrant_rows.jsonl", [{"row": 1}])
    (root / "merge_summary.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    (root / "summary.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    (root / "manifest.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    (root / "report.md").write_text("# ok", encoding="utf-8")
    (root / "gallery").mkdir()
    write_jsonl(root / "gallery" / "gallery_rows.jsonl", [{"row": 1}])
    (root / "gallery" / "index.md").write_text("# gallery", encoding="utf-8")
    (root / "gallery" / "gallery_summary.json").write_text(json.dumps({"ok": True}), encoding="utf-8")

    status = build_status_report(artifact_root=root)

    assert status["stage_status"] == "final_artifacts_present"
    assert status["final_ready"] is True


def test_status_reports_index_ready_pending_gpu(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    (root / "resolved_config.yaml").parent.mkdir(parents=True, exist_ok=True)
    (root / "resolved_config.yaml").write_text("{}", encoding="utf-8")
    write_jsonl(root / "prefix_state_index.jsonl", [{"row": 1}])
    write_jsonl(root / "prefix_state_sampled_rows.jsonl", [{"prefix_state_id": "a", "shard_id": 0}])
    (root / "prefix_state_index_summary.json").write_text(
        json.dumps({"launch_eligible": True, "failed_launch_gates": []}),
        encoding="utf-8",
    )

    status = build_status_report(artifact_root=root)

    assert status["stage_status"] == "index_ready_pending_gpu"
    assert status["launch_eligible"] is True
    assert status["failed_launch_gates"] == []
