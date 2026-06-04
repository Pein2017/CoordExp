from __future__ import annotations

import os

from src.analysis.candidate_field_cardinality_tomography.artifacts import write_json, write_jsonl
from src.analysis.candidate_field_cardinality_tomography.finalize import finalize_if_ready


def test_finalize_if_ready_noops_while_shards_running(tmp_path) -> None:
    root = tmp_path / "artifact"
    log_root = tmp_path / "logs"
    shard_root = root / "shards" / "shard_000"
    shard_root.mkdir(parents=True)
    log_root.mkdir()
    config = tmp_path / "config.yaml"
    config.write_text("project_id: candidate_field_cardinality_tomography\n", encoding="utf-8")
    write_json(root / "probe_plan_summary.json", {"num_shards": 1, "gpu_probe_planned_cases": 1})
    write_jsonl(shard_root / "x1_candidate_field_rows.jsonl.inprogress", [{"row": 1}])
    (log_root / "shard_pids.tsv").write_text(f"0\t0\t{os.getpid()}\t{log_root / 'shard.log'}\n", encoding="utf-8")
    (log_root / "shard.log").write_text("running\n", encoding="utf-8")

    result = finalize_if_ready(config_path=config, artifact_root=root, log_root=log_root)

    assert result["action"] == "not_ready"
    assert result["status"]["stage_status"] == "x1_shards_running"
    assert result["result"] is None


def test_finalize_if_ready_detects_existing_final_artifacts(tmp_path) -> None:
    root = tmp_path / "artifact"
    config = tmp_path / "config.yaml"
    config.write_text("project_id: candidate_field_cardinality_tomography\n", encoding="utf-8")
    write_jsonl(root / "x1_candidate_field_rows.jsonl", [{"ok": True}])
    write_jsonl(root / "phase_a_case_taxonomy_rows.jsonl", [{"ok": True}])
    for filename in ("merge_summary.json", "taxonomy_summary.json", "summary.json", "manifest.json"):
        write_json(root / filename, {"ok": True})
    (root / "report.md").write_text("# report\n", encoding="utf-8")

    result = finalize_if_ready(config_path=config, artifact_root=root)

    assert result["action"] == "already_final"
    assert result["status"]["stage_status"] == "final_artifacts_present"
    assert result["result"] is None
