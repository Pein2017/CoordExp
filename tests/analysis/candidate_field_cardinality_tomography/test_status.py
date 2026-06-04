from __future__ import annotations

import os

from src.analysis.candidate_field_cardinality_tomography.artifacts import write_json, write_jsonl
from src.analysis.candidate_field_cardinality_tomography.status import build_status_report


def test_status_report_identifies_running_shards_and_missing_final_artifacts(tmp_path) -> None:
    root = tmp_path / "artifact"
    log_root = tmp_path / "logs"
    shard_root = root / "shards" / "shard_000"
    log_root.mkdir(parents=True)
    shard_root.mkdir(parents=True)
    write_json(root / "probe_plan_summary.json", {"num_shards": 1, "gpu_probe_planned_cases": 3})
    write_jsonl(shard_root / "x1_candidate_field_rows.jsonl.inprogress", [{"row": 1}, {"row": 2}])
    (log_root / "shard_pids.tsv").write_text(
        f"0\t0\t{os.getpid()}\t{log_root / 'candidate_field_x1_shard_000.log'}\n",
        encoding="utf-8",
    )
    (log_root / "candidate_field_x1_shard_000.log").write_text("loading\n", encoding="utf-8")

    status = build_status_report(artifact_root=root, log_root=log_root)

    assert status["stage_status"] == "x1_shards_running"
    assert status["planned_cases"] == 3
    assert status["expected_shards"] == 1
    assert status["alive_shard_processes"] == 1
    assert status["final_ready"] is False
    assert status["final_files"]["manifest.json"]["exists"] is False
    assert status["shards"][0]["files"]["x1_candidate_field_rows.jsonl.inprogress"]["rows"] == 2
    assert status["processes"][0]["alive"] is True


def test_status_report_identifies_final_artifacts(tmp_path) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    for filename in (
        "x1_candidate_field_rows.jsonl",
        "phase_a_case_taxonomy_rows.jsonl",
    ):
        write_jsonl(root / filename, [{"ok": True}])
    for filename in (
        "merge_summary.json",
        "taxonomy_summary.json",
        "summary.json",
        "manifest.json",
    ):
        write_json(root / filename, {"ok": True})
    (root / "report.md").write_text("# report\n", encoding="utf-8")

    status = build_status_report(artifact_root=root)

    assert status["stage_status"] == "final_artifacts_present"
    assert status["final_ready"] is True
