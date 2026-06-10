from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

FINAL_ARTIFACTS = (
    "x1_candidate_field_rows.jsonl",
    "phase_a_case_taxonomy_rows.jsonl",
    "merge_summary.json",
    "taxonomy_summary.json",
    "summary.json",
    "report.md",
    "manifest.json",
)

SHARD_ARTIFACTS = (
    "x1_candidate_field_rows.jsonl",
    "x1_candidate_field_rows.jsonl.inprogress",
    "x1_candidate_field_progress.json",
    "x1_candidate_field_summary.json",
    "shard_manifest.json",
)


def build_status_report(*, artifact_root: Path, log_root: Path | None = None) -> dict[str, Any]:
    root = Path(artifact_root)
    plan_summary = _read_json(root / "probe_plan_summary.json")
    expected_shards = int(plan_summary.get("num_shards") or 0)
    planned_cases = int(plan_summary.get("gpu_probe_planned_cases") or 0)
    shard_reports = [_build_shard_report(root, shard_id) for shard_id in range(expected_shards)]
    process_reports = _read_shard_process_reports(log_root / "shard_pids.tsv") if log_root else []
    final_files = {name: _file_status(root / name) for name in FINAL_ARTIFACTS}
    final_ready = all(item["exists"] for item in final_files.values())
    shards_complete = bool(shard_reports) and all(
        shard["files"]["x1_candidate_field_rows.jsonl"]["exists"]
        and shard["files"]["shard_manifest.json"]["exists"]
        for shard in shard_reports
    )
    alive_count = sum(1 for process in process_reports if process["alive"])
    return {
        "artifact_root": str(root),
        "log_root": None if log_root is None else str(log_root),
        "stage_status": _stage_status(
            final_ready=final_ready,
            shards_complete=shards_complete,
            alive_count=alive_count,
            root_exists=root.exists(),
        ),
        "planned_cases": planned_cases,
        "expected_shards": expected_shards,
        "alive_shard_processes": alive_count,
        "final_ready": final_ready,
        "final_files": final_files,
        "shards_complete": shards_complete,
        "shards": shard_reports,
        "processes": process_reports,
    }


def _stage_status(*, final_ready: bool, shards_complete: bool, alive_count: int, root_exists: bool) -> str:
    if final_ready:
        return "final_artifacts_present"
    if alive_count:
        return "x1_shards_running"
    if shards_complete:
        return "shards_complete_pending_merge"
    if root_exists:
        return "incomplete_or_waiting"
    return "missing_artifact_root"


def _build_shard_report(root: Path, shard_id: int) -> dict[str, Any]:
    shard_root = root / "shards" / f"shard_{shard_id:03d}"
    files = {name: _file_status(shard_root / name) for name in SHARD_ARTIFACTS}
    return {
        "shard_id": shard_id,
        "shard_root": str(shard_root),
        "files": files,
    }


def _read_shard_process_reports(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        shard_id, gpu_id, pid_text, log_file = parts[:4]
        pid = int(pid_text)
        rows.append(
            {
                "shard_id": int(shard_id),
                "gpu_id": str(gpu_id),
                "pid": pid,
                "alive": _pid_alive(pid),
                "log_file": log_file,
                "log": _file_status(Path(log_file)),
            }
        )
    return rows


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _file_status(path: Path) -> dict[str, Any]:
    exists = path.exists()
    status: dict[str, Any] = {
        "path": str(path),
        "exists": exists,
        "bytes": path.stat().st_size if exists else 0,
    }
    if exists and ".jsonl" in path.name:
        status["rows"] = _count_lines(path)
    return status


def _count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
