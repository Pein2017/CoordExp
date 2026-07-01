from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .jsonl import read_jsonl


FINAL_ARTIFACTS = (
    "boundary_score_rows.jsonl",
    "forced_x1_rows.jsonl",
    "paired_state_rows.jsonl",
    "quadrant_rows.jsonl",
    "merge_summary.json",
    "summary.json",
    "manifest.json",
    "report.md",
    "gallery/gallery_rows.jsonl",
    "gallery/index.md",
    "gallery/gallery_summary.json",
)

INDEX_ARTIFACTS = (
    "resolved_config.yaml",
    "prefix_state_index.jsonl",
    "prefix_state_index_summary.json",
    "prefix_state_sampled_rows.jsonl",
)

SHARD_ARTIFACTS = (
    "boundary_score_rows.jsonl",
    "boundary_score_rows.jsonl.inprogress",
    "forced_x1_rows.jsonl",
    "forced_x1_rows.jsonl.inprogress",
    "paired_probe_progress.json",
    "shard_summary.json",
    "shard_manifest.json",
)


def build_status_report(*, artifact_root: Path, log_root: Path | None = None) -> dict[str, Any]:
    root = Path(artifact_root)
    index_files = {name: _file_status(root / name) for name in INDEX_ARTIFACTS}
    index_summary = _read_json(root / "prefix_state_index_summary.json")
    launch_eligible = index_summary.get("launch_eligible")
    failed_launch_gates = index_summary.get("failed_launch_gates", [])
    expected_shards = _expected_shards(root)
    shard_reports = [_build_shard_report(root, shard_id) for shard_id in range(expected_shards)]
    process_reports = _read_shard_process_reports(log_root / "shard_pids.tsv") if log_root else []
    final_files = {name: _file_status(root / name) for name in FINAL_ARTIFACTS}
    final_ready = all(item["exists"] for item in final_files.values())
    index_complete = all(item["exists"] for item in index_files.values())
    shards_complete = bool(shard_reports) and all(
        shard["files"]["boundary_score_rows.jsonl"]["exists"]
        and shard["files"]["forced_x1_rows.jsonl"]["exists"]
        and shard["files"]["shard_manifest.json"]["exists"]
        for shard in shard_reports
    )
    alive_count = sum(1 for process in process_reports if process["alive"])
    planned_rows = _planned_rows(root)
    return {
        "artifact_root": str(root),
        "log_root": None if log_root is None else str(log_root),
        "stage_status": _stage_status(
            final_ready=final_ready,
            index_complete=index_complete,
            launch_eligible=launch_eligible,
            shards_complete=shards_complete,
            alive_count=alive_count,
            root_exists=root.exists(),
        ),
        "launch_eligible": launch_eligible,
        "failed_launch_gates": failed_launch_gates,
        "expected_shards": expected_shards,
        "planned_prefix_state_rows": planned_rows,
        "sampled_prefix_state_rows": planned_rows,
        "alive_shard_processes": alive_count,
        "index_files": index_files,
        "final_ready": final_ready,
        "final_files": final_files,
        "shards_complete": shards_complete,
        "shards": shard_reports,
        "processes": process_reports,
    }


def _stage_status(
    *,
    final_ready: bool,
    index_complete: bool,
    launch_eligible: object,
    shards_complete: bool,
    alive_count: int,
    root_exists: bool,
) -> str:
    if final_ready:
        return "final_artifacts_present"
    if alive_count:
        return "paired_probe_shards_running"
    if shards_complete:
        return "shards_complete_pending_merge"
    if index_complete and launch_eligible is False:
        return "index_launch_blocked"
    if index_complete and launch_eligible is True:
        return "index_ready_pending_gpu"
    if root_exists:
        return "index_missing_or_incomplete"
    return "missing_artifact_root"


def _expected_shards(root: Path) -> int:
    sample_path = root / "prefix_state_sampled_rows.jsonl"
    if sample_path.exists():
        shard_ids = []
        for row in read_jsonl(sample_path):
            value = row.get("shard_id", row.get("planned_shard_id"))
            if value is not None:
                shard_ids.append(int(value))
        if shard_ids:
            return max(shard_ids) + 1
    shard_dirs = sorted((root / "shards").glob("shard_*")) if (root / "shards").exists() else []
    if shard_dirs:
        return len(shard_dirs)
    return 0


def _planned_rows(root: Path) -> int:
    sample_path = root / "prefix_state_sampled_rows.jsonl"
    if not sample_path.exists():
        return 0
    return len(read_jsonl(sample_path))


def _build_shard_report(root: Path, shard_id: int) -> dict[str, Any]:
    shard_root = root / "shards" / f"shard_{shard_id:02d}"
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
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


__all__ = ["FINAL_ARTIFACTS", "INDEX_ARTIFACTS", "SHARD_ARTIFACTS", "build_status_report"]
