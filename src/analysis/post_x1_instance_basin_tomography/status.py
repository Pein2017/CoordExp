from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from . import CHECKPOINT_ROLES, SCHEMA_VERSION
from .jsonl import read_jsonl
from .merge_report import validate_report_language


FINAL_ARTIFACTS = ("summary.json", "report.md", "gallery/gallery_summary.json")


def evaluate_status(root: str | Path) -> dict[str, Any]:
    artifact_root = Path(root)
    failed: list[str] = []
    if not (artifact_root / "template_contracts.json").is_file():
        failed.append("template_contracts_present")
    contracts = _read_json(artifact_root / "template_contracts.json").get("template_contracts", {})
    slot_rows = _read_rows_if_exists(artifact_root / "slot_posterior_rows.jsonl")
    if not slot_rows:
        failed.append("slot_posterior_rows_present")
    shard_rows = _read_shard_rows(artifact_root)
    shard_summaries = _read_rows_if_exists(artifact_root / "slot_posterior_shard_summaries.jsonl")
    if shard_summaries:
        expected = sum(int(row.get("row_count", 0)) for row in shard_summaries)
        if expected != len(shard_rows):
            failed.append("slot_shard_row_counts_match")
    elif shard_rows:
        failed.append("slot_shard_summaries_present")
    if slot_rows and shard_rows and len(slot_rows) != len(shard_rows):
        failed.append("merged_slot_rows_match_shards")
    if slot_rows and contracts and not _contracts_match(slot_rows, contracts):
        failed.append("checkpoint_role_template_contracts_match")
    for role in CHECKPOINT_ROLES:
        if slot_rows and not any(row.get("checkpoint_role") == role for row in slot_rows):
            failed.append(f"checkpoint_role_nonempty:{role}")
    report_path = artifact_root / "report.md"
    if report_path.exists():
        try:
            validate_report_language(report_path.read_text(encoding="utf-8"))
        except ValueError:
            failed.append("report_language_no_uncaveated_detector_ranking")
    else:
        failed.append("report_present")
    final_present = all((artifact_root / name).exists() for name in FINAL_ARTIFACTS)
    if not final_present:
        failed.append("final_artifacts_present")
    row_counts = {
        "case_universe_rows": _count_jsonl(artifact_root / "case_universe.jsonl"),
        "prefix_state_rows": _count_jsonl(artifact_root / "prefix_states.jsonl"),
        "slot_posterior_rows": len(slot_rows),
        "slot_posterior_shard_rows": len(shard_rows),
    }
    status = "final_artifacts_present" if not failed and final_present else "incomplete"
    return {
        "status": status,
        "failed_gates": sorted(set(failed)),
        "final_artifacts_present": final_present,
        "row_counts": row_counts,
        "checkpoint_role_counts": dict(Counter(str(row.get("checkpoint_role")) for row in slot_rows)),
        "schema_versions": dict(Counter(str(row.get("artifact_schema_version")) for row in slot_rows)),
    }


def build_status_report(*, artifact_root: Path) -> dict[str, Any]:
    root = Path(artifact_root)
    rows = _read_shard_rows(root)
    stage_status = "posterior_shards_present_pending_merge" if rows else "artifact_root_incomplete"
    if all((root / name).exists() for name in FINAL_ARTIFACTS):
        stage_status = "final_artifacts_present"
    return {
        "artifact_root": str(root),
        "stage_status": stage_status,
        "row_counts": {
            "case_universe_rows": _count_jsonl(root / "case_universe.jsonl"),
            "prefix_state_rows": _count_jsonl(root / "prefix_states.jsonl"),
            "slot_posterior_rows": len(rows),
        },
        "checkpoint_role_counts": dict(Counter(str(row.get("checkpoint_role")) for row in rows)),
        "schema_versions": dict(Counter(str(row.get("artifact_schema_version")) for row in rows)),
    }


def _contracts_match(rows: list[Mapping[str, Any]], contracts: Mapping[str, Any]) -> bool:
    for row in rows:
        role = str(row.get("checkpoint_role"))
        expected = contracts.get(role)
        if expected and row.get("template_contract") != expected:
            return False
    return True


def _read_shard_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    shard_root = root / "slot_posterior_shards"
    if shard_root.exists():
        for path in sorted(shard_root.glob("shard_*.jsonl")):
            rows.extend(read_jsonl(path))
    return rows


def _read_rows_if_exists(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.exists() else []


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


__all__ = ["FINAL_ARTIFACTS", "build_status_report", "evaluate_status"]
