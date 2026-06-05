from __future__ import annotations

import json
import re
import hashlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import SCHEMA_VERSION
from .jsonl import read_jsonl, write_jsonl


_UNCAVEATED_RANKING_RE = re.compile(r"\b(best|wins?|better|AP50|AP75|mAP|F1-score|AR100)\b", re.I)


def build_summary(
    *,
    slot_rows: Sequence[Mapping[str, Any]],
    trajectory_rows: Sequence[Mapping[str, Any]],
    prefix_sensitivity_rows: Sequence[Mapping[str, Any]],
    greedy_rows: Sequence[Mapping[str, Any]],
    config_path: str | None = None,
    config_sha256: str | None = None,
    checkpoint_fingerprints: Mapping[str, Any] | None = None,
    template_contracts: Mapping[str, Any] | None = None,
    shard_merge: Mapping[str, Any] | None = None,
    status_gate_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    boundary_counts = Counter(
        str(row.get("checkpoint_role"))
        for row in slot_rows
        if bool(row.get("boundary_extreme_flag"))
    )
    winner_counts = Counter(str(row.get("winner_bucket")) for row in slot_rows)
    return {
        "artifact_schema_version": SCHEMA_VERSION,
        "comparison_semantics": "mechanism_traits_not_detector_accuracy",
        "reference_anchor_caveat": "template_objective_confounded_reference",
        "row_counts": {
            "slot_posterior_rows": len(slot_rows),
            "trajectory_rows": len(trajectory_rows),
            "prefix_sensitivity_rows": len(prefix_sensitivity_rows),
            "greedy_continuation_rows": len(greedy_rows),
        },
        "winner_bucket_counts": dict(winner_counts),
        "boundary_extreme_counts_by_checkpoint": dict(boundary_counts),
        "config_path": config_path,
        "config_sha256": config_sha256,
        "checkpoint_fingerprints": dict(checkpoint_fingerprints or {}),
        "template_contracts": dict(template_contracts or {}),
        "shard_merge": dict(shard_merge or {}),
        "status_gate_result": dict(status_gate_result or {}),
        "interpretation_boundaries": [
            "ET-RMP-CE is reference_anchor, not a clean controlled baseline.",
            "ET-RMP-CE carries template_objective_confounded_reference caveat.",
            "IoU50 is secondary to slot-level and trajectory-level taxonomy.",
        ],
    }


def build_report_markdown(summary: Mapping[str, Any]) -> str:
    text = [
        "# A3.3 Post-X1 Instance-Basin Tomography",
        "",
        "Evidence scope: mechanism traits, not by final detector accuracy.",
        "",
        "The ET-RMP checkpoint is a reference_anchor with template_objective_confounded_reference.",
        "",
        "## Row Counts",
        "",
        "```json",
        json.dumps(summary.get("row_counts", {}), indent=2, sort_keys=True),
        "```",
        "",
        "## Winner Buckets",
        "",
        "```json",
        json.dumps(summary.get("winner_bucket_counts", {}), indent=2, sort_keys=True),
        "```",
        "",
    ]
    report = "\n".join(text)
    validate_report_language(report)
    return report


def validate_report_language(text: str) -> None:
    if _UNCAVEATED_RANKING_RE.search(text):
        raise ValueError("uncaveated detector ranking language is not allowed")
    if "not by final detector accuracy" not in text.lower():
        raise ValueError("report must include scope caveat: not by final detector accuracy")


def write_report(root: str | Path, summary: Mapping[str, Any]) -> dict[str, str]:
    artifact_root = Path(root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    report = build_report_markdown(summary)
    summary_path = artifact_root / "summary.json"
    report_path = artifact_root / "report.md"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(report, encoding="utf-8")
    return {"summary_path": str(summary_path), "report_path": str(report_path)}


def merge_slot_posterior_shards(root: str | Path) -> dict[str, Any]:
    artifact_root = Path(root)
    shard_root = artifact_root / "slot_posterior_shards"
    merged_rows: list[dict[str, Any]] = []
    input_shards: list[dict[str, Any]] = []
    for shard_id in range(8):
        path = shard_root / f"shard_{shard_id}.jsonl"
        rows = read_jsonl(path) if path.exists() else []
        merged_rows.extend(rows)
        input_shards.append(
            {
                "path": f"slot_posterior_shards/shard_{shard_id}.jsonl",
                "sha256": _sha256_file(path) if path.exists() else "0" * 64,
                "row_count": len(rows),
            }
        )
    write_jsonl(artifact_root / "slot_posterior_rows.jsonl", merged_rows)
    manifest = {
        "artifact_schema_version": SCHEMA_VERSION,
        "input_shards": input_shards,
        "merged_row_count": len(merged_rows),
        "merge_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (artifact_root / "merge_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "build_report_markdown",
    "build_summary",
    "merge_slot_posterior_shards",
    "validate_report_language",
    "write_report",
]
