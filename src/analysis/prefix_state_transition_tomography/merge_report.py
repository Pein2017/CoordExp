from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .jsonl import read_jsonl, write_jsonl


PAIRED_KEY_FIELDS = (
    "image_id",
    "source_line_idx",
    "prefix_state_id",
    "prefix_condition",
    "prefix_depth",
    "prefix_order_policy_id",
    "probe_desc",
)


def paired_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in PAIRED_KEY_FIELDS)


def assign_quadrant(*, boundary_good: bool, x1_good: bool) -> str:
    if boundary_good and x1_good:
        return "boundary_good_x1_good"
    if boundary_good and not x1_good:
        return "boundary_good_x1_bad"
    if not boundary_good and x1_good:
        return "boundary_bad_x1_good"
    return "boundary_bad_x1_bad"


def build_quadrant_rows(
    boundary_rows: Sequence[Mapping[str, Any]],
    forced_x1_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    boundary_by_role = _group_one_by_role(boundary_rows)
    x1_by_role = _group_one_by_role(forced_x1_rows)
    out: list[dict[str, Any]] = []
    for key in sorted(set(boundary_by_role) & set(x1_by_role)):
        boundary_roles = boundary_by_role[key]
        x1_roles = x1_by_role[key]
        if set(boundary_roles) != {"et_rmp_ce", "pure_ce"}:
            continue
        if set(x1_roles) != {"et_rmp_ce", "pure_ce"}:
            continue
        for role in ("et_rmp_ce", "pure_ce"):
            boundary = boundary_roles[role]
            x1 = x1_roles[role]
            boundary_good = boundary.get("boundary_alignment") == "residual_favored"
            x1_good = float(x1.get("forced_x1_residual_coverage", 0.0)) > 0.0
            out.append(
                {
                    "paired_key": "|".join(str(part) for part in key),
                    "checkpoint_role": role,
                    "split": boundary.get("split"),
                    "transition_type": boundary.get("transition_type"),
                    "prefix_depth": boundary.get("prefix_depth"),
                    "prefix_state_id": boundary.get("prefix_state_id"),
                    "probe_desc": boundary.get("probe_desc"),
                    "boundary_alignment": boundary.get("boundary_alignment"),
                    "forced_x1_residual_coverage": float(x1.get("forced_x1_residual_coverage", 0.0)),
                    "emitted_attraction_rate": float(x1.get("emitted_attraction_rate", 0.0)),
                    "quadrant": assign_quadrant(boundary_good=boundary_good, x1_good=x1_good),
                }
            )
    return out


def summarize_quadrants(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "row_count": len(rows),
        "quadrant_counts": dict(Counter(str(row.get("quadrant")) for row in rows)),
        "by_split": _count_by(rows, "split", "quadrant"),
        "by_transition_type": _count_by(rows, "transition_type", "quadrant"),
        "by_checkpoint_role": _count_by(rows, "checkpoint_role", "quadrant"),
        "paired_delta_metrics": _paired_delta_metrics(rows),
    }


def merge_artifacts(*, artifact_root: Path) -> dict[str, Any]:
    root = Path(artifact_root)
    boundary = _read_shard_rows(root, "boundary_score_rows.jsonl")
    forced = _read_shard_rows(root, "forced_x1_rows.jsonl")
    write_jsonl(root / "boundary_score_rows.jsonl", boundary)
    write_jsonl(root / "forced_x1_rows.jsonl", forced)
    quadrants = build_quadrant_rows(boundary, forced)
    write_jsonl(root / "paired_state_rows.jsonl", quadrants)
    write_jsonl(root / "quadrant_rows.jsonl", quadrants)
    summary = summarize_quadrants(quadrants)
    summary = {
        **summary,
        "row_counts": {
            "boundary_score_rows": len(boundary),
            "forced_x1_rows": len(forced),
            "paired_state_rows": len(quadrants),
            "quadrant_rows": len(quadrants),
        },
    }
    (root / "merge_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "status": "ok",
        "artifact_root": str(root),
        "artifacts": {
            "boundary_score_rows": "boundary_score_rows.jsonl",
            "forced_x1_rows": "forced_x1_rows.jsonl",
            "paired_state_rows": "paired_state_rows.jsonl",
            "quadrant_rows": "quadrant_rows.jsonl",
            "merge_summary": "merge_summary.json",
            "summary": "summary.json",
        },
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def write_report(*, artifact_root: Path) -> Path:
    root = Path(artifact_root)
    summary_path = root / "merge_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    report = [
        "# Prefix-State Transition Tomography Phase A3.1",
        "",
        "## Scope",
        "",
        "Evidence scope: paired prefix-state transition diagnostics.",
        "",
        "## Four-Quadrant Mechanism Table",
        "",
        "```json",
        json.dumps(summary.get("quadrant_counts", {}), indent=2, sort_keys=True),
        "```",
        "",
        "## Evidence Boundaries",
        "",
        "- This report summarizes boundary/full-desc and forced-x1 rows.",
        "- It does not include full attention dumps.",
        "- Category-sorted training remains a Phase B hypothesis, not a conclusion.",
        "",
    ]
    path = root / "report.md"
    path.write_text("\n".join(report), encoding="utf-8")
    return path


def _group_one_by_role(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[Any, ...], dict[str, Mapping[str, Any]]]:
    grouped: dict[tuple[Any, ...], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        role = str(row.get("checkpoint_role"))
        grouped[paired_key(row)][role] = row
    return grouped


def _count_by(rows: Sequence[Mapping[str, Any]], *fields: str) -> dict[str, int]:
    counter = Counter("|".join(str(row.get(field)) for field in fields) for row in rows)
    return dict(counter)


def _paired_delta_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[str(row.get("paired_key"))][str(row.get("checkpoint_role"))] = row
    deltas = []
    disagreements = 0
    for pair in grouped.values():
        if set(pair) != {"et_rmp_ce", "pure_ce"}:
            continue
        et = pair["et_rmp_ce"]
        pure = pair["pure_ce"]
        deltas.append(
            float(pure.get("forced_x1_residual_coverage", 0.0))
            - float(et.get("forced_x1_residual_coverage", 0.0))
        )
        if pure.get("quadrant") != et.get("quadrant"):
            disagreements += 1
    return {
        "paired_state_count": len(deltas),
        "mean_pure_minus_et_forced_x1_residual_coverage": (
            0.0 if not deltas else sum(deltas) / len(deltas)
        ),
        "quadrant_disagreement_count": disagreements,
    }


def _read_shard_rows(root: Path, filename: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    shard_root = root / "shards"
    for path in sorted(shard_root.glob(f"shard_*/{filename}")):
        rows.extend(read_jsonl(path))
    return rows


__all__ = [
    "PAIRED_KEY_FIELDS",
    "assign_quadrant",
    "build_quadrant_rows",
    "merge_artifacts",
    "paired_key",
    "summarize_quadrants",
    "write_report",
]
