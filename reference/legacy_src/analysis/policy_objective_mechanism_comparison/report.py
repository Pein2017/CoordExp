from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from . import PROJECT_ID, RUN_ID, SCHEMA_VERSION
from .config import ComparisonConfig


def build_summary(config: ComparisonConfig) -> dict[str, Any]:
    role_rows = [
        {
            "checkpoint_role": role,
            "objective_policy": role_config.objective_policy,
            "training_ordering": role_config.training_ordering,
            "template_contract_id": role_config.template_contract_id,
            "comparison_group": role_config.comparison_group,
        }
        for role, role_config in config.checkpoint_roles.items()
    ]
    return {
        "project_id": PROJECT_ID,
        "schema_version": SCHEMA_VERSION,
        "run_id": RUN_ID,
        "artifact_root": str(config.artifact_root),
        "checkpoint_role_count": len(config.checkpoint_roles),
        "checkpoint_roles": list(config.checkpoint_roles),
        "role_rows": role_rows,
        "role_counts_by_objective": dict(Counter(row["objective_policy"] for row in role_rows)),
        "role_counts_by_ordering": dict(Counter(row["training_ordering"] for row in role_rows)),
        "clean_2x2_roles": [
            row["checkpoint_role"]
            for row in role_rows
            if row["comparison_group"] == "fullobj_2x2_20260601"
        ],
        "legacy_reference_anchor_roles": [
            row["checkpoint_role"]
            for row in role_rows
            if row["comparison_group"] == "legacy_reference_anchor"
        ],
        "artifact_status": _artifact_status(config),
    }


def write_outputs(config: ComparisonConfig) -> dict[str, Any]:
    summary = build_summary(config)
    config.artifact_root.mkdir(parents=True, exist_ok=True)
    plots_dir = config.artifact_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    _write_json(config.artifact_root / "summary.json", summary)
    (config.artifact_root / "comparison_report.md").write_text(
        render_report(summary),
        encoding="utf-8",
    )
    (plots_dir / "role_matrix.tsv").write_text(
        _role_matrix_tsv(summary),
        encoding="utf-8",
    )
    return {
        "status": "ok",
        "summary_json": str(config.artifact_root / "summary.json"),
        "comparison_report_md": str(config.artifact_root / "comparison_report.md"),
        "role_matrix_tsv": str(plots_dir / "role_matrix.tsv"),
    }


def render_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# FullObj Policy/Objectivity Mechanism Comparison",
        "",
        f"- project_id: `{summary['project_id']}`",
        f"- run_id: `{summary['run_id']}`",
        f"- checkpoint_role_count: `{summary['checkpoint_role_count']}`",
        "",
        "## Clean 2x2 Cohort",
    ]
    for row in summary["role_rows"]:
        if row["comparison_group"] != "fullobj_2x2_20260601":
            continue
        lines.append(
            "- `{checkpoint_role}`: objective=`{objective_policy}`, "
            "ordering=`{training_ordering}`, template=`{template_contract_id}`".format(
                **row
            )
        )
    lines.extend(["", "## Legacy Reference Anchor"])
    for row in summary["role_rows"]:
        if row["comparison_group"] != "legacy_reference_anchor":
            continue
        lines.append(
            "- `{checkpoint_role}`: objective=`{objective_policy}`, "
            "ordering=`{training_ordering}`, template=`{template_contract_id}`".format(
                **row
            )
        )
    lines.extend(["", "## Artifact Inputs"])
    for name, status in summary["artifact_status"].items():
        lines.append(
            "- `{}`: exists=`{}`, scope=`{}`, root=`{}`".format(
                name,
                status["exists"],
                status["evidence_scope"],
                status["artifact_root"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            "- The four `fullobj_2x2_20260601` roles form the controlled 2x2 cohort.",
            "- The `legacy_reference_anchor` role is reported separately because its checkpoint step and template contract differ.",
            "- This report is a consolidation surface; it does not itself prove model behavior without populated child artifacts.",
            "",
        ]
    )
    return "\n".join(lines)


def _artifact_status(config: ComparisonConfig) -> dict[str, dict[str, Any]]:
    status: dict[str, dict[str, Any]] = {}
    for name, artifact in config.artifact_sources.items():
        root = artifact.root
        status[name] = {
            "artifact_root": str(root),
            "evidence_scope": artifact.evidence_scope,
            "required": artifact.required,
            "exists": root.exists(),
            "summary_files": _existing_summary_files(root),
            "row_counts": _row_counts(root),
        }
    return status


def _existing_summary_files(root: Path) -> list[str]:
    candidates = (
        "summary.json",
        "summary/prefix_readout_summary.json",
        "prefix_state_index_summary.json",
        "slot_posterior_summary.json",
        "report.md",
        "summary/report.md",
    )
    return [rel for rel in candidates if (root / rel).is_file()]


def _row_counts(root: Path) -> dict[str, int]:
    rel_paths = (
        "prefix_state_sampled_rows.jsonl",
        "summary/prefix_readout_merged_rows.jsonl",
        "slot_posterior_rows.jsonl",
        "trajectory_rows.jsonl",
        "basin_attraction_matrix.jsonl",
        "prefix_sensitivity_rows.jsonl",
        "greedy_continuation_rows.jsonl",
        "fn_probe/fn_probe_rows.jsonl",
    )
    return {
        rel: _count_jsonl_rows(root / rel)
        for rel in rel_paths
        if (root / rel).is_file()
    }


def _count_jsonl_rows(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _role_matrix_tsv(summary: Mapping[str, Any]) -> str:
    rows_by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in summary["role_rows"]:
        rows_by_group[str(row["comparison_group"])].append(row)
    lines = [
        "comparison_group\tcheckpoint_role\tobjective_policy\ttraining_ordering\ttemplate_contract_id"
    ]
    for group in sorted(rows_by_group):
        for row in rows_by_group[group]:
            lines.append(
                "{comparison_group}\t{checkpoint_role}\t{objective_policy}\t{training_ordering}\t{template_contract_id}".format(
                    **row
                )
            )
    return "\n".join(lines) + "\n"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
