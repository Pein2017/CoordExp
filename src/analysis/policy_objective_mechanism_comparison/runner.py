from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import PROJECT_ID, RUN_ID, SCHEMA_VERSION
from .config import ArtifactSourceConfig, ComparisonConfig, load_config


ROLE_JSONL_FILES = (
    "summary/prefix_readout_merged_rows.jsonl",
    "boundary_score_rows.jsonl",
    "forced_x1_rows.jsonl",
    "slot_posterior_rows.jsonl",
    "trajectory_rows.jsonl",
    "basin_attraction_matrix.jsonl",
    "prefix_sensitivity_rows.jsonl",
    "greedy_continuation_rows.jsonl",
    "rollout/rollout_phenotype_rows.jsonl",
    "fn_probe/fn_probe_rows.jsonl",
    "fn_probe/fn_candidate_scores.jsonl",
    "fn_probe/fn_slot_evidence.jsonl",
)

SUMMARY_JSON_FILES = (
    "sample_manifest.json",
    "prefix_state_index_summary.json",
    "case_universe_summary.json",
    "prefix_mode_summary.json",
    "slot_posterior_summary.json",
    "summary/prefix_readout_summary.json",
    "rollout/rollout_summary.json",
    "fn_probe/fn_slot_rescue_summary.json",
)


def run_from_config(config_path: str | Path, *, allow_overwrite: bool = False) -> dict[str, Any]:
    config = load_config(config_path)
    return run(config, config_path=Path(config_path), allow_overwrite=allow_overwrite)


def run(
    config: ComparisonConfig,
    *,
    config_path: Path | None = None,
    allow_overwrite: bool = False,
) -> dict[str, Any]:
    root = config.artifact_root
    if root.exists() and not allow_overwrite and any(root.iterdir()):
        raise FileExistsError(f"artifact root already exists: {root}")
    root.mkdir(parents=True, exist_ok=True)

    source_summaries = {
        source_id: _summarize_source(source, known_roles=tuple(config.checkpoint_roles))
        for source_id, source in config.artifact_sources.items()
    }
    summary = {
        "project_id": PROJECT_ID,
        "run_id": RUN_ID,
        "schema_version": SCHEMA_VERSION,
        "config_path": None if config_path is None else str(config_path),
        "artifact_root": str(root),
        "checkpoint_roles": {
            role: asdict(meta) for role, meta in config.checkpoint_roles.items()
        },
        "clean_2x2_roles": [
            role
            for role, meta in config.checkpoint_roles.items()
            if meta.comparison_group == "fullobj_2x2_20260601"
        ],
        "legacy_reference_roles": [
            role
            for role, meta in config.checkpoint_roles.items()
            if meta.comparison_group == "legacy_reference_anchor"
        ],
        "artifact_sources": {
            source_id: {
                "root": str(source.root),
                "required": source.required,
            }
            for source_id, source in config.artifact_sources.items()
        },
        "source_summaries": source_summaries,
        "role_counts_by_source": {
            source_id: source_summary["role_counts"]
            for source_id, source_summary in source_summaries.items()
        },
        "missing_required_sources": [
            source_id
            for source_id, source in config.artifact_sources.items()
            if source.required and not source_summaries[source_id]["root_exists"]
        ],
    }
    _write_json(root / "summary.json", summary)
    _write_markdown(root / "comparison_report.md", _render_report(summary))
    plot_dir = root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    _write_text(plot_dir / "role_row_counts.tsv", _render_role_counts_tsv(summary))
    _write_text(plot_dir / "role_row_counts.svg", _render_role_counts_svg(summary))
    gallery_dir = root / "gallery"
    gallery_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        gallery_dir / "gallery_index.json",
        {
            "status": "index_only",
            "source_gallery_candidates": _gallery_candidates(config.artifact_sources.values()),
        },
    )
    return {
        "status": "ok",
        "artifact_root": str(root),
        "summary": str(root / "summary.json"),
        "report": str(root / "comparison_report.md"),
        "plots": str(plot_dir),
    }


def _summarize_source(
    source: ArtifactSourceConfig,
    *,
    known_roles: tuple[str, ...],
) -> dict[str, Any]:
    root = source.root
    role_counts = {role: 0 for role in known_roles}
    jsonl_files: dict[str, dict[str, Any]] = {}
    summary_files: dict[str, dict[str, Any]] = {}
    if root.exists():
        for rel_path in ROLE_JSONL_FILES:
            path = root / rel_path
            file_summary = _summarize_jsonl(path, known_roles=known_roles)
            if file_summary["exists"]:
                jsonl_files[rel_path] = file_summary
                for role, count in file_summary["role_counts"].items():
                    role_counts[role] = role_counts.get(role, 0) + int(count)
        for rel_path in SUMMARY_JSON_FILES:
            path = root / rel_path
            if path.is_file():
                payload = _read_json(path)
                summary_files[rel_path] = {
                    "exists": True,
                    "keys": sorted(str(key) for key in payload) if isinstance(payload, Mapping) else [],
                }
    return {
        "source_id": source.source_id,
        "root": str(root),
        "root_exists": root.exists(),
        "required": source.required,
        "role_counts": role_counts,
        "jsonl_files": jsonl_files,
        "summary_files": summary_files,
    }


def _summarize_jsonl(path: Path, *, known_roles: tuple[str, ...]) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False, "row_count": 0, "role_counts": {role: 0 for role in known_roles}}
    role_counts = {role: 0 for role in known_roles}
    row_count = 0
    for row in _iter_jsonl(path):
        row_count += 1
        role = row.get("checkpoint_role")
        if role is not None:
            role_text = str(role)
            role_counts[role_text] = role_counts.get(role_text, 0) + 1
    return {"exists": True, "row_count": row_count, "role_counts": role_counts}


def _render_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# FullObj Policy/Objectivity Mechanism Comparison",
        "",
        f"- project_id: `{summary['project_id']}`",
        f"- run_id: `{summary['run_id']}`",
        f"- evidence_scope: artifact aggregation only; no production training launched",
        "",
        "## Role Groups",
        "",
        f"- clean_2x2_roles: {', '.join(summary['clean_2x2_roles'])}",
        f"- legacy_reference_roles: {', '.join(summary['legacy_reference_roles'])}",
        "",
        "## Source Status",
        "",
        "| source | root exists | required | observed rows |",
        "| --- | ---: | ---: | ---: |",
    ]
    for source_id, source_summary in summary["source_summaries"].items():
        observed = sum(int(value) for value in source_summary["role_counts"].values())
        lines.append(
            f"| `{source_id}` | {source_summary['root_exists']} | "
            f"{source_summary['required']} | {observed} |"
        )
    lines.extend(["", "## Role Row Counts", "", "| source | role | rows |", "| --- | --- | ---: |"])
    for source_id, counts in summary["role_counts_by_source"].items():
        for role, count in counts.items():
            lines.append(f"| `{source_id}` | `{role}` | {count} |")
    if summary["missing_required_sources"]:
        lines.extend(["", "## Missing Required Sources", ""])
        lines.extend(f"- `{source_id}`" for source_id in summary["missing_required_sources"])
    return "\n".join(lines) + "\n"


def _render_role_counts_tsv(summary: Mapping[str, Any]) -> str:
    lines = ["source\trole\trows"]
    for source_id, counts in summary["role_counts_by_source"].items():
        for role, count in counts.items():
            lines.append(f"{source_id}\t{role}\t{count}")
    return "\n".join(lines) + "\n"


def _render_role_counts_svg(summary: Mapping[str, Any]) -> str:
    rows: list[tuple[str, str, int]] = []
    for source_id, counts in summary["role_counts_by_source"].items():
        for role, count in counts.items():
            rows.append((str(source_id), str(role), int(count)))
    max_count = max([count for _, _, count in rows], default=1) or 1
    width = 960
    row_h = 22
    height = 40 + row_h * max(1, len(rows))
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="12" y="22" font-family="monospace" font-size="14">Role row counts by source</text>',
    ]
    for idx, (source_id, role, count) in enumerate(rows):
        y = 40 + idx * row_h
        bar_w = int((count / max_count) * 360)
        label = f"{source_id} / {role}"
        out.append(f'<text x="12" y="{y + 14}" font-family="monospace" font-size="11">{_xml_escape(label[:72])}</text>')
        out.append(f'<rect x="540" y="{y + 3}" width="{bar_w}" height="14" fill="#3366cc"/>')
        out.append(f'<text x="{550 + bar_w}" y="{y + 14}" font-family="monospace" font-size="11">{count}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def _gallery_candidates(sources: Iterable[ArtifactSourceConfig]) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    for source in sources:
        for rel_path in ("gallery/index.md", "fn_probe/gallery/index.md", "gallery/gallery_summary.json"):
            path = source.root / rel_path
            if path.exists():
                candidates.append({"source_id": source.source_id, "path": str(path)})
    return candidates


def _iter_jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, Mapping):
                yield payload


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_markdown(path: Path, text: str) -> None:
    _write_text(path, text)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _xml_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
