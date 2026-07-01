from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RunConfig:
    name: str
    output_dir: str


@dataclass(frozen=True)
class InputPaths:
    base_metadata_yaml: str
    contract_summary_json: str
    eval_bundle_summary_json: str
    recall_summary_json: str
    basin_summary_json: str


@dataclass(frozen=True)
class ReportConfig:
    run: RunConfig
    inputs: InputPaths


def _artifact_root_for_repo(repo_root: Path) -> Path:
    repo_root = repo_root.resolve()
    parts = list(repo_root.parts)
    if ".worktrees" in parts:
        marker = parts.index(".worktrees")
        return Path(*parts[:marker])
    return repo_root


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return payload


def _require_mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping.")
    return value


def _require_nonempty_str(parent: dict[str, Any], key: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string.")
    return value.strip()


def _require_summary_mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping.")
    return dict(value)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _resolve_input_path(path_str: str, *, config_dir: Path, repo_root: Path) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return _artifact_root_for_repo(repo_root) / path


def _resolve_output_dir(path_str: str, *, config_dir: Path, repo_root: Path) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    config_relative = config_dir / path
    if config_relative.exists():
        return config_relative
    return _artifact_root_for_repo(repo_root) / path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_report_config(config_path: Path) -> ReportConfig:
    payload = _load_yaml(config_path)
    run_raw = _require_mapping(payload, "run")
    inputs_raw = _require_mapping(payload, "inputs")
    return ReportConfig(
        run=RunConfig(
            name=_require_nonempty_str(run_raw, "name"),
            output_dir=_require_nonempty_str(run_raw, "output_dir"),
        ),
        inputs=InputPaths(
            base_metadata_yaml=_require_nonempty_str(inputs_raw, "base_metadata_yaml"),
            contract_summary_json=_require_nonempty_str(inputs_raw, "contract_summary_json"),
            eval_bundle_summary_json=_require_nonempty_str(inputs_raw, "eval_bundle_summary_json"),
            recall_summary_json=_require_nonempty_str(inputs_raw, "recall_summary_json"),
            basin_summary_json=_require_nonempty_str(inputs_raw, "basin_summary_json"),
        ),
    )


def build_report_summary(
    *,
    target_alias: str,
    eval_summary: dict[str, Any],
    recall_summary: dict[str, Any],
    basin_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "target_alias": target_alias,
        "eval": dict(eval_summary),
        "recall": dict(recall_summary),
        "basin": dict(basin_summary),
    }


def _render_report_md(summary: dict[str, Any], *, run_name: str) -> str:
    lines = [
        "# Mixed-Objective SOTA Probe",
        "",
        f"- run: `{run_name}`",
        f"- target alias: `{summary['target_alias']}`",
        f"- contract rows: {summary['contract'].get('family_count', '(n/a)')}",
        f"- eval keys: {', '.join(sorted(summary['eval'])) or '(none)'}",
        f"- recall keys: {', '.join(sorted(summary['recall'])) or '(none)'}",
        f"- basin keys: {', '.join(sorted(summary['basin'])) or '(none)'}",
        "",
    ]
    return "\n".join(lines)


def build_mixed_objective_sota_probe_report(
    config_path: Path,
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    config = load_report_config(config_path)
    run_dir = _resolve_output_dir(config.run.output_dir, config_dir=config_path.parent, repo_root=repo_root)
    run_dir.mkdir(parents=True, exist_ok=True)

    base_metadata = _load_yaml(
        _resolve_input_path(config.inputs.base_metadata_yaml, config_dir=config_path.parent, repo_root=repo_root)
    )
    target_alias = _require_mapping(base_metadata, "target_family")["alias"]
    if not isinstance(target_alias, str) or not target_alias.strip():
        raise ValueError("base metadata target_family.alias must be a non-empty string.")
    contract_summary = _read_json(
        _resolve_input_path(config.inputs.contract_summary_json, config_dir=config_path.parent, repo_root=repo_root)
    )
    eval_summary = _read_json(
        _resolve_input_path(config.inputs.eval_bundle_summary_json, config_dir=config_path.parent, repo_root=repo_root)
    )
    recall_summary = _read_json(
        _resolve_input_path(config.inputs.recall_summary_json, config_dir=config_path.parent, repo_root=repo_root)
    )
    basin_summary = _read_json(
        _resolve_input_path(config.inputs.basin_summary_json, config_dir=config_path.parent, repo_root=repo_root)
    )

    summary = build_report_summary(
        target_alias=target_alias.strip(),
        eval_summary=eval_summary,
        recall_summary=recall_summary,
        basin_summary=basin_summary,
    )
    summary["run_name"] = config.run.name
    summary["contract"] = contract_summary

    summary_path = run_dir / "summary.json"
    report_path = run_dir / "report.md"
    _write_json(summary_path, summary)
    report_path.write_text(_render_report_md(summary, run_name=config.run.name), encoding="utf-8")

    return {
        "run_dir": str(run_dir),
        "summary_json": str(summary_path),
        "report_md": str(report_path),
    }
