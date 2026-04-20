from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from src.infer.checkpoints import resolve_inference_checkpoint

REPO_ROOT = Path(__file__).resolve().parents[2]
_VALID_CHECKPOINT_HINTS = frozenset({"auto", "adapter", "merged"})
_VALID_INFER_MODES = frozenset({"auto", "coord", "text"})


@dataclass(frozen=True)
class RunSpec:
    name: str
    output_dir: str


@dataclass(frozen=True)
class FamilySpec:
    alias: str
    checkpoint_path: str
    infer_mode: str
    bbox_format: str
    checkpoint_hint: str = "auto"
    pred_coord_mode: str = "pixel"
    eval_compatibility_path: str = "confidence_postop"
    is_headline_2b_family: bool = False


@dataclass(frozen=True)
class FamilyAuditConfig:
    run: RunSpec
    families: tuple[FamilySpec, ...]


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


def _optional_nonempty_str(parent: dict[str, Any], key: str, default: str) -> str:
    value = parent.get(key, default)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string when provided.")
    normalized = value.strip()
    return normalized or default


def _optional_bool(parent: dict[str, Any], key: str, default: bool) -> bool:
    value = parent.get(key, default)
    if isinstance(value, bool):
        return value
    raise ValueError(f"{key} must be a boolean when provided.")


def _resolve_family_specs(payload: dict[str, Any]) -> tuple[FamilySpec, ...]:
    defaults = payload.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise ValueError("defaults must be a mapping when provided.")

    families_raw = payload.get("families")
    if not isinstance(families_raw, list) or not families_raw:
        raise ValueError("families must be a non-empty list.")

    families: list[FamilySpec] = []
    for index, item in enumerate(families_raw):
        if not isinstance(item, dict):
            raise ValueError(f"families[{index}] must be a mapping.")
        alias = _require_nonempty_str(item, "alias")
        checkpoint_path = _require_nonempty_str(item, "checkpoint_path")
        infer_mode = _optional_nonempty_str(
            item,
            "infer_mode",
            _optional_nonempty_str(defaults, "infer_mode", "coord"),
        )
        if infer_mode not in _VALID_INFER_MODES:
            raise ValueError(
                f"families[{index}].infer_mode must be one of {sorted(_VALID_INFER_MODES)}."
            )
        bbox_format = _optional_nonempty_str(
            item,
            "bbox_format",
            _optional_nonempty_str(defaults, "bbox_format", "xyxy"),
        )
        checkpoint_hint = _optional_nonempty_str(
            item,
            "checkpoint_hint",
            _optional_nonempty_str(defaults, "checkpoint_hint", "auto"),
        )
        if checkpoint_hint not in _VALID_CHECKPOINT_HINTS:
            raise ValueError(
                f"families[{index}].checkpoint_hint must be one of "
                f"{sorted(_VALID_CHECKPOINT_HINTS)}."
            )
        pred_coord_mode = _optional_nonempty_str(
            item,
            "pred_coord_mode",
            _optional_nonempty_str(defaults, "pred_coord_mode", "pixel"),
        )
        eval_compatibility_path = _optional_nonempty_str(
            item,
            "eval_compatibility_path",
            _optional_nonempty_str(defaults, "eval_compatibility_path", "confidence_postop"),
        )
        is_headline_2b_family = _optional_bool(
            item,
            "is_headline_2b_family",
            _optional_bool(defaults, "is_headline_2b_family", False),
        )
        families.append(
            FamilySpec(
                alias=alias,
                checkpoint_path=checkpoint_path,
                infer_mode=infer_mode,
                bbox_format=bbox_format,
                checkpoint_hint=checkpoint_hint,
                pred_coord_mode=pred_coord_mode,
                eval_compatibility_path=eval_compatibility_path,
                is_headline_2b_family=is_headline_2b_family,
            )
        )
    return tuple(families)


def load_family_audit_config(config_path: Path) -> FamilyAuditConfig:
    payload = _load_yaml(config_path)
    run_raw = _require_mapping(payload, "run")
    run = RunSpec(
        name=_require_nonempty_str(run_raw, "name"),
        output_dir=_require_nonempty_str(run_raw, "output_dir"),
    )
    return FamilyAuditConfig(run=run, families=_resolve_family_specs(payload))


def load_contract_audit_config(config_path: Path) -> FamilyAuditConfig:
    return load_family_audit_config(config_path)


def _resolve_checkpoint_path(
    checkpoint_path: str | Path,
    *,
    repo_root: Path | None = None,
) -> Path:
    path = Path(checkpoint_path).expanduser()
    if path.is_absolute() or repo_root is None:
        return path
    return _artifact_root_for_repo(repo_root) / path


def _artifact_root_for_repo(repo_root: Path) -> Path:
    repo_root = repo_root.resolve()
    parts = list(repo_root.parts)
    if ".worktrees" in parts:
        marker = parts.index(".worktrees")
        return Path(*parts[:marker])
    return repo_root


def infer_checkpoint_runtime_mode(
    checkpoint_path: Path,
    *,
    repo_root: Path | None = None,
) -> Literal["adapter", "merged"]:
    resolved_path = _resolve_checkpoint_path(checkpoint_path, repo_root=repo_root)
    resolved = resolve_inference_checkpoint(model_checkpoint=str(resolved_path))
    if resolved.checkpoint_mode == "adapter_shorthand":
        return "adapter"
    return "merged"


def _resolve_inventory_row(
    spec: FamilySpec,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    raw_path = spec.checkpoint_path.strip()
    path = _resolve_checkpoint_path(raw_path, repo_root=repo_root)
    path_exists = path.exists()
    resolved = resolve_inference_checkpoint(model_checkpoint=str(path))
    runtime_type = (
        "adapter" if resolved.checkpoint_mode == "adapter_shorthand" else "merged"
    )

    notes: list[str] = []
    if runtime_type == "adapter":
        checkpoint_type = "adapter"
        resolution_source = "runtime_detection"
        resolved_base_model_checkpoint = resolved.resolved_base_model_checkpoint
        resolved_adapter_checkpoint = resolved.resolved_adapter_checkpoint
        runtime_contract_ready = True
    elif spec.checkpoint_hint == "adapter":
        checkpoint_type = "adapter"
        resolution_source = "checkpoint_hint"
        resolved_base_model_checkpoint = None
        resolved_adapter_checkpoint = raw_path
        runtime_contract_ready = False
        notes.append(
            "Adapter classification came from checkpoint_hint; a local "
            "adapter_config.json is still required to resolve the base model at runtime."
        )
    else:
        checkpoint_type = "merged"
        resolution_source = "runtime_detection" if path_exists else "checkpoint_hint"
        resolved_base_model_checkpoint = resolved.resolved_base_model_checkpoint
        resolved_adapter_checkpoint = None
        runtime_contract_ready = path_exists
        if not path_exists:
            notes.append("Merged checkpoint path does not exist at audit time.")

    if checkpoint_type == "adapter":
        notes.append("vLLM local/server inference does not support adapter shorthand checkpoints.")

    adapter_base_model_name_or_path = None
    adapter_modules_to_save: list[str] = []
    if resolved.adapter_info is not None:
        adapter_base_model_name_or_path = resolved.adapter_info.base_model_name_or_path
        adapter_modules_to_save = list(resolved.adapter_info.modules_to_save)

    return {
        "alias": spec.alias,
        "checkpoint_path": raw_path,
        "resolved_checkpoint_path": str(path),
        "checkpoint_hint": spec.checkpoint_hint,
        "checkpoint_exists": path_exists,
        "checkpoint_type": checkpoint_type,
        "contract_resolution_source": resolution_source,
        "runtime_checkpoint_mode": resolved.checkpoint_mode,
        "runtime_load_pattern": (
            "model_checkpoint + adapter_checkpoint"
            if checkpoint_type == "adapter"
            else "model_checkpoint only"
        ),
        "infer_mode": spec.infer_mode,
        "bbox_format": spec.bbox_format,
        "pred_coord_mode": spec.pred_coord_mode,
        "eval_compatibility_path": spec.eval_compatibility_path,
        "is_headline_2b_family": spec.is_headline_2b_family,
        "resolved_base_model_checkpoint": resolved_base_model_checkpoint,
        "resolved_adapter_checkpoint": resolved_adapter_checkpoint,
        "adapter_base_model_name_or_path": adapter_base_model_name_or_path,
        "adapter_modules_to_save": adapter_modules_to_save,
        "vllm_supported": checkpoint_type == "merged",
        "runtime_contract_ready": runtime_contract_ready,
        "notes": notes,
    }


def build_family_inventory(
    specs: list[FamilySpec] | tuple[FamilySpec, ...],
    *,
    repo_root: Path | None = None,
) -> list[dict[str, Any]]:
    return [_resolve_inventory_row(spec, repo_root=repo_root) for spec in specs]


def summarize_family_inventory(rows: list[dict[str, Any]]) -> dict[str, Any]:
    checkpoint_type_counts = Counter(str(row["checkpoint_type"]) for row in rows)
    infer_mode_counts = Counter(str(row["infer_mode"]) for row in rows)
    bbox_format_counts = Counter(str(row["bbox_format"]) for row in rows)
    not_ready_aliases = [
        str(row["alias"]) for row in rows if not bool(row["runtime_contract_ready"])
    ]
    return {
        "family_count": len(rows),
        "checkpoint_type_counts": dict(sorted(checkpoint_type_counts.items())),
        "infer_mode_counts": dict(sorted(infer_mode_counts.items())),
        "bbox_format_counts": dict(sorted(bbox_format_counts.items())),
        "runtime_contract_ready_count": len(rows) - len(not_ready_aliases),
        "runtime_contract_not_ready_aliases": not_ready_aliases,
    }


def _run_dir(run: RunSpec, repo_root: Path) -> Path:
    output_dir = Path(run.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    return output_dir / run.name


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _render_markdown(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# Coord Family Contract Audit",
        "",
        f"- Family count: {summary['family_count']}",
        f"- Runtime-ready families: {summary['runtime_contract_ready_count']}",
        f"- Checkpoint types: {json.dumps(summary['checkpoint_type_counts'], sort_keys=True)}",
        "",
        "| Alias | Type | Load Pattern | Infer Mode | BBox Format | Pred Coord Mode | Eval Path | Headline 2B | Runtime Ready |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {alias} | {checkpoint_type} | {runtime_load_pattern} | {infer_mode} | "
            "{bbox_format} | {pred_coord_mode} | {eval_compatibility_path} | "
            "{is_headline_2b_family} | {runtime_contract_ready} |".format(**row)
        )
    return "\n".join(lines) + "\n"


def run_contract_audit(
    config_path: Path,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    repo_root = (repo_root or REPO_ROOT).resolve()
    config = load_family_audit_config(config_path)
    rows = build_family_inventory(config.families, repo_root=repo_root)
    summary = summarize_family_inventory(rows)
    run_dir = _run_dir(config.run, repo_root)

    inventory_json = run_dir / "inventory.json"
    family_inventory_json = run_dir / "family_inventory.json"
    inventory_jsonl = run_dir / "inventory.jsonl"
    summary_json = run_dir / "summary.json"
    report_md = run_dir / "family_contract_audit.md"

    _write_json(inventory_json, rows)
    _write_json(family_inventory_json, rows)
    _write_jsonl(inventory_jsonl, rows)
    _write_json(
        summary_json,
        {
            "run_name": config.run.name,
            **summary,
            "families": rows,
        },
    )
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text(_render_markdown(rows, summary), encoding="utf-8")

    return {
        "config_path": str(config_path),
        "run_dir": str(run_dir),
        "inventory_json": str(inventory_json),
        "family_inventory_json": str(family_inventory_json),
        "inventory_jsonl": str(inventory_jsonl),
        "summary_json": str(summary_json),
        "report_md": str(report_md),
        "family_count": summary["family_count"],
        "checkpoint_type_counts": summary["checkpoint_type_counts"],
    }


__all__ = [
    "FamilyAuditConfig",
    "FamilySpec",
    "RunSpec",
    "build_family_inventory",
    "infer_checkpoint_runtime_mode",
    "load_contract_audit_config",
    "load_family_audit_config",
    "run_contract_audit",
    "summarize_family_inventory",
]
