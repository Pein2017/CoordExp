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
class FamilyProgressSpec:
    family_alias: str
    sample_size: int | None = None
    verifier_summary_json: str | None = None
    oracle_summary_json: str | None = None
    recall_probe_summary_json: str | None = None
    oracle_blocker_kind: str | None = None
    oracle_blocker_detail: str | None = None


@dataclass(frozen=True)
class RecallProgressConfig:
    run: RunConfig
    families: tuple[FamilyProgressSpec, ...]


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return payload


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


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


def _optional_nonempty_str(parent: dict[str, Any], key: str) -> str | None:
    value = parent.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string when provided.")
    return value.strip()


def _optional_int(parent: dict[str, Any], key: str) -> int | None:
    value = parent.get(key)
    if value is None:
        return None
    if not isinstance(value, int):
        raise ValueError(f"{key} must be an integer when provided.")
    return int(value)


def _artifact_root_for_repo(repo_root: Path) -> Path:
    repo_root = repo_root.resolve()
    parts = list(repo_root.parts)
    if ".worktrees" in parts:
        marker = parts.index(".worktrees")
        return Path(*parts[:marker])
    return repo_root


def _resolve_input_path(
    path_str: str,
    *,
    config_dir: Path,
    artifact_root: Path,
) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    config_relative = config_dir / path
    if config_relative.exists():
        return config_relative
    return artifact_root / path


def load_recall_progress_config(config_path: Path) -> RecallProgressConfig:
    payload = _load_yaml(config_path)
    run_raw = _require_mapping(payload, "run")
    families_raw = payload.get("families")
    if not isinstance(families_raw, list) or not families_raw:
        raise ValueError("families must be a non-empty list.")

    families: list[FamilyProgressSpec] = []
    for index, item in enumerate(families_raw):
        if not isinstance(item, dict):
            raise ValueError(f"families[{index}] must be a mapping.")
        families.append(
            FamilyProgressSpec(
                family_alias=_require_nonempty_str(item, "family_alias"),
                sample_size=_optional_int(item, "sample_size"),
                verifier_summary_json=_optional_nonempty_str(item, "verifier_summary_json"),
                oracle_summary_json=_optional_nonempty_str(item, "oracle_summary_json"),
                recall_probe_summary_json=_optional_nonempty_str(
                    item,
                    "recall_probe_summary_json",
                ),
                oracle_blocker_kind=_optional_nonempty_str(item, "oracle_blocker_kind"),
                oracle_blocker_detail=_optional_nonempty_str(item, "oracle_blocker_detail"),
            )
        )

    return RecallProgressConfig(
        run=RunConfig(
            name=_require_nonempty_str(run_raw, "name"),
            output_dir=_require_nonempty_str(run_raw, "output_dir"),
        ),
        families=tuple(families),
    )


def _extract_verifier_summary(payload: dict[str, Any]) -> dict[str, Any]:
    collection = payload.get("collection_health", {})
    if not isinstance(collection, dict):
        collection = {}
    gt_vs_hard_negative = payload.get("gt_vs_hard_negative", {})
    if not isinstance(gt_vs_hard_negative, dict):
        gt_vs_hard_negative = {}
    matched_vs_unmatched = payload.get("matched_vs_unmatched", {})
    if not isinstance(matched_vs_unmatched, dict):
        matched_vs_unmatched = {}
    return {
        "pred_count_total": collection.get("pred_count_total"),
        "matched_count": collection.get("matched_count"),
        "unmatched_count": collection.get("unmatched_count"),
        "duplicate_like_rate": collection.get("duplicate_like_rate"),
        "invalid_rollout_count": collection.get("invalid_rollout_count"),
        "collection_valid": collection.get("collection_valid"),
        "parser_failure_counts": collection.get("parser_failure_counts"),
        "gt_vs_hard_negative_auroc": gt_vs_hard_negative,
        "matched_vs_unmatched_auroc": matched_vs_unmatched,
        "commitment_counterfactual_correlation": payload.get(
            "commitment_counterfactual_correlation"
        ),
    }


def _extract_oracle_summary(payload: dict[str, Any]) -> dict[str, Any]:
    iou_thresholds = payload.get("iou_thresholds", {})
    if not isinstance(iou_thresholds, dict):
        iou_thresholds = {}
    iou_050 = iou_thresholds.get("0.50", {})
    if not isinstance(iou_050, dict):
        iou_050 = {}
    baseline = iou_050.get("baseline", {})
    oracle_k = iou_050.get("oracle_k", {})
    primary_recovery = payload.get("primary_recovery", {})
    if not isinstance(baseline, dict):
        baseline = {}
    if not isinstance(oracle_k, dict):
        oracle_k = {}
    if not isinstance(primary_recovery, dict):
        primary_recovery = {}
    return {
        "baseline_recall_loc": baseline.get("recall_loc"),
        "oracle_k_recall_loc": oracle_k.get("recall_loc"),
        "baseline_fn_count_loc": primary_recovery.get("baseline_fn_count_loc"),
        "recoverable_fn_count_loc": primary_recovery.get("recoverable_fn_count_loc"),
        "systematic_fn_count_loc": primary_recovery.get("systematic_fn_count_loc"),
        "recoverable_fraction_of_baseline_fn_loc": primary_recovery.get(
            "recover_fraction_loc"
        ),
    }


def _extract_recall_probe_summary(payload: dict[str, Any], family_alias: str) -> dict[str, Any] | None:
    rows = payload.get("family_metrics")
    if not isinstance(rows, list):
        return None
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("family_alias")) == family_alias:
            return dict(row)
    return None


def _derive_status(
    *,
    verifier_exists: bool,
    oracle_exists: bool,
    probe_exists: bool,
    oracle_blocker_kind: str | None,
) -> str:
    if verifier_exists and oracle_exists and probe_exists:
        return "oracle_and_verifier_complete"
    if verifier_exists and oracle_exists:
        return "oracle_and_verifier_complete_probe_pending"
    if verifier_exists and oracle_blocker_kind:
        return f"verifier_complete_oracle_blocked_by_{oracle_blocker_kind}"
    if verifier_exists:
        return "verifier_complete_oracle_pending"
    if oracle_exists:
        return "oracle_complete_verifier_missing"
    return "artifact_pending"


def _build_family_payload(
    spec: FamilyProgressSpec,
    *,
    config_dir: Path,
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact_root = _artifact_root_for_repo(repo_root)

    verifier_path = (
        _resolve_input_path(
            spec.verifier_summary_json,
            config_dir=config_dir,
            artifact_root=artifact_root,
        )
        if spec.verifier_summary_json
        else None
    )
    oracle_path = (
        _resolve_input_path(
            spec.oracle_summary_json,
            config_dir=config_dir,
            artifact_root=artifact_root,
        )
        if spec.oracle_summary_json
        else None
    )
    probe_path = (
        _resolve_input_path(
            spec.recall_probe_summary_json,
            config_dir=config_dir,
            artifact_root=artifact_root,
        )
        if spec.recall_probe_summary_json
        else None
    )

    verifier_exists = bool(verifier_path and verifier_path.exists())
    oracle_exists = bool(oracle_path and oracle_path.exists())
    probe_exists = bool(probe_path and probe_path.exists())

    family_payload: dict[str, Any] = {
        "status": _derive_status(
            verifier_exists=verifier_exists,
            oracle_exists=oracle_exists,
            probe_exists=probe_exists,
            oracle_blocker_kind=spec.oracle_blocker_kind,
        ),
    }
    if spec.sample_size is not None:
        family_payload["sample_size"] = spec.sample_size

    if verifier_exists and verifier_path is not None:
        family_payload["verifier"] = _extract_verifier_summary(_read_json(verifier_path))

    if oracle_exists and oracle_path is not None:
        family_payload.update(_extract_oracle_summary(_read_json(oracle_path)))

    if probe_exists and probe_path is not None:
        probe_payload = _extract_recall_probe_summary(_read_json(probe_path), spec.family_alias)
        if probe_payload is not None:
            family_payload["recall_probe"] = probe_payload

    if spec.oracle_blocker_kind or spec.oracle_blocker_detail:
        family_payload["oracle_blocker"] = {
            key: value
            for key, value in {
                "kind": spec.oracle_blocker_kind,
                "detail": spec.oracle_blocker_detail,
            }.items()
            if value is not None
        }

    source_payload = {
        "verifier_summary_json": str(verifier_path) if verifier_path is not None else None,
        "verifier_exists": verifier_exists,
        "oracle_summary_json": str(oracle_path) if oracle_path is not None else None,
        "oracle_exists": oracle_exists,
        "recall_probe_summary_json": str(probe_path) if probe_path is not None else None,
        "recall_probe_exists": probe_exists,
    }
    return family_payload, source_payload


def _render_report(summary: dict[str, Any]) -> str:
    families = summary.get("families", {})
    lines = [
        "# Coord Family Recall Progress",
        "",
        f"- Run: `{summary['run_name']}`",
        f"- Family count: `{len(families)}`",
        "",
    ]
    for family_alias, payload in sorted(families.items()):
        lines.append(f"## {family_alias}")
        lines.append("")
        lines.append(f"- status: `{payload.get('status')}`")
        sample_size = payload.get("sample_size")
        if sample_size is not None:
            lines.append(f"- sample_size: `{sample_size}`")
        if "baseline_recall_loc" in payload:
            lines.append(
                f"- recall_loc: baseline `{payload.get('baseline_recall_loc'):.4f}` -> "
                f"oracle-k `{payload.get('oracle_k_recall_loc'):.4f}`"
            )
            lines.append(
                f"- fn_loc: baseline `{payload.get('baseline_fn_count_loc')}` | "
                f"recoverable `{payload.get('recoverable_fn_count_loc')}` | "
                f"systematic `{payload.get('systematic_fn_count_loc')}`"
            )
        verifier = payload.get("verifier")
        if isinstance(verifier, dict):
            lines.append(
                f"- verifier: pred_count_total `{verifier.get('pred_count_total')}`, "
                f"unmatched_count `{verifier.get('unmatched_count')}`, "
                f"duplicate_like_rate `{verifier.get('duplicate_like_rate')}`"
            )
        recall_probe = payload.get("recall_probe")
        if isinstance(recall_probe, dict):
            lines.append(
                f"- recall_probe: suppressed `{recall_probe.get('suppressed_fn_rate'):.4f}`, "
                f"competitive `{recall_probe.get('competitive_fn_rate'):.4f}`, "
                f"weak_visual `{recall_probe.get('weak_visual_fn_rate'):.4f}`"
            )
        oracle_blocker = payload.get("oracle_blocker")
        if isinstance(oracle_blocker, dict):
            kind = oracle_blocker.get("kind")
            detail = oracle_blocker.get("detail")
            if kind is not None:
                lines.append(f"- oracle_blocker_kind: `{kind}`")
            if detail is not None:
                lines.append(f"- oracle_blocker_detail: {detail}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_recall_progress(
    config_path: Path,
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    config = load_recall_progress_config(config_path)
    artifact_root = _artifact_root_for_repo(repo_root)
    output_root = artifact_root / config.run.output_dir
    run_dir = output_root / config.run.name
    run_dir.mkdir(parents=True, exist_ok=True)

    families: dict[str, Any] = {}
    source_artifacts: dict[str, Any] = {}
    for spec in config.families:
        family_payload, source_payload = _build_family_payload(
            spec,
            config_dir=config_path.parent,
            repo_root=repo_root,
        )
        families[spec.family_alias] = family_payload
        source_artifacts[spec.family_alias] = source_payload

    summary = {
        "run_name": config.run.name,
        "families": families,
        "source_artifacts": source_artifacts,
    }

    _write_json(run_dir / "summary.json", summary)
    (run_dir / "report.md").write_text(_render_report(summary), encoding="utf-8")
    return {
        "run_dir": str(run_dir),
        "summary_json": str(run_dir / "summary.json"),
        "report_md": str(run_dir / "report.md"),
        "family_count": len(families),
    }
