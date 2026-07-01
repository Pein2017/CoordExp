from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = 1
ANALYSIS_NAME = "autoreg_supervision_credit"

LANE_A_SUMMARY = "rollout_anatomy/summary.json"
LANE_B_MERGED_FILES = (
    "prefix_boundary/summary.json",
    "prefix_boundary/per_case.jsonl",
    "prefix_boundary/merge_summary.json",
)

COORDINATE_LOCALITY_CANDIDATES = (
    "coordinate_locality/summary.json",
    "hard_ce_coord_logit_locality/summary.json",
    "coord_logit_locality/summary.json",
    "coord_soft_ce_locality/summary.json",
)

LANE_C_CANDIDATES = (
    "x1_basin_attribution/summary.json",
    "next_object_posterior/summary.json",
    "coordinate_locality/x1_basin_attribution_summary.json",
)


def run_supervision_credit(
    *,
    analysis_root: str | Path,
    training_run_dir: str | Path,
    checkpoint: str,
    dataset_slice: str | None = None,
    metric_family: str | None = None,
) -> dict[str, Any]:
    """Build and write the Lane F supervision-credit evidence ledger."""

    analysis_root_path = Path(analysis_root).resolve()
    training_run_path = Path(training_run_dir).resolve()
    checkpoint_path = Path(checkpoint).resolve()

    summary = build_supervision_credit_summary(
        analysis_root=analysis_root_path,
        training_run_dir=training_run_path,
        checkpoint=str(checkpoint_path),
        dataset_slice=dataset_slice,
        metric_family=metric_family,
    )
    output_dir = analysis_root_path / "supervision_credit"
    _write_json(output_dir / "summary.json", summary)
    _write_report(output_dir / "report.md", summary)
    return summary


def build_supervision_credit_summary(
    *,
    analysis_root: str | Path,
    training_run_dir: str | Path,
    checkpoint: str,
    dataset_slice: str | None = None,
    metric_family: str | None = None,
) -> dict[str, Any]:
    analysis_root_path = Path(analysis_root).resolve()
    training_run_path = Path(training_run_dir).resolve()
    checkpoint_path = Path(checkpoint).resolve()

    rollout_summary = _read_json_if_present(analysis_root_path / LANE_A_SUMMARY)
    rollout_scope = _dict_or_empty(rollout_summary.get("scope") if rollout_summary else {})
    inferred_dataset_slice = dataset_slice or _string_or_none(rollout_scope.get("dataset_slice"))
    inferred_metric_family = metric_family or _string_or_none(rollout_scope.get("metric_family"))

    training_artifacts = _training_artifact_ledger(training_run_path, checkpoint_path)
    lane_a_status = _single_file_status(analysis_root_path / LANE_A_SUMMARY)
    coordinate_locality = _candidate_status(analysis_root_path, COORDINATE_LOCALITY_CANDIDATES)
    lane_b_status = _prefix_boundary_status(analysis_root_path)
    lane_c_status = _candidate_status(analysis_root_path, LANE_C_CANDIDATES)

    resolved_config = _read_json_if_present(training_run_path / "resolved_config.json")
    trainer_state = _read_json_if_present(checkpoint_path / "trainer_state.json")
    logging_rows = _read_jsonl_if_present(training_run_path / "logging.jsonl")

    observed_logs = _summarize_training_logs(logging_rows, trainer_state)
    inputs = {
        "training_artifacts": training_artifacts,
        "lane_a_rollout_anatomy": lane_a_status,
        "coordinate_locality": coordinate_locality,
        "lane_b_prefix_boundary": lane_b_status,
        "lane_c_x1_basin_attribution": lane_c_status,
    }

    prefix_boundary = _summarize_prefix_boundary(analysis_root_path, lane_b_status)
    h5_readout = _h5_readout(inputs, observed_logs)

    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_name": ANALYSIS_NAME,
        "scope": {
            "checkpoint": str(checkpoint_path),
            "training_run_dir": str(training_run_path),
            "analysis_root": str(analysis_root_path),
            "dataset_slice": inferred_dataset_slice or "unknown",
            "metric_family": inferred_metric_family or "unknown",
            "claim_scope": "artifact_ledger_only_no_training_claim",
        },
        "inputs": inputs,
        "training_objective": _training_objective(resolved_config),
        "observed_training_logs": observed_logs,
        "code_capability": {
            "current_repo_emits_target_mix_events": True,
            "capability_is_not_checkpoint_evidence": True,
        },
        "rollout_anatomy": {
            "status": lane_a_status["status"],
            "counts": _dict_or_empty(rollout_summary.get("counts") if rollout_summary else {}),
        },
        "prefix_boundary": prefix_boundary,
        "coordinate_locality": coordinate_locality,
        "h5_readout": h5_readout,
    }


def _training_artifact_ledger(training_run_dir: Path, checkpoint: Path) -> dict[str, Any]:
    files = {
        "logging_jsonl": training_run_dir / "logging.jsonl",
        "resolved_config_json": training_run_dir / "resolved_config.json",
        "trainer_state_json": checkpoint / "trainer_state.json",
    }
    entries = {name: _single_file_status(path) for name, path in files.items()}
    missing = [name for name, status in entries.items() if status["status"] == "missing"]
    entries["status"] = "complete" if not missing else "missing"
    entries["missing"] = missing
    return entries


def _single_file_status(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "status": "present" if path.exists() else "missing",
    }


def _candidate_status(root: Path, candidates: Iterable[str]) -> dict[str, Any]:
    entries = [_single_file_status(root / rel) | {"relative_path": rel} for rel in candidates]
    present = [entry for entry in entries if entry["status"] == "present"]
    return {
        "status": "present" if present else "missing",
        "present_paths": [entry["path"] for entry in present],
        "candidate_paths": entries,
    }


def _prefix_boundary_status(root: Path) -> dict[str, Any]:
    merged = {rel: _single_file_status(root / rel) for rel in LANE_B_MERGED_FILES}
    missing = [rel for rel, status in merged.items() if status["status"] == "missing"]
    shards_dir = root / "prefix_boundary/shards"
    shard_files = sorted(str(path) for path in shards_dir.glob("**/*") if path.is_file())
    if not missing:
        status = "complete"
    elif shard_files:
        status = "incomplete_shards_only"
    else:
        status = "missing"
    return {
        "status": status,
        "merged_files": merged,
        "missing_merged_files": missing,
        "shard_file_count": len(shard_files),
        "shards_dir": str(shards_dir),
    }


def _summarize_prefix_boundary(root: Path, status: Mapping[str, Any]) -> dict[str, Any]:
    if status.get("status") != "complete":
        return {
            "status": status.get("status", "missing"),
            "counts": {},
            "complete_requires": list(LANE_B_MERGED_FILES),
        }

    summary = _read_json_if_present(root / "prefix_boundary/summary.json")
    merge_summary = _read_json_if_present(root / "prefix_boundary/merge_summary.json")
    per_case = root / "prefix_boundary/per_case.jsonl"
    counts = {
        "per_case_rows": _jsonl_line_count(per_case),
    }
    for key in ("row_count", "selected_record_count", "expected_shards"):
        if key in merge_summary:
            counts[key] = merge_summary[key]
    for key in ("row_count", "case_count"):
        if key in summary:
            counts[f"summary_{key}"] = summary[key]
    return {
        "status": "complete",
        "counts": counts,
        "complete_requires": list(LANE_B_MERGED_FILES),
    }


def _training_objective(resolved_config: Mapping[str, Any]) -> dict[str, Any]:
    resolved = _dict_or_empty(resolved_config.get("resolved"))
    objective = _dict_or_empty(resolved.get("objective") or resolved_config.get("objective"))
    coord_soft_ce = objective.get("coord_soft_ce")
    return {
        "id": objective.get("id"),
        "objective_id": objective.get("id"),
        "variant": objective.get("variant"),
        "trie_support_weight": objective.get("trie_support_weight"),
        "trie_balance_weight": objective.get("trie_balance_weight"),
        "support": objective.get("trie_support_weight"),
        "balance": objective.get("trie_balance_weight"),
        "state_weighting": objective.get("state_weighting"),
        "normalization": objective.get("normalization"),
        "coord_soft_ce_configured": coord_soft_ce is not None,
        "coord_soft_ce": coord_soft_ce if isinstance(coord_soft_ce, Mapping) else None,
    }


def _summarize_training_logs(
    rows: list[dict[str, Any]],
    trainer_state: Mapping[str, Any],
) -> dict[str, Any]:
    eval_rows = [row for row in rows if _is_eval_log_row(row)]
    train_rows = [row for row in rows if not _is_eval_log_row(row)]
    target_mix_keys = sorted({key for row in rows for key in row if "/target_mix/" in key})
    coord_soft_ce_keys = sorted({key for row in rows for key in row if "coord_soft_ce" in key})
    keys = sorted({key for row in rows for key in row})
    return {
        "row_counts": {
            "total": len(rows),
            "train": len(train_rows),
            "eval": len(eval_rows),
        },
        "final_global_step": _final_global_step(rows, trainer_state),
        "target_mix_keys_present": bool(target_mix_keys),
        "target_mix_keys": target_mix_keys,
        "target_mix_numeric": _numeric_key_summary(rows, target_mix_keys),
        "coord_soft_ce_keys_present": bool(coord_soft_ce_keys),
        "coord_soft_ce_keys": coord_soft_ce_keys,
        "coord_soft_ce_numeric": _numeric_key_summary(rows, coord_soft_ce_keys),
        "available_key_prefixes": _available_key_prefixes(keys),
    }


def _is_eval_log_row(row: Mapping[str, Any]) -> bool:
    return any(key == "eval_loss" or key.startswith("eval_") or key.startswith("eval/") for key in row)


def _final_global_step(rows: list[dict[str, Any]], trainer_state: Mapping[str, Any]) -> int | float | None:
    value = trainer_state.get("global_step")
    if isinstance(value, (int, float)):
        return value
    numeric_steps = [
        row["step"]
        for row in rows
        if isinstance(row.get("step"), (int, float)) and not isinstance(row.get("step"), bool)
    ]
    if numeric_steps:
        return max(numeric_steps)
    return None


def _numeric_key_summary(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, Any]:
    last: dict[str, int | float] = {}
    mins: dict[str, int | float] = {}
    maxes: dict[str, int | float] = {}
    counts: dict[str, int] = {}
    for key in keys:
        values = [
            row[key]
            for row in rows
            if isinstance(row.get(key), (int, float)) and not isinstance(row.get(key), bool)
        ]
        if not values:
            continue
        last[key] = values[-1]
        mins[key] = min(values)
        maxes[key] = max(values)
        counts[key] = len(values)
    return {
        "last": last,
        "min": mins,
        "max": maxes,
        "non_null_counts": counts,
    }


def _available_key_prefixes(keys: list[str]) -> list[str]:
    prefixes: set[str] = set()
    for key in keys:
        if "/" in key:
            prefixes.add(key.split("/", 1)[0])
        elif "_" in key:
            prefixes.add(key.split("_", 1)[0])
        else:
            prefixes.add(key)
    return sorted(prefixes)


def _h5_readout(inputs: Mapping[str, Any], observed_logs: Mapping[str, Any]) -> dict[str, Any]:
    missing_discriminators: list[str] = []
    if inputs["lane_c_x1_basin_attribution"]["status"] != "present":
        missing_discriminators.append("lane_c_x1_basin_attribution")
    if inputs["lane_b_prefix_boundary"]["status"] != "complete":
        missing_discriminators.append("lane_b_prefix_boundary")
    if inputs["coordinate_locality"]["status"] != "present":
        missing_discriminators.append("coordinate_locality")
    if not observed_logs["target_mix_keys_present"]:
        missing_discriminators.append("checkpoint_target_mix_logs")
    claim_bounds = [
        "CPU-only evidence ledger",
        "No GPU inference, model load, tmux launch, or training run",
        "Current repo metric capability is not evidence that this checkpoint logged those metrics",
    ]
    if missing_discriminators:
        claim_bounds.append(
            "No H5 objective recommendation until missing discriminators are resolved: "
            + ", ".join(missing_discriminators)
        )
    else:
        claim_bounds.append(
            "No production training recommendation from this ledger alone; require matched rollout evidence"
        )
    return {
        "status": "inconclusive_missing_lane",
        "support_level": "compatible_only",
        "missing_discriminators": missing_discriminators,
        "production_training_recommendation": "none",
        "claim_bounds": claim_bounds,
    }


def _write_report(path: Path, summary: Mapping[str, Any]) -> None:
    scope = _dict_or_empty(summary.get("scope"))
    observed = _dict_or_empty(summary.get("observed_training_logs"))
    h5 = _dict_or_empty(summary.get("h5_readout"))
    prefix_boundary = _dict_or_empty(summary.get("prefix_boundary"))
    rollout = _dict_or_empty(summary.get("rollout_anatomy"))
    objective = _dict_or_empty(summary.get("training_objective"))
    lines = [
        "# Supervision Credit Evidence Ledger",
        "",
        f"- analysis_name: {summary.get('analysis_name')}",
        f"- schema_version: {summary.get('schema_version')}",
        f"- checkpoint: {scope.get('checkpoint')}",
        f"- training_run_dir: {scope.get('training_run_dir')}",
        f"- analysis_root: {scope.get('analysis_root')}",
        f"- dataset_slice: {scope.get('dataset_slice')}",
        f"- metric_family: {scope.get('metric_family')}",
        f"- claim_scope: {scope.get('claim_scope')}",
        "",
        "## H5 Readout",
        "",
        f"- status: {h5.get('status')}",
        f"- support_level: {h5.get('support_level')}",
        f"- production_training_recommendation: {h5.get('production_training_recommendation')}",
        f"- missing_discriminators: {', '.join(h5.get('missing_discriminators') or []) or 'none'}",
        "",
        "## Training Objective",
        "",
        f"- objective_id: {objective.get('objective_id')}",
        f"- variant: {objective.get('variant')}",
        f"- support: {objective.get('support')}",
        f"- balance: {objective.get('balance')}",
        f"- state_weighting: {objective.get('state_weighting')}",
        f"- normalization: {objective.get('normalization')}",
        f"- coord_soft_ce_configured: {objective.get('coord_soft_ce_configured')}",
        "",
        "## Observed Training Logs",
        "",
        f"- row_counts: {json.dumps(observed.get('row_counts'), sort_keys=True)}",
        f"- final_global_step: {observed.get('final_global_step')}",
        f"- target_mix_keys_present: {observed.get('target_mix_keys_present')}",
        f"- target_mix_keys: {', '.join(observed.get('target_mix_keys') or []) or 'none'}",
        f"- coord_soft_ce_keys_present: {observed.get('coord_soft_ce_keys_present')}",
        f"- available_key_prefixes: {', '.join(observed.get('available_key_prefixes') or []) or 'none'}",
        "",
        "## Lane Inputs",
        "",
        f"- rollout_anatomy_status: {rollout.get('status')}",
        f"- rollout_anatomy_counts: {json.dumps(rollout.get('counts'), sort_keys=True)}",
        f"- prefix_boundary_status: {prefix_boundary.get('status')}",
        f"- prefix_boundary_counts: {json.dumps(prefix_boundary.get('counts'), sort_keys=True)}",
        "",
        "## Claim Bounds",
        "",
    ]
    lines.extend(f"- {bound}" for bound in h5.get("claim_bounds", []))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json_if_present(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _read_jsonl_if_present(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(payload)
    return rows


def _jsonl_line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
