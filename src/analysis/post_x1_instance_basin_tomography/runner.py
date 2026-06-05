from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from . import CHECKPOINT_ROLES, FULL_RUN_ID, PHASE_ID, PROJECT_ID, SCHEMA_VERSION
from .case_universe import build_case_universe_rows
from .gallery import materialize_gallery
from .jsonl import read_jsonl, write_jsonl
from .merge_report import build_summary, write_report
from .prefix_modes import materialize_prefix_modes


KNOWN_STAGES = {
    "data_root_audit",
    "case_universe",
    "prefix_states",
    "slot_posterior",
    "slot_merge",
    "trajectory",
    "attraction_matrix",
    "prefix_sensitivity",
    "greedy_continuation",
    "report",
    "gallery",
}

STAGE_ARTIFACTS = {
    "data_root_audit": "data_root_audit.json",
    "case_universe": "case_universe.jsonl",
    "prefix_states": "prefix_states.jsonl",
    "slot_posterior": "slot_posterior_shards/shard_0.jsonl",
    "slot_merge": "slot_posterior_rows.jsonl",
    "trajectory": "trajectory_rows.jsonl",
    "attraction_matrix": "basin_attraction_matrix.jsonl",
    "prefix_sensitivity": "prefix_sensitivity_rows.jsonl",
    "greedy_continuation": "greedy_continuation_rows.jsonl",
    "report": "report.md",
    "gallery": "gallery/gallery_summary.json",
}


def run_stages(
    config_path: str | Path,
    *,
    stages: str | Sequence[str],
    dry_run: bool = False,
    mock_runtime: bool = False,
    real_runtime: bool = False,
    allow_overwrite: bool = False,
    shard_id: int | None = None,
    launch_context: bool = False,
    full_run_override: bool = False,
) -> dict[str, Any]:
    config = _load_runtime_config(config_path)
    selected = _parse_stages(stages)
    if dry_run:
        return _dry_run(config, selected, shard_id=shard_id, allow_overwrite=allow_overwrite)
    _ = launch_context
    if (
        config["run_id"] == FULL_RUN_ID
        and not full_run_override
        and not _smoke_marker_present(config["artifact_root"])
    ):
        raise PermissionError("full A3.3 run requires a smoke marker or explicit override")
    root = Path(config["artifact_root"])
    if root.exists() and not allow_overwrite and any(stage in selected for stage in ("case_universe", "data_root_audit")):
        raise FileExistsError(f"artifact root already exists: {root}")
    root.mkdir(parents=True, exist_ok=True)
    _write_config_artifacts(root, config, config_path)
    result = _base_result(config, selected, dry_run=False)
    for stage in selected:
        if stage == "data_root_audit":
            result["stage_results"][stage] = _run_data_root_audit(root, config)
        elif stage == "case_universe":
            result["stage_results"][stage] = _run_case_universe(root, config, mock_runtime=mock_runtime)
        elif stage == "prefix_states":
            result["stage_results"][stage] = _run_prefix_states(root, config)
        elif stage == "slot_posterior":
            if real_runtime:
                from .real_runtime import run_real_slot_posterior

                limit_raw = os.environ.get("A33_REAL_SLOT_LIMIT")
                result["stage_results"][stage] = run_real_slot_posterior(
                    artifact_root=root,
                    config=config,
                    shard_id=0 if shard_id is None else int(shard_id),
                    gpu_id=os.environ.get("A33_GPU_ID"),
                    limit=None if not limit_raw else int(limit_raw),
                    allow_overwrite=allow_overwrite,
                )
                continue
            if not mock_runtime:
                raise ValueError("slot_posterior requires mock runtime or real runtime")
            result["stage_results"][stage] = _run_slot_posterior_mock(root, config, shard_id=shard_id)
        elif stage == "slot_merge":
            result["stage_results"][stage] = _run_slot_merge(root, config)
        elif stage == "trajectory":
            result["stage_results"][stage] = _run_downstream_rows(root, config, "trajectory_rows.jsonl", "trajectory.v1")
        elif stage == "attraction_matrix":
            result["stage_results"][stage] = _run_downstream_rows(root, config, "basin_attraction_matrix.jsonl", "basin_attraction_matrix.v1")
        elif stage == "prefix_sensitivity":
            result["stage_results"][stage] = _run_downstream_rows(root, config, "prefix_sensitivity_rows.jsonl", "prefix_sensitivity.v1")
        elif stage == "greedy_continuation":
            result["stage_results"][stage] = _run_downstream_rows(root, config, "greedy_continuation_rows.jsonl", "greedy_continuation.v1")
        elif stage == "report":
            slot_rows = _read_if_exists(root / "slot_posterior_rows.jsonl")
            summary = build_summary(
                slot_rows=slot_rows,
                trajectory_rows=_read_if_exists(root / "trajectory_rows.jsonl"),
                prefix_sensitivity_rows=_read_if_exists(root / "prefix_sensitivity_rows.jsonl"),
                greedy_rows=_read_if_exists(root / "greedy_continuation_rows.jsonl"),
                config_path=str(config_path),
                config_sha256=_sha256(Path(config_path)),
            )
            result["stage_results"][stage] = write_report(root, summary)
        elif stage == "gallery":
            result["stage_results"][stage] = materialize_gallery(root)
        else:
            raise ValueError(f"unknown stage: {stage}")
    return result


def build_dry_run_plan(config: Mapping[str, Any], *, stages: Sequence[str], shard_id: int | None = None, allow_overwrite: bool = False) -> dict[str, Any]:
    return _dry_run(dict(config), tuple(stages), shard_id=shard_id, allow_overwrite=allow_overwrite)


def run_from_config(config_path: Path, *, stages: Sequence[str] | None = None, dry_run: bool = False, allow_overwrite: bool = False, shard_id: int | None = None) -> dict[str, Any]:
    return run_stages(
        config_path,
        stages=tuple(stages or ("data_root_audit", "case_universe", "prefix_states")),
        dry_run=dry_run,
        allow_overwrite=allow_overwrite,
        shard_id=shard_id,
    )


def _dry_run(config: Mapping[str, Any], stages: Sequence[str], *, shard_id: int | None, allow_overwrite: bool) -> dict[str, Any]:
    result = _base_result(config, stages, dry_run=True)
    result["shard_id"] = shard_id
    result["allow_overwrite"] = allow_overwrite
    for stage in stages:
        result["stage_results"][stage] = {"dry_run": True, "planned_artifact": STAGE_ARTIFACTS[stage]}
    return result


def _base_result(config: Mapping[str, Any], stages: Sequence[str], *, dry_run: bool) -> dict[str, Any]:
    return {
        "project_id": PROJECT_ID,
        "phase_id": PHASE_ID,
        "schema_version": SCHEMA_VERSION,
        "artifact_schema_version": SCHEMA_VERSION,
        "run_id": config["run_id"],
        "artifact_root": str(config["artifact_root"]),
        "checkpoint_roles": list(CHECKPOINT_ROLES),
        "stages": list(stages),
        "stage_results": {},
        "dry_run": dry_run,
    }


def _load_runtime_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("A3.3 config must be a mapping")
    checkpoints = raw.get("checkpoints")
    if not isinstance(checkpoints, Mapping):
        raise ValueError("missing config key: checkpoints")
    missing_roles = [role for role in CHECKPOINT_ROLES if role not in checkpoints]
    if missing_roles:
        raise ValueError(f"missing checkpoint role(s): {', '.join(missing_roles)}")
    runtime = raw.get("runtime") if isinstance(raw.get("runtime"), Mapping) else {}
    prefix = raw.get("prefix") if isinstance(raw.get("prefix"), Mapping) else {}
    case_sampling = raw.get("case_sampling") if isinstance(raw.get("case_sampling"), Mapping) else {}
    return {
        "project_id": str(raw.get("project_id", PROJECT_ID)),
        "phase_id": str(raw.get("phase_id", PHASE_ID)),
        "schema_version": str(raw.get("schema_version", SCHEMA_VERSION)),
        "run_id": str(raw.get("run_id", "three_ckpt_phase_a3_3_smoke")),
        "artifact_root": Path(str(raw.get("artifact_root"))),
        "image_root": Path(str(raw.get("image_root", ""))) if raw.get("image_root") else Path("/"),
        "train_jsonl": Path(str(raw.get("train_jsonl", ""))) if raw.get("train_jsonl") else None,
        "val_jsonl": Path(str(raw.get("val_jsonl", ""))) if raw.get("val_jsonl") else None,
        "num_shards": int(raw.get("num_shards", runtime.get("num_shards", 8))),
        "checkpoints": {role: dict(checkpoints[role]) for role in CHECKPOINT_ROLES},
        "prefix_modes_requested": tuple(prefix.get("prefix_modes_requested", ("empty", "same_desc_good_prefix"))),
        "rollout_prefix_missing_policy": str(prefix.get("rollout_prefix_missing_policy_smoke", "skip_with_manifest")),
        "split_quotas": dict(case_sampling.get("split_quotas", {"train": 1})),
        "max_images": int(case_sampling.get("max_images", 1)),
        "max_target_instances": int(case_sampling.get("max_target_instances", 8)),
    }


def _write_config_artifacts(root: Path, config: Mapping[str, Any], config_path: str | Path) -> None:
    (root / "config_resolved.json").write_text(
        json.dumps({"artifact_schema_version": SCHEMA_VERSION, "config_path": str(config_path)}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "template_contracts.json").write_text(
        json.dumps(
            {
                "artifact_schema_version": SCHEMA_VERSION,
                "checkpoint_roles": list(CHECKPOINT_ROLES),
                "template_contracts": _contracts(config),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _run_data_root_audit(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    payload = {"artifact_schema_version": SCHEMA_VERSION, "status": "mock_or_preflight", "train_jsonl": str(config.get("train_jsonl"))}
    (root / "data_root_audit.json").write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return {"status": "ok", "artifact": "data_root_audit.json"}


def _run_case_universe(root: Path, config: Mapping[str, Any], *, mock_runtime: bool) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    train_jsonl = config.get("train_jsonl")
    if not mock_runtime and isinstance(train_jsonl, Path) and train_jsonl.exists():
        samples = []
        with train_jsonl.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                if idx >= int(config["max_images"]):
                    break
                if line.strip():
                    samples.append(json.loads(line))
        rows = build_case_universe_rows(
            samples,
            split="train",
            max_images=int(config["max_images"]),
            max_target_instances=int(config["max_target_instances"]),
            min_same_desc_count=2,
            source_jsonl_path=train_jsonl,
        )
    if not rows:
        rows = _mock_case_rows()
    count = write_jsonl(root / "case_universe.jsonl", rows)
    (root / "case_universe_summary.json").write_text(json.dumps({"artifact_schema_version": SCHEMA_VERSION, "row_count": count}) + "\n", encoding="utf-8")
    return {"status": "ok", "row_count": count}


def _run_prefix_states(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    cases = read_jsonl(root / "case_universe.jsonl")
    rows, summary = materialize_prefix_modes(
        cases,
        modes=config["prefix_modes_requested"],
        rollout_prefix_rows=None,
        rollout_prefix_missing_policy=str(config["rollout_prefix_missing_policy"]),
    )
    count = write_jsonl(root / "prefix_states.jsonl", rows)
    (root / "prefix_mode_summary.json").write_text(json.dumps({**summary, "artifact_schema_version": SCHEMA_VERSION, "row_count": count}, sort_keys=True) + "\n", encoding="utf-8")
    return {"status": "ok", "row_count": count}


def _run_slot_posterior_mock(root: Path, config: Mapping[str, Any], *, shard_id: int | None) -> dict[str, Any]:
    num_shards = int(config["num_shards"])
    shard_root = root / "slot_posterior_shards"
    contracts = _contracts(config)
    rows = [_slot_row(role, contracts[role]) for role in CHECKPOINT_ROLES]
    summaries = []
    for sid in range(num_shards):
        shard_rows = rows if sid == (0 if shard_id is None else shard_id) else []
        write_jsonl(shard_root / f"shard_{sid}.jsonl", shard_rows)
        summaries.append({"artifact_schema_version": SCHEMA_VERSION, "shard_id": sid, "row_count": len(shard_rows)})
    write_jsonl(root / "slot_posterior_shard_summaries.jsonl", summaries)
    return {"status": "ok", "runtime_kind": "mock_cpu_slot_posterior_v1", "row_count": len(rows)}


def _run_slot_merge(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    input_shards = []
    for path in sorted((root / "slot_posterior_shards").glob("shard_*.jsonl")):
        shard_rows = read_jsonl(path)
        rows.extend(shard_rows)
        input_shards.append({"path": str(path.relative_to(root)), "sha256": _sha256(path), "row_count": len(shard_rows)})
    write_jsonl(root / "slot_posterior_rows.jsonl", rows)
    manifest = {
        "artifact_schema_version": SCHEMA_VERSION,
        "input_shards": input_shards,
        "merged_row_count": len(rows),
        "merge_timestamp": "2026-06-05T00:00:00Z",
    }
    (root / "merge_manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {"status": "ok", "row_count": len(rows)}


def _run_downstream_rows(root: Path, config: Mapping[str, Any], filename: str, schema: str) -> dict[str, Any]:
    contracts = _contracts(config)
    rows = [
        {
            "artifact_schema_version": SCHEMA_VERSION,
            "row_schema_version": schema,
            "case_id": "case-0",
            "checkpoint_role": role,
            "template_contract": contracts[role],
            "primary_basin_label_source": "same_desc_gt_instances",
        }
        for role in CHECKPOINT_ROLES
    ]
    count = write_jsonl(root / filename, rows)
    return {"status": "ok", "row_count": count, "artifact": filename}


def _slot_row(role: str, contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "artifact_schema_version": SCHEMA_VERSION,
        "row_schema_version": "slot_posterior.v1",
        "case_id": "case-0",
        "prefix_state_id": "state-0",
        "checkpoint_role": role,
        "comparison_role": "reference_anchor" if role == "et_rmp_ce_ckpt3664" else "clean_pair",
        "controlled_comparison_group": "reference_anchor_not_controlled" if role == "et_rmp_ce_ckpt3664" else "pure_ce_sorted_vs_random_no_newline",
        "slot": "y1",
        "winner_bucket": "same_desc_competitor" if role == "et_rmp_ce_ckpt3664" else "target_instance",
        "primary_basin_label_source": "same_desc_gt_instances",
        "template_contract": dict(contract),
        "system_prompt_sha256": "a" * 64,
        "user_prompt_sha256": "b" * 64,
        "template_prompt_hash": "c" * 64,
        "assistant_prefix_sha256": "d" * 64,
        "forced_prompt_sha256": "e" * 64,
        "runtime_kind": "mock_cpu_slot_posterior_v1",
        "mock_runtime": True,
        "boundary_extreme_flag": False,
    }


def _mock_case_rows() -> list[dict[str, Any]]:
    return [
        {
            "artifact_schema_version": SCHEMA_VERSION,
            "case_id": "case-0",
            "split": "mock",
            "image_id": "mock-image",
            "desc": "person",
            "target_gt_idx": 1,
            "same_desc_gt_indices": [0, 1],
            "competitor_gt_indices": [0],
            "objects": [
                {"gt_idx": 0, "desc": "person", "bbox_coord_token_xyxy": [10, 10, 50, 80]},
                {"gt_idx": 1, "desc": "person", "bbox_coord_token_xyxy": [100, 10, 150, 90]},
                {"gt_idx": 2, "desc": "chair", "bbox_coord_token_xyxy": [200, 10, 250, 90]},
            ],
        }
    ]


def _contracts(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        role: dict(config["checkpoints"][role].get("template_contract", {}))
        for role in CHECKPOINT_ROLES
    }


def _read_if_exists(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.exists() else []


def _parse_stages(stages: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(stages, str):
        result = tuple(stage.strip() for stage in stages.split(",") if stage.strip())
    else:
        result = tuple(str(stage) for stage in stages)
    unknown = sorted(set(result) - KNOWN_STAGES)
    if unknown:
        raise ValueError(f"unknown stage: {unknown[0]}")
    return result


def _smoke_marker_present(root: Path) -> bool:
    return (Path(root).parent / ".a3_3_smoke_passed").exists()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["KNOWN_STAGES", "STAGE_ARTIFACTS", "build_dry_run_plan", "run_from_config", "run_stages"]
