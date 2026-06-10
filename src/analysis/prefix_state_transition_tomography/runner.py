from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Sequence

from .config import KNOWN_STAGES, PrefixStateTransitionConfig, load_config
from .gallery import materialize_gallery
from .jsonl import read_jsonl, write_jsonl
from .merge_report import merge_artifacts, write_report
from .paired_probe import run_paired_checkpoint_probe, validate_sampled_image_paths
from .prefix_state_index import build_prefix_state_index


def build_dry_run_plan(
    config: PrefixStateTransitionConfig,
    *,
    stages: Sequence[str],
    shard_id: int | None = None,
    allow_overwrite: bool = False,
) -> dict[str, Any]:
    _validate_stages(stages)
    _validate_shard_id(shard_id, config.sampling.num_shards)
    return {
        "project_id": config.project_id,
        "run_id": config.run_id,
        "index_checkpoint_id": config.index_checkpoint_id,
        "index_checkpoint_role": config.index_checkpoint_role,
        "artifact_root": str(config.artifact_root),
        "train_jsonl": str(config.train_jsonl),
        "val_jsonl": str(config.val_jsonl),
        "stages": list(stages),
        "checkpoints": {
            role: str(checkpoint.checkpoint_path)
            for role, checkpoint in sorted(config.checkpoints.items())
        },
        "sampling": _json_safe(config.sampling),
        "peak": _json_safe(config.peak),
        "shard_id": shard_id,
        "allow_overwrite": allow_overwrite,
        "dry_run": True,
    }


def run_from_config(
    config_path: Path,
    *,
    stages: Sequence[str] | None = None,
    dry_run: bool = False,
    allow_overwrite: bool = False,
    shard_id: int | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    selected_stages = tuple(stages or config.stages)
    _validate_stages(selected_stages)
    _validate_shard_id(shard_id, config.sampling.num_shards)
    if dry_run:
        return build_dry_run_plan(
            config,
            stages=selected_stages,
            shard_id=shard_id,
            allow_overwrite=allow_overwrite,
        )
    result: dict[str, Any] = {
        "project_id": config.project_id,
        "artifact_root": str(config.artifact_root),
        "stages": list(selected_stages),
        "stage_results": {},
    }
    if config.artifact_root.exists() and not allow_overwrite and "prefix_state_index" in selected_stages:
        raise FileExistsError(f"artifact root already exists: {config.artifact_root}")
    config.artifact_root.mkdir(parents=True, exist_ok=True)
    for stage in selected_stages:
        if stage == "prefix_state_index":
            result["stage_results"][stage] = _run_prefix_state_index(config, config_path)
        elif stage == "validate":
            result["stage_results"][stage] = _run_validate(config)
        elif stage == "paired_checkpoint_probe":
            if shard_id is None:
                raise ValueError("paired_checkpoint_probe requires --shard-id")
            result["stage_results"][stage] = run_paired_checkpoint_probe(
                config=config,
                shard_id=shard_id,
            )
        elif stage == "merge":
            result["stage_results"][stage] = merge_artifacts(artifact_root=config.artifact_root)
        elif stage == "report":
            report_path = write_report(artifact_root=config.artifact_root)
            result["stage_results"][stage] = {"status": "ok", "report_path": str(report_path)}
        elif stage == "gallery":
            result["stage_results"][stage] = materialize_gallery(artifact_root=config.artifact_root)
        else:
            raise NotImplementedError(f"stage not implemented yet: {stage}")
    return result


def _validate_stages(stages: Sequence[str]) -> None:
    if not stages:
        raise ValueError("stages must not be empty")
    unknown = sorted(set(stages) - KNOWN_STAGES)
    if unknown:
        raise ValueError(f"unknown stage: {unknown[0]}")


def _validate_shard_id(shard_id: int | None, num_shards: int) -> None:
    if shard_id is None:
        return
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"shard_id must satisfy 0 <= shard_id < {num_shards}")


def _run_prefix_state_index(
    config: PrefixStateTransitionConfig,
    config_path: Path,
) -> dict[str, Any]:
    rows, sampled_rows, summary = build_prefix_state_index(
        train_jsonl=config.train_jsonl,
        val_jsonl=config.val_jsonl,
        run_id=config.run_id,
        checkpoint_id=config.index_checkpoint_id,
        checkpoint_role=config.index_checkpoint_role,
        max_prefix_states=config.sampling.max_prefix_states,
        num_shards=config.sampling.num_shards,
        seed=config.sampling.seed,
        easy_sanity_max_fraction=config.sampling.easy_sanity_max_fraction,
    )
    (config.artifact_root / "resolved_config.yaml").write_text(
        Path(config_path).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    index_count = write_jsonl(config.artifact_root / "prefix_state_index.jsonl", rows)
    sampled_count = write_jsonl(
        config.artifact_root / "prefix_state_sampled_rows.jsonl",
        sampled_rows,
    )
    summary_path = config.artifact_root / "prefix_state_index_summary.json"
    summary_path.write_text(
        _json_dumps(summary),
        encoding="utf-8",
    )
    return {
        "status": "ok",
        "prefix_state_index_rows": index_count,
        "prefix_state_sampled_rows": sampled_count,
        "launch_eligible": summary["launch_eligible"],
        "failed_launch_gates": summary["failed_launch_gates"],
    }


def _run_validate(config: PrefixStateTransitionConfig) -> dict[str, Any]:
    required = [
        "resolved_config.yaml",
        "prefix_state_index.jsonl",
        "prefix_state_index_summary.json",
        "prefix_state_sampled_rows.jsonl",
    ]
    missing = [name for name in required if not (config.artifact_root / name).exists()]
    if missing:
        return {"status": "missing_artifacts", "missing": missing}
    rows = read_jsonl(config.artifact_root / "prefix_state_sampled_rows.jsonl")
    summary = _read_json(config.artifact_root / "prefix_state_index_summary.json")
    image_validation = validate_sampled_image_paths(config=config)
    if image_validation.get("status") != "ok":
        raise FileNotFoundError(f"sampled image validation failed: {image_validation}")
    return {
        "status": "ok",
        "sampled_prefix_state_rows": len(rows),
        "launch_eligible": summary.get("launch_eligible"),
        "failed_launch_gates": summary.get("failed_launch_gates"),
        "image_validation": image_validation,
    }


def _read_json(path: Path) -> dict[str, Any]:
    import json

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _json_dumps(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
