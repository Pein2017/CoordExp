from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPERIMENT_MANIFEST_SCHEMA_VERSION = 1


def _module_names(raw: Any) -> list[str]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    names: list[str] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("name", "") or "").strip()
        if name:
            names.append(name)
    return names


def build_experiment_manifest_payload(
    *,
    output_dir: str | Path,
    config_path: str,
    base_config_path: str | None,
    run_name: str,
    dataset_seed: int,
    experiment: Mapping[str, Any] | None,
    effective_runtime: Mapping[str, Any] | None,
    pipeline_manifest: Mapping[str, Any] | None,
    run_metadata: Mapping[str, Any] | None,
    manifest_files: Mapping[str, Any] | None,
) -> dict[str, Any]:
    artifact_files: dict[str, str] = {}
    if isinstance(manifest_files, Mapping):
        for key, value in manifest_files.items():
            artifact_files[str(key)] = str(value)
    artifact_files["run_metadata"] = "run_metadata.json"
    artifact_files["experiment_manifest"] = "experiment_manifest.json"

    runtime_summary: dict[str, Any] = {}
    if isinstance(effective_runtime, Mapping):
        for key in (
            "trainer_variant",
            "checkpoint_mode",
            "resume_from_checkpoint",
            "save_only_model",
            "save_strategy",
            "save_last_epoch",
            "seed",
            "per_device_train_batch_size",
            "per_device_eval_batch_size",
            "gradient_accumulation_steps",
            "max_steps",
            "num_train_epochs",
            "dataloader_drop_last",
            "global_max_length",
            "template_max_length",
            "template_max_pixels",
            "packing",
            "encoded_sample_cache",
            "dataset_source_train_jsonl",
            "dataset_source_val_jsonl",
            "launcher",
        ):
            if key in effective_runtime:
                runtime_summary[key] = copy.deepcopy(effective_runtime[key])

    if isinstance(pipeline_manifest, Mapping):
        runtime_summary["pipeline"] = {
            "checksum": str(pipeline_manifest.get("checksum", "") or ""),
            "objective": _module_names(pipeline_manifest.get("objective")),
            "diagnostics": _module_names(pipeline_manifest.get("diagnostics")),
        }

    provenance_summary: dict[str, Any] = {}
    if isinstance(run_metadata, Mapping):
        for key in (
            "created_at",
            "git_sha",
            "git_branch",
            "git_dirty",
            "git_status_porcelain",
            "upstream",
        ):
            if key in run_metadata:
                provenance_summary[key] = copy.deepcopy(run_metadata[key])

    authored_experiment: dict[str, Any] | None = None
    if isinstance(experiment, Mapping) and experiment:
        authored_experiment = copy.deepcopy(dict(experiment))

    return {
        "schema_version": EXPERIMENT_MANIFEST_SCHEMA_VERSION,
        "identity": {
            "config_path": str(config_path or ""),
            "base_config_path": str(base_config_path or ""),
            "run_name": str(run_name or ""),
            "output_dir": str(output_dir),
            "dataset_seed": int(dataset_seed),
        },
        "experiment": {
            "authored": authored_experiment,
        },
        "runtime_summary": runtime_summary,
        "provenance_summary": provenance_summary,
        "artifacts": artifact_files,
    }


def write_experiment_manifest_file(
    *,
    output_dir: str | Path,
    config_path: str,
    base_config_path: str | None,
    run_name: str,
    dataset_seed: int,
    experiment: Mapping[str, Any] | None,
    effective_runtime: Mapping[str, Any] | None,
    pipeline_manifest: Mapping[str, Any] | None,
    run_metadata: Mapping[str, Any] | None,
    manifest_files: Mapping[str, Any] | None,
) -> Path:
    payload = build_experiment_manifest_payload(
        output_dir=output_dir,
        config_path=config_path,
        base_config_path=base_config_path,
        run_name=run_name,
        dataset_seed=dataset_seed,
        experiment=experiment,
        effective_runtime=effective_runtime,
        pipeline_manifest=pipeline_manifest,
        run_metadata=run_metadata,
        manifest_files=manifest_files,
    )
    out_path = Path(str(output_dir)) / "experiment_manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return out_path
