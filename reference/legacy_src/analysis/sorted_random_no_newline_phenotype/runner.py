"""A3.2 sorted-vs-random no-newline phenotype runner."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from . import PHASE_ID, PROJECT_ID, RUN_ID, SCHEMA_VERSION
from .config import A32Config, load_config
from .data_root_audit import build_data_root_audit
from .fn_probe import (
    build_fn_candidate_score_rows,
    build_fn_probe_rows,
    build_fn_slot_evidence_rows,
)
from .fn_matching import (
    MATCH_POLICY_ID,
    build_fn_case_universe,
    build_replayable_fn_cases,
    match_fn_cases_for_image,
)
from .gallery import build_gallery
from .merge_report import (
    DELTA_METRICS,
    build_report_markdown,
    merge_prefix_readout_rows,
    write_jsonl as write_report_jsonl,
    write_report,
)
from .paired_probe import (
    REAL_PREFIX_RUNTIME_KIND,
    build_mocked_paired_readout_rows,
    build_shard_manifest_row,
    checkpoint_roles_from_config,
)
from .prefix_index import build_prefix_state_index
from .rollout_phenotype import materialize_rollout_phenotype
from .status import REAL_FN_HINT_RUNTIME_KIND, evaluate_status


STAGE_NAMES = (
    "data_root_audit",
    "prefix_state_index",
    "validate",
    "paired_checkpoint_probe",
    "prefix_merge",
    "prefix_report",
    "prefix_gallery",
    "native_rollout",
    "rollout_phenotype",
    "fn_case_index",
    "fn_hint_probe",
    "fn_merge",
    "fn_report",
    "fn_gallery",
    "finalize",
)
GPU_STAGES = frozenset(
    {
        "paired_checkpoint_probe",
        "native_rollout",
        "fn_hint_probe",
    }
)
SHARDED_GPU_STAGES = frozenset(
    {
        "paired_checkpoint_probe",
        "fn_hint_probe",
    }
)
PREFIX_EVIDENCE_LABEL = "a3_2_prefix4096_hardbiased_len12000_canonical_sorted"
ROLLOUT_EVIDENCE_LABEL = "a3_2_rollout1024_greedy_len12000_native"
FN_EVIDENCE_LABEL = "a3_2_fn512_greedyfn_len12000"


def run_stages(
    config_path: str | Path,
    *,
    stages: Sequence[str] | str | None = None,
    dry_run: bool = False,
    allow_overwrite: bool = False,
    shard_id: int | None = None,
    launch_context: bool = False,
    mock_runtime: bool = False,
) -> dict[str, Any]:
    """Run or plan A3.2 stages.

    GPU-facing stages are orchestration-only here: they either dry-run, require a
    shard id, or require an explicit launch context. They do not import model
    runtimes or silently execute CPU fallbacks.
    """

    config = load_config(config_path)
    selected_stages = _normalize_stages(stages)
    _validate_shard_id(config, shard_id)
    if dry_run:
        return _json_safe(
            _base_result(
                config,
                config_path=config_path,
                stages=selected_stages,
                dry_run=True,
                allow_overwrite=allow_overwrite,
                shard_id=shard_id,
                launch_context=launch_context,
                mock_runtime=mock_runtime,
                stage_results={
                    stage: _dry_stage_result(config, stage=stage, shard_id=shard_id)
                    for stage in selected_stages
                },
            )
        )

    _validate_gpu_stage_context(
        config,
        stages=selected_stages,
        shard_id=shard_id,
        launch_context=launch_context,
        mock_runtime=mock_runtime,
    )

    stage_results: dict[str, Any] = {}
    for stage in selected_stages:
        handler = _STAGE_HANDLERS[stage]
        stage_results[stage] = handler(
            config,
            allow_overwrite=allow_overwrite,
            shard_id=shard_id,
            launch_context=launch_context,
        )

    status_payload = _status_payload(config)
    if "validate" in stage_results:
        status_payload = stage_results["validate"]
    if "finalize" in stage_results:
        status_payload = stage_results["finalize"]

    result = _base_result(
        config,
        config_path=config_path,
        stages=selected_stages,
        dry_run=False,
        allow_overwrite=allow_overwrite,
        shard_id=shard_id,
        launch_context=launch_context,
        mock_runtime=mock_runtime,
        stage_results=stage_results,
    )
    result.update(_launch_gate_fields(config, status_payload.get("status", status_payload)))
    return _json_safe(result)


def _normalize_stages(stages: Sequence[str] | str | None) -> list[str]:
    if stages is None:
        return list(STAGE_NAMES)
    if isinstance(stages, str):
        raw_stages = [stage.strip() for stage in stages.split(",")]
    else:
        raw_stages = [str(stage).strip() for stage in stages]
    selected = [stage for stage in raw_stages if stage]
    unknown = [stage for stage in selected if stage not in STAGE_NAMES]
    if unknown:
        raise ValueError(f"unknown stage(s): {', '.join(unknown)}")
    if not selected:
        raise ValueError("at least one stage is required")
    return selected


def _validate_shard_id(config: A32Config, shard_id: int | None) -> None:
    if shard_id is None:
        return
    if shard_id < 0 or shard_id >= config.sampling.num_shards:
        raise ValueError(
            f"shard_id must be in [0, {config.sampling.num_shards - 1}], got {shard_id}"
        )


def _validate_gpu_stage_context(
    config: A32Config,
    *,
    stages: Sequence[str],
    shard_id: int | None,
    launch_context: bool,
    mock_runtime: bool,
) -> None:
    for stage in stages:
        if stage in SHARDED_GPU_STAGES and shard_id is None and not launch_context:
            raise ValueError(f"{stage} requires --shard-id or launch context")
        if stage == "native_rollout" and not launch_context:
            raise ValueError("native_rollout requires controlled launch context")
        if stage in GPU_STAGES and not mock_runtime:
            raise ValueError(
                f"{stage} real GPU runtime is not wired in the generic runner; "
                "use --dry-run for planning or the dedicated runtime stage once implemented"
            )
    _validate_shard_id(config, shard_id)


def _base_result(
    config: A32Config,
    *,
    config_path: str | Path,
    stages: Sequence[str],
    dry_run: bool,
    allow_overwrite: bool,
    shard_id: int | None,
    launch_context: bool,
    mock_runtime: bool,
    stage_results: Mapping[str, Any],
) -> dict[str, Any]:
    roles = checkpoint_roles_from_config(config)
    return {
        "dry_run": dry_run,
        "config_path": str(config_path),
        "project_id": config.project_id,
        "phase_id": config.phase_id,
        "schema_version": config.schema_version,
        "run_id": config.run_id,
        "artifact_root": str(config.artifact_root),
        "checkpoint_roles": roles,
        "roles": _checkpoint_provenance(config),
        "template_contract": _template_contract(config),
        "sampling": {
            "max_prefix_states": config.sampling.max_prefix_states,
            "num_shards": config.sampling.num_shards,
            "seed": config.sampling.seed,
            "easy_sanity_max_fraction": config.sampling.easy_sanity_max_fraction,
        },
        "rollout": {
            "limit_images": config.rollout.limit_images,
            "decode_policy": config.rollout.decode_policy,
            "native_prompt_ordering": config.rollout.native_prompt_ordering,
            "constraint_policy": config.rollout.constraint_policy,
        },
        "fn_probe": {
            "max_fn_objects_per_checkpoint": config.fn_probe.max_fn_objects_per_checkpoint,
            "hint_policy_id": config.fn_probe.hint_policy_id,
            "strict_r95_axis_fraction": config.fn_probe.strict_r95_axis_fraction,
            "strict_r95_cap_bins": config.fn_probe.strict_r95_cap_bins,
            "broad_x1_radius": config.fn_probe.broad_x1_radius,
        },
        "stages": list(stages),
        "stage_results": dict(stage_results),
        "shard": {
            "shard_id": shard_id,
            "num_shards": config.sampling.num_shards,
            "selected_shards": _selected_shards(config, shard_id=shard_id),
            "launch_context": launch_context,
        },
        "mock_runtime": mock_runtime,
        "allow_overwrite": allow_overwrite,
    }


def _dry_stage_result(
    config: A32Config,
    *,
    stage: str,
    shard_id: int | None,
) -> dict[str, Any]:
    root = config.artifact_root
    selected_shards = _selected_shards(config, shard_id=shard_id)
    stage_writes: dict[str, list[str]] = {
        "data_root_audit": ["data_root_audit.json"],
        "prefix_state_index": [
            "resolved_config.yaml",
            "prefix_state_index.jsonl",
            "prefix_state_sampled_rows.jsonl",
            "prefix_state_index_summary.json",
            "sample_manifest.json",
        ],
        "validate": [],
        "paired_checkpoint_probe": [
            f"prefix_readout_shards/shard_{sid}.jsonl" for sid in selected_shards
        ]
        + ["prefix_state_shard_summaries.jsonl"],
        "prefix_merge": [
            "summary/prefix_readout_merged_rows.jsonl",
            "summary/prefix_readout_summary.json",
        ],
        "prefix_report": ["summary/report.md"],
        "prefix_gallery": ["gallery/index.md", "gallery/metadata.json"],
        "native_rollout": [
            f"rollout/{role}/{name}"
            for role in checkpoint_roles_from_config(config)
            for name in ("gt_vs_pred.jsonl", "pred_token_trace.jsonl", "summary.json")
        ],
        "rollout_phenotype": [
            "rollout/rollout_phenotype_rows.jsonl",
            "rollout/rollout_summary.json",
        ],
        "fn_case_index": [
            "fn_probe/fn_case_universe.jsonl",
            "fn_probe/fn_cases.jsonl",
            "fn_probe/fn_matching_summary.json",
        ],
        "fn_hint_probe": [
            f"fn_probe/fn_hint_shards/shard_{sid}_probe_rows.jsonl"
            for sid in selected_shards
        ],
        "fn_merge": [
            "fn_probe/fn_probe_rows.jsonl",
            "fn_probe/fn_candidate_scores.jsonl",
            "fn_probe/fn_slot_evidence.jsonl",
            "fn_probe/fn_bucket_summary.json",
            "fn_probe/fn_prefix_sensitivity.json",
            "fn_probe/fn_slot_rescue_summary.json",
        ],
        "fn_report": ["summary/report.md"],
        "fn_gallery": ["fn_probe/gallery/index.md", "fn_probe/gallery/metadata.json"],
        "finalize": [],
    }
    return {
        "stage": stage,
        "dry_run": True,
        "requires_gpu": stage in GPU_STAGES,
        "requires_shard": stage in SHARDED_GPU_STAGES,
        "selected_shards": selected_shards,
        "artifact_root": str(root),
        "would_write": [str(root / rel_path) for rel_path in stage_writes[stage]],
        "required_inputs": _dry_required_inputs(stage, config),
    }


def _dry_required_inputs(stage: str, config: A32Config) -> list[str]:
    root = config.artifact_root
    if stage == "validate":
        return [
            str(root / "data_root_audit.json"),
            str(root / "prefix_state_index.jsonl"),
            str(root / "prefix_state_sampled_rows.jsonl"),
            str(root / "prefix_state_index_summary.json"),
            str(root / "sample_manifest.json"),
        ]
    if stage == "paired_checkpoint_probe":
        return [str(root / "prefix_state_sampled_rows.jsonl")]
    if stage == "prefix_merge":
        return [str(root / "prefix_readout_shards")]
    if stage == "rollout_phenotype":
        return [
            str(root / "rollout" / role / "gt_vs_pred.jsonl")
            for role in checkpoint_roles_from_config(config)
        ]
    if stage in {"fn_hint_probe", "fn_merge", "fn_report", "fn_gallery"}:
        return [str(root / "fn_probe" / "fn_cases.jsonl")]
    return []


def _stage_data_root_audit(
    config: A32Config,
    *,
    allow_overwrite: bool,
    shard_id: int | None,
    launch_context: bool,
) -> dict[str, Any]:
    del shard_id, launch_context
    path = config.artifact_root / "data_root_audit.json"
    _ensure_can_write(path, allow_overwrite=allow_overwrite)
    payload = build_data_root_audit(config)
    _write_json(path, payload)
    return {"stage": "data_root_audit", "path": str(path), "status": payload["status"]}


def _stage_prefix_state_index(
    config: A32Config,
    *,
    allow_overwrite: bool,
    shard_id: int | None,
    launch_context: bool,
) -> dict[str, Any]:
    del shard_id, launch_context
    root = config.artifact_root
    paths = {
        "resolved_config": root / "resolved_config.yaml",
        "index": root / "prefix_state_index.jsonl",
        "sampled_rows": root / "prefix_state_sampled_rows.jsonl",
        "summary": root / "prefix_state_index_summary.json",
        "sample_manifest": root / "sample_manifest.json",
    }
    for path in paths.values():
        _ensure_can_write(path, allow_overwrite=allow_overwrite)

    rows, sampled_rows, summary, sample_manifest = build_prefix_state_index(config)
    summary = {
        **summary,
        "checkpoint_roles": checkpoint_roles_from_config(config),
        "template_contract": _template_contract(config),
        "artifact_root": str(root),
    }
    sample_manifest = {
        **sample_manifest,
        "checkpoint_roles": checkpoint_roles_from_config(config),
        "checkpoint_provenance": _checkpoint_provenance(config),
        "template_contract": _template_contract(config),
        "artifact_root": str(root),
    }
    _write_yaml(paths["resolved_config"], _config_to_dict(config))
    _write_jsonl(paths["index"], rows)
    _write_jsonl(paths["sampled_rows"], sampled_rows)
    _write_json(paths["summary"], summary)
    _write_json(paths["sample_manifest"], sample_manifest)
    return {
        "stage": "prefix_state_index",
        "index_rows": len(rows),
        "sampled_rows": len(sampled_rows),
        "paths": {name: str(path) for name, path in paths.items()},
    }


def _stage_validate(
    config: A32Config,
    *,
    allow_overwrite: bool,
    shard_id: int | None,
    launch_context: bool,
) -> dict[str, Any]:
    del allow_overwrite, shard_id, launch_context
    status = evaluate_status(config.artifact_root)
    result = {"stage": "validate", "status": status}
    result.update(_launch_gate_fields(config, status))
    return result


def _stage_paired_checkpoint_probe(
    config: A32Config,
    *,
    allow_overwrite: bool,
    shard_id: int | None,
    launch_context: bool,
) -> dict[str, Any]:
    del launch_context
    sampled_rows = _read_jsonl(config.artifact_root / "prefix_state_sampled_rows.jsonl")
    selected = _selected_shards(config, shard_id=shard_id)
    _ensure_paths_can_write(
        [
            *[
                config.artifact_root
                / "prefix_readout_shards"
                / f"shard_{selected_shard_id}.jsonl"
                for selected_shard_id in selected
            ],
            config.artifact_root / "prefix_state_shard_summaries.jsonl",
        ],
        allow_overwrite=allow_overwrite,
    )
    written: list[str] = []
    summaries: list[dict[str, Any]] = []
    for selected_shard_id in selected:
        shard_rows = [
            row for row in sampled_rows if int(row.get("shard_id", -1)) == selected_shard_id
        ]
        shard_path = (
            config.artifact_root
            / "prefix_readout_shards"
            / f"shard_{selected_shard_id}.jsonl"
        )
        manifest_row = {
            "row_type": "shard_manifest",
            **build_shard_manifest_row(
                config,
                shard_id=selected_shard_id,
                prefix_state_rows=shard_rows,
            ),
        }
        boundary_summaries = _mock_boundary_summaries(config, shard_rows)
        readout_rows = build_mocked_paired_readout_rows(
            config,
            prefix_state_rows=shard_rows,
            boundary_summaries_by_role=boundary_summaries,
        )
        _write_jsonl(shard_path, [manifest_row, *readout_rows])
        written.append(str(shard_path))
        summaries.append(
            {
                "project_id": PROJECT_ID,
                "phase_id": PHASE_ID,
                "schema_version": SCHEMA_VERSION,
                "run_id": RUN_ID,
                "shard_id": selected_shard_id,
                "checkpoint_roles": checkpoint_roles_from_config(config),
                "prefix_source_policy": "canonical_sorted_teacher_prefix_readout",
                "prefix_state_count": len(shard_rows),
                "readout_row_count": len(readout_rows),
                "mocked_runtime": True,
            }
        )
    _upsert_shard_summaries(
        config.artifact_root,
        summaries,
        allow_overwrite=allow_overwrite,
    )
    return {
        "stage": "paired_checkpoint_probe",
        "selected_shards": selected,
        "written": written,
        "mocked_runtime": True,
    }


def _stage_prefix_merge(
    config: A32Config,
    *,
    allow_overwrite: bool,
    shard_id: int | None,
    launch_context: bool,
) -> dict[str, Any]:
    del shard_id, launch_context
    sidecar_summary_path = _materialize_prefix_shard_summary_jsonl(
        config,
        allow_overwrite=allow_overwrite,
    )
    rows = _read_prefix_readout_rows(config)
    merged_rows = merge_prefix_readout_rows(rows)
    merged_path = config.artifact_root / "summary" / "prefix_readout_merged_rows.jsonl"
    summary_path = config.artifact_root / "summary" / "prefix_readout_summary.json"
    _ensure_can_write(merged_path, allow_overwrite=allow_overwrite)
    _ensure_can_write(summary_path, allow_overwrite=allow_overwrite)
    write_report_jsonl(merged_path, merged_rows)
    _write_json(
        summary_path,
        _prefix_readout_summary(
            merged_rows,
            checkpoint_roles=checkpoint_roles_from_config(config),
        ),
    )
    return {
        "stage": "prefix_merge",
        "readout_rows": len(rows),
        "merged_rows": len(merged_rows),
        "merged_path": str(merged_path),
        "summary_path": str(summary_path),
        "shard_summary_path": (
            None if sidecar_summary_path is None else str(sidecar_summary_path)
        ),
    }


def _stage_prefix_report(
    config: A32Config,
    *,
    allow_overwrite: bool,
    shard_id: int | None,
    launch_context: bool,
) -> dict[str, Any]:
    del shard_id, launch_context
    report_path = config.artifact_root / "summary" / "report.md"
    _ensure_can_write(report_path, allow_overwrite=allow_overwrite)
    merged_rows = _read_jsonl(config.artifact_root / "summary" / "prefix_readout_merged_rows.jsonl")
    report = build_report_markdown(
        evidence_labels={
            "prefix_readout": PREFIX_EVIDENCE_LABEL,
            "native_rollout": ROLLOUT_EVIDENCE_LABEL,
            "fn_probe": FN_EVIDENCE_LABEL,
        },
        checkpoint_provenance=_checkpoint_provenance(config),
        data_roots=_data_roots_for_report(config),
        template_contract=_template_contract(config),
        prefix_readout_rows=merged_rows,
        rollout_summary=_read_json_or_empty(
            config.artifact_root / "rollout" / "rollout_summary.json"
        ),
        fn_universe_counts=_read_json_or_empty(
            config.artifact_root / "fn_probe" / "fn_matching_summary.json"
        ),
        fn_bucket_summary=_read_json_or_empty(
            config.artifact_root / "fn_probe" / "fn_bucket_summary.json"
        ),
        prefix_sensitivity=_read_json_or_empty(
            config.artifact_root / "fn_probe" / "fn_prefix_sensitivity.json"
        ),
    )
    write_report(report_path, report)
    return {"stage": "prefix_report", "path": str(report_path)}


def _stage_prefix_gallery(
    config: A32Config,
    *,
    allow_overwrite: bool,
    shard_id: int | None,
    launch_context: bool,
) -> dict[str, Any]:
    del shard_id, launch_context
    _ensure_paths_can_write(
        _placeholder_gallery_paths(
            config.artifact_root / "gallery",
            case_id="native-rollout-placeholder",
        ),
        allow_overwrite=allow_overwrite,
    )
    metadata = build_gallery(
        config.artifact_root / "gallery",
        [_placeholder_gallery_case(case_id="native-rollout-placeholder")],
        title="A3.2 Native Rollout Gallery",
    )
    return {"stage": "prefix_gallery", "items": len(metadata)}


def _stage_native_rollout(
    config: A32Config,
    *,
    allow_overwrite: bool,
    shard_id: int | None,
    launch_context: bool,
) -> dict[str, Any]:
    del shard_id
    if not launch_context:
        raise ValueError("native_rollout requires controlled launch context")
    native_paths: list[Path] = []
    for role in checkpoint_roles_from_config(config):
        rollout_dir = config.artifact_root / "rollout" / role
        native_paths.extend(
            [
                rollout_dir / "gt_vs_pred.jsonl",
                rollout_dir / "pred_token_trace.jsonl",
                rollout_dir / "summary.json",
            ]
        )
    _ensure_paths_can_write(native_paths, allow_overwrite=allow_overwrite)
    written: list[str] = []
    for role in checkpoint_roles_from_config(config):
        rollout_dir = config.artifact_root / "rollout" / role
        paths = {
            "gt_vs_pred": rollout_dir / "gt_vs_pred.jsonl",
            "pred_token_trace": rollout_dir / "pred_token_trace.jsonl",
            "summary": rollout_dir / "summary.json",
        }
        _write_jsonl(paths["gt_vs_pred"], [])
        _write_jsonl(paths["pred_token_trace"], [])
        _write_json(
            paths["summary"],
            {
                "project_id": PROJECT_ID,
                "phase_id": PHASE_ID,
                "schema_version": SCHEMA_VERSION,
                "run_id": RUN_ID,
                "checkpoint_role": role,
                "checkpoint_roles": checkpoint_roles_from_config(config),
                "decode_policy": config.rollout.decode_policy,
                "constraint_policy": config.rollout.constraint_policy,
                "native_prompt_ordering": True,
                "runtime_status": "planned_or_external_rollout_required",
            },
        )
        written.extend(str(path) for path in paths.values())
    return {
        "stage": "native_rollout",
        "launch_context": launch_context,
        "written": written,
        "gpu_runtime_invoked": False,
    }


def _stage_rollout_phenotype(
    config: A32Config,
    *,
    allow_overwrite: bool,
    shard_id: int | None,
    launch_context: bool,
) -> dict[str, Any]:
    del shard_id, launch_context
    rollout_rows = _read_rollout_input_rows(config)
    rollout_dir = config.artifact_root / "rollout"
    rows_path = rollout_dir / "rollout_phenotype_rows.jsonl"
    summary_path = rollout_dir / "rollout_summary.json"
    _ensure_paths_can_write(
        (rows_path, summary_path),
        allow_overwrite=allow_overwrite,
    )
    if rollout_rows:
        paths = materialize_rollout_phenotype(
            config.artifact_root,
            _read_jsonl(config.val_jsonl),
            rollout_rows,
        )
        return {
            "stage": "rollout_phenotype",
            "source_rows": len(rollout_rows),
            "paths": paths,
        }

    _write_jsonl(rows_path, [])
    _write_json(
        summary_path,
        {
            "project_id": PROJECT_ID,
            "phase_id": PHASE_ID,
            "schema_version": SCHEMA_VERSION,
            "run_id": RUN_ID,
            "status": "rollout_inputs_missing",
            "required_artifacts": _dry_required_inputs("rollout_phenotype", config),
        },
    )
    return {
        "stage": "rollout_phenotype",
        "source_rows": 0,
        "paths": {"rows_path": str(rows_path), "summary_path": str(summary_path)},
    }


def _stage_fn_case_index(
    config: A32Config,
    *,
    allow_overwrite: bool,
    shard_id: int | None,
    launch_context: bool,
) -> dict[str, Any]:
    del shard_id, launch_context
    fn_dir = config.artifact_root / "fn_probe"
    _ensure_paths_can_write(
        (
            fn_dir / "fn_case_universe.jsonl",
            fn_dir / "fn_cases.jsonl",
            fn_dir / "fn_matching_summary.json",
        ),
        allow_overwrite=allow_overwrite,
    )
    roles = checkpoint_roles_from_config(config)
    rollout_rows_by_role = _rollout_rows_by_role(config)
    if not all(rollout_rows_by_role.get(role) for role in roles):
        universe_rows: list[dict[str, Any]] = []
        fn_cases: list[dict[str, Any]] = []
        summary = {
            "project_id": PROJECT_ID,
            "phase_id": PHASE_ID,
            "schema_version": SCHEMA_VERSION,
            "run_id": RUN_ID,
            "checkpoint_roles": roles,
            "universe_rows": 0,
            "fn_cases": 0,
            "status": "rollout_inputs_missing",
            "required_artifacts": _dry_required_inputs("fn_case_index", config),
        }
        _write_jsonl(fn_dir / "fn_case_universe.jsonl", universe_rows)
        _write_jsonl(fn_dir / "fn_cases.jsonl", fn_cases)
        _write_json(fn_dir / "fn_matching_summary.json", summary)
        return {
            "stage": "fn_case_index",
            "universe_rows": 0,
            "fn_cases": 0,
            "status": "rollout_inputs_missing",
        }

    image_packages = _build_fn_image_packages(config, rollout_rows_by_role)
    universe_all = [
        row for package in image_packages for row in package["universe_rows"]
    ]
    selected_keys, sampling_reasons = _sample_fn_probe_keys(
        universe_all,
        max_per_checkpoint=int(config.fn_probe.max_fn_objects_per_checkpoint),
    )
    universe_rows: list[dict[str, Any]] = []
    fn_cases: list[dict[str, Any]] = []
    for package in image_packages:
        image_universe = build_fn_case_universe(
            package["gt_rows"],
            match_ledgers_by_role=package["match_ledgers_by_role"],
            split=str(package["split"]),
            image_id=package["image_id"],
            sampled_gt_object_keys=selected_keys,
            sampling_reasons=sampling_reasons,
        )
        universe_rows.extend(image_universe)
        fn_cases.extend(
            build_replayable_fn_cases(
                image_universe,
                match_ledgers_by_role=package["match_ledgers_by_role"],
                contexts_by_role=package["contexts_by_role"],
            )
        )

    summary = {
        "project_id": PROJECT_ID,
        "phase_id": PHASE_ID,
        "schema_version": SCHEMA_VERSION,
        "run_id": RUN_ID,
        "checkpoint_roles": roles,
        "match_policy_id": MATCH_POLICY_ID,
        "universe_rows": len(universe_rows),
        "fn_cases": len(fn_cases),
        "fn_membership_counts": _count_by_key(universe_rows, "fn_membership"),
        "sampled_for_probe": sum(bool(row.get("sampled_for_probe")) for row in universe_rows),
        "source": "native_rollout_gt_vs_pred",
    }
    _write_jsonl(fn_dir / "fn_case_universe.jsonl", universe_rows)
    _write_jsonl(fn_dir / "fn_cases.jsonl", fn_cases)
    _write_json(fn_dir / "fn_matching_summary.json", summary)
    return {
        "stage": "fn_case_index",
        "universe_rows": len(universe_rows),
        "fn_cases": len(fn_cases),
    }


def _stage_fn_hint_probe(
    config: A32Config,
    *,
    allow_overwrite: bool,
    shard_id: int | None,
    launch_context: bool,
) -> dict[str, Any]:
    del launch_context
    selected = _selected_shards(config, shard_id=shard_id)
    fn_cases = _read_jsonl(config.artifact_root / "fn_probe" / "fn_cases.jsonl")
    shard_dir = config.artifact_root / "fn_probe" / "fn_hint_shards"
    _ensure_paths_can_write(
        [
            path
            for selected_shard_id in selected
            for path in (
                shard_dir / f"shard_{selected_shard_id}_probe_rows.jsonl",
                shard_dir / f"shard_{selected_shard_id}_candidate_scores.jsonl",
                shard_dir / f"shard_{selected_shard_id}_slot_evidence.jsonl",
            )
        ],
        allow_overwrite=allow_overwrite,
    )
    written: list[str] = []
    for selected_shard_id in selected:
        shard_cases = [
            case
            for index, case in enumerate(fn_cases)
            if index % config.sampling.num_shards == selected_shard_id
        ]
        if not shard_cases and fn_cases:
            shard_cases = []
        probe_specs = [
            {
                "probe_id": f"{case['fn_case_id']}:desc_x1",
                "fn_case_id": case["fn_case_id"],
                "hint_level": "desc_x1",
                "prefix_condition": "teacher_sorted_prefix",
                "valid_parse": True,
                "prefix_objects": [
                    {
                        "source": "teacher",
                        "gt_idx": 0,
                        "pred_idx": None,
                        "desc": case.get("fn_desc", "person"),
                        "bbox": case.get("fn_bbox", [50, 10, 100, 90]),
                        "order_idx": 0,
                    }
                ],
            }
            for case in shard_cases
        ]
        candidate_scores = build_fn_candidate_score_rows(
            [
                {
                    "probe_id": spec["probe_id"],
                    "fn_case_id": spec["fn_case_id"],
                    "candidate_id": "target",
                    "candidate_gt_idx": 0,
                    "desc": "person",
                    "role": "residual_same_desc",
                    "score": 1.0,
                }
                for spec in probe_specs
            ]
        )
        slot_rows = build_fn_slot_evidence_rows(
            [
                {
                    "probe_id": spec["probe_id"],
                    "fn_case_id": spec["fn_case_id"],
                    "slot": "x1",
                    "axis_len": 1024,
                    "gt_idx": 0,
                    "gt_value": 50,
                    "peak_value": 50,
                    "score": 1.0,
                }
                for spec in probe_specs
            ],
            broad_x1_radius=config.fn_probe.broad_x1_radius,
        )
        probe_rows = build_fn_probe_rows(
            shard_cases,
            probe_specs=probe_specs,
            candidate_score_rows=candidate_scores,
            slot_evidence_rows=slot_rows,
            hint_policy_id=config.fn_probe.hint_policy_id,
        )
        paths = {
            "probe_rows": shard_dir / f"shard_{selected_shard_id}_probe_rows.jsonl",
            "candidate_scores": shard_dir
            / f"shard_{selected_shard_id}_candidate_scores.jsonl",
            "slot_evidence": shard_dir / f"shard_{selected_shard_id}_slot_evidence.jsonl",
        }
        _write_jsonl(paths["probe_rows"], probe_rows)
        _write_jsonl(paths["candidate_scores"], candidate_scores)
        _write_jsonl(paths["slot_evidence"], slot_rows)
        written.extend(str(path) for path in paths.values())
    return {
        "stage": "fn_hint_probe",
        "selected_shards": selected,
        "written": written,
        "mocked_runtime": True,
    }


def _stage_fn_merge(
    config: A32Config,
    *,
    allow_overwrite: bool,
    shard_id: int | None,
    launch_context: bool,
) -> dict[str, Any]:
    del shard_id, launch_context
    fn_dir = config.artifact_root / "fn_probe"
    _ensure_paths_can_write(
        (
            fn_dir / "fn_probe_rows.jsonl",
            fn_dir / "fn_candidate_scores.jsonl",
            fn_dir / "fn_slot_evidence.jsonl",
            fn_dir / "fn_bucket_summary.json",
            fn_dir / "fn_prefix_sensitivity.json",
            fn_dir / "fn_slot_rescue_summary.json",
        ),
        allow_overwrite=allow_overwrite,
    )
    probe_rows = _read_sharded_rows(fn_dir / "fn_hint_shards", "*_probe_rows.jsonl")
    candidate_rows = _read_sharded_rows(fn_dir / "fn_hint_shards", "*_candidate_scores.jsonl")
    slot_rows = _read_sharded_rows(fn_dir / "fn_hint_shards", "*_slot_evidence.jsonl")
    if not probe_rows:
        probe_rows = _read_jsonl_or_empty(fn_dir / "fn_probe_rows.jsonl")
    if not candidate_rows:
        candidate_rows = _read_jsonl_or_empty(fn_dir / "fn_candidate_scores.jsonl")
    if not slot_rows:
        slot_rows = _read_jsonl_or_empty(fn_dir / "fn_slot_evidence.jsonl")
    runtime_provenance = _fn_runtime_provenance_from_rows(probe_rows)
    _write_jsonl(fn_dir / "fn_probe_rows.jsonl", probe_rows)
    _write_jsonl(fn_dir / "fn_candidate_scores.jsonl", candidate_rows)
    _write_jsonl(fn_dir / "fn_slot_evidence.jsonl", slot_rows)
    _write_json(
        fn_dir / "fn_bucket_summary.json",
        {**_fn_bucket_summary(probe_rows), **runtime_provenance},
    )
    _write_json(
        fn_dir / "fn_prefix_sensitivity.json",
        {**_fn_prefix_sensitivity(probe_rows), **runtime_provenance},
    )
    _write_json(
        fn_dir / "fn_slot_rescue_summary.json",
        {**_fn_slot_rescue_summary(probe_rows), **runtime_provenance},
    )
    return {
        "stage": "fn_merge",
        "probe_rows": len(probe_rows),
        "candidate_score_rows": len(candidate_rows),
        "slot_evidence_rows": len(slot_rows),
    }


def _stage_fn_report(
    config: A32Config,
    *,
    allow_overwrite: bool,
    shard_id: int | None,
    launch_context: bool,
) -> dict[str, Any]:
    return _stage_prefix_report(
        config,
        allow_overwrite=allow_overwrite,
        shard_id=shard_id,
        launch_context=launch_context,
    )


def _stage_fn_gallery(
    config: A32Config,
    *,
    allow_overwrite: bool,
    shard_id: int | None,
    launch_context: bool,
) -> dict[str, Any]:
    del shard_id, launch_context
    _ensure_paths_can_write(
        _placeholder_gallery_paths(
            config.artifact_root / "fn_probe" / "gallery",
            case_id="fn-probe-placeholder",
        ),
        allow_overwrite=allow_overwrite,
    )
    metadata = build_gallery(
        config.artifact_root / "fn_probe" / "gallery",
        [_placeholder_gallery_case(case_id="fn-probe-placeholder")],
        title="A3.2 FN Probe Gallery",
    )
    return {"stage": "fn_gallery", "items": len(metadata)}


def _stage_finalize(
    config: A32Config,
    *,
    allow_overwrite: bool,
    shard_id: int | None,
    launch_context: bool,
) -> dict[str, Any]:
    del allow_overwrite, shard_id, launch_context
    status = evaluate_status(config.artifact_root)
    result = {
        "stage": "finalize",
        "status": status,
        "semantic_report": {
            "project_id": PROJECT_ID,
            "phase_id": PHASE_ID,
            "schema_version": SCHEMA_VERSION,
            "run_id": RUN_ID,
            "stage_status": status["status"],
            "final_artifacts_present": status["final_artifacts_present"],
            "failed_gates": status["failed_gates"],
        },
    }
    result.update(_launch_gate_fields(config, status))
    return result


def _status_payload(config: A32Config) -> dict[str, Any]:
    status = evaluate_status(config.artifact_root)
    payload = {"stage": "status", "status": status}
    payload.update(_launch_gate_fields(config, status))
    return payload


def _launch_gate_fields(config: A32Config, status: Mapping[str, Any]) -> dict[str, Any]:
    stage_status = str(status.get("status", "incomplete"))
    launch_eligible = stage_status == "index_ready_pending_gpu"
    failed_gates = [] if launch_eligible else list(status.get("failed_gates", ()))
    return {
        "stage_status": stage_status,
        "launch_eligible": launch_eligible,
        "failed_launch_gates": failed_gates,
        "expected_shards": config.sampling.num_shards,
        "planned_prefix_state_rows": _planned_prefix_state_rows(config.artifact_root),
    }


def _planned_prefix_state_rows(root: Path) -> int:
    manifest = _read_json_or_empty(root / "sample_manifest.json")
    if "selected_rows" in manifest:
        return int(manifest["selected_rows"])
    sampled_path = root / "prefix_state_sampled_rows.jsonl"
    if sampled_path.is_file():
        return len(_read_jsonl(sampled_path))
    return 0


def _selected_shards(config: A32Config, *, shard_id: int | None) -> list[int]:
    if shard_id is not None:
        return [int(shard_id)]
    return list(range(config.sampling.num_shards))


def _checkpoint_provenance(config: A32Config) -> dict[str, dict[str, str]]:
    return {
        role: {
            "checkpoint_path": str(checkpoint.checkpoint_path),
            "training_ordering": checkpoint.training_ordering,
            "readout_prompt_ordering": checkpoint.readout_prompt_ordering,
            "objective_policy": checkpoint.objective_policy,
            "comparison_group": checkpoint.comparison_group,
            "template_contract_id": checkpoint.template_contract_id,
        }
        for role, checkpoint in config.checkpoints.items()
    }


def _template_contract(config: A32Config) -> dict[str, str]:
    return {
        "detection_sequence_format": config.template_contract.detection_sequence_format,
        "coordinate_surface": config.template_contract.coordinate_surface,
        "bbox_format": config.template_contract.bbox_format,
        "row_separator": config.template_contract.row_separator,
    }


def _data_roots_for_report(config: A32Config) -> dict[str, Any]:
    audit = _read_json_or_empty(config.artifact_root / "data_root_audit.json")
    if audit:
        return {
            "train_jsonl": audit.get("actual_train_jsonl"),
            "val_jsonl": audit.get("actual_val_jsonl"),
            "image_root": audit.get("image_root"),
            "evidence_scope": audit.get("evidence_scope"),
            "row_counts": audit.get("row_counts"),
        }
    return {
        "train_jsonl": str(config.train_jsonl),
        "val_jsonl": str(config.val_jsonl),
        "image_root": str(config.image_root),
        "evidence_scope": "len12000-jsonl-local-mechanism-probe",
    }


def _mock_boundary_summaries(
    config: A32Config,
    shard_rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    summaries: dict[str, dict[str, dict[str, Any]]] = {}
    roles = checkpoint_roles_from_config(config)
    for role_index, role in enumerate(roles):
        role_summaries: dict[str, dict[str, Any]] = {}
        for row in shard_rows:
            prefix_state_id = str(row["prefix_state_id"])
            candidate_descs = list(row.get("candidate_descs", ()))
            base_margin = 0.05 + 0.05 * role_index
            role_summaries[prefix_state_id] = {
                "winner_desc": candidate_descs[0] if candidate_descs else "eos",
                "winner_roles": ["residual_same_desc"],
                "boundary_winner_class": "residual_same_desc_favored",
                "residual_vs_eos_margin": base_margin,
                "residual_vs_winner_margin": base_margin / 2.0,
                "low_margin_flag": False,
                "candidate_descs_with_roles": list(row.get("candidate_descs_with_roles", ())),
                "strict_r95_x1_hit_rate": float(role_index),
                "boundary_residual_favored_rate": 1.0,
            }
        summaries[role] = role_summaries
    return summaries


def _read_prefix_readout_rows(config: A32Config) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    shard_root = config.artifact_root / "prefix_readout_shards"
    for shard_path in sorted(shard_root.glob("shard_*.jsonl")):
        for row in _read_jsonl(shard_path):
            if "checkpoint_role" in row:
                rows.append(row)
    return rows


def _prefix_readout_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    checkpoint_roles: Sequence[str] | None = None,
) -> dict[str, Any]:
    roles = (
        [str(role) for role in checkpoint_roles]
        if checkpoint_roles is not None
        else _checkpoint_roles_from_rows(rows)
    )
    summary: dict[str, Any] = {
        "project_id": PROJECT_ID,
        "phase_id": PHASE_ID,
        "schema_version": SCHEMA_VERSION,
        "run_id": RUN_ID,
        "checkpoint_roles": roles,
        "merged_prefix_state_count": len(rows),
    }
    by_delta_role: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        delta_role = row.get("delta_role")
        values = row.get("delta_metric_values")
        if isinstance(delta_role, str) and isinstance(values, Mapping):
            metric_values = by_delta_role.setdefault(delta_role, {})
            for metric, value in values.items():
                try:
                    metric_values.setdefault(str(metric), []).append(float(value))
                except (TypeError, ValueError):
                    continue
    summary["delta_metric_means_by_role"] = {
        delta_role: {
            metric: (sum(values) / len(values) if values else 0.0)
            for metric, values in sorted(metrics.items())
        }
        for delta_role, metrics in sorted(by_delta_role.items())
    }
    for _, delta_name in DELTA_METRICS:
        values = [float(row[delta_name]) for row in rows if delta_name in row]
        summary[f"mean_{delta_name}"] = 0.0 if not values else sum(values) / len(values)
    return summary


def _checkpoint_roles_from_rows(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    roles: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in ("left_role", "right_role", "checkpoint_role"):
            value = row.get(key)
            if isinstance(value, str) and value and value not in seen:
                seen.add(value)
                roles.append(value)
    return roles


def _materialize_prefix_shard_summary_jsonl(
    config: A32Config,
    *,
    allow_overwrite: bool,
) -> Path | None:
    sidecar_root = config.artifact_root / "prefix_state_shard_summaries"
    if not sidecar_root.is_dir():
        return None
    sidecars = sorted(sidecar_root.glob("shard_*.json"))
    if not sidecars:
        return None
    rows: list[dict[str, Any]] = []
    seen_shards: set[int] = set()
    for path in sidecars:
        row = _read_json_or_empty(path)
        if not row:
            continue
        if str(row.get("runtime_kind")) != REAL_PREFIX_RUNTIME_KIND:
            raise ValueError(f"prefix shard summary is not real runtime output: {path}")
        shard_id = int(row["shard_id"])
        if shard_id in seen_shards:
            raise ValueError(f"duplicate prefix shard summary for shard {shard_id}")
        seen_shards.add(shard_id)
        rows.append(row)
    if not rows:
        return None
    rows.sort(key=lambda row: int(row["shard_id"]))
    path = config.artifact_root / "prefix_state_shard_summaries.jsonl"
    _ensure_can_write(path, allow_overwrite=allow_overwrite)
    _write_jsonl(path, rows)
    return path


def _upsert_shard_summaries(
    root: Path,
    summaries: Sequence[Mapping[str, Any]],
    *,
    allow_overwrite: bool,
) -> None:
    path = root / "prefix_state_shard_summaries.jsonl"
    _ensure_can_write(path, allow_overwrite=allow_overwrite)
    existing: dict[int, dict[str, Any]] = {}
    if path.is_file():
        for row in _read_jsonl(path):
            existing[int(row["shard_id"])] = row
    for summary in summaries:
        existing[int(summary["shard_id"])] = dict(summary)
    _write_jsonl(path, [existing[shard_id] for shard_id in sorted(existing)])


def _read_rollout_input_rows(config: A32Config) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for role in checkpoint_roles_from_config(config):
        path = config.artifact_root / "rollout" / role / "gt_vs_pred.jsonl"
        if path.is_file():
            rows.extend(_read_jsonl(path))
    return rows


def _rollout_rows_by_role(config: A32Config) -> dict[str, list[dict[str, Any]]]:
    rows_by_role: dict[str, list[dict[str, Any]]] = {}
    for role in checkpoint_roles_from_config(config):
        path = config.artifact_root / "rollout" / role / "gt_vs_pred.jsonl"
        rows_by_role[role] = _read_jsonl(path) if path.is_file() else []
    return rows_by_role


def _build_fn_image_packages(
    config: A32Config,
    rollout_rows_by_role: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    roles = checkpoint_roles_from_config(config)
    indexed_by_role = {
        role: {
            _rollout_image_key(row, fallback_index=index): row
            for index, row in enumerate(rollout_rows_by_role.get(role, ()))
        }
        for role in roles
    }
    key_sets = [set(indexed_by_role[role]) for role in roles]
    common_keys = set.intersection(*key_sets) if key_sets else set()
    packages: list[dict[str, Any]] = []
    val_jsonl_sha = _file_sha256(config.val_jsonl)
    for key in sorted(common_keys):
        role_rows = {role: indexed_by_role[role][key] for role in roles}
        reference = role_rows[roles[0]]
        gt_rows = _gt_rows_from_rollout(reference)
        if not gt_rows:
            continue
        match_ledgers_by_role = {
            role: match_fn_cases_for_image(
                gt_rows,
                _pred_rows_from_rollout(row),
                split=str(reference.get("split", "val")),
                image_id=_rollout_image_id(reference, fallback=key),
            )
            for role, row in role_rows.items()
        }
        contexts_by_role = {
            role: _fn_context_from_rollout(
                config,
                row,
                role=role,
                jsonl_sha256=val_jsonl_sha,
            )
            for role, row in role_rows.items()
        }
        packages.append(
            {
                "split": str(reference.get("split", "val")),
                "image_id": _rollout_image_id(reference, fallback=key),
                "gt_rows": gt_rows,
                "match_ledgers_by_role": match_ledgers_by_role,
                "contexts_by_role": contexts_by_role,
                "universe_rows": build_fn_case_universe(
                    gt_rows,
                    match_ledgers_by_role=match_ledgers_by_role,
                    split=str(reference.get("split", "val")),
                    image_id=_rollout_image_id(reference, fallback=key),
                ),
            }
        )
    return packages


def _gt_rows_from_rollout(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    gt_raw = row.get("gt", ())
    if not isinstance(gt_raw, Sequence) or isinstance(gt_raw, (str, bytes)):
        return []
    object_count = len(gt_raw)
    desc_counts: dict[str, int] = {}
    for gt in gt_raw:
        if isinstance(gt, Mapping):
            desc = str(gt.get("desc") or gt.get("label") or gt.get("category") or "")
            desc_counts[desc] = desc_counts.get(desc, 0) + 1
    rows: list[dict[str, Any]] = []
    for index, gt in enumerate(gt_raw):
        if not isinstance(gt, Mapping):
            continue
        desc = str(gt.get("desc") or gt.get("label") or gt.get("category") or "")
        out = dict(gt)
        out.setdefault("gt_idx", index)
        out.setdefault("desc", desc)
        out.setdefault("gt_sorted_rank", index + 1)
        out.setdefault("same_desc_gt_count", desc_counts.get(desc, 0))
        out.setdefault("object_count", object_count)
        rows.append(out)
    return rows


def _pred_rows_from_rollout(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    pred_raw = row.get("pred", ())
    if not isinstance(pred_raw, Sequence) or isinstance(pred_raw, (str, bytes)):
        return []
    rows: list[dict[str, Any]] = []
    for index, pred in enumerate(pred_raw):
        if not isinstance(pred, Mapping):
            continue
        out = dict(pred)
        out.setdefault("pred_idx", index)
        if "desc" not in out:
            for key in ("label", "category", "description"):
                if key in out:
                    out["desc"] = out[key]
                    break
        rows.append(out)
    return rows


def _fn_context_from_rollout(
    config: A32Config,
    row: Mapping[str, Any],
    *,
    role: str,
    jsonl_sha256: str,
) -> dict[str, Any]:
    image_ref = str(row.get("image") or row.get("image_path") or "")
    image_path = Path(image_ref)
    if image_ref and not image_path.is_absolute():
        image_path = config.image_root / image_ref
    return {
        "split": str(row.get("split", "val")),
        "image_id": _rollout_image_id(row, fallback=str(row.get("source_line_idx", ""))),
        "source_line_idx": int(row.get("source_line_idx", row.get("line_idx", 0))),
        "image_path": str(image_path) if image_ref else "",
        "width": int(row.get("width", 0)),
        "height": int(row.get("height", 0)),
        "coord_mode": str(row.get("coord_mode") or "unknown"),
        "bbox_surface": str(row.get("bbox_surface") or "xyxy"),
        "data_root": str(config.val_jsonl.parent),
        "jsonl_sha256": jsonl_sha256,
        "checkpoint_fingerprint": str(row.get("checkpoint_fingerprint", "")),
        "decode_policy": str(row.get("decode_policy", config.rollout.decode_policy)),
        "template_contract": row.get("template_contract") or _template_contract(config),
        "invalid_pred_count": len(row.get("errors", ()) or ()),
        "checkpoint_role": role,
    }


def _rollout_image_key(row: Mapping[str, Any], *, fallback_index: int) -> str:
    source_line_idx = row.get("source_line_idx")
    if source_line_idx is not None:
        return f"line:{int(source_line_idx)}"
    image_id = row.get("image_id")
    if image_id is not None:
        return f"image_id:{image_id}"
    image = row.get("image") or row.get("image_path")
    if image is not None:
        return f"image:{image}"
    return f"fallback:{fallback_index}"


def _rollout_image_id(row: Mapping[str, Any], *, fallback: Any) -> Any:
    if row.get("image_id") is not None:
        return row.get("image_id")
    if row.get("source_line_idx") is not None:
        return row.get("source_line_idx")
    return fallback


def _sample_fn_probe_keys(
    universe_rows: Sequence[Mapping[str, Any]],
    *,
    max_per_checkpoint: int,
) -> tuple[set[str], dict[str, str]]:
    roles = _fn_roles_from_universe(universe_rows)
    selected: set[str] = set()
    counts = {role: 0 for role in roles}
    reasons: dict[str, str] = {}
    for row in sorted(universe_rows, key=lambda item: str(item["gt_object_key"])):
        key = str(row["gt_object_key"])
        selected_here = False
        for role in roles:
            if bool(row.get(f"is_fn_{role}")) and counts[role] < max_per_checkpoint:
                counts[role] += 1
                selected_here = True
        if selected_here:
            selected.add(key)
            reasons[key] = str(row.get("fn_membership", "selected_for_probe"))
    return selected, reasons


def _fn_roles_from_universe(universe_rows: Sequence[Mapping[str, Any]]) -> list[str]:
    roles: list[str] = []
    seen: set[str] = set()
    for row in universe_rows:
        for key in row:
            if not str(key).startswith("is_fn_"):
                continue
            role = str(key)[len("is_fn_") :]
            if role and role not in seen:
                seen.add(role)
                roles.append(role)
    return roles


def _file_sha256(path: Path) -> str:
    digest = __import__("hashlib").sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _count_by_key(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, "<missing>"))
        counts[value] = counts.get(value, 0) + 1
    return counts


def _mock_fn_case(
    config: A32Config,
    *,
    checkpoint_role: str,
    seed_row: Mapping[str, Any],
    fn_case_id: str,
) -> dict[str, Any]:
    image_path = str(seed_row.get("image_path") or config.image_root / "missing.jpg")
    return {
        "project_id": PROJECT_ID,
        "phase_id": PHASE_ID,
        "schema_version": SCHEMA_VERSION,
        "run_id": RUN_ID,
        "fn_case_id": fn_case_id,
        "gt_object_key": "val:mock-image:0",
        "checkpoint_role": checkpoint_role,
        "checkpoint_roles": checkpoint_roles_from_config(config),
        "split": str(seed_row.get("split", "val")),
        "image_id": str(seed_row.get("image_id", "mock-image")),
        "source_line_idx": int(seed_row.get("source_line_idx", 0)),
        "image_path": image_path,
        "width": 1024,
        "height": 768,
        "coord_mode": "coord_token",
        "bbox_surface": "xyxy",
        "data_root": str(config.val_jsonl.parent),
        "jsonl_sha256": "mocked-jsonl-sha256",
        "checkpoint_fingerprint": str(config.checkpoints[checkpoint_role].checkpoint_path),
        "decode_policy": config.rollout.decode_policy,
        "template_contract": _template_contract(config),
        "fn_gt_idx": 0,
        "fn_desc": "person",
        "fn_bbox": [50, 10, 100, 90],
        "gt_sorted_rank": 0,
        "same_desc_gt_count": 2,
        "object_count": 3,
        "pred_rows_ordered": [],
        "match_policy_id": "same_desc_greedy_iou_0_5_v1",
        "match_candidates_same_desc": [],
        "accepted_matches": [],
        "best_same_desc_iou": 0.0,
        "near_miss": False,
        "wrong_desc_overlap": False,
        "same_desc_duplicate": False,
        "duplicate_source": None,
        "invalid_pred_count": 0,
        "sample_stratum": "random_only_fn",
        "emitted_gt_indices": [],
        "residual_gt_indices": [0],
    }


def _fn_bucket_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    axis_counts: dict[str, int] = {}
    for row in rows:
        bucket = str(row.get("primary_bucket", "unbucketed"))
        counts[bucket] = counts.get(bucket, 0) + 1
        for axis in row.get("bucket_trace", ()):
            axis_counts[str(axis)] = axis_counts.get(str(axis), 0) + 1
    return {
        "project_id": PROJECT_ID,
        "phase_id": PHASE_ID,
        "schema_version": SCHEMA_VERSION,
        "run_id": RUN_ID,
        "row_count": len(rows),
        "primary_bucket_counts": counts,
        "axis_counts": axis_counts,
    }


def _fn_prefix_sensitivity(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    prefix_counts: dict[str, int] = {}
    flip_count = 0
    for row in rows:
        prefix = str(row.get("prefix_condition", "unknown"))
        prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
        if bool(row.get("prefix_suppression_flip")):
            flip_count += 1
    return {
        "project_id": PROJECT_ID,
        "phase_id": PHASE_ID,
        "schema_version": SCHEMA_VERSION,
        "run_id": RUN_ID,
        "row_count": len(rows),
        "prefix_condition_counts": prefix_counts,
        "prefix_suppression_flip_count": flip_count,
    }


def _fn_slot_rescue_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    strict_hits = sum(bool(row.get("x1_strict_r95_hit")) for row in rows)
    broad_hits = sum(bool(row.get("x1_broad_near_24")) for row in rows)
    return {
        "project_id": PROJECT_ID,
        "phase_id": PHASE_ID,
        "schema_version": SCHEMA_VERSION,
        "run_id": RUN_ID,
        "row_count": len(rows),
        "x1_strict_r95_hit_count": strict_hits,
        "x1_broad_near_24_count": broad_hits,
        "x1_strict_r95_hit_rate": _safe_rate(strict_hits, len(rows)),
    }


def _fn_runtime_provenance_from_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    for row in rows:
        if str(row.get("runtime_kind")) == REAL_FN_HINT_RUNTIME_KIND:
            return {
                "runtime_kind": REAL_FN_HINT_RUNTIME_KIND,
                "probe_runtime_id": str(row.get("probe_runtime_id")),
                "decode_policy": str(row.get("decode_policy")),
                "constraint_policy": str(row.get("constraint_policy")),
            }
    return {}


def _placeholder_gallery_case(*, case_id: str) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "image_path": "",
        "width": 256,
        "height": 192,
        "gt_objects": [{"desc": "person", "bbox_xyxy": [50, 20, 110, 120]}],
        "random_predictions": [{"desc": "person", "bbox_xyxy": [55, 25, 115, 125]}],
        "sorted_predictions": [{"desc": "person", "bbox_xyxy": [50, 20, 110, 120]}],
        "fn_target": {"desc": "person", "bbox_xyxy": [50, 20, 110, 120]},
        "emitted_same_desc": [],
        "residual_same_desc": [{"desc": "person", "bbox_xyxy": [50, 20, 110, 120]}],
        "x1_peaks": [{"x1": 50, "label": "strict R95 hit"}],
    }


def _read_sharded_rows(root: Path, pattern: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.is_dir():
        return rows
    for path in sorted(root.glob(pattern)):
        rows.extend(_read_jsonl(path))
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_safe(row), allow_nan=False, sort_keys=True))
            handle.write("\n")


def _write_yaml(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(_json_safe(payload), sort_keys=False), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must contain JSON objects")
            rows.append(row)
    return rows


def _read_jsonl_or_empty(path: Path) -> list[dict[str, Any]]:
    return _read_jsonl(path) if path.is_file() else []


def _read_json_or_empty(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _ensure_can_write(path: Path, *, allow_overwrite: bool) -> None:
    if path.exists() and not allow_overwrite:
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")


def _ensure_paths_can_write(
    paths: Sequence[Path],
    *,
    allow_overwrite: bool,
) -> None:
    for path in paths:
        _ensure_can_write(path, allow_overwrite=allow_overwrite)


def _placeholder_gallery_paths(root: Path, *, case_id: str) -> tuple[Path, ...]:
    return (
        root / "index.md",
        root / "metadata.json",
        root / "images" / f"{case_id}.jpg",
    )


def _config_to_dict(config: A32Config) -> dict[str, Any]:
    return _json_safe(asdict(config))


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite float is not JSON-safe: {value!r}")
        return value
    return value


def _safe_rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator <= 0 else numerator / denominator


_STAGE_HANDLERS: dict[str, Callable[..., dict[str, Any]]] = {
    "data_root_audit": _stage_data_root_audit,
    "prefix_state_index": _stage_prefix_state_index,
    "validate": _stage_validate,
    "paired_checkpoint_probe": _stage_paired_checkpoint_probe,
    "prefix_merge": _stage_prefix_merge,
    "prefix_report": _stage_prefix_report,
    "prefix_gallery": _stage_prefix_gallery,
    "native_rollout": _stage_native_rollout,
    "rollout_phenotype": _stage_rollout_phenotype,
    "fn_case_index": _stage_fn_case_index,
    "fn_hint_probe": _stage_fn_hint_probe,
    "fn_merge": _stage_fn_merge,
    "fn_report": _stage_fn_report,
    "fn_gallery": _stage_fn_gallery,
    "finalize": _stage_finalize,
}


__all__ = ["GPU_STAGES", "SHARDED_GPU_STAGES", "STAGE_NAMES", "run_stages"]
