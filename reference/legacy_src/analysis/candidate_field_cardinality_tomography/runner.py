from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import yaml

from .artifacts import validate_manifest, write_json, write_jsonl, write_manifest
from .case_index import build_case_index
from .config import CandidateFieldConfig, KNOWN_STAGES, load_config
from .controls import REQUIRED_CONTROL_TYPES
from .probe_plan import build_probe_plan
from .report import build_probe_breakdowns, build_summary
from .taxonomy import materialize_phase_a_taxonomy_rows
from .x1_candidate_field import X1ProbeRuntime, materialize_x1_candidate_field_rows

UNIMPLEMENTED_STAGES = {
    "residual_row_scoring",
    "basin_attraction",
    "attention_components",
    "gallery",
}


def build_dry_run_plan(config: CandidateFieldConfig, *, stages: Sequence[str]) -> dict[str, Any]:
    _validate_stages(stages)
    return {
        "project_id": config.project_id,
        "artifact_root": str(config.artifact_root),
        "checkpoint_path": str(config.checkpoint_path),
        "train_jsonl": str(config.train_jsonl),
        "val_jsonl": str(config.val_jsonl),
        "stages": list(stages),
        "num_shards": config.sampling.num_shards,
        "max_cases": config.sampling.max_cases,
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
    _reject_unimplemented_stages(selected_stages)
    if dry_run:
        return build_dry_run_plan(config, stages=selected_stages)
    root = config.artifact_root
    if root.exists() and not allow_overwrite:
        raise FileExistsError(f"artifact root exists: {root}")
    root.mkdir(parents=True, exist_ok=True)
    resolved_config_path = (
        root / "resolved_config.yaml"
        if shard_id is None
        else root / "shards" / f"shard_{shard_id:03d}" / "resolved_config.yaml"
    )
    resolved_config_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_config_path.write_text(
        yaml.safe_dump(
            json.loads(
                json.dumps(
                    _config_to_dict(
                        config,
                        config_path=config_path,
                        effective_stages=selected_stages,
                        shard_id=shard_id,
                    )
                )
            )
        ),
        encoding="utf-8",
    )
    case_rows: list[dict[str, Any]] = []
    plan_rows: list[dict[str, Any]] = []
    if "case_index" in selected_stages:
        case_rows, case_summary = build_case_index(
            train_jsonl=config.train_jsonl,
            val_jsonl=config.val_jsonl,
            checkpoint_id=config.checkpoint_path.name,
            run_id="candidate-field-run",
        )
        write_jsonl(root / "case_index.jsonl", case_rows)
        write_json(root / "case_index_summary.json", case_summary)
    elif (root / "case_index.jsonl").exists():
        from .artifacts import read_jsonl

        case_rows = read_jsonl(root / "case_index.jsonl")
    if "probe_plan" in selected_stages:
        plan_rows, plan_summary = build_probe_plan(
            case_rows,
            num_shards=config.sampling.num_shards,
            max_cases=config.sampling.max_cases,
            seed=config.sampling.seed,
        )
        write_jsonl(root / "probe_plan.jsonl", plan_rows)
        write_json(root / "probe_plan_summary.json", plan_summary)
    elif (root / "probe_plan.jsonl").exists():
        from .artifacts import read_jsonl

        plan_rows = read_jsonl(root / "probe_plan.jsonl")
    if "x1_candidate_field" in selected_stages:
        materialize_x1_candidate_field_rows(
            artifact_root=root,
            checkpoint_path=config.checkpoint_path,
            runtime=X1ProbeRuntime(
                max_cases=config.sampling.max_cases,
                shard_id=shard_id,
                raw_topk_k=config.peak.raw_topk_k,
                primary_merge_radius=config.peak.primary_merge_radius,
                absolute_mass_floor=config.peak.absolute_mass_floor,
                relative_floor=config.peak.relative_floor,
                gt_x1_neighborhood_radius=config.peak.gt_x1_neighborhood_radius,
            ),
        )
    if "merge" in selected_stages:
        _merge_x1_shards(root)
    if "taxonomy" in selected_stages:
        from .artifacts import read_jsonl

        x1_rows = read_jsonl(root / "x1_candidate_field_rows.jsonl")
        materialize_phase_a_taxonomy_rows(root, x1_rows)
    if "validate" in selected_stages or "report" in selected_stages:
        from .artifacts import read_jsonl

        x1_rows = read_jsonl(root / "x1_candidate_field_rows.jsonl")
        taxonomy_rows = read_jsonl(root / "phase_a_case_taxonomy_rows.jsonl")
        case_total = _case_total(root, case_rows)
        case_index_rows = _case_index_rows(root, case_rows)
        planned = _planned_total(root, plan_rows)
        valid = sum(1 for row in x1_rows if row.get("probe_status") == "ok")
        validation_status = _validation_status(
            root=root,
            planned=planned,
            attempted=len(x1_rows),
            taxonomy_count=len(taxonomy_rows),
        )
        summary = build_summary(
            case_index_total=case_total,
            case_index_rows=case_index_rows,
            planned=planned,
            attempted=len(x1_rows),
            valid=valid,
            assigned=len(taxonomy_rows),
            control_status_by_type=_control_status_from_x1_rows(x1_rows),
            breakdowns=build_probe_breakdowns(x1_rows, taxonomy_rows),
            checkpoint_id=config.checkpoint_path.name,
            checkpoint_path=str(config.checkpoint_path),
            validation_status=validation_status,
        )
        write_json(root / "summary.json", summary)
        (root / "report.md").write_text(_report_markdown(summary), encoding="utf-8")
        _write_artifact_manifest(
            root,
            metadata={
                "project_id": config.project_id,
                "checkpoint_path": str(config.checkpoint_path),
                "effective_stages": list(selected_stages),
                "shard_id": shard_id,
                "validation_status": validation_status,
            },
        )
    return {"artifact_root": str(root), "stages": list(selected_stages)}


def _merge_x1_shards(root: Path) -> None:
    from .artifacts import read_jsonl

    expected_shards = _expected_shard_count(root)
    expected_by_shard = _expected_sampled_counts_by_shard(root)
    rows: list[dict[str, Any]] = []
    for shard_id in range(expected_shards):
        shard_dir = root / "shards" / f"shard_{shard_id:03d}"
        manifest_path = shard_dir / "shard_manifest.json"
        rows_path = shard_dir / "x1_candidate_field_rows.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing shard manifest: {manifest_path}")
        if not rows_path.exists():
            raise FileNotFoundError(f"missing shard rows: {rows_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("stage_status") != "ok":
            raise ValueError(f"shard {shard_id} is not complete")
        shard_rows = read_jsonl(rows_path)
        expected_rows = expected_by_shard.get(shard_id)
        if expected_rows is not None and len(shard_rows) != expected_rows:
            raise ValueError(
                f"shard {shard_id} row count mismatch: {len(shard_rows)} != {expected_rows}"
            )
        manifest_rows = int(
            manifest.get("output_row_counts", {}).get("x1_candidate_field_rows.jsonl", -1)
        )
        if manifest_rows != len(shard_rows):
            raise ValueError(f"shard {shard_id} manifest row count mismatch")
        rows.extend(shard_rows)
    if len({row.get("probe_plan_row_id") for row in rows}) != len(rows):
        raise ValueError("duplicate probe_plan_row_id in merged x1 rows")
    write_jsonl(root / "x1_candidate_field_rows.jsonl", rows)
    write_json(
        root / "merge_summary.json",
        {
            "stage": "merge",
            "x1_candidate_field_rows": len(rows),
            "shard_count": expected_shards,
        },
    )


def _case_total(root: Path, case_rows: list[dict[str, Any]]) -> int:
    if case_rows:
        return len({row["case_id"] for row in case_rows})
    summary_path = root / "case_index_summary.json"
    if summary_path.exists():
        import json

        return int(json.loads(summary_path.read_text(encoding="utf-8")).get("case_index_total_cases", 0))
    return 0


def _case_index_rows(root: Path, case_rows: list[dict[str, Any]]) -> int:
    if case_rows:
        return len(case_rows)
    summary_path = root / "case_index_summary.json"
    if summary_path.exists():
        return int(json.loads(summary_path.read_text(encoding="utf-8")).get("case_index_total_rows", 0))
    return 0


def _planned_total(root: Path, plan_rows: list[dict[str, Any]]) -> int:
    if plan_rows:
        return len([row for row in plan_rows if row.get("probe_sampled")])
    summary_path = root / "probe_plan_summary.json"
    if summary_path.exists():
        import json

        return int(json.loads(summary_path.read_text(encoding="utf-8")).get("gpu_probe_planned_cases", 0))
    return 0


def _control_status_from_x1_rows(rows: list[dict[str, Any]]) -> dict[str, str]:
    roles = {str(row.get("pool_role")) for row in rows if row.get("probe_status") == "ok"}
    statuses: dict[str, str] = {control: "missing" for control in REQUIRED_CONTROL_TYPES}
    for role in ("same_desc_count_1_control", "same_desc_count_2_control"):
        statuses[role] = "pass" if role in roles else "missing"
    if any(row.get("x1_projection_collision") for row in rows if row.get("probe_status") == "ok"):
        statuses["x1_projection_collision_slice"] = "present"
    else:
        statuses["x1_projection_collision_slice"] = "missing"
    coord_masses = [
        float(row.get("coord_vocab_mass") or 0.0)
        for row in rows
        if row.get("probe_status") == "ok"
    ]
    if coord_masses:
        statuses["p_cond_vs_coord_vocab_mass"] = "pass" if min(coord_masses) >= 0.05 else "fail"
    else:
        statuses["p_cond_vs_coord_vocab_mass"] = "missing"
    return statuses


def _validate_stages(stages: Sequence[str]) -> None:
    unknown = sorted(set(stages) - KNOWN_STAGES)
    if unknown:
        raise ValueError(f"unknown stage: {unknown[0]}")


def _reject_unimplemented_stages(stages: Sequence[str]) -> None:
    selected = sorted(set(stages) & UNIMPLEMENTED_STAGES)
    if selected:
        raise NotImplementedError(f"stage not implemented: {selected[0]}")


def _config_to_dict(
    config: CandidateFieldConfig,
    *,
    config_path: Path,
    effective_stages: Sequence[str],
    shard_id: int | None,
) -> dict[str, Any]:
    return {
        "project_id": config.project_id,
        "artifact_root": str(config.artifact_root),
        "checkpoint_path": str(config.checkpoint_path),
        "train_jsonl": str(config.train_jsonl),
        "val_jsonl": str(config.val_jsonl),
        "fn_rescue_overlay_root": None if config.fn_rescue_overlay_root is None else str(config.fn_rescue_overlay_root),
        "phase5_overlay_root": None if config.phase5_overlay_root is None else str(config.phase5_overlay_root),
        "configured_stages": list(config.stages),
        "effective_stages": list(effective_stages),
        "config_path": str(config_path),
        "shard_id": shard_id,
        "sampling": {
            "num_shards": config.sampling.num_shards,
            "max_cases": config.sampling.max_cases,
            "seed": config.sampling.seed,
        },
        "peak": {
            "absolute_mass_floor": config.peak.absolute_mass_floor,
            "relative_floor": config.peak.relative_floor,
            "primary_merge_radius": config.peak.primary_merge_radius,
            "gt_x1_neighborhood_radius": config.peak.gt_x1_neighborhood_radius,
            "raw_topk_k": config.peak.raw_topk_k,
        },
        "policies": {
            "desc_normalization_policy_id": config.policies.desc_normalization_policy_id,
            "row_score_policy_id": config.policies.row_score_policy_id,
            "decode_policy_id": config.policies.decode_policy_id,
        },
    }


def _validation_status(root: Path, *, planned: int, attempted: int, taxonomy_count: int) -> str:
    if planned > 0 and attempted == 0:
        return "incomplete_missing_stage_outputs"
    if attempted > 0 and taxonomy_count == 0 and not (root / "phase_a_case_taxonomy_rows.jsonl").exists():
        return "incomplete_missing_stage_outputs"
    if planned > 0 and attempted > planned:
        return "invalid_row_count_mismatch"
    return "ok"


def _expected_shard_count(root: Path) -> int:
    summary_path = root / "probe_plan_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing probe plan summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return int(summary.get("num_shards") or 0)


def _expected_sampled_counts_by_shard(root: Path) -> dict[int, int]:
    from .artifacts import read_jsonl

    plan_path = root / "probe_plan.jsonl"
    if not plan_path.exists():
        return {}
    counts: dict[int, int] = {}
    for row in read_jsonl(plan_path):
        if row.get("probe_sampled") is True:
            shard_id = int(row.get("planned_shard_id", -1))
            counts[shard_id] = counts.get(shard_id, 0) + 1
    return counts


def _write_artifact_manifest(root: Path, metadata: dict[str, Any]) -> None:
    files = [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    ]
    write_manifest(root, files, metadata)
    validate_manifest(root)


def _report_markdown(summary: dict[str, Any]) -> str:
    counts = summary.get("denominator_counts", {})
    return "\n".join(
        [
            "# Candidate-Field Cardinality Tomography Report",
            "",
            "Scope: smoke/analysis artifact, not production training.",
            "",
            f"Checkpoint: `{summary.get('checkpoint_id')}`",
            f"Checkpoint path: `{summary.get('checkpoint_path')}`",
            "",
            f"Validation status: `{summary.get('validation_status')}`",
            f"Headline eligibility: `{summary.get('headline_eligibility_status')}`",
            "",
            "## Denominators",
            "",
            f"- indexed_cases: `{counts.get('indexed_cases')}`",
            f"- indexed_gt_rows: `{counts.get('indexed_gt_rows')}`",
            f"- sampled_gpu_cases: `{counts.get('sampled_gpu_cases')}`",
            f"- attempted_gpu_cases: `{counts.get('attempted_gpu_cases')}`",
            f"- valid_gpu_cases: `{counts.get('valid_gpu_cases')}`",
            f"- taxonomy_cases: `{counts.get('taxonomy_cases')}`",
            "",
            "Current taxonomy is based on the first x1 posterior probe only; residual, basin, and attention-component stages remain separate evidence layers.",
            "",
        ]
    )
