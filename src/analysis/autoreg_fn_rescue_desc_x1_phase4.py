"""Phase-4 desc->x1 binding mechanism linkage analyses."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


PHASE4_STAGES = ("instance_attention_binding", "desc_x1_probe_linkage", "report")
PHASE4_DEFAULT_REGION_KINDS = (
    "target_gt",
    "same_desc_competitor_gt_object",
    "same_desc_rollout_prediction",
    "wrong_control_source_region",
    "context_ring",
    "far_background",
)


@dataclass(frozen=True)
class Phase4Paths:
    artifact_root: Path
    phase3_root: Path
    fn_rescue_root: Path
    lane_d_root: Path


@dataclass(frozen=True)
class Phase4InstanceAttentionConfig:
    max_attention_rows: int | None
    roles: tuple[str, ...]
    region_kinds: tuple[str, ...]


@dataclass(frozen=True)
class Phase4Config:
    paths: Phase4Paths
    evidence_scope: str
    instance_attention: Phase4InstanceAttentionConfig


def load_phase4_config(path: Path) -> Phase4Config:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("phase4 config must be a mapping")
    paths_raw = _required_mapping(raw, "paths")
    instance_raw = _optional_mapping(raw, "instance_attention")
    max_attention_rows_raw = instance_raw.get("max_attention_rows")
    return Phase4Config(
        paths=Phase4Paths(
            artifact_root=_required_path(paths_raw, "artifact_root"),
            phase3_root=_required_path(paths_raw, "phase3_root"),
            fn_rescue_root=_required_path(paths_raw, "fn_rescue_root"),
            lane_d_root=_required_path(paths_raw, "lane_d_root"),
        ),
        evidence_scope=str(raw.get("evidence_scope") or "val200_fn_rescue_desc_x1_phase4_binding_mechanism"),
        instance_attention=Phase4InstanceAttentionConfig(
            max_attention_rows=(
                None
                if max_attention_rows_raw is None
                else _nonnegative_int(max_attention_rows_raw, "instance_attention.max_attention_rows")
            ),
            roles=_string_tuple(instance_raw.get("roles", ("pre_x1", "pre_y1")), "instance_attention.roles"),
            region_kinds=_string_tuple(
                instance_raw.get("region_kinds", PHASE4_DEFAULT_REGION_KINDS),
                "instance_attention.region_kinds",
            ),
        ),
    )


def build_phase4_dry_run_plan(config: Phase4Config, *, stages: Sequence[str]) -> dict[str, Any]:
    _validate_stages(stages)
    phase3_case_rows = config.paths.phase3_root / "case_linked" / "case_mechanism_rows.jsonl"
    attention_rows = config.paths.fn_rescue_root / "rescue_attention_region_rows.jsonl"
    probe_rows = config.paths.lane_d_root / "probe_rows.jsonl"
    return {
        "artifact_root": str(config.paths.artifact_root),
        "phase3_root": str(config.paths.phase3_root),
        "fn_rescue_root": str(config.paths.fn_rescue_root),
        "lane_d_root": str(config.paths.lane_d_root),
        "evidence_scope": config.evidence_scope,
        "stages": list(stages),
        "phase3_case_rows": str(phase3_case_rows),
        "attention_rows": str(attention_rows),
        "probe_rows": str(probe_rows),
        "phase3_case_rows_exists": phase3_case_rows.exists(),
        "attention_rows_exists": attention_rows.exists(),
        "probe_rows_exists": probe_rows.exists(),
        "roles": list(config.instance_attention.roles),
        "region_kinds": list(config.instance_attention.region_kinds),
        "max_attention_rows": config.instance_attention.max_attention_rows,
    }


def materialize_instance_attention_binding(config: Phase4Config) -> dict[str, Any]:
    phase3_rows = list(_iter_jsonl(config.paths.phase3_root / "case_linked" / "case_mechanism_rows.jsonl"))
    wanted_keys = {(_required_str(row, "case_id"), _required_str(row, "rescue_tier")) for row in phase3_rows}
    wanted_roles = set(config.instance_attention.roles)
    wanted_regions = set(config.instance_attention.region_kinds)
    totals: dict[tuple[str, str, str], float] = defaultdict(float)
    counts: dict[tuple[str, str, str], int] = defaultdict(int)
    head_totals: dict[tuple[str, str, str, int, int], float] = defaultdict(float)
    head_counts: dict[tuple[str, str, str, int, int], int] = defaultdict(int)
    processed_attention_rows = 0
    joined_attention_rows = 0
    attention_path = config.paths.fn_rescue_root / "rescue_attention_region_rows.jsonl"
    for attention_row in _iter_jsonl(attention_path):
        if (
            config.instance_attention.max_attention_rows is not None
            and processed_attention_rows >= config.instance_attention.max_attention_rows
        ):
            break
        processed_attention_rows += 1
        if str(attention_row.get("aggregation_scope") or "") != "instance":
            continue
        if str(attention_row.get("role") or "") not in wanted_roles:
            continue
        region_kind = str(attention_row.get("region_kind") or "")
        if region_kind not in wanted_regions:
            continue
        key = (_required_str(attention_row, "case_id"), _required_str(attention_row, "rescue_tier"))
        if key not in wanted_keys:
            continue
        mass = float(attention_row.get("attention_mass_normalized", attention_row.get("attention_mass", 0.0)))
        totals[(key[0], key[1], region_kind)] += mass
        counts[(key[0], key[1], region_kind)] += 1
        layer = int(attention_row.get("layer") or -1)
        head = int(attention_row.get("head") or -1)
        role = str(attention_row.get("role") or "")
        head_key = (key[0], key[1], role, layer, head)
        head_totals[head_key] += mass
        head_counts[head_key] += 1
        joined_attention_rows += 1

    masses_by_case: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for (case_id, tier, region_kind), total in totals.items():
        masses_by_case[(case_id, tier)][region_kind] = total / counts[(case_id, tier, region_kind)]
    top_heads_by_case: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for (case_id, tier, role, layer, head), total in head_totals.items():
        mean_mass = total / head_counts[(case_id, tier, role, layer, head)]
        top_heads_by_case[(case_id, tier)].append(
            {
                "aggregation_scope": "instance",
                "role": role,
                "layer": layer,
                "head": head,
                "_mean_mass": mean_mass,
            }
        )
    for heads in top_heads_by_case.values():
        heads.sort(key=lambda row: (-float(row["_mean_mass"]), str(row["role"]), int(row["layer"]), int(row["head"])))

    output_rows: list[dict[str, Any]] = []
    for phase3_row in phase3_rows:
        key = (_required_str(phase3_row, "case_id"), _required_str(phase3_row, "rescue_tier"))
        masses = masses_by_case.get(key, {})
        target_mass = masses.get("target_gt")
        competitor_mass = _first_non_none(
            masses.get("same_desc_competitor_gt_object"),
            masses.get("same_desc_rollout_prediction"),
        )
        output_rows.append(
            {
                "case_id": key[0],
                "rescue_tier": key[1],
                "intervention_kind": phase3_row.get("intervention_kind"),
                "mechanism_bucket": phase3_row.get("mechanism_bucket"),
                "target_iou_delta": phase3_row.get("target_iou_delta"),
                "primary_success_changed": phase3_row.get("primary_success_changed"),
                "valid_paired_row": phase3_row.get("valid_paired_row"),
                "target_attention_mass": target_mass,
                "competitor_attention_mass": competitor_mass,
                "rollout_prediction_attention_mass": masses.get("same_desc_rollout_prediction"),
                "wrong_source_attention_mass": masses.get("wrong_control_source_region"),
                "context_ring_attention_mass": masses.get("context_ring"),
                "far_background_attention_mass": masses.get("far_background"),
                "target_minus_competitor_instance_attention": (
                    None
                    if target_mass is None or competitor_mass is None
                    else float(target_mass) - float(competitor_mass)
                ),
                "top_instance_attention_heads": [
                    {k: v for k, v in row.items() if k != "_mean_mass"}
                    for row in top_heads_by_case.get(key, [])[:5]
                ],
            }
        )

    output_root = config.paths.artifact_root / "instance_attention_binding"
    output_root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_root / "rows.jsonl", output_rows)
    summary = {
        "artifact_root": str(config.paths.artifact_root),
        "phase3_root": str(config.paths.phase3_root),
        "fn_rescue_root": str(config.paths.fn_rescue_root),
        "evidence_scope": config.evidence_scope,
        "stage": "instance_attention_binding",
        "row_count": len(output_rows),
        "case_count": len(wanted_keys),
        "processed_attention_rows": processed_attention_rows,
        "joined_attention_rows": joined_attention_rows,
        "bucket_summaries": _bucket_summaries(output_rows),
        "interpretation_boundary": (
            "instance-level attention is linked to image-region causal outcomes; "
            "this is not attention-head causality"
        ),
    }
    _write_json(output_root / "summary.json", summary)
    _write_instance_report(output_root / "report.md", summary)
    return summary


def materialize_desc_x1_probe_linkage(config: Phase4Config) -> dict[str, Any]:
    phase3_rows = list(_iter_jsonl(config.paths.phase3_root / "case_linked" / "case_mechanism_rows.jsonl"))
    phase3_case_ids = {_required_str(row, "case_id") for row in phase3_rows}
    probe_path = config.paths.lane_d_root / "probe_rows.jsonl"
    output_root = config.paths.artifact_root / "desc_x1_probe_linkage"
    output_root.mkdir(parents=True, exist_ok=True)
    if not probe_path.exists():
        summary = {
            "artifact_root": str(config.paths.artifact_root),
            "lane_d_root": str(config.paths.lane_d_root),
            "evidence_scope": config.evidence_scope,
            "stage": "desc_x1_probe_linkage",
            "status": "blocked_missing_probe_rows",
            "phase3_case_rows": len(phase3_rows),
            "probe_rows": 0,
            "joined_case_count": 0,
            "available_roles": [],
            "available_layer_groups": [],
            "interpretation_boundary": "hidden/logit binding claims require Lane-D probe_rows.jsonl",
        }
        _write_json(output_root / "summary.json", summary)
        _write_probe_report(output_root / "report.md", summary)
        return summary

    probe_rows = list(_iter_jsonl(probe_path))
    joined_case_ids = {_required_str(row, "case_id") for row in probe_rows if _required_str(row, "case_id") in phase3_case_ids}
    summary = {
        "artifact_root": str(config.paths.artifact_root),
        "lane_d_root": str(config.paths.lane_d_root),
        "evidence_scope": config.evidence_scope,
        "stage": "desc_x1_probe_linkage",
        "status": "linked_probe_rows_available",
        "phase3_case_rows": len(phase3_rows),
        "probe_rows": len(probe_rows),
        "joined_case_count": len(joined_case_ids),
        "available_roles": sorted({str(row.get("role")) for row in probe_rows if row.get("role") is not None}),
        "available_layer_groups": sorted(
            {str(row.get("layer_group")) for row in probe_rows if row.get("layer_group") is not None}
        ),
        "interpretation_boundary": "linkage availability only; detailed x1-rank analysis depends on probe row schema",
    }
    _write_json(output_root / "summary.json", summary)
    _write_probe_report(output_root / "report.md", summary)
    return summary


def write_phase4_report(config: Phase4Config) -> Path:
    stage_summaries: dict[str, Any] = {}
    for subdir in ("instance_attention_binding", "desc_x1_probe_linkage"):
        path = config.paths.artifact_root / subdir / "summary.json"
        if path.exists():
            stage_summaries[subdir] = json.loads(path.read_text(encoding="utf-8"))
    summary = {
        "artifact_root": str(config.paths.artifact_root),
        "phase3_root": str(config.paths.phase3_root),
        "fn_rescue_root": str(config.paths.fn_rescue_root),
        "lane_d_root": str(config.paths.lane_d_root),
        "evidence_scope": config.evidence_scope,
        "stages_materialized": sorted(stage_summaries),
        "stage_summaries": stage_summaries,
    }
    config.paths.artifact_root.mkdir(parents=True, exist_ok=True)
    _write_json(config.paths.artifact_root / "summary.json", summary)
    lines = [
        "# FN-Rescue Desc-X1 Binding Mechanism Phase 4",
        "",
        f"- Evidence scope: `{config.evidence_scope}`",
        f"- Artifact root: `{config.paths.artifact_root}`",
        "",
        "## Instance-Level Attention Binding",
        "",
    ]
    instance_summary = stage_summaries.get("instance_attention_binding")
    if instance_summary is None:
        lines.append("Not materialized.")
    else:
        lines.append(f"- Rows: `{instance_summary.get('row_count')}`")
        lines.append(f"- Joined attention rows: `{instance_summary.get('joined_attention_rows')}`")
        lines.append(f"- Bucket summaries: `{instance_summary.get('bucket_summaries')}`")
    lines.extend(["", "## Desc-X1 Probe Linkage", ""])
    probe_summary = stage_summaries.get("desc_x1_probe_linkage")
    if probe_summary is None:
        lines.append("Not materialized.")
    else:
        lines.append(f"- Status: `{probe_summary.get('status')}`")
        lines.append(f"- Joined cases: `{probe_summary.get('joined_case_count')}`")
    report_path = config.paths.artifact_root / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _bucket_summaries(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    by_bucket = Counter(str(row.get("mechanism_bucket")) for row in rows)
    summaries: dict[str, dict[str, Any]] = {}
    for bucket in sorted(by_bucket):
        bucket_rows = [row for row in rows if str(row.get("mechanism_bucket")) == bucket]
        margins = [
            float(row["target_minus_competitor_instance_attention"])
            for row in bucket_rows
            if row.get("target_minus_competitor_instance_attention") is not None
        ]
        summaries[bucket] = {
            "rows": len(bucket_rows),
            "valid_paired_rows": sum(1 for row in bucket_rows if row.get("valid_paired_row") is True),
            "mean_target_minus_competitor_instance_attention": None
            if not margins
            else sum(margins) / len(margins),
        }
    return summaries


def _write_instance_report(path: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# Instance-Level Attention Binding",
        "",
        f"- Rows: `{summary.get('row_count')}`",
        f"- Joined attention rows: `{summary.get('joined_attention_rows')}`",
        f"- Interpretation boundary: {summary.get('interpretation_boundary')}",
        "",
    ]
    for bucket, payload in sorted(summary.get("bucket_summaries", {}).items()):
        lines.append(
            f"- `{bucket}`: rows `{payload.get('rows')}`, "
            f"mean target-minus-competitor `{payload.get('mean_target_minus_competitor_instance_attention')}`"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_probe_report(path: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# Desc-X1 Probe Linkage",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Phase-3 rows: `{summary.get('phase3_case_rows')}`",
        f"- Probe rows: `{summary.get('probe_rows')}`",
        f"- Joined cases: `{summary.get('joined_case_count')}`",
        f"- Interpretation boundary: {summary.get('interpretation_boundary')}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _validate_stages(stages: Sequence[str]) -> None:
    if not stages:
        raise ValueError("stages must not be empty")
    unknown = sorted(set(stages) - set(PHASE4_STAGES))
    if unknown:
        raise ValueError(f"unknown phase4 stage(s): {', '.join(unknown)}")


def _iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _required_mapping(row: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    value = row.get(field)
    if not isinstance(value, Mapping):
        raise ValueError(f"missing mapping field: {field}")
    return value


def _optional_mapping(row: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    value = row.get(field, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _required_path(row: Mapping[str, Any], field: str) -> Path:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"missing path field: {field}")
    return Path(value).expanduser()


def _required_str(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"missing string field: {field}")
    return value


def _string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be a string or sequence")
    parsed = tuple(str(item) for item in value)
    if not parsed or any(not item for item in parsed):
        raise ValueError(f"{field_name} must contain at least one non-empty value")
    return parsed


def _nonnegative_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a nonnegative integer")
    return value


def _first_non_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None
