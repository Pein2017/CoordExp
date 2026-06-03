"""Phase-5 desc->x1 state-dependent logit binding analyses."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


PHASE5_STAGES = ("x1_logit_binding_probe", "coord_slot_logit_binding_probe", "report")
PHASE5_DEFAULT_ROLES = ("desc_end", "box_start", "pre_x1", "post_x1", "pre_y1")
PHASE5_DEFAULT_LAYER_GROUPS = ("middle", "late", "last")


@dataclass(frozen=True)
class Phase5Paths:
    artifact_root: Path
    phase4_root: Path
    lane_d_root: Path


@dataclass(frozen=True)
class Phase5ProbeConfig:
    max_probe_rows: int | None
    requested_roles: tuple[str, ...]
    layer_groups: tuple[str, ...]
    poor_rank_threshold: int


@dataclass(frozen=True)
class Phase5Config:
    paths: Phase5Paths
    evidence_scope: str
    x1_logit_probe: Phase5ProbeConfig


def load_phase5_config(path: Path) -> Phase5Config:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("phase5 config must be a mapping")
    paths_raw = _required_mapping(raw, "paths")
    probe_raw = _optional_mapping(raw, "x1_logit_probe")
    max_probe_rows_raw = probe_raw.get("max_probe_rows")
    return Phase5Config(
        paths=Phase5Paths(
            artifact_root=_required_path(paths_raw, "artifact_root"),
            phase4_root=_required_path(paths_raw, "phase4_root"),
            lane_d_root=_required_path(paths_raw, "lane_d_root"),
        ),
        evidence_scope=str(raw.get("evidence_scope") or "val200_fn_rescue_desc_x1_phase5_logit_binding"),
        x1_logit_probe=Phase5ProbeConfig(
            max_probe_rows=(
                None
                if max_probe_rows_raw is None
                else _nonnegative_int(max_probe_rows_raw, "x1_logit_probe.max_probe_rows")
            ),
            requested_roles=_string_tuple(
                probe_raw.get("requested_roles", PHASE5_DEFAULT_ROLES),
                "x1_logit_probe.requested_roles",
            ),
            layer_groups=_string_tuple(
                probe_raw.get("layer_groups", PHASE5_DEFAULT_LAYER_GROUPS),
                "x1_logit_probe.layer_groups",
            ),
            poor_rank_threshold=_positive_int(
                probe_raw.get("poor_rank_threshold", 100),
                "x1_logit_probe.poor_rank_threshold",
            ),
        ),
    )


def build_phase5_dry_run_plan(config: Phase5Config, *, stages: Sequence[str]) -> dict[str, Any]:
    _validate_stages(stages)
    phase4_rows = config.paths.phase4_root / "instance_attention_binding" / "rows.jsonl"
    probe_rows = config.paths.lane_d_root / "probe_rows.jsonl"
    return {
        "artifact_root": str(config.paths.artifact_root),
        "phase4_root": str(config.paths.phase4_root),
        "lane_d_root": str(config.paths.lane_d_root),
        "evidence_scope": config.evidence_scope,
        "stages": list(stages),
        "phase4_rows": str(phase4_rows),
        "probe_rows": str(probe_rows),
        "phase4_rows_exists": phase4_rows.exists(),
        "probe_rows_exists": probe_rows.exists(),
        "requested_roles": list(config.x1_logit_probe.requested_roles),
        "layer_groups": list(config.x1_logit_probe.layer_groups),
        "max_probe_rows": config.x1_logit_probe.max_probe_rows,
        "poor_rank_threshold": config.x1_logit_probe.poor_rank_threshold,
    }


def materialize_x1_logit_binding_probe(config: Phase5Config) -> dict[str, Any]:
    phase4_rows = list(_iter_jsonl(config.paths.phase4_root / "instance_attention_binding" / "rows.jsonl"))
    phase4_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in phase4_rows:
        phase4_by_case[_required_str(row, "case_id")].append(dict(row))

    requested_roles = set(config.x1_logit_probe.requested_roles)
    requested_layer_groups = set(config.x1_logit_probe.layer_groups)
    output_rows: list[dict[str, Any]] = []
    processed_probe_rows = 0
    joined_probe_rows = 0
    probe_case_ids: set[str] = set()
    available_case_ids: set[str] = set()
    coverage: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {"rows": 0, "available_rows": 0})

    for probe_row in _iter_jsonl(config.paths.lane_d_root / "probe_rows.jsonl"):
        if (
            config.x1_logit_probe.max_probe_rows is not None
            and processed_probe_rows >= config.x1_logit_probe.max_probe_rows
        ):
            break
        processed_probe_rows += 1
        role = str(probe_row.get("role") or "")
        layer_group = str(probe_row.get("layer_group") or "")
        if role not in requested_roles or layer_group not in requested_layer_groups:
            continue
        case_id = _required_str(probe_row, "case_id")
        phase4_matches = phase4_by_case.get(case_id)
        if not phase4_matches:
            continue
        joined_probe_rows += 1
        probe_case_ids.add(case_id)
        available = probe_row.get("x1_logit_lens_available") is True
        coverage[(role, layer_group)]["rows"] += 1
        if available:
            coverage[(role, layer_group)]["available_rows"] += 1
            available_case_ids.add(case_id)
        for phase4_row in phase4_matches:
            output_rows.append(_joined_row(phase4_row, probe_row, config=config, available=available))

    output_root = config.paths.artifact_root / "x1_logit_binding_probe"
    output_root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_root / "rows.jsonl", output_rows)
    summary = {
        "artifact_root": str(config.paths.artifact_root),
        "phase4_root": str(config.paths.phase4_root),
        "lane_d_root": str(config.paths.lane_d_root),
        "evidence_scope": config.evidence_scope,
        "stage": "x1_logit_binding_probe",
        "phase4_rows": len(phase4_rows),
        "processed_probe_rows": processed_probe_rows,
        "joined_probe_rows": joined_probe_rows,
        "row_count": len(output_rows),
        "joined_case_count": len(probe_case_ids),
        "available_case_count": len(available_case_ids),
        "available_logit_rows": sum(1 for row in output_rows if row["x1_logit_lens_available"] is True),
        "unique_available_probe_rows": len(_unique_available_probe_rows(output_rows)),
        "requested_roles": list(config.x1_logit_probe.requested_roles),
        "layer_groups": list(config.x1_logit_probe.layer_groups),
        "role_layer_coverage": _role_layer_coverage(config, coverage),
        "missing_requested_roles": _missing_requested_roles(config, coverage),
        "bucket_summaries": _bucket_summaries(output_rows),
        "role_layer_summaries": _role_layer_summaries(output_rows),
        "unique_role_layer_summaries": _role_layer_summaries(_unique_available_probe_rows(output_rows)),
        "critical_slice": _critical_slice(output_rows),
        "schema_audit": _schema_audit(output_rows),
        "interpretation_boundary": (
            "x1_logit_lens_* fields are state-dependent; x1_target_rank and "
            "x1_top_peak_attribution are Lane-C case-selection metadata"
        ),
    }
    _write_json(output_root / "summary.json", summary)
    _write_probe_report(output_root / "report.md", summary)
    return summary


def materialize_coord_slot_logit_binding_probe(config: Phase5Config) -> dict[str, Any]:
    phase4_rows = list(_iter_jsonl(config.paths.phase4_root / "instance_attention_binding" / "rows.jsonl"))
    phase4_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in phase4_rows:
        phase4_by_case[_required_str(row, "case_id")].append(dict(row))

    requested_layer_groups = set(config.x1_logit_probe.layer_groups)
    output_rows: list[dict[str, Any]] = []
    processed_probe_rows = 0
    joined_probe_rows = 0
    probe_case_ids: set[str] = set()
    available_case_ids: set[str] = set()
    for probe_row in _iter_jsonl(config.paths.lane_d_root / "probe_rows.jsonl"):
        if (
            config.x1_logit_probe.max_probe_rows is not None
            and processed_probe_rows >= config.x1_logit_probe.max_probe_rows
        ):
            break
        processed_probe_rows += 1
        if str(probe_row.get("layer_group") or "") not in requested_layer_groups:
            continue
        if "coord_slot_logit_lens_available" not in probe_row:
            continue
        case_id = _required_str(probe_row, "case_id")
        phase4_matches = phase4_by_case.get(case_id)
        if not phase4_matches:
            continue
        joined_probe_rows += 1
        probe_case_ids.add(case_id)
        available = probe_row.get("coord_slot_logit_lens_available") is True
        if available:
            available_case_ids.add(case_id)
        for phase4_row in phase4_matches:
            output_rows.append(_coord_slot_joined_row(phase4_row, probe_row, config=config, available=available))

    output_root = config.paths.artifact_root / "coord_slot_logit_binding_probe"
    output_root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_root / "rows.jsonl", output_rows)
    summary = {
        "artifact_root": str(config.paths.artifact_root),
        "phase4_root": str(config.paths.phase4_root),
        "lane_d_root": str(config.paths.lane_d_root),
        "evidence_scope": config.evidence_scope,
        "stage": "coord_slot_logit_binding_probe",
        "phase4_rows": len(phase4_rows),
        "processed_probe_rows": processed_probe_rows,
        "joined_probe_rows": joined_probe_rows,
        "row_count": len(output_rows),
        "joined_case_count": len(probe_case_ids),
        "available_case_count": len(available_case_ids),
        "available_logit_rows": sum(1 for row in output_rows if row["coord_slot_logit_lens_available"] is True),
        "unique_available_probe_rows": len(_unique_available_coord_slot_probe_rows(output_rows)),
        "role_slot_summaries": _role_slot_summaries(output_rows),
        "unique_role_slot_summaries": _role_slot_summaries(_unique_available_coord_slot_probe_rows(output_rows)),
        "bucket_summaries": _coord_slot_bucket_summaries(output_rows),
        "critical_slice": _coord_slot_critical_slice(output_rows),
        "interpretation_boundary": (
            "coord_slot_logit_lens_* fields use the target coordinate for the role slot; "
            "post_x1 probes target_y1, not target_x1"
        ),
    }
    _write_json(output_root / "summary.json", summary)
    _write_coord_slot_report(output_root / "report.md", summary)
    return summary


def write_phase5_report(config: Phase5Config) -> Path:
    config.paths.artifact_root.mkdir(parents=True, exist_ok=True)
    summary_path = config.paths.artifact_root / "x1_logit_binding_probe" / "summary.json"
    probe_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
    coord_slot_summary_path = config.paths.artifact_root / "coord_slot_logit_binding_probe" / "summary.json"
    coord_slot_summary = (
        json.loads(coord_slot_summary_path.read_text(encoding="utf-8"))
        if coord_slot_summary_path.exists()
        else None
    )
    top_summary = {
        "artifact_root": str(config.paths.artifact_root),
        "phase4_root": str(config.paths.phase4_root),
        "lane_d_root": str(config.paths.lane_d_root),
        "evidence_scope": config.evidence_scope,
        "stages_materialized": [
            stage
            for stage, payload in (
                ("x1_logit_binding_probe", probe_summary),
                ("coord_slot_logit_binding_probe", coord_slot_summary),
            )
            if payload is not None
        ],
        "x1_logit_binding_probe": probe_summary,
        "coord_slot_logit_binding_probe": coord_slot_summary,
    }
    _write_json(config.paths.artifact_root / "summary.json", top_summary)
    report_path = config.paths.artifact_root / "report.md"
    lines = [
        "# FN-Rescue Desc-X1 Logit Binding Phase 5",
        "",
        f"- Evidence scope: `{config.evidence_scope}`",
        f"- Artifact root: `{config.paths.artifact_root}`",
        "",
    ]
    if probe_summary is not None:
        lines.extend(
            [
                "## X1 Logit Binding Probe",
                "",
                f"- Rows: `{probe_summary.get('row_count')}`",
                f"- Available logit rows: `{probe_summary.get('available_logit_rows')}`",
                f"- Missing requested roles: `{probe_summary.get('missing_requested_roles')}`",
                f"- Critical slice: `{probe_summary.get('critical_slice')}`",
                "",
            ]
        )
    if coord_slot_summary is not None:
        lines.extend(
            [
                "## Coord-Slot Logit Binding Probe",
                "",
                f"- Rows: `{coord_slot_summary.get('row_count')}`",
                f"- Available logit rows: `{coord_slot_summary.get('available_logit_rows')}`",
                f"- Critical slice: `{coord_slot_summary.get('critical_slice')}`",
                "",
            ]
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _joined_row(
    phase4_row: Mapping[str, Any],
    probe_row: Mapping[str, Any],
    *,
    config: Phase5Config,
    available: bool,
) -> dict[str, Any]:
    rank = probe_row.get("x1_logit_lens_rank")
    target_minus_top1 = probe_row.get("x1_logit_lens_target_minus_top1")
    target_attention_margin = phase4_row.get("target_minus_competitor_instance_attention")
    target_attention_positive = isinstance(target_attention_margin, (int, float)) and float(target_attention_margin) > 0.0
    poor_rank = isinstance(rank, int) and rank > config.x1_logit_probe.poor_rank_threshold
    negative_logit_margin = isinstance(target_minus_top1, (int, float)) and float(target_minus_top1) < 0.0
    positive_attention_but_poor = bool(
        target_attention_positive and available and (poor_rank or negative_logit_margin)
    )
    return {
        "case_id": _required_str(phase4_row, "case_id"),
        "rescue_tier": phase4_row.get("rescue_tier"),
        "mechanism_bucket": phase4_row.get("mechanism_bucket"),
        "intervention_kind": phase4_row.get("intervention_kind"),
        "valid_paired_row": phase4_row.get("valid_paired_row"),
        "target_attention_mass": phase4_row.get("target_attention_mass"),
        "competitor_attention_mass": phase4_row.get("competitor_attention_mass"),
        "target_minus_competitor_instance_attention": target_attention_margin,
        "target_attention_positive": target_attention_positive,
        "role": probe_row.get("role"),
        "layer_group": probe_row.get("layer_group"),
        "layer": probe_row.get("layer"),
        "x1_logit_lens_available": available,
        "x1_logit_lens_rank": rank,
        "x1_logit_lens_top1_bin": probe_row.get("x1_logit_lens_top1_bin"),
        "x1_logit_lens_target_minus_top1": target_minus_top1,
        "x1_logit_lens_target_logit": probe_row.get("x1_logit_lens_target_logit"),
        "x1_logit_lens_top1_logit": probe_row.get("x1_logit_lens_top1_logit"),
        "target_x1_bin": probe_row.get("target_x1_bin"),
        "lane_c_x1_target_rank": probe_row.get("x1_target_rank"),
        "lane_c_x1_top_peak_attribution": probe_row.get("x1_top_peak_attribution"),
        "poor_x1_logit_rank": poor_rank if available else None,
        "negative_x1_target_minus_top1": negative_logit_margin if available else None,
        "positive_attention_but_poor_x1_logit": positive_attention_but_poor,
    }


def _coord_slot_joined_row(
    phase4_row: Mapping[str, Any],
    probe_row: Mapping[str, Any],
    *,
    config: Phase5Config,
    available: bool,
) -> dict[str, Any]:
    rank = probe_row.get("coord_slot_logit_lens_rank")
    target_minus_top1 = probe_row.get("coord_slot_logit_lens_target_minus_top1")
    target_attention_margin = phase4_row.get("target_minus_competitor_instance_attention")
    target_attention_positive = isinstance(target_attention_margin, (int, float)) and float(target_attention_margin) > 0.0
    poor_rank = isinstance(rank, int) and rank > config.x1_logit_probe.poor_rank_threshold
    negative_logit_margin = isinstance(target_minus_top1, (int, float)) and float(target_minus_top1) < 0.0
    return {
        "case_id": _required_str(phase4_row, "case_id"),
        "rescue_tier": phase4_row.get("rescue_tier"),
        "mechanism_bucket": phase4_row.get("mechanism_bucket"),
        "intervention_kind": phase4_row.get("intervention_kind"),
        "valid_paired_row": phase4_row.get("valid_paired_row"),
        "target_minus_competitor_instance_attention": target_attention_margin,
        "target_attention_positive": target_attention_positive,
        "role": probe_row.get("role"),
        "layer_group": probe_row.get("layer_group"),
        "layer": probe_row.get("layer"),
        "target_slot": probe_row.get("coord_slot_logit_lens_target_slot"),
        "coord_slot_logit_lens_available": available,
        "coord_slot_logit_lens_rank": rank,
        "coord_slot_logit_lens_top1_bin": probe_row.get("coord_slot_logit_lens_top1_bin"),
        "coord_slot_logit_lens_target_bin": probe_row.get("coord_slot_logit_lens_target_bin"),
        "coord_slot_logit_lens_target_minus_top1": target_minus_top1,
        "coord_slot_logit_lens_target_logit": probe_row.get("coord_slot_logit_lens_target_logit"),
        "coord_slot_logit_lens_top1_logit": probe_row.get("coord_slot_logit_lens_top1_logit"),
        "poor_coord_slot_logit_rank": poor_rank if available else None,
        "negative_coord_slot_target_minus_top1": negative_logit_margin if available else None,
        "positive_attention_but_poor_coord_slot_logit": bool(
            target_attention_positive and available and (poor_rank or negative_logit_margin)
        ),
    }


def _role_layer_coverage(
    config: Phase5Config,
    coverage: Mapping[tuple[str, str], Mapping[str, int]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for role in config.x1_logit_probe.requested_roles:
        for layer_group in config.x1_logit_probe.layer_groups:
            counts = coverage.get((role, layer_group), {})
            row_count = int(counts.get("rows", 0))
            available_rows = int(counts.get("available_rows", 0))
            rows.append(
                {
                    "role": role,
                    "layer_group": layer_group,
                    "rows": row_count,
                    "available_rows": available_rows,
                    "has_state_dependent_x1_logit_lens": available_rows > 0,
                }
            )
    return rows


def _missing_requested_roles(
    config: Phase5Config,
    coverage: Mapping[tuple[str, str], Mapping[str, int]],
) -> list[str]:
    missing: list[str] = []
    for role in config.x1_logit_probe.requested_roles:
        available = sum(int(coverage.get((role, layer_group), {}).get("available_rows", 0)) for layer_group in config.x1_logit_probe.layer_groups)
        if available == 0:
            missing.append(role)
    return missing


def _bucket_summaries(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("mechanism_bucket") or "unknown")].append(row)
    return {bucket: _summary_for_rows(items) for bucket, items in sorted(grouped.items())}


def _role_layer_summaries(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = f"{row.get('role')}::{row.get('layer_group')}"
        grouped[key].append(row)
    return {key: _summary_for_rows(items) for key, items in sorted(grouped.items())}


def _role_slot_summaries(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = f"{row.get('role')}::{row.get('target_slot')}"
        grouped[key].append(row)
    return {key: _coord_slot_summary_for_rows(items) for key, items in sorted(grouped.items())}


def _unique_available_probe_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    unique: dict[tuple[str, str, str, int], Mapping[str, Any]] = {}
    for row in rows:
        if row.get("x1_logit_lens_available") is not True:
            continue
        layer = row.get("layer")
        if not isinstance(layer, int):
            continue
        key = (
            str(row.get("case_id") or ""),
            str(row.get("role") or ""),
            str(row.get("layer_group") or ""),
            layer,
        )
        unique[key] = row
    return [unique[key] for key in sorted(unique)]


def _unique_available_coord_slot_probe_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    unique: dict[tuple[str, str, str, str, int], Mapping[str, Any]] = {}
    for row in rows:
        if row.get("coord_slot_logit_lens_available") is not True:
            continue
        layer = row.get("layer")
        if not isinstance(layer, int):
            continue
        key = (
            str(row.get("case_id") or ""),
            str(row.get("role") or ""),
            str(row.get("target_slot") or ""),
            str(row.get("layer_group") or ""),
            layer,
        )
        unique[key] = row
    return [unique[key] for key in sorted(unique)]


def _summary_for_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    available = [row for row in rows if row.get("x1_logit_lens_available") is True]
    ranks = [int(row["x1_logit_lens_rank"]) for row in available if isinstance(row.get("x1_logit_lens_rank"), int)]
    margins = [
        float(row["x1_logit_lens_target_minus_top1"])
        for row in available
        if isinstance(row.get("x1_logit_lens_target_minus_top1"), (int, float))
    ]
    return {
        "rows": len(rows),
        "available_rows": len(available),
        "mean_x1_logit_lens_rank": _mean(ranks),
        "median_x1_logit_lens_rank": _median(ranks),
        "mean_x1_target_minus_top1": _mean(margins),
        "negative_target_minus_top1_fraction": (
            None if not margins else sum(1 for value in margins if value < 0.0) / len(margins)
        ),
        "positive_attention_but_poor_x1_logit_rows": sum(
            1 for row in rows if row.get("positive_attention_but_poor_x1_logit") is True
        ),
    }


def _critical_slice(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    available = [row for row in rows if row.get("x1_logit_lens_available") is True]
    critical = [row for row in available if row.get("positive_attention_but_poor_x1_logit") is True]
    return {
        "available_rows": len(available),
        "positive_attention_but_poor_x1_logit_rows": len(critical),
        "fraction": None if not available else len(critical) / len(available),
    }


def _coord_slot_bucket_summaries(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("mechanism_bucket") or "unknown")].append(row)
    return {bucket: _coord_slot_summary_for_rows(items) for bucket, items in sorted(grouped.items())}


def _coord_slot_summary_for_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    available = [row for row in rows if row.get("coord_slot_logit_lens_available") is True]
    ranks = [
        int(row["coord_slot_logit_lens_rank"])
        for row in available
        if isinstance(row.get("coord_slot_logit_lens_rank"), int)
    ]
    margins = [
        float(row["coord_slot_logit_lens_target_minus_top1"])
        for row in available
        if isinstance(row.get("coord_slot_logit_lens_target_minus_top1"), (int, float))
    ]
    return {
        "rows": len(rows),
        "available_rows": len(available),
        "mean_coord_slot_logit_lens_rank": _mean(ranks),
        "median_coord_slot_logit_lens_rank": _median(ranks),
        "mean_coord_slot_target_minus_top1": _mean(margins),
        "negative_target_minus_top1_fraction": (
            None if not margins else sum(1 for value in margins if value < 0.0) / len(margins)
        ),
        "positive_attention_but_poor_coord_slot_logit_rows": sum(
            1 for row in rows if row.get("positive_attention_but_poor_coord_slot_logit") is True
        ),
    }


def _coord_slot_critical_slice(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    available = [row for row in rows if row.get("coord_slot_logit_lens_available") is True]
    critical = [row for row in available if row.get("positive_attention_but_poor_coord_slot_logit") is True]
    return {
        "available_rows": len(available),
        "positive_attention_but_poor_coord_slot_logit_rows": len(critical),
        "fraction": None if not available else len(critical) / len(available),
    }


def _schema_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    lens_fields = (
        "x1_logit_lens_rank",
        "x1_logit_lens_top1_bin",
        "x1_logit_lens_target_minus_top1",
    )
    return {
        "state_dependent_fields": list(lens_fields),
        "case_level_metadata_fields": ["lane_c_x1_target_rank", "lane_c_x1_top_peak_attribution"],
        "rows_with_available_lens": sum(1 for row in rows if row.get("x1_logit_lens_available") is True),
        "rows_with_lane_c_metadata": sum(
            1
            for row in rows
            if row.get("lane_c_x1_target_rank") is not None or row.get("lane_c_x1_top_peak_attribution") is not None
        ),
    }


def _write_probe_report(path: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# X1 Logit Binding Probe",
        "",
        f"- Rows: `{summary.get('row_count')}`",
        f"- Available logit rows: `{summary.get('available_logit_rows')}`",
        f"- Joined cases: `{summary.get('joined_case_count')}`",
        f"- Missing requested roles: `{summary.get('missing_requested_roles')}`",
        f"- Critical slice: `{summary.get('critical_slice')}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_coord_slot_report(path: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# Coord-Slot Logit Binding Probe",
        "",
        f"- Rows: `{summary.get('row_count')}`",
        f"- Available logit rows: `{summary.get('available_logit_rows')}`",
        f"- Joined cases: `{summary.get('joined_case_count')}`",
        f"- Critical slice: `{summary.get('critical_slice')}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
            if not isinstance(row, Mapping):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            yield dict(row)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _required_mapping(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return value


def _optional_mapping(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return value


def _required_path(raw: Mapping[str, Any], key: str) -> Path:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"paths.{key} must be a nonempty string")
    return Path(value)


def _required_str(raw: Mapping[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a nonempty string")
    return value


def _string_tuple(value: Any, key: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{key} must be a sequence of strings")
    parsed = tuple(str(item) for item in value if str(item))
    if not parsed:
        raise ValueError(f"{key} must not be empty")
    return parsed


def _nonnegative_int(value: Any, key: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{key} must be a nonnegative integer")
    return value


def _positive_int(value: Any, key: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _validate_stages(stages: Sequence[str]) -> None:
    unknown = sorted(set(stages) - set(PHASE5_STAGES))
    if unknown:
        raise ValueError(f"unknown phase5 stage(s): {', '.join(unknown)}")


def _mean(values: Sequence[float | int]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _median(values: Sequence[int]) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2 == 1:
        return float(sorted_values[mid])
    return float((sorted_values[mid - 1] + sorted_values[mid]) / 2)
