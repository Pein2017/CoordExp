from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import PHASE_ID, PROJECT_ID, RUN_ID, SCHEMA_VERSION


RANDOM_ROLE = "fullobj_random_pure_ce_ckpt3668"
SORTED_ROLE = "fullobj_sorted_pure_ce_ckpt3668"
EXPECTED_A32_ROLES = (RANDOM_ROLE, SORTED_ROLE)

LEGACY_A31_LABELS = (
    "pure_minus_et",
    "et_rmp_ce",
    "ckpt3664",
    "phase_a3_1",
)
BANNED_CAUSAL_PHRASES = (
    "proved",
    "root cause",
    "visual encoder failed",
    "sorted solves recall",
    "multiple positive is unnecessary",
)

DELTA_METRICS = (
    (
        "residual_vs_eos_margin",
        "sorted_minus_random_residual_vs_eos_margin",
    ),
    (
        "strict_r95_x1_hit_rate",
        "sorted_minus_random_strict_r95_x1_hit_rate",
    ),
    (
        "boundary_residual_favored_rate",
        "sorted_minus_random_boundary_residual_favored_rate",
    ),
)

EXTERNAL_VAL200_CONTEXT = (
    ("AP@[.50:.95]", "+0.0158", "0.4072 - 0.3914"),
    ("AP50", "+0.0291", "0.5642 - 0.5351"),
    ("AP75", "+0.0173", "0.4226 - 0.4054"),
    ("AR100", "+0.0338", "0.4836 - 0.4498"),
    ("F1-ish@0.50", "+0.1233", "0.5892 - 0.4659"),
    ("recall@0.50", "+0.0686", "0.5637 - 0.4952"),
    ("precision@0.50", "+0.1771", "0.6171 - 0.4400"),
    ("FN@0.50", "-99", "630 - 729"),
    ("FP@0.50", "-405", "505 - 910"),
)


def merge_prefix_readout_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    role_a: str = RANDOM_ROLE,
    role_b: str = SORTED_ROLE,
) -> list[dict[str, Any]]:
    """Merge paired readout rows into explicit sorted-minus-random rows."""

    _validate_a32_roles(role_a, role_b)
    validate_no_legacy_labels(rows)
    grouped: dict[str, dict[str, Mapping[str, Any]]] = {}
    for index, row in enumerate(rows):
        checkpoint_role = str(row.get("checkpoint_role", ""))
        if checkpoint_role not in EXPECTED_A32_ROLES:
            raise ValueError(
                f"rows[{index}].checkpoint_role must be one of {EXPECTED_A32_ROLES}; "
                f"got {checkpoint_role!r}"
            )
        prefix_state_id = str(row.get("prefix_state_id", ""))
        if not prefix_state_id:
            raise ValueError(f"rows[{index}].prefix_state_id is required")
        role_rows = grouped.setdefault(prefix_state_id, {})
        if checkpoint_role in role_rows:
            raise ValueError(
                f"duplicate readout row for prefix_state_id {prefix_state_id} "
                f"and checkpoint_role {checkpoint_role}"
            )
        role_rows[checkpoint_role] = row

    merged_rows: list[dict[str, Any]] = []
    for prefix_state_id in sorted(grouped):
        role_rows = grouped[prefix_state_id]
        missing = [role for role in EXPECTED_A32_ROLES if role not in role_rows]
        if missing:
            raise ValueError(
                f"missing paired readout rows for prefix_state_id {prefix_state_id}: "
                f"{', '.join(missing)}"
            )
        random_row = role_rows[role_a]
        sorted_row = role_rows[role_b]
        merged: dict[str, Any] = {
            "project_id": PROJECT_ID,
            "phase_id": PHASE_ID,
            "schema_version": SCHEMA_VERSION,
            "run_id": RUN_ID,
            "prefix_state_id": prefix_state_id,
            "role_a": role_a,
            "role_b": role_b,
            "checkpoint_roles": [role_a, role_b],
            "delta_role": "sorted_minus_random",
            "random_row": random_row,
            "sorted_row": sorted_row,
        }
        for metric_name, delta_name in DELTA_METRICS:
            merged[delta_name] = _metric_value(sorted_row, metric_name) - _metric_value(
                random_row,
                metric_name,
            )
        merged_rows.append(_json_safe(merged, "merged_row"))
    validate_no_legacy_labels(merged_rows)
    return merged_rows


def build_report_markdown(
    *,
    evidence_labels: Mapping[str, Any],
    checkpoint_provenance: Mapping[str, Any],
    data_roots: Mapping[str, Any],
    template_contract: Mapping[str, Any],
    prefix_readout_rows: Sequence[Mapping[str, Any]],
    rollout_summary: Mapping[str, Any],
    fn_universe_counts: Mapping[str, Any],
    fn_bucket_summary: Mapping[str, Any],
    prefix_sensitivity: Mapping[str, Any],
    interpretive_caveats: Sequence[str] | None = None,
) -> str:
    """Build the A3.2 report with explicit evidence scope and cautious language."""

    payloads = (
        evidence_labels,
        checkpoint_provenance,
        data_roots,
        template_contract,
        prefix_readout_rows,
        rollout_summary,
        fn_universe_counts,
        fn_bucket_summary,
        prefix_sensitivity,
        interpretive_caveats or (),
    )
    validate_no_legacy_labels(payloads)
    _validate_report_roles(checkpoint_provenance)

    prefix_summary = _prefix_readout_summary(prefix_readout_rows)
    caveats = list(interpretive_caveats or _default_interpretive_caveats())
    sections = [
        "# A3.2 Sorted Vs Random No-Newline Phenotype Report",
        "## Scope And Evidence Labels",
        _bullets(evidence_labels)
        or "- No evidence labels were provided for this materialization.",
        (
            "Evidence is scoped to local len12000 mechanism probes and supports "
            "under this evidence scope only."
        ),
        "## Checkpoint Provenance",
        _bullets(checkpoint_provenance)
        or "- Checkpoint provenance is not available in this materialization.",
        "## Data And Image Roots",
        _bullets(data_roots)
        or "- Data and image roots are not available in this materialization.",
        "## External Offline Val-200 Eval Context",
        (
            "The external offline val-200 detector metrics show sorted > random "
            "across AP/AR/F1/FN/FP. These numbers are phenotype context, not "
            "mechanism proof."
        ),
        _external_val200_table(),
        "## Template Contract",
        _bullets(template_contract)
        or "- Template contract is not available in this materialization.",
        "## Prefix Readout",
        _bullets(prefix_summary)
        or "- No merged prefix readout rows are available yet.",
        (
            "The prefix readout is consistent with checkpoint-dependent transition "
            "differences when interpreted with the canonical sorted teacher-prefix "
            "conditioning."
        ),
        "## Native Rollout Phenotype",
        _bullets(rollout_summary)
        or "- Native rollout phenotype is not available yet.",
        "## FN Universe Counts",
        _bullets(fn_universe_counts) or "- FN universe counts are not available yet.",
        "## FN Multi-Axis Bucket Counts",
        _bullets(fn_bucket_summary)
        or "- FN multi-axis bucket counts are not available yet.",
        "## Prefix Sensitivity",
        _bullets(prefix_sensitivity)
        or "- Prefix sensitivity summary is not available yet.",
        "## A3.1 Compatibility And Caveats",
        (
            "Metric compatibility is historical and descriptive only: A3.2 uses a "
            "new ckpt3668 random-vs-sorted pair, canonical sorted teacher-prefix "
            "readout, and no-newline compact-full prompting."
        ),
        "## Interpretive Caveats",
        "\n".join(f"- {caveat}" for caveat in caveats),
    ]
    report = "\n\n".join(section.rstrip() for section in sections) + "\n"
    validate_no_legacy_labels(report)
    validate_cautious_language(report)
    return report


def write_report(path: str | Path, report_markdown: str) -> Path:
    validate_no_legacy_labels(report_markdown)
    validate_cautious_language(report_markdown)
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_markdown, encoding="utf-8")
    return report_path


def write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    validate_no_legacy_labels(rows)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            safe_row = _json_safe(row, f"rows[{index}]")
            handle.write(json.dumps(safe_row, allow_nan=False, sort_keys=True))
            handle.write("\n")
    return out_path


def validate_no_legacy_labels(value: Any) -> None:
    text = value if isinstance(value, str) else json.dumps(_json_safe(value, "value"))
    for label in LEGACY_A31_LABELS:
        if label in text:
            raise ValueError(f"legacy A3.1 label is not allowed in A3.2 artifacts: {label}")


def validate_cautious_language(text: str) -> None:
    lowered = text.lower()
    for phrase in BANNED_CAUSAL_PHRASES:
        pattern = rf"\b{re.escape(phrase.lower())}\b"
        if re.search(pattern, lowered):
            raise ValueError(
                f"unqualified causal language is not allowed in A3.2 reports: {phrase}"
            )


def _validate_a32_roles(role_a: str, role_b: str) -> None:
    if (role_a, role_b) != EXPECTED_A32_ROLES:
        raise ValueError(
            "A3.2 merge roles must be "
            f"role_a={RANDOM_ROLE!r}, role_b={SORTED_ROLE!r}"
        )


def _validate_report_roles(checkpoint_provenance: Mapping[str, Any]) -> None:
    if not checkpoint_provenance:
        return
    roles = tuple(str(role) for role in checkpoint_provenance)
    if set(roles) != set(EXPECTED_A32_ROLES):
        raise ValueError(
            f"checkpoint provenance roles must be exactly {EXPECTED_A32_ROLES}; "
            f"got {roles}"
        )


def _metric_value(row: Mapping[str, Any], metric_name: str) -> float:
    if metric_name in row:
        return _finite_float(row[metric_name], metric_name)
    boundary_summary = row.get("boundary_summary")
    if isinstance(boundary_summary, Mapping) and metric_name in boundary_summary:
        return _finite_float(boundary_summary[metric_name], f"boundary_summary.{metric_name}")
    raise ValueError(
        f"readout row for prefix_state_id={row.get('prefix_state_id')!r} "
        f"checkpoint_role={row.get('checkpoint_role')!r} is missing metric {metric_name}"
    )


def _finite_float(value: Any, path: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{path} must be finite")
    return number


def _prefix_readout_summary(
    prefix_readout_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not prefix_readout_rows:
        return {}
    summary: dict[str, Any] = {"merged_prefix_state_count": len(prefix_readout_rows)}
    for _, delta_name in DELTA_METRICS:
        values = [
            _finite_float(row[delta_name], delta_name)
            for row in prefix_readout_rows
            if delta_name in row
        ]
        if values:
            summary[f"mean_{delta_name}"] = sum(values) / len(values)
    return _json_safe(summary, "prefix_readout_summary")


def _external_val200_table() -> str:
    lines = ["| Metric | Sorted Minus Random | Source Arithmetic |", "| --- | ---: | --- |"]
    for metric, delta, arithmetic in EXTERNAL_VAL200_CONTEXT:
        lines.append(f"| {metric} | {delta} | `{arithmetic}` |")
    return "\n".join(lines)


def _default_interpretive_caveats() -> tuple[str, ...]:
    return (
        "Findings are consistent with the measured artifacts, but remain scoped to local mechanism probes.",
        "External val-200 metrics provide phenotype context and do not by themselves identify a mechanism.",
        "Prefix readout and FN probes support under this evidence scope; production-scale validation remains separate.",
    )


def _bullets(mapping: Mapping[str, Any]) -> str:
    if not mapping:
        return ""
    lines = []
    for key in sorted(mapping):
        value = mapping[key]
        if isinstance(value, Mapping):
            rendered = json.dumps(_json_safe(value, str(key)), allow_nan=False, sort_keys=True)
        else:
            rendered = str(_json_safe(value, str(key)))
        lines.append(f"- `{key}`: {rendered}")
    return "\n".join(lines)


def _json_safe(value: Any, path: str) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item, f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, tuple | list):
        return [_json_safe(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must be finite")
        return value
    if isinstance(value, str | int):
        return value
    return str(value)


__all__ = [
    "BANNED_CAUSAL_PHRASES",
    "DELTA_METRICS",
    "EXPECTED_A32_ROLES",
    "EXTERNAL_VAL200_CONTEXT",
    "LEGACY_A31_LABELS",
    "RANDOM_ROLE",
    "SORTED_ROLE",
    "build_report_markdown",
    "merge_prefix_readout_rows",
    "validate_cautious_language",
    "validate_no_legacy_labels",
    "write_jsonl",
    "write_report",
]
