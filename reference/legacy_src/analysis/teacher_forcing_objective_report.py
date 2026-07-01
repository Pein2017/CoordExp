from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.metrics.events import flatten_metric_events
from src.training.teacher_forcing.metrics import (
    builder_rejection_events,
    decode_quality_events,
)

from .teacher_forcing_atom_probe import build_atom_probe_summary


def build_objective_report_bundle(
    *,
    metrics_jsonl: Path,
    output_dir: Path,
    atom_probe_jsonl: Path | None = None,
    builder_jsonl: Path | None = None,
    decode_jsonl: Path | None = None,
) -> dict[str, str]:
    summary = build_objective_report_summary(
        metrics_jsonl=metrics_jsonl,
        atom_probe_jsonl=atom_probe_jsonl,
        builder_jsonl=builder_jsonl,
        decode_jsonl=decode_jsonl,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    _write_json(summary_path, summary)
    report_path.write_text(_render_markdown(summary), encoding="utf-8")
    return {"summary_json": str(summary_path), "report_md": str(report_path)}


def build_objective_report_summary(
    *,
    metrics_jsonl: Path,
    atom_probe_jsonl: Path | None = None,
    builder_jsonl: Path | None = None,
    decode_jsonl: Path | None = None,
) -> dict[str, Any]:
    metrics = _aggregate_metric_jsonl(metrics_jsonl)

    atom_probe: dict[str, Any] | None = None
    if atom_probe_jsonl is not None:
        atom_probe = build_atom_probe_summary(atom_probe_jsonl)
        metrics.update(atom_probe["metrics"])

    builder = _builder_summary(builder_jsonl)
    metrics.update(builder["metrics"])

    decode = _decode_summary(decode_jsonl)
    metrics.update(decode["metrics"])

    return {
        "inputs": {
            "metrics_jsonl": str(metrics_jsonl),
            "atom_probe_jsonl": str(atom_probe_jsonl) if atom_probe_jsonl else None,
            "builder_jsonl": str(builder_jsonl) if builder_jsonl else None,
            "decode_jsonl": str(decode_jsonl) if decode_jsonl else None,
        },
        "metrics": dict(sorted(metrics.items())),
        "metric_semantics": {
            "flat_input_metrics": (
                "count-like keys are summed; non-count flat metrics are "
                "reported with a /macro_avg suffix because source denominators "
                "are not available in flat JSONL rows"
            )
        },
        "atom_probe": atom_probe,
        "builder": {
            "artifact_status": builder["artifact_status"],
            "rejected_samples": builder["rejected_samples"],
            "rejection_reason_counts": builder["rejection_reason_counts"],
        },
        "decode": {
            "artifact_status": decode["artifact_status"],
            "sample_count": decode["sample_count"],
        },
    }


def _aggregate_metric_jsonl(path: Path) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in _read_jsonl(path):
        metrics = row.get("metrics", row)
        if not isinstance(metrics, dict):
            continue
        for key, value in metrics.items():
            if not isinstance(key, str) or not key.startswith(
                ("teacher_forcing/", "infer/parse/compact_full/")
            ):
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            values[key].append(float(value))

    reduced: dict[str, float] = {}
    for key, key_values in values.items():
        if _is_count_metric(key):
            reduced[key] = float(sum(key_values))
        else:
            reduced[f"{key}/macro_avg"] = float(sum(key_values) / len(key_values))
    return reduced


def _builder_summary(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "artifact_status": "absent",
            "rejected_samples": 0,
            "rejection_reason_counts": {},
            "metrics": {},
        }

    reason_counts: Counter[str] = Counter()
    for row in _read_jsonl(path):
        reason = row.get("reason") or row.get("rejection_reason")
        if row.get("status") == "rejected" or reason is not None:
            reason_counts[str(reason or "unknown").strip() or "unknown"] += 1
    metrics = flatten_metric_events(builder_rejection_events(reason_counts))
    return {
        "artifact_status": "present",
        "rejected_samples": int(sum(reason_counts.values())),
        "rejection_reason_counts": dict(sorted(reason_counts.items())),
        "metrics": metrics,
    }


def _decode_summary(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"artifact_status": "absent", "sample_count": 0, "metrics": {}}

    rows = _read_jsonl(path)
    sample_count = len(rows)
    object_coherent = sum(_truthy(row.get("object_coherent")) for row in rows)
    duplicate = sum(_truthy(row.get("duplicate")) for row in rows)
    missed_object = sum(_truthy(row.get("missed_object")) for row in rows)
    malformed_sequence = sum(_truthy(row.get("malformed_sequence")) for row in rows)
    metrics = flatten_metric_events(
        decode_quality_events(
            artifact_present=True,
            object_coherent=object_coherent,
            duplicate=duplicate,
            missed_object=missed_object,
            malformed_sequence=malformed_sequence,
            total=sample_count,
        )
    )
    return {
        "artifact_status": "present",
        "sample_count": sample_count,
        "metrics": metrics,
    }


def _is_count_metric(key: str) -> bool:
    return (
        key.endswith("_count")
        or key.endswith("/rejected_samples")
        or "/rejection_reason/" in key
        or key.startswith("infer/parse/compact_full/error/")
    )


def _truthy(value: object) -> int:
    return 1 if bool(value) else 0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no} must contain a JSON object")
        rows.append(row)
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Teacher-Forcing Objective Diagnostics",
        "",
        f"- metric keys: {len(summary['metrics'])}",
        f"- builder artifact: {summary['builder']['artifact_status']}",
        f"- builder rejections: {summary['builder']['rejected_samples']}",
        f"- decode artifact: {summary['decode']['artifact_status']}",
        f"- decode rows: {summary['decode']['sample_count']}",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--atom-probe-jsonl", type=Path)
    parser.add_argument("--builder-jsonl", type=Path)
    parser.add_argument("--decode-jsonl", type=Path)
    args = parser.parse_args(argv)
    bundle = build_objective_report_bundle(
        metrics_jsonl=args.metrics_jsonl,
        output_dir=args.output_dir,
        atom_probe_jsonl=args.atom_probe_jsonl,
        builder_jsonl=args.builder_jsonl,
        decode_jsonl=args.decode_jsonl,
    )
    print(json.dumps(bundle, sort_keys=True))


if __name__ == "__main__":
    main()
