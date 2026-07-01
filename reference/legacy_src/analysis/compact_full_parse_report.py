from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.metrics.events import flatten_metric_events
from src.training.teacher_forcing.metrics import compact_full_parse_error_events


def build_compact_full_parse_summary(parse_jsonl: Path) -> dict[str, Any]:
    rows = _read_jsonl(parse_jsonl)
    error_counts: Counter[str] = Counter()
    ok_count = 0
    for row in rows:
        if bool(row.get("ok", False)):
            ok_count += 1
            continue
        code = row.get("error_code") or row.get("error") or row.get("reason") or "unknown"
        error_counts[str(code).strip() or "unknown"] += 1

    metrics = flatten_metric_events(compact_full_parse_error_events(error_counts))
    return {
        "artifact": str(parse_jsonl),
        "rows": len(rows),
        "ok_count": ok_count,
        "error_counts": dict(sorted(error_counts.items())),
        "metrics": metrics,
    }


def write_compact_full_parse_report(parse_jsonl: Path, output_dir: Path) -> dict[str, str]:
    summary = build_compact_full_parse_summary(parse_jsonl)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    _write_json(summary_path, summary)
    report_path.write_text(_render_markdown(summary), encoding="utf-8")
    return {"summary_json": str(summary_path), "report_md": str(report_path)}


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
        "# Compact-Full Parse Diagnostics",
        "",
        f"- rows: {summary['rows']}",
        f"- ok rows: {summary['ok_count']}",
        "- errors:",
    ]
    for code, count in summary["error_counts"].items():
        lines.append(f"  - `{code}`: {count}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parse-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    bundle = write_compact_full_parse_report(args.parse_jsonl, args.output_dir)
    print(json.dumps(bundle, sort_keys=True))


if __name__ == "__main__":
    main()
