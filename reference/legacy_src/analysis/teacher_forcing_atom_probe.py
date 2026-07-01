from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.metrics.events import flatten_metric_events
from src.training.teacher_forcing.metrics import teacher_forcing_diagnostic_events


def build_atom_probe_summary(atom_jsonl: Path) -> dict[str, Any]:
    rows = _read_jsonl(atom_jsonl)
    coordinate_onset_count = 0
    text_count = 0
    mixed_role_pairs: Counter[str] = Counter()

    for row in rows:
        for ambiguity in _ambiguities(row):
            kind = str(ambiguity.get("kind", "")).strip().lower()
            roles = _normalized_roles(ambiguity.get("roles"))
            if kind == "coordinate_onset":
                coordinate_onset_count += 1
            elif kind == "text":
                text_count += 1
            if len(roles) > 1:
                mixed_role_pairs["|".join(sorted(roles))] += 1

    mixed_role_count = sum(mixed_role_pairs.values())
    metrics = flatten_metric_events(
        teacher_forcing_diagnostic_events(
            coordinate_onset_count=coordinate_onset_count,
            mixed_role_count=mixed_role_count,
        )
    )
    return {
        "artifact": str(atom_jsonl),
        "rows": len(rows),
        "ambiguity": {
            "coordinate_onset_count": coordinate_onset_count,
            "text_count": text_count,
            "mixed_role_count": mixed_role_count,
            "mixed_role_pairs": dict(sorted(mixed_role_pairs.items())),
        },
        "metrics": metrics,
    }


def write_atom_probe_report(atom_jsonl: Path, output_dir: Path) -> dict[str, str]:
    summary = build_atom_probe_summary(atom_jsonl)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    _write_json(summary_path, summary)
    report_path.write_text(_render_atom_probe_markdown(summary), encoding="utf-8")
    return {"summary_json": str(summary_path), "report_md": str(report_path)}


def _ambiguities(row: dict[str, Any]) -> list[dict[str, Any]]:
    value = row.get("ambiguities", [])
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _normalized_roles(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    roles = []
    for item in value:
        text = str(item).strip().upper()
        if text:
            roles.append(text)
    return tuple(dict.fromkeys(roles))


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


def _render_atom_probe_markdown(summary: dict[str, Any]) -> str:
    ambiguity = summary["ambiguity"]
    return "\n".join(
        [
            "# Teacher-Forcing Atom Probe",
            "",
            f"- rows: {summary['rows']}",
            f"- coordinate onset ambiguities: {ambiguity['coordinate_onset_count']}",
            f"- text ambiguities: {ambiguity['text_count']}",
            f"- mixed-role ambiguities: {ambiguity['mixed_role_count']}",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atom-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    bundle = write_atom_probe_report(args.atom_jsonl, args.output_dir)
    print(json.dumps(bundle, sort_keys=True))


if __name__ == "__main__":
    main()
