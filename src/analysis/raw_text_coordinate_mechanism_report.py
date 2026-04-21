from __future__ import annotations

import csv
import json
from pathlib import Path


def write_report_bundle(
    *,
    output_dir: Path,
    summary: dict[str, object],
    review_rows: list[dict[str, object]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        "# Raw-Text Coordinate Mechanism Report\n\n"
        f"- questions: {len(summary)}\n"
        f"- review_rows: {len(review_rows)}\n",
        encoding="utf-8",
    )
    if summary.get("review_gallery_path"):
        (output_dir / "report.md").write_text(
            "# Raw-Text Coordinate Mechanism Report\n\n"
            f"- questions: {len(summary)}\n"
            f"- review_rows: {len(review_rows)}\n"
            f"- review_gallery: {summary['review_gallery_path']}\n",
            encoding="utf-8",
        )
    with (output_dir / "review_queue.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(review_rows[0].keys()) if review_rows else ["case_uid"],
        )
        writer.writeheader()
        for row in review_rows:
            writer.writerow(row)
