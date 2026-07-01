from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import SCHEMA_VERSION
from .jsonl import read_jsonl, write_jsonl


def select_gallery_rows(rows: Sequence[Mapping[str, Any]], *, max_rows: int = 32) -> list[dict[str, Any]]:
    def priority(row: Mapping[str, Any]) -> tuple[int, str]:
        competitor = str(row.get("winner_bucket")) == "same_desc_competitor"
        reference = str(row.get("comparison_role")) == "reference_anchor"
        return (0 if competitor and reference else 1 if competitor else 2, str(row.get("case_id")))

    selected = [dict(row) for row in sorted(rows, key=priority)[:max_rows]]
    for row in selected:
        if row.get("comparison_role") == "reference_anchor":
            row["reference_anchor_caveat"] = "template_objective_confounded_reference"
    return selected


def materialize_gallery(root: str | Path, *, max_rows: int = 32) -> dict[str, Any]:
    artifact_root = Path(root)
    rows = read_jsonl(artifact_root / "slot_posterior_rows.jsonl")
    selected = select_gallery_rows(rows, max_rows=max_rows)
    gallery_root = artifact_root / "gallery"
    gallery_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(gallery_root / "gallery_rows.jsonl", selected)
    lines = [
        "# A3.3 Gallery",
        "",
        "Mechanism examples only; reference_anchor means template_objective_confounded_reference.",
        "",
    ]
    for row in selected:
        contract = row.get("template_contract") or {}
        lines.extend(
            [
                f"## {row.get('case_id')} / {row.get('checkpoint_role')}",
                "",
                f"- winner_bucket: {row.get('winner_bucket')}",
                f"- template_contract: {contract.get('template_contract_id')}",
                f"- caveat: {row.get('reference_anchor_caveat', '')}",
                "",
            ]
        )
    (gallery_root / "index.md").write_text("\n".join(lines), encoding="utf-8")
    summary = {"artifact_schema_version": SCHEMA_VERSION, "gallery_rows": len(selected)}
    (gallery_root / "gallery_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


__all__ = ["materialize_gallery", "select_gallery_rows"]
