from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .jsonl import read_jsonl, write_jsonl


PRIORITY_QUADRANTS = {"boundary_bad_x1_good", "boundary_good_x1_bad"}


def select_gallery_rows(
    quadrant_rows: Sequence[Mapping[str, Any]],
    *,
    max_rows: int = 128,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in quadrant_rows:
        grouped.setdefault(str(row.get("paired_key")), []).append(row)
    scored: list[tuple[tuple[int, float, str], dict[str, Any]]] = []
    for paired_key, rows in grouped.items():
        quadrants = {str(row.get("checkpoint_role")): str(row.get("quadrant")) for row in rows}
        disagreement = len(set(quadrants.values())) > 1
        priority_quadrant = any(q in PRIORITY_QUADRANTS for q in quadrants.values())
        emitted = max(float(row.get("emitted_attraction_rate", 0.0)) for row in rows)
        class_block = any(str(row.get("prefix_depth")) == "class_block_done" for row in rows)
        priority = 0
        if priority_quadrant:
            priority += 100
        if disagreement:
            priority += 50
        if class_block:
            priority += 10
        representative = dict(rows[0])
        representative.update(
            {
                "gallery_case_id": f"gallery-{len(scored):05d}",
                "paired_key": paired_key,
                "quadrant_et_rmp_ce": quadrants.get("et_rmp_ce"),
                "quadrant_pure_ce": quadrants.get("pure_ce"),
                "max_emitted_attraction_rate": emitted,
                "manual_label": "",
                "manual_notes": "",
            }
        )
        scored.append(((priority, emitted, paired_key), representative))
    scored.sort(key=lambda item: (-item[0][0], -item[0][1], item[0][2]))
    return [row for _, row in scored[:max_rows]]


def materialize_gallery(*, artifact_root: Path, max_rows: int = 128) -> dict[str, Any]:
    root = Path(artifact_root)
    quadrant_path = root / "quadrant_rows.jsonl"
    rows = read_jsonl(quadrant_path) if quadrant_path.exists() else []
    selected = select_gallery_rows(rows, max_rows=max_rows)
    gallery_root = root / "gallery"
    gallery_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(gallery_root / "gallery_rows.jsonl", selected)
    index = _render_index(selected)
    (gallery_root / "index.md").write_text(index, encoding="utf-8")
    summary = {
        "gallery_rows": len(selected),
        "max_rows": max_rows,
        "gallery_root": str(gallery_root),
    }
    (gallery_root / "gallery_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def _render_index(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Prefix-State Transition Gallery",
        "",
        "| Case | Split | Transition | Prefix Depth | ET Quadrant | Pure Quadrant |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {case} | {split} | {transition} | {depth} | {et} | {pure} |".format(
                case=row.get("gallery_case_id", ""),
                split=row.get("split", ""),
                transition=row.get("transition_type", ""),
                depth=row.get("prefix_depth", ""),
                et=row.get("quadrant_et_rmp_ce", ""),
                pure=row.get("quadrant_pure_ce", ""),
            )
        )
    return "\n".join(lines) + "\n"


__all__ = ["PRIORITY_QUADRANTS", "materialize_gallery", "select_gallery_rows"]

