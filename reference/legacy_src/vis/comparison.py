"""Shared canonical comparison helpers for GT-vs-Pred visualization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from .gt_vs_pred import canonicalize_gt_vs_pred_record


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                yield payload


def _require_equal_identity(
    *,
    base_label: str,
    base_record: Mapping[str, Any],
    other_label: str,
    other_record: Mapping[str, Any],
) -> None:
    base_record_idx = int(base_record.get("record_idx") or 0)
    for field in ("image_id", "file_name", "image"):
        base_value = base_record.get(field)
        other_value = other_record.get(field)
        if base_value is None or other_value is None:
            continue
        if base_value != other_value:
            raise ValueError(
                f"Comparison alignment failure at record_idx={base_record_idx}: "
                f"{field} mismatch between '{base_label}' and '{other_label}' "
                f"({base_value!r} != {other_value!r})"
            )


def compose_comparison_scene_members(
    members_by_label: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    labels = list(members_by_label.keys())
    if not labels:
        raise ValueError("Comparison composition requires at least one member")

    base_label = labels[0]
    base_record = members_by_label[base_label]
    base_record_idx = int(base_record.get("record_idx") or 0)
    base_width = int(base_record["width"])
    base_height = int(base_record["height"])
    base_gt = list(base_record.get("gt") or [])

    members: List[Dict[str, Any]] = []
    for label in labels:
        record = members_by_label[label]
        if int(record.get("record_idx") or 0) != base_record_idx:
            raise ValueError(
                f"Comparison alignment failure: record_idx mismatch for member '{label}'"
            )
        if int(record["width"]) != base_width or int(record["height"]) != base_height:
            raise ValueError(
                f"Comparison alignment failure at record_idx={base_record_idx}: "
                f"size mismatch for member '{label}'"
            )
        _require_equal_identity(
            base_label=base_label,
            base_record=base_record,
            other_label=label,
            other_record=record,
        )
        if list(record.get("gt") or []) != base_gt:
            raise ValueError(
                f"Comparison alignment failure at record_idx={base_record_idx}: "
                f"canonical GT mismatch for member '{label}'"
            )
        members.append(
            {
                "label": label,
                "record": dict(record),
            }
        )

    scene: Dict[str, Any] = {
        "record_idx": base_record_idx,
        "image": base_record.get("image"),
        "width": base_width,
        "height": base_height,
        "gt": base_gt,
        "members": members,
    }
    for field in ("image_id", "file_name"):
        value = base_record.get(field)
        if value is not None:
            scene[field] = value
    return scene


def compose_comparison_scenes_from_jsonls(
    member_jsonls: Mapping[str, Path],
) -> List[Dict[str, Any]]:
    labels = list(member_jsonls.keys())
    if len(labels) < 2:
        raise ValueError("Comparison composition requires at least two member JSONLs")

    canonical_by_label: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for label in labels:
        jsonl_path = Path(member_jsonls[label])
        records: Dict[int, Dict[str, Any]] = {}
        for fallback_record_idx, record in enumerate(_iter_jsonl(jsonl_path)):
            canonical = canonicalize_gt_vs_pred_record(
                record,
                fallback_record_idx=fallback_record_idx,
                source_kind=f"comparison_member_{label}",
                explicit_matching=None,
                materialize_matching=True,
            )
            record_idx = int(canonical["record_idx"])
            if record_idx in records:
                raise ValueError(
                    f"Comparison input '{label}' contains duplicate record_idx={record_idx}"
                )
            records[record_idx] = canonical
        canonical_by_label[label] = records

    common_record_indices = set(canonical_by_label[labels[0]].keys())
    for label in labels[1:]:
        common_record_indices &= set(canonical_by_label[label].keys())
    if not common_record_indices:
        raise ValueError("Comparison composition found no common record_idx values")

    scenes: List[Dict[str, Any]] = []
    for record_idx in sorted(common_record_indices):
        members = {
            label: canonical_by_label[label][record_idx]
            for label in labels
        }
        scenes.append(compose_comparison_scene_members(members))
    return scenes


__all__ = [
    "compose_comparison_scene_members",
    "compose_comparison_scenes_from_jsonls",
]
