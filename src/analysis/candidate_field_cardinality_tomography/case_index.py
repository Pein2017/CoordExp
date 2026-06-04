from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from . import PHASE_ID, PROJECT_ID, SCHEMA_VERSION


def canonical_desc(desc: str) -> str:
    return " ".join(str(desc).strip().lower().split())


def build_case_index(
    *,
    train_jsonl: Path,
    val_jsonl: Path,
    checkpoint_id: str,
    run_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, path in (("train", Path(train_jsonl)), ("val", Path(val_jsonl))):
        for source_line_idx, record in _iter_jsonl(path):
            groups: dict[str, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
            objects = record.get("objects") or []
            if not isinstance(objects, list):
                continue
            for gt_idx, obj in enumerate(objects):
                if not isinstance(obj, Mapping):
                    continue
                groups[canonical_desc(str(obj.get("desc") or ""))].append((gt_idx, obj))
            for desc_id, (desc_text, members) in enumerate(sorted(groups.items())):
                if not desc_text:
                    continue
                count = len(members)
                pool_role = _pool_role(count)
                if pool_role is None:
                    continue
                case_id = _case_id(split, record, desc_text, source_line_idx)
                for member_offset, (gt_idx, obj) in enumerate(members):
                    bbox = tuple(obj.get("bbox_2d") or ())
                    rows.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "project_id": PROJECT_ID,
                            "phase_id": PHASE_ID,
                            "run_id": run_id,
                            "checkpoint_id": checkpoint_id,
                            "case_id": case_id,
                            "case_index_row_id": f"{case_id}:gt{gt_idx}",
                            "split": split,
                            "pool_role": pool_role,
                            "source_dataset_jsonl": str(path),
                            "dataset_manifest_id": _dataset_manifest_id(path),
                            "dataset_manifest_sha256": _path_identity_hash(path),
                            "fn_rescue_overlay_membership": False,
                            "source_line_idx": source_line_idx,
                            "image_id": record.get("image_id"),
                            "image_path": record.get("file_name") or (record.get("images") or [""])[0],
                            "desc_id": desc_id,
                            "desc_text_raw": str(obj.get("desc") or ""),
                            "desc_text_stripped": str(obj.get("desc") or "").strip(),
                            "desc_text_canonical": desc_text,
                            "desc_text": desc_text,
                            "same_desc_cluster_id": f"{case_id}:cluster",
                            "same_desc_gt_count_annotated": count,
                            "same_desc_count_bucket": _same_desc_bucket(count),
                            "target_gt_idx": gt_idx,
                            "gt_idx": gt_idx,
                            "gt_member_offset": member_offset,
                            "bbox_tokens": list(bbox),
                            "bbox_xyxy": _bbox_values(bbox),
                            "object_size_bucket": _object_size_bucket(bbox),
                            "overlap_bucket": "unknown_overlap",
                            "gt_universe": "coco_annotated",
                        }
                    )
    summary = _summarize(rows)
    return rows, summary


def _iter_jsonl(path: Path):
    if not path.exists():
        return
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if line:
            yield idx, json.loads(line)


def _pool_role(count: int) -> str | None:
    if count >= 3:
        return "headline_crowded"
    if count == 2:
        return "same_desc_count_2_control"
    if count == 1:
        return "same_desc_count_1_control"
    return None


def _same_desc_bucket(count: int) -> str:
    if count == 1:
        return "same_desc_1"
    if count == 2:
        return "same_desc_2"
    if count == 3:
        return "same_desc_3"
    if count in (4, 5):
        return "same_desc_4_5"
    return "same_desc_6_plus"


def _case_id(split: str, record: Mapping[str, Any], desc: str, source_line_idx: int) -> str:
    image_id = record.get("image_id", source_line_idx)
    digest = hashlib.sha1(f"{split}|{image_id}|{desc}|{source_line_idx}".encode()).hexdigest()[:12]
    return f"{split}:{image_id}:{desc.replace(' ', '_')}:{digest}"


def _dataset_manifest_id(path: Path) -> str:
    return f"jsonl:{path.name}"


def _path_identity_hash(path: Path) -> str:
    payload = f"{path.resolve()}:{path.stat().st_size if path.exists() else -1}"
    return hashlib.sha256(payload.encode()).hexdigest()


def _bbox_values(tokens: tuple[Any, ...]) -> list[int] | None:
    if len(tokens) != 4:
        return None
    values = []
    for token in tokens:
        text = str(token)
        if not text.startswith("<|coord_") or not text.endswith("|>"):
            return None
        values.append(int(text.removeprefix("<|coord_").removesuffix("|>")))
    return values


def _object_size_bucket(tokens: tuple[Any, ...]) -> str:
    values = _bbox_values(tokens)
    if values is None:
        return "unknown_size"
    x1, y1, x2, y2 = values
    area = max(0, x2 - x1) * max(0, y2 - y1)
    if area < 32 * 32:
        return "small"
    if area < 128 * 128:
        return "medium"
    return "large"


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pool_case_sets: dict[str, set[str]] = defaultdict(set)
    split_case_sets: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        pool_case_sets[str(row["pool_role"])].add(str(row["case_id"]))
        split_case_sets[str(row["split"])].add(str(row["case_id"]))
    return {
        "case_index_total_rows": len(rows),
        "case_index_total_cases": len({row["case_id"] for row in rows}),
        "pool_role_counts": {key: len(value) for key, value in pool_case_sets.items()},
        "split_counts": {key: len(value) for key, value in split_case_sets.items()},
        "desc_normalization_policy_id": "lower_strip_collapse_ws_v1",
    }
