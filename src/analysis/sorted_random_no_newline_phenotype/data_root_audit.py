from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from .config import A32Config


EVIDENCE_SCOPE = "len12000-jsonl-local-mechanism-probe"


def build_data_root_audit(
    config: A32Config,
    *,
    max_missing_image_examples: int = 20,
) -> dict[str, Any]:
    summary = _JsonlSummary()
    missing_image_examples: list[dict[str, str | int]] = []

    for split, jsonl_path in (
        ("train", config.train_jsonl),
        ("val", config.val_jsonl),
    ):
        for row_index, record in enumerate(_iter_jsonl_records(jsonl_path), start=1):
            summary.add(split, record)
            image_ref = _extract_image_ref(record)
            if image_ref is None:
                continue
            image_path = _resolve_image_path(config.image_root, image_ref)
            if image_path.exists():
                continue
            if len(missing_image_examples) < max_missing_image_examples:
                missing_image_examples.append(
                    {
                        "split": split,
                        "row_index": row_index,
                        "image_ref": image_ref,
                        "resolved_path": str(image_path),
                    }
                )

    return {
        "status": "ok" if not missing_image_examples else "missing_images",
        "actual_train_jsonl": str(config.train_jsonl),
        "actual_val_jsonl": str(config.val_jsonl),
        "image_root": str(config.image_root),
        "evidence_scope": EVIDENCE_SCOPE,
        "row_counts": summary.row_counts,
        "jsonl_sha256": {
            "train": _sha256(config.train_jsonl),
            "val": _sha256(config.val_jsonl),
        },
        "object_count_histogram": _sorted_histogram(summary.object_count_histogram),
        "desc_count_histogram": _sorted_histogram(summary.desc_count_histogram),
        "same_desc_multi_instance_count": summary.same_desc_multi_instance_count,
        "missing_image_examples": missing_image_examples,
    }


class _JsonlSummary:
    def __init__(self) -> None:
        self.row_counts: dict[str, int] = {"train": 0, "val": 0}
        self.object_count_histogram: Counter[int] = Counter()
        self.desc_count_histogram: Counter[int] = Counter()
        self.same_desc_multi_instance_count = 0

    def add(self, split: str, record: Mapping[str, Any]) -> None:
        self.row_counts[split] += 1
        objects = record.get("objects")
        if not isinstance(objects, list):
            objects = []

        descs = [_object_desc(obj) for obj in objects if isinstance(obj, Mapping)]
        descs = [desc for desc in descs if desc]
        desc_counter = Counter(descs)
        self.object_count_histogram[len(objects)] += 1
        self.desc_count_histogram[len(desc_counter)] += 1
        if any(count > 1 for count in desc_counter.values()):
            self.same_desc_multi_instance_count += 1


def _iter_jsonl_records(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if not isinstance(record, Mapping):
                raise ValueError(f"{path}:{line_number} JSONL row must be an object")
            yield record


def _object_desc(obj: Mapping[str, Any]) -> str | None:
    for key in ("desc", "category_name"):
        value = obj.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _extract_image_ref(record: Mapping[str, Any]) -> str | None:
    file_name = record.get("file_name")
    if isinstance(file_name, str) and file_name:
        return file_name

    images = record.get("images")
    if not isinstance(images, list):
        return None
    for image in images:
        if isinstance(image, str) and image:
            return image
        if isinstance(image, Mapping):
            nested_file_name = image.get("file_name")
            if isinstance(nested_file_name, str) and nested_file_name:
                return nested_file_name
    return None


def _resolve_image_path(image_root: Path, image_ref: str) -> Path:
    path = Path(image_ref)
    if path.is_absolute():
        return path
    return image_root / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sorted_histogram(histogram: Counter[int]) -> dict[str, int]:
    return {str(key): histogram[key] for key in sorted(histogram)}
