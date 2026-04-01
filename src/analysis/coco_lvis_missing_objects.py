from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

DEFAULT_COCO_ANNOTATION_PATHS: tuple[Path, ...] = (
    Path("public_data/coco/raw/annotations/instances_train2017.json"),
    Path("public_data/coco/raw/annotations/instances_val2017.json"),
)
DEFAULT_LVIS_ANNOTATION_PATHS: tuple[Path, ...] = (
    Path("public_data/lvis/raw/annotations/lvis_v1_train.json"),
    Path("public_data/lvis/raw/annotations/lvis_v1_val.json"),
)

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

_MANUAL_ALIAS_MAP: dict[str, tuple[str, ...]] = {
    "tie": ("necktie",),
    "skis": ("ski",),
    "wine glass": ("wineglass",),
    "orange": ("orange_(fruit)",),
    "laptop": ("laptop_computer",),
    "mouse": ("mouse_(computer_equipment)",),
    "remote": ("remote_control",),
    "keyboard": ("computer_keyboard",),
    "cell phone": ("cellular_telephone",),
    "microwave": ("microwave_oven",),
    "hair drier": ("hair_dryer",),
}

_MANUAL_BROAD_MAP: dict[str, tuple[str, ...]] = {
    "car": ("car_(automobile)", "police_cruiser", "race_car"),
    "bus": ("bus_(vehicle)", "school_bus"),
    "train": ("train_(railroad_vehicle)",),
    "sports ball": (
        "baseball",
        "basketball",
        "soccer_ball",
        "softball",
        "tennis_ball",
    ),
    "potted plant": ("flowerpot",),
}

_EXACT_CANONICAL_ALIAS_MAP: dict[str, tuple[str, ...]] = {
    "couch": ("sofa",),
    "donut": ("doughnut",),
    "fire hydrant": ("fireplug",),
    "tv": ("television_set", "television"),
}

_CATEGORY_MAPPING_FIELDNAMES = [
    "coco_category_id",
    "coco_category_name",
    "mapping_strategy",
    "lvis_category_ids",
    "lvis_category_names",
]

_PER_IMAGE_FIELDNAMES = [
    "image_id",
    "coco_image_split",
    "lvis_source_name",
    "coco_annotation_count",
    "lvis_annotation_count",
    "mappable_lvis_instances",
    "candidate_lvis_instances",
    "matched_lvis_instances",
    "unmatched_recoverable_instances",
    "skipped_lvis_crowd",
    "skipped_unmapped_category",
    "skipped_not_exhaustive",
    "unmatched_lvis_categories",
]

_PER_CATEGORY_FIELDNAMES = [
    "coco_category_id",
    "coco_category_name",
    "lvis_category_id",
    "lvis_category_name",
    "lvis_frequency",
    "mapping_strategy",
    "mappable_lvis_instances",
    "candidate_lvis_instances",
    "matched_lvis_instances",
    "unmatched_recoverable_instances",
    "skipped_not_exhaustive",
]

_MAPPING_EVIDENCE_FIELDNAMES = [
    "lvis_category_id",
    "lvis_category_name",
    "lvis_frequency",
    "coco_category_id",
    "coco_category_name",
    "candidate_source",
    "candidate_kind",
    "prior_kind",
    "has_exact_canonical_prior",
    "n_match",
    "precision_like",
    "coverage_like",
    "mean_iou",
    "median_iou",
    "iou_ge_05_rate",
    "iou_ge_075_rate",
    "n_images",
    "eligible_lvis_instances",
    "matched_lvis_instances",
]

_RECOVERED_PER_IMAGE_FIELDNAMES = [
    "image_id",
    "coco_image_split",
    "lvis_source_name",
    "recovered_strict_count",
    "recovered_strict_plus_usable_count",
    "blocked_not_exhaustive_coco_categories",
    "recovered_coco_categories",
]

_RECOVERED_PER_CATEGORY_FIELDNAMES = [
    "coco_category_id",
    "coco_category_name",
    "recovered_strict_count",
    "recovered_strict_plus_usable_count",
    "recovered_lvis_categories",
]


@dataclass(frozen=True)
class MappingDecisionThresholds:
    min_match_count: int
    min_precision_like: float
    min_coverage_like: float
    min_mean_iou: float
    min_median_iou: float
    min_iou_ge_05_rate: float
    min_iou_ge_075_rate: float
    min_image_count: int
    max_runner_up_ratio: float


def _default_strict_exact_thresholds() -> MappingDecisionThresholds:
    return MappingDecisionThresholds(
        min_match_count=10,
        min_precision_like=0.90,
        min_coverage_like=0.02,
        min_mean_iou=0.85,
        min_median_iou=0.90,
        min_iou_ge_05_rate=0.95,
        min_iou_ge_075_rate=0.75,
        min_image_count=5,
        max_runner_up_ratio=0.25,
    )


def _default_usable_exact_thresholds() -> MappingDecisionThresholds:
    return MappingDecisionThresholds(
        min_match_count=3,
        min_precision_like=0.75,
        min_coverage_like=0.005,
        min_mean_iou=0.75,
        min_median_iou=0.75,
        min_iou_ge_05_rate=0.80,
        min_iou_ge_075_rate=0.40,
        min_image_count=2,
        max_runner_up_ratio=0.50,
    )


def _default_strict_semantic_thresholds() -> MappingDecisionThresholds:
    return MappingDecisionThresholds(
        min_match_count=15,
        min_precision_like=0.95,
        min_coverage_like=0.02,
        min_mean_iou=0.85,
        min_median_iou=0.90,
        min_iou_ge_05_rate=0.95,
        min_iou_ge_075_rate=0.80,
        min_image_count=5,
        max_runner_up_ratio=0.20,
    )


def _default_usable_semantic_thresholds() -> MappingDecisionThresholds:
    return MappingDecisionThresholds(
        min_match_count=5,
        min_precision_like=0.85,
        min_coverage_like=0.005,
        min_mean_iou=0.75,
        min_median_iou=0.80,
        min_iou_ge_05_rate=0.85,
        min_iou_ge_075_rate=0.50,
        min_image_count=3,
        max_runner_up_ratio=0.35,
    )


@dataclass(frozen=True)
class AnalysisConfig:
    iou_threshold: float = 0.5
    ignore_crowd: bool = True
    mapping_mode: str = "strict"
    max_images: int | None = None
    allowed_coco_image_splits: tuple[str, ...] = ()
    evidence_pair_iou_threshold: float = 0.3
    recovery_iou_threshold: float = 0.5
    recovery_min_lvis_box_area: float = 0.0
    recovery_max_conflicting_coco_iou: float = 0.5
    strict_exact_thresholds: MappingDecisionThresholds = field(
        default_factory=_default_strict_exact_thresholds
    )
    usable_exact_thresholds: MappingDecisionThresholds = field(
        default_factory=_default_usable_exact_thresholds
    )
    strict_semantic_thresholds: MappingDecisionThresholds = field(
        default_factory=_default_strict_semantic_thresholds
    )
    usable_semantic_thresholds: MappingDecisionThresholds = field(
        default_factory=_default_usable_semantic_thresholds
    )


@dataclass(frozen=True)
class CategoryMapping:
    coco_category_id: int
    coco_category_name: str
    lvis_category_ids: tuple[int, ...]
    lvis_category_names: tuple[str, ...]
    strategy: str


@dataclass(frozen=True)
class CategoryMappingBundle:
    coco_to_lvis: dict[int, CategoryMapping]
    lvis_to_coco: dict[int, CategoryMapping]
    rows: list[dict[str, Any]]
    summary: dict[str, Any]


@dataclass(frozen=True)
class LoadedDataset:
    dataset_kind: str
    source_name: str
    categories_by_id: dict[int, dict[str, Any]]
    images_by_id: dict[int, dict[str, Any]]
    annotations_by_image: dict[int, list[dict[str, Any]]]
    invalid_annotation_count: int


@dataclass(frozen=True)
class AnalysisResult:
    summary: dict[str, Any]
    category_mapping_rows: list[dict[str, Any]]
    per_image_rows: list[dict[str, Any]]
    per_category_rows: list[dict[str, Any]]
    unmatched_instances: list[dict[str, Any]]


@dataclass(frozen=True)
class MappingEvidenceResult:
    rows: list[dict[str, Any]]
    rows_by_lvis_category: dict[int, list[dict[str, Any]]]
    exact_canonical_candidates: dict[int, dict[int, str]]
    eligible_instances_by_lvis_category: dict[int, int]
    matched_instances_by_lvis_category: dict[int, int]
    summary: dict[str, Any]


@dataclass(frozen=True)
class LearnedMapping:
    lvis_category_id: int
    lvis_category_name: str
    lvis_frequency: str
    confidence_tier: str
    mapping_kind: str | None
    mapped_coco_category_id: int | None
    mapped_coco_category_name: str | None
    prior_kind: str | None
    evidence_summary: dict[str, Any] | None
    top_candidates: list[dict[str, Any]]
    rejection_reason: str | None


@dataclass(frozen=True)
class LearnedMappingResult:
    rows: list[dict[str, Any]]
    by_lvis_category_id: dict[int, LearnedMapping]
    lvis_ids_by_coco_and_tier: dict[str, dict[int, set[int]]]
    summary: dict[str, Any]


@dataclass(frozen=True)
class RecoveryResult:
    recovered_instances: list[dict[str, Any]]
    per_image_rows: list[dict[str, Any]]
    per_category_rows: list[dict[str, Any]]
    summary: dict[str, Any]


@dataclass(frozen=True)
class ProjectionAnalysisResult:
    summary: dict[str, Any]
    mapping_evidence_rows: list[dict[str, Any]]
    learned_mapping_rows: list[dict[str, Any]]
    recovered_instances: list[dict[str, Any]]
    recovered_per_image_rows: list[dict[str, Any]]
    recovered_per_category_rows: list[dict[str, Any]]
    report_markdown: str


def _normalize_category_name(text: str) -> str:
    value = str(text).lower().replace("_", " ")
    return _NON_ALNUM_RE.sub(" ", value).strip()


def _source_name_from_path(path: Path) -> str:
    return path.stem


def _infer_coco_split_from_source_name(source_name: str) -> str | None:
    if source_name.endswith("train2017"):
        return "train2017"
    if source_name.endswith("val2017"):
        return "val2017"
    return None


def _infer_coco_split_from_url(coco_url: Any) -> str | None:
    if not coco_url:
        return None
    parts = str(coco_url).strip().split("/")
    if len(parts) < 2:
        return None
    candidate = parts[-2]
    if candidate in {"train2017", "val2017"}:
        return candidate
    return None


def _file_name_from_url(coco_url: Any) -> str | None:
    if not coco_url:
        return None
    return str(coco_url).strip().split("/")[-1] or None


def _xywh_to_xyxy(bbox_xywh: Sequence[float]) -> tuple[float, float, float, float]:
    if len(bbox_xywh) != 4:
        raise ValueError(f"Expected bbox with 4 values, got {bbox_xywh!r}")
    x, y, w, h = [float(value) for value in bbox_xywh]
    return x, y, x + w, y + h


def _clip_xyxy(
    box_xyxy: Sequence[float],
    *,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    if len(box_xyxy) != 4:
        raise ValueError(f"Expected box with 4 values, got {box_xyxy!r}")
    x1, y1, x2, y2 = [float(value) for value in box_xyxy]
    return (
        max(0.0, min(float(width), x1)),
        max(0.0, min(float(height), y1)),
        max(0.0, min(float(width), x2)),
        max(0.0, min(float(height), y2)),
    )


def _is_valid_box(box_xyxy: Sequence[float], *, eps: float = 1e-6) -> bool:
    if len(box_xyxy) != 4:
        return False
    x1, y1, x2, y2 = [float(value) for value in box_xyxy]
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        return False
    return (x2 - x1) > eps and (y2 - y1) > eps


def _bbox_area(box_xyxy: Sequence[float]) -> float:
    if len(box_xyxy) != 4:
        return 0.0
    x1, y1, x2, y2 = [float(value) for value in box_xyxy]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _bbox_iou_xyxy(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    if len(box_a) != 4 or len(box_b) != 4:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(value) for value in box_a]
    bx1, by1, bx2, by2 = [float(value) for value in box_b]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    union = _bbox_area(box_a) + _bbox_area(box_b) - inter
    return float(inter / union) if union > 0.0 else 0.0


def _summarize_segmentation(segmentation: Any) -> dict[str, Any] | None:
    if segmentation is None:
        return None
    if isinstance(segmentation, list):
        polygon_count = 0
        point_count = 0
        for polygon in segmentation:
            if not isinstance(polygon, list):
                continue
            polygon_count += 1
            point_count += len(polygon) // 2
        return {
            "type": "polygon",
            "polygon_count": polygon_count,
            "point_count": point_count,
        }
    if isinstance(segmentation, dict):
        size = segmentation.get("size")
        if isinstance(size, list):
            size_value: list[int] | None = [int(value) for value in size]
        else:
            size_value = None
        counts = segmentation.get("counts")
        return {
            "type": "rle",
            "size": size_value,
            "counts_type": type(counts).__name__,
        }
    return {"type": type(segmentation).__name__}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Annotation JSON must be an object: {path}")
    return payload


def _load_dataset_from_payload(
    payload: Mapping[str, Any],
    *,
    dataset_kind: str,
    source_name: str,
) -> LoadedDataset:
    categories_raw = payload.get("categories")
    images_raw = payload.get("images")
    annotations_raw = payload.get("annotations")
    if not isinstance(categories_raw, list):
        raise ValueError(f"{source_name}: missing list payload['categories']")
    if not isinstance(images_raw, list):
        raise ValueError(f"{source_name}: missing list payload['images']")
    if not isinstance(annotations_raw, list):
        raise ValueError(f"{source_name}: missing list payload['annotations']")

    categories_by_id: dict[int, dict[str, Any]] = {}
    for category in categories_raw:
        category_id = int(category["id"])
        categories_by_id[category_id] = dict(category)

    images_by_id: dict[int, dict[str, Any]] = {}
    for image in images_raw:
        image_id = int(image["id"])
        image_record = dict(image)
        image_record["source_name"] = source_name
        if dataset_kind == "coco":
            image_record["coco_image_split"] = _infer_coco_split_from_source_name(source_name)
        else:
            image_record["coco_image_split"] = _infer_coco_split_from_url(
                image_record.get("coco_url")
            )
            image_record.setdefault("file_name", _file_name_from_url(image_record.get("coco_url")))
        images_by_id[image_id] = image_record

    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    invalid_annotation_count = 0
    for annotation in annotations_raw:
        image_id = int(annotation["image_id"])
        image_info = images_by_id.get(image_id)
        if image_info is None:
            invalid_annotation_count += 1
            continue
        try:
            clipped_box = _clip_xyxy(
                _xywh_to_xyxy(annotation["bbox"]),
                width=int(image_info["width"]),
                height=int(image_info["height"]),
            )
        except (KeyError, TypeError, ValueError):
            invalid_annotation_count += 1
            continue
        if not _is_valid_box(clipped_box):
            invalid_annotation_count += 1
            continue
        category_id = int(annotation["category_id"])
        category_info = categories_by_id.get(category_id, {})
        record = {
            "annotation_id": int(annotation["id"]),
            "image_id": image_id,
            "category_id": category_id,
            "category_name": str(category_info.get("name", category_id)),
            "bbox_xyxy": [float(value) for value in clipped_box],
            "bbox_xywh": [
                float(clipped_box[0]),
                float(clipped_box[1]),
                float(clipped_box[2] - clipped_box[0]),
                float(clipped_box[3] - clipped_box[1]),
            ],
            "bbox_area": float(annotation.get("area", _bbox_area(clipped_box))),
            "iscrowd": int(annotation.get("iscrowd", 0)),
            "segmentation_summary": _summarize_segmentation(annotation.get("segmentation")),
        }
        if dataset_kind == "lvis":
            record["frequency"] = str(category_info.get("frequency", "unknown"))
        annotations_by_image[image_id].append(record)

    return LoadedDataset(
        dataset_kind=dataset_kind,
        source_name=source_name,
        categories_by_id=categories_by_id,
        images_by_id=images_by_id,
        annotations_by_image={key: value for key, value in annotations_by_image.items()},
        invalid_annotation_count=invalid_annotation_count,
    )


def _merge_loaded_datasets(datasets: Sequence[LoadedDataset], *, dataset_kind: str) -> LoadedDataset:
    if not datasets:
        raise ValueError(f"No {dataset_kind} datasets were loaded")
    categories_by_id = dict(datasets[0].categories_by_id)
    images_by_id: dict[int, dict[str, Any]] = {}
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    invalid_annotation_count = 0
    for dataset in datasets:
        current_signature = {
            key: (
                str(value.get("name")),
                tuple(str(item) for item in value.get("synonyms", [])),
                str(value.get("frequency", "")),
            )
            for key, value in dataset.categories_by_id.items()
        }
        base_signature = {
            key: (
                str(value.get("name")),
                tuple(str(item) for item in value.get("synonyms", [])),
                str(value.get("frequency", "")),
            )
            for key, value in categories_by_id.items()
        }
        if current_signature != base_signature:
            raise ValueError(
                f"{dataset_kind} category definitions differ across sources: {dataset.source_name}"
            )
        for image_id, image_info in dataset.images_by_id.items():
            if image_id in images_by_id:
                raise ValueError(
                    f"Duplicate {dataset_kind} image_id {image_id} across sources; "
                    f"already saw {images_by_id[image_id]['source_name']} and {dataset.source_name}"
                )
            images_by_id[image_id] = dict(image_info)
        for image_id, annotations in dataset.annotations_by_image.items():
            annotations_by_image[image_id].extend(dict(annotation) for annotation in annotations)
        invalid_annotation_count += int(dataset.invalid_annotation_count)

    return LoadedDataset(
        dataset_kind=dataset_kind,
        source_name=f"{dataset_kind}_merged",
        categories_by_id=categories_by_id,
        images_by_id=images_by_id,
        annotations_by_image={key: value for key, value in annotations_by_image.items()},
        invalid_annotation_count=invalid_annotation_count,
    )


def _resolve_manual_mapping(
    coco_name: str,
    *,
    mapping_mode: str,
) -> tuple[str, tuple[str, ...]] | None:
    normalized = _normalize_category_name(coco_name)
    if normalized in _MANUAL_ALIAS_MAP:
        return "manual_alias", _MANUAL_ALIAS_MAP[normalized]
    if mapping_mode == "expanded" and normalized in _MANUAL_BROAD_MAP:
        return "manual_broad", _MANUAL_BROAD_MAP[normalized]
    return None


def build_category_mapping(
    coco_categories_by_id: Mapping[int, Mapping[str, Any]],
    lvis_categories_by_id: Mapping[int, Mapping[str, Any]],
    *,
    mapping_mode: str = "strict",
) -> CategoryMappingBundle:
    if mapping_mode not in {"strict", "expanded"}:
        raise ValueError(f"Unsupported mapping_mode={mapping_mode!r}")

    lvis_by_name = {str(category["name"]): int(category_id) for category_id, category in lvis_categories_by_id.items()}
    lvis_exact_index: dict[str, list[str]] = defaultdict(list)
    lvis_synonym_index: dict[str, list[str]] = defaultdict(list)
    for category in lvis_categories_by_id.values():
        canonical_name = str(category["name"])
        lvis_exact_index[_normalize_category_name(canonical_name)].append(canonical_name)
        for synonym in category.get("synonyms", []):
            lvis_synonym_index[_normalize_category_name(str(synonym))].append(canonical_name)

    coco_to_lvis: dict[int, CategoryMapping] = {}
    lvis_to_coco: dict[int, CategoryMapping] = {}
    rows: list[dict[str, Any]] = []
    strategy_counts: Counter[str] = Counter()
    unmapped_coco_categories: list[str] = []

    for coco_category_id in sorted(coco_categories_by_id):
        coco_category = coco_categories_by_id[coco_category_id]
        coco_name = str(coco_category["name"])
        normalized_name = _normalize_category_name(coco_name)

        strategy: str | None = None
        matched_lvis_names: tuple[str, ...] = ()
        exact_matches = tuple(sorted(set(lvis_exact_index.get(normalized_name, ()))))
        if exact_matches:
            strategy = "exact"
            matched_lvis_names = exact_matches
        else:
            synonym_matches = tuple(sorted(set(lvis_synonym_index.get(normalized_name, ()))))
            if synonym_matches:
                strategy = "synonym"
                matched_lvis_names = synonym_matches
            else:
                manual_mapping = _resolve_manual_mapping(
                    coco_name,
                    mapping_mode=mapping_mode,
                )
                if manual_mapping is not None:
                    strategy, manual_names = manual_mapping
                    matched_lvis_names = tuple(
                        name for name in manual_names if name in lvis_by_name
                    )
                    if not matched_lvis_names:
                        strategy = None

        if not matched_lvis_names or strategy is None:
            unmapped_coco_categories.append(coco_name)
            continue

        lvis_category_ids = tuple(sorted(lvis_by_name[name] for name in matched_lvis_names))
        mapping = CategoryMapping(
            coco_category_id=int(coco_category_id),
            coco_category_name=coco_name,
            lvis_category_ids=lvis_category_ids,
            lvis_category_names=matched_lvis_names,
            strategy=strategy,
        )
        coco_to_lvis[int(coco_category_id)] = mapping
        rows.append(
            {
                "coco_category_id": int(coco_category_id),
                "coco_category_name": coco_name,
                "mapping_strategy": strategy,
                "lvis_category_ids": list(lvis_category_ids),
                "lvis_category_names": list(matched_lvis_names),
            }
        )
        strategy_counts[strategy] += 1
        for lvis_category_id in lvis_category_ids:
            previous = lvis_to_coco.get(lvis_category_id)
            if previous is not None and previous.coco_category_id != int(coco_category_id):
                raise ValueError(
                    "LVIS category mapped to multiple COCO categories: "
                    f"{lvis_categories_by_id[lvis_category_id]['name']} -> "
                    f"{previous.coco_category_name}, {coco_name}"
                )
            lvis_to_coco[lvis_category_id] = mapping

    summary = {
        "mapping_mode": mapping_mode,
        "mapped_coco_categories": len(coco_to_lvis),
        "mapped_lvis_categories": len(lvis_to_coco),
        "strategy_counts": dict(sorted(strategy_counts.items())),
        "unmapped_coco_categories": unmapped_coco_categories,
    }
    return CategoryMappingBundle(
        coco_to_lvis=coco_to_lvis,
        lvis_to_coco=lvis_to_coco,
        rows=rows,
        summary=summary,
    )


def _best_iou_against_annotations(
    box_xyxy: Sequence[float],
    annotations: Sequence[Mapping[str, Any]],
) -> float:
    best_iou = 0.0
    for annotation in annotations:
        best_iou = max(best_iou, _bbox_iou_xyxy(box_xyxy, annotation["bbox_xyxy"]))
    return float(best_iou)


def _best_iou_against_other_categories(
    box_xyxy: Sequence[float],
    annotations: Sequence[Mapping[str, Any]],
    *,
    excluded_category_id: int,
) -> tuple[float, int | None, str | None]:
    best_iou = 0.0
    best_category_id: int | None = None
    best_category_name: str | None = None
    for annotation in annotations:
        category_id = int(annotation["category_id"])
        if category_id == excluded_category_id:
            continue
        iou = _bbox_iou_xyxy(box_xyxy, annotation["bbox_xyxy"])
        if iou <= best_iou:
            continue
        best_iou = float(iou)
        best_category_id = category_id
        best_category_name = str(annotation.get("category_name", category_id))
    return best_iou, best_category_id, best_category_name


def _greedy_collect_matches_by_iou(
    query_annotations: Sequence[Mapping[str, Any]],
    target_annotations: Sequence[Mapping[str, Any]],
    *,
    iou_threshold: float,
) -> tuple[list[tuple[int, int, float]], list[float]]:
    best_ious = [0.0 for _ in query_annotations]
    candidate_pairs: list[tuple[float, int, int]] = []
    for query_index, query_annotation in enumerate(query_annotations):
        for target_index, target_annotation in enumerate(target_annotations):
            iou = _bbox_iou_xyxy(
                query_annotation["bbox_xyxy"],
                target_annotation["bbox_xyxy"],
            )
            if iou > best_ious[query_index]:
                best_ious[query_index] = float(iou)
            if iou >= iou_threshold:
                candidate_pairs.append((float(iou), query_index, target_index))

    matched_query_indices: set[int] = set()
    matched_target_indices: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    candidate_pairs.sort(key=lambda item: item[0], reverse=True)
    for iou, query_index, target_index in candidate_pairs:
        if query_index in matched_query_indices or target_index in matched_target_indices:
            continue
        matched_query_indices.add(query_index)
        matched_target_indices.add(target_index)
        matches.append((query_index, target_index, float(iou)))
    return matches, best_ious


def _greedy_match_by_iou(
    lvis_annotations: Sequence[Mapping[str, Any]],
    coco_annotations: Sequence[Mapping[str, Any]],
    *,
    iou_threshold: float,
) -> tuple[set[int], list[float]]:
    matches, best_same_category_ious = _greedy_collect_matches_by_iou(
        lvis_annotations,
        coco_annotations,
        iou_threshold=iou_threshold,
    )
    matched_lvis_indices = {match[0] for match in matches}
    return matched_lvis_indices, best_same_category_ious


def analyze_coco_lvis_overlap(
    coco_payloads: Sequence[Mapping[str, Any]],
    lvis_payloads: Sequence[Mapping[str, Any]],
    *,
    coco_source_names: Sequence[str] | None = None,
    lvis_source_names: Sequence[str] | None = None,
    config: AnalysisConfig | None = None,
) -> AnalysisResult:
    cfg = config or AnalysisConfig()
    loaded_coco = [
        _load_dataset_from_payload(
            payload,
            dataset_kind="coco",
            source_name=(
                coco_source_names[index]
                if coco_source_names is not None
                else f"coco_source_{index}"
            ),
        )
        for index, payload in enumerate(coco_payloads)
    ]
    loaded_lvis = [
        _load_dataset_from_payload(
            payload,
            dataset_kind="lvis",
            source_name=(
                lvis_source_names[index]
                if lvis_source_names is not None
                else f"lvis_source_{index}"
            ),
        )
        for index, payload in enumerate(lvis_payloads)
    ]
    coco_dataset = _merge_loaded_datasets(loaded_coco, dataset_kind="coco")
    lvis_dataset = _merge_loaded_datasets(loaded_lvis, dataset_kind="lvis")
    mapping_bundle = build_category_mapping(
        coco_dataset.categories_by_id,
        lvis_dataset.categories_by_id,
        mapping_mode=cfg.mapping_mode,
    )

    shared_image_ids = sorted(
        set(coco_dataset.images_by_id.keys()) & set(lvis_dataset.images_by_id.keys())
    )
    if cfg.allowed_coco_image_splits:
        allowed_splits = set(cfg.allowed_coco_image_splits)
        shared_image_ids = [
            image_id
            for image_id in shared_image_ids
            if (
                lvis_dataset.images_by_id[image_id].get("coco_image_split")
                or coco_dataset.images_by_id[image_id].get("coco_image_split")
            )
            in allowed_splits
        ]
    if cfg.max_images is not None:
        shared_image_ids = shared_image_ids[: int(cfg.max_images)]

    analysis_counts: Counter[str] = Counter()
    shared_by_coco_split: Counter[str] = Counter()
    shared_by_lvis_source: Counter[str] = Counter()
    per_image_rows: list[dict[str, Any]] = []
    unmatched_instances: list[dict[str, Any]] = []
    per_category_counts: dict[tuple[int, int], Counter[str]] = defaultdict(Counter)

    for image_id in shared_image_ids:
        coco_image = coco_dataset.images_by_id[image_id]
        lvis_image = lvis_dataset.images_by_id[image_id]
        coco_image_split = (
            lvis_image.get("coco_image_split")
            or coco_image.get("coco_image_split")
            or "unknown"
        )
        lvis_source_name = str(lvis_image.get("source_name", "unknown"))
        shared_by_coco_split[str(coco_image_split)] += 1
        shared_by_lvis_source[lvis_source_name] += 1

        coco_annotations_all = coco_dataset.annotations_by_image.get(image_id, [])
        if cfg.ignore_crowd:
            analysis_counts["skipped_coco_crowd"] += sum(
                int(annotation.get("iscrowd", 0))
                for annotation in coco_annotations_all
            )
            coco_annotations = [
                annotation
                for annotation in coco_annotations_all
                if int(annotation.get("iscrowd", 0)) == 0
            ]
        else:
            coco_annotations = list(coco_annotations_all)
        coco_annotations_by_category: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in coco_annotations:
            coco_annotations_by_category[int(annotation["category_id"])].append(annotation)

        image_not_exhaustive_ids = {
            int(value)
            for value in lvis_image.get("not_exhaustive_category_ids", [])
        }
        image_negative_ids = {
            int(value)
            for value in lvis_image.get("neg_category_ids", [])
        }
        image_counts: Counter[str] = Counter()
        image_counts["coco_annotation_count"] = len(coco_annotations)
        image_counts["lvis_annotation_count"] = len(
            lvis_dataset.annotations_by_image.get(image_id, [])
        )
        image_unmatched_categories: set[str] = set()
        candidate_groups: dict[int, list[tuple[dict[str, Any], CategoryMapping]]] = defaultdict(list)

        for lvis_annotation in lvis_dataset.annotations_by_image.get(image_id, []):
            analysis_counts["lvis_annotations_total"] += 1
            if cfg.ignore_crowd and int(lvis_annotation.get("iscrowd", 0)) == 1:
                analysis_counts["skipped_lvis_crowd"] += 1
                image_counts["skipped_lvis_crowd"] += 1
                continue
            mapping = mapping_bundle.lvis_to_coco.get(int(lvis_annotation["category_id"]))
            if mapping is None:
                analysis_counts["skipped_unmapped_category"] += 1
                image_counts["skipped_unmapped_category"] += 1
                continue
            per_category_key = (mapping.coco_category_id, int(lvis_annotation["category_id"]))
            per_category_counter = per_category_counts[per_category_key]
            per_category_counter["mappable_lvis_instances"] += 1
            analysis_counts["mappable_lvis_instances"] += 1
            image_counts["mappable_lvis_instances"] += 1
            if int(lvis_annotation["category_id"]) in image_not_exhaustive_ids:
                per_category_counter["skipped_not_exhaustive"] += 1
                analysis_counts["skipped_not_exhaustive"] += 1
                image_counts["skipped_not_exhaustive"] += 1
                continue
            per_category_counter["candidate_lvis_instances"] += 1
            analysis_counts["candidate_lvis_instances"] += 1
            image_counts["candidate_lvis_instances"] += 1
            candidate_groups[mapping.coco_category_id].append((dict(lvis_annotation), mapping))

        for coco_category_id, lvis_candidates in candidate_groups.items():
            same_category_coco_annotations = coco_annotations_by_category.get(coco_category_id, [])
            lvis_annotations_for_match = [item[0] for item in lvis_candidates]
            matched_lvis_indices, best_same_category_ious = _greedy_match_by_iou(
                lvis_annotations_for_match,
                same_category_coco_annotations,
                iou_threshold=float(cfg.iou_threshold),
            )
            for candidate_index, (lvis_annotation, mapping) in enumerate(lvis_candidates):
                per_category_key = (
                    int(mapping.coco_category_id),
                    int(lvis_annotation["category_id"]),
                )
                per_category_counter = per_category_counts[per_category_key]
                if candidate_index in matched_lvis_indices:
                    per_category_counter["matched_lvis_instances"] += 1
                    analysis_counts["matched_lvis_instances"] += 1
                    image_counts["matched_lvis_instances"] += 1
                    continue
                per_category_counter["unmatched_recoverable_instances"] += 1
                analysis_counts["unmatched_recoverable_instances"] += 1
                image_counts["unmatched_recoverable_instances"] += 1
                image_unmatched_categories.add(str(lvis_annotation["category_name"]))
                unmatched_instances.append(
                    {
                        "image_id": int(image_id),
                        "coco_image_split": str(coco_image_split),
                        "lvis_source_name": lvis_source_name,
                        "lvis_annotation_id": int(lvis_annotation["annotation_id"]),
                        "lvis_category_id": int(lvis_annotation["category_id"]),
                        "lvis_category_name": str(lvis_annotation["category_name"]),
                        "lvis_frequency": str(lvis_annotation.get("frequency", "unknown")),
                        "mapped_coco_category_id": int(mapping.coco_category_id),
                        "mapped_coco_category_name": str(mapping.coco_category_name),
                        "mapping_strategy": str(mapping.strategy),
                        "bbox_xyxy": [float(value) for value in lvis_annotation["bbox_xyxy"]],
                        "bbox_area": float(lvis_annotation["bbox_area"]),
                        "best_same_category_iou": float(best_same_category_ious[candidate_index]),
                        "best_any_coco_iou": float(
                            _best_iou_against_annotations(
                                lvis_annotation["bbox_xyxy"],
                                coco_annotations,
                            )
                        ),
                        "image_has_negative_flag_for_lvis_category": (
                            int(lvis_annotation["category_id"]) in image_negative_ids
                        ),
                    }
                )

        per_image_rows.append(
            {
                "image_id": int(image_id),
                "coco_image_split": str(coco_image_split),
                "lvis_source_name": lvis_source_name,
                "coco_annotation_count": int(image_counts["coco_annotation_count"]),
                "lvis_annotation_count": int(image_counts["lvis_annotation_count"]),
                "mappable_lvis_instances": int(image_counts["mappable_lvis_instances"]),
                "candidate_lvis_instances": int(image_counts["candidate_lvis_instances"]),
                "matched_lvis_instances": int(image_counts["matched_lvis_instances"]),
                "unmatched_recoverable_instances": int(
                    image_counts["unmatched_recoverable_instances"]
                ),
                "skipped_lvis_crowd": int(image_counts["skipped_lvis_crowd"]),
                "skipped_unmapped_category": int(image_counts["skipped_unmapped_category"]),
                "skipped_not_exhaustive": int(image_counts["skipped_not_exhaustive"]),
                "unmatched_lvis_categories": "|".join(sorted(image_unmatched_categories)),
            }
        )

    per_category_rows: list[dict[str, Any]] = []
    for (coco_category_id, lvis_category_id), counter in sorted(per_category_counts.items()):
        mapping = mapping_bundle.lvis_to_coco[lvis_category_id]
        lvis_category = lvis_dataset.categories_by_id[lvis_category_id]
        per_category_rows.append(
            {
                "coco_category_id": int(coco_category_id),
                "coco_category_name": str(mapping.coco_category_name),
                "lvis_category_id": int(lvis_category_id),
                "lvis_category_name": str(lvis_category["name"]),
                "lvis_frequency": str(lvis_category.get("frequency", "unknown")),
                "mapping_strategy": str(mapping.strategy),
                "mappable_lvis_instances": int(counter["mappable_lvis_instances"]),
                "candidate_lvis_instances": int(counter["candidate_lvis_instances"]),
                "matched_lvis_instances": int(counter["matched_lvis_instances"]),
                "unmatched_recoverable_instances": int(
                    counter["unmatched_recoverable_instances"]
                ),
                "skipped_not_exhaustive": int(counter["skipped_not_exhaustive"]),
            }
        )

    unmatched_distribution = Counter(
        int(row["unmatched_recoverable_instances"]) for row in per_image_rows
    )
    summary = {
        "config": {
            "iou_threshold": float(cfg.iou_threshold),
            "ignore_crowd": bool(cfg.ignore_crowd),
            "mapping_mode": str(cfg.mapping_mode),
            "max_images": cfg.max_images,
            "allowed_coco_image_splits": list(cfg.allowed_coco_image_splits),
        },
        "inputs": {
            "coco_image_count": len(coco_dataset.images_by_id),
            "lvis_image_count": len(lvis_dataset.images_by_id),
            "coco_invalid_annotations": int(coco_dataset.invalid_annotation_count),
            "lvis_invalid_annotations": int(lvis_dataset.invalid_annotation_count),
        },
        "shared_images": {
            "count": len(shared_image_ids),
            "by_coco_image_split": dict(sorted(shared_by_coco_split.items())),
            "by_lvis_source_name": dict(sorted(shared_by_lvis_source.items())),
        },
        "category_mapping": mapping_bundle.summary,
        "analysis": {
            key: int(value)
            for key, value in sorted(analysis_counts.items())
        },
        "unmatched_per_image_distribution": {
            str(key): int(value)
            for key, value in sorted(unmatched_distribution.items())
        },
        "top_unmatched_images": sorted(
            [
                {
                    "image_id": int(row["image_id"]),
                    "coco_image_split": str(row["coco_image_split"]),
                    "lvis_source_name": str(row["lvis_source_name"]),
                    "unmatched_recoverable_instances": int(
                        row["unmatched_recoverable_instances"]
                    ),
                    "unmatched_lvis_categories": str(row["unmatched_lvis_categories"]),
                }
                for row in per_image_rows
                if int(row["unmatched_recoverable_instances"]) > 0
            ],
            key=lambda row: (
                int(row["unmatched_recoverable_instances"]),
                int(row["image_id"]),
            ),
            reverse=True,
        )[:20],
        "top_unmatched_categories": sorted(
            [
                {
                    "coco_category_name": str(row["coco_category_name"]),
                    "lvis_category_name": str(row["lvis_category_name"]),
                    "mapping_strategy": str(row["mapping_strategy"]),
                    "unmatched_recoverable_instances": int(
                        row["unmatched_recoverable_instances"]
                    ),
                }
                for row in per_category_rows
                if int(row["unmatched_recoverable_instances"]) > 0
            ],
            key=lambda row: (
                int(row["unmatched_recoverable_instances"]),
                str(row["coco_category_name"]),
                str(row["lvis_category_name"]),
            ),
            reverse=True,
        )[:20],
    }
    return AnalysisResult(
        summary=summary,
        category_mapping_rows=mapping_bundle.rows,
        per_image_rows=per_image_rows,
        per_category_rows=per_category_rows,
        unmatched_instances=unmatched_instances,
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_analysis_outputs(result: AnalysisResult, *, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    category_mapping_path = output_dir / "category_mapping.json"
    per_image_path = output_dir / "per_image.csv"
    per_category_path = output_dir / "per_category.csv"
    unmatched_path = output_dir / "unmatched_instances.jsonl"

    _write_json(summary_path, result.summary)
    _write_json(
        category_mapping_path,
        {
            "summary": result.summary["category_mapping"],
            "mappings": result.category_mapping_rows,
        },
    )
    _write_csv(
        per_image_path,
        result.per_image_rows,
        fieldnames=_PER_IMAGE_FIELDNAMES,
    )
    _write_csv(
        per_category_path,
        result.per_category_rows,
        fieldnames=_PER_CATEGORY_FIELDNAMES,
    )
    _write_jsonl(unmatched_path, result.unmatched_instances)
    return {
        "summary_json": str(summary_path),
        "category_mapping_json": str(category_mapping_path),
        "per_image_csv": str(per_image_path),
        "per_category_csv": str(per_category_path),
        "unmatched_instances_jsonl": str(unmatched_path),
    }


def run_coco_lvis_missing_object_analysis(
    *,
    output_dir: Path,
    coco_annotation_paths: Sequence[Path] = DEFAULT_COCO_ANNOTATION_PATHS,
    lvis_annotation_paths: Sequence[Path] = DEFAULT_LVIS_ANNOTATION_PATHS,
    config: AnalysisConfig | None = None,
) -> dict[str, Any]:
    coco_paths = [Path(path) for path in coco_annotation_paths]
    lvis_paths = [Path(path) for path in lvis_annotation_paths]
    coco_payloads = [_load_json(path) for path in coco_paths]
    lvis_payloads = [_load_json(path) for path in lvis_paths]
    result = analyze_coco_lvis_overlap(
        coco_payloads,
        lvis_payloads,
        coco_source_names=[_source_name_from_path(path) for path in coco_paths],
        lvis_source_names=[_source_name_from_path(path) for path in lvis_paths],
        config=config,
    )
    artifact_paths = write_analysis_outputs(result, output_dir=output_dir)
    summary_with_artifacts = dict(result.summary)
    summary_with_artifacts["artifacts"] = artifact_paths
    _write_json(output_dir / "summary.json", summary_with_artifacts)
    return summary_with_artifacts


def _load_merged_analysis_datasets(
    coco_payloads: Sequence[Mapping[str, Any]],
    lvis_payloads: Sequence[Mapping[str, Any]],
    *,
    coco_source_names: Sequence[str] | None = None,
    lvis_source_names: Sequence[str] | None = None,
) -> tuple[LoadedDataset, LoadedDataset]:
    loaded_coco = [
        _load_dataset_from_payload(
            payload,
            dataset_kind="coco",
            source_name=(
                coco_source_names[index]
                if coco_source_names is not None
                else f"coco_source_{index}"
            ),
        )
        for index, payload in enumerate(coco_payloads)
    ]
    loaded_lvis = [
        _load_dataset_from_payload(
            payload,
            dataset_kind="lvis",
            source_name=(
                lvis_source_names[index]
                if lvis_source_names is not None
                else f"lvis_source_{index}"
            ),
        )
        for index, payload in enumerate(lvis_payloads)
    ]
    return (
        _merge_loaded_datasets(loaded_coco, dataset_kind="coco"),
        _merge_loaded_datasets(loaded_lvis, dataset_kind="lvis"),
    )


def _resolve_shared_image_ids(
    coco_dataset: LoadedDataset,
    lvis_dataset: LoadedDataset,
    *,
    config: AnalysisConfig,
) -> tuple[list[int], dict[str, int], dict[str, int]]:
    shared_image_ids = sorted(
        set(coco_dataset.images_by_id.keys()) & set(lvis_dataset.images_by_id.keys())
    )
    if config.allowed_coco_image_splits:
        allowed_splits = set(config.allowed_coco_image_splits)
        shared_image_ids = [
            image_id
            for image_id in shared_image_ids
            if (
                lvis_dataset.images_by_id[image_id].get("coco_image_split")
                or coco_dataset.images_by_id[image_id].get("coco_image_split")
            )
            in allowed_splits
        ]
    if config.max_images is not None:
        shared_image_ids = shared_image_ids[: int(config.max_images)]

    shared_by_coco_split: Counter[str] = Counter()
    shared_by_lvis_source: Counter[str] = Counter()
    for image_id in shared_image_ids:
        coco_image = coco_dataset.images_by_id[image_id]
        lvis_image = lvis_dataset.images_by_id[image_id]
        coco_image_split = (
            lvis_image.get("coco_image_split")
            or coco_image.get("coco_image_split")
            or "unknown"
        )
        shared_by_coco_split[str(coco_image_split)] += 1
        shared_by_lvis_source[str(lvis_image.get("source_name", "unknown"))] += 1
    return (
        shared_image_ids,
        dict(sorted(shared_by_coco_split.items())),
        dict(sorted(shared_by_lvis_source.items())),
    )


def _build_exact_canonical_candidates(
    coco_categories_by_id: Mapping[int, Mapping[str, Any]],
    lvis_categories_by_id: Mapping[int, Mapping[str, Any]],
) -> dict[int, dict[int, str]]:
    coco_name_to_id: dict[str, int] = {}
    coco_alias_to_ids: dict[str, set[int]] = defaultdict(set)
    for coco_category_id, coco_category in coco_categories_by_id.items():
        coco_name = str(coco_category["name"])
        normalized_coco_name = _normalize_category_name(coco_name)
        coco_name_to_id[normalized_coco_name] = int(coco_category_id)
        for alias in _MANUAL_ALIAS_MAP.get(normalized_coco_name, ()):
            coco_alias_to_ids[_normalize_category_name(alias)].add(int(coco_category_id))
        for alias in _EXACT_CANONICAL_ALIAS_MAP.get(normalized_coco_name, ()):
            coco_alias_to_ids[_normalize_category_name(alias)].add(int(coco_category_id))

    exact_candidates: dict[int, dict[int, str]] = defaultdict(dict)
    for lvis_category_id, lvis_category in lvis_categories_by_id.items():
        lvis_name = _normalize_category_name(str(lvis_category["name"]))
        synonym_terms = {
            _normalize_category_name(str(synonym))
            for synonym in lvis_category.get("synonyms", [])
        }
        direct_coco_id = coco_name_to_id.get(lvis_name)
        if direct_coco_id is not None:
            exact_candidates[int(lvis_category_id)][direct_coco_id] = "exact_name"
        for synonym_term in synonym_terms:
            synonym_coco_id = coco_name_to_id.get(synonym_term)
            if synonym_coco_id is not None:
                exact_candidates[int(lvis_category_id)][synonym_coco_id] = "synonym_alias"
        candidate_terms = {lvis_name, *synonym_terms}
        for candidate_term in candidate_terms:
            for coco_category_id in coco_alias_to_ids.get(candidate_term, set()):
                exact_candidates[int(lvis_category_id)].setdefault(
                    coco_category_id,
                    "canonical_alias",
                )
    return {
        int(lvis_category_id): dict(sorted(candidates.items()))
        for lvis_category_id, candidates in exact_candidates.items()
    }


def mine_lvis_to_coco80_mapping_evidence(
    coco_dataset: LoadedDataset,
    lvis_dataset: LoadedDataset,
    *,
    shared_image_ids: Sequence[int],
    config: AnalysisConfig,
) -> MappingEvidenceResult:
    exact_canonical_candidates = _build_exact_canonical_candidates(
        coco_dataset.categories_by_id,
        lvis_dataset.categories_by_id,
    )
    pair_ious: dict[tuple[int, int], list[float]] = defaultdict(list)
    pair_image_ids: dict[tuple[int, int], set[int]] = defaultdict(set)
    eligible_instances_by_lvis_category: Counter[int] = Counter()
    matched_instances_by_lvis_category: Counter[int] = Counter()
    analysis_counts: Counter[str] = Counter()

    for image_id in shared_image_ids:
        lvis_image = lvis_dataset.images_by_id[image_id]
        image_not_exhaustive_ids = {
            int(value)
            for value in lvis_image.get("not_exhaustive_category_ids", [])
        }
        coco_annotations_all = coco_dataset.annotations_by_image.get(image_id, [])
        lvis_annotations_all = lvis_dataset.annotations_by_image.get(image_id, [])
        if config.ignore_crowd:
            coco_annotations = [
                annotation
                for annotation in coco_annotations_all
                if int(annotation.get("iscrowd", 0)) == 0
            ]
            lvis_annotations = [
                annotation
                for annotation in lvis_annotations_all
                if int(annotation.get("iscrowd", 0)) == 0
            ]
            analysis_counts["skipped_coco_crowd"] += len(coco_annotations_all) - len(coco_annotations)
            analysis_counts["skipped_lvis_crowd"] += len(lvis_annotations_all) - len(lvis_annotations)
        else:
            coco_annotations = list(coco_annotations_all)
            lvis_annotations = list(lvis_annotations_all)

        eligible_lvis_annotations: list[dict[str, Any]] = []
        for lvis_annotation in lvis_annotations:
            analysis_counts["lvis_annotations_total"] += 1
            lvis_category_id = int(lvis_annotation["category_id"])
            if lvis_category_id in image_not_exhaustive_ids:
                analysis_counts["skipped_not_exhaustive"] += 1
                continue
            if float(lvis_annotation["bbox_area"]) < float(config.recovery_min_lvis_box_area):
                analysis_counts["skipped_small_lvis_box"] += 1
                continue
            eligible_instances_by_lvis_category[lvis_category_id] += 1
            eligible_lvis_annotations.append(dict(lvis_annotation))
        analysis_counts["eligible_lvis_instances"] += len(eligible_lvis_annotations)
        analysis_counts["eligible_coco_instances"] += len(coco_annotations)

        matches, _ = _greedy_collect_matches_by_iou(
            eligible_lvis_annotations,
            coco_annotations,
            iou_threshold=float(config.evidence_pair_iou_threshold),
        )
        analysis_counts["matched_pairs"] += len(matches)
        for lvis_index, coco_index, iou in matches:
            lvis_annotation = eligible_lvis_annotations[lvis_index]
            coco_annotation = coco_annotations[coco_index]
            lvis_category_id = int(lvis_annotation["category_id"])
            coco_category_id = int(coco_annotation["category_id"])
            pair_key = (lvis_category_id, coco_category_id)
            pair_ious[pair_key].append(float(iou))
            pair_image_ids[pair_key].add(int(image_id))
            matched_instances_by_lvis_category[lvis_category_id] += 1

    rows_by_lvis_category: dict[int, list[dict[str, Any]]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    for lvis_category_id, lvis_category in sorted(lvis_dataset.categories_by_id.items()):
        candidate_coco_ids = {
            int(coco_category_id)
            for pair_lvis_category_id, coco_category_id in pair_ious.keys()
            if int(pair_lvis_category_id) == int(lvis_category_id)
        }
        candidate_coco_ids.update(
            exact_canonical_candidates.get(int(lvis_category_id), {}).keys()
        )
        if not candidate_coco_ids:
            continue
        lvis_name = str(lvis_category["name"])
        lvis_frequency = str(lvis_category.get("frequency", "unknown"))
        matched_total = int(matched_instances_by_lvis_category.get(int(lvis_category_id), 0))
        eligible_total = int(eligible_instances_by_lvis_category.get(int(lvis_category_id), 0))
        for coco_category_id in sorted(candidate_coco_ids):
            coco_category = coco_dataset.categories_by_id[int(coco_category_id)]
            ious = pair_ious.get((int(lvis_category_id), int(coco_category_id)), [])
            n_match = len(ious)
            prior_kind = exact_canonical_candidates.get(int(lvis_category_id), {}).get(
                int(coco_category_id)
            )
            if prior_kind and n_match > 0:
                candidate_source = "exact_canonical+empirical_match"
            elif prior_kind:
                candidate_source = "exact_canonical_prior"
            else:
                candidate_source = "empirical_match"
            mean_iou = float(sum(ious) / n_match) if n_match > 0 else 0.0
            median_iou = float(median(ious)) if n_match > 0 else 0.0
            row = {
                "lvis_category_id": int(lvis_category_id),
                "lvis_category_name": lvis_name,
                "lvis_frequency": lvis_frequency,
                "coco_category_id": int(coco_category_id),
                "coco_category_name": str(coco_category["name"]),
                "candidate_source": candidate_source,
                "candidate_kind": (
                    "exact_canonical" if prior_kind is not None else "semantic_evidence"
                ),
                "prior_kind": prior_kind,
                "has_exact_canonical_prior": prior_kind is not None,
                "n_match": int(n_match),
                "precision_like": (
                    float(n_match / matched_total) if matched_total > 0 else 0.0
                ),
                "coverage_like": (
                    float(n_match / eligible_total) if eligible_total > 0 else 0.0
                ),
                "mean_iou": mean_iou,
                "median_iou": median_iou,
                "iou_ge_05_rate": (
                    float(sum(iou >= 0.5 for iou in ious) / n_match)
                    if n_match > 0
                    else 0.0
                ),
                "iou_ge_075_rate": (
                    float(sum(iou >= 0.75 for iou in ious) / n_match)
                    if n_match > 0
                    else 0.0
                ),
                "n_images": int(len(pair_image_ids.get((int(lvis_category_id), int(coco_category_id)), set()))),
                "eligible_lvis_instances": int(eligible_total),
                "matched_lvis_instances": int(matched_total),
            }
            rows.append(row)
            rows_by_lvis_category[int(lvis_category_id)].append(row)

    for category_rows in rows_by_lvis_category.values():
        category_rows.sort(
            key=lambda row: (
                -int(row["n_match"]),
                -float(row["precision_like"]),
                -float(row["mean_iou"]),
                -int(row["n_images"]),
                str(row["coco_category_name"]),
            )
        )
    rows.sort(
        key=lambda row: (
            str(row["lvis_category_name"]),
            -int(row["n_match"]),
            -float(row["precision_like"]),
            str(row["coco_category_name"]),
        )
    )
    summary = {
        "evidence_pair_iou_threshold": float(config.evidence_pair_iou_threshold),
        "lvis_categories_with_candidate_pairs": int(len(rows_by_lvis_category)),
        "lvis_categories_with_eligible_instances": int(
            sum(value > 0 for value in eligible_instances_by_lvis_category.values())
        ),
        "candidate_pair_rows": int(len(rows)),
        "exact_canonical_candidate_rows": int(
            sum(int(row["has_exact_canonical_prior"]) for row in rows)
        ),
        "matched_pair_rows": int(sum(int(row["n_match"]) > 0 for row in rows)),
        "analysis": {key: int(value) for key, value in sorted(analysis_counts.items())},
    }
    return MappingEvidenceResult(
        rows=rows,
        rows_by_lvis_category={
            int(lvis_category_id): list(category_rows)
            for lvis_category_id, category_rows in rows_by_lvis_category.items()
        },
        exact_canonical_candidates=exact_canonical_candidates,
        eligible_instances_by_lvis_category={
            int(key): int(value)
            for key, value in sorted(eligible_instances_by_lvis_category.items())
        },
        matched_instances_by_lvis_category={
            int(key): int(value)
            for key, value in sorted(matched_instances_by_lvis_category.items())
        },
        summary=summary,
    )


def _threshold_failures(
    row: Mapping[str, Any],
    thresholds: MappingDecisionThresholds,
    *,
    runner_up_ratio: float,
) -> list[str]:
    failures: list[str] = []
    if int(row["n_match"]) < thresholds.min_match_count:
        failures.append(
            f"n_match {int(row['n_match'])} < {thresholds.min_match_count}"
        )
    if float(row["precision_like"]) < thresholds.min_precision_like:
        failures.append(
            "precision_like "
            f"{float(row['precision_like']):.3f} < {thresholds.min_precision_like:.3f}"
        )
    if float(row["coverage_like"]) < thresholds.min_coverage_like:
        failures.append(
            "coverage_like "
            f"{float(row['coverage_like']):.3f} < {thresholds.min_coverage_like:.3f}"
        )
    if float(row["mean_iou"]) < thresholds.min_mean_iou:
        failures.append(
            f"mean_iou {float(row['mean_iou']):.3f} < {thresholds.min_mean_iou:.3f}"
        )
    if float(row["median_iou"]) < thresholds.min_median_iou:
        failures.append(
            f"median_iou {float(row['median_iou']):.3f} < {thresholds.min_median_iou:.3f}"
        )
    if float(row["iou_ge_05_rate"]) < thresholds.min_iou_ge_05_rate:
        failures.append(
            "iou_ge_05_rate "
            f"{float(row['iou_ge_05_rate']):.3f} < {thresholds.min_iou_ge_05_rate:.3f}"
        )
    if float(row["iou_ge_075_rate"]) < thresholds.min_iou_ge_075_rate:
        failures.append(
            "iou_ge_075_rate "
            f"{float(row['iou_ge_075_rate']):.3f} < {thresholds.min_iou_ge_075_rate:.3f}"
        )
    if int(row["n_images"]) < thresholds.min_image_count:
        failures.append(
            f"n_images {int(row['n_images'])} < {thresholds.min_image_count}"
        )
    if runner_up_ratio > thresholds.max_runner_up_ratio:
        failures.append(
            "runner_up_ratio "
            f"{runner_up_ratio:.3f} > {thresholds.max_runner_up_ratio:.3f}"
        )
    return failures


def _serialize_candidate_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "coco_category_id": int(row["coco_category_id"]),
        "coco_category_name": str(row["coco_category_name"]),
        "candidate_kind": str(row["candidate_kind"]),
        "candidate_source": str(row["candidate_source"]),
        "prior_kind": row.get("prior_kind"),
        "n_match": int(row["n_match"]),
        "precision_like": float(row["precision_like"]),
        "coverage_like": float(row["coverage_like"]),
        "mean_iou": float(row["mean_iou"]),
        "median_iou": float(row["median_iou"]),
        "iou_ge_05_rate": float(row["iou_ge_05_rate"]),
        "iou_ge_075_rate": float(row["iou_ge_075_rate"]),
        "n_images": int(row["n_images"]),
    }


def infer_lvis_to_coco80_mapping(
    evidence: MappingEvidenceResult,
    lvis_categories_by_id: Mapping[int, Mapping[str, Any]],
    *,
    config: AnalysisConfig,
) -> LearnedMappingResult:
    rows: list[dict[str, Any]] = []
    by_lvis_category_id: dict[int, LearnedMapping] = {}
    lvis_ids_by_coco_and_tier: dict[str, dict[int, set[int]]] = {
        "strict": defaultdict(set),
        "usable": defaultdict(set),
        "strict_plus_usable": defaultdict(set),
    }
    tier_counts: Counter[str] = Counter()
    kind_counts: Counter[str] = Counter()
    considered_lvis_category_count = 0

    for lvis_category_id, lvis_category in sorted(lvis_categories_by_id.items()):
        candidate_rows = list(evidence.rows_by_lvis_category.get(int(lvis_category_id), []))
        eligible_instances = int(
            evidence.eligible_instances_by_lvis_category.get(int(lvis_category_id), 0)
        )
        if eligible_instances > 0 or candidate_rows:
            considered_lvis_category_count += 1
        exact_candidates = [
            row for row in candidate_rows if bool(row["has_exact_canonical_prior"])
        ]
        if exact_candidates:
            considered_candidates = exact_candidates
            mapping_kind = "exact_canonical"
        else:
            considered_candidates = [
                row for row in candidate_rows if int(row["n_match"]) > 0
            ]
            mapping_kind = "semantic_evidence" if considered_candidates else None
        considered_candidates.sort(
            key=lambda row: (
                -int(row["n_match"]),
                -float(row["precision_like"]),
                -float(row["mean_iou"]),
                -int(row["n_images"]),
                str(row["coco_category_name"]),
            )
        )
        top_candidates = [
            _serialize_candidate_summary(row)
            for row in candidate_rows[:5]
        ]
        confidence_tier = "reject"
        mapped_coco_category_id: int | None = None
        mapped_coco_category_name: str | None = None
        prior_kind: str | None = None
        evidence_summary: dict[str, Any] | None = None
        rejection_reason: str | None = None

        if not considered_candidates:
            if eligible_instances == 0:
                rejection_reason = "no eligible LVIS instances on the analyzed overlap"
            elif candidate_rows:
                rejection_reason = (
                    "only exact/canonical candidates with zero matched-instance evidence"
                )
            else:
                rejection_reason = "no candidate COCO classes observed for this LVIS category"
        else:
            best_row = considered_candidates[0]
            runner_up_matches = (
                int(considered_candidates[1]["n_match"])
                if len(considered_candidates) > 1
                else 0
            )
            runner_up_ratio = (
                float(runner_up_matches / int(best_row["n_match"]))
                if int(best_row["n_match"]) > 0
                else 0.0
            )
            if mapping_kind == "exact_canonical":
                strict_thresholds = config.strict_exact_thresholds
                usable_thresholds = config.usable_exact_thresholds
            else:
                strict_thresholds = config.strict_semantic_thresholds
                usable_thresholds = config.usable_semantic_thresholds
            strict_failures = _threshold_failures(
                best_row,
                strict_thresholds,
                runner_up_ratio=runner_up_ratio,
            )
            usable_failures = _threshold_failures(
                best_row,
                usable_thresholds,
                runner_up_ratio=runner_up_ratio,
            )
            if int(best_row["n_match"]) == 0:
                rejection_reason = (
                    "candidate mapping has no matched-instance evidence on the analyzed overlap"
                )
            elif not strict_failures:
                confidence_tier = "strict"
            elif not usable_failures:
                confidence_tier = "usable"
            else:
                rejection_reason = (
                    "insufficient evidence: "
                    + "; ".join(usable_failures[:4])
                )
            prior_kind = str(best_row["prior_kind"]) if best_row.get("prior_kind") else None
            evidence_summary = {
                **_serialize_candidate_summary(best_row),
                "runner_up_match_count": int(runner_up_matches),
                "runner_up_ratio": float(runner_up_ratio),
            }
            if confidence_tier != "reject":
                mapped_coco_category_id = int(best_row["coco_category_id"])
                mapped_coco_category_name = str(best_row["coco_category_name"])

        learning_row = LearnedMapping(
            lvis_category_id=int(lvis_category_id),
            lvis_category_name=str(lvis_category["name"]),
            lvis_frequency=str(lvis_category.get("frequency", "unknown")),
            confidence_tier=confidence_tier,
            mapping_kind=mapping_kind if confidence_tier != "reject" else None,
            mapped_coco_category_id=mapped_coco_category_id,
            mapped_coco_category_name=mapped_coco_category_name,
            prior_kind=prior_kind,
            evidence_summary=evidence_summary,
            top_candidates=top_candidates,
            rejection_reason=rejection_reason,
        )
        by_lvis_category_id[int(lvis_category_id)] = learning_row
        row_payload = {
            "lvis_category_id": int(learning_row.lvis_category_id),
            "lvis_category_name": learning_row.lvis_category_name,
            "lvis_frequency": learning_row.lvis_frequency,
            "confidence_tier": learning_row.confidence_tier,
            "mapping_kind": learning_row.mapping_kind,
            "mapped_coco_category_id": learning_row.mapped_coco_category_id,
            "mapped_coco_category_name": learning_row.mapped_coco_category_name,
            "prior_kind": learning_row.prior_kind,
            "evidence_summary": learning_row.evidence_summary,
            "top_candidates": learning_row.top_candidates,
            "rejection_reason": learning_row.rejection_reason,
        }
        rows.append(row_payload)
        tier_counts[learning_row.confidence_tier] += 1
        if learning_row.mapping_kind is not None:
            kind_counts[str(learning_row.mapping_kind)] += 1
        if learning_row.mapped_coco_category_id is not None:
            if learning_row.confidence_tier == "strict":
                lvis_ids_by_coco_and_tier["strict"][
                    int(learning_row.mapped_coco_category_id)
                ].add(int(learning_row.lvis_category_id))
            if learning_row.confidence_tier in {"strict", "usable"}:
                lvis_ids_by_coco_and_tier["strict_plus_usable"][
                    int(learning_row.mapped_coco_category_id)
                ].add(int(learning_row.lvis_category_id))
                if learning_row.confidence_tier == "usable":
                    lvis_ids_by_coco_and_tier["usable"][
                        int(learning_row.mapped_coco_category_id)
                    ].add(int(learning_row.lvis_category_id))

    summary = {
        "lvis_categories_total": int(len(lvis_categories_by_id)),
        "lvis_categories_considered": int(considered_lvis_category_count),
        "strict_mapping_count": int(tier_counts["strict"]),
        "usable_mapping_count": int(tier_counts["usable"]),
        "reject_mapping_count": int(tier_counts["reject"]),
        "accepted_mapping_kind_counts": dict(sorted(kind_counts.items())),
    }
    return LearnedMappingResult(
        rows=rows,
        by_lvis_category_id=by_lvis_category_id,
        lvis_ids_by_coco_and_tier={
            tier: {
                int(coco_category_id): set(lvis_ids)
                for coco_category_id, lvis_ids in tier_mapping.items()
            }
            for tier, tier_mapping in lvis_ids_by_coco_and_tier.items()
        },
        summary=summary,
    )


def _recover_instances_for_tier_view(
    coco_dataset: LoadedDataset,
    lvis_dataset: LoadedDataset,
    *,
    shared_image_ids: Sequence[int],
    learned_mapping: LearnedMappingResult,
    allowed_tiers: set[str],
    tier_view_name: str,
    config: AnalysisConfig,
) -> dict[str, Any]:
    allowed_mappings = {
        lvis_category_id: mapping
        for lvis_category_id, mapping in learned_mapping.by_lvis_category_id.items()
        if mapping.confidence_tier in allowed_tiers
        and mapping.mapped_coco_category_id is not None
    }
    mapped_lvis_ids_by_coco = defaultdict(set)
    for mapping in allowed_mappings.values():
        mapped_lvis_ids_by_coco[int(mapping.mapped_coco_category_id)].add(
            int(mapping.lvis_category_id)
        )

    recovered_rows: list[dict[str, Any]] = []
    recovered_counts_by_image: Counter[int] = Counter()
    recovered_counts_by_coco: Counter[int] = Counter()
    recovered_coco_categories_by_image: dict[int, set[str]] = defaultdict(set)
    blocked_coco_categories_by_image: dict[int, set[str]] = defaultdict(set)
    recovered_lvis_categories_by_coco: dict[int, set[str]] = defaultdict(set)

    for image_id in shared_image_ids:
        coco_image = coco_dataset.images_by_id[image_id]
        lvis_image = lvis_dataset.images_by_id[image_id]
        coco_image_split = (
            lvis_image.get("coco_image_split")
            or coco_image.get("coco_image_split")
            or "unknown"
        )
        lvis_source_name = str(lvis_image.get("source_name", "unknown"))
        coco_annotations_all = coco_dataset.annotations_by_image.get(image_id, [])
        lvis_annotations_all = lvis_dataset.annotations_by_image.get(image_id, [])
        if config.ignore_crowd:
            coco_annotations = [
                annotation
                for annotation in coco_annotations_all
                if int(annotation.get("iscrowd", 0)) == 0
            ]
            lvis_annotations = [
                annotation
                for annotation in lvis_annotations_all
                if int(annotation.get("iscrowd", 0)) == 0
            ]
        else:
            coco_annotations = list(coco_annotations_all)
            lvis_annotations = list(lvis_annotations_all)
        coco_annotations_by_category: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in coco_annotations:
            coco_annotations_by_category[int(annotation["category_id"])].append(annotation)

        image_not_exhaustive_ids = {
            int(value)
            for value in lvis_image.get("not_exhaustive_category_ids", [])
        }
        blocked_coco_category_ids = {
            int(coco_category_id)
            for coco_category_id, lvis_ids in mapped_lvis_ids_by_coco.items()
            if image_not_exhaustive_ids.intersection(lvis_ids)
        }
        for coco_category_id in blocked_coco_category_ids:
            blocked_coco_categories_by_image[int(image_id)].add(
                str(coco_dataset.categories_by_id[int(coco_category_id)]["name"])
            )

        candidate_groups: dict[int, list[tuple[dict[str, Any], LearnedMapping]]] = defaultdict(list)
        for lvis_annotation in lvis_annotations:
            lvis_category_id = int(lvis_annotation["category_id"])
            mapping = allowed_mappings.get(lvis_category_id)
            if mapping is None or mapping.mapped_coco_category_id is None:
                continue
            if int(mapping.mapped_coco_category_id) in blocked_coco_category_ids:
                continue
            if float(lvis_annotation["bbox_area"]) < float(config.recovery_min_lvis_box_area):
                continue
            candidate_groups[int(mapping.mapped_coco_category_id)].append(
                (dict(lvis_annotation), mapping)
            )

        for coco_category_id, candidates in candidate_groups.items():
            same_category_coco_annotations = coco_annotations_by_category.get(
                int(coco_category_id),
                [],
            )
            lvis_annotations_for_match = [candidate[0] for candidate in candidates]
            matched_indices, best_same_category_ious = _greedy_match_by_iou(
                lvis_annotations_for_match,
                same_category_coco_annotations,
                iou_threshold=float(config.recovery_iou_threshold),
            )
            for candidate_index, (lvis_annotation, mapping) in enumerate(candidates):
                if candidate_index in matched_indices:
                    continue
                best_any_coco_iou = _best_iou_against_annotations(
                    lvis_annotation["bbox_xyxy"],
                    coco_annotations,
                )
                (
                    best_conflicting_coco_iou,
                    best_conflicting_coco_category_id,
                    best_conflicting_coco_category_name,
                ) = _best_iou_against_other_categories(
                    lvis_annotation["bbox_xyxy"],
                    coco_annotations,
                    excluded_category_id=int(coco_category_id),
                )
                if (
                    best_conflicting_coco_iou
                    >= float(config.recovery_max_conflicting_coco_iou)
                ):
                    continue
                evidence_summary = mapping.evidence_summary or {}
                recovered_rows.append(
                    {
                        "recovery_view": tier_view_name,
                        "image_id": int(image_id),
                        "coco_image_split": str(coco_image_split),
                        "lvis_source_name": lvis_source_name,
                        "file_name": str(
                            lvis_image.get("file_name")
                            or coco_image.get("file_name")
                            or ""
                        ),
                        "lvis_ann_id": int(lvis_annotation["annotation_id"]),
                        "lvis_category_id": int(lvis_annotation["category_id"]),
                        "lvis_category_name": str(lvis_annotation["category_name"]),
                        "lvis_frequency": str(lvis_annotation.get("frequency", "unknown")),
                        "mapped_coco_category_id": int(coco_category_id),
                        "mapped_coco_category_name": str(mapping.mapped_coco_category_name),
                        "mapping_tier": str(mapping.confidence_tier),
                        "mapping_kind": mapping.mapping_kind,
                        "prior_kind": mapping.prior_kind,
                        "bbox_xyxy": [float(value) for value in lvis_annotation["bbox_xyxy"]],
                        "bbox_xywh": [float(value) for value in lvis_annotation["bbox_xywh"]],
                        "bbox_area": float(lvis_annotation["bbox_area"]),
                        "segmentation_summary": lvis_annotation.get("segmentation_summary"),
                        "best_same_category_iou": float(
                            best_same_category_ious[candidate_index]
                        ),
                        "best_any_coco_iou": float(best_any_coco_iou),
                        "best_conflicting_coco_iou": float(best_conflicting_coco_iou),
                        "best_conflicting_coco_category_id": best_conflicting_coco_category_id,
                        "best_conflicting_coco_category_name": best_conflicting_coco_category_name,
                        "mapping_evidence_n_match": int(evidence_summary.get("n_match", 0)),
                        "mapping_evidence_precision_like": float(
                            evidence_summary.get("precision_like", 0.0)
                        ),
                        "mapping_evidence_coverage_like": float(
                            evidence_summary.get("coverage_like", 0.0)
                        ),
                        "mapping_evidence_mean_iou": float(
                            evidence_summary.get("mean_iou", 0.0)
                        ),
                        "mapping_evidence_median_iou": float(
                            evidence_summary.get("median_iou", 0.0)
                        ),
                        "mapping_evidence_n_images": int(
                            evidence_summary.get("n_images", 0)
                        ),
                        "why_recovered": (
                            f"{mapping.confidence_tier} {mapping.mapping_kind} mapping; "
                            f"no COCO {mapping.mapped_coco_category_name} annotation "
                            f"at IoU >= {float(config.recovery_iou_threshold):.2f}; "
                            f"best conflicting other-class IoU "
                            f"{float(best_conflicting_coco_iou):.3f} < "
                            f"{float(config.recovery_max_conflicting_coco_iou):.2f}"
                        ),
                    }
                )
                recovered_counts_by_image[int(image_id)] += 1
                recovered_counts_by_coco[int(coco_category_id)] += 1
                recovered_coco_categories_by_image[int(image_id)].add(
                    str(mapping.mapped_coco_category_name)
                )
                recovered_lvis_categories_by_coco[int(coco_category_id)].add(
                    str(lvis_annotation["category_name"])
                )

    return {
        "rows": recovered_rows,
        "recovered_counts_by_image": recovered_counts_by_image,
        "recovered_counts_by_coco": recovered_counts_by_coco,
        "recovered_coco_categories_by_image": {
            int(image_id): set(category_names)
            for image_id, category_names in recovered_coco_categories_by_image.items()
        },
        "blocked_coco_categories_by_image": {
            int(image_id): set(category_names)
            for image_id, category_names in blocked_coco_categories_by_image.items()
        },
        "recovered_lvis_categories_by_coco": {
            int(coco_category_id): set(category_names)
            for coco_category_id, category_names in recovered_lvis_categories_by_coco.items()
        },
    }


def recover_projected_coco80_instances(
    coco_dataset: LoadedDataset,
    lvis_dataset: LoadedDataset,
    *,
    shared_image_ids: Sequence[int],
    learned_mapping: LearnedMappingResult,
    config: AnalysisConfig,
) -> RecoveryResult:
    strict_view = _recover_instances_for_tier_view(
        coco_dataset,
        lvis_dataset,
        shared_image_ids=shared_image_ids,
        learned_mapping=learned_mapping,
        allowed_tiers={"strict"},
        tier_view_name="strict_only",
        config=config,
    )
    strict_plus_usable_view = _recover_instances_for_tier_view(
        coco_dataset,
        lvis_dataset,
        shared_image_ids=shared_image_ids,
        learned_mapping=learned_mapping,
        allowed_tiers={"strict", "usable"},
        tier_view_name="strict_plus_usable",
        config=config,
    )

    recovered_union: dict[tuple[int, int], dict[str, Any]] = {}
    for row in strict_view["rows"]:
        key = (int(row["image_id"]), int(row["lvis_ann_id"]))
        recovered_union[key] = {
            **row,
            "included_in_strict_only": True,
            "included_in_strict_plus_usable": False,
        }
    for row in strict_plus_usable_view["rows"]:
        key = (int(row["image_id"]), int(row["lvis_ann_id"]))
        existing = recovered_union.get(key)
        if existing is None:
            recovered_union[key] = {
                **row,
                "included_in_strict_only": False,
                "included_in_strict_plus_usable": True,
            }
            continue
        existing["included_in_strict_plus_usable"] = True

    recovered_instances = [
        recovered_union[key]
        for key in sorted(recovered_union.keys())
    ]

    per_image_rows: list[dict[str, Any]] = []
    for image_id in shared_image_ids:
        coco_image = coco_dataset.images_by_id[image_id]
        lvis_image = lvis_dataset.images_by_id[image_id]
        coco_image_split = (
            lvis_image.get("coco_image_split")
            or coco_image.get("coco_image_split")
            or "unknown"
        )
        lvis_source_name = str(lvis_image.get("source_name", "unknown"))
        per_image_rows.append(
            {
                "image_id": int(image_id),
                "coco_image_split": str(coco_image_split),
                "lvis_source_name": lvis_source_name,
                "recovered_strict_count": int(
                    strict_view["recovered_counts_by_image"].get(int(image_id), 0)
                ),
                "recovered_strict_plus_usable_count": int(
                    strict_plus_usable_view["recovered_counts_by_image"].get(
                        int(image_id),
                        0,
                    )
                ),
                "blocked_not_exhaustive_coco_categories": "|".join(
                    sorted(
                        strict_plus_usable_view["blocked_coco_categories_by_image"].get(
                            int(image_id),
                            set(),
                        )
                    )
                ),
                "recovered_coco_categories": "|".join(
                    sorted(
                        strict_plus_usable_view["recovered_coco_categories_by_image"].get(
                            int(image_id),
                            set(),
                        )
                    )
                ),
            }
        )

    per_category_rows: list[dict[str, Any]] = []
    for coco_category_id, coco_category in sorted(coco_dataset.categories_by_id.items()):
        strict_count = int(strict_view["recovered_counts_by_coco"].get(int(coco_category_id), 0))
        combined_count = int(
            strict_plus_usable_view["recovered_counts_by_coco"].get(int(coco_category_id), 0)
        )
        if strict_count == 0 and combined_count == 0:
            continue
        recovered_lvis_categories = set(
            strict_view["recovered_lvis_categories_by_coco"].get(int(coco_category_id), set())
        )
        recovered_lvis_categories.update(
            strict_plus_usable_view["recovered_lvis_categories_by_coco"].get(
                int(coco_category_id),
                set(),
            )
        )
        per_category_rows.append(
            {
                "coco_category_id": int(coco_category_id),
                "coco_category_name": str(coco_category["name"]),
                "recovered_strict_count": strict_count,
                "recovered_strict_plus_usable_count": combined_count,
                "recovered_lvis_categories": "|".join(sorted(recovered_lvis_categories)),
            }
        )

    top_images_strict = sorted(
        (
            {
                "image_id": int(row["image_id"]),
                "coco_image_split": str(row["coco_image_split"]),
                "lvis_source_name": str(row["lvis_source_name"]),
                "recovered_count": int(row["recovered_strict_count"]),
                "recovered_coco_categories": str(row["recovered_coco_categories"]),
            }
            for row in per_image_rows
            if int(row["recovered_strict_count"]) > 0
        ),
        key=lambda row: (int(row["recovered_count"]), int(row["image_id"])),
        reverse=True,
    )[:20]
    top_images_strict_plus_usable = sorted(
        (
            {
                "image_id": int(row["image_id"]),
                "coco_image_split": str(row["coco_image_split"]),
                "lvis_source_name": str(row["lvis_source_name"]),
                "recovered_count": int(row["recovered_strict_plus_usable_count"]),
                "recovered_coco_categories": str(row["recovered_coco_categories"]),
            }
            for row in per_image_rows
            if int(row["recovered_strict_plus_usable_count"]) > 0
        ),
        key=lambda row: (int(row["recovered_count"]), int(row["image_id"])),
        reverse=True,
    )[:20]
    summary = {
        "strict_only_count": int(len(strict_view["rows"])),
        "strict_plus_usable_count": int(len(strict_plus_usable_view["rows"])),
        "per_coco_category_recovered_counts": per_category_rows,
        "top_images_strict": top_images_strict,
        "top_images_strict_plus_usable": top_images_strict_plus_usable,
    }
    return RecoveryResult(
        recovered_instances=recovered_instances,
        per_image_rows=per_image_rows,
        per_category_rows=per_category_rows,
        summary=summary,
    )


def _render_projection_report(
    learned_mapping: LearnedMappingResult,
    recovery: RecoveryResult,
) -> str:
    exact_supported = [
        row
        for row in learned_mapping.rows
        if row["confidence_tier"] in {"strict", "usable"}
        and row["mapping_kind"] == "exact_canonical"
    ]
    exact_supported.sort(
        key=lambda row: (
            -int((row.get("evidence_summary") or {}).get("n_match", 0)),
            str(row["lvis_category_name"]),
        )
    )
    semantic_supported = [
        row
        for row in learned_mapping.rows
        if row["confidence_tier"] in {"strict", "usable"}
        and row["mapping_kind"] == "semantic_evidence"
    ]
    semantic_supported.sort(
        key=lambda row: (
            -int((row.get("evidence_summary") or {}).get("n_match", 0)),
            str(row["lvis_category_name"]),
        )
    )
    rejected_rows = [
        row
        for row in learned_mapping.rows
        if row["confidence_tier"] == "reject" and row.get("top_candidates")
    ]
    rejected_rows.sort(
        key=lambda row: (
            -int((row["top_candidates"][0]).get("n_match", 0)),
            str(row["lvis_category_name"]),
        )
    )
    lines = [
        "# LVIS to COCO-80 Projection Report",
        "",
        "## Caveats",
        "- Local LVIS JSONs expose `neg_category_ids` and `not_exhaustive_category_ids`, but not `pos_category_ids`.",
        "- Matching is box-based; segmentation is preserved only as an output summary, not as a matching signal.",
        "- Exact/canonical mappings and semantic evidence-backed mappings are inferred separately and stay explicit in `learned_mapping.json`.",
        "",
        "## Top Exact/Canonical Mappings",
    ]
    if exact_supported:
        for row in exact_supported[:12]:
            evidence_summary = row.get("evidence_summary") or {}
            lines.append(
                f"- {row['lvis_category_name']} -> {row['mapped_coco_category_name']} "
                f"({row['confidence_tier']}, n_match={int(evidence_summary.get('n_match', 0))}, "
                f"precision={float(evidence_summary.get('precision_like', 0.0)):.3f}, "
                f"mean_iou={float(evidence_summary.get('mean_iou', 0.0)):.3f})"
            )
    else:
        lines.append("- No exact/canonical mappings reached the acceptance thresholds.")
    lines.extend(["", "## Top Semantic-but-Supported Mappings"])
    if semantic_supported:
        for row in semantic_supported[:12]:
            evidence_summary = row.get("evidence_summary") or {}
            lines.append(
                f"- {row['lvis_category_name']} -> {row['mapped_coco_category_name']} "
                f"({row['confidence_tier']}, n_match={int(evidence_summary.get('n_match', 0))}, "
                f"precision={float(evidence_summary.get('precision_like', 0.0)):.3f}, "
                f"mean_iou={float(evidence_summary.get('mean_iou', 0.0)):.3f})"
            )
    else:
        lines.append("- No semantic-only mappings cleared the current thresholds.")
    lines.extend(["", "## Rejected / Ambiguous Mappings"])
    if rejected_rows:
        for row in rejected_rows[:12]:
            top_candidate = row["top_candidates"][0]
            lines.append(
                f"- {row['lvis_category_name']}: rejected. Top candidate "
                f"{top_candidate['coco_category_name']} had n_match={int(top_candidate['n_match'])}, "
                f"precision={float(top_candidate['precision_like']):.3f}. "
                f"Reason: {row['rejection_reason']}"
            )
    else:
        lines.append("- No rejected mappings had enough candidate evidence to highlight.")
    lines.extend(["", "## Crowded-Scene Recovery Examples"])
    top_images = recovery.summary.get("top_images_strict_plus_usable", [])
    if top_images:
        for row in top_images[:12]:
            lines.append(
                f"- image {int(row['image_id'])} ({row['coco_image_split']}): "
                f"{int(row['recovered_count'])} recovered instances; "
                f"categories={row['recovered_coco_categories']}"
            )
    else:
        lines.append("- No recoveries were produced under the current thresholds.")
    return "\n".join(lines) + "\n"


def analyze_lvis_to_coco80_projection(
    coco_payloads: Sequence[Mapping[str, Any]],
    lvis_payloads: Sequence[Mapping[str, Any]],
    *,
    coco_source_names: Sequence[str] | None = None,
    lvis_source_names: Sequence[str] | None = None,
    config: AnalysisConfig | None = None,
) -> ProjectionAnalysisResult:
    cfg = config or AnalysisConfig()
    coco_dataset, lvis_dataset = _load_merged_analysis_datasets(
        coco_payloads,
        lvis_payloads,
        coco_source_names=coco_source_names,
        lvis_source_names=lvis_source_names,
    )
    shared_image_ids, shared_by_coco_split, shared_by_lvis_source = _resolve_shared_image_ids(
        coco_dataset,
        lvis_dataset,
        config=cfg,
    )
    evidence = mine_lvis_to_coco80_mapping_evidence(
        coco_dataset,
        lvis_dataset,
        shared_image_ids=shared_image_ids,
        config=cfg,
    )
    learned_mapping = infer_lvis_to_coco80_mapping(
        evidence,
        lvis_dataset.categories_by_id,
        config=cfg,
    )
    recovery = recover_projected_coco80_instances(
        coco_dataset,
        lvis_dataset,
        shared_image_ids=shared_image_ids,
        learned_mapping=learned_mapping,
        config=cfg,
    )
    report_markdown = _render_projection_report(learned_mapping, recovery)
    lvis_pos_category_ids_present = any(
        "pos_category_ids" in image_info
        for image_info in lvis_dataset.images_by_id.values()
    )
    summary = {
        "config": {
            "ignore_crowd": bool(cfg.ignore_crowd),
            "mapping_mode": str(cfg.mapping_mode),
            "max_images": cfg.max_images,
            "allowed_coco_image_splits": list(cfg.allowed_coco_image_splits),
            "evidence_pair_iou_threshold": float(cfg.evidence_pair_iou_threshold),
            "recovery_iou_threshold": float(cfg.recovery_iou_threshold),
            "recovery_min_lvis_box_area": float(cfg.recovery_min_lvis_box_area),
            "recovery_max_conflicting_coco_iou": float(
                cfg.recovery_max_conflicting_coco_iou
            ),
            "strict_exact_thresholds": cfg.strict_exact_thresholds.__dict__,
            "usable_exact_thresholds": cfg.usable_exact_thresholds.__dict__,
            "strict_semantic_thresholds": cfg.strict_semantic_thresholds.__dict__,
            "usable_semantic_thresholds": cfg.usable_semantic_thresholds.__dict__,
        },
        "limitations": {
            "lvis_pos_category_ids_present": bool(lvis_pos_category_ids_present),
            "note": (
                "Local LVIS JSONs do not expose pos_category_ids; the pipeline relies on "
                "LVIS annotations plus neg/not_exhaustive metadata only."
            ),
        },
        "inputs": {
            "coco_image_count": len(coco_dataset.images_by_id),
            "lvis_image_count": len(lvis_dataset.images_by_id),
            "coco_invalid_annotations": int(coco_dataset.invalid_annotation_count),
            "lvis_invalid_annotations": int(lvis_dataset.invalid_annotation_count),
        },
        "shared_images": {
            "count": len(shared_image_ids),
            "by_coco_image_split": shared_by_coco_split,
            "by_lvis_source_name": shared_by_lvis_source,
        },
        "mapping_evidence": evidence.summary,
        "learned_mapping": learned_mapping.summary,
        "recovery": recovery.summary,
    }
    return ProjectionAnalysisResult(
        summary=summary,
        mapping_evidence_rows=evidence.rows,
        learned_mapping_rows=learned_mapping.rows,
        recovered_instances=recovery.recovered_instances,
        recovered_per_image_rows=recovery.per_image_rows,
        recovered_per_category_rows=recovery.per_category_rows,
        report_markdown=report_markdown,
    )


def write_projection_outputs(
    result: ProjectionAnalysisResult,
    *,
    output_dir: Path,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    mapping_evidence_path = output_dir / "mapping_evidence.csv"
    learned_mapping_path = output_dir / "learned_mapping.json"
    recovered_instances_path = output_dir / "recovered_coco80_instances.jsonl"
    per_image_path = output_dir / "per_image.csv"
    per_category_path = output_dir / "per_category.csv"
    report_path = output_dir / "report.md"

    _write_json(summary_path, result.summary)
    _write_csv(
        mapping_evidence_path,
        result.mapping_evidence_rows,
        fieldnames=_MAPPING_EVIDENCE_FIELDNAMES,
    )
    _write_json(
        learned_mapping_path,
        {
            "summary": result.summary["learned_mapping"],
            "mappings": result.learned_mapping_rows,
        },
    )
    _write_jsonl(recovered_instances_path, result.recovered_instances)
    _write_csv(
        per_image_path,
        result.recovered_per_image_rows,
        fieldnames=_RECOVERED_PER_IMAGE_FIELDNAMES,
    )
    _write_csv(
        per_category_path,
        result.recovered_per_category_rows,
        fieldnames=_RECOVERED_PER_CATEGORY_FIELDNAMES,
    )
    report_path.write_text(result.report_markdown, encoding="utf-8")
    return {
        "summary_json": str(summary_path),
        "mapping_evidence_csv": str(mapping_evidence_path),
        "learned_mapping_json": str(learned_mapping_path),
        "recovered_coco80_instances_jsonl": str(recovered_instances_path),
        "per_image_csv": str(per_image_path),
        "per_category_csv": str(per_category_path),
        "report_md": str(report_path),
    }


def run_coco_lvis_projection_analysis(
    *,
    output_dir: Path,
    coco_annotation_paths: Sequence[Path] = DEFAULT_COCO_ANNOTATION_PATHS,
    lvis_annotation_paths: Sequence[Path] = DEFAULT_LVIS_ANNOTATION_PATHS,
    config: AnalysisConfig | None = None,
) -> dict[str, Any]:
    coco_paths = [Path(path) for path in coco_annotation_paths]
    lvis_paths = [Path(path) for path in lvis_annotation_paths]
    coco_payloads = [_load_json(path) for path in coco_paths]
    lvis_payloads = [_load_json(path) for path in lvis_paths]
    result = analyze_lvis_to_coco80_projection(
        coco_payloads,
        lvis_payloads,
        coco_source_names=[_source_name_from_path(path) for path in coco_paths],
        lvis_source_names=[_source_name_from_path(path) for path in lvis_paths],
        config=config,
    )
    artifact_paths = write_projection_outputs(result, output_dir=output_dir)
    summary_with_artifacts = dict(result.summary)
    summary_with_artifacts["artifacts"] = artifact_paths
    _write_json(output_dir / "summary.json", summary_with_artifacts)
    return summary_with_artifacts
