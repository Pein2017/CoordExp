from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.analysis.coco_lvis_missing_objects import (
    DEFAULT_COCO_ANNOTATION_PATHS,
    DEFAULT_LVIS_ANNOTATION_PATHS,
    LoadedDataset,
    _load_json,
    _load_merged_analysis_datasets,
    _source_name_from_path,
)
from src.vis.gt_vs_pred import (
    materialize_gt_vs_pred_vis_resource,
    render_gt_vs_pred_review,
)


@dataclass(frozen=True)
class SemanticVisConfig:
    examples_per_mapping: int = 2
    max_total_gt_objects: int = 12
    max_total_pred_objects: int = 12
    max_sibling_lvis_instances: int = 1
    auto_top_mappings: int = 5
    include_sibling_lvis_in_gt: bool = False


@dataclass(frozen=True)
class SemanticVisExample:
    record_idx: int
    image_id: int
    image: str
    width: int
    height: int
    coco_image_split: str
    lvis_source_name: str
    lvis_category_id: int
    lvis_category_name: str
    mapped_coco_category_id: int
    mapped_coco_category_name: str
    mapping_tier: str
    mapping_evidence_n_match: int
    mapping_evidence_precision_like: float
    recovered_target_count: int
    target_lvis_count: int
    coco_pred_count: int
    sibling_lvis_count: int
    sibling_lvis_categories: tuple[str, ...]
    target_lvis_annotations: tuple[dict[str, Any], ...]
    sibling_lvis_annotations: tuple[dict[str, Any], ...]
    coco_annotations: tuple[dict[str, Any], ...]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object rows in {path}")
            rows.append(payload)
    return rows


def _projection_image_relpath(
    image_info: Mapping[str, Any],
    *,
    fallback_split: str,
) -> str:
    file_name = str(image_info.get("file_name") or "").strip()
    if not file_name:
        raise ValueError(f"Missing file_name in image info: {image_info}")
    image_split = str(image_info.get("coco_image_split") or fallback_split or "").strip()
    if image_split:
        return f"{image_split}/{file_name}"
    return file_name


def _build_target_category_names(
    recovered_rows: Sequence[Mapping[str, Any]],
    *,
    explicit_lvis_category_names: Sequence[str],
    auto_top_mappings: int,
) -> list[str]:
    if explicit_lvis_category_names:
        return [str(name) for name in explicit_lvis_category_names]
    recovered_counts = Counter(
        str(row["lvis_category_name"])
        for row in recovered_rows
        if str(row.get("mapping_tier")) == "usable"
        and str(row.get("mapping_kind")) == "semantic_evidence"
        and bool(row.get("included_in_strict_plus_usable"))
    )
    return [
        category_name
        for category_name, _ in recovered_counts.most_common(int(auto_top_mappings))
    ]


def _load_projection_artifacts(
    projection_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    learned_mapping_path = projection_dir / "learned_mapping.json"
    recovered_instances_path = projection_dir / "recovered_coco80_instances.jsonl"
    learned_mapping_payload = _read_json(learned_mapping_path)
    mappings = learned_mapping_payload.get("mappings")
    if not isinstance(mappings, list):
        raise ValueError(f"Missing list mappings in {learned_mapping_path}")
    recovered_rows = _read_jsonl(recovered_instances_path)
    return [dict(row) for row in mappings], recovered_rows


def select_representative_usable_semantic_examples(
    coco_dataset: LoadedDataset,
    lvis_dataset: LoadedDataset,
    *,
    learned_mapping_rows: Sequence[Mapping[str, Any]],
    recovered_rows: Sequence[Mapping[str, Any]],
    explicit_lvis_category_names: Sequence[str] = (),
    config: SemanticVisConfig | None = None,
) -> list[SemanticVisExample]:
    cfg = config or SemanticVisConfig()
    target_category_names = _build_target_category_names(
        recovered_rows,
        explicit_lvis_category_names=explicit_lvis_category_names,
        auto_top_mappings=cfg.auto_top_mappings,
    )
    learned_by_lvis_name = {
        str(row["lvis_category_name"]): dict(row)
        for row in learned_mapping_rows
    }
    accepted_mappings_by_coco_id: dict[int, set[int]] = defaultdict(set)
    for row in learned_mapping_rows:
        mapped_coco_category_id = row.get("mapped_coco_category_id")
        if mapped_coco_category_id is None:
            continue
        confidence_tier = str(row.get("confidence_tier") or "")
        if confidence_tier not in {"strict", "usable"}:
            continue
        accepted_mappings_by_coco_id[int(mapped_coco_category_id)].add(
            int(row["lvis_category_id"])
        )

    recovered_by_mapping_and_image: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in recovered_rows:
        if not (
            str(row.get("mapping_tier")) == "usable"
            and str(row.get("mapping_kind")) == "semantic_evidence"
            and bool(row.get("included_in_strict_plus_usable"))
        ):
            continue
        recovered_by_mapping_and_image[
            (int(row["lvis_category_id"]), int(row["image_id"]))
        ].append(dict(row))

    examples: list[SemanticVisExample] = []
    record_idx = 0
    for lvis_category_name in target_category_names:
        learned_row = learned_by_lvis_name.get(str(lvis_category_name))
        if learned_row is None:
            continue
        if not (
            str(learned_row.get("confidence_tier")) == "usable"
            and str(learned_row.get("mapping_kind")) == "semantic_evidence"
            and learned_row.get("mapped_coco_category_id") is not None
        ):
            continue
        lvis_category_id = int(learned_row["lvis_category_id"])
        mapped_coco_category_id = int(learned_row["mapped_coco_category_id"])
        candidate_examples: list[tuple[tuple[Any, ...], SemanticVisExample]] = []
        for (candidate_lvis_category_id, image_id), recovered_items in recovered_by_mapping_and_image.items():
            if candidate_lvis_category_id != lvis_category_id:
                continue
            coco_image = coco_dataset.images_by_id.get(int(image_id))
            lvis_image = lvis_dataset.images_by_id.get(int(image_id))
            if coco_image is None or lvis_image is None:
                continue
            target_lvis_annotations = tuple(
                annotation
                for annotation in lvis_dataset.annotations_by_image.get(int(image_id), [])
                if int(annotation["category_id"]) == lvis_category_id
                and int(annotation.get("iscrowd", 0)) == 0
            )
            sibling_lvis_annotations = tuple(
                annotation
                for annotation in lvis_dataset.annotations_by_image.get(int(image_id), [])
                if int(annotation["category_id"]) != lvis_category_id
                and int(annotation["category_id"])
                in accepted_mappings_by_coco_id.get(mapped_coco_category_id, set())
                and int(annotation.get("iscrowd", 0)) == 0
            )
            coco_annotations = tuple(
                annotation
                for annotation in coco_dataset.annotations_by_image.get(int(image_id), [])
                if int(annotation["category_id"]) == mapped_coco_category_id
                and int(annotation.get("iscrowd", 0)) == 0
            )
            if not target_lvis_annotations or not coco_annotations and not recovered_items:
                continue
            if len(target_lvis_annotations) > int(cfg.max_total_gt_objects):
                continue
            if len(coco_annotations) > int(cfg.max_total_pred_objects):
                continue
            if len(sibling_lvis_annotations) > int(cfg.max_sibling_lvis_instances):
                continue
            coco_image_split = str(
                lvis_image.get("coco_image_split")
                or coco_image.get("coco_image_split")
                or "unknown"
            )
            sibling_categories = tuple(
                sorted(
                    {
                        str(annotation["category_name"])
                        for annotation in sibling_lvis_annotations
                    }
                )
            )
            # `record_idx` must be unique per rendered record because it is used
            # downstream for stable filenames (e.g. vis_0007.png). We assign it
            # after selection so multiple candidates for the same mapping don't
            # accidentally share the same value.
            example = SemanticVisExample(
                record_idx=-1,
                image_id=int(image_id),
                image=_projection_image_relpath(
                    lvis_image,
                    fallback_split=coco_image_split,
                ),
                width=int(lvis_image["width"]),
                height=int(lvis_image["height"]),
                coco_image_split=coco_image_split,
                lvis_source_name=str(lvis_image.get("source_name", "unknown")),
                lvis_category_id=lvis_category_id,
                lvis_category_name=str(lvis_category_name),
                mapped_coco_category_id=mapped_coco_category_id,
                mapped_coco_category_name=str(learned_row["mapped_coco_category_name"]),
                mapping_tier=str(learned_row["confidence_tier"]),
                mapping_evidence_n_match=int(
                    (learned_row.get("evidence_summary") or {}).get("n_match", 0)
                ),
                mapping_evidence_precision_like=float(
                    (learned_row.get("evidence_summary") or {}).get(
                        "precision_like",
                        0.0,
                    )
                ),
                recovered_target_count=len(recovered_items),
                target_lvis_count=len(target_lvis_annotations),
                coco_pred_count=len(coco_annotations),
                sibling_lvis_count=len(sibling_lvis_annotations),
                sibling_lvis_categories=sibling_categories,
                target_lvis_annotations=target_lvis_annotations,
                sibling_lvis_annotations=sibling_lvis_annotations,
                coco_annotations=coco_annotations,
            )
            score = (
                int(example.sibling_lvis_count),
                -int(example.recovered_target_count),
                int(abs(example.target_lvis_count - example.coco_pred_count)),
                int(example.target_lvis_count + example.coco_pred_count),
                int(example.image_id),
            )
            candidate_examples.append((score, example))
        candidate_examples.sort(key=lambda item: item[0])
        for _, example in candidate_examples[: int(cfg.examples_per_mapping)]:
            examples.append(replace(example, record_idx=record_idx))
            record_idx += 1
    return examples


def _object_from_annotation(
    annotation: Mapping[str, Any],
    *,
    desc: str,
) -> dict[str, Any]:
    return {
        "index": int(annotation["annotation_id"]),
        "desc": desc,
        "bbox_2d": [int(round(float(value))) for value in annotation["bbox_xyxy"]],
    }


def build_semantic_example_record(
    example: SemanticVisExample,
    *,
    include_sibling_lvis_in_gt: bool = False,
) -> dict[str, Any]:
    gt_objects = [
        _object_from_annotation(
            annotation,
            desc=f"LVIS:{example.lvis_category_name}",
        )
        for annotation in example.target_lvis_annotations
    ]
    if include_sibling_lvis_in_gt:
        gt_objects.extend(
            _object_from_annotation(
                annotation,
                desc=f"LVIS-sibling:{annotation['category_name']}",
            )
            for annotation in example.sibling_lvis_annotations
        )
    pred_objects = [
        _object_from_annotation(
            annotation,
            desc=f"COCO:{example.mapped_coco_category_name}",
        )
        for annotation in example.coco_annotations
    ]
    return {
        "record_idx": int(example.record_idx),
        "image_id": int(example.image_id),
        "file_name": example.image.split("/")[-1],
        "image": example.image,
        "width": int(example.width),
        "height": int(example.height),
        "gt": gt_objects,
        "pred": pred_objects,
        "provenance": {
            "scene_type": "coco_lvis_semantic_mapping_audit",
            "lvis_category_id": int(example.lvis_category_id),
            "lvis_category_name": example.lvis_category_name,
            "mapped_coco_category_id": int(example.mapped_coco_category_id),
            "mapped_coco_category_name": example.mapped_coco_category_name,
            "recovered_target_count": int(example.recovered_target_count),
            "sibling_lvis_categories": list(example.sibling_lvis_categories),
        },
        "debug": {
            "mapping_tier": example.mapping_tier,
            "mapping_evidence_n_match": int(example.mapping_evidence_n_match),
            "mapping_evidence_precision_like": float(
                example.mapping_evidence_precision_like
            ),
        },
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_examples_csv(path: Path, examples: Sequence[SemanticVisExample]) -> None:
    fieldnames = [
        "record_idx",
        "image_id",
        "image",
        "coco_image_split",
        "lvis_source_name",
        "lvis_category_name",
        "mapped_coco_category_name",
        "mapping_tier",
        "mapping_evidence_n_match",
        "mapping_evidence_precision_like",
        "recovered_target_count",
        "target_lvis_count",
        "coco_pred_count",
        "sibling_lvis_count",
        "sibling_lvis_categories",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for example in examples:
            writer.writerow(
                {
                    "record_idx": int(example.record_idx),
                    "image_id": int(example.image_id),
                    "image": example.image,
                    "coco_image_split": example.coco_image_split,
                    "lvis_source_name": example.lvis_source_name,
                    "lvis_category_name": example.lvis_category_name,
                    "mapped_coco_category_name": example.mapped_coco_category_name,
                    "mapping_tier": example.mapping_tier,
                    "mapping_evidence_n_match": int(example.mapping_evidence_n_match),
                    "mapping_evidence_precision_like": float(
                        example.mapping_evidence_precision_like
                    ),
                    "recovered_target_count": int(example.recovered_target_count),
                    "target_lvis_count": int(example.target_lvis_count),
                    "coco_pred_count": int(example.coco_pred_count),
                    "sibling_lvis_count": int(example.sibling_lvis_count),
                    "sibling_lvis_categories": "|".join(example.sibling_lvis_categories),
                }
            )


def _render_index_markdown(
    examples: Sequence[SemanticVisExample],
    *,
    review_dir: Path,
) -> str:
    lines = [
        "# Usable Semantic Mapping Audit",
        "",
        "GT is LVIS target-subtype boxes for the selected mapping.",
        "Pred is COCO boxes for the mapped COCO-80 category.",
        "Orange on GT means LVIS target objects missing from COCO for this view.",
        "Red on Pred means COCO boxes without a matching LVIS target object for this view.",
        "",
        "## Examples",
    ]
    for example in examples:
        png_name = f"vis_{int(example.record_idx):04d}.png"
        lines.append(
            "- "
            f"[{png_name}](review/{png_name}) | "
            f"{example.lvis_category_name} -> {example.mapped_coco_category_name} | "
            f"image_id={int(example.image_id)} | "
            f"recovered_target_count={int(example.recovered_target_count)} | "
            f"target_lvis_count={int(example.target_lvis_count)} | "
            f"coco_pred_count={int(example.coco_pred_count)} | "
            f"sibling_lvis_count={int(example.sibling_lvis_count)} | "
            f"sibling_lvis_categories={','.join(example.sibling_lvis_categories) or 'none'} | "
            f"evidence_n_match={int(example.mapping_evidence_n_match)} | "
            f"evidence_precision_like={float(example.mapping_evidence_precision_like):.3f}"
        )
    return "\n".join(lines) + "\n"


def render_usable_semantic_mapping_audit(
    *,
    projection_dir: Path,
    output_dir: Path,
    coco_annotation_paths: Sequence[Path] = DEFAULT_COCO_ANNOTATION_PATHS,
    lvis_annotation_paths: Sequence[Path] = DEFAULT_LVIS_ANNOTATION_PATHS,
    explicit_lvis_category_names: Sequence[str] = (),
    root_image_dir: Path = Path("public_data/coco/raw/images"),
    config: SemanticVisConfig | None = None,
) -> dict[str, Any]:
    cfg = config or SemanticVisConfig()
    learned_mapping_rows, recovered_rows = _load_projection_artifacts(projection_dir)
    coco_paths = [Path(path) for path in coco_annotation_paths]
    lvis_paths = [Path(path) for path in lvis_annotation_paths]
    coco_payloads = [_load_json(path) for path in coco_paths]
    lvis_payloads = [_load_json(path) for path in lvis_paths]
    coco_dataset, lvis_dataset = _load_merged_analysis_datasets(
        coco_payloads,
        lvis_payloads,
        coco_source_names=[_source_name_from_path(path) for path in coco_paths],
        lvis_source_names=[_source_name_from_path(path) for path in lvis_paths],
    )
    examples = select_representative_usable_semantic_examples(
        coco_dataset,
        lvis_dataset,
        learned_mapping_rows=learned_mapping_rows,
        recovered_rows=recovered_rows,
        explicit_lvis_category_names=explicit_lvis_category_names,
        config=cfg,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_input_path = output_dir / "semantic_mapping_examples.jsonl"
    examples_csv_path = output_dir / "examples.csv"
    review_dir = output_dir / "review"
    canonical_path = output_dir / "vis_resources" / "gt_vs_pred.jsonl"
    index_path = output_dir / "index.md"

    scene_rows = [
        build_semantic_example_record(
            example,
            include_sibling_lvis_in_gt=bool(cfg.include_sibling_lvis_in_gt),
        )
        for example in examples
    ]
    _write_jsonl(scene_input_path, scene_rows)
    _write_examples_csv(examples_csv_path, examples)
    materialize_gt_vs_pred_vis_resource(
        scene_input_path,
        output_path=canonical_path,
        source_kind="coco_lvis_semantic_mapping_audit",
        materialize_matching=True,
    )
    render_gt_vs_pred_review(
        canonical_path,
        out_dir=review_dir,
        limit=0,
        root_image_dir=root_image_dir,
        root_source="semantic_mapping_audit",
        record_order="input",
    )
    index_path.write_text(
        _render_index_markdown(examples, review_dir=review_dir),
        encoding="utf-8",
    )
    return {
        "scene_input_jsonl": str(scene_input_path),
        "canonical_jsonl": str(canonical_path),
        "examples_csv": str(examples_csv_path),
        "index_md": str(index_path),
        "review_dir": str(review_dir),
        "example_count": int(len(examples)),
        "lvis_categories": [example.lvis_category_name for example in examples],
    }
