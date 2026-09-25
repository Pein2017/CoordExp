"""CPU-only deterministic admission for the 2026-09-22 pilot.

Selection is based on source annotations and frozen exclusions only.  The
processor is used for image/prompt planning, never for a model forward.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable, Mapping

from PIL import Image
from src.artifacts.utf8_json import canonical
from src.config.inference import InferConfig
from src.data.examples import raw_example_from_jsonl_row
from src.inference.inputs import plan_examples
from src.qwen.runtime_loading import QwenLoadOptions, load_qwen_components_from_options


OUTPUT_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-22-address-readout-pilot"
)
SELECTION_ROOT = OUTPUT_ROOT / "selection"
SOURCE_ROOT = Path("/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000_xy_sorted")
TRAIN_SOURCE = SOURCE_ROOT / "train.coord.jsonl"
VAL_SOURCE = SOURCE_ROOT / "val.coord.jsonl"
PANEL = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-18-untied-highconfidence18-natural/panel.json"
)
RECURRENCE_PANEL = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-19-recurrence-distribution-census/shared-panel.json"
)
FRESH_PANEL = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-17-readout-norm-fresh128/panel.json"
)
CURRENT_TRAIN = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-05-sft256-dev128-baseline/inputs-v3/train.jsonl"
)
CURRENT_DEV = CURRENT_TRAIN.with_name("dev.jsonl")
BASE_MODEL = Path(
    "/data/Qwen3-VL/model_cache/models/Qwen/"
    "Qwen3-VL-2B-Instruct-coordexp-natural-adjacent"
)
CHECKPOINT = Path(
    "/data/CoordExp/outputs/research/eight-coordinate-bbox-supervision/"
    "2026-08-05-closeout/artifacts/training/four-coordinate-xy/checkpoints/step-2444"
)
LOADER = Path(__file__).resolve().parents[3] / 'probes/model_profiles/mature_tied_untied.py'
SEED = 92219
TRAIN_COUNT, CAL_COUNT, EVAL_COUNT = 128, 32, 32
SCHEMA = "address_readout_pilot.admission.v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def binding(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    require(path.is_file(), f"expected file: {path}")
    return {"path": str(path), "sha256": file_hash(path), "size_bytes": path.stat().st_size}


def publish(path: Path, value: Any) -> None:
    data = canonical(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.is_file() and path.read_bytes() == data, f"immutable collision: {path}")
        return
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def publish_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    data = b"".join(canonical(dict(row)) for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.is_file() and path.read_bytes() == data, f"immutable collision: {path}")
        return
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _rows(path: Path) -> list[dict[str, Any]]:
    result = []
    with path.open() as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                require(isinstance(row, dict), f"non-object source row {path}:{number}")
                row["_source_row_number"] = number
                row["_source_row_sha256"] = hashlib.sha256(line.encode()).hexdigest()
                result.append(row)
    return result


def _panel_ids(path: Path) -> set[int]:
    x = json.loads(path.read_text())
    result: set[int] = set()
    def collect(value: Any) -> None:
        if isinstance(value, Mapping):
            if type(value.get("image_id")) is int:
                result.add(int(value["image_id"]))
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)
    collect(x)
    groups = x.get("groups", []) if isinstance(x, dict) else []
    if isinstance(groups, list):
        for group in groups:
            for case in group.get("cases", []):
                record = case.get("input_record", {})
                if isinstance(record.get("image_id"), int):
                    result.add(int(record["image_id"]))
    for case in x.get("cases", []) if isinstance(x, dict) else []:
        record = case.get("input_record", {})
        if isinstance(record.get("image_id"), int):
            result.add(int(record["image_id"]))
    return result


def _jsonl_ids(path: Path) -> set[int]:
    return {int(row["image_id"]) for row in _rows(path)}


def _coord(value: Any) -> int:
    require(isinstance(value, str) and value.startswith("<|coord_") and value.endswith("|>"), f"bad coordinate token: {value!r}")
    number = int(value[8:-2])
    require(0 <= number <= 999, f"coordinate outside 0..999: {number}")
    return number


def _validate_source(row: Mapping[str, Any], source: Path) -> tuple[Path | None, str | None]:
    objects = row.get("objects")
    if not isinstance(objects, list) or not objects:
        return None, "no_positive_objects"
    try:
        for obj in objects:
            require(isinstance(obj, dict) and isinstance(obj.get("desc"), str) and obj["desc"], "description")
            coords = [_coord(x) for x in obj.get("bbox_2d", [])]
            require(len(coords) == 4 and coords[0] < coords[2] and coords[1] < coords[3], "positive geometry")
        image = (source.parent / str(row["images"][0])).resolve(strict=True)
        require(image.is_file(), "image unavailable")
        require(type(row.get("width")) is int and type(row.get("height")) is int, "dimensions")
        require(row["width"] % 32 == 0 and row["height"] % 32 == 0, "grid dimensions")
        return image, None
    except (KeyError, TypeError, ValueError, OSError) as exc:
        return None, str(exc)


def _stratum(row: Mapping[str, Any]) -> dict[str, Any]:
    objects = list(row["objects"])
    counts = collections.Counter(str(obj["desc"]) for obj in objects)
    density = len(objects)
    density_bin = "1" if density == 1 else "2-4" if density <= 4 else "5-9" if density <= 9 else "10+"
    return {
        "density_bin": density_bin,
        "object_count": density,
        "class_counts": dict(sorted(counts.items())),
        "unique_class_descriptions": sorted(name for name, count in counts.items() if count == 1),
        "dominant_classes": [name for name, count in sorted(counts.items()) if count == max(counts.values())],
    }


def _select(rows: list[dict[str, Any]], count: int, seed: int, *, unique_class: bool = False) -> list[dict[str, Any]]:
    pool = [r for r in rows if not unique_class or _stratum(r)["unique_class_descriptions"]]
    require(len(pool) >= count, f"availability only {len(pool)} for requested {count}")
    ordered = sorted(pool, key=lambda r: (int(r["image_id"]), str(r["file_name"])))
    rng = random.Random(seed)
    chosen = rng.sample(ordered, count)
    return sorted(chosen, key=lambda r: (int(r["image_id"]), str(r["file_name"])))


def _case(row: Mapping[str, Any], source: Path, index: int, plan: Any) -> dict[str, Any]:
    image = (source.parent / str(row["images"][0])).resolve(strict=True)
    split = str(row["metadata"]["split"])
    original = Path("/data/CoordExp/public_data/coco/raw/images") / f"{split}2017" / Path(str(row["file_name"])).name
    original = original.resolve(strict=True)
    with Image.open(original) as opened:
        original_width, original_height = opened.size
    more_detail = original_width >= int(row["width"]) and original_height >= int(row["height"]) and (original_width > int(row["width"]) or original_height > int(row["height"]))
    row_id = f"coco2017_{row['metadata']['split']}_{int(row['image_id']):012d}"
    image_plan = {
        "backend_prompt_token_count": len(plan.prompt.expected_executed_prompt_token_ids),
        "image_content_sha256": plan.image.image_content_sha256,
        "logical_transform_id": plan.image.logical_transform_id,
        "merged_visual_tokens": int(plan.image.merged_visual_tokens),
        "observed_image_grid_thw": list(plan.image.expected_image_grid_thw),
        "expected_image_grid_thw": list(plan.image.expected_image_grid_thw),
        "patch_size": int(plan.image.patch_size),
        "merge_size": int(plan.image.merge_size),
        "temporal_patch_size": int(plan.image.temporal_patch_size),
        "do_resize": False,
        "planning": "CPU plan_qwen_image + prompt; no model forward",
    }
    return {
        "row_id": row_id,
        "row_index": index,
        "input_record": {k: v for k, v in row.items() if not k.startswith("_")},
        "image_path": str(image),
        "processed_image_path": str(image),
        "original_image_path": str(original),
        "image_width": int(row["width"]),
        "image_height": int(row["height"]),
        "width": int(row["width"]),
        "height": int(row["height"]),
        "original_width": int(original_width),
        "original_height": int(original_height),
        "processed_image_binding": binding(image),
        "original_image_binding": binding(original),
        "lane_b_detail": {"eligible": more_detail, "reason": "original_contains_more_native_pixels" if more_detail else "original_dimensions_do_not_exceed_baseline", "baseline_dimensions": [int(row["width"]), int(row["height"])], "original_dimensions": [int(original_width), int(original_height)]},
        "image_plan": image_plan,
        "annotation_provenance": {
            "source_path": str(source.resolve()),
            "source_row_number": int(row["_source_row_number"]),
            "source_row_sha256": str(row["_source_row_sha256"]),
            "source_split": split,
            "positive_target_semantics": "retained source positives; missing annotations are not negatives",
            "serialization": "source geo_sorted_xy view, lexicographic x1,y1",
        },
        "stratum": _stratum(row),
    }


def _source_identity(config: Mapping[str, Any], components: Any) -> dict[str, Any]:
    files = []
    for path in [BASE_MODEL / "config.json", BASE_MODEL / "preprocessor_config.json", BASE_MODEL / "tokenizer.json", BASE_MODEL / "tokenizer_config.json", CHECKPOINT / "adapter" / "adapter_config.json", CHECKPOINT / "adapter" / "adapter_model.safetensors", CHECKPOINT / "special_token_embeddings" / "special_token_embeddings.json", CHECKPOINT / "special_token_embeddings" / "special_token_embeddings.safetensors"]:
        files.append(binding(path))
    return {
        "base_model_path": str(BASE_MODEL.resolve(strict=True)),
        "checkpoint": str(CHECKPOINT.resolve(strict=True)),
        "loader": binding(LOADER),
        "panel": binding(PANEL),
        "bound_files": files,
        "config": dict(config),
        "processor_identity": components.to_artifact_dict()["processor"],
        "model_identity": components.to_artifact_dict()["model"],
    }


def prepare(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    selection_root = output_root.resolve() / "selection"
    for path in [TRAIN_SOURCE, VAL_SOURCE, PANEL, RECURRENCE_PANEL, FRESH_PANEL, CURRENT_TRAIN, CURRENT_DEV, BASE_MODEL / "config.json", CHECKPOINT / "adapter" / "adapter_model.safetensors"]:
        path.resolve(strict=True)
    panel = json.loads(PANEL.read_text())
    tied = dict(panel["configs"]["tied"])
    require(tied["model"]["base_model"] == str(BASE_MODEL), "base model config drift")
    require(tied["model"]["processor"]["do_resize"] is False, "processor.do_resize must be false")
    tied["data"] = dict(tied["data"])
    tied["data"]["input_jsonl"] = str(TRAIN_SOURCE.resolve())

    excluded = _jsonl_ids(CURRENT_TRAIN) | _jsonl_ids(CURRENT_DEV)
    panel_exclusions = {}
    for name, path in [("mature_panel", PANEL), ("recurrence_shared_panel", RECURRENCE_PANEL), ("prior_fresh_panel", FRESH_PANEL)]:
        ids = _panel_ids(path)
        panel_exclusions[name] = {"source": binding(path), "image_count": len(ids), "image_ids": sorted(ids)}
        excluded |= ids
    excluded |= {632, 885, 5586, 7511, 14038, 417044, 309264}

    source_rows = {"train": _rows(TRAIN_SOURCE), "val": _rows(VAL_SOURCE)}
    candidates: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    rejected: list[dict[str, Any]] = []
    for split, source in [("train", TRAIN_SOURCE), ("val", VAL_SOURCE)]:
        for row in source_rows[split]:
            image_id = int(row.get("image_id", -1))
            if image_id in excluded:
                continue
            image, reason = _validate_source(row, source)
            if reason:
                rejected.append({"split": split, "image_id": image_id, "reason": reason})
            else:
                candidates[split].append(row)

    train_rows = _select(candidates["train"], TRAIN_COUNT, SEED)
    cal_rows = _select(candidates["val"], CAL_COUNT, SEED + 1, unique_class=True)
    reserved = {int(r["image_id"]) for r in cal_rows}
    eval_rows = _select([r for r in candidates["val"] if int(r["image_id"]) not in reserved], EVAL_COUNT, SEED + 2)

    components = load_qwen_components_from_options(QwenLoadOptions(base_model=str(BASE_MODEL), dtype="fp32", attn_implementation="sdpa", load_model=False))
    infer = InferConfig.model_validate(tied)
    case_sets = {}
    for name, rows, source in [("train", train_rows, TRAIN_SOURCE), ("calibration", cal_rows, VAL_SOURCE), ("evaluation", eval_rows, VAL_SOURCE)]:
        cases = []
        for row in rows:
            raw = raw_example_from_jsonl_row({k: v for k, v in row.items() if not k.startswith("_")}, jsonl_path=source, row_number=int(row["_source_row_number"]), raw_line=json.dumps({k: v for k, v in row.items() if not k.startswith("_")}, ensure_ascii=False))
            planned = plan_examples([raw], config=infer, components=components, row_indices=[int(row["_source_row_number"]) - 1])[0]
            cases.append(_case(row, source, int(row["_source_row_number"]) - 1, planned))
        case_sets[name] = cases

    diagnostic_cases = []
    for case in case_sets["calibration"]:
        names = case["stratum"]["unique_class_descriptions"]
        require(names, f"calibration case lacks unique class: {case['row_id']}")
        description = names[0]
        diagnostic_cases.append({"row_id": case["row_id"], "image_id": case["input_record"]["image_id"], "referent_description": description, "referent_source": "naturally unique source class; coordinates intentionally omitted", "source_row_number": case["row_index"] + 1})

    source_identity = _source_identity(tied, components)
    config = dict(tied)
    config["data"] = dict(config["data"])
    config["data"]["input_jsonl"] = str(TRAIN_SOURCE.resolve())
    configs = {}
    for name, source in [("train", TRAIN_SOURCE), ("calibration", VAL_SOURCE), ("evaluation", VAL_SOURCE)]:
        cohort_config = dict(tied)
        cohort_config["data"] = dict(cohort_config["data"])
        cohort_config["data"]["input_jsonl"] = str(source.resolve())
        configs[name] = cohort_config
    all_ids = {name: [int(case["input_record"]["image_id"]) for case in cases] for name, cases in case_sets.items()}
    require(not (set(all_ids["train"]) & set(all_ids["calibration"])), "train/cal overlap")
    require(not (set(all_ids["train"]) & set(all_ids["evaluation"])), "train/eval overlap")
    require(not (set(all_ids["calibration"]) & set(all_ids["evaluation"])), "cal/eval overlap")
    manifest = {
        "schema": SCHEMA,
        "status": "candidate_ready_for_lead_review",
        "selection": {"seed": SEED, "counts": {"train": TRAIN_COUNT, "calibration": CAL_COUNT, "evaluation": EVAL_COUNT}, "method": "sorted source image IDs; Python Random.sample; calibration requires naturally unique class", "source_order": "canonical geo_sorted_xy rows; x1 then y1", "excluded_image_count": len(excluded), "excluded_ids_sha256": digest(sorted(excluded)), "development_exclusions": panel_exclusions, "current_source_exclusions": {"train": binding(CURRENT_TRAIN), "dev": binding(CURRENT_DEV)}, "rejected_unavailable_count": len(rejected)},
        "sources": {"train": binding(TRAIN_SOURCE), "calibration": binding(VAL_SOURCE), "evaluation": binding(VAL_SOURCE), "source_identity": source_identity},
        "config": config,
        "configs": configs,
        "processor": components.to_artifact_dict(),
        "cohorts": {name: {"count": len(ids), "image_ids": ids, "cases_path": str(selection_root / f"{name}.jsonl")} for name, ids in all_ids.items()},
        "diagnostic_cases": diagnostic_cases,
        "availability": {"eligible_train": len(candidates["train"]), "eligible_val": len(candidates["val"]), "eligible_val_unique_class": sum(bool(_stratum(r)["unique_class_descriptions"]) for r in candidates["val"]), "requested": {"train": TRAIN_COUNT, "calibration": CAL_COUNT, "evaluation": EVAL_COUNT}, "unavailable_cases": rejected},
        "checks": {"model_calls": 0, "processor_mode": "CPU plan only; load_model=false", "all_files_resolve": True, "all_dimensions_bound": True, "all_cases_have_image_plan": True, "all_diagnostics_omit_coordinates": True, "disjoint_original_image_identity": True},
    }
    for name, cases in case_sets.items():
        publish_jsonl(selection_root / f"{name}.jsonl", cases)
    publish_jsonl(selection_root / "diagnostic_cases.jsonl", diagnostic_cases)
    publish(selection_root / "rejected.json", {"schema": SCHEMA + ".rejected", "rows": rejected})
    publish(selection_root / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    manifest = prepare(args.output_root)
    print(json.dumps({"status": manifest["status"], "manifest": str(args.output_root.resolve() / "selection" / "manifest.json"), "counts": manifest["cohorts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
