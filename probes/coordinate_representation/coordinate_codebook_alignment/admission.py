"""CPU-only admission for the coordinate-codebook alignment fit pilot.

The source rows and labels are frozen before execution.  This module performs
no model forward; it uses the maintained renderer and the local tokenizer only
to certify canonical full-response lengths against the native 3084-token cap.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from PIL import Image
from transformers import AutoTokenizer

# Keep the maintained CLI usable both as ``python -m`` and by its explicit
# path, which is how preparation receipts invoke bounded admissions.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.artifacts.utf8_json import canonical
from src.config.models import TemplateConfig
from src.data.examples import raw_example_from_jsonl_row
from src.templates.renderer import render_example


ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-22-coordinate-codebook-alignment"
)
SELECTION = ROOT / "selection"
COCO_ROOT = Path("/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000_xy_sorted")
COCO_VAL = COCO_ROOT / "val.coord.jsonl"
HUMAN13 = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-08-05-static-dynamic-owner-interface-crossover/inputs/"
    "human-refined-13.geo_sorted_xy.coord.jsonl"
)
REFINED_RUNTIME = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-18-untied-highconfidence18-natural/refined18.runtime.jsonl"
)
REFINED_SNAPSHOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-17-history-rereading-mechanism/human-evaluation/"
    "annotation-snapshot-v1/working.norm.jsonl"
)
PANEL = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-18-untied-highconfidence18-natural/panel.json"
)
BASE_MODEL = Path(
    "/data/Qwen3-VL/model_cache/models/Qwen/"
    "Qwen3-VL-2B-Instruct-coordexp-natural-adjacent"
)
CHECKPOINT = Path(
    "/data/CoordExp/outputs/infra_base/train/"
    "qwen3-vl-2b-geo-sorted-xy-untied-axis001-ebs24-4epoch/checkpoints/step-2444"
)
FIT_HUMAN_IDS = {1584, 2299, 2685, 4134, 5001, 6040, 7511, 10707, 13348, 13923, 14038, 14439, 16228}
FIT_REFINED_IDS = {7116, 309264, 351017, 417044, 477415}
FIT_FIXED_IDS = FIT_HUMAN_IDS | FIT_REFINED_IDS
FIT_FILL_SEED = 20260922
MONITOR_SEED = 20260923
TARGET_CAP = 3084
SCHEMA = "coordinate_codebook_alignment.admission.v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def binding(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    require(path.is_file(), f"expected file: {path}")
    return {"path": str(path), "sha256": file_hash(path), "size_bytes": path.stat().st_size}


def publish(path: Path, value: Any) -> None:
    data = canonical(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == data, f"immutable collision: {path}")
        return
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def publish_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    data = b"".join(canonical(dict(row)) for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == data, f"immutable collision: {path}")
        return
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            require(isinstance(row, dict), f"non-object row at {path}:{line_number}")
            row["_source_line_number"] = line_number
            row["_source_line_sha256"] = hashlib.sha256(line.encode()).hexdigest()
            rows.append(row)
    return rows


def row_id(row: Mapping[str, Any]) -> str:
    split = str(row["metadata"]["split"])
    return f"coco2017_{split}_{int(row['image_id']):012d}"


def object_coords(obj: Mapping[str, Any]) -> tuple[int, int, int, int]:
    values = obj.get("bbox_2d")
    require(isinstance(values, list) and len(values) == 4, "bbox must have four coordinates")
    out: list[int] = []
    for value in values:
        require(isinstance(value, str) and value.startswith("<|coord_") and value.endswith("|>"), f"non-canonical coordinate token {value!r}")
        number = int(value[len("<|coord_") : -2])
        require(0 <= number <= 999, f"coordinate outside norm1000: {number}")
        out.append(number)
    require(out[0] <= out[2] and out[1] <= out[3], "invalid box geometry")
    return tuple(out)  # type: ignore[return-value]


def stratum(row: Mapping[str, Any]) -> dict[str, Any]:
    objects = row["objects"]
    counts = Counter(str(obj["desc"]) for obj in objects)
    count = len(objects)
    dominant = max(counts.values()) if counts else 0
    same_class = count >= 5 and dominant / count >= 0.75
    return {
        "object_count": count,
        "density": "ordinary" if count <= 4 else "dense" if count >= 10 else "middle",
        "same_class": same_class,
        "class_counts": dict(sorted(counts.items())),
    }


def stable_order(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: hashlib.sha256(f"{seed}:{row_id(row)}".encode()).hexdigest())


def resolve_processed_image(row: Mapping[str, Any], source: Path) -> Path:
    values = row.get("images")
    require(isinstance(values, list) and values and isinstance(values[0], str), f"missing image path for {row_id(row)}")
    image = Path(values[0])
    if not image.is_absolute():
        image = source.parent / image
    return image.resolve(strict=True)


def raw_for_render(row: Mapping[str, Any], source: Path) -> Any:
    clean = {k: v for k, v in row.items() if not k.startswith("_")}
    return raw_example_from_jsonl_row(
        clean,
        jsonl_path=source,
        row_number=int(row["_source_line_number"]),
        raw_line=json.dumps(clean, ensure_ascii=False),
    )


def validate_canonical_row(row: Mapping[str, Any]) -> None:
    require(row.get("metadata", {}).get("source") == "coco2017", f"unexpected source for {row_id(row)}")
    require(str(row.get("metadata", {}).get("split")) in {"train", "val"}, f"unexpected split for {row_id(row)}")
    require(type(row.get("width")) is int and type(row.get("height")) is int, f"missing dimensions for {row_id(row)}")
    objects = row.get("objects")
    require(isinstance(objects, list) and objects, f"empty positives for {row_id(row)}")
    previous: tuple[int, int] | None = None
    for obj in objects:
        require(isinstance(obj.get("desc"), str) and obj["desc"], f"missing description for {row_id(row)}")
        require(isinstance(obj.get("coco_ann_id"), int), f"missing owner ID for {row_id(row)}")
        coords = object_coords(obj)
        key = (coords[0], coords[1])
        require(previous is None or key >= previous, f"row is not geo_sorted_xy for {row_id(row)}")
        previous = key


def target_info(row: Mapping[str, Any], source: Path, tokenizer: Any, template: TemplateConfig) -> dict[str, Any]:
    rendered = render_example(raw_for_render(row, source), template)
    target_text = rendered.supervised_response_text
    ids = tokenizer.encode(target_text, add_special_tokens=False)
    require(ids and target_text.endswith("<|im_end|>\n"), f"bad target suffix for {row_id(row)}")
    require(len(ids) <= TARGET_CAP, f"target exceeds cap for {row_id(row)}: {len(ids)}")
    require(rendered.object_ordering == "geo_sorted_xy", f"renderer ordering changed for {row_id(row)}")
    return {
        "token_count": len(ids),
        "token_ids_sha256": hashlib.sha256(bytes().join(int(i).to_bytes(4, "little") for i in ids)).hexdigest(),
        "target_text_sha256": hashlib.sha256(target_text.encode()).hexdigest(),
        "object_count": len(row["objects"]),
        "rendered_object_ids": [str(x.object_id) for x in rendered.realized_object_order],
    }


def lineage_for(image_id: int, source: Path, panel: Mapping[str, Any]) -> dict[str, Any]:
    if image_id in FIT_HUMAN_IDS:
        return {
            "source_group": "human13",
            "membership": "human13",
            "reference_source": str(HUMAN13.resolve()),
            "annotation_boundary": "user-refined positive labels; not exhaustive scene negatives",
        }
    if image_id in FIT_REFINED_IDS:
        return {
            "source_group": "refined5",
            "membership": "refined5",
            "reference_source": str(REFINED_SNAPSHOT.resolve()),
            "reference_derivative": str(REFINED_RUNTIME.resolve()),
            "annotation_boundary": "user-refined positive labels; not exhaustive scene negatives",
        }
    return {"source_group": "coco12k-geo-sorted-xy-v1", "membership": "canonical_coco_positive_source"}


def output_row(row: Mapping[str, Any], source: Path, group: str, tokenizer: Any, template: TemplateConfig, panel: Mapping[str, Any]) -> dict[str, Any]:
    validate_canonical_row(row)
    image = resolve_processed_image(row, source)
    with Image.open(image) as opened:
        processed_dimensions = [int(opened.width), int(opened.height)]
    require(processed_dimensions == [int(row["width"]), int(row["height"])], f"processed dimension mismatch for {row_id(row)}")
    original_path = (Path("/data/CoordExp/public_data/coco/raw/images") / f"{row['metadata']['split']}2017" / Path(str(row["file_name"])).name).resolve(strict=True)
    with Image.open(original_path) as opened:
        original_dimensions = [int(opened.width), int(opened.height)]
    info = target_info(row, source, tokenizer, template)
    result = {k: copy.deepcopy(v) for k, v in row.items() if not k.startswith("_")}
    result["images"] = [str(image)]
    result["_admission"] = {
        "row_id": row_id(row),
        "cohort": group,
        "source_binding": binding(source),
        "source_line_number": int(row["_source_line_number"]),
        "source_line_sha256": str(row["_source_line_sha256"]),
        "image_binding": binding(image),
        "processed_dimensions": processed_dimensions,
        "original_image_path": str(original_path),
        "original_image_binding": binding(original_path),
        "original_dimensions": original_dimensions,
        "stratum": stratum(row),
        "source_lineage": lineage_for(int(row["image_id"]), source, panel),
        "owner_ids": [int(obj["coco_ann_id"]) for obj in row["objects"]],
        "target": info,
        "serialization": "geo_sorted_xy; desc_first; xyxy norm1000; full response with EOS",
    }
    return result


def source_map(path: Path) -> dict[int, dict[str, Any]]:
    return {int(row["image_id"]): row for row in read_rows(path)}


def compare_refined_snapshot(runtime: Mapping[int, dict[str, Any]], snapshot: Mapping[int, dict[str, Any]]) -> None:
    for image_id in FIT_REFINED_IDS:
        live = runtime[image_id]
        reference = snapshot[image_id]
        require(len(live["objects"]) == len(reference["objects"]), f"refined5 object count conflict {image_id}")
        live_by_owner = {int(obj["coco_ann_id"]): obj for obj in live["objects"]}
        reference_by_owner = {int(obj["coco_ann_id"]): obj for obj in reference["objects"]}
        require(set(live_by_owner) == set(reference_by_owner), f"refined5 owner set conflict {image_id}")
        for owner_id, live_obj in live_by_owner.items():
            ref_obj = reference_by_owner[owner_id]
            require(live_obj["desc"] == ref_obj["desc"], f"refined5 description conflict {image_id}/{owner_id}")
            require(tuple(object_coords(live_obj)) == tuple(int(x) for x in ref_obj["bbox_2d"]), f"refined5 geometry conflict {image_id}/{owner_id}")


def build(selection_dir: Path = SELECTION) -> dict[str, Any]:
    panel = json.loads(PANEL.read_text())
    require(panel["configs"]["untied"]["model"]["processor"]["do_resize"] is False, "source processor resize changed")
    human = source_map(HUMAN13)
    refined_runtime = source_map(REFINED_RUNTIME)
    refined_snapshot = source_map(REFINED_SNAPSHOT)
    require(set(human) == FIT_HUMAN_IDS, "Human13 membership changed")
    require(FIT_REFINED_IDS <= set(refined_runtime) and FIT_REFINED_IDS <= set(refined_snapshot), "Refined5 availability changed")
    compare_refined_snapshot(refined_runtime, refined_snapshot)

    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL), use_fast=True)
    template = TemplateConfig(**{key: panel["configs"]["untied"]["template"][key] for key in ("object_field_order", "object_ordering", "assistant_format", "prompt")})
    require(template.object_field_order == "desc_first" and template.object_ordering == "geo_sorted_xy", "canonical template changed")

    coco_rows = read_rows(COCO_VAL)
    coco_by_id = {int(row["image_id"]): row for row in coco_rows}
    require(len(coco_by_id) == len(coco_rows), "COCO val duplicate image identity")
    available = [row for row in coco_rows if int(row["image_id"]) not in FIT_FIXED_IDS]
    ordinary = [row for row in available if stratum(row)["density"] == "ordinary"]
    dense_same = [row for row in available if stratum(row)["density"] == "dense" and stratum(row)["same_class"]]
    require(len(ordinary) >= 39 and len(dense_same) >= 39, f"COCO source availability ordinary={len(ordinary)} dense_same={len(dense_same)}")
    fit_ordinary = stable_order(ordinary, FIT_FILL_SEED)[:7]
    fit_dense = stable_order(dense_same, FIT_FILL_SEED + 1)[:7]
    fit_ids = FIT_FIXED_IDS | {int(row["image_id"]) for row in fit_ordinary + fit_dense}
    monitor_pool = [row for row in available if int(row["image_id"]) not in fit_ids]
    monitor_ordinary = [row for row in monitor_pool if stratum(row)["density"] == "ordinary"]
    monitor_dense = [row for row in monitor_pool if stratum(row)["density"] == "dense"]
    require(len(monitor_ordinary) >= 32 and len(monitor_dense) >= 32, f"monitor availability ordinary={len(monitor_ordinary)} dense={len(monitor_dense)}")
    monitor_ordinary_pool_count = len(monitor_ordinary)
    monitor_dense_pool_count = len(monitor_dense)
    monitor_ordinary = stable_order(monitor_ordinary, MONITOR_SEED)[:32]
    monitor_dense = stable_order(monitor_dense, MONITOR_SEED + 1)[:32]
    require(len({int(row["image_id"]) for row in monitor_ordinary + monitor_dense}) == 64, "monitor duplicate identity")
    require(not ({int(row["image_id"]) for row in monitor_ordinary + monitor_dense} & fit_ids), "fit/monitor identity overlap")

    fit_rows: list[dict[str, Any]] = []
    for image_id in sorted(FIT_HUMAN_IDS):
        fit_rows.append(output_row(human[image_id], HUMAN13, "fit_human13", tokenizer, template, panel))
    for image_id in sorted(FIT_REFINED_IDS):
        fit_rows.append(output_row(refined_runtime[image_id], REFINED_RUNTIME, "fit_refined5", tokenizer, template, panel))
    for row in sorted(fit_ordinary, key=lambda x: int(x["image_id"])):
        fit_rows.append(output_row(row, COCO_VAL, "fit_coco_ordinary", tokenizer, template, panel))
    for row in sorted(fit_dense, key=lambda x: int(x["image_id"])):
        fit_rows.append(output_row(row, COCO_VAL, "fit_coco_dense_sameclass", tokenizer, template, panel))
    monitor_rows: list[dict[str, Any]] = []
    for row in sorted(monitor_ordinary, key=lambda x: int(x["image_id"])):
        monitor_rows.append(output_row(row, COCO_VAL, "monitor_coco_ordinary", tokenizer, template, panel))
    for row in sorted(monitor_dense, key=lambda x: int(x["image_id"])):
        monitor_rows.append(output_row(row, COCO_VAL, "monitor_coco_dense", tokenizer, template, panel))
    require(len(fit_rows) == 32 and len(monitor_rows) == 64, "final admission count changed")
    fit_ids_out = [x["_admission"]["row_id"] for x in fit_rows]
    monitor_ids_out = [x["_admission"]["row_id"] for x in monitor_rows]
    require(len(set(fit_ids_out)) == 32 and len(set(monitor_ids_out)) == 64 and set(fit_ids_out).isdisjoint(monitor_ids_out), "output identity disjointness")
    require(all(x["_admission"]["target"]["token_count"] <= TARGET_CAP for x in fit_rows + monitor_rows), "target cap check")

    source_paths = [HUMAN13, REFINED_RUNTIME, REFINED_SNAPSHOT, COCO_VAL, PANEL,
                    BASE_MODEL / "config.json", BASE_MODEL / "preprocessor_config.json", BASE_MODEL / "tokenizer.json", BASE_MODEL / "tokenizer_config.json",
                    CHECKPOINT / "adapter" / "adapter_config.json", CHECKPOINT / "adapter" / "adapter_model.safetensors",
                    CHECKPOINT / "special_token_embeddings" / "special_token_embeddings.json", CHECKPOINT / "special_token_embeddings" / "special_token_embeddings.safetensors",
                    CHECKPOINT / "inference_payload_manifest.json", Path(__file__), Path("probes/model_profiles/mature_tied_untied.py"), Path("src/templates/renderer.py"), Path("src/data/examples.py")]
    source_bindings: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in source_paths:
        path = path.resolve(strict=True)
        if str(path) not in seen:
            source_bindings.append(binding(path))
            seen.add(str(path))

    strata = Counter(x["_admission"]["stratum"]["density"] for x in fit_rows + monitor_rows)
    return {
        "schema": SCHEMA,
        "status": "candidate_ready_for_parent_review",
        "model_calls": 0,
        "source_asset": "mature-untied-axis001-xy-step2444",
        "checkpoint_root": str(CHECKPOINT.resolve(strict=True)),
        "source_config": copy.deepcopy(panel["configs"]["untied"]),
        "availability": {"human13_exact_images": len(FIT_HUMAN_IDS), "refined5_exact_images": len(FIT_REFINED_IDS), "coco_val_images": len(coco_rows), "coco_after_fixed_ids": len(available), "coco_fit_ordinary_pool": len(ordinary), "coco_fit_dense_sameclass_pool": len(dense_same), "coco_monitor_ordinary_pool_after_fit": monitor_ordinary_pool_count, "coco_monitor_dense_pool_after_fit": monitor_dense_pool_count},
        "fit": {"count": 32, "fixed_human_refined_count": 18, "coco_fill_count": 14, "cases_path": str((selection_dir / "fit.coord.jsonl").resolve()), "image_ids": [int(x["image_id"]) for x in sorted(fit_rows, key=lambda x: int(x["image_id"]))], "groups": {key: sum(x["_admission"]["cohort"] == key for x in fit_rows) for key in sorted({x["_admission"]["cohort"] for x in fit_rows})}},
        "monitor": {"count": 64, "cases_path": str((selection_dir / "monitor.coord.jsonl").resolve()), "image_ids": [int(x["image_id"]) for x in sorted(monitor_rows, key=lambda x: int(x["image_id"]))], "groups": {key: sum(x["_admission"]["cohort"] == key for x in monitor_rows) for key in sorted({x["_admission"]["cohort"] for x in monitor_rows})}},
        "selection": {"fit_fill_seed": FIT_FILL_SEED, "monitor_seed": MONITOR_SEED, "source_pool": str(COCO_VAL.resolve()), "fixed_ids": sorted(FIT_FIXED_IDS), "exclusion_rule": "fit/monitor image identity disjoint; Human13 and Refined5 retained exactly", "strata_definition": {"ordinary": "1-4 positive objects", "dense": "10 or more positive objects", "same_class": "at least 5 objects and dominant description share >= 0.75"}},
        "target_contract": {"format": "full response; description first; object_box_closed; geo_sorted_xy", "coordinates": "xyxy norm1000 coordinate tokens 0..999", "eos": "<|im_end|> newline", "max_new_tokens": TARGET_CAP, "tokenizer": binding(BASE_MODEL / "tokenizer.json"), "processor_resize": False, "loss": "full-vocabulary response CE; segment_balanced denominator in parent trainer"},
        "checks": {"human13_count": 13, "refined5_count": 5, "fit_count": len(fit_rows), "monitor_count": len(monitor_rows), "fit_monitor_disjoint": True, "target_cap_pass": True, "all_geo_sorted_xy": True, "owner_ids_preserved": True, "model_calls": 0, "strata_counts": dict(sorted(strata.items()))},
        "source_bindings": source_bindings,
    }, fit_rows, monitor_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT)
    parser.add_argument("--selection-dir", type=Path, default=None)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    selection_dir = (args.selection_dir or (args.output_root / "selection")).resolve()
    manifest, fit_rows, monitor_rows = build(selection_dir)
    if not args.check_only:
        publish_jsonl(selection_dir / "fit.coord.jsonl", fit_rows)
        publish_jsonl(selection_dir / "monitor.coord.jsonl", monitor_rows)
        publish(selection_dir / "admission.json", manifest)
    print(json.dumps({"status": manifest["status"], "fit": len(fit_rows), "monitor": len(monitor_rows), "model_calls": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
