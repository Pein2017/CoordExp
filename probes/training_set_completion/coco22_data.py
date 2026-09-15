"""Freeze a COCO train2017 cohort by annotation metadata, then render GT review sources.

This preparation reads the canonical coordinate JSONL and existing exposure
records. It does not use model outcomes or promote additional owners.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


ROOT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-15-coco22-cumulative-expansion/cohort-v1")
ROOT_V2 = ROOT.with_name("cohort-v2")
ROOT_V3 = ROOT.with_name("cohort-v3")
CANONICAL = Path("/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000_xy_sorted/train.coord.jsonl")
IMAGE_ROOT = Path("/data/CoordExp/public_data/coco/rescale_32_1024_bbox/images/train2017")
OLD_ANNOTATIONS = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum/annotations-with-unlabeled-v4/annotations.jsonl")
EXPOSURE_FILES = (
    Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-05-sft256-dev128-baseline/inputs-v1/train.jsonl"),
    Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-05-sft256-dev128-baseline/inputs-v1/dev.jsonl"),
    Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-05-sft256-dev128-baseline/inputs-v2/train.jsonl"),
    Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-05-sft256-dev128-baseline/inputs-v2/dev.jsonl"),
    Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-05-sft256-dev128-baseline/inputs-v3/train.jsonl"),
    Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-05-sft256-dev128-baseline/inputs-v3/dev.jsonl"),
    Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-06-ce-controls-rloo-successor/data2048-v1/train.jsonl"),
    Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-08-row-boundary-continuous-feedback/data-v1/train.jsonl"),
    Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-08-row-boundary-continuous-feedback/data-v1/dev.jsonl"),
    OLD_ANNOTATIONS,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def binding(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve(strict=True)), "sha256": sha256(path), "size_bytes": path.stat().st_size}


def rows(path: Path):
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            yield line_no, json.loads(line)


def bbox(obj: dict[str, Any]) -> tuple[int, int, int, int]:
    raw = obj["bbox_2d"]
    values = tuple(int(v.removeprefix("<|coord_").removesuffix("|>")) for v in raw)
    if len(values) != 4 or not (0 <= values[0] < values[2] <= 999 and 0 <= values[1] < values[3] <= 999):
        raise ValueError(f"bad native bbox: {raw}")
    return values


def features(row: dict[str, Any]) -> dict[str, Any]:
    objects = row["objects"]
    classes = collections.Counter(str(o["desc"]) for o in objects)
    areas = [(b[2] - b[0]) * (b[3] - b[1]) / 999**2 for b in map(bbox, objects)]
    return {
        "gt_count": len(objects),
        "max_same_class": max(classes.values(), default=0),
        "same_class_pairs": sum(n * (n - 1) // 2 for n in classes.values()),
        "small_share": round(sum(a < .005 for a in areas) / max(1, len(areas)), 6),
        "small_count": sum(a < .005 for a in areas),
        "category_count": len(classes),
        "classes": dict(sorted(classes.items())),
        "pixel_count": int(row["width"]) * int(row["height"]),
    }


def score(a: dict[str, Any], b: dict[str, Any]) -> float:
    # Keep count/density dominant; match repetition and small-object burden.
    return (
        abs(a["gt_count"] - b["gt_count"]) / 3
        + abs(math.log1p(a["same_class_pairs"]) - math.log1p(b["same_class_pairs"]))
        + abs(a["small_share"] - b["small_share"]) * 3
        + abs(a["category_count"] - b["category_count"]) / 3
        + abs(math.log(a["pixel_count"] / b["pixel_count"]))
    )


def image_path(image_id: int) -> Path:
    return IMAGE_ROOT / f"{image_id:012d}.jpg"


def dhash(path: Path) -> int:
    with Image.open(path) as original:
        image = original.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        pixels = list(image.getdata())
    value = 0
    for y in range(8):
        for x in range(8):
            value = value * 2 + (pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return value


def prepare(out: Path = ROOT) -> dict[str, Any]:
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"cohort output occupied: {out}")
    old = {r["image_id"]: r for _, r in rows(OLD_ANNOTATIONS)}
    if len(old) != 11:
        raise ValueError("expected frozen 11 old images")
    exposure: dict[int, list[str]] = collections.defaultdict(list)
    exposure_sources = []
    for path in EXPOSURE_FILES:
        if not path.is_file():
            raise FileNotFoundError(path)
        ids = {int(r["image_id"]) for _, r in rows(path)}
        for image_id in ids:
            exposure[image_id].append(str(path))
        exposure_sources.append({**binding(path), "rows": len(ids)})
    old_features = {i: features(r) for i, r in old.items()}
    candidates = []
    source_rows: dict[int, dict[str, Any]] = {}
    source_lines: dict[int, int] = {}
    rejection = collections.Counter()
    for line_no, row in rows(CANONICAL):
        image_id = row["image_id"]
        if image_id in exposure:
            rejection["known_specialized_input_or_old"] += 1
            continue
        if row.get("metadata") != {"source": "coco2017", "split": "train"}:
            rejection["source_metadata"] += 1
            continue
        try:
            f = features(row)
        except (ValueError, KeyError):
            rejection["invalid_annotation_geometry"] += 1
            continue
        if not 5 <= f["gt_count"] <= 32:
            rejection["gt_count_outside_old_range"] += 1
            continue
        if not image_path(image_id).is_file():
            rejection["missing_image"] += 1
            continue
        # Severe same-class overlapping duplicates make teacher identity ambiguous.
        boxes = [bbox(o) for o in row["objects"]]
        if len(set((str(o["desc"]), box) for o, box in zip(row["objects"], boxes))) != len(boxes):
            rejection["exact_duplicate_annotation"] += 1
            continue
        candidates.append((image_id, f))
        source_rows[image_id] = row
        source_lines[image_id] = line_no
    anchors = sorted(old)
    # Take a bounded metadata shortlist per anchor, then apply an image near-duplicate guard.
    shortlists: dict[int, list[tuple[float, int]]] = {}
    for anchor in anchors:
        shortlists[anchor] = sorted(
            ((score(old_features[anchor], f), i) for i, f in candidates),
            key=lambda x: (round(x[0], 8), hashlib.sha256(f"coco22-v1:{x[1]}".encode()).hexdigest()),
        )[:80]
    exposed_hashes = [(i, dhash(image_path(i))) for i in sorted(exposure) if image_path(i).is_file()]
    chosen: list[dict[str, Any]] = []
    chosen_hashes: list[tuple[int, int]] = []
    nearest_rejects: list[dict[str, Any]] = []
    for anchor in anchors:
        for value, image_id in shortlists[anchor]:
            if image_id in {x["image_id"] for x in chosen}:
                continue
            h = dhash(image_path(image_id))
            neighbors = sorted(((h ^ other).bit_count(), other_id) for other_id, other in exposed_hashes + chosen_hashes)
            nearest_distance, nearest_id = neighbors[0]
            if nearest_distance <= 4:
                nearest_rejects.append({"image_id": image_id, "nearest_id": nearest_id, "hamming64": nearest_distance})
                continue
            f = features(source_rows[image_id])
            chosen.append({"image_id": image_id, "matched_old_image_id": anchor, "metadata_score": round(value, 8), "features": f, "canonical_line": source_lines[image_id], "image_dhash64": f"{h:016x}", "nearest_exposure_or_selected": {"image_id": nearest_id, "dhash_hamming64": nearest_distance}})
            chosen_hashes.append((image_id, h))
            break
        else:
            raise RuntimeError(f"no near-duplicate-safe metadata match for old {anchor}")
    out.mkdir(parents=True)
    source_out = out / "new-originals.jsonl"
    with source_out.open("w", encoding="utf-8") as f:
        for item in chosen:
            row = source_rows[item["image_id"]]
            # Rebase only the media reference so this original-format copy is consumable.
            row = {**row, "images": [os.path.relpath(image_path(item["image_id"]), source_out.parent)]}
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    receipt = {
        "schema": "coco22.metadata_frozen_cohort.v1", "status": "selected_pending_visual_admission",
        "selection_policy": "old GT annotation count, same-class repetition, small-area share, category count and rescaled pixel count; deterministic SHA256 tie-break; no model outcome",
        "near_duplicate_policy": "64-bit horizontal dHash on 9x8 luminance; reject Hamming<=4 against specialized inputs and selected images; diagnostic proxy, visual review still required",
        "source": binding(CANONICAL), "old_annotations": binding(OLD_ANNOTATIONS), "specialized_exposure_sources": exposure_sources,
        "exposure_unique_image_count": len(exposure), "source_candidate_count": len(candidates), "rejections": dict(rejection), "near_duplicate_rejections": nearest_rejects,
        "old_image_ids": anchors, "new_images": chosen, "new_originals": binding(source_out),
        "old_gt_features": {str(i): old_features[i] for i in anchors},
        "exclusion_claim_scope": "IDs in the listed specialized input/teacher records only; no base-model or upstream Source-unseen claim",
    }
    (out / "selection.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    return receipt


def prepare_v2(out: Path = ROOT_V2, predecessor: Path = ROOT) -> dict[str, Any]:
    """Replace only pre-model visual-quality failures under the original scorer."""
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"cohort output occupied: {out}")
    prior_path = predecessor / "selection.json"
    prior = json.loads(prior_path.read_text())
    if prior["status"] != "selected_pending_visual_admission" or len(prior["new_images"]) != 11:
        raise ValueError("predecessor selection identity")
    visual_exclusions = {
        109707: "primary GT includes surfboards depicted in wall photos and sofa/bed instance ambiguity",
        260604: "primary GT covers LEGO toy figures and vehicles rather than real scene instances",
        346877: "GT bottle class is visually ambiguous with the drinking vessel",
        4360: "two GT motorcycle boxes appear to overlap one heavily obscured physical vehicle",
    }
    replacements = {210457, 219546}
    retained = [item for item in prior["new_images"] if item["matched_old_image_id"] not in replacements]
    if len(retained) != 9 or {x["image_id"] for x in prior["new_images"] if x["matched_old_image_id"] in replacements} != {109707, 260604}:
        raise ValueError("visual exclusion does not match predecessor")
    old = {r["image_id"]: r for _, r in rows(OLD_ANNOTATIONS)}
    exposure = {int(r["image_id"]) for path in EXPOSURE_FILES for _, r in rows(path)}
    old_features = {i: features(old[i]) for i in replacements}
    source_rows = {}
    source_lines = {}
    shortlists = {i: [] for i in replacements}
    skip = exposure | {x["image_id"] for x in retained} | set(visual_exclusions)
    for line_no, row in rows(CANONICAL):
        image_id = row["image_id"]
        if image_id in skip:
            continue
        try:
            f = features(row)
        except (ValueError, KeyError):
            continue
        if not 5 <= f["gt_count"] <= 32 or not image_path(image_id).is_file():
            continue
        source_rows[image_id] = row
        source_lines[image_id] = line_no
        for anchor in replacements:
            shortlists[anchor].append((score(old_features[anchor], f), image_id))
    exposed_hashes = [(i, dhash(image_path(i))) for i in sorted(exposure | {x["image_id"] for x in retained}) if image_path(i).is_file()]
    chosen = list(retained)
    for anchor in sorted(replacements):
        ranked = sorted(shortlists[anchor], key=lambda x: (round(x[0], 8), hashlib.sha256(f"coco22-v1:{x[1]}".encode()).hexdigest()))
        for value, image_id in ranked:
            if image_id in {x["image_id"] for x in chosen}:
                continue
            h = dhash(image_path(image_id))
            distance, nearest_id = min(((h ^ other).bit_count(), other_id) for other_id, other in exposed_hashes)
            if distance <= 4:
                continue
            chosen.append({"image_id": image_id, "matched_old_image_id": anchor, "metadata_score": round(value, 8), "features": features(source_rows[image_id]), "canonical_line": source_lines[image_id], "image_dhash64": f"{h:016x}", "nearest_exposure_or_selected": {"image_id": nearest_id, "dhash_hamming64": distance}})
            exposed_hashes.append((image_id, h))
            break
        else:
            raise RuntimeError(f"no visual-quality eligible match for {anchor}")
    chosen.sort(key=lambda x: prior["old_image_ids"].index(x["matched_old_image_id"]))
    if {x["image_id"] for x in chosen} != ({x["image_id"] for x in retained} | {200288, 335722}):
        raise ValueError("replacement changed from visually screened metadata shortlist")
    # Fetch exact canonical records for retained images and visual exclusions too.
    needed = {x["image_id"] for x in chosen} | set(visual_exclusions)
    canonical_records = {}
    for line_no, row in rows(CANONICAL):
        if row["image_id"] in needed:
            canonical_records[row["image_id"]] = (line_no, row)
    if set(canonical_records) != needed:
        raise ValueError("canonical source record missing")
    out.mkdir(parents=True)
    source_out = out / "new-originals.jsonl"
    rebased = []
    with source_out.open("w", encoding="utf-8") as f:
        for item in chosen:
            image_id = item["image_id"]
            line_no, original = canonical_records[image_id]
            reference = os.path.relpath(image_path(image_id), source_out.parent)
            emitted = {**original, "images": [reference]}
            if {k: v for k, v in emitted.items() if k != "images"} != {k: v for k, v in original.items() if k != "images"}:
                raise ValueError("non-media source field changed")
            if (source_out.parent / reference).resolve(strict=True) != image_path(image_id).resolve(strict=True):
                raise ValueError("rebased image path is wrong")
            if line_no != item["canonical_line"]:
                raise ValueError("canonical source line changed")
            f.write(json.dumps(emitted, ensure_ascii=False, separators=(",", ":")) + "\n")
            rebased.append({"image_id": image_id, "canonical_line": line_no, "original_images": original["images"], "emitted_images": emitted["images"], "resolved_image": binding(image_path(image_id))})
    quality = []
    for image_id, reason in visual_exclusions.items():
        _, row = canonical_records[image_id]
        image = Image.open(image_path(image_id)).convert("RGB")
        original_path = out / "quality-exclusions" / f"image-{image_id:012d}-original.png"
        overlay_path = out / "quality-exclusions" / f"image-{image_id:012d}-gt-overlay.png"
        original_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(original_path)
        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        for n, obj in enumerate(row["objects"]):
            b = bbox(obj)
            pixel = [round(b[0] * image.width / 999), round(b[1] * image.height / 999), round(b[2] * image.width / 999), round(b[3] * image.height / 999)]
            draw.rectangle(pixel, outline=(255, 0, 0), width=3)
            draw.text((pixel[0], max(0, pixel[1] - 12)), f"{n}:{obj['desc']}", fill=(255, 255, 0), stroke_width=2, stroke_fill=(0, 0, 0))
        overlay.save(overlay_path)
        quality.append({"image_id": image_id, "reason": reason, "visual": {"original": binding(original_path), "gt_overlay": binding(overlay_path)}})
    receipt = {**prior, "schema": "coco22.metadata_frozen_cohort.v2", "status": "selected_pending_visual_admission", "predecessor": binding(prior_path), "new_images": chosen, "new_originals": binding(source_out), "visual_quality_exclusions": quality, "media_rebase": {"rule": "only images path rebased; all other original-format record fields equal canonical source", "checked": True, "records": rebased}, "selection_transition": "two replacements before model outputs under user-accepted real-scene rule; nine predecessor selections retained"}
    (out / "selection.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    return receipt


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / (area_a + area_b - intersection)


def prepare_v3(out: Path = ROOT_V3, predecessor: Path = ROOT_V2) -> dict[str, Any]:
    """Replace a cross-class duplicate GT while preserving ten v2 selections."""
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"cohort output occupied: {out}")
    prior_path = predecessor / "selection.json"
    prior = json.loads(prior_path.read_text())
    retained = [x for x in prior["new_images"] if x["image_id"] != 29802]
    if len(retained) != 10 or next(x for x in prior["new_images"] if x["image_id"] == 29802)["matched_old_image_id"] != 323322:
        raise ValueError("v3 predecessor image conflict identity")
    exposure = {int(r["image_id"]) for path in EXPOSURE_FILES for _, r in rows(path)}
    old = {r["image_id"]: r for _, r in rows(OLD_ANNOTATIONS)}
    anchor = 323322
    old_f = features(old[anchor])
    candidates = []
    conflicts = []
    for line_no, row in rows(CANONICAL):
        image_id = row["image_id"]
        if image_id in exposure or image_id in {x["image_id"] for x in retained} or image_id == 29802:
            continue
        try:
            f = features(row)
        except (ValueError, KeyError):
            continue
        if not 5 <= f["gt_count"] <= 32 or not image_path(image_id).is_file():
            continue
        pairs = [(j, k, _iou(bbox(a), bbox(b))) for j, a in enumerate(row["objects"]) for k, b in enumerate(row["objects"]) if j < k and a["desc"] != b["desc"] and _iou(bbox(a), bbox(b)) >= .9]
        if pairs:
            conflicts.append({"image_id": image_id, "pairs": pairs})
            continue
        candidates.append((score(old_f, f), image_id, line_no, f))
    ranked = sorted(candidates, key=lambda x: (round(x[0], 8), hashlib.sha256(f"coco22-v1:{x[1]}".encode()).hexdigest()))
    if not ranked or ranked[0][1] != 124185:
        raise ValueError("first metadata-ranked visually screened replacement changed")
    value, image_id, line_no, f = ranked[0]
    hashes = [(i, dhash(image_path(i))) for i in sorted(exposure | {x["image_id"] for x in retained}) if image_path(i).is_file()]
    h = dhash(image_path(image_id))
    distance, nearest_id = min(((h ^ other).bit_count(), other_id) for other_id, other in hashes)
    if distance <= 4:
        raise ValueError("candidate is a near duplicate of specialized input")
    new = {"image_id": image_id, "matched_old_image_id": anchor, "metadata_score": round(value, 8), "features": f, "canonical_line": line_no, "image_dhash64": f"{h:016x}", "nearest_exposure_or_selected": {"image_id": nearest_id, "dhash_hamming64": distance}}
    chosen = sorted([*retained, new], key=lambda x: prior["old_image_ids"].index(x["matched_old_image_id"]))
    needed = {x["image_id"] for x in chosen} | {29802}
    originals = {r["image_id"]: (n, r) for n, r in rows(CANONICAL) if r["image_id"] in needed}
    if set(originals) != needed:
        raise ValueError("canonical source records missing")
    out.mkdir(parents=True)
    source_out = out / "new-originals.jsonl"
    rebased = []
    with source_out.open("w", encoding="utf-8") as f:
        for item in chosen:
            i = item["image_id"]
            n, original = originals[i]
            reference = os.path.relpath(image_path(i), source_out.parent)
            emitted = {**original, "images": [reference]}
            if n != item["canonical_line"] or {k: v for k, v in emitted.items() if k != "images"} != {k: v for k, v in original.items() if k != "images"}:
                raise ValueError("source record identity changed")
            if (source_out.parent / reference).resolve(strict=True) != image_path(i).resolve(strict=True):
                raise ValueError("rebased media path differs")
            f.write(json.dumps(emitted, ensure_ascii=False, separators=(",", ":")) + "\n")
            rebased.append({"image_id": i, "canonical_line": n, "original_images": original["images"], "emitted_images": emitted["images"], "resolved_image": binding(image_path(i))})
    image = Image.open(image_path(29802)).convert("RGB")
    quality_dir = out / "quality-exclusions"
    quality_dir.mkdir()
    original_path = quality_dir / "image-000000029802-original.png"
    overlay_path = quality_dir / "image-000000029802-gt-overlay.png"
    image.save(original_path)
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    for n, obj in enumerate(originals[29802][1]["objects"]):
        b = bbox(obj)
        pixel = [round(b[0] * image.width / 999), round(b[1] * image.height / 999), round(b[2] * image.width / 999), round(b[3] * image.height / 999)]
        draw.rectangle(pixel, outline=(255, 0, 0), width=3)
        draw.text((pixel[0], max(0, pixel[1] - 12)), f"{n}:{obj['desc']}", fill=(255, 255, 0), stroke_width=2, stroke_fill=(0, 0, 0))
    overlay.save(overlay_path)
    quality = [*prior["visual_quality_exclusions"], {"image_id": 29802, "reason": "GT chair and couch label the same right foreground recliner (cross-class IoU about .955); incompatible with one-owner, correct-class teacher", "visual": {"original": binding(original_path), "gt_overlay": binding(overlay_path)}}]
    receipt = {**prior, "schema": "coco22.metadata_frozen_cohort.v3", "status": "selected_pending_visual_admission", "predecessor": binding(prior_path), "new_images": chosen, "new_originals": binding(source_out), "visual_quality_exclusions": quality, "media_rebase": {"rule": "only images path rebased; all other original-format record fields equal canonical source", "checked": True, "records": rebased}, "selection_transition": "one cross-class duplicate-GT replacement before model outputs; ten v2 selections retained", "cross_class_iou_screen": {"threshold": .9, "candidate_conflict_count": len(conflicts), "note": "screen only; chosen image visually checked, legitimate overlaps require visual adjudication"}}
    (out / "selection.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    return receipt


def render(out: Path = ROOT, *, review_version: int = 1) -> dict[str, Any]:
    from src.data.geometry import coord_bins_to_pixel_xyxy

    if review_version not in (1, 2):
        raise ValueError("unsupported review version")
    receipt = json.loads((out / "selection.json").read_text())
    source_path = Path(receipt["new_originals"]["path"])
    if sha256(source_path) != receipt["new_originals"]["sha256"]:
        raise ValueError("selected source changed")
    selected = {r["image_id"]: r for _, r in rows(source_path)}
    index = []
    for image_id, row in sorted(selected.items()):
        image_dir = out / ("review" if review_version == 1 else "review-v2") / f"image-{image_id:012d}"
        if image_dir.exists() and any(image_dir.iterdir()):
            raise FileExistsError(f"review image output occupied: {image_dir}")
        image_dir.mkdir(parents=True, exist_ok=True)
        path = image_path(image_id)
        with Image.open(path) as f:
            image = f.convert("RGB")
        if image.size != (row["width"], row["height"]):
            raise ValueError(f"image dimensions differ: {image_id}")
        original_path = image_dir / "original.png"
        image.save(original_path)
        canvas = image.copy()
        draw = ImageDraw.Draw(canvas)
        for n, obj in enumerate(row["objects"]):
            x1, y1, x2, y2 = bbox(obj)
            if review_version == 1:
                pixel = (round(x1 * image.width / 999), round(y1 * image.height / 999), round(x2 * image.width / 999), round(y2 * image.height / 999))
            else:
                pixel = coord_bins_to_pixel_xyxy((x1, y1, x2, y2), image_width=image.width, image_height=image.height, field=f"gt:{image_id}:{n}")
            draw.rectangle(pixel, outline=(255, 0, 0), width=3)
            draw.text((pixel[0], max(0, pixel[1] - 12)), f"{n}:{obj['desc']}", fill=(255, 255, 0), stroke_width=2, stroke_fill=(0, 0, 0))
            pad_x = max(8, round(.1 * (pixel[2] - pixel[0])))
            pad_y = max(8, round(.1 * (pixel[3] - pixel[1])))
            crop = image.crop((max(0, pixel[0] - pad_x), max(0, pixel[1] - pad_y), min(image.width, pixel[2] + pad_x), min(image.height, pixel[3] + pad_y)))
            one = image.copy()
            ImageDraw.Draw(one).rectangle(pixel, outline=(255, 0, 0), width=3)
            crop_path = image_dir / f"gt-{n:03d}-crop.png"
            overlay_path = image_dir / f"gt-{n:03d}-overlay.png"
            crop.save(crop_path)
            one.save(overlay_path)
            index.append({"image_id": image_id, "owner_id": f"gt:{image_id}:{n}", "coco_ann_id": obj.get("coco_ann_id"), "category": obj["desc"], "bbox_2d": obj["bbox_2d"], "bbox_pixel_xyxy": list(pixel), "coordinate_conversion": "src.data.geometry.coord_bins_to_pixel_xyxy" if review_version == 2 else "superseded_divisor_999", "visual": {"original": binding(original_path), "bbox_overlay": binding(overlay_path), "crop": binding(crop_path)}, "decision": None, "status": "pending_visual_review"})
        canvas.save(image_dir / "all-gt-overlay.png")
    index_path = out / ("gt-review-index.jsonl" if review_version == 1 else "gt-review-index-v2.jsonl")
    if index_path.exists():
        raise FileExistsError(index_path)
    with index_path.open("w", encoding="utf-8") as f:
        for item in index:
            f.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    result = {"schema": f"coco22.gt_review_sources.v{review_version}", "status": "rendered_pending_visual_review", "gt_owner_count": len(index), "review_index": binding(index_path), "image_count": len(selected), "coordinate_conversion": "src.data.geometry.coord_bins_to_pixel_xyxy" if review_version == 2 else "superseded_divisor_999"}
    render_receipt = out / ("render-receipt.json" if review_version == 1 else "render-receipt-v2.json")
    if render_receipt.exists():
        raise FileExistsError(render_receipt)
    render_receipt.write_text(json.dumps(result, indent=2) + "\n")
    return result


OLD_227_BANK = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-15-coco227-ce-normalization/data-v1/bank.json")
DISCOVERY_CASES = ROOT.parent / "discovery-cases-v3-reviewed.json"
GT_ADMISSION = ROOT.parent / "gt-review-admission-v1/lead-admission.json"
ANNOTATIONS_V1 = ROOT.parent / "annotations-v1/annotations.jsonl"
TEACHER_OUTPUT = ROOT.parent / "data-v1"
TEACHER_SCHEMA = "training_set_completion.coco22_teacher.v1"
OLD_227_FILE_SHA256 = "d65126ec827a9cd070b6a06c4557f44e2bb6c5a37b3fb3fc1b1fb6e94d172aac"


def _teacher_sources(
    *, old_bank_path: Path, case_packet_path: Path, cohort_selection_path: Path,
    annotation_jsonl_path: Path, gt_admission_path: Path,
) -> dict[str, Any]:
    source = {"old227_bank": binding(old_bank_path), "discovery_case_packet": binding(case_packet_path),
              "cohort_selection": binding(cohort_selection_path), "annotations": binding(annotation_jsonl_path),
              "gt_review_admission": binding(gt_admission_path)}
    if source["old227_bank"]["sha256"] != OLD_227_FILE_SHA256:
        raise ValueError("old 227 bank bytes changed")
    annotation_manifest = annotation_jsonl_path.parent / "manifest.json"
    manifest = json.loads(annotation_manifest.read_text())
    if manifest.get("annotations") != source["annotations"] or manifest.get("status") != "candidate_ready_for_lead_annotation_admission":
        raise ValueError("annotation version manifest does not bind actual image JSONL")
    source["annotation_manifest"] = binding(annotation_manifest)
    receipt = manifest["sources"].get("owner_root_admission")
    source["new_unlabeled_admission"] = receipt
    if receipt is not None and binding(Path(receipt["path"])) != receipt:
        raise ValueError("new unlabeled lead receipt changed")
    source["producer"] = binding(Path(__file__))
    return source


def _selected_new_owners(
    *, annotation_rows: list[dict[str, Any]], new_image_ids: list[int], admission: dict[str, Any],
    sources: dict[str, Any],
) -> dict[int, list[dict[str, Any]]]:
    from src.data.geometry import parse_source_bbox_tokens
    from src.eval.detection_categories import COCO_80_CLASS_NAMES

    if admission.get("status") != "lead-accepted" or not isinstance(admission.get("gt_owners"), list):
        raise ValueError("full lead GT admission is required")
    rows_by_image = {r["image_id"]: r for r in annotation_rows}
    raw_gt = {(image_id, obj["coco_ann_id"]): (index, obj)
              for image_id in new_image_ids for index, obj in enumerate(rows_by_image[image_id]["objects"])}
    if len(raw_gt) != sum(len(rows_by_image[image_id]["objects"]) for image_id in new_image_ids):
        raise ValueError("duplicate raw new GT identity")
    indexed: dict[tuple[int, int], tuple[int, dict[str, Any]]] = {}
    for entry_index, entry in enumerate(admission["gt_owners"]):
        if type(entry.get("image_id")) is not int or type(entry.get("coco_ann_id")) is not int:
            raise ValueError("GT review image/annotation identity")
        key = (entry["image_id"], entry["coco_ann_id"])
        if key not in raw_gt or key in indexed:
            raise ValueError(f"GT review must bind exactly one raw owner: {key}")
        decision = entry.get("decision")
        if decision not in {"verified_real_gt", "reject_false_gt", "reject_duplicate_gt", "hold"}:
            raise ValueError(f"unrecognized GT review decision: {key}")
        _, obj = raw_gt[key]
        if entry.get("bbox_2d") != obj["bbox_2d"]:
            raise ValueError(f"GT review geometry differs from original record: {key}")
        if decision == "verified_real_gt" and entry.get("category") != obj["desc"]:
            raise ValueError(f"GT review category differs from original record: {key}")
        visual = entry.get("evidence")
        if not isinstance(visual, dict):
            raise ValueError(f"GT review requires original, overlay, and crop: {key}")
        for name in ("original", "overlay", "crop"):
            value = visual.get(name)
            path = value.get("path") if isinstance(value, dict) else value
            if not isinstance(path, str):
                raise ValueError(f"GT review missing visual {name}: {key}")
            observed = binding(Path(path))
            if isinstance(value, dict) and value != observed:
                raise ValueError(f"GT review visual changed: {key}:{name}")
        indexed[key] = (entry_index, entry)
    if set(indexed) != set(raw_gt):
        raise ValueError("GT review omits a raw object; pending decisions must explicitly hold")
    lead_receipts: dict[str, dict[str, Any]] = {}
    def lead_entry(source: dict[str, Any], image_id: int, owner_id: str) -> dict[str, Any]:
        if not isinstance(source, dict) or binding(Path(source["path"])) != source:
            raise ValueError(f"unlabeled owner admission changed: {image_id}:{owner_id}")
        receipt = lead_receipts.setdefault(source["path"], json.loads(Path(source["path"]).read_text()))
        if receipt.get("status") != "lead-accepted" or not isinstance(receipt.get("owners"), list):
            raise ValueError("unlabeled admission receipt is not lead-accepted")
        matched = [e for e in receipt["owners"] if e.get("image_id") == image_id and e.get("stable_owner_id") == owner_id]
        if len(matched) != 1:
            raise ValueError(f"unlabeled owner lead decision absent/duplicated: {image_id}:{owner_id}")
        return matched[0]
    selected: dict[int, list[dict[str, Any]]] = {}
    for image_id in new_image_ids:
        row = rows_by_image[image_id]
        accepted = []
        for index, obj in enumerate(row["objects"]):
            entry_index, entry = indexed[(image_id, obj["coco_ann_id"])]
            if entry["decision"] != "verified_real_gt":
                continue
            description = obj["desc"]
            if description not in COCO_80_CLASS_NAMES:
                raise ValueError(f"trusted GT description outside COCO80: {image_id}:{index}")
            bins = list(parse_source_bbox_tokens(obj["bbox_2d"], field=f"gt:{image_id}:{index}"))
            accepted.append({"owner_id": str(obj["coco_ann_id"]), "description": description, "bins": bins,
                             "source_kind": "visually_verified_original_gt", "source_index": index,
                             "source_decision": {"coco_ann_id": obj["coco_ann_id"], "decision": entry["decision"],
                                                 "gt_review_entry_index": entry_index},
                             "source_trace": {"object": obj, "gt_review_admission": sources["gt_review_admission"]}})
        for index, obj in enumerate(row["unlabeled"]):
            if obj.get("class_status") != "verified":
                continue
            owner_id = obj.get("stable_owner_id")
            history = obj.get("provenance", {})
            past = history.get("admission", [])
            if not isinstance(owner_id, str) or not isinstance(past, list) or len(past) != 1:
                raise ValueError(f"verified new unlabeled owner lacks lead decision: {image_id}:{owner_id}")
            original_decision = lead_entry(past[0], image_id, owner_id)
            resolution = history.get("category_resolution_admission", [])
            decision = lead_entry(resolution[0], image_id, owner_id) if resolution else original_decision
            if resolution and (len(resolution) != 4 or decision.get("decision") != "resolve_prior_unknown_category"):
                raise ValueError(f"unlabeled category resolution provenance: {image_id}:{owner_id}")
            if obj.get("physical_status") != "valid_unlabeled" or obj.get("geometry_status") != "reasonable" or obj.get("scene_scope") != "real_original_scene":
                raise ValueError(f"unlabeled physical/geometry/scene status: {image_id}:{owner_id}")
            if decision.get("class_status") != "verified" or decision.get("category_name") != obj.get("desc") or decision.get("bbox_2d_bins_1000") != obj.get("bbox_2d_bins_1000"):
                raise ValueError(f"unlabeled decision differs from annotation: {image_id}:{owner_id}")
            if not resolution and original_decision.get("decision") != "distinct_real_owner":
                raise ValueError(f"unlabeled physical owner admission: {image_id}:{owner_id}")
            description = obj["desc"]
            if description not in COCO_80_CLASS_NAMES:
                raise ValueError(f"unlabeled description outside COCO80: {image_id}:{owner_id}")
            bins = obj["bbox_2d_bins_1000"]
            bbox({"bbox_2d": obj["bbox_2d"]})
            if bins != [int(value.removeprefix("<|coord_").removesuffix("|>")) for value in obj["bbox_2d"]]:
                raise ValueError(f"unlabeled coordinate spellings differ: {image_id}:{owner_id}")
            accepted.append({"owner_id": owner_id, "description": description, "bins": list(bins),
                             "source_kind": "lead_admitted_real_unlabeled", "source_index": index,
                             "source_decision": {"stable_owner_id": owner_id, "decision": decision["decision"]},
                             "source_trace": {"unlabeled": obj, "owner_root_admission": past[0],
                                              "category_resolution_admission": resolution[0] if resolution else None}})
        if not accepted:
            raise ValueError(f"new image has zero trusted targets: {image_id}")
        accepted.sort(key=lambda r: (r["bins"][0], r["bins"][1], r["source_kind"], r["owner_id"]))
        if len({x["owner_id"] for x in accepted}) != len(accepted):
            raise ValueError(f"duplicate selected teacher owner: {image_id}")
        selected[image_id] = accepted
    return selected


def _teacher_new_route(record: dict[str, Any], owners: list[dict[str, Any]], *, tokenizer: Any, coordinate_ids: list[int]) -> dict[str, Any]:
    from probes.training_set_completion.complete_bank import EOS, _field_ids
    from probes.training_set_completion.training import validate_route

    image_id = record["image_id"]
    case = record["case"]
    plan = case["image_plan"]
    continuation: list[int] = []
    weights: list[int] = []
    boxes: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    for order, owner in enumerate(owners):
        description, bins = owner["description"], owner["bins"]
        ids = tokenizer.encode(description, add_special_tokens=False)
        if not ids or tokenizer.decode(ids, skip_special_tokens=False) != description:
            raise ValueError(f"literal COCO description tokenization: {owner['owner_id']}")
        start = len(continuation)
        continuation.extend(_field_ids(tokenizer, description, bins, coordinate_ids))
        weights.extend([1] * (len(ids) + 8))
        positions = list(range(start + 1 + len(ids) + 2, start + 1 + len(ids) + 6))
        boxes.append({"x1_position": positions[0], "y1_position": positions[1],
                      "x2_position": positions[2], "y2_position": positions[3], "expected_bins": bins})
        trace.append({"owner_id": owner["owner_id"], "order": order, "source_kind": owner["source_kind"],
                      "source_order": owner["source_index"], "source_decision": owner["source_decision"],
                      "edited_fields": {"description_source": owner["source_kind"],
                                        "selected_description": description, "description_ce_positive": True,
                                        "description_token_positions": list(range(start + 1, start + 1 + len(ids))),
                                        "catalog_reference_coord_bins_1000": bins, "geometry_replaced": False},
                      "source_trace": owner["source_trace"]})
    continuation.append(EOS)
    weights.append(1)
    if len(continuation) > 3084:
        raise ValueError(f"teacher exceeds frozen 3084-token decode cap: {image_id}")
    route = {"route_id": f"coco22:image-{image_id:012d}", "image_id": image_id,
             "example_id": record["example_id"], "case": case,
             "image_identity": {"image_path": case["image_path"],
                                "image_content_sha256": plan["image_content_sha256"],
                                "executed_media_sha256": plan["executed_media_sha256"],
                                "observed_image_grid_thw": plan["observed_image_grid_thw"]},
             "prompt_token_ids": record["prompt_token_ids"], "continuation_token_ids": continuation,
             "ce_weights": weights, "trusted_boxes": boxes, "trusted_complete_support_endpoint": True,
             "provenance": {"synthetic_teacher": True, "selected_owner_ids": [r["owner_id"] for r in owners],
                            "trace": trace, "fixed_order": "geo_sorted_xy by x1 then y1; stable owner tie"}}
    validate_route(route, eos_token_id=EOS, coordinate_token_ids=coordinate_ids)
    return route


def validate_22_bank(bank: dict[str, Any]) -> dict[str, int]:
    """Recheck frozen old routes and every new trusted owner against physical admissions."""
    from probes.training_set_completion.complete_bank import EOS
    from probes.training_set_completion.training import coordinate_token_table, validate_route

    if bank.get("schema") != TEACHER_SCHEMA or bank.get("status") != "candidate_ready":
        raise ValueError("22-bank schema/lifecycle")
    content = {key: value for key, value in bank.items() if key != "content_sha256"}
    if hashlib.sha256((json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()).hexdigest() != bank.get("content_sha256"):
        raise ValueError("22-bank content identity")
    sources = bank["sources"]
    expected = _teacher_sources(old_bank_path=Path(sources["old227_bank"]["path"]),
                                case_packet_path=Path(sources["discovery_case_packet"]["path"]),
                                cohort_selection_path=Path(sources["cohort_selection"]["path"]),
                                annotation_jsonl_path=Path(sources["annotations"]["path"]),
                                gt_admission_path=Path(sources["gt_review_admission"]["path"]))
    if sources != expected:
        raise ValueError("22-bank source bytes changed")
    old = json.loads(Path(sources["old227_bank"]["path"]).read_text())
    packet = json.loads(Path(sources["discovery_case_packet"]["path"]).read_text())
    selection = json.loads(Path(sources["cohort_selection"]["path"]).read_text())
    new_image_ids = [entry["image_id"] for entry in selection["new_images"]]
    annotations = [entry for _, entry in rows(Path(sources["annotations"]["path"]))]
    accepted = _selected_new_owners(annotation_rows=annotations, new_image_ids=new_image_ids,
                                    admission=json.loads(Path(sources["gt_review_admission"]["path"]).read_text()),
                                    sources=sources)
    routes = bank.get("routes")
    if (not isinstance(routes, list) or len(routes) != 22 or routes[:11] != old["routes"] or
            [r["image_id"] for r in routes[11:]] != new_image_ids or
            [r["image_id"] for r in packet["records"]] != new_image_ids):
        raise ValueError("old11 routes or new11 case/image order changed")
    if bank.get("image_count") != 22 or old.get("fixed_owner_count") != 227:
        raise ValueError("22-bank image/old owner denominator")
    model_config = json.loads(Path(old["sources"]["first_fit_preparation"]["path"]).read_text())
    coordinate_ids = coordinate_token_table(Path(model_config["model_config"]["model"]["base_model"]))["ids"]
    old_ids = [card["owner_id"] for r in routes[:11] for card in r["provenance"]["trace"]]
    new_ids = []
    for route, record in zip(routes[11:], packet["records"]):
        if route["case"] != record["case"] or route["prompt_token_ids"] != record["prompt_token_ids"] or route["example_id"] != record["example_id"]:
            raise ValueError(f"new route case/prompt changed: {route['image_id']}")
        validate_route(route, eos_token_id=EOS, coordinate_token_ids=coordinate_ids)
        targets = accepted[route["image_id"]]
        if (len(route["trusted_boxes"]) != len(targets) or
                route["provenance"]["selected_owner_ids"] != [entry["owner_id"] for entry in targets]):
            raise ValueError(f"new trusted target selection changed: {route['image_id']}")
        for card, box, target in zip(route["provenance"]["trace"], route["trusted_boxes"], targets):
            if (card["owner_id"] != target["owner_id"] or
                    card["edited_fields"]["selected_description"] != target["description"] or
                    card["edited_fields"]["catalog_reference_coord_bins_1000"] != target["bins"] or
                    box["expected_bins"] != target["bins"] or card["source_decision"] != target["source_decision"]):
                raise ValueError(f"new teacher owner differs from visual admission: {route['image_id']}:{target['owner_id']}")
            new_ids.append(target["owner_id"])
    if (bank.get("old227_owner_ids") != old_ids or bank.get("new_cohort_owner_ids") != new_ids or
            len(old_ids) != 227 or len(set(old_ids + new_ids)) != len(old_ids) + len(new_ids)):
        raise ValueError("22-bank owner partition")
    if bank.get("fixed_owner_count") != len(old_ids) + len(new_ids) or any(len(r["continuation_token_ids"]) > 3084 for r in routes):
        raise ValueError("22-bank owner count/decode cap")
    return {"images": 22, "old_owners": len(old_ids), "new_owners": len(new_ids)}


def build_22_bank(
    *, gt_admission_path: Path = GT_ADMISSION, annotation_jsonl_path: Path = ANNOTATIONS_V1,
    discovery_case_packet_path: Path = DISCOVERY_CASES, old_bank_path: Path = OLD_227_BANK,
    cohort_selection_path: Path = ROOT_V3 / "selection.json", output: Path = TEACHER_OUTPUT,
) -> dict[str, Any]:
    """Build only from full lead GT decisions and the current versioned 22-row ledger."""
    from probes.training_set_completion.complete_bank import EOS
    from probes.training_set_completion.route_bank import _load_tokenizer
    from probes.training_set_completion.training import coordinate_token_table, validate_route
    from probes.source_rweak_row_cross.run import native_record

    if output.exists() or output.is_symlink():
        raise FileExistsError(f"teacher output occupied: {output}")
    sources = _teacher_sources(old_bank_path=old_bank_path, case_packet_path=discovery_case_packet_path,
                               cohort_selection_path=cohort_selection_path,
                               annotation_jsonl_path=annotation_jsonl_path, gt_admission_path=gt_admission_path)
    old_bank = json.loads(old_bank_path.read_text())
    if old_bank.get("fixed_owner_count") != 227 or old_bank.get("image_count") != 11:
        raise ValueError("old bank 227/11 invariant")
    old_routes = old_bank["routes"]
    if sum(len(r["trusted_boxes"]) for r in old_routes) != 227:
        raise ValueError("old route owner count")
    selection = json.loads(cohort_selection_path.read_text())
    new_image_ids = [r["image_id"] for r in selection["new_images"]]
    if selection.get("schema") != "coco22.metadata_frozen_cohort.v3" or len(new_image_ids) != 11 or len(set(new_image_ids)) != 11:
        raise ValueError("frozen v3 cohort identity")
    packet = json.loads(discovery_case_packet_path.read_text())
    if packet.get("selected_image_ids") != new_image_ids or packet.get("original_jsonl") != selection["new_originals"]:
        raise ValueError("discovery packet cohort/source differs")
    discovery_records = packet.get("records")
    if not isinstance(discovery_records, list) or [r.get("image_id") for r in discovery_records] != new_image_ids:
        raise ValueError("discovery case order/denominator")
    originals = {r["image_id"]: r for _, r in rows(Path(selection["new_originals"]["path"]))}
    if len(originals) != 11:
        raise ValueError("new source originals population")
    annotations = [r for _, r in rows(annotation_jsonl_path)]
    annotations_by_image = {r["image_id"]: r for r in annotations}
    if len(annotations) != 22 or len(annotations_by_image) != 22 or set(annotations_by_image) != {r["image_id"] for r in old_routes} | set(new_image_ids):
        raise ValueError("annotation image population")
    prior_old = {r["image_id"]: r for _, r in rows(OLD_ANNOTATIONS)}
    if any(annotations_by_image[i] != prior_old[i] for i in prior_old):
        raise ValueError("old accepted annotation records changed")
    for record in discovery_records:
        image_id, case = record["image_id"], record["case"]
        new = annotations_by_image[image_id]
        if {k: v for k, v in new.items() if k != "unlabeled"} != originals[image_id]:
            raise ValueError(f"new raw GT/source record differs: {image_id}")
        if case["input_record"] != originals[image_id] or case["row_id"] != record["example_id"]:
            raise ValueError(f"new case differs from frozen original/prompt identity: {image_id}")
        media = binding(Path(case["image_path"]))
        if case["image_plan"]["image_content_sha256"] != media["sha256"] or case["image_plan"]["observed_image_grid_thw"] != case["image_plan"]["expected_image_grid_thw"]:
            raise ValueError(f"new case media/grid differs: {image_id}")
        if (case["image_width"], case["image_height"]) != (new["width"], new["height"]):
            raise ValueError(f"new case image dimensions differ: {image_id}")
    admission = json.loads(gt_admission_path.read_text())
    selected = _selected_new_owners(annotation_rows=annotations, new_image_ids=new_image_ids,
                                    admission=admission, sources=sources)
    model_root = Path(old_bank["sources"]["first_fit_preparation"]["path"])
    first_fit = json.loads(model_root.read_text())
    tokenizer_root = Path(first_fit["model_config"]["model"]["base_model"])
    tokenizer = _load_tokenizer(tokenizer_root)
    coordinate_ids = coordinate_token_table(tokenizer_root)["ids"]
    new_routes = [_teacher_new_route(record, selected[record["image_id"]], tokenizer=tokenizer,
                                     coordinate_ids=coordinate_ids) for record in discovery_records]
    old_acquisition = json.loads(Path(old_bank["sources"]["acquisition_manifest"]["path"]).read_text())
    goldens = {r["image_id"]: r["golden"] for r in old_acquisition["records"] + discovery_records}
    routes = [*old_routes, *new_routes]
    if len(routes) != 22 or len({r["image_id"] for r in routes}) != 22:
        raise ValueError("one image per teacher route")
    parsed_count = 0
    for route in routes:
        validate_route(route, eos_token_id=EOS, coordinate_token_ids=coordinate_ids)
        text = tokenizer.decode(route["continuation_token_ids"], skip_special_tokens=False)
        parsed = native_record(text, route["case"], goldens[route["image_id"]], "im_end")
        if parsed.get("parse_status") != "accepted" or parsed.get("dropped_prediction_count") != 0 or len(parsed.get("pred", [])) != len(route["trusted_boxes"]):
            raise ValueError(f"teacher literal parser roundtrip failed: {route['image_id']}")
        for prediction, box, card in zip(parsed["pred"], route["trusted_boxes"], route["provenance"]["trace"]):
            if prediction.get("coord_bins") != box["expected_bins"] or prediction.get("description") != card["edited_fields"]["selected_description"]:
                raise ValueError(f"teacher parser owner identity differs: {route['image_id']}")
        parsed_count += len(parsed["pred"])
    old227_ids = [card["owner_id"] for r in old_routes for card in r["provenance"]["trace"]]
    new_ids = [card["owner_id"] for r in new_routes for card in r["provenance"]["trace"]]
    if len(old227_ids) != 227 or len(set(old227_ids + new_ids)) != len(old227_ids) + len(new_ids):
        raise ValueError("teacher owner identity collision")
    bank = {"schema": TEACHER_SCHEMA, "status": "candidate_ready",
            "target_version": "coco22-old227-plus-visually-verified-new-scene-v1",
            "sources": sources, "old227_owner_ids": old227_ids, "new_cohort_owner_ids": new_ids,
            "fixed_owner_count": parsed_count, "image_count": 22,
            "parser_receipt": {"schema": "coco22.teacher_parser_roundtrip.v1", "images": 22,
                               "owner_rows": parsed_count, "old227_rows": 227, "new_cohort_rows": len(new_ids),
                               "parser": "src.inference.parsing.parse_compact_object_box_closed"},
            "routes": routes}
    bank["content_sha256"] = hashlib.sha256(json.dumps(bank, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode() + b"\n").hexdigest()
    validate_22_bank(bank)
    # Only a complete sidecar can create a candidate; this remains subject to lead admission.
    output.mkdir(parents=True)
    bank_path = output / "bank.json"
    with bank_path.open("xb") as f:
        data = (json.dumps(bank, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    if json.loads(bank_path.read_text()) != bank:
        raise ValueError("teacher bank publication readback differs")
    result = {"schema": "coco22.teacher_cpu_candidate.v1", "status": "pending_lead_teacher_admission",
              "teacher_bank": binding(bank_path), "old227_route_count": 11,
              "new_route_count": 11, "new_trusted_owner_count": len(new_ids),
              "new_gt_review_decisions": collections.Counter(r["decision"] for r in admission["gt_owners"]),
              "annotation_jsonl": sources["annotations"], "no_model_forward": True}
    result["new_gt_review_decisions"] = dict(result["new_gt_review_decisions"])
    (output / "cpu-candidate.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "prepare-v2", "prepare-v3", "render", "render-v2", "teacher"))
    parser.add_argument("--out", type=Path, default=ROOT)
    parser.add_argument("--gt-admission", type=Path, default=GT_ADMISSION)
    parser.add_argument("--annotations", type=Path, default=ANNOTATIONS_V1)
    parser.add_argument("--case-packet", type=Path, default=DISCOVERY_CASES)
    args = parser.parse_args()
    result = (build_22_bank(gt_admission_path=args.gt_admission, annotation_jsonl_path=args.annotations,
                            discovery_case_packet_path=args.case_packet,
                            output=TEACHER_OUTPUT if args.out == ROOT else args.out) if args.command == "teacher" else
              prepare(args.out) if args.command == "prepare" else prepare_v2(args.out) if args.command == "prepare-v2" else
              prepare_v3(args.out) if args.command == "prepare-v3" else
              render(args.out, review_version=2 if args.command == "render-v2" else 1))
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
