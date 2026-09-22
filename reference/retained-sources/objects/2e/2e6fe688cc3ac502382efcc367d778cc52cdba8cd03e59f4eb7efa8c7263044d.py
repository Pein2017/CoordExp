"""CPU preparation and reduction helpers for the recurrence census.

This file freezes the source population and selection before new model output.
It intentionally writes only under the Lane-A output root.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-distribution-census")
MATURE = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-untied-highconfidence18-natural")
FEEDBACK = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-numerical-recurrence-feedback")
DATA = Path("/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000_xy_sorted")
IMAGE_DATA = Path("/data/CoordExp/public_data/coco/rescale_32_1024_bbox")
PIPELINE = DATA / "pipeline_manifest.json"
TRAIN256 = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-05-sft256-dev128-baseline/inputs-v3/train.jsonl")
DEV128 = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-05-sft256-dev128-baseline/inputs-v3/dev.jsonl")
BASE = Path("/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent")
COORD = re.compile(r"<\|coord_(\d+)\|>")
ROLES = ("x1", "y1", "x2", "y2")


def binding(path: str | Path) -> dict[str, Any]:
    p = Path(path).resolve()
    if p.is_dir():
        entries = []
        total = 0
        for child in sorted(x for x in p.rglob("*") if x.is_file()):
            data = child.read_bytes()
            rel = str(child.relative_to(p))
            entries.append((rel, hashlib.sha256(data).hexdigest(), len(data)))
            total += len(data)
        payload = json.dumps(entries, separators=(",", ":")).encode()
        return {"path": str(p), "sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": total, "kind": "directory", "file_count": len(entries)}
    b = p.read_bytes()
    return {"path": str(p), "sha256": hashlib.sha256(b).hexdigest(), "size_bytes": len(b), "kind": "file"}


def path_binding(path: str | Path) -> dict[str, Any]:
    """Bind a large model directory without recursively reading model weights."""
    p = Path(path).resolve()
    return {"path": str(p), "kind": "directory", "identity": "existing-loader-path-binding", "exists": p.is_dir()}


def canonical_digest(items: list[Any]) -> str:
    return hashlib.sha256(json.dumps(items, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def key_of(row: dict[str, Any]) -> tuple[str, int, str]:
    return (str(row["metadata"]["split"]), int(row["image_id"]), str(row["file_name"]))


def image_path(row: dict[str, Any]) -> Path:
    p = IMAGE_DATA / str(row["file_name"])
    if not p.exists():
        raise FileNotFoundError(p)
    return p.resolve()


def _image_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _identity(row: dict[str, Any]) -> tuple[str, int]:
    """The exclusion identity; file_name is a cross-check, never the identity."""
    return (str(row["metadata"]["split"]), int(row["image_id"]))


def _record_signature(row: dict[str, Any], source_path: Path | None = None) -> dict[str, Any]:
    path = image_path(row)
    return {
        "identity": list(_identity(row)),
        "file_name": str(row.get("file_name", "")),
        "canonical_path": str(path),
        "basename": path.name,
        "sha256": _image_sha256(path),
        "source": None if source_path is None else str(source_path),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def to_runtime_record(row: dict[str, Any]) -> dict[str, Any]:
    d = copy.deepcopy(row)
    img = image_path(d)
    d["images"] = [os.path.relpath(img, OUT)]
    d["objects"] = sorted(
        [
            {
                **o,
                "bbox_2d": [f"<|coord_{int(v) if isinstance(v, int) else COORD.fullmatch(v).group(1)}|>" for v in o["bbox_2d"]],
            }
            for o in d["objects"]
        ],
        key=lambda o: (int(COORD.fullmatch(o["bbox_2d"][0]).group(1)), int(COORD.fullmatch(o["bbox_2d"][1]).group(1))),
    )
    return d


def load_exclusion_keys() -> tuple[
    set[tuple[str, int, str]],
    dict[str, list[tuple[str, int, str]]],
    dict[str, Any],
]:
    by_source: dict[str, list[tuple[str, int, str]]] = {}
    by_identity: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    records: list[dict[str, Any]] = []
    for label, path in (("train256", TRAIN256), ("dev128", DEV128)):
        rows = read_jsonl(path)
        keys = [key_of(x) for x in rows]
        by_source[label] = keys
        for row in rows:
            sig = _record_signature(row, path)
            by_identity[_identity(row)].append({"source": label, "key": list(key_of(row)), "signature": sig})
            records.append({"source": label, "key": list(key_of(row)), "signature": sig})
    panel = json.loads((MATURE / "panel.json").read_text())
    mature_cases = [c for g in panel["groups"] for c in g["cases"]]
    mature_keys = [key_of(c["input_record"]) for c in mature_cases]
    assert len(set(mature_keys)) == 145, len(set(mature_keys))
    by_source["fresh128_sentinel_highconfidence"] = mature_keys
    for case in mature_cases:
        row = case["input_record"]
        sig = _record_signature(row, MATURE / "panel.json")
        by_identity[_identity(row)].append({"source": "fresh128_sentinel_highconfidence", "key": list(key_of(row)), "signature": sig})
        records.append({"source": "fresh128_sentinel_highconfidence", "key": list(key_of(row)), "signature": sig})

    aliases = []
    conflicts = []
    for ident, items in sorted(by_identity.items()):
        signatures = {x["signature"]["sha256"] for x in items}
        if len(signatures) > 1:
            conflicts.append({"identity": list(ident), "records": items})
        elif len({x["signature"]["canonical_path"] for x in items}) > 1 or len({tuple(x["key"]) for x in items}) > 1:
            aliases.append({"identity": list(ident), "records": items})
    audit = {
        "schema": "recurrence_census.identity_audit.v1",
        "identity": "(metadata.split,image_id); file_name/path and image sha256 are cross-checks",
        "split_is_part_of_identity": True,
        "prior_record_count": len(records),
        "prior_identity_count": len(by_identity),
        "prior_aliases": aliases,
        "prior_conflicts": conflicts,
        "status": "conflict" if conflicts else "no_conflict",
        "records": records,
    }
    return set().union(*map(set, by_source.values())), by_source, audit


def freeze_cohort() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    exclusions, exclusion_parts, identity_audit = load_exclusion_keys()
    panel = json.loads((MATURE / "panel.json").read_text())
    if identity_audit["prior_conflicts"]:
        write_json(OUT / "identity-audit.json", identity_audit)
        raise RuntimeError("prior source identity conflict; new cohort is held")
    excluded_signatures = defaultdict(set)
    for item in identity_audit["records"]:
        excluded_signatures[tuple(item["signature"]["identity"])].add(item["signature"]["sha256"])
    rows: list[dict[str, Any]] = []
    source_lines: dict[tuple[str, int, str], tuple[Path, int, str]] = {}
    source_bindings: dict[Path, dict[str, Any]] = {}
    source_identity_paths: dict[tuple[str, int], dict[str, Any]] = {}
    processed_aliases = []
    processed_conflicts = []
    exclusion_alias_hits = []
    source_record_count = 0
    for split in ("train", "val"):
        p = DATA / f"{split}.coord.jsonl"
        source_bindings[p] = binding(p)
        for line_no, line in enumerate(p.read_text().splitlines(), 1):
            row = json.loads(line)
            k = key_of(row)
            source_record_count += 1
            source_lines[k] = (p, line_no, line)
            ident = _identity(row)
            candidate_sig = None
            if ident in excluded_signatures or ident in source_identity_paths:
                candidate_sig = _record_signature(row, p)
            if ident in source_identity_paths:
                prior = source_identity_paths[ident]
                if "sha256" not in prior:
                    prior = {**prior, "sha256": _image_sha256(Path(prior["canonical_path"]))}
                    source_identity_paths[ident] = prior
                if candidate_sig["sha256"] != prior["sha256"]:
                    processed_conflicts.append({"identity": list(ident), "records": [prior, candidate_sig]})
                elif candidate_sig["canonical_path"] != prior["canonical_path"] or candidate_sig["file_name"] != prior["file_name"]:
                    processed_aliases.append({"identity": list(ident), "records": [prior, candidate_sig]})
            else:
                if candidate_sig is None:
                    # The common path is cheap to audit and avoids hashing every processed image.
                    candidate_sig = {"identity": list(ident), "file_name": str(row.get("file_name", "")), "canonical_path": str(image_path(row)), "basename": image_path(row).name}
                source_identity_paths[ident] = candidate_sig
            excluded = k in exclusions
            if ident in excluded_signatures:
                if candidate_sig is None or candidate_sig.get("sha256") is None:
                    candidate_sig = _record_signature(row, p)
                if candidate_sig["sha256"] in excluded_signatures[ident]:
                    excluded = True
                    if k not in exclusions:
                        exclusion_alias_hits.append({"identity": list(ident), "selected_key": list(k), "matched_sha256": candidate_sig["sha256"]})
                else:
                    processed_conflicts.append({"identity": list(ident), "records": [candidate_sig, {"prior_sha256": sorted(excluded_signatures[ident])}]})
            if not excluded:
                rows.append(row)
    keys = [key_of(x) for x in rows]
    assert len(keys) == len(set(keys)), "processed manifest has identity duplicates"
    ordered = sorted(rows, key=key_of)
    selected = random.Random(19).sample(ordered, 128)
    selected.sort(key=key_of)
    assert len(selected) == 128 and len({key_of(x) for x in selected}) == 128

    selected_rows = []
    runtime = OUT / "new128.runtime.jsonl"
    with runtime.open("w") as f:
        for out_index, row in enumerate(selected):
            k = key_of(row)
            source_path, source_line_number, raw_line = source_lines[k]
            img = image_path(row)
            with Image.open(img) as im:
                dimensions = [int(im.width), int(im.height)]
            assert dimensions == [int(row["width"]), int(row["height"])], (k, dimensions, row["width"], row["height"])
            runtime_row = to_runtime_record(row)
            f.write(json.dumps(runtime_row, separators=(",", ":")) + "\n")
            selected_rows.append(
                {
                    "selection_index": out_index,
                    "key": list(k),
                    "source_line_number": source_line_number,
                    "source_line_sha256": hashlib.sha256(raw_line.encode()).hexdigest(),
                    "image": binding(img),
                    "image_dimensions": dimensions,
                    "annotation_count": len(row.get("objects", [])),
                }
            )

    eligible_keys = [list(key_of(x)) for x in ordered]
    manifest = {
        "schema": "recurrence_census.eligible_processed_manifest.v1",
        "status": "frozen_before_new_outputs",
        "selection_seed": 19,
        "selection_algorithm": "sort eligible records by (metadata.split,image_id,file_name); random.Random(19).sample(128); sort selected by the same key",
        "processed_root": str(DATA),
        "pipeline_manifest": binding(PIPELINE),
        "source_files": [source_bindings[DATA / "train.coord.jsonl"], source_bindings[DATA / "val.coord.jsonl"]],
        "source_record_counts": {"train": 117266, "val": 4952, "total": 122218},
        "source_records_seen": source_record_count,
        "eligible_unique_records": len(eligible_keys),
        "eligible_key_sha256": canonical_digest(eligible_keys),
        "exclusion_counts": {k: len(v) for k, v in exclusion_parts.items()},
        "exclusion_key_sha256": {k: canonical_digest([list(x) for x in sorted(v)]) for k, v in exclusion_parts.items()},
        "exclusion_union_count": len(exclusions),
        "selected_count": len(selected_rows),
        "selected_key_sha256": canonical_digest([x["key"] for x in selected_rows]),
        "selected_rows": selected_rows,
        "runtime_records": binding(runtime),
        "prior_source_manifests": [binding(TRAIN256), binding(DEV128), binding(MATURE / "panel.json")],
        "production_val200": {"status": "not_found_in_current_research-output-roots"},
        "identity_audit": {
            "identity": identity_audit["identity"],
            "split_is_part_of_identity": True,
            "prior_alias_count": len(identity_audit["prior_aliases"]),
            "prior_conflict_count": len(identity_audit["prior_conflicts"]),
            "processed_alias_count": len(processed_aliases),
            "processed_conflict_count": len(processed_conflicts),
            "exclusion_alias_hits": exclusion_alias_hits,
            "status": "conflict" if processed_conflicts else "no_conflict",
            "processed_aliases": processed_aliases,
            "processed_conflicts": processed_conflicts,
        },
    }
    write_json(OUT / "eligible-manifest.json", manifest)
    write_json(OUT / "exclusions.json", {k: [list(x) for x in sorted(v)] for k, v in exclusion_parts.items()})
    write_json(OUT / "identity-audit.json", {**identity_audit, "processed_aliases": processed_aliases, "processed_conflicts": processed_conflicts, "exclusion_alias_hits": exclusion_alias_hits})
    if processed_conflicts:
        raise RuntimeError("processed manifest identity conflict; new cohort is held")
    return {"manifest": manifest, "selected": selected, "runtime": runtime, "panel_source": panel}


def _old_reduce_module():
    path = MATURE / "reduce.py"
    spec = importlib.util.spec_from_file_location("mature_reduce", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    # The mature package reducer is a thin driver whose ``m`` member is the
    # accepted score implementation used by its saved reduction.
    return getattr(mod, "m", mod)


def _rows_from_tokens(tokens: list[int]) -> list[dict[str, Any]]:
    rows = []
    for start, token in enumerate(tokens):
        if token != 151646:
            continue
        try:
            end = tokens.index(151647, start + 1)
        except ValueError:
            continue
        if 151646 in tokens[start + 1 : end] or end + 6 >= len(tokens):
            continue
        if tokens[end + 1] != 151648 or tokens[end + 6] != 151649:
            continue
        coords = tokens[end + 2 : end + 6]
        if not all(151670 <= t <= 152669 for t in coords):
            continue
        values = [t - 151670 for t in coords]
        rows.append(
            {
                "row_index": len(rows),
                "start": start,
                "end": end + 7,
                "description_tokens": tokens[start + 1 : end],
                "values": values,
                "valid": values[0] < values[2] and values[1] < values[3],
            }
        )
    return rows


def _same(a: dict[str, Any], b: dict[str, Any], eps: int) -> bool:
    return a["description_tokens"] == b["description_tokens"] and max(abs(x - y) for x, y in zip(a["values"], b["values"])) <= eps


def recurrence_accounting(tokens: list[int]) -> dict[str, Any]:
    rows = _rows_from_tokens(tokens)
    exact = Counter((tuple(r["description_tokens"]), tuple(r["values"])) for r in rows)
    exact_later_indices = []
    exact_seen = set()
    for i, row in enumerate(rows):
        key = (tuple(row["description_tokens"]), tuple(row["values"]))
        if key in exact_seen:
            exact_later_indices.append(i)
        exact_seen.add(key)
    exact_later = len(exact_later_indices)
    exact_pairs = sum(n * (n - 1) // 2 for n in exact.values())
    near_pairs = 0
    near_later = 0
    near_seen = set()
    for i, row in enumerate(rows):
        hit = False
        for prev in rows[:i]:
            if _same(row, prev, 8):
                near_pairs += 1
                hit = True
        if hit:
            near_later += 1
            near_seen.add(i)
    exact_onset = next((i for i, r in enumerate(rows) if any(_same(r, p, 0) for p in rows[:i])), None)
    near_onset = next((i for i, r in enumerate(rows) if any(_same(r, p, 8) for p in rows[:i])), None)
    runs = []
    for row in rows:
        key = (tuple(row["description_tokens"]), tuple(row["values"]))
        if runs and runs[-1]["key"] == key:
            runs[-1]["length"] += 1
        else:
            runs.append({"key": key, "start_row": row["row_index"], "length": 1})
    longest = max(runs, key=lambda x: x["length"], default=None)
    return {
        "parsed_rows": len(rows),
        "invalid_rows": sum(not r["valid"] for r in rows),
        "exact_repeat_rows": exact_later,
        "exact_invalid_repeat_rows": sum(not rows[i]["valid"] for i in exact_later_indices),
        "exact_pair_edges": exact_pairs,
        "near8_repeat_rows": near_later,
        "near8_invalid_repeat_rows": sum(not rows[i]["valid"] for i in near_seen),
        "near8_pair_edges": near_pairs,
        "near8_repeated_row_indices": sorted(near_seen),
        "exact_onset_row": exact_onset,
        "near8_onset_row": near_onset,
        "longest_exact_run": None if longest is None else {"start_row": longest["start_row"], "length": longest["length"]},
        "rows": rows,
    }


def _area_bin(box: list[int]) -> str:
    area = max(0, box[2] - box[0]) * max(0, box[3] - box[1]) / 1_000_000
    return "tiny" if area < 0.01 else "small" if area < 0.05 else "medium" if area < 0.2 else "large"


def _region(box: list[int]) -> str:
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    return ("left" if cx < 500 else "right") + ("-top" if cy < 500 else "-bottom")


def _phase(row: int, n: int) -> str:
    if n <= 0:
        return "unknown"
    q = row / n
    return "early" if q < 0.25 else "middle" if q < 0.75 else "late"


def _annotation_meta(case: dict[str, Any]) -> dict[str, Any]:
    objs = case["input_record"].get("objects", [])
    boxes = [[int(COORD.fullmatch(v).group(1)) for v in o["bbox_2d"]] for o in objs]
    desc = [str(o.get("desc", "")).strip().lower() for o in objs]
    areas = [max(0, b[2] - b[0]) * max(0, b[3] - b[1]) / 1_000_000 for b in boxes]
    return {
        "annotation_count": len(objs),
        "annotation_density_bin": "low" if len(objs) <= 4 else "medium" if len(objs) <= 15 else "high",
        "annotation_area_median": float(np.median(areas)) if areas else None,
        "annotation_small_fraction": float(sum(a < 0.01 for a in areas) / len(areas)) if areas else None,
        "annotation_category_count": len(set(desc)),
        "annotation_duplicate_description_count": sum(max(0, n - 1) for n in Counter(desc).values()),
        "annotation_overlap_proxy": "unknown",
    }


def mature_census() -> dict[str, Any]:
    panel = json.loads((MATURE / "panel.json").read_text())
    reduce_mod = _old_reduce_module()
    cells: dict[str, dict[str, Any]] = {}
    image_meta: dict[str, dict[str, Any]] = {}
    condition_names = list(panel["conditions"])
    for condition in condition_names:
        image_cells = {}
        for group in panel["groups"]:
            runtime = MATURE / "runtime" / condition / group["key"]
            raw = json.loads((runtime / "raw.json").read_text())
            rec = json.loads((runtime / "receipt.json").read_text())
            assert rec["status"] == "candidate_complete"
            for j, case in enumerate(group["cases"]):
                iid = str(case["input_record"]["image_id"])
                bank_name = "sentinel" if iid in panel.get("sentinel_banks", {}) else "refined"
                bank = panel[f"{bank_name}_banks"][iid]
                scored = reduce_mod.score(raw["rows"][j], case, bank)
                account = recurrence_accounting(raw["rows"][j]["token_ids"])
                preds = scored["valid_predictions"]
                complete_rows = scored["complete_rows"]
                if len(complete_rows) != len(account["rows"]):
                    raise RuntimeError(
                        f"complete-row alignment mismatch for {condition}/{iid}: "
                        f"token_account={len(account['rows'])} scored={len(complete_rows)}"
                    )
                for i, row in enumerate(account["rows"]):
                    if list(row["values"]) != list(complete_rows[i]["box"]):
                        raise RuntimeError(f"complete-row coordinate mismatch for {condition}/{iid}/row{i}")
                recurrent_rows = set(account["near8_repeated_row_indices"])
                pred_categories = Counter(str(p["description"]).strip().lower() for p in preds)
                complete_categories = Counter(str(p["description"]).strip().lower() for p in complete_rows)
                valid_complete_categories = Counter(
                    str(p["description"]).strip().lower()
                    for p in complete_rows
                    if p["box"][0] < p["box"][2] and p["box"][1] < p["box"][3]
                )
                invalid_complete_categories = complete_categories - valid_complete_categories
                exact_category = Counter()
                exact_category_valid = Counter()
                exact_category_invalid = Counter()
                near_category = Counter()
                near_category_valid = Counter()
                near_category_invalid = Counter()
                exact_region = Counter()
                exact_region_valid = Counter()
                exact_region_invalid = Counter()
                near_region = Counter()
                near_region_valid = Counter()
                near_region_invalid = Counter()
                exact_size = Counter()
                exact_size_valid = Counter()
                exact_size_invalid = Counter()
                near_size = Counter()
                near_size_valid = Counter()
                near_size_invalid = Counter()

                def add_repeat(
                    i: int,
                    category: Counter,
                    category_valid: Counter,
                    category_invalid: Counter,
                    region: Counter,
                    region_valid: Counter,
                    region_invalid: Counter,
                    size: Counter,
                    size_valid: Counter,
                    size_invalid: Counter,
                ) -> None:
                    p = complete_rows[i]
                    desc = str(p["description"]).strip().lower()
                    box = [int(x) for x in p["box"]]
                    valid_geometry = box[0] < box[2] and box[1] < box[3]
                    category[desc] += 1
                    region[_region(box)] += 1
                    size[_area_bin(box)] += 1
                    (category_valid if valid_geometry else category_invalid)[desc] += 1
                    (region_valid if valid_geometry else region_invalid)[_region(box)] += 1
                    (size_valid if valid_geometry else size_invalid)[_area_bin(box)] += 1

                exact_indices = []
                seen_exact = set()
                for i, row in enumerate(account["rows"]):
                    key = (tuple(row["description_tokens"]), tuple(row["values"]))
                    if key in seen_exact:
                        exact_indices.append(i)
                    seen_exact.add(key)
                for i in exact_indices:
                    add_repeat(i, exact_category, exact_category_valid, exact_category_invalid, exact_region, exact_region_valid, exact_region_invalid, exact_size, exact_size_valid, exact_size_invalid)
                for i in recurrent_rows:
                    add_repeat(i, near_category, near_category_valid, near_category_invalid, near_region, near_region_valid, near_region_invalid, near_size, near_size_valid, near_size_invalid)
                z = {
                    "condition": condition,
                    "model": condition.split("-")[0],
                    "policy": condition.split("-", 1)[1],
                    "runtime": "native_hf_fp32_sdpa_rp1_cap3084",
                    "image_id": int(iid),
                    "group": group["key"],
                    "batch_index": j,
                    "cohort": group["cohort"],
                    "runtime_receipt": binding(runtime / "receipt.json"),
                    "raw": binding(runtime / "raw.json"),
                    "trace": binding(runtime / "trace.json"),
                    "source_row": case["row_id"],
                    "input_meta": _annotation_meta(case),
                    "output": {
                        "complete_rows": scored["burden"]["complete_rows"],
                        "valid_rows": scored["burden"]["valid"],
                        "invalid_rows": scored["burden"]["invalid"],
                        "malformed_rows": scored["burden"]["malformed"],
                        "strict_iou95_repeat_rows": scored["burden"]["strict_valid_repeats"],
                        "literal_exact_repeat_rows": account["exact_repeat_rows"],
                        "literal_exact_invalid_repeat_rows": account["exact_invalid_repeat_rows"],
                        "literal_exact_pair_edges": account["exact_pair_edges"],
                        "near8_same_description_repeat_rows": account["near8_repeat_rows"],
                        "near8_literal_invalid_repeat_rows": account["near8_invalid_repeat_rows"],
                        "near8_same_description_pair_edges": account["near8_pair_edges"],
                        "exact_onset_row": account["exact_onset_row"],
                        "near8_onset_row": account["near8_onset_row"],
                        "longest_exact_run": account["longest_exact_run"],
                        "eos": scored["burden"]["eos"],
                        "cap": scored["burden"]["cap"],
                        "token_count": scored["token_count"],
                        "endpoint_occupancy": scored["endpoint_occupancy"],
                        "category_image_exposure": sorted(pred_categories),
                        "category_output_rows": dict(pred_categories),
                        "category_complete_rows": dict(complete_categories),
                        "category_valid_complete_rows": dict(valid_complete_categories),
                        "category_invalid_complete_rows": dict(invalid_complete_categories),
                        "exact_repeat_categories": dict(exact_category),
                        "exact_repeat_categories_valid": dict(exact_category_valid),
                        "exact_repeat_categories_invalid": dict(exact_category_invalid),
                        "near8_repeat_categories": dict(near_category),
                        "near8_repeat_categories_valid": dict(near_category_valid),
                        "near8_repeat_categories_invalid": dict(near_category_invalid),
                        "exact_repeat_regions": dict(exact_region),
                        "exact_repeat_regions_valid": dict(exact_region_valid),
                        "exact_repeat_regions_invalid": dict(exact_region_invalid),
                        "near8_repeat_regions": dict(near_region),
                        "near8_repeat_regions_valid": dict(near_region_valid),
                        "near8_repeat_regions_invalid": dict(near_region_invalid),
                        "exact_repeat_sizes": dict(exact_size),
                        "exact_repeat_sizes_valid": dict(exact_size_valid),
                        "exact_repeat_sizes_invalid": dict(exact_size_invalid),
                        "near8_repeat_sizes": dict(near_size),
                        "near8_repeat_sizes_valid": dict(near_size_valid),
                        "near8_repeat_sizes_invalid": dict(near_size_invalid),
                    },
                }
                image_cells[iid] = z
                image_meta.setdefault(iid, {"cohort": group["cohort"], "group": group["key"], "case": case, "input_meta": z["input_meta"]})
        cells[condition] = image_cells

    def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
        out = {"images": len(items)}
        numeric = [
            "complete_rows", "valid_rows", "invalid_rows", "malformed_rows", "strict_iou95_repeat_rows",
            "literal_exact_repeat_rows", "literal_exact_invalid_repeat_rows", "literal_exact_pair_edges", "near8_same_description_repeat_rows",
            "near8_literal_invalid_repeat_rows",
            "near8_same_description_pair_edges", "token_count", "eos", "cap",
        ]
        for k in numeric:
            out[k] = sum(int(x["output"][k]) for x in items)
        out["image_exposure_exact"] = sum(x["output"]["literal_exact_repeat_rows"] > 0 for x in items)
        out["image_exposure_near8"] = sum(x["output"]["near8_same_description_repeat_rows"] > 0 for x in items)
        out["category"] = {}
        for x in items:
            y = x["output"]
            for cat in (
                set(y["category_image_exposure"])
                | set(y["category_complete_rows"])
                | set(y["exact_repeat_categories"])
                | set(y["near8_repeat_categories"])
            ):
                z = out["category"].setdefault(
                    cat,
                    {
                        "image_exposure": 0,
                        "complete_row_images": 0,
                        "valid_complete_row_images": 0,
                        "invalid_complete_row_images": 0,
                        "exact_repeat_images": 0,
                        "near8_repeat_images": 0,
                        "output_rows": 0,
                        "complete_rows": 0,
                        "valid_complete_rows": 0,
                        "invalid_complete_rows": 0,
                        "exact_repeat_rows": 0,
                        "exact_repeat_rows_valid": 0,
                        "exact_repeat_rows_invalid": 0,
                        "near8_repeat_rows": 0,
                        "near8_repeat_rows_valid": 0,
                        "near8_repeat_rows_invalid": 0,
                    },
                )
                if cat in y["category_image_exposure"]:
                    z["image_exposure"] += 1
                if cat in y["category_complete_rows"]:
                    z["complete_row_images"] += 1
                if cat in y["category_valid_complete_rows"]:
                    z["valid_complete_row_images"] += 1
                if cat in y["category_invalid_complete_rows"]:
                    z["invalid_complete_row_images"] += 1
                if cat in y["exact_repeat_categories"]:
                    z["exact_repeat_images"] += 1
                if cat in y["near8_repeat_categories"]:
                    z["near8_repeat_images"] += 1
                z["output_rows"] += y["category_output_rows"].get(cat, 0)
                z["complete_rows"] += y["category_complete_rows"].get(cat, 0)
                z["valid_complete_rows"] += y["category_valid_complete_rows"].get(cat, 0)
                z["invalid_complete_rows"] += y["category_invalid_complete_rows"].get(cat, 0)
                z["exact_repeat_rows"] += y["exact_repeat_categories"].get(cat, 0)
                z["exact_repeat_rows_valid"] += y["exact_repeat_categories_valid"].get(cat, 0)
                z["exact_repeat_rows_invalid"] += y["exact_repeat_categories_invalid"].get(cat, 0)
                z["near8_repeat_rows"] += y["near8_repeat_categories"].get(cat, 0)
                z["near8_repeat_rows_valid"] += y["near8_repeat_categories_valid"].get(cat, 0)
                z["near8_repeat_rows_invalid"] += y["near8_repeat_categories_invalid"].get(cat, 0)
        # Rare-category uncertainty is resampled at the image unit. Token and
        # pair edges never enter this bootstrap.
        if items:
            rng = np.random.default_rng(19)
            draws = rng.integers(0, len(items), (10000, len(items)))
            for cat, z in out["category"].items():
                exposure = np.asarray([cat in x["output"]["category_image_exposure"] for x in items], dtype=float)
                exact = np.asarray([cat in x["output"]["exact_repeat_categories"] for x in items], dtype=float)
                near = np.asarray([cat in x["output"]["near8_repeat_categories"] for x in items], dtype=float)
                z["image_exposure_rate"] = float(exposure.mean())
                z["image_exposure_rate_ci95"] = np.quantile(exposure[draws].mean(axis=1), [0.025, 0.975]).tolist()
                z["exact_repeat_image_rate_ci95"] = np.quantile(exact[draws].mean(axis=1), [0.025, 0.975]).tolist()
                z["near8_repeat_image_rate_ci95"] = np.quantile(near[draws].mean(axis=1), [0.025, 0.975]).tolist()
        return out

    summary = {}
    for condition, by_image in cells.items():
        values = list(by_image.values())
        summary[condition] = aggregate(values)
        # Image-unit bootstrap for the two recurrence exposure rates.
        rng = np.random.default_rng(19)
        exact = np.asarray([x["output"]["literal_exact_repeat_rows"] > 0 for x in values], dtype=float)
        near = np.asarray([x["output"]["near8_same_description_repeat_rows"] > 0 for x in values], dtype=float)
        draws = rng.integers(0, len(values), (10000, len(values)))
        summary[condition]["image_bootstrap_seed19"] = {
            "unit": "image",
            "resamples": 10000,
            "exact_exposure_rate_ci95": np.quantile(exact[draws].mean(axis=1), [0.025, 0.975]).tolist(),
            "near8_exposure_rate_ci95": np.quantile(near[draws].mean(axis=1), [0.025, 0.975]).tolist(),
        }
    out = {
        "schema": "recurrence_census.mature_saved_output.v1",
        "status": "candidate_cpu_reduced",
        "source_panel": binding(MATURE / "panel.json"),
        "source_reduction": binding(MATURE / "reduction.json"),
        "source_unit": binding(REPO / "research/experiments/2026-09-18-untied-highconfidence18-natural/unit.md"),
        "images": image_meta,
        "cells": cells,
        "summary": summary,
        "denominator": {"unique_images": len(image_meta), "outputs": len(image_meta) * len(condition_names), "conditions": condition_names},
        "runtime_strata": {"mature": len(image_meta), "production_val200": {"status": "not_available"}},
    }
    write_json(OUT / "mature-census.json", out)
    return out


def _case_for_boundary(b: dict[str, Any], mature_panel: dict[str, Any], feedback_panel: dict[str, Any]) -> dict[str, Any] | None:
    for g in mature_panel["groups"]:
        if g["key"] == b["group"]:
            for c in g["cases"]:
                if int(c["input_record"]["image_id"]) == int(b["image_id"]):
                    return c
    for g in feedback_panel["groups"]:
        if g["key"] == b["group"]:
            for c in g["cases"]:
                if int(c["input_record"]["image_id"]) == int(b["image_id"]):
                    return c
    return None


def _boundary_metadata(b: dict[str, Any], case: dict[str, Any] | None) -> dict[str, Any]:
    source = b["source_row"]
    box = [int(x) for x in source["values"]]
    category = "token_ids:" + ",".join(map(str, source.get("description_tokens", [])))
    meta = _annotation_meta(case) if case else {"annotation_count": None, "annotation_density_bin": "unknown"}
    return {
        "predicted_category": category,
        "coordinate_region": _region(box),
        "box_size_bin": _area_bin(box),
        "sequence_phase": "source_row",
        "annotation_density_proxy": meta.get("annotation_density_bin", "unknown"),
        "annotation_context_proxy": meta,
    }


def existing_shared_boundaries() -> list[dict[str, Any]]:
    selection = json.loads((FEEDBACK / "selection.json").read_text())
    mature_panel = json.loads((MATURE / "panel.json").read_text())
    feedback_panel = json.loads((FEEDBACK / "panel.json").read_text())
    out = []
    for raw in selection["boundaries"]:
        b = copy.deepcopy(raw)
        for field in ("raw_path", "trace_path", "receipt_path"):
            p = Path(b[field])
            b.setdefault("bindings", {})[field.removesuffix("_path")] = binding(p)
        c = _case_for_boundary(b, mature_panel, feedback_panel)
        b.update(
            {
                "condition": f"{b['model']}-original",
                "split": None if c is None else str(c["input_record"]["metadata"]["split"]),
                "runtime": "native_hf_fp32_sdpa_rp1_cap3084",
                "selection_stratum": "mature_saved_numerical_feedback_boundary",
                "source_prefix_hash": b.get("prefix_hash"),
                "target_slots": b.get("target_slots", []),
                "native_tokens": b.get("native_tokens", []),
                "metadata": _boundary_metadata(b, c),
            }
        )
        out.append(b)
    return out


def freeze_panel_prelaunch() -> dict[str, Any]:
    manifest = json.loads((OUT / "eligible-manifest.json").read_text())
    panel_source = json.loads((MATURE / "panel.json").read_text())
    feedback_panel = json.loads((FEEDBACK / "panel.json").read_text())
    boundaries = existing_shared_boundaries()
    shared_sources = {
        "schema": "recurrence_census.shared_sources.v1",
        "status": "frozen_before_new_outputs",
        "unit_id": "2026-09-19-recurrence-distribution-census",
        "processed_manifest": manifest,
        "mature_panel": binding(MATURE / "panel.json"),
        "mature_reduction": binding(MATURE / "reduction.json"),
        "feedback_panel": binding(FEEDBACK / "panel.json"),
        "feedback_selection": binding(FEEDBACK / "selection.json"),
        "checkpoint_bindings": {
            "tied_step2444": path_binding(Path("/data/CoordExp/outputs/research/eight-coordinate-bbox-supervision/2026-08-05-closeout/artifacts/training/four-coordinate-xy/checkpoints/step-2444")),
            "untied_axis_step2444": path_binding(Path("/data/CoordExp/outputs/infra_base/train/qwen3-vl-2b-geo-sorted-xy-untied-axis001-ebs24-4epoch/checkpoints/step-2444")),
            "base": path_binding(BASE),
        },
        "existing_prospective_exclusions": json.loads((OUT / "exclusions.json").read_text()),
        "known_boundary_ids": [b["id"] for b in boundaries],
        "production_val200": {"status": "not_available"},
        "identity_rule": "(metadata.split,image_id) is the identity; canonical resolved image path, file_name and image sha256 are cross-checks; split is never dropped",
        "mature_image_count": 145,
        "new_image_count": 128,
        "new_output_count": 256,
        "conditions": ["tied-original", "untied-original"],
    }
    selection_rule = {
        "failure_selector": "first completed row of earliest >=3 exact same-description+coordinate run, else earliest >=3 pairwise <=8-bin same-description near run; invalid geometry retained",
        "proxy_selector": "first available non-recurrent complete row when no failure row exists; otherwise the existing healthy matching key (validity, coordinate L1 distance, prefix length, row order); no outcome rescue",
        "stratification": ["source", "model", "predicted_category", "coordinate_region", "box_size_bin", "sequence_phase", "annotation_density_proxy"],
        "per_image_model": {"max_failure": 1, "max_nonrecurrent_proxy": 1},
        "caps": {"failure": 24, "nonrecurrent_proxy": 24},
        "known_sources": ["val885", "val5586", "val7511", "val14038", "val632", "donut417044"],
        "control_rule": "bird309264 may be selected as a non-recurrent proxy when qualified; never force a failure boundary",
        "new_cohort": "apply exactly this function after new original-policy outputs are saved; no manual difficulty/outcome replacement",
        "freeze_point": "before new intervention outcomes",
        "selector_source": binding(REPO / "probes/training_set_completion/numerical_feedback/select.py"),
        "panel_scope": "fixed mechanism panel selected from mature accepted boundaries plus the prospective seed19 cohort; not a census of all recurrence",
    }
    shared_sources["selection_rule"] = selection_rule
    write_json(OUT / "shared-sources.json", shared_sources)
    prelaunch_panel = {
        "schema": "recurrence_census.prelaunch_mechanism_rule.v1",
        "status": "rule_frozen_before_new_outputs",
        "unit_id": "2026-09-19-recurrence-distribution-census",
        "caps": {"failure": 24, "nonrecurrent_proxy": 24},
        "selection_rule": selection_rule,
        "existing_boundaries": boundaries,
        "new_cohort": {"count": 128, "output_count": 256, "source": binding(OUT / "new128.runtime.jsonl"), "pending": True},
        "sources": [binding(OUT / "shared-sources.json"), binding(FEEDBACK / "selection.json"), binding(MATURE / "panel.json")],
        "conditions": ["tied-original", "untied-original"],
    }
    write_json(OUT / "selection-rule.json", {"selection_rule": selection_rule, "shared_sources": binding(OUT / "shared-sources.json"), "status": "frozen_before_new_outputs"})
    write_json(OUT / "prelaunch-panel.json", prelaunch_panel)
    write_json(OUT / "prelaunch-receipt.json", {"status": "candidate", "manifest": binding(OUT / "eligible-manifest.json"), "shared_sources": binding(OUT / "shared-sources.json"), "prelaunch_panel": binding(OUT / "prelaunch-panel.json"), "selection_rule": binding(OUT / "selection-rule.json"), "mature_census": binding(OUT / "mature-census.json"), "new_generation_authorized_after": "this receipt"})
    return prelaunch_panel


def main() -> None:
    freeze_cohort()
    mature_census()
    panel = freeze_panel_prelaunch()
    print(json.dumps({"status": panel["status"], "new_images": 128, "existing_boundaries": len(panel["existing_boundaries"]), "shared_sources": str(OUT / "shared-sources.json"), "prelaunch_panel": str(OUT / "prelaunch-panel.json"), "final_shared_panel": "pending_new_outputs"}, indent=2))


if __name__ == "__main__":
    main()
