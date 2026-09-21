"""Reduce Lane-A original natural outputs and freeze the final mechanism panel."""
from __future__ import annotations

import collections
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np

from probes.training_set_completion.recurrence_census.prepare import (
    COORD,
    OUT,
    _annotation_meta,
    _area_bin,
    _boundary_metadata,
    _old_reduce_module,
    _region,
    binding,
    recurrence_accounting,
    write_json,
)
from probes.training_set_completion.numerical_feedback import select as frozen_select


def _cell(condition: str, group: dict[str, Any], j: int, raw: dict[str, Any], case: dict[str, Any], score_mod: Any, bank: list[dict[str, Any]]) -> dict[str, Any]:
    row = raw["rows"][j]
    score = score_mod.score(row, case, bank)
    account = recurrence_accounting(row["token_ids"])
    complete = score["complete_rows"]
    if len(complete) != len(account["rows"]):
        raise RuntimeError(f"new complete-row alignment mismatch {condition}/{group['key']}/{j}")
    for i, parsed in enumerate(complete):
        if list(parsed["box"]) != list(account["rows"][i]["values"]):
            raise RuntimeError(f"new complete-row coordinate mismatch {condition}/{group['key']}/{j}/{i}")
    preds = score["valid_predictions"]
    pred_categories = collections.Counter(str(p["description"]).strip().lower() for p in preds)
    complete_categories = collections.Counter(str(p["description"]).strip().lower() for p in complete)
    valid_categories = collections.Counter(
        str(p["description"]).strip().lower()
        for p in complete
        if p["box"][0] < p["box"][2] and p["box"][1] < p["box"][3]
    )
    invalid_categories = complete_categories - valid_categories
    exact_indices = []
    seen = set()
    for i, parsed in enumerate(account["rows"]):
        key = (tuple(parsed["description_tokens"]), tuple(parsed["values"]))
        if key in seen:
            exact_indices.append(i)
        seen.add(key)
    near_indices = set(account["near8_repeated_row_indices"])

    def repeat_counts(indices: list[int] | set[int]) -> tuple[dict[str, int], dict[str, int], dict[str, int], dict[str, int], dict[str, int], dict[str, Any]]:
        cats: collections.Counter[str] = collections.Counter()
        cats_valid: collections.Counter[str] = collections.Counter()
        cats_invalid: collections.Counter[str] = collections.Counter()
        regions: collections.Counter[str] = collections.Counter()
        sizes: collections.Counter[str] = collections.Counter()
        for i in indices:
            parsed = complete[i]
            desc = str(parsed["description"]).strip().lower()
            box = [int(x) for x in parsed["box"]]
            valid = box[0] < box[2] and box[1] < box[3]
            cats[desc] += 1
            (cats_valid if valid else cats_invalid)[desc] += 1
            regions[_region(box)] += 1
            sizes[_area_bin(box)] += 1
        return dict(cats), dict(cats_valid), dict(cats_invalid), dict(regions), dict(sizes), {"valid": sum(cats_valid.values()), "invalid": sum(cats_invalid.values())}

    exact_c, exact_cv, exact_ci, exact_r, exact_s, exact_vi = repeat_counts(exact_indices)
    near_c, near_cv, near_ci, near_r, near_s, near_vi = repeat_counts(near_indices)
    return {
        "condition": condition,
        "model": condition.split("-", 1)[0],
        "policy": "original",
        "runtime": "native_hf_fp32_sdpa_rp1_cap3084",
        "split": str(case["input_record"]["metadata"]["split"]),
        "image_id": int(case["input_record"]["image_id"]),
        "image_key": f"{case['input_record']['metadata']['split']}:{case['input_record']['image_id']}",
        "group": group["key"],
        "batch_index": j,
        "source_row": case["row_index"],
        "runtime_receipt": binding(OUT / "runtime" / condition / group["key"] / "receipt.json"),
        "raw": binding(OUT / "runtime" / condition / group["key"] / "raw.json"),
        "trace": binding(OUT / "runtime" / condition / group["key"] / "trace.json"),
        "input_meta": _annotation_meta(case),
        "output": {
            "complete_rows": score["burden"]["complete_rows"],
            "valid_rows": score["burden"]["valid"],
            "invalid_rows": score["burden"]["invalid"],
            "malformed_rows": score["burden"]["malformed"],
            "strict_iou95_repeat_rows": score["burden"]["strict_valid_repeats"],
            "literal_exact_repeat_rows": account["exact_repeat_rows"],
            "literal_exact_invalid_repeat_rows": account["exact_invalid_repeat_rows"],
            "literal_exact_pair_edges": account["exact_pair_edges"],
            "near8_same_description_repeat_rows": account["near8_repeat_rows"],
            "near8_literal_invalid_repeat_rows": account["near8_invalid_repeat_rows"],
            "near8_same_description_pair_edges": account["near8_pair_edges"],
            "exact_onset_row": account["exact_onset_row"],
            "near8_onset_row": account["near8_onset_row"],
            "longest_exact_run": account["longest_exact_run"],
            "eos": score["burden"]["eos"],
            "cap": score["burden"]["cap"],
            "token_count": score["token_count"],
            "endpoint_occupancy": score["endpoint_occupancy"],
            "category_image_exposure": sorted(pred_categories),
            "category_output_rows": dict(pred_categories),
            "category_complete_rows": dict(complete_categories),
            "category_valid_complete_rows": dict(valid_categories),
            "category_invalid_complete_rows": dict(invalid_categories),
            "exact_repeat_categories": exact_c,
            "exact_repeat_categories_valid": exact_cv,
            "exact_repeat_categories_invalid": exact_ci,
            "near8_repeat_categories": near_c,
            "near8_repeat_categories_valid": near_cv,
            "near8_repeat_categories_invalid": near_ci,
            "exact_repeat_regions": exact_r,
            "near8_repeat_regions": near_r,
            "exact_repeat_sizes": exact_s,
            "near8_repeat_sizes": near_s,
            "exact_repeat_valid_invalid": exact_vi,
            "near8_repeat_valid_invalid": near_vi,
        },
    }


def _aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
    numeric = [
        "complete_rows", "valid_rows", "invalid_rows", "malformed_rows", "strict_iou95_repeat_rows",
        "literal_exact_repeat_rows", "literal_exact_invalid_repeat_rows", "literal_exact_pair_edges",
        "near8_same_description_repeat_rows", "near8_literal_invalid_repeat_rows", "near8_same_description_pair_edges",
        "token_count", "eos", "cap",
    ]
    out: dict[str, Any] = {"images": len({x["image_key"] for x in items})}
    for k in numeric:
        out[k] = sum(int(x["output"][k]) for x in items)
    out["image_exposure_exact"] = len({x["image_key"] for x in items if x["output"]["literal_exact_repeat_rows"] > 0})
    out["image_exposure_near8"] = len({x["image_key"] for x in items if x["output"]["near8_same_description_repeat_rows"] > 0})
    out["category"] = {}
    for x in items:
        y = x["output"]
        categories = set(y["category_image_exposure"]) | set(y["category_complete_rows"]) | set(y["exact_repeat_categories"]) | set(y["near8_repeat_categories"])
        for cat in categories:
            z = out["category"].setdefault(cat, {
                "image_exposure": 0, "complete_row_images": 0, "valid_complete_row_images": 0,
                "invalid_complete_row_images": 0, "exact_repeat_images": 0, "near8_repeat_images": 0,
                "output_rows": 0, "complete_rows": 0, "valid_complete_rows": 0, "invalid_complete_rows": 0,
                "exact_repeat_rows": 0, "exact_repeat_rows_valid": 0, "exact_repeat_rows_invalid": 0,
                "near8_repeat_rows": 0, "near8_repeat_rows_valid": 0, "near8_repeat_rows_invalid": 0,
            })
            z["image_exposure"] += int(cat in y["category_image_exposure"])
            z["complete_row_images"] += int(cat in y["category_complete_rows"])
            z["valid_complete_row_images"] += int(cat in y["category_valid_complete_rows"])
            z["invalid_complete_row_images"] += int(cat in y["category_invalid_complete_rows"])
            z["exact_repeat_images"] += int(cat in y["exact_repeat_categories"])
            z["near8_repeat_images"] += int(cat in y["near8_repeat_categories"])
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
    # Rare-category intervals use images as the bootstrap unit.
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


def _round_robin(candidates: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    strata: dict[tuple[str, ...], list[dict[str, Any]]] = collections.defaultdict(list)
    for candidate in candidates:
        meta = candidate["metadata"]
        strata[
            (
                candidate["model"],
                meta.get("predicted_category", "unknown"),
                meta.get("coordinate_region", "unknown"),
                meta.get("box_size_bin", "unknown"),
                meta.get("sequence_phase", "unknown"),
                meta.get("annotation_density_proxy", "unknown"),
                "prospective_new_cohort_seed19",
            )
        ].append(candidate)
    for values in strata.values():
        values.sort(key=lambda x: (x["split"], x["image_id"], x["group"], x["batch_index"], x["source_row"]["index"]))
    selected = []
    for key in sorted(strata):
        strata[key].sort(key=lambda x: (x["split"], x["image_id"], x["group"], x["batch_index"], x["source_row"]["index"]))
    while len(selected) < limit and any(strata.values()):
        for key in sorted(strata):
            if strata[key] and len(selected) < limit:
                selected.append(strata[key].pop(0))
    return selected


def _new_boundaries(cells: dict[str, dict[str, Any]], panel: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    candidates: dict[str, list[dict[str, Any]]] = {"failure": [], "healthy": []}
    exclusions = []
    for condition in panel["conditions"]:
        model = condition.split("-", 1)[0]
        for group in panel["groups"]:
            for j, case in enumerate(group["cases"]):
                cell = cells[condition][f"{case['input_record']['metadata']['split']}:{case['input_record']['image_id']}"]
                raw_path = OUT / "runtime" / condition / group["key"] / "raw.json"
                trace_path = OUT / "runtime" / condition / group["key"] / "trace.json"
                receipt_path = OUT / "runtime" / condition / group["key"] / "receipt.json"
                raw_row = json.loads(raw_path.read_text())["rows"][j]
                rr = frozen_select.rows(raw_row["token_ids"])
                idx, episode = frozen_select.choose_episode(rr)
                if idx is None:
                    exclusions.append({"model": model, "split": cell["split"], "image_id": cell["image_id"], "condition": condition, "status": "no_qualifying_episode", "complete_rows": len(rr)})
                else:
                    candidates["failure"].append(_make_boundary("failure", episode, idx, rr, cell, condition, model, group, j, case, raw_path, trace_path, receipt_path))
                recurrent = {
                    j2
                    for i in range(len(rr) - 2)
                    if all(frozen_select.same(rr[k], rr[l], 8) for k, l in [(i, i + 1), (i, i + 2), (i + 1, i + 2)])
                    for j2 in range(i, i + 3)
                }
                healthy = [i2 for i2 in range(len(rr) - 1) if i2 not in recurrent and not any(frozen_select.same(rr[i2], prev, 8) for prev in rr[:i2])]
                if healthy:
                    if idx is not None:
                        ref = rr[idx]
                        hidx = min(healthy, key=lambda x: (rr[x]["valid"] != ref["valid"], sum(abs(a - b) for a, b in zip(rr[x]["values"], ref["values"])), abs(rr[x]["end"] - ref["end"]), x))
                    else:
                        hidx = healthy[0]
                    candidates["healthy"].append(_make_boundary("healthy", "nonrecurrent_proxy", hidx, rr, cell, condition, model, group, j, case, raw_path, trace_path, receipt_path))
                else:
                    exclusions.append({"model": model, "split": cell["split"], "image_id": cell["image_id"], "condition": condition, "status": "healthy_boundary_absent", "complete_rows": len(rr)})
    existing = json.loads((OUT / "prelaunch-panel.json").read_text())["existing_boundaries"]
    existing_failure = sum(b["kind"] == "failure" for b in existing)
    existing_healthy = sum(b["kind"] in {"healthy", "nonrecurrent_proxy"} for b in existing)
    selected = existing + _round_robin(candidates["failure"], max(0, 24 - existing_failure)) + _round_robin(candidates["healthy"], max(0, 24 - existing_healthy))
    return selected, exclusions, {
        "failure_candidates": len(candidates["failure"]),
        "nonrecurrent_proxy_candidates": len(candidates["healthy"]),
        "failure_selected_new": min(len(candidates["failure"]), max(0, 24 - existing_failure)),
        "nonrecurrent_proxy_selected_new": min(len(candidates["healthy"]), max(0, 24 - existing_healthy)),
    }


def _make_boundary(kind: str, episode: str, i: int, rr: list[dict[str, Any]], cell: dict[str, Any], condition: str, model: str, group: dict[str, Any], batch_index: int, case: dict[str, Any], raw_path: Path, trace_path: Path, receipt_path: Path) -> dict[str, Any]:
    row = rr[i]
    targets = [
        {"row_delay": delay - 1, "row_index": i + delay, "role": role, "offset": rr[i + delay]["coordinate_offsets"][k]}
        for delay in [1, 2, 4]
        if i + delay < len(rr)
        for k, role in enumerate(("x1", "y1", "x2", "y2"))
    ]
    source = {
        "index": row["index"],
        "start": row["start"],
        "end": row["end"],
        "description_tokens": row["description_tokens"],
        "coordinate_offsets": row["coordinate_offsets"],
        "values": row["values"],
        "valid": row["valid"],
    }
    boundary = {
        "id": f"{model}-{cell['split']}-{cell['image_id']}-{kind}",
        "model": model,
        "condition": condition,
        "image_id": cell["image_id"],
        "split": cell["split"],
        "group": group["key"],
        "batch_index": batch_index,
        "kind": kind,
        "episode_stratum": episode,
        "selection_stratum": "prospective_new_cohort_seed19",
        "source_row": source,
        "previous_row": rr[i - 1] if i else None,
        "next_row": rr[i + 1] if i + 1 < len(rr) else None,
        "target_slots": targets,
        "fixed_suffix_end": rr[max(x["row_index"] for x in targets)]["end"] if targets else row["end"],
        "native_tokens": [],
        "source_prefix_hash": frozen_select.token_hash([]),
        "raw_path": str(raw_path),
        "trace_path": str(trace_path),
        "receipt_path": str(receipt_path),
        "healthy_matching": "same image/model; validity, coordinate L1 distance, prefix length priority; no exact-match claim",
        "physical_status": "not_required_UNKNOWN",
        "metadata": {},
    }
    raw_row = json.loads(raw_path.read_text())["rows"][batch_index]
    boundary["native_tokens"] = list(raw_row["token_ids"])
    boundary["native_token_hash"] = frozen_select.token_hash(boundary["native_tokens"])
    boundary["source_prefix_hash"] = frozen_select.token_hash(boundary["native_tokens"][: row["end"]])
    boundary["metadata"] = _boundary_metadata(boundary, case)
    boundary["bindings"] = {"raw": binding(raw_path), "trace": binding(trace_path), "receipt": binding(receipt_path)}
    return boundary


def main() -> None:
    panel = json.loads((OUT / "panel.json").read_text())
    score_mod = _old_reduce_module()
    cells: dict[str, dict[str, Any]] = {}
    receipts = []
    failed = []
    for condition in panel["conditions"]:
        by_image: dict[str, dict[str, Any]] = {}
        for group in panel["groups"]:
            runtime = OUT / "runtime" / condition / group["key"]
            receipt_path = runtime / "receipt.json"
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("status") != "candidate_complete":
                failed.append({"condition": condition, "group": group["key"], "receipt": binding(receipt_path), "status": receipt.get("status")})
                continue
            receipts.append(receipt)
            raw = json.loads((runtime / "raw.json").read_text())
            for j, case in enumerate(group["cases"]):
                key = f"{case['input_record']['metadata']['split']}:{case['input_record']['image_id']}"
                bank = panel["new_banks"][f"{case['input_record']['metadata']['split']}:{case['input_record']['image_id']}"]
                by_image[key] = _cell(condition, group, j, raw, case, score_mod, bank)
        cells[condition] = by_image
    summary = {}
    for condition, values_by_image in cells.items():
        values = list(values_by_image.values())
        s = _aggregate(values)
        rng = np.random.default_rng(19)
        n = len(values)
        draws = rng.integers(0, n, (10000, n)) if n else np.empty((0, 0), dtype=int)
        exact = np.asarray([x["output"]["literal_exact_repeat_rows"] > 0 for x in values], dtype=float)
        near = np.asarray([x["output"]["near8_same_description_repeat_rows"] > 0 for x in values], dtype=float)
        s["image_bootstrap_seed19"] = {
            "unit": "image",
            "resamples": 10000,
            "exact_exposure_rate_ci95": np.quantile(exact[draws].mean(axis=1), [0.025, 0.975]).tolist() if n else None,
            "near8_exposure_rate_ci95": np.quantile(near[draws].mean(axis=1), [0.025, 0.975]).tolist() if n else None,
        }
        summary[condition] = s
    out = {
        "schema": "recurrence_census.new_saved_output.v1",
        "status": "candidate_cpu_reduced" if not failed else "technical_invalid",
        "panel": binding(OUT / "panel.json"),
        "cells": cells,
        "summary": summary,
        "denominator": {"unique_images": len({x["image_key"] for values in cells.values() for x in values.values()}), "outputs": sum(len(x) for x in cells.values()), "conditions": panel["conditions"]},
        "failed_groups": failed,
        "cost": {
            "completed_groups": len(receipts),
            "failed_groups": len(failed),
            "model_forwards": sum(int(r.get("model_forwards", 0)) for r in receipts),
            "vision_forwards": sum(int(r.get("vision_forwards", 0)) for r in receipts),
            "gpu_seconds": sum(float(r.get("elapsed_seconds", 0.0)) for r in receipts),
            "retained_bytes": sum(Path(p).stat().st_size for p in OUT.glob("runtime/*/*/*.json") if p.is_file()),
        },
    }
    write_json(OUT / "new-census.json", out)
    if failed:
        raise RuntimeError(f"failed output groups: {len(failed)}")
    selected, exclusions, candidate_pool = _new_boundaries(cells, panel)
    existing = json.loads((OUT / "prelaunch-panel.json").read_text())["existing_boundaries"]
    final = {
        "schema": "recurrence_census.shared_mechanism_panel.v1",
        "status": "final_frozen",
        "unit_id": "2026-09-19-recurrence-distribution-census",
        "caps": {"failure": 24, "nonrecurrent_proxy": 24},
        "selection_rule": json.loads((OUT / "selection-rule.json").read_text())["selection_rule"],
        "existing_boundaries": existing,
        "new_boundaries": [x for x in selected if x not in existing],
        "all_boundaries": selected,
        "new_exclusions": exclusions,
        "candidate_pool": candidate_pool,
        "new_cohort": {"count": 128, "output_count": 256, "source": binding(OUT / "new128.runtime.jsonl"), "pending": False},
        "sources": [binding(OUT / "shared-sources.json"), binding(OUT / "prelaunch-panel.json"), binding(OUT / "panel.json"), binding(OUT / "new-census.json")],
        "conditions": panel["conditions"],
        "counts": {
            "existing": len(existing),
            "new": len(selected) - len(existing),
            "all": len(selected),
            "failure": sum(x["kind"] == "failure" for x in selected),
            "nonrecurrent_proxy": sum(x["kind"] in {"healthy", "nonrecurrent_proxy"} for x in selected),
        },
    }
    write_json(OUT / "shared-panel.json", final)
    write_json(OUT / "finalization-receipt.json", {"status": "candidate", "shared_panel": binding(OUT / "shared-panel.json"), "new_census": binding(OUT / "new-census.json"), "selected_boundaries": final["counts"]})
    print(json.dumps({"status": final["status"], "counts": final["counts"], "exclusions": len(exclusions), "cost": out["cost"]}, indent=2))


if __name__ == "__main__":
    main()
