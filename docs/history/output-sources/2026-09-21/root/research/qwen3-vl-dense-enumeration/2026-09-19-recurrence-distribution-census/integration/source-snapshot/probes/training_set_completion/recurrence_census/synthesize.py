"""CPU-only scientific synthesis from saved Lane-A census JSON.

This reads saved reductions and writes a separate summary artifact. It never
reselects or rewrites the shared mechanism panel.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np


OUT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-distribution-census")


def _ci(values: list[float], seed: int = 19) -> list[float] | None:
    if not values:
        return None
    a = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(a), (10000, len(a)))
    return np.quantile(a[draws].mean(axis=1), [0.025, 0.975]).tolist()


def _rate(items: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
    values = [float(predicate(x)) for x in items]
    return {"images": len(items), "exposed_images": int(sum(values)), "rate": float(np.mean(values)) if values else None, "image_bootstrap_ci95": _ci(values)}


def _distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "median": None, "q25": None, "q75": None, "max": None, "image_bootstrap_mean_ci95": None}
    a = np.asarray(values, dtype=float)
    return {"n": len(values), "median": float(np.median(a)), "q25": float(np.quantile(a, 0.25)), "q75": float(np.quantile(a, 0.75)), "max": float(np.max(a)), "image_bootstrap_mean_ci95": _ci(values)}


def _items(doc: dict[str, Any], condition: str) -> list[dict[str, Any]]:
    return list(doc["cells"][condition].values())


def _category_table(doc: dict[str, Any], condition: str) -> dict[str, Any]:
    items = _items(doc, condition)
    summary = doc["summary"][condition]
    table = {}
    for cat, value in summary["category"].items():
        table[cat] = {
            "total_images": len(items),
            "image_emitting": value["image_exposure"],
            "image_emitting_rate": value.get("image_exposure_rate"),
            "image_emitting_ci95": value.get("image_exposure_rate_ci95"),
            "exact_repeat_images": value["exact_repeat_images"],
            "exact_repeat_image_rate_ci95": value.get("exact_repeat_image_rate_ci95"),
            "near8_repeat_images": value["near8_repeat_images"],
            "near8_repeat_image_rate_ci95": value.get("near8_repeat_image_rate_ci95"),
            "output_rows": value["output_rows"],
            "complete_rows": value["complete_rows"],
            "valid_complete_rows": value["valid_complete_rows"],
            "invalid_complete_rows": value["invalid_complete_rows"],
            "exact_repeat_rows": value["exact_repeat_rows"],
            "exact_repeat_rows_valid": value["exact_repeat_rows_valid"],
            "exact_repeat_rows_invalid": value["exact_repeat_rows_invalid"],
            "near8_repeat_rows": value["near8_repeat_rows"],
            "near8_repeat_rows_valid": value["near8_repeat_rows_valid"],
            "near8_repeat_rows_invalid": value["near8_repeat_rows_invalid"],
            "rare_category": value["image_exposure"] < 5,
        }
    def rank(field: str) -> list[dict[str, Any]]:
        return sorted(
            ({"category": cat, **value} for cat, value in table.items()),
            key=lambda x: (-x[field], -x["image_emitting"], x["category"]),
        )[:15]
    return {"all": table, "top_by_image_emitting": rank("image_emitting"), "top_by_exact_repeat_images": rank("exact_repeat_images"), "top_by_near8_repeat_images": rank("near8_repeat_images")}


def _region_size(items: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for label, field in (("region", "exact_repeat_regions"), ("size", "exact_repeat_sizes")):
        exact_rows = Counter()
        near_rows = Counter()
        exact_exposure = Counter()
        near_exposure = Counter()
        for item in items:
            y = item["output"]
            for key, n in y[field].items():
                exact_rows[key] += int(n)
                exact_exposure[key] += 1
            near_field = "near8_repeat_regions" if label == "region" else "near8_repeat_sizes"
            for key, n in y[near_field].items():
                near_rows[key] += int(n)
                near_exposure[key] += 1
        output[label] = {
            "total_images": len(items),
            "exact_repeat_rows": dict(sorted(exact_rows.items())),
            "near8_repeat_rows": dict(sorted(near_rows.items())),
            "exact_repeat_image_exposure": {k: {"images": v, "rate": v / len(items) if items else None, "ci95": _ci([float(k in x["output"][field]) for x in items])} for k, v in sorted(exact_exposure.items())},
            "near8_repeat_image_exposure": {k: {"images": v, "rate": v / len(items) if items else None, "ci95": _ci([float(k in x["output"]["near8_repeat_regions" if label == "region" else "near8_repeat_sizes"]) for x in items])} for k, v in sorted(near_exposure.items())},
        }
    return output


def _onset(items: list[dict[str, Any]]) -> dict[str, Any]:
    exact = [float(x["output"]["exact_onset_row"]) for x in items if x["output"]["exact_onset_row"] is not None]
    near = [float(x["output"]["near8_onset_row"]) for x in items if x["output"]["near8_onset_row"] is not None]
    exact_norm = [float(x["output"]["exact_onset_row"]) / max(1, x["output"]["complete_rows"]) for x in items if x["output"]["exact_onset_row"] is not None]
    near_norm = [float(x["output"]["near8_onset_row"]) / max(1, x["output"]["complete_rows"]) for x in items if x["output"]["near8_onset_row"] is not None]
    runs = [float(x["output"]["longest_exact_run"]["length"]) for x in items if x["output"]["longest_exact_run"] is not None]
    repeated_runs = [x for x in runs if x > 1]
    return {
        "exact_onset_row": _distribution(exact),
        "near8_onset_row": _distribution(near),
        "exact_onset_fraction_of_complete_rows": _distribution(exact_norm),
        "near8_onset_fraction_of_complete_rows": _distribution(near_norm),
        "longest_exact_run": _distribution(runs),
        "longest_exact_run_gt1": _distribution(repeated_runs),
    }


def _proxy_groups(items: list[dict[str, Any]], key_fn: Callable[[dict[str, Any]], str]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        groups[key_fn(item)].append(item)
    result = {}
    for key, values in sorted(groups.items()):
        result[key] = {
            "images": len(values),
            "complete_rows": sum(x["output"]["complete_rows"] for x in values),
            "mean_complete_rows_per_image": float(np.mean([x["output"]["complete_rows"] for x in values])),
            "token_count": sum(x["output"]["token_count"] for x in values),
            "exact": _rate(values, lambda x: x["output"]["literal_exact_repeat_rows"] > 0),
            "near8": _rate(values, lambda x: x["output"]["near8_same_description_repeat_rows"] > 0),
            "exact_repeat_rows": sum(x["output"]["literal_exact_repeat_rows"] for x in values),
            "exact_pair_edges": sum(x["output"]["literal_exact_pair_edges"] for x in values),
            "near8_repeat_rows": sum(x["output"]["near8_same_description_repeat_rows"] for x in values),
            "near8_pair_edges": sum(x["output"]["near8_same_description_pair_edges"] for x in values),
            "invalid_rows": sum(x["output"]["invalid_rows"] for x in values),
        }
    return result


def _condition(doc: dict[str, Any], condition: str) -> dict[str, Any]:
    items = _items(doc, condition)
    summary = doc["summary"][condition]
    return {
        "images": len(items),
        "outputs": len(items),
        "image_exposure": {
            "exact": _rate(items, lambda x: x["output"]["literal_exact_repeat_rows"] > 0),
            "near8": _rate(items, lambda x: x["output"]["near8_same_description_repeat_rows"] > 0),
        },
        "rows_and_pairs": {k: summary[k] for k in ("literal_exact_repeat_rows", "literal_exact_pair_edges", "near8_same_description_repeat_rows", "near8_same_description_pair_edges", "literal_exact_invalid_repeat_rows", "near8_literal_invalid_repeat_rows")},
        "validity": {k: summary[k] for k in ("complete_rows", "valid_rows", "invalid_rows", "malformed_rows")},
        "onset_and_runs": _onset(items),
        "region_and_size": _region_size(items),
        "category": _category_table(doc, condition),
        "annotation_density_proxy": _proxy_groups(items, lambda x: x["input_meta"].get("annotation_density_bin", "unknown")),
        "duplicate_description_proxy": _proxy_groups(items, lambda x: "none" if x["input_meta"].get("annotation_duplicate_description_count", 0) == 0 else "1-5" if x["input_meta"].get("annotation_duplicate_description_count", 0) <= 5 else "6+"),
        "small_object_fraction_proxy": _proxy_groups(items, lambda x: "unknown" if x["input_meta"].get("annotation_small_fraction") is None else "low" if x["input_meta"]["annotation_small_fraction"] < 0.25 else "mixed" if x["input_meta"]["annotation_small_fraction"] < 0.75 else "high"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT / "scientific-synthesis.json")
    args = parser.parse_args()
    mature = json.loads((OUT / "mature-census.json").read_text())
    new = json.loads((OUT / "new-census.json").read_text())
    result = {
        "schema": "recurrence_census.scientific_synthesis.v1",
        "status": "candidate_cpu_synthesis",
        "estimand": "image-unit descriptive distribution in the 145-image mature package and the deterministic 128-image eligible seed19 cohort",
        "interpretation_limits": [
            "Category, region, size and annotation fields are saved-output proxies; annotation density based on duplicated output is endogenous.",
            "Repeat-row counts and all-pairs edges are separate; a run of length r contributes r choose 2 pair edges.",
            "Invalid geometry and malformed rows remain numerical outcomes; they are not physical false positives.",
            "Rare-category intervals are image-unit bootstrap intervals and should not support claims that only a few categories recur.",
            "Mature tied/untied is a package comparison, and the prospective cohort estimates its eligible processed population rather than COCO generally.",
        ],
        "strata": {
            "mature": {condition: _condition(mature, condition) for condition in mature["cells"]},
            "prospective_seed19": {condition: _condition(new, condition) for condition in new["cells"]},
        },
        "source_artifacts": {"mature": str(OUT / "mature-census.json"), "prospective": str(OUT / "new-census.json"), "panel": str(OUT / "shared-panel.json")},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "output": str(args.output), "mature_conditions": len(result["strata"]["mature"]), "new_conditions": len(result["strata"]["prospective_seed19"])}, indent=2))


if __name__ == "__main__":
    main()
