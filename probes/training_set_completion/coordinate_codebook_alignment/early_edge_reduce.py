"""Saved-cell reducer for the frozen early-edge comparison."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from probes.training_set_completion.coordinate_codebook_alignment.reduce import reduce_cell
from probes.training_set_completion.coordinate_codebook_alignment.scale_reduce import (
    _aggregate,
    _cases,
    _cell_path,
    _file_binding,
    _flags,
    _format_summary,
    _json,
    _paired,
    _stratum,
)

EXPECTED = {"source": 1280, "three_loss_epoch16": 1280, "three_loss_epoch8": 96,
            "early_edge_epoch16": 1280, "early_edge_epoch8": 96}
REUSED = {"source": 1280, "three_loss_epoch16": 1280, "three_loss_epoch8": 96,
          "early_edge_epoch16": 0, "early_edge_epoch8": 0}
FAILURES = ("bad", "cap", "owner_recurrent", "severe")
LIMITS = {"train": {"bad": 51, "cap": 10, "owner_recurrent": 51, "severe": 10},
          "validation": {"bad": 12, "cap": 2, "owner_recurrent": 12, "severe": 2}}


def _retained32(manifest: Mapping[str, Any], analytical: Mapping[str, Any], specs: list[Mapping[str, Any]], cases: Mapping[str, Mapping[str, Any]]) -> set[str]:
    raw = analytical.get("retained32_row_ids", manifest.get("retained32_row_ids"))
    if not isinstance(raw, list) or len(raw) != 32 or len({str(x) for x in raw}) != 32:
        raise ValueError("explicit retained32_row_ids must contain exactly 32 unique identities")
    retained = {str(x) for x in raw}
    train_source = {str(s["row_id"]) for s in specs if s["condition"] == "source" and s.get("panel") == "train"}
    if not retained <= train_source or not retained <= cases.keys():
        raise ValueError("retained32_row_ids contains an unknown or non-training source identity")
    return retained


def _group(spec: Mapping[str, Any], retained: set[str]) -> str:
    if spec.get("panel") != "train":
        return "validation256"
    return "retained32" if str(spec["row_id"]) in retained else "additions992"


def _endpoint(spec: Mapping[str, Any]) -> str:
    condition, panel = spec["condition"], spec.get("panel")
    if condition in {"three_loss_epoch8", "early_edge_epoch8"}:
        return "sentinel96"
    return "train1024" if panel == "train" else "validation256"


def _transition(base: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    before, after = _flags(base), _flags(candidate)
    return {name: {"baseline": before[name], "candidate": after[name],
                   "repaired": before[name] and not after[name],
                   "persistent": before[name] and after[name],
                   "newly_introduced": not before[name] and after[name]}
            for name in FAILURES}


def reduce_early_edge(manifest: Mapping[str, Any], analytical: Mapping[str, Any], run_root: Path) -> dict[str, Any]:
    specs = list(analytical.get("specs", []))
    if len(specs) != 4032:
        raise ValueError(f"early-edge analytical denominator must contain 4032 cells, got {len(specs)}")
    keys = [(str(s["row_id"]), str(s["condition"])) for s in specs]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate early-edge analytical cell identity")
    counts = Counter(str(s["condition"]) for s in specs)
    if {name: counts.get(name, 0) for name in EXPECTED} != EXPECTED or set(counts) != set(EXPECTED):
        raise ValueError(f"early-edge condition counts differ: {dict(counts)}")
    panels = Counter((str(s["condition"]), str(s.get("panel", ""))) for s in specs)
    expected_panels = {("source", "train"): 1024, ("source", "validation"): 256,
                       ("three_loss_epoch16", "train"): 1024,
                       ("three_loss_epoch16", "validation"): 256,
                       ("three_loss_epoch8", "train"): 96,
                       ("early_edge_epoch16", "train"): 1024,
                       ("early_edge_epoch16", "validation"): 256,
                       ("early_edge_epoch8", "train"): 96}
    if dict(panels) != expected_panels:
        raise ValueError(f"early-edge panel counts differ: {dict(panels)}")
    reuse_counts = Counter(str(s["condition"]) for s in specs if isinstance(s.get("reuse"), Mapping))
    if {name: reuse_counts.get(name, 0) for name in EXPECTED} != REUSED:
        raise ValueError("frozen early-edge reuse counts differ")

    cases = _cases(specs)
    retained = _retained32(manifest, analytical, specs, cases)
    loaded: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    mutations: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = []
    for spec in specs:
        row_id, condition = str(spec["row_id"]), str(spec["condition"])
        path, source_kind = _cell_path(run_root, spec)
        binding = _file_binding(path)
        binding.update(cell_key=spec["cell_key"], row_id=row_id, condition=condition, source_kind=source_kind)
        reuse = spec.get("reuse")
        expected_hash = reuse.get("sha256") if isinstance(reuse, Mapping) else None
        if isinstance(reuse, Mapping):
            binding["expected_sha256"] = expected_hash
        bindings.append(binding)
        if not path.exists():
            missing.append({"cell_key": spec["cell_key"], "row_id": row_id, "condition": condition, "path": str(path)})
            continue
        reasons = []
        if isinstance(reuse, Mapping) and (not isinstance(expected_hash, str) or not expected_hash or expected_hash != binding["sha256"]):
            reasons.append("reuse_sha256_missing_or_mismatch")
        if reasons:
            mutations.append({"cell_key": spec["cell_key"], "row_id": row_id, "condition": condition,
                              "path": str(path), "observed_condition": None,
                              "observed_row_id": None, "reason": reasons})
            continue
        cell = _json(path)
        actual_condition = str(cell.get("condition", ""))
        allowed = {condition}
        actual_row = str(cell.get("case", {}).get("row_id", cell.get("row_id", "")))
        if actual_condition not in allowed or actual_row != row_id:
            reasons.append("cell_identity_mismatch")
        if reasons:
            mutations.append({"cell_key": spec["cell_key"], "row_id": row_id, "condition": condition,
                              "path": str(path), "observed_condition": actual_condition,
                              "observed_row_id": actual_row, "reason": reasons})
            continue
        reduced = reduce_cell(cell)
        reduced.update(row_id=row_id, condition=condition, panel=spec.get("panel"),
                       endpoint=_endpoint(spec), group=_group(spec, retained),
                       stratum=_stratum(cases[row_id], spec), source_kind=source_kind,
                       cell_key=spec["cell_key"], cell_path=str(path),
                       format_summary=_format_summary(cell, reduced))
        loaded.append(reduced)

    by_key = {(r["row_id"], r["condition"]): r for r in loaded}
    spec_by_key = {(str(s["row_id"]), str(s["condition"])): s for s in specs}
    holds = []
    for spec in specs:
        key = (str(spec["row_id"]), str(spec["condition"]))
        if key in by_key:
            continue
        path, source_kind = _cell_path(run_root, spec)
        reason = "missing_saved_cell" if not path.exists() else "mutated_saved_cell"
        holds.append({"status": "HOLD", "row_id": key[0], "condition": key[1], "panel": spec.get("panel"),
                      "endpoint": _endpoint(spec), "group": _group(spec, retained),
                      "stratum": _stratum(cases[key[0]], spec), "source_kind": source_kind,
                      "cell_key": spec["cell_key"], "cell_path": str(path), "reason": reason,
                      "teacher": {"status": "missing", "reason": reason, "metrics": None},
                      "clean_known_positive_proxy": {"eligible": False, "checks": {}}})

    aggregate = {}
    for condition in EXPECTED:
        cond_specs = [s for s in specs if s["condition"] == condition]
        rows = [by_key[(str(s["row_id"]), condition)] for s in cond_specs if (str(s["row_id"]), condition) in by_key]
        aggregate[condition] = _aggregate(rows, len(cond_specs))

    pairs = []
    transition_counts: dict[str, dict[str, Any]] = {}
    for candidate_condition, baseline_condition in (("early_edge_epoch16", "source"),
                                                     ("early_edge_epoch16", "three_loss_epoch16"),
                                                     ("early_edge_epoch8", "source"),
                                                     ("early_edge_epoch8", "three_loss_epoch8")):
        expected_ids = {str(s["row_id"]) for s in specs if s["condition"] == candidate_condition}
        comparison_rows = []
        for row_id in sorted(expected_ids):
            base, candidate = by_key.get((row_id, baseline_condition)), by_key.get((row_id, candidate_condition))
            if not (base and candidate and base.get("status") == candidate.get("status") == "complete"):
                continue
            spec = spec_by_key[(row_id, candidate_condition)]
            transitions = _transition(base, candidate)
            entry = {"row_id": row_id, "baseline_condition": baseline_condition,
                     "condition": candidate_condition, "panel": spec["panel"],
                     "endpoint": _endpoint(spec), "group": _group(spec, retained),
                     "stratum": _stratum(cases[row_id], spec), "transitions": transitions,
                     "per_image_severity": {"baseline": _flags(base), "candidate": _flags(candidate)},
                     "baseline_repeat_proxy": base.get("repeat_proxy", {}),
                     "candidate_repeat_proxy": candidate.get("repeat_proxy", {}),
                     "iou50_class_consistent": _paired(base, candidate)["iou50_class_consistent"],
                     "iou80_class_consistent": _paired(base, candidate)["iou80_class_consistent"]}
            pairs.append(entry)
            comparison_rows.append(entry)
        name = f"{candidate_condition}_vs_{baseline_condition}"
        buckets: dict[str, list[Mapping[str, Any]]] = {}
        for group in ("all", "ordinary", "dense", "middle", "unknown", "retained32", "additions992"):
            buckets[group] = [r for r in comparison_rows if group == "all" or r["stratum"] == group or r["group"] == group]
        transition_counts[name] = {group: {metric: {
            "paired_complete": len(rows),
            "repaired": sum(r["transitions"][metric]["repaired"] for r in rows),
            "persistent": sum(r["transitions"][metric]["persistent"] for r in rows),
            "newly_introduced": sum(r["transitions"][metric]["newly_introduced"] for r in rows),
        } for metric in FAILURES} for group, rows in buckets.items()}

    expected_pairs = {"early_edge_epoch16_vs_source": 1280,
                      "early_edge_epoch16_vs_three_loss_epoch16": 1280,
                      "early_edge_epoch8_vs_source": 96,
                      "early_edge_epoch8_vs_three_loss_epoch8": 96}
    paired_counts = Counter(f"{r['condition']}_vs_{r['baseline_condition']}" for r in pairs)
    incomplete = any(paired_counts.get(name, 0) != expected for name, expected in expected_pairs.items())
    source_guardrails = {}
    for endpoint, panel in (("train1024", "train"), ("validation256", "validation")):
        selected = [r for r in pairs if r["condition"] == "early_edge_epoch16" and
                    r["baseline_condition"] == "source" and r["panel"] == panel]
        expected = 1024 if panel == "train" else 256
        entry = {"paired_expected": expected, "paired_complete": len(selected),
                 "eligibility": "HOLD" if len(selected) != expected else None, "metrics": {}}
        for metric, limit in LIMITS[panel].items():
            count = sum(r["transitions"][metric]["newly_introduced"] for r in selected)
            entry["metrics"][metric] = {"source_negative_early_positive": count,
                                        "prospective_limit": limit}
        if entry["eligibility"] is None:
            entry["eligibility"] = "pass" if all(v["source_negative_early_positive"] <= v["prospective_limit"]
                                                   for v in entry["metrics"].values()) else "fail"
        source_guardrails[endpoint] = entry
    status = "HOLD" if missing or mutations or holds or incomplete or any(r.get("status") != "complete" for r in loaded) else "complete"
    return {"schema": "coordinate_codebook_alignment.early_edge_reduction.v1", "status": status,
            "model_calls": 0, "denominator": {"analytical_cells": 4032, "distinct_cells": 4032,
            "new_cells": 1376, "reused_cells": 2656, "expected_by_condition": EXPECTED,
            "reused_by_condition": REUSED, "missing_cells": len(missing), "mutation_cells": len(mutations)},
            "retained_membership": {"method": "explicit_retained32_row_ids", "count": 32, "row_ids": sorted(retained)},
            "aggregates": aggregate, "transition_counts": transition_counts, "paired_complete_by_comparison": dict(paired_counts),
            "guardrails": {"source_vs_early_edge_epoch16": source_guardrails,
                           "late_relative": "per-image transitions reported; no prospective threshold applied"},
            "per_image_comparisons": pairs, "missing_cells": missing, "mutation_cells": mutations,
            "per_image": sorted(loaded + holds, key=lambda r: (str(r["condition"]), str(r["row_id"]))),
            "input_bindings": bindings,
            "unknown_policy": "annotation-unmatched predictions remain UNKNOWN; no physical-false conversion",
            "incomplete_comparison_policy": "HOLD; unmatched cells are not failures or physical negatives",
            "source_manifest": str(manifest.get("schema", ""))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--analytical", type=Path)
    args = parser.parse_args()
    analytical = _json(args.analytical or (args.manifest.resolve().parent / "analytical-cells.json"))
    result = reduce_early_edge(_json(args.manifest), analytical, args.run_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "analytical_cells": result["denominator"]["analytical_cells"],
                      "missing": result["denominator"]["missing_cells"]}, sort_keys=True))


if __name__ == "__main__":
    main()
