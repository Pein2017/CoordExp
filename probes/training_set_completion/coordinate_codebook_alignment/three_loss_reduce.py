"""Saved-cell reduction for the fixed three-loss 16-epoch comparison."""
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
    LIMITS,
    _aggregate,
    _cases,
    _cell_path,
    _file_binding,
    _flags,
    _format_summary,
    _json,
    _paired,
    _retained_membership,
    _stratum,
)

EXPECTED = {
    "source": 1280,
    "ce_only_epoch16": 1280,
    "three_loss_epoch8": 96,
    "three_loss_epoch16": 1280,
}
REUSED = {"source": 1280, "ce_only_epoch16": 96, "three_loss_epoch8": 0, "three_loss_epoch16": 0}
FINAL = "three_loss_epoch16"
CE = "ce_only_epoch16"


def _group(spec: Mapping[str, Any], row: Mapping[str, Any], retained: set[str]) -> str:
    if str(spec.get("panel")) != "train":
        return "validation256"
    return "retained32" if str(spec["row_id"]) in retained else "additions992"


def _endpoint(spec: Mapping[str, Any]) -> str:
    if str(spec["condition"]) == "three_loss_epoch8":
        return "sentinel8"
    return "train1024" if str(spec.get("panel")) == "train" else "validation256"


def _objective(cell: Mapping[str, Any]) -> dict[str, Any]:
    for key in ("training_objective", "objective", "losses"):
        value = cell.get(key)
        if isinstance(value, Mapping):
            return {"status": "present", "source_key": key, "metrics": dict(value)}
    return {"status": "missing", "reason": "native_cell_has_no_composite_training_objective"}


def _reduce(manifest: Mapping[str, Any], analytical: Mapping[str, Any], run_root: Path) -> dict[str, Any]:
    specs = list(analytical.get("specs", []))
    if len(specs) != 3936:
        raise ValueError(f"three-loss analytical denominator must contain 3936 cells, got {len(specs)}")
    keys = [(str(s["row_id"]), str(s["condition"])) for s in specs]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate three-loss analytical cell identity")
    counts = Counter(str(s["condition"]) for s in specs)
    if dict(counts) != EXPECTED:
        raise ValueError(f"three-loss condition counts differ: {dict(counts)}")
    panels = Counter((str(s["condition"]), str(s.get("panel", ""))) for s in specs)
    expected_panels = {
        ("source", "train"): 1024, ("source", "validation"): 256,
        ("ce_only_epoch16", "train"): 1024, ("ce_only_epoch16", "validation"): 256,
        ("three_loss_epoch8", "train"): 96,
        ("three_loss_epoch16", "train"): 1024, ("three_loss_epoch16", "validation"): 256,
    }
    if dict(panels) != expected_panels:
        raise ValueError(f"three-loss panel counts differ: {dict(panels)}")
    reuse_counts = Counter(str(s["condition"]) for s in specs if isinstance(s.get("reuse"), Mapping))
    normalized_reuse = {condition: reuse_counts.get(condition, 0) for condition in EXPECTED}
    if normalized_reuse != REUSED:
        raise ValueError(f"three-loss reuse counts differ: {normalized_reuse}")
    cases = _cases(specs)
    retained, retained_method = _retained_membership(manifest, analytical, specs, cases)
    loaded: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    mutations: list[dict[str, Any]] = []
    input_bindings: list[dict[str, Any]] = []
    for spec in specs:
        row_id, condition = str(spec["row_id"]), str(spec["condition"])
        path, source_kind = _cell_path(run_root, spec)
        binding = _file_binding(path)
        binding.update({"cell_key": spec["cell_key"], "row_id": row_id, "condition": condition, "source_kind": source_kind})
        if isinstance(spec.get("reuse"), Mapping):
            binding["expected_sha256"] = spec["reuse"].get("sha256")
        input_bindings.append(binding)
        if not path.exists():
            missing.append({"cell_key": spec["cell_key"], "row_id": row_id, "condition": condition, "path": str(path)})
            continue
        cell = _json(path)
        reasons: list[str] = []
        if isinstance(spec.get("reuse"), Mapping) and binding["expected_sha256"] and binding["expected_sha256"] != binding["sha256"]:
            reasons.append("reuse_sha256_mismatch")
        observed_id = str(cell.get("case", {}).get("row_id", cell.get("row_id", "")))
        allowed_conditions = {condition}
        # The historical CE-only evaluator wrote its epoch-16 cells under the
        # predecessor condition name; the analytical packet explicitly maps
        # those saved cells into ce_only_epoch16.
        if condition == "ce_only_epoch16" and source_kind == "reused_source":
            allowed_conditions.add("epoch16")
        if str(cell.get("condition", "")) not in allowed_conditions or observed_id != row_id:
            reasons.append("cell_identity_mismatch")
        if reasons:
            mutations.append({"cell_key": spec["cell_key"], "row_id": row_id, "condition": condition, "path": str(path), "observed_condition": cell.get("condition"), "observed_row_id": observed_id, "reason": reasons})
            continue
        reduced = reduce_cell(cell)
        reduced.update({
            "row_id": row_id, "condition": condition, "panel": spec.get("panel"),
            "endpoint": _endpoint(spec), "group": _group(spec, cases[row_id], retained),
            "stratum": _stratum(cases[row_id], spec), "source_kind": source_kind,
            "cell_key": spec["cell_key"], "cell_path": str(path),
            "format_summary": _format_summary(cell, reduced),
            "training_objective": _objective(cell),
        })
        loaded.append(reduced)
    loaded_keys = {(str(r["row_id"]), str(r["condition"])) for r in loaded}
    holds: list[dict[str, Any]] = []
    for spec in specs:
        key = (str(spec["row_id"]), str(spec["condition"]))
        if key in loaded_keys:
            continue
        path, source_kind = _cell_path(run_root, spec)
        reason = "missing_saved_cell" if not path.exists() else "mutated_saved_cell"
        holds.append({"status": "HOLD", "row_id": key[0], "condition": key[1], "panel": spec.get("panel"), "endpoint": _endpoint(spec), "group": _group(spec, cases[key[0]], retained), "stratum": _stratum(cases[key[0]], spec), "source_kind": source_kind, "cell_key": spec["cell_key"], "cell_path": str(path), "reason": reason, "teacher": {"status": "missing", "reason": reason, "metrics": None}, "clean_known_positive_proxy": {"eligible": False, "checks": {}}})
    by_key = {(str(r["row_id"]), str(r["condition"])): r for r in loaded}
    aggregate: dict[str, Any] = {}
    for condition in EXPECTED:
        cond_specs = [s for s in specs if str(s["condition"]) == condition]
        cond_rows = [by_key[k] for k in ((str(s["row_id"]), condition) for s in cond_specs) if k in by_key]
        item: dict[str, Any] = {"cells": _aggregate(cond_rows, len(cond_specs)), "by_endpoint": {}, "by_group": {}, "by_stratum": {}}
        for field, values in (("endpoint", {_endpoint(s) for s in cond_specs}), ("group", {_group(s, cases[str(s["row_id"])], retained) for s in cond_specs}), ("stratum", {_stratum(cases[str(s["row_id"])], s) for s in cond_specs})):
            key_name = f"by_{field}"
            for value in sorted(values):
                selected = [r for r in cond_rows if r[field] == value]
                expected = sum(1 for s in cond_specs if ( _endpoint(s) if field == "endpoint" else _group(s, cases[str(s["row_id"])], retained) if field == "group" else _stratum(cases[str(s["row_id"])], s)) == value)
                item[key_name][value] = _aggregate(selected, expected)
        item["training_objective_images"] = sum(r.get("training_objective", {}).get("status") == "present" for r in cond_rows)
        aggregate[condition] = item
    comparisons: list[dict[str, Any]] = []
    for condition in (CE, "three_loss_epoch8", FINAL):
        for key, candidate in sorted(by_key.items()):
            if key[1] == condition:
                base = by_key.get((key[0], "source"))
                if base and base.get("status") == "complete" and candidate.get("status") == "complete":
                    comparisons.append(_paired(base, candidate) | {"comparison": "source_vs_condition", "panel": candidate["panel"], "endpoint": candidate["endpoint"], "group": candidate["group"], "stratum": candidate["stratum"]})
    triplet_comparisons: list[dict[str, Any]] = []
    for row_id in sorted({str(s["row_id"]) for s in specs if str(s["condition"]) == FINAL}):
        source, ce, final = (by_key.get((row_id, c)) for c in ("source", CE, FINAL))
        if all(r and r.get("status") == "complete" for r in (source, ce, final)):
            entry = {"row_id": row_id, "panel": final["panel"], "endpoint": final["endpoint"], "group": final["group"], "stratum": final["stratum"], "source_vs_final": _paired(source, final), "ce_only_vs_final": _paired(ce, final)}
            for metric in ("bad", "cap", "owner_recurrent", "severe"):
                sf, cf, ff = _flags(source)[metric], _flags(ce)[metric], _flags(final)[metric]
                entry.setdefault("ce_relative_failures", {})[metric] = {"repaired": cf and not ff, "persistent": cf and ff, "newly_introduced": (not cf) and ff}
            triplet_comparisons.append(entry)
    guardrails: dict[str, Any] = {}
    for endpoint in ("train1024", "validation256"):
        panel = "train" if endpoint == "train1024" else "validation"
        expected = 1024 if panel == "train" else 256
        pairs = [x for x in triplet_comparisons if x["endpoint"] == endpoint]
        entry: dict[str, Any] = {"paired_expected": expected, "paired_complete": len(pairs), "paired_denominator_complete": len(pairs) == expected, "eligibility": "HOLD" if len(pairs) != expected else None, "metrics": {}}
        for metric, limit in LIMITS[panel].items():
            count = sum(not _flags(by_key[(x["row_id"], "source")])[metric] and _flags(by_key[(x["row_id"], FINAL)])[metric] for x in pairs)
            entry["metrics"][metric] = {"source_negative_final_positive": int(count), "prospective_limit": limit}
        if entry["eligibility"] is None:
            entry["eligibility"] = "pass" if all(v["source_negative_final_positive"] <= v["prospective_limit"] for v in entry["metrics"].values()) else "fail"
        guardrails[endpoint] = entry
    status = "HOLD" if missing or mutations or any(r.get("status") != "complete" for r in loaded) else "complete"
    return {
        "schema": "coordinate_codebook_alignment.three_loss_reduction.v1", "status": status, "model_calls": 0,
        "denominator": {"analytical_cells": len(specs), "distinct_cells": len(specs), "new_cells": len(specs) - sum(REUSED.values()), "reused_cells": sum(REUSED.values()), "expected_by_condition": EXPECTED, "reused_by_condition": REUSED, "missing_cells": len(missing), "mutation_cells": len(mutations)},
        "retained_membership": {"method": retained_method, "count": len(retained), "row_ids": sorted(retained)},
        "endpoints": {"train1024": 1024, "validation256": 256, "sentinel8": 96},
        "aggregate": aggregate, "source_comparisons": comparisons, "triplet_comparisons": triplet_comparisons,
        "guardrails": {"source_vs_final": guardrails, "ce_only_relative": "per-image repaired/persistent/newly_introduced in triplet_comparisons; no threshold applied"},
        "missing_cells": missing, "mutation_cells": mutations, "per_image": sorted(loaded + holds, key=lambda r: (str(r["condition"]), str(r["row_id"]))),
        "input_bindings": input_bindings, "unknown_policy": "annotation-unmatched predictions remain UNKNOWN; no physical-false conversion",
        "teacher_ce_note": "teacher metrics are ordinary full-vocabulary CE and remain separate from any saved composite training objective",
        "source_manifest": str(manifest.get("schema", "")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--analytical", type=Path)
    args = parser.parse_args()
    analytical = _json(args.analytical or (args.manifest.resolve().parent / "analytical-cells.json"))
    result = _reduce(_json(args.manifest), analytical, args.run_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "analytical_cells": result["denominator"]["analytical_cells"], "missing": result["denominator"]["missing_cells"]}, sort_keys=True))


if __name__ == "__main__":
    main()
