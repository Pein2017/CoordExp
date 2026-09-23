"""Saved-cell reduction for the frozen 1024/256 scale packet.

This is intentionally a thin scale-specific view over ``reduce.py``: cell
matching, owner proxies, teacher metrics, and geometry stay in that module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from probes.training_set_completion.coordinate_codebook_alignment.reduce import reduce_cell, _teacher_aggregate


EXPECTED = {"source": 1280, "epoch32": 1280, "epoch4": 96, "epoch16": 96}
LIMITS = {
    "train": {"bad": 51, "cap": 10, "owner_recurrent": 51, "severe": 10},
    "validation": {"bad": 12, "cap": 2, "owner_recurrent": 12, "severe": 2},
}


def _json(path: Path) -> Any:
    return json.loads(path.read_text())


def _file_binding(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "sha256": None, "size_bytes": None}
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "sha256": digest, "size_bytes": path.stat().st_size}


def _cell_path(run_root: Path, spec: Mapping[str, Any]) -> tuple[Path, str]:
    reuse = spec.get("reuse")
    if isinstance(reuse, Mapping):
        return Path(str(reuse["path"])), "reused_source"
    condition = str(spec["condition"])
    return run_root / "production" / condition / "cells" / f"{spec['cell_key']}.json", "new"


def _cases(specs: list[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    paths = {str(s["dataset"]) for s in specs}
    result: dict[str, Mapping[str, Any]] = {}
    for raw_path in sorted(paths):
        for line in Path(raw_path).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row.get("_admission", {}).get("row_id", ""))
            if not key:
                raise ValueError(f"admission row has no row_id: {raw_path}")
            if key in result:
                raise ValueError(f"duplicate admitted image identity: {key}")
            result[key] = row
    return result


def _retained_membership(manifest: Mapping[str, Any], analytical: Mapping[str, Any], specs: list[Mapping[str, Any]], cases: Mapping[str, Mapping[str, Any]]) -> tuple[set[str], str]:
    """Resolve the original 32 identities independently of reuse provenance."""
    raw = analytical.get("retained32_row_ids", manifest.get("retained32_row_ids"))
    method = "explicit_retained32_row_ids"
    if raw is None:
        method = "trusted_admission_cohort_fit_prefix"
        raw = [
            str(s["row_id"])
            for s in specs
            if str(s.get("condition")) == "source"
            and str(s.get("panel")) == "train"
            and str(cases[str(s["row_id"])]
                    .get("_admission", {}).get("cohort", "")).startswith("fit_")
        ]
    if not isinstance(raw, list) or len(raw) != 32 or len({str(item) for item in raw}) != 32:
        raise ValueError("retained32 membership must contain exactly 32 unique row IDs")
    retained = {str(item) for item in raw}
    train_source = {str(s["row_id"]) for s in specs if str(s.get("condition")) == "source" and str(s.get("panel")) == "train"}
    if not retained <= train_source:
        raise ValueError("retained32 membership contains an unknown or non-training source identity")
    if any(row_id not in cases for row_id in retained):
        raise ValueError("retained32 membership contains an admitted-row identity absent from input cases")
    return retained, method


def _stratum(row: Mapping[str, Any], spec: Mapping[str, Any]) -> str:
    density = str(row.get("_admission", {}).get("stratum", {}).get("density", spec.get("density", "unknown")))
    return density if density in {"ordinary", "dense", "middle"} else "unknown"


def _group(spec: Mapping[str, Any], row: Mapping[str, Any], retained: set[str]) -> str:
    panel = str(spec.get("panel", "unknown"))
    row_id = str(spec["row_id"])
    if panel == "train":
        return "retained32" if row_id in retained else "additions992"
    return "validation256"


def _endpoint(spec: Mapping[str, Any]) -> str:
    condition = str(spec["condition"])
    panel = str(spec.get("panel", "unknown"))
    if condition in {"epoch4", "epoch16"}:
        return "sentinel96"
    return "train1024" if panel == "train" else "validation256"


def _format_summary(cell: Mapping[str, Any], reduced: Mapping[str, Any]) -> dict[str, Any]:
    parser = cell.get("parser", {})
    generation = cell.get("generation", {})
    dropped = parser.get("dropped_predictions", [])
    spans = []
    for item in dropped if isinstance(dropped, list) else []:
        if isinstance(item, Mapping):
            spans.append({k: item[k] for k in ("reason", "char_start", "char_end", "raw_span_text") if k in item})
    text = generation.get("text")
    tokens = generation.get("token_ids")
    malformed = sum(str(item.get("reason", "")) == "malformed_object_span" for item in dropped if isinstance(item, Mapping))
    return {
        "total_dropped_span_count": len(spans),
        "malformed_object_span_count": malformed,
        "malformed_spans": spans,
        "generated_characters": len(text) if isinstance(text, str) else None,
        "generated_tokens": len(tokens) if isinstance(tokens, list) else None,
        "parser_dropped": int(reduced.get("parser_dropped", 0)),
        "invalid_geometry": int(reduced.get("invalid_geometry", 0)),
        "stop_reason": reduced.get("stop_reason"),
        "cap": reduced.get("cap"),
        "owner_max_run_iou50": int(reduced.get("repeat_proxy", {}).get("owner_max_run_iou50", 0)),
    }


def _flags(row: Mapping[str, Any]) -> dict[str, bool]:
    if row.get("status") != "complete":
        return {"bad": False, "cap": False, "owner_recurrent": False, "severe": False}
    stop = row.get("stop_reason")
    repeat = row.get("repeat_proxy", {})
    cap = stop in {"length", "cap", "max_new_tokens"}
    return {
        "bad": bool(row.get("parser_dropped", 0) or row.get("invalid_geometry", 0) or stop != "im_end"),
        "cap": cap,
        "owner_recurrent": int(repeat.get("owner_revisit_count_iou50", 0)) > 0,
        "severe": int(repeat.get("owner_max_run_iou50", 0)) >= 5,
    }


def _paired(base: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"row_id": candidate["row_id"], "condition": candidate["condition"]}
    result["baseline_flags"] = _flags(base)
    result["condition_flags"] = _flags(candidate)
    result["baseline_format_summary"] = base.get("format_summary", {})
    result["condition_format_summary"] = candidate.get("format_summary", {})
    for metric in ("iou50_class_consistent", "iou80_class_consistent"):
        old = set(base.get(metric, {}).get("covered_owner_ids", []))
        new = set(candidate.get(metric, {}).get("covered_owner_ids", []))
        result[metric] = {
            "baseline_coverage": base.get(metric, {}).get("coverage"),
            "condition_coverage": candidate.get(metric, {}).get("coverage"),
            "gained_owner_ids": sorted(new - old),
            "lost_owner_ids": sorted(old - new),
            "equal": new == old,
        }
    return result


def _aggregate(rows: list[Mapping[str, Any]], expected: int) -> dict[str, Any]:
    complete = [r for r in rows if r.get("status") == "complete"]
    target = sum(int(r.get("target_count", 0)) for r in complete)
    def matched(metric: str) -> int:
        return sum(int(r.get(metric, {}).get("matched_count", 0)) for r in complete)
    def coverage(metric: str) -> float | None:
        return matched(metric) / target if target else None
    repeats = [r.get("repeat_proxy", {}) for r in complete]
    formats = [r.get("format_summary", {}) for r in complete]
    tokens = [int(item["generated_tokens"]) for item in formats if item.get("generated_tokens") is not None]
    characters = [int(item["generated_characters"]) for item in formats if item.get("generated_characters") is not None]
    return {
        "cells_expected": expected,
        "cells_complete": len(complete),
        "cells_hold": expected - len(complete),
        "matched_target_denominators": {
            "iou50_class_consistent": {"matched": matched("iou50_class_consistent"), "target": target},
            "iou80_class_consistent": {"matched": matched("iou80_class_consistent"), "target": target},
        },
        "coverage_iou50_class_consistent": coverage("iou50_class_consistent"),
        "coverage_iou80_class_consistent": coverage("iou80_class_consistent"),
        "clean_known_positive_images": sum(bool(r.get("clean_known_positive_proxy", {}).get("eligible")) for r in complete),
        "invalid_geometry": sum(int(r.get("invalid_geometry", 0)) for r in complete),
        "parser_dropped": sum(int(r.get("parser_dropped", 0)) for r in complete),
        "format_burden": {
            "images_with_dropped_spans": sum(int(f.get("total_dropped_span_count", 0)) > 0 for f in formats),
            "total_dropped_spans": sum(int(f.get("total_dropped_span_count", 0)) for f in formats),
            "images_with_malformed_object_spans": sum(int(f.get("malformed_object_span_count", 0)) > 0 for f in formats),
            "malformed_object_spans": sum(int(f.get("malformed_object_span_count", 0)) for f in formats),
        },
        "unknown_predictions": sum(len(r.get("unknown_prediction_indices", [])) for r in complete),
        "natural_im_end": sum(r.get("stop_reason") == "im_end" for r in complete),
        "cap_cells": sum(_flags(r)["cap"] for r in complete),
        "repeat_burden": {
            "exact_row_recurrent_images": sum(int(p.get("exact_row_revisit_count", 0)) > 0 for p in repeats),
            "exact_row_revisits": sum(int(p.get("exact_row_revisit_count", 0)) for p in repeats),
            "owner_recurrent_images": sum(int(p.get("owner_revisit_count_iou50", 0)) > 0 for p in repeats),
            "owner_revisits_iou50": sum(int(p.get("owner_revisit_count_iou50", 0)) for p in repeats),
            "max_owner_run_iou50": max((int(p.get("owner_max_run_iou50", 0)) for p in repeats), default=0),
        },
        "generated_length": {
            "images_with_tokens": len(tokens), "token_total": sum(tokens), "token_mean": sum(tokens) / len(tokens) if tokens else None, "token_max": max(tokens, default=None),
            "images_with_characters": len(characters), "character_total": sum(characters), "character_mean": sum(characters) / len(characters) if characters else None, "character_max": max(characters, default=None),
        },
        "teacher": _teacher_aggregate(complete, expected),
    }


def reduce_scale(manifest: Mapping[str, Any], analytical: Mapping[str, Any], run_root: Path) -> dict[str, Any]:
    specs = list(analytical.get("specs", []))
    if len(specs) != 2752:
        raise ValueError(f"frozen analytical denominator must contain 2752 cells, got {len(specs)}")
    keys = [(str(s["row_id"]), str(s["condition"])) for s in specs]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate frozen analytical cell identity")
    condition_counts = Counter(str(s["condition"]) for s in specs)
    if dict(condition_counts) != EXPECTED:
        raise ValueError(f"frozen analytical condition counts differ: {dict(condition_counts)}")
    panel_counts = Counter((str(s["condition"]), str(s.get("panel", ""))) for s in specs)
    expected_panels = {("source", "train"): 1024, ("source", "validation"): 256, ("epoch32", "train"): 1024, ("epoch32", "validation"): 256, ("epoch4", "train"): 96, ("epoch16", "train"): 96}
    if dict(panel_counts) != expected_panels:
        raise ValueError(f"frozen analytical panel counts differ: {dict(panel_counts)}")
    cases = _cases(specs)
    retained, retained_method = _retained_membership(manifest, analytical, specs, cases)
    loaded: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    mutations: list[dict[str, Any]] = []
    sources: dict[str, str] = {}
    input_bindings: list[dict[str, Any]] = []
    for spec in specs:
        row_id, condition = str(spec["row_id"]), str(spec["condition"])
        path, source_kind = _cell_path(run_root, spec)
        key = f"{condition}:{row_id}"
        sources[key] = str(path)
        binding = _file_binding(path)
        binding.update({"cell_key": spec["cell_key"], "row_id": row_id, "condition": condition, "source_kind": source_kind})
        if isinstance(spec.get("reuse"), Mapping):
            binding["expected_sha256"] = spec["reuse"].get("sha256")
        input_bindings.append(binding)
        if not path.exists():
            missing.append({"cell_key": spec["cell_key"], "row_id": row_id, "condition": condition, "path": str(path)})
            continue
        cell = _json(path)
        reasons = []
        if isinstance(spec.get("reuse"), Mapping) and binding["expected_sha256"] and binding["expected_sha256"] != binding["sha256"]:
            reasons.append("reuse_sha256_mismatch")
        if str(cell.get("condition", "")) != condition or str(cell.get("case", {}).get("row_id", cell.get("row_id", ""))) != row_id:
            reasons.append("cell_identity_mismatch")
        if reasons:
            mutations.append({"cell_key": spec["cell_key"], "row_id": row_id, "condition": condition, "path": str(path), "observed_condition": cell.get("condition"), "observed_row_id": cell.get("case", {}).get("row_id", cell.get("row_id")), "reason": reasons})
            continue
        reduced = reduce_cell(cell)
        reduced.update({"row_id": row_id, "condition": condition, "panel": spec.get("panel"), "endpoint": _endpoint(spec), "group": _group(spec, cases[row_id], retained), "stratum": _stratum(cases[row_id], spec), "source_kind": source_kind, "cell_key": spec["cell_key"], "cell_path": str(path), "format_summary": _format_summary(cell, reduced)})
        loaded.append(reduced)
    hold_rows: list[dict[str, Any]] = []
    loaded_keys = {(str(item["row_id"]), str(item["condition"])) for item in loaded}
    for spec in specs:
        row_id, condition = str(spec["row_id"]), str(spec["condition"])
        key = (row_id, condition)
        if key in loaded_keys:
            continue
        path, source_kind = _cell_path(run_root, spec)
        if path.exists() and not any(item.get("row_id") == row_id and item.get("condition") == condition for item in loaded):
            # Identity-mismatched cells are represented below as HOLD rows.
            reason = "mutated_saved_cell"
        elif not path.exists():
            reason = "missing_saved_cell"
        else:
            continue
        hold_rows.append({"status": "HOLD", "row_id": row_id, "condition": condition, "panel": spec.get("panel"), "endpoint": _endpoint(spec), "group": _group(spec, cases[row_id], retained), "stratum": _stratum(cases[row_id], spec), "source_kind": source_kind, "cell_key": spec["cell_key"], "cell_path": str(path), "reason": reason, "teacher": {"status": "missing", "reason": reason, "metrics": None}, "clean_known_positive_proxy": {"eligible": False, "checks": {}}})
    by_key = {(str(r["row_id"]), str(r["condition"])): r for r in loaded}
    conditions = ["source", "epoch4", "epoch16", "epoch32"]
    aggregate: dict[str, Any] = {}
    for condition in conditions:
        cond_specs = [s for s in specs if str(s["condition"]) == condition]
        cond_rows = [by_key[k] for k in ((str(s["row_id"]), condition) for s in cond_specs) if k in by_key]
        aggregate[condition] = {"cells": _aggregate(cond_rows, len(cond_specs)), "by_endpoint": {}, "by_group": {}, "by_stratum": {}}
        for field, values in (("endpoint", {_endpoint(s) for s in cond_specs}), ("group", {_group(s, cases[str(s["row_id"])], retained) for s in cond_specs}), ("stratum", {_stratum(cases[str(s["row_id"])], s) for s in cond_specs})):
            aggregate[condition][f"by_{field}"] = {}
            for value in sorted(values):
                selected = [r for r in cond_rows if r[field] == value]
                expected = sum(1 for s in cond_specs if ( _endpoint(s) if field == "endpoint" else _group(s, cases[str(s["row_id"])], retained) if field == "group" else _stratum(cases[str(s["row_id"])], s)) == value)
                aggregate[condition][f"by_{field}"][value] = _aggregate(selected, expected)
    comparisons: list[dict[str, Any]] = []
    for condition in ("epoch4", "epoch16", "epoch32"):
        for key, candidate in sorted(by_key.items()):
            if key[1] != condition:
                continue
            base = by_key.get((key[0], "source"))
            if base and base.get("status") == "complete" and candidate.get("status") == "complete":
                comparisons.append(_paired(base, candidate) | {"panel": candidate["panel"], "endpoint": candidate["endpoint"], "group": candidate["group"], "stratum": candidate["stratum"]})
    guardrails: dict[str, Any] = {}
    for condition in ("epoch4", "epoch16", "epoch32"):
        rows_out: dict[str, Any] = {}
        for endpoint in ("train1024", "validation256", "sentinel96"):
            endpoint_specs = [s for s in specs if str(s["condition"]) == condition and _endpoint(s) == endpoint]
            expected_pairs = len(endpoint_specs)
            paired_rows = []
            for metric in ("bad", "cap", "owner_recurrent", "severe"):
                pairs = []
                for key, candidate in by_key.items():
                    if key[1] != condition or candidate.get("endpoint") != endpoint:
                        continue
                    base = by_key.get((key[0], "source"))
                    if base and base.get("status") == "complete" and candidate.get("status") == "complete":
                        pairs.append((not _flags(base)[metric], _flags(candidate)[metric], key[0]))
                        if key[0] not in {item[2] for item in paired_rows}:
                            paired_rows.append((not _flags(base)[metric], _flags(candidate)[metric], key[0]))
                rows_out.setdefault(endpoint, {})[metric] = {"source_negative_trained_positive": sum(a and b for a,b,_ in pairs), "paired_complete": len(pairs), "new_positive_row_ids": [rid for a,b,rid in pairs if a and b]}
            entry = rows_out[endpoint]
            paired_complete = len({rid for _, _, rid in paired_rows})
            entry["paired_expected"] = expected_pairs
            entry["paired_complete"] = paired_complete
            entry["paired_denominator_complete"] = paired_complete == expected_pairs
            if condition == "epoch32" and endpoint in {"train1024", "validation256"}:
                panel = "train" if endpoint == "train1024" else "validation"
                for metric, limit in LIMITS[panel].items():
                    entry[metric]["prospective_limit"] = limit
                entry["eligibility"] = "HOLD" if paired_complete != expected_pairs else ("pass" if all(entry[m]["source_negative_trained_positive"] <= LIMITS[panel][m] for m in LIMITS[panel]) else "fail")
            elif expected_pairs:
                entry["eligibility"] = "diagnostic_only"
            else:
                entry["eligibility"] = "not_applicable"
        guardrails[condition] = rows_out
    status = "HOLD" if missing or mutations or any(r.get("status") != "complete" for r in loaded) else "complete"
    return {
        "schema": "coordinate_codebook_alignment.scale_reduction.v1",
        "status": status,
        "model_calls": 0,
        "denominator": {"analytical_cells": len(specs), "new_cells": len(specs) - sum(isinstance(s.get("reuse"), Mapping) for s in specs), "reused_source_cells": sum(isinstance(s.get("reuse"), Mapping) for s in specs), "expected_by_condition": EXPECTED, "missing_cells": len(missing), "mutation_cells": len(mutations)},
        "endpoints": {"train1024": 1024, "validation256": 256, "sentinel96": 96},
        "strata": {"training_groups": ["retained32", "additions992"], "density": ["ordinary", "dense", "middle"]},
        "aggregate": aggregate,
        "paired_source_comparisons": comparisons,
        "guardrails": {"definition": "source-negative to trained-positive; UNKNOWN excluded; HOLD cells never count", "limits": LIMITS, "by_condition": guardrails},
        "missing_cells": missing,
        "mutation_cells": mutations,
        "per_image": sorted(loaded + hold_rows, key=lambda r: (str(r["condition"]), str(r["row_id"]))),
        "cell_paths": sources,
        "input_bindings": input_bindings,
        "unknown_policy": "annotation-unmatched predictions remain UNKNOWN; no physical-false conversion",
        "source_manifest": str(manifest.get("schema", "")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True, help="frozen v3 manifest.json")
    parser.add_argument("--run-root", type=Path, required=True, help="scale output root containing production/{condition}/cells")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet = args.manifest.resolve().parent
    result = reduce_scale(_json(args.manifest), _json(packet / "analytical-cells.json"), args.run_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "analytical_cells": result["denominator"]["analytical_cells"], "missing": result["denominator"]["missing_cells"]}, sort_keys=True))


if __name__ == "__main__":
    main()
