"""Saved-cell reducer for the frozen injection-off discriminator panel."""
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

from probes.training_set_completion.coordinate_codebook_alignment.reduce import reduce_cell
from probes.training_set_completion.coordinate_codebook_alignment.scale_reduce import (
    LIMITS,
    _aggregate,
    _cases,
    _endpoint,
    _flags,
    _format_summary,
    _group,
    _paired,
    _stratum,
)

EXPECTED = {"source": 1280, "three_loss_epoch16": 1280,
            "early_edge_epoch16": 1280, "off_epoch16": 1280}
REUSED = {"source": 1280, "three_loss_epoch16": 1280,
          "early_edge_epoch16": 1280, "off_epoch16": 0}
CONDITIONS = tuple(EXPECTED)
FAILURES = ("bad", "cap", "owner_recurrent", "severe")


def _cell_path(run_root: Path, spec: Mapping[str, Any]) -> tuple[Path, str]:
    reuse = spec.get("reuse")
    if isinstance(reuse, Mapping):
        return Path(str(reuse["path"])), "reused_source"
    return run_root / "production" / str(spec["condition"]) / "cells" / f"{spec['cell_key']}.json", "new"


def _retained32(analytical: Mapping[str, Any], specs: list[Mapping[str, Any]], cases: Mapping[str, Mapping[str, Any]]) -> set[str]:
    raw = analytical.get("retained32_row_ids")
    if not isinstance(raw, list) or len(raw) != 32 or len({str(x) for x in raw}) != 32:
        raise ValueError("explicit retained32_row_ids must contain exactly 32 unique identities")
    retained = {str(x) for x in raw}
    train_source = {str(s["row_id"]) for s in specs
                    if s.get("condition") == "source" and s.get("panel") == "train"}
    if not retained <= train_source or not retained <= cases.keys():
        raise ValueError("retained32_row_ids must name admitted source rows in the train panel")
    return retained


def _validate_specs(specs: list[Mapping[str, Any]]) -> None:
    if len(specs) != 5120:
        raise ValueError(f"frozen analytical denominator must contain 5120 cells, got {len(specs)}")
    keys = [(str(s["row_id"]), str(s["condition"])) for s in specs]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate frozen analytical cell identity")
    counts = Counter(str(s["condition"]) for s in specs)
    if dict(counts) != EXPECTED:
        raise ValueError(f"frozen analytical condition counts differ: {dict(counts)}")
    panels = Counter((str(s["condition"]), str(s.get("panel", ""))) for s in specs)
    expected_panels = {(condition, panel): size for condition in CONDITIONS
                       for panel, size in (("train", 1024), ("validation", 256))}
    if dict(panels) != expected_panels:
        raise ValueError(f"frozen analytical panel counts differ: {dict(panels)}")
    reuse = Counter(str(s["condition"]) for s in specs if isinstance(s.get("reuse"), Mapping))
    if {name: reuse.get(name, 0) for name in CONDITIONS} != REUSED:
        raise ValueError("frozen reuse counts differ")
    panels_by_row: dict[str, set[str]] = {}
    for spec in specs:
        panels_by_row.setdefault(str(spec["row_id"]), set()).add(str(spec["panel"]))
    if len(panels_by_row) != 1280 or any(len(value) != 1 for value in panels_by_row.values()):
        raise ValueError("frozen conditions must share exactly 1280 panel-aligned row identities")
    for row_id, panelset in panels_by_row.items():
        expected = set(CONDITIONS)
        actual = {str(s["condition"]) for s in specs if str(s["row_id"]) == row_id}
        if actual != expected or len(panelset) != 1:
            raise ValueError(f"frozen row-condition identity mismatch: {row_id}")


def _hold(spec: Mapping[str, Any], path: Path, source_kind: str, retained: set[str],
          cases: Mapping[str, Mapping[str, Any]], reason: str) -> dict[str, Any]:
    row_id, condition = str(spec["row_id"]), str(spec["condition"])
    return {"status": "HOLD", "row_id": row_id, "condition": condition,
            "panel": spec.get("panel"), "endpoint": _endpoint(spec),
            "group": _group(spec, cases[row_id], retained),
            "stratum": _stratum(cases[row_id], spec), "source_kind": source_kind,
            "cell_key": spec["cell_key"], "cell_path": str(path), "reason": reason,
            "teacher": {"status": "missing", "reason": reason, "metrics": None},
            "clean_known_positive_proxy": {"eligible": False, "checks": {}}}


def _pair(base: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    pair = _paired(base, candidate)
    base_flags, candidate_flags = _flags(base), _flags(candidate)
    base_teacher = base.get("teacher", {}).get("metrics") or {}
    candidate_teacher = candidate.get("teacher", {}).get("metrics") or {}
    fidelity_fields = ("ce_mean", "minimum_target_margin", "mean_target_margin",
                       "coordinate_mean_absolute_error")
    pair.update({
        "panel": candidate["panel"], "endpoint": candidate["endpoint"],
        "group": candidate["group"], "stratum": candidate["stratum"],
        "baseline_clean": bool(base.get("clean_known_positive_proxy", {}).get("eligible")),
        "candidate_clean": bool(candidate.get("clean_known_positive_proxy", {}).get("eligible")),
        "clean_delta": int(bool(candidate.get("clean_known_positive_proxy", {}).get("eligible"))) -
                       int(bool(base.get("clean_known_positive_proxy", {}).get("eligible"))),
        "failure_transitions": {name: {
            "baseline": base_flags[name], "candidate": candidate_flags[name],
            "repaired": base_flags[name] and not candidate_flags[name],
            "persistent": base_flags[name] and candidate_flags[name],
            "newly_introduced": not base_flags[name] and candidate_flags[name],
        } for name in FAILURES},
        "teacher_fidelity": {field: {
            "baseline": base_teacher.get(field), "candidate": candidate_teacher.get(field),
            "delta": (float(candidate_teacher[field]) - float(base_teacher[field])
                      if isinstance(base_teacher.get(field), (int, float)) and
                      isinstance(candidate_teacher.get(field), (int, float)) else None),
        } for field in fidelity_fields},
        "severity": {
            "baseline": {"flags": base_flags, "repeat_proxy": base.get("repeat_proxy", {}),
                         "parser_dropped": base.get("parser_dropped", 0),
                         "invalid_geometry": base.get("invalid_geometry", 0),
                         "format_summary": base.get("format_summary", {})},
            "candidate": {"flags": candidate_flags, "repeat_proxy": candidate.get("repeat_proxy", {}),
                          "parser_dropped": candidate.get("parser_dropped", 0),
                          "invalid_geometry": candidate.get("invalid_geometry", 0),
                          "format_summary": candidate.get("format_summary", {})},
        },
    })
    return pair


def _summarize_pairs(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"paired_complete": len(rows),
                              "clean_delta": sum(int(r["clean_delta"]) for r in rows),
                              "clean_improved": sum(int(r["clean_delta"]) > 0 for r in rows),
                              "clean_regressed": sum(int(r["clean_delta"]) < 0 for r in rows)}
    for metric in ("iou50_class_consistent", "iou80_class_consistent"):
        net = [len(r[metric]["gained_owner_ids"]) - len(r[metric]["lost_owner_ids"])
               for r in rows]
        result[metric] = {
            "gained_matches": sum(len(r[metric]["gained_owner_ids"]) for r in rows),
            "lost_matches": sum(len(r[metric]["lost_owner_ids"]) for r in rows),
            "net_match_delta": sum(net),
            "image_improvements": sum(delta > 0 for delta in net),
            "image_regressions": sum(delta < 0 for delta in net),
            "images_with_zero_net_matches": sum(delta == 0 for delta in net),
            "images_with_both_gain_and_loss": sum(bool(r[metric]["lost_owner_ids"]) and bool(r[metric]["gained_owner_ids"]) for r in rows),
        }
    result["failure_transitions"] = {name: {
        "repaired": sum(bool(r["failure_transitions"][name]["repaired"]) for r in rows),
        "persistent": sum(bool(r["failure_transitions"][name]["persistent"]) for r in rows),
        "newly_introduced": sum(bool(r["failure_transitions"][name]["newly_introduced"]) for r in rows),
    } for name in FAILURES}
    return result


def reduce_off(manifest: Mapping[str, Any], analytical: Mapping[str, Any], run_root: Path) -> dict[str, Any]:
    specs = list(analytical.get("specs", []))
    _validate_specs(specs)
    cases = _cases(specs)
    retained = _retained32(analytical, specs, cases)
    loaded: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    mutations: list[dict[str, Any]] = []
    technical_invalid: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = []
    for spec in specs:
        row_id, condition = str(spec["row_id"]), str(spec["condition"])
        path, source_kind = _cell_path(run_root, spec)
        raw = path.read_bytes() if path.exists() else None
        digest = hashlib.sha256(raw).hexdigest() if raw is not None else None
        binding = {"path": str(path), "sha256": digest,
                   "size_bytes": len(raw) if raw is not None else None,
                   "cell_key": spec["cell_key"], "row_id": row_id,
                   "condition": condition, "source_kind": source_kind}
        reuse = spec.get("reuse")
        expected_hash = reuse.get("sha256") if isinstance(reuse, Mapping) else None
        if isinstance(reuse, Mapping):
            binding["expected_sha256"] = expected_hash
        bindings.append(binding)
        if raw is None:
            missing.append({"cell_key": spec["cell_key"], "row_id": row_id,
                            "condition": condition, "path": str(path)})
            continue
        if isinstance(reuse, Mapping) and (not isinstance(expected_hash, str) or not expected_hash or expected_hash != digest):
            mutations.append({"cell_key": spec["cell_key"], "row_id": row_id,
                              "condition": condition, "path": str(path),
                              "observed_condition": None, "observed_row_id": None,
                              "reason": ["reuse_sha256_missing_or_mismatch"]})
            continue
        try:
            cell = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            entry = {"cell_key": spec["cell_key"], "row_id": row_id,
                     "condition": condition, "path": str(path),
                     "reason": "invalid_json", "detail": str(exc)}
            if isinstance(reuse, Mapping):
                mutations.append({**entry, "observed_condition": None,
                                  "observed_row_id": None,
                                  "reason": ["reused_cell_invalid_json"]})
            else:
                technical_invalid.append(entry)
            continue
        if not isinstance(cell, Mapping):
            mismatch = True
            actual_condition, actual_row = None, None
        else:
            actual_condition = cell.get("condition")
            case = cell.get("case")
            actual_row = case.get("row_id", cell.get("row_id")) if isinstance(case, Mapping) else cell.get("row_id")
            mismatch = str(actual_condition or "") != condition or str(actual_row or "") != row_id
        if mismatch:
            mutations.append({"cell_key": spec["cell_key"], "row_id": row_id,
                              "condition": condition, "path": str(path),
                              "observed_condition": actual_condition,
                              "observed_row_id": actual_row,
                              "reason": ["cell_identity_mismatch"]})
            continue
        reduced = reduce_cell(cell)
        reduced.update(row_id=row_id, condition=condition, panel=spec["panel"],
                       endpoint=_endpoint(spec), group=_group(spec, cases[row_id], retained),
                       stratum=_stratum(cases[row_id], spec), source_kind=source_kind,
                       cell_key=spec["cell_key"], cell_path=str(path),
                       format_summary=_format_summary(cell, reduced))
        loaded.append(reduced)

    by_key = {(str(r["row_id"]), str(r["condition"])): r for r in loaded}
    holds = []
    for spec in specs:
        key = (str(spec["row_id"]), str(spec["condition"]))
        if key in by_key:
            continue
        path, source_kind = _cell_path(run_root, spec)
        if not path.exists():
            reason = "missing_saved_cell"
        elif any(x["row_id"] == key[0] and x["condition"] == key[1] for x in technical_invalid):
            reason = "invalid_new_saved_cell"
        else:
            reason = "mutated_saved_cell"
        holds.append(_hold(spec, path, source_kind, retained, cases, reason))

    aggregates: dict[str, Any] = {}
    for condition in CONDITIONS:
        cond_specs = [s for s in specs if str(s["condition"]) == condition]
        rows = [by_key[(str(s["row_id"]), condition)] for s in cond_specs
                if (str(s["row_id"]), condition) in by_key]
        groups = {str(r["group"]) for r in rows}
        strata = {str(r["stratum"]) for r in rows}
        aggregates[condition] = {
            "all": _aggregate(rows, len(cond_specs)),
            "by_panel": {panel: _aggregate([r for r in rows if r["panel"] == panel],
                                           sum(s["panel"] == panel for s in cond_specs))
                         for panel in ("train", "validation")},
            "by_group": {group: _aggregate([r for r in rows if r["group"] == group],
                                           sum(_group(s, cases[str(s["row_id"])], retained) == group for s in cond_specs))
                         for group in sorted(groups | {"retained32", "additions992", "validation256"})},
            "by_stratum": {stratum: _aggregate([r for r in rows if r["stratum"] == stratum],
                                               sum(_stratum(cases[str(s["row_id"])], s) == stratum for s in cond_specs))
                           for stratum in sorted(strata | {"ordinary", "dense", "middle", "unknown"})},
        }

    pair_specs = (("off_epoch16", "source"),
                  ("early_edge_epoch16", "off_epoch16"),
                  ("early_edge_epoch16", "source"),
                  ("three_loss_epoch16", "off_epoch16"),
                  ("three_loss_epoch16", "source"))
    comparisons: list[dict[str, Any]] = []
    pair_summaries: dict[str, Any] = {}
    for candidate_condition, baseline_condition in pair_specs:
        rows = []
        for spec in specs:
            if str(spec["condition"]) != candidate_condition:
                continue
            row_id = str(spec["row_id"])
            base, candidate = by_key.get((row_id, baseline_condition)), by_key.get((row_id, candidate_condition))
            if base and candidate and base.get("status") == candidate.get("status") == "complete":
                rows.append(_pair(base, candidate))
        name = f"{candidate_condition}_vs_{baseline_condition}"
        comparisons.extend({"baseline_condition": baseline_condition,
                            "condition": candidate_condition, **r} for r in rows)
        buckets = {"all": rows}
        for value in ("ordinary", "dense", "middle", "unknown"):
            buckets[value] = [r for r in rows if r["stratum"] == value]
        for value in ("retained32", "additions992", "validation256"):
            buckets[value] = [r for r in rows if r["group"] == value]
        pair_summaries[name] = {key: _summarize_pairs(value) for key, value in buckets.items()}

    guardrails: dict[str, Any] = {}
    for condition in ("off_epoch16",):
        guardrails[condition] = {}
        for panel, endpoint in (("train", "train1024"), ("validation", "validation256")):
            selected = [r for r in comparisons if r["condition"] == condition and
                        r["baseline_condition"] == "source" and r["panel"] == panel]
            limits = LIMITS[panel]
            entry = {"paired_expected": 1024 if panel == "train" else 256,
                     "paired_complete": len(selected),
                     "eligibility": "HOLD" if len(selected) != (1024 if panel == "train" else 256) else None,
                     "metrics": {}}
            for metric in FAILURES:
                count = sum(r["failure_transitions"][metric]["newly_introduced"] for r in selected)
                entry["metrics"][metric] = {"source_negative_off_positive": count,
                                           "prospective_limit": limits[metric]}
            if entry["eligibility"] is None:
                entry["eligibility"] = "pass" if all(v["source_negative_off_positive"] <= v["prospective_limit"]
                                                       for v in entry["metrics"].values()) else "fail"
            guardrails[condition][endpoint] = entry

    status = "HOLD" if missing or mutations or technical_invalid or holds or any(r.get("status") != "complete" for r in loaded) else "complete"
    return {
        "schema": "coordinate_codebook_alignment.off_reduction.v1", "status": status,
        "model_calls": 0,
        "denominator": {"analytical_cells": 5120, "distinct_cells": 5120,
                        "images": 1280, "new_cells": 1280, "reused_cells": 3840,
                        "expected_by_condition": EXPECTED, "reused_by_condition": REUSED,
                        "missing_cells": len(missing), "mutation_cells": len(mutations),
                        "technical_invalid_cells": len(technical_invalid)},
        "retained_membership": {"method": "explicit_retained32_row_ids", "count": 32,
                                "row_ids": sorted(retained)},
        "aggregates": aggregates, "paired_summaries": pair_summaries,
        "guardrails": {"off_vs_source": guardrails["off_epoch16"],
                       "definition": "source-negative to off-positive; HOLD and UNKNOWN never count",
                       "limits": LIMITS},
        "per_image_comparisons": comparisons,
        "missing_cells": missing, "mutation_cells": mutations,
        "technical_invalid_cells": technical_invalid,
        "per_image": sorted(loaded + holds, key=lambda r: (str(r["condition"]), str(r["row_id"]))),
        "input_bindings": bindings,
        "teacher_ce_scope": "teacher CE and margins are fitted-panel evaluation metrics, distinct from training objective losses",
        "training_objective": manifest.get("effective_losses"),
        "unknown_policy": "annotation-unmatched predictions remain UNKNOWN; no physical-false conversion",
        "incomplete_comparison_policy": "HOLD; unmatched cells are not failures or physical negatives",
        "source_manifest": str(manifest.get("schema", "")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--analytical", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing reduction: {args.output}")
    result = reduce_off(json.loads(args.manifest.read_text()),
                        json.loads(args.analytical.read_text()), args.run_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": result["status"],
                      "analytical_cells": result["denominator"]["analytical_cells"],
                      "missing": result["denominator"]["missing_cells"]}, sort_keys=True))


if __name__ == "__main__":
    main()
