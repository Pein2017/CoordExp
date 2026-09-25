"""Deterministic CPU reduction of saved coordinate-codebook evaluation cells."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data.geometry import iou_xyxy
from src.eval.assignment import global_matches


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _case_key(cell: Mapping[str, Any]) -> str:
    return str(cell.get("case", {}).get("row_id", cell.get("row_id", "")))


def _cohort(case_id: str, cases: Mapping[str, Mapping[str, Any]]) -> str:
    return str(cases[case_id].get("_admission", {}).get("cohort", cases[case_id].get("cohort", "unknown")))


def _split(cohort: str) -> str:
    if cohort.startswith("fit_"):
        return "fit"
    if cohort.startswith("monitor_"):
        return "monitor"
    return cohort


def _pairs(cell: Mapping[str, Any], *, threshold: float, class_aware: bool) -> list[tuple[int, int, float]]:
    gt = [(str(item["description"]) if class_aware else "", tuple(float(x) for x in item["bbox"])) for item in cell.get("gt", [])]
    predictions = cell.get("parser", {}).get("predictions", [])
    pred = [(str(item.get("description", "")) if class_aware else "", tuple(float(x) for x in item["bbox"])) for item in predictions]
    return global_matches(gt, pred, threshold)


def _repeat_proxy(cell: Mapping[str, Any]) -> dict[str, Any]:
    gt = list(cell.get("gt", []))
    predictions = cell.get("parser", {}).get("predictions", [])
    keys = [(str(item.get("description", "")), tuple(item.get("coord_bins", ()))) for item in predictions]
    max_run = run = 0
    previous = None
    for key in keys:
        run = run + 1 if key == previous else 1
        max_run = max(max_run, run)
        previous = key
    owner_sequence: list[str | None] = []
    for prediction in predictions:
        box = tuple(float(x) for x in prediction["bbox"])
        candidates = [(iou_xyxy(box, tuple(float(x) for x in target["bbox"])), str(target.get("owner_id"))) for target in gt]
        best = max(candidates, default=(0.0, None), key=lambda item: (item[0], item[1] or ""))
        owner_sequence.append(best[1] if best[0] >= 0.5 else None)
    owner_runs = 0
    owner_max_run = 0
    previous_owner: str | None = None
    for owner in owner_sequence:
        if owner is not None and owner == previous_owner:
            owner_runs += 1
        else:
            owner_runs = 1 if owner is not None else 0
        owner_max_run = max(owner_max_run, owner_runs)
        previous_owner = owner
    owner_revisits = sum(1 for index, owner in enumerate(owner_sequence) if owner is not None and owner in owner_sequence[:index])
    return {"prediction_count": len(predictions), "exact_row_max_run": max_run, "exact_row_revisit_count": len(keys) - len(set(keys)), "owner_sequence_iou50": owner_sequence, "owner_max_run_iou50": owner_max_run, "owner_revisit_count_iou50": owner_revisits}


def _teacher_artifact(cell: Mapping[str, Any]) -> dict[str, Any]:
    raw = cell.get("teacher")
    if not isinstance(raw, Mapping):
        return {"status": "missing", "reason": "teacher_metrics_absent", "metrics": None}
    required = ("token_count", "ce_sum", "ce_mean", "minimum_target_margin", "mean_target_margin")
    if not isinstance(raw.get("token_count"), int) or raw["token_count"] <= 0:
        return {"status": "missing", "reason": "teacher_token_count_invalid", "metrics": dict(raw)}
    for field in required[1:]:
        value = raw.get(field)
        if not isinstance(value, (int, float)) or not float(value) == float(value) or abs(float(value)) == float("inf"):
            return {"status": "missing", "reason": f"teacher_{field}_invalid", "metrics": dict(raw)}
    coordinate_error = raw.get("coordinate_mean_absolute_error")
    if coordinate_error is not None and (not isinstance(coordinate_error, (int, float)) or not float(coordinate_error) == float(coordinate_error) or abs(float(coordinate_error)) == float("inf")):
        return {"status": "missing", "reason": "teacher_coordinate_error_invalid", "metrics": dict(raw)}
    return {"status": "present", "reason": None, "metrics": dict(raw)}


def _teacher_aggregate(rows: Sequence[Mapping[str, Any]], expected_count: int) -> dict[str, Any]:
    present = [row["teacher"]["metrics"] for row in rows if row.get("teacher", {}).get("status") == "present"]
    token_total = sum(int(item["token_count"]) for item in present)
    ce_sum = sum(float(item["ce_sum"]) for item in present)
    macro = lambda field: sum(float(item[field]) for item in present) / len(present) if present else None
    return {"successful_images": len(present), "missing_images": int(expected_count) - len(present), "token_count": token_total, "ce_sum": ce_sum if present else None, "ce_macro_image": macro("ce_mean"), "ce_token_weighted": ce_sum / token_total if token_total else None, "minimum_target_margin_macro_image": macro("minimum_target_margin"), "mean_target_margin_macro_image": macro("mean_target_margin"), "coordinate_mean_absolute_error_macro_image": macro("coordinate_mean_absolute_error") if present and all(item.get("coordinate_mean_absolute_error") is not None for item in present) else None}


def _clean_proxy(out: Mapping[str, Any]) -> dict[str, Any]:
    repeat = out["repeat_proxy"]
    checks = {"complete_iou50_class_consistent": out["iou50_class_consistent"]["matched_count"] == out["target_count"], "zero_owner_revisit_iou50": repeat["owner_revisit_count_iou50"] == 0, "zero_exact_row_revisit": repeat["exact_row_revisit_count"] == 0, "zero_invalid_geometry": out["invalid_geometry"] == 0, "zero_parser_dropped": out["parser_dropped"] == 0, "natural_im_end": out["stop_reason"] == "im_end"}
    return {"eligible": all(checks.values()), "checks": checks}


def reduce_cell(cell: Mapping[str, Any]) -> dict[str, Any]:
    if cell.get("status") != "complete":
        return {"status": "HOLD", "reason": "technical_invalid_cell", "error": cell.get("error"), "teacher": {"status": "missing", "reason": "technical_invalid_cell", "metrics": None}, "clean_known_positive_proxy": {"eligible": False, "checks": {}}}
    gt = list(cell.get("gt", []))
    predictions = list(cell.get("parser", {}).get("predictions", []))
    out: dict[str, Any] = {"status": "complete", "row_id": _case_key(cell), "condition": str(cell.get("condition", "")), "cohort": str(cell.get("case", {}).get("cohort", "unknown")), "target_count": len(gt), "prediction_count": len(predictions), "stop_reason": cell.get("generation", {}).get("stop_reason"), "cap": cell.get("generation", {}).get("cap"), "parser_status": cell.get("parser", {}).get("parse_status"), "invalid_geometry": sum(str(item.get("reason", "")) == "geometry_invalid" for item in cell.get("parser", {}).get("dropped_predictions", [])), "parser_dropped": len(cell.get("parser", {}).get("dropped_predictions", [])), "repeat_proxy": _repeat_proxy(cell), "timing": cell.get("timing", {})}
    for threshold_name, threshold in (("iou50", 0.5), ("iou80", 0.8)):
        for class_aware in (False, True):
            key = f"{threshold_name}_{'class_consistent' if class_aware else 'class_agnostic'}"
            matches = _pairs(cell, threshold=threshold, class_aware=class_aware)
            owners = [str(gt[gt_index].get("owner_id")) for gt_index, _, _ in matches]
            out[key] = {"matched_count": len(matches), "coverage": len(matches) / len(gt) if gt else 0.0, "covered_owner_ids": owners, "missing_owner_ids": [str(item.get("owner_id")) for item in gt if str(item.get("owner_id")) not in set(owners)], "matches": [{"gt_index": a, "prediction_index": b, "iou": c} for a, b, c in matches]}
    # A prediction not assigned to a known positive remains an annotation
    # proxy/unknown; it is not silently converted into a false physical object.
    assigned = {b for a, b, _ in _pairs(cell, threshold=0.5, class_aware=False)}
    out["unknown_prediction_indices"] = [index for index in range(len(predictions)) if index not in assigned]
    out["teacher"] = _teacher_artifact(cell)
    out["clean_known_positive_proxy"] = _clean_proxy(out)
    return out


def _expected_cases(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    for name in ("fit", "monitor"):
        path = Path(manifest[name]["cases_path"])
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                key = str(row["_admission"]["row_id"])
                if key in cases:
                    raise ValueError(f"duplicate admitted image identity: {key}")
                cases[key] = row
    return cases


def _planned_pairs(manifest: Mapping[str, Any], admitted: Mapping[str, Mapping[str, Any]], conditions: Iterable[str] | None, planned_cells: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None) -> set[tuple[str, str]]:
    plan: Any = planned_cells
    if plan is None:
        plan = manifest.get("planned_cells")
    if plan is None and isinstance(manifest.get("evaluation"), Mapping):
        plan = manifest["evaluation"].get("planned_cells") or manifest["evaluation"].get("condition_map")
    selected_conditions = None if conditions is None else {str(item) for item in conditions}
    if plan is None:
        names = selected_conditions or set()
        return {(row_id, condition) for row_id in admitted for condition in names}
    if isinstance(plan, Mapping) and "cells" in plan:
        plan = plan["cells"]
    pairs: set[tuple[str, str]] = set()
    if isinstance(plan, Sequence) and not isinstance(plan, (str, bytes)):
        for item in plan:
            if not isinstance(item, Mapping) or "row_id" not in item or "condition" not in item:
                raise ValueError("planned cell entries require row_id and condition")
            pair = (str(item["row_id"]), str(item["condition"]))
            if pair[0] not in admitted:
                raise ValueError(f"planned cell row_id is not admitted: {pair[0]}")
            if selected_conditions is None or pair[1] in selected_conditions:
                pairs.add(pair)
        return pairs
    if isinstance(plan, Mapping):
        for condition, selection in plan.items():
            condition = str(condition)
            if selected_conditions is not None and condition not in selected_conditions:
                continue
            if isinstance(selection, Mapping):
                if "row_ids" in selection:
                    selection = selection["row_ids"]
                elif "cohort" in selection:
                    selection = [row_id for row_id, row in admitted.items() if _cohort(row_id, admitted) == str(selection["cohort"])]
                elif "split" in selection:
                    selection = [row_id for row_id, row in admitted.items() if _split(_cohort(row_id, admitted)) == str(selection["split"])]
                else:
                    raise ValueError(f"unsupported condition map selection: {condition}")
            if not isinstance(selection, Sequence) or isinstance(selection, (str, bytes)):
                raise ValueError(f"condition map selection must be a row_id list: {condition}")
            for row_id in selection:
                row_id = str(row_id)
                if row_id not in admitted:
                    raise ValueError(f"condition map row_id is not admitted: {row_id}")
                pairs.add((row_id, condition))
        return pairs
    raise ValueError("planned_cells must be a list or condition map")


def reduce_cells(manifest: Mapping[str, Any], cells: Iterable[Mapping[str, Any]], conditions: Iterable[str] | None = None, planned_cells: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    admitted = _expected_cases(manifest)
    loaded = list(cells)
    observed = sorted({str(cell.get("condition", "")) for cell in loaded})
    expected_conditions = sorted(set(conditions or observed))
    if not expected_conditions:
        raise ValueError("no evaluation conditions supplied or observed")
    index: dict[tuple[str, str], Mapping[str, Any]] = {}
    duplicates: list[dict[str, str]] = []
    for cell in loaded:
        key = (_case_key(cell), str(cell.get("condition", "")))
        if key in index:
            duplicates.append({"row_id": key[0], "condition": key[1]})
        else:
            index[key] = cell
    planned = _planned_pairs(manifest, admitted, expected_conditions, planned_cells)
    if planned_cells is None and manifest.get("planned_cells") is not None:
        expected_conditions = sorted({condition for _, condition in planned})
    expected = planned if (planned_cells is not None or manifest.get("planned_cells") is not None or isinstance(manifest.get("evaluation"), Mapping) and (manifest["evaluation"].get("planned_cells") or manifest["evaluation"].get("condition_map"))) else {(row_id, condition) for row_id in admitted for condition in expected_conditions}
    missing = sorted(expected - set(index))
    rows: list[dict[str, Any]] = []
    for row_id in sorted(admitted):
        for condition in expected_conditions:
            if (row_id, condition) not in expected:
                continue
            cell = index.get((row_id, condition))
            reduced = reduce_cell(cell) if cell is not None else {"status": "HOLD", "row_id": row_id, "condition": condition, "reason": "missing_saved_cell", "teacher": {"status": "missing", "reason": "missing_saved_cell", "metrics": None}, "clean_known_positive_proxy": {"eligible": False, "checks": {}}}
            reduced["row_id"] = row_id
            reduced["condition"] = condition
            reduced["cohort"] = _cohort(row_id, admitted)
            rows.append(reduced)
    complete = [row for row in rows if row.get("status") == "complete"]
    baseline = "source" if "source" in expected_conditions else (expected_conditions[0] if expected_conditions else None)
    comparisons: list[dict[str, Any]] = []
    if baseline is not None:
        base_rows = {row["row_id"]: row for row in complete if row["condition"] == baseline}
        for condition in expected_conditions:
            if condition == baseline:
                continue
            for row in sorted((item for item in complete if item["condition"] == condition), key=lambda x: x["row_id"]):
                base = base_rows.get(row["row_id"])
                if base is None:
                    continue
                old = set(base["iou50_class_agnostic"]["covered_owner_ids"])
                new = set(row["iou50_class_agnostic"]["covered_owner_ids"])
                comparisons.append({"row_id": row["row_id"], "condition": condition, "baseline": baseline, "gained_owner_ids": sorted(new - old), "lost_owner_ids": sorted(old - new), "baseline_coverage": base["iou50_class_agnostic"]["coverage"], "condition_coverage": row["iou50_class_agnostic"]["coverage"]})
    aggregate: dict[str, Any] = {}
    aggregate_by_split: dict[str, dict[str, Any]] = {}
    for condition in expected_conditions:
        subset = [row for row in complete if row["condition"] == condition]
        cells_expected = sum(1 for _, planned_condition in expected if planned_condition == condition)
        clean_count = sum(bool(row.get("clean_known_positive_proxy", {}).get("eligible")) for row in subset)
        aggregate[condition] = {"cells_complete": len(subset), "cells_expected": cells_expected, "coverage_iou50": sum(row["iou50_class_agnostic"]["matched_count"] for row in subset) / sum(row["target_count"] for row in subset) if subset else None, "coverage_iou80": sum(row["iou80_class_agnostic"]["matched_count"] for row in subset) / sum(row["target_count"] for row in subset) if subset else None, "class_consistent_iou50": sum(row["iou50_class_consistent"]["matched_count"] for row in subset) / sum(row["target_count"] for row in subset) if subset else None, "invalid_geometry": sum(row.get("invalid_geometry", 0) for row in subset), "parser_dropped": sum(row.get("parser_dropped", 0) for row in subset), "unknown_predictions": sum(len(row.get("unknown_prediction_indices", [])) for row in subset), "natural_eos_cells": sum(row.get("stop_reason") == "im_end" for row in subset), "cap_cells": sum(row.get("stop_reason") == "length" for row in subset), "teacher": _teacher_aggregate(subset, cells_expected), "clean_known_positive_proxy": {"successful_images": clean_count, "nonclean_images": cells_expected - clean_count, "denominator_cells_expected": cells_expected, "definition": "IoU50 class-consistent complete coverage, zero owner and exact-row revisits, zero invalid geometry/parser drops, natural im_end; UNKNOWN predictions remain annotation-unknown"}}
        for split in ("fit", "monitor"):
            split_subset = [row for row in subset if _split(str(row.get("cohort", "unknown"))) == split]
            denominator = sum(row["target_count"] for row in split_subset)
            split_expected = sum(1 for row_id, planned_condition in expected if planned_condition == condition and _split(_cohort(row_id, admitted)) == split)
            split_clean = sum(bool(row.get("clean_known_positive_proxy", {}).get("eligible")) for row in split_subset)
            aggregate_by_split.setdefault(split, {})[condition] = {"cells_complete": len(split_subset), "cells_expected": split_expected, "coverage_iou50": sum(row["iou50_class_agnostic"]["matched_count"] for row in split_subset) / denominator if denominator else None, "coverage_iou80": sum(row["iou80_class_agnostic"]["matched_count"] for row in split_subset) / denominator if denominator else None, "class_consistent_iou50": sum(row["iou50_class_consistent"]["matched_count"] for row in split_subset) / denominator if denominator else None, "invalid_geometry": sum(row.get("invalid_geometry", 0) for row in split_subset), "parser_dropped": sum(row.get("parser_dropped", 0) for row in split_subset), "unknown_predictions": sum(len(row.get("unknown_prediction_indices", [])) for row in split_subset), "natural_eos_cells": sum(row.get("stop_reason") == "im_end" for row in split_subset), "cap_cells": sum(row.get("stop_reason") == "length" for row in split_subset), "teacher": _teacher_aggregate(split_subset, split_expected), "clean_known_positive_proxy": {"successful_images": split_clean, "nonclean_images": split_expected - split_clean, "denominator_cells_expected": split_expected}}
    return {"schema": "coordinate_codebook_alignment.reduction.v1", "status": "complete" if not missing and not duplicates and all(row.get("status") == "complete" for row in rows) else "HOLD", "denominator": {"fit": int(manifest["fit"]["count"]), "monitor": int(manifest["monitor"]["count"]), "all_images": len(admitted), "conditions": expected_conditions, "cells_expected": len(expected)}, "observed": {"cells_loaded": len(loaded), "cells_complete": len(complete), "missing_cells": len(missing), "duplicate_cells": len(duplicates)}, "missing_cells": [{"row_id": row_id, "condition": condition} for row_id, condition in missing], "duplicate_cells": duplicates, "aggregate": aggregate, "aggregate_by_split": aggregate_by_split, "per_image": rows, "comparisons_vs_baseline": comparisons, "matching_contract": {"iou": "cardinality-first one-to-one via src.eval.assignment.global_matches", "class_agnostic": "annotation/owner proxy", "class_consistent": "description equality", "unknown": "annotation-unmatched predictions remain unknown; no physical judge upgrade"}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conditions", help="comma-separated expected condition names")
    parser.add_argument("--planned-cells", type=Path, help="JSON list or condition map defining the frozen cell denominator")
    parser.add_argument("--cell-list", type=Path, help="Frozen JSON list of exact saved cell paths to replay")
    args = parser.parse_args()
    manifest = _read_json(args.manifest)
    paths = (sorted(Path(p) for p in json.loads(args.cell_list.read_text()))
             if args.cell_list else sorted(args.input_root.rglob("*.json")))
    cells = []
    for path in paths:
        if path.resolve() == args.output.resolve():
            continue
        value = json.loads(path.read_text())
        if args.cell_list and (not isinstance(value, dict) or value.get("schema") != "coordinate_codebook_alignment.evaluation_cell.v1"):
            raise ValueError(f"listed input is not an evaluation cell: {path}")
        if isinstance(value, dict) and value.get("schema") == "coordinate_codebook_alignment.evaluation_cell.v1":
            cells.append(value)
    conditions = None if args.conditions is None else [item for item in args.conditions.split(",") if item]
    planned = None if args.planned_cells is None else json.loads(args.planned_cells.read_text())
    result = reduce_cells(manifest, cells, conditions, planned_cells=planned)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "cells_expected": result["denominator"]["cells_expected"], "missing": result["observed"]["missing_cells"]}, sort_keys=True))


if __name__ == "__main__":
    main()
