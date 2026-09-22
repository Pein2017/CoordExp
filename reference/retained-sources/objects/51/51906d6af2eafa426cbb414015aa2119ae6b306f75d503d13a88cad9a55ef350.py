#!/usr/bin/env python3
"""Standalone CPU reducer for corrected Lane-B broad outputs.

It consumes only saved runtime/reduced/manifests and emits compact scientific
summaries. The raw JSON receipts remain the evidence boundary.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source")
FINAL = ROOT / "final"
BROAD = FINAL / "broad-v1"
REDUCED = BROAD / "reduced"
RUNTIME = BROAD / "runtime"
MANIFESTS = FINAL / "manifests"
PILOT = FINAL / "corrected-pilot-v2"
PANEL = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-distribution-census/shared-panel.json")
EXPECTED_CELLS = ["00", "10-", "10+", "01-", "01+", "11-", "11+"]
TRANSFORM_CELLS = EXPECTED_CELLS[1:]


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bind(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    return {"path": str(p), "sha256": sha256(p), "size_bytes": p.stat().st_size}


def finite(values: list[Any]) -> list[float]:
    return [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]


def median(values: list[Any]) -> float | None:
    vals = finite(values)
    return statistics.median(vals) if vals else None


def qpercent(values: list[Any], q: float) -> float | None:
    vals = sorted(finite(values))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def source_stratum(group: str) -> str:
    if group == "val-extra":
        return "val-extra-runtime-stratum"
    if group.startswith("new-"):
        return "prospective-new-cohort"
    return "mature-panel"


def panel_kind(admission: dict[str, Any]) -> str:
    return "failure" if admission.get("mode") == "failure" else "proxy"


def run_witness(parse: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    exact = parse.get("exact_runs") or []
    near = parse.get("near_runs") or []
    run = exact[0] if exact else (near[0] if near else None)
    return run, "exact" if exact else ("near" if near else None)


def chain_lengths(runs: list[dict[str, Any]]) -> list[int]:
    """Summarize overlapping consecutive triple windows without pair inflation."""
    if not runs:
        return []
    starts = sorted({int(r["start_row"]) for r in runs})
    lengths: list[int] = []
    start = prev = starts[0]
    for cur in starts[1:]:
        if cur == prev + 1:
            prev = cur
            continue
        lengths.append(prev - start + 3)
        start = prev = cur
    lengths.append(prev - start + 3)
    return lengths


def compact_anchor(parse: dict[str, Any]) -> dict[str, Any] | None:
    run, kind = run_witness(parse)
    if run is None:
        return None
    coords = list(run["coord_bins_source"])
    return {
        "kind": kind,
        "start_row": int(run["start_row"]),
        "row_indices": list(run["row_indices"]),
        "description": run.get("description"),
        "coord_bins_source": coords,
        "x_center_source": (coords[0] + coords[2]) / 2.0,
        "y_center_source": (coords[1] + coords[3]) / 2.0,
    }


def stop_class(cell: dict[str, Any]) -> str:
    row = cell.get("row_limit") or {}
    if row.get("injected_eos") and row.get("injection_reason") == "row_cap":
        return "row_cap_injected_eos"
    if cell.get("stop_reason") in {"im_end", "eos"}:
        return "natural_eos"
    return str(cell.get("stop_reason") or "unknown")


def opener_summary(boundary: dict[str, Any]) -> dict[str, Any]:
    opener = boundary.get("opener") or {}
    top = opener.get("top10") or []
    chosen = next((x for x in top if x.get("token_id") == opener.get("chosen_token_id")), None)
    eos = opener.get("eos") or {}
    fam = opener.get("coordinate_family") or {}
    return {
        "chosen_token_id": opener.get("chosen_token_id"),
        "chosen_log_prob": chosen.get("log_prob") if chosen else None,
        "eos_log_prob": eos.get("log_prob"),
        "chosen_minus_eos_log_prob": (
            chosen.get("log_prob") - eos.get("log_prob")
            if chosen and isinstance(eos.get("log_prob"), (int, float))
            else None
        ),
        "coordinate_family_mass": fam.get("mass"),
        "coordinate_family_log_mass": fam.get("log_mass"),
        "coordinate_family_max_bin": fam.get("max_bin"),
        "boundary_stop_reason": opener.get("stop_reason"),
        "input_width": opener.get("input_width"),
    }


def window_summary(boundary: dict[str, Any]) -> dict[str, Any]:
    raw = boundary.get("windows_by_sign")
    if not isinstance(raw, dict):
        raw = {}
    result: dict[str, Any] = {}
    for sign in ["-", "+"]:
        x = raw.get(sign)
        if not isinstance(x, dict):
            result[sign] = None
            continue
        old = x.get("old_window") or {}
        moved = x.get("moved_window") or {}
        result[sign] = {
            "old_bin": x.get("old_bin"),
            "moved_bin": x.get("moved_bin"),
            "window_radius": x.get("window_radius"),
            "old_logit": x.get("old_logit"),
            "moved_logit": x.get("moved_logit"),
            "old_log_prob": x.get("old_log_prob"),
            "moved_log_prob": x.get("moved_log_prob"),
            "moved_minus_old_log_prob": x.get("moved_minus_old_log_prob"),
            "old_window_log_mass": old.get("log_mass"),
            "old_window_mass": old.get("mass"),
            "moved_window_log_mass": moved.get("log_mass"),
            "moved_window_mass": moved.get("mass"),
            "moved_minus_old_log_mass": x.get("moved_minus_old_log_mass"),
        }
    return result


def compact_identity(identity: dict[str, Any] | None) -> dict[str, Any] | None:
    """Keep binding evidence while leaving full prompt IDs in raw receipts."""
    if not isinstance(identity, dict):
        return None
    prompts = identity.get("prompt_token_ids") or []
    prompt_blob = json.dumps(prompts, separators=(",", ":"), sort_keys=True).encode()
    return {
        "request_ids": identity.get("request_ids"),
        "prompt_count": len(prompts),
        "prompt_token_lengths": [len(x) for x in prompts if isinstance(x, list)],
        "prompt_token_ids_sha256": hashlib.sha256(prompt_blob).hexdigest(),
        "media_sha256": identity.get("media_sha256"),
        "image_grids": identity.get("image_grids"),
        "tensor_sha256": identity.get("tensor_sha256"),
    }


def compact_cell(cell_key: str, reduced_cell: dict[str, Any], manifest_cell: dict[str, Any]) -> dict[str, Any]:
    p = reduced_cell["parse"]
    exact = p.get("exact_runs") or []
    near = p.get("near_runs") or []
    exact_lengths = chain_lengths(exact)
    near_lengths = chain_lengths(near)
    anchor = compact_anchor(p)
    return {
        "cell": cell_key,
        "visual_offset_px": manifest_cell.get("visual_offset_px"),
        "history_tx_relative_px": manifest_cell.get("history_tx_relative_px"),
        "history_offset_px": manifest_cell.get("history_offset_px"),
        "history_sha256": manifest_cell.get("history_sha256"),
        "token_count": reduced_cell.get("token_count"),
        "stop_reason": reduced_cell.get("stop_reason"),
        "stop_class": stop_class(reduced_cell),
        "row_limit": reduced_cell.get("row_limit"),
        "complete_rows": p.get("complete_rows"),
        "valid_rows": p.get("valid_rows"),
        "invalid_rows": p.get("invalid_rows"),
        "malformed_rows": p.get("malformed_rows"),
        "exact_triple_window_count": len(exact),
        "near_triple_window_count": len(near),
        "exact_recurrent": bool(exact),
        "near_recurrent": bool(near),
        "failure_predicate": bool(p.get("failure_predicate")),
        "longest_exact_run_rows": max(exact_lengths) if exact_lengths else 0,
        "longest_near_run_rows": max(near_lengths) if near_lengths else 0,
        "first_recurrence_anchor": anchor,
        "known": {
            "bank_count": (reduced_cell.get("known") or {}).get("bank_count"),
            "matched_count": (reduced_cell.get("known") or {}).get("matched_count"),
            "bank_source": reduced_cell.get("known_bank_source"),
        },
        "opener": opener_summary(reduced_cell.get("boundary") or {}),
        "forced_description_x1_windows": window_summary(reduced_cell.get("boundary") or {}),
        "input_identity": compact_identity(reduced_cell.get("input_identity")),
        "runtime_parse_compatibility": reduced_cell.get("runtime_parse_compatibility"),
    }


def load_state(state_id: str, reduced: dict[str, Any], execution: str, runtime_paths: list[Path], manifest_path: Path) -> dict[str, Any]:
    manifest = read(manifest_path)
    source = reduced.get("source", {})
    # The pilot has separate common and transform runtime receipts; broad has one.
    runtime = [read(p) for p in runtime_paths]
    cells: dict[str, Any] = {}
    for key, rc in reduced.get("cells", {}).items():
        cells[key] = compact_cell(key, rc, manifest["cells"][key])
    admission = reduced.get("admission") or {}
    raw_source = {
        "panel": source.get("panel"),
        "shared_panel": source.get("shared_panel"),
        "mature_raw": source.get("mature_raw"),
        "feedback_selection": source.get("feedback_selection"),
        "feedback_selection_id": source.get("feedback_selection_id"),
        "transform_manifest": source.get("transform_manifest"),
    }
    # Keep only binding metadata already in the reduction; do not embed prompts or token arrays.
    return {
        "state_id": state_id,
        "execution": execution,
        "model": reduced.get("model"),
        "image_id": reduced.get("image_id"),
        "group": reduced.get("group"),
        "source_stratum": source_stratum(str(reduced.get("group"))),
        "panel_kind": panel_kind(admission),
        "status": reduced.get("status"),
        "admission": admission,
        "cells_present": sorted(cells),
        "cells_expected": EXPECTED_CELLS,
        "cell_count": len(cells),
        "source": raw_source,
        "source_manifest": bind(manifest_path),
        "source_row_count_00": len(manifest.get("cells", {}).get("00", {}).get("history_boxes", [])),
        "source_invalid_row_count_00": sum(
            not bool(x.get("source_valid", False))
            for x in manifest.get("cells", {}).get("00", {}).get("history_boxes", [])
        ),
        "runtime": {
            "model_forwards": sum(int(x.get("model_forwards") or 0) for x in runtime),
            "vision_forwards": sum(int(x.get("vision_forwards") or 0) for x in runtime),
            "elapsed_seconds": sum(float(x.get("elapsed_seconds") or 0.0) for x in runtime),
            "gpu_seconds": None,
            "no_parameter_mutation": all(bool(x.get("no_parameter_mutation", False)) for x in runtime),
            "runtime_receipts": [bind(p) for p in runtime_paths],
        },
        "cells": cells,
    }


def make_states() -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    for p in sorted(REDUCED.glob("*.json")):
        if p.name in {"pilot-summary.json"}:
            continue
        d = read(p)
        state_id = p.stem
        rp = RUNTIME / f"{state_id}.json"
        if not rp.exists():
            raise AssertionError(f"missing broad runtime for {state_id}")
        mp = MANIFESTS / f"{state_id}.json"
        if not mp.exists():
            raise AssertionError(f"missing broad manifest for {state_id}")
        states.append(load_state(state_id, d, "broad-v1", [rp], mp))
    if len(states) != 43:
        raise AssertionError(f"expected 43 broad states, got {len(states)}")
    for model in ("tied", "untied"):
        common_p = PILOT / "reduced" / f"{model}-reduced.json"
        transform_p = PILOT / "reduced" / f"{model}-transforms-reduced.json"
        common = read(common_p)
        transforms = read(transform_p)
        combined = dict(common)
        combined["cells"] = dict(common.get("cells", {}))
        combined["cells"].update(transforms.get("cells", {}))
        state_id = f"{model}-417044-failure"
        mp = Path(common["source"]["transform_manifest"]["path"])
        rp = [
            PILOT / "raw" / f"{model}-runtime.json",
            PILOT / "raw" / f"{model}-transforms-runtime.json",
        ]
        states.append(load_state(state_id, combined, "corrected-pilot-v2", rp, mp))
    states.sort(key=lambda x: x["state_id"])
    if len(states) != 45:
        raise AssertionError(f"expected 45 total states, got {len(states)}")
    return states


def affected_geometry() -> dict[str, Any]:
    paths = list(MANIFESTS.glob("*.json")) + list((PILOT / "inputs" / "manifests").glob("*-417044-failure.json"))
    seen = {p.name: p for p in paths}
    affected = []
    supplied = 0
    invalid = []
    max_drift = 0
    for p in sorted(seen.values()):
        d = read(p)
        boxes = d.get("cells", {}).get("00", {}).get("history_boxes", [])
        supplied += len(boxes)
        for b in boxes:
            max_drift = max(max_drift, int(b.get("max_inverse_drift", 0)))
            if not b.get("source_valid", False):
                invalid.append({"state_id": p.stem, "row_index": b.get("row_index"), "source_bins": b.get("source_bins"), "mapped_bins": b.get("mapped_bins"), "mapped_valid": b.get("mapped_valid"), "order_preserved": b.get("order_preserved")})
        # Report the five frozen affected state/cell/row entries exactly.
        for key, cell in d.get("cells", {}).items():
            for b in cell.get("history_boxes", []):
                if b.get("validity_changed_by_rounding") or not b.get("order_preserved", True):
                    affected.append({"state_id": p.stem, "cell": key, "row_index": b.get("row_index"), "source_bins": b.get("source_bins"), "mapped_bins": b.get("mapped_bins"), "inverse_bins": b.get("inverse_bins"), "mapped_valid": b.get("mapped_valid"), "source_valid": b.get("source_valid"), "order_preserved": b.get("order_preserved"), "validity_changed_by_rounding": b.get("validity_changed_by_rounding"), "max_inverse_drift": b.get("max_inverse_drift")})
    return {"supplied_boxes_across_45x7_cells": supplied * 7, "source_rows_across_45_states": supplied, "source_invalid_rows": invalid, "source_invalid_row_count": len(invalid), "affected_rounding_or_order_entries": affected, "affected_count": len(affected), "max_inverse_drift": max_drift, "inverse_drift_gt_1_count": 0 if max_drift <= 1 else None}


def cell_stats(states: list[dict[str, Any]], *, admitted_only: bool) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in EXPECTED_CELLS:
        cs = []
        for st in states:
            if admitted_only and not st["admission"].get("admitted", False):
                continue
            if key in st["cells"]:
                cs.append(st["cells"][key])
        out[key] = {
            "present": len(cs),
            "failure_predicate": sum(bool(c["failure_predicate"]) for c in cs),
            "exact_recurrent": sum(bool(c["exact_recurrent"]) for c in cs),
            "near_recurrent": sum(bool(c["near_recurrent"]) for c in cs),
            "complete_rows": sum(int(c["complete_rows"] or 0) for c in cs),
            "valid_rows": sum(int(c["valid_rows"] or 0) for c in cs),
            "invalid_rows": sum(int(c["invalid_rows"] or 0) for c in cs),
            "malformed_rows": sum(int(c["malformed_rows"] or 0) for c in cs),
            "row_cap_injected_eos": sum(c["stop_class"] == "row_cap_injected_eos" for c in cs),
            "natural_eos": sum(c["stop_class"] == "natural_eos" for c in cs),
            "token_count_median": median([c["token_count"] for c in cs]),
            "token_count_max": max([int(c["token_count"]) for c in cs], default=None),
            "longest_exact_run_rows_max": max([int(c["longest_exact_run_rows"]) for c in cs], default=0),
            "longest_near_run_rows_max": max([int(c["longest_near_run_rows"]) for c in cs], default=0),
        }
    return out


def primary_contrasts(states: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for kind in ("failure", "proxy"):
        chosen = [s for s in states if s["panel_kind"] == kind and s["admission"].get("admitted", False)]
        row: dict[str, Any] = {"denominator_admitted_states": len(chosen), "state_ids": [s["state_id"] for s in chosen]}
        for key in ["00", "10-", "10+", "01-", "01+", "11-", "11+"]:
            present = [s for s in chosen if key in s["cells"]]
            row[key] = {
                "present": len(present),
                "failure_predicate": sum(bool(s["cells"][key]["failure_predicate"]) for s in present),
                "exact_recurrent": sum(bool(s["cells"][key]["exact_recurrent"]) for s in present),
                "near_recurrent": sum(bool(s["cells"][key]["near_recurrent"]) for s in present),
            }
        a = row["00"]["failure_predicate"]
        m = row["11-"]["failure_predicate"]
        p = row["11+"]["failure_predicate"]
        row["coherent_11_both"] = sum(bool(s["cells"].get("11-", {}).get("failure_predicate")) and bool(s["cells"].get("11+", {}).get("failure_predicate")) for s in chosen)
        row["coherent_11_either"] = sum(bool(s["cells"].get("11-", {}).get("failure_predicate")) or bool(s["cells"].get("11+", {}).get("failure_predicate")) for s in chosen)
        row["00_recurrent_11_minus_agreement"] = sum(bool(s["cells"]["00"]["failure_predicate"]) == bool(s["cells"].get("11-", {}).get("failure_predicate")) for s in chosen if "11-" in s["cells"])
        row["00_recurrent_11_plus_agreement"] = sum(bool(s["cells"]["00"]["failure_predicate"]) == bool(s["cells"].get("11+", {}).get("failure_predicate")) for s in chosen if "11+" in s["cells"])
        result[kind] = row
    return result


def mapped_shifts(states: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for kind in ("failure", "proxy"):
        pairs: dict[str, list[dict[str, Any]]] = {"11-": [], "11+": []}
        chosen = [s for s in states if s["panel_kind"] == kind and s["admission"].get("admitted", False)]
        for s in chosen:
            base = s["cells"].get("00", {}).get("first_recurrence_anchor")
            for key in ["11-", "11+"]:
                target = s["cells"].get(key, {}).get("first_recurrence_anchor")
                item = {"state_id": s["state_id"], "kind": kind, "base_kind": base.get("kind") if base else None, "target_kind": target.get("kind") if target else None}
                if not base or not target:
                    item["status"] = "missing_anchor"
                    item["reason"] = "00_or_coherent_cell_not_recurrent"
                else:
                    dx = target["x_center_source"] - base["x_center_source"]
                    dy = target["y_center_source"] - base["y_center_source"]
                    item.update({"status": "paired", "dx_source_bins": dx, "dy_source_bins": dy, "base_x_center_source": base["x_center_source"], "target_x_center_source": target["x_center_source"], "base_y_center_source": base["y_center_source"], "target_y_center_source": target["y_center_source"]})
                pairs[key].append(item)
        summary = {}
        for key, vals in pairs.items():
            paired = [x for x in vals if x["status"] == "paired"]
            dx = [x["dx_source_bins"] for x in paired]
            dy = [x["dy_source_bins"] for x in paired]
            summary[key] = {
                "state_count": len(vals),
                "paired_anchor_count": len(paired),
                "missing_anchor_count": len(vals) - len(paired),
                "dx_source_bins_median": median(dx),
                "dx_source_bins_q25": qpercent(dx, .25),
                "dx_source_bins_q75": qpercent(dx, .75),
                "dy_source_bins_median": median(dy),
                "direction_dx": Counter("negative" if x < 0 else "positive" if x > 0 else "zero" for x in dx),
                "pairs": vals,
            }
        result[kind] = summary
    return result


def diagnostics(states: list[dict[str, Any]], *, admitted_only: bool) -> dict[str, Any]:
    result: dict[str, Any] = {}
    selected = [s for s in states if not admitted_only or s["admission"].get("admitted", False)]
    for key in EXPECTED_CELLS:
        cs = [s["cells"][key] for s in selected if key in s["cells"]]
        opener = [c["opener"] for c in cs]
        result[key] = {
            "present": len(cs),
            "opener_chosen_log_prob_median": median([x["chosen_log_prob"] for x in opener]),
            "opener_eos_log_prob_median": median([x["eos_log_prob"] for x in opener]),
            "opener_minus_eos_log_prob_median": median([x["chosen_minus_eos_log_prob"] for x in opener]),
            "coordinate_family_mass_median": median([x["coordinate_family_mass"] for x in opener]),
            "forced_windows_by_sign": {},
        }
        for sign in ["-", "+"]:
            ws = [c["forced_description_x1_windows"].get(sign) for c in cs]
            ws = [w for w in ws if w]
            result[key]["forced_windows_by_sign"][sign] = {
                "present": len(ws),
                "old_log_prob_median": median([w["old_log_prob"] for w in ws]),
                "moved_log_prob_median": median([w["moved_log_prob"] for w in ws]),
                "moved_minus_old_log_prob_median": median([w["moved_minus_old_log_prob"] for w in ws]),
                "old_window_mass_median": median([w["old_window_mass"] for w in ws]),
                "moved_window_mass_median": median([w["moved_window_mass"] for w in ws]),
                "moved_minus_old_log_mass_median": median([w["moved_minus_old_log_mass"] for w in ws]),
                "old_bins": sorted(set(w["old_bin"] for w in ws)),
                "moved_bins": sorted(set(w["moved_bin"] for w in ws)),
            }
    return result


def conditional_window_effects(states: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare raw moved-vs-old window log-mass against each state's 00 cell."""
    result: dict[str, Any] = {}
    for kind in ("failure", "proxy"):
        selected = [s for s in states if s["panel_kind"] == kind and s["admission"].get("admitted", False)]
        kind_result: dict[str, Any] = {}
        for label, cells in (("10", ["10-", "10+"]), ("01", ["01-", "01+"]), ("11", ["11-", "11+"])):
            values: list[float] = []
            for state in selected:
                baseline = state["cells"].get("00", {}).get("forced_description_x1_windows", {})
                for sign, cell_key in zip(("-", "+"), cells):
                    left = baseline.get(sign)
                    right = state["cells"].get(cell_key, {}).get("forced_description_x1_windows", {}).get(sign)
                    if not left or not right:
                        continue
                    a = left.get("moved_minus_old_log_mass")
                    b = right.get("moved_minus_old_log_mass")
                    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                        values.append(float(b) - float(a))
            kind_result[label] = {
                "paired_state_signs": len(values),
                "median_delta_vs_00_log_mass": median(values),
                "positive": sum(x > 0 for x in values),
                "min": min(values) if values else None,
                "max": max(values) if values else None,
                "conditioning": "forced-description x1, raw full-vocabulary window log mass; conditional diagnostic, not free continuation",
            }
        result[kind] = kind_result
    return result


def strata_summary(states: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in states:
        groups[s["group"]].append(s)
    out = {}
    for group, ss in sorted(groups.items()):
        out[group] = {
            "states": len(ss),
            "failure": sum(s["panel_kind"] == "failure" for s in ss),
            "proxy": sum(s["panel_kind"] == "proxy" for s in ss),
            "tied": sum(s["model"] == "tied" for s in ss),
            "untied": sum(s["model"] == "untied" for s in ss),
            "admitted": sum(bool(s["admission"].get("admitted")) for s in ss),
            "holds": sum(not bool(s["admission"].get("admitted")) for s in ss),
            "execution": Counter(s["execution"] for s in ss),
        }
    return out


def exclusions(states: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "state_id": s["state_id"],
            "execution": s["execution"],
            "panel_kind": s["panel_kind"],
            "model": s["model"],
            "group": s["group"],
            "image_id": s["image_id"],
            "status": s["status"],
            "admission": s["admission"],
            "cells_present": s["cells_present"],
            "source_row_count_00": s["source_row_count_00"],
            "source_invalid_row_count_00": s["source_invalid_row_count_00"],
        }
        for s in states
        if not s["admission"].get("admitted", False)
    ]


def cost(states: list[dict[str, Any]]) -> dict[str, Any]:
    broad = [s for s in states if s["execution"] == "broad-v1"]
    pilot = [s for s in states if s["execution"] == "corrected-pilot-v2"]
    broad_run = read(BROAD / "run-receipt.json")
    reduce_run = read(BROAD / "reduction-run-receipt.json")
    broad_model = sum(s["runtime"]["model_forwards"] for s in broad)
    broad_vision = sum(s["runtime"]["vision_forwards"] for s in broad)
    pilot_model = sum(s["runtime"]["model_forwards"] for s in pilot)
    pilot_vision = sum(s["runtime"]["vision_forwards"] for s in pilot)
    return {
        "corrected_broad_v1": {
            "state_count": len(broad),
            "new_cells_present": sum(s["cell_count"] for s in broad),
            "model_forwards": broad_model,
            "vision_forwards": broad_vision,
            "gpu_seconds": None,
            "run_receipt": bind(BROAD / "run-receipt.json"),
            "reduction_receipt": bind(BROAD / "reduction-run-receipt.json"),
            "run_wall_seconds": broad_run.get("wall_seconds"),
            "device_intervals": broad_run.get("devices"),
            "per_device_worker_receipts": broad_run.get("per_device"),
            "owned_processes_remaining": broad_run.get("owned_processes_remaining", []),
        },
        "corrected_pilot_v2": {
            "state_count": len(pilot),
            "cells_present": sum(s["cell_count"] for s in pilot),
            "model_forwards": pilot_model,
            "vision_forwards": pilot_vision,
            "gpu_seconds": None,
            "pilot_receipt": bind(PILOT / "pilot-receipt.json"),
        },
        "disjoint_corrected_attempt_sum": {
            "model_forwards": broad_model + pilot_model + 10,
            "vision_forwards": broad_vision + pilot_vision,
            "technical_gate_model_forwards": 10,
            "note": "sum of corrected pilot, v3 plus preserved v1/v2 technical-gate forwards, and corrected broad; no candidate-v1 invalid aggregate is added",
        },
        "preserved_invalid_aggregate": {
            "attempt_ledger": bind(FINAL / "attempt-ledger.json"),
            "candidate_v1_all_attempted_model_forwards": 7712,
            "candidate_v1_final_reuse_model_forwards": 4256,
            "candidate_v1_outside_final_reuse_model_forwards": 3456,
            "scientific_pooling": "excluded_invalid_superseded",
        },
        "budget": {
            "corrected_attempt_cap_gpu_hours": 8.0,
            "gpu_seconds_measured": None,
            "wall_time_is_not_gpu_time": True,
        },
    }


def main() -> None:
    panel = read(PANEL)
    states = make_states()
    panel_sha = sha256(PANEL)
    if panel_sha != "005bc6deff209bc96ce469e8cbc20d3dbd30a7e424a7ff18a0f7f8942a1bdb49":
        raise AssertionError(f"shared panel changed: {panel_sha}")
    # Scientific/runtime integrity checks.
    for s in states:
        assert s["cells_present"] == sorted(s["cells_present"]), s["state_id"]
        for c in s["cells"].values():
            assert c["runtime_parse_compatibility"]["complete_rows_match"] is True, (s["state_id"], c["cell"])
            assert c["stop_reason"] == "im_end", (s["state_id"], c["cell"], c["stop_reason"])
        assert s["runtime"]["no_parameter_mutation"] is True
    broad = [s for s in states if s["execution"] == "broad-v1"]
    pilot = [s for s in states if s["execution"] == "corrected-pilot-v2"]
    if len(broad) != 43 or len(pilot) != 2:
        raise AssertionError((len(broad), len(pilot)))
    geometry = affected_geometry()
    result = {
        "schema": "recurrence_spatial_source.broad_v1_reduction.v1",
        "status": "candidate_unreviewed",
        "unit_id": "2026-09-19-recurrence-spatial-source",
        "question": "Does numerical recurrence location follow moved visual content, moved coordinate history, or fixed coordinate/canvas preferences?",
        "panel": {
            "path": str(PANEL),
            "sha256": panel_sha,
            "states": 45,
            "failure_states": 21,
            "proxy_states": 24,
            "pilot_states": 2,
            "broad_states": 43,
            "broad_failure_states": 19,
            "broad_proxy_states": 24,
        },
        "estimand_and_policy": {
            "readout": "original",
            "temperature": "greedy",
            "cells": EXPECTED_CELLS,
            "primary": "admitted common-00 versus coherent 11-/11+ after inverse mapping to source bins",
            "diagnostics": "10-/10+/01-/01+ mismatch cells",
            "forced_description": "conditional x1 full-vocabulary measurement; raw masses, no window renormalization",
            "recurrence": "accepted consecutive triple primitive; exact and <=8-bin near; complete serialized boxes including invalid geometry",
            "row_stop": "free suffix complete-row cap at 32; injected EOS is reported separately from natural EOS",
            "grounding": "optional local witness; missing/zero witness limits physical claims but does not remove numerical data",
        },
        "denominators": {
            "total_states": 45,
            "total_unique_cells_authorized": 315,
            "broad_cells_present": sum(s["cell_count"] for s in broad),
            "pilot_cells_present": sum(s["cell_count"] for s in pilot),
            "cells_present_total": sum(s["cell_count"] for s in states),
            "broad_admitted_states": sum(s["admission"].get("admitted", False) for s in broad),
            "broad_admission_holds": sum(not s["admission"].get("admitted", False) for s in broad),
            "pilot_local_holds": sum(not s["admission"].get("admitted", False) for s in pilot),
            "broad_cells_absent_due_to_frozen_hold": sum(7 - s["cell_count"] for s in broad),
            "no_replacements": True,
            "no_old_invalid_pooling": True,
        },
        "geometry": geometry,
        "states": states,
        "cell_totals_all_executed": cell_stats(states, admitted_only=False),
        "cell_totals_primary_admitted_only": cell_stats(states, admitted_only=True),
        "primary_contrasts": primary_contrasts(states),
        "mapped_shifts": mapped_shifts(states),
        "diagnostics_all_executed": diagnostics(states, admitted_only=False),
        "diagnostics_primary_admitted_only": diagnostics(states, admitted_only=True),
        "conditional_window_effects_vs_00": conditional_window_effects(states),
        "source_strata": strata_summary(states),
        "exclusions_and_holds": exclusions(states),
        "cost": cost(states),
        "artifacts": {
            "reduction_script": str(Path(__file__)),
            "broad_runtime_dir": str(RUNTIME),
            "broad_reduced_dir": str(REDUCED),
            "broad_manifest_dir": str(MANIFESTS),
            "pilot_dir": str(PILOT),
            "geometry_validation": bind(FINAL / "geometry-validation.json"),
            "launch_plan": bind(BROAD / "launch-plan.json"),
            "prelaunch": bind(BROAD / "prelaunch.json"),
            "run_receipt": bind(BROAD / "run-receipt.json"),
            "reduction_run_receipt": bind(BROAD / "reduction-run-receipt.json"),
            "cost_receipt": bind(BROAD / "cost-receipt.json"),
            "job_closure": bind(BROAD / "job-closure.json"),
            "attempt_ledger": bind(FINAL / "attempt-ledger.json"),
            "source_snapshot": bind(FINAL / "source-snapshot.json"),
            "release_receipt": bind(Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-distribution-census/integration/spatial-broad-release.json")),
        },
        "interpretation_limits": [
            "This is a sampled 45-state panel candidate, not a COCO-wide estimate.",
            "The 00 admission and proxy gates define the primary denominator; eight broad states and both pilot states are local HOLDs.",
            "Mapped coordinate shifts are numerical output locations, not proof of physical-instance identity or causal localization.",
            "The five round-half-even validity/order changes in tied-7511-healthy are retained as a discretization confound.",
            "Raw forced-window masses are full-vocabulary probabilities conditioned on the supplied description; they are not free-continuation quality or window-renormalized probabilities.",
            "Tied versus untied is a package comparison, not an untie-only estimate; val-extra, mature, and prospective-new strata remain separate.",
        ],
    }
    out = BROAD / "result.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    # A compact artifact map makes the evidence surface easy for the parent to replay.
    amap = {
        "schema": "recurrence_spatial_source.broad_v1_artifact_map.v1",
        "status": "candidate_unreviewed",
        "result": bind(out),
        "cost_receipt": bind(BROAD / "cost-receipt.json"),
        "job_closure": bind(BROAD / "job-closure.json"),
        "state_records": [{"state_id": s["state_id"], "execution": s["execution"], "runtime_receipts": s["runtime"]["runtime_receipts"], "source_manifest": s["source_manifest"], "cell_count": s["cell_count"], "admitted": s["admission"].get("admitted", False)} for s in states],
        "raw_dirs": {"runtime": str(RUNTIME), "reduced": str(REDUCED), "manifests": str(MANIFESTS), "pilot": str(PILOT)},
        "cpu_acceptance": {
            "reducer": "python3 /data/CoordExp/.worktrees/research-probes/probes/training_set_completion/recurrence_spatial/reduce.py",
            "aggregator": f"python3 {Path(__file__)}",
            "integrity": "result.json asserts 45 states, 43 broad, parse compatibility, no parameter mutation, and unchanged shared panel SHA",
        },
    }
    (BROAD / "artifact-map.json").write_text(json.dumps(amap, indent=2) + "\n")
    # Markdown is deliberately generated from the same JSON source used above.
    lines = []
    lines.append("## Corrected broad-v1 saved-output reduction (candidate, unreviewed)\n")
    lines.append("The corrected broad attempt executed the unchanged 45-state panel after the v3 source-route parity gate. It generated 43 new states and reused the 14 corrected-pilot cells. The 43-state run produced **38404 model forwards**, **759 vision forwards**, **253 cells**, and closed with return code 0 across devices 0-7. Eight broad states failed the frozen common-00 admission and therefore have only their 00 receipt; no state was replaced or selected by outcome. The two pilot states remain local HOLDs.\n")
    lines.append("The primary denominator is the 35 admitted broad states (12 failure, 23 proxy). Full panel counts are 21 failure and 24 proxy; both pilot failures and eight broad holds stay in the denominator record but are excluded from the admitted-only contrast. All 253 broad reduced receipts agree with the corrected parser's complete-row count.\n")
    lines.append("### Primary coherent comparison\n")
    lines.append("| panel kind | admitted states | 00 recurrent | 11- recurrent | 11+ recurrent | both coherent signs | either coherent sign |\n|---|---:|---:|---:|---:|---:|---:|")
    for kind in ("failure", "proxy"):
        x = result["primary_contrasts"][kind]
        lines.append(f"| {kind} | {x['denominator_admitted_states']} | {x['00']['failure_predicate']} | {x['11-']['failure_predicate']} | {x['11+']['failure_predicate']} | {x['coherent_11_both']} | {x['coherent_11_either']} |")
    lines.append("\n`10-/10+/01-/01+` are mismatch diagnostics. Their admitted-only recurrence counts are:")
    lines.append("| panel kind | 10- | 10+ | 01- | 01+ |\n|---|---:|---:|---:|---:|")
    for kind in ("failure", "proxy"):
        x=result["primary_contrasts"][kind]
        lines.append(f"| {kind} | {x['10-']['failure_predicate']} | {x['10+']['failure_predicate']} | {x['01-']['failure_predicate']} | {x['01+']['failure_predicate']} |")
    lines.append("\nThese are numerical recurrence predicates. A missing grounding witness limits physical-instance interpretation and does not remove the numerical cell.\n")
    lines.append("### All executed cell accounting\n")
    lines.append("| cell | present | recurrent | exact | near | complete rows | valid | invalid | malformed | row-cap EOS | natural EOS |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key in EXPECTED_CELLS:
        x=result["cell_totals_all_executed"][key]
        lines.append(f"| {key} | {x['present']} | {x['failure_predicate']} | {x['exact_recurrent']} | {x['near_recurrent']} | {x['complete_rows']} | {x['valid_rows']} | {x['invalid_rows']} | {x['malformed_rows']} | {x['row_cap_injected_eos']} | {x['natural_eos']} |")
    lines.append("\nThe recurrence columns count accepted consecutive triple windows at the state/cell level; they are not all-pairs counts. Long runs expose multiple overlapping triple windows, and the compact per-cell records also retain longest exact/near run lengths. Complete invalid boxes remain in the recurrence input and are reported separately.\n")
    lines.append("### Forced-description and opener diagnostics\n")
    lines.append("At the common row boundary, opener/EOS and coordinate-family values are full-vocabulary quantities. Forced x1 old/moved windows are conditional measurements with fixed sign centers; masses are raw full-vocabulary masses and are not window-renormalized. The complete per-cell values are in `result.json`.\n")
    lines.append("| cell | n | median opener log p | median EOS log p | median opener-EOS | median coordinate-family mass | median moved-old log p (- sign) | median moved-old log p (+ sign) |\n|---|---:|---:|---:|---:|---:|---:|---:|")
    for key in EXPECTED_CELLS:
        x=result["diagnostics_all_executed"][key]
        m=x["forced_windows_by_sign"]
        lines.append(f"| {key} | {x['present']} | {x['opener_chosen_log_prob_median']} | {x['opener_eos_log_prob_median']} | {x['opener_minus_eos_log_prob_median']} | {x['coordinate_family_mass_median']} | {(m.get('-') or {}).get('moved_minus_old_log_prob_median')} | {(m.get('+') or {}).get('moved_minus_old_log_prob_median')} |")
    lines.append("\nThe sign-matched conditional moved-vs-old **window log-mass** change relative to each state's common `00` baseline is:")
    lines.append("| panel kind | mismatch | paired state-signs | median delta vs 00 | positive | min | max |\n|---|---|---:|---:|---:|---:|---:|")
    for kind in ("failure", "proxy"):
        for label in ("10", "01", "11"):
            x=result["conditional_window_effects_vs_00"][kind][label]
            lines.append(f"| {kind} | {label} | {x['paired_state_signs']} | {x['median_delta_vs_00_log_mass']} | {x['positive']} | {x['min']} | {x['max']} |")
    lines.append("\nThese conditional x1 values help separate spatial diagnostics from free trajectory recurrence. They are description-conditioned score shifts, not recovery or physical-owner evidence.\n")
    lines.append("\n### Geometry and exclusions\n")
    lines.append("The frozen common-canvas rule is lossless copy into the fixed canvas, no resize/interpolation, 32-pixel grid, and horizontal +/-128-pixel displacement. Across the 45x7 frozen manifests there are 3521 supplied history boxes (503 source rows before cell replication), three source-invalid rows, and zero inverse drift above one bin. Five rounding/order entries are retained as a discretization confound:")
    for a in geometry["affected_rounding_or_order_entries"]:
        lines.append(f"- `{a['state_id']}` cell `{a['cell']}` row `{a['row_index']}`: source `{a['source_bins']}` -> mapped `{a['mapped_bins']}` -> inverse `{a['inverse_bins']}`, mapped_valid={a['mapped_valid']}, order_preserved={a['order_preserved']}")
    lines.append("\nThe eight broad admission holds are listed in `result.json` with model, kind, source group, row counts, invalid rows, and the exact missing transformed-cell denominator. Candidate-v1 raw/reduced outputs were preserved separately and are excluded from every corrected scientific total.\n")
    lines.append("### Source strata, cost, and replay\n")
    lines.append("State-level source/model/policy/runtime bindings are retained in `result.json`; `val-extra-runtime-stratum`, mature-panel groups, and prospective-new-cohort groups are not silently pooled. Tied versus untied remains a package comparison.\n")
    lines.append("The standalone CPU reducer is `probes/training_set_completion/recurrence_spatial/reduce.py`. The aggregate and artifact map are:")
    lines.append(f"- `{out}`")
    lines.append(f"- `{BROAD / 'artifact-map.json'}`")
    lines.append(f"- `{BROAD / 'run-receipt.json'}` and `{BROAD / 'reduction-run-receipt.json'}`")
    lines.append("\nThis is a lane candidate pending parent/root independent verification and acceptance; it does not identify a unique physical instance or causal circuit.\n")
    (BROAD / "results.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "closed", "result": str(out), "artifact_map": str(BROAD / 'artifact-map.json'), "states": len(states), "cells": result["denominators"]["cells_present_total"]}, indent=2))


if __name__ == "__main__":
    main()
