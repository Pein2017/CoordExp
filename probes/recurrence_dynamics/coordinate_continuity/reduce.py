"""CPU-only reduction of the saved coordinate-continuity captures.

This reducer never loads a model.  It compares each saved +/-1 capture with
the native capture for the same frozen state, at both score sites, and writes
the complete contrast ledger plus compact scientific summaries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


COORD_BASE = 151670
COORD_COUNT = 1000
PARITY_ATOL = 2e-4
SITE_NAMES = ("immediate", "later")
MODES = ("native_no_hook", "observational_hook")


def binding(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def panel_boundaries(panel: dict[str, Any]) -> dict[str, dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for key in ("boundaries", "existing_boundaries", "new_boundaries"):
        if isinstance(panel.get(key), list):
            values.extend(x for x in panel[key] if isinstance(x, dict))
    return {
        str(x.get("id", x.get("boundary_id"))): x
        for x in values
        if x.get("id", x.get("boundary_id")) is not None
    }


def source_panel_for(boundary: dict[str, Any]) -> Path:
    return Path(boundary["raw_path"]).resolve().parents[3] / "panel.json"


def safe_float(value: Any) -> float:
    return float(value)


def vector_stats(delta: torch.Tensor) -> dict[str, float]:
    x = delta.detach().double().flatten()
    norm = float(torch.linalg.vector_norm(x))
    denom = float(torch.linalg.vector_norm(x))
    return {
        "l2": norm,
        "mean_abs": float(x.abs().mean()),
        "max_abs": float(x.abs().max()),
        "nonzero": int(torch.count_nonzero(x)),
        "cosine_self": 1.0 if denom else 0.0,
    }


def difference_stats(value: torch.Tensor, native: torch.Tensor) -> dict[str, float]:
    delta = value.detach().double() - native.detach().double()
    value_d = value.detach().double().flatten()
    native_d = native.detach().double().flatten()
    delta_norm = float(torch.linalg.vector_norm(delta))
    value_norm = float(torch.linalg.vector_norm(value_d))
    native_norm = float(torch.linalg.vector_norm(native_d))
    cosine = 0.0
    if value_norm and native_norm:
        cosine = float(torch.dot(value_d, native_d) / (value_norm * native_norm))
    return {
        "l2": delta_norm,
        "mean_abs": float(delta.abs().mean()),
        "max_abs": float(delta.abs().max()),
        "cosine_to_native": cosine,
        "native_l2": native_norm,
        "value_l2": value_norm,
    }


def rank_of(logits: torch.Tensor, token_id: int) -> int:
    score = logits[int(token_id)].double()
    return 1 + int((logits.double() > score).sum())


def js_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    p = p.double()
    q = q.double()
    m = (p + q) / 2
    eps = torch.finfo(torch.float64).tiny
    return float(
        0.5 * (p * ((p + eps).log() - (m + eps).log())).sum()
        + 0.5 * (q * ((q + eps).log() - (m + eps).log())).sum()
    )


def coordinate_distribution(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits.detach().double(), dim=-1)


def site_metrics(native: dict[str, Any], variant: dict[str, Any], site_index: int) -> dict[str, Any]:
    n_logits = native["logits"][site_index].detach().double()
    v_logits = variant["logits"][site_index].detach().double()
    n_coord = native["coordinate_logits"][site_index].detach().double()
    v_coord = variant["coordinate_logits"][site_index].detach().double()
    n_coord_center = n_coord - n_coord.mean()
    v_coord_center = v_coord - v_coord.mean()
    coordinate_centered_delta = v_coord_center - n_coord_center
    n_full_center = n_logits - n_logits.mean()
    v_full_center = v_logits - v_logits.mean()
    full_vocab_centered_delta = v_full_center - n_full_center
    n_top = torch.topk(n_logits, 2)
    v_top = torch.topk(v_logits, 2)
    n_coord_top = torch.topk(n_coord, 2)
    v_coord_top = torch.topk(v_coord, 2)
    old_token = int(native["variant"]["token_id"])
    new_token = int(variant["variant"]["token_id"])
    old_index = old_token - COORD_BASE
    new_index = new_token - COORD_BASE
    native_full_winner = int(n_top.indices[0])
    native_full_winner_coord_index = native_full_winner - COORD_BASE
    native_full_winner_is_coordinate = 0 <= native_full_winner_coord_index < COORD_COUNT
    n_coord_prob = coordinate_distribution(n_coord)
    v_coord_prob = coordinate_distribution(v_coord)
    native_margin = float(n_top.values[0] - n_top.values[1])
    pair_native = float(n_coord[old_index] - n_coord[new_index])
    pair_variant = float(v_coord[old_index] - v_coord[new_index])
    coordinate_centered_linf = float(coordinate_centered_delta.abs().max())
    full_vocab_centered_linf = float(full_vocab_centered_delta.abs().max())
    return {
        "position": int(native["positions"][site_index]),
        "native_full_winner": native_full_winner,
        "variant_full_winner": int(v_top.indices[0]),
        "full_winner_changed": int(n_top.indices[0]) != int(v_top.indices[0]),
        "native_margin": native_margin,
        "variant_margin": float(v_top.values[0] - v_top.values[1]),
        "margin_delta": float((v_top.values[0] - v_top.values[1]) - (n_top.values[0] - n_top.values[1])),
        "native_winner_rank_under_variant": rank_of(v_logits, int(n_top.indices[0])),
        "variant_winner_rank_under_native": rank_of(n_logits, int(v_top.indices[0])),
        "replacement_rank_under_native": rank_of(n_logits, new_token),
        "replacement_rank_under_variant": rank_of(v_logits, new_token),
        "native_coordinate_winner": COORD_BASE + int(n_coord_top.indices[0]),
        "variant_coordinate_winner": COORD_BASE + int(v_coord_top.indices[0]),
        "coordinate_winner_changed": int(n_coord_top.indices[0]) != int(v_coord_top.indices[0]),
        "native_coordinate_margin": float(n_coord_top.values[0] - n_coord_top.values[1]),
        "variant_coordinate_margin": float(v_coord_top.values[0] - v_coord_top.values[1]),
        "coordinate_margin_delta": float((v_coord_top.values[0] - v_coord_top.values[1]) - (n_coord_top.values[0] - n_coord_top.values[1])),
        "native_coordinate_winner_rank_under_variant": rank_of(v_coord, int(n_coord_top.indices[0])),
        "variant_coordinate_winner_rank_under_native": rank_of(n_coord, int(v_coord_top.indices[0])),
        "replacement_coordinate_rank_under_native": rank_of(n_coord, new_index),
        "replacement_coordinate_rank_under_variant": rank_of(v_coord, new_index),
        # Existing centered_* fields are deliberately coordinate-only for
        # backwards compatibility.  Full-vocabulary fields below use the
        # complete logits vector and have an explicit prefix.
        "centered_logit_scope": "coordinate_logits_only_1000",
        "centered_logit_change_l2": float(torch.linalg.vector_norm(coordinate_centered_delta)),
        "centered_logit_change_linf": coordinate_centered_linf,
        "centered_logit_change_mean_abs": float(coordinate_centered_delta.abs().mean()),
        "centered_change_over_native_margin": coordinate_centered_linf / max(abs(native_margin), 1e-12),
        "full_vocab_centered_logit_scope": "full_logits_vocabulary",
        "full_vocab_centered_logit_change_l2": float(torch.linalg.vector_norm(full_vocab_centered_delta)),
        "full_vocab_centered_logit_change_linf": full_vocab_centered_linf,
        "full_vocab_centered_logit_change_mean_abs": float(full_vocab_centered_delta.abs().mean()),
        "full_vocab_centered_change_over_native_margin": full_vocab_centered_linf / max(abs(native_margin), 1e-12),
        "native_full_winner_is_coordinate": native_full_winner_is_coordinate,
        "centered_change_at_native_winner": (
            float(coordinate_centered_delta[native_full_winner_coord_index])
            if native_full_winner_is_coordinate
            else None
        ),
        "centered_change_at_replacement": float(coordinate_centered_delta[new_token - COORD_BASE]),
        "full_vocab_centered_change_at_native_winner": float(full_vocab_centered_delta[native_full_winner]),
        "full_vocab_centered_change_at_replacement": float(full_vocab_centered_delta[new_token]),
        "coordinate_pair_margin_native": pair_native,
        "coordinate_pair_margin_variant": pair_variant,
        "coordinate_pair_margin_delta": pair_variant - pair_native,
        "coordinate_distribution_l1": float((v_coord_prob - n_coord_prob).abs().sum()),
        "coordinate_distribution_js": js_divergence(n_coord_prob, v_coord_prob),
        "coordinate_distribution_scope": "softmax_over_coordinate_family_only_1000",
        "native_coordinate_family_probability": float(native["coordinate_family_probability"][site_index]),
        "variant_coordinate_family_probability": float(variant["coordinate_family_probability"][site_index]),
        "coordinate_family_probability_delta": float(variant["coordinate_family_probability"][site_index] - native["coordinate_family_probability"][site_index]),
        "native_eos_probability": float(native["eos_probability"][site_index]),
        "variant_eos_probability": float(variant["eos_probability"][site_index]),
        "eos_probability_delta": float(variant["eos_probability"][site_index] - native["eos_probability"][site_index]),
    }


def layer_metrics(native_hook: dict[str, Any], variant_hook: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for layer in sorted(native_hook["layer_inputs"], key=int):
        layer_out: dict[str, Any] = {}
        for site_index, site_name in enumerate(SITE_NAMES):
            layer_out[site_name] = {
                "input": difference_stats(variant_hook["layer_inputs"][layer][site_index], native_hook["layer_inputs"][layer][site_index]),
                "output": difference_stats(variant_hook["layer_outputs"][layer][site_index], native_hook["layer_outputs"][layer][site_index]),
                "residual": difference_stats(variant_hook["layer_residuals"][layer][site_index], native_hook["layer_residuals"][layer][site_index]),
            }
        output[layer] = layer_out
    return output


def mean_or_none(values: list[float]) -> float | None:
    return float(statistics.mean(values)) if values else None


def median_or_none(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def summarize(rows: list[dict[str, Any]], site_name: str | None = None) -> dict[str, Any]:
    selected_site_names = (site_name,) if site_name else SITE_NAMES
    sites = [row["sites"][name] for row in rows for name in selected_site_names]
    def vals(key: str) -> list[float]:
        return [float(site[key]) for site in sites]
    return {
        "n_contrasts": len(rows),
        "n_sites": len(sites),
        "input_delta_l2_median": median_or_none([float(row["input_delta"]["l2"]) for row in rows]),
        "input_delta_l2_mean": mean_or_none([float(row["input_delta"]["l2"]) for row in rows]),
        "centered_change_over_native_margin_median": median_or_none(vals("centered_change_over_native_margin")),
        "coordinate_pair_margin_delta_median": median_or_none(vals("coordinate_pair_margin_delta")),
        "coordinate_distribution_l1_median": median_or_none(vals("coordinate_distribution_l1")),
        "coordinate_distribution_js_median": median_or_none(vals("coordinate_distribution_js")),
        "centered_logit_change_linf_median": median_or_none(vals("centered_logit_change_linf")),
        "native_winner_rank_under_variant_median": median_or_none(vals("native_winner_rank_under_variant")),
        "replacement_rank_under_variant_median": median_or_none(vals("replacement_rank_under_variant")),
        "full_vocab_centered_change_over_native_margin_median": median_or_none(vals("full_vocab_centered_change_over_native_margin")),
        "full_vocab_centered_logit_change_linf_median": median_or_none(vals("full_vocab_centered_logit_change_linf")),
        "full_winner_changed_sites": int(sum(bool(site["full_winner_changed"]) for site in sites)),
        "coordinate_winner_changed_sites": int(sum(bool(site["coordinate_winner_changed"]) for site in sites)),
        "eos_probability_delta_median": median_or_none(vals("eos_probability_delta")),
        "coordinate_family_probability_delta_median": median_or_none(vals("coordinate_family_probability_delta")),
        "head_delta_l2_median": median_or_none([float(row["head_delta"][name]["l2"]) for row in rows for name in selected_site_names]),
        "norm_delta_l2_median": median_or_none([float(row["norm_delta"][name]["l2"]) for row in rows for name in selected_site_names]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    root = args.output_root
    plan_path = root / "execution-plan.json"
    panel_path = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-distribution-census/shared-panel.json")
    plan = read_json(plan_path)
    panel = read_json(panel_path)
    boundaries = panel_boundaries(panel)
    contrasts: list[dict[str, Any]] = []
    verification_errors: list[str] = []
    native_cache: dict[str, dict[str, Any]] = {}
    for state in plan["states"]:
        sid = state["id"]
        state_root = root / "runtime" / sid
        release = read_json(state_root / "release.json")
        if release.get("status") != "candidate_complete":
            verification_errors.append(f"{sid}: release status")
            continue
        boundary = boundaries[state["boundary_id"]]
        source_panel = source_panel_for(boundary)
        source_name = source_panel.parent.name
        native_hook = torch.load(state_root / "native" / "observational_hook" / "capture.pt", map_location="cpu", weights_only=False)
        native_no_hook = torch.load(state_root / "native" / "native_no_hook" / "capture.pt", map_location="cpu", weights_only=False)
        native_cache[sid] = native_no_hook
        if not torch.equal(native_hook["logits"], native_no_hook["logits"]):
            verification_errors.append(f"{sid}: native hook mismatch")
        for variant in state["variants"]:
            if variant["name"] == "native":
                continue
            vroot = state_root / variant["name"]
            v_hook = torch.load(vroot / "observational_hook" / "capture.pt", map_location="cpu", weights_only=False)
            v_no_hook = torch.load(vroot / "native_no_hook" / "capture.pt", map_location="cpu", weights_only=False)
            if not torch.equal(v_hook["logits"], v_no_hook["logits"]):
                verification_errors.append(f"{sid}/{variant['name']}: hook mismatch")
            input_delta = v_no_hook["input_delta"].detach().double()
            row = {
                "state_id": sid,
                "boundary_id": state["boundary_id"],
                "model": state["model"],
                "stratum": state["stratum"],
                "source_panel": source_name,
                "source_panel_path": str(source_panel),
                "group": state["group"],
                "image_id": int(state["image_id"]),
                "role": state["site"]["role"],
                "source_offset": int(state["site"]["offset"]),
                "source_value": int(state["site"]["value"]),
                "replacement_value": int(variant["values"][state["site"]["role_index"]]),
                "delta": int(variant["delta"]),
                "geometry_valid": bool(variant["geometry_valid"]),
                "order_valid": bool(variant["order_valid"]),
                "input_delta": vector_stats(input_delta),
                "sites": {
                    name: site_metrics(native_no_hook, v_no_hook, index)
                    for index, name in enumerate(SITE_NAMES)
                },
                "norm_delta": {
                    name: difference_stats(v_hook["norm_inputs"][index], native_hook["norm_inputs"][index])
                    for index, name in enumerate(SITE_NAMES)
                },
                "head_delta": {
                    name: difference_stats(v_hook["head_inputs"][index], native_hook["head_inputs"][index])
                    for index, name in enumerate(SITE_NAMES)
                },
                "layers": layer_metrics(native_hook, v_hook),
            }
            contrasts.append(row)
    if len(contrasts) != 32:
        verification_errors.append(f"expected 32 +/-1 contrasts, got {len(contrasts)}")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    role_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in contrasts:
        groups[f"{row['stratum']}|{row['model']}"] .append(row)
        source_groups[row["source_panel"]].append(row)
        role_groups[row["role"]].append(row)
    summary = {
        "by_stratum_model": {key: summarize(rows) for key, rows in sorted(groups.items())},
        "by_source_panel": {key: summarize(rows) for key, rows in sorted(source_groups.items())},
        "by_role": {key: summarize(rows) for key, rows in sorted(role_groups.items())},
        "by_stratum_model_site": {
            key: {site_name: summarize(rows, site_name) for site_name in SITE_NAMES}
            for key, rows in sorted(groups.items())
        },
        "by_source_panel_site": {
            key: {site_name: summarize(rows, site_name) for site_name in SITE_NAMES}
            for key, rows in sorted(source_groups.items())
        },
        "by_role_site": {
            key: {site_name: summarize(rows, site_name) for site_name in SITE_NAMES}
            for key, rows in sorted(role_groups.items())
        },
    }
    # Layer-wise summaries retain the mechanism-facing path without copying
    # every tensor into a second artifact.
    layer_summary: dict[str, Any] = {}
    for key, rows in {**groups, **source_groups}.items():
        per_layer: dict[str, Any] = {}
        for layer in sorted(contrasts[0]["layers"], key=int):
            per_layer[layer] = {}
            for site_name in SITE_NAMES:
                per_layer[layer][site_name] = {}
                for branch in ("input", "output", "residual"):
                    values = [float(row["layers"][layer][site_name][branch]["l2"]) for row in rows]
                    per_layer[layer][site_name][f"{branch}_delta_l2_median"] = median_or_none(values)
                    per_layer[layer][site_name][f"{branch}_delta_l2_mean"] = mean_or_none(values)
        layer_summary[key] = per_layer
    closure_path = root / "runtime" / "closure.json"
    cpu_path = root / "cpu-geometry.json"
    closure = read_json(closure_path)
    result = {
        "schema": "coordinate_continuity.reduction.v1",
        "status": "candidate" if not verification_errors else "technical_invalid",
        "plan": binding(plan_path),
        "panel": binding(panel_path),
        "closure": binding(closure_path),
        "cpu_geometry": binding(cpu_path),
        "producer": binding(Path(__file__)),
        "contrast_count": len(contrasts),
        "site_count": len(contrasts) * 2,
        "sites": list(SITE_NAMES),
        "contrast_definition": "Each row is a saved +/-1 variant minus its same-state native capture; legacy centered_logit_change_* fields subtract the mean over the 1000 coordinate logits, full_vocab_centered_logit_change_* fields subtract the mean over the complete vocabulary logits, coordinate distributions are softmaxes over the 1000-coordinate family only, and ranks use strict greater-logit counts plus one.",
        "centered_metric_scopes": {
            "coordinate": "legacy centered_logit_change_* and centered_change_over_native_margin fields: coordinate_logits only (1000 values)",
            "full_vocabulary": "full_vocab_centered_logit_change_* and full_vocab_centered_change_over_native_margin fields: complete logits vector",
            "coordinate_distribution": "coordinate_distribution_l1/js: softmax over coordinate_logits only (1000 values)",
        },
        "contrasts": contrasts,
        "summary": summary,
        "layer_summary": layer_summary,
        "verification": {
            "errors": verification_errors,
            "native_hook_and_no_hook_exact": not verification_errors,
            "all_saved_states": len(contrasts) == 32,
        },
    }
    out = args.out or (root / "reduction.json")
    write_json(out, result)
    verification = {
        "schema": "coordinate_continuity.reduction_verification.v1",
        "status": result["status"],
        "reduction": binding(out),
        "contrast_count": len(contrasts),
        "site_count": len(contrasts) * 2,
        "errors": verification_errors,
        "saved_tensor_only": True,
        "model_calls": 0,
        "centered_metric_scopes": result["centered_metric_scopes"],
        "full_vocab_metrics_present": all(
            "full_vocab_centered_logit_change_linf" in site
            and "full_vocab_centered_change_over_native_margin" in site
            for row in contrasts
            for site in row["sites"].values()
        ),
    }
    write_json(root / "verification.json", verification)
    compact = {
        "schema": "coordinate_continuity.result.v1",
        "status": result["status"],
        "reduction": binding(out),
        "verification": binding(root / "verification.json"),
        "closure": binding(closure_path),
        "contrast_count": len(contrasts),
        "sites": len(contrasts) * 2,
        "summary": summary,
        "role_scope": {key: value["n_contrasts"] for key, value in summary["by_role"].items()},
        "source_strata": {key: value["n_contrasts"] for key, value in summary["by_source_panel"].items()},
        "runtime_observation": {
            "planned_states": closure.get("denominator", {}).get("planned_states"),
            "completed_states": closure.get("completed_state_count"),
            "candidate_forwards": closure.get("candidate_forwards"),
            "failed_attempt_forwards": closure.get("failed_attempt_forwards"),
            "total_model_forwards_observed": closure.get("total_model_forwards_observed"),
            "source_parity": closure.get("source_parity"),
            "hook_parity": closure.get("hook_parity"),
            "input_mutation": closure.get("input_mutation"),
            "source_plan_correction_count": closure.get("source_plan_correction_count"),
            "cost": closure.get("cost"),
        },
        "limits": [
            "The panel is a frozen selected failure/proxy cohort; results are conditional and not prevalence estimates.",
            "All selected sites are y2 under the latest-coordinate rule; no other role was swept.",
            "A one-bin input edit changes context; centered-logit and layer responses do not identify a causal module or physical object identity.",
            "Companion outputs are not interpreted as independent outcomes.",
        ],
    }
    write_json(root / "result.json", compact)
    print(json.dumps({"status": result["status"], "contrasts": len(contrasts), "sites": len(contrasts) * 2, "errors": verification_errors}, indent=2))


if __name__ == "__main__":
    main()
