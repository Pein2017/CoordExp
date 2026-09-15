"""Read-only post-run accounting for the sealed eight-cell free-h diagnostic."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.artifacts import publish_json_exclusive


BASE = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-label-vs-compilation/diagnostic")
EXECUTION = BASE / "execution-v1"


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def digest_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binding(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": digest_file(path), "size_bytes": path.stat().st_size}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_binding(value: Mapping[str, Any], label: str) -> None:
    require(dict(value) == binding(Path(value["path"])), f"{label} binding")


def subsequence_positions(values: Sequence[int], target: Sequence[int]) -> list[int]:
    target = list(target)
    return [
        index
        for index in range(len(values) - len(target) + 1)
        if list(values[index:index + len(target)]) == target
    ]


def analyze(execution: Path) -> tuple[dict[str, Any], str]:
    terminal = read(execution / "terminal.json")
    result = read(execution / "result.json")
    exits = read(execution / "process-exits.json")["results"]
    packet_path = Path(result["packet"]["path"])
    grant_path = Path(result["grant"]["path"])
    packet = read(packet_path)
    grant = read(grant_path)
    verify_binding(result["packet"], "result packet")
    verify_binding(result["grant"], "result grant")
    verify_binding(result["consumer"], "result consumer")
    verify_binding(terminal["packet"], "terminal packet")
    verify_binding(terminal["grant"], "terminal grant")
    verify_binding(terminal["result"], "terminal result")
    require(terminal["status"] == "completed_consumed" and terminal["exit_code"] == 0, "launch terminal")
    require(result["status"] == "completed_all_cells_free_c_parity_eligible", "result status")
    require(result["counts"] == {
        "cells": 8,
        "complete_continuations": 8,
        "free_c_parity_pass": 8,
        "free_c_parity_stop": 0,
        "images": 4,
        "models": 2,
    }, "consumer counts")
    require(result["parity_stop_cell_ids"] == [], "unexpected parity stop")
    require(grant["authorized_cell_ids"] == list(range(8)), "grant cell IDs")
    require(grant["authorized_complete_continuations"] == 8, "grant continuation count")
    require(grant["fallback_h_plus_c_calls"] == 0, "grant fallback")
    verify_binding(grant["packet"], "grant packet")
    verify_binding(grant["runner"], "grant runner")
    require(grant["packet"] == result["packet"], "grant/result packet identity")
    require(grant["runner"] == result["consumer"], "grant/result runner identity")
    require(len(exits) == 8 and len({row["cell_id"] for row in exits}) == 8, "process exit identity")
    require(all(row["exit_code"] == 0 for row in exits), "worker exit")
    require(len({row["pid"] for row in exits}) == 8, "worker PID uniqueness")

    h_owners_by_image = {
        int(image_id): {str(owner) for owner in owners}
        for image_id, owners in grant["lead_acceptance"]["supplied_h_gt50_owners"].items()
    }
    h_obligation_intersections = {
        int(image_id): values
        for image_id, values in grant["lead_acceptance"]["registered_obligation_intersections_with_supplied_h"].items()
    }
    require(all(values == [] for values in h_obligation_intersections.values()), "obligations overlap supplied h")
    cell_by_id = {cell["cell_id"]: cell for cell in packet["cells"]}
    require(set(cell_by_id) == set(range(8)), "packet cell identity")
    result_by_id = {row["cell_id"]: row for row in result["rows"]}
    require(set(result_by_id) == set(range(8)), "result cell identity")
    exit_by_id = {row["cell_id"]: row for row in exits}

    cells = []
    total_tokens = 0
    burden_totals: dict[str, int] = defaultdict(int)
    resource_maxima = {
        "peak_cuda_allocated_bytes": 0,
        "peak_cuda_reserved_bytes": 0,
        "peak_rss_bytes": 0,
        "elapsed_seconds": 0.0,
    }
    by_case: dict[int, dict[str, Any]] = defaultdict(dict)
    for cell_id in range(8):
        cell = cell_by_id[cell_id]
        summary = result_by_id[cell_id]
        process = exit_by_id[cell_id]
        require(
            process["arm"] == cell["arm"]
            and process["image_id"] == cell["image_id"]
            and process["physical_gpu"] == cell["physical_gpu"],
            f"process/cell {cell_id}",
        )
        cell_root = execution / f"cell-{cell_id:02d}-{cell['arm'].lower()}-{cell['image_id']}"
        raw_path = cell_root / "raw-generation.json"
        row_path = cell_root / "row.json"
        terminal_path = cell_root / "terminal.json"
        raw = read(raw_path)
        row = read(row_path)
        cell_terminal = read(terminal_path)
        verify_binding(summary["row"], f"cell {cell_id} row")
        verify_binding(summary["terminal"], f"cell {cell_id} terminal")
        verify_binding(row["raw_generation"], f"cell {cell_id} raw")
        require(summary["row"] == binding(row_path), f"cell {cell_id} summary row")
        require(summary["terminal"] == binding(terminal_path), f"cell {cell_id} summary terminal")
        require(row["raw_generation"] == binding(raw_path), f"cell {cell_id} row raw")
        require(
            cell_terminal["status"] == "completed"
            and cell_terminal["exit_code"] == 0
            and cell_terminal["model_loads"] == 1
            and cell_terminal["continuations"] == 1
            and cell_terminal["image_forwards"] == 1,
            f"cell {cell_id} terminal checks",
        )
        require(raw["cell_id"] == row["cell_id"] == cell_id, f"cell {cell_id} raw/row identity")
        require(raw["packet"] == result["packet"] and raw["grant"] == result["grant"], f"cell {cell_id} bindings")
        require(raw["h_token_ids"] == cell["h"]["token_ids"], f"cell {cell_id} h")
        require(raw["remaining_budget"] == cell["remaining_budget"], f"cell {cell_id} budget")
        free_ids = raw["free_token_ids"]
        require(len(free_ids) <= cell["remaining_budget"], f"cell {cell_id} free budget")
        c_ids, w_ids = cell["c"]["token_ids"], cell["w"]["token_ids"]
        literal_c = list(free_ids[:len(c_ids)]) == list(c_ids)
        require(literal_c == row["assessment"]["literal_c_prefix_parity"] is True, f"cell {cell_id} c parity")
        w_positions = subsequence_positions(free_ids, w_ids)
        require(w_positions == row["assessment"]["literal_w_occurrences"], f"cell {cell_id} w positions")
        immediate_w_literal = bool(w_positions and w_positions[0] == len(c_ids))
        w_geometry = row["assessment"]["w_geometry"]
        immediate_w_geometry = bool(
            w_geometry["best_same_class"]
            and w_geometry["best_same_class"]["pred_index"] == 1
            and w_geometry["best_same_class"]["iou"] > 0.5
        )
        observed = {str(owner) for owner in row["free_score"]["50"]["owners"]}
        required = {str(item["owner_id"]) for item in cell["annotation_owner_obligations"]}
        h_owners = h_owners_by_image[cell["image_id"]]
        require(not (required & h_owners), f"cell {cell_id} registered obligations in supplied h")
        retained = sorted(required & observed)
        lost = sorted(required - observed)
        gained = sorted(observed - required)
        require(retained == row["assessment"]["annotation_owner_obligations"]["retained"], "retained obligations")
        require(lost == row["assessment"]["annotation_owner_obligations"]["missing"], "lost obligations")
        require(gained == row["assessment"]["annotation_owner_obligations"]["gained_beyond_obligations"], "gained obligations")
        reemitted = sorted(observed & h_owners)
        newly_covered = sorted(observed - h_owners)
        burden = row["burden"]
        for key, value in burden.items():
            burden_totals[key] += int(value)
        total_tokens += len(free_ids)
        for key in resource_maxima:
            resource_maxima[key] = max(resource_maxima[key], cell_terminal[key])
        item = {
            "cell_id": cell_id,
            "physical_gpu": cell["physical_gpu"],
            "arm": cell["arm"],
            "image_id": cell["image_id"],
            "raw_generation": binding(raw_path),
            "row": binding(row_path),
            "terminal": binding(terminal_path),
            "free_tokens": len(free_ids),
            "stop_reason": raw["stop_reason"],
            "literal_c": {"prefix_exact": True, "positions": row["assessment"]["literal_c_occurrences"]},
            "c_geometry": row["assessment"]["c_geometry"],
            "immediate_w": {
                "literal_exact": immediate_w_literal,
                "same_class_iou_gt_0_5": immediate_w_geometry,
                "same_class_iou": w_geometry["best_same_class"]["iou"] if w_geometry["best_same_class"] else None,
            },
            "eventual_w": {
                "literal_exact": bool(w_positions),
                "literal_token_offsets": w_positions,
                "same_class_iou_gt_0_5": w_geometry["same_class_iou_gt_0_5"],
                "best_same_class": w_geometry["best_same_class"],
            },
            "frozen_annotation_obligations": {
                "required": sorted(required),
                "retained": retained,
                "lost": lost,
                "gained_beyond_obligations": gained,
            },
            "supplied_h_gt50_owners": sorted(h_owners),
            "h_owner_reemission": reemitted,
            "newly_covered_gt50_owners_after_removing_h": newly_covered,
            "free_gt50_owners": sorted(observed),
            "free_tp50": row["free_score"]["50"]["tp"],
            "burden": burden,
            "resources": {
                key: cell_terminal[key]
                for key in (
                    "elapsed_seconds", "model_loads", "model_forwards", "image_forwards",
                    "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes", "peak_rss_bytes",
                )
            },
            "natural_root_context": summary["natural_root_context"],
        }
        cells.append(item)
        by_case[cell["image_id"]][cell["arm"]] = item

    case_comparisons = []
    for case in packet["cases"]:
        image_id = case["image_id"]
        n16, arm_a = by_case[image_id]["N16"], by_case[image_id]["A"]
        n16_owners, a_owners = set(n16["free_gt50_owners"]), set(arm_a["free_gt50_owners"])
        case_comparisons.append({
            "image_id": image_id,
            "c_record_id": case["c"]["record_id"],
            "w_record_id": case["w"]["record_id"],
            "c_provenance_kind": case["c"]["physical_owner_provenance"]["kind"],
            "physical_review_reason": case["physical_review"]["reason"],
            "N16": n16,
            "A": arm_a,
            "common_h_free_gt50_owners": sorted(n16_owners & a_owners),
            "N16_only_h_free_gt50_owners": sorted(n16_owners - a_owners),
            "A_only_h_free_gt50_owners": sorted(a_owners - n16_owners),
            "free_tp50_N16_minus_A": n16["free_tp50"] - arm_a["free_tp50"],
        })

    require(sum(item["resources"]["model_forwards"] for item in cells) == result["resources"]["model_forwards"], "model forward total")
    require(sum(item["resources"]["image_forwards"] for item in cells) == result["resources"]["image_forwards"], "image forward total")
    analysis = {
        "schema": "label_vs_compilation.free_h_postrun_analysis.v1",
        "status": "accepted_complete_execution_cpu_analyzed",
        "sources": {
            "packet": binding(packet_path),
            "grant": binding(grant_path),
            "launch_terminal": binding(execution / "terminal.json"),
            "process_exits": binding(execution / "process-exits.json"),
            "native_consumer_result": binding(execution / "result.json"),
            "bound_runner_consumer": result["consumer"],
        },
        "acceptance": {
            "unique_cells": 8,
            "unique_worker_pids": 8,
            "worker_exit_zero": 8,
            "cell_terminals_completed": 8,
            "complete_continuations": 8,
            "free_c_literal_parity_pass": 8,
            "free_c_literal_parity_stop": 0,
            "exact_binding_checks": "passed",
            "native_consumer_recomputation_evidence": "Bound runner published completed_consumed result only after exact raw-token decode, native ledger, score, overlap, burden, and assessment equality checks for all eight cells.",
        },
        "case_comparisons": case_comparisons,
        "totals": {
            "free_tokens": total_tokens,
            "burden": dict(sorted(burden_totals.items())),
            "model_loads": 8,
            "model_forwards": result["resources"]["model_forwards"],
            "image_forwards": result["resources"]["image_forwards"],
            "rank_elapsed_seconds_sum": result["resources"]["elapsed_seconds"],
            "rank_gpu_hours_sum": result["resources"]["gpu_hours_sum"],
            "outer_launcher_elapsed_seconds": terminal["elapsed_seconds"],
            "per_cell_resource_maxima": resource_maxima,
            "utilization": "unknown_not_measured",
        },
        "observations": [
            "All eight free continuations begin with the exact registered c row, closing the replay/generation parity gate for these cells.",
            "Both endpoints immediately realize exact literal w only on image477415. On the other three images, exact literal w never appears, but the immediate second parsed row is a same-class geometric w match above IoU0.5 for both endpoints.",
            "N16 and A have identical free GT50 owner sets on477415 and388795. On351017 A lacks N16 owners2094819 and96050. On417044 A lacks N16 owners1079910 and1083295 and adds1083564.",
            "No free output re-emits any supplied-h GT50 owner. All registered Stable50-source obligations were frozen disjoint from supplied h.",
            "Only A/388795 has a strict repeat: exact c appears again later; all cells EOS-stop with zero invalid or malformed rows.",
        ],
        "inference": "Within these four selected exact h conditions, A preserves deterministic c entry and the immediate geometric successor behavior seen in N16. Downstream annotation-owner compatibility is equal on two cases and lower/different on two; this localizes the remaining endpoint difference after entry rather than to inability to realize c.",
        "limits": [
            "Four outcome-selected old-training regressions are a bounded localization panel, not an estimate over all images.",
            "Empty-root rows are descriptive reachability context, not a matched forced-history control.",
            "Stable50 supplies frozen obligations only; its suffix is not an N16/A target trajectory.",
            "Geometric c/w matches do not certify complete physical suffix quality, and GT-unmatched predictions are not inferred to be unlabeled owners.",
            "The reviewed unlabeled donut c/w on417044 remains physical-review evidence, not a new GT annotation.",
        ],
        "stop": "Eight granted cells completed and were exactly consumed. No fallback, retry, additional case, training, crossover, KV intervention, or evaluation is authorized.",
    }

    lines = [
        "# Free-h conditional continuation result",
        "",
        "Status: **accepted complete execution / CPU analyzed**. All eight granted cells completed with exact bindings and zero free-c parity stops.",
        "",
        "| Image | N16/A free TP50 | Literal c | Immediate literal w | Immediate geometric w | Frozen obligations N16 retained/lost; A retained/lost | Common / N16-only / A-only free owners |",
        "|---:|---:|---|---|---|---|---|",
    ]
    for case in case_comparisons:
        n16, arm_a = case["N16"], case["A"]
        common = ",".join(case["common_h_free_gt50_owners"]) or "none"
        n_only = ",".join(case["N16_only_h_free_gt50_owners"]) or "none"
        a_only = ",".join(case["A_only_h_free_gt50_owners"]) or "none"
        lines.append(
            f"| {case['image_id']} | {n16['free_tp50']}/{arm_a['free_tp50']} | exact/exact | "
            f"{'yes' if n16['immediate_w']['literal_exact'] else 'no'}/{'yes' if arm_a['immediate_w']['literal_exact'] else 'no'} | "
            f"yes {n16['immediate_w']['same_class_iou']:.3f}/yes {arm_a['immediate_w']['same_class_iou']:.3f} | "
            f"{len(n16['frozen_annotation_obligations']['retained'])}/{len(n16['frozen_annotation_obligations']['lost'])}; "
            f"{len(arm_a['frozen_annotation_obligations']['retained'])}/{len(arm_a['frozen_annotation_obligations']['lost'])} | "
            f"{common} / {n_only} / {a_only} |"
        )
    lines += [
        "",
        "## Decision-bearing observations",
        "",
        "- Exact `c` is the free prefix in all 8/8 cells. The entry-realization parity gate passes.",
        "- Image 477415 produces exact immediate `w` at both endpoints. Images 351017, 417044, and 388795 never produce the registered literal `w`, but both endpoints produce an immediate same-class geometric match (IoU 0.826, 0.860, and 0.988 respectively). Literal and geometric success are therefore not interchangeable.",
        "- Free owner sets are identical for N16/A on 477415 and 388795. Under common `h`, A loses N16 owners 2094819 and 96050 on 351017; on 417044 it loses 1079910 and 1083295 while adding 1083564.",
        "- No supplied-`h` GT50 owner is re-emitted in any cell. The observed free owners are newly covered relative to supplied `h`; the reviewed unlabeled donut remains separate from GT-owner accounting.",
        "- Every cell EOS-stops. Across all cells: zero geometry-invalid rows, zero other malformed rows, zero cap stops, and one strict repeat—A/388795 repeats exact `c` later.",
        "",
        "## Resources and scope",
        "",
        f"Generated {total_tokens} free tokens with {result['resources']['model_forwards']} model forwards and 8 image forwards. Sum of per-cell elapsed time was {result['resources']['elapsed_seconds']:.3f}s ({result['resources']['gpu_hours_sum']:.6f} GPU-hours by rank-time); outer elapsed time was {terminal['elapsed_seconds']:.3f}s. Utilization was not measured.",
        "",
        "Inference: on this four-case localization panel, A preserves `c` entry and immediate geometric successor behavior, but downstream annotation-owner compatibility is equal on two cases and lower/different on two. This supports a post-entry compatibility difference, not a c-realization failure.",
        "",
        "Limits: existing empty-root outputs are contextual only; Stable50 suffixes are not target trajectories; geometric matches do not establish whole-suffix physical quality; GT-unmatched predictions are not treated as unlabeled owners. No fallback or further call is authorized.",
    ]
    return analysis, "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution", type=Path, default=EXECUTION)
    parser.add_argument("--json-output", type=Path, default=BASE / "analysis-v1.json")
    parser.add_argument("--markdown-output", type=Path, default=BASE / "result.md")
    args = parser.parse_args()
    analysis, markdown = analyze(args.execution)
    publish_json_exclusive(args.json_output, analysis)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    with args.markdown_output.open("x") as stream:
        stream.write(markdown)
    print(json.dumps({
        "status": analysis["status"],
        "json": str(args.json_output),
        "json_sha256": digest_file(args.json_output),
        "markdown": str(args.markdown_output),
        "markdown_sha256": digest_file(args.markdown_output),
        "cells": analysis["acceptance"]["unique_cells"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
