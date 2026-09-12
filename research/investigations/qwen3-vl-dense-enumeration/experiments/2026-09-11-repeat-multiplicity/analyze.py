"""CPU-only closeout reducer and first-free-row overlays."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


HERE = Path(__file__).resolve().parent
R = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-repeat-multiplicity")
PACKET = R / "packet-v2.json"
EXPECTED_PACKET_SHA = "26f16b3074456d03556d525b8ca2f6cc7a4187ab8d1a7c4d1a60d65a0af2e7b4"
RUNS = {
    "9813": R / "smoke" / "case-9813",
    "417044": R / "full" / "case-417044",
}


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def publish(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical_bytes(value))


def load_runner() -> Any:
    spec = importlib.util.spec_from_file_location("repeat_multiplicity_runner", HERE / "run_probe.py")
    require(spec is not None and spec.loader is not None, "cannot load runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def render(case: dict[str, Any], conditional: list[dict[str, Any]], target: Path) -> None:
    source = Image.open(case["source_case"]["image_path"]).convert("RGB")
    font = ImageFont.load_default()
    panels = []
    for row in conditional:
        image = source.copy()
        draw = ImageDraw.Draw(image)
        for label, color in (("A", "#ff3030"), ("B", "#00a0ff")):
            box = case["candidates"][label]["coord_bins"]
            pixels = runner.coord_bins_to_pixel_xyxy(box, image_width=image.width, image_height=image.height, field=label)
            draw.rectangle(pixels, outline=color, width=max(3, min(image.size) // 250))
        first = row["free_behavior"]["first_complete_free_row"]
        if first is not None and first["pixel_box_xyxy"] is not None:
            draw.rectangle(first["pixel_box_xyxy"], outline="#ff00ff", width=max(4, min(image.size) // 180))
        counts = {label: row["prefix_row_labels"].count(label) for label in "AB"}
        title = f"{row['cell_id']} A:B={counts['A']}:{counts['B']}  logodds={row['scores']['A_minus_B']['sum_log_odds']:.3f}"
        subtitle = f"first={None if first is None else first['coord_bins']}  rows={row['free_behavior']['complete_free_rows']} {row['stop_reason']}"
        fitted = ImageOps.contain(image, (520, 520))
        panel = Image.new("RGB", (540, 570), "white")
        panel.paste(fitted, ((540 - fitted.width) // 2, 45))
        pd = ImageDraw.Draw(panel)
        pd.text((8, 6), title, fill="black", font=font)
        pd.text((8, 22), subtitle, fill="black", font=font)
        panels.append(panel)
    canvas = Image.new("RGB", (1620, 1140), "white")
    for index, panel in enumerate(panels):
        canvas.paste(panel, ((index % 3) * 540, (index // 3) * 570))
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target)


def main() -> int:
    require(file_hash(PACKET) == EXPECTED_PACKET_SHA, "packet-v2 identity changed")
    packet = json.loads(PACKET.read_text())
    require(packet["admitted_case_ids"] == ["9813", "417044"], "admitted set changed")
    cases = {case["case_id"]: case for case in packet["cases"]}
    case_results = []
    source_files = {str(PACKET): file_hash(PACKET), str(HERE / "analyze.py"): file_hash(HERE / "analyze.py"), str(HERE / "run_probe.py"): file_hash(HERE / "run_probe.py")}
    costs = {key: 0 for key in ("model_loads", "generation_calls", "score_replays", "generated_free_tokens", "model_forwards", "image_forwards")}
    costs["summed_worker_wall_seconds"] = 0.0
    peaks = {"cuda_allocated_bytes": 0, "cuda_reserved_bytes": 0, "rss_bytes": 0}
    for case_id, run_dir in RUNS.items():
        terminal_path, records_path, readback_path = (run_dir / name for name in ("terminal.json", "records.jsonl", "readback.json"))
        terminal = json.loads(terminal_path.read_text())
        readback = json.loads(readback_path.read_text())
        rows = load_rows(records_path)
        require(terminal["status"] == "complete" and terminal["exit_code"] == 0 and terminal["packet_sha256"] == EXPECTED_PACKET_SHA, f"{case_id}: terminal")
        require(readback["records"] == len(rows) == 7 and readback["conditional_cells"] == 6 and readback["score_replays"] == 24, f"{case_id}: cold readback")
        require(rows[0]["kind"] == "natural_anchor" and rows[0]["exact_saved_anchor_match"] is True, f"{case_id}: anchor parity")
        conditional = rows[1:]
        case = cases[case_id]
        require([row["cell_id"] for row in conditional] == [cell["cell_id"] for cell in case["cells"]], f"{case_id}: cell identity")
        for row in conditional:
            recomputed = runner.free_behavior(row["prefix_ids"], row["free_ids"], case["candidates"], row["stop_reason"], image_width=case["source_case"]["image_width"], image_height=case["source_case"]["image_height"])
            require(recomputed == row["free_behavior"], f"{case_id}/{row['cell_id']}: free behavior changed")
            require(abs((row["scores"]["A"]["sum_logprob"] - row["scores"]["B"]["sum_logprob"]) - row["scores"]["A_minus_B"]["sum_log_odds"]) < 1e-12, f"{case_id}/{row['cell_id']}: score difference")
        by_cell = {row["cell_id"]: row for row in conditional}
        pairs = {
            "Aheavy_5_3": ("a5_b3_fwd", "a5_b3_rev"),
            "balanced_4_4": ("a4_b4_fwd", "a4_b4_rev"),
            "Bheavy_3_5": ("a3_b5_fwd", "a3_b5_rev"),
        }
        pair_stats = {}
        for label, ids in pairs.items():
            values = [by_cell[cell]["scores"]["A_minus_B"]["sum_log_odds"] for cell in ids]
            pair_stats[label] = {"cell_ids": list(ids), "A_minus_B_sum_logodds": values, "mean": sum(values) / 2, "absolute_reversal_gap": abs(values[0] - values[1])}
        means = [pair_stats[label]["mean"] for label in pairs]
        overlay = R / "free-row-overlays" / f"case-{case_id}-first-free.png"
        render(case, conditional, overlay)
        first_rows = [row["free_behavior"]["first_complete_free_row"] for row in conditional]
        case_results.append({
            "case_id": case_id,
            "candidate_description": case["candidates"]["A"]["description"],
            "pair_stats": pair_stats,
            "pair_means_in_Aheavy_balanced_Bheavy_order": means,
            "monotone_increasing_as_B_multiplicity_increases": means[0] < means[1] < means[2],
            "registered_direct_count_direction_Aheavy_gt_balanced_gt_Bheavy": means[0] > means[1] > means[2],
            "first_free_exact_A_or_B_cells": sum(bool(first and set(first["exact_candidate_labels"]) & {"A", "B"}) for first in first_rows),
            "first_free_native_pixel_A_or_B_cells": sum(bool(first and set(first["native_pixel_iou_gt_0_95_candidate_labels_same_description"]) & {"A", "B"}) for first in first_rows),
            "all_free_native_pixel_A_or_B_incidences": {label: sum(row["free_behavior"]["native_pixel_iou_gt_0_95_candidate_recurrences_same_description"][label] for row in conditional) for label in ("A", "B")},
            "free_complete_rows": sum(row["free_behavior"]["complete_free_rows"] for row in conditional),
            "native_pixel_class_blind_strict_repeat_rows": sum(row["free_behavior"]["native_pixel_class_blind_strict_repeat_rows_against_prefix_and_prior_free"] for row in conditional),
            "geometry_invalid_complete_free_rows": sum(row["free_behavior"]["geometry_invalid_complete_free_rows"] for row in conditional),
            "parser_drops": sum(row["parsed_free"]["dropped_prediction_count"] for row in conditional),
            "eos_cells": sum(row["stop_reason"] == "im_end" for row in conditional),
            "horizon_cells": sum(row["stop_reason"] == "length" for row in conditional),
            "overlay": {"path": str(overlay), "sha256": file_hash(overlay)},
        })
        for key in ("model_loads", "generation_calls", "score_replays", "generated_free_tokens", "model_forwards", "image_forwards"):
            costs[key] += terminal[key]
        costs["summed_worker_wall_seconds"] += terminal["elapsed_seconds"]
        peaks["cuda_allocated_bytes"] = max(peaks["cuda_allocated_bytes"], terminal["peak_cuda_allocated_bytes"])
        peaks["cuda_reserved_bytes"] = max(peaks["cuda_reserved_bytes"], terminal["peak_cuda_reserved_bytes"])
        peaks["rss_bytes"] = max(peaks["rss_bytes"], terminal["peak_rss_bytes"])
        for path in (terminal_path, records_path, readback_path, run_dir / "launch.json", run_dir / "config.json", run_dir / "model.json", run_dir / "batch.json"):
            source_files[str(path)] = file_hash(path)
    costs["summed_worker_wall_gpu_hours"] = costs["summed_worker_wall_seconds"] / 3600
    result = {
        "schema": "repeat_multiplicity.reduction.v1",
        "status": "candidate_complete_pending_root_acceptance",
        "scope": "Two visually admitted hand-selected cases; synthetic fixed complete-row prefixes; exact-row scores and greedy free continuations only",
        "cases": case_results,
        "aggregate": {
            "admitted_cases": 2,
            "conditional_cells": 12,
            "cases_with_inverse_pair_mean_order": sum(case["monotone_increasing_as_B_multiplicity_increases"] for case in case_results),
            "cases_supporting_registered_direct_count_direction": sum(case["registered_direct_count_direction_Aheavy_gt_balanced_gt_Bheavy"] for case in case_results),
            "first_free_exact_A_or_B_cells": sum(case["first_free_exact_A_or_B_cells"] for case in case_results),
            "first_free_native_pixel_A_or_B_cells": sum(case["first_free_native_pixel_A_or_B_cells"] for case in case_results),
            "all_free_native_pixel_A_incidences": sum(case["all_free_native_pixel_A_or_B_incidences"]["A"] for case in case_results),
            "all_free_native_pixel_B_incidences": sum(case["all_free_native_pixel_A_or_B_incidences"]["B"] for case in case_results),
            "free_complete_rows": sum(case["free_complete_rows"] for case in case_results),
            "native_pixel_class_blind_strict_repeat_rows": sum(case["native_pixel_class_blind_strict_repeat_rows"] for case in case_results),
            "geometry_invalid_complete_free_rows": sum(case["geometry_invalid_complete_free_rows"] for case in case_results),
            "parser_drops": sum(case["parser_drops"] for case in case_results),
            "eos_cells": sum(case["eos_cells"] for case in case_results),
            "horizon_cells": sum(case["horizon_cells"] for case in case_results),
        },
        "cost": costs,
        "peak_resources_across_workers": peaks,
        "interpretation": {
            "observation": "In both admitted cases, reversal-pair mean exact-row A-minus-B log-odds increased as older B multiplicity increased, opposite the registered direct-count direction. No free continuation returned to A or B by exact row or same-description native-pixel IoU>0.95.",
            "supported_inference": "Older synthetic history reweights relative exact-row likelihood, but the panel supplies no free recurrence event selecting A versus B.",
            "strong_alternative": "Inverse-frequency avoidance or another older-order/content interaction; reversal pairing and common local suffix do not identify a pure multiplicity mechanism.",
            "not_supported": "Natural-policy prevalence, physical-owner memory, IoU-neighborhood probability mass, learning or training origin, circuits/KV, generalization, or a usable repair.",
        },
        "source_files": source_files,
    }
    publish(R / "reduction.json", result)
    publish(R / "reduction-receipt.json", {"schema": "repeat_multiplicity.reduction_receipt.v1", "reduction_sha256": file_hash(R / "reduction.json"), "source_files": source_files, "overlays": {case["case_id"]: case["overlay"] for case in case_results}})
    print(json.dumps({"status": result["status"], "aggregate": result["aggregate"], "cost": costs, "peak_resources": peaks, "reduction_sha256": file_hash(R / "reduction.json")}, indent=2, sort_keys=True))
    return 0


runner = load_runner()


if __name__ == "__main__":
    raise SystemExit(main())
