"""Prepare the immutable CPU-side repeat-multiplicity draft and overlays."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
WORKTREE = Path("/data/CoordExp/.worktrees/research-probes")
OLD = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-small-owner-repeat-origin")
DEFAULT_OUTPUT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-repeat-multiplicity")
EXPECTED_HEAD = "296a9097e36d5854aafa6d7d5b714000063205cf"
EOS = 151645
ROW_START = 151646
REF_END = 151647
BOX_START = 151648
ROW_END = 151649

# Literal source rows. Indices address complete rows in the saved action, not
# parser-filtered rows. Root still owns physical-instance admission.
SELECTIONS = {
    "9813": {
        "A": ("natural", 3, "person", [390, 305, 421, 347]),
        "B": ("natural", 7, "person", [616, 295, 664, 355]),
        "C": ("natural", 2, "horse", [322, 145, 666, 724]),
    },
    "158044": {
        "A": ("natural", 4, "book", [782, 0, 999, 202]),
        "B": ("natural", 13, "book", [860, 405, 999, 517]),
        "C": ("natural", 3, "teddy bear", [549, 233, 653, 308]),
    },
    "417044": {
        "A": ("early_original_translated", 8, "donut", [243, 388, 303, 440]),
        "B": ("early_original_translated", 10, "donut", [361, 441, 432, 510]),
        "C": ("natural", 0, "person", [0, 406, 500, 999]),
    },
    "502725": {
        "A": ("natural", 0, "knife", [0, 656, 211, 848]),
        "B": ("natural", 2, "knife", [0, 555, 218, 605]),
        "C": ("early_original_translated", 5, "cake", [256, 222, 953, 859]),
    },
}

BLOCKS = {
    "a5_b3_fwd": "AAAB",
    "a5_b3_rev": "BAAA",
    "a4_b4_fwd": "AABB",
    "a4_b4_rev": "BBAA",
    "a3_b5_fwd": "ABBB",
    "a3_b5_rev": "BBBA",
}


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def publish(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical_bytes(value))


def records_path(case_id: str) -> Path:
    rank = {"9813": 0, "158044": 1, "417044": 5}[case_id] if case_id != "502725" else 7
    lane = "repair-01" if case_id == "502725" else "full"
    return OLD / lane / f"rank-{rank}" / "records.jsonl"


def read_records(path: Path) -> dict[str, dict[str, Any]]:
    values = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {value["job_id"]: value for value in values}


def split_complete_rows(ids: list[int]) -> list[list[int]]:
    rows: list[list[int]] = []
    index = 0
    while index < len(ids):
        if ids[index] == EOS:
            break
        require(ids[index] == ROW_START, f"non-row token at action offset {index}")
        try:
            end = ids.index(ROW_END, index) + 1
        except ValueError:
            break
        rows.append(ids[index:end])
        index = end
    return rows


def row_coord_bins(row: list[int]) -> list[int]:
    require(row[0] == ROW_START and row[-1] == ROW_END, "row wrapper mismatch")
    require(REF_END in row and BOX_START in row, "row inner wrapper mismatch")
    start = row.index(BOX_START) + 1
    coords = row[start:-1]
    require(len(coords) == 4, "row does not have four coordinate tokens")
    return [value - 151670 for value in coords]


def row_iou(a: list[int], b: list[int]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def source_ref(case_id: str, label: str, spec: tuple[str, int, str, list[int]], records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    job_id, row_index, description, coords = spec
    record = records[job_id]
    rows = split_complete_rows(record["action_ids"])
    require(row_index < len(rows), f"{case_id}/{label}: source row index missing")
    ids = rows[row_index]
    require(row_coord_bins(ids) == coords, f"{case_id}/{label}: coordinate source changed")
    require(ids[0] == ROW_START and ids[-1] == ROW_END and EOS not in ids, f"{case_id}/{label}: invalid literal row")
    return {
        "label": label,
        "description": description,
        "coord_bins": coords,
        "token_ids": ids,
        "token_count": len(ids),
        "token_ids_sha256": digest(ids),
        "source": {
            "records_path": str(records_path(case_id)),
            "records_sha256": file_hash(records_path(case_id)),
            "job_id": job_id,
            "complete_action_row_index": row_index,
            "record_packet_sha256": record["packet_sha256"],
        },
    }


def make_prefix(rows: dict[str, dict[str, Any]], block: str) -> tuple[list[str], list[int]]:
    labels = ["A", "B", *list(block), "A", "B", "C"]
    ids = [token for label in labels for token in rows[label]["token_ids"]]
    return labels, ids


def render_overlay(case: dict[str, Any], output: Path) -> dict[str, Any]:
    image_path = Path(case["source_case"]["image_path"])
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    colors = {"A": "#ff3030", "B": "#00a0ff", "C": "#00c853"}
    width, height = image.size
    for label in ("A", "B", "C"):
        box = case["candidates"][label]["coord_bins"]
        xy = [round(box[0] * width / 999), round(box[1] * height / 999),
              round(box[2] * width / 999), round(box[3] * height / 999)]
        line = max(3, round(min(width, height) / 250))
        draw.rectangle(xy, outline=colors[label], width=line)
        text = f"{label}: {case['candidates'][label]['description']} {box}"
        anchor = (max(0, xy[0]), max(0, xy[1] - 18))
        draw.rectangle([anchor[0], anchor[1], min(width, anchor[0] + 8 * len(text)), anchor[1] + 17], fill="black")
        draw.text((anchor[0] + 2, anchor[1] + 1), text, fill=colors[label], font=ImageFont.load_default())
    target = output / "overlays" / f"case-{case['case_id']}-abc.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target)
    return {"path": str(target), "sha256": file_hash(target), "image_size": [width, height]}


def build_draft(output: Path) -> dict[str, Any]:
    require(not output.exists(), f"occupied output root: {output}")
    observed_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=WORKTREE, text=True
    ).strip()
    require(observed_head == EXPECTED_HEAD, f"checkout HEAD changed: {observed_head}")
    old_packet_path = OLD / "packet.json"
    old_packet = json.loads(old_packet_path.read_text())
    old_cases = {str(case["case_id"]): case for case in old_packet["cases"]}
    cases = []
    for case_id in SELECTIONS:
        old_case = old_cases[case_id]
        records = read_records(records_path(case_id))
        candidates = {label: source_ref(case_id, label, spec, records) for label, spec in SELECTIONS[case_id].items()}
        require(candidates["A"]["description"] == candidates["B"]["description"], f"{case_id}: A/B description differs")
        require(candidates["A"]["token_count"] == candidates["B"]["token_count"], f"{case_id}: A/B serialized length differs")
        iou = row_iou(candidates["A"]["coord_bins"], candidates["B"]["coord_bins"])
        require(iou <= 0.95, f"{case_id}: A/B IoU does not establish geometric distinction")
        cells = []
        for cell_id, block in BLOCKS.items():
            labels, prefix_ids = make_prefix(candidates, block)
            cells.append({
                "cell_id": cell_id,
                "varied_block": list(block),
                "prefix_row_labels": labels,
                "prefix_ids": prefix_ids,
                "prefix_token_count": len(prefix_ids),
                "prefix_ids_sha256": digest(prefix_ids),
                "counts": {"A": labels.count("A"), "B": labels.count("B"), "C": labels.count("C")},
                "budget": min(512, 3084 - len(prefix_ids)),
                "score_candidates": ["A", "B", "C", "EOS"],
            })
        case = {
            "case_id": case_id,
            "admission_status": "pending_root_visual_review",
            "source_case": deepcopy(old_case["source_case"]),
            "source_row_id": old_case["source_case"]["row_id"],
            "prompt_token_ids": old_case["prompt_token_ids"],
            "baseline_action_ids": old_case["baseline_action_ids"],
            "golden": deepcopy(old_case["golden"]),
            "candidates": candidates,
            "candidate_A_B_iou": iou,
            "cells": cells,
        }
        cases.append(case)

    source_files = dict(old_packet["source_files"])
    for case_id in SELECTIONS:
        source_files[str(records_path(case_id))] = file_hash(records_path(case_id))
    for path in (HERE / "unit.md", HERE / "prepare.py", HERE / "run_probe.py", HERE / "test_repeat_multiplicity.py"):
        source_files[str(path)] = file_hash(path)
    packet = {
        "schema": "repeat_multiplicity.v1",
        "status": "draft_pending_root_visual_admission",
        "claim_boundary": "Synthetic fixed-prefix conditional A/B recurrence; no natural-policy, learning, owner-ledger, circuit, training-origin, or generalization claim",
        "question": "Holding image, unique literal rows, prefix token count, recent A/B positions, and final row fixed, does older A/B multiplicity reweight exact-row preference and free recurrence?",
        "strong_alternative": "Older order/history interactions or last-geometry dependence rather than independent multiplicity",
        "falsifier": "Pair-mean A-minus-B full-row log-odds is not monotone A-heavy > balanced > B-heavy, reversal-pair disagreement dominates, or free first-row recurrence repeatedly opposes scored direction",
        "head": EXPECTED_HEAD,
        "prior_packet": {"path": str(old_packet_path), "sha256": file_hash(old_packet_path)},
        "anchor_adapter": old_packet["anchor_adapter"],
        "model": old_packet["model"],
        "config": old_packet["config"],
        "policy": {"dtype": "fp32", "attention": "sdpa", "temperature": 0.0, "repetition_penalty": 1.0, "top_p": 1.0, "top_k": 0, "use_model_defaults": False, "eos_token_id": EOS},
        "prefix_family": {"base": ["A", "B"], "varied_blocks": BLOCKS, "common_suffix": ["A", "B", "C"], "conditional_cells_per_case": 6},
        "limits": {"candidate_cases": 4, "conditional_cells_max": 24, "generation_calls_max": 28, "score_replays_max": 96, "score_replays_per_cell": 4, "conditional_free_tokens_max": 512, "total_action_tokens_max": 3084, "model_loads_max": 4, "seconds_per_worker": 1500, "peak_cuda_bytes_max": 12 * 1024**3, "peak_rss_bytes_max": 16 * 1024**3, "training_steps": 0},
        "cases": cases,
        "source_files": source_files,
    }
    output.mkdir(parents=True)
    overlays = {case["case_id"]: render_overlay(case, output) for case in cases}
    for case in cases:
        case["overlay"] = overlays[case["case_id"]]
    invariant = validate_packet(packet, require_final=False)
    packet["cpu_invariants"] = invariant
    publish(output / "packet-draft.json", packet)
    publish(output / "draft-receipt.json", {"schema": "repeat_multiplicity.draft_receipt.v1", "packet_draft_sha256": file_hash(output / "packet-draft.json"), "overlays": overlays, "cpu_invariants": invariant})
    return packet


def validate_packet(packet: dict[str, Any], *, require_final: bool) -> dict[str, Any]:
    require(packet["schema"] == "repeat_multiplicity.v1", "schema mismatch")
    cases = packet["cases"]
    require([case["case_id"] for case in cases] == list(SELECTIONS), "cohort/order changed")
    admitted = []
    for case in cases:
        if case["admission_status"] == "admitted_by_root":
            admitted.append(case["case_id"])
        elif require_final:
            require(case["admission_status"] == "HOLD_visual_owner_unresolved", f"{case['case_id']}: unresolved final admission")
        candidates = case["candidates"]
        require(candidates["A"]["description"] == candidates["B"]["description"], f"{case['case_id']}: A/B class")
        require(len(candidates["A"]["token_ids"]) == len(candidates["B"]["token_ids"]), f"{case['case_id']}: A/B length")
        require(case["candidate_A_B_iou"] <= 0.95, f"{case['case_id']}: A/B IoU")
        require(len(case["cells"]) == 6, f"{case['case_id']}: cell count")
        lengths = {cell["prefix_token_count"] for cell in case["cells"]}
        unique_sets = {tuple(sorted(set(cell["prefix_row_labels"]))) for cell in case["cells"]}
        suffixes = {tuple(cell["prefix_row_labels"][-3:]) for cell in case["cells"]}
        require(len(lengths) == len(unique_sets) == len(suffixes) == 1, f"{case['case_id']}: matched family invariant")
        require(unique_sets == {("A", "B", "C")} and suffixes == {("A", "B", "C")}, f"{case['case_id']}: row set/suffix")
        for cell in case["cells"]:
            rebuilt = [token for label in cell["prefix_row_labels"] for token in candidates[label]["token_ids"]]
            require(rebuilt == cell["prefix_ids"], f"{case['case_id']}/{cell['cell_id']}: prefix bytes")
            require(cell["budget"] <= 512 and len(rebuilt) + cell["budget"] <= 3084, f"{case['case_id']}/{cell['cell_id']}: budget")
    if require_final:
        require(packet["status"] == "frozen_root_admitted", "final packet status")
        require(packet["admitted_case_ids"] == admitted and bool(admitted), "final admitted cases")
    return {"candidate_cases": len(cases), "admitted_cases": len(admitted), "cells_per_case": 6, "all_prefix_token_counts_fixed_within_case": True, "all_unique_row_sets_fixed_within_case": True, "all_common_suffixes_fixed_within_case": True, "all_A_B_lengths_equal": True, "all_total_actions_within_3084": True}


def finalize(output: Path, admitted_ids: list[str], admission_receipt: Path, admission_receipt_sha256: str) -> dict[str, Any]:
    draft_path = output / "packet-draft-v2.json"
    require(draft_path.is_file(), "missing packet-draft.json")
    require(not (output / "packet.json").exists(), "packet.json already exists")
    packet = json.loads(draft_path.read_text())
    require(admission_receipt.is_absolute() and admission_receipt.is_file(), "missing absolute visual-admission receipt")
    require(file_hash(admission_receipt) == admission_receipt_sha256, "visual-admission receipt hash mismatch")
    require(admitted_ids and len(set(admitted_ids)) == len(admitted_ids), "--admit must name unique cases")
    require(set(admitted_ids) <= set(SELECTIONS), "--admit names an out-of-cohort case")
    for case in packet["cases"]:
        case["admission_status"] = "admitted_by_root" if case["case_id"] in admitted_ids else "HOLD_visual_owner_unresolved"
    packet["status"] = "frozen_root_admitted"
    packet["admitted_case_ids"] = [case_id for case_id in SELECTIONS if case_id in admitted_ids]
    packet["visual_admission"] = {"owner": "root", "receipt_path": str(admission_receipt), "receipt_sha256": admission_receipt_sha256}
    packet["source_files"][str(admission_receipt)] = admission_receipt_sha256
    for path in (HERE / "unit.md", HERE / "prepare.py", HERE / "run_probe.py", HERE / "test_repeat_multiplicity.py"):
        packet["source_files"][str(path)] = file_hash(path)
    packet["cpu_invariants"] = validate_packet(packet, require_final=True)
    publish(output / "packet.json", packet)
    publish(output / "packet-receipt.json", {"schema": "repeat_multiplicity.packet_receipt.v1", "packet_sha256": file_hash(output / "packet.json"), "draft_sha256": file_hash(draft_path), "admitted_case_ids": packet["admitted_case_ids"], "cpu_invariants": packet["cpu_invariants"]})
    return packet


def refresh_draft(output: Path) -> dict[str, Any]:
    old_draft = output / "packet-draft.json"
    target = output / "packet-draft-v2.json"
    require(old_draft.is_file(), "missing packet-draft.json")
    require(not target.exists(), "packet-draft-v2.json already exists")
    packet = json.loads(old_draft.read_text())
    for path in (HERE / "unit.md", HERE / "prepare.py", HERE / "run_probe.py", HERE / "test_repeat_multiplicity.py"):
        packet["source_files"][str(path)] = file_hash(path)
    packet["cpu_invariants"] = validate_packet(packet, require_final=False)
    publish(target, packet)
    publish(output / "draft-receipt-v2.json", {"schema": "repeat_multiplicity.draft_receipt.v2", "packet_draft_sha256": file_hash(target), "supersedes": {"path": str(old_draft), "sha256": file_hash(old_draft)}, "overlays": {case["case_id"]: case["overlay"] for case in packet["cases"]}, "cpu_invariants": packet["cpu_invariants"]})
    return packet


def repair_final(output: Path) -> dict[str, Any]:
    old_packet = output / "packet.json"
    target = output / "packet-v2.json"
    require(old_packet.is_file(), "missing packet.json")
    require(not target.exists(), "packet-v2.json already exists")
    packet = json.loads(old_packet.read_text())
    require(packet["admitted_case_ids"] == ["9813", "417044"], "unexpected admitted set for v2 repair")
    packet["supersedes_unlaunched_packet"] = {
        "path": str(old_packet),
        "sha256": file_hash(old_packet),
        "reason": "native-pixel repeat incidence and process-peak/token-bound accounting correction before any GPU call",
    }
    packet["limits"] = {
        "candidate_cases": 4,
        "admitted_cases": 2,
        "conditional_cells": 12,
        "generation_calls_max": 14,
        "generated_tokens_hard_max": 12312,
        "generated_tokens_expected_if_exact_anchor_lengths_match": 9328,
        "score_replays_max": 48,
        "score_replays_per_cell": 4,
        "conditional_free_tokens_max": 512,
        "natural_generation_budget_per_case": 3084,
        "total_action_tokens_max_per_call": 3084,
        "model_loads_max": 2,
        "seconds_per_worker": 1500,
        "peak_cuda_bytes_max": 12 * 1024**3,
        "peak_rss_bytes_max": 16 * 1024**3,
        "training_steps": 0,
    }
    for path in (HERE / "unit.md", HERE / "prepare.py", HERE / "run_probe.py", HERE / "test_repeat_multiplicity.py"):
        packet["source_files"][str(path)] = file_hash(path)
    packet["cpu_invariants"] = validate_packet(packet, require_final=True)
    publish(target, packet)
    publish(output / "packet-receipt-v2.json", {
        "schema": "repeat_multiplicity.packet_receipt.v2",
        "packet_sha256": file_hash(target),
        "supersedes_unlaunched_packet": packet["supersedes_unlaunched_packet"],
        "admitted_case_ids": packet["admitted_case_ids"],
        "limits": packet["limits"],
        "cpu_invariants": packet["cpu_invariants"],
    })
    return packet


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--draft", action="store_true")
    mode.add_argument("--refresh-draft", action="store_true")
    mode.add_argument("--repair-final", action="store_true")
    mode.add_argument("--final", action="store_true")
    parser.add_argument("--admit", default="")
    parser.add_argument("--admission-receipt", type=Path)
    parser.add_argument("--admission-receipt-sha256", default="")
    args = parser.parse_args()
    if args.draft:
        require(not args.admit and args.admission_receipt is None and not args.admission_receipt_sha256, "admission arguments are final-only")
        packet = build_draft(args.output.resolve())
    elif args.refresh_draft:
        require(not args.admit and args.admission_receipt is None and not args.admission_receipt_sha256, "admission arguments are final-only")
        packet = refresh_draft(args.output.resolve())
    elif args.repair_final:
        require(not args.admit and args.admission_receipt is None and not args.admission_receipt_sha256, "admission arguments are not used by repair-final")
        packet = repair_final(args.output.resolve())
    else:
        admitted = [value for value in args.admit.split(",") if value]
        require(args.admission_receipt is not None and bool(args.admission_receipt_sha256), "final requires visual-admission receipt and hash")
        packet = finalize(args.output.resolve(), admitted, args.admission_receipt.resolve(), args.admission_receipt_sha256)
    print(json.dumps({"status": packet["status"], "cases": len(packet["cases"]), "admitted": packet.get("admitted_case_ids", []), "output": str(args.output.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
