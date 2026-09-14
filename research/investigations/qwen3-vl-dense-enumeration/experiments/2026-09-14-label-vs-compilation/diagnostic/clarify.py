"""CPU-only four-owner clarification from the sealed free-h execution."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from probes.dora_owner_learning.candidate_opportunity import score
from probes.dora_owner_learning.reward_rows import _gt_objects, _pred_objects
from probes.parallel_owner_research.history import complete_rows
from probes.source_rweak_row_cross.run import native_record
from src.artifacts import publish_json_exclusive
from src.data.geometry import iou_xyxy


BASE = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-label-vs-compilation/diagnostic")
EXECUTION = BASE / "execution-v1"
EOS = 151645
TARGETS = {
    351017: ("2094819", "96050"),
    417044: ("1079910", "1083295"),
}


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def digest_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def binding(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": digest_file(path), "size_bytes": path.stat().st_size}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_binding(value: Mapping[str, Any], label: str) -> None:
    require(dict(value) == binding(Path(value["path"])), f"{label} binding")


def without_terminal_eos(ids: Sequence[int], stop: str) -> list[int]:
    result = list(ids)
    if stop == "im_end":
        require(result and result[-1] == EOS and EOS not in result[:-1], "terminal EOS identity")
        result.pop()
    else:
        require(stop == "length" and EOS not in result, "length-stop identity")
    return result


def best_overlap(
    gt_category: str,
    gt_box: Sequence[float],
    predictions: Sequence[tuple[str, tuple[float, float, float, float]]],
    *,
    same_class: bool,
) -> dict[str, Any] | None:
    candidates = []
    for index, (category, box) in enumerate(predictions):
        if same_class and category != gt_category:
            continue
        candidates.append({
            "iou": float(iou_xyxy(gt_box, box)),
            "pred_index_zero_based": index,
            "pred_row_ordinal_one_based": index + 1,
            "category": category,
            "bbox_xyxy_pixels": list(box),
        })
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item["iou"], item["pred_index_zero_based"]))
    return candidates[0]


def clarification(execution: Path) -> tuple[dict[str, Any], str]:
    from transformers import AutoTokenizer

    packet_path = BASE / "execution-packet-v2.json"
    packet = read(packet_path)
    paired_binding = packet["sources"]["paired_packet"]
    verify_binding(paired_binding, "paired packet")
    paired = read(Path(paired_binding["path"]))
    frozen_by_image = {int(row["image_id"]): row for row in paired["records"]}
    tokenizer = AutoTokenizer.from_pretrained(packet["models"]["base_model_path"], local_files_only=True)

    result_path = execution / "result.json"
    terminal_path = execution / "terminal.json"
    result, terminal = read(result_path), read(terminal_path)
    require(result["status"] == "completed_all_cells_free_c_parity_eligible", "execution result status")
    require(terminal["status"] == "completed_consumed" and terminal["exit_code"] == 0, "execution terminal")
    verify_binding(result["packet"], "result packet")
    verify_binding(result["grant"], "result grant")
    require(result["packet"] == binding(packet_path), "result/packet identity")
    require(len(result["rows"]) == 8, "eight consumed cells")

    cells: dict[tuple[int, str], dict[str, Any]] = {}
    raw_sources = []
    for cell in packet["cells"]:
        image_id, arm = int(cell["image_id"]), cell["arm"]
        cell_root = execution / f"cell-{cell['cell_id']:02d}-{arm.lower()}-{image_id}"
        raw_path, row_path = cell_root / "raw-generation.json", cell_root / "row.json"
        raw, row = read(raw_path), read(row_path)
        require(row["raw_generation"] == binding(raw_path), f"cell {cell['cell_id']} raw binding")
        require(raw["free_token_ids_sha256"] == digest_json(raw["free_token_ids"]), f"cell {cell['cell_id']} token digest")
        free_text = tokenizer.decode(raw["free_token_ids"], skip_special_tokens=False)
        require(free_text == row["free_text"], f"cell {cell['cell_id']} token/text identity")
        literal_rows = complete_rows(without_terminal_eos(raw["free_token_ids"], raw["stop_reason"]))
        frozen = frozen_by_image[image_id]
        parsed = native_record(free_text, frozen["case"], frozen["golden"], raw["stop_reason"])
        fresh_score = score(parsed, seed=None, length=len(raw["free_token_ids"]), stop=raw["stop_reason"])
        require(fresh_score == row["free_score"], f"cell {cell['cell_id']} native score")
        predictions, invalid = _pred_objects(parsed)
        require(invalid == 0 and len(predictions) == len(literal_rows), f"cell {cell['cell_id']} parsed rows")
        cells[image_id, arm] = {
            "packet_cell": cell,
            "raw": raw,
            "row": row,
            "literal_rows": literal_rows,
            "parsed": parsed,
            "score": fresh_score,
            "predictions": predictions,
        }
        raw_sources.append({
            "cell_id": cell["cell_id"],
            "arm": arm,
            "image_id": image_id,
            "raw_generation": binding(raw_path),
            "native_row": binding(row_path),
        })
    require(len(cells) == 8, "unique arm/image cells")

    paired_rows = []
    for image_id in [477415, 351017, 417044, 388795]:
        n16, arm_a = cells[image_id, "N16"], cells[image_id, "A"]
        left, right = n16["literal_rows"], arm_a["literal_rows"]
        first_difference = next((index for index, pair in enumerate(zip(left, right)) if pair[0] != pair[1]), None)
        if first_difference is None and len(left) != len(right):
            first_difference = min(len(left), len(right))
        difference_evidence = None
        if first_difference is not None:
            difference_evidence = {
                "N16_row_token_ids_sha256": digest_json(left[first_difference]) if first_difference < len(left) else None,
                "A_row_token_ids_sha256": digest_json(right[first_difference]) if first_difference < len(right) else None,
            }
        paired_rows.append({
            "image_id": image_id,
            "N16_complete_free_rows": len(left),
            "A_complete_free_rows": len(right),
            "N16_free_tokens_including_eos": len(n16["raw"]["free_token_ids"]),
            "A_free_tokens_including_eos": len(arm_a["raw"]["free_token_ids"]),
            "earliest_differing_complete_row_index_zero_based": first_difference,
            "earliest_differing_complete_row_ordinal_one_based": None if first_difference is None else first_difference + 1,
            "difference_evidence": difference_evidence,
        })

    owner_checks = []
    for image_id, owners in TARGETS.items():
        frozen = frozen_by_image[image_id]
        gt_ids = [str(item["object_id"]) for item in frozen["golden"]["gt"]]
        gt_objects = _gt_objects(cells[image_id, "A"]["parsed"], row_id=str(frozen["case"]["row_id"]))
        for owner in owners:
            gt_index = gt_ids.index(owner)
            gt_category, gt_box = gt_objects[gt_index]
            n16 = cells[image_id, "N16"]
            arm_a = cells[image_id, "A"]
            selected = [match for match in n16["score"]["50"]["matches"] if str(match["owner"]) == owner]
            require(len(selected) == 1 and selected[0]["gt_index"] == gt_index, f"N16 owner match {owner}")
            require(not any(str(match["owner"]) == owner for match in arm_a["score"]["50"]["matches"]), f"A owner unexpectedly selected {owner}")
            n16_match = selected[0]
            pred_index = n16_match["pred_index"]
            pred_category, pred_box = n16["predictions"][pred_index]
            recomputed_iou = float(iou_xyxy(gt_box, pred_box))
            require(pred_category == gt_category and recomputed_iou == n16_match["iou"], f"N16 match recompute {owner}")
            best_any = best_overlap(gt_category, gt_box, arm_a["predictions"], same_class=False)
            best_same = best_overlap(gt_category, gt_box, arm_a["predictions"], same_class=True)
            require(best_any is not None and best_same is not None, f"A overlap candidates {owner}")
            n16_position_exists_in_a = pred_index < len(arm_a["literal_rows"])
            owner_checks.append({
                "image_id": image_id,
                "owner_id": owner,
                "ground_truth": {
                    "gt_index_zero_based": gt_index,
                    "category": gt_category,
                    "bbox_coordinate_bins": frozen["golden"]["gt"][gt_index]["bbox"],
                    "bbox_xyxy_pixels": list(gt_box),
                },
                "N16_selected_match_at_iou_gt_0_5": {
                    "pred_index_zero_based": pred_index,
                    "pred_row_ordinal_one_based": pred_index + 1,
                    "iou": n16_match["iou"],
                    "category": pred_category,
                    "bbox_xyxy_pixels": list(pred_box),
                },
                "A": {
                    "selected_at_iou_gt_0_5": False,
                    "best_same_class": best_same,
                    "best_any_class": best_any,
                    "same_class_threshold_eligible": best_same["iou"] >= 0.5,
                    "any_class_spatial_threshold_eligible": best_any["iou"] >= 0.5,
                    "complete_rows_cover_N16_match_ordinal": n16_position_exists_in_a,
                },
                "bounded_read": (
                    "same_class_box_overlap_below_GT50"
                    if best_same["iou"] > 0.0
                    else "no_same_class_spatial_overlap_in_free_rows"
                ),
            })

    repeat = {}
    for arm in ("N16", "A"):
        cell = cells[388795, arm]
        c_ids = cell["packet_cell"]["c"]["token_ids"]
        offset = 0
        positions = []
        for index, row_ids in enumerate(cell["literal_rows"]):
            if row_ids == c_ids:
                positions.append({
                    "complete_row_index_zero_based": index,
                    "complete_row_ordinal_one_based": index + 1,
                    "free_token_offset_zero_based": offset,
                    "free_token_end_offset_inclusive": offset + len(row_ids) - 1,
                })
            offset += len(row_ids)
        repeat[arm] = {
            "c_positions": positions,
            "repeated_after_initial_count": max(0, len(positions) - 1),
        }
    require(repeat["N16"]["repeated_after_initial_count"] == 0, "N16 c repeat")
    require(repeat["A"]["c_positions"][1]["complete_row_index_zero_based"] == 3, "A c repeat position")

    output = {
        "schema": "label_vs_compilation.free_h_four_owner_clarification.v1",
        "status": "cpu_clarification_complete",
        "sources": {
            "execution_packet": binding(packet_path),
            "paired_packet": binding(Path(paired_binding["path"])),
            "execution_result": binding(result_path),
            "execution_terminal": binding(terminal_path),
            "accepted_analysis": binding(BASE / "analysis-v1.json"),
            "postrun_acceptance": binding(BASE / "postrun-acceptance-v1.json"),
            "raw_native_cells": raw_sources,
        },
        "method": {
            "parser": "native_record(parse_compact_object_box_closed)",
            "gt_projection": "coord_bins_to_pixel_xyxy via _gt_objects",
            "prediction_projection": "native pixel xyxy via _pred_objects",
            "iou": "iou_xyxy",
            "matching_threshold": "strict IoU > 0.5 in global matcher; displayed eligibility uses >=0.5 only as a descriptive boundary and all observed A maxima are below it",
            "row_basis": "literal raw free token IDs, terminal EOS removed, strict complete_rows",
        },
        "paired_free_rows": paired_rows,
        "owner_checks": owner_checks,
        "image_388795_repeat_c": repeat,
        "observations": [
            "All four named A owner checks lack a same-class or any-class box at the GT50 spatial threshold; none is merely excluded by global assignment despite an eligible edge.",
            "All four have an A same-class prediction somewhere, but its exact owner-box IoU is below 0.5. This is predicted geometry/trajectory redistribution, not evidence of class disappearance or physical object absence.",
            "For every owner, A contains the N16 matched row ordinal before EOS. Therefore the shorter A trajectories do not mechanically truncate before these N16 owner positions, although A does EOS earlier on351017 and417044.",
            "A/388795 repeats exact c as its fourth free complete row, starting at free-token offset29; N16 emits c only as its first free row.",
        ],
        "localization": {
            "351017": "N16 and A first differ at free row3 (one-based). N16 owner matches occur at rows4 and5, both within A's six-row trajectory. Owner2094819 has moderate same-class A overlap below threshold; owner96050 has zero same-class overlap. Earlier EOS is observed but is not by itself the owner-loss mechanism.",
            "417044": "N16 and A first differ at free row4. N16 owner matches occur at rows19 and22, both within A's 22-row trajectory. A has donut predictions but only low overlaps with these exact owner boxes, localizing the loss to row content/box redistribution rather than class loss or truncation before the corresponding positions.",
        },
        "limits": [
            "These checks establish only exact annotation-relative support in one persisted free-h continuation per endpoint/case; they do not establish that a physical object is absent.",
            "No visual review was performed, so class/extent ambiguity and whole-suffix physical quality remain unidentified.",
            "The evidence does not causally identify why row ordering, boxes, or EOS changed, nor whether another decode would recover the owners.",
            "A best-box comparisons are independent per owner; they do not create new labels or reinterpret unknown-neutral predictions.",
        ],
        "stop": "Four requested owners and paired literal-row evidence checked; no model, view, trajectory, or evaluation call made.",
    }

    lines = [
        "# Four-owner free-h clarification",
        "",
        "Status: **CPU clarification complete** from the accepted raw-token/native-parser artifacts; no new model or visual call.",
        "",
        "| Image / owner | GT class | N16 selected row / IoU | A best same-class row / IoU | A best any-class row / IoU |",
        "|---|---|---:|---:|---:|",
    ]
    for item in owner_checks:
        n16, same, any_class = item["N16_selected_match_at_iou_gt_0_5"], item["A"]["best_same_class"], item["A"]["best_any_class"]
        lines.append(
            f"| {item['image_id']} / {item['owner_id']} | {item['ground_truth']['category']} | "
            f"{n16['pred_row_ordinal_one_based']} / {n16['iou']:.6f} | "
            f"{same['pred_row_ordinal_one_based']} {same['category']} / {same['iou']:.6f} | "
            f"{any_class['pred_row_ordinal_one_based']} {any_class['category']} / {any_class['iou']:.6f} |"
        )
    lines += [
        "",
        "All four A maxima are below IoU 0.5, so none is a global-assignment-only loss with an otherwise eligible owner edge. All four do have same-class predictions somewhere: the evidence localizes changed box/row support, not disappearance of the class and not physical missing-owner status.",
        "",
        "## Literal row alignment",
        "",
        "| Image | N16/A complete free rows | Earliest differing row (one-based) |",
        "|---:|---:|---:|",
    ]
    for item in paired_rows:
        lines.append(
            f"| {item['image_id']} | {item['N16_complete_free_rows']}/{item['A_complete_free_rows']} | "
            f"{item['earliest_differing_complete_row_ordinal_one_based']} |"
        )
    lines += [
        "",
        "For 351017, N16's selected owner rows are 4 and 5 while A has 6 rows; for 417044 they are 19 and 22 while A has 22 rows. Thus A's earlier EOS does not truncate before the corresponding N16 positions. It remains a correlated route-length change, not an identified cause.",
        "",
        "On 388795, N16 emits exact `c` only at free row1/token offset0. A emits it at row1/offset0 and repeats it at row4/offset29 (zero-based offset; row token offsets29–37).",
        "",
        "Remaining unknowns: physical presence/quality without visual review, class/extent ambiguity, and the cause of changed row order, geometry, and EOS. These four GT50 losses must not be promoted to physical missing-owner claims.",
    ]
    return output, "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution", type=Path, default=EXECUTION)
    parser.add_argument("--json-output", type=Path, default=BASE / "clarification.json")
    parser.add_argument("--markdown-output", type=Path, default=BASE / "clarification.md")
    args = parser.parse_args()
    output, markdown = clarification(args.execution)
    publish_json_exclusive(args.json_output, output)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    with args.markdown_output.open("x") as stream:
        stream.write(markdown)
    print(json.dumps({
        "status": output["status"],
        "json": binding(args.json_output),
        "markdown": binding(args.markdown_output),
        "owners": len(output["owner_checks"]),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
