"""CPU-only source and candidate proposal for the first-arrivals unit.

All physical labels here are proposals.  The lead admits a frozen selection
before this package may make a model call.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from probes.training_set_completion.artifacts import digest, literal_binding, write_pretty_json
from probes.training_set_completion.numerical_feedback.select import rows, same
from src.data.geometry import iou_xyxy, parse_source_bbox_tokens
from src.templates.renderer import BOX_END_TOKEN, BOX_START_TOKEN, OBJECT_REF_END_TOKEN, OBJECT_REF_START_TOKEN


MATURE = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-untied-highconfidence18-natural")
CENSUS = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-distribution-census")
UNIT = Path(__file__).resolve().parents[3] / "research/experiments/2026-09-23-recurrence-first-arrivals"
BASE = Path("/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent")
COORD = 151670

# Reviewed source-order proposals; annotations and visual handles are rechecked
# below.  14038 is an explicit owner HOLD preceding later valid proposals.
TARGETS = (
    ("mature", 2685, 3, "first_valid", -78, "one wine glass; lead review pending"),
    ("mature", 7511, 0, "invalidity_HOLD", None, "far-left shoreline proposal; person/owner identity UNKNOWN"),
    ("mature", 14038, 6, "owner_HOLD", -141, "adjacent book owners overlap at row 7"),
    ("mature", 313465, 0, "first_valid", 716308, "left bowl; same image also has a separate carrot proposal"),
    ("mature", 313465, 7, "invalidity_HOLD", None, "grouped carrot annotation; physical proposal validity UNKNOWN"),
    ("mature", 351017, 1, "first_invalid", None, "severe part-only hair localization; physical person arrived row 0; row 43 is later degeneration"),
    ("mature", 356238, 1, "owner_HOLD", 1663359, "group of adjacent book/DVD spines; singleton owner and earlier row 0 UNKNOWN"),
    ("mature", 477415, 4, "first_invalid", None, "one chair box spans several seats"),
    ("new", 131490, 0, "first_valid", 1820512, "left cow"),
)

CONTROL_FOR = (
    ("mature", 441325, 4, "mature:2685:3"),
    ("new", 231097, 0, "mature:313465:0"),
    ("new", 151704, 2, "mature:356238:1"),
    ("mature", 479075, 0, "new:131490:0"),
)


def _require(test: bool, message: str) -> None:
    if not test:
        raise ValueError(message)


def _matches(binding: dict, saved: dict) -> bool:
    return all(binding[key] == saved.get(key) for key in ("path", "sha256", "size_bytes"))


def _load(tokenizer):
    mature_panel = json.loads((MATURE / "panel.json").read_text())
    new_panel = json.loads((CENSUS / "panel.json").read_text())
    mature_records = {
        case["input_record"]["image_id"]: case["input_record"]
        for group in mature_panel["groups"] for case in group["cases"]
    }
    new_records = [json.loads(line) for line in (CENSUS / "new128.runtime.jsonl").read_text().splitlines()]
    sources = (
        ("mature", json.loads((MATURE / "event-selection-order.json").read_text())["image_ids"], mature_records,
         json.loads((CENSUS / "mature-census.json").read_text())["cells"]["untied-original"], MATURE),
        ("new", [record["image_id"] for record in new_records],
         {record["image_id"]: record for record in new_records},
         json.loads((CENSUS / "new-census.json").read_text())["cells"]["untied-original"], CENSUS),
    )
    output = {}
    seen = set()
    raw_cache = {}
    for source, order, records, cell_map, base in sources:
        by_image = {int(cell["image_id"]): cell for cell in cell_map.values()}
        _require(len(order) == len(by_image) == len(records), f"{source} source denominator changed")
        for rank, image_id in enumerate(order):
            cell = by_image[image_id]
            record = records[image_id]
            key = (record["metadata"]["split"], int(image_id))
            _require(key not in seen, f"overlapping image identity: {key}")
            seen.add(key)
            _require(cell["condition"] == "untied-original" and cell["policy"] == "original", "non-original source")
            raw_path = Path(cell["raw"]["path"])
            raw = raw_cache.setdefault(raw_path, json.loads(raw_path.read_text()))["rows"][int(cell["batch_index"])]
            _require(int(raw["image_id"]) == image_id, "source row/image mismatch")
            parsed = rows(raw["token_ids"])
            annotation = []
            for obj in record["objects"]:
                annotation.append({"owner_id": int(obj["coco_ann_id"]), "description": obj["desc"],
                                   "bbox": list(parse_source_bbox_tokens(obj["bbox_2d"], field="bbox_2d"))})
            rr = []
            for row in parsed:
                description = tokenizer.decode(row["description_tokens"], clean_up_tokenization_spaces=False)
                _require(tokenizer.encode(description, add_special_tokens=False) == row["description_tokens"],
                         f"noncanonical description tokens: {source}:{image_id}:{row['index']}")
                matches = sorted(((iou_xyxy(row["values"], obj["bbox"]), obj["owner_id"])
                                  for obj in annotation if obj["description"] == description), reverse=True)
                owner = (matches[0][1] if matches and matches[0][0] >= .5
                         and (len(matches) == 1 or matches[1][0] < .5) else None)
                rr.append({**row, "description": description, "proxy_owner": owner,
                           "top_iou": round(matches[0][0], 6) if matches else None,
                           "second_iou": round(matches[1][0], 6) if len(matches) > 1 else None})
            output[(source, image_id)] = {
                "source": source, "source_rank": rank, "split": key[0], "image_id": image_id,
                "cell": cell, "record": record, "image_path": str((base / record["images"][0]).resolve()),
                "raw": raw, "rows": rr, "annotation": annotation,
            }
    return output


def _source_bindings(item: dict) -> dict:
    cell = item["cell"]
    out = {}
    for key in ("raw", "trace", "runtime_receipt"):
        path = Path(cell[key]["path"])
        current = literal_binding(path)
        _require(_matches(current, cell[key]), f"changed source {key}: {path}")
        out[key] = current
    receipt = json.loads(Path(out["runtime_receipt"]["path"]).read_text())
    _require(receipt["status"] == "candidate_complete" and receipt["condition"] == "untied-original", "source receipt invalid")
    _require(receipt["group"] == cell["group"], "source group changed")
    _require(_matches(out["raw"], receipt["raw"]) and _matches(out["trace"], receipt["trace"]), "receipt artifact mismatch")
    trace = json.loads(Path(out["trace"]["path"]).read_text())["steps"]
    tokens = item["raw"]["token_ids"]
    index = int(cell["batch_index"])
    _require(tokens == [int(step["chosen"][index]) for step in trace[:len(tokens)]], "raw/trace token mismatch")
    image_binding = literal_binding(Path(item["image_path"]))
    identity = receipt["identity"]
    compact = {key: identity[key] for key in ("model", "shared_delta", "selected_ids_sha256",
                                                "input_rows_sha256", "output_rows_sha256",
                                                "input_delta_sha256", "output_delta_sha256", "dtype", "attention")}
    input_identity = receipt["input_identity"]
    input_summary = {key: input_identity[key] for key in ("request_ids", "media_sha256", "image_grids", "tensor_sha256")}
    return {**out, "image": image_binding, "source_identity": compact,
            "input_identity_sha256": digest(input_identity), "input_identity_summary": input_summary,
            "adapter_path": identity["adapter"]["adapter_path"],
            "embedding_delta_path": identity["embedding"]["identity"]["delta_path"],
            "loader_source": identity["loader_source"]}


def _candidate_rows(item: dict, first: dict, last_arrival: int, owner: int | None) -> list[dict]:
    visited = {row["proxy_owner"] for row in item["rows"][:last_arrival + 1] if row["proxy_owner"] is not None}
    if owner is not None:
        visited.add(owner)
    center = ((first["values"][0] + first["values"][2]) / 2,
              (first["values"][1] + first["values"][3]) / 2)
    # A crowded row may overlap two owners even though the strict unique-IoU
    # proxy assigns neither.  Such an owner is possibly visited, not new.
    possibly_visited = {
        obj["owner_id"] for row in item["rows"][:last_arrival + 1]
        for obj in item["annotation"]
        if obj["description"] == row["description"] and row["valid"]
        and iou_xyxy(row["values"], obj["bbox"]) >= .5
    }
    eligible = [obj for obj in item["annotation"] if obj["description"] == first["description"]
                and obj["owner_id"] not in visited | possibly_visited]
    eligible.sort(key=lambda obj: (((obj["bbox"][0] + obj["bbox"][2]) / 2 - center[0]) ** 2
                                   + ((obj["bbox"][1] + obj["bbox"][3]) / 2 - center[1]) ** 2,
                                   obj["owner_id"]))
    out = []
    for obj in eligible[:2]:
        row_tokens = [151646, *first["description_tokens"], 151647, 151648,
                      *(COORD + value for value in obj["bbox"]), 151649]
        actual = item["raw"]["token_ids"][first["start"]:first["end"]]
        _require(actual[:len(first["description_tokens"]) + 3] == row_tokens[:len(first["description_tokens"]) + 3],
                 "candidate noncoordinate prefix differs")
        j = next((j for j in range(4) if first["values"][j] != obj["bbox"][j]), None)
        out.append({"owner_id": obj["owner_id"], "bbox": obj["bbox"], "row_token_ids": row_tokens,
                    "first_differing_coordinate": j,
                    "branch_status": ("HOLD_physical_novelty_unconfirmed" if item["image_id"] == 2685
                                      else "planned" if j is not None and j < 3
                                      else "HOLD_no_free_coordinate"),
                    "prior_same_class_iou_by_row": {
                        str(row["index"]): round(iou_xyxy(row["values"], obj["bbox"]), 6)
                        for row in item["rows"][:last_arrival + 1] if row["description"] == first["description"]
                    },
                    "physical_novelty": "HOLD_lead_review" if item["image_id"] == 2685 else "proposed_lead_review",
                    "selection_rule": "nearest unvisited same-class annotation center; owner ID tie-break"})
    return out


def _family(item: dict, first_index: int, kind: str, owner: int | None, note: str) -> dict:
    rr = item["rows"]
    _require(0 <= first_index < len(rr), "first row absent")
    first = rr[first_index]
    if kind in ("first_valid", "normal_control"):
        _require(first["valid"] and first["proxy_owner"] == owner, "valid owner proxy mismatch")
        _require(owner not in (row["proxy_owner"] for row in rr[:first_index]),
                 "selected target has an earlier proxy owner visit")
        arrival_indices = [i for i in range(first_index, len(rr)) if rr[i]["proxy_owner"] == owner]
        if kind == "normal_control":
            _require(arrival_indices == [first_index] and item["raw"]["stop"] == "im_end", "control target returned or censored")
            _require(first_index + 1 < len(rr) and rr[first_index + 1]["proxy_owner"] not in (None, owner),
                     "normal control has no clear next new owner")
        else:
            _require(len(arrival_indices) >= 2, "no proxy owner return")
    elif kind == "first_invalid":
        arrival_indices = [i for i in range(first_index, len(rr)) if same(first, rr[i], 8)]
        _require(len(arrival_indices) >= 2, "no erroneous proposal return")
    else:
        arrival_indices = [i for i in range(first_index, len(rr)) if same(first, rr[i], 8)]
    arrivals = arrival_indices[:3] if kind != "normal_control" else [first_index]
    candidates = _candidate_rows(item, first, arrivals[-1], owner)
    tokens = item["raw"]["token_ids"]
    covered = [False] * len(tokens)
    for row in rr:
        covered[row["start"]:row["end"]] = [True] * (row["end"] - row["start"])
    unparsed = []
    start = None
    for offset in range(len(tokens) + 1):
        is_unparsed = offset < len(tokens) and not covered[offset]
        if is_unparsed and start is None:
            start = offset
        elif not is_unparsed and start is not None:
            segment = tokens[start:offset]
            unparsed.append({"start": start, "end": offset, "token_ids": segment,
                             "has_opener": 151646 in segment, "has_eos": 151645 in segment})
            start = None
    return {
        "id": f"{item['source']}:{item['image_id']}:{first_index}", "kind": kind,
        "status": "proposed_lead_review" if kind != "owner_HOLD" else "HOLD_owner_ambiguity",
        "review_note": note, "source": item["source"], "source_rank": item["source_rank"],
        "split": item["split"], "image_id": item["image_id"], "group": item["cell"]["group"],
        "batch_index": item["cell"]["batch_index"], "first_row": first_index,
        "owner_id": owner if kind in ("first_valid", "normal_control") else None,
        "description": first["description"], "first_bbox": first["values"],
        "arrival_row_indices": arrivals, "all_proxy_or_near_return_indices": arrival_indices,
        "intervening_rows": [list(range(a + 1, b)) for a, b in zip(arrivals, arrivals[1:])],
        "stop": item["raw"]["stop"], "complete_rows": len(rr), "unparsed_segments": unparsed,
        "censored": item["raw"]["stop"] == "length", "malformed_rows": item["cell"]["output"]["malformed_rows"],
        "row_evidence": [{"index": row["index"], "start": row["start"], "end": row["end"],
                          "description": row["description"], "bbox": row["values"], "valid_geometry": row["valid"],
                          "proxy_owner": row["proxy_owner"], "top_iou": row["top_iou"],
                          "second_iou": row["second_iou"]} for row in rr],
        "candidates": candidates, "source_bindings": _source_bindings(item),
    }


def prepare() -> dict:
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    sources = _load(tokenizer)
    families = [_family(sources[source, image_id], row, kind, owner, note)
                for source, image_id, row, kind, owner, note in TARGETS]
    ruling_path = UNIT / "lead-stage0-rulings.json"
    ruling = json.loads(ruling_path.read_text())
    _require(ruling["reviewed_draft"]["sha256"] ==
             "a8552c1fd0bfde52ab752cee8dc8637593b66ccf6abd7b9ba9291d8f1fa63b1a",
             "lead ruling changed")
    by_id = {family["id"]: family for family in families}
    held = by_id["mature:14038:6"]
    _require(held["source_rank"] == 10, "held source ordering changed")
    held["status"] = "HOLD_owner_UNKNOWN"
    held["physical_same_owner"] = "UNKNOWN"
    held["qualified_family_eligible"] = False
    held["lead_ruling"] = next(x for x in ruling["rulings"] if x["family_id"] == held["id"])
    wine = by_id["mature:2685:3"]
    wine["status"] = "native_family_retained_candidate_HOLD"
    wine["qualified_family_eligible"] = "lead_review_pending"
    wine["lead_ruling"] = next(x for x in ruling["rulings"] if x["family_id"] == wine["id"])
    item = sources["mature", 2685]
    first = item["rows"][3]
    center = ((first["values"][0] + first["values"][2]) / 2,
              (first["values"][1] + first["values"][3]) / 2)
    screen = []
    for obj in item["annotation"]:
        if obj["description"] != first["description"]:
            continue
        overlap = {str(row["index"]): round(iou_xyxy(row["values"], obj["bbox"]), 6)
                   for row in item["rows"][:5] if row["description"] == first["description"]}
        visited = any(value >= .5 for value in overlap.values())
        screen.append({"owner_id": obj["owner_id"], "bbox": obj["bbox"],
                       "center_distance_squared": ((obj["bbox"][0] + obj["bbox"][2]) / 2 - center[0]) ** 2
                       + ((obj["bbox"][1] + obj["bbox"][3]) / 2 - center[1]) ** 2,
                       "same_class_iou_by_native_row": overlap,
                       "status": "HOLD_possible_prior_visit" if visited else "HOLD_physical_novelty_unconfirmed"})
    wine["candidate_screen"] = sorted(screen, key=lambda x: (x["center_distance_squared"], x["owner_id"]))
    _require([x["owner_id"] for x in wine["candidate_screen"]] == [-78, -83, -81, -92, -85, -82],
             "wine candidate screen ordering changed")
    _require([x["owner_id"] for x in wine["candidates"]] == [-85, -82],
             "nearest nonoverlapping wine candidates changed")
    admission_path = UNIT / "lead-stage0-admission-v1.json"
    admission = json.loads(admission_path.read_text())
    _require(literal_binding(admission_path)["sha256"] ==
             "146605205625078300ef00016c8e1f715e9f78b4afa7edf4282d61a32a44b555",
             "lead partial admission changed")
    lead_labels = {r["family"]: r for r in admission["family_rulings"]}
    used_control_images = {(family["split"], family["image_id"]) for family in families}
    by_family = {family["id"]: family for family in families}
    for source, image_id, row, related in CONTROL_FOR:
        target = by_family[related]
        ranked = []
        for item in sources.values():
            if (item["split"], item["image_id"]) in used_control_images or item["raw"]["stop"] != "im_end":
                continue
            rr = item["rows"]
            for index, candidate_row in enumerate(rr[:-1]):
                owner_id = candidate_row["proxy_owner"]
                if owner_id is None or owner_id in (r["proxy_owner"] for r in rr[:index]) or owner_id in (r["proxy_owner"] for r in rr[index + 1:]):
                    continue
                if rr[index + 1]["proxy_owner"] in (None, owner_id):
                    continue
                if not _candidate_rows(item, candidate_row, index, owner_id):
                    continue
                rank = (candidate_row["description"] != target["description"],
                        abs(len(item["annotation"]) - len(sources[target["source"], target["image_id"]]["annotation"])),
                        abs(index - target["first_row"]), item["source"] != "mature", item["source_rank"])
                ranked.append((rank, item["source"], item["image_id"], index))
                break  # earliest eligible target within this image
        _require(bool(ranked), f"normal control support absent for {related}")
        _rank, best_source, best_image, best_row = min(ranked)
        _require((source, image_id, row) == (best_source, best_image, best_row),
                 f"normal control rank changed for {related}: {(best_source, best_image, best_row)}")
        item = sources[source, image_id]
        owner = item["rows"][row]["proxy_owner"]
        family = _family(item, row, "normal_control", owner, f"matched control for {related}; lead review pending")
        family["matched_target"] = related
        family["matching_rank"] = list(_rank)
        families.append(family)
        used_control_images.add((item["split"], item["image_id"]))
    _require(set(lead_labels) == {f["id"] for f in families if f["id"] != "mature:351017:1"}
             | {"mature:351017:43"}, "lead registry coverage changed")
    for family in families:
        lead_key = "mature:351017:43" if family["id"] == "mature:351017:1" else family["id"]
        decision = lead_labels[lead_key]
        family["lead_stage0_ruling"] = decision
        family["status"] = ("proposed_corrected_first_invalid_part_only" if family["id"] == "mature:351017:1"
                            else decision["native_label"])
        family["candidate_admission_status"] = decision["candidates"]
        admitted = set(decision.get("candidate_annotation_ids", []))
        for candidate in family["candidates"]:
            if candidate["owner_id"] not in admitted:
                candidate["branch_status"] = "HOLD_candidate_not_admitted"
                candidate["physical_novelty"] = "HOLD_lead_ruling"
            else:
                _require(candidate["branch_status"] == "planned", "admitted candidate lacks free coordinate")
    families.sort(key=lambda f: (f["source"] != "mature", f["source_rank"], f["first_row"]))
    first_valid = [f for f in families if f["kind"] == "first_valid"]
    first_invalid = [f for f in families if f["kind"] == "first_invalid"]
    controls = [f for f in families if f["kind"] == "normal_control"]
    _require((len(first_valid), len(first_invalid), len(controls)) == (3, 2, 4), "revised cohort count changed")
    _require(len({(f["split"], f["image_id"]) for f in controls}) == 4, "controls overlap")
    identities = {tuple(f["source_bindings"]["source_identity"].items()) for f in families}
    _require(len(identities) == 1, "effective independent untied rows differ between sources")
    for family in families:
        source = family["source"]
        image_id = family["image_id"]
        first = family["first_row"]
        family["visual_review_handles"] = [str(Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-first-arrivals/stage0-visual-review") / f"{source}-{image_id}-row{first}.png")]
        family["visual_review_bindings"] = [literal_binding(Path(path)) for path in family["visual_review_handles"]]
        family["candidate_count"] = len(family["candidates"])
        family["branch_candidate_count_at_first"] = sum(c["branch_status"] == "planned" for c in family["candidates"])
        for candidate in family["candidates"]:
            canonical = (f"{OBJECT_REF_START_TOKEN}{family['description']}{OBJECT_REF_END_TOKEN}"
                         f"{BOX_START_TOKEN}"
                         + "".join(f"<|coord_{value}|>" for value in candidate["bbox"])
                         + BOX_END_TOKEN)
            _require(tokenizer.encode(canonical, add_special_tokens=False) == candidate["row_token_ids"],
                     f"candidate serializer mismatch: {family['id']}:{candidate['owner_id']}")
    cells = []
    for family in families:
        if family["kind"] in ("owner_HOLD", "invalidity_HOLD"):
            continue
        for arrival in family["arrival_row_indices"]:
            cells.append({"family": family["id"], "arrival_row": arrival, "mode": "native_replay_and_sham"})
            row = family["row_evidence"][arrival]
            for candidate in family["candidates"]:
                if candidate["branch_status"].startswith("HOLD"):
                    continue
                _require(row["description"] == family["description"], "arrival description changed")
                j = next((index for index in range(4) if row["bbox"][index] != candidate["bbox"][index]), None)
                cells.append({"family": family["id"], "arrival_row": arrival, "owner_id": candidate["owner_id"],
                              "mode": "complete_row_score"})
                if j is not None and j < 3:
                    cells.append({"family": family["id"], "arrival_row": arrival, "owner_id": candidate["owner_id"],
                                  "mode": "one_coordinate_supplied_branch", "coordinate_slot": j,
                                  "native_token_id": COORD + row["bbox"][j],
                                  "supplied_token_id": COORD + candidate["bbox"][j]})
    mode_counts = {mode: sum(cell["mode"] == mode for cell in cells)
                   for mode in ("native_replay_and_sham", "complete_row_score", "one_coordinate_supplied_branch")}
    admitted_cells = admission["stage1_qualification"]["cells"]
    _require(all(cell in cells for cell in admitted_cells), "admitted bowl cells lost during registry revision")
    return {
        "schema": "recurrence_first_arrivals.stage0_selection_proposal.v1", "status": "proposed_lead_admission",
        "source_order": "mature event-selection-order then prospective new128.runtime.jsonl; split/image deduplicated",
        "source_population_counts": {"mature": 145, "new": 128},
        "source_files": [literal_binding(MATURE / "panel.json"), literal_binding(MATURE / "event-selection-order.json"),
                         literal_binding(CENSUS / "panel.json"), literal_binding(CENSUS / "new128.runtime.jsonl"),
                         literal_binding(CENSUS / "mature-census.json"), literal_binding(CENSUS / "new-census.json"),
                         literal_binding(CENSUS / "shared-sources.json"),
                         literal_binding(CENSUS / "selection-rule.json"), literal_binding(ruling_path),
                         literal_binding(admission_path)],
        "effective_identity": first_valid[0]["source_bindings"]["source_identity"],
        "families": families, "cells": cells,
        "counts": {"source_images": 273, "reviewed_families": len(families),
                   "proposed_first_valid": len(first_valid), "proposed_first_invalid": len(first_invalid),
                   "proposed_normal_controls": len(controls),
                   "owner_HOLD": sum(f["kind"] == "owner_HOLD" for f in families),
                   "invalidity_HOLD": sum(f["kind"] == "invalidity_HOLD" for f in families),
                   "reviewed_HOLD": sum(f["kind"].endswith("HOLD") for f in families),
                   "planned_cells": len(cells),
                   "planned_cell_modes": mode_counts,
                   "families_with_candidate": sum(any(c["branch_status"] == "planned" for c in f["candidates"])
                                                  for f in families if not f["kind"].endswith("HOLD")),
                   "candidate_HOLD_families": sum(bool(f["candidates"]) and
                                                  all(c["branch_status"].startswith("HOLD") for c in f["candidates"])
                                                  for f in families),
                   "lead_reviewed_family_rulings": len(admission["family_rulings"]),
                   "lead_native_label_admitted": sum(f["status"].startswith("lead-admitted") for f in families),
                   "owner_adjudicated_by_lead": sum(f["status"].startswith("lead-admitted") and
                                                    f["kind"] in ("first_valid", "normal_control") for f in families),
                   "replay_qualified": 0},
        "budget": {"allocated_gpu_hours_ceiling": 8, "used_stage0": 0,
                   "planned_free_token_cap_if_native_sham_shared": 256 * (
                       mode_counts["native_replay_and_sham"] + mode_counts["one_coordinate_supplied_branch"]),
                   "complete_row_scores": mode_counts["complete_row_score"],
                   "source_groups_to_reconstruct": len({(f["source"], f["group"]) for f in families}),
                   "historical_natural_cost_source": literal_binding(
                       CENSUS / "integration/natural-cost-check.json"),
                   "historical_cost_source": literal_binding(CENSUS / "integration/cost.json"),
                   "forecast": "Historical natural forwards are a scale hint only; first qualified case must measure full-prefix, vision, scoring, continuation, finalizer and allocated device intervals before pilot scale. Stop and return to lead if the frozen plan no longer fits 8 allocated GPU-hours."},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=UNIT / "selection.json")
    args = parser.parse_args()
    proposal = prepare()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_pretty_json(args.output, proposal)
    print(json.dumps({"status": proposal["status"], "counts": proposal["counts"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
