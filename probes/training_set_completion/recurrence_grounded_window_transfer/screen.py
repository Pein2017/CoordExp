"""Frozen 273-source, high-confidence annotation-proxy screen; no model calls."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from probes.training_set_completion.artifacts import digest, literal_binding
from probes.training_set_completion.numerical_feedback.select import rows
from probes.training_set_completion.recurrence_first_arrivals.prepare import BASE, _load, _source_bindings
from src.data.geometry import iou_xyxy, parse_source_bbox_tokens
from src.inference.parsing import parse_compact_object_box_closed


ROOT = Path(__file__).resolve().parents[3]
UNIT = ROOT / "research/experiments/2026-09-24-recurrence-grounded-window-transfer"
FREEZE = UNIT / "supporting/source-screen-freeze-v1.json"
REGISTRY = UNIT / "supporting/cpu-candidate-registry.json"
READBACK = UNIT / "supporting/cpu-readback-v1.json"
EXCLUDED = {313465, 151704, 477415}
FLAG_KEYS = ("iscrowd", "is_crowd", "crowd", "is_group", "group", "group_of")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def binding(path):
    return literal_binding(Path(path))


def annotations(record):
    result = []
    for obj in record["objects"]:
        flags = {key: obj[key] for key in FLAG_KEYS if key in obj}
        result.append({
            "id": int(obj["coco_ann_id"]), "description": obj["desc"],
            "box": list(parse_source_bbox_tokens(obj["bbox_2d"], field="bbox_2d")),
            "flags": flags, "explicit_group": any(bool(value) for value in flags.values()),
        })
    return result


def unique_proxy(row, anns):
    matches = sorted(
        ((iou_xyxy(row["values"], ann["box"]), ann) for ann in anns
         if ann["description"] == row["description"]),
        key=lambda pair: (-pair[0], pair[1]["id"]),
    )
    strong = [(score, ann) for score, ann in matches if score >= .75]
    if len(strong) != 1:
        return None, "ambiguous_or_missing_high_confidence_annotation"
    score, ann = strong[0]
    if any(other_score >= .25 for other_score, other in matches if other is not ann):
        return None, "ambiguous_or_missing_high_confidence_annotation"
    if ann["explicit_group"]:
        return None, "explicit_crowd_or_group_annotation"
    return {"annotation_id": ann["id"], "annotation_box": ann["box"],
            "row_box": row["values"], "iou": score,
            "other_same_class_iou_max": max((s for s, a in matches if a is not ann), default=0.0),
            "flags": ann["flags"], "flags_absent_unknown": not bool(ann["flags"])}, None


def prior_guard(raw, rr, target, ann, tokenizer, width, height):
    start = target["start"]
    prefix = raw["token_ids"][:start]
    if 151645 in prefix:
        return {"status": "HOLD", "reason": "terminal_in_prior_prefix"}
    text = tokenizer.decode(prefix, clean_up_tokenization_spaces=False)
    parsed = parse_compact_object_box_closed(text, row_id="grounded-prefix", row_index=0,
                                             image_width=width, image_height=height)
    earlier = [row for row in rr if row["end"] <= start]
    if parsed.dropped_predictions or len(parsed.predictions) != len(earlier):
        return {"status": "HOLD", "reason": "unparsed_or_malformed_prior_object",
                "parser_status": parsed.parse_status,
                "drop_reasons": [drop["reason"] for drop in parsed.dropped_predictions],
                "parsed_count": len(parsed.predictions), "canonical_count": len(earlier)}
    overlaps = [{"row": row["index"], "description": row["description"],
                 "box": row["values"], "iou": iou_xyxy(row["values"], ann["box"])}
                for row in earlier]
    offending = [entry for entry in overlaps if entry["iou"] >= .1]
    return {"status": "overlap" if offending else "ok",
            "reason": "earlier_parsed_box_overlaps_proxy" if offending else None,
            "prior_boxes": overlaps, "offending_rows": [entry["row"] for entry in offending]}


def scan_rows(raw, rr, anns, tokenizer, width, height):
    found = {"R": None, "D": None}
    patterns = collections.Counter()
    rejects = collections.Counter()
    holds = []
    ann_by_id = {ann["id"]: ann for ann in anns}
    require(len(ann_by_id) == len(anns), "duplicate annotation ID")
    for index in range(len(rr) - 2):
        a, b, c = rr[index:index + 3]
        if not all(row["valid"] for row in (a, b, c)):
            rejects["invalid_geometry"] += 1
            continue
        if a["end"] != b["start"] or b["end"] != c["start"]:
            rejects["raw_gap"] += 1
            continue
        if not (a["description_tokens"] == b["description_tokens"] == c["description_tokens"]):
            rejects["class_mismatch"] += 1
            continue
        mapped = [unique_proxy(row, anns) for row in (a, b, c)]
        if any(proxy is None for proxy, _ in mapped):
            rejects[next(reason for proxy, reason in mapped if proxy is None)] += 1
            continue
        proxies = [proxy for proxy, _ in mapped]
        owner_ids = [proxy["annotation_id"] for proxy in proxies]
        kind = ("R" if owner_ids[0] != owner_ids[1] == owner_ids[2] else
                "D" if len(set(owner_ids)) == 3 else None)
        if kind is None:
            rejects["other_proxy_pattern"] += 1
            continue
        patterns[kind] += 1
        guards = {"B": prior_guard(raw, rr, b, ann_by_id[owner_ids[1]], tokenizer, width, height)}
        if kind == "D":
            guards["C"] = prior_guard(raw, rr, c, ann_by_id[owner_ids[2]], tokenizer, width, height)
        statuses = [guard["status"] for guard in guards.values()]
        if "HOLD" in statuses:
            holds.append({"kind": kind, "rows": [a["index"], b["index"], c["index"]],
                          "guards": guards})
            rejects["chronology_HOLD"] += 1
            continue
        if "overlap" in statuses:
            rejects["earlier_proxy_overlap"] += 1
            continue
        if found[kind] is None:
            found[kind] = {
                "kind": kind, "rows": [a["index"], b["index"], c["index"]],
                "raw_spans": [[row["start"], row["end"]] for row in (a, b, c)],
                "description": a["description"], "description_tokens": a["description_tokens"],
                "proxies": proxies, "prior_guards": guards,
            }
    return {"earliest_R": found["R"], "earliest_D": found["D"],
            "pattern_counts": dict(patterns), "rejection_counts": dict(rejects),
            "chronology_HOLDs": holds,
            "R_absent_reason": None if found["R"] else "chronology_HOLD" if any(x["kind"] == "R" for x in holds) else "no_eligible_R",
            "D_absent_reason": None if found["D"] else "chronology_HOLD" if any(x["kind"] == "D" for x in holds) else "no_eligible_D"}


def pick(screen):
    require(len(screen) == 273 and [row["global_source_rank"] for row in screen] == list(range(273)),
            "source order/denominator changed")
    selected = []
    for kind in ("R", "D"):
        for row in screen:
            if row["selection_exclusion"] or row[f"earliest_{kind}"] is None:
                continue
            if any(chosen["image_id"] == row["image_id"] for chosen in selected):
                continue
            selected.append({"kind": kind, "source": row["source"], "image_id": row["image_id"],
                             "global_source_rank": row["global_source_rank"],
                             "window": row[f"earliest_{kind}"]})
            if sum(chosen["kind"] == kind for chosen in selected) == 2:
                break
    return selected


def selfcheck(tokenizer):
    book = tokenizer.encode("book", add_special_tokens=False)
    def row(box, desc=book):
        return [151646, *desc, 151647, 151648, *(151670 + value for value in box), 151649]
    A, B, C = [100, 100, 200, 200], [300, 300, 400, 400], [500, 500, 600, 600]
    anns = [{"id": i, "description": "book", "box": box, "flags": {}, "explicit_group": False}
            for i, box in enumerate((A, B, C), 1)]
    def run(tokens, aa=anns):
        rr = rows(tokens)
        for r in rr:
            r["description"] = tokenizer.decode(r["description_tokens"], clean_up_tokenization_spaces=False)
        return scan_rows({"token_ids": tokens}, rr, aa, tokenizer, 1000, 1000)
    good = row(A) + row(B) + row(B)
    require(run(good)["earliest_R"] is not None, "base R selfcheck failed")
    require(run(row(B) + good)["earliest_R"] is None, "earlier proxy visit not rejected")
    ambiguous = anns + [{"id": 9, "description": "book", "box": [310, 310, 410, 410],
                         "flags": {}, "explicit_group": False}]
    require(run(good, ambiguous)["earliest_R"] is None, "ambiguous match not rejected")
    malformed = [151646, *book, 151647, 151648, 151670] + good
    m = run(malformed)
    require(m["earliest_R"] is None and m["chronology_HOLDs"], "malformed prior object not held")
    require(run(row(A) + [1234] + row(B) + row(B))["earliest_R"] is None, "raw gap not rejected")
    require(run(row(A) + row(B, tokenizer.encode("cat", add_special_tokens=False)) + row(B))["earliest_R"] is None,
            "class mismatch not rejected")
    fake = [{"global_source_rank": i, "source": "mature", "image_id": i,
             "selection_exclusion": None, "earliest_R": {"rows": [0, 1, 2]} if i in (3, 7, 10) else None,
             "earliest_D": {"rows": [0, 1, 2]} if i in (3, 4, 8) else None} for i in range(273)]
    require([(x["kind"], x["image_id"]) for x in pick(fake)] == [("R", 3), ("R", 7), ("D", 4), ("D", 8)],
            "R then D source priority failed")
    fake[3], fake[4] = fake[4], fake[3]
    try:
        pick(fake)
    except ValueError:
        pass
    else:
        raise ValueError("source-order mutation was not rejected")
    return ["earlier_proxy_visit", "ambiguous_annotation", "malformed_prior_object",
            "raw_gap", "class_mismatch", "R_then_D_priority", "wrong_source_order"]


def build():
    freeze = json.loads(FREEZE.read_text())
    require(binding(UNIT / "unit.md")["sha256"] == "71a369966dfe8fe5a73719fb5fd4cf1f734ba08f77f71e41c338d3ecfbdf5353", "unit changed")
    require(freeze["status"] == "frozen_before_candidate_screen", "missing frozen screen")
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    checks = selfcheck(tokenizer)
    sources = _load(tokenizer)
    order = freeze["source_order"]
    require(len(sources) == len(order) == 273, "source denominator changed")
    selection = json.loads(Path(freeze["authority"]["selection"]["path"]).read_text())
    rulings = collections.defaultdict(list)
    for family in selection["families"]:
        rulings[(family["source"], family["image_id"])].append({
            key: family[key] for key in ("id", "first_row", "kind", "status", "review_note") if key in family
        } | {"lead_ruling": family.get("lead_ruling"),
             "physical_same_owner": family.get("physical_same_owner")})
    screen = []
    binding_conflicts = []
    for source_id in order:
        key = (source_id["source"], source_id["image_id"])
        item = sources[key]
        require(all(item[k] == source_id[k] for k in ("source", "source_rank", "split", "image_id")),
                f"source order drift: {key}")
        require(item["cell"]["group"] == source_id["group"] and
                int(item["cell"]["batch_index"]) == source_id["batch_index"],
                f"source cell drift: {key}")
        anns = annotations(item["record"])
        require(sorted((a["id"], a["description"], a["box"]) for a in anns) ==
                sorted((a["owner_id"], a["description"], a["bbox"]) for a in item["annotation"]),
                f"annotation projection drift: {key}")
        row = {**source_id, "processed_record_sha256": digest(item["record"]),
               "annotation_count": len(anns), "complete_rows": len(item["rows"]),
               "native_stop": item["raw"]["stop"],
               "existing_row_rulings": rulings[key],
               "selection_exclusion": "previously_probed_image" if source_id["image_id"] in EXCLUDED else None}
        try:
            bound = _source_bindings(item)
            require(bound["source_identity"] == selection["effective_identity"], f"effective identity drift: {key}")
        except Exception as exc:
            row["source_binding"] = {"status": "HOLD", "reason": str(exc)}
            row.update({"earliest_R": None, "earliest_D": None, "pattern_counts": {},
                        "rejection_counts": {}, "chronology_HOLDs": [],
                        "R_absent_reason": "source_binding_HOLD", "D_absent_reason": "source_binding_HOLD"})
            binding_conflicts.append({"source": source_id, "reason": str(exc)})
        else:
            row["source_binding"] = {"status": "verified", **{name: bound[name] for name in ("raw", "trace", "runtime_receipt", "image")}}
            row.update(scan_rows(item["raw"], item["rows"], anns, tokenizer,
                                 item["record"]["width"], item["record"]["height"]))
        screen.append(row)
    selected = [] if binding_conflicts else pick(screen)
    for chosen in selected:
        item = sources[(chosen["source"], chosen["image_id"])]
        chosen["source_bindings"] = _source_bindings(item)
        chosen["processed_record_sha256"] = digest(item["record"])
        chosen["processed_annotation_projection"] = annotations(item["record"])
        chosen["request_id"] = item["raw"]["row_id"]
        chosen["raw_stop"] = item["raw"]["stop"]
        chosen["native_token_count"] = len(item["raw"]["token_ids"])
        trace = json.loads(Path(chosen["source_bindings"]["trace"]["path"]).read_text())["steps"]
        chosen["rows_detail"] = []
        for index in chosen["window"]["rows"]:
            native = item["rows"][index]
            tokens = item["raw"]["token_ids"][native["start"]:native["end"]]
            observed = [step["chosen"][item["cell"]["batch_index"]] for step in trace[native["start"]:native["end"]]]
            require(tokens == observed, "selected row/native trace mismatch")
            chosen["rows_detail"].append({"index": index, "raw_span": [native["start"], native["end"]],
                                          "tokens": tokens, "trace_chosen": observed,
                                          "box": native["values"], "description": native["description"]})
        first = chosen["window"]["rows"][0]
        chosen["immediate_previous_row"] = ({key: item["rows"][first - 1][key]
                                             for key in ("index", "start", "end", "description", "values", "valid")}
                                            if first else None)
    counts = {"all_sources": 273, "mature": 145, "prospective": 128,
              "excluded_named_images": sum(row["selection_exclusion"] is not None for row in screen),
              "source_binding_HOLD": len(binding_conflicts),
              "R_eligible_images": sum(row["earliest_R"] is not None and row["selection_exclusion"] is None for row in screen),
              "D_eligible_images": sum(row["earliest_D"] is not None and row["selection_exclusion"] is None for row in screen),
              "R_pattern_windows": sum(row["pattern_counts"].get("R", 0) for row in screen),
              "D_pattern_windows": sum(row["pattern_counts"].get("D", 0) for row in screen),
              "chronology_HOLD_windows": sum(len(row["chronology_HOLDs"]) for row in screen),
              "R_selected": sum(x["kind"] == "R" for x in selected),
              "D_selected": sum(x["kind"] == "D" for x in selected)}
    return {"status": "HOLD_source_binding_conflict" if binding_conflicts else "candidate_cpu_qualified",
            "unit": binding(UNIT / "unit.md"), "freeze": binding(FREEZE),
            "selection": binding(freeze["authority"]["selection"]["path"]),
            "prior_registry_crosswalk_only": binding(freeze["authority"]["prior_registry_crosswalk_only"]["path"]),
            "selfcheck_rejections": checks, "counts": counts,
            "binding_conflicts": binding_conflicts, "source_screen": screen, "selected": selected,
            "model_loads": 0, "cuda_calls": 0, "model_forwards": 0, "vision_forwards": 0,
            "generation_tokens": 0, "gpu_hours": 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("screen", "readback", "selfcheck"))
    args = parser.parse_args()
    if args.mode == "selfcheck":
        tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
        print(json.dumps({"passed": selfcheck(tokenizer)}))
        return
    fresh = build()
    if args.mode == "screen":
        require(not REGISTRY.exists(), "registry already exists")
        REGISTRY.write_text(json.dumps(fresh, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"status": fresh["status"], "counts": fresh["counts"],
                          "selected": [(x["kind"], x["source"], x["image_id"], x["window"]["rows"])
                                       for x in fresh["selected"]]}))
    else:
        saved = json.loads(REGISTRY.read_text())
        require(fresh == saved, "cold native-source screen/readback mismatch")
        result = {"status": "cold_readback_passed", "registry": binding(REGISTRY),
                  "counts": fresh["counts"],
                  "selected": [(x["kind"], x["source"], x["image_id"], x["window"]["rows"])
                               for x in fresh["selected"]],
                  "source_order_sha256": json.loads(FREEZE.read_text())["source_order_sha256"]}
        require(not READBACK.exists(), "readback already exists")
        READBACK.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result))


if __name__ == "__main__":
    main()
