"""Retrospective CPU audit of saved numerical returns in the frozen 273 sources."""

from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from probes.training_set_completion.artifacts import digest, literal_binding
from probes.training_set_completion.numerical_feedback.select import rows
from probes.training_set_completion.recurrence_first_arrivals.prepare import BASE, _load, _source_bindings
from probes.training_set_completion.recurrence_grounded_window_transfer.screen import (
    FREEZE, REGISTRY, UNIT, annotations, prior_guard, require, unique_proxy,
)
from src.data.geometry import iou_xyxy


OUT = UNIT / "supporting/numerical-return-attrition-v1.json"
BRIEF = UNIT / "supporting/lead-attrition-brief-v1.md"
ACCEPTANCE = UNIT / "lead-cpu-acceptance-v1.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def partners(tokens, rr):
    """All qualifying previous rows, retaining exact and nonexact matches."""
    result = []
    for j, row in enumerate(rr):
        exact_valid, exact_invalid, near_valid = [], [], []
        for i in range(j):
            previous = rr[i]
            if previous["description_tokens"] != row["description_tokens"]:
                continue
            equal = tokens[previous["start"]:previous["end"]] == tokens[row["start"]:row["end"]]
            if equal:
                destination = exact_valid if row["valid"] and previous["valid"] else exact_invalid
                destination.append({"i": i, "iou": iou_xyxy(previous["values"], row["values"])
                                    if destination is exact_valid else None})
            elif row["valid"] and previous["valid"]:
                overlap = iou_xyxy(previous["values"], row["values"])
                if overlap >= .5:
                    near_valid.append({"i": i, "iou": overlap})
        all_partners = exact_valid + exact_invalid + near_valid
        if all_partners:
            result.append({"j": j, "raw_span": [row["start"], row["end"]],
                           "description": row["description"], "valid": row["valid"],
                           "box": row["values"], "exact_valid": exact_valid,
                           "exact_invalid": exact_invalid, "near_valid_nonexact": near_valid,
                           "earliest_predecessor": min(partner["i"] for partner in all_partners),
                           "adjacent": any(partner["i"] == j - 1 for partner in all_partners),
                           "has_preceding_third": j >= 2})
    return result


def proxy_diagnostic(row, anns):
    matches = [(iou_xyxy(row["values"], ann["box"]), ann)
               for ann in anns if ann["description"] == row["description"]]
    matches.sort(key=lambda value: (-value[0], value[1]["id"]))
    best = matches[0] if matches else None
    runner = matches[1] if len(matches) > 1 else None
    proxy, _ = unique_proxy(row, anns)
    reasons = []
    if not matches:
        reasons.append("annotation_missing")
    else:
        if best[0] < .75:
            reasons.append("best_below_0.75")
        if runner is not None and runner[0] >= .25:
            reasons.append("runner_not_below_0.25")
        if best[1]["explicit_group"]:
            reasons.append("explicit_crowd_or_group")
    return {"same_class_annotation_count": len(matches),
            "best_iou": best[0] if best else None, "best_annotation_id": best[1]["id"] if best else None,
            "best_flags": best[1]["flags"] if best else None,
            "runner_iou": runner[0] if runner else None,
            "runner_annotation_id": runner[1]["id"] if runner else None,
            "qualified_proxy_id": proxy["annotation_id"] if proxy else None,
            "reasons": reasons}


def triple(raw, rr, j, pair_kind, pair_iou, anns, tokenizer, width, height):
    third = j >= 2
    result = {"pair_rows": [j - 1, j], "pair_kind": pair_kind, "pair_iou": pair_iou,
              "has_preceding_third": third}
    if not third:
        result.update({"rows": None, "reason_flags": ["missing_preceding_third"],
                       "first_failure": "missing_preceding_third"})
        return result
    a, b, c = rr[j - 2:j + 1]
    rowset = (a, b, c)
    flags = []
    validity = [bool(row["valid"]) for row in rowset]
    for label, valid in zip("ABC", validity):
        if not valid:
            flags.append(f"invalid_geometry_{label}")
    raw_contiguous = a["end"] == b["start"] and b["end"] == c["start"]
    class_equal = a["description_tokens"] == b["description_tokens"] == c["description_tokens"]
    if not raw_contiguous:
        flags.append("raw_gap")
    if not class_equal:
        flags.append("class_mismatch")
    proxies = [proxy_diagnostic(row, anns) for row in rowset]
    for label, diag in zip("ABC", proxies):
        flags.extend(f"{reason}_{label}" for reason in diag["reasons"])
    ids = [diag["qualified_proxy_id"] for diag in proxies]
    pattern = (ids[0] != ids[1] == ids[2]) if None not in ids else None
    if pattern is False:
        flags.append("not_A_B_B_proxy_pattern")
    guard = None
    if pattern and class_equal:
        by_id = {ann["id"]: ann for ann in anns}
        guard = prior_guard(raw, rr, b, by_id[ids[1]], tokenizer, width, height)
        if guard["status"] != "ok":
            flags.append("chronology_HOLD" if guard["status"] == "HOLD" else "earlier_proxy_overlap")
    order = ["invalid_geometry_A", "invalid_geometry_B", "invalid_geometry_C",
             "raw_gap", "class_mismatch"]
    order += [f"{reason}_{label}" for label in "ABC"
              for reason in ("annotation_missing", "best_below_0.75",
                             "runner_not_below_0.25", "explicit_crowd_or_group")]
    order += ["not_A_B_B_proxy_pattern", "chronology_HOLD", "earlier_proxy_overlap"]
    first = next((reason for reason in order if reason in flags), "eligible_R_triple")
    result.update({"rows": [a["index"], b["index"], c["index"]],
                   "raw_spans": [[row["start"], row["end"]] for row in rowset],
                   "raw_contiguous": raw_contiguous, "equal_description_tokens": class_equal,
                   "row_validity": validity, "proxy_diagnostics": proxies,
                   "A_ne_B_eq_B": pattern, "chronology_guard": guard,
                   "reason_flags": flags, "first_failure": first})
    return result


def selfcheck():
    def rawrow(box, cls=100):
        return [151646, cls, 151647, 151648, *(151670 + value for value in box), 151649]
    A = [100, 100, 200, 200]
    near = [110, 100, 210, 200]
    invalid = [200, 100, 100, 200]
    def classify(tokens):
        parsed = rows(tokens)
        for row in parsed:
            row["description"] = str(row["description_tokens"])
        return partners(tokens, parsed)
    exact = classify(rawrow(A) * 2)
    require(len(exact) == 1 and exact[0]["exact_valid"] == [{"i": 0, "iou": 1.0}],
            "exact-repeat classification failed")
    nonexact = classify(rawrow(A) + rawrow(near))
    require(len(nonexact) == 1 and nonexact[0]["near_valid_nonexact"][0]["iou"] >= .5
            and not nonexact[0]["exact_valid"], "near-nonexact classification failed")
    bad = classify(rawrow(invalid) * 2)
    require(len(bad) == 1 and bad[0]["exact_invalid"] == [{"i": 0, "iou": None}]
            and not bad[0]["near_valid_nonexact"], "invalid exact classification failed")
    nonadjacent = classify(rawrow(A) + rawrow(A, 101) + rawrow(A))
    require(len(nonadjacent) == 1 and not nonadjacent[0]["adjacent"]
            and nonadjacent[0]["earliest_predecessor"] == 0,
            "nonadjacent classification failed")
    many = classify(rawrow(A) + rawrow(near) + rawrow(A) + rawrow(A))
    require(many[-1]["earliest_predecessor"] == 0
            and {x["i"] for x in many[-1]["exact_valid"]} == {0, 2}
            and {x["i"] for x in many[-1]["near_valid_nonexact"]} == {1},
            "earliest/all-partner classification failed")
    return ["exact_full_row", "nonexact_valid_iou_ge_0.5", "invalid_exact_separate",
            "nonadjacent", "earliest_and_all_qualifying_predecessors"]


def build():
    require(sha(BRIEF) == "d7e0754bc05caee29b732dfec756e40dbcc533e522d03d638df0e2aa0b2f98fd",
            "attrition brief changed")
    require(sha(ACCEPTANCE) == "68d23d266693abff9727706b2263942a4f4c080e3c4e165d54e80553606b503f",
            "screen acceptance changed")
    require(sha(REGISTRY) == "e79234c7360427508d9846b6290087231ca1ea05d57f12e6246f14d3e054652d",
            "accepted registry changed")
    helper = Path(__file__).with_name("screen.py")
    require(sha(helper) == "cb135f584db3309b594ec041cb63132e932727b74e6fc8e88d369cc48ea7d1e8",
            "accepted screen helper changed")
    checks = selfcheck()
    freeze = json.loads(FREEZE.read_text())
    accepted = json.loads(REGISTRY.read_text())
    selection = json.loads(Path(freeze["authority"]["selection"]["path"]).read_text())
    order = freeze["source_order"]
    require(len(order) == len(accepted["source_screen"]) == 273, "273 source denominator changed")
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    sources = _load(tokenizer)
    require(len(sources) == 273, "native source denominator changed")
    image_rows = []
    counts = collections.Counter()
    first_failures = collections.Counter()
    overlapping_failures = collections.Counter()
    image_flags = collections.defaultdict(set)
    book = None
    for identity in order:
        key = (identity["source"], identity["image_id"])
        item = sources[key]
        bound = _source_bindings(item)
        require(bound["source_identity"] == selection["effective_identity"], f"source identity drift: {key}")
        require(all(item[k] == identity[k] for k in ("source", "source_rank", "split", "image_id")),
                f"source order drift: {key}")
        rr = item["rows"]
        raw = item["raw"]
        returns = partners(raw["token_ids"], rr)
        anns = annotations(item["record"])
        adjacent = []
        for record in returns:
            j = record["j"]
            counts["return_rows"] += 1
            image_flags[key].add("any")
            for name, field in (("exact_valid", "exact_valid"),
                                ("exact_invalid", "exact_invalid"),
                                ("near_valid_nonexact", "near_valid_nonexact")):
                if record[field]:
                    counts[f"{name}_return_rows"] += 1
                    image_flags[key].add(name)
                counts[f"{name}_pairs"] += len(record[field])
                counts[f"{name}_adjacent_pairs"] += sum(p["i"] == j - 1 for p in record[field])
                counts[f"{name}_nonadjacent_pairs"] += sum(p["i"] != j - 1 for p in record[field])
            kinds = sum(bool(record[name]) for name in ("exact_valid", "exact_invalid", "near_valid_nonexact"))
            if kinds == 1:
                counts["return_rows_single_pair_kind"] += 1
            else:
                counts["return_rows_multiple_pair_kinds"] += 1
            counts["adjacent_return_rows" if record["adjacent"] else "nonadjacent_only_return_rows"] += 1
            for field in ("exact_valid", "near_valid_nonexact"):
                for partner in record[field]:
                    if partner["i"] == j - 1:
                        detail = triple(raw, rr, j, field, partner["iou"], anns, tokenizer,
                                        item["record"]["width"], item["record"]["height"])
                        adjacent.append(detail)
                        counts["adjacent_valid_geometry_return_pairs"] += 1
                        counts["adjacent_valid_with_third" if detail["has_preceding_third"] else
                               "adjacent_valid_missing_third"] += 1
                        first_failures[detail["first_failure"]] += 1
                        overlapping_failures.update(detail["reason_flags"])
        counts["complete_canonical_rows"] += len(rr)
        counts["valid_geometry_rows"] += sum(bool(row["valid"]) for row in rr)
        counts["invalid_geometry_rows"] += sum(not row["valid"] for row in rr)
        if identity["image_id"] in (313465, 151704, 477415):
            counts["named_excluded_images_included_in_audit"] += 1
        screen_row = accepted["source_screen"][identity["global_source_rank"]]
        require((screen_row["source"], screen_row["image_id"]) == key, "accepted registry order drift")
        require(all(bound[name] == screen_row["source_binding"][name]
                    for name in ("raw", "trace", "runtime_receipt", "image")),
                f"accepted source binding drift: {key}")
        image_rows.append({**identity, "excluded_from_new_picks": bool(screen_row["selection_exclusion"]),
                           "source_bindings": {name: bound[name] for name in ("raw", "trace", "runtime_receipt", "image")},
                           "processed_record_sha256": digest(item["record"]),
                           "canonical_row_count": len(rr), "native_stop": raw["stop"],
                           "returns": returns, "earliest_return": (None if not returns else
                               {"j": returns[0]["j"], "earliest_predecessor": returns[0]["earliest_predecessor"]}),
                           "adjacent_valid_pair_triples": adjacent})
        if key == ("new", 151704):
            require(len(rr) > 4, "known book rows missing")
            a, b = rr[3], rr[4]
            book = {"source": identity, "pair_rows": [3, 4],
                    "pair_exact_full_row": raw["token_ids"][a["start"]:a["end"]] ==
                                            raw["token_ids"][b["start"]:b["end"]],
                    "pair_valid": bool(a["valid"] and b["valid"]),
                    "pair_iou": iou_xyxy(a["values"], b["values"]),
                    "diagnostic_triple": triple(raw, rr, 4, "known_book_diagnostic",
                                               iou_xyxy(a["values"], b["values"]), anns, tokenizer,
                                               item["record"]["width"], item["record"]["height"])}
    require(book is not None, "known book diagnostic absent")
    counts["all_images"] = len(image_rows)
    counts["mature_images"] = sum(x["source"] == "mature" for x in image_rows)
    counts["prospective_images"] = sum(x["source"] == "new" for x in image_rows)
    for kind in ("any", "exact_valid", "exact_invalid", "near_valid_nonexact"):
        counts[f"{kind}_return_images"] = sum(kind in flags for flags in image_flags.values())
    first_return_kinds = collections.Counter()
    first_adjacent_valid_failure = collections.Counter()
    for image in image_rows:
        if not image["returns"]:
            continue
        earliest = image["returns"][0]
        first_return_kinds["+".join(name for name in ("exact_valid", "exact_invalid", "near_valid_nonexact")
                                    if earliest[name])] += 1
        triples = image["adjacent_valid_pair_triples"]
        first_adjacent_valid_failure[
            min(triples, key=lambda row: row["pair_rows"][1])["first_failure"]
            if triples else "no_adjacent_valid_pair"
        ] += 1
    result = {"status": "retrospective_cpu_candidate", "scope": "same frozen 273 original-policy sources",
              "brief": literal_binding(BRIEF), "acceptance": literal_binding(ACCEPTANCE),
              "accepted_registry": literal_binding(REGISTRY), "accepted_screen_helper": literal_binding(helper),
              "audit_helper": literal_binding(Path(__file__)),
              "source_freeze": literal_binding(FREEZE), "selfcheck_rejections": checks,
              "predicates": {"exact": "full serialized row-token equality; invalid exact separate",
                             "near_nonexact": "both valid geometry, same description tokens, nonexact row tokens, native box IoU>=0.5",
                             "physical_owner": "UNKNOWN; no owner inferred"},
              "first_failure_order": ["missing_preceding_third", "invalid_geometry_A/B/C", "raw_gap",
                                      "class_mismatch", "proxy_A/B/C: missing, best<.75, runner>=.25, group",
                                      "not_A_B_B_proxy_pattern", "chronology_HOLD", "earlier_proxy_overlap"],
              "counts": dict(counts), "first_failure_counts": dict(first_failures),
              "overlapping_failure_counts": dict(overlapping_failures),
              "earliest_return_kind_by_image": dict(first_return_kinds),
              "first_adjacent_valid_failure_by_image": dict(first_adjacent_valid_failure),
              "images": image_rows, "known_book_diagnostic": book,
              "model_loads": 0, "cuda_calls": 0, "model_forwards": 0,
              "vision_forwards": 0, "generation_tokens": 0, "gpu_hours": 0}
    return result


if __name__ == "__main__":
    require(not OUT.exists(), "attrition output already exists")
    result = build()
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "counts": result["counts"],
                      "first_failure_counts": result["first_failure_counts"],
                      "known_book": result["known_book_diagnostic"]["diagnostic_triple"]["first_failure"]}))
