"""CPU-only, source-ordered numerical antecedent cohort proposal."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

from probes.training_set_completion.artifacts import literal_binding, write_pretty_json
from probes.training_set_completion.coordinate_continuity.runtime import _source
from probes.training_set_completion.numerical_feedback.runtime import _prefix_tokens
from probes.training_set_completion.numerical_feedback.select import token_hash
from probes.training_set_completion.recurrence_first_arrivals.prepare import BASE, CENSUS, MATURE, _load, _source_bindings
from src.qwen.input_identity import input_identity
from src.qwen.native import padded_histories
from src.qwen.runtime_loading import QwenLoadOptions, load_qwen_components_from_options


REPO = Path(__file__).resolve().parents[3]
UNIT = REPO / "research/experiments/2026-09-23-recurrence-native-x1-phase"
STAGE0 = REPO / "research/experiments/2026-09-23-recurrence-first-arrivals/selection.json"
BRIEF = UNIT / "supporting/cross-image-cpu-brief.md"
REGISTRY = UNIT / "supporting/cross-image-candidate-registry.json"
BRIEF_SHA = "d421a4cfdfcd528f66bc840634809c0eec1f86fe9830c5b61f705c6509a29457"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def earliest(item):
    result = {"R": None, "D": None}
    rr, tokens = item["rows"], item["raw"]["token_ids"]
    for earlier, latest, current in zip(rr, rr[1:], rr[2:]):
        if earlier["end"] != latest["start"] or latest["end"] != current["start"]:
            continue
        if any(row["end"] - row["start"] != 9 for row in (earlier, latest, current)):
            continue
        if not (earlier["description_tokens"] == latest["description_tokens"] == current["description_tokens"]):
            continue
        kind = ("R" if tokens[earlier["start"]:earlier["end"]] ==
                tokens[latest["start"]:latest["end"]] else
                "D" if earlier["values"] != latest["values"] else None)
        if kind and result[kind] is None:
            result[kind] = (earlier, latest, current)
    return result


def selfcheck():
    from probes.training_set_completion.numerical_feedback.select import rows
    def row(desc, x):
        return [151646, desc, 151647, 151648, 151670+x, 151672, 151680, 151690, 151649]
    a, b = row(100, 1), row(100, 2)
    def item(tokens):
        return {"rows": rows(tokens), "raw": {"token_ids": tokens}}
    require(earliest(item(a+a+a))["R"][2]["index"] == 2, "R selfcheck")
    require(earliest(item(a+b+a))["D"][2]["index"] == 2, "D selfcheck")
    require(all(v is None for v in earliest(item(a+a+[151645]+a)).values()), "gap selfcheck")
    require(all(v is None for v in earliest(item(a+row(101,1)+a)).values()), "class selfcheck")


def geometry(item, triplet, batch, raw, trace, config, pad):
    earlier, latest, current = triplet
    idx = int(item["cell"]["batch_index"])
    offset = current["coordinate_offsets"][1]
    require(offset == current["start"] + 5 and current["coordinate_offsets"][0] == offset-1,
            "wrong first-y1 slot")
    require(raw[idx]["token_ids"][offset] == int(trace["steps"][offset]["chosen"][idx]),
            "native y1/trace mismatch")
    tails = _prefix_tokens(raw, offset, pad)
    histories = [list(prompt)+tail for prompt, tail in zip(batch.prompt_token_ids, tails, strict=True)]
    ids, mask = padded_histories(histories, pad_token_id=pad)
    positions, _ = Qwen3VLModel.get_rope_index(SimpleNamespace(config=config), ids,
                                                batch.inputs["image_grid_thw"], None, mask)
    left_pad = ids.shape[1]-len(histories[idx])
    start = left_pad+len(batch.prompt_token_ids[idx])+current["start"]
    require(ids[idx, start:start+5].tolist() == item["raw"]["token_ids"][current["start"]:offset],
            "current native S input mismatch")
    require(mask[idx, start-18:start+5].tolist() == [1]*23, "history/S mask mismatch")
    spans = {"earlier": [start-18, start-9], "latest": [start-9, start], "S": [start, start+5]}
    rotary = {name: positions[:, idx, lo:hi].tolist() for name, (lo, hi) in spans.items()}
    for name, axes in rotary.items():
        require(all(axis == list(range(axis[0], axis[0]+len(axis))) for axis in axes),
                f"nonconsecutive {name} rotary span")
    require(all(rotary["latest"][a][0]-rotary["earlier"][a][0] == 9 and
                rotary["S"][a][0]-rotary["latest"][a][0] == 9 for a in range(3)),
            "nine-position geometry changed")
    return {"prompt_width_target": len(batch.prompt_token_ids[idx]),
            "history_raw_offset": current["start"], "first_y1_raw_offset": offset,
            "physical_unpadded": {k: [v[0]-left_pad, v[1]-left_pad] for k, v in spans.items()},
            "physical_full_batch_padded": spans, "rotary_position_ids": rotary,
            "source_step_full_batch_width": int(ids.shape[1]),
            "source_step_companion_tail_lengths": [min(offset,len(r["token_ids"])) for r in raw],
            "left_pad_target": int(left_pad)}


def main():
    selfcheck()
    require(literal_binding(BRIEF)["sha256"] == BRIEF_SHA, "brief changed")
    stage0 = json.loads(STAGE0.read_text())
    for saved in stage0["source_files"]:
        current = literal_binding(Path(saved["path"]))
        require(current == saved, f"stage0 source file changed: {saved['path']}")
    accepted = json.loads((UNIT / "manifest.json").read_text())
    require(accepted["source_bindings"]["source_identity"] == stage0["effective_identity"],
            "accepted native-x1 / Stage0 effective identity mismatch")
    sources = _load(AutoTokenizer.from_pretrained(BASE, local_files_only=True))
    require(len(sources) == 273, "source denominator changed")
    ledger, eligible = [], {"R": [], "D": []}
    for global_rank, (key, item) in enumerate(sources.items()):
        source, image_id = key
        entry = {"global_source_rank": global_rank, "source": source,
                 "source_rank": item["source_rank"], "split": item["split"],
                 "image_id": image_id, "group": item["cell"]["group"],
                 "batch_index": int(item["cell"]["batch_index"]),
                 "complete_rows": len(item["rows"]), "native_termination": item["raw"]["stop"]}
        try:
            bindings = _source_bindings(item)
            require(bindings["source_identity"] == stage0["effective_identity"],
                    "effective identity changed")
            entry["provenance"] = "verified_raw_trace_receipt_image_effective_identity"
        except (ValueError, KeyError, OSError) as exc:
            entry["provenance"] = "HOLD"
            entry["hold_reason"] = str(exc)
            ledger.append(entry)
            continue
        if (item["split"], image_id) == ("train", 477415):
            entry["excluded"] = "previously_studied_train477415"
            ledger.append(entry)
            continue
        found = earliest(item)
        for kind in ("R", "D"):
            triplet = found[kind]
            entry[f"earliest_{kind}"] = None if triplet is None else {
                "antecedent_rows": [triplet[0]["index"], triplet[1]["index"]],
                "current_row": triplet[2]["index"],
                "first_y1_raw_offset": triplet[2]["coordinate_offsets"][1]}
            if triplet is not None:
                eligible[kind].append((key, item, triplet, global_rank, bindings))
        entry["screen_exclusion"] = ("no_adjacent_canonical_same_class_repeated_antecedents"
                                      if entry["earliest_R"] is None else None)
        entry["different_antecedent_exclusion"] = ("no_adjacent_canonical_same_class_different_antecedents"
                                                   if entry["earliest_D"] is None else None)
        ledger.append(entry)
    r = eligible["R"][:4]
    rkeys = {x[0] for x in r}
    d = [x for x in eligible["D"] if x[0] not in rkeys][:4]
    picked = [("R", x) for x in r]+[("D", x) for x in d]
    require(len(picked) <= 8 and len({x[1][0] for x in picked}) == len(picked), "selection duplication")
    q = load_qwen_components_from_options(QwenLoadOptions(
        base_model=str(BASE), dtype="fp32", attn_implementation="sdpa",
        patch_embed_linearization="enabled", load_model=False))
    require(q.model is None, "CPU proposal loaded language model")
    config = AutoConfig.from_pretrained(BASE, local_files_only=True)
    panels = {"mature": json.loads((MATURE/"panel.json").read_text()),
              "new": json.loads((CENSUS/"panel.json").read_text())}
    group_cache = {}
    selected = []
    reviewed = stage0["families"]
    for kind, (key, item, triplet, global_rank, bindings) in picked:
        group_key = (item["source"], item["cell"]["group"])
        if group_key not in group_cache:
            boundary = {"group": item["cell"]["group"], "batch_index": item["cell"]["batch_index"],
                        "image_id": item["image_id"], "raw_path": bindings["raw"]["path"],
                        "trace_path": bindings["trace"]["path"],
                        "receipt_path": bindings["runtime_receipt"]["path"],
                        "native_tokens": item["raw"]["token_ids"],
                        "native_token_hash": token_hash(item["raw"]["token_ids"])}
            group_cache[group_key] = _source(boundary, "untied", panels[item["source"]], q, torch.device("cpu"))
        batch, raw, trace, group, planning = group_cache[group_key]
        receipt = json.loads(Path(bindings["runtime_receipt"]["path"]).read_text())
        require(input_identity(batch) == receipt["input_identity"], "processor full-batch identity drift")
        earlier, latest, current = triplet
        offset = current["coordinate_offsets"][1]
        require(item["raw"]["token_ids"][offset-1] == 151670+current["values"][0] and
                item["raw"]["token_ids"][offset] == 151670+current["values"][1],
                "native x1/y1 mismatch")
        geo = geometry(item, triplet, batch, raw, trace, config, int(q.tokenizer.pad_token_id))
        selected.append({"cohort": kind, "id": f"{item['source']}:{item['image_id']}:{current['index']}",
                         "global_source_rank": global_rank, "source_rank": item["source_rank"],
                         "source": item["source"], "split": item["split"], "image_id": item["image_id"],
                         "group": item["cell"]["group"], "batch_index": int(item["cell"]["batch_index"]),
                         "request_id": batch.request_ids[int(item["cell"]["batch_index"])],
                         "class": current["description"], "antecedent_rows": [earlier["index"],latest["index"]],
                         "current_row": current["index"], "antecedent_boxes": [earlier["values"],latest["values"]],
                         "current_box": current["values"],
                         "current_equals_earlier": item["raw"]["token_ids"][current["start"]:current["end"]] == item["raw"]["token_ids"][earlier["start"]:earlier["end"]],
                         "current_equals_latest": item["raw"]["token_ids"][current["start"]:current["end"]] == item["raw"]["token_ids"][latest["start"]:latest["end"]],
                         "native_x1_token": item["raw"]["token_ids"][offset-1],
                         "native_y1_token": item["raw"]["token_ids"][offset],
                         "native_y1_trace_parity": True,
                         "native_termination": item["raw"]["stop"], "native_token_count": len(item["raw"]["token_ids"]),
                         "source_bindings": bindings,
                         "source_input_identity_sha256": bindings["input_identity_sha256"],
                         "batch_companions": [{"batch_index": i, "request_id": request_id,
                                               "image_id": int(source_row["image_id"]),
                                               "native_token_count": len(source_row["token_ids"]),
                                               "stop": source_row["stop"],
                                               "prompt_width": len(batch.prompt_token_ids[i]),
                                               "image_grid": batch.image_grids[i]}
                                              for i, (request_id, source_row) in enumerate(zip(batch.request_ids, raw, strict=True))],
                         "pixel_elements_full_batch": int(batch.inputs["pixel_values"].numel()),
                         "geometry": geo,
                         "processor_planning": {k: planning[k] for k in planning
                                                if k not in ("receipt_input_identity", "replayed_input_identity")},
                         "existing_review_annotations": [{"id": f["id"], "kind": f["kind"],
                                                         "status": f["status"], "review_note": f["review_note"],
                                                         "candidate_holds": [{"owner_id": c["owner_id"],
                                                                              "branch_status": c["branch_status"]}
                                                                             for c in f["candidates"]
                                                                             if "HOLD" in c["branch_status"]]}
                                                         for f in reviewed if f["source"] == item["source"] and f["image_id"] == item["image_id"]]})
    registry = {"schema": "recurrence_native_x1_phase.cross_image_cpu.v1", "status": "candidate_lead_review",
                "brief": literal_binding(BRIEF), "stage0_selection": literal_binding(STAGE0),
                "accepted_native_x1_manifest": literal_binding(UNIT/"manifest.json"),
                "accepted_native_x1_result": literal_binding(UNIT/"candidate-results.md"),
                "effective_identity": stage0["effective_identity"], "source_files": stage0["source_files"],
                "counts": {"source_images": len(ledger), "excluded_prior_image": 1,
                           "provenance_hold": sum(x["provenance"] == "HOLD" for x in ledger),
                           "R_eligible_images": len(eligible["R"]), "D_eligible_images": len(eligible["D"]),
                           "R_selected": len(r), "D_selected": len(d),
                           "R_D_overlap_eligible": len({x[0] for x in eligible["R"]}&{x[0] for x in eligible["D"]})},
                "source_screen": ledger, "selected": selected,
                "call_plan_per_case": ["native_historical_prefill", "native_first_y1_anchor",
                                       "S_minus9_cross", "native_S_latest_identity_sham",
                                       "latest_K_plus9", "earlier_K_plus9"],
                "model_forwards": 6*len(selected), "vision_forwards": len(selected),
                "free_generation_tokens": 0, "gpu_hours_spent": 0}
    write_pretty_json(REGISTRY, registry)
    print(json.dumps({"counts": registry["counts"], "selected": [(x["cohort"],x["id"]) for x in selected],
                      "registry": literal_binding(REGISTRY)}))


if __name__ == "__main__":
    main()
