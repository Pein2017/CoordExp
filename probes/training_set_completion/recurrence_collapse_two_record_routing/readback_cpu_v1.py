"""Versioned CPU readback of the frozen two-record production attempt."""
from __future__ import annotations

import argparse
import inspect
import json
import shutil
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from . import run as frozen


UNIT = frozen.UNIT
RAW = frozen.OUT
OUT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-24-recurrence-collapse-two-record-routing/cpu-readback-v1")
RULING = UNIT / "lead-readback-repair-ruling-v1.json"
PREFLIGHT = UNIT / "supporting/cpu-readback-v1-preflight.json"
RULING_SHA = "958e4a1de44a6a648ad6d1d812b812576e10c10ed0fe8e9711e4dde887119da1"
bind, require, write_new = frozen.bind, frozen.require, frozen.write_new


def guard_written_records(value, arm):
    """The receipt is JSON: accept exactly an ordered two-element list."""
    return type(value) is list and value == list(frozen.PATTERNS[arm])


def checked():
    admission, original = frozen.checked()
    require(bind(RULING)["sha256"] == RULING_SHA, "readback ruling changed")
    ruling = json.loads(RULING.read_text())
    for name in ("protocol", "original_admission", "failure_report", "failure_receipt",
                 "lead_verification", "frozen_producer", "preflight",
                 "production_receipt", "production_outer"):
        require(bind(ruling[name]["path"]) == ruling[name], f"ruling binding changed: {name}")
    require(ruling["new_code_path"] == str(Path(__file__).resolve()) and
            ruling["new_output_root"] == str(OUT) and
            ruling["max_model_loads"] == ruling["max_model_forwards"] ==
            ruling["max_vision_forwards"] == ruling["max_cuda_calls"] ==
            ruling["max_gpu_seconds"] == 0, "CPU-only scope changed")
    return admission, original, ruling


def selfcheck():
    admission, original, ruling = checked()
    require(not PREFLIGHT.exists() and not OUT.exists(), "CPU readback already prepared")
    receipt = json.loads((RAW / "receipt.json").read_text())
    require([x["arm"] for x in receipt["arms"]] == list(frozen.ARMS), "arm order changed")
    checks = []
    for record in receipt["arms"]:
        arm = record["arm"]
        saved = json.loads(json.dumps(record["written_records"]))
        require(type(saved) is list and not (saved == frozen.PATTERNS[arm]) and
                guard_written_records(saved, arm), f"RED/GREEN failed: {arm}")
        checks.append({"arm": arm, "old_actual_guard_rejects": True,
                       "new_actual_guard_accepts": True})
        wrong = [
            ("dropped", saved[:1]),
            ("extra", saved + ["A"]),
            ("wrong_earlier", ["F" if saved[0] == "A" else "A", saved[1]]),
            ("wrong_latest", [saved[0], "F" if saved[1] == "A" else "A"]),
            ("wrong_container", tuple(saved)),
        ]
        if saved[0] != saved[1]:
            wrong.append(("swapped", saved[::-1]))
        for label, value in wrong:
            require(not guard_written_records(value, arm), f"guard accepted {arm}/{label}")
            checks.append({"arm": arm, "rejected": label})
    OUT.mkdir(parents=True, exist_ok=False)
    captures = []
    paths = [Path(__file__), Path(frozen.__file__), Path(frozen.base.__file__),
             Path(inspect.getfile(frozen.parse_row)), Path(inspect.getfile(frozen.iou_xyxy)),
             Path(inspect.getfile(frozen.create_causal_mask)),
             Path(inspect.getfile(frozen.modeling_qwen3_vl))]
    for path in dict.fromkeys(paths):
        relative = (path.relative_to(frozen.REPO) if path.is_relative_to(frozen.REPO)
                    else Path("external") / path.name)
        copy = OUT / "sources" / relative
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, copy)
        require(bind(path)["sha256"] == bind(copy)["sha256"], "CPU source copy changed")
        captures.append({"maintained": bind(path), "capture": bind(copy)})
    packet = {"status": "cpu_reader_qualified_before_saved_vectors",
              "ruling": bind(RULING), "original_preflight": bind(frozen.PREFLIGHT),
              "original_receipt": bind(RAW / "receipt.json"),
              "original_outer": bind(RAW / "outer.json"),
              "failed_reader": bind(frozen.__file__), "reader": bind(Path(__file__)),
              "checks": checks, "direct_source_captures": captures,
              "commands": {"selfcheck": ["python", "-B", "-m",
                           "probes.training_set_completion.recurrence_collapse_two_record_routing.readback_cpu_v1",
                           "selfcheck"],
                           "readback": ["python", "-B", "-m",
                           "probes.training_set_completion.recurrence_collapse_two_record_routing.readback_cpu_v1",
                           "readback"]},
              "model_loads": 0, "model_forwards": 0, "vision_forwards": 0,
              "cuda_calls": 0, "gpu_seconds": 0}
    write_new(PREFLIGHT, packet)
    print(json.dumps({"status": packet["status"], "guard_checks": len(checks),
                      "captures": len(captures)}))


def readback():
    admission, original, ruling = checked()
    pre = json.loads(PREFLIGHT.read_text())
    require(pre["status"] == "cpu_reader_qualified_before_saved_vectors" and
            pre["reader"] == bind(Path(__file__)) and pre["ruling"] == bind(RULING) and
            pre["original_receipt"] == bind(RAW / "receipt.json") and
            not (OUT / "readback.json").exists(), "CPU reader/preflight changed")
    for item in pre["direct_source_captures"]:
        require(bind(item["maintained"]["path"]) == item["maintained"] and
                bind(item["capture"]["path"]) == item["capture"],
                "CPU direct source capture changed")
    receipt = json.loads((RAW / "receipt.json").read_text())
    outer = json.loads((RAW / "outer.json").read_text())
    counts = receipt["counts"]
    require(receipt["status"] == "candidate_complete" and
            receipt["admission"] == bind(frozen.ADMISSION) and
            [x["arm"] for x in receipt["arms"]] == list(frozen.ARMS) and
            [len(x["steps"]) for x in receipt["arms"]] == [9] * 5 and
            counts == {"model_forwards": 45, "vision_forwards": 45,
                       "emitted_target_tokens": 45, "reused_calls": 0} and
            outer["returncode"] == 0 and outer["terminal"] and
            outer["child_pid"] == receipt["terminal_pid"] and
            not Path(f"/proc/{outer['child_pid']}").exists() and
            abs(outer["outer_seconds"] - 129.19039377570152) < 1e-9,
            "terminal production/count/cost changed")
    q = frozen.base.load_qwen_components_from_options(frozen.base.QwenLoadOptions(
        base_model=str(frozen.base.BASE), dtype="fp32", attn_implementation="sdpa",
        patch_embed_linearization="enabled", load_model=False))
    require(q.model is None, "CPU reader loaded a language model")
    batch, raw, trace, source_receipt, planning = frozen.source(q, admission, torch.device("cpu"))
    require(int(q.tokenizer.pad_token_id) == original["pad_id"] and
            sorted(q.tokenizer.all_special_ids) == original["special_ids"],
            "CPU tokenizer/source changed")
    source_identity = {"status": "original_full_batch_cpu_reconstructed",
                       "admission": bind(frozen.ADMISSION),
                       "source_runtime_receipt": bind(admission["source_bindings"]["runtime_receipt"]["path"]),
                       "input_identity_sha256": admission["source_bindings"]["input_identity_sha256"],
                       "request_ids": list(batch.request_ids),
                       "prompt_lengths": list(map(len, batch.prompt_token_ids)),
                       "raw_lengths": [len(x["token_ids"]) for x in raw],
                       "pixel_elements": int(batch.inputs["pixel_values"].numel()),
                       "image_grids": [list(x) for x in batch.image_grids],
                       "pad_id": original["pad_id"], "model_loads": 0,
                       "model_forwards": 0, "vision_forwards": 0, "cuda_calls": 0}
    write_new(OUT / "source-identity.json", source_identity)
    pad = original["pad_id"]
    special = frozenset(original["special_ids"])
    fixture = frozen.base.ConfigOnlyRope()
    result = {"status": "candidate_cold_readback_passed", "ruling": bind(RULING),
              "reader": bind(Path(__file__)), "cpu_preflight": bind(PREFLIGHT),
              "source_identity": bind(OUT / "source-identity.json"),
              "production_receipt": bind(RAW / "receipt.json"),
              "production_outer": bind(RAW / "outer.json"),
              "prior_readback_failure": bind(ruling["failure_receipt"]["path"]),
              "arms": [], "counts": counts, "vectors_loaded": 0,
              "input_files_loaded": 0, "model_loads": 0,
              "model_forwards": 0, "vision_forwards": 0,
              "cuda_calls": 0, "gpu_seconds": 0}
    native = []
    for record in receipt["arms"]:
        arm = record["arm"]
        require(guard_written_records(record["written_records"], arm),
                "cold written donors changed")
        emitted = []
        summaries = []
        for t, entry in enumerate(record["steps"]):
            require(entry["step"] == t and entry["actual_layers"] == list(range(28)) and
                    entry["changed_history_raw"] == frozen.CHANGED[arm] and
                    bind(entry["raw"]["path"]) == entry["raw"] and
                    bind(entry["inputs"]["path"]) == entry["inputs"],
                    "cold vector/input/layer binding changed")
            full = {key: torch.tensor(value, dtype=torch.long) for key, value in
                    json.loads(Path(entry["inputs"]["path"]).read_text()).items()}
            result["input_files_loaded"] += 1
            payload = torch.load(entry["raw"]["path"], map_location="cpu", weights_only=True)
            result["vectors_loaded"] += 1
            logits = payload["logits"]
            require(payload["arm"] == arm and payload["step"] == t and
                    logits.shape == (4, 152670) and torch.isfinite(logits).all().item() and
                    payload["historical_by_layer"].shape[0] ==
                    payload["companions_by_layer"].shape[0] == 28,
                    "cold full vectors/states incomplete")
            expected = frozen.step_inputs(fixture, batch, raw, pad, arm, emitted,
                                          previous=emitted[-1] if t else None)
            mask, native_mask = frozen.mask_for(expected, arm)
            require(all(torch.equal(full[key], expected[key]) for key in frozen.KEYS) and
                    torch.equal(payload["actual_mask"], native_mask) and
                    entry["native_mask_hash"] == entry["actual_mask_hash"] ==
                    frozen.base.tensor_hash(native_mask) and
                    entry["actual_layer_mask_hashes"] == [entry["actual_mask_hash"]] * 28,
                    "cold original input/position/native mask/consumer changed")
            seen = {"actual_input": entry["input_hashes"], "layers": entry["actual_layers"],
                    "layer_mask_hashes": entry["actual_layer_mask_hashes"],
                    "expected_mask": native_mask}
            frozen.verify_step(fixture, batch, raw, pad, arm, emitted, full, seen,
                               logits, entry["chosen"], previous=emitted[-1] if t else None)
            if arm == frozen.ARMS[0]:
                parity = [frozen.base._trace_compare(
                    logits=logits[i], trace=trace, batch_index=i,
                    absolute_offset=frozen.OFFSET + t,
                    token_id=raw[i]["token_ids"][frozen.OFFSET + t],
                    role="cpu_v1_native_AF") for i in range(4)]
                require(all(x["passed"] for x in parity) and entry["chosen"] == frozen.ROW2[t],
                        "cold original native source trace failed")
                native.append(payload)
            else:
                parity = [frozen.base._trace_compare(
                    logits=logits[i], trace=trace, batch_index=i,
                    absolute_offset=frozen.OFFSET + t,
                    token_id=raw[i]["token_ids"][frozen.OFFSET + t],
                    role="cpu_v1_companion") for i in (0, 1, 3)]
                require(all(x["passed"] for x in parity), "cold companion source trace failed")
                if t < 9:
                    old = native[t]
                    require(max(
                        float((payload["companions_by_layer"] - old["companions_by_layer"]).abs().max()),
                        max(float((logits[i] - old["logits"][i]).abs().max()) for i in (0, 1, 3)))
                        <= frozen.TOL, "cold companion state/vector differs")
                    if arm == frozen.ARMS[1]:
                        require(max(
                            float((payload["historical_by_layer"] - old["historical_by_layer"]).abs().max()),
                            float((logits - old["logits"]).abs().max())) <= frozen.TOL and
                            entry["chosen"] == frozen.ROW2[t],
                            "cold independent sham state/vector differs")
                else:
                    require(entry["matched_native_full_vector_state"] ==
                            "UNAVAILABLE_not_executed", "unexecuted native comparator fabricated")
            vector = logits[frozen.TARGET].double()
            top = torch.topk(vector, 2)
            summaries.append({"step": t, "chosen": entry["chosen"],
                              "top2_ids": top.indices.tolist(),
                              "top2_logits": top.values.tolist(),
                              "log_normalizer": float(torch.logsumexp(vector, -1)),
                              "chosen_logprob": float(torch.log_softmax(vector, -1)[entry["chosen"]]),
                              "raw": entry["raw"], "inputs": entry["inputs"]})
            emitted.append(entry["chosen"])
            stop = frozen.parse_row(emitted, special)["stop"]
            require(stop is None if t < len(record["steps"]) - 1 else stop == record["stop"],
                    "cold parser/terminator changed")
        require(emitted == record["emitted"], "cold own greedy prefix changed")
        if arm in frozen.ARMS[:2]:
            require(record["stop"] == "complete" and emitted == list(frozen.ROW2),
                    "cold native/sham source qualification failed")
        parsed = frozen.parse_row(emitted, special)
        description = q.tokenizer.decode(parsed["description_ids"],
                                         skip_special_tokens=False,
                                         clean_up_tokenization_spaces=False)
        box = [x - 151670 for x in parsed["box_ids"]] if len(parsed["box_ids"]) == 4 else None
        geometry = ("valid" if box[0] < box[2] and box[1] < box[3] else "invalid") if box else "no_complete_box"
        divergence = next((i for i, (x, y) in enumerate(zip(emitted, frozen.ROW2)) if x != y),
                          min(len(emitted), len(frozen.ROW2)))
        result["arms"].append({"arm": arm, "written_records": list(frozen.PATTERNS[arm]),
                               "emitted": emitted, "stop": record["stop"],
                               "description_ids": parsed["description_ids"],
                               "description": description, "box": box,
                               "geometry": geometry, "first_native_divergence": divergence,
                               "steps": summaries})
    require(result["vectors_loaded"] == result["input_files_loaded"] == 45 and
            len(result["arms"]) == 5, "cold 45-vector/90-path coverage changed")
    a_box, f_box = admission["source"]["A_box"], admission["source"]["F_box"]
    def classify(row):
        if row["stop"] != "complete":
            return row["stop"]
        if row["description_ids"] != [8987]:
            return "other_class"
        if row["geometry"] != "valid":
            return "invalid_geometry"
        ia, iff = frozen.iou_xyxy(row["box"], a_box), frozen.iou_xyxy(row["box"], f_box)
        row["iou_A"], row["iou_F"] = ia, iff
        if min(abs(ia - .5), abs(iff - .1), abs(iff - .5), abs(ia - .1)) <= 1e-6:
            return "numerical_HOLD"
        if ia >= .5 and iff <= .1:
            return "broad_A"
        if iff >= .5 and ia <= .1:
            return "fragment_F"
        return "neither_region"
    for row in result["arms"]:
        row["region"] = classify(row)
    components = {"FF_broad_A": result["arms"][2]["region"] == "broad_A",
                  "AA_fragment_F": result["arms"][3]["region"] == "fragment_F",
                  "FA_fragment_F": result["arms"][4]["region"] == "fragment_F"}
    result["primary_components"] = components
    result["shared_primary_pass"] = all(components.values())
    image = Image.open(admission["source_bindings"]["image"]["path"]).convert("RGB")
    draw = ImageDraw.Draw(image)
    for label, box, color in (("A", a_box, "#00ee77"), ("F", f_box, "#ff5533"),
                              ("native AF", result["arms"][0]["box"], "#eeb000"),
                              ("FF", result["arms"][2]["box"], "#00d9ff"),
                              ("AA", result["arms"][3]["box"], "#ee00dd"),
                              ("FA", result["arms"][4]["box"], "#ffffff")):
        if box is None:
            continue
        xy = [round(v * (image.width if k % 2 == 0 else image.height) / 1000)
              for k, v in enumerate(box)]
        draw.rectangle(xy, outline=color, width=4)
        draw.text((xy[0], max(0, xy[1] - 14)), label, fill=color)
    image.save(OUT / "overlay.png")
    result["overlay"] = bind(OUT / "overlay.png")
    write_new(OUT / "readback.json", result)
    print(json.dumps({"status": result["status"],
                      "regions": {x["arm"]: x["region"] for x in result["arms"]},
                      "components": components,
                      "shared_primary": result["shared_primary_pass"],
                      "vectors_loaded": result["vectors_loaded"],
                      "gpu_seconds_added": 0}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("selfcheck", "readback"))
    {"selfcheck": selfcheck, "readback": readback}[parser.parse_args().action]()


if __name__ == "__main__":
    main()
