"""Cold CPU qualification of the three finalized bowl-progress cells only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path("/data/CoordExp/.worktrees/research-probes")
UNIT = ROOT / "research/experiments/2026-09-24-recurrence-first-revisit-routing"
OUT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-24-recurrence-first-revisit-routing")
FAILED = OUT / "native-read-remaining-v1"
ACCEPTED = OUT / "native-read-attempt-001"
DEST = UNIT / "supporting/native-read-bowl-progress-cpu-readback-v1.json"
CONDITIONS = ("native", "identity-mask-sham", "latest-row-mask")
TOKENS = {"B": 151827, "A1": 151675, "A0": 151670}
TOL = 2e-4


def require(value, message):
    if not value:
        raise ValueError(message)


def bind(path):
    path = Path(path)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size}


def checked(expected):
    actual = bind(expected["path"])
    require(actual["sha256"] == expected["sha256"] and
            ("size_bytes" not in expected or actual["size_bytes"] == expected["size_bytes"]),
            f"binding changed: {expected['path']}")
    return actual


def thash(value):
    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def stats(logits):
    v = np.asarray(logits[3], dtype=np.float64)
    require(v.ndim == 1 and np.isfinite(v).all(), "nonfinite target vector")
    peak = float(v.max())
    logz = peak + float(np.log(np.exp(v - peak).sum()))
    p = np.exp(v - logz)
    top = np.argsort(-v, kind="stable")[:2]
    return {"global_top2_ids": top.tolist(), "global_top2_logits": v[top].tolist(),
            "global_gap": float(v[top[0]] - v[top[1]]),
            "tokens": {name: {"token_id": tid, "logit": float(v[tid]),
                               "prob": float(p[tid]), "logprob": float(v[tid] - logz),
                               "rank": int(np.count_nonzero(v > v[tid]) + 1)}
                       for name, tid in TOKENS.items()}}, p


def run():
    require(not DEST.exists(), "versioned CPU readback already exists")
    ruling_path = UNIT / "lead-native-read-failure-ruling-v1.json"
    ruling = bind(ruling_path)
    require(ruling["sha256"] == "2d6d85de592a5eb06fa807e0f303900f22e5e78deb2ee11c151d215287c544ea",
            "lead failure ruling changed")
    rule = json.loads(ruling_path.read_text())
    require(rule["status"] == "lead-ruling-GPU-stopped-CPU-only-readback-admitted" and
            rule["cost"]["CPU_assignment_max_model_calls"] == 0 and
            rule["cost"]["CPU_assignment_max_GPU_seconds"] == 0,
            "CPU-only authority changed")
    for name in ("failed_attempt", "failure_audit", "independent_failure_verification",
                 "preserve_failed_producer", "protocol", "supersedes_launch_authority",
                 "first_case_acceptance"):
        checked(rule[name])
    receipt = json.loads(Path(rule["failed_attempt"]["path"]).read_text())
    require(receipt["status"] == "technical_invalid" and
            receipt["counts"] == {"model_forwards": 4, "vision_forwards": 4, "free_tokens": 0} and
            [(c["state"], c["condition"]) for c in receipt["cells"]] ==
            [("bowl-row2", c) for c in CONDITIONS], "failed seven-cell denominator changed")
    preflight_binding = checked(receipt["preflight"])
    preflight = json.loads(Path(preflight_binding["path"]).read_text())
    checked(receipt["producer"])
    require(receipt["producer"] == preflight["producer"] and
            receipt["admission"] == preflight["admission"] and
            receipt["effective_identity"]["model"] == "untied", "producer/model/admission changed")
    for item in preflight["direct_source_captures"]:
        require(checked(item["maintained"])["sha256"] == checked(item["capture"])["sha256"],
                "source capture differs from maintained import")
    acceptance = json.loads(Path(rule["first_case_acceptance"]["path"]).read_text())
    require(acceptance["status"] == "lead-accepted", "first case acceptance changed")
    checked(acceptance["receipt"])
    prior_binding = checked(acceptance["readback"])
    prior = json.loads(Path(prior_binding["path"]).read_text())
    require(prior["status"] == "cold_readback_passed", "prior bowl readback changed")
    prior_inputs_binding = bind(ACCEPTED / "inputs.json")
    prior_inputs = json.loads(Path(prior_inputs_binding["path"]).read_text())

    registry_binding = bind(UNIT / "supporting/native-window-candidates.json")
    registry = json.loads(Path(registry_binding["path"]).read_text())
    image = registry["images"]["mature:313465:0"]
    source = preflight["sources"]["bowl-row2"]
    require(source["bindings"] == image["source_bindings"] and
            image["group"] == "fresh-18" and image["batch_index"] == 3 and
            image["full_batch_target_lengths"] == [286, 109, 11, 138] and
            receipt["effective_identity"]["model"] == image["model_identity"]["model"],
            "frozen bowl source identity changed")
    source_bindings = {name: checked(expected) for name, expected in source["bindings"].items()}
    original_receipt = json.loads(Path(source_bindings["runtime_receipt"]["path"]).read_text())
    require(source["input_identity"] == original_receipt["input_identity"] and
            source["input_identity"]["request_ids"] == image["request_ids"],
            "original four-request input identity changed")
    raw = json.loads(Path(source_bindings["raw"]["path"]).read_text())["rows"]
    trace = json.loads(Path(source_bindings["trace"]["path"]).read_text())
    require(len(raw) == 4 and [len(r["token_ids"]) for r in raw] == [286, 109, 11, 138] and
            raw[3]["token_ids"][:26] == [151646,65,9605,151647,151648,151670,151827,151947,152174,151649,
                                         151646,65,9605,151647,151648,151675,151867,151887,152077,151649,
                                         151646,65,9605,151647,151648,151827] and
            raw[2]["token_ids"][-1] == 151645, "original target rows/EOS changed")

    saved_binding = bind(FAILED / "inputs-bowl-row2.json")
    saved = json.loads(Path(saved_binding["path"]).read_text())
    full = {name: torch.tensor(saved[name], dtype=torch.long) for name in
            ("input_ids", "attention_mask", "position_ids", "cache_position")}
    require(all(thash(tensor) == preflight["inputs"]["bowl-row2"][name]
                for name, tensor in full.items()), "saved full-batch input/position hash changed")
    ids, mask, pos, cache = (full[k] for k in ("input_ids", "attention_mask", "position_ids", "cache_position"))
    require(tuple(ids.shape) == tuple(mask.shape) == (4, 1387) and
            tuple(pos.shape) == (3, 4, 1387) and cache.tolist() == list(range(1387)) and
            preflight["shapes"]["bowl-row2"]["prompt_width"] == 1362 and
            preflight["shapes"]["bowl-row2"]["query_physical"] == [1382, 1387] and
            preflight["shapes"]["bowl-row2"]["latest_physical"] == [1372, 1382],
            "full-batch shape/physical positions changed")
    prompt_rows = source["input_identity"]["prompt_token_ids"]
    for i, (prompt, row) in enumerate(zip(prompt_rows, raw, strict=True)):
        suffix = row["token_ids"][:25]
        if len(suffix) < 25:
            suffix = suffix[:suffix.index(151645) + 1] + [151643] * (25 - suffix.index(151645) - 1)
        history = prompt + suffix
        left = 1387 - len(history)
        require(left >= 0 and ids[i].tolist() == [151643] * left + history and
                mask[i].tolist() == [0] * left + [1] * len(history),
                f"full-batch tokens/EOS/pad/mask differ: row {i}")
    require(raw[2]["token_ids"][:11] == ids[2, 1362:1373].tolist() and
            ids[2, 1373:].tolist() == [151643] * 14,
            "ended companion EOS/pad boundary changed")
    require(prior_inputs["full_input_ids"] == ids[:, :1377].tolist() and
            prior_inputs["source_attention_mask"] == mask[:, :1377].tolist() and
            prior_inputs["position_ids"] == pos[:, :, :1377].tolist(),
            "shared native prefix/position differs from accepted bowl-row1")
    for axis in range(3):
        for i in range(4):
            last = int(pos[axis, i, 1376])
            require(pos[axis, i, 1377:].tolist() == list(range(last + 1, last + 11)),
                    "continuation text positions changed")

    width = 1387
    base = ((torch.arange(width)[:, None] >= torch.arange(width)[None, :])[None, None] &
            mask[:, None, None, :].bool())
    rect = (3, 0, slice(1382, 1387), slice(1372, 1382))
    require(bool(base[rect].all()), "latest row not readable in native mask")
    observed, vectors, complements, gate = {}, {}, [], {}
    for index, cell in enumerate(receipt["cells"], 1):
        condition = cell["condition"]
        checkpoint_binding = bind(FAILED / f"checkpoint-{index}.json")
        checkpoint = json.loads(Path(checkpoint_binding["path"]).read_text())
        require(checkpoint["cells"] == receipt["cells"][:index], "finalized checkpoint differs")
        checked(cell["raw"])
        require(cell["actual_mask_layers"] == list(range(28)) and
                cell["query_physical"] == [1382, 1387] and
                cell["key_physical"] == [1372, 1382],
                f"actual layer/rectangle evidence missing: {condition}")
        actual = mask if condition == "native" else base.clone()
        if condition == "latest-row-mask":
            actual[rect] = False
        seen = base if condition == "native" else actual
        complement = seen.clone()
        complement[rect] = False
        require(cell["actual_input_hashes"] == {
                    "input_ids": thash(ids), "attention_mask": thash(actual),
                    "position_ids": thash(pos), "cache_position": thash(cache)} and
                cell["selected_true_count"] == int(seen[rect].sum()) and
                cell["complement_sha256"] == thash(complement),
                f"actual input/selected/complement differs: {condition}")
        complements.append(cell["complement_sha256"])
        payload = torch.load(cell["raw"]["path"], map_location="cpu", weights_only=True)
        require(payload["state"] == "bowl-row2" and payload["condition"] == condition and
                tuple(payload["logits"].shape)[0] == 4 and
                torch.isfinite(payload["logits"]).all().item() and
                payload["all_prior_history_by_layer"].shape[0] == 28 and
                payload["companion_last_by_layer"].shape[:2] == (28, 3),
                f"raw payload incomplete/nonfinite: {condition}")
        vectors[condition] = payload
        observed[condition], _ = stats(payload["logits"].numpy())
        gate[condition] = {"checkpoint": checkpoint_binding, "raw": cell["raw"],
                           "actual_mask_layers": cell["actual_mask_layers"],
                           "selected_true_count": cell["selected_true_count"],
                           "complement_sha256": cell["complement_sha256"]}
    require(len(set(complements)) == 1 and complements[0] ==
            "d67ae8026916f6d20b91ed5823df19abdd2ca3f07d43e7e6be65747ea888f169",
            "same-rectangle mask complements differ")
    native = vectors["native"]
    parity = []
    active = preflight["shapes"]["bowl-row2"]["active_trace_rows"]
    require(active == [0, 1, 3] and len(receipt["cells"][0]["source_trace_parity"]) == 3,
            "active source-trace boundary changed")
    step = trace["steps"][25]
    for i, recorded in zip(active, receipt["cells"][0]["source_trace_parity"], strict=True):
        v = native["logits"][i].float()
        chosen = int(raw[i]["token_ids"][25])
        top = torch.topk(v, 2)
        source_top = step["raw_top2"][i]
        if source_top and isinstance(source_top[0], (list, tuple)):
            source_ids = [int(item[0]) for item in source_top]
            source_logits = [float(item[1]) for item in source_top]
        else:
            source_ids = [int(step["raw_winners"][i]), int(step["raw_runnerups"][i])]
            source_logits = [float(item) for item in source_top]
        errors = {"chosen": abs(float(v[chosen]) - float(step["chosen_raw_logits"][i])),
                  "logsumexp": abs(float(torch.logsumexp(v, -1)) - float(step["logsumexp"][i])),
                  "logprob": abs(float(torch.log_softmax(v, -1)[chosen]) -
                                 (float(step["chosen_raw_logits"][i]) - float(step["logsumexp"][i]))),
                  "top2": max(abs(float(a) - b) for a, b in zip(top.values, source_logits, strict=True))}
        passed = (int(step["chosen"][i]) == chosen and top.indices.tolist() == source_ids and
                  max(errors.values()) <= TOL and recorded["passed"] and
                  recorded["token_id"] == chosen and recorded["current_top2_token_ids"] == source_ids)
        require(passed, f"cold active source trace mismatch: row {i}")
        parity.append({"batch_index": i, "chosen_token_id": chosen,
                       "global_top2_ids": source_ids, "errors": errors, "passed": passed})
    state_errors = {}
    for condition in CONDITIONS[1:]:
        x = vectors[condition]
        e = {"all_prior_history": float((x["all_prior_history_by_layer"] -
                                         native["all_prior_history_by_layer"]).abs().max()),
             "companion_last_hidden": float((x["companion_last_by_layer"] -
                                             native["companion_last_by_layer"]).abs().max())}
        if condition == "identity-mask-sham":
            e["all_batch_logits"] = float((x["logits"] - native["logits"]).abs().max())
            require(e["all_batch_logits"] == receipt["cells"][1]["all_batch_vector_max_error"],
                    "sham vector receipt differs")
        else:
            e["companion_logits"] = [float((x["logits"][i] - native["logits"][i]).abs().max())
                                     for i in range(3)]
            require(e["companion_logits"] == receipt["cells"][2]["companion_vector_max_errors"],
                    "treatment companion receipt differs")
        cell = receipt["cells"][CONDITIONS.index(condition)]
        require(e["all_prior_history"] == cell["all_prior_history_max_error"] and
                e["companion_last_hidden"] == cell["companion_last_states_max_error"] and
                max([e["all_prior_history"], e["companion_last_hidden"]] +
                    ([e["all_batch_logits"]] if condition == "identity-mask-sham" else e["companion_logits"])) <= TOL,
                f"historical/sham/companion state changed: {condition}")
        state_errors[condition] = e

    margins = {condition: {"B_minus_A1": float(v["logits"][3, TOKENS["B"]].double() -
                                               v["logits"][3, TOKENS["A1"]].double()),
                           "B_minus_A0": float(v["logits"][3, TOKENS["B"]].double() -
                                               v["logits"][3, TOKENS["A0"]].double())}
               for condition, v in vectors.items()}
    native_p = stats(native["logits"].numpy())[1]
    masked_p = stats(vectors["latest-row-mask"]["logits"].numpy())[1]
    deltas = {name: margins["latest-row-mask"][name] - margins["native"][name]
              for name in ("B_minus_A1", "B_minus_A0")}
    prior_delta = {"B_minus_A1": prior["primary_B_minus_A1"]["delta_mask_minus_native"],
                   "B_minus_A0": prior["secondary_B_minus_A0"]["delta_mask_minus_native"]}
    result = {"schema": "native_read_bowl_progress_cpu_readback.v1", "status": "candidate_cpu_qualified",
              "authority": ruling, "failed_receipt": rule["failed_attempt"],
              "failure_verification": rule["independent_failure_verification"],
              "preflight": preflight_binding, "producer": receipt["producer"],
              "direct_capture_count": len(preflight["direct_source_captures"]),
              "registry": registry_binding, "source_bindings": source_bindings,
              "original_source_request_ids": source["input_identity"]["request_ids"],
              "saved_inputs": saved_binding, "accepted_row1_inputs": prior_inputs_binding,
              "accepted_row1_readback": prior_binding,
              "full_batch": {"batch": 4, "width": width, "target": 3, "prompt_width": 1362,
                             "raw_lengths": [len(r["token_ids"]) for r in raw],
                             "prompt_lengths": [len(x) for x in prompt_rows],
                             "ended_companion": {"batch_index": 2, "eos_token_id": 151645,
                                                 "pad_token_id": 151643, "pad_tail_count": 14,
                                                 "after_eos_trace": "undefined"},
                             "position_check": "all shared 1377 slots equal accepted input; all three axes extend by one per text slot",
                             "input_tensor_hashes": preflight["inputs"]["bowl-row2"]},
              "cells": gate, "source_trace_parity": parity, "state_errors": state_errors,
              "observations": observed, "margins": margins,
              "latest_mask_minus_native": {"signed_deltas": deltas,
                                           "full_vocabulary_tv": float(0.5 * np.abs(native_p - masked_p).sum())},
              "accepted_row1_deltas": prior_delta,
              "row2_minus_accepted_row1_deltas": {k: deltas[k] - prior_delta[k] for k in deltas},
              "fixed_seven_cell_denominator": {"bowl_row2_native_sham_latest": "3 CPU-qualified candidate cells",
                                               "bowl_row2_earlier": "1 executed technical-invalid/unanswered",
                                               "cow_row1": "3 HOLD/unrun"},
              "cost": {**rule["cost"], "additional_model_forwards": 0,
                       "additional_vision_forwards": 0, "additional_GPU_seconds": 0}}
    DEST.parent.mkdir(parents=True, exist_ok=True)
    with DEST.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({"status": result["status"], "deltas": deltas,
                      "row2_minus_row1": result["row2_minus_accepted_row1_deltas"],
                      "TV": result["latest_mask_minus_native"]["full_vocabulary_tv"],
                      "readback": bind(DEST)}))


if __name__ == "__main__":
    run()
