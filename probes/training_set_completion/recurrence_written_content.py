"""Fixed-position row-content swap at one accepted native recurrence exit."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch

from probes.training_set_completion.artifacts import literal_binding
from probes.training_set_completion.numerical_feedback.runtime import _prefix_tokens
from probes.training_set_completion.recurrence_position_history import PANEL, TARGET, source_and_rows
from probes.training_set_completion.untied_shared import load_model
from src.artifacts.source_provenance import preserve_source
from src.inference.bound_requests import build_bound_native_requests
from src.qwen.input_identity import input_identity, tensor_hash
from src.qwen.native import exact_history_inputs, prepare_native_inputs


OUT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-written-content/attempt-001")
SPLIT_OUT = OUT.parent / "coordinate-split-001"
PRIMARY = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-position-history/attempt-002")
PAIR = OUT
PAIR_ACCEPTANCE = OUT.parent / "content-pair-acceptance.json"
MAX_FORWARDS = 4
MAX_SECONDS = 600
ATOL = 2e-4
R = [151646, 8987, 151647, 151648, 151670, 152241, 151708, 152245, 151649]
B = [151646, 8987, 151647, 151648, 151670, 152241, 152669, 152669, 151649]
X = R[:6] + [B[6], R[7], R[8]]
Y = R[:7] + [B[7], R[8]]


def require(value, message):
    if not value:
        raise ValueError(message)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def changed_inputs(native, current_S, replacement=B):
    ids = native["input_ids"]
    n = len(current_S)
    require(ids.ndim == 2 and ids.shape[0] == 4 and n == 6, "native batch/S length changed")
    require(ids[TARGET, -n:].tolist() == current_S, "current S changed before intervention")
    require(ids[TARGET, -(n + 9):-n].tolist() == R, "preceding row is not exact R")
    edited = ids.clone()
    require(replacement in (B, X, Y), "unsupported row-coordinate mutation")
    edited[TARGET, -(n + 9):-n] = torch.tensor(replacement, dtype=ids.dtype, device=ids.device)
    changed = torch.nonzero(edited != ids, as_tuple=False).tolist()
    expected = [[TARGET, ids.shape[1] - n - 3]] if replacement == X else [[TARGET, ids.shape[1] - n - 2]] if replacement == Y else [[TARGET, ids.shape[1] - n - 3], [TARGET, ids.shape[1] - n - 2]]
    require(changed == expected, f"wrong changed physical token slots: {changed}")
    require(edited[TARGET, -n:].tolist() == current_S and edited.shape == ids.shape, "mutation changed S or length")
    result = dict(native)
    result["input_ids"] = edited
    require(result["attention_mask"] is native["attention_mask"] and result["position_ids"] is native["position_ids"], "mask or positions changed")
    return result, changed


def selfcheck():
    s = [1, 2, 3, 4, 5, 6]
    sample = {"input_ids": torch.tensor([[0] * 15, [0] * 15, R + s, [0] * 15]),
              "attention_mask": torch.ones(4, 15, dtype=torch.long), "position_ids": torch.arange(15).expand(3, 4, 15).clone()}
    changed, indices = changed_inputs(sample, s)
    require(indices == [[2, 6], [2, 7]] and changed["input_ids"][2, 6:8].tolist() == B[6:8], "two-token mutation selfcheck")
    require(changed_inputs(sample, s, X)[1] == [[2, 6]] and changed_inputs(sample, s, Y)[1] == [[2, 7]], "singleton-coordinate mutation selfcheck")
    wrong = R.copy(); wrong[4] = 151671
    try:
        changed_inputs(sample, s, wrong)
    except ValueError:
        pass
    else:
        raise AssertionError("wrong coordinate token edit was admitted")
    for wrong in (s[:-1], s[:-1] + [7]):
        try:
            changed_inputs(sample, wrong)
        except ValueError:
            pass
        else:
            raise AssertionError("wrong S/length was admitted")


def hooks(model, expected_ids, expected_positions):
    text = model.model.language_model
    seen = {"embedding_inputs": [], "rotary_positions": [], "masks": [], "cache_slots": []}
    def embedding(_module, args):
        require(torch.equal(args[0], expected_ids), "embedding consumed altered or wrong input tokens")
        seen["embedding_inputs"].append(tensor_hash(args[0]))
    def rotary(_module, args):
        require(torch.equal(args[1], expected_positions), "rotary consumed changed MRoPE positions")
        seen["rotary_positions"].append(tensor_hash(args[1]))
    def attention(_module, _args, kwargs):
        mask, cache = kwargs.get("attention_mask"), kwargs.get("cache_position")
        seen["masks"].append(None if mask is None else tensor_hash(mask))
        seen["cache_slots"].append(None if cache is None else tensor_hash(cache))
        require(isinstance(kwargs.get("position_embeddings"), tuple), "attention missing consumed rotary embeddings")
    handles = [text.embed_tokens.register_forward_pre_hook(embedding), text.rotary_emb.register_forward_pre_hook(rotary),
               text.layers[0].self_attn.register_forward_pre_hook(attention, with_kwargs=True),
               text.layers[-1].self_attn.register_forward_pre_hook(attention, with_kwargs=True)]
    return seen, handles


def run(device, *, coordinate_split=False):
    selfcheck()
    require(torch.cuda.is_available() and device.startswith("cuda"), "CUDA required")
    output = SPLIT_OUT if coordinate_split else OUT
    require(not output.exists(), "attempt path already exists")
    chosen, raw, trace, receipt, panel, group, S, offsets, _ = source_and_rows()
    require(offsets[1] == 807 and raw[TARGET]["token_ids"][792:801] == R and raw[TARGET]["token_ids"][801:810] == B, "frozen row contents changed")
    primary_manifest_path = PRIMARY / "source-to-cell.json"
    primary_vector_path = PRIMARY / "L_L.pt"
    primary_result_path = PRIMARY / "result.json"
    primary_manifest = json.loads(primary_manifest_path.read_text())
    require(primary_manifest["cells"]["L_L"]["history"] == "L" and primary_manifest["cells"]["L_L"]["S_positions"] == "L", "primary late diagonal changed")
    primary_result = json.loads(primary_result_path.read_text())
    require(primary_result["status"] == "candidate", "primary candidate missing")
    require(literal_binding(primary_vector_path) == primary_result["cells"]["L_L"]["logits_binding"], "primary full-vocabulary vector binding changed")
    pair_result_path, pair_R_path, pair_B_path = PAIR / "result.json", PAIR / "R.pt", PAIR / "B.pt"
    pair_result = json.loads(pair_result_path.read_text()) if coordinate_split else None
    if coordinate_split:
        acceptance = json.loads(PAIR_ACCEPTANCE.read_text())
        require(acceptance["status"] == "lead-accepted" and pair_result["status"] == "candidate", "accepted pair unavailable")
        require(literal_binding(pair_R_path) == pair_result["cells"]["R"]["full_logits"] and literal_binding(pair_B_path) == pair_result["cells"]["B"]["full_logits"], "accepted pair vectors changed")
    output.mkdir(parents=True)
    state = {"status": "preparing", "pid": os.getpid(), "device": device, "model_forwards": 0, "vision_forwards": 0,
             "source": {"primary_manifest": literal_binding(primary_manifest_path), "primary_vector": literal_binding(primary_vector_path), "primary_result": literal_binding(primary_result_path),
                        "selection": primary_manifest["source"]["selection"], **{k: chosen[k] for k in ("raw", "trace", "runtime_receipt", "image")},
                        **({"pair_acceptance": literal_binding(PAIR_ACCEPTANCE), "pair_result": literal_binding(pair_result_path),
                            "pair_R": literal_binding(pair_R_path), "pair_B": literal_binding(pair_B_path)} if coordinate_split else {})}, "started_unix": time.time()}
    write(output / "receipt.json", state)
    started = time.monotonic()
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        q, identity = load_model("untied", torch.device(device))
        current, saved = dict(identity), dict(receipt["identity"])
        current_loader, saved_loader = current.pop("loader_source"), saved.pop("loader_source")
        require(current == saved and (current_loader["sha256"], current_loader["size_bytes"]) == (saved_loader["sha256"], saved_loader["size_bytes"]), "source/effective model identity changed")
        config = dict(panel["configs"]["untied"])
        config["data"] = {"input_jsonl": group["input_jsonl"]}
        requests, _ = build_bound_native_requests(q, config, group["cases"])
        batch = prepare_native_inputs(q.processor, requests, device=device, record_media_identity=True)
        require(input_identity(batch) == receipt["input_identity"], "source native batch identity changed")
        end = offsets[1]
        suffixes = _prefix_tokens(raw, end, int(q.tokenizer.pad_token_id))
        histories = [list(prompt) + tail for prompt, tail in zip(batch.prompt_token_ids, suffixes, strict=True)]
        native = exact_history_inputs(q.model, batch.inputs, histories, pad_token_id=int(q.tokenizer.pad_token_id), logits_to_keep=1)
        variants = {name: changed_inputs(native, S, tokens) for name, tokens in (("X", X), ("Y", Y))} if coordinate_split else {"B": changed_inputs(native, S)}
        primary_hashes = primary_manifest["cells"]["L_L"]["tensor_hashes"]
        require(all(tensor_hash(native[k]) == primary_hashes[k] for k in ("input_ids", "attention_mask", "position_ids")), "primary L_L replay inputs changed")
        require(all(native["input_ids"].shape == inputs["input_ids"].shape and torch.equal(native["position_ids"], inputs["position_ids"]) and torch.equal(native["attention_mask"], inputs["attention_mask"]) for inputs, _ in variants.values()), "content edit changed length, positions or mask")
        producer_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        producer_capture = preserve_source(Path(__file__), run_root=output, relative_name=f"recurrence_written_content-{producer_hash[:12]}.py")
        dependency_paths = [Path(p) for p in ("probes/training_set_completion/recurrence_position_history.py", "probes/training_set_completion/untied_shared.py",
                            "probes/training_set_completion/numerical_feedback/runtime.py", "src/qwen/native.py", "src/inference/bound_requests.py", "src/qwen/untied_embeddings.py")]
        dependency_captures = [str(preserve_source(p, run_root=output, relative_name=str(p))) for p in dependency_paths]
        require(literal_binding(Path(dependency_captures[-1]))["sha256"] == current_loader["sha256"], "current loader capture differs")
        manifest = {"schema": "recurrence_written_content.coordinate_split.v1" if coordinate_split else "recurrence_written_content.cells.v1", "status": "frozen_before_forward", "source": state["source"],
                    "producer": literal_binding(Path(__file__)), "producer_capture": literal_binding(producer_capture), "dependency_captures": [literal_binding(Path(p)) for p in dependency_captures],
                    "model_identity": identity, "native_batch_identity": input_identity(batch), "source_panel": literal_binding(PANEL),
                    "target_batch": TARGET, "target_offset": end, "source_rows": [88, 89], "current_S": S,
                    "row_tokens": {"R": R, **({"X": X, "Y": Y, "B": B} if coordinate_split else {"B": B})},
                    "actual_changed_physical_indices": {name: changed for name, (_, changed) in variants.items()},
                    "actual_changed_raw_offsets": {"X": [798], "Y": [799]} if coordinate_split else {"B": [798, 799]},
                    "inputs": {name: {k: tensor_hash(inputs[k]) for k in ("input_ids", "attention_mask", "position_ids")}
                               for name, inputs in {"R": native, **{name: pair[0] for name, pair in variants.items()}}.items()},
                    "max_forwards": MAX_FORWARDS, "max_model_seconds": MAX_SECONDS, "parity_atol": ATOL, "delta_guard": 4e-4}
        write(output / "source-to-cell.json", manifest)
        state.update(status="executing", manifest=literal_binding(output / "source-to-cell.json"))
        write(output / "receipt.json", state)
        clock = time.monotonic()
        def count(_module, _args, _kwargs):
            state["model_forwards"] += 1
            require(state["model_forwards"] <= MAX_FORWARDS and time.monotonic() - clock <= MAX_SECONDS, "forward/time cap exceeded")
        def vision(*_):
            state["vision_forwards"] += 1
        handles = [q.model.register_forward_pre_hook(count, with_kwargs=True), q.model.model.visual.register_forward_pre_hook(vision)]
        cells = {}
        try:
            for name, inputs in (("R", native), *((name, pair[0]) for name, pair in variants.items())):
                seen, observers = hooks(q.model, inputs["input_ids"], inputs["position_ids"])
                try:
                    with torch.inference_mode():
                        vector = q.model(**inputs).logits[TARGET, -1].detach().float().cpu()
                finally:
                    for observer in observers:
                        observer.remove()
                require(torch.isfinite(vector).all() and len(seen["embedding_inputs"]) == len(seen["rotary_positions"]) == 1 and len(seen["masks"]) == 2, "actual consumption invalid")
                require(seen["masks"][0] == seen["masks"][1] and seen["cache_slots"][0] == seen["cache_slots"][1], "decoder causal consumption varies")
                top = torch.topk(vector, 5)
                path = output / f"{name}.pt"
                torch.save(vector, path)
                cells[name] = {"argmax_token": int(top.indices[0]), "top1_top2_gap": float(top.values[0] - top.values[1]),
                               "top5": [{"token": int(t), "logit": float(v)} for t, v in zip(top.indices, top.values, strict=True)],
                               "z38": float(vector[151708]), "z999": float(vector[152669]), "z38_minus_z999": float(vector[151708] - vector[152669]),
                               "full_logits": literal_binding(path), "consumed": seen}
                if name == "R":
                    old = torch.load(primary_vector_path, map_location="cpu", weights_only=True)
                    cells[name]["max_abs_vs_primary_full_vocab"] = float((vector - old).abs().max())
                    saved = trace["steps"][end]
                    topids = (saved["raw_winners"][TARGET], saved["raw_runnerups"][TARGET])
                    topvals = saved["raw_top2"][TARGET]
                    cells[name]["saved_top2_max_abs_error"] = max(abs(float(vector[t]) - float(v)) for t, v in zip(topids, topvals, strict=True))
                    require(cells[name]["max_abs_vs_primary_full_vocab"] <= ATOL and cells[name]["saved_top2_max_abs_error"] <= ATOL and cells[name]["argmax_token"] == 152669, "native full-vocabulary qualification failed")
                    if coordinate_split:
                        accepted_R = torch.load(pair_R_path, map_location="cpu", weights_only=True)
                        cells[name]["max_abs_vs_accepted_pair_R"] = float((vector - accepted_R).abs().max())
                        require(cells[name]["max_abs_vs_accepted_pair_R"] <= ATOL, "fresh R differs from accepted pair")
                else:
                    require(seen["masks"] == cells["R"]["consumed"]["masks"] and seen["cache_slots"] == cells["R"]["consumed"]["cache_slots"] and seen["rotary_positions"] == cells["R"]["consumed"]["rotary_positions"], "content edit changed consumed mask/position/cache")
                state["last_cell"] = name
                write(output / "partial-results.json", {"status": "running", "cells": cells})
                write(output / "receipt.json", state)
                require(time.monotonic() - clock <= MAX_SECONDS, "model execution time cap exceeded")
        finally:
            for handle in handles:
                handle.remove()
        if coordinate_split:
            accepted_B = torch.load(pair_B_path, map_location="cpu", weights_only=True)
            d = {name: cells[name]["z38_minus_z999"] for name in ("R", "X", "Y")}
            d["B"] = float(accepted_B[151708] - accepted_B[152669])
            require(abs(d["B"] - pair_result["cells"]["B"]["z38_minus_z999"]) <= ATOL, "accepted B readback differs")
            effects = {"x2_at_y2_575": d["X"] - d["R"], "x2_at_y2_999": d["B"] - d["Y"],
                       "y2_at_x2_38": d["Y"] - d["R"], "y2_at_x2_999": d["B"] - d["X"],
                       "interaction": d["B"] - d["X"] - d["Y"] + d["R"]}
            result = {"schema": "recurrence_written_content.coordinate_split_result.v1", "status": "candidate", "cells": cells,
                      "accepted_B": {"full_logits": literal_binding(pair_B_path), "argmax_token": pair_result["cells"]["B"]["argmax_token"],
                                     "top1_top2_gap": pair_result["cells"]["B"]["top1_top2_gap"], "z38_minus_z999": d["B"]},
                      "margins": d, "effects": effects, "effect_guard_4e_4": {k: abs(v) > 4e-4 for k, v in effects.items()},
                      "model_forwards": state["model_forwards"], "vision_forwards": state["vision_forwards"], "model_seconds": time.monotonic() - clock}
        else:
            delta = cells["B"]["z38_minus_z999"] - cells["R"]["z38_minus_z999"]
            primary_effect = primary_result["cells"]["L_L"]["z38_minus_z999"] - primary_result["cells"]["E_L"]["z38_minus_z999"]
            result = {"schema": "recurrence_written_content.result.v1", "status": "candidate", "cells": cells, "delta_B_minus_R": delta,
                      "delta_guard_met": abs(delta) > 4e-4, "primary_added_history_effect": primary_effect,
                      "delta_fraction_of_primary_effect_magnitude": abs(delta) / abs(primary_effect),
                      "B_conditional_reversal": cells["B"]["argmax_token"] == 151708 and cells["B"]["top1_top2_gap"] > 4e-4,
                      "model_forwards": state["model_forwards"], "vision_forwards": state["vision_forwards"], "model_seconds": time.monotonic() - clock}
        write(output / "result.json", result)
        state.update(status="candidate_complete", result=literal_binding(output / "result.json"), model_seconds=result["model_seconds"],
                     elapsed_seconds=time.monotonic() - started, peak_reserved_bytes=int(torch.cuda.max_memory_reserved()))
        write(output / "receipt.json", state)
        print(json.dumps({"status": state["status"], "result": str(output / "result.json"), "margins": result.get("margins"), "effects": result.get("effects"), "forwards": state["model_forwards"]}))
    except BaseException as error:
        state.update(status="technical_invalid", error=repr(error), elapsed_seconds=time.monotonic() - started)
        write(output / "receipt.json", state)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--selfcheck", action="store_true")
    parser.add_argument("--coordinate-split", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck(); print("selfcheck ok")
    else:
        run(args.device, coordinate_split=args.coordinate_split)
