"""One-image, four-cell native history/current-prefix MRoPE crossing."""
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
from probes.training_set_completion.recurrence_census.prepare import _rows_from_tokens
from probes.training_set_completion.untied_shared import ROOT as SOURCE_ROOT, load_model
from src.artifacts.source_provenance import preserve_source
from src.inference.bound_requests import build_bound_native_requests
from src.qwen.input_identity import input_identity, tensor_hash
from src.qwen.native import exact_history_inputs, prepare_native_inputs


OUT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-position-history/attempt-002")
SELECTION = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-transition-readback/dynamics/selection.json")
PANEL = SOURCE_ROOT / "panel.json"
TARGET = 2
ATOL = 2e-4
FORWARDS = 12
SECONDS = 20 * 60


def require(ok, message):
    if not ok:
        raise ValueError(message)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, allow_nan=False).encode() + b"\n"
    path.write_bytes(data)


def bound(binding):
    actual = literal_binding(Path(binding["path"]))
    require(actual["sha256"] == binding["sha256"] and actual["size_bytes"] == binding["size_bytes"], f"source binding drift: {binding['path']}")
    return Path(actual["path"])


def swap_prefix_positions(base, donor, target, suffix, expected_tokens):
    """Only the target's entire final S position span may change."""
    require(base["position_ids"].ndim == 3 and base["position_ids"].shape[0] == 3, "expected three-axis MRoPE")
    require(tuple(base["input_ids"][target, -suffix:].tolist()) == tuple(expected_tokens), "wrong S slot")
    require(tuple(donor.shape) == (3, suffix), "donor S positions have wrong shape")
    crossed = dict(base)
    positions = base["position_ids"].clone()
    positions[:, target, -suffix:] = donor
    require(torch.equal(positions[:, target, :-suffix], base["position_ids"][:, target, :-suffix]), "history positions changed")
    require(torch.equal(positions[:, [i for i in range(positions.shape[1]) if i != target]], base["position_ids"][:, [i for i in range(positions.shape[1]) if i != target]]), "companion positions changed")
    require(torch.equal(positions[:, target, -suffix:], donor), "S position intervention not applied")
    crossed["position_ids"] = positions
    require(crossed["input_ids"] is base["input_ids"] and crossed["attention_mask"] is base["attention_mask"], "history tokens or mask changed")
    return crossed


def selfcheck():
    base = {"input_ids": torch.tensor([[9, 4, 5, 6]]), "attention_mask": torch.ones(1, 4, dtype=torch.long),
            "position_ids": torch.arange(4).expand(3, 1, 4).clone()}
    donor = torch.tensor([[8, 9], [8, 9], [8, 9]])
    crossed = swap_prefix_positions(base, donor, 0, 2, (5, 6))
    require(torch.equal(crossed["position_ids"][:, 0, :2], base["position_ids"][:, 0, :2]), "selfcheck history")
    require(torch.equal(crossed["attention_mask"], base["attention_mask"]), "selfcheck mask")
    for suffix, tokens in ((2, (4, 5)), (1, (5,))):
        try:
            swap_prefix_positions(base, donor, 0, suffix, tokens)
        except ValueError:
            pass
        else:
            raise AssertionError("off-by-one x2 slot escaped selfcheck")
    bad = dict(base)
    bad["input_ids"] = base["input_ids"].clone()
    bad["input_ids"][0, 0] = 10
    require(not torch.equal(bad["input_ids"], crossed["input_ids"]), "history mutation insensitive")


def source_and_rows():
    selection_binding = literal_binding(SELECTION)
    selection = json.loads(SELECTION.read_text())
    matches = [x for x in selection["selected"] if x["image_key"] == "val:7511"]
    require(len(matches) == 1 and selection["condition"] == "untied-original", "frozen source selection changed")
    chosen = matches[0]
    require(chosen["group"] == "refined-01" and chosen["batch_index"] == TARGET, "native group or target index changed")
    for key in ("raw", "trace", "runtime_receipt", "image"):
        bound(chosen[key])
    raw = json.loads(Path(chosen["raw"]["path"]).read_text())["rows"]
    trace = json.loads(Path(chosen["trace"]["path"]).read_text())
    receipt = json.loads(Path(chosen["runtime_receipt"]["path"]).read_text())
    require(receipt["status"] == "candidate_complete" and receipt["condition"] == "untied-original", "source runtime not complete")
    require(receipt["raw"]["sha256"] == chosen["raw"]["sha256"] and receipt["trace"]["sha256"] == chosen["trace"]["sha256"], "runtime receipt source mismatch")
    require(raw[TARGET]["image_id"] == chosen["image_id"] and len(trace["steps"]) >= 808, "source batch mismatch")
    tokens = raw[TARGET]["token_ids"]
    rows = _rows_from_tokens(tokens)
    require(len(rows) > 89 and rows[88]["end"] == rows[89]["start"], "native row boundary changed")
    early, late = rows[88], rows[89]
    suffix = tokens[early["start"]:early["end"] - 3]
    require(suffix == tokens[late["start"]:late["end"] - 3] and len(suffix) == 6, "current row prefix S differs")
    require((early["start"], late["start"], early["values"], late["values"]) == (792, 801, [0, 571, 38, 575], [0, 571, 999, 999]), "row witness changed")
    offsets = (early["start"] + len(suffix), late["start"] + len(suffix))
    require(offsets == (798, 807) and [trace["steps"][o]["chosen"][TARGET] for o in offsets] == [151708, 152669], "x2 predictor alignment changed")
    panel_binding = literal_binding(PANEL)
    require(receipt["panel"]["sha256"] == panel_binding["sha256"], "source panel changed")
    panel = json.loads(PANEL.read_text())
    group = next(g for g in panel["groups"] if g["key"] == chosen["group"])
    require([c["input_record"]["image_id"] for c in group["cases"]] == [x["image_id"] for x in raw], "batch companions changed")
    return chosen, raw, trace, receipt, panel, group, suffix, offsets, {"selection": selection_binding, "panel": panel_binding}


def observed_hooks(model, expected_positions, suffix, target):
    text = model.model.language_model
    seen = {"rotary": 0, "masks": [], "caches": [], "embedding": 0}
    def rotary_hook(_module, args):
        pos = args[1]
        require(torch.equal(pos, expected_positions), "rotary consumed different MRoPE positions")
        seen["rotary"] += 1
    def attention_hook(_module, _args, kwargs):
        mask = kwargs.get("attention_mask")
        seen["masks"].append(None if mask is None else tensor_hash(mask))
        cache = kwargs.get("cache_position")
        seen["caches"].append(None if cache is None else tensor_hash(cache))
        embeds = kwargs.get("position_embeddings")
        require(isinstance(embeds, tuple) and len(embeds) == 2, "attention did not consume rotary embeddings")
        for value in embeds:
            require(torch.isfinite(value).all(), "nonfinite consumed rotary embeddings")
        seen["embedding"] += 1
    handles = [text.rotary_emb.register_forward_pre_hook(rotary_hook),
               text.layers[0].self_attn.register_forward_pre_hook(attention_hook, with_kwargs=True),
               text.layers[-1].self_attn.register_forward_pre_hook(attention_hook, with_kwargs=True)]
    return seen, handles


def run(args):
    selfcheck()
    chosen, raw, trace, receipt, panel, group, suffix, offsets, source_bindings = source_and_rows()
    require(torch.cuda.is_available() and args.device.startswith("cuda"), "one CUDA device required")
    require(not OUT.exists(), "output root already exists; retain attempts")
    OUT.mkdir(parents=True)
    start = time.monotonic()
    state = {"status": "preparing", "pid": os.getpid(), "device": args.device, "forward_count": 0, "vision_forward_count": 0,
             "source": {**source_bindings, **{key: chosen[key] for key in ("raw", "trace", "runtime_receipt", "image")}}, "started_unix": time.time()}
    write(OUT / "receipt.json", state)
    model = None
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        q, identity = load_model("untied", torch.device(args.device))
        model = q.model
        current_identity, saved_identity = dict(identity), dict(receipt["identity"])
        current_loader, saved_loader = current_identity.pop("loader_source"), saved_identity.pop("loader_source")
        require((current_loader["sha256"], current_loader["size_bytes"]) == (saved_loader["sha256"], saved_loader["size_bytes"]), "loader source bytes differ from source")
        require(current_identity == saved_identity, "loaded model/effective-row identity differs from source")
        config = dict(panel["configs"]["untied"])
        config["data"] = {"input_jsonl": group["input_jsonl"]}
        requests, _ = build_bound_native_requests(q, config, group["cases"])
        batch = prepare_native_inputs(q.processor, requests, device=args.device, record_media_identity=True)
        require(input_identity(batch) == receipt["input_identity"], "native prompt/image/batch identity differs from source")
        require(q.tokenizer.pad_token_id is not None, "pad token missing")
        bases = {}
        for name, end in zip(("E", "L"), offsets):
            tails = _prefix_tokens(raw, end, int(q.tokenizer.pad_token_id))
            histories = [list(prompt) + tail for prompt, tail in zip(batch.prompt_token_ids, tails, strict=True)]
            inputs = exact_history_inputs(model, batch.inputs, histories, pad_token_id=int(q.tokenizer.pad_token_id), logits_to_keep=1)
            require(inputs["input_ids"].shape[0] == 4 and inputs["attention_mask"].ndim == 2, "native batch/mask shape changed")
            require(inputs["input_ids"][TARGET, -len(suffix):].tolist() == suffix, "S source tokens not at causal tail")
            require(inputs["attention_mask"][TARGET, -len(suffix):].tolist() == [1] * len(suffix), "S padding mask changed")
            bases[name] = inputs
        positions = {name: inputs["position_ids"][:, TARGET, -len(suffix):].clone() for name, inputs in bases.items()}
        delta = positions["L"] - positions["E"]
        require(len(torch.unique(delta)) == 1 and int(delta.flatten()[0]) == 9, "S MRoPE translation is not shared nine-position shift")
        cells = {"E_E": bases["E"], "E_L": swap_prefix_positions(bases["E"], positions["L"], TARGET, len(suffix), suffix),
                 "L_E": swap_prefix_positions(bases["L"], positions["E"], TARGET, len(suffix), suffix), "L_L": bases["L"]}
        producer_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        capture = preserve_source(Path(__file__), run_root=OUT, relative_name=f"recurrence_position_history-{producer_hash[:12]}.py")
        dependencies = [Path(p) for p in ("probes/training_set_completion/untied_shared.py", "probes/training_set_completion/untied_natural.py",
                       "probes/training_set_completion/numerical_feedback/runtime.py", "src/qwen/native.py", "src/inference/bound_requests.py")]
        captures = [str(preserve_source(p, run_root=OUT, relative_name=str(p))) for p in dependencies]
        before = {name: {key: tensor_hash(inputs[key]) for key in ("input_ids", "attention_mask", "position_ids")} for name, inputs in cells.items()}
        require(before["E_E"]["input_ids"] == before["E_L"]["input_ids"] and before["E_E"]["attention_mask"] == before["E_L"]["attention_mask"], "E history or mask differs")
        require(before["L_E"]["input_ids"] == before["L_L"]["input_ids"] and before["L_E"]["attention_mask"] == before["L_L"]["attention_mask"], "L history or mask differs")
        manifest = {"schema": "recurrence_position_history.cells.v1", "status": "frozen_before_model_forward", "source": state["source"],
                    "producer": literal_binding(Path(__file__)), "source_capture": str(capture), "dependency_captures": captures,
                    "runtime_identity": q.to_artifact_dict(), "model_identity": identity, "native_input_identity": input_identity(batch),
                    "target_batch_index": TARGET, "image_key": "val:7511", "source_rows": [88, 89], "next_token_offsets": list(offsets),
                    "S_tokens": suffix, "S_length": len(suffix), "native_S_positions": {k: v.detach().cpu().tolist() for k, v in positions.items()},
                    "position_delta_all_axes": int(delta.flatten()[0]), "crossed_gap_overlap_intentional": True,
                    "cells": {name: {"history": name[0], "S_positions": name[-1], "tensor_hashes": before[name],
                                      "shape": list(inputs["input_ids"].shape), "target_S_physical_range": [inputs["input_ids"].shape[1] - len(suffix), inputs["input_ids"].shape[1]],
                              "history_end_before_S": offsets[0] - len(suffix) if name[0] == "E" else offsets[1] - len(suffix)} for name, inputs in cells.items()},
                    "forward_cap": FORWARDS, "forward_seconds_cap": SECONDS, "parity_atol": ATOL, "cross_argmax_min_gap": 4e-4}
        write(OUT / "source-to-cell.json", manifest)
        state["status"] = "executing"
        state["manifest"] = literal_binding(OUT / "source-to-cell.json")
        write(OUT / "receipt.json", state)
        first_forward = time.monotonic()
        def count_model(_module, _args, _kwargs):
            state["forward_count"] += 1
            require(state["forward_count"] <= FORWARDS and time.monotonic() - first_forward <= SECONDS, "forward/time cap exceeded")
        def count_vision(*_):
            state["vision_forward_count"] += 1
        count_handle = model.register_forward_pre_hook(count_model, with_kwargs=True)
        vision_handle = model.model.visual.register_forward_pre_hook(count_vision)
        observations = {}
        try:
            for name in ("E_E", "L_L", "E_noop", "E_L", "L_E"):
                inputs = cells["E_E"] if name == "E_noop" else cells[name]
                observed, handles = observed_hooks(model, inputs["position_ids"], len(suffix), TARGET)
                try:
                    with torch.inference_mode():
                        logits = model(**inputs).logits[TARGET, -1].detach().float().cpu()
                finally:
                    for handle in handles:
                        handle.remove()
                require(torch.isfinite(logits).all(), "nonfinite full vocabulary logits")
                require(observed["rotary"] == 1 and observed["embedding"] == 2 and len(observed["masks"]) == 2, "position/mask consumption not attested")
                require(observed["masks"][0] == observed["masks"][1] and observed["caches"][0] == observed["caches"][1], "decoder layers used different masks/cache slots")
                top = torch.topk(logits, 5)
                record = {"cell": name, "logits_file": f"{name}.pt", "argmax_token": int(top.indices[0]), "top5": [{"token": int(t), "logit": float(v)} for t, v in zip(top.indices, top.values, strict=True)],
                          "z38_minus_z999": float(logits[151708] - logits[152669]), "z38": float(logits[151708]), "z999": float(logits[152669]),
                          "top1_top2_gap": float(top.values[0] - top.values[1]), "consumed": observed,
                          "full_vocab_count": int(logits.numel()), "elapsed_model_seconds": time.monotonic() - first_forward}
                torch.save(logits, OUT / f"{name}.pt")
                record["logits_binding"] = literal_binding(OUT / f"{name}.pt")
                if name in ("E_E", "L_L"):
                    offset = offsets[0] if name == "E_E" else offsets[1]
                    saved = trace["steps"][offset]
                    winners = [saved["raw_winners"][TARGET], saved["raw_runnerups"][TARGET]]
                    values = saved["raw_top2"][TARGET]
                    record["saved_top2"] = [{"token": int(t), "logit": float(v)} for t, v in zip(winners, values, strict=True)]
                    record["top2_max_abs_error"] = max(abs(float(logits[t]) - float(v)) for t, v in zip(winners, values, strict=True))
                    require(record["argmax_token"] == winners[0] and record["top2_max_abs_error"] <= ATOL, f"native diagonal parity failure {name}")
                if name == "E_noop":
                    baseline = torch.load(OUT / "E_E.pt", map_location="cpu", weights_only=True)
                    record["max_abs_vs_E_E"] = float((logits - baseline).abs().max())
                    require(record["max_abs_vs_E_E"] <= ATOL, "no-op full-vocabulary replay mismatch")
                if name in ("E_L", "L_E"):
                    native = observations["E_E" if name[0] == "E" else "L_L"]["consumed"]
                    require(observed["masks"] == native["masks"] and observed["caches"] == native["caches"], "crossing changed consumed causal mask/cache positions")
                observations[name] = record
                write(OUT / "partial-results.json", {"status": "running", "cells": observations})
                state["last_cell"] = name
                write(OUT / "receipt.json", state)
                require(time.monotonic() - first_forward <= SECONDS, "model execution time cap exceeded")
        finally:
            count_handle.remove(); vision_handle.remove()
        margins = {name: observations[name]["z38_minus_z999"] for name in ("E_E", "E_L", "L_E", "L_L")}
        results = {"schema": "recurrence_position_history.result.v1", "status": "candidate", "cells": observations, "factorial_margin_interaction": margins["L_L"] - margins["L_E"] - margins["E_L"] + margins["E_E"],
                   "transport_prediction_met": observations["E_L"]["argmax_token"] == 152669 and observations["L_E"]["argmax_token"] == 151708 and
                   observations["E_L"]["top1_top2_gap"] > 4e-4 and observations["L_E"]["top1_top2_gap"] > 4e-4,
                   "history_following": observations["E_E"]["argmax_token"] == observations["E_L"]["argmax_token"] == 151708 and observations["L_E"]["argmax_token"] == observations["L_L"]["argmax_token"] == 152669,
                   "forward_count": state["forward_count"], "vision_forward_count": state["vision_forward_count"], "model_execution_seconds": time.monotonic() - first_forward}
        write(OUT / "result.json", results)
        state.update(status="candidate_complete", result=literal_binding(OUT / "result.json"), model_execution_seconds=results["model_execution_seconds"], elapsed_seconds=time.monotonic() - start,
                     peak_reserved_bytes=int(torch.cuda.max_memory_reserved()))
        write(OUT / "receipt.json", state)
        print(json.dumps({"status": state["status"], "result": str(OUT / "result.json"), "margins": margins, "transport": results["transport_prediction_met"], "forwards": state["forward_count"]}))
    except BaseException as exc:
        state.update(status="technical_invalid", error=repr(exc), elapsed_seconds=time.monotonic() - start)
        write(OUT / "receipt.json", state)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck(); print("selfcheck ok")
    else:
        run(args)
