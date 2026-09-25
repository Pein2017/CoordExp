"""Three admitted full-prefix bowl forwards with a checked row-read mask."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import resource
import time
import traceback
from pathlib import Path

import torch
from transformers import AutoConfig
from transformers.integrations import sdpa_attention
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3_vl import modeling_qwen3_vl

from src.artifacts.utf8_json import literal_binding
from probes.model_profiles.mature_source import load_saved_source as _source
from probes.recurrence_dynamics.coordinate_continuity.runtime import tensor_hash
from src.qwen.native_row_scores import compare_saved_trace as _trace_compare
from src.qwen.saved_prefix import prefix_tokens as _prefix_tokens
from probes.recurrence_dynamics.numerical_feedback.select import token_hash
from probes.recurrence_dynamics.recurrence_first_arrivals.prepare import MATURE
from probes.model_profiles.mature_tied_untied import BASE, load_model
from src.artifacts.source_provenance import preserve_source
from src.qwen.input_identity import input_identity
from src.qwen.native import exact_history_inputs
from src.qwen.runtime_loading import QwenLoadOptions, load_qwen_components_from_options


REPO = Path(__file__).resolve().parents[3]
UNIT = REPO / "research/experiments/2026-09-24-recurrence-first-revisit-routing"
ADMISSION = UNIT / "lead-native-read-admission-v1.json"
PROTOCOL = UNIT / "native-history-read-protocol.md"
REGISTRY = UNIT / "supporting/native-window-candidates.json"
NOTE = UNIT / "lead-interpretation-note-v1.md"
PREFLIGHT = UNIT / "supporting/native-read-preflight-001.json"
OUT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-24-recurrence-first-revisit-routing/native-read-attempt-001")
SHAS = {ADMISSION: "887875d1ccf7376881e2faddf74645ed560828a3d7a854df1bf96896a8576dab",
        PROTOCOL: "9ccece16a05d07aca61da9311ffea954382d306d23ad85f106baebe16ebd955c",
        REGISTRY: "dcdc2f20a6dc64b6ec1110faf7cc2a7c4e4cf3dcbd4355c9ee2338b4c07996c4"}
TARGET, OFFSET, OLD, NEW, OLD0 = 3, 15, 151675, 151827, 151670
QUERY, KEY = (10, 15), (0, 10)
CONDITIONS = ("native", "identity-mask-sham", "latest-row-mask")
CAP = 360.0
TOL = 2e-4


def require(ok, message):
    if not ok:
        raise ValueError(message)


def bind(path):
    return literal_binding(Path(path))


def write_new(path, value):
    path = Path(path)
    require(not path.exists(), f"artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return bind(path)


def contract():
    for path, expected in SHAS.items():
        require(bind(path)["sha256"] == expected, f"frozen binding changed: {path}")
    admission = json.loads(ADMISSION.read_text())
    registry = json.loads(REGISTRY.read_text())
    require(admission["admitted_cells"] == [
        {"state": "bowl-row1", "condition": condition, "image": "mature:313465:0",
         "target_batch_index": TARGET, "current_raw_query_span": list(QUERY),
         "selected_raw_key_span": list(KEY), "old_token_id": OLD, "new_token_id": NEW}
        for condition in CONDITIONS], "admitted cells changed")
    require(admission["max_model_forwards"] == admission["max_vision_forwards"] == 3
            and admission["max_free_tokens"] == 0
            and admission["first_case_cap_gpu_seconds"] == CAP
            and admission["raw_output_root"] == str(OUT), "admission budget/root changed")
    require(math.isclose(admission["prior_sequence_gpu_hours"], .2381858424494664, abs_tol=1e-12)
            and admission["sequence_cap_gpu_hours"] == 8, "sequence budget changed")
    for key in ("protocol", "candidate_registry", "cpu_acceptance", "prior_cost_ledger"):
        b = admission[key]
        require(bind(b["path"])["sha256"] == b["sha256"], f"admission dependency changed: {key}")
    image = registry["images"]["mature:313465:0"]
    require(image["group"] == "fresh-18" and image["batch_index"] == TARGET
            and image["important_rows"]["0"]["span"] == [0, 10]
            and image["important_rows"]["1"]["span"] == [10, 20]
            and image["important_rows"]["1"]["token_ids"][:5] ==
                image["important_rows"]["2"]["token_ids"][:5]
            and image["important_rows"]["0"]["token_ids"][5] == OLD0
            and image["important_rows"]["1"]["token_ids"][5] == OLD
            and image["important_rows"]["2"]["token_ids"][5] == NEW,
            "bowl source/extent geometry changed")
    for key in ("raw", "trace", "runtime_receipt", "image"):
        b = image["source_bindings"][key]
        require(bind(b["path"])["sha256"] == b["sha256"], f"source changed: {key}")
    return admission, image


def source_boundary(image, raw):
    b = image["source_bindings"]
    return {"group": image["group"], "batch_index": TARGET, "image_id": 313465,
            "raw_path": b["raw"]["path"], "trace_path": b["trace"]["path"],
            "receipt_path": b["runtime_receipt"]["path"],
            "native_tokens": raw[TARGET]["token_ids"],
            "native_token_hash": token_hash(raw[TARGET]["token_ids"])}


class ConfigOnlyRope:
    def __init__(self):
        self.config = AutoConfig.from_pretrained(BASE, local_files_only=True)
    get_rope_index = modeling_qwen3_vl.Qwen3VLModel.get_rope_index


def source_inputs(model, batch, raw, pad):
    tails = _prefix_tokens(raw, OFFSET, pad)
    histories = [list(prompt) + tail for prompt, tail in zip(batch.prompt_token_ids, tails, strict=True)]
    full = exact_history_inputs(model, batch.inputs, histories, pad_token_id=pad, logits_to_keep=1)
    width = int(full["input_ids"].shape[1])
    full["cache_position"] = torch.arange(width, device=full["input_ids"].device)
    prompt = width - OFFSET
    require(full["input_ids"].shape == full["attention_mask"].shape == (4, width)
            and full["position_ids"].shape == (3, 4, width)
            and full["input_ids"][:, prompt:].tolist() == tails
            and full["attention_mask"][:, prompt:].tolist() == [[1] * OFFSET] * 4,
            "source full-prefix input shape/tokens/masks changed")
    require(full["input_ids"][TARGET, prompt + QUERY[0]:prompt + QUERY[1]].tolist()
            == raw[TARGET]["token_ids"][QUERY[0]:QUERY[1]]
            and raw[TARGET]["token_ids"][OFFSET] == OLD
            and raw[TARGET]["token_ids"][0:10][-1] == 151649
            and len(raw) == 4 and [len(r["token_ids"]) for r in raw] == [286, 109, 11, 138],
            "source bowl row/companion geometry changed")
    return full, prompt


def native_4d(mask2d):
    require(mask2d.ndim == 2 and mask2d.shape[0] == 4, "source mask rank/batch changed")
    width = mask2d.shape[1]
    good = (torch.arange(width, device=mask2d.device)[:, None] >=
            torch.arange(width, device=mask2d.device)[None, :])
    good = good[None, None] & mask2d[:, None, None, :].bool()
    return good


def mask_for(mask2d, prompt, condition, *, target=TARGET, query=QUERY, key=KEY,
             source_mask_hash=None):
    require(condition in CONDITIONS and target == TARGET and query == QUERY and key == KEY,
            "wrong target, query span or historical row span")
    require(prompt == mask2d.shape[1] - OFFSET and prompt > 0, "wrong source/prefix width")
    if source_mask_hash is not None:
        require(tensor_hash(mask2d) == source_mask_hash, "source attention mask mutated")
    base = native_4d(mask2d)
    if condition == "native":
        return mask2d, base
    actual = base.clone()
    if condition == "latest-row-mask":
        actual[target, 0, prompt + query[0]:prompt + query[1], prompt + key[0]:prompt + key[1]] = False
    verify_mask(base, actual, prompt, condition)
    return actual, base


def verify_mask(base, actual, prompt, condition):
    require(base.shape == actual.shape and base.ndim == 4, "mask shape changed")
    rect = (TARGET, 0, slice(prompt + QUERY[0], prompt + QUERY[1]),
            slice(prompt + KEY[0], prompt + KEY[1]))
    expected = base.clone()
    if condition == "latest-row-mask":
        expected[rect] = False
        require(torch.all(base[rect]).item(), "selected native causal rectangle is not readable")
    require(torch.equal(actual, expected), "mask changed companion, history, complement or selected rectangle")


def forward_cell(model, full, prompt, condition, observer=None, source_mask_hash=None):
    mask, base = mask_for(full["attention_mask"], prompt, condition,
                          source_mask_hash=source_mask_hash)
    payload = {**full, "attention_mask": mask}
    if observer is not None:
        observer["expected_mask"] = base if condition == "native" else mask
        observer["condition"] = condition
    return model(**payload)


def cpu_fixture(full, prompt):
    source = full["attention_mask"]
    source_hash = tensor_hash(source)
    class Consumer:
        def __init__(self): self.seen = []
        def __call__(self, **kwargs):
            self.seen.append(kwargs["attention_mask"].clone())
            return kwargs["attention_mask"]
    fake = Consumer()
    for condition in CONDITIONS:
        forward_cell(fake, full, prompt, condition, source_mask_hash=source_hash)
    base = native_4d(source)
    require(torch.equal(fake.seen[0], source) and torch.equal(fake.seen[1], base),
            "actual caller did not preserve native/identity masks")
    verify_mask(base, fake.seen[2], prompt, "latest-row-mask")
    # Installed Qwen SDPA uses a boolean 4D mask (True=readable). Check the
    # exact source shape without loading a language model.
    config = AutoConfig.from_pretrained(BASE, local_files_only=True).text_config
    config._attn_implementation = "sdpa"
    reference = create_causal_mask(config, torch.empty((*source.shape, 1)), source,
                                   torch.arange(source.shape[1]), None,
                                   position_ids=full["position_ids"][0])
    require(isinstance(reference, torch.Tensor) and torch.equal(base, reference),
            "constructed identity mask differs from installed native SDPA causal mask")
    checks = []
    for label, kwargs in (("wrong_target", {"target": 2}),
                          ("wrong_query", {"query": (9, 15)}),
                          ("wrong_key", {"key": (0, 9)}),
                          ("wrong_prompt", {"prompt": prompt - 1})):
        bad_prompt = kwargs.pop("prompt", prompt)
        try: mask_for(source, bad_prompt, "latest-row-mask", source_mask_hash=source_hash, **kwargs)
        except ValueError: checks.append(label)
        else: raise AssertionError(f"CPU caller failed to reject {label}")
    changed_source = source.clone(); changed_source[0, 0] = 1 - changed_source[0, 0]
    try: mask_for(changed_source, prompt, "latest-row-mask", source_mask_hash=source_hash)
    except ValueError: checks.append("changed_source_mask")
    else: raise AssertionError("CPU caller accepted changed source mask")
    for label, at in (("companion", (0, 0, prompt + 11, prompt)),
                      ("historical_query", (TARGET, 0, prompt + 1, prompt)),
                      ("other_key", (TARGET, 0, prompt + 11, prompt + 10))):
        bad = fake.seen[2].clone(); bad[at] = False
        try: verify_mask(base, bad, prompt, "latest-row-mask")
        except ValueError: checks.append(label)
        else: raise AssertionError(f"CPU verifier accepted changed {label}")
    require(tensor_hash(source) == source_hash, "CPU fixture mutated source mask")
    return checks


def prepare_source(q, image, device):
    raw = json.loads(Path(image["source_bindings"]["raw"]["path"]).read_text())["rows"]
    panel = json.loads((MATURE / "panel.json").read_text())
    batch, raw, trace, group, planning = _source(source_boundary(image, raw), "untied", panel, q, device)
    receipt = json.loads(Path(image["source_bindings"]["runtime_receipt"]["path"]).read_text())
    require(len(raw) == len(group["cases"]) == 4 and input_identity(batch) == receipt["input_identity"]
            and list(batch.request_ids) == image["request_ids"], "original four-request source identity changed")
    return batch, raw, trace, receipt, planning


def preflight():
    admission, image = contract()
    require(not PREFLIGHT.exists() and not OUT.exists(), "attempt or preflight already exists")
    q = load_qwen_components_from_options(QwenLoadOptions(
        base_model=str(BASE), dtype="fp32", attn_implementation="sdpa",
        patch_embed_linearization="enabled", load_model=False))
    require(q.model is None, "CPU preflight loaded language model")
    batch, raw, _trace, receipt, planning = prepare_source(q, image, torch.device("cpu"))
    full, prompt = source_inputs(ConfigOnlyRope(), batch, raw, int(q.tokenizer.pad_token_id))
    checks = cpu_fixture(full, prompt)
    width = int(full["input_ids"].shape[1])
    pixel_elements = int(batch.inputs["pixel_values"].numel())
    # Measured 7-call/2-vision cross-image predecessor: 360.09s / 9 attempts.
    # This route has three full vision passes, so use a deliberately loose 3x
    # 40-second case estimate plus 60 seconds for setup, below the 360s cap.
    forecast = 180.0 * (width / 1421) * (pixel_elements / (4 * 52 * 78 * 1536))
    require(forecast < CAP and admission["prior_sequence_gpu_hours"] + CAP / 3600 < 8,
            "shape-aware forecast or sequence cap fails")
    direct = [Path(__file__), Path(inspect.getfile(_source)), Path(inspect.getfile(_prefix_tokens)),
              Path(inspect.getfile(token_hash)), Path(inspect.getfile(_trace_compare)),
              Path(__file__).resolve().parents[3] / 'probes/recurrence_dynamics/recurrence_first_arrivals/prepare.py',
              Path(inspect.getfile(load_model)),
              Path(inspect.getfile(exact_history_inputs)), Path(inspect.getfile(input_identity)),
              Path(inspect.getfile(literal_binding)), Path(inspect.getfile(preserve_source)),
              Path(inspect.getfile(load_qwen_components_from_options)),
              Path(inspect.getfile(modeling_qwen3_vl)), Path(inspect.getfile(create_causal_mask)),
              Path(inspect.getfile(sdpa_attention))]
    captures = []
    for path in dict.fromkeys(direct):
        rel = path.relative_to(REPO) if path.is_relative_to(REPO) else Path("transformers") / path.name
        saved = preserve_source(path, run_root=OUT, relative_name=rel)
        captures.append({"maintained": bind(path), "capture": bind(saved)})
    packet = {"schema": "recurrence_native_history_read.preflight.v1", "status": "cpu_qualified_before_gpu",
              "admission": bind(ADMISSION), "protocol": bind(PROTOCOL), "registry": bind(REGISTRY),
              "interpretation_note": bind(NOTE), "source_bindings": image["source_bindings"],
              "source_identity": receipt["identity"], "input_identity": receipt["input_identity"],
              "planning": planning, "producer": bind(Path(__file__)), "direct_source_captures": captures,
              "shape": {"batch": 4, "target": TARGET, "width": width, "prompt_width": prompt,
                        "query_physical": [prompt + QUERY[0], prompt + QUERY[1]],
                        "key_physical": [prompt + KEY[0], prompt + KEY[1]],
                        "image_grids": [list(g) for g in batch.image_grids], "pixel_elements": pixel_elements,
                        "raw_lengths": [len(r["token_ids"]) for r in raw]},
              "source_tensor_hashes": {k: tensor_hash(full[k]) for k in
                                      ("input_ids", "attention_mask", "position_ids", "cache_position")},
              "cpu_actual_caller_checks": checks,
              "forecast_allocated_gpu_seconds": forecast, "cap_allocated_gpu_seconds": CAP,
              "commands": {"gpu": ["python", "-B", "-m", "probes.recurrence_dynamics.recurrence_native_history_read.run", "run"],
                           "readback": ["python", "-B", "-m", "probes.recurrence_dynamics.recurrence_native_history_read.run", "readback"]}}
    write_new(PREFLIGHT, packet)
    print(json.dumps({"status": packet["status"], "width": width, "pixel_elements": pixel_elements,
                      "forecast_seconds": forecast, "cpu_checks": checks, "captures": len(captures)}))


def checked_preflight():
    admission, image = contract()
    p = json.loads(PREFLIGHT.read_text())
    require(p["status"] == "cpu_qualified_before_gpu" and p["admission"] == bind(ADMISSION)
            and p["protocol"] == bind(PROTOCOL) and p["registry"] == bind(REGISTRY)
            and p["producer"] == bind(Path(__file__)), "preflight or producer changed")
    for cap in p["direct_source_captures"]:
        require(bind(cap["maintained"]["path"]) == cap["maintained"]
                and bind(cap["capture"]["path"]) == cap["capture"], "direct import/capture changed")
    return admission, image, p


def run():
    admission, image, pre = checked_preflight()
    require(not OUT.exists(), "attempt already exists")
    OUT.mkdir(parents=True)
    started = time.monotonic()
    begun = time.time()
    counts = {"model_forwards": 0, "vision_forwards": 0, "free_tokens": 0}
    receipt = {"schema": "recurrence_native_history_read.attempt.v1", "status": "running", "pid": os.getpid(),
               "begun_unix": begun, "preflight": bind(PREFLIGHT), "producer": bind(Path(__file__)),
               "admission": bind(ADMISSION), "conditions": list(CONDITIONS), "counts": counts,
               "allocated_gpu_seconds": 0, "cells": []}
    write_new(OUT / "launch.json", receipt)
    device = torch.device("cuda:0")
    handles = []
    try:
        torch.cuda.set_device(device)
        torch.empty(1, device=device)
        torch.cuda.reset_peak_memory_stats(device)
        q, identity = load_model("untied", device)
        source_identity = pre["source_identity"]
        require({k: v for k, v in identity.items() if k != "loader_source"} ==
                {k: v for k, v in source_identity.items() if k != "loader_source"}
                and all(identity["loader_source"][k] == source_identity["loader_source"][k]
                        for k in ("sha256", "size_bytes")), "effective model identity changed")
        model = q.model.eval()
        batch, raw, trace, source_receipt, planning = prepare_source(q, image, device)
        full, prompt = source_inputs(model, batch, raw, int(q.tokenizer.pad_token_id))
        require(prompt == pre["shape"]["prompt_width"] and
                all(tensor_hash(full[k]) == pre["source_tensor_hashes"][k]
                    for k in ("input_ids", "attention_mask", "position_ids", "cache_position")),
                "GPU source geometry/input differs from CPU preflight")
        require(planning == pre["planning"] and source_receipt["input_identity"] == pre["input_identity"],
                "source planning/receipt changed")
        text_layers = [m for m in model.modules() if isinstance(m, modeling_qwen3_vl.Qwen3VLTextDecoderLayer)]
        attentions = [m for m in model.modules() if isinstance(m, modeling_qwen3_vl.Qwen3VLTextAttention)]
        require(len(text_layers) == len(attentions) == 28 and
                [m.self_attn for m in text_layers] == attentions,
                "installed model text-layer route is not exact 28-layer SDPA")
        active = {"condition": None, "expected_mask": None, "seen": [], "history": [], "companions": [],
                  "top_input": None}
        def top_hook(_module, _args, kwargs):
            counts["model_forwards"] += 1
            require(counts["model_forwards"] <= 3 and time.monotonic() - started < CAP,
                    "model-forward or allocated-time cap")
            require(torch.equal(kwargs["input_ids"], full["input_ids"]) and
                    torch.equal(kwargs["position_ids"], full["position_ids"]),
                    "actual model input/positions changed")
            active["top_input"] = {k: tensor_hash(kwargs[k]) for k in
                                   ("input_ids", "attention_mask", "position_ids", "cache_position")}
        def vision_hook(_module, _args):
            counts["vision_forwards"] += 1
            require(counts["vision_forwards"] <= 3, "vision-forward cap")
        handles.append(model.register_forward_pre_hook(top_hook, with_kwargs=True))
        handles.append(model.model.visual.register_forward_pre_hook(vision_hook))
        for i, attn in enumerate(attentions):
            def attention_hook(_module, _args, kwargs, layer=i):
                mask = kwargs.get("attention_mask")
                expected = active["expected_mask"]
                require(isinstance(mask, torch.Tensor) and mask.ndim == 4 and
                        torch.equal(mask, expected), f"layer {layer} did not consume exact mask")
                active["seen"].append(layer)
            handles.append(attn.register_forward_pre_hook(attention_hook, with_kwargs=True))
        for i, layer in enumerate(text_layers):
            def layer_hook(_module, _args, output, layer_idx=i):
                states = output[0] if isinstance(output, tuple) else output
                require(isinstance(states, torch.Tensor) and states.ndim == 3,
                        "text layer output unavailable")
                active["history"].append(states[TARGET, prompt:prompt+10].detach().cpu().float())
                active["companions"].append(states[[0, 1, 2], -1].detach().cpu().float())
            handles.append(layer.register_forward_hook(layer_hook))
        receipt["effective_identity"] = identity
        receipt["actual_geometry"] = pre["shape"]
        write_new(OUT / "inputs.json", {"source_hashes": pre["source_tensor_hashes"],
                                        "full_input_ids": full["input_ids"].cpu().tolist(),
                                        "source_attention_mask": full["attention_mask"].cpu().tolist(),
                                        "position_ids": full["position_ids"].cpu().tolist(),
                                        "cache_position": full["cache_position"].cpu().tolist()})
        source_mask_hash = tensor_hash(full["attention_mask"])
        with torch.inference_mode():
            for condition in CONDITIONS:
                require(time.monotonic() - started + pre["forecast_allocated_gpu_seconds"] / 3 < CAP,
                        "remaining allocated budget below shape-aware next-call allowance")
                active.update(condition=condition, expected_mask=None, seen=[], history=[], companions=[], top_input=None)
                out = forward_cell(model, full, prompt, condition, active,
                                   source_mask_hash=source_mask_hash)
                torch.cuda.synchronize(device)
                require(active["seen"] == list(range(28)) and len(active["history"]) ==
                        len(active["companions"]) == 28 and active["top_input"] is not None,
                        "actual all-layer mask/state consumer incomplete")
                logits = out.logits[:, -1, :].detach().cpu().float()
                require(logits.shape[0] == 4 and torch.isfinite(logits).all().item(), "invalid full vectors")
                item = {"condition": condition, "logits": logits,
                        "history_row0_by_layer": torch.stack(active["history"]),
                        "companion_last_by_layer": torch.stack(active["companions"])}
                torch.save(item, OUT / f"{condition}.pt")
                cell = {"condition": condition, "vector": bind(OUT / f"{condition}.pt"),
                        "actual_input_hashes": active["top_input"], "actual_mask_layers": active["seen"],
                        "selected_query_physical": pre["shape"]["query_physical"],
                        "selected_key_physical": pre["shape"]["key_physical"],
                        "elapsed_allocated_gpu_seconds": time.monotonic() - started}
                if condition == "native":
                    active_ids = [i for i, r in enumerate(raw) if len(r["token_ids"]) > OFFSET]
                    cell["source_trace_parity"] = [
                        _trace_compare(logits=logits[i], trace=trace, batch_index=i,
                                       absolute_offset=OFFSET, token_id=int(raw[i]["token_ids"][OFFSET]),
                                       role="native_unforced_x1") for i in active_ids]
                    require(active_ids == [0, 1, 3] and all(x["passed"] for x in cell["source_trace_parity"]),
                            "first native source parity failed before sham/mask")
                else:
                    native = torch.load(OUT / "native.pt", map_location="cpu", weights_only=True)
                    if condition == "identity-mask-sham":
                        error = float((logits - native["logits"]).abs().max())
                        cell["native_sham_all_batch_max_error"] = error
                        require(error <= TOL, "native/4D identity sham full-vector parity failed")
                    else:
                        for i in (0, 1, 2):
                            require(float((logits[i] - native["logits"][i]).abs().max()) <= TOL,
                                    f"treatment companion {i} changed")
                    hist_error = float((item["history_row0_by_layer"] - native["history_row0_by_layer"]).abs().max())
                    cell["historical_row0_all_layer_max_error"] = hist_error
                    require(hist_error <= TOL, "historical row0 states changed")
                receipt["cells"].append(cell)
                receipt["allocated_gpu_seconds"] = time.monotonic() - started
                receipt["counts"] = dict(counts)
                write_new(OUT / f"checkpoint-{len(receipt['cells'])}.json", receipt)
        receipt["status"] = "candidate_complete"
    except BaseException as exc:
        receipt["status"] = "technical_invalid"
        receipt["failure"] = {"type": type(exc).__name__, "message": str(exc),
                              "traceback": traceback.format_exc()}
    finally:
        for handle in handles: handle.remove()
        if torch.cuda.is_available(): torch.cuda.synchronize(device)
        receipt["allocated_gpu_seconds"] = time.monotonic() - started
        receipt["wall_seconds"] = receipt["allocated_gpu_seconds"]
        receipt["counts"] = dict(counts)
        receipt["rss_peak_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        receipt["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device) if torch.cuda.is_available() else 0
        receipt["gpu_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device) if torch.cuda.is_available() else 0
        receipt["terminal_pid"] = os.getpid()
        receipt["artifact_bytes"] = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
        write_new(OUT / "receipt.json", receipt)
    print(json.dumps({"status": receipt["status"], "counts": counts,
                      "allocated_gpu_seconds": receipt["allocated_gpu_seconds"],
                      "failure": receipt.get("failure", {}).get("message")}))
    require(receipt["status"] == "candidate_complete", "three-call attempt failed; no retry")


def readback():
    admission, image, pre = checked_preflight()
    receipt = json.loads((OUT / "receipt.json").read_text())
    require(receipt["status"] == "candidate_complete" and receipt["counts"] ==
            {"model_forwards": 3, "vision_forwards": 3, "free_tokens": 0}
            and len(receipt["cells"]) == 3 and receipt["admission"] == bind(ADMISSION),
            "attempt incomplete or changed")
    require(not (OUT / "readback.json").exists(), "readback exists")
    vectors = {}
    for cell in receipt["cells"]:
        require(bind(cell["vector"]["path"]) == cell["vector"] and
                cell["actual_mask_layers"] == list(range(28)), "raw vector/mask evidence changed")
        vectors[cell["condition"]] = torch.load(cell["vector"]["path"], map_location="cpu", weights_only=True)
    native, sham, masked = (vectors[name] for name in CONDITIONS)
    require(float((native["logits"] - sham["logits"]).abs().max()) <= TOL,
            "cold native/sham parity failed")
    require(all(float((native["logits"][i] - masked["logits"][i]).abs().max()) <= TOL
                for i in (0, 1, 2)), "cold companion parity failed")
    require(float((native["history_row0_by_layer"] - masked["history_row0_by_layer"]).abs().max()) <= TOL,
            "cold historical-state equality failed")
    observed_inputs = json.loads((OUT / "inputs.json").read_text())
    reconstructed = {"input_ids": torch.tensor(observed_inputs["full_input_ids"]),
                     "attention_mask": torch.tensor(observed_inputs["source_attention_mask"]),
                     "position_ids": torch.tensor(observed_inputs["position_ids"]),
                     "cache_position": torch.tensor(observed_inputs["cache_position"])}
    require(all(tensor_hash(reconstructed[k]) == pre["source_tensor_hashes"][k]
                for k in reconstructed), "cold source input/positions changed")
    for cell in receipt["cells"]:
        condition = cell["condition"]
        mask, _base = mask_for(reconstructed["attention_mask"], pre["shape"]["prompt_width"], condition,
                               source_mask_hash=pre["source_tensor_hashes"]["attention_mask"])
        require(cell["actual_input_hashes"] == {
            "input_ids": pre["source_tensor_hashes"]["input_ids"],
            "attention_mask": tensor_hash(mask),
            "position_ids": pre["source_tensor_hashes"]["position_ids"],
            "cache_position": pre["source_tensor_hashes"]["cache_position"]},
            f"cold actual consumer input/mask changed: {condition}")
    source = json.loads(Path(image["source_bindings"]["trace"]["path"]).read_text())
    raw = json.loads(Path(image["source_bindings"]["raw"]["path"]).read_text())["rows"]
    parity = [_trace_compare(logits=native["logits"][i], trace=source, batch_index=i,
                             absolute_offset=OFFSET, token_id=raw[i]["token_ids"][OFFSET],
                             role="cold_native_x1") for i in (0, 1, 3)]
    require(all(x["passed"] for x in parity), "cold source parity failed")
    def stats(logits):
        v = logits[TARGET].double();p = torch.softmax(v, dim=-1); logp = torch.log_softmax(v, dim=-1)
        top = torch.topk(v, 2)
        return {"global_top2_ids": top.indices.tolist(), "global_top2_logits": top.values.tolist(),
                "global_gap": float(top.values[0] - top.values[1]),
                "tokens": {str(t): {"logit": float(v[t]), "prob": float(p[t]),
                                    "logprob": float(logp[t]), "rank": int((v > v[t]).sum()) + 1}
                           for t in (OLD0, OLD, NEW)}}
    reductions = {name: stats(vectors[name]["logits"]) for name in CONDITIONS}
    def margin(v, old): return float(v[TARGET, NEW].double() - v[TARGET, old].double())
    primary = {name: margin(vectors[name]["logits"], OLD) for name in CONDITIONS}
    secondary = {name: margin(vectors[name]["logits"], OLD0) for name in CONDITIONS}
    p = native["logits"][TARGET].double().softmax(-1)
    q = masked["logits"][TARGET].double().softmax(-1)
    summary = {"schema": "recurrence_native_history_read.readback.v1", "status": "cold_readback_passed",
               "receipt": bind(OUT / "receipt.json"), "admission": bind(ADMISSION),
               "interpretation_note": bind(NOTE), "source_parity": parity,
               "native_sham_max_error": float((native["logits"] - sham["logits"]).abs().max()),
               "treatment_companion_max_errors": [float((native["logits"][i]-masked["logits"][i]).abs().max()) for i in (0,1,2)],
               "historical_row0_all_layer_max_error": float((native["history_row0_by_layer"]-masked["history_row0_by_layer"]).abs().max()),
               "reductions": reductions, "primary_B_minus_A1": {"margins": primary,
                    "delta_mask_minus_native": primary["latest-row-mask"] - primary["native"], "deadband": .01},
               "secondary_B_minus_A0": {"margins": secondary,
                    "delta_mask_minus_native": secondary["latest-row-mask"] - secondary["native"]},
               "full_vocabulary_tv_native_mask": float(.5 * (p-q).abs().sum()),
               "counts": receipt["counts"], "allocated_gpu_seconds": receipt["allocated_gpu_seconds"],
               "prior_sequence_gpu_hours": admission["prior_sequence_gpu_hours"],
               "cumulative_sequence_gpu_hours": admission["prior_sequence_gpu_hours"] + receipt["allocated_gpu_seconds"] / 3600}
    write_new(OUT / "readback.json", summary)
    print(json.dumps({"status": summary["status"], "primary_delta": summary["primary_B_minus_A1"]["delta_mask_minus_native"],
                      "secondary_delta": summary["secondary_B_minus_A0"]["delta_mask_minus_native"],
                      "tv": summary["full_vocabulary_tv_native_mask"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("preflight", "run", "readback"))
    action = parser.parse_args().action
    {"preflight": preflight, "run": run, "readback": readback}[action]()
